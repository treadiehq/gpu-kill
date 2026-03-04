use crate::nvml_api::{GpuProc, GpuSnapshot};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use tokio::io::{AsyncBufReadExt, BufReader};

/// Audit record for GPU usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub id: i64,
    pub timestamp: DateTime<Utc>,
    pub gpu_index: u16,
    pub gpu_name: String,
    pub pid: Option<u32>,
    pub user: Option<String>,
    pub process_name: Option<String>,
    pub memory_used_mb: u32,
    pub utilization_pct: f32,
    pub temperature_c: i32,
    pub power_w: f32,
    pub container: Option<String>,
    /// When set, record is from a cluster node; used to group by (node_id, pid).
    #[serde(default)]
    pub node_id: Option<String>,
}

/// Audit summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSummary {
    pub total_records: u64,
    pub time_range_hours: u32,
    pub top_users: Vec<(String, u64, u32)>, // (user, count, total_memory_mb)
    pub top_processes: Vec<(String, u64, u32)>, // (process, count, total_memory_mb)
    pub gpu_usage_by_hour: Vec<(u32, u32)>, // (hour, avg_memory_mb)
}

/// Monotonically increasing record ID generator — eliminates timestamp+PID collisions.
fn next_record_id() -> i64 {
    static COUNTER: AtomicI64 = AtomicI64::new(0);
    let ts = Utc::now().timestamp_millis();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Pack timestamp (high bits) + sequence (low 20 bits) to stay unique and ordered.
    (ts << 20) | (seq & 0xFFFFF)
}

/// Audit manager for GPU usage tracking
#[derive(Clone)]
pub struct AuditManager {
    data_dir: PathBuf,
}

#[allow(dead_code)]
impl AuditManager {
    /// Initialize the audit manager with JSON file storage
    pub async fn new() -> Result<Self> {
        let data_dir = Self::get_data_dir()?;

        tracing::debug!("Initializing audit storage at: {}", data_dir.display());

        // Create directory if it doesn't exist
        fs::create_dir_all(&data_dir)
            .map_err(|e| anyhow::anyhow!("Failed to create audit directory: {}", e))?;

        Ok(Self { data_dir })
    }

    /// Get the data directory path
    fn get_data_dir() -> Result<PathBuf> {
        // Try multiple fallback locations for the data directory
        let mut path = if let Some(data_dir) = dirs::data_dir() {
            data_dir
        } else if let Some(home_dir) = dirs::home_dir() {
            home_dir.join(".local").join("share")
        } else {
            std::env::current_dir()?
        };

        path.push("gpukill");
        Ok(path)
    }

    /// Log GPU usage snapshot.
    ///
    /// `node_id` should be the hostname or cluster node identifier of the machine
    /// generating this snapshot. Pass `None` when the node identity is unknown.
    pub async fn log_snapshot(
        &self,
        snapshots: &[GpuSnapshot],
        processes: &[GpuProc],
        node_id: Option<String>,
    ) -> Result<()> {
        let timestamp = Utc::now();
        let mut records = Vec::new();

        for snapshot in snapshots {
            let gpu_record = AuditRecord {
                id: next_record_id(),
                timestamp,
                gpu_index: snapshot.gpu_index,
                gpu_name: snapshot.name.clone(),
                pid: None,
                user: None,
                process_name: None,
                memory_used_mb: snapshot.mem_used_mb,
                utilization_pct: snapshot.util_pct,
                temperature_c: snapshot.temp_c,
                power_w: snapshot.power_w,
                container: None,
                node_id: node_id.clone(),
            };

            records.push(gpu_record);

            // Attribute GPU utilization proportionally across processes on this GPU.
            let gpu_processes: Vec<_> = processes
                .iter()
                .filter(|p| p.gpu_index == snapshot.gpu_index)
                .collect();
            let process_count = gpu_processes.len().max(1);
            let util_per_process = snapshot.util_pct / process_count as f32;

            for process in gpu_processes {
                let process_record = AuditRecord {
                    id: next_record_id(),
                    timestamp,
                    gpu_index: snapshot.gpu_index,
                    gpu_name: snapshot.name.clone(),
                    pid: Some(process.pid),
                    user: Some(process.user.clone()),
                    process_name: Some(process.proc_name.clone()),
                    memory_used_mb: process.used_mem_mb,
                    utilization_pct: util_per_process,
                    temperature_c: 0,
                    power_w: 0.0,
                    container: process.container.clone(),
                    node_id: node_id.clone(),
                };

                records.push(process_record);
            }
        }

        self.append_records(&records).await?;
        Ok(())
    }

    /// Public wrapper for appending pre-built records (e.g. from cluster coordinator).
    pub async fn append_records_pub(&self, records: &[AuditRecord]) -> Result<()> {
        self.append_records(records).await
    }

    /// Append records to JSON file
    async fn append_records(&self, records: &[AuditRecord]) -> Result<()> {
        let file_path = self.data_dir.join("audit.jsonl");

        // Create a JSON Lines file (one JSON object per line)
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .map_err(|e| anyhow::anyhow!("Failed to open audit file: {}", e))?;

        for record in records {
            let json_line = serde_json::to_string(record)
                .map_err(|e| anyhow::anyhow!("Failed to serialize record: {}", e))?;
            writeln!(file, "{}", json_line)
                .map_err(|e| anyhow::anyhow!("Failed to write to audit file: {}", e))?;
        }

        Ok(())
    }

    /// Query audit records with filters.
    ///
    /// Streams the file line-by-line using `tokio::io::BufReader` to avoid loading
    /// the entire log into memory.
    pub async fn query_records(
        &self,
        hours: u32,
        user_filter: Option<&str>,
        process_filter: Option<&str>,
    ) -> Result<Vec<AuditRecord>> {
        let since = Utc::now() - chrono::Duration::hours(hours as i64);
        let file_path = self.data_dir.join("audit.jsonl");

        if !file_path.exists() {
            return Ok(Vec::new());
        }

        let file = tokio::fs::File::open(&file_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to open audit file: {}", e))?;
        let mut lines = BufReader::new(file).lines();

        let mut records = Vec::new();
        while let Some(line) = lines.next_line().await? {
            let line = line.trim().to_owned();
            if line.is_empty() {
                continue;
            }

            let record: AuditRecord = serde_json::from_str(&line)
                .map_err(|e| anyhow::anyhow!("Failed to parse audit record: {}", e))?;

            if record.timestamp < since {
                continue;
            }

            if let Some(user) = user_filter {
                match &record.user {
                    Some(u) if u == user => {}
                    _ => continue,
                }
            }

            if let Some(process) = process_filter {
                match &record.process_name {
                    Some(p) if p.contains(process) => {}
                    _ => continue,
                }
            }

            records.push(record);
        }

        records.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(records)
    }

    /// Get audit summary statistics.
    ///
    /// Single-pass O(N) aggregation over a streamed file — no full-file load,
    /// no O(N·H) nested loop.
    pub async fn get_summary(&self, hours: u32) -> Result<AuditSummary> {
        let since = Utc::now() - chrono::Duration::hours(hours as i64);
        let file_path = self.data_dir.join("audit.jsonl");

        if !file_path.exists() {
            return Ok(AuditSummary {
                total_records: 0,
                time_range_hours: hours,
                top_users: Vec::new(),
                top_processes: Vec::new(),
                gpu_usage_by_hour: Vec::new(),
            });
        }

        let file = tokio::fs::File::open(&file_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to open audit file: {}", e))?;
        let mut lines = BufReader::new(file).lines();

        let mut total_records: u64 = 0;
        let mut user_stats: std::collections::HashMap<String, (u64, u32)> =
            std::collections::HashMap::new();
        let mut process_stats: std::collections::HashMap<String, (u64, u32)> =
            std::collections::HashMap::new();
        // hour_bucket[h] = (sum_memory, count) for records in hour h since `since`
        let mut hour_buckets: Vec<(u64, u64)> = vec![(0, 0); hours as usize];

        while let Some(line) = lines.next_line().await? {
            let line = line.trim().to_owned();
            if line.is_empty() {
                continue;
            }

            let record: AuditRecord = serde_json::from_str(&line)
                .map_err(|e| anyhow::anyhow!("Failed to parse audit record: {}", e))?;

            if record.timestamp < since {
                continue;
            }

            total_records += 1;

            if let Some(ref user) = record.user {
                let e = user_stats.entry(user.clone()).or_insert((0, 0));
                e.0 += 1;
                e.1 = e.1.saturating_add(record.memory_used_mb);
            }

            if let Some(ref process) = record.process_name {
                let e = process_stats.entry(process.clone()).or_insert((0, 0));
                e.0 += 1;
                e.1 = e.1.saturating_add(record.memory_used_mb);
            }

            // Determine which hour bucket this record falls into (single pass).
            let elapsed = record.timestamp - since;
            let bucket = elapsed.num_hours().clamp(0, hours as i64 - 1) as usize;
            hour_buckets[bucket].0 += record.memory_used_mb as u64;
            hour_buckets[bucket].1 += 1;
        }

        let mut top_users: Vec<(String, u64, u32)> = user_stats
            .into_iter()
            .map(|(user, (count, memory))| (user, count, memory))
            .collect();
        top_users.sort_by(|a, b| b.2.cmp(&a.2));
        top_users.truncate(10);

        let mut top_processes: Vec<(String, u64, u32)> = process_stats
            .into_iter()
            .map(|(process, (count, memory))| (process, count, memory))
            .collect();
        top_processes.sort_by(|a, b| b.2.cmp(&a.2));
        top_processes.truncate(10);

        let gpu_usage_by_hour: Vec<(u32, u32)> = hour_buckets
            .iter()
            .enumerate()
            .map(|(h, (sum, count))| {
                let avg = if *count > 0 { sum / count } else { 0 };
                (h as u32, avg as u32)
            })
            .collect();

        Ok(AuditSummary {
            total_records,
            time_range_hours: hours,
            top_users,
            top_processes,
            gpu_usage_by_hour,
        })
    }

    /// Clean up old audit records (keep only last N days).
    ///
    /// Streams input line-by-line, writes survivors to a temp file, then
    /// atomically renames it over the original so a crash cannot corrupt the log.
    pub async fn cleanup_old_records(&self, keep_days: u32) -> Result<u64> {
        let cutoff = Utc::now() - chrono::Duration::days(keep_days as i64);
        let file_path = self.data_dir.join("audit.jsonl");

        if !file_path.exists() {
            return Ok(0);
        }

        let tmp_path = self.data_dir.join("audit.jsonl.tmp");

        let input = tokio::fs::File::open(&file_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to open audit file: {}", e))?;
        let mut lines = BufReader::new(input).lines();

        // Write to a temp file; sync + rename for atomicity.
        let mut tmp_file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
            .map_err(|e| anyhow::anyhow!("Failed to open tmp audit file: {}", e))?;

        let mut total_read: u64 = 0;
        let mut kept: u64 = 0;

        while let Some(line) = lines.next_line().await? {
            let line = line.trim().to_owned();
            if line.is_empty() {
                continue;
            }

            total_read += 1;

            let record: AuditRecord = serde_json::from_str(&line)
                .map_err(|e| anyhow::anyhow!("Failed to parse audit record: {}", e))?;

            if record.timestamp >= cutoff {
                writeln!(tmp_file, "{}", line)
                    .map_err(|e| anyhow::anyhow!("Failed to write tmp audit file: {}", e))?;
                kept += 1;
            }
        }

        tmp_file
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush tmp audit file: {}", e))?;
        drop(tmp_file);

        fs::rename(&tmp_path, &file_path)
            .map_err(|e| anyhow::anyhow!("Failed to replace audit file atomically: {}", e))?;

        Ok(total_read.saturating_sub(kept))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::nvml_api::{GpuSnapshot, GpuProc}; // Unused for now

    #[tokio::test]
    async fn test_audit_manager() {
        // This test would require a test database setup
        // For now, just test that the manager can be created
        if let Ok(_manager) = AuditManager::new().await {
            // Test passed - manager created successfully
        }
    }
}
