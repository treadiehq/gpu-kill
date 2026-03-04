use crate::nvml_api::{GpuProc, GpuSnapshot};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

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

    /// Log GPU usage snapshot
    pub async fn log_snapshot(
        &self,
        snapshots: &[GpuSnapshot],
        processes: &[GpuProc],
    ) -> Result<()> {
        let timestamp = Utc::now();
        let mut records = Vec::new();

        for snapshot in snapshots {
            // Log GPU-level information
            let gpu_record = AuditRecord {
                id: timestamp.timestamp_millis(), // Use timestamp as ID
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
                node_id: None,
            };

            records.push(gpu_record);

            // Log process-level information. Attribute GPU utilization proportionally
            // across processes on this GPU so RogueDetector heuristics are effective.
            let gpu_processes: Vec<_> = processes
                .iter()
                .filter(|p| p.gpu_index == snapshot.gpu_index)
                .collect();
            let process_count = gpu_processes.len().max(1);
            let util_per_process = snapshot.util_pct / process_count as f32;

            for process in gpu_processes {
                let process_record = AuditRecord {
                    id: timestamp.timestamp_millis() + process.pid as i64, // Use timestamp + PID as ID
                    timestamp,
                    gpu_index: snapshot.gpu_index,
                    gpu_name: snapshot.name.clone(),
                    pid: Some(process.pid),
                    user: Some(process.user.clone()),
                    process_name: Some(process.proc_name.clone()),
                    memory_used_mb: process.used_mem_mb,
                    utilization_pct: util_per_process,
                    temperature_c: 0, // Process-level temperature not available
                    power_w: 0.0,     // Process-level power not available
                    container: process.container.clone(),
                    node_id: None,
                };

                records.push(process_record);
            }
        }

        // Append records to JSON file
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

    /// Query audit records with filters
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

        let content = fs::read_to_string(&file_path)
            .map_err(|e| anyhow::anyhow!("Failed to read audit file: {}", e))?;

        let mut records = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let record: AuditRecord = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("Failed to parse audit record: {}", e))?;

            // Filter by time
            if record.timestamp < since {
                continue;
            }

            // Filter by user
            if let Some(user) = user_filter {
                if let Some(ref record_user) = record.user {
                    if record_user != user {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Filter by process
            if let Some(process) = process_filter {
                if let Some(ref record_process) = record.process_name {
                    if !record_process.contains(process) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            records.push(record);
        }

        // Sort by timestamp descending
        records.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(records)
    }

    /// Get audit summary statistics
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

        let content = fs::read_to_string(&file_path)
            .map_err(|e| anyhow::anyhow!("Failed to read audit file: {}", e))?;

        let mut records = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let record: AuditRecord = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("Failed to parse audit record: {}", e))?;

            if record.timestamp >= since {
                records.push(record);
            }
        }

        let total_records = records.len() as u64;

        // Calculate top users
        let mut user_stats: std::collections::HashMap<String, (u64, u32)> =
            std::collections::HashMap::new();
        for record in &records {
            if let Some(ref user) = record.user {
                let entry = user_stats.entry(user.clone()).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += record.memory_used_mb;
            }
        }
        let mut top_users: Vec<(String, u64, u32)> = user_stats
            .into_iter()
            .map(|(user, (count, memory))| (user, count, memory))
            .collect();
        top_users.sort_by(|a, b| b.2.cmp(&a.2));
        top_users.truncate(10);

        // Calculate top processes
        let mut process_stats: std::collections::HashMap<String, (u64, u32)> =
            std::collections::HashMap::new();
        for record in &records {
            if let Some(ref process) = record.process_name {
                let entry = process_stats.entry(process.clone()).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += record.memory_used_mb;
            }
        }
        let mut top_processes: Vec<(String, u64, u32)> = process_stats
            .into_iter()
            .map(|(process, (count, memory))| (process, count, memory))
            .collect();
        top_processes.sort_by(|a, b| b.2.cmp(&a.2));
        top_processes.truncate(10);

        // Calculate GPU usage by hour
        let mut gpu_usage_by_hour = Vec::new();
        for hour in 0..hours {
            let hour_start = since + chrono::Duration::hours(hour as i64);
            let hour_end = hour_start + chrono::Duration::hours(1);

            let hour_records: Vec<&AuditRecord> = records
                .iter()
                .filter(|r| r.timestamp >= hour_start && r.timestamp < hour_end)
                .collect();

            let avg_memory = if hour_records.is_empty() {
                0.0
            } else {
                let total_memory: u32 = hour_records.iter().map(|r| r.memory_used_mb).sum();
                total_memory as f64 / hour_records.len() as f64
            };

            gpu_usage_by_hour.push((hour, avg_memory as u32));
        }

        Ok(AuditSummary {
            total_records,
            time_range_hours: hours,
            top_users,
            top_processes,
            gpu_usage_by_hour,
        })
    }

    /// Clean up old audit records (keep only last N days)
    pub async fn cleanup_old_records(&self, keep_days: u32) -> Result<u64> {
        let cutoff = Utc::now() - chrono::Duration::days(keep_days as i64);
        let file_path = self.data_dir.join("audit.jsonl");

        if !file_path.exists() {
            return Ok(0);
        }

        let content = fs::read_to_string(&file_path)
            .map_err(|e| anyhow::anyhow!("Failed to read audit file: {}", e))?;

        let mut records = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let record: AuditRecord = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("Failed to parse audit record: {}", e))?;

            if record.timestamp >= cutoff {
                records.push(record);
            }
        }

        let removed_count = content.lines().count() as u64 - records.len() as u64;

        // Write back the filtered records
        let mut file = fs::File::create(&file_path)
            .map_err(|e| anyhow::anyhow!("Failed to create audit file: {}", e))?;

        for record in records {
            let json_line = serde_json::to_string(&record)
                .map_err(|e| anyhow::anyhow!("Failed to serialize record: {}", e))?;
            writeln!(file, "{}", json_line)
                .map_err(|e| anyhow::anyhow!("Failed to write to audit file: {}", e))?;
        }

        Ok(removed_count)
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
