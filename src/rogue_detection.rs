use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::audit::{AuditManager, AuditRecord};
use crate::nvml_api::GpuProc;

/// Rogue detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RogueDetectionResult {
    pub timestamp: DateTime<Utc>,
    pub suspicious_processes: Vec<SuspiciousProcess>,
    pub crypto_miners: Vec<CryptoMiner>,
    pub resource_abusers: Vec<ResourceAbuser>,
    pub data_exfiltrators: Vec<DataExfiltrator>,
    pub risk_score: f32,
    pub recommendations: Vec<String>,
}

/// Suspicious process detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousProcess {
    pub process: GpuProc,
    pub reasons: Vec<String>,
    pub confidence: f32,
    pub risk_level: RiskLevel,
}

/// Crypto miner detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoMiner {
    pub process: GpuProc,
    pub mining_indicators: Vec<String>,
    pub confidence: f32,
    pub estimated_hashrate: Option<f32>,
}

/// Resource abuse detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAbuser {
    pub process: GpuProc,
    pub abuse_type: AbuseType,
    pub severity: f32,
    pub duration_hours: f32,
}

/// Data exfiltration detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExfiltrator {
    pub process: GpuProc,
    pub exfil_indicators: Vec<String>,
    pub confidence: f32,
    pub data_volume_mb: Option<f32>,
}

/// Risk levels for suspicious activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Types of resource abuse
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AbuseType {
    MemoryHog,
    LongRunning,
    ExcessiveUtilization,
    UnauthorizedAccess,
}

/// Rogue detection heuristics and rules
pub struct RogueDetector {
    audit_manager: AuditManager,
    detection_rules: DetectionRules,
}

/// Which detection types are active
#[derive(Debug, Clone)]
pub struct EnabledDetections {
    pub crypto_miners: bool,
    pub suspicious_processes: bool,
    pub resource_abusers: bool,
    pub data_exfiltrators: bool,
}

impl Default for EnabledDetections {
    fn default() -> Self {
        Self {
            crypto_miners: true,
            suspicious_processes: true,
            resource_abusers: true,
            data_exfiltrators: false,
        }
    }
}

/// Scoring weights for each threat category
#[derive(Debug, Clone)]
pub struct ThreatWeights {
    pub crypto_miner: f32,
    pub suspicious_process: f32,
    pub resource_abuser: f32,
    pub data_exfiltrator: f32,
}

impl Default for ThreatWeights {
    fn default() -> Self {
        Self {
            crypto_miner: 0.8,
            suspicious_process: 0.6,
            resource_abuser: 0.3,
            data_exfiltrator: 0.9,
        }
    }
}

/// Thresholds mapping confidence → RiskLevel
#[derive(Debug, Clone)]
pub struct RiskThresholds {
    pub critical: f32,
    pub high: f32,
    pub medium: f32,
}

impl Default for RiskThresholds {
    fn default() -> Self {
        Self {
            critical: 0.9,
            high: 0.7,
            medium: 0.5,
        }
    }
}

/// A single custom detection pattern carried from config
#[derive(Debug, Clone)]
pub struct CustomPattern {
    pub name: String,
    pub pattern: String,
    pub confidence_boost: f32,
}

/// Configurable detection rules
#[derive(Debug, Clone)]
pub struct DetectionRules {
    pub crypto_miner_patterns: Vec<String>,
    pub suspicious_process_names: Vec<String>,
    pub max_memory_usage_gb: f32,
    pub max_utilization_pct: f32,
    pub max_duration_hours: f32,
    pub min_confidence_threshold: f32,
    /// Users in this list are exempt from rogue detection
    pub user_whitelist: Vec<String>,
    /// Processes in this list are exempt from rogue detection
    pub process_whitelist: Vec<String>,
    /// Which detection categories are active
    pub enabled_detections: EnabledDetections,
    /// Scoring weights per threat type
    pub threat_weights: ThreatWeights,
    /// Confidence thresholds for risk levels
    pub risk_thresholds: RiskThresholds,
    /// User-defined custom patterns applied in suspicious-process detection
    pub custom_patterns: Vec<CustomPattern>,
}

impl Default for DetectionRules {
    fn default() -> Self {
        Self {
            crypto_miner_patterns: vec![
                "cuda".to_string(),
                "opencl".to_string(),
                "miner".to_string(),
                "hash".to_string(),
                "cryptonight".to_string(),
                "ethash".to_string(),
                "equihash".to_string(),
            ],
            suspicious_process_names: vec![
                "xmrig".to_string(),
                "ccminer".to_string(),
                "cgminer".to_string(),
                "bfgminer".to_string(),
                "sgminer".to_string(),
                "ethminer".to_string(),
                "t-rex".to_string(),
                "lolminer".to_string(),
                "nbminer".to_string(),
                "gminer".to_string(),
            ],
            max_memory_usage_gb: 20.0,
            max_utilization_pct: 95.0,
            max_duration_hours: 24.0,
            min_confidence_threshold: 0.7,
            user_whitelist: vec![
                "root".to_string(),
                "admin".to_string(),
                "system".to_string(),
            ],
            process_whitelist: vec![
                "python".to_string(),
                "jupyter".to_string(),
                "tensorflow".to_string(),
                "pytorch".to_string(),
                "nvidia-smi".to_string(),
            ],
            enabled_detections: EnabledDetections::default(),
            threat_weights: ThreatWeights::default(),
            risk_thresholds: RiskThresholds::default(),
            custom_patterns: Vec::new(),
        }
    }
}

impl RogueDetector {
    /// Create a new rogue detector
    pub fn new(audit_manager: AuditManager) -> Self {
        Self {
            audit_manager,
            detection_rules: DetectionRules::default(),
        }
    }

    /// Create a new rogue detector with configuration
    pub fn with_config(
        audit_manager: AuditManager,
        config_manager: &crate::rogue_config::RogueConfigManager,
    ) -> Self {
        Self {
            audit_manager,
            detection_rules: config_manager.to_detection_rules(),
        }
    }

    /// Create a new rogue detector with custom rules
    #[allow(dead_code)]
    pub fn with_rules(audit_manager: AuditManager, rules: DetectionRules) -> Self {
        Self {
            audit_manager,
            detection_rules: rules,
        }
    }

    /// Analyze audit records for suspicious activity
    pub async fn detect_rogue_activity(&self, hours: u32) -> Result<RogueDetectionResult> {
        info!("Starting rogue activity detection for last {} hours", hours);

        let audit_records = self.audit_manager.query_records(hours, None, None).await?;
        debug!("Analyzing {} audit records", audit_records.len());

        let mut suspicious_processes = Vec::new();
        let mut crypto_miners = Vec::new();
        let mut resource_abusers = Vec::new();
        let mut data_exfiltrators = Vec::new();

        // Group audit records by (node_id, pid) for analysis
        let process_groups = self.group_records_by_pid(&audit_records);

        let en = &self.detection_rules.enabled_detections;

        for (_key, records) in process_groups {
            if en.crypto_miners {
                if let Some(miner) = self.detect_crypto_miner(&records) {
                    crypto_miners.push(miner);
                }
            }
            if en.suspicious_processes {
                if let Some(suspicious) = self.detect_suspicious_process(&records) {
                    suspicious_processes.push(suspicious);
                }
            }
            if en.resource_abusers {
                if let Some(abuser) = self.detect_resource_abuser(&records) {
                    resource_abusers.push(abuser);
                }
            }
            if en.data_exfiltrators {
                if let Some(exfiltrator) = self.detect_data_exfiltrator(&records) {
                    data_exfiltrators.push(exfiltrator);
                }
            }
        }

        let risk_score = self.calculate_risk_score(
            &suspicious_processes,
            &crypto_miners,
            &resource_abusers,
            &data_exfiltrators,
        );

        let recommendations = self.generate_recommendations(
            &suspicious_processes,
            &crypto_miners,
            &resource_abusers,
            &data_exfiltrators,
        );

        let result = RogueDetectionResult {
            timestamp: Utc::now(),
            suspicious_processes,
            crypto_miners,
            resource_abusers,
            data_exfiltrators,
            risk_score,
            recommendations,
        };

        info!("Rogue detection completed. Risk score: {:.2}", risk_score);
        Ok(result)
    }

    /// Analyze a provided list of audit records (e.g. from cluster snapshots) for rogue activity.
    /// Does not query the audit manager; use this for cluster-wide analysis.
    #[allow(dead_code)]
    pub async fn detect_rogue_activity_from_records(
        &self,
        records: Vec<AuditRecord>,
    ) -> Result<RogueDetectionResult> {
        debug!(
            "Analyzing {} audit records (cluster or ad-hoc)",
            records.len()
        );

        let mut suspicious_processes = Vec::new();
        let mut crypto_miners = Vec::new();
        let mut resource_abusers = Vec::new();
        let mut data_exfiltrators = Vec::new();

        let process_groups = self.group_records_by_pid(&records);

        let en = &self.detection_rules.enabled_detections;

        for (_key, group_records) in process_groups {
            if en.crypto_miners {
                if let Some(miner) = self.detect_crypto_miner(&group_records) {
                    crypto_miners.push(miner);
                }
            }
            if en.suspicious_processes {
                if let Some(suspicious) = self.detect_suspicious_process(&group_records) {
                    suspicious_processes.push(suspicious);
                }
            }
            if en.resource_abusers {
                if let Some(abuser) = self.detect_resource_abuser(&group_records) {
                    resource_abusers.push(abuser);
                }
            }
            if en.data_exfiltrators {
                if let Some(exfiltrator) = self.detect_data_exfiltrator(&group_records) {
                    data_exfiltrators.push(exfiltrator);
                }
            }
        }

        let risk_score = self.calculate_risk_score(
            &suspicious_processes,
            &crypto_miners,
            &resource_abusers,
            &data_exfiltrators,
        );
        let recommendations = self.generate_recommendations(
            &suspicious_processes,
            &crypto_miners,
            &resource_abusers,
            &data_exfiltrators,
        );

        Ok(RogueDetectionResult {
            timestamp: Utc::now(),
            suspicious_processes,
            crypto_miners,
            resource_abusers,
            data_exfiltrators,
            risk_score,
            recommendations,
        })
    }

    /// Group audit records by (node_id, pid) for analysis. When node_id is None (local),
    /// uses empty string so single-node behavior is unchanged.
    fn group_records_by_pid(
        &self,
        records: &[AuditRecord],
    ) -> HashMap<(String, u32), Vec<AuditRecord>> {
        let mut groups = HashMap::new();

        for record in records {
            if let Some(pid) = record.pid {
                let node_key = record.node_id.clone().unwrap_or_default();
                groups
                    .entry((node_key, pid))
                    .or_insert_with(Vec::new)
                    .push(record.clone());
            }
        }

        groups
    }

    /// Check if a user is whitelisted
    fn is_user_whitelisted(&self, user: &str) -> bool {
        self.detection_rules
            .user_whitelist
            .iter()
            .any(|u| u.eq_ignore_ascii_case(user))
    }

    /// Check if a process is whitelisted. Matches exact name or process name that
    /// starts with a whitelist entry (e.g. "python" matches "python3", "python3.10").
    fn is_process_whitelisted(&self, process_name: &str) -> bool {
        let name_lower = process_name.to_lowercase();
        self.detection_rules.process_whitelist.iter().any(|entry| {
            let entry_lower = entry.to_lowercase();
            name_lower == entry_lower || name_lower.starts_with(&entry_lower)
        })
    }

    /// Only skip detection if every record is whitelisted (user and process).
    /// If any record was non-whitelisted or had a suspicious name, we do not skip.
    fn all_records_whitelisted(&self, records: &[AuditRecord]) -> bool {
        records.iter().all(|r| {
            let user_ok = r.user.as_ref().is_none_or(|u| self.is_user_whitelisted(u));
            let process_ok = r
                .process_name
                .as_ref()
                .is_none_or(|p| self.is_process_whitelisted(p));
            // OR: either a trusted user OR a trusted process is sufficient for exemption.
            user_ok || process_ok
        })
    }

    /// Compute name-based confidence for crypto miner detection (patterns + known miner names).
    /// Returns (confidence, indicators, optional best record index).
    fn crypto_name_confidence_from_records(
        &self,
        records: &[AuditRecord],
    ) -> (f32, Vec<String>, Option<usize>) {
        let mut indicators = Vec::new();
        let mut pattern_matched = std::collections::HashSet::<String>::new();
        let mut miner_matched = std::collections::HashSet::<String>::new();
        let mut best_idx = None;
        let mut best_score = 0.0f32;

        for (idx, record) in records.iter().enumerate() {
            let mut score = 0.0f32;
            if let Some(process_name) = &record.process_name {
                let process_name_lower = process_name.to_lowercase();
                for pattern in &self.detection_rules.crypto_miner_patterns {
                    if process_name_lower.contains(pattern)
                        && pattern_matched.insert(pattern.clone())
                    {
                        indicators.push(format!("Process name contains '{}'", pattern));
                        score += 0.3;
                    }
                }
                for miner_name in &self.detection_rules.suspicious_process_names {
                    if process_name_lower.contains(miner_name)
                        && miner_matched.insert(miner_name.clone())
                    {
                        indicators.push(format!("Known miner process: {}", miner_name));
                        score += 0.5;
                    }
                }
            }
            if score > best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }
        let confidence = pattern_matched.len() as f32 * 0.3 + miner_matched.len() as f32 * 0.5;
        (confidence, indicators, best_idx)
    }

    /// Detect crypto mining activity. Evaluates all records so that evasion by
    /// renaming to a whitelisted name or PID recycling does not hide prior rogue activity.
    fn detect_crypto_miner(&self, records: &[AuditRecord]) -> Option<CryptoMiner> {
        if records.is_empty() {
            return None;
        }

        // Skip only if every record is whitelisted; if any record was non-whitelisted, run detection
        if self.all_records_whitelisted(records) {
            debug!("Skipping crypto miner detection: all records whitelisted");
            return None;
        }

        let (name_confidence, mut indicators, best_idx) =
            self.crypto_name_confidence_from_records(records);
        let mut confidence = name_confidence;

        // Use the most suspicious record (by name) for output, so we report the rogue name
        let record = best_idx.and_then(|i| records.get(i)).unwrap_or(&records[0]);

        // Check for high GPU utilization (aggregate over all records)
        if let Some(avg_util) = self.calculate_average_utilization(records) {
            if avg_util > self.detection_rules.max_utilization_pct {
                indicators.push(format!("High GPU utilization: {:.1}%", avg_util));
                confidence += 0.2;
            }
        }

        // Check for sustained high memory usage
        if let Some(avg_memory) = self.calculate_average_memory_usage(records) {
            if avg_memory > self.detection_rules.max_memory_usage_gb {
                indicators.push(format!("High memory usage: {:.1} GB", avg_memory));
                confidence += 0.1;
            }
        }

        // Check for long-running processes
        if let Some(duration) = self.calculate_process_duration(records) {
            if duration > 2.0 {
                indicators.push(format!("Long-running process: {:.1} hours", duration));
                confidence += 0.1;
            }
        }

        if confidence >= self.detection_rules.min_confidence_threshold {
            let process = GpuProc {
                gpu_index: record.gpu_index,
                pid: record.pid.unwrap_or(0),
                user: record.user.clone().unwrap_or_else(|| "unknown".to_string()),
                proc_name: record
                    .process_name
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                used_mem_mb: record.memory_used_mb,
                start_time: "unknown".to_string(),
                container: record.container.clone(),
                node_id: record.node_id.clone(),
            };

            Some(CryptoMiner {
                process,
                mining_indicators: indicators,
                confidence,
                estimated_hashrate: self.estimate_hashrate(records),
            })
        } else {
            None
        }
    }

    /// Detect suspicious processes. Evaluates all records so that evasion by
    /// renaming or PID recycling does not hide prior suspicious activity.
    fn detect_suspicious_process(&self, records: &[AuditRecord]) -> Option<SuspiciousProcess> {
        if records.is_empty() {
            return None;
        }

        if self.all_records_whitelisted(records) {
            debug!("Skipping suspicious process detection: all records whitelisted");
            return None;
        }

        let mut reasons = Vec::new();
        let mut confidence = 0.0;
        let mut representative_idx = 0usize;

        // Check if any record had unusual process name or unusual user
        for (idx, record) in records.iter().enumerate() {
            if let Some(process_name) = &record.process_name {
                if self.is_unusual_process_name(process_name) {
                    reasons.push("Unusual process name pattern".to_string());
                    confidence += 0.3;
                    representative_idx = idx;
                    break;
                }
            }
        }
        for (idx, record) in records.iter().enumerate() {
            if let Some(user) = &record.user {
                if self.is_unusual_user(user) {
                    reasons.push(format!("Unusual user: {}", user));
                    confidence += 0.2;
                    if representative_idx == 0 {
                        representative_idx = idx;
                    }
                    break;
                }
            }
        }

        let representative = &records[representative_idx];

        // Apply user-configured custom patterns
        if !self.detection_rules.custom_patterns.is_empty() {
            for record in records.iter() {
                if let Some(proc_name) = &record.process_name {
                    let proc_lower = proc_name.to_lowercase();
                    for cp in &self.detection_rules.custom_patterns {
                        if proc_lower.contains(&cp.pattern.to_lowercase()) {
                            reasons.push(format!(
                                "Custom pattern '{}' matched process '{}'",
                                cp.name, proc_name
                            ));
                            confidence += cp.confidence_boost;
                            break;
                        }
                    }
                }
            }
        }

        // Check for high resource usage (aggregate)
        if let Some(avg_util) = self.calculate_average_utilization(records) {
            if avg_util > self.detection_rules.max_utilization_pct {
                reasons.push(format!("Excessive GPU utilization: {:.1}%", avg_util));
                confidence += 0.4;
            }
        }

        if let Some(avg_memory) = self.calculate_average_memory_usage(records) {
            if avg_memory > self.detection_rules.max_memory_usage_gb {
                reasons.push(format!("Excessive memory usage: {:.1} GB", avg_memory));
                confidence += 0.3;
            }
        }

        if confidence >= self.detection_rules.min_confidence_threshold {
            let process = GpuProc {
                gpu_index: representative.gpu_index,
                pid: representative.pid.unwrap_or(0),
                user: representative
                    .user
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                proc_name: representative
                    .process_name
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                used_mem_mb: representative.memory_used_mb,
                start_time: "unknown".to_string(),
                container: representative.container.clone(),
                node_id: representative.node_id.clone(),
            };

            Some(SuspiciousProcess {
                process,
                reasons,
                confidence,
                risk_level: self.determine_risk_level(confidence),
            })
        } else {
            None
        }
    }

    /// Detect resource abuse. Only skips if all records are whitelisted.
    fn detect_resource_abuser(&self, records: &[AuditRecord]) -> Option<ResourceAbuser> {
        if records.is_empty() {
            return None;
        }

        if self.all_records_whitelisted(records) {
            debug!("Skipping resource abuser detection: all records whitelisted");
            return None;
        }

        let record = &records[0];

        let mut abuse_type = AbuseType::MemoryHog;
        let mut severity = 0.0;

        // Check for memory abuse
        if let Some(avg_memory) = self.calculate_average_memory_usage(records) {
            if avg_memory > self.detection_rules.max_memory_usage_gb {
                let memory_severity =
                    (avg_memory / self.detection_rules.max_memory_usage_gb).min(2.0);
                if memory_severity > severity {
                    abuse_type = AbuseType::MemoryHog;
                    severity = memory_severity;
                }
            }
        }

        // Check for excessive utilization
        if let Some(avg_util) = self.calculate_average_utilization(records) {
            if avg_util > self.detection_rules.max_utilization_pct {
                let util_severity = (avg_util / self.detection_rules.max_utilization_pct).min(2.0);
                if util_severity > severity {
                    abuse_type = AbuseType::ExcessiveUtilization;
                    severity = util_severity;
                }
            }
        }

        // Check for long-running processes
        if let Some(duration) = self.calculate_process_duration(records) {
            if duration > self.detection_rules.max_duration_hours {
                let duration_severity =
                    (duration / self.detection_rules.max_duration_hours).min(2.0);
                if duration_severity > severity {
                    abuse_type = AbuseType::LongRunning;
                    severity = duration_severity;
                }
            }
        }

        if severity > 1.0 {
            // Create a GpuProc from the AuditRecord for compatibility
            let process = GpuProc {
                gpu_index: record.gpu_index,
                pid: record.pid.unwrap_or(0),
                user: record.user.clone().unwrap_or_else(|| "unknown".to_string()),
                proc_name: record
                    .process_name
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                used_mem_mb: record.memory_used_mb,
                start_time: "unknown".to_string(),
                container: record.container.clone(),
                node_id: record.node_id.clone(),
            };

            Some(ResourceAbuser {
                process,
                abuse_type,
                severity,
                duration_hours: self.calculate_process_duration(records).unwrap_or(0.0),
            })
        } else {
            None
        }
    }

    /// Detect data exfiltration (placeholder - would need network monitoring)
    fn detect_data_exfiltrator(&self, _records: &[AuditRecord]) -> Option<DataExfiltrator> {
        // This would require network monitoring data
        // For now, we'll implement basic heuristics
        None
    }

    /// Calculate average GPU utilization for a process
    fn calculate_average_utilization(&self, records: &[AuditRecord]) -> Option<f32> {
        if records.is_empty() {
            return None;
        }

        let total_util: f32 = records.iter().map(|r| r.utilization_pct).sum();

        Some(total_util / records.len() as f32)
    }

    /// Calculate average memory usage for a process
    fn calculate_average_memory_usage(&self, records: &[AuditRecord]) -> Option<f32> {
        if records.is_empty() {
            return None;
        }

        let total_memory: f32 = records
            .iter()
            .map(|r| r.memory_used_mb as f32 / 1024.0)
            .sum();

        Some(total_memory / records.len() as f32)
    }

    /// Calculate process duration in hours
    fn calculate_process_duration(&self, records: &[AuditRecord]) -> Option<f32> {
        if records.len() < 2 {
            return None;
        }

        let timestamps: Vec<DateTime<Utc>> = records.iter().map(|r| r.timestamp).collect();

        if timestamps.len() < 2 {
            return None;
        }

        let min_time = timestamps.iter().min()?;
        let max_time = timestamps.iter().max()?;
        let duration = (*max_time - *min_time).num_seconds() as f32 / 3600.0;

        Some(duration)
    }

    /// Estimate hashrate for crypto mining (placeholder)
    fn estimate_hashrate(&self, _records: &[AuditRecord]) -> Option<f32> {
        // This would require more sophisticated analysis
        None
    }

    /// Check if process name is unusual
    fn is_unusual_process_name(&self, name: &str) -> bool {
        let name_lower = name.to_lowercase();

        // Check for random-looking names
        if name.len() > 20 && name.chars().filter(|c| c.is_ascii_digit()).count() > 5 {
            return true;
        }

        // Check for suspicious patterns
        let suspicious_patterns = ["temp", "tmp", "random", "test", "unknown"];
        for pattern in suspicious_patterns {
            if name_lower.contains(pattern) {
                return true;
            }
        }

        false
    }

    /// Check if user is unusual
    fn is_unusual_user(&self, user: &str) -> bool {
        // root, admin, system are already in the default user whitelist and should
        // never be penalised as "unusual". Only flag genuinely low-privilege accounts
        // that are unlikely to legitimately run GPU workloads.
        let unusual_users = ["daemon", "nobody"];
        unusual_users.contains(&user.to_lowercase().as_str()) && !self.is_user_whitelisted(user)
    }

    /// Determine risk level based on confidence using configured thresholds.
    fn determine_risk_level(&self, confidence: f32) -> RiskLevel {
        let t = &self.detection_rules.risk_thresholds;
        if confidence >= t.critical {
            RiskLevel::Critical
        } else if confidence >= t.high {
            RiskLevel::High
        } else if confidence >= t.medium {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Calculate overall risk score using configured threat weights.
    fn calculate_risk_score(
        &self,
        suspicious: &[SuspiciousProcess],
        miners: &[CryptoMiner],
        abusers: &[ResourceAbuser],
        exfiltrators: &[DataExfiltrator],
    ) -> f32 {
        let w = &self.detection_rules.threat_weights;
        let t = &self.detection_rules.risk_thresholds;
        let mut score = 0.0;

        for process in suspicious {
            let level_weight = match process.risk_level {
                RiskLevel::Critical => 1.0,
                RiskLevel::High => t.high,
                RiskLevel::Medium => t.medium,
                RiskLevel::Low => 0.1,
            };
            score += level_weight * w.suspicious_process;
        }

        for miner in miners {
            score += miner.confidence * w.crypto_miner;
        }

        for abuser in abusers {
            score += abuser.severity * w.resource_abuser;
        }

        for exfiltrator in exfiltrators {
            score += exfiltrator.confidence * w.data_exfiltrator;
        }

        (score / 10.0).min(1.0)
    }

    /// Generate recommendations based on detected threats
    fn generate_recommendations(
        &self,
        suspicious: &[SuspiciousProcess],
        miners: &[CryptoMiner],
        abusers: &[ResourceAbuser],
        _exfiltrators: &[DataExfiltrator],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !miners.is_empty() {
            recommendations.push(
                "🚨 CRITICAL: Crypto miners detected! Consider immediate termination.".to_string(),
            );
            recommendations
                .push("Review system security and check for unauthorized access.".to_string());
        }

        if !suspicious.is_empty() {
            recommendations
                .push("⚠️ Suspicious processes detected. Review and investigate.".to_string());
        }

        if !abusers.is_empty() {
            recommendations.push(
                "📊 Resource abuse detected. Consider implementing usage limits.".to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("✅ No suspicious activity detected.".to_string());
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::AuditManager;
    // use chrono::Utc; // Unused for now

    #[test]
    fn test_detection_rules_default() {
        let rules = DetectionRules::default();
        assert!(!rules.crypto_miner_patterns.is_empty());
        assert!(!rules.suspicious_process_names.is_empty());
        assert!(rules.max_memory_usage_gb > 0.0);
        // Verify whitelist fields exist and have defaults
        assert!(!rules.user_whitelist.is_empty());
        assert!(!rules.process_whitelist.is_empty());
        assert!(rules.user_whitelist.contains(&"root".to_string()));
        assert!(rules.process_whitelist.contains(&"python".to_string()));
    }

    #[tokio::test]
    async fn test_rogue_detector_creation() {
        let audit_manager = AuditManager::new().await.unwrap();
        let detector = RogueDetector::new(audit_manager);
        assert_eq!(detector.detection_rules.min_confidence_threshold, 0.7);
    }

    #[tokio::test]
    async fn test_whitelist_functionality() {
        let audit_manager = AuditManager::new().await.unwrap();
        let detector = RogueDetector::new(audit_manager);

        // Test user whitelist checking
        assert!(detector.is_user_whitelisted("root"));
        assert!(detector.is_user_whitelisted("ROOT")); // Case insensitive
        assert!(detector.is_user_whitelisted("admin"));
        assert!(!detector.is_user_whitelisted("hacker"));
        assert!(!detector.is_user_whitelisted("unknown_user"));

        // Test process whitelist checking (prefix match: "python" matches "python3", etc.)
        assert!(detector.is_process_whitelisted("python"));
        assert!(detector.is_process_whitelisted("python3"));
        assert!(detector.is_process_whitelisted("python3.10"));
        assert!(detector.is_process_whitelisted("jupyter-notebook"));
        assert!(detector.is_process_whitelisted("tensorflow_training"));
        assert!(detector.is_process_whitelisted("pytorch_model"));
        assert!(detector.is_process_whitelisted("nvidia-smi"));
        assert!(!detector.is_process_whitelisted("xmrig"));
        assert!(!detector.is_process_whitelisted("suspicious_miner"));
    }

    #[tokio::test]
    async fn test_crypto_miner_uses_configured_thresholds() {
        use crate::audit::AuditRecord;
        use chrono::Utc;

        let rules = DetectionRules {
            max_memory_usage_gb: 20.0,
            max_utilization_pct: 95.0,
            min_confidence_threshold: 0.1,
            ..DetectionRules::default()
        };
        let detector = RogueDetector::with_rules(AuditManager::new().await.unwrap(), rules);

        let now = Utc::now();
        let records = vec![
            AuditRecord {
                id: 1,
                timestamp: now,
                gpu_index: 0,
                gpu_name: "Test GPU".to_string(),
                pid: Some(1234),
                user: Some("user1".to_string()),
                process_name: Some("miner".to_string()),
                memory_used_mb: 19 * 1024,
                utilization_pct: 94.0,
                temperature_c: 70,
                power_w: 150.0,
                container: None,
                node_id: None,
            },
            AuditRecord {
                id: 2,
                timestamp: now + chrono::Duration::minutes(1),
                gpu_index: 0,
                gpu_name: "Test GPU".to_string(),
                pid: Some(1234),
                user: Some("user1".to_string()),
                process_name: Some("miner".to_string()),
                memory_used_mb: 19 * 1024,
                utilization_pct: 94.0,
                temperature_c: 70,
                power_w: 150.0,
                container: None,
                node_id: None,
            },
        ];

        let miner = detector
            .detect_crypto_miner(&records)
            .expect("Should detect miner based on patterns");
        assert!(miner
            .mining_indicators
            .iter()
            .all(|entry| !entry.contains("High GPU utilization")
                && !entry.contains("High memory usage")));
    }

    #[tokio::test]
    async fn test_crypto_miner_detection_evaluates_all_records_not_just_newest() {
        // Evasion: newest record is "python3" but older records show "xmrig" — should still detect
        use crate::audit::AuditRecord;
        use chrono::Utc;

        let rules = DetectionRules {
            min_confidence_threshold: 0.5,
            ..DetectionRules::default()
        };
        let detector = RogueDetector::with_rules(AuditManager::new().await.unwrap(), rules);

        let now = Utc::now();
        let records = vec![
            AuditRecord {
                id: 1,
                timestamp: now - chrono::Duration::minutes(10),
                gpu_index: 0,
                gpu_name: "Test GPU".to_string(),
                pid: Some(1234),
                user: Some("attacker".to_string()),
                process_name: Some("xmrig".to_string()),
                memory_used_mb: 1024,
                utilization_pct: 0.0,
                temperature_c: 70,
                power_w: 150.0,
                container: None,
                node_id: None,
            },
            AuditRecord {
                id: 2,
                timestamp: now,
                gpu_index: 0,
                gpu_name: "Test GPU".to_string(),
                pid: Some(1234),
                user: Some("attacker".to_string()),
                process_name: Some("python3".to_string()), // renamed to evade
                memory_used_mb: 1024,
                utilization_pct: 0.0,
                temperature_c: 70,
                power_w: 150.0,
                container: None,
                node_id: None,
            },
        ];

        let miner = detector
            .detect_crypto_miner(&records)
            .expect("Should detect miner from history even though newest record is python3");
        assert!(
            miner.mining_indicators.iter().any(|s| s.contains("xmrig")),
            "Indicators should mention xmrig from history"
        );
        assert!(
            miner.process.proc_name == "xmrig",
            "Representative record should be the suspicious one (xmrig), not python3"
        );
    }

    #[tokio::test]
    async fn test_resource_abuser_preserves_highest_severity() {
        use crate::audit::AuditRecord;
        use chrono::Utc;

        let rules = DetectionRules {
            max_memory_usage_gb: 20.0,
            max_utilization_pct: 95.0,
            ..DetectionRules::default()
        };
        let detector = RogueDetector::with_rules(AuditManager::new().await.unwrap(), rules);

        let now = Utc::now();
        let records = vec![
            AuditRecord {
                id: 1,
                timestamp: now,
                gpu_index: 0,
                gpu_name: "Test GPU".to_string(),
                pid: Some(1234),
                user: Some("user1".to_string()),
                process_name: Some("hog".to_string()),
                memory_used_mb: 100 * 1024,
                utilization_pct: 96.0,
                temperature_c: 70,
                power_w: 150.0,
                container: None,
                node_id: None,
            },
            AuditRecord {
                id: 2,
                timestamp: now + chrono::Duration::minutes(1),
                gpu_index: 0,
                gpu_name: "Test GPU".to_string(),
                pid: Some(1234),
                user: Some("user1".to_string()),
                process_name: Some("hog".to_string()),
                memory_used_mb: 100 * 1024,
                utilization_pct: 96.0,
                temperature_c: 70,
                power_w: 150.0,
                container: None,
                node_id: None,
            },
        ];

        let abuser = detector
            .detect_resource_abuser(&records)
            .expect("Should detect abuse");

        assert_eq!(abuser.abuse_type, AbuseType::MemoryHog);
        assert!(abuser.severity >= 2.0);
    }
}
