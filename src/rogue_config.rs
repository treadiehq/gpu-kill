use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tracing::info;

use crate::rogue_detection::DetectionRules;

/// Rogue detection configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RogueConfig {
    /// Detection rules and thresholds
    pub detection: DetectionConfig,
    /// Risk scoring weights
    pub scoring: ScoringConfig,
    /// Custom detection patterns
    pub patterns: PatternConfig,
    /// Alerting configuration
    pub alerts: AlertConfig,
    /// Configuration metadata
    pub metadata: ConfigMetadata,
}

/// Detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Maximum memory usage threshold (GB)
    pub max_memory_usage_gb: f32,
    /// Maximum GPU utilization threshold (%)
    pub max_utilization_pct: f32,
    /// Maximum process duration threshold (hours)
    pub max_duration_hours: f32,
    /// Minimum confidence threshold for detection
    pub min_confidence_threshold: f32,
    /// Enable/disable specific detection types
    pub enabled_detections: DetectionTypes,
}

/// Detection types configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionTypes {
    pub crypto_miners: bool,
    pub suspicious_processes: bool,
    pub resource_abusers: bool,
    pub data_exfiltrators: bool,
}

/// Risk scoring configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScoringConfig {
    /// Weights for different threat types
    pub threat_weights: ThreatWeights,
    /// Risk level thresholds
    pub risk_thresholds: RiskThresholds,
}

/// Threat type weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatWeights {
    pub crypto_miner: f32,
    pub suspicious_process: f32,
    pub resource_abuser: f32,
    pub data_exfiltrator: f32,
}

/// Risk level thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskThresholds {
    pub critical: f32,
    pub high: f32,
    pub medium: f32,
    pub low: f32,
}

/// Pattern configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    /// Crypto miner patterns
    pub crypto_miner_patterns: Vec<String>,
    /// Suspicious process names
    pub suspicious_process_names: Vec<String>,
    /// Custom regex patterns
    pub custom_patterns: Vec<CustomPattern>,
    /// User whitelist (processes from these users are ignored)
    pub user_whitelist: Vec<String>,
    /// Process whitelist (these processes are ignored)
    pub process_whitelist: Vec<String>,
}

/// Custom detection pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomPattern {
    pub name: String,
    pub description: String,
    pub pattern: String,
    pub pattern_type: PatternType,
    pub risk_level: RiskLevel,
    pub confidence_boost: f32,
}

/// Pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    ProcessName,
    UserName,
    CommandLine,
    MemoryUsage,
    Utilization,
    Duration,
}

/// Risk levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable/disable alerts
    pub enabled: bool,
    /// Alert thresholds
    pub thresholds: AlertThresholds,
    /// Notification channels
    pub channels: NotificationChannels,
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub risk_score_threshold: f32,
    pub crypto_miner_threshold: u32,
    pub suspicious_process_threshold: u32,
    pub resource_abuser_threshold: u32,
}

/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannels {
    pub console: bool,
    pub log_file: bool,
    pub webhook: Option<WebhookConfig>,
    pub email: Option<EmailConfig>,
}

/// Webhook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    pub url: String,
    pub headers: HashMap<String, String>,
    pub timeout_seconds: u32,
}

/// Email configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailConfig {
    pub smtp_server: String,
    pub smtp_port: u16,
    pub username: String,
    pub password: String,
    pub from_address: String,
    pub to_addresses: Vec<String>,
}

/// Configuration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMetadata {
    pub version: String,
    pub created_at: String,
    pub last_modified: String,
    pub description: String,
}

// Default implementation is now derived

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            max_memory_usage_gb: 20.0,
            max_utilization_pct: 95.0,
            max_duration_hours: 24.0,
            min_confidence_threshold: 0.7,
            enabled_detections: DetectionTypes::default(),
        }
    }
}

impl Default for DetectionTypes {
    fn default() -> Self {
        Self {
            crypto_miners: true,
            suspicious_processes: true,
            resource_abusers: true,
            data_exfiltrators: false, // Disabled by default as it requires network monitoring
        }
    }
}

// Default implementation is now derived

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

impl Default for RiskThresholds {
    fn default() -> Self {
        Self {
            critical: 0.9,
            high: 0.7,
            medium: 0.5,
            low: 0.3,
        }
    }
}

impl Default for PatternConfig {
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
            custom_patterns: Vec::new(),
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
        }
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thresholds: AlertThresholds::default(),
            channels: NotificationChannels::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            risk_score_threshold: 0.7,
            crypto_miner_threshold: 1,
            suspicious_process_threshold: 3,
            resource_abuser_threshold: 2,
        }
    }
}

impl Default for NotificationChannels {
    fn default() -> Self {
        Self {
            console: true,
            log_file: true,
            webhook: None,
            email: None,
        }
    }
}

impl Default for ConfigMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            last_modified: chrono::Utc::now().to_rfc3339(),
            description: "Default GPU Kill rogue detection configuration".to_string(),
        }
    }
}

/// Rogue configuration manager
pub struct RogueConfigManager {
    config_path: PathBuf,
    config: RogueConfig,
}

impl RogueConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Result<Self> {
        let config_path = Self::get_config_path()?;
        let config = if config_path.exists() {
            Self::load_config(&config_path)?
        } else {
            let default_config = RogueConfig::default();
            Self::save_config(&config_path, &default_config)?;
            default_config
        };

        Ok(Self {
            config_path,
            config,
        })
    }

    /// Get the configuration file path
    fn get_config_path() -> Result<PathBuf> {
        let mut path = if let Some(config_dir) = dirs::config_dir() {
            config_dir
        } else if let Some(home_dir) = dirs::home_dir() {
            home_dir.join(".config")
        } else {
            std::env::current_dir()?
        };

        path.push("gpukill");
        fs::create_dir_all(&path)
            .map_err(|e| anyhow::anyhow!("Failed to create config directory: {}", e))?;

        path.push("rogue_config.toml");
        Ok(path)
    }

    /// Load configuration from file
    fn load_config(path: &PathBuf) -> Result<RogueConfig> {
        let content = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file: {}", e))?;

        let config: RogueConfig = toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))?;

        info!(
            "Loaded rogue detection configuration from: {}",
            path.display()
        );
        Ok(config)
    }

    /// Save configuration to file
    fn save_config(path: &PathBuf, config: &RogueConfig) -> Result<()> {
        let content = toml::to_string_pretty(config)
            .map_err(|e| anyhow::anyhow!("Failed to serialize config: {}", e))?;

        fs::write(path, content)
            .map_err(|e| anyhow::anyhow!("Failed to write config file: {}", e))?;

        info!("Saved rogue detection configuration to: {}", path.display());
        Ok(())
    }

    /// Get the current configuration
    pub fn get_config(&self) -> &RogueConfig {
        &self.config
    }

    /// Update the configuration
    pub fn update_config(&mut self, new_config: RogueConfig) -> Result<()> {
        self.config = new_config;
        Self::save_config(&self.config_path, &self.config)?;
        Ok(())
    }

    /// Reload configuration from file
    #[allow(dead_code)]
    pub fn reload(&mut self) -> Result<()> {
        self.config = Self::load_config(&self.config_path)?;
        Ok(())
    }

    /// Convert to DetectionRules, propagating all configured settings.
    pub fn to_detection_rules(&self) -> DetectionRules {
        use crate::rogue_detection::{
            CustomPattern as DetCustomPattern, EnabledDetections, RiskThresholds, ThreatWeights,
        };

        let cfg_en = &self.config.detection.enabled_detections;
        let cfg_w = &self.config.scoring.threat_weights;
        let cfg_t = &self.config.scoring.risk_thresholds;

        DetectionRules {
            crypto_miner_patterns: self.config.patterns.crypto_miner_patterns.clone(),
            suspicious_process_names: self.config.patterns.suspicious_process_names.clone(),
            max_memory_usage_gb: self.config.detection.max_memory_usage_gb,
            max_utilization_pct: self.config.detection.max_utilization_pct,
            max_duration_hours: self.config.detection.max_duration_hours,
            min_confidence_threshold: self.config.detection.min_confidence_threshold,
            user_whitelist: self.config.patterns.user_whitelist.clone(),
            process_whitelist: self.config.patterns.process_whitelist.clone(),
            enabled_detections: EnabledDetections {
                crypto_miners: cfg_en.crypto_miners,
                suspicious_processes: cfg_en.suspicious_processes,
                resource_abusers: cfg_en.resource_abusers,
                data_exfiltrators: cfg_en.data_exfiltrators,
            },
            threat_weights: ThreatWeights {
                crypto_miner: cfg_w.crypto_miner,
                suspicious_process: cfg_w.suspicious_process,
                resource_abuser: cfg_w.resource_abuser,
                data_exfiltrator: cfg_w.data_exfiltrator,
            },
            risk_thresholds: RiskThresholds {
                critical: cfg_t.critical,
                high: cfg_t.high,
                medium: cfg_t.medium,
            },
            custom_patterns: self
                .config
                .patterns
                .custom_patterns
                .iter()
                .map(|cp| DetCustomPattern {
                    name: cp.name.clone(),
                    pattern: cp.pattern.clone(),
                    confidence_boost: cp.confidence_boost,
                })
                .collect(),
        }
    }

    /// Add a custom pattern
    #[allow(dead_code)]
    pub fn add_custom_pattern(&mut self, pattern: CustomPattern) -> Result<()> {
        self.config.patterns.custom_patterns.push(pattern);
        self.config.metadata.last_modified = chrono::Utc::now().to_rfc3339();
        Self::save_config(&self.config_path, &self.config)?;
        Ok(())
    }

    /// Remove a custom pattern by name
    #[allow(dead_code)]
    pub fn remove_custom_pattern(&mut self, name: &str) -> Result<()> {
        self.config
            .patterns
            .custom_patterns
            .retain(|p| p.name != name);
        self.config.metadata.last_modified = chrono::Utc::now().to_rfc3339();
        Self::save_config(&self.config_path, &self.config)?;
        Ok(())
    }

    /// Add a user to the whitelist
    pub fn add_user_to_whitelist(&mut self, user: String) -> Result<()> {
        let user_lower = user.to_lowercase();
        if !self
            .config
            .patterns
            .user_whitelist
            .iter()
            .any(|u| u.eq_ignore_ascii_case(&user_lower))
        {
            self.config.patterns.user_whitelist.push(user_lower);
            self.config.metadata.last_modified = chrono::Utc::now().to_rfc3339();
            Self::save_config(&self.config_path, &self.config)?;
        }
        Ok(())
    }

    /// Remove a user from the whitelist
    pub fn remove_user_from_whitelist(&mut self, user: &str) -> Result<()> {
        self.config
            .patterns
            .user_whitelist
            .retain(|u| !u.eq_ignore_ascii_case(user));
        self.config.metadata.last_modified = chrono::Utc::now().to_rfc3339();
        Self::save_config(&self.config_path, &self.config)?;
        Ok(())
    }

    /// Add a process to the whitelist
    pub fn add_process_to_whitelist(&mut self, process: String) -> Result<()> {
        let process_lower = process.to_lowercase();
        if !self
            .config
            .patterns
            .process_whitelist
            .iter()
            .any(|p| p.eq_ignore_ascii_case(&process_lower))
        {
            self.config.patterns.process_whitelist.push(process_lower);
            self.config.metadata.last_modified = chrono::Utc::now().to_rfc3339();
            Self::save_config(&self.config_path, &self.config)?;
        }
        Ok(())
    }

    /// Remove a process from the whitelist
    pub fn remove_process_from_whitelist(&mut self, process: &str) -> Result<()> {
        self.config
            .patterns
            .process_whitelist
            .retain(|p| !p.eq_ignore_ascii_case(process));
        self.config.metadata.last_modified = chrono::Utc::now().to_rfc3339();
        Self::save_config(&self.config_path, &self.config)?;
        Ok(())
    }

    /// Update detection thresholds
    pub fn update_thresholds(
        &mut self,
        max_memory: Option<f32>,
        max_utilization: Option<f32>,
        max_duration: Option<f32>,
        min_confidence: Option<f32>,
    ) -> Result<()> {
        if let Some(memory) = max_memory {
            self.config.detection.max_memory_usage_gb = memory;
        }
        if let Some(utilization) = max_utilization {
            self.config.detection.max_utilization_pct = utilization;
        }
        if let Some(duration) = max_duration {
            self.config.detection.max_duration_hours = duration;
        }
        if let Some(confidence) = min_confidence {
            self.config.detection.min_confidence_threshold = confidence;
        }

        self.config.metadata.last_modified = chrono::Utc::now().to_rfc3339();
        Self::save_config(&self.config_path, &self.config)?;
        Ok(())
    }

    /// Enable/disable detection types
    #[allow(dead_code)]
    pub fn toggle_detection_type(&mut self, detection_type: &str, enabled: bool) -> Result<()> {
        match detection_type {
            "crypto_miners" => self.config.detection.enabled_detections.crypto_miners = enabled,
            "suspicious_processes" => {
                self.config
                    .detection
                    .enabled_detections
                    .suspicious_processes = enabled
            }
            "resource_abusers" => {
                self.config.detection.enabled_detections.resource_abusers = enabled
            }
            "data_exfiltrators" => {
                self.config.detection.enabled_detections.data_exfiltrators = enabled
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown detection type: {}",
                    detection_type
                ))
            }
        }

        self.config.metadata.last_modified = chrono::Utc::now().to_rfc3339();
        Self::save_config(&self.config_path, &self.config)?;
        Ok(())
    }

    /// Get configuration file path
    pub fn get_config_file_path(&self) -> &PathBuf {
        &self.config_path
    }

    /// Export configuration to JSON
    pub fn export_to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.config)
            .map_err(|e| anyhow::anyhow!("Failed to export config to JSON: {}", e))
    }

    /// Import configuration from JSON
    pub fn import_from_json(&mut self, json: &str) -> Result<()> {
        let config: RogueConfig = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("Failed to import config from JSON: {}", e))?;

        self.update_config(config)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_default_config() {
        let config = RogueConfig::default();
        assert!(config.detection.max_memory_usage_gb > 0.0);
        assert!(config.detection.max_utilization_pct > 0.0);
        assert!(!config.patterns.crypto_miner_patterns.is_empty());
    }

    #[test]
    fn test_config_serialization() {
        let config = RogueConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RogueConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            config.detection.max_memory_usage_gb,
            deserialized.detection.max_memory_usage_gb
        );
    }

    #[test]
    fn test_whitelist_case_insensitive_management() {
        let temp_dir = tempdir().unwrap();
        std::env::set_var("HOME", temp_dir.path());
        std::env::set_var("XDG_CONFIG_HOME", temp_dir.path());

        let mut manager = RogueConfigManager::new().unwrap();

        manager.add_user_to_whitelist("Root".to_string()).unwrap();
        manager.add_user_to_whitelist("root".to_string()).unwrap();
        manager.remove_user_from_whitelist("ROOT").unwrap();

        let users = &manager.get_config().patterns.user_whitelist;
        assert!(!users.iter().any(|u| u.eq_ignore_ascii_case("root")));

        manager
            .add_process_to_whitelist("Python".to_string())
            .unwrap();
        manager
            .add_process_to_whitelist("python".to_string())
            .unwrap();
        manager.remove_process_from_whitelist("PYTHON").unwrap();

        let processes = &manager.get_config().patterns.process_whitelist;
        assert!(!processes.iter().any(|p| p.eq_ignore_ascii_case("python")));
    }
}
