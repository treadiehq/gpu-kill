//! Hot Aisle API client for GPU instance provisioning and management
//!
//! This module provides integration with Hot Aisle's infrastructure
//! for on-demand GPU testing in CI/CD pipelines.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;

/// Hot Aisle API client for managing GPU instances
pub struct HotAisleClient {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

/// GPU instance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInstanceConfig {
    /// GPU type (nvidia, amd, intel, apple-silicon)
    pub gpu_type: String,
    /// Instance duration in minutes
    pub duration_minutes: u32,
    /// Instance size/type
    pub instance_type: Option<String>,
    /// Custom labels for the instance
    pub labels: Option<Vec<String>>,
}

/// GPU instance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInstance {
    /// Unique instance ID
    pub id: String,
    /// Instance IP address — absent while the instance is still provisioning.
    #[serde(default)]
    pub ip_address: Option<String>,
    /// SSH connection details — absent while the instance is still provisioning.
    #[serde(default)]
    pub ssh_config: Option<SshConfig>,
    /// GPU type
    pub gpu_type: String,
    /// Instance status
    pub status: String,
    /// Creation timestamp
    pub created_at: String,
    /// Expiration timestamp
    pub expires_at: String,
}

/// SSH connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshConfig {
    /// SSH username
    pub username: String,
    /// SSH port — defaults to 22 when the API omits the field.
    #[serde(default = "default_ssh_port")]
    pub port: u16,
    /// SSH key path or content
    pub key_path: Option<String>,
}

fn default_ssh_port() -> u16 {
    22
}

/// Test results from GPU instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTestResults {
    /// Instance ID where tests were run
    pub instance_id: String,
    /// Test execution status
    pub status: String,
    /// Test output/logs
    pub output: String,
    /// Test duration in seconds
    pub duration_seconds: u64,
    /// Number of tests passed
    pub tests_passed: u32,
    /// Number of tests failed
    pub tests_failed: u32,
    /// Number of tests skipped
    pub tests_skipped: u32,
}

impl HotAisleClient {
    /// Create a new Hot Aisle client
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let base_url = base_url.unwrap_or_else(|| "https://admin.hotaisle.app/api".to_string());

        Self {
            api_key,
            base_url,
            client: reqwest::Client::new(),
        }
    }

    /// Provision a new GPU instance
    pub async fn provision_gpu_instance(&self, config: GpuInstanceConfig) -> Result<GpuInstance> {
        let url = format!("{}/instances", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&config)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!(
                "Failed to provision GPU instance: {} - {}",
                status,
                error_text
            ));
        }

        let instance: GpuInstance = response.json().await?;
        Ok(instance)
    }

    /// Wait for instance to be ready
    pub async fn wait_for_instance_ready(
        &self,
        instance_id: &str,
        timeout_minutes: u32,
    ) -> Result<GpuInstance> {
        let timeout = Duration::from_secs(timeout_minutes as u64 * 60);
        let start = std::time::Instant::now();

        while start.elapsed() < timeout {
            let instance = self.get_instance(instance_id).await?;

            match instance.status.as_str() {
                "ready" | "running" => return Ok(instance),
                "failed" | "error" => {
                    return Err(anyhow::anyhow!("Instance {} failed to start", instance_id));
                }
                _ => {
                    // Still provisioning, wait and retry
                    sleep(Duration::from_secs(10)).await;
                }
            }
        }

        Err(anyhow::anyhow!(
            "Instance {} did not become ready within {} minutes",
            instance_id,
            timeout_minutes
        ))
    }

    /// Get instance information
    pub async fn get_instance(&self, instance_id: &str) -> Result<GpuInstance> {
        let url = format!("{}/instances/{}", self.base_url, instance_id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!(
                "Failed to get instance {}: {} - {}",
                instance_id,
                status,
                error_text
            ));
        }

        let instance: GpuInstance = response.json().await?;
        Ok(instance)
    }

    /// Execute GPU tests on an instance
    pub async fn run_gpu_tests(
        &self,
        instance: &GpuInstance,
        test_config: &GpuTestConfig,
    ) -> Result<GpuTestResults> {
        let url = format!("{}/instances/{}/execute", self.base_url, instance.id);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(test_config)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!(
                "Failed to run tests on instance {}: {} - {}",
                instance.id,
                status,
                error_text
            ));
        }

        let results: GpuTestResults = response.json().await?;
        Ok(results)
    }

    /// Terminate an instance
    pub async fn terminate_instance(&self, instance_id: &str) -> Result<()> {
        let url = format!("{}/instances/{}", self.base_url, instance_id);

        let response = self
            .client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!(
                "Failed to terminate instance {}: {} - {}",
                instance_id,
                status,
                error_text
            ));
        }

        Ok(())
    }

    /// List available GPU types
    pub async fn list_available_gpu_types(&self) -> Result<Vec<String>> {
        let url = format!("{}/gpu-types", self.base_url);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!(
                "Failed to list GPU types: {} - {}",
                status,
                error_text
            ));
        }

        let gpu_types: Vec<String> = response.json().await?;
        Ok(gpu_types)
    }
}

/// GPU test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTestConfig {
    /// Test command to execute
    pub test_command: String,
    /// Test timeout in minutes
    pub timeout_minutes: u32,
    /// Environment variables
    pub env_vars: Option<std::collections::HashMap<String, String>>,
    /// Working directory
    pub working_dir: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hotaisle_client_creation() {
        let client = HotAisleClient::new("test-key".to_string(), None);
        assert_eq!(client.base_url, "https://admin.hotaisle.app/api");
    }

    #[tokio::test]
    async fn test_gpu_instance_config() {
        let config = GpuInstanceConfig {
            gpu_type: "nvidia".to_string(),
            duration_minutes: 30,
            instance_type: Some("g4dn.xlarge".to_string()),
            labels: Some(vec!["ci-test".to_string(), "gpu-kill".to_string()]),
        };

        assert_eq!(config.gpu_type, "nvidia");
        assert_eq!(config.duration_minutes, 30);
    }

    #[test]
    fn test_ssh_config_defaults_port_when_missing() {
        let json = r#"{"username": "root", "key_path": "/path/to/key"}"#;
        let config: SshConfig = serde_json::from_str(json)
            .expect("SshConfig should deserialize with port defaulting to 22");
        assert_eq!(config.port, 22);
        assert_eq!(config.username, "root");
    }

    #[test]
    fn test_gpu_instance_deserializes_without_ip_or_ssh() {
        let json = r#"{
            "id": "inst-123",
            "gpu_type": "nvidia",
            "status": "provisioning",
            "created_at": "2023-01-01T00:00:00Z",
            "expires_at": "2023-01-01T01:00:00Z"
        }"#;
        let instance: GpuInstance = serde_json::from_str(json).expect(
            "GpuInstance should deserialize even when ip_address and ssh_config are absent",
        );
        assert_eq!(instance.status, "provisioning");
        assert!(instance.ip_address.is_none());
        assert!(instance.ssh_config.is_none());
    }

    #[test]
    fn test_gpu_instance_deserializes_fully_when_ready() {
        let json = r#"{
            "id": "inst-456",
            "ip_address": "10.0.0.1",
            "ssh_config": {"username": "ubuntu", "port": 22},
            "gpu_type": "nvidia",
            "status": "ready",
            "created_at": "2023-01-01T00:00:00Z",
            "expires_at": "2023-01-01T01:00:00Z"
        }"#;
        let instance: GpuInstance = serde_json::from_str(json)
            .expect("GpuInstance should deserialize with all fields present");
        assert_eq!(instance.ip_address.as_deref(), Some("10.0.0.1"));
        assert_eq!(instance.ssh_config.as_ref().map(|s| s.port), Some(22));
    }
}
