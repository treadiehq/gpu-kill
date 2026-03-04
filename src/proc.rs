use crate::nvml_api::NvmlApi;
use crate::util::parse_process_start_time;
use anyhow::{Context, Result};
#[cfg(unix)]
use nix::sys::signal::{kill, Signal};
#[cfg(unix)]
use nix::unistd::Pid;
// use std::process::Command; // Used conditionally below
use std::time::{Duration, Instant, SystemTime};
use sysinfo::{Pid as SysPid, System};

/// Process information for a running process
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    #[allow(dead_code)]
    pub pid: u32,
    pub user: String,
    pub name: String,
    #[allow(dead_code)]
    pub start_time: SystemTime,
    #[allow(dead_code)]
    pub cmdline: String,
}

/// Process management utilities
pub struct ProcessManager {
    nvml_api: NvmlApi,
    system: System,
}

#[allow(dead_code)]
impl ProcessManager {
    /// Create a new process manager
    pub fn new(nvml_api: NvmlApi) -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        Self { nvml_api, system }
    }

    /// Get process information by PID
    pub fn get_process_info(&mut self, pid: u32) -> Result<ProcessInfo> {
        self.system.refresh_processes();

        let sys_pid = SysPid::from_u32(pid);
        let process = self
            .system
            .process(sys_pid)
            .ok_or_else(|| anyhow::anyhow!("Process with PID {} not found", pid))?;

        let user = get_process_user(pid).unwrap_or_else(|_| "unknown".to_string());

        let start_time = process.start_time();
        let start_time_system = SystemTime::UNIX_EPOCH + Duration::from_secs(start_time);

        Ok(ProcessInfo {
            pid,
            user,
            name: process.name().to_string(),
            start_time: start_time_system,
            cmdline: process.cmd().join(" "),
        })
    }

    /// Check if a process is using any GPU
    pub fn is_process_using_gpu(&self, pid: u32) -> Result<bool> {
        self.nvml_api.is_process_using_gpu(pid)
    }

    /// Gracefully terminate a process with timeout and escalation.
    ///
    /// Captures the process start time before sending any signal, then verifies
    /// on every poll that the PID still belongs to the same process (guards against
    /// PID reuse races). Uses `Instant` for a monotonic timeout so system clock
    /// adjustments cannot cause premature exit or runaway loops.
    #[cfg(unix)]
    pub fn graceful_kill(&mut self, pid: u32, timeout_secs: u16, force: bool) -> Result<()> {
        let nix_pid = Pid::from_raw(pid as i32);

        // Snapshot the start time of the target process before sending any signal.
        // If we cannot find it, it has already exited — nothing to do.
        let original_start = self.get_process_start_secs(pid);
        if original_start.is_none() {
            tracing::info!("Process {} is already gone before SIGTERM", pid);
            return Ok(());
        }

        tracing::info!("Sending SIGTERM to process {}", pid);
        kill(nix_pid, Signal::SIGTERM)
            .map_err(|e| anyhow::anyhow!("Failed to send SIGTERM: {}", e))?;

        // Use Instant so the timeout is unaffected by wall-clock adjustments.
        let timeout = Duration::from_secs(timeout_secs as u64);
        let deadline = Instant::now() + timeout;

        while Instant::now() < deadline {
            if !self.is_same_process_running(pid, original_start)? {
                tracing::info!("Process {} terminated gracefully", pid);
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        if force {
            tracing::warn!("Process {} did not terminate, escalating to SIGKILL", pid);
            kill(nix_pid, Signal::SIGKILL)
                .map_err(|e| anyhow::anyhow!("Failed to send SIGKILL: {}", e))?;

            std::thread::sleep(Duration::from_millis(500));

            if !self.is_same_process_running(pid, original_start)? {
                tracing::info!("Process {} terminated with SIGKILL", pid);
                Ok(())
            } else {
                // A zombie will never pass is_same_process_running (its status differs),
                // but report clearly if somehow still present.
                Err(anyhow::anyhow!(
                    "Process {} still running after SIGKILL",
                    pid
                ))
            }
        } else {
            Err(anyhow::anyhow!(
                "Process {} did not terminate within {} seconds. Use --force to escalate to SIGKILL",
                pid,
                timeout_secs
            ))
        }
    }

    /// Gracefully terminate a process with timeout and escalation (Windows stub)
    #[cfg(windows)]
    pub fn graceful_kill(&mut self, _pid: u32, _timeout_secs: u16, _force: bool) -> Result<()> {
        Err(anyhow::anyhow!(
            "Process termination not yet implemented for Windows"
        ))
    }

    /// Return the start time (seconds since UNIX epoch) of the process at `pid`,
    /// or `None` if the process does not exist or is a zombie.
    fn get_process_start_secs(&mut self, pid: u32) -> Option<u64> {
        let sys_pid = SysPid::from_u32(pid);
        if !self.system.refresh_process(sys_pid) {
            return None;
        }
        let proc = self.system.process(sys_pid)?;
        // Treat zombies as gone — they have already released their resources.
        #[cfg(unix)]
        if proc.status() == sysinfo::ProcessStatus::Zombie {
            return None;
        }
        Some(proc.start_time())
    }

    /// Check whether the process at `pid` is still the *same* process that was
    /// originally targeted (identified by its start time).
    ///
    /// Returns `false` (meaning "the original process is gone") when:
    /// - the PID no longer exists,
    /// - the PID has been reused by a different process (different start time), or
    /// - the process is a zombie (resources already released).
    fn is_same_process_running(&mut self, pid: u32, original_start: Option<u64>) -> Result<bool> {
        let current_start = self.get_process_start_secs(pid);
        // If either is None, or the start times differ, the original process is gone.
        Ok(current_start.is_some() && current_start == original_start)
    }

    /// Check if a process is still running (identity-unaware, kept for external callers).
    ///
    /// Prefers `is_same_process_running` internally; this variant is left for
    /// `validate_process` and other callers that do not need start-time anchoring.
    fn is_process_running(&mut self, pid: u32) -> Result<bool> {
        let sys_pid = SysPid::from_u32(pid);
        Ok(self.system.refresh_process(sys_pid))
    }

    /// Enrich GPU processes with system information.
    ///
    /// Does a single `refresh_processes` up front, then looks up each PID in the
    /// already-populated cache — O(N) instead of O(N·M).
    pub fn enrich_gpu_processes(
        &mut self,
        mut processes: Vec<crate::nvml_api::GpuProc>,
    ) -> Result<Vec<crate::nvml_api::GpuProc>> {
        self.system.refresh_processes();

        for process in &mut processes {
            let sys_pid = SysPid::from_u32(process.pid);
            if let Some(proc) = self.system.process(sys_pid) {
                process.proc_name = proc.name().to_string();
                let start = SystemTime::UNIX_EPOCH + Duration::from_secs(proc.start_time());
                process.start_time = parse_process_start_time(start);
                // User resolution is best-effort; fall back to existing value on error.
                if let Ok(user) = get_process_user(process.pid) {
                    process.user = user;
                }
            }
        }

        Ok(processes)
    }

    /// Get all processes using GPUs with enriched information
    pub fn get_enriched_gpu_processes(&mut self) -> Result<Vec<crate::nvml_api::GpuProc>> {
        let processes = self.nvml_api.get_gpu_processes()?;
        self.enrich_gpu_processes(processes)
    }

    /// Validate that a process exists and optionally check GPU usage
    pub fn validate_process(&self, pid: u32, check_gpu_usage: bool) -> Result<()> {
        // Check if process exists
        let sys_pid = SysPid::from_u32(pid);
        if self.system.process(sys_pid).is_none() {
            return Err(anyhow::anyhow!("Process with PID {} not found", pid));
        }

        // Check GPU usage if requested
        if check_gpu_usage {
            let is_using_gpu = self.is_process_using_gpu(pid)?;
            if !is_using_gpu {
                return Err(anyhow::anyhow!(
                    "Process {} is not using any GPU. Use --force to kill anyway.",
                    pid
                ));
            }
        }

        Ok(())
    }

    /// Get device count
    pub fn device_count(&self) -> Result<u32> {
        self.nvml_api.device_count()
    }

    /// Create snapshot
    pub fn create_snapshot(&self) -> Result<crate::nvml_api::Snapshot> {
        self.nvml_api.create_snapshot()
    }

    /// Reset GPU
    pub fn reset_gpu(&self, index: u32) -> Result<()> {
        self.nvml_api.reset_gpu(index)
    }
}

/// Get the username for a process (cross-platform)
fn get_process_user(pid: u32) -> Result<String> {
    #[cfg(target_os = "linux")]
    {
        // On Linux, read from /proc/<pid>/status
        let status_path = format!("/proc/{}/status", pid);
        let status = std::fs::read_to_string(&status_path)
            .with_context(|| format!("Failed to read process status from {}", status_path))?;

        for line in status.lines() {
            if line.starts_with("Uid:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let uid = parts[1]
                        .parse::<u32>()
                        .with_context(|| format!("Failed to parse UID: {}", parts[1]))?;

                    // Get username from UID
                    return get_username_from_uid(uid);
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        // On macOS, use ps command
        let output = Command::new("ps")
            .args(["-o", "user=", "-p", &pid.to_string()])
            .output()
            .context("Failed to execute ps command")?;

        if output.status.success() {
            let user = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !user.is_empty() {
                return Ok(user);
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        // On Windows, use wmic command
        let output = Command::new("wmic")
            .args([
                "process",
                "where",
                &format!("ProcessId={}", pid),
                "get",
                "ExecutablePath",
                "/format:value",
            ])
            .output()
            .context("Failed to execute wmic command")?;

        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.starts_with("ExecutablePath=") {
                    let path = line.strip_prefix("ExecutablePath=").unwrap_or("");
                    if !path.is_empty() {
                        // Extract username from path or use a default
                        return Ok("windows_user".to_string());
                    }
                }
            }
        }
    }

    Ok("unknown".to_string())
}

#[cfg(target_os = "linux")]
fn get_username_from_uid(uid: u32) -> Result<String> {
    use std::ffi::CStr;

    unsafe {
        let passwd = libc::getpwuid(uid as libc::uid_t);
        if passwd.is_null() {
            return Ok(format!("uid_{}", uid));
        }

        let username = CStr::from_ptr((*passwd).pw_name);
        Ok(username.to_string_lossy().to_string())
    }
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn get_username_from_uid(_uid: u32) -> Result<String> {
    Ok("unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvml_api::NvmlApi;

    #[test]
    fn test_process_info_creation() {
        // Skip this test if NVML is not available
        let nvml_api = match NvmlApi::new() {
            Ok(api) => api,
            Err(_) => {
                // Skip test if NVML is not available
                return;
            }
        };

        let mut proc_mgr = ProcessManager::new(nvml_api);

        // Test with a known process (init/systemd)
        if let Ok(info) = proc_mgr.get_process_info(1) {
            assert_eq!(info.pid, 1);
            assert!(!info.name.is_empty());
        }
    }

    #[test]
    fn test_process_validation() {
        // Skip this test if NVML is not available
        let nvml_api = match NvmlApi::new() {
            Ok(api) => api,
            Err(_) => {
                // Skip test if NVML is not available
                return;
            }
        };

        let proc_mgr = ProcessManager::new(nvml_api);

        // Test validation of non-existent process
        let result = proc_mgr.validate_process(999999, false);
        assert!(result.is_err());
    }
}
