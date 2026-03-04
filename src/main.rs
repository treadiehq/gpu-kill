use crate::args::{Cli, OutputFormat, VendorFilter};
use crate::config::get_config;
use crate::coordinator::{create_router, CoordinatorState};
use crate::nvml_api::{NvmlApi, Snapshot};
use crate::proc::ProcessManager;
use crate::process_mgmt::EnhancedProcessManager;
use crate::render::{render_error, render_info, render_success, render_warning, Renderer};
use crate::vendor::GpuManager;
use crate::version::get_version_string;
use anyhow::{Context, Result};
use std::process;
use std::time::Duration;
use tracing::{debug, error, info, warn};

mod args;
mod audit;
mod config;
mod coordinator;
mod guard_mode;
mod nvml_api;
mod proc;
mod process_mgmt;
mod remote;
mod render;
mod rogue_config;
mod rogue_detection;
mod util;
mod vendor;
mod version;

fn main() -> Result<()> {
    // Initialize error handling
    color_eyre::install().map_err(|e| anyhow::anyhow!("Failed to install error handler: {}", e))?;

    // Parse command line arguments
    let cli = Cli::parse();

    // Initialize logging
    init_logging(&cli.log_level.to_string())?;

    // Load configuration
    let config_manager = get_config(cli.config.clone()).context("Failed to load configuration")?;

    info!("Starting gpukill {}", get_version_string());

    // Execute the requested operation
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create tokio runtime: {}", e))?;
    match rt.block_on(execute_operation(cli, config_manager)) {
        Ok(()) => {
            info!("Operation completed successfully");
            Ok(())
        }
        Err(e) => {
            error!("Operation failed: {}", e);
            render_error(&e.to_string());

            // Set appropriate exit codes
            let exit_code = if e.to_string().contains("NVML") {
                2 // NVML initialization failure
            } else if e.to_string().contains("Invalid argument") {
                3 // Invalid arguments
            } else if e.to_string().contains("permission") || e.to_string().contains("Permission") {
                4 // Permission errors
            } else if e.to_string().contains("not supported")
                || e.to_string().contains("unsupported")
            {
                5 // Operation not supported
            } else {
                1 // General error
            };

            process::exit(exit_code);
        }
    }
}

/// Initialize logging system
fn init_logging(log_level: &str) -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .init();

    Ok(())
}

/// Execute the requested operation
async fn execute_operation(cli: Cli, config_manager: crate::config::ConfigManager) -> Result<()> {
    // Check if this is a remote operation
    if let Some(remote_host) = cli.remote.clone() {
        return execute_remote_operation(cli, &remote_host).await;
    }

    // Initialize GPU manager for local operations
    let gpu_manager = GpuManager::initialize().context("Failed to initialize GPU manager")?;

    if cli.list {
        execute_list_operation(
            cli.details,
            cli.watch,
            cli.output,
            cli.vendor,
            cli.containers,
            gpu_manager,
            config_manager,
        )
        .await
    } else if cli.kill {
        execute_kill_operation(
            cli.pid,
            cli.timeout_secs,
            cli.force,
            cli.filter,
            cli.batch,
            cli.gpu,
            cli.dry_run,
            gpu_manager,
            config_manager,
        )
    } else if cli.reset {
        execute_reset_operation(
            cli.gpu,
            cli.all,
            cli.force,
            cli.dry_run,
            gpu_manager,
            config_manager,
        )
    } else if cli.audit {
        execute_audit_operation(
            cli.audit_user.clone(),
            cli.audit_process.clone(),
            cli.audit_hours,
            cli.audit_summary,
            cli.rogue,
            &cli,
            cli.output.clone(),
        )
        .await
    } else if cli.server {
        let host = cli.server_host.clone();
        let port = cli.server_port;
        if cli.open {
            // Spawn server so we can open the browser once it is listening (instead of blocking forever)
            let server_handle =
                tokio::spawn(
                    async move { execute_server_operation(host, port, gpu_manager).await },
                );
            tokio::time::sleep(Duration::from_millis(500)).await;
            open_browser_at_port(port);
            server_handle
                .await
                .context("Server task panicked")?
                .context("Server exited with error")?;
        } else {
            execute_server_operation(host, port, gpu_manager).await?;
        }
        Ok(())
    } else if cli.guard {
        execute_guard_operation(&cli, gpu_manager).await
    } else if let Some(coordinator_url) = cli.register_node {
        execute_register_node_operation(coordinator_url, gpu_manager).await
    } else {
        Err(anyhow::anyhow!("No operation specified"))
    }
}

/// Execute list operation
async fn execute_list_operation(
    details: bool,
    watch: bool,
    output: OutputFormat,
    vendor_filter: Option<VendorFilter>,
    containers: bool,
    gpu_manager: GpuManager,
    config_manager: crate::config::ConfigManager,
) -> Result<()> {
    let renderer = Renderer::new(output);

    if watch {
        execute_watch_mode(
            details,
            containers,
            vendor_filter,
            renderer,
            gpu_manager,
            config_manager,
        )
        .await
    } else {
        execute_single_list(details, containers, &vendor_filter, &renderer, &gpu_manager).await
    }
}

/// Execute single list operation
async fn execute_single_list(
    details: bool,
    containers: bool,
    vendor_filter: &Option<VendorFilter>,
    renderer: &Renderer,
    gpu_manager: &GpuManager,
) -> Result<()> {
    // Get all GPU snapshots
    let mut gpus = gpu_manager.get_all_snapshots()?;

    // Filter by vendor if specified
    if let Some(filter) = vendor_filter {
        if let Some(target_vendor) = filter.to_gpu_vendor() {
            gpus.retain(|gpu| gpu.vendor == target_vendor);
        }
    }

    // Get all processes
    let mut procs = gpu_manager.get_all_processes()?;

    // Enrich with container information if requested (uses sysinfo; NVML not required)
    if containers {
        match NvmlApi::new() {
            Ok(nvml_api) => {
                let proc_manager = ProcessManager::new(nvml_api);
                let mut enhanced_manager = EnhancedProcessManager::new(proc_manager);
                procs = enhanced_manager.enrich_with_containers(procs)?;
            }
            Err(e) => {
                tracing::warn!(
                    "Skipping container enrichment: NVML unavailable ({}). Container names will not be shown.",
                    e
                );
            }
        }
    }

    // Create snapshot for rendering
    let snapshot = Snapshot {
        host: crate::util::get_hostname(),
        ts: crate::util::get_current_timestamp_iso(),
        gpus: gpus.clone(),
        procs: procs.clone(),
    };

    // Log to audit database (async)
    // Now that execute_single_list is async, we can directly log to audit
    match crate::audit::AuditManager::new().await {
        Ok(audit_manager) => match audit_manager.log_snapshot(&gpus, &procs).await {
            Ok(()) => {
                tracing::debug!(
                    "Successfully logged audit snapshot with {} GPUs and {} processes",
                    gpus.len(),
                    procs.len()
                );
            }
            Err(e) => {
                tracing::warn!("Failed to log audit snapshot: {}", e);
            }
        },
        Err(e) => {
            tracing::warn!("Failed to initialize audit manager: {}", e);
        }
    }

    renderer
        .render_snapshot(&snapshot, details)
        .map_err(|e| anyhow::anyhow!("Render error: {}", e))?;
    Ok(())
}

/// Execute watch mode
async fn execute_watch_mode(
    details: bool,
    containers: bool,
    vendor_filter: Option<VendorFilter>,
    renderer: Renderer,
    gpu_manager: GpuManager,
    config_manager: crate::config::ConfigManager,
) -> Result<()> {
    let _interval = Duration::from_secs(config_manager.config().watch_interval_secs);

    info!(
        "Starting watch mode (refresh every {}s). Press Ctrl-C to stop.",
        config_manager.config().watch_interval_secs
    );

    loop {
        // Clear screen BEFORE rendering new data so users see the data
        // during the entire sleep interval (matches standard `watch` behavior)
        if matches!(renderer.get_output_format(), OutputFormat::Table) {
            renderer.clear_screen();
        }

        match execute_single_list(details, containers, &vendor_filter, &renderer, &gpu_manager)
            .await
        {
            Ok(()) => {
                // Data is now visible during the entire sleep interval
            }
            Err(e) => {
                warn!("Failed to refresh data: {}", e);
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(
            config_manager.config().watch_interval_secs,
        ))
        .await;
    }
}

/// Execute kill operation
#[allow(clippy::too_many_arguments)]
fn execute_kill_operation(
    pid: Option<u32>,
    timeout_secs: u16,
    force: bool,
    filter: Option<String>,
    batch: bool,
    gpu_id: Option<u16>,
    dry_run: bool,
    gpu_manager: GpuManager,
    _config_manager: crate::config::ConfigManager,
) -> Result<()> {
    // Initialize process manager for enhanced operations
    let nvml_api = match NvmlApi::new() {
        Ok(api) => api,
        Err(e) => {
            // Friendlier error when non-NVIDIA vendors are present
            let available_vendors = gpu_manager.get_vendors();
            if !available_vendors.is_empty()
                && !available_vendors.contains(&crate::vendor::GpuVendor::Nvidia)
            {
                return Err(anyhow::anyhow!(
                    "Kill operations currently require NVIDIA/NVML. Detected vendors: {:?}. Use --list/watch/audit, or run on a NVIDIA node.",
                    available_vendors
                ));
            }
            return Err(anyhow::anyhow!(
                "Failed to initialize NVML. Ensure NVIDIA drivers are installed and GPU is accessible. ({})",
                e
            ));
        }
    };
    let proc_manager = ProcessManager::new(nvml_api);
    let mut enhanced_manager = EnhancedProcessManager::new(proc_manager);

    if let Some(filter_pattern) = filter {
        // Batch kill based on filter
        let all_processes = gpu_manager.get_all_processes()?;
        let filtered_processes =
            enhanced_manager.filter_processes_by_name(&all_processes, &filter_pattern)?;

        if filtered_processes.is_empty() {
            render_warning(&format!(
                "No processes found matching pattern: {}",
                filter_pattern
            ));
            return Ok(());
        }

        render_info(&format!(
            "Found {} processes matching pattern '{}'",
            filtered_processes.len(),
            filter_pattern
        ));

        if batch {
            let killed_pids = if dry_run {
                // Preview only
                render_info("Dry-run: would kill the following processes:");
                for p in &filtered_processes {
                    render_info(&format!(
                        "  PID {}: {} ({}) - {} MB",
                        p.pid, p.proc_name, p.user, p.used_mem_mb
                    ));
                }
                Vec::new()
            } else {
                enhanced_manager.batch_kill_processes(&filtered_processes, timeout_secs, force)?
            };
            render_success(&format!(
                "Successfully killed {} processes: {:?}",
                killed_pids.len(),
                killed_pids
            ));
        } else {
            // Show processes and ask for confirmation (for now, just show them)
            for proc in &filtered_processes {
                render_info(&format!(
                    "  PID {}: {} ({}) - {} MB",
                    proc.pid, proc.proc_name, proc.user, proc.used_mem_mb
                ));
            }
            render_warning("Use --batch flag to actually kill these processes");
        }
    } else if let Some(target_pid) = pid {
        // Single process kill
        let check_gpu_usage = !force;
        enhanced_manager
            .process_manager
            .validate_process(target_pid, check_gpu_usage)?;

        // Get process info for display
        let process_info = enhanced_manager
            .process_manager
            .get_process_info(target_pid)?;
        render_info(&format!(
            "Terminating process {} ({}: {})",
            target_pid, process_info.user, process_info.name
        ));

        if dry_run {
            render_info(&format!(
                "Dry-run: would terminate process {} (timeout {}s, force: {})",
                target_pid, timeout_secs, force
            ));
        } else {
            // Perform graceful kill
            enhanced_manager
                .process_manager
                .graceful_kill(target_pid, timeout_secs, force)?;
            render_success(&format!("Process {} terminated successfully", target_pid));
        }
    } else if let Some(target_gpu) = gpu_id {
        // Kill all processes on a specific GPU
        let all_processes = gpu_manager.get_all_processes()?;
        let gpu_processes: Vec<_> = all_processes
            .into_iter()
            .filter(|p| p.gpu_index == target_gpu)
            .collect();

        if gpu_processes.is_empty() {
            render_warning(&format!("No processes found on GPU {}", target_gpu));
            return Ok(());
        }

        render_info(&format!(
            "Found {} processes on GPU {}",
            gpu_processes.len(),
            target_gpu
        ));

        if dry_run {
            render_info("Dry-run: would kill the following processes:");
            for p in &gpu_processes {
                render_info(&format!(
                    "  PID {}: {} ({}) - {} MB",
                    p.pid, p.proc_name, p.user, p.used_mem_mb
                ));
            }
            return Ok(());
        }

        if !batch {
            render_warning("Use --batch to confirm killing all processes on this GPU");
            for p in &gpu_processes {
                render_info(&format!(
                    "  PID {}: {} ({}) - {} MB",
                    p.pid, p.proc_name, p.user, p.used_mem_mb
                ));
            }
            return Ok(());
        }

        let killed_pids =
            enhanced_manager.batch_kill_processes(&gpu_processes, timeout_secs, force)?;
        render_success(&format!(
            "Successfully killed {} processes on GPU {}: {:?}",
            killed_pids.len(),
            target_gpu,
            killed_pids
        ));
    } else {
        return Err(anyhow::anyhow!(
            "Either --pid, --filter, or --gpu must be specified"
        ));
    }

    Ok(())
}

/// Execute reset operation
fn execute_reset_operation(
    gpu: Option<u16>,
    all: bool,
    force: bool,
    dry_run: bool,
    gpu_manager: GpuManager,
    _config_manager: crate::config::ConfigManager,
) -> Result<()> {
    if all {
        execute_reset_all_gpus(&gpu_manager, force, dry_run)
    } else if let Some(gpu_id) = gpu {
        execute_reset_single_gpu(&gpu_manager, gpu_id, force, dry_run)
    } else {
        Err(anyhow::anyhow!("No GPU specified for reset operation"))
    }
}

/// Execute reset for all GPUs
fn execute_reset_all_gpus(gpu_manager: &GpuManager, force: bool, dry_run: bool) -> Result<()> {
    let device_count = gpu_manager.total_device_count()?;

    if device_count == 0 {
        return Err(anyhow::anyhow!("No GPUs found"));
    }

    if dry_run {
        render_info(&format!("Dry-run: would reset all {} GPUs", device_count));
        return Ok(());
    } else {
        render_info(&format!("Resetting all {} GPUs", device_count));
    }

    // Check for active processes if not forcing
    if !force {
        let active_processes = gpu_manager.get_all_processes()?;

        if !active_processes.is_empty() {
            render_warning("Active GPU processes found:");
            for proc in &active_processes {
                render_warning(&format!(
                    "  GPU {}: PID {} ({})",
                    proc.gpu_index, proc.pid, proc.proc_name
                ));
            }
            return Err(anyhow::anyhow!(
                "Cannot reset GPUs with active processes. Use --force to override."
            ));
        }
    }

    // Reset each GPU
    for i in 0..device_count {
        match gpu_manager.reset_gpu(i) {
            Ok(()) => {
                render_success(&format!("GPU {} reset successfully", i));
            }
            Err(e) => {
                render_error(&format!("Failed to reset GPU {}: {}", i, e));
            }
        }
    }

    Ok(())
}

/// Execute reset for a single GPU
fn execute_reset_single_gpu(
    gpu_manager: &GpuManager,
    gpu_id: u16,
    force: bool,
    dry_run: bool,
) -> Result<()> {
    let device_count = gpu_manager.total_device_count()?;

    if gpu_id as u32 >= device_count {
        return Err(anyhow::anyhow!(
            "GPU {} not found. Available GPUs: 0-{}",
            gpu_id,
            device_count - 1
        ));
    }

    if dry_run {
        render_info(&format!("Dry-run: would reset GPU {}", gpu_id));
        return Ok(());
    } else {
        render_info(&format!("Resetting GPU {}", gpu_id));
    }

    // Check for active processes on this GPU if not forcing
    if !force {
        let all_processes = gpu_manager.get_all_processes()?;
        let gpu_processes: Vec<_> = all_processes
            .iter()
            .filter(|p| p.gpu_index == gpu_id)
            .collect();

        if !gpu_processes.is_empty() {
            render_warning(&format!("Active processes found on GPU {}:", gpu_id));
            for proc in &gpu_processes {
                render_warning(&format!("  PID {} ({})", proc.pid, proc.proc_name));
            }
            return Err(anyhow::anyhow!(
                "Cannot reset GPU {} with active processes. Use --force to override.",
                gpu_id
            ));
        }
    }

    // Reset the GPU
    gpu_manager.reset_gpu(gpu_id as u32)?;
    render_success(&format!("GPU {} reset successfully", gpu_id));

    Ok(())
}

/// Execute audit operation
async fn execute_audit_operation(
    user_filter: Option<String>,
    process_filter: Option<String>,
    hours: u32,
    summary: bool,
    rogue: bool,
    cli: &crate::args::Cli,
    output_format: crate::args::OutputFormat,
) -> Result<()> {
    use crate::audit::AuditManager;
    use crate::render::{render_info, render_warning};

    // Initialize audit manager
    let audit_manager = AuditManager::new()
        .await
        .context("Failed to initialize audit manager")?;

    // Handle configuration management
    if cli.rogue_config
        || cli.rogue_memory_threshold.is_some()
        || cli.rogue_utilization_threshold.is_some()
        || cli.rogue_duration_threshold.is_some()
        || cli.rogue_confidence_threshold.is_some()
        || cli.rogue_whitelist_process.is_some()
        || cli.rogue_unwhitelist_process.is_some()
        || cli.rogue_whitelist_user.is_some()
        || cli.rogue_unwhitelist_user.is_some()
        || cli.rogue_export_config
        || cli.rogue_import_config.is_some()
    {
        use crate::rogue_config::RogueConfigManager;

        let mut config_manager =
            RogueConfigManager::new().context("Failed to initialize rogue config manager")?;

        // Show current configuration
        if cli.rogue_config {
            let config = config_manager.get_config();
            if output_format == crate::args::OutputFormat::Json {
                let json = config_manager
                    .export_to_json()
                    .context("Failed to export config to JSON")?;
                println!("{}", json);
            } else {
                render_info("🕵️ Rogue Detection Configuration:");
                render_info(&format!(
                    "  Memory Threshold: {:.1} GB",
                    config.detection.max_memory_usage_gb
                ));
                render_info(&format!(
                    "  Utilization Threshold: {:.1}%",
                    config.detection.max_utilization_pct
                ));
                render_info(&format!(
                    "  Duration Threshold: {:.1} hours",
                    config.detection.max_duration_hours
                ));
                render_info(&format!(
                    "  Confidence Threshold: {:.2}",
                    config.detection.min_confidence_threshold
                ));
                render_info(&format!(
                    "  Crypto Miners: {}",
                    if config.detection.enabled_detections.crypto_miners {
                        "enabled"
                    } else {
                        "disabled"
                    }
                ));
                render_info(&format!(
                    "  Suspicious Processes: {}",
                    if config.detection.enabled_detections.suspicious_processes {
                        "enabled"
                    } else {
                        "disabled"
                    }
                ));
                render_info(&format!(
                    "  Resource Abusers: {}",
                    if config.detection.enabled_detections.resource_abusers {
                        "enabled"
                    } else {
                        "disabled"
                    }
                ));
                render_info(&format!(
                    "  Data Exfiltrators: {}",
                    if config.detection.enabled_detections.data_exfiltrators {
                        "enabled"
                    } else {
                        "disabled"
                    }
                ));

                render_info("\n📋 Whitelisted Users:");
                for user in &config.patterns.user_whitelist {
                    render_info(&format!("  - {}", user));
                }

                render_info("\n📋 Whitelisted Processes:");
                for process in &config.patterns.process_whitelist {
                    render_info(&format!("  - {}", process));
                }

                render_info(&format!(
                    "\n📁 Config file: {}",
                    config_manager.get_config_file_path().display()
                ));
            }
        }

        // Update thresholds
        if cli.rogue_memory_threshold.is_some()
            || cli.rogue_utilization_threshold.is_some()
            || cli.rogue_duration_threshold.is_some()
            || cli.rogue_confidence_threshold.is_some()
        {
            config_manager
                .update_thresholds(
                    cli.rogue_memory_threshold,
                    cli.rogue_utilization_threshold,
                    cli.rogue_duration_threshold,
                    cli.rogue_confidence_threshold,
                )
                .context("Failed to update thresholds")?;

            render_info("✅ Rogue detection thresholds updated successfully");
        }

        // Manage whitelists
        if let Some(process) = &cli.rogue_whitelist_process {
            config_manager
                .add_process_to_whitelist(process.clone())
                .context("Failed to add process to whitelist")?;
            render_info(&format!("✅ Added '{}' to process whitelist", process));
        }

        if let Some(process) = &cli.rogue_unwhitelist_process {
            config_manager
                .remove_process_from_whitelist(process)
                .context("Failed to remove process from whitelist")?;
            render_info(&format!("✅ Removed '{}' from process whitelist", process));
        }

        if let Some(user) = &cli.rogue_whitelist_user {
            config_manager
                .add_user_to_whitelist(user.clone())
                .context("Failed to add user to whitelist")?;
            render_info(&format!("✅ Added '{}' to user whitelist", user));
        }

        if let Some(user) = &cli.rogue_unwhitelist_user {
            config_manager
                .remove_user_from_whitelist(user)
                .context("Failed to remove user from whitelist")?;
            render_info(&format!("✅ Removed '{}' from user whitelist", user));
        }

        // Export configuration
        if cli.rogue_export_config {
            let json = config_manager
                .export_to_json()
                .context("Failed to export config to JSON")?;
            println!("{}", json);
        }

        // Import configuration
        if let Some(file_path) = &cli.rogue_import_config {
            let content =
                std::fs::read_to_string(file_path).context("Failed to read import file")?;
            config_manager
                .import_from_json(&content)
                .context("Failed to import config from JSON")?;
            render_info(&format!("✅ Imported configuration from: {}", file_path));
        }

        return Ok(());
    }

    if rogue {
        // Perform rogue activity detection
        use crate::rogue_config::RogueConfigManager;
        use crate::rogue_detection::RogueDetector;

        let config_manager =
            RogueConfigManager::new().context("Failed to initialize rogue config manager")?;

        let detector = RogueDetector::with_config(audit_manager, &config_manager);
        let result = detector
            .detect_rogue_activity(hours)
            .await
            .context("Failed to perform rogue detection")?;

        if output_format == crate::args::OutputFormat::Json {
            // JSON output
            let json = serde_json::to_string_pretty(&result)
                .context("Failed to serialize rogue detection results to JSON")?;
            println!("{}", json);
        } else {
            // Table output
            render_info(&format!(
                "🕵️ Rogue Activity Detection Results (Last {} hours)",
                hours
            ));
            render_info(&format!("Overall Risk Score: {:.2}/1.0", result.risk_score));

            if !result.crypto_miners.is_empty() {
                render_warning(&format!(
                    "🚨 CRITICAL: {} crypto miners detected!",
                    result.crypto_miners.len()
                ));
                for (i, miner) in result.crypto_miners.iter().enumerate() {
                    render_warning(&format!(
                        "  {}. PID {}: {} (confidence: {:.2})",
                        i + 1,
                        miner.process.pid,
                        miner.process.proc_name,
                        miner.confidence
                    ));
                    for indicator in &miner.mining_indicators {
                        render_info(&format!("     - {}", indicator));
                    }
                }
            }

            if !result.suspicious_processes.is_empty() {
                render_warning(&format!(
                    "⚠️ {} suspicious processes detected!",
                    result.suspicious_processes.len()
                ));
                for (i, process) in result.suspicious_processes.iter().enumerate() {
                    let risk_emoji = match process.risk_level {
                        crate::rogue_detection::RiskLevel::Critical => "🚨",
                        crate::rogue_detection::RiskLevel::High => "⚠️",
                        crate::rogue_detection::RiskLevel::Medium => "⚡",
                        crate::rogue_detection::RiskLevel::Low => "ℹ️",
                    };
                    render_warning(&format!(
                        "  {}. {} PID {}: {} (confidence: {:.2})",
                        i + 1,
                        risk_emoji,
                        process.process.pid,
                        process.process.proc_name,
                        process.confidence
                    ));
                    for reason in &process.reasons {
                        render_info(&format!("     - {}", reason));
                    }
                }
            }

            if !result.resource_abusers.is_empty() {
                render_warning(&format!(
                    "📊 {} resource abusers detected!",
                    result.resource_abusers.len()
                ));
                for (i, abuser) in result.resource_abusers.iter().enumerate() {
                    let abuse_type = match abuser.abuse_type {
                        crate::rogue_detection::AbuseType::MemoryHog => "Memory Hog",
                        crate::rogue_detection::AbuseType::LongRunning => "Long Running",
                        crate::rogue_detection::AbuseType::ExcessiveUtilization => {
                            "Excessive Utilization"
                        }
                        crate::rogue_detection::AbuseType::UnauthorizedAccess => {
                            "Unauthorized Access"
                        }
                    };
                    render_warning(&format!(
                        "  {}. PID {}: {} - {} (severity: {:.2})",
                        i + 1,
                        abuser.process.pid,
                        abuser.process.proc_name,
                        abuse_type,
                        abuser.severity
                    ));
                }
            }

            if result.crypto_miners.is_empty()
                && result.suspicious_processes.is_empty()
                && result.resource_abusers.is_empty()
            {
                render_info("✅ No suspicious activity detected!");
            }

            if !result.recommendations.is_empty() {
                render_info("\n📋 Recommendations:");
                for recommendation in &result.recommendations {
                    render_info(&format!("  {}", recommendation));
                }
            }
        }
        return Ok(());
    }

    if summary {
        // Show audit summary
        let summary = audit_manager
            .get_summary(hours)
            .await
            .context("Failed to get audit summary")?;

        render_info(&format!("GPU Usage Audit Summary (Last {} hours)", hours));
        render_info(&format!("Total records: {}", summary.total_records));

        if !summary.top_users.is_empty() {
            render_info("\nTop Users by Memory Usage:");
            for (i, (user, count, memory_mb)) in summary.top_users.iter().enumerate() {
                render_info(&format!(
                    "  {}. {}: {} records, {} MB total",
                    i + 1,
                    user,
                    count,
                    memory_mb
                ));
            }
        }

        if !summary.top_processes.is_empty() {
            render_info("\nTop Processes by Memory Usage:");
            for (i, (process, count, memory_mb)) in summary.top_processes.iter().enumerate() {
                render_info(&format!(
                    "  {}. {}: {} records, {} MB total",
                    i + 1,
                    process,
                    count,
                    memory_mb
                ));
            }
        }

        render_info("\nHourly GPU Memory Usage:");
        for (hour, avg_memory) in &summary.gpu_usage_by_hour {
            render_info(&format!("  Hour {}: {} MB average", hour, avg_memory));
        }
    } else {
        // Show detailed audit records
        let records = audit_manager
            .query_records(hours, user_filter.as_deref(), process_filter.as_deref())
            .await
            .context("Failed to query audit records")?;

        if records.is_empty() {
            render_warning(&format!(
                "No audit records found for the last {} hours",
                hours
            ));
            if user_filter.is_some() || process_filter.is_some() {
                render_info("Try removing filters to see all records");
            }
            return Ok(());
        }

        render_info(&format!(
            "Found {} audit records (Last {} hours)",
            records.len(),
            hours
        ));

        if output_format == crate::args::OutputFormat::Json {
            // JSON output
            let json = serde_json::to_string_pretty(&records)
                .context("Failed to serialize audit records to JSON")?;
            println!("{}", json);
        } else {
            // Table output
            use tabled::{Table, Tabled};

            #[derive(Tabled)]
            struct AuditTableRow {
                #[tabled(rename = "Time")]
                time: String,
                #[tabled(rename = "GPU")]
                gpu: String,
                #[tabled(rename = "PID")]
                pid: String,
                #[tabled(rename = "User")]
                user: String,
                #[tabled(rename = "Process")]
                process: String,
                #[tabled(rename = "Memory (MB)")]
                memory: u32,
                #[tabled(rename = "Container")]
                container: String,
            }

            let table_rows: Vec<AuditTableRow> = records
                .iter()
                .map(|record| AuditTableRow {
                    time: record.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                    gpu: format!("{} ({})", record.gpu_index, record.gpu_name),
                    pid: record
                        .pid
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "-".to_string()),
                    user: record.user.clone().unwrap_or_else(|| "-".to_string()),
                    process: record
                        .process_name
                        .clone()
                        .unwrap_or_else(|| "-".to_string()),
                    memory: record.memory_used_mb,
                    container: record.container.clone().unwrap_or_else(|| "-".to_string()),
                })
                .collect();

            let table = Table::new(table_rows);
            println!("{}", table);
        }
    }

    Ok(())
}

/// Open the default browser to http://localhost:{port} (used for --server --open).
fn open_browser_at_port(port: u16) {
    let url = format!("http://localhost:{}", port);
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(&url).status();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open").arg(&url).status();
    }
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", &url])
            .status();
    }
}

/// Execute server operation
async fn execute_server_operation(host: String, port: u16, gpu_manager: GpuManager) -> Result<()> {
    use axum::serve;
    use std::net::SocketAddr;

    info!("Starting GPU Kill Coordinator Server on {}:{}", host, port);

    // Initialize coordinator state
    let state = CoordinatorState::new()
        .await
        .context("Failed to initialize coordinator audit manager")?;

    // Start background tasks for cluster management
    state.start_background_tasks();

    // Register this node as the coordinator
    let node_id = uuid::Uuid::new_v4().to_string();
    let hostname = crate::util::get_hostname();

    // Get initial GPU information
    let gpu_snapshots = gpu_manager.get_all_snapshots()?;
    let gpu_processes = gpu_manager.get_all_processes()?;
    let total_memory_gb = gpu_snapshots
        .iter()
        .map(|gpu| gpu.mem_total_mb as f32 / 1024.0)
        .sum();

    let node_info = crate::coordinator::NodeInfo {
        id: node_id.clone(),
        hostname: hostname.clone(),
        ip_address: "127.0.0.1".to_string(), // TODO: Get actual IP
        last_seen: chrono::Utc::now(),
        status: crate::coordinator::NodeStatus::Online,
        gpu_count: gpu_snapshots.len() as u32,
        total_memory_gb,
        tags: std::collections::HashMap::new(),
    };

    state.register_node(node_info).await?;

    // Create initial snapshot
    let initial_snapshot = crate::coordinator::NodeSnapshot {
        node_id: node_id.clone(),
        hostname,
        timestamp: chrono::Utc::now(),
        gpus: gpu_snapshots,
        processes: gpu_processes,
        status: crate::coordinator::NodeStatus::Online,
    };

    state.update_snapshot(node_id, initial_snapshot).await?;

    // Create router
    let app = create_router(state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("GPU Kill Coordinator Server listening on http://{}", addr);
    info!("Dashboard will be available at http://{}:{}", host, port);
    info!("API endpoints:");
    info!("  GET  /api/nodes - List all nodes");
    info!("  GET  /api/cluster/snapshot - Get cluster snapshot");
    info!("  GET  /api/cluster/contention - Get contention analysis");
    info!("  WS   /ws - WebSocket for real-time updates");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .context("Failed to bind to address")?;

    serve(listener, app)
        .await
        .context("Failed to start server")?;

    Ok(())
}

/// Execute operation on remote host via SSH
async fn execute_remote_operation(cli: Cli, remote_host: &str) -> Result<()> {
    use crate::remote::{execute_remote_operation as remote_exec, SshConfig};
    use std::time::Duration;

    info!("Executing remote operation on {}", remote_host);

    // Build SSH configuration
    let username = cli
        .ssh_user
        .unwrap_or_else(|| std::env::var("USER").unwrap_or_else(|_| "root".to_string()));

    let mut ssh_config = SshConfig::new(remote_host.to_string(), cli.ssh_port, username)
        .with_timeout(Duration::from_secs(cli.ssh_timeout as u64));

    // Add authentication options
    if let Some(key_path) = &cli.ssh_key {
        ssh_config = ssh_config.with_key_path(key_path.clone());
    }

    if let Some(password) = &cli.ssh_password {
        ssh_config = ssh_config.with_password(password.clone());
    }

    // Build command arguments for remote execution
    let mut remote_args = Vec::new();

    // Add operation flags
    if cli.list {
        remote_args.push("--list".to_string());
        if cli.details {
            remote_args.push("--details".to_string());
        }
        if cli.watch {
            remote_args.push("--watch".to_string());
        }
        if cli.containers {
            remote_args.push("--containers".to_string());
        }
    } else if cli.kill {
        remote_args.push("--kill".to_string());
        if let Some(pid) = cli.pid {
            remote_args.push("--pid".to_string());
            remote_args.push(pid.to_string());
        }
        if let Some(filter) = &cli.filter {
            remote_args.push("--filter".to_string());
            remote_args.push(filter.clone());
        }
        if let Some(gpu_id) = cli.gpu {
            remote_args.push("--gpu".to_string());
            remote_args.push(gpu_id.to_string());
        }
        if cli.batch {
            remote_args.push("--batch".to_string());
        }
        if cli.force {
            remote_args.push("--force".to_string());
        }
        remote_args.push("--timeout-secs".to_string());
        remote_args.push(cli.timeout_secs.to_string());
    } else if cli.reset {
        remote_args.push("--reset".to_string());
        if let Some(gpu_id) = cli.gpu {
            remote_args.push("--gpu".to_string());
            remote_args.push(gpu_id.to_string());
        }
        if cli.all {
            remote_args.push("--all".to_string());
        }
        if cli.force {
            remote_args.push("--force".to_string());
        }
    } else if cli.audit {
        remote_args.push("--audit".to_string());
        if let Some(user) = &cli.audit_user {
            remote_args.push("--audit-user".to_string());
            remote_args.push(user.clone());
        }
        if let Some(process) = &cli.audit_process {
            remote_args.push("--audit-process".to_string());
            remote_args.push(process.clone());
        }
        remote_args.push("--audit-hours".to_string());
        remote_args.push(cli.audit_hours.to_string());
        if cli.audit_summary {
            remote_args.push("--audit-summary".to_string());
        }
    } else if cli.server {
        return Err(anyhow::anyhow!(
            "Server mode cannot be used with remote operations"
        ));
    }

    // Add output format
    match cli.output {
        crate::args::OutputFormat::Json => {
            remote_args.push("--output".to_string());
            remote_args.push("json".to_string());
        }
        crate::args::OutputFormat::Table => {
            // Table is default, no need to specify
        }
    }

    // Propagate dry-run/safe
    if cli.dry_run {
        remote_args.push("--dry-run".to_string());
    }

    // Add vendor filter if specified
    if let Some(vendor) = &cli.vendor {
        remote_args.push("--vendor".to_string());
        remote_args.push(format!("{:?}", vendor).to_lowercase());
    }

    // Execute the remote operation
    remote_exec(ssh_config, &remote_args)?;

    Ok(())
}

/// Execute Guard Mode operation
async fn execute_guard_operation(
    cli: &crate::args::Cli,
    _gpu_manager: crate::vendor::GpuManager,
) -> Result<()> {
    use crate::guard_mode::GuardModeManager;
    use crate::render::render_info;

    // Initialize guard mode manager
    let mut guard_manager =
        GuardModeManager::new().context("Failed to initialize Guard Mode manager")?;

    // Handle configuration management
    if cli.guard_config
        || cli.guard_enable
        || cli.guard_disable
        || cli.guard_dry_run
        || cli.guard_enforce
        || cli.guard_add_user.is_some()
        || cli.guard_remove_user.is_some()
        || cli.guard_memory_limit.is_some()
        || cli.guard_utilization_limit.is_some()
        || cli.guard_process_limit.is_some()
        || cli.guard_add_group.is_some()
        || cli.guard_remove_group.is_some()
        || cli.guard_add_gpu.is_some()
        || cli.guard_remove_gpu.is_some()
        || cli.guard_group_memory_limit.is_some()
        || cli.guard_group_utilization_limit.is_some()
        || cli.guard_group_process_limit.is_some()
        || cli.guard_gpu_memory_limit.is_some()
        || cli.guard_gpu_utilization_limit.is_some()
        || cli.guard_gpu_reserved_memory.is_some()
        || cli.guard_export_config
        || cli.guard_import_config.is_some()
        || cli.guard_test_policies
        || cli.guard_toggle_dry_run
    {
        // Show current configuration
        if cli.guard_config {
            let config = guard_manager.get_config();
            render_info("🛡️ Guard Mode Configuration:");
            render_info(&format!("  Enabled: {}", config.global.enabled));
            render_info(&format!("  Dry Run: {}", config.global.dry_run));
            render_info(&format!(
                "  Default Memory Limit: {:.1} GB",
                config.global.default_memory_limit_gb
            ));
            render_info(&format!(
                "  Default Utilization Limit: {:.1}%",
                config.global.default_utilization_limit_pct
            ));
            render_info(&format!(
                "  Default Duration Limit: {:.1} hours",
                config.global.default_duration_limit_hours
            ));
            render_info(&format!(
                "  Check Interval: {} seconds",
                config.global.check_interval_seconds
            ));

            render_info(&format!(
                "  Soft Enforcement: {}",
                config.enforcement.soft_enforcement
            ));
            render_info(&format!(
                "  Hard Enforcement: {}",
                config.enforcement.hard_enforcement
            ));
            render_info(&format!(
                "  Grace Period: {} seconds",
                config.enforcement.grace_period_seconds
            ));

            render_info("\n👥 User Policies:");
            for (username, policy) in &config.user_policies {
                render_info(&format!(
                    "  - {}: {:.1}GB memory, {:.1}% util, {} processes",
                    username,
                    policy.memory_limit_gb,
                    policy.utilization_limit_pct,
                    policy.max_concurrent_processes
                ));
            }

            render_info("\n👥 Group Policies:");
            for (group_name, policy) in &config.group_policies {
                let members_info = if !policy.members.is_empty() {
                    format!(
                        ", {} members: {}",
                        policy.members.len(),
                        policy.members.join(", ")
                    )
                } else {
                    "".to_string()
                };
                render_info(&format!(
                    "  - {}: {:.1}GB memory, {:.1}% util, {} processes{}",
                    group_name,
                    policy.total_memory_limit_gb,
                    policy.total_utilization_limit_pct,
                    policy.max_concurrent_processes,
                    members_info
                ));
            }

            render_info("\n🖥️ GPU Policies:");
            for (gpu_index, policy) in &config.gpu_policies {
                let users_info = if !policy.allowed_users.is_empty() {
                    format!(
                        ", {} allowed users: {}",
                        policy.allowed_users.len(),
                        policy.allowed_users.join(", ")
                    )
                } else {
                    "".to_string()
                };
                render_info(&format!(
                    "  - GPU {}: {:.1}GB memory, {:.1}% util, {:.1}GB reserved{}",
                    gpu_index,
                    policy.max_memory_gb,
                    policy.max_utilization_pct,
                    policy.reserved_memory_gb,
                    users_info
                ));
            }

            render_info(&format!(
                "\n📁 Config file: {}",
                guard_manager.get_config_file_path().display()
            ));
        }

        // Enable/disable guard mode
        if cli.guard_enable {
            guard_manager
                .set_enabled(true)
                .context("Failed to enable Guard Mode")?;
            render_info("✅ Guard Mode enabled");
        }

        if cli.guard_disable {
            guard_manager
                .set_enabled(false)
                .context("Failed to disable Guard Mode")?;
            render_info("✅ Guard Mode disabled");
        }

        // Set dry-run mode
        if cli.guard_dry_run {
            guard_manager
                .set_dry_run(true)
                .context("Failed to set dry-run mode")?;
            render_info("✅ Guard Mode set to dry-run (no enforcement)");
        }

        if cli.guard_enforce {
            guard_manager
                .set_dry_run(false)
                .context("Failed to set enforcement mode")?;
            render_info("✅ Guard Mode set to enforce policies");
        }

        // Add user policy
        if let Some(username) = &cli.guard_add_user {
            let memory_limit = cli.guard_memory_limit.unwrap_or(16.0);
            let utilization_limit = cli.guard_utilization_limit.unwrap_or(80.0);
            let process_limit = cli.guard_process_limit.unwrap_or(5);

            let user_policy = crate::guard_mode::UserPolicy {
                username: username.clone(),
                memory_limit_gb: memory_limit,
                utilization_limit_pct: utilization_limit,
                duration_limit_hours: 12.0,
                max_concurrent_processes: process_limit,
                priority: 5,
                allowed_gpus: Vec::new(),
                blocked_gpus: Vec::new(),
                time_overrides: Vec::new(),
            };

            guard_manager
                .add_user_policy(user_policy)
                .context("Failed to add user policy")?;
            render_info(&format!(
                "✅ Added policy for user '{}': {:.1}GB memory, {:.1}% util, {} processes",
                username, memory_limit, utilization_limit, process_limit
            ));
        }

        // Remove user policy
        if let Some(username) = &cli.guard_remove_user {
            guard_manager
                .remove_user_policy(username)
                .context("Failed to remove user policy")?;
            render_info(&format!("✅ Removed policy for user '{}'", username));
        }

        // Add group policy
        if let Some(group_name) = &cli.guard_add_group {
            let memory_limit = cli.guard_group_memory_limit.unwrap_or(32.0);
            let utilization_limit = cli.guard_group_utilization_limit.unwrap_or(80.0);
            let process_limit = cli.guard_group_process_limit.unwrap_or(10);

            // Parse members from comma-separated input
            let members = if let Some(members_str) = &cli.guard_group_members {
                members_str
                    .split(',')
                    .map(|m| m.trim().to_string())
                    .filter(|m| !m.is_empty())
                    .collect()
            } else {
                vec![]
            };

            let members_info = if !members.is_empty() {
                format!(", {} members: {}", members.len(), members.join(", "))
            } else {
                "".to_string()
            };

            let group_policy = crate::guard_mode::GroupPolicy {
                group_name: group_name.clone(),
                total_memory_limit_gb: memory_limit,
                total_utilization_limit_pct: utilization_limit,
                max_concurrent_processes: process_limit,
                priority: 5,
                allowed_gpus: vec![],
                blocked_gpus: vec![],
                members,
            };

            guard_manager
                .add_group_policy(group_policy)
                .context("Failed to add group policy")?;

            render_info(&format!(
                "✅ Added policy for group '{}': {:.1}GB memory, {:.1}% util, {} processes{}",
                group_name, memory_limit, utilization_limit, process_limit, members_info
            ));
        }

        // Remove group policy
        if let Some(group_name) = &cli.guard_remove_group {
            guard_manager
                .remove_group_policy(group_name)
                .context("Failed to remove group policy")?;
            render_info(&format!("✅ Removed policy for group '{}'", group_name));
        }

        // Add GPU policy
        if let Some(gpu_index) = cli.guard_add_gpu {
            let memory_limit = cli.guard_gpu_memory_limit.unwrap_or(24.0);
            let utilization_limit = cli.guard_gpu_utilization_limit.unwrap_or(90.0);
            let reserved_memory = cli.guard_gpu_reserved_memory.unwrap_or(2.0);

            // Parse allowed users from comma-separated input
            let allowed_users = if let Some(users_str) = &cli.guard_gpu_allowed_users {
                users_str
                    .split(',')
                    .map(|u| u.trim().to_string())
                    .filter(|u| !u.is_empty())
                    .collect()
            } else {
                vec![]
            };

            let users_info = if !allowed_users.is_empty() {
                format!(
                    ", {} allowed users: {}",
                    allowed_users.len(),
                    allowed_users.join(", ")
                )
            } else {
                "".to_string()
            };

            let gpu_policy = crate::guard_mode::GpuPolicy {
                gpu_index,
                max_memory_gb: memory_limit,
                max_utilization_pct: utilization_limit,
                reserved_memory_gb: reserved_memory,
                allowed_users,
                blocked_users: vec![],
                maintenance_window: None,
            };

            guard_manager
                .add_gpu_policy(gpu_policy)
                .context("Failed to add GPU policy")?;

            render_info(&format!(
                "✅ Added policy for GPU {}: {:.1}GB memory, {:.1}% util, {:.1}GB reserved{}",
                gpu_index, memory_limit, utilization_limit, reserved_memory, users_info
            ));
        }

        // Remove GPU policy
        if let Some(gpu_index) = cli.guard_remove_gpu {
            guard_manager
                .remove_gpu_policy(gpu_index)
                .context("Failed to remove GPU policy")?;
            render_info(&format!("✅ Removed policy for GPU {}", gpu_index));
        }

        // Export configuration
        if cli.guard_export_config {
            let json = guard_manager
                .export_to_json()
                .context("Failed to export Guard Mode config to JSON")?;
            println!("{}", json);
        }

        // Import configuration
        if let Some(file_path) = &cli.guard_import_config {
            let content =
                std::fs::read_to_string(file_path).context("Failed to read import file")?;
            guard_manager
                .import_from_json(&content)
                .context("Failed to import Guard Mode config from JSON")?;
            render_info(&format!(
                "✅ Imported Guard Mode configuration from: {}",
                file_path
            ));
        }

        // Test policies in dry-run mode
        if cli.guard_test_policies {
            render_info("🧪 Testing policies in dry-run mode...");

            // Get current GPU processes for testing
            let gpu_manager = crate::vendor::GpuManager::initialize()
                .context("Failed to initialize GPU manager")?;
            let test_processes = gpu_manager
                .get_all_processes()
                .context("Failed to get GPU processes")?;

            let result = guard_manager
                .simulate_policy_check(&test_processes)
                .context("Failed to simulate policy check")?;

            render_info("📊 Simulation Results:");
            render_info(&format!("  Violations found: {}", result.violations.len()));
            render_info(&format!("  Warnings found: {}", result.warnings.len()));
            render_info(&format!(
                "  Actions simulated: {}",
                result.actions_taken.len()
            ));

            if !result.violations.is_empty() {
                render_info("\n🚨 Simulated Violations:");
                for (i, violation) in result.violations.iter().enumerate() {
                    render_info(&format!(
                        "  {}. {} - {:?} ({:?}): {}",
                        i + 1,
                        violation.user,
                        violation.violation_type,
                        violation.severity,
                        violation.message
                    ));
                }
            }

            if !result.actions_taken.is_empty() {
                render_info("\n⚡ Simulated Actions:");
                for (i, action) in result.actions_taken.iter().enumerate() {
                    render_info(&format!(
                        "  {}. {:?}: {}",
                        i + 1,
                        action.action_type,
                        action.message
                    ));
                }
            }

            if result.violations.is_empty() && result.warnings.is_empty() {
                render_info("✅ No policy violations detected in simulation!");
            }
        }

        // Toggle dry-run mode
        if cli.guard_toggle_dry_run {
            let new_dry_run = guard_manager
                .toggle_dry_run()
                .context("Failed to toggle dry-run mode")?;
            render_info(&format!(
                "✅ Dry-run mode {} (simulation only)",
                if new_dry_run { "enabled" } else { "disabled" }
            ));
        }

        return Ok(());
    }

    // If no specific guard operations, show help
    render_info("🛡️ Guard Mode - Soft Policy Enforcement");
    render_info("Use --guard-config to view current configuration");
    render_info("Use --guard-enable to enable Guard Mode");
    render_info("Use --guard-dry-run to test policies without enforcement");
    render_info("Use --guard-add-user <username> to add user policies");

    Ok(())
}

/// Execute node registration operation
async fn execute_register_node_operation(
    coordinator_url: String,
    gpu_manager: GpuManager,
) -> Result<()> {
    use crate::coordinator::{NodeInfo, NodeSnapshot, NodeStatus};
    use crate::render::render_info;
    use reqwest::Client;
    use std::collections::HashMap;
    use uuid::Uuid;

    info!("Registering node with coordinator: {}", coordinator_url);

    // Get node information
    let node_id = Uuid::new_v4().to_string();
    let hostname = crate::util::get_hostname();
    let ip_address = "127.0.0.1".to_string(); // Simplified for now

    // Get GPU information
    let gpus = gpu_manager
        .get_all_snapshots()
        .context("Failed to get GPU snapshots")?;
    let procs = gpu_manager
        .get_all_processes()
        .context("Failed to get GPU processes")?;

    let total_memory_gb: f32 = gpus
        .iter()
        .map(|gpu| gpu.mem_total_mb as f32 / 1024.0)
        .sum();

    // Create node info
    let node_info = NodeInfo {
        id: node_id.clone(),
        hostname: hostname.clone(),
        ip_address,
        last_seen: chrono::Utc::now(),
        status: NodeStatus::Online,
        gpu_count: gpus.len() as u32,
        total_memory_gb,
        tags: HashMap::new(),
    };

    // Create node snapshot
    let snapshot = NodeSnapshot {
        node_id: node_id.clone(),
        hostname,
        timestamp: chrono::Utc::now(),
        gpus,
        processes: procs,
        status: NodeStatus::Online,
    };

    let client = Client::new();

    // Register node
    let register_url = format!("{}/api/nodes/{}/register", coordinator_url, node_id);
    match client.post(&register_url).json(&node_info).send().await {
        Ok(response) => {
            if response.status().is_success() {
                render_info(&format!(
                    "✅ Successfully registered node {} with coordinator",
                    node_id
                ));
            } else {
                return Err(anyhow::anyhow!(
                    "Failed to register node: HTTP {}",
                    response.status()
                ));
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to register node: {}", e));
        }
    }

    // Send initial snapshot
    let snapshot_url = format!("{}/api/nodes/{}/snapshot", coordinator_url, node_id);
    match client.post(&snapshot_url).json(&snapshot).send().await {
        Ok(response) => {
            if response.status().is_success() {
                render_info("✅ Successfully sent initial snapshot to coordinator");
            } else {
                return Err(anyhow::anyhow!(
                    "Failed to send snapshot: HTTP {}",
                    response.status()
                ));
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to send snapshot: {}", e));
        }
    }

    // Start periodic snapshot updates
    render_info("🔄 Starting periodic snapshot updates...");
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

    loop {
        interval.tick().await;

        // Get fresh snapshot
        let gpus = match gpu_manager.get_all_snapshots() {
            Ok(gpus) => gpus,
            Err(e) => {
                warn!("Failed to get GPU snapshots: {}", e);
                continue;
            }
        };

        let procs = match gpu_manager.get_all_processes() {
            Ok(procs) => procs,
            Err(e) => {
                warn!("Failed to get GPU processes: {}", e);
                continue;
            }
        };

        let snapshot = NodeSnapshot {
            node_id: node_id.clone(),
            hostname: node_info.hostname.clone(),
            timestamp: chrono::Utc::now(),
            gpus,
            processes: procs,
            status: NodeStatus::Online,
        };

        // Send snapshot
        match client.post(&snapshot_url).json(&snapshot).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    debug!("Successfully sent snapshot update");
                } else {
                    warn!("Failed to send snapshot update: HTTP {}", response.status());
                }
            }
            Err(e) => {
                warn!("Failed to send snapshot update: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_initialization() {
        // This test just ensures the function doesn't panic
        let result = init_logging("info");
        assert!(result.is_ok());
    }

    #[test]
    fn test_version_string() {
        let version = get_version_string();
        assert!(version.contains("gpukill"));
    }
}
