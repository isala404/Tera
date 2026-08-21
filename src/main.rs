use tera::config::Config;
use tera::codex::tier;
use tera::codex::CodexSupervisor;
use tera::conversation::{ConversationSession, Phoenix, TurnEngine};
use tera::history::{backup, HistoryDb, ProjectionEngine};
use tera::memory::generations::GenerationManager;
use tera::scheduler::db::SchedulerDb;
use tera::scheduler::recurrence;
use tera::secrets::SecretStore;
use tera::mcp::{DaemonRpcServer, StdioMcpProxy};
use tera::memory::optimizer::OptimizerOutcome;
use tera::memory::{MaintenanceRunner, MemoryOptimizer, MemoryRebuilder};
use tera::runtime::{self, ActivityTracker, DaemonLock, RuntimeDb};
use tera::scheduler::SchedulerRunner;
use tera::transport::{InboundEvent, MockTransport, Transport, WhatsAppWebTransport};
use tera::workspace::WorkspaceInit;
use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;

/// How many times Phoenix retries before giving up on reaching the owner. Ten
/// tries at ten-second spacing outlasts any normal WhatsApp reconnect.
const PHOENIX_REPORT_ATTEMPTS: u32 = 10;

use tracing::{error, info, warn};

#[derive(Parser)]
#[command(
    name = "tera",
    version = env!("CARGO_PKG_VERSION"),
    long_version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("TERA_GIT_SHA"), ", built ", env!("TERA_BUILD_TIME"), ")")
)]
#[command(about = "Persistent Personal Helper Daemon", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print Tera and Codex version details
    Version {
        #[arg(long)]
        json: bool,
    },
    /// Update Tera and Codex, then restart the daemon safely
    Update {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
        #[arg(long, value_enum, default_value_t = UpdateComponent::All)]
        component: UpdateComponent,
        /// Reinstall the current Tera release
        #[arg(long)]
        force: bool,
    },
    /// Start the long-running assistant daemon
    Daemon {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
        #[arg(long)]
        mock_transport: bool,
    },
    /// Idempotent workspace initialization
    Init {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
    },
    /// Stdio MCP server proxy for Codex App Server
    Mcp {
        #[arg(long)]
        socket: PathBuf,
    },
    /// Print system health and state status
    Status {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
    },
    /// History tools
    History {
        #[command(subcommand)]
        sub: HistorySubcommands,
    },
    /// Memory tools
    Memory {
        #[command(subcommand)]
        sub: MemorySubcommands,
    },
    /// Credentials skills read. Normally set from chat, not here.
    Secret {
        #[command(subcommand)]
        sub: SecretSubcommands,
    },
    #[command(name = "__signal-update", hide = true)]
    InternalSignalUpdate {
        #[arg(long)]
        lock: PathBuf,
        #[arg(long)]
        pid: u32,
    },
    #[command(name = "__restart-daemon", hide = true)]
    InternalRestartDaemon {
        #[arg(long)]
        workspace: PathBuf,
        #[arg(long)]
        old_pid: u32,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum UpdateComponent {
    All,
    Tera,
    Codex,
}

impl From<UpdateComponent> for tera::update::Component {
    fn from(value: UpdateComponent) -> Self {
        match value {
            UpdateComponent::All => Self::All,
            UpdateComponent::Tera => Self::Tera,
            UpdateComponent::Codex => Self::Codex,
        }
    }
}

#[derive(Subcommand)]
enum HistorySubcommands {
    /// Rebuild JSONL projection from canonical SQLite history
    RebuildJsonl {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
    },
    /// Snapshot canonical history into history/backups/
    Backup {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
    },
    /// Check canonical history, its projection, and its assets
    Check {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
    },
}

#[derive(Subcommand)]
enum MemorySubcommands {
    /// Trigger full memory regeneration from SQLite history
    Rebuild {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
    },
    /// Trigger nightly memory optimization pass
    Optimize {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
    },
    /// List memory generations and which one is active
    Status {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
    },
    /// Point active memory back at an earlier generation
    Rollback {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
        generation: u64,
    },
}

#[derive(Subcommand)]
enum SecretSubcommands {
    /// List stored names and when each was set. Never prints a value.
    List {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
    },
    /// Store a credential, read from stdin.
    ///
    /// Stdin rather than an argument, so the value does not end up in shell
    /// history or in the process list of every other account on the machine.
    Set {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
        name: String,
    },
    /// Forget a credential.
    Rm {
        #[arg(long, default_value = "/workspace")]
        workspace: PathBuf,
        name: String,
    },
}

impl SecretSubcommands {
    fn workspace(&self) -> &PathBuf {
        match self {
            SecretSubcommands::List { workspace }
            | SecretSubcommands::Set { workspace, .. }
            | SecretSubcommands::Rm { workspace, .. } => workspace,
        }
    }
}

impl Commands {
    /// The workspace this invocation acts on. `mcp` is a child process addressed
    /// by socket and has none, so it logs to stderr only, which is also the only
    /// safe place for it, since Codex owns its stdout.
    fn workspace(&self) -> Option<&PathBuf> {
        match self {
            Commands::Daemon { workspace, .. }
            | Commands::Init { workspace }
            | Commands::Status { workspace }
            | Commands::Update { workspace, .. }
            | Commands::InternalRestartDaemon { workspace, .. } => Some(workspace),
            Commands::History { sub } => Some(sub.workspace()),
            Commands::Memory { sub } => Some(sub.workspace()),
            Commands::Secret { sub } => Some(sub.workspace()),
            Commands::Mcp { .. }
            | Commands::Version { .. }
            | Commands::InternalSignalUpdate { .. } => None,
        }
    }
}

impl HistorySubcommands {
    fn workspace(&self) -> &PathBuf {
        match self {
            HistorySubcommands::RebuildJsonl { workspace }
            | HistorySubcommands::Backup { workspace }
            | HistorySubcommands::Check { workspace } => workspace,
        }
    }
}

impl MemorySubcommands {
    fn workspace(&self) -> &PathBuf {
        match self {
            MemorySubcommands::Rebuild { workspace }
            | MemorySubcommands::Optimize { workspace }
            | MemorySubcommands::Status { workspace }
            | MemorySubcommands::Rollback { workspace, .. } => workspace,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse before logging: where the log file goes depends on which workspace
    // this invocation is acting on.
    let cli = Cli::parse();
    let log_dir = cli
        .command
        .workspace()
        .map(|w| Config::new(w.clone(), false).logs_dir());
    tera::observability::init_tracing(log_dir.as_deref());

    match cli.command {
        Commands::Version { json } => {
            tera::version::VersionReport::current().print(json)?;
        }

        Commands::Update {
            workspace,
            component,
            force,
        } => {
            let config = Config::new(workspace, false);
            let outcome = tera::update::run(&config, component.into(), force)?;
            println!(
                "Tera: {} ({})",
                outcome.tera.version,
                outcome.tera.short_sha()
            );
            if let Some(codex) = outcome.codex {
                println!("Codex: {codex}");
            }
            if outcome.restart_scheduled {
                println!("The daemon will restart in a few seconds. Phoenix will report the result.");
            }
        }

        Commands::InternalSignalUpdate { lock, pid } => {
            tera::update::signal_update(&lock, pid)?;
        }

        Commands::InternalRestartDaemon { workspace, old_pid } => {
            tera::update::restart_daemon(workspace, old_pid)?;
        }

        Commands::Init { workspace } => {
            let config = Config::new(workspace, false);
            WorkspaceInit::init(&config)?;
            let _ = HistoryDb::open_for(&config)?;
            let _ = RuntimeDb::open(&config.runtime_db_path())?;
            info!("Initialization finished successfully!");
        }

        // Reports what is degraded, not just what is configured: a status command
        // that only echoes paths back cannot tell you why the assistant is quiet
        // (PLAN.md section 95).
        Commands::Status { workspace } => {
            let config = Config::new(workspace, false);
            println!("=== tera status ===");
            println!("Workspace:   {}", config.workspace_dir.display());
            println!(
                "Daemon:      {}",
                if config.socket_path().exists() {
                    "socket present"
                } else {
                    "not running (no runtime socket)"
                }
            );

            println!(
                "Log:         {}",
                tera::observability::current_log_path(&config.logs_dir()).display()
            );

            let writable = |dir: &std::path::Path| {
                if dir.is_dir() && !dir.metadata().map(|m| m.permissions().readonly()).unwrap_or(true) {
                    "writable"
                } else {
                    "NOT WRITABLE"
                }
            };
            println!(
                "Writable:    workspace {}, runtime {}",
                writable(&config.workspace_dir),
                writable(&config.runtime_dir())
            );

            if config.history_db_path().exists() {
                let db = HistoryDb::open_for(&config)?;
                match backup::check_integrity(&config, &db) {
                    Ok(report) => {
                        println!(
                            "History:     {} events, {} projected{}{}",
                            report.event_count,
                            report.projected_records,
                            if report.sqlite_ok { "" } else { ", SQLITE INTEGRITY FAILED" },
                            if report.missing_assets.is_empty() {
                                String::new()
                            } else {
                                format!(", {} missing assets", report.missing_assets.len())
                            }
                        );
                        if report.event_count != report.projected_records {
                            println!("             projection is out of sync; it rebuilds on next daemon start");
                        }
                    }
                    Err(e) => println!("History:     could not be checked: {e}"),
                }
            } else {
                println!("History:     not initialized");
            }

            match GenerationManager::active_generation(&config) {
                Some(generation) if config.memories_link().join("INDEX.md").is_file() => {
                    println!("Memory:      generation {generation:08}")
                }
                _ => println!("Memory:      MEMORIES link is BROKEN"),
            }

            println!("Owner:       {} (set TERA_OWNER to change)", config.owner_name);
            println!(
                "Models:      {} ({}) conversation / {} ({}) heavy / {} ({}) routine",
                tier::CONVERSATION.model,
                tier::CONVERSATION.effort,
                tier::HEAVY.model,
                tier::HEAVY.effort,
                tier::ROUTINE.model,
                tier::ROUTINE.effort,
            );

            if config.runtime_db_path().exists() {
                let rdb = RuntimeDb::open(&config.runtime_db_path())?;
                match rdb.get_main_thread()? {
                    Some(thread) => println!(
                        "Thread:      {} (model {}, last active {})",
                        thread.thread_id,
                        thread.model_id,
                        chrono::DateTime::from_timestamp_millis(thread.last_activity_at_ms)
                            .map(|d| d.to_rfc3339())
                            .unwrap_or_else(|| "unknown".to_string())
                    ),
                    None => println!("Thread:      none recorded"),
                }

                let schedules = SchedulerDb::list_schedules(&rdb).unwrap_or_default();
                println!("Schedules:   {} active", schedules.len());
                for item in schedules.iter().take(10) {
                    // Local time, because that is the timezone the rule is written
                    // in, printing UTC here is how a schedule looks wrong when it
                    // is right, and vice versa.
                    println!(
                        "  {} '{}' [{}] next {}",
                        item.id,
                        item.name,
                        item.tier,
                        item.next_run_at_ms
                            .map(tera::scheduler::recurrence::local_time)
                            .unwrap_or_else(|| "never".to_string())
                    );
                }

                let runs = SchedulerDb::recent_runs(&rdb, 5).unwrap_or_default();
                if !runs.is_empty() {
                    println!("Recent runs:");
                    for run in runs {
                        println!(
                            "  {} {} {}",
                            run.state,
                            run.schedule_id,
                            run.error.as_deref().unwrap_or("")
                        );
                    }
                }

                for (label, key) in [
                    ("Rebuild pending", tera::memory::rebuild::REBUILD_PENDING_KEY),
                    ("Optimizer retry", tera::memory::optimizer::RETRY_PENDING_KEY),
                ] {
                    if rdb.get_state_value(key)?.as_deref() == Some("true") {
                        println!("{label}: yes");
                    }
                }
            }
        }

        Commands::Mcp { socket } => {
            // The proxy is a child process Codex spawns, addressed only by socket:
            // it has no workspace flag. It still needs the owner's name, because the
            // tool descriptions it serves are prompt text.
            let owner = Config::new(std::path::PathBuf::from("."), false).owner_name;
            let proxy = StdioMcpProxy::new(socket, owner);
            proxy.run().await?;
        }

        Commands::History { sub } => match sub {
            HistorySubcommands::RebuildJsonl { workspace } => {
                let config = Config::new(workspace, false);
                let history_db = HistoryDb::open_for(&config)?;
                ProjectionEngine::rebuild_all(
                    &config.history_jsonl_dir(),
                    &config.runtime_dir().join("tmp"),
                    &history_db,
                )?;
            }

            HistorySubcommands::Backup { workspace } => {
                let config = Config::new(workspace, false);
                let path = backup::backup_history(&config, &backup::timestamp_now())?;
                println!("Backed up history to {}", path.display());
            }

            HistorySubcommands::Check { workspace } => {
                let config = Config::new(workspace, false);
                let history_db = HistoryDb::open_for(&config)?;
                let report = backup::check_integrity(&config, &history_db)?;

                println!("SQLite integrity:   {}", if report.sqlite_ok { "ok" } else { "FAILED" });
                println!("Canonical events:   {}", report.event_count);
                println!("Projected records:  {}", report.projected_records);
                println!("Projection dirty:   {}", report.projection_dirty);
                println!("Missing assets:     {}", report.missing_assets.len());
                for path in report.missing_assets.iter().take(20) {
                    println!("  - {path}");
                }

                if !report.is_healthy() {
                    // Exit non-zero so this is usable from a script or a cron job.
                    std::process::exit(1);
                }
            }
        },

        // Both of these drive a real Codex turn, so they need a workspace that is
        // set up and a live app-server, the same path the daemon uses.
        Commands::Memory { sub } => match sub {
            MemorySubcommands::Rebuild { workspace } => {
                let config = Config::new(workspace, false);
                WorkspaceInit::init(&config)?;
                let history_db = HistoryDb::open_for(&config)?;
                let runtime_db = RuntimeDb::open(&config.runtime_db_path())?;
                let codex =
                    CodexSupervisor::new(config.clone(), runtime_db.clone(), history_db.clone());
                let generation =
                    MemoryRebuilder::run(&config, &history_db, &runtime_db, &codex).await?;
                println!("Memory rebuilt; generation {generation} is active");
            }

            MemorySubcommands::Optimize { workspace } => {
                let config = Config::new(workspace, false);
                WorkspaceInit::init(&config)?;
                let runtime_db = RuntimeDb::open(&config.runtime_db_path())?;
                let history_db = HistoryDb::open_for(&config)?;
                let codex = CodexSupervisor::new(config.clone(), runtime_db.clone(), history_db);
                // No other work can be in flight in a one-shot command, so an
                // empty tracker is the honest answer to "is anything active".
                let activity = ActivityTracker::new();
                match MemoryOptimizer::run(&config, &runtime_db, &codex, &activity).await? {
                    OptimizerOutcome::Promoted(generation) => {
                        println!("Memory optimized; generation {generation} is active")
                    }
                    OptimizerOutcome::Interrupted => println!("Optimization was interrupted"),
                    OptimizerOutcome::Rejected(reason) => {
                        println!("Optimization rejected, memory unchanged: {reason}")
                    }
                }
            }

            MemorySubcommands::Status { workspace } => {
                let config = Config::new(workspace, false);
                let active = GenerationManager::active_generation(&config);
                let latest = GenerationManager::get_current_generation_num(&config)?;
                match active {
                    Some(generation) => println!("Active:  {generation:08}"),
                    None => println!("Active:  none, the MEMORIES link is broken"),
                }
                if active != Some(latest) {
                    // Normal after a rollback; alarming if it is a surprise.
                    println!("Latest:  {latest:08} (not active)");
                }
                println!("\nGenerations:");
                let mut generations = std::fs::read_dir(config.generations_dir())?
                    .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
                    .collect::<Vec<_>>();
                generations.sort();
                let active_name = active.map(|g| format!("{g:08}"));
                for name in generations {
                    let marker = if Some(&name) == active_name.as_ref() { " (active)" } else { "" };
                    println!("  {name}{marker}");
                }
            }

            MemorySubcommands::Rollback {
                workspace,
                generation,
            } => {
                let config = Config::new(workspace, false);
                let target = config.generations_dir().join(format!("{generation:08}"));
                if !target.is_dir() {
                    return Err(anyhow::anyhow!("No memory generation {generation:08} at {target:?}"));
                }
                GenerationManager::validate_generation_dir(&target)?;
                GenerationManager::point_memories_at(&config, generation)?;
                println!("Active memory rolled back to generation {generation:08}");
            }
        },

        Commands::Secret { sub } => {
            let store = SecretStore::new(Config::new(sub.workspace().clone(), false).secrets_path());
            match sub {
                SecretSubcommands::List { .. } => {
                    let names = store.names()?;
                    if names.is_empty() {
                        println!("No secrets are stored.");
                    }
                    for (name, set_at_ms) in names {
                        println!("{name}  set {}", recurrence::local_time(set_at_ms));
                    }
                }

                SecretSubcommands::Set { name, .. } => {
                    // Stdin, so the value never reaches the shell history or the
                    // argv every other account on this machine can read.
                    let mut value = String::new();
                    std::io::Read::read_to_string(&mut std::io::stdin(), &mut value)?;
                    store.set(&name, value.trim(), chrono::Utc::now().timestamp_millis())?;
                    println!("Stored {name}");
                }

                SecretSubcommands::Rm { name, .. } => {
                    if store.remove(&name)? {
                        println!("Removed {name}");
                    } else {
                        println!("No secret named {name}");
                    }
                }
            }
        }

        Commands::Daemon {
            workspace,
            mock_transport,
        } => {
            let config = Config::new(workspace, mock_transport);
            info!("Starting tera daemon...");

            // 1. Acquire singleton daemon lock
            let _lock = DaemonLock::acquire(&config.lock_file_path())?;

            // Armed before anything else can fail, and removed only by a clean
            // exit further down. Whatever this returns is the previous life.
            let crashed = runtime::phoenix::arm(&config.runtime_dir())?;
            if let Some(mark) = &crashed {
                warn!("Previous tera {} (started {})", mark.describe(), mark.started_at_ms);
            }
            let update_notice = match tera::update::startup_action(&config, crashed.is_some())? {
                tera::update::StartupAction::Continue(notice) => notice.map(|notice| *notice),
                tera::update::StartupAction::RestartAfterRollback => {
                    runtime::phoenix::disarm(&config.runtime_dir());
                    return Err(anyhow::anyhow!(
                        "the failed update was rolled back; restarting the restored binary"
                    ));
                }
            };

            // 2. Initialize workspace & databases
            WorkspaceInit::init(&config)?;
            let history_db = HistoryDb::open_for(&config)?;
            let runtime_db = RuntimeDb::open(&config.runtime_db_path())?;

            // 2a. Repair what a crash or an older build may have left behind,
            //     before anything is served (PLAN.md section 96.3).
            match backup::clear_stale_staging(&config) {
                Ok(removed) if !removed.is_empty() => {
                    info!("Cleared {} stale staging director(ies)", removed.len())
                }
                Err(e) => warn!("Could not clear stale staging: {:?}", e),
                _ => {}
            }

            if let Err(e) = backup::verify_memories_link(&config) {
                warn!("Active memory is not usable: {:?}", e);
            }

            // The assistant only wakes on a message, so its instruction to look
            // after this machine needs something that fires on its own.
            tera::scheduler::defaults::seed(&runtime_db);

            // The projection is what the agent reads. If it has drifted from
            // canonical history for any reason, regenerate it before serving.
            if let Err(e) = ProjectionEngine::verify_and_repair(
                &config.history_jsonl_dir(),
                &config.runtime_dir().join("tmp"),
                &history_db,
            ) {
                warn!("Could not repair the JSONL projection: {:?}", e);
            }

            // A snapshot per start. Cheap next to what it protects: history is the
            // only thing here that cannot be regenerated.
            match backup::backup_history(&config, &backup::timestamp_now()) {
                Ok(path) => info!("Startup history backup at {:?}", path),
                Err(e) => warn!("Could not back up history at startup: {:?}", e),
            }

            // Shared between the MCP server and the turn engine so the engine can
            // tell whether a turn already replied through the send_message tool.
            let session = ConversationSession::new();

            // Conversation turns and scheduled runs both register here, so memory
            // maintenance can tell when it is safe to run, and get out of the way.
            let activity = ActivityTracker::new();

            // One app-server process, shared by the conversation and the scheduler.
            let codex =
                CodexSupervisor::new(config.clone(), runtime_db.clone(), history_db.clone());
            if update_notice.is_some() {
                // An update is not healthy until the exact app-server protocol
                // this daemon depends on has completed its handshake.
                codex.ensure().await?;
            } else {
                codex.warm_in_background();
            }

            // 3. Initialize transport & spawn bot loop
            let transport: Arc<dyn Transport> = if config.mock_transport {
                warn!("Starting daemon with MockTransport for testing");
                Arc::new(MockTransport::new())
            } else {
                let wa_session_db = config.runtime_dir().join("whatsapp_session.db");
                let wa_transport = Arc::new(WhatsAppWebTransport::new(wa_session_db));
                let wa_clone = wa_transport.clone();

                let turn_engine = Arc::new(TurnEngine::new(
                    config.clone(),
                    history_db.clone(),
                    runtime_db.clone(),
                    wa_transport.clone(),
                    session.clone(),
                    codex.clone(),
                    activity.clone(),
                ));

                // Phoenix speaks before it repairs, and it cannot speak until the
                // transport is up, which happens below. Retrying the whole job is
                // safe: nothing is recorded until the owner has been told.
                let phoenix = Phoenix::new(
                    config.clone(),
                    history_db.clone(),
                    runtime_db.clone(),
                    wa_transport.clone(),
                    codex.clone(),
                    activity.clone(),
                );
                tokio::spawn(async move {
                    for attempt in 1..=PHOENIX_REPORT_ATTEMPTS {
                        match phoenix.run(crashed.clone(), update_notice.clone()).await {
                            Ok(()) => return,
                            Err(e) => {
                                warn!("Phoenix could not run (attempt {attempt}): {e:?}");
                                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                            }
                        }
                    }
                    error!("Phoenix gave up; an interrupted turn is still open in the database");
                });

                // Reconnect rather than going quiet. `start_bot` returns when the
                // bot stops for any reason, and a daemon that keeps running with
                // no transport looks alive while receiving nothing at all.
                tokio::spawn(async move {
                    let mut backoff = std::time::Duration::from_secs(2);
                    loop {
                        let engine_for_run = turn_engine.clone();
                        let result = wa_clone
                            .start_bot(move |event| {
                                let engine = engine_for_run.clone();
                                tokio::spawn(async move {
                                    match event {
                                        InboundEvent::Message(msg) => {
                                            if let Err(err) =
                                                engine.handle_inbound_message(msg).await
                                            {
                                                tracing::error!("TurnEngine error: {:?}", err);
                                            }
                                        }
                                        InboundEvent::Reaction(_) => {}
                                    }
                                });
                            })
                            .await;

                        match result {
                            Ok(()) => warn!("WhatsApp transport stopped; reconnecting in {backoff:?}"),
                            Err(e) => tracing::error!(
                                "WhatsApp transport failed ({e:?}); reconnecting in {backoff:?}"
                            ),
                        }

                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(std::time::Duration::from_secs(120));
                    }
                });

                wa_transport
            };

            // 4. Start MCP RPC server on Unix socket
            let rpc_server = Arc::new(DaemonRpcServer::new(
                config.clone(),
                history_db.clone(),
                runtime_db.clone(),
                transport.clone(),
                session.clone(),
            ));
            tokio::spawn(async move {
                if let Err(e) = rpc_server.run().await {
                    tracing::error!("RPC Server failure: {:?}", e);
                }
            });

            // 5. Start background scheduler runner
            let scheduler_runner = Arc::new(SchedulerRunner::new(
                config.clone(),
                runtime_db.clone(),
                codex.clone(),
                activity.clone(),
            ));
            scheduler_runner.start_loop();

            // 6. Memory maintenance: nightly optimization, and rebuilds after a
            //    model change. Lowest priority in the system; it only runs in an
            //    idle window and abandons its work when anything else starts.
            let maintenance = Arc::new(MaintenanceRunner::new(
                config.clone(),
                history_db.clone(),
                runtime_db.clone(),
                codex.clone(),
                activity.clone(),
            ));
            maintenance.start_loop();

            // Everything required to serve a turn is now running, including a
            // post-update Codex handshake. The rollback copy is no longer needed.
            tera::update::mark_healthy(&config);

            info!("Daemon is fully initialized and operational. Press Ctrl+C to stop.");
            let update_restart = {
                let mut update_signal = tokio::signal::unix::signal(
                    tokio::signal::unix::SignalKind::user_defined1(),
                )?;
                tokio::select! {
                    result = tokio::signal::ctrl_c() => {
                        result?;
                        false
                    }
                    _ = update_signal.recv() => true,
                }
            };

            // Graceful shutdown (PLAN.md section 52). Systemd may restart us, so
            // what matters is leaving no state that a fresh start would misread:
            // no phantom typing indicator, no stale socket.
            info!("Shutting down: clearing typing state and releasing the socket");
            if let Some(chat) = session.chat() {
                if let Err(e) = transport.set_typing_status(&chat, false).await {
                    warn!("Could not clear typing state on shutdown: {:?}", e);
                }
            }
            if let Err(e) = std::fs::remove_file(config.socket_path()) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    warn!("Could not remove the runtime socket: {:?}", e);
                }
            }

            // Last thing, so anything that stops us before here reads as a crash.
            runtime::phoenix::disarm(&config.runtime_dir());
            if update_restart {
                tera::update::spawn_manual_restart(&config, std::process::id())?;
            }
            info!("tera stopped.");
        }
    }

    Ok(())
}
