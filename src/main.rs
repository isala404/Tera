use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;
use tera::codex::CodexSupervisor;
use tera::config::Config;
use tera::conversation::{ConversationSession, Phoenix, TurnEngine};
use tera::history::{backup, HistoryDb, ProjectionEngine};
use tera::mcp::{DaemonRpcServer, StdioMcpProxy};
use tera::runtime::{self, DaemonLock, RuntimeDb};
use tera::scheduler::db::SchedulerDb;
use tera::scheduler::recurrence;
use tera::scheduler::SchedulerRunner;
use tera::secrets::SecretStore;
use tera::transport::{MockTransport, Transport, WhatsAppWebTransport};
use tera::workspace::WorkspaceInit;

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

#[derive(clap::Args, Debug, Clone)]
struct WorkspaceArg {
    #[arg(long, default_value = "/workspace")]
    workspace: PathBuf,
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
        #[command(flatten)]
        workspace: WorkspaceArg,
        #[arg(long, value_enum, default_value_t = UpdateComponent::All)]
        component: UpdateComponent,
        /// Reinstall the current Tera release
        #[arg(long)]
        force: bool,
    },
    /// Start the long-running assistant daemon
    Daemon {
        #[command(flatten)]
        workspace: WorkspaceArg,
        #[arg(long)]
        mock_transport: bool,
    },
    /// Idempotent workspace initialization
    Init {
        #[command(flatten)]
        workspace: WorkspaceArg,
    },
    /// Pair / log in to Codex using device authorization grant
    Login {
        #[command(flatten)]
        workspace: WorkspaceArg,
        /// Force re-pairing even if credentials already exist
        #[arg(long)]
        force: bool,
    },
    /// Stdio MCP server proxy for Codex App Server
    Mcp {
        #[arg(long)]
        socket: PathBuf,
    },
    /// Print system health and state status
    Status {
        #[command(flatten)]
        workspace: WorkspaceArg,
    },
    /// History tools
    History {
        #[command(subcommand)]
        sub: HistorySubcommands,
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
        #[command(flatten)]
        workspace: WorkspaceArg,
    },
    /// Snapshot canonical history into history/backups/
    Backup {
        #[command(flatten)]
        workspace: WorkspaceArg,
    },
    /// Check canonical history, its projection, and its assets
    Check {
        #[command(flatten)]
        workspace: WorkspaceArg,
    },
}

#[derive(Subcommand)]
enum SecretSubcommands {
    /// List stored names and when each was set. Never prints a value.
    List {
        #[command(flatten)]
        workspace: WorkspaceArg,
    },
    /// Store a credential, read from stdin.
    ///
    /// Stdin rather than an argument, so the value does not end up in shell
    /// history or in the process list of every other account on the machine.
    Set {
        #[command(flatten)]
        workspace: WorkspaceArg,
        name: String,
    },
    /// Forget a credential.
    Rm {
        #[command(flatten)]
        workspace: WorkspaceArg,
        name: String,
    },
}

impl SecretSubcommands {
    fn workspace(&self) -> &PathBuf {
        match self {
            SecretSubcommands::List { workspace }
            | SecretSubcommands::Set { workspace, .. }
            | SecretSubcommands::Rm { workspace, .. } => &workspace.workspace,
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
            | Commands::Login { workspace, .. }
            | Commands::Status { workspace }
            | Commands::Update { workspace, .. } => Some(&workspace.workspace),
            Commands::History { sub } => Some(sub.workspace()),
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
            | HistorySubcommands::Check { workspace } => &workspace.workspace,
        }
    }
}

/// One line describing the memory repository for `tera status`.
///
/// Git is the whole story here, so the useful facts are how many commits deep
/// it is and what the last one said. A directory that is not a repository is
/// worth reporting loudly, because nothing will be versioned until it is.
fn memory_status(config: &Config) -> String {
    let dir = config.memories_dir();
    if !dir.join(".git").exists() {
        return format!("{} is not a git repository", dir.display());
    }
    match std::process::Command::new("git")
        .current_dir(&dir)
        .args(["log", "-1", "--format=%h %s (%ar)"])
        .output()
    {
        Ok(out) if out.status.success() => {
            let last = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if last.is_empty() {
                "git repository, no commits yet".to_string()
            } else {
                last
            }
        }
        _ => "git repository, no commits yet".to_string(),
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
            workspace: WorkspaceArg { workspace },
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
                println!(
                    "The daemon will restart in a few seconds. Phoenix will report the result."
                );
            }
        }

        Commands::InternalSignalUpdate { lock, pid } => {
            tera::update::signal_update(&lock, pid)?;
        }

        Commands::Init {
            workspace: WorkspaceArg { workspace },
        } => {
            let config = Config::new(workspace, false);
            WorkspaceInit::init(&config)?;
            let _ = HistoryDb::open_for(&config)?;
            let _ = RuntimeDb::open(&config.runtime_db_path())?;
            info!("Initialization finished successfully!");
        }

        Commands::Login {
            workspace: WorkspaceArg { workspace },
            force,
        } => {
            let config = Config::new(workspace, false);
            WorkspaceInit::login(&config, force)?;
        }

        // Reports what is degraded, not just what is configured: a status command
        // that only echoes paths back cannot tell you why the assistant is quiet.
        Commands::Status {
            workspace: WorkspaceArg { workspace },
        } => {
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
                if dir.is_dir()
                    && !dir
                        .metadata()
                        .map(|m| m.permissions().readonly())
                        .unwrap_or(true)
                {
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
                            if report.sqlite_ok {
                                ""
                            } else {
                                ", SQLITE INTEGRITY FAILED"
                            },
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

            println!("Memory:      {}", memory_status(&config));

            println!(
                "Owner:       {} (set TERA_OWNER to change)",
                config.owner_name
            );
            println!("Model config: {}", config.codex_config_path().display());
            let codex_auth_path = config.codex_home_dir().join("auth.json");
            println!(
                "Codex auth:   {}",
                if codex_auth_path.exists() {
                    "authenticated (auth.json linked)"
                } else {
                    "NOT AUTHENTICATED (run `tera login` to pair)"
                }
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
                        "  {} '{}' next {}",
                        item.id,
                        item.name,
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
            HistorySubcommands::RebuildJsonl {
                workspace: WorkspaceArg { workspace },
            } => {
                let config = Config::new(workspace, false);
                let history_db = HistoryDb::open_for(&config)?;
                ProjectionEngine::rebuild_all(
                    &config.history_jsonl_dir(),
                    &config.runtime_dir().join("tmp"),
                    &history_db,
                )?;
            }

            HistorySubcommands::Backup {
                workspace: WorkspaceArg { workspace },
            } => {
                let config = Config::new(workspace, false);
                let path = backup::backup_history(&config, &backup::timestamp_now())?;
                println!("Backed up history to {}", path.display());
            }

            HistorySubcommands::Check {
                workspace: WorkspaceArg { workspace },
            } => {
                let config = Config::new(workspace, false);
                let history_db = HistoryDb::open_for(&config)?;
                let report = backup::check_integrity(&config, &history_db)?;

                println!(
                    "SQLite integrity:   {}",
                    if report.sqlite_ok { "ok" } else { "FAILED" }
                );
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

        Commands::Secret { sub } => {
            let store =
                SecretStore::new(Config::new(sub.workspace().clone(), false).secrets_path());
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
            workspace: WorkspaceArg { workspace },
            mock_transport,
        } => {
            let config = Config::new(workspace, mock_transport);
            info!("Starting tera daemon...");

            let _lock = DaemonLock::acquire(&config.lock_file_path())?;

            // Armed before anything else can fail, and removed only by a clean
            // exit further down. Whatever this returns is the previous life.
            let crashed = runtime::crash_mark::arm(&config.runtime_dir())?;
            if let Some(mark) = &crashed {
                warn!(
                    "Previous tera {} (started {})",
                    mark.describe(),
                    mark.started_at_ms
                );
            }
            let update_notice = match tera::update::startup_action(&config, crashed.is_some())? {
                tera::update::StartupAction::Continue(notice) => notice.map(|notice| *notice),
                tera::update::StartupAction::RestartAfterRollback => {
                    runtime::crash_mark::disarm(&config.runtime_dir());
                    return Err(anyhow::anyhow!(
                        "the failed update was rolled back; restarting the restored binary"
                    ));
                }
            };

            WorkspaceInit::init(&config)?;
            let history_db = HistoryDb::open_for(&config)?;
            let runtime_db = RuntimeDb::open(&config.runtime_db_path())?;

            // 2a. Repair what a crash may have left behind, before anything is served.
            match backup::clear_stale_scratch(&config) {
                Ok(removed) if !removed.is_empty() => {
                    info!("Cleared {} stale staging director(ies)", removed.len())
                }
                Err(e) => warn!("Could not clear stale staging: {:?}", e),
                _ => {}
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
                ));

                // The startup assistant writes the restart message after the
                // transport and Codex are both usable. It retries while WhatsApp
                // is still connecting.
                let phoenix = Phoenix::new(
                    config.clone(),
                    history_db.clone(),
                    runtime_db.clone(),
                    wa_transport.clone(),
                    codex.clone(),
                    session.clone(),
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
                        let engine_for_presence = turn_engine.clone();
                        let result = wa_clone
                            .start_bot(
                                move |message| {
                                    let engine = engine_for_run.clone();
                                    tokio::spawn(async move {
                                        if let Err(err) =
                                            engine.handle_inbound_message(message).await
                                        {
                                            tracing::error!("TurnEngine error: {:?}", err);
                                        }
                                    });
                                },
                                move |presence| {
                                    let engine = engine_for_presence.clone();
                                    tokio::spawn(async move {
                                        engine.handle_presence(presence).await;
                                    });
                                },
                            )
                            .await;

                        match result {
                            Ok(()) => {
                                warn!("WhatsApp transport stopped; reconnecting in {backoff:?}")
                            }
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

            let scheduler_runner = Arc::new(SchedulerRunner::new(
                config.clone(),
                runtime_db.clone(),
                codex.clone(),
            ));
            scheduler_runner.start_loop();

            // Everything required to serve a turn is now running, including a
            // post-update Codex handshake. The rollback copy is no longer needed.
            tera::update::mark_healthy(&config);

            info!("Daemon is fully initialized and operational. Press Ctrl+C to stop.");
            // Both signals mean the same thing: stop cleanly and let the
            // supervisor start whatever binary is on disk now. SIGUSR1 is what
            // `tera update` sends once the replacement is installed; ctrl-c is a
            // person. Nothing here restarts the daemon itself, because a process
            // that supervises itself is a worse supervisor than the one the OS
            // already runs.
            let mut update_signal =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined1())?;
            let mut terminate_signal =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
            tokio::select! {
                result = tokio::signal::ctrl_c() => result?,
                _ = update_signal.recv() => {}
                _ = terminate_signal.recv() => {}
            }

            // Graceful shutdown. Systemd may restart us, so what matters is
            // leaving no state that a fresh start would misread:
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
            runtime::crash_mark::disarm(&config.runtime_dir());
            info!("tera stopped.");
        }
    }

    Ok(())
}
