//! End to end behaviour against a real Codex app-server and model.
//!
//! The daemon's pieces are wired as in `main.rs`, with the mock transport in
//! place of WhatsApp, so every message and reaction the agent sends is recorded
//! and printed. Ignored by default because it burns real tokens. Run with the
//! provider config the deployed host uses:
//!
//!     TERA_E2E_CODEX_CONFIG=/path/to/config.toml \
//!     cargo test --test e2e_live_test -- --ignored --nocapture --test-threads=1

use chrono::Utc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tera::codex::CodexSupervisor;
use tera::config::Config;
use tera::conversation::phoenix::interrupted_turns;
use tera::conversation::{ConversationSession, Phoenix, TurnEngine};
use tera::history::HistoryDb;
use tera::mcp::DaemonRpcServer;
use tera::runtime::crash_mark::CrashMark;
use tera::runtime::RuntimeDb;
use tera::scheduler::db::{RunState, SchedulerDb};
use tera::scheduler::recurrence::ScheduleTiming;
use tera::scheduler::SchedulerRunner;
use tera::transport::{InboundMessage, MockTransport};
use tera::workspace::WorkspaceInit;

const CHAT: &str = "owner@s.whatsapp.net";
const TURN_LIMIT: Duration = Duration::from_secs(600);

struct Daemon {
    _dir: TempDir,
    config: Config,
    history_db: HistoryDb,
    runtime_db: RuntimeDb,
    transport: Arc<MockTransport>,
    session: ConversationSession,
    codex: CodexSupervisor,
    engine: Arc<TurnEngine>,
}

impl Daemon {
    async fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        Self::start_in(dir).await
    }

    async fn start_in(dir: TempDir) -> Self {
        if std::env::var_os("TERA_BIN").is_none() {
            let deps = std::env::current_exe().unwrap();
            std::env::set_var(
                "TERA_BIN",
                deps.parent().unwrap().parent().unwrap().join("tera"),
            );
        }
        std::env::set_var("TERA_OWNER", "Isala");
        let config = Config::new(dir.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();
        if let Some(provider) = std::env::var_os("TERA_E2E_CODEX_CONFIG") {
            std::fs::copy(provider, config.codex_config_path()).unwrap();
        }

        let history_db = HistoryDb::open_for(&config).unwrap();
        let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
        let transport = Arc::new(MockTransport::new());
        let session = ConversationSession::new();
        let codex = CodexSupervisor::new(config.clone(), runtime_db.clone(), history_db.clone());

        let rpc = Arc::new(DaemonRpcServer::new(
            config.clone(),
            history_db.clone(),
            runtime_db.clone(),
            transport.clone(),
            session.clone(),
        ));
        let listener = rpc.bind().unwrap();
        tokio::spawn(rpc.run_listener(listener));

        let engine = Arc::new(TurnEngine::new(
            config.clone(),
            history_db.clone(),
            runtime_db.clone(),
            transport.clone(),
            session.clone(),
            codex.clone(),
        ));
        Self {
            _dir: dir,
            config,
            history_db,
            runtime_db,
            transport,
            session,
            codex,
            engine,
        }
    }

    fn send(&self, text: &str) {
        let engine = self.engine.clone();
        let msg = InboundMessage {
            provider_msg_id: format!("wa_{}", Utc::now().timestamp_nanos_opt().unwrap()),
            sender: CHAT.to_string(),
            text: Some(text.to_string()),
            timestamp_ms: Utc::now().timestamp_millis(),
            reply_to_provider_msg_id: None,
            media_attachment: None,
            media_error: None,
            chat_jid: CHAT.to_string(),
            from_own_account: true,
            is_group: false,
        };
        tokio::spawn(async move { engine.handle_inbound_message(msg).await.unwrap() });
    }

    async fn settle(&self) {
        let started = Instant::now();
        tokio::time::sleep(Duration::from_secs(5)).await;
        while !self.runtime_db.unfinished_turns().unwrap().is_empty() {
            assert!(started.elapsed() < TURN_LIMIT, "turn did not finish");
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    fn messages(&self) -> Vec<String> {
        let sent = self.transport.sent_messages.lock().unwrap();
        sent.iter().map(|(_, text, _)| text.clone()).collect()
    }

    fn reactions(&self) -> Vec<String> {
        let sent = self.transport.sent_reactions.lock().unwrap();
        sent.iter().map(|(_, _, emoji)| emoji.clone()).collect()
    }
}

/// Every message the agent sent since `from`, printed with seconds since
/// `since`, so the transcript shows pacing as well as wording.
struct Watch {
    start: Instant,
    from: usize,
    stamps: Vec<(f32, String)>,
}

impl Watch {
    fn new(daemon: &Daemon) -> Self {
        Self {
            start: Instant::now(),
            from: daemon.messages().len(),
            stamps: Vec::new(),
        }
    }

    async fn until_settled(mut self, daemon: &Daemon) -> Vec<(f32, String)> {
        let poll = async {
            loop {
                let all = daemon.messages();
                while self.from + self.stamps.len() < all.len() {
                    let text = all[self.from + self.stamps.len()].clone();
                    self.stamps.push((self.start.elapsed().as_secs_f32(), text));
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        };
        tokio::select! {
            _ = poll => unreachable!(),
            _ = daemon.settle() => {}
        }
        let all = daemon.messages();
        while self.from + self.stamps.len() < all.len() {
            let text = all[self.from + self.stamps.len()].clone();
            self.stamps.push((self.start.elapsed().as_secs_f32(), text));
        }
        self.stamps
    }
}

fn print(scenario: &str, said: &str, stamps: &[(f32, String)], reactions: &[String]) {
    println!("\n=== {scenario}\n> {said}");
    for emoji in reactions {
        println!("  [react {emoji}]");
    }
    for (at, text) in stamps {
        println!("  +{at:>5.1}s  {}", text.replace('\n', " / "));
    }
}

fn is_only_emoji(text: &str) -> bool {
    let text = text.trim();
    !text.is_empty() && !text.chars().any(|c| c.is_alphanumeric())
}

fn no_markdown(stamps: &[(f32, String)]) {
    for (_, text) in stamps {
        assert!(!text.contains("**"), "markdown bold: {text}");
        assert!(
            !text
                .lines()
                .any(|l| l.starts_with("- ") || l.starts_with('#')),
            "markdown list or heading: {text}"
        );
    }
}

#[tokio::test]
#[ignore = "spawns a real codex app-server and consumes account tokens"]
async fn e2e_a_clean_idle_restart_says_nothing() {
    let daemon = Daemon::start().await;
    daemon
        .runtime_db
        .start_turn("turn_old", CHAT, "wa_old")
        .unwrap();
    daemon
        .runtime_db
        .finish_turn("turn_old", tera::runtime::TurnState::Completed)
        .unwrap();

    let interrupted = interrupted_turns(&daemon.runtime_db).unwrap();
    phoenix(&daemon)
        .run(None, None, &interrupted)
        .await
        .unwrap();

    print(
        "clean idle restart",
        "(daemon restarted, nothing in flight)",
        &[],
        &daemon.reactions(),
    );
    assert!(daemon.messages().is_empty(), "{:?}", daemon.messages());
    assert!(daemon.reactions().is_empty());
}

#[tokio::test]
#[ignore = "spawns a real codex app-server and consumes account tokens"]
async fn e2e_conversation_acks_reacts_browses_and_finds_its_version() {
    let daemon = Daemon::start().await;

    let said =
        "can you look up the latest stable Rust release on the web and tell me when it came out";
    daemon.send(said);
    let web = Watch::new(&daemon).until_settled(&daemon).await;
    print("web lookup", said, &web, &[]);
    assert!(web.len() >= 2, "expected an ack and an answer");
    assert!(
        web[0].0 + 5.0 < web.last().unwrap().0,
        "the ack came with the answer"
    );
    no_markdown(&web);

    let said = "thanks, perfect";
    let reactions_before = daemon.reactions().len();
    daemon.send(said);
    let thanks = Watch::new(&daemon).until_settled(&daemon).await;
    let reactions = daemon.reactions()[reactions_before..].to_vec();
    print("thanks", said, &thanks, &reactions);
    if !reactions.is_empty() {
        assert!(
            !thanks.iter().any(|(_, text)| is_only_emoji(text)),
            "reacted and sent an emoji message"
        );
    }

    let said = "which version of tera are you running right now?";
    daemon.send(said);
    let version = Watch::new(&daemon).until_settled(&daemon).await;
    print("version", said, &version, &[]);
    assert!(
        version
            .iter()
            .any(|(_, text)| text.contains(env!("CARGO_PKG_VERSION"))),
        "the self-update script did not report the running version"
    );
    no_markdown(&version);
}

#[tokio::test]
#[ignore = "spawns a real codex app-server and consumes account tokens"]
async fn e2e_a_crash_mid_task_is_owned_and_finished() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    // The first life runs on its own runtime so it can die mid turn the way a
    // killed process does: every task dropped, the app-server's stdin closed.
    let said = "go through the folders in this workspace and tell me in a few lines what each top level one is for";
    let (dir, first_life) = std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (daemon, before_crash) = rt.block_on(async {
            let daemon = Daemon::start_in(dir).await;
            daemon.send(said);
            tokio::time::sleep(Duration::from_secs(8)).await;
            let before = daemon.messages();
            assert!(
                !daemon.runtime_db.unfinished_turns().unwrap().is_empty(),
                "the turn finished before the crash"
            );
            (daemon, before)
        });
        let Daemon { _dir, .. } = daemon;
        rt.shutdown_background();
        (_dir, before_crash)
    })
    .join()
    .unwrap();
    println!("\n=== crash mid task\n> {said}");
    for text in &first_life {
        println!("  before crash  {}", text.replace('\n', " / "));
    }
    assert_eq!(dir.path(), path);

    let daemon = Daemon::start_in(dir).await;
    let interrupted = interrupted_turns(&daemon.runtime_db).unwrap();
    assert_eq!(interrupted.len(), 1, "the cut off turn should be open");
    let crash = CrashMark {
        started_at_ms: Utc::now().timestamp_millis() - 60_000,
        panic: None,
        consecutive: 0,
    };
    let watch = Watch::new(&daemon);
    let runner = phoenix(&daemon);
    let recovered = async { runner.run(Some(crash), None, &interrupted).await.unwrap() };
    let (stamps, ()) = tokio::join!(watch.until_settled(&daemon), recovered);
    print("after restart", "(phoenix)", &stamps, &daemon.reactions());
    assert!(
        !stamps.is_empty(),
        "the interrupted request was never answered"
    );
    for (_, text) in &stamps {
        let lower = text.to_lowercase();
        assert!(
            !lower.contains("i'm back") && !lower.contains("im back"),
            "{text}"
        );
    }
    no_markdown(&stamps);
}

#[tokio::test]
#[ignore = "spawns a real codex app-server and consumes account tokens"]
async fn e2e_a_daily_check_with_nothing_new_stays_quiet() {
    let daemon = Daemon::start().await;
    daemon.session.set_chat(CHAT);
    let runner = SchedulerRunner::new(
        daemon.config.clone(),
        daemon.runtime_db.clone(),
        daemon.history_db.clone(),
        daemon.codex.clone(),
        daemon.session.clone(),
    );
    let item = SchedulerDb::create_schedule(
        &daemon.runtime_db,
        "Morning machine check",
        "Check disk headroom and memory on this machine and how this workspace's disk use looks, then let Isala know how the machine is doing.",
        &ScheduleTiming::Recurring {
            rrule: "30 9 * * *".to_string(),
        },
        "tasks/morning-check",
    )
    .unwrap();

    for run in ["first run", "second run, nothing changed"] {
        let before = daemon.messages().len();
        let started = Instant::now();
        runner.run_schedule(&item).await.unwrap();
        let last = &SchedulerDb::recent_runs(&daemon.runtime_db, 1).unwrap()[0];
        assert_eq!(last.state, RunState::Completed, "{:?}", last.error);
        let sent = daemon.messages()[before..].to_vec();
        let stamps: Vec<_> = sent
            .into_iter()
            .map(|t| (started.elapsed().as_secs_f32(), t))
            .collect();
        print(run, "(scheduled morning check)", &stamps, &[]);
        if run.starts_with("second") {
            assert!(stamps.is_empty(), "repeated the same news");
        }
    }
}

fn phoenix(daemon: &Daemon) -> Phoenix {
    Phoenix::new(
        daemon.config.clone(),
        daemon.history_db.clone(),
        daemon.runtime_db.clone(),
        daemon.transport.clone(),
        daemon.codex.clone(),
        daemon.session.clone(),
    )
}

#[tokio::test]
#[ignore = "spawns a real codex app-server and consumes account tokens"]
async fn e2e_a_person_mentioned_in_passing_lands_in_memory() {
    let daemon = Daemon::start().await;
    let memories = daemon.config.workspace_dir.join("MEMORIES");
    let commits = || {
        let out = std::process::Command::new("git")
            .args(["rev-list", "--count", "HEAD"])
            .current_dir(&memories)
            .output()
            .unwrap();
        String::from_utf8(out.stdout)
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap()
    };
    let before = commits();

    let said =
        "my sister Nethmi's birthday is on March 12, she's into orchids. remind me a week before";
    daemon.send(said);
    let stamps = Watch::new(&daemon).until_settled(&daemon).await;
    print("learning", said, &stamps, &daemon.reactions());

    let remembered = walkdir(&memories)
        .into_iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
        .map(|path| std::fs::read_to_string(path).unwrap().to_lowercase())
        .any(|text| text.contains("nethmi") && text.contains("orchid"));
    assert!(remembered, "Nethmi and the orchids never reached MEMORIES");
    assert!(commits() > before, "memory changed without a commit");
    let schedules = SchedulerDb::list_schedules(&daemon.runtime_db).unwrap();
    assert!(!schedules.is_empty(), "no reminder was scheduled");
    println!(
        "  memory commits {before} -> {}, schedules {:?}",
        commits(),
        schedules.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
}

#[tokio::test]
#[ignore = "spawns a real codex app-server and consumes account tokens"]
async fn e2e_the_nightly_pass_learns_the_day_and_compacts_memory() {
    let daemon = Daemon::start().await;
    daemon.session.set_chat(CHAT);
    let memories = daemon.config.workspace_dir.join("MEMORIES");
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(&memories)
            .output()
            .unwrap();
        String::from_utf8(out.stdout).unwrap()
    };

    // Memory the way a weaker pass leaves it: headings, bold, the same fact
    // twice, a fix story and a loop that closed weeks ago.
    std::fs::write(
        memories.join("USER.md"),
        "# User Profile\n\n## Basics\n\n- **Name:** Isala\n- **Location:** Colombo, Sri Lanka\n- Isala lives in Colombo.\n\n## Preferences\n\n- **Coffee:** likes it black, no sugar\n",
    )
    .unwrap();
    std::fs::write(
        memories.join("HORIZON.md"),
        "# Horizon\n\n## Open\n\n- **2026-10-20** dentist appointment at 4pm\n\n## Done\n\n- ~~Fix the printer driver~~ resolved on 2026-09-02 after reinstalling cups\n",
    )
    .unwrap();
    std::fs::write(
        memories.join("INFRA.md"),
        "# Infrastructure Notes\n\n## Network\n\n- The router is a TP-Link Archer AX55.\n- **Router:** TP-Link Archer AX55 (confirmed)\n\n## Incident log\n\nOn 2026-09-14 the backups failed. I checked the logs, found the disk was full, deleted old snapshots, reran the job and it worked. Then I verified it again the next morning and it was still fine.\n",
    )
    .unwrap();
    git(&["add", "-A"]);
    git(&["commit", "-q", "-m", "Seed sloppy memory"]);
    let words = || -> usize {
        walkdir(&memories)
            .iter()
            .map(|path| {
                std::fs::read_to_string(path)
                    .unwrap_or_default()
                    .split_whitespace()
                    .count()
            })
            .sum()
    };
    let before = words();

    // A day of chat the day's turns never saved anything from.
    let now = Utc::now().timestamp_millis();
    let day = [
        "my colleague Kasun handles the office VPN, he only answers on Signal",
        "convert this bank statement pdf into a csv for me",
        "please stop using bullet lists when you text me, plain sentences only",
        "another bank statement, same thing, pdf to csv please",
        "and the savings account statement too, csv again",
    ];
    for (i, text) in day.iter().enumerate() {
        let at = now - (day.len() - i) as i64 * 3_600_000;
        for (actor, said) in [
            ("user", text.to_string()),
            ("assistant", "done".to_string()),
        ] {
            daemon
                .history_db
                .insert_event(tera::history::db::ConversationEvent::message(
                    format!("msg_{}", uuid::Uuid::new_v4().simple()),
                    at,
                    actor,
                    Some(said),
                    None,
                    Some(format!("turn_day_{i}")),
                    vec![],
                ))
                .unwrap();
        }
    }

    tera::scheduler::defaults::seed(&daemon.runtime_db);
    let item = SchedulerDb::list_schedules(&daemon.runtime_db)
        .unwrap()
        .into_iter()
        .find(|item| item.name == "Memory compaction")
        .unwrap();
    let runner = SchedulerRunner::new(
        daemon.config.clone(),
        daemon.runtime_db.clone(),
        daemon.history_db.clone(),
        daemon.codex.clone(),
        daemon.session.clone(),
    );
    let started = Instant::now();
    runner.run_schedule(&item).await.unwrap();
    let last = &SchedulerDb::recent_runs(&daemon.runtime_db, 1).unwrap()[0];
    assert_eq!(last.state, RunState::Completed, "{:?}", last.error);
    let sent: Vec<_> = daemon
        .messages()
        .into_iter()
        .map(|t| (started.elapsed().as_secs_f32(), t))
        .collect();
    print("nightly", "(nightly memory pass)", &sent, &[]);

    let tree: Vec<(String, String)> = walkdir(&memories)
        .into_iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
        .map(|path| {
            let name = path.strip_prefix(&memories).unwrap().display().to_string();
            (name, std::fs::read_to_string(path).unwrap())
        })
        .collect();
    for (name, text) in &tree {
        println!("  --- {name}\n{text}");
    }
    let all = tree
        .iter()
        .map(|(_, text)| text.to_lowercase())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        git(&["log", "-1", "--format=%s"]).starts_with("Nightly"),
        "no Nightly commit"
    );
    assert!(
        all.contains("kasun") && all.contains("signal"),
        "missed the new person"
    );
    assert!(
        ["bullet", "plain sentence", "lists"]
            .iter()
            .any(|word| all.contains(word)),
        "missed the corrected preference"
    );
    assert!(all.contains("dentist"), "dropped a live open loop");
    assert!(!all.contains("printer"), "kept a closed loop");
    assert!(!all.contains("snapshots"), "kept a fix story");
    for (name, text) in &tree {
        assert!(
            !text.lines().any(|line| line.starts_with('#'))
                && !text.contains("**")
                && !text.contains("]("),
            "{name} still has markdown"
        );
    }
    let after = words();
    println!("  words {before} -> {after}, messages {}", sent.len());
    assert!(after < before, "memory grew");
    assert!(sent.len() <= 1, "the pass may send at most one message");
}

fn walkdir(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.file_name().is_some_and(|name| name == ".git") {
            continue;
        }
        if path.is_dir() {
            found.extend(walkdir(&path));
        } else {
            found.push(path);
        }
    }
    found
}
