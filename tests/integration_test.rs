use chrono::Utc;
use rusqlite::Connection;
use std::fs;
use std::sync::Arc;
use tempfile::TempDir;
use tera::codex::CodexSupervisor;
use tera::config::Config;
use tera::conversation::{ConversationSession, TurnEngine};
use tera::history::db::{ConversationEvent, HistoryDb};
use tera::history::projection::ProjectionEngine;
use tera::runtime::RuntimeDb;
use tera::scheduler::db::{RunState, ScheduleStatus, SchedulerDb};
use tera::scheduler::recurrence::ScheduleTiming;
use tera::scheduler::SchedulerRunner;
use tera::secrets::SecretStore;
use tera::transport::InboundMessage;
use tera::workspace::init::WorkspaceInit;

#[tokio::test]
async fn test_workspace_init() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);

    WorkspaceInit::init(&config).unwrap();

    assert!(config.workspace_dir.join("AGENTS.md").exists());
    assert!(config.projects_dir().join("AGENTS.md").exists());
    assert!(config.tasks_dir().join("AGENTS.md").exists());
    assert!(config.codex_home_dir().join("config.toml").exists());
    assert!(!tera::data::BUILTIN_SKILLS.is_empty());
    for skill in tera::data::BUILTIN_SKILLS {
        for file in skill.files {
            assert!(config
                .builtin_skills_dir()
                .join(skill.name)
                .join(file.relative_path)
                .exists());
        }
    }
    assert!(config.memories_dir().exists());
    assert!(config.memories_dir().join("INDEX.md").exists());
}

/// The agent searches history with `sqlite3` against `conversation_fts`, not
/// through a Rust tool. What has to hold is that the index is populated by the
/// insert trigger and joinable back to the events. Exactly the query shape the
/// generated SCHEMA.md hands the agent.
#[tokio::test]
async fn test_fts_index_is_queryable_the_way_the_agent_queries_it() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    history_db
        .insert_event(ConversationEvent::message(
            "m_test1",
            Utc::now().timestamp_millis(),
            "user",
            Some("OpenChoreo deployment setup in progress".to_string()),
            None,
            Some("turn_1".to_string()),
            vec![],
        ))
        .unwrap();

    let retrieved = history_db.get_event("m_test1").unwrap().unwrap();
    assert_eq!(
        retrieved.text().unwrap(),
        "OpenChoreo deployment setup in progress"
    );

    let conn = Connection::open(config.history_db_path()).unwrap();
    let found: String = conn
        .query_row(
            "SELECT e.id FROM conversation_fts f
               JOIN conversation_events e ON e.id = f.event_id
              WHERE conversation_fts MATCH ?1",
            ["OpenChoreo"],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(found, "m_test1");
}

#[tokio::test]
async fn test_jsonl_projection_and_rebuild() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();

    for (id, actor, text) in [
        ("m_test2", "user", "Meeting at 3pm"),
        ("m_test3", "assistant", "Noted, 3pm."),
    ] {
        history_db
            .insert_event(ConversationEvent::message(
                id,
                Utc::now().timestamp_millis(),
                actor,
                Some(text.to_string()),
                None,
                Some("turn_2".to_string()),
                vec![],
            ))
            .unwrap();
    }

    // Inserting is enough; the projection is not a separate step callers can skip.
    let projected = read_projection(&config);
    assert_eq!(projected.len(), 2, "{projected:?}");

    // A rebuild is deterministic: same events in, same lines out.
    ProjectionEngine::rebuild_all(
        &config.history_jsonl_dir(),
        &config.runtime_dir().join("tmp"),
        &history_db,
    )
    .unwrap();
    assert_eq!(read_projection(&config), projected);
}

/// The drift that shipped: events written straight to SQLite by an older build,
/// with no projection record. A start must notice and repair it.
#[tokio::test]
async fn test_start_repairs_a_projection_that_drifted() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    history_db
        .insert_event(ConversationEvent::message(
            "m_kept",
            Utc::now().timestamp_millis(),
            "assistant",
            Some("I replied through the tool".to_string()),
            None,
            None,
            vec![],
        ))
        .unwrap();

    // Simulate the old behaviour: canonical event present, projection empty.
    for path in fs::read_dir(config.history_jsonl_dir()).unwrap() {
        fs::remove_file(path.unwrap().path()).unwrap();
    }
    assert_eq!(
        ProjectionEngine::projected_line_count(&config.history_jsonl_dir()).unwrap(),
        0
    );

    ProjectionEngine::verify_and_repair(
        &config.history_jsonl_dir(),
        &config.runtime_dir().join("tmp"),
        &history_db,
    )
    .unwrap();

    assert_eq!(read_projection(&config).len(), 1);
}

/// A rebuild must not leave behind a month file whose events are gone from
/// canonical history, the agent would read it as history that happened.
#[tokio::test]
async fn test_rebuild_drops_stale_month_files() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    let orphan = config.history_jsonl_dir().join("1999-01.jsonl");
    fs::write(&orphan, "{\"id\":\"m_ghost\",\"from\":\"user\"}\n").unwrap();

    ProjectionEngine::rebuild_all(
        &config.history_jsonl_dir(),
        &config.runtime_dir().join("tmp"),
        &history_db,
    )
    .unwrap();

    assert!(!orphan.exists());
}

fn read_projection(config: &Config) -> Vec<String> {
    let mut lines = Vec::new();
    let mut files: Vec<_> = fs::read_dir(config.history_jsonl_dir())
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    for path in files {
        lines.extend(
            fs::read_to_string(path)
                .unwrap()
                .lines()
                .map(str::to_string),
        );
    }
    lines
}

#[tokio::test]
async fn test_scheduler_persistence() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();

    // Databases created by the tier implementation keep this nullable column.
    // New code must ignore it without requiring a destructive migration.
    runtime_db
        .conn
        .lock()
        .unwrap()
        .execute("ALTER TABLE schedules ADD COLUMN tier TEXT", [])
        .unwrap();

    // Asserted as a delta: the daemon seeds a machine-health schedule at startup,
    // and this test should keep passing if that ever moves into workspace init.
    let before = SchedulerDb::list_schedules(&runtime_db).unwrap().len();

    let timing = ScheduleTiming::parse(
        &serde_json::json!({"type": "recurring", "rrule": "EVERY_24H"}),
        Utc::now().timestamp_millis(),
    )
    .unwrap();

    let item = SchedulerDb::create_schedule(
        &runtime_db,
        "Daily Report",
        "Generate status summary",
        &timing,
        "tasks/schedule-test",
    )
    .unwrap();

    let list = SchedulerDb::list_schedules(&runtime_db).unwrap();
    assert_eq!(list.len(), before + 1);
    assert!(list.iter().any(|s| s.id == item.id));

    let cancelled = SchedulerDb::cancel_schedule(&runtime_db, &item.id).unwrap();
    assert!(cancelled);

    let list_after = SchedulerDb::list_schedules(&runtime_db).unwrap();
    assert_eq!(list_after.len(), before);
}

/// Memory is versioned by git, and the commits are tera's own.
///
/// The identity matters: these commits are the assistant editing its notes, and
/// signing them with the owner's global git identity would be a lie about who
/// wrote them. It is set on the repository so nothing outside the workspace is
/// touched.
#[tokio::test]
async fn test_memory_is_a_git_repository_tera_owns() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let dir = config.memories_dir();
    assert!(dir.join(".git").exists());

    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .current_dir(&dir)
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?} failed");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };

    assert_eq!(git(&["config", "user.name"]), "Tera");
    assert_eq!(git(&["config", "user.email"]), "tera@localhost");

    // The seed is committed, so the very first edit the agent makes has a
    // parent to diff against and can be reverted.
    assert_eq!(git(&["status", "--porcelain"]), "");
    assert!(git(&["log", "--format=%an", "-1"]) == "Tera");

    // Init runs on every daemon start, so it has to be safe to repeat.
    WorkspaceInit::init(&config).unwrap();
    assert_eq!(git(&["rev-list", "--count", "HEAD"]), "1");
}

/// A credential typed into the chat must never reach canonical history.
///
/// This is the whole point of `tera::secrets`, and it holds only because the
/// capture happens before the insert. That ordering is one line in
/// `TurnEngine::handle_inbound_message` and nothing about moving it would look
/// wrong in review, so the guarantee is pinned here rather than described in a
/// comment. History is the strictest place to check: everything else the model
/// ever sees, the JSONL projection, a resumed thread, a rebuilt memory
/// generation, is derived from it.
#[tokio::test]
async fn test_a_secret_sent_through_chat_never_lands_in_history() {
    const VALUE: &str = "65b708073fc0480ea92a077233ca87bd";

    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
    let transport = std::sync::Arc::new(tera::transport::MockTransport::new());
    let engine = TurnEngine::new(
        config.clone(),
        history_db.clone(),
        runtime_db.clone(),
        transport.clone(),
        ConversationSession::new(),
        CodexSupervisor::new(config.clone(), runtime_db, history_db.clone()),
    );

    engine
        .handle_inbound_message(InboundMessage {
            provider_msg_id: "wa_1".to_string(),
            sender: "owner@s.whatsapp.net".to_string(),
            text: Some(format!("/secret SPOTIFY_CLIENT_ID {VALUE}")),
            timestamp_ms: Utc::now().timestamp_millis(),
            reply_to_provider_msg_id: None,
            media_attachment: None,
            media_error: None,
            chat_jid: "owner@s.whatsapp.net".to_string(),
            // No explicit owner is configured here, so the policy accepts the
            // paired account and nobody else.
            from_own_account: true,
            is_group: false,
        })
        .await
        .unwrap();

    let events = history_db.list_events_all().unwrap();
    assert_eq!(events.len(), 1, "the message should still be recorded");
    let recorded = events[0].text().unwrap();
    assert!(
        !recorded.contains(VALUE),
        "history holds the secret: {recorded}"
    );
    assert!(
        recorded.contains("SPOTIFY_CLIENT_ID"),
        "the agent still has to learn which credential arrived: {recorded}"
    );

    // And the value went somewhere a skill can actually read it.
    let store = SecretStore::new(config.secrets_path());
    assert_eq!(
        store.get("SPOTIFY_CLIENT_ID").unwrap().as_deref(),
        Some(VALUE)
    );
}

#[tokio::test]
async fn test_a_buffered_message_activates_typing_for_the_canonical_chat() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
    let transport = std::sync::Arc::new(tera::transport::MockTransport::new());
    let engine = TurnEngine::new(
        config.clone(),
        history_db.clone(),
        runtime_db.clone(),
        transport.clone(),
        ConversationSession::new(),
        CodexSupervisor::new(config, runtime_db, history_db),
    );

    engine
        .handle_inbound_message(InboundMessage {
            provider_msg_id: "wa_typing".to_string(),
            sender: "owner:26@s.whatsapp.net".to_string(),
            text: Some("hello".to_string()),
            timestamp_ms: Utc::now().timestamp_millis(),
            reply_to_provider_msg_id: None,
            media_attachment: None,
            media_error: None,
            chat_jid: "owner@s.whatsapp.net".to_string(),
            from_own_account: true,
            is_group: false,
        })
        .await
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(
        transport.typing_states.lock().unwrap().first(),
        Some(&("owner@s.whatsapp.net".to_string(), true))
    );
}

/// WhatsApp replays protocol traffic through the same callback as real chat when
/// a device is linked, and those messages carry neither text nor an attachment.
/// A burst of them opened a turn on the paired account's own chat once, and the
/// owner's first real message was steered into it and answered there. Nothing
/// with nothing in it may reach history or start a turn.
#[tokio::test]
async fn test_an_empty_message_is_neither_recorded_nor_answered() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
    let transport = std::sync::Arc::new(tera::transport::MockTransport::new());
    let engine = TurnEngine::new(
        config.clone(),
        history_db.clone(),
        runtime_db.clone(),
        transport.clone(),
        ConversationSession::new(),
        CodexSupervisor::new(config, runtime_db, history_db.clone()),
    );

    for (id, text) in [("wa_sync_1", None), ("wa_sync_2", Some("   ".to_string()))] {
        engine
            .handle_inbound_message(InboundMessage {
                provider_msg_id: id.to_string(),
                sender: "owner@s.whatsapp.net".to_string(),
                text,
                timestamp_ms: Utc::now().timestamp_millis(),
                reply_to_provider_msg_id: None,
                media_attachment: None,
                media_error: None,
                chat_jid: "owner@s.whatsapp.net".to_string(),
                from_own_account: true,
                is_group: false,
            })
            .await
            .unwrap();
    }

    // Long enough that a burst opened by these would have fired its turn.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    assert!(
        history_db.list_events_all().unwrap().is_empty(),
        "an empty message reached canonical history"
    );
    assert!(
        transport.typing_states.lock().unwrap().is_empty(),
        "an empty message started a turn"
    );
}

#[tokio::test]
async fn test_media_download_failure_reports_to_owner() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
    let transport = std::sync::Arc::new(tera::transport::MockTransport::new());
    let engine = TurnEngine::new(
        config.clone(),
        history_db.clone(),
        runtime_db.clone(),
        transport.clone(),
        ConversationSession::new(),
        CodexSupervisor::new(config, runtime_db, history_db.clone()),
    );

    // 1. File-only message with failed download
    engine
        .handle_inbound_message(InboundMessage {
            provider_msg_id: "wa_fail_1".to_string(),
            sender: "owner@s.whatsapp.net".to_string(),
            text: None,
            timestamp_ms: Utc::now().timestamp_millis(),
            reply_to_provider_msg_id: None,
            media_attachment: None,
            media_error: Some("connection reset by peer".to_string()),
            chat_jid: "owner@s.whatsapp.net".to_string(),
            from_own_account: true,
            is_group: false,
        })
        .await
        .unwrap();

    {
        let sent = transport.sent_messages.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert!(sent[0]
            .1
            .contains("Could not download attachment: connection reset by peer"));
    }

    let events = history_db.list_events_all().unwrap();
    assert_eq!(events.len(), 2);
    assert!(events[0]
        .text()
        .unwrap()
        .contains("[Attachment download failed: connection reset by peer]"));

    // 2. Captioned message with failed download
    engine
        .handle_inbound_message(InboundMessage {
            provider_msg_id: "wa_fail_2".to_string(),
            sender: "owner@s.whatsapp.net".to_string(),
            text: Some("Summarize this invoice".to_string()),
            timestamp_ms: Utc::now().timestamp_millis(),
            reply_to_provider_msg_id: None,
            media_attachment: None,
            media_error: Some("file corrupted".to_string()),
            chat_jid: "owner@s.whatsapp.net".to_string(),
            from_own_account: true,
            is_group: false,
        })
        .await
        .unwrap();

    {
        let sent = transport.sent_messages.lock().unwrap();
        assert_eq!(sent.len(), 2);
        assert!(sent[1]
            .1
            .contains("Could not download attachment: file corrupted"));
    }

    let events = history_db.list_events_all().unwrap();
    assert!(events.iter().any(|e| e
        .text()
        .unwrap_or("")
        .contains("Summarize this invoice\n\n[Attachment download failed: file corrupted]")));
}

#[tokio::test]
async fn test_unauthorized_senders_and_groups_produce_zero_history() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
    let transport = std::sync::Arc::new(tera::transport::MockTransport::new());
    let engine = TurnEngine::new(
        config.clone(),
        history_db.clone(),
        runtime_db.clone(),
        transport.clone(),
        ConversationSession::new(),
        CodexSupervisor::new(config, runtime_db, history_db.clone()),
    );

    // Non-owner sender
    engine
        .handle_inbound_message(InboundMessage {
            provider_msg_id: "wa_stranger".to_string(),
            sender: "stranger@s.whatsapp.net".to_string(),
            text: Some("malicious prompt".to_string()),
            timestamp_ms: Utc::now().timestamp_millis(),
            reply_to_provider_msg_id: None,
            media_attachment: None,
            media_error: None,
            chat_jid: "stranger@s.whatsapp.net".to_string(),
            from_own_account: false,
            is_group: false,
        })
        .await
        .unwrap();

    // Group chat
    engine
        .handle_inbound_message(InboundMessage {
            provider_msg_id: "wa_group".to_string(),
            sender: "owner@s.whatsapp.net".to_string(),
            text: Some("group message".to_string()),
            timestamp_ms: Utc::now().timestamp_millis(),
            reply_to_provider_msg_id: None,
            media_attachment: None,
            media_error: None,
            chat_jid: "12345@g.us".to_string(),
            from_own_account: true,
            is_group: true,
        })
        .await
        .unwrap();

    assert!(history_db.list_events_all().unwrap().is_empty());
    assert!(transport.sent_messages.lock().unwrap().is_empty());
}

#[test]
fn test_linux_service_lifecycle_and_signal_shutdown() {
    let content =
        fs::read_to_string("deploy/tera.service").expect("deploy/tera.service must exist");
    assert!(
        !content.contains("PLAN.md"),
        "service unit contains obsolete PLAN.md reference"
    );
    assert!(content.contains("[Unit]"));
    assert!(content.contains("[Service]"));
    assert!(content.contains("[Install]"));
    assert!(
        content.contains("ExecStart=%h/.local/bin/tera daemon --workspace %h/assistant-workspace")
    );
    assert!(content.contains("WorkingDirectory=%h"));
    assert!(content.contains("Restart=always"));
    assert!(content.contains("Type=simple"));
    assert!(content.contains("KillSignal="));

    let temp_dir = TempDir::new().unwrap();
    let workspace_dir = temp_dir.path().to_path_buf();
    let config = Config::new(workspace_dir.clone(), true);

    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_tera"));
    cmd.arg("daemon")
        .arg("--workspace")
        .arg(&workspace_dir)
        .arg("--mock-transport")
        .env("RUST_LOG", "info");

    let mut child = cmd.spawn().expect("failed to spawn tera daemon binary");
    let pid = child.id() as i32;

    // Poll until config.socket_path() exists (socket is bound and daemon is ready)
    let start = std::time::Instant::now();
    let mut ready = false;
    while start.elapsed() < std::time::Duration::from_secs(10) {
        if config.socket_path().exists() {
            ready = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(ready, "Daemon socket was not created within 10 seconds");
    assert!(
        config.lock_file_path().exists(),
        "Lock file must exist while daemon is running"
    );

    // Send SIGTERM to the daemon
    #[cfg(unix)]
    {
        let ret = unsafe { libc::kill(pid, libc::SIGTERM) };
        assert_eq!(ret, 0, "kill SIGTERM should succeed");
    }

    // Wait for the child process to exit gracefully
    let status = child.wait().expect("child wait should succeed");
    assert!(
        status.success(),
        "Daemon should exit cleanly (status 0) on SIGTERM"
    );

    // Assert socket and lock file are cleaned up
    assert!(
        !config.socket_path().exists(),
        "Runtime socket must be removed on clean shutdown"
    );
    assert!(
        !config.lock_file_path().exists(),
        "Daemon lock file must be removed on clean shutdown"
    );
}

#[tokio::test]
async fn test_fake_app_server_deterministic_turn_and_fault_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("fake_codex.sh");
    let script_content = r#"#!/usr/bin/env python3
import sys, json, time

for line in sys.stdin:
    if not line.strip():
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    method = req.get("method")
    req_id = req.get("id")
    if method == "getAuthStatus":
        sys.stdout.write(json.dumps({"id": req_id, "result": {"requiresOpenaiAuth": False, "authMethod": "chatgpt"}}) + "\n")
        sys.stdout.flush()
    elif method == "thread/start":
        sys.stdout.write(json.dumps({"id": req_id, "result": {"thread": {"id": "fake_th_1", "model": "fake-model"}}}) + "\n")
        sys.stdout.flush()
    elif method == "turn/start":
        prompt_str = json.dumps(req)
        if "INJECT_FAULT_CRASH" in prompt_str:
            sys.exit(42)
        if "SLOW_TURN" in prompt_str:
            sys.stdout.write(json.dumps({"id": req_id, "result": {"turn": {"id": "fake_turn_slow"}}}) + "\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({"id": req_id, "result": {"turn": {"id": "fake_turn_1"}}}) + "\n")
        sys.stdout.write(json.dumps({"method": "turn/started", "params": {"threadId": "fake_th_1", "turn": {"id": "fake_turn_1"}}}) + "\n")
        sys.stdout.write(json.dumps({"method": "item/started", "params": {"threadId": "fake_th_1", "item": {"type": "agentMessage", "id": "msg_1"}}}) + "\n")
        sys.stdout.write(json.dumps({"method": "item/agentMessage/delta", "params": {"threadId": "fake_th_1", "delta": "Deterministic fake response"}}) + "\n")
        sys.stdout.write(json.dumps({"method": "item/completed", "params": {"threadId": "fake_th_1", "item": {"type": "agentMessage", "id": "msg_1", "text": "Deterministic fake response"}}}) + "\n")
        sys.stdout.write(json.dumps({"method": "turn/completed", "params": {"threadId": "fake_th_1"}}) + "\n")
        sys.stdout.flush()
    elif req_id is not None:
        sys.stdout.write(json.dumps({"id": req_id, "result": {}}) + "\n")
        sys.stdout.flush()
"#;

    fs::write(&script_path, script_content).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&script_path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script_path, perms).unwrap();
    }

    let mut config = Config::new(temp_dir.path().to_path_buf(), true);
    config.codex_bin = script_path.clone();
    WorkspaceInit::init(&config).unwrap();

    let mgr = tera::codex::CodexProcessManager::spawn_for(&config)
        .await
        .expect("fake app-server should spawn");

    let opts = tera::codex::ThreadOptions::new(&config.workspace_dir);
    let info = mgr.start_thread(&opts).await.expect("thread should start");
    assert_eq!(info.id, "fake_th_1");

    let reply = tokio::time::timeout(std::time::Duration::from_secs(3), mgr.run_turn("hello"))
        .await
        .expect("turn should complete within bounded deadline")
        .expect("run_turn should succeed");
    assert_eq!(reply, "Deterministic fake response");

    // Production cancellation: dropping run_turn_on future removes listener and active turn
    let slow_inputs = vec![tera::codex::TurnInput::Text("SLOW_TURN".to_string())];
    let cancel_fut = mgr.run_turn_on(&info.id, &slow_inputs);
    tokio::select! {
        _ = cancel_fut => panic!("turn should not complete before cancellation"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {}
    }
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert_eq!(
        mgr.turn_listener_count().await,
        0,
        "listeners must be cleared on cancellation"
    );
    assert_eq!(
        mgr.active_turn_count().await,
        0,
        "active turns must be cleared on cancellation"
    );

    // Fault injection: turn with INJECT_FAULT_CRASH causes fake app-server to terminate mid-turn
    let crash_res = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        mgr.run_turn("INJECT_FAULT_CRASH"),
    )
    .await
    .expect("crashed turn should not hang")
    .unwrap_err();
    assert!(
        crash_res.to_string().contains("closed")
            || crash_res.to_string().contains("exited")
            || crash_res.to_string().contains("not running")
    );
}

#[tokio::test]
async fn test_injected_dependent_insert_rollback() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();

    // Inject constraint failure by creating a trigger on delivery_events
    {
        let conn = rusqlite::Connection::open(config.history_db_path()).unwrap();
        conn.execute(
            "CREATE TRIGGER fail_delivery BEFORE INSERT ON delivery_events BEGIN SELECT RAISE(ABORT, 'injected constraint failure'); END;",
            [],
        ).unwrap();
    }

    let event_id = "msg_rollback_test";
    let event = ConversationEvent::message(
        event_id,
        Utc::now().timestamp_millis(),
        "assistant",
        Some("Should be rolled back".to_string()),
        None,
        Some("turn_rb".to_string()),
        vec![tera::history::db::Attachment {
            id: None,
            event_id: event_id.to_string(),
            position: 0,
            media_type: "document".to_string(),
            relative_path: "assets/2026/09/doc.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            original_name: Some("doc.txt".to_string()),
        }],
    );

    let pref = tera::history::db::ProviderRef::whatsapp(
        event_id,
        "wa_msg_rb",
        "12345@s.whatsapp.net",
        true,
    );

    let res = history_db.insert_event_full(event, Some(&pref), Some(("sent", None)));

    assert!(res.is_err(), "insert_event_full must fail due to trigger");

    // Verify zero partial rows remain across all affected tables
    let conn = rusqlite::Connection::open(config.history_db_path()).unwrap();
    let ev_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM conversation_events WHERE id = ?1",
            rusqlite::params![event_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(ev_count, 0, "conversation_events should have zero rows");

    let att_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM attachments WHERE event_id = ?1",
            rusqlite::params![event_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(att_count, 0, "attachments should have zero rows");

    let pref_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM provider_refs WHERE event_id = ?1",
            rusqlite::params![event_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(pref_count, 0, "provider_refs should have zero rows");

    let deliv_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM delivery_events WHERE event_id = ?1",
            rusqlite::params![event_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(deliv_count, 0, "delivery_events should have zero rows");

    // Verify JSONL projection was not written
    let events = history_db.list_events_all().unwrap();
    assert!(!events.iter().any(|e| e.id == event_id));
}

#[tokio::test]
async fn test_concurrent_and_restart_inbound_replay() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
    let transport = Arc::new(tera::transport::MockTransport::new());
    let engine = Arc::new(TurnEngine::new(
        config.clone(),
        history_db.clone(),
        runtime_db.clone(),
        transport.clone(),
        ConversationSession::new(),
        CodexSupervisor::new(config.clone(), runtime_db.clone(), history_db.clone()),
    ));

    let msg = InboundMessage {
        provider_msg_id: "concurrent_replay_wa_1".to_string(),
        sender: "owner@s.whatsapp.net".to_string(),
        text: Some("Hello concurrent test".to_string()),
        timestamp_ms: Utc::now().timestamp_millis(),
        reply_to_provider_msg_id: None,
        media_attachment: None,
        media_error: None,
        chat_jid: "owner@s.whatsapp.net".to_string(),
        from_own_account: true,
        is_group: false,
    };

    // Spawn 5 concurrent tasks delivering the exact same provider message ID
    let mut tasks = Vec::new();
    for _ in 0..5 {
        let eng = engine.clone();
        let m = msg.clone();
        tasks.push(tokio::spawn(async move {
            let _ = eng.handle_inbound_message(m).await;
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }

    // Exactly 1 event must be recorded in history
    assert_eq!(history_db.count_events().unwrap(), 1);

    // Simulate restart: construct fresh TurnEngine pointing to same state
    let restarted_engine = TurnEngine::new(
        config.clone(),
        history_db.clone(),
        runtime_db.clone(),
        transport.clone(),
        ConversationSession::new(),
        CodexSupervisor::new(config.clone(), runtime_db.clone(), history_db.clone()),
    );

    // Replay after restart
    let res = restarted_engine.handle_inbound_message(msg).await;
    assert!(res.is_ok());

    // Still exactly 1 event
    assert_eq!(history_db.count_events().unwrap(), 1);

    // Verify exactly 1 projection record exists
    let proj_lines = read_projection(&config);
    assert_eq!(proj_lines.len(), 1);
}

#[tokio::test]
async fn test_mcp_readiness_failure_preserves_rollback_state() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let updates_dir = config.runtime_dir().join("updates");
    fs::create_dir_all(&updates_dir).unwrap();
    let journal_path = updates_dir.join("update.json");
    let backup_bin = updates_dir.join("tera.previous");
    fs::write(&backup_bin, "fake backup binary").unwrap();

    let current = tera::version::BuildInfo::current();
    let journal_content = serde_json::json!({
        "phase": "installed",
        "previous": current,
        "next": current,
        "target": config.workspace_dir.join("tera"),
        "backup": backup_bin,
        "codex_before": null,
        "codex_after": null,
        "codex_backup": null,
        "codex_target": null,
    });
    fs::write(
        &journal_path,
        serde_json::to_string(&journal_content).unwrap(),
    )
    .unwrap();

    // Create a directory at the MCP socket path so bind() will fail with EISDIR/EADDRINUSE
    let sock_path = config.socket_path();
    fs::create_dir_all(&sock_path).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
    let transport = Arc::new(tera::transport::MockTransport::new());
    let session = ConversationSession::new();
    let rpc_server = Arc::new(tera::mcp::daemon_rpc::DaemonRpcServer::new(
        config.clone(),
        history_db,
        runtime_db,
        transport,
        session,
    ));

    // bind() must fail
    let bind_res = rpc_server.bind();
    assert!(bind_res.is_err(), "binding to a directory must fail");

    // Verify that because readiness failed, update state was preserved (mark_healthy not called)
    assert!(journal_path.exists(), "update journal must still exist");
    assert!(backup_bin.exists(), "backup binary must still exist");

    // Remove the blocking directory and bind successfully
    fs::remove_dir_all(&sock_path).unwrap();
    let listener = rpc_server
        .bind()
        .expect("bind should succeed after removing directory");
    drop(listener);

    // Simulate mark_healthy (called only after daemon readiness)
    tera::update::mark_healthy(&config);
    assert!(
        !journal_path.exists(),
        "journal must be committed and removed after healthy start"
    );
    assert!(
        !backup_bin.exists(),
        "backup binary must be cleaned up after healthy start"
    );
}

#[tokio::test]
async fn test_all_five_outgoing_media_kinds_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
    let transport = Arc::new(tera::transport::MockTransport::new());
    let session = ConversationSession::new();
    session.set_chat("owner@s.whatsapp.net");

    let server = Arc::new(tera::mcp::daemon_rpc::DaemonRpcServer::new(
        config.clone(),
        history_db.clone(),
        runtime_db,
        transport.clone(),
        session,
    ));

    let scratch_dir = config.workspace_dir.join("scratch_media");
    fs::create_dir_all(&scratch_dir).unwrap();

    let media_specs: [(&str, &str, &str, &[u8]); 5] = [
        ("image_path", "photo.png", "image", b"fake png image bytes"),
        ("video_path", "clip.mp4", "video", b"fake mp4 video bytes"),
        ("audio_path", "song.mp3", "audio", b"fake mp3 audio bytes"),
        (
            "voice_note_path",
            "voice.ogg",
            "voice_note",
            b"fake ogg voice bytes",
        ),
        (
            "file_path",
            "report.pdf",
            "document",
            b"fake pdf document bytes",
        ),
    ];

    for (arg_key, filename, _media_type, data) in &media_specs {
        let scratch_file = scratch_dir.join(filename);
        fs::write(&scratch_file, data).unwrap();

        let tool_args = serde_json::json!({
            "text": format!("Sending {filename}"),
            *arg_key: scratch_file.to_str().unwrap(),
        });

        let res = server
            .execute_tool("send_message", &tool_args, None)
            .await
            .unwrap();
        assert_eq!(res["status"], "sent");
    }

    // Delete the entire scratch directory: original files are gone
    fs::remove_dir_all(&scratch_dir).unwrap();
    assert!(!scratch_dir.exists());

    // Rebuild the JSONL projection from canonical SQLite history
    ProjectionEngine::rebuild_all(
        &config.history_jsonl_dir(),
        &config.runtime_dir().join("tmp"),
        &history_db,
    )
    .unwrap();

    // Verify all 5 media events exist in rebuilt projection and assets are preserved on disk
    let proj_lines = read_projection(&config);
    assert_eq!(proj_lines.len(), 5);

    let events = history_db.list_events_all().unwrap();
    assert_eq!(events.len(), 5);

    for (i, (_, filename, expected_type, expected_data)) in media_specs.iter().enumerate() {
        let ev = &events[i];
        assert_eq!(ev.attachments.len(), 1);
        let att = &ev.attachments[0];
        assert_eq!(att.media_type, *expected_type);
        assert_eq!(att.original_name.as_deref(), Some(*filename));

        let asset_path = config.resolve_asset(&att.relative_path);
        assert!(
            asset_path.exists(),
            "Asset file must exist at {:?}",
            asset_path
        );
        let recovered_data = fs::read(&asset_path).unwrap();
        assert_eq!(
            &recovered_data[..],
            *expected_data,
            "Recovered data must match original"
        );
    }
}

#[tokio::test]
async fn test_worker_isolation_and_send_accounting() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let history_db = HistoryDb::open_for(&config).unwrap();
    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
    let transport = Arc::new(tera::transport::MockTransport::new());
    let session = ConversationSession::new();
    session.set_chat("owner@s.whatsapp.net");

    let server = Arc::new(tera::mcp::daemon_rpc::DaemonRpcServer::new(
        config.clone(),
        history_db.clone(),
        runtime_db,
        transport.clone(),
        session.clone(),
    ));

    session.set_turn(Some("foreground_turn_1"));
    let fg_before = session.count_for(None);

    let sched_worker = "schedule:daily_brief";
    session.set_turn_for(sched_worker, Some("run_sched_1"));
    let sched_before = session.count_for(Some(sched_worker));

    // Scheduled task worker sends a message via MCP execute_tool
    let sched_args = serde_json::json!({"text": "Scheduled task notification"});
    let sched_res = server
        .execute_tool("send_message", &sched_args, Some(sched_worker))
        .await
        .unwrap();
    assert_eq!(sched_res["status"], "sent");
    let sched_msg_id = sched_res["message_id"].as_str().unwrap();

    // Verify worker send counter incremented, but foreground counter was untouched
    assert_eq!(session.sends_since_for(Some(sched_worker), sched_before), 1);
    assert_eq!(session.sends_since(fg_before), 0);

    // Verify history event SQLite turn_id attribution matches the worker turn
    let sched_event = history_db.get_event(sched_msg_id).unwrap().unwrap();
    assert_eq!(sched_event.turn_id(), Some("run_sched_1"));

    // Foreground work sends a message via MCP execute_tool (worker_id: None)
    let fg_args = serde_json::json!({"text": "Foreground conversation reply"});
    let fg_res = server
        .execute_tool("send_message", &fg_args, None)
        .await
        .unwrap();
    assert_eq!(fg_res["status"], "sent");
    let fg_msg_id = fg_res["message_id"].as_str().unwrap();

    // Foreground counter must now be 1
    assert_eq!(session.sends_since(fg_before), 1);

    // Verify history event SQLite turn_id attribution matches foreground turn
    let fg_event = history_db.get_event(fg_msg_id).unwrap().unwrap();
    assert_eq!(fg_event.turn_id(), Some("foreground_turn_1"));

    // Phoenix recovery uses isolated worker "phoenix"
    let ph_before = session.count_for(Some("phoenix"));
    session.set_turn_for("phoenix", Some("ph_turn_1"));
    assert_eq!(
        session.turn_for(Some("phoenix")),
        Some("ph_turn_1".to_string())
    );
    assert_eq!(
        session.turn_for(None),
        Some("foreground_turn_1".to_string())
    );

    let ph_args = serde_json::json!({"text": "Phoenix crash recovery notice"});
    let ph_res = server
        .execute_tool("send_message", &ph_args, Some("phoenix"))
        .await
        .unwrap();
    let ph_msg_id = ph_res["message_id"].as_str().unwrap();
    assert_eq!(session.sends_since_for(Some("phoenix"), ph_before), 1);
    assert_eq!(session.sends_since(fg_before), 1);

    let ph_event = history_db.get_event(ph_msg_id).unwrap().unwrap();
    assert_eq!(ph_event.turn_id(), Some("ph_turn_1"));

    // Clearing the worker does not touch foreground state
    session.clear_worker(sched_worker);
    assert_eq!(session.turn_for(Some(sched_worker)), None);
    assert_eq!(
        session.turn_for(None),
        Some("foreground_turn_1".to_string())
    );
}

#[tokio::test]
async fn test_scheduler_injected_completion_write_and_interrupted_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let script_path = temp_dir.path().join("fake_codex_sched.sh");
    let script_content = r#"#!/usr/bin/env python3
import sys, json

for line in sys.stdin:
    if not line.strip():
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    method = req.get("method")
    req_id = req.get("id")
    if method == "getAuthStatus":
        sys.stdout.write(json.dumps({"id": req_id, "result": {"requiresOpenaiAuth": False, "authMethod": "chatgpt"}}) + "\n")
        sys.stdout.flush()
    elif method == "thread/start":
        sys.stdout.write(json.dumps({"id": req_id, "result": {"thread": {"id": "fake_sched_th_1", "model": "fake-model"}}}) + "\n")
        sys.stdout.flush()
    elif method == "turn/start":
        sys.stdout.write(json.dumps({"id": req_id, "result": {"turn": {"id": "fake_turn_sched_1"}}}) + "\n")
        sys.stdout.write(json.dumps({"method": "turn/started", "params": {"threadId": "fake_sched_th_1", "turn": {"id": "fake_turn_sched_1"}}}) + "\n")
        sys.stdout.write(json.dumps({"method": "item/started", "params": {"threadId": "fake_sched_th_1", "item": {"type": "agentMessage", "id": "msg_1"}}}) + "\n")
        sys.stdout.write(json.dumps({"method": "item/completed", "params": {"threadId": "fake_sched_th_1", "item": {"type": "agentMessage", "id": "msg_1", "text": "Task finished successfully"}}}) + "\n")
        sys.stdout.write(json.dumps({"method": "turn/completed", "params": {"threadId": "fake_sched_th_1"}}) + "\n")
        sys.stdout.flush()
    elif req_id is not None:
        sys.stdout.write(json.dumps({"id": req_id, "result": {}}) + "\n")
        sys.stdout.flush()
"#;
    fs::write(&script_path, script_content).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&script_path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script_path, perms).unwrap();
    }

    let mut config = Config::new(temp_dir.path().to_path_buf(), true);
    config.codex_bin = script_path.clone();
    WorkspaceInit::init(&config).unwrap();

    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
    let history_db = HistoryDb::open_for(&config).unwrap();
    let session = ConversationSession::new();
    let codex = CodexSupervisor::new(config.clone(), runtime_db.clone(), history_db.clone());
    let runner = SchedulerRunner::new(config.clone(), runtime_db.clone(), codex, session);

    // Scenario 1: Exercise run_schedule() failing to persist completion via injected SQLite write failure.
    // The run completes on disk, writes sentinel, but finish_run fails and leaves run in 'running'.
    // Then recover_stale_runs() reconciles it to Completed and deletes sentinel.
    let at_ms1 = Utc::now().timestamp_millis() - 5000;
    let timing1 = ScheduleTiming::Once { at_ms: at_ms1 };
    let item1 = SchedulerDb::create_schedule(
        &runtime_db,
        "Injected Write Failure Task",
        "Backup task",
        &timing1,
        "tasks/test-injected-failure",
    )
    .unwrap();

    // Inject trigger causing SQLite UPDATE state='completed' on schedule_runs to fail
    {
        let conn = runtime_db.conn.lock().unwrap();
        conn.execute(
            "CREATE TRIGGER inject_finish_run_fail
             BEFORE UPDATE OF state ON schedule_runs
             FOR EACH ROW WHEN NEW.state = 'completed'
             BEGIN
                 SELECT RAISE(FAIL, 'injected SQLite write failure on completion');
             END;",
            [],
        )
        .unwrap();
    }

    // Call run_schedule - must fail due to injected trigger on finish_run
    let run_res = runner.run_schedule(&item1).await;
    assert!(
        run_res.is_err(),
        "run_schedule must fail when completion persistence fails"
    );

    // Verify run is still 'running' in the database
    let running = SchedulerDb::running_runs(&runtime_db).unwrap();
    assert_eq!(running.len(), 1);
    let run1_id = running[0].id.clone();

    // Verify sentinel was written to disk by run_schedule
    let task_dir1 = config.workspace_dir.join(&item1.task_path);
    let sentinel1 = task_dir1.join(format!(".completed_{}", run1_id));
    assert!(
        sentinel1.exists(),
        "run_schedule must have left the completion sentinel on disk"
    );

    // Drop the injected trigger so recovery can persist completion
    {
        let conn = runtime_db.conn.lock().unwrap();
        conn.execute("DROP TRIGGER inject_finish_run_fail", [])
            .unwrap();
    }

    // Recover stale runs
    runner.recover_stale_runs().unwrap();

    // Sentinel must be removed
    assert!(
        !sentinel1.exists(),
        "sentinel file must be deleted upon reconciliation"
    );

    // Run must be reconciled to Completed in the database
    let run1 = SchedulerDb::get_run(&runtime_db, &run1_id)
        .unwrap()
        .unwrap();
    assert_eq!(run1.state, RunState::Completed);

    // One-shot schedule must be marked Completed with no next run
    let item1_after = SchedulerDb::get_schedule(&runtime_db, &item1.id)
        .unwrap()
        .unwrap();
    assert_eq!(item1_after.status, ScheduleStatus::Completed);
    assert_eq!(item1_after.next_run_at_ms, None);

    // Scenario 2: Interrupted run without sentinel (process was killed mid-turn).
    // Run must be marked Failed, PHOENIX_RECOVERY.md written, and one-shot schedule re-queued as Active.
    let at_ms2 = Utc::now().timestamp_millis() - 5000;
    let timing2 = ScheduleTiming::Once { at_ms: at_ms2 };
    let item2 = SchedulerDb::create_schedule(
        &runtime_db,
        "Interrupted One-Shot Stale Run",
        "Critical sync task",
        &timing2,
        "tasks/test-stale-interrupted",
    )
    .unwrap();
    let task_dir2 = config.workspace_dir.join(&item2.task_path);
    fs::create_dir_all(&task_dir2).unwrap();

    let run2_id = SchedulerDb::start_run(&runtime_db, &item2.id, at_ms2).unwrap();
    // Simulate runner clearing next_run_at_ms when claiming task
    SchedulerDb::update_next_run(&runtime_db, &item2.id, None, None).unwrap();

    // No sentinel exists. Run stale recovery.
    runner.recover_stale_runs().unwrap();

    // Run must be marked Failed
    let run2 = SchedulerDb::get_run(&runtime_db, &run2_id)
        .unwrap()
        .unwrap();
    assert_eq!(run2.state, RunState::Failed);
    assert!(
        run2.error
            .as_deref()
            .unwrap_or("")
            .contains("Phoenix recovered"),
        "error must document Phoenix recovery"
    );

    // PHOENIX_RECOVERY.md must be written for the worker
    let recovery_file = task_dir2.join("PHOENIX_RECOVERY.md");
    assert!(
        recovery_file.exists(),
        "PHOENIX_RECOVERY.md must exist in task dir"
    );
    let recovery_content = fs::read_to_string(&recovery_file).unwrap();
    assert!(recovery_content.contains("Phoenix recovered schedule"));

    // Interrupted one-shot schedule must be re-queued as Active with next_run_at_ms set
    let item2_after = SchedulerDb::get_schedule(&runtime_db, &item2.id)
        .unwrap()
        .unwrap();
    assert_eq!(item2_after.status, ScheduleStatus::Active);
    assert!(
        item2_after.next_run_at_ms.is_some(),
        "interrupted one-shot schedule must be re-queued with next_run_at_ms"
    );
}
