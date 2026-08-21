use tera::config::Config;
use tera::history::db::{ConversationEvent, EventKind, HistoryDb};
use tera::history::projection::ProjectionEngine;
use tera::memory::generations::GenerationManager;
use tera::memory::NIGHTLY;
use tera::runtime::RuntimeDb;
use tera::codex::tier;
use tera::scheduler::db::SchedulerDb;
use tera::scheduler::recurrence::ScheduleTiming;
use tera::secrets::SecretStore;
use tera::codex::CodexSupervisor;
use tera::conversation::{ConversationSession, TurnEngine};
use tera::runtime::ActivityTracker;
use tera::transport::InboundMessage;
use tera::workspace::init::WorkspaceInit;
use chrono::Utc;
use rusqlite::Connection;
use std::fs;
use tempfile::TempDir;

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
            assert!(config.skills_dir().join(skill.name).join(file.relative_path).exists());
        }
    }
    assert!(config.memories_link().exists());
    assert!(config.memories_link().join("INDEX.md").exists());
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
        .insert_event(ConversationEvent {
            seq: None,
            id: "m_test1".to_string(),
            occurred_at_ms: Utc::now().timestamp_millis(),
            kind: EventKind::Message,
            actor: "user".to_string(),
            text: Some("OpenChoreo deployment setup in progress".to_string()),
            reply_to_id: None,
            turn_id: Some("turn_1".to_string()),
            reaction_target_id: None,
            reaction_emoji: None,
            attachments: vec![],
        })
        .unwrap();

    let retrieved = history_db.get_event("m_test1").unwrap().unwrap();
    assert_eq!(retrieved.text.unwrap(), "OpenChoreo deployment setup in progress");

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
            .insert_event(ConversationEvent {
                seq: None,
                id: id.to_string(),
                occurred_at_ms: Utc::now().timestamp_millis(),
                kind: EventKind::Message,
                actor: actor.to_string(),
                text: Some(text.to_string()),
                reply_to_id: None,
                turn_id: Some("turn_2".to_string()),
                reaction_target_id: None,
                reaction_emoji: None,
                attachments: vec![],
            })
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
        .insert_event(ConversationEvent {
            seq: None,
            id: "m_kept".to_string(),
            occurred_at_ms: Utc::now().timestamp_millis(),
            kind: EventKind::Message,
            actor: "assistant".to_string(),
            text: Some("I replied through the tool".to_string()),
            reply_to_id: None,
            turn_id: None,
            reaction_target_id: None,
            reaction_emoji: None,
            attachments: vec![],
        })
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
        tier::ROUTINE,
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

/// The memory generation transaction, the part Rust owns. What a generation
/// should *say* is the model's job and needs a live app-server, so it is covered
/// by the live tests rather than here.
#[tokio::test]
async fn test_memory_generation_promotion_is_atomic() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let staging = NIGHTLY.prepare_staging(&config).unwrap();
    fs::write(staging.join("people.md"), "# Amaya\n\nMoving in December.\n").unwrap();

    let generation = GenerationManager::atomic_swap_generation(&config, &staging).unwrap();
    assert!(generation >= 2);
    assert_eq!(
        GenerationManager::get_current_generation_num(&config).unwrap(),
        generation
    );

    // The symlink swings over to the new generation, and never dangles.
    assert!(config.memories_link().join("people.md").is_file());
    assert!(!config.workspace_dir.join("memories.new").exists());
}

/// A generation that would break the workspace must not become active, and the
/// failure must leave the current memory in place.
#[tokio::test]
async fn test_invalid_generation_is_refused() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::new(temp_dir.path().to_path_buf(), true);
    WorkspaceInit::init(&config).unwrap();

    let before = GenerationManager::get_current_generation_num(&config).unwrap();

    // Missing HORIZON.md.
    let bad = config.staging_dir().join("bad");
    fs::create_dir_all(&bad).unwrap();
    fs::write(bad.join("INDEX.md"), "# Index\n").unwrap();
    assert!(GenerationManager::atomic_swap_generation(&config, &bad).is_err());

    // A symlink escaping the generation.
    let escaping = config.staging_dir().join("escaping");
    fs::create_dir_all(&escaping).unwrap();
    fs::write(escaping.join("INDEX.md"), "# Index\n").unwrap();
    fs::write(escaping.join("HORIZON.md"), "# Horizon\n").unwrap();
    std::os::unix::fs::symlink("/etc/passwd", escaping.join("secrets.md")).unwrap();
    assert!(GenerationManager::atomic_swap_generation(&config, &escaping).is_err());

    assert_eq!(
        GenerationManager::get_current_generation_num(&config).unwrap(),
        before
    );
    assert!(config.memories_link().join("INDEX.md").is_file());
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
        ActivityTracker::new(),
    );

    engine
        .handle_inbound_message(InboundMessage {
            provider_msg_id: "wa_1".to_string(),
            sender: "owner@s.whatsapp.net".to_string(),
            text: Some(format!("/secret SPOTIFY_CLIENT_ID {VALUE}")),
            timestamp_ms: Utc::now().timestamp_millis(),
            reply_to_provider_msg_id: None,
            media_attachment: None,
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
    let recorded = events[0].text.as_deref().unwrap();
    assert!(!recorded.contains(VALUE), "history holds the secret: {recorded}");
    assert!(
        recorded.contains("SPOTIFY_CLIENT_ID"),
        "the agent still has to learn which credential arrived: {recorded}"
    );

    // And the value went somewhere a skill can actually read it.
    let store = SecretStore::new(config.secrets_path());
    assert_eq!(store.get("SPOTIFY_CLIENT_ID").unwrap().as_deref(), Some(VALUE));
}
