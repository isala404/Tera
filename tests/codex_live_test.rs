//! Live tests against a real `codex app-server`.
//!
//! Ignored by default because they spawn the actual binary, need a logged-in
//! Codex account and burn real tokens. Run them after touching anything in
//! `codex::process`, `workspace::init` or `mcp`:
//!
//!     cargo test --test codex_live_test -- --ignored --nocapture

use tera::codex::process::ThreadOptions;
use tera::codex::tier;
use tera::codex::CodexProcessManager;
use tera::config::Config;
use tera::conversation::ConversationSession;
use tera::history::HistoryDb;
use tera::mcp::DaemonRpcServer;
use tera::runtime::RuntimeDb;
use tera::transport::{MockTransport, Transport};
use tera::workspace::WorkspaceInit;
use std::sync::Arc;

/// Point the generated Codex config at the real daemon binary. Under `cargo
/// test` the current executable is the test harness, which would make Codex
/// spawn the wrong process for the required MCP server.
fn workspace_config(dir: &std::path::Path) -> Config {
    std::env::set_var(
        "TERA_BIN",
        std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tera"),
    );
    Config::new(dir.to_path_buf(), true)
}

#[tokio::test]
#[ignore = "spawns a real codex app-server and consumes account tokens"]
async fn live_handshake_and_turn_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = CodexProcessManager::spawn(None)
        .await
        .expect("app-server handshake should complete");
    mgr.start_thread(&ThreadOptions::new(tmp.path()))
        .await
        .expect("thread should start");

    let reply = mgr
        .run_turn("Reply with exactly the word: pong", tier::CONVERSATION)
        .await
        .expect("turn should produce text");

    println!("codex replied: {reply:?}");
    assert!(
        reply.to_lowercase().contains("pong"),
        "expected the model's reply to contain 'pong', got {reply:?}"
    );
}

/// PLAN.md Phase 1 success criterion:
/// "Codex can start in /workspace and call a dummy MCP tool."
///
/// Exercises the whole bootstrap path: workspace init writes `.codex-home`,
/// the daemon binds its Unix socket, the app-server starts against that Codex
/// home, and Codex reaches back through the stdio MCP proxy into the daemon.
#[tokio::test]
#[ignore = "spawns a real codex app-server and consumes account tokens"]
async fn live_codex_starts_in_workspace_and_calls_mcp_tool() {
    let tmp = tempfile::tempdir().unwrap();
    let config = workspace_config(tmp.path());

    WorkspaceInit::init(&config).unwrap();
    let history_db = HistoryDb::open_for(&config).unwrap();
    let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();

    // The MCP proxy Codex spawns talks to this socket.
    let transport: Arc<dyn Transport> = Arc::new(MockTransport::new());
    let rpc = Arc::new(DaemonRpcServer::new(
        config.clone(),
        history_db.clone(),
        runtime_db.clone(),
        transport.clone(),
        ConversationSession::new(),
    ));
    tokio::spawn(async move {
        let _ = rpc.run().await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert!(config.socket_path().exists(), "daemon socket should be bound");

    let mgr = CodexProcessManager::spawn(Some(&config.codex_home_dir()))
        .await
        .expect("app-server should start against the workspace codex home");

    let opts = ThreadOptions::new(&config.workspace_dir);
    let info = mgr.start_thread(&opts).await.expect("thread should start");
    println!("thread {} on model {}", info.id, info.model);

    let reply = mgr
        .run_turn(
            "Call the tera MCP tool `list_schedules` with no arguments. \
             Then reply with exactly: TOOL_OK",
            tier::CONVERSATION,
        )
        .await
        .expect("turn should complete");

    println!("codex replied: {reply:?}");
    assert!(
        reply.contains("TOOL_OK"),
        "expected Codex to reach the tera MCP server, got {reply:?}"
    );
}

/// The main thread must survive a daemon restart, or every restart silently
/// drops the user's conversation context (PLAN.md section 12).
#[tokio::test]
#[ignore = "spawns a real codex app-server and consumes account tokens"]
async fn live_thread_resume_preserves_context() {
    let tmp = tempfile::tempdir().unwrap();
    let config = workspace_config(tmp.path());
    WorkspaceInit::init(&config).unwrap();

    let opts = ThreadOptions::new(&config.workspace_dir);

    let first = CodexProcessManager::spawn(Some(&config.codex_home_dir()))
        .await
        .unwrap();
    let info = first.start_thread(&opts).await.unwrap();
    first
        .run_turn("Remember this codeword: BANYAN. Reply with just: ok", tier::CONVERSATION)
        .await
        .unwrap();
    drop(first);

    // A second process resumes the same thread id, as the daemon does on boot.
    let second = CodexProcessManager::spawn(Some(&config.codex_home_dir()))
        .await
        .unwrap();
    let resumed = second
        .ensure_thread(Some(&info.id), &opts)
        .await
        .expect("thread should resume");
    assert_eq!(resumed.id, info.id, "resume should rejoin the same thread");

    let reply = second
        .run_turn("What was the codeword I gave you? Reply with just the word.", tier::CONVERSATION)
        .await
        .unwrap();

    println!("codex recalled: {reply:?}");
    assert!(
        reply.to_uppercase().contains("BANYAN"),
        "resumed thread lost its context, got {reply:?}"
    );
}
