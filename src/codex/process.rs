use crate::codex::log::{is_stderr_problem, log_notification, strip_ansi, truncate};
use crate::codex::rpc::{JsonRpcRequest, JsonRpcResponse};
use crate::codex::tier::{self, ModelTier};
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Streaming events tera cares about, routed per Codex thread.
#[derive(Debug, Clone)]
enum TurnEvent {
    /// Incremental assistant text.
    Delta(String),
    /// Final text of a completed `agentMessage` item. Authoritative over deltas.
    Message(String),
    Completed,
    Failed(String),
}

struct TurnListener {
    thread_id: String,
    tx: mpsc::Sender<TurnEvent>,
}

/// Where a Codex thread runs, under what permissions, and on which model.
#[derive(Debug, Clone)]
pub struct ThreadOptions {
    pub cwd: std::path::PathBuf,
    /// Model for the thread. Turns can still override it per turn; this is what
    /// the thread reports as its model, which is what thread rotation compares.
    pub tier: ModelTier,
}

impl ThreadOptions {
    /// A conversation thread. Task threads pick their tier explicitly.
    pub fn new(cwd: impl Into<std::path::PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            tier: tier::CONVERSATION,
        }
    }

    pub fn with_tier(mut self, tier: ModelTier) -> Self {
        self.tier = tier;
        self
    }
}

/// One element of a user turn. Media is passed by path rather than inlined:
/// the app-server reads the file itself, and the bytes already live in the
/// asset store, so there is no second copy to keep consistent.
#[derive(Debug, Clone)]
pub enum TurnInput {
    Text(String),
    LocalImage(std::path::PathBuf),
    LocalAudio(std::path::PathBuf),
}

impl TurnInput {
    fn to_json(&self) -> Value {
        match self {
            TurnInput::Text(text) => json!({"type": "text", "text": text}),
            TurnInput::LocalImage(path) => {
                json!({"type": "localImage", "path": path.to_string_lossy()})
            }
            TurnInput::LocalAudio(path) => {
                json!({"type": "localAudio", "path": path.to_string_lossy()})
            }
        }
    }
}

/// Whether a thread carries prior conversation or starts empty.
///
/// The difference matters beyond logging: a resumed thread already knows the
/// user, while a fresh one needs bootstrap context before it can behave like the
/// same assistant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadOrigin {
    /// Rejoined a thread from a previous daemon run; history intact.
    Resumed,
    /// Brand new thread; no prior context.
    Created,
}

impl std::fmt::Display for ThreadOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThreadOrigin::Resumed => write!(f, "RESUMED"),
            ThreadOrigin::Created => write!(f, "NEW"),
        }
    }
}

/// Identity of a live Codex thread, as reported by `thread/start` or `thread/resume`.
#[derive(Debug, Clone)]
pub struct ThreadInfo {
    pub id: String,
    pub model: String,
    pub origin: ThreadOrigin,
}

impl ThreadInfo {
    fn from_result(res: &Value, origin: ThreadOrigin) -> Result<Self> {
        let id = res
            .get("thread")
            .and_then(|t| t.get("id"))
            .and_then(|i| i.as_str())
            .ok_or_else(|| anyhow!("app-server response missing thread.id: {res}"))?
            .to_string();
        let model = res
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown")
            .to_string();
        Ok(Self { id, model, origin })
    }
}

pub struct CodexProcessManager {
    next_id: AtomicU64,
    stdin_tx: mpsc::Sender<String>,
    response_waiters: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    turn_listeners: Arc<Mutex<HashMap<String, TurnListener>>>,
    active_thread_id: Mutex<Option<String>>,
    /// Turn currently running on each thread, keyed by thread id. `turn/steer`
    /// requires the id of the turn it is steering as a precondition.
    active_turns: Arc<Mutex<HashMap<String, String>>>,
    /// Set once the child process is known to be gone, so callers stop waiting
    /// 15 seconds for a reply that will never come.
    dead: Arc<AtomicBool>,
}

impl CodexProcessManager {
    pub async fn spawn(codex_home: Option<&std::path::Path>) -> Result<Self> {
        info!("Spawning persistent 'codex app-server' process (codex_home={:?})", codex_home);

        let mut cmd = Command::new("codex");
        cmd.arg("app-server")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Quieten the child's own tracing. Without this it inherits the daemon's
        // RUST_LOG, so turning on debug logging for tera also turned on
        // Codex's internal telemetry, which is far noisier than anything it says
        // about the actual turn. Set CODEX_LOG to raise it when debugging Codex
        // itself rather than the assistant.
        cmd.env(
            "RUST_LOG",
            std::env::var("CODEX_LOG").unwrap_or_else(|_| "error".to_string()),
        );

        if let Some(path) = codex_home {
            cmd.env("CODEX_HOME", path);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| "Failed to spawn 'codex app-server' process")?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow!("Failed to capture stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("Failed to capture stdout"))?;
        let stderr = child.stderr.take().ok_or_else(|| anyhow!("Failed to capture stderr"))?;

        let (stdin_tx, mut stdin_rx) = mpsc::channel::<String>(100);

        // Stdin writer task
        tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(line) = stdin_rx.recv().await {
                if let Err(e) = stdin.write_all(line.as_bytes()).await {
                    error!("Failed to write line to Codex stdin: {:?}", e);
                    break;
                }
                if let Err(e) = stdin.flush().await {
                    error!("Failed to flush Codex stdin: {:?}", e);
                    break;
                }
            }
        });

        // Stderr reader task.
        //
        // Codex writes its own tracing output here, coloured and at its own
        // levels. Echoing every line at warn turned routine startup telemetry
        // into warnings in our log, so classify by what the line actually says
        // and keep the rest at debug.
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                let line = strip_ansi(&line);
                if line.trim().is_empty() {
                    continue;
                }
                if is_stderr_problem(&line) {
                    warn!(target: "codex::stderr", "{line}");
                } else {
                    debug!(target: "codex::stderr", "{line}");
                }
            }
        });

        let waiters: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let waiters_clone = waiters.clone();

        let turn_listeners: Arc<Mutex<HashMap<String, TurnListener>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let listeners_clone = turn_listeners.clone();

        let active_turns: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
        let turns_clone = active_turns.clone();

        let dead = Arc::new(AtomicBool::new(false));

        // Reap the child so its exit is observable. Without this the handle was
        // dropped at the end of spawn and a crashed app-server looked identical to
        // a slow one: requests and turns could wait indefinitely instead of failing.
        let dead_on_exit = dead.clone();
        let listeners_on_exit = turn_listeners.clone();
        tokio::spawn(async move {
            match child.wait().await {
                Ok(status) => error!("codex app-server exited: {status}"),
                Err(e) => error!("Could not wait on codex app-server: {e}"),
            }
            dead_on_exit.store(true, Ordering::SeqCst);
            Self::fail_listeners(&listeners_on_exit, "codex app-server exited").await;
        });

        // Stdout reader task
        let dead_on_eof = dead.clone();
        let reply_tx = stdin_tx.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<Value>(&line) else {
                    warn!("Unparseable line from codex app-server: {}", truncate(&line, 300));
                    continue;
                };

                // Three kinds of line, and they must be told apart by `method`
                // rather than by which struct happens to parse. A server request
                // carries both an id and a method, and its ids come from the
                // app-server's own sequence, parsing one as a response resolved
                // whichever of OUR requests shared that number, with an empty
                // result, and left the request itself unanswered.
                match (v.get("method").and_then(Value::as_str), v.get("id")) {
                    (Some(method), Some(id)) => {
                        let reply = Self::answer_server_request(id, method, v.get("params"));
                        if let Ok(line) = serde_json::to_string(&reply) {
                            let _ = reply_tx.send(line + "\n").await;
                        }
                    }
                    (Some(_), None) => {
                        log_notification(&v);
                        Self::track_active_turn(&turns_clone, &v).await;
                        if let Some((thread_id, event)) = Self::classify_notification(&v) {
                            Self::dispatch(&listeners_clone, &thread_id, event).await;
                        }
                    }
                    (None, _) => {
                        if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(&line) {
                            let mut lock = waiters_clone.lock().await;
                            if let Some(tx) = lock.remove(&resp.id) {
                                let _ = tx.send(resp);
                            }
                        }
                    }
                }
            }
            warn!("codex app-server stdout closed");
            dead_on_eof.store(true, Ordering::SeqCst);
            Self::fail_listeners(&listeners_clone, "codex app-server stdout closed").await;
        });

        let mgr = Self {
            next_id: AtomicU64::new(1),
            stdin_tx,
            response_waiters: waiters,
            turn_listeners,
            active_thread_id: Mutex::new(None),
            active_turns,
            dead,
        };

        // 1. Send initialize handshake, then the `initialized` notification the
        //    app-server expects before it will accept thread requests.
        let init_res = mgr
            .send_request(
                "initialize",
                Some(json!({
                    "clientInfo": {
                        "name": "tera",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                })),
            )
            .await?;
        info!("Codex app-server initialized: {:?}", init_res);
        mgr.send_notification("initialized", None).await?;

        Ok(mgr)
    }

    /// Thread creation settings. The assistant runs unattended over WhatsApp, so
    /// it cannot answer approval prompts. Hence `never` and full access.
    fn thread_params(&self, opts: &ThreadOptions) -> Value {
        json!({
            "cwd": opts.cwd.to_string_lossy(),
            "approvalPolicy": "never",
            "sandbox": "danger-full-access",
            "model": opts.tier.model,
        })
    }

    /// Answer a request the app-server made of us.
    ///
    /// tera runs unattended: there is nobody to ask, and the thread is already
    /// started with `never` / `danger-full-access` because the agent is meant to
    /// be able to read any path, write any path, reach the network and install
    /// what it needs. The app-server can still ask, for an explicit permission
    /// grant, or through a legacy approval path, and an unanswered request is
    /// not a refusal, it is a stall: the turn blocks until our timeout and the
    /// agent reports that it was denied.
    ///
    /// So every approval is granted, at session scope, and anything we genuinely
    /// cannot answer gets an error rather than silence.
    fn answer_server_request(id: &Value, method: &str, params: Option<&Value>) -> Value {
        let result = match method {
            "item/commandExecution/requestApproval" | "item/fileChange/requestApproval" => {
                json!({ "decision": "acceptForSession" })
            }

            // Pre-v2 spelling of the same two, still emitted by some builds.
            "execCommandApproval" | "applyPatchApproval" => {
                json!({ "decision": "approved_for_session" })
            }

            // The agent asking for a wider sandbox than it was given. Grant the
            // whole filesystem and the network: that is the configured posture,
            // and a partial grant here would silently re-narrow it.
            "item/permissions/requestApproval" => json!({
                "permissions": {
                    "fileSystem": {
                        "entries": [{ "path": { "type": "path", "path": "/" }, "access": "write" }]
                    },
                    "network": { "enabled": true }
                },
                "scope": "session"
            }),

            // These need a human with an answer, which is exactly what is missing.
            // Declining lets the turn continue; leaving it unanswered would not.
            "item/tool/requestUserInput" => json!({ "answers": {} }),
            "mcpServer/elicitation/request" => json!({ "action": "decline" }),

            other => {
                let detail = truncate(&params.map(|p| p.to_string()).unwrap_or_default(), 300);
                warn!(target: "codex", "Unhandled app-server request '{other}': {detail}");
                return json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": format!("tera does not implement '{other}'") }
                });
            }
        };

        info!(target: "codex", "Auto-approved app-server request '{method}'");
        json!({ "jsonrpc": "2.0", "id": id, "result": result })
    }

    /// Create a thread without making it the main conversation.
    ///
    /// Scheduled tasks get their own short-lived thread rooted in their task
    /// directory, and must not disturb the thread the user is talking to.
    pub async fn create_thread(&self, opts: &ThreadOptions) -> Result<ThreadInfo> {
        let res = self
            .send_request("thread/start", Some(self.thread_params(opts)))
            .await?;
        let info = ThreadInfo::from_result(&res, ThreadOrigin::Created)?;
        debug!("thread/start -> {} (model {}) at {:?}", info.id, info.model, opts.cwd);
        Ok(info)
    }

    /// Start a brand new Codex thread and make it the main conversation.
    pub async fn start_thread(&self, opts: &ThreadOptions) -> Result<ThreadInfo> {
        let info = self.create_thread(opts).await?;
        self.set_active_thread(&info.id).await;
        Ok(info)
    }

    /// Rejoin a previously persisted thread so conversation context survives a
    /// daemon restart.
    pub async fn resume_thread(&self, thread_id: &str, opts: &ThreadOptions) -> Result<ThreadInfo> {
        let mut params = self.thread_params(opts);
        params["threadId"] = json!(thread_id);
        let res = self.send_request("thread/resume", Some(params)).await?;
        let info = ThreadInfo::from_result(&res, ThreadOrigin::Resumed)?;
        debug!("thread/resume -> {} (model {})", info.id, info.model);
        self.set_active_thread(&info.id).await;
        Ok(info)
    }

    /// Resume `persisted` if it is still loadable, otherwise start fresh.
    /// A thread that Codex has garbage-collected must not take the daemon down.
    pub async fn ensure_thread(
        &self,
        persisted: Option<&str>,
        opts: &ThreadOptions,
    ) -> Result<ThreadInfo> {
        if let Some(thread_id) = persisted {
            match self.resume_thread(thread_id, opts).await {
                Ok(info) => {
                    info!(
                        "RESUMED main conversation thread {} (model {}). Earlier context intact",
                        info.id, info.model
                    );
                    return Ok(info);
                }
                Err(e) => warn!(
                    "Could not resume main conversation thread {thread_id}: {e}. Starting a new one."
                ),
            }
        } else {
            info!("No previous main conversation thread recorded; starting fresh");
        }

        let info = self.start_thread(opts).await?;
        info!(
            "NEW main conversation thread {} (model {}). Starts with no prior context",
            info.id, info.model
        );
        Ok(info)
    }

    async fn set_active_thread(&self, thread_id: &str) {
        let mut lock = self.active_thread_id.lock().await;
        *lock = Some(thread_id.to_string());
    }

    pub async fn active_thread(&self) -> Option<String> {
        self.active_thread_id.lock().await.clone()
    }

    /// Ask the app-server which models it offers and which is default, so memory
    /// regeneration can be triggered when the default changes.
    pub async fn list_models(&self) -> Result<Value> {
        self.send_request("model/list", Some(json!({}))).await
    }

    /// Narrate what the agent is doing, so a turn is traceable from the daemon
    /// log without attaching to Codex.
    ///
    /// Map an app-server notification to the thread it belongs to and the event
    /// tera cares about. Returns `None` for notifications we ignore
    /// (MCP startup status, token usage, rate limits, presence, ...).
    fn classify_notification(v: &Value) -> Option<(String, TurnEvent)> {
        let method = v.get("method")?.as_str()?;
        let params = v.get("params")?;
        let thread_id = params.get("threadId")?.as_str()?.to_string();

        let event = match method {
            "item/agentMessage/delta" => TurnEvent::Delta(params.get("delta")?.as_str()?.to_string()),
            "item/completed" => {
                let item = params.get("item")?;
                if item.get("type")?.as_str()? != "agentMessage" {
                    return None;
                }
                TurnEvent::Message(item.get("text")?.as_str()?.to_string())
            }
            "turn/completed" => TurnEvent::Completed,
            "turn/failed" => {
                let reason = params
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error");
                TurnEvent::Failed(reason.to_string())
            }
            _ => return None,
        };

        Some((thread_id, event))
    }

    /// Keep the per-thread active turn id current from the event stream.
    ///
    /// `turn/steer` needs the id of the turn it is joining, and the window in
    /// which steering is legal is exactly the window between these two
    /// notifications, so the stream, not our own bookkeeping, defines it.
    async fn track_active_turn(turns: &Arc<Mutex<HashMap<String, String>>>, v: &Value) {
        let Some(method) = v.get("method").and_then(|m| m.as_str()) else {
            return;
        };
        let Some(params) = v.get("params") else { return };
        let Some(thread_id) = params.get("threadId").and_then(|t| t.as_str()) else {
            return;
        };

        match method {
            "turn/started" => {
                if let Some(turn_id) = params
                    .get("turn")
                    .and_then(|t| t.get("id"))
                    .or_else(|| params.get("turnId"))
                    .and_then(|t| t.as_str())
                {
                    turns
                        .lock()
                        .await
                        .insert(thread_id.to_string(), turn_id.to_string());
                }
            }
            "turn/completed" | "turn/failed" | "turn/aborted" => {
                turns.lock().await.remove(thread_id);
            }
            _ => {}
        }
    }

    /// The turn currently running on a thread, if any.
    pub async fn active_turn_of(&self, thread_id: &str) -> Option<String> {
        self.active_turns.lock().await.get(thread_id).cloned()
    }

    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::SeqCst)
    }

    /// Deliver more input into a turn that is already running.
    ///
    /// This is what makes "actually, make it Japanese" work while the agent is
    /// mid-search, instead of starting a second concurrent turn on the same
    /// thread.
    pub async fn steer(&self, thread_id: &str, inputs: &[TurnInput]) -> Result<()> {
        let turn_id = self
            .active_turn_of(thread_id)
            .await
            .ok_or_else(|| anyhow!("no active turn on thread {thread_id} to steer"))?;

        self.send_request(
            "turn/steer",
            Some(json!({
                "threadId": thread_id,
                "expectedTurnId": turn_id,
                "input": inputs.iter().map(TurnInput::to_json).collect::<Vec<_>>(),
            })),
        )
        .await?;

        info!("Steered new input into running turn {turn_id} on thread {thread_id}");
        Ok(())
    }

    /// Stop the turn running on a thread. Used to get out of the way of real work
    /// when maintenance is holding the app-server.
    pub async fn interrupt(&self, thread_id: &str) -> Result<()> {
        let Some(turn_id) = self.active_turn_of(thread_id).await else {
            return Ok(());
        };
        self.send_request(
            "turn/interrupt",
            Some(json!({"threadId": thread_id, "turnId": turn_id})),
        )
        .await?;
        info!("Interrupted turn {turn_id} on thread {thread_id}");
        Ok(())
    }

    async fn dispatch(
        listeners: &Arc<Mutex<HashMap<String, TurnListener>>>,
        thread_id: &str,
        event: TurnEvent,
    ) {
        let lock = listeners.lock().await;
        for listener in lock.values() {
            if listener.thread_id == thread_id {
                let _ = listener.tx.try_send(event.clone());
            }
        }
    }

    async fn fail_listeners(
        listeners: &Arc<Mutex<HashMap<String, TurnListener>>>,
        reason: &str,
    ) {
        let lock = listeners.lock().await;
        for listener in lock.values() {
            let _ = listener
                .tx
                .try_send(TurnEvent::Failed(reason.to_string()));
        }
    }

    async fn send_notification(&self, method: &str, params: Option<Value>) -> Result<()> {
        let note = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params.unwrap_or(json!({})),
        });
        self.stdin_tx
            .send(serde_json::to_string(&note)? + "\n")
            .await
            .map_err(|_| anyhow!("Failed to send notification to Codex stdin worker"))
    }

    pub async fn send_request(&self, method: &str, params: Option<Value>) -> Result<Value> {
        // Fail fast rather than waiting out the timeout on a process that has
        // already exited; the supervisor replaces a dead manager on the next call.
        if self.is_dead() {
            return Err(anyhow!(
                "codex app-server is not running (request '{method}' not sent)"
            ));
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let req = JsonRpcRequest::new(id, method, params);
        let line = serde_json::to_string(&req)? + "\n";

        let (tx, rx) = oneshot::channel();
        {
            let mut lock = self.response_waiters.lock().await;
            lock.insert(id, tx);
        }

        self.stdin_tx
            .send(line)
            .await
            .map_err(|_| anyhow!("Failed to send request to Codex stdin worker"))?;

        let resp = tokio::time::timeout(std::time::Duration::from_secs(15), rx)
            .await
            .map_err(|_| anyhow!("Timeout waiting for response to method '{}'", method))??;

        if let Some(err) = resp.error {
            return Err(anyhow!("Codex JSON-RPC error ({}): {}", err.code, err.message));
        }

        Ok(resp.result.unwrap_or(Value::Null))
    }

    /// Convenience for text-only turns (tests, scheduled prompts).
    pub async fn run_turn(&self, prompt: &str, tier: ModelTier) -> Result<String> {
        self.run_turn_inputs(&[TurnInput::Text(prompt.to_string())], tier)
            .await
    }

    pub async fn run_turn_inputs(&self, inputs: &[TurnInput], tier: ModelTier) -> Result<String> {
        let thread_id = {
            let lock = self.active_thread_id.lock().await;
            lock.clone().ok_or_else(|| anyhow!("No active Codex thread"))?
        };
        self.run_turn_on(&thread_id, inputs, tier).await
    }

    /// Run a turn on a specific thread, leaving the main conversation alone.
    pub async fn run_turn_on(
        &self,
        thread_id: &str,
        inputs: &[TurnInput],
        tier: ModelTier,
    ) -> Result<String> {
        let thread_id = thread_id.to_string();

        // Register before turn/start so no event can be missed in the gap.
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(256);
        let listener_id = Uuid::new_v4().to_string();
        {
            let mut lock = self.turn_listeners.lock().await;
            lock.insert(
                listener_id.clone(),
                TurnListener {
                    thread_id: thread_id.clone(),
                    tx,
                },
            );
        }

        // Model and effort are set per turn, not just per thread: a scheduled
        // sweep and a hard debugging session can share a thread and should not
        // share a price.
        let turn_req = json!({
            "threadId": thread_id,
            "input": inputs.iter().map(TurnInput::to_json).collect::<Vec<_>>(),
            "model": tier.model,
            "effort": tier.effort,
        });

        info!(
            target: "codex::turn",
            "Sending turn/start to Codex thread {} on {} ({} effort)",
            thread_id, tier.model, tier.effort
        );
        let start_result = self.send_request("turn/start", Some(turn_req)).await;

        let outcome = match start_result {
            Err(e) => Err(e),
            Ok(res) => {
                // Record the turn id from the response as well as the event
                // stream: a burst arriving immediately after turn/start must be
                // steerable without waiting for the turn/started notification.
                if let Some(turn_id) = res.get("turn").and_then(|t| t.get("id")).and_then(|i| i.as_str()) {
                    self.active_turns
                        .lock()
                        .await
                        .insert(thread_id.clone(), turn_id.to_string());
                }
                Self::collect_turn(&mut rx).await
            }
        };

        {
            let mut lock = self.turn_listeners.lock().await;
            lock.remove(&listener_id);
        }
        self.active_turns.lock().await.remove(&thread_id);

        outcome
    }

    /// Drain turn events until completion, preferring the final `agentMessage`
    /// text over accumulated deltas (identical in practice, but the completed
    /// item is what the app-server considers authoritative). There is no wall
    /// clock cap because a turn may be installing dependencies or waiting on a
    /// long-running tool; process death and a closed event stream still fail it.
    async fn collect_turn(rx: &mut mpsc::Receiver<TurnEvent>) -> Result<String> {
        let mut deltas = String::new();
        let mut final_message: Option<String> = None;

        loop {
            match rx.recv().await {
                Some(TurnEvent::Delta(d)) => deltas.push_str(&d),
                Some(TurnEvent::Message(text)) => final_message = Some(text),
                Some(TurnEvent::Completed) => break,
                Some(TurnEvent::Failed(reason)) => {
                    return Err(anyhow!("Codex turn failed: {}", reason));
                }
                None => return Err(anyhow!("Codex event stream closed mid-turn")),
            }
        }

        // An empty final message is not a failure. When the agent answers through
        // the send_message tool it has already said everything it needs to, and
        // ends the turn with no closing text, reporting that as an error made
        // every successful tool-based reply look broken in the log.
        Ok(final_message.unwrap_or(deltas))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_thread_origin_reads_distinctly_in_logs() {
        assert_eq!(ThreadOrigin::Resumed.to_string(), "RESUMED");
        assert_eq!(ThreadOrigin::Created.to_string(), "NEW");
        assert_ne!(ThreadOrigin::Resumed, ThreadOrigin::Created);
    }

    #[test]
    fn test_thread_info_carries_its_origin() {
        let res = json!({"thread": {"id": "t1"}, "model": "gpt-5.6-sol"});
        let resumed = ThreadInfo::from_result(&res, ThreadOrigin::Resumed).unwrap();
        let created = ThreadInfo::from_result(&res, ThreadOrigin::Created).unwrap();
        assert_eq!(resumed.origin, ThreadOrigin::Resumed);
        assert_eq!(created.origin, ThreadOrigin::Created);
        assert_eq!(created.model, "gpt-5.6-sol");
    }

    /// Nobody is at the other end of an approval prompt, and the whole posture is
    /// full access. Every approval path must therefore answer, and answer yes.
    #[test]
    fn test_every_approval_request_is_granted() {
        for method in [
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "execCommandApproval",
            "applyPatchApproval",
            "item/permissions/requestApproval",
        ] {
            let reply = CodexProcessManager::answer_server_request(&json!(7), method, None);
            assert_eq!(reply["id"], 7);
            assert!(reply.get("error").is_none(), "{method} was refused: {reply}");
            let result = &reply["result"];
            let granted = matches!(
                result["decision"].as_str(),
                Some("accept" | "acceptForSession" | "approved" | "approved_for_session")
            ) || result.get("permissions").is_some();
            assert!(granted, "{method} did not grant: {result}");
        }
    }

    /// The permission grant has to be the whole machine and the network, or the
    /// agent cannot install a package or read a file outside its workspace.
    #[test]
    fn test_permission_grant_covers_the_filesystem_and_the_network() {
        let reply = CodexProcessManager::answer_server_request(
            &json!(1),
            "item/permissions/requestApproval",
            None,
        );
        let permissions = &reply["result"]["permissions"];
        assert_eq!(permissions["fileSystem"]["entries"][0]["path"]["path"], "/");
        assert_eq!(permissions["fileSystem"]["entries"][0]["access"], "write");
        assert_eq!(permissions["network"]["enabled"], true);
        assert_eq!(reply["result"]["scope"], "session");
    }

    /// An unknown request still has to be answered. Silence stalls the turn until
    /// the 300s timeout, which the agent reports as a refusal.
    #[test]
    fn test_an_unknown_request_gets_an_error_not_silence() {
        let reply = CodexProcessManager::answer_server_request(&json!(9), "some/newThing", None);
        assert_eq!(reply["id"], 9);
        assert_eq!(reply["error"]["code"], -32601);
    }

    #[test]
    fn test_classify_agent_message_delta() {
        let v = json!({
            "method": "item/agentMessage/delta",
            "params": {"threadId": "t1", "turnId": "u1", "delta": "pong"}
        });
        let (thread, event) = CodexProcessManager::classify_notification(&v).unwrap();
        assert_eq!(thread, "t1");
        assert!(matches!(event, TurnEvent::Delta(d) if d == "pong"));
    }

    #[test]
    fn test_classify_completed_agent_message() {
        let v = json!({
            "method": "item/completed",
            "params": {
                "threadId": "t1",
                "item": {"type": "agentMessage", "id": "m1", "text": "pong"}
            }
        });
        let (_, event) = CodexProcessManager::classify_notification(&v).unwrap();
        assert!(matches!(event, TurnEvent::Message(t) if t == "pong"));
    }

    #[test]
    fn test_user_message_item_is_ignored() {
        // Otherwise the assistant would echo the user's own text back at them.
        let v = json!({
            "method": "item/completed",
            "params": {
                "threadId": "t1",
                "item": {"type": "userMessage", "id": "m1", "text": "hi"}
            }
        });
        assert!(CodexProcessManager::classify_notification(&v).is_none());
    }

    #[test]
    fn test_noise_notifications_are_ignored() {
        for method in ["mcpServer/startupStatus/updated", "thread/tokenUsage/updated"] {
            let v = json!({"method": method, "params": {"threadId": "t1"}});
            assert!(CodexProcessManager::classify_notification(&v).is_none());
        }
    }

    #[tokio::test]
    async fn test_collect_turn_prefers_final_message() {
        let (tx, mut rx) = mpsc::channel(8);
        tx.send(TurnEvent::Delta("po".into())).await.unwrap();
        tx.send(TurnEvent::Delta("ng".into())).await.unwrap();
        tx.send(TurnEvent::Message("pong".into())).await.unwrap();
        tx.send(TurnEvent::Completed).await.unwrap();
        assert_eq!(CodexProcessManager::collect_turn(&mut rx).await.unwrap(), "pong");
    }

    /// Regression: a turn that answers entirely through send_message ends with
    /// an empty final message. Treating that as an error made every successful
    /// tool-based reply log as a failure.
    #[tokio::test]
    async fn test_empty_final_message_is_not_an_error() {
        let (tx, mut rx) = mpsc::channel(8);
        tx.send(TurnEvent::Message(String::new())).await.unwrap();
        tx.send(TurnEvent::Completed).await.unwrap();
        assert_eq!(CodexProcessManager::collect_turn(&mut rx).await.unwrap(), "");
    }

    #[tokio::test]
    async fn test_collect_turn_falls_back_to_deltas() {
        let (tx, mut rx) = mpsc::channel(8);
        tx.send(TurnEvent::Delta("pong".into())).await.unwrap();
        tx.send(TurnEvent::Completed).await.unwrap();
        assert_eq!(CodexProcessManager::collect_turn(&mut rx).await.unwrap(), "pong");
    }

    /// Steering is only legal between `turn/started` and the turn ending, and
    /// needs that turn's id. Both come from the event stream.
    #[tokio::test]
    async fn test_active_turn_is_tracked_across_a_turn_lifecycle() {
        let turns = Arc::new(Mutex::new(HashMap::new()));

        CodexProcessManager::track_active_turn(
            &turns,
            &json!({"method": "turn/started", "params": {"threadId": "t1", "turn": {"id": "u1"}}}),
        )
        .await;
        assert_eq!(turns.lock().await.get("t1").map(String::as_str), Some("u1"));

        CodexProcessManager::track_active_turn(
            &turns,
            &json!({"method": "turn/completed", "params": {"threadId": "t1"}}),
        )
        .await;
        assert!(turns.lock().await.is_empty());
    }

    #[tokio::test]
    async fn test_a_failed_turn_is_no_longer_steerable() {
        let turns = Arc::new(Mutex::new(HashMap::new()));
        turns.lock().await.insert("t1".to_string(), "u1".to_string());

        CodexProcessManager::track_active_turn(
            &turns,
            &json!({"method": "turn/failed", "params": {"threadId": "t1"}}),
        )
        .await;
        assert!(turns.lock().await.is_empty());
    }

    /// Other threads' turns must not be mistaken for this one's: a scheduled task
    /// runs on its own thread while the conversation is idle.
    #[tokio::test]
    async fn test_turns_are_tracked_per_thread() {
        let turns = Arc::new(Mutex::new(HashMap::new()));
        for (thread, turn) in [("t1", "u1"), ("t2", "u2")] {
            CodexProcessManager::track_active_turn(
                &turns,
                &json!({"method": "turn/started", "params": {"threadId": thread, "turnId": turn}}),
            )
            .await;
        }
        CodexProcessManager::track_active_turn(
            &turns,
            &json!({"method": "turn/completed", "params": {"threadId": "t2"}}),
        )
        .await;

        let lock = turns.lock().await;
        assert_eq!(lock.get("t1").map(String::as_str), Some("u1"));
        assert!(!lock.contains_key("t2"));
    }

    #[tokio::test]
    async fn test_collect_turn_surfaces_failure() {
        let (tx, mut rx) = mpsc::channel(8);
        tx.send(TurnEvent::Failed("model exploded".into())).await.unwrap();
        let err = CodexProcessManager::collect_turn(&mut rx).await.unwrap_err();
        assert!(err.to_string().contains("model exploded"));
    }

    #[tokio::test]
    async fn test_process_death_fails_waiting_turn() {
        let listeners = Arc::new(Mutex::new(HashMap::new()));
        let (tx, mut rx) = mpsc::channel(8);
        listeners.lock().await.insert(
            "listener".to_string(),
            TurnListener {
                thread_id: "thread".to_string(),
                tx,
            },
        );

        let waiting = tokio::spawn(async move {
            CodexProcessManager::collect_turn(&mut rx).await
        });
        CodexProcessManager::fail_listeners(&listeners, "codex app-server exited").await;

        let error = waiting.await.unwrap().unwrap_err();
        assert!(error.to_string().contains("codex app-server exited"));
    }
}

