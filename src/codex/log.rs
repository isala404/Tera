//! Turning the app-server's event stream into a log a human can read.
//!
//! Nothing here affects a turn; it exists so that when a turn misbehaves you can
//! see which command ran, which tool was called and what it cost.

use crate::secrets::SecretStore;
use serde_json::Value;
use tracing::{debug, info, warn};

/// Remove ANSI colour sequences so Codex's coloured output does not arrive as
/// literal `\x1b[2m` noise in our log.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        // CSI sequence: ESC [ ... <final byte in @..~>
        if chars.next() == Some('[') {
            for c in chars.by_ref() {
                if ('\u{40}'..='\u{7e}').contains(&c) {
                    break;
                }
            }
        }
    }
    out
}

/// Whether a Codex stderr line is something the operator should see.
///
/// Codex logs its own INFO telemetry to stderr; only genuine failures deserve a
/// warning in the daemon's log.
pub fn is_stderr_problem(line: &str) -> bool {
    let lowered = line.to_lowercase();
    ["error", "panic", "fatal", " warn"]
        .iter()
        .any(|needle| lowered.contains(needle))
}

/// Keep a log line readable: agent output and command dumps can be megabytes.
pub fn truncate(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}… (+{} more chars)", s.chars().count() - max)
}

fn text_of(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(str::to_string)
}

fn num_of(v: &Value, key: &str) -> i64 {
    v.get(key).and_then(|x| x.as_i64()).unwrap_or_default()
}

use std::collections::HashMap;
use std::sync::Mutex;

static STREAM_BUFFERS: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);

fn redact_stream_delta(stream_key: &str, delta: &str, secrets: &SecretStore) -> Option<String> {
    let mut lock = STREAM_BUFFERS.lock().unwrap();
    let map = lock.get_or_insert_with(HashMap::new);
    let buf = map.entry(stream_key.to_string()).or_default();
    buf.push_str(delta);

    let active_secrets: Vec<String> = secrets
        .names()
        .ok()
        .map(|names| {
            names
                .into_iter()
                .filter_map(|(name, _)| secrets.get(&name).ok().flatten())
                .filter(|v| v.len() >= 6)
                .collect()
        })
        .unwrap_or_default();

    if active_secrets.is_empty() {
        let out = std::mem::take(buf);
        return if out.is_empty() { None } else { Some(out) };
    }

    let redacted = secrets.redact(buf);

    let mut max_prefix_len = 0;
    for secret in &active_secrets {
        for k in 1..=secret.len().min(redacted.len()) {
            if secret.is_char_boundary(k) && redacted.ends_with(&secret[..k]) {
                max_prefix_len = max_prefix_len.max(k);
            }
        }
    }

    if max_prefix_len > 0 {
        let split_idx = redacted.len().saturating_sub(max_prefix_len);
        if redacted.is_char_boundary(split_idx) {
            let to_emit = redacted[..split_idx].to_string();
            *buf = redacted[split_idx..].to_string();
            if to_emit.is_empty() {
                None
            } else {
                Some(to_emit)
            }
        } else {
            None
        }
    } else {
        *buf = String::new();
        Some(redacted)
    }
}

fn flush_stream_delta(stream_key: &str, secrets: &SecretStore) -> Option<String> {
    let mut lock = STREAM_BUFFERS.lock().unwrap();
    let map = lock.as_mut()?;
    let buf = map.remove(stream_key)?;
    if buf.is_empty() {
        return None;
    }
    Some(secrets.redact(&buf))
}

fn clear_turn_buffers(thread_id: &str, secrets: &SecretStore) {
    let mut lock = STREAM_BUFFERS.lock().unwrap();
    if let Some(map) = lock.as_mut() {
        let prefix = format!("{thread_id}:");
        let keys: Vec<String> = map
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();
        for k in keys {
            if let Some(buf) = map.remove(&k) {
                if !buf.is_empty() {
                    let text = secrets.redact(&buf);
                    debug!(target: "codex::stream", "flushed turn buffer: {}", truncate(&text, 400));
                }
            }
        }
    }
}

/// `info` carries the things worth seeing on every turn. Shell commands,
/// MCP tool calls, file edits, web searches, token usage. `debug` carries the
/// streaming firehose (output deltas, reasoning text) that is only useful
/// when chasing a specific problem; enable with
/// `RUST_LOG=tera::codex=debug`.
pub fn log_notification(v: &Value, secrets: &SecretStore) {
    let Some(method) = v.get("method").and_then(|m| m.as_str()) else {
        return;
    };
    let params = v.get("params").unwrap_or(&Value::Null);

    match method {
        "item/started" | "item/completed" => {
            if method == "item/completed" {
                let thread_id = params
                    .get("threadId")
                    .and_then(Value::as_str)
                    .unwrap_or("global");
                if let Some(item_id) = params
                    .get("item")
                    .and_then(|i| i.get("id"))
                    .and_then(Value::as_str)
                {
                    let stream_key = format!("{thread_id}:{item_id}");
                    if let Some(flushed) = flush_stream_delta(&stream_key, secrets) {
                        debug!(target: "codex::stream", "flushed output: {}", truncate(&flushed, 400));
                    }
                }
            }
            log_item(method, params, secrets);
        }

        "item/commandExecution/outputDelta" | "process/outputDelta" => {
            let chunk = text_of(params, "chunk")
                .or_else(|| text_of(params, "delta"))
                .unwrap_or_default();
            let thread_id = params
                .get("threadId")
                .and_then(Value::as_str)
                .unwrap_or("global");
            let item_id = params
                .get("itemId")
                .and_then(Value::as_str)
                .unwrap_or("cmd_out");
            let stream_key = format!("{thread_id}:{item_id}");
            if let Some(text) = redact_stream_delta(&stream_key, &chunk, secrets) {
                debug!(target: "codex::stream", "command output: {}", truncate(&text, 400));
            }
        }
        "item/reasoning/summaryTextDelta" | "item/reasoning/textDelta" => {
            let delta = text_of(params, "delta").unwrap_or_default();
            let thread_id = params
                .get("threadId")
                .and_then(Value::as_str)
                .unwrap_or("global");
            let item_id = params
                .get("itemId")
                .and_then(Value::as_str)
                .unwrap_or("reasoning");
            let stream_key = format!("{thread_id}:{item_id}");
            if let Some(text) = redact_stream_delta(&stream_key, &delta, secrets) {
                debug!(target: "codex::stream", "reasoning: {}", truncate(&text, 400));
            }
        }
        "item/agentMessage/delta" => {
            let delta = text_of(params, "delta").unwrap_or_default();
            let thread_id = params
                .get("threadId")
                .and_then(Value::as_str)
                .unwrap_or("global");
            let item_id = params
                .get("itemId")
                .and_then(Value::as_str)
                .unwrap_or("agent_msg");
            let stream_key = format!("{thread_id}:{item_id}");
            if let Some(text) = redact_stream_delta(&stream_key, &delta, secrets) {
                debug!(target: "codex::stream", "answer: {}", truncate(&text, 200));
            }
        }
        "item/mcpToolCall/progress" => {
            let raw_params = params.to_string();
            let redacted_params = secrets.redact(&raw_params);
            debug!(target: "codex::mcp", "MCP tool progress: {}", truncate(&redacted_params, 300));
        }

        "turn/started" => info!(target: "codex::turn", "turn started"),
        "turn/completed" => {
            let thread_id = params
                .get("threadId")
                .and_then(Value::as_str)
                .unwrap_or("global");
            clear_turn_buffers(thread_id, secrets);
            info!(target: "codex::turn", "turn completed");
        }
        "turn/failed" => {
            let thread_id = params
                .get("threadId")
                .and_then(Value::as_str)
                .unwrap_or("global");
            clear_turn_buffers(thread_id, secrets);
            let raw_params = params.to_string();
            let redacted_params = secrets.redact(&raw_params);
            warn!(target: "codex::turn", "turn failed: {}", truncate(&redacted_params, 500));
        }

        "thread/tokenUsage/updated" => {
            if let Some(last) = params.get("tokenUsage").and_then(|u| u.get("last")) {
                info!(
                    target: "codex::usage",
                    "tokens: in={} cached={} out={} (context window {})",
                    num_of(last, "inputTokens"),
                    num_of(last, "cachedInputTokens"),
                    num_of(last, "outputTokens"),
                    params.get("tokenUsage").and_then(|u| u.get("modelContextWindow")).and_then(|c| c.as_i64()).unwrap_or(0),
                );
            }
        }

        "mcpServer/startupStatus/updated" => {
            let name = text_of(params, "name").unwrap_or_default();
            let status = text_of(params, "status").unwrap_or_default();
            match params.get("error").and_then(|e| e.as_str()) {
                Some(err) => warn!(target: "codex::mcp", "MCP server '{name}' {status}: {err}"),
                None => info!(target: "codex::mcp", "MCP server '{name}' {status}"),
            }
        }

        "error" | "guardianWarning" | "configWarning" | "deprecationNotice" => {
            warn!(target: "codex", "{method}: {}", truncate(&secrets.redact(&params.to_string()), 500));
        }
        "model/rerouted" => {
            info!(target: "codex", "model rerouted: {}", truncate(&secrets.redact(&params.to_string()), 200))
        }

        _ => {
            debug!(target: "codex::raw", "{method}: {}", truncate(&secrets.redact(&params.to_string()), 300))
        }
    }
}

/// One line per thread item, with the detail that makes it identifiable:
/// which command ran, which MCP tool was called, which files changed.
fn log_item(method: &str, params: &Value, secrets: &SecretStore) {
    let Some(item) = params.get("item") else {
        return;
    };
    let Some(kind) = item.get("type").and_then(|t| t.as_str()) else {
        return;
    };
    let finished = method == "item/completed";

    match kind {
        "commandExecution" => {
            let cmd = truncate(
                &secrets.redact(&text_of(item, "command").unwrap_or_default()),
                300,
            );
            if finished {
                let exit = num_of(item, "exitCode");
                let ms = num_of(item, "durationMs");
                let out = truncate(
                    &secrets.redact(&text_of(item, "aggregatedOutput").unwrap_or_default()),
                    500,
                );
                if exit == 0 {
                    info!(target: "codex::exec", "$ {cmd} -> exit {exit} in {ms}ms\n{out}");
                } else {
                    warn!(target: "codex::exec", "$ {cmd} -> exit {exit} in {ms}ms\n{out}");
                }
            } else {
                info!(target: "codex::exec", "$ {cmd}");
            }
        }

        "mcpToolCall" => {
            let server = text_of(item, "server").unwrap_or_default();
            let tool = text_of(item, "tool").unwrap_or_default();
            if finished {
                let ms = num_of(item, "durationMs");
                match item
                    .get("error")
                    .and_then(|e| if e.is_null() { None } else { Some(e) })
                {
                    Some(err) => {
                        warn!(target: "codex::mcp", "{server}.{tool} failed in {ms}ms: {}", truncate(&secrets.redact(&err.to_string()), 400))
                    }
                    None => {
                        info!(target: "codex::mcp", "{server}.{tool} ok in {ms}ms -> {}", truncate(&secrets.redact(&item.get("result").map(|r| r.to_string()).unwrap_or_default()), 400))
                    }
                }
            } else {
                info!(target: "codex::mcp", "{server}.{tool} calling with {}", truncate(&secrets.redact(&item.get("arguments").map(|a| a.to_string()).unwrap_or_default()), 400));
            }
        }

        "fileChange" if finished => {
            let files: Vec<String> = item
                .get("changes")
                .and_then(|c| c.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|c| c.get("path").and_then(|p| p.as_str()).map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            info!(target: "codex::files", "edited {} file(s): {}", files.len(), files.join(", "));
        }

        "webSearch" if finished => {
            let query = text_of(item, "query").unwrap_or_default();
            info!(target: "codex::web", "searched: {}", truncate(&secrets.redact(&query), 200));
        }

        "agentMessage" if finished => {
            let text = text_of(item, "text").unwrap_or_default();
            info!(target: "codex::answer", "{}", truncate(&secrets.redact(&text), 500));
        }

        "reasoning" if finished => {
            let raw = item.to_string();
            debug!(target: "codex::reasoning", "{}", truncate(&secrets.redact(&raw), 600));
        }

        "plan" if finished => {
            let text = text_of(item, "text").unwrap_or_default();
            info!(target: "codex::plan", "{}", truncate(&secrets.redact(&text), 400));
        }

        other if finished => debug!(target: "codex::item", "{other} completed"),
        _ => {}
    }
}

#[cfg(test)]
mod stderr_tests {
    use super::*;

    #[test]
    fn test_strip_ansi_removes_colour_sequences() {
        // Verbatim shape of a codex app-server stderr line.
        let raw = "\u{1b}[2m2026-08-17T14:47:49Z\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m \u{1b}[2mcodex_otel\u{1b}[0m: ready";
        let clean = strip_ansi(raw);
        assert_eq!(clean, "2026-08-17T14:47:49Z  INFO codex_otel: ready");
        assert!(!clean.contains('\u{1b}'));
    }

    #[test]
    fn test_startup_telemetry_is_not_a_problem() {
        let line = "2026-08-17T14:47:49Z  INFO codex_otel.trace_safe: \
                    event.name=\"codex.startup_phase\" startup.status=\"ready\" duration_ms=3421";
        assert!(!is_stderr_problem(&strip_ansi(line)));
    }

    #[test]
    fn test_real_failures_are_still_surfaced() {
        for line in [
            "2026-08-17T14:47:49Z ERROR codex: failed to reach model provider",
            "thread 'main' panicked at src/lib.rs:1:1",
            "2026-08-17T14:47:49Z  WARN codex: retrying request",
        ] {
            assert!(is_stderr_problem(line), "should have been surfaced: {line}");
        }
    }

    #[test]
    fn test_streaming_redaction_unicode_credential_no_panic() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = SecretStore::new(temp_dir.path().join("secrets.json"));
        store.set("UNICODE_PASS", "p@sswörd🔥secret", 0).unwrap();

        let stream_key = "thread_u:item_u";
        let delta1 = "The password is p@ss";
        let out1 = redact_stream_delta(stream_key, delta1, &store);
        let delta2 = "wörd🔥secret confirmed";
        let out2 = redact_stream_delta(stream_key, delta2, &store);
        let flushed = flush_stream_delta(stream_key, &store);
        let full = format!(
            "{}{}{}",
            out1.unwrap_or_default(),
            out2.unwrap_or_default(),
            flushed.unwrap_or_default()
        );
        assert!(!full.contains("p@sswörd🔥secret"));
        assert!(full.contains("[redacted UNICODE_PASS]"));
    }

    #[test]
    fn test_overlapping_turns_streaming_redaction_isolation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = SecretStore::new(temp_dir.path().join("secrets.json"));
        store.set("API_KEY", "super_secret_token_12345", 0).unwrap();

        // Thread 1 sends first half of secret
        let t1_key = "thread_1:item_1";
        let t1_chunk = "Bearer super_secret_";
        let _ = redact_stream_delta(t1_key, t1_chunk, &store);

        // Thread 2 completes its turn
        clear_turn_buffers("thread_2", &store);

        // Thread 1's buffer must NOT be cleared by thread 2's completion
        let t1_chunk2 = "token_12345 in header";
        let out = redact_stream_delta(t1_key, t1_chunk2, &store);
        let flushed = flush_stream_delta(t1_key, &store);
        let combined = format!("{}{}", out.unwrap_or_default(), flushed.unwrap_or_default());
        assert!(
            !combined.contains("super_secret_token_12345"),
            "secret leaked after overlapping turn completion"
        );
        assert!(combined.contains("[redacted API_KEY]"));
    }

    #[test]
    fn test_completed_items_redaction() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = SecretStore::new(temp_dir.path().join("secrets.json"));
        store.set("DB_PASS", "super_secret_pass_999", 0).unwrap();

        let raw_answer = "Your database password is super_secret_pass_999";
        let item = serde_json::json!({
            "type": "agentMessage",
            "text": raw_answer
        });
        let params = serde_json::json!({ "item": item });

        // log_item directly uses secrets.redact on agentMessage text
        let text = text_of(&item, "text").unwrap();
        let redacted = store.redact(&text);
        assert!(!redacted.contains("super_secret_pass_999"));
        assert!(redacted.contains("[redacted DB_PASS]"));

        // Verify log_notification does not panic and processes completed items
        log_notification(
            &serde_json::json!({
                "method": "item/completed",
                "params": params
            }),
            &store,
        );
    }
}
