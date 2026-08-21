//! Turning the app-server's event stream into a log a human can read.
//!
//! Nothing here affects a turn; it exists so that when a turn misbehaves you can
//! see which command ran, which tool was called and what it cost.

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

/// `info` carries the things worth seeing on every turn. Shell commands,
/// MCP tool calls, file edits, web searches, token usage. `debug` carries the
/// streaming firehose (output deltas, reasoning text) that is only useful
/// when chasing a specific problem; enable with
/// `RUST_LOG=tera::codex=debug`.
pub fn log_notification(v: &Value) {
    let Some(method) = v.get("method").and_then(|m| m.as_str()) else {
        return;
    };
    let params = v.get("params").unwrap_or(&Value::Null);

    match method {
        "item/started" | "item/completed" => log_item(method, params),

        "item/commandExecution/outputDelta" | "process/outputDelta" => {
            debug!(target: "codex::stream", "command output: {}", truncate(&text_of(params, "chunk").or_else(|| text_of(params, "delta")).unwrap_or_default(), 400));
        }
        "item/reasoning/summaryTextDelta" | "item/reasoning/textDelta" => {
            debug!(target: "codex::stream", "reasoning: {}", truncate(&text_of(params, "delta").unwrap_or_default(), 400));
        }
        "item/agentMessage/delta" => {
            debug!(target: "codex::stream", "answer: {}", truncate(&text_of(params, "delta").unwrap_or_default(), 200));
        }
        "item/mcpToolCall/progress" => {
            debug!(target: "codex::mcp", "MCP tool progress: {}", truncate(&params.to_string(), 300));
        }

        "turn/started" => info!(target: "codex::turn", "turn started"),
        "turn/completed" => info!(target: "codex::turn", "turn completed"),
        "turn/failed" => warn!(target: "codex::turn", "turn failed: {}", truncate(&params.to_string(), 500)),

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
            warn!(target: "codex", "{method}: {}", truncate(&params.to_string(), 500));
        }
        "model/rerouted" => info!(target: "codex", "model rerouted: {}", truncate(&params.to_string(), 200)),

        _ => debug!(target: "codex::raw", "{method}: {}", truncate(&params.to_string(), 300)),
    }
}

/// One line per thread item, with the detail that makes it identifiable:
/// which command ran, which MCP tool was called, which files changed.
fn log_item(method: &str, params: &Value) {
    let Some(item) = params.get("item") else { return };
    let Some(kind) = item.get("type").and_then(|t| t.as_str()) else { return };
    let finished = method == "item/completed";

    match kind {
        "commandExecution" => {
            let cmd = truncate(&text_of(item, "command").unwrap_or_default(), 300);
            if finished {
                let exit = num_of(item, "exitCode");
                let ms = num_of(item, "durationMs");
                let out = truncate(&text_of(item, "aggregatedOutput").unwrap_or_default(), 500);
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
                match item.get("error").and_then(|e| if e.is_null() { None } else { Some(e) }) {
                    Some(err) => warn!(target: "codex::mcp", "{server}.{tool} failed in {ms}ms: {}", truncate(&err.to_string(), 400)),
                    None => info!(target: "codex::mcp", "{server}.{tool} ok in {ms}ms -> {}", truncate(&item.get("result").map(|r| r.to_string()).unwrap_or_default(), 400)),
                }
            } else {
                info!(target: "codex::mcp", "{server}.{tool} calling with {}", truncate(&item.get("arguments").map(|a| a.to_string()).unwrap_or_default(), 400));
            }
        }

        "fileChange" if finished => {
            let files: Vec<String> = item
                .get("changes")
                .and_then(|c| c.as_array())
                .map(|arr| arr.iter().filter_map(|c| c.get("path").and_then(|p| p.as_str()).map(str::to_string)).collect())
                .unwrap_or_default();
            info!(target: "codex::files", "edited {} file(s): {}", files.len(), files.join(", "));
        }

        "webSearch" if finished => {
            info!(target: "codex::web", "searched: {}", truncate(&text_of(item, "query").unwrap_or_default(), 200));
        }

        "agentMessage" if finished => {
            info!(target: "codex::answer", "{}", truncate(&text_of(item, "text").unwrap_or_default(), 500));
        }

        "reasoning" if finished => {
            debug!(target: "codex::reasoning", "{}", truncate(&item.to_string(), 600));
        }

        "plan" if finished => {
            info!(target: "codex::plan", "{}", truncate(&text_of(item, "text").unwrap_or_default(), 400));
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
}
