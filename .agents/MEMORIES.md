# Tera

Rust daemon paired to a WhatsApp account. Each inbound message becomes a turn run by a spawned `codex app-server` against a local workspace. Codex replies through the `send_message` MCP tool.

Build: `cargo build`  Test: `cargo test --all`  Lint: `cargo clippy --all-targets`

## The core idea

Tera is a thin durable substrate around Codex. It owns what a model cannot: the WhatsApp pairing, the event store, the workspace on disk, the process lifecycle. It does not own intelligence. Anything that encodes judgement in Rust is in the wrong place and belongs in a prompt under `data/`, where a better model improves it for free.

Test for new code: would this need rewriting when the model gets better? If yes, it belongs in `data/`, not `src/`.

## Layout

- `data/` is the single source for every prompt, instruction file and template. Nothing user-facing is authored in Rust; `include_str!` bakes it into the binary. Substitution is `{{PLACEHOLDER}}` plain string replacement, deliberately not a template engine.
- Two SQLite stores, and the split is intentional. `history/history.sqlite3` is authoritative, backed up, and queried directly by the agent with `jq`/`sqlite3`. `.runtime/state.sqlite3` is disposable daemon state.
- Reading history is deliberately not a tool. The agent shells out.
- `codex/tier.rs` pins which model each kind of work asks for. `memory/maintenance.rs` notices when the available default changes.

## Conventions

- Comments record why, and most cite a specific past regression. They are the only durable "why" in a repo maintained mostly by agents with no memory between sessions. Do not cull them for tidiness.
- Tests are named as behavioural sentences (`test_a_crashed_first_start_restores_the_previous_binary`).
- `anyhow::Result` everywhere, `.with_context()` on IO.
- Unix-only by design; both release targets are unix. Do not add `#[cfg(unix)]` guards.
- `write_atomic` in `runtime/fs.rs` is the one way to write a file that must not tear.

## Gotchas

- `rusqlite` held at 0.39 on purpose. 0.40 wants libsqlite3-sys 0.38 while whatsapp-rust-sqlite-storage pins ^0.37, and only one crate may link native sqlite3.
- Wire format verified against Codex CLI 0.147.0. A materially different version may break parsing.
- The `cron` crate rejects 5-field crontab syntax and numbers weekdays 1=Sunday against crontab's 0=Sunday. `scheduler/recurrence.rs` translates; that code is scar tissue, leave it.
- WhatsApp media URLs are single-use, so media bytes are carried, never a handle.
