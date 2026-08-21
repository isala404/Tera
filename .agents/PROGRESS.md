# Progress

## 2026-08-21  Simplification pass

Full audit of all 14,345 lines, executed across three tiers. 200 tests before, 206 after. `cargo clippy --all-targets` clean.

Guiding decision: Tera is a thin substrate around Codex. Machinery that encodes judgement in Rust gets deleted or moved to `data/`, because a better model should improve the system without a code change.

### Done

Tier 1, mechanical:
- [x] T1.1  `history/db.rs` row-mapper and attachment-loader shared across its three queries
- [x] T1.2  `templates.rs` 10 wrappers and `init.rs` 12 call sites collapsed to two tables; markdown seeds moved to `data/`, which also fixed three copies of the memory seeds that had drifted apart
- [x] T1.3  `runtime/fs.rs::write_atomic` replaced 6 hand-rolled temp-write-rename blocks, two of which never fsynced the parent directory and could lose the rename
- [x] T1.4  `memory/rebuild.rs` and `optimizer.rs` became one `memory/pass.rs` with a `Pass` describing what differs
- [x] T1.5  dead code swept, including a `maintenance_runs` table nothing had ever read or written
- [x] T1.6  clap `--workspace` flattened across 14 variants; `require_str` and stdio `respond` shared
- [x] T1.7  the two divergent `add_column_if_missing` bodies became `src/sqlite.rs`
- [x] T1.8  PLAN.md decided, see below
- [x] T1.9  misc dedup: `send_text`->`client_for`, `send_media` dead tuple, `run_from_row`, generation enumeration, secret capture outcomes

Tier 2, structural:
- [x] T2.1  the scheduler owns its two tables, their structs and its migration, next to the queries that read them. `RuntimeDb::open` calls `scheduler::db::init_schema`
- [x] T2.2  `codex/log.rs` took the 200-line notification formatter out of the 1200-line process manager
- [x] T2.3  `ConversationEvent.kind` is an `EventKind` enum. The stored text is unchanged because the agent queries that column with `sqlite3`
- [x] T2.4  `runtime/phoenix.rs` -> `runtime/crash_mark.rs`

Tier 3, accepted bets:
- [x] T3.1  the self-supervising restarter is gone, narrowed, see below
- [x] T3.2  one memory pipeline instead of two, narrowed, see below

### Decisions

- SCOPE: T3.1 deleted only the self-restarting half. The 15-second detached SIGUSR1 signaller stays, because without it the daemon dies mid-turn and the owner never hears the reply to the update they asked for. Process lifecycle is substrate, which is exactly what Tera is supposed to own.
- SCOPE: T3.2 kept `memory/generations.rs`. Numbered generations, the atomic symlink swap and `tera memory rollback` are documented, user-facing durability, not encoded judgement. What collapsed was the two near-identical pipelines above it.
- TRADEOFF: T3.1 means a daemon started by hand stays down after an update. `deploy/tera.service` carries `Restart=always`; README says so.
- DECIDED: the 44 `PLAN.md section N` citations are gone. PLAN.md was never committed, so they pointed at nothing anyone else could read. The surrounding prose already carried the why.
- BEHAVIOUR: a successful rebuild now also stamps `LAST_OPTIMIZED_KEY`. A rebuild re-derives the whole tree from history, which is strictly more than the nightly pass does, so it counts as the day's compaction.
- KEEP: the comment lines are load-bearing. They cite specific past regressions and are the only durable why in a repo maintained by agents that start each session blank. Only restatements, section banners and dead citations were cut.
- KEEP: `secrets.rs`, `history/backup.rs`, `scheduler/recurrence.rs`, the `Transport` trait, and the inline test lines. All earn their length.

Net Rust is 14,868 -> 14,730 lines. The line count is not the result; roughly 1,500 lines were deleted and a similar number rewritten in fewer places with more tests. What changed is that there is now one way to do each of these things instead of two or three.

### Open

- TODO: outbound quoted replies never reach the wire. `WhatsAppWebTransport::send_text` takes `_reply_to_provider_id` and discards it, and `send_media` has no ContextInfo the SDK exposes. `MockTransport` records it faithfully, so every test passes. Predates this pass, not caused by it. Fixing it needs a WhatsApp `ContextInfo` on the outbound message, which whatsapp-rust 0.7 does not surface on these calls.
