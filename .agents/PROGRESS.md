# Progress

## 2026-08-21  Simplification pass

Full audit of all 14,345 lines. Baseline before work: 200 tests passing.

Guiding decision: Tera is a thin substrate around Codex. Machinery that encodes judgement in Rust gets deleted or moved to `data/`, because a better model should improve the system without a code change.

### Task list

Tier 1, mechanical:
- [ ] T1.1  `history/db.rs` row-mapper + attachment-loader triplicated across three queries
- [ ] T1.2  `templates.rs` 10 wrapper fns + `init.rs` 12 call sites collapse to one table; markdown seeds move to `data/`
- [ ] T1.3  one `write_atomic` helper replaces 6 hand-rolled temp-write-rename blocks
- [ ] T1.4  `memory/rebuild.rs` + `optimizer.rs` share one stage/turn/validate/promote helper
- [ ] T1.5  dead code sweep (InboundReaction, render_burst, scheduler dead columns, `_ttl_ms`, Config derive, cache_ttl_minutes, run_turn, MockTransport fields, JsonRpcError.data, PendingRequest.reason, ModelTier re-export)
- [ ] T1.6  boilerplate: clap `--workspace` x14, `require_str` x7, stdio `respond` x5
- [ ] T1.7  `add_column_if_missing` deduplicated (two different bodies today)
- [ ] T1.8  comment cull, ~30 lines only, plus the PLAN.md citation decision
- [ ] T1.9  misc dedup: `send_text`->`client_for`, send_media dead tuple, UpdateJournal ctor, `run_from_row`, generations iterator chains, secrets `capture` dedup

Tier 2, structural, ~0 net lines:
- [ ] T2.1  schema ownership: `runtime/state.rs` stops being landlord for scheduler, memory and codex tables
- [ ] T2.2  split `codex/process.rs` (1203) into wire / semantics / events
- [ ] T2.3  `ConversationEvent.kind` String -> enum
- [ ] T2.4  rename `runtime/phoenix.rs` -> `runtime/crash_mark.rs`

Tier 3, accepted bets:
- [ ] T3.1  delete restart orchestration from `update.rs`, let the OS supervisor bounce the process
- [ ] T3.2  collapse the memory generation pipeline; memory is a rebuildable cache derived from history

### Decisions

- TRADEOFF: T3.1 trades in-process restart for a supervisor dependency. Running bare in a terminal no longer auto-restarts after an update. `deploy/tera.service` must carry `Restart=always`.
- TRADEOFF: T3.2 gives up promotion-time validation of memory. A bad Codex turn can degrade memory until the next rebuild. Acceptable because history is the source of truth and memory is derived.
- KEEP: the 1,708 comment lines are load-bearing. They cite specific past regressions and are the only durable why in a repo maintained by agents that start each session blank. Only restatements and section banners were cut.
- KEEP: `secrets.rs`, `history/backup.rs`, `scheduler/recurrence.rs`, the `Transport` trait, and the 3,300 inline test lines. All earn their length; reasons in the audit.

### Open

- TODO: outbound quoted replies never reach the wire. `WhatsAppWebTransport::send_text` takes `_reply_to_provider_id` and discards it; `send_media` does `let _ = (...)`. `MockTransport` records it faithfully, so every test passes. Predates this pass, not caused by it.
- TODO: 50 comments cite `PLAN.md section N`. PLAN.md was never committed and is not gitignored, so it is local to one machine and unresolvable for anyone else.
