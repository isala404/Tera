<!-- generated: tera, edits are overwritten; put yours in PERSONA.md -->
# Conversation history

Two views of the same events. SQLite is canonical; the JSONL is a projection, rebuilt with `tera history rebuild-jsonl`. Both read-only to you.

## JSONL. `history/jsonl/YYYY-MM.jsonl`

One object per line, event order, one file per month. The fast path for `jq`, `rg`, `tail`, Python.

```json
{"id":"msg_a1b2","t":"2026-08-17T10:31:04.123Z","from":"user","turn":"turn_9f","reply_to":"msg_98","text":"Find somewhere for dinner","assets":[{"type":"image","path":"../assets/2026/08/msg_a1b2/photo.jpg"}]}
{"id":"r_77","t":"2026-08-17T10:32:20.000Z","from":"user","reaction":"❤️","to":"msg_a1b2"}
```

| field | meaning |
| ----- | ------- |
| `id` | `msg_*` for messages, `r_*` for reactions |
| `t` | RFC3339 UTC, milliseconds. Sorts lexically |
| `from` | `user` or `assistant` |
| `turn` | logical turn id; one exchange shares it |
| `reply_to` | id of the message being replied to |
| `text` | message text |
| `assets` | `{type, path}`, path relative to `history/jsonl/` |
| `reaction`, `to` | the emoji, and the message it is on |

Absent fields are omitted rather than `null`, so `select(.text?)` and `select(.assets)` are safe.

```bash
cat {{WORKSPACE}}/history/jsonl/*.jsonl | tail -5 | jq -c .                                        # last 5
cat {{WORKSPACE}}/history/jsonl/*.jsonl | jq -r 'select(.text) | "\(.t) \(.from): \(.text)"'       # transcript
jq -c 'select(.from=="user")' {{WORKSPACE}}/history/jsonl/$(date -u +%Y-%m).jsonl                   # theirs, this month
jq -c 'select(.t >= "2026-08-01" and .t < "2026-08-15")' {{WORKSPACE}}/history/jsonl/*.jsonl        # date range
jq -c 'select(.turn=="turn_9f")' {{WORKSPACE}}/history/jsonl/*.jsonl                                # one exchange
jq -r '.t[0:10]' {{WORKSPACE}}/history/jsonl/*.jsonl | sort | uniq -c                               # per day
rg -i 'flight|invoice' {{WORKSPACE}}/history/jsonl/                                                 # by content
```

Asset paths resolve from `history/jsonl/`, so `cd` there before `realpath`. When the analysis is real work (counting, grouping, correlating across months) write Python instead of a longer pipeline.

## SQLite. `history/history.sqlite3`

For what the flat files cannot answer: full-text search, reply chains, attachment joins, long ranges.

- `conversation_events`. `seq`, `id`, `occurred_at_ms`, `kind` (`message`|`reaction`), `actor`, `text`, `reply_to_id`, `turn_id`, `reaction_target_id`, `reaction_emoji`. Append-only.
- `attachments`. `event_id`, `position`, `media_type`, `relative_path`, `mime_type`, `original_name`.
- `provider_refs`. Event id ↔ WhatsApp message id, plus `chat_jid` and `from_me`. What `react` resolves through.
- `delivery_events`. Per-event transport state.
- `conversation_fts`. FTS5 over event text, kept current by a trigger.

```bash
sqlite3 {{WORKSPACE}}/history/history.sqlite3 \
  "SELECT e.occurred_at_ms, e.actor, e.text FROM conversation_fts f
     JOIN conversation_events e ON e.id = f.event_id
    WHERE conversation_fts MATCH 'dinner AND italian'
    ORDER BY e.occurred_at_ms DESC LIMIT 20;"

sqlite3 {{WORKSPACE}}/history/history.sqlite3 \
  "WITH RECURSIVE chain(id, text, reply_to_id) AS (
     SELECT id, text, reply_to_id FROM conversation_events WHERE id = 'msg_a1b2'
     UNION ALL SELECT e.id, e.text, e.reply_to_id FROM conversation_events e JOIN chain c ON e.id = c.reply_to_id
   ) SELECT * FROM chain;"
```

Timestamps are epoch milliseconds: `datetime(occurred_at_ms/1000, 'unixepoch', 'localtime')`.

Originals live at `history/assets/YYYY/MM/<event-id>/<file>`, byte-for-byte, never rewritten.
