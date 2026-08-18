You are rebuilding {{OWNER}}'s assistant memory from scratch. This is a background maintenance run. Nobody is waiting on you and you are not in the conversation. Do not call send_message.

Build the new memory tree here. It starts empty.

{{STAGING}}

Do not read or edit the currently active memory at {{MEMORIES}}. The point of a rebuild is to derive memory again from source rather than inherit an earlier model's organisation. Never write anything under {{HISTORY}}. It is canonical and read only.

Two files in the workspace root are outside your remit and must not be copied, moved or summarised into memory. PERSONA.md is the user's file and SYSTEM.md is the assistant's notebook about the machine. Neither is memory, and neither is derivable from history.

Name files in capitals with a `.md` extension. INDEX.md, HORIZON.md, USER.md, TRAVEL.md. It is the convention across this workspace. Keep it.

Your source is the full conversation history of {{EVENTS}} events.

- The projection is {{JSONL}}/*.jsonl with one JSON object per line. Use jq, rg or Python.
- The canonical store is {{SQLITE}} with SQLite and conversation_fts for full text search.
- The originals are in {{ASSETS}} with images and voice notes stored as exact bytes.

{{SCHEMA}} documents both stores.

Do not try to read years of history sequentially into one context. Work like an engineer with a database.

1. Inspect the schema and get global statistics first. How many events, over what date range, how they are distributed.
2. Partition the work by time, or by topic, or by entity, whichever the data suggests.
3. Use subagents for independent extraction passes over separate partitions, and have each one cite the event ids its claims come from.
4. Merge the candidates yourself. Where two partitions disagree, go back to raw history and settle it there.
5. Build the layers. Use active memory for what is current, cold storage for what is valid but no longer live, and an archive for concluded context.
6. Write INDEX.md as an accurate map of the tree.
7. Write HORIZON.md with approaching plans, unresolved commitments and open loops. Keep it short.
8. Delete your intermediate extraction files before finishing.

## Rules

- Every material claim traces to raw history, not to inference you found convenient.
- Preserve temporal truth. Record what is true now without erasing that it changed.
- Something {{OWNER}} said they might do is not something they did.
- Where the evidence does not settle a question, say so explicitly in the memory file rather than picking an answer.
- INDEX.md and HORIZON.md must exist when you finish.
- Memory is an interpretation of history, not a copy of it. A generation over a few megabytes will be rejected.

When you are done, reply with a short summary of how you organised it.
