# Rebuilding memory from history

This derives the whole tree again from source, so it is expensive and the owner has to ask for it. Nobody is waiting and you are not in the conversation. Do not call `send_message`.

Work on a branch so the current memory stays reachable.

```bash
cd MEMORIES && git switch -c rebuild
```

Delete the existing files on that branch and build from nothing. Inheriting the old organisation defeats the point.

## Source

`history/SCHEMA.md` documents both stores. The projection is `history/jsonl/*.jsonl`, one JSON object per line, for `jq`, `rg` or Python. The canonical store is `history/history.sqlite3` with `conversation_fts` for full text search. Originals sit in `history/assets/` as exact bytes. Never write anything under `history/`.

Work like an engineer with a database. Measure before you decide how to read.

1. Get global statistics first. How many events, over what date range, and how many bytes of text in total.
2. Decide from that measurement whether to partition at all. History you can read in one context, you read in one context. Partition only when the text genuinely does not fit, then by time, topic or entity, whichever the data suggests.
3. Only if you partitioned, use one subagent per partition, at most four, each citing the event ids behind its claims. Every subagent pays a large startup cost before it reads anything, so splitting work that already fits is waste.
4. Merge the candidates yourself. Where two partitions disagree, go back to raw history.
5. Write `INDEX.md` as an accurate map, and `HORIZON.md` with approaching plans and unresolved commitments.
6. Delete your intermediate extraction files.

## Rules

Every material claim traces to raw history, not to convenient inference. Record what is true now without erasing that it changed. `INDEX.md` and `HORIZON.md` must exist when you finish. A tree over a few megabytes means you copied history instead of interpreting it.

## Finishing

Commit the branch, then merge it into `main` so the old tree stays in the history of the repository.

```bash
git add -A && git commit -q -m "Rebuild memory from history"
git switch main && git merge --no-ff rebuild -m "Adopt rebuilt memory"
```

If the result looks worse than what it replaced, `git switch main` and delete the branch. Nothing is lost.
