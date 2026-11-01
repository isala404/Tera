---
name: memory
description: Read, write and version the assistant's memory tree, and run its nightly compaction.
---

Memory lives in `MEMORIES/`, a git repository whose author is already set to Tera. It is your interpretation. `history/` is the read only truth, so settle disagreements there.

## Reading

`INDEX.md` is the map and `HORIZON.md` is what's coming up. Read those, then only what the request needs.

## Shape

A personal CRM, so the next conversation starts where the last one left off.

- `USER.md` is the owner, their routines, tastes and how they like things done.
- `PEOPLE/` has one file per recurring person.
- Topic files like `HEALTH.md` or `WORK.md` hold the current state of an area of life.
- `HORIZON.md` holds dated upcoming things and open loops.

Names are capitals with `.md`, and `INDEX.md` stays accurate.

## Format

Memory is loaded every conversation, so every word costs. Plain lines, one current fact per line, dated when it can change. Keep a line only if it would change what you say or do later. Update in place rather than appending, and leave out stories, closed loops and what `SYSTEM.md`, task memory or history already holds. Something the owner might do isn't something they did, and an unsettled question stays marked as one.

## Writing

Edit, then commit with a message saying what changed.

```bash
cd MEMORIES && git add -A && git commit -q -m "Note the December move"
```

Commit small and often, and `git revert` mistakes rather than rewriting them.

## Maintenance

The nightly pass is `references/nightly.md`. A full rebuild from history is `references/rebuild.md`, which costs millions of tokens, so only when the owner asks.
