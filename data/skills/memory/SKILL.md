---
name: memory
description: Read, write and version the assistant's memory tree, and run its nightly compaction.
---

Memory lives in `MEMORIES/`, a git repository tera owns. Its author is already set to Tera on the repository itself, so nothing global changes and every commit is honestly yours.

Memory is your interpretation. History under `history/` is the truth, and it is read only. When the two disagree, go back to history and settle it there.

## Reading

`MEMORIES/INDEX.md` is the map and `MEMORIES/HORIZON.md` is what is coming up. Read those two, then open only the files the request actually needs.

## Shape

Run it like a personal CRM, so the next conversation starts where the last one left off.

- `USER.md` is the owner. Who they are, routines, tastes, and how they like things done.
- `PEOPLE/` has one file per person who comes up more than once, named after them. Who they are to the owner, how to reach them, what they care about, dates worth remembering, and open threads with them.
- Topic files such as `HEALTH.md`, `WORK.md` or `HOME.md` hold an ongoing area of life with its current state and decisions.
- `HORIZON.md` holds what is coming up and open loops, each with a date.

Update facts in place rather than appending, and date anything that can change. Note a preference when the owner corrects you, since that is the cheapest lesson there is.

## Format

Memory is read into a context window every conversation, so every word costs. Write plain lines, one fact per line, with no headings, bold, tables, links or nested lists, since the file name already says the topic. Say each fact as it stands now in the fewest words that carry it. A line earns its place only if it would change what you say or do in a later conversation. How something got fixed, closed loops, and anything `SYSTEM.md`, a task's `MEMORY.md` or history already holds do not belong here. Facts about this machine go in `SYSTEM.md`.

## Writing

Edit the files, then commit in one step. A message that says what changed is what makes `git log` worth reading later.

```bash
cd MEMORIES && git add -A && git commit -q -m "Note the December move"
```

Commit whenever you learn something durable. Small commits are the point. If a past edit turns out wrong, `git revert` it rather than quietly rewriting, and if you want to see how a fact drifted, `git log -p FILE.md` shows every version of it.

Names are capitals with a `.md` suffix. Keep `INDEX.md` accurate whenever you add or remove a file, and keep `HORIZON.md` short.

Something the owner said they might do is not something they did. Where the evidence does not settle a question, write that down instead of picking an answer.

## Maintenance

The nightly pass is in `references/nightly.md`. Read it when that schedule fires.

A full rebuild from history is in `references/rebuild.md`. It costs millions of tokens, so read it only when the owner asks for a rebuild.
