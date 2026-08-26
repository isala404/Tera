---
name: memory
description: Read, write and version the assistant's memory tree, and run its nightly compaction.
---

Memory lives in `MEMORIES/`, a git repository tera owns. Its author is already set to Tera on the repository itself, so nothing global changes and every commit is honestly yours.

Memory is your interpretation. History under `history/` is the truth, and it is read only. When the two disagree, go back to history and settle it there.

## Reading

`MEMORIES/INDEX.md` is the map and `MEMORIES/HORIZON.md` is what is coming up. Read those two, then open only the files the request actually needs.

## Writing

Edit the files, then commit in one step. A message that says what changed is what makes `git log` worth reading later.

```bash
cd MEMORIES && git add -A && git commit -q -m "Note the December move"
```

Commit whenever you learn something durable. Small commits are the point. If a past edit turns out wrong, `git revert` it rather than quietly rewriting, and if you want to see how a fact drifted, `git log -p FILE.md` shows every version of it.

Names are capitals with a `.md` suffix. Keep `INDEX.md` accurate whenever you add or remove a file, and keep `HORIZON.md` short.

Record durable facts and open loops. Not a diary, not anything trivially recoverable from history, and never a copy of `PERSONA.md` or `SYSTEM.md`. Something the owner said they might do is not something they did. Where the evidence does not settle a question, write that down instead of picking an answer.

## Maintenance

The nightly pass is in `references/nightly.md`. Read it when that schedule fires.

A full rebuild from history is in `references/rebuild.md`. It costs millions of tokens, so read it only when the owner asks for a rebuild.
