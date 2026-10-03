# Nightly compaction

Nobody is waiting. Work from the workspace root, edit `MEMORIES/` directly, and commit once at the end.

## Catch up on the day

Read every message since the last nightly pass.

```bash
since=$(TZ=UTC git -C MEMORIES log -1 --grep='^Nightly' --format=%cd --date=format-local:%Y-%m-%dT%H:%M:%S)
jq -r --arg s "${since:-$(date -u +%Y-%m)}" 'select(.t >= $s and .text) | "\(.t[0:16]) \(.from) \(.text)"' history/jsonl/*.jsonl
```

Anything durable that the day's turns did not save goes in now. A person and who they are to the owner, a preference shown by a correction, a plan, a date, a decision.

## Compact

Read every file in `MEMORIES/`. Rewrite each one in the format `SKILL.md` describes, and move every fact to where the shape says it lives. Merge duplicates, settle contradictions against history rather than older memory, keep uncertainty and dates, and delete closed loops, fix stories and anything another file already holds. Merge tiny overlapping files and split one that has grown hard to scan. The goal is a smaller tree that keeps every live fact. `INDEX.md` gets one line per file.

Commit once with `git add -A && git commit -q --allow-empty -m "Nightly ..."`, finishing the message with what changed. The commit is how the next pass knows where this one stopped, so make it even when nothing changed.

## Skill candidates

Compare what the owner asked for today with the last two weeks of history and with `.agents/skills/`. You are looking for one repeated workflow worth a new skill, or concrete friction showing an existing skill should improve. Tera rewrites its own skills in `.codex-home/skills/tera/` every start, so friction in one of those is worth raising rather than editing. Prefer steps that can be scripted. Ignore work done once, work already solved, and anything that would put a credential in a skill package.

Keep at most one candidate in `HORIZON.md` with its evidence, the date, and whether the owner has been asked. If it is unchanged and nothing new supports it, say nothing. If a new candidate is strong, call `send_message` once with the workflow, what repetition it removes, and a short approval question. That is the only message this pass may send.

Do not create or edit skills in this pass.
