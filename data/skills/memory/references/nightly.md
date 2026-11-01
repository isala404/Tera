# Nightly compaction

Nobody is waiting. Work from the workspace root, edit `MEMORIES/` directly, and commit once at the end.

## Catch up on the day

Read every message since the last nightly pass.

```bash
since=$(TZ=UTC git -C MEMORIES log -1 --grep='^Nightly' --format=%cd --date=format-local:%Y-%m-%dT%H:%M:%S)
jq -r --arg s "${since:-$(date -u +%Y-%m)}" 'select(.t >= $s and .text) | "\(.t[0:16]) \(.from) \(.text)"' history/jsonl/*.jsonl
```

Save anything durable the day's turns missed.

## Compact

Rewrite every file in `MEMORIES/` to the format in `SKILL.md`, ending with a smaller tree that keeps every live fact. Settle contradictions against history. Give `SYSTEM.md` and any `tasks/*/MEMORY.md` that has grown into a diary the same treatment.

Commit once with `git add -A && git commit -q --allow-empty -m "Nightly ..."`, finishing the message with what changed. The commit is how the next pass knows where this one stopped, so make it even when nothing changed.

## Skill candidates

Look for one repeated workflow in the last two weeks worth a new skill, or friction showing an existing one in `.agents/skills/` or `.codex-home/skills/tera/` should improve. Ignore one offs and anything that would put a credential in a skill.

Keep at most one candidate in `HORIZON.md` with its evidence and date. When a new candidate is strong, `send_message` once with the workflow and a short approval question. That is the only message this pass may send, and it creates or edits no skills.
