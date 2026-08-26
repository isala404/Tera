# Nightly compaction

Nobody is waiting. Work directly in `MEMORIES/` and commit at the end.

Read `INDEX.md` and `HORIZON.md` first. Then deduplicate durable facts, settle contradictions against raw history rather than against older memory, keep uncertainty and temporal change intact, and keep open loops.

Merge tiny overlapping files, split files that are hard to retrieve from, and delete anything trivially recoverable from history. Update `INDEX.md` to match whatever the tree looks like when you finish, and keep `HORIZON.md` short.

Commit once with a message saying what actually changed. If nothing changed, commit nothing.

## Skill candidates

Review completed work from the last 14 days and look through `.agents/skills/`. You are looking for one repeated workflow worth a new skill, or concrete friction showing an existing skill should improve. Tera rewrites its own skills in `.codex-home/skills/tera/` every start, so friction in one of those is worth raising rather than editing. Prefer steps that can be scripted. Ignore work done once, work already solved, and anything that would put a credential in a skill package.

Keep at most one candidate in `HORIZON.md` with its evidence, the date, and whether the owner has been asked. If it is unchanged and nothing new supports it, say nothing. If a new candidate is strong, call `send_message` once with the workflow, what repetition it removes, and a short approval question.

Do not create or edit skills in this pass.
