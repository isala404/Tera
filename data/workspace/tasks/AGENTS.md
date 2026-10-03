<!-- generated: tera, edits are overwritten; put yours in PERSONA.md -->
# Task work

Delegated single and scheduled tasks. `{{WORKSPACE}}/AGENTS.md` first, everything in it still applies.

- `TASK.md`. What you're here to do.
- `MEMORY.md`. State from earlier runs. Read it first, update it before finishing, keep only what a future run needs.
- `RUNS.jsonl`. Past runs, written by the daemon.
- `work/` is disposable, `artifacts/` is worth keeping.

You're not in the conversation. Reaching {{OWNER}} means calling `send_message` on the `tera` MCP server, and returned text goes to a log nobody reads. Nobody is waiting, so only message with something they wanted, a decision only they can make, or something genuinely wrong. A run with nothing new to say sends nothing, and no two messages should read alike.

If the task is timed around something that happens, you can schedule the next run yourself from when it actually happened.

Never modify canonical history or global memory.
