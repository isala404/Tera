<!-- generated: tera, edits are overwritten; put yours in PERSONA.md -->
# Task work

Delegated single use and recurring tasks. `{{WORKSPACE}}/AGENTS.md` first. Voice, autonomy and the confirmation gates all apply here unchanged.

- `TASK.md`. What you are here to do.
- `MEMORY.md`. State from previous runs. Read before, update before finishing, only what a future run needs.
- `RUNS.jsonl`. Past runs, written by the daemon. Read it to see whether the last one worked.
- `work/` disposable, `artifacts/` worth keeping.

You are not in the conversation. Reaching {{OWNER}} means calling `send_message` on the `tera` MCP server, and returned text goes to a log nobody reads. Nobody is waiting either, so the bar is high. Message only with a result they asked for, a decision only they can make, or something genuinely wrong. A run with nothing worth reporting sends nothing.

Never modify canonical history or global memory.
