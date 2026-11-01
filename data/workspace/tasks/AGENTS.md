<!-- generated: tera, edits are overwritten; put yours in PERSONA.md -->
# Task work

Delegated single and scheduled tasks. `{{WORKSPACE}}/AGENTS.md` still applies.

- `TASK.md`, what you're here to do.
- `MEMORY.md`, state that survives runs. Keep it to what the next run needs.
- `RUNS.jsonl`, past runs, written by the daemon.
- `work/` is disposable, `artifacts/` is worth keeping.

You're not in the conversation. Only `send_message` on the `tera` MCP server reaches {{OWNER}}, and nobody is waiting, so message only when it's worth their attention. You can schedule your own next run when the timing depends on what happens.

Never modify canonical history or global memory.
