<!-- generated: tera, edits are overwritten; put yours in PERSONA.md -->
# Daemon log

`{{WORKSPACE}}/logs/tera-YYYY-MM-DD.log`, one per local day, 14 days, read only to you. The only record of what happened around the conversation rather than in it.

```bash
tail -n 100 "{{WORKSPACE}}/logs/tera-$(date +%F).log"        # today
rg -n 'ERROR|WARN' {{WORKSPACE}}/logs/                        # everything wrong
rg -n -C5 '18:43' "{{WORKSPACE}}/logs/tera-$(date +%F).log"   # around a timestamp
```

Lines use `TIMESTAMP LEVEL target: message` with no ANSI. The target narrows it fast.

| target | covers |
| ------ | ------ |
| `codex::turn` | turns starting, completing and failing, plus model and effort |
| `codex::exec` | shell commands you ran, exit codes, output |
| `codex::mcp` | MCP servers starting, and how each call went from Codex |
| `codex::stderr` | the application server's own complaints |
| `codex` | approvals granted, model reroutes, anything unhandled |
| `mcp::tool` | your tool calls with arguments, result, timing and failures |
| `tera::scheduler` | scheduled runs firing, skipped and failing, plus seeding |
| `tera::memory` | nightly optimization, rebuilds, generation promotion |
| `tera::transport` | WhatsApp connection, pairing, send failures |

Start from the symptom's timestamp and read outwards, not from the top of the file.

`mcp::tool` logs both sides of every call, so a reply that never arrived is either a failed `send_message` or no `send_message` at all. Those are very different problems. A turn that timed out looks like slowness in `codex::turn` with nothing after it. Check `codex::stderr` for the same window before blaming the task.
