Review {{OWNER}}'s recent completed work for one concrete reusable workflow that deserves a native Codex skill.

Read MEMORY.md here first. Read {{WORKSPACE}}/AGENTS.md and {{WORKSPACE}}/history/SCHEMA.md. Inspect recent conversation history from the last 14 days with the documented SQLite or JSONL queries. Check {{WORKSPACE}}/.agents/skills/ and read the relevant SKILL.md files before deciding.

Only suggest a skill when the same kind of work involved repeated steps, trial and error, or a scriptable integration. Ignore one-off work, unfinished work, work already covered by a suitable skill, and anything that would need secrets in the skill package. Do not create or edit a skill. Skill creation happens in a conversation after {{OWNER}} agrees and uses the built-in $skill-creator.

Use MEMORY.md to avoid repeating the same suggestion. Record the candidate, evidence, date, and whether you messaged {{OWNER}}. If the candidate is unchanged and there is no new evidence, finish silently.

If there is a new strong candidate, call send_message once with a short plain-text message that names the workflow, says what repeated work it would capture, and ends with exactly: Should I create a skill for this? Otherwise send no message. Update MEMORY.md before finishing.
