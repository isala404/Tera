---
name: model-config
description: Report or change Tera's active Codex model or provider safely, including checks and restart.
---

Use this when the owner asks what model is running or asks to use, switch, change or reset the conversation model, provider or reasoning effort. This is configuration work, not a Tera code change.

## Report the current agent model

Ask the Codex harness instead of relying on model self identification. Run the bundled helper from the workspace root.

```bash
.agents/skills/model-config/scripts/current-model
```

Report the exact model id it prints. It uses `CODEX_THREAD_ID`, so a subagent reports its own model rather than the main conversation model.

Do not infer the model from config, documentation or self identification. An empty config uses a Codex default that can change. If the helper fails, report that and do not guess.

If the owner asks for the main conversation model instead of this agent's model, run `tera status --workspace "$PWD"` and report its `Thread` line. Never use it to identify a subagent.

When a configuration change is waiting for restart, distinguish the currently running model from the model selected for the next start.

The source of truth is `.codex-home/config.toml`. Read it fully first. Tera preserves it and an empty file means native Codex defaults.

Use native keys such as `model`, `model_provider` and `model_reasoning_effort`. For a custom provider, use its documented `[model_providers.<id>]`, `base_url` and `wire_api = "responses"`. Verify model ids and endpoints against official docs. Never add a Tera specific format.

Keep credentials out of chat and Codex config. API keys live in the workspace `.env`. Confirm Git ignores it and its mode is `0600`. Never ask the owner to paste one into chat or print its value.

Get the variable name from provider docs. Check only whether it exists. Never display its line or source `.env` in an agent shell.

```bash
rg -q '^PROVIDER_API_KEY=' .env
```

Configure native Codex command auth to load the value when Codex needs it. Use absolute paths and replace both placeholders with the real workspace path and documented variable name.

```toml
[model_providers.example.auth]
command = "/bin/sh"
args = ["-c", "set -a; . '/absolute/workspace/.env'; printf %s \"$PROVIDER_API_KEY\""]
```

Codex is the only consumer of stdout. Never run this auth command yourself or combine it with another auth method.

If it is missing, ask the owner to add it from a private terminal. Never populate `.env` from chat.

Make a targeted edit that preserves unrelated settings and provider tables. To return to native Codex defaults, remove only `model`, `model_provider` and `model_reasoning_effort`. Keep unrelated user configuration.

Validate before restarting.

```bash
CODEX_HOME="$PWD/.codex-home" codex --strict-config doctor --summary
```

The command's exit code is the validation result. An explanation for a nonzero exit does not make the configuration valid. If validation fails for a transient reason, fix that reason and retry and require exit zero. If it still fails, restore only your edit and report the error.

After validation passes, send the owner one short confirmation through `send_message`, naming the model and provider and saying the restart is scheduled. Then run the bundled helper and finish the turn immediately. The helper delays the service restart long enough for the reply and turn state to be committed, preventing Phoenix from treating the planned restart as interrupted work.

```bash
.agents/skills/model-config/scripts/restart-after-turn
```

Do not restart the service directly from the active turn. Do not add or call a model configuration MCP tool. Do not change Tera source, build or deploy a binary, update Codex, or touch runtime database state, schedules or memory to switch the conversation model.
