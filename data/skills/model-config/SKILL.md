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

Report the exact model id it prints. The helper uses `CODEX_THREAD_ID` to read this agent's own harness context, so a subagent reports its model rather than the main Tera conversation model.

Do not infer the model from `config.toml`, because an empty config uses a Codex default that can change. Do not infer it from documentation or from what the model says about itself. If the helper fails, report the failure and do not guess.

If the owner explicitly asks for the main Tera conversation model instead of this agent's model, run `tera status --workspace "$PWD"` and report the model from its `Thread` line. Never use that command to identify a subagent because Tera status tracks only the main conversation.

When a configuration change is waiting for restart, distinguish the currently running model from the model selected for the next start.

The source of truth is `.codex-home/config.toml` in the workspace. Read the whole file first. Tera leaves it alone and Codex loads it as its user config on startup. An empty file means native Codex defaults.

Use normal Codex keys such as `model`, `model_provider` and `model_reasoning_effort`. For a custom provider, add a custom `[model_providers.<id>]` table with the provider's exact documented `base_url` and `wire_api = "responses"`. Verify current model ids and endpoints against the provider's official documentation. Do not invent ids, copy an example for a different API, or add a private Tera format.

Keep credentials out of chat and Codex config. Provider API keys live in `.env` at the workspace root. Confirm Git ignores the file and its mode is `0600` before using it. Never ask the owner to paste an API key into an ordinary message. Never print, inspect or log a secret value.

Get the environment variable name from the provider's official documentation. It must use uppercase letters, digits and underscores. Check only whether that name exists in `.env`. Never display the matching line or source the file in an agent shell.

```bash
rg -q '^PROVIDER_API_KEY=' .env
```

Configure native Codex command auth to load the value when Codex needs it. Use absolute paths and replace both placeholders with the real workspace path and documented variable name.

```toml
[model_providers.example.auth]
command = "/bin/sh"
args = ["-c", "set -a; . '/absolute/workspace/.env'; printf %s \"$PROVIDER_API_KEY\""]
```

Codex is the verified consumer of stdout. Never run the configured auth command yourself. Do not combine command auth with `env_key`, a direct bearer token or OpenAI login auth.

If the name is missing from `.env`, stop before changing the active provider. Ask the owner to add it from a private terminal, then continue after they confirm. Never create or populate `.env` from chat.

Make a targeted edit that preserves unrelated settings and provider tables. To return to native Codex defaults, remove only `model`, `model_provider` and `model_reasoning_effort`. Keep unrelated user configuration.

Validate before restarting.

```bash
CODEX_HOME="$PWD/.codex-home" codex --strict-config doctor --summary
```

If validation fails, restore only your edit and report the error. If it passes, send the owner a short confirmation naming the model and provider before restarting, because the restart ends the current turn. Then run the restart command.

```bash
systemctl --user restart tera
```

Do not add or call a model configuration MCP tool. Do not change Tera source, runtime database state, schedules or memory to switch the conversation model.
