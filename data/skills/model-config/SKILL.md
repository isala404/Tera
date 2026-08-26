---
name: model-config
description: Report or change Tera's Codex model or provider with live validation and restart verification.
---

Use this when the owner asks what model is running or asks to use, switch, change, update, or reset the conversation model, provider, or reasoning effort. This is native Codex configuration work, not a Tera code change.

## Reporting the model

Ask the Codex harness instead of relying on self identification.

```bash
.agents/skills/model-config/scripts/current-model
```

Report the exact id. It uses `CODEX_THREAD_ID`, so a subagent reports its own model. If the owner explicitly asks for the main conversation model, use `tera status --workspace "$PWD"` and its `Thread` line. Never infer a running model from config, documentation, memory, or the model's claim about itself. When a change is waiting for restart, distinguish the live model from the selection for the next start.

## Research before every change

Search the web every time. Model names, aliases, provider endpoints, and compatibility change too often to reuse remembered values. Prefer the provider's current official model and Codex integration documentation. Confirm all of these from current sources.

- the latest stable model when the owner asked for a family or provider rather than an exact id
- the exact model id
- the OpenAI Responses compatible base URL, ending immediately before `/responses`
- the documented API key environment variable
- the publication or update date when the provider exposes one.

For native OpenAI models, also ask Codex with `model/list`. Its live catalog is stronger evidence than a stale page. If current official evidence is unavailable, do not guess and do not change the provider.

The earlier GLM failure happened because a plausible looking URL passed static config checks but returned 404 from `/responses`. Documentation research and a real inference request are both required now.

## Editing native Codex config

Read `.codex-home/config.toml` fully before editing. Preserve unrelated settings and provider tables. An empty file means native Codex defaults. Use native keys such as `model`, `model_provider`, and `model_reasoning_effort`. Custom providers use `[model_providers.<id>]`, `base_url`, and the Responses wire API. Codex currently supports only `wire_api = "responses"` for custom providers.

For a custom provider, read `references/provider-auth.md` before touching credentials. Keep credentials out of ordinary chat and Codex config.

Before editing, copy the current config to `.runtime/model-config.backup.toml`. Make a targeted edit. To restore native defaults, remove only `model`, `model_provider`, and `model_reasoning_effort`.

## Proving the change before restart

First validate syntax and local health.

```bash
CODEX_HOME="$PWD/.codex-home" codex --strict-config doctor --summary
```

Then exercise the selected provider and model with a real isolated inference.

```bash
.agents/skills/model-config/scripts/preflight
```

Both commands must exit zero. The preflight is what proves the model id, auth path, base URL, `/responses` route, and response streaming work together. If either fails, restore `.runtime/model-config.backup.toml`, validate the restored config, remove any restart context, and explain the failure naturally. Do not restart.

After both pass, write `.runtime/restart-context.md` as a few factual Markdown lines covering what the owner requested, the selected model and provider, the official documentation URLs checked, and that live inference passed. This is context for the startup agent, not a message template.

Send one short natural confirmation through `send_message`, naming the selection and saying the verified restart is scheduled. Then run the helper and end the turn immediately.

```bash
.agents/skills/model-config/scripts/restart-after-turn
```

Do not restart the service directly inside the active turn. On the new process, the startup agent independently checks the running model and tells the owner Tera is back. Do not claim the switch completed before that check.
