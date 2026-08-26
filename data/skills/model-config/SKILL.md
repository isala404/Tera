---
name: model-config
description: Report or change Tera's Codex model or provider with live validation and restart verification.
---

Use this when the owner asks what model is running or asks to use, switch, change, update, or reset the conversation model, provider, or reasoning effort. This is native Codex configuration, not a Tera code change.

## Reporting the model

Ask the Codex harness rather than trusting self identification.

```bash
.codex-home/skills/tera/model-config/scripts/current-model
```

Report the exact id. The script reads `CODEX_THREAD_ID`, so a subagent reports its own model. If the owner explicitly asks about the main conversation, use `tera status --workspace "$PWD"` and read its `Thread` line. Never infer the running model from config, documentation, memory, or the model's own claim about itself. When a change is waiting on a restart, say which model is live and which one starts next.

## Research before every change

Search the web every time. Model names, aliases, endpoints and compatibility move too fast to reuse a remembered value. Confirm all of these from the provider's current official documentation.

- the latest stable model, when the owner named a family or provider rather than an exact id
- the exact model id
- the OpenAI Responses compatible base URL, ending immediately before `/responses`
- the documented API key environment variable
- the publication date, where the provider gives one

For native OpenAI models, also ask Codex with `model/list`. A live catalog beats a stale page. For anything else, read `references/providers.md` first. If current official evidence is unavailable, do not guess and do not change the provider.

The GLM failure happened because a plausible looking URL passed every static check and then returned 404 from `/responses`. Research and a real inference request are both required now.

## Editing the config

Read `.codex-home/config.toml` in full before touching it. An empty file means native Codex defaults. Preserve unrelated settings and provider tables.

Copy the current file to `.runtime/model-config.backup.toml`, then make a targeted edit. The native keys are `model`, `model_provider` and `model_reasoning_effort`. A custom provider adds a `[model_providers.<id>]` table with `base_url`, and Codex currently accepts only `wire_api = "responses"` there. Restoring defaults means removing those three keys, nothing else.

## Proving it before restarting

```bash
.codex-home/skills/tera/model-config/scripts/verify
```

That runs a real isolated inference against the selection, under `--strict-config`. It has to exit zero. Passing is what proves the model id, the auth path, the base URL, the `/responses` route and streaming all work together. On failure, restore `.runtime/model-config.backup.toml`, run `scripts/verify` again, remove any restart context, explain what broke, and do not restart.

Once it passes, write `.runtime/restart-context.md` as a few factual lines covering what the owner asked for, the selection, the documentation URLs you checked, and that live inference passed. That is context for the startup agent, not a message template.

Send one short confirmation with `send_message`, naming the selection and saying the verified restart is scheduled. Then run the helper and end the turn immediately.

```bash
.codex-home/skills/tera/model-config/scripts/restart-after-turn
```

Do not restart the service directly inside the active turn. On the new process the startup agent checks the running model itself and tells the owner Tera is back. Do not claim the switch completed before that check.
