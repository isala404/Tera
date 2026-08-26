# Custom providers

Read this before configuring anything other than native OpenAI models.

## Finding the right settings

Search the provider's own documentation for a Codex integration guide first. A generic OpenAI API guide is not enough, because Codex speaks the Responses protocol and some providers need a local model catalog on top of that.

Follow any published `models.json` and `model_catalog_json` instructions exactly. Use an absolute catalog path inside `.codex-home` and treat the catalog as part of the provider configuration. Back up an existing catalog before editing, and move a newly created one aside if verification fails.

## Credentials

Prefer a credential helper that streams the key straight to Codex. Inspect only secret metadata to find the reference, then point Codex at the absolute helper path.

```toml
[model_providers.example.auth]
command = "/usr/local/bin/secretctl"
args = ["reveal", "secret://BACKEND/VAULT/SECRET_ID"]
```

Codex is the only consumer of that stdout. Never run the reveal command yourself. Do not combine command authentication with `env_key`, `experimental_bearer_token`, or any other auth method.

With no credential helper, API keys live in `.env`, which must be ignored by git and mode `0600`. Check only that the documented variable exists. Never print it, never source it in an agent shell.

```bash
rg -q '^PROVIDER_API_KEY=' .env
```

```toml
[model_providers.example.auth]
command = "/bin/sh"
args = ["-c", "set -a; . '/absolute/workspace/.env'; printf %s \"$PROVIDER_API_KEY\""]
```

If the value is missing, use `request_secret`. Never ask for a key in ordinary chat.
