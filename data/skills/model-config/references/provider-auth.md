# Provider authentication

API keys live in `.env`, which must be ignored by Git and mode `0600`. Check only whether the documented variable exists; never print or source it in an agent shell.

```bash
rg -q '^PROVIDER_API_KEY=' .env
```

Use native Codex command-backed auth with absolute paths and the documented variable:

```toml
[model_providers.example.auth]
command = "/bin/sh"
args = ["-c", "set -a; . '/absolute/workspace/.env'; printf %s \"$PROVIDER_API_KEY\""]
```

Codex is the only consumer of stdout. Never run this auth command yourself or combine it with another auth method. If the value is missing, use `request_secret`; never ask for it in ordinary chat.
