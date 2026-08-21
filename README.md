# Tera

Tera is a Rust daemon that pairs to a WhatsApp account. Every message you send becomes a turn run by [Codex](https://github.com/openai/codex) against a workspace on your own machine, and Codex has a shell, a filesystem and the network. So "what's eating my disk", "clean up Downloads" and "summarise this PR" are all the same kind of request.

Read the privilege posture section before running this. It is not a chatbot in a sandbox. It is closer to giving a phone number a shell on your laptop.

## How it works

It connects to WhatsApp Web through [WhatsApp Rust](https://github.com/oxidezap/whatsapp-rust), so no Cloud API and no Meta developer account, and spawns one persistent `codex app-server`. An inbound message is checked against an owner allowlist, buffered 2.5 seconds in case more are coming, then handed to Codex as a turn. Codex replies through the `send_message` MCP tool. The text a turn ends with is only a fallback if that tool was never called.

Everything said in either direction lands in a SQLite event store with a JSONL projection that Codex queries with `jq` and `sqlite3`. Reading history is deliberately not a tool. Memory is a few markdown files Codex regenerates itself, and a bundled scheduler lets it queue its own future work instead of touching `cron`.

## Before you run it

- **`codex` on PATH and logged in.** Tera has no API key of its own. It symlinks `<workspace>/.codex-home/auth.json` to `~/.codex/auth.json` and borrows your login. Run `codex login` once first or every turn fails to authenticate.
- **A WhatsApp account to link a device to.** Your own number is the easy path.
- Optionally `git`, `sqlite3`, `jq` and `ffmpeg`. `tera init` warns if they are missing but starts anyway.

The wire format was verified against **Codex CLI 0.147.0**. A materially different version may break parsing.

## First run

```bash
codex login                                   # once, if you haven't
cargo build --release
tera daemon --workspace ~/assistant-workspace  # calls init itself
```

With no prior session on disk the daemon prints a QR code to the terminal. Scan it from WhatsApp under Linked devices, Link a device. The pairing lives in `<workspace>/.runtime/whatsapp_session.db`, so restarts reconnect silently. Don't lose that file.

Owner filtering is closed by default. With `WHATSAPP_OWNER_JID` unset, Tera answers only the account it paired to, so messaging that same number from your phone just works. It logs the sender id the first time it ignores someone, ready to paste into the env var. Group chats are never served, whatever you set.

## Configuration

| variable | effect | default |
| --- | --- | --- |
| `TERA_OWNER` | what the assistant calls you in every prompt | `$USER`, then `$LOGNAME`, then "the owner" |
| `WHATSAPP_OWNER_JID` | which sender is served | unset means only the paired account |
| `TERA_BIN` | absolute path written into the Codex config so it can spawn `tera mcp` | the running executable |
| `CODEX_LOG` | log level for the spawned application server | `error` |
| `RUST_LOG` | Tera's own tracing filter, e.g. `info,tera=debug` | subscriber default |

## Commands

Every subcommand takes `--workspace <path>`, default `/workspace`. Point it somewhere durable, not `/tmp`.

| command | what it does |
| --- | --- |
| `daemon` | the assistant itself with WhatsApp, MCP socket, scheduler and memory maintenance |
| `init` | idempotent workspace setup, called automatically by `daemon` |
| `version [--json]` | binary version, commit SHA, build time and installed Codex version |
| `update [--component all\|tera\|codex]` | update Codex and Tera, then restart with Phoenix rollback protection |
| `mcp --socket <path>` | stdio proxy Codex spawns to reach the daemon's tools, not for humans |
| `status` | daemon state, history health, active memory generation, schedules |
| `history rebuild-jsonl \| backup \| check` | projection, snapshot, integrity check (`check` exits 1 on failure) |
| `memory rebuild \| optimize` | regenerate or tidy memory, each via its own Codex turn |
| `memory status \| rollback <generation>` | list generations, or point active memory at an earlier one |

## Updates

Ask the assistant to update itself, or run `tera update --workspace ~/assistant-workspace`. The bundled update skill calls the native updater directly without MCP. Codex updates through its own `codex update` command. Tera downloads the release binary and checksum, verifies both the target and embedded version details, keeps the current executable as a rollback copy, and replaces it atomically.

The daemon restarts only after the update command returns. A systemd service comes back through `Restart=always`. A daemon started by hand uses a small detached restarter. The update journal is committed only after Tera initializes the workspace and completes a real Codex app server handshake. If the replacement crashes before that point, the next Phoenix start restores the prior Tera and Codex executables and reports the rollback to the owner.

Official release assets are `tera-x86_64-unknown-linux-gnu` and `tera-aarch64-apple-darwin`, each with a matching `.sha256` file. Builds made outside Git carry `unknown` as the commit SHA. Set `TERA_GIT_SHA` in the build environment when packaging from an exported source tree.

## Workspace

```
<workspace>/
  AGENTS.md WORKING.md      generated, rewritten every start
  PERSONA.md                yours, written once, never touched again
  SYSTEM.md                 the agent's notebook on this machine
  MEMORIES -> .memory/generations/NNNNNNNN
  .agents/skills/           native Codex skills, seeded and versioned from data/skills
  .codex-home/              private CODEX_HOME: config.toml, auth.json symlink
  .runtime/                 socket, state.sqlite3, whatsapp_session.db
  history/                  history.sqlite3, jsonl/, assets/, backups/
  logs/                     daily log, pruned after 14 days
  projects/ tasks/          Codex's working directories
```

Generated files carry an HTML comment marker and are rewritten every start, so an improved template reaches an existing workspace. Edit one by hand and Tera backs your copy up to `<file>.md.user-backup` and installs its own, so put your instructions in `PERSONA.md`, which is written once and left alone.

Bundled skills are stored under `.agents/skills/`, the native Codex repository location. Every package in `data/skills/` is discovered at build time, with no individual Rust registration. Tera installs each package once, updates an untouched managed package when its embedded files change, and remembers user edits, existing paths, symlinks, and deletions. The nightly memory pass also compacts repeated technical work into one candidate for a new or improved skill. It only suggests. Implementation waits for approval and runs on the heavy Sol tier.

Codex reaches the daemon through five tools. They are `send_message`, `react`, `schedule`, `list_schedules` and `cancel_schedule`. Schedules name a tier rather than a model id in `src/codex/tier.rs`. The `routine` tier is luna at low effort and the default, `default` is luna at xhigh for conversation, and `heavy` is sol at high.

## Privilege posture

Every Codex thread runs with `approvalPolicy: "never"` and `sandbox: "danger-full-access"`. The generated config repeats it and enables network access. If the application server asks for permission anyway, `answer_server_request` grants it automatically, including full filesystem write at `/`. Nothing pauses for review, because there is no human on the other end of an unattended WhatsApp turn.

So any message past the owner check runs as an agent that can read, write and delete anything on the machine, run arbitrary shell commands, and reach the network, unreviewed. That is intentional, and it means the WhatsApp owner check is the entire security boundary here. Pair this to a shared or public number and anyone who can message it has a shell on your machine.

## Deployment

`deploy/tera.service` is a systemd **user** unit meant to run as the account that owns the Codex login and the pairing.

```bash
cp deploy/tera.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now tera
```

Add `sudo loginctl enable-linger "$USER"` to keep it running while logged out. It expects the binary at `~/.local/bin/tera` and installs nothing for you. Stop it with `SIGINT`, not `SIGTERM`. The shutdown path listens for that to clear the typing indicator and remove the socket. Written but never run on Linux, since development happened on macOS.

## Known rough edges

- `rusqlite` is pinned at 0.39 on purpose. 0.40 needs `libsqlite3-sys` 0.38 while `whatsapp-rust-sqlite-storage` pins `^0.37`, and only one crate may link the native sqlite3 library. It moves when `whatsapp-rust` does.
- The memory optimizer and rebuild prompts run as real Codex turns over your history and are the least tested part of this. Nobody has watched a nightly pass against a real model and confirmed the output is any good.
- Turns have no elapsed time cap, so a hung tool can hold a worker indefinitely. Process death and a closed event stream are the recovery boundaries.
- Schedules use the host's local time, not a fixed timezone. Fly somewhere with the laptop and a `07:30` brief stays at `07:30` wherever it now thinks it is.
- History backups accumulate forever, one per daemon start, with nothing pruning them.
- No fault injection suite. Recovery paths have unit tests, but nothing kills a live daemon during a write to see what happens.

## Licence

[AGPL 3.0 or later](LICENSE)
