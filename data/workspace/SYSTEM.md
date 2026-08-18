# This machine

Your notebook on the host, and your standing orders for keeping it healthy. Nothing else writes here, so it is as accurate as you keep it. Read before touching the system. Write whenever you learn something durable.

Facts, not narrative. Anything needing a paragraph belongs in a task's notes with a single line pointer here.

## Identity

`uname -a`, `hostname`, `echo $SHELL`, then this platform's own version command. `sw_vers` on macOS, `/etc/os-release` on Linux.

- Host / OS / arch
- Shell
- RAM / cores

## Software

The authoritative package manager, where its config lives, and the exact commands. Wrong answers here rot a declaratively managed machine.

- Package manager
- Config / dotfiles
- Rebuild command
- Cleanup command
- Container runtime and path

## Disks

`df -h`, then `du -sh` the suspects. Record what actually grows.

- Volumes
- Space sinks
- Safe to prune, and how
- Never touch

## Services

What should be running, how to check, how to restart. Include tera.

## Quirks

Record anything not discoverable from a man page. This includes an env var that must be unset, a surprising symlink, a command slow enough to look hung, or an ordering dependency. One line each.

## Health duties

Learn by looking. `uname -a`, `df -h`, `du -sh`, `docker system df`, and `pgrep` work anywhere. The package manager, service manager and version command depend on the platform. Record which ones are right here.

Do these quietly without asking. Check disk headroom and where space went. Prune the OS cache directory, use the package manager's cleanup, run `docker system prune`, and remove abandoned `target/` and `node_modules/`. Clear your own old logs and backups. Confirm the services that should run are running. Note pending updates and how stale they are.

Report only if you freed meaningful space, something is trending wrong, or an upgrade wants approving. A clean bill of health is not a message.

Never upgrade unasked. Package upgrades, OS updates, rebuilds, or anything restarting a service need approval. Say what is pending, then wait. Never do it during other work.

Deleting is not cleanup. Know what it is and that it regenerates.

## Maintenance log

Newest first, one line per pass. Include the date, what was checked and changed, what was left alone and why, upgrades flagged, and whether {{OWNER}} approved. Collapse anything older than a few months into one summary line.
