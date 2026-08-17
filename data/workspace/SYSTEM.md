# This machine

Your notebook on the host, and your standing orders for keeping it healthy. Nothing else writes here, so it is as accurate as you keep it. Read before touching the system; write whenever you learn something durable.

Facts, not narrative. Anything needing a paragraph belongs in a task's notes with a one-line pointer here.

## Identity

`uname -a`, `hostname`, `echo $SHELL`, then this platform's own version command. `sw_vers` on macOS, `/etc/os-release` on Linux.

- Host / OS / arch:
- Shell:
- RAM / cores:

## Software

The authoritative package manager, where its config lives, and the exact commands. Wrong answers here rot a declaratively managed machine.

- Package manager:
- Config / dotfiles:
- Rebuild command:
- Cleanup command:
- Container runtime and path:

## Disks

`df -h`, then `du -sh` the suspects. Record what actually grows.

- Volumes:
- Space sinks:
- Safe to prune, and how:
- Never touch:

## Services

What should be running, how to check, how to restart. Include tera.

## Quirks

Not discoverable from a man page: an env var that must be unset, a surprising symlink, a command slow enough to look hung, an ordering dependency. One line each.

## Health duties

Learn by looking. `uname -a`, `df -h`, `du -sh`, `docker system df`, `pgrep` work anywhere; the package manager, service manager and version command are platform-specific. Record which ones are right here.

Do quietly, without asking: check disk headroom and where space went; prune the OS cache directory, the package manager's cleanup, `docker system prune`, abandoned `target/` and `node_modules/`; clear your own old logs and backups; confirm the services that should run are running; note pending updates and how stale they are.

Report only if you freed meaningful space, something is trending wrong, or an upgrade wants approving. A clean bill of health is not a message.

Never upgrade unasked. Package upgrades, OS updates, rebuilds, anything restarting a service. Say what is pending, then wait. Never mid-work.

Deleting is not cleanup. Know what it is and that it regenerates.

## Maintenance log

Newest first, one line per pass: date, checked, changed, left alone and why, upgrades flagged and whether {{OWNER}} approved. Collapse anything older than a few months into one summary line.
