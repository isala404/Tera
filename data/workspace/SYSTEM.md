# This machine

Your notebook on the host. Nothing else writes here. Read it before touching the system and record durable facts, not narrative.

## Identity

- Host / OS / arch
- Shell
- RAM / cores

## Software

The authoritative package manager and its exact commands.

- Package manager
- Config / dotfiles
- Rebuild command
- Cleanup command
- Container runtime and path

## Disks

Record what actually grows.

- Volumes
- Space sinks
- Safe to prune, and how
- Never touch

## Services

What should run, how to check it and how to restart it, tera included.

## Quirks

Anything a man page won't tell you, one line each.

## Health duties

Keep disk headroom, prune caches and dead build directories, clear your own old logs, confirm services run, and note pending updates. Know what something is before deleting it. Upgrades, rebuilds and restarts wait for {{OWNER}}'s yes. Message them only when there's something to know or decide.

## Maintenance log

Newest first, only for passes that changed or found something, with the date, what and why. Collapse old lines into summaries.
