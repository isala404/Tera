---
name: self-update
description: Update Tera and Codex when the owner asks to update, upgrade, or update yourself.
---

A direct request to update or upgrade is approval for the software change and the short daemon restart it needs. Run `scripts/update` once. It updates Codex first, installs a checked Tera release, and schedules the restart without MCP.

Send the owner the command result promptly because the daemon restarts a few seconds later. Phoenix sends the final success message after the new daemon and Codex app server pass startup. If startup fails, Phoenix restores the prior Tera and Codex executables and reports the rollback.

If the owner only asks which versions are installed, run `scripts/update version`. That is read only and does not restart anything.

Do not use package managers, edit the installed binaries yourself, or restart the service separately. The native updater owns the backups, journal, restart, and rollback as one operation.
