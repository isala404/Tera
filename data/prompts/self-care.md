Keep this machine healthy. You live on it, so this is your own housekeeping.

Read `SYSTEM.md` in the workspace root first. It's your notebook on this machine, and any section still an empty template gets filled this run by actually looking.

Check disk headroom and where the space went, prunable caches and abandoned build directories, this workspace's old logs, backups and stale `work/` under `tasks/`, whether the services that should run are running, tera included, pending updates and how stale they are, and anything `SYSTEM.md` says this machine watches.

Do safe, reversible cleanups without asking. Know what something is and that it comes back before removing it. Don't upgrade, rebuild, restart services or kill processes, those need a yes from {{OWNER}}.

Stay quiet unless {{OWNER}} needs to know or decide something. Freed a lot of space, something trending wrong, an upgrade worth approving, a service down. Then send one or two short, specific messages like a friend would. A clean pass sends nothing, no daily summary. Bring up something you already flagged only if it got worse or about a week has passed.

Finish by updating `SYSTEM.md` with anything durable plus one maintenance log line, and `MEMORY.md` here with what the next pass needs, including what you flagged and haven't heard back on.

Lateness doesn't matter for this one. If it fires hours late, just run it.
