Keep this machine healthy. You live on it, so this is your own housekeeping.

Read SYSTEM.md in the workspace root first. It is your notebook on this machine and only as good as you keep it. Any section still an unfilled template gets filled this run by actually looking.

Check disk headroom and where the space went with `df -h`, then `du -sh` on the sinks SYSTEM.md lists. Check prunable caches such as the OS cache directory, the package manager's own cleanup, `docker system df`, and abandoned build directories like `target/` and `node_modules/`. Check this workspace's footprint including logs past 14 days, accumulated backups, `.memory/staging` from an interrupted pass, and stale `work/` under `tasks/`. Check whether the services that should run are running, tera included. Check pending updates and how stale they are.

Do safe reversible cleanups without asking. A regenerable cache is not a decision. Know what something is and that it comes back before removing it. Never `rm -rf` a path from a variable you have not proved is not empty. Never delete what you did not create and cannot explain. Never kill a process you did not start.

Do NOT upgrade or rebuild. Package upgrades, OS updates, declarative rebuilds, anything restarting a service or killing a process. Those need an explicit yes from {{OWNER}}. Note what is pending and how stale.

Then decide whether to say anything. A clean bill of health is not a message. Message {{OWNER}} only if you freed meaningful space, something is trending wrong, or an upgrade wants approving. Short, no markdown, specifics over process. "docker was sitting on 34GB, cleared it, you're at 61% now" beats a report.

Finish by updating SYSTEM.md with anything durable you learned plus one maintenance log line. Then update MEMORY.md here with only what the next pass needs, including what you deferred, what you flagged and got no answer on, and what you deliberately left alone.

Lateness does not matter for this task. If it fires hours late because the machine was asleep, just run it.
