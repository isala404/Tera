---
name: health
description: Track the user's health from Garmin Connect and Bluetooth devices, keep the history in a local database, and give trends and advice.
---

Run `scripts/health status` first. Any line that says `missing` or `failed` tells you what to do next. Everything lands in one SQLite database under `$HEALTH_HOME`, which defaults to `~/.local/share/health`. It lives outside this folder, so updating the skill never loses history, and every answer can lean on months of data.

If `dependencies` is missing, run `scripts/health setup`. It installs the pinned Garmin and Bluetooth packages into a private virtualenv under `$HEALTH_HOME`, and needs uv or Python 3.12. The brief, trends and manual log work without it.

If `garmin-login` is missing, log in once. The command takes the email from `--email` or `GARMIN_EMAIL` and the password from `GARMIN_PASSWORD` or the first line of stdin, and never stores the password. If the user keeps it in a secret manager, pipe it straight in, like `secretctl reveal secret://... | scripts/health login --email you@example.com`, so it never enters your context. If they have no secret manager, ask them to run `scripts/health login` in their own terminal, where it prompts with the password hidden. Never ask them to paste a password into the chat. If it says Garmin sent a code, ask for the code from their email and run `scripts/health login --mfa CODE`. The code dies after about ten minutes, and a wrong one can simply be sent again. The saved login then lasts for months, and `scripts/health logout` removes it.

Run `scripts/health sync` before answering anything about recent data. The first run backfills 30 days and later runs catch up, and `--days 90` reaches further back. Garmin rate limits this unofficial API hard, so never loop it, and if sync reports rate limiting stop for an hour.

For how am I doing, run `scripts/health brief`. It shows the day against the user's own 28 day normal, then recommendations with the most urgent first. `scripts/health trends` compares the last week with the four weeks before it, `scripts/health history METRIC` lists one metric a day, and `scripts/health metrics` lists the names. Add `--json` to brief or trends when you want to reason over the numbers. For anything the commands do not answer, query the database with sqlite3, after reading `references/data.md` for the tables.

When the user tells you a reading, record it, like `scripts/health log weight_kg=72.4` or `scripts/health log bp_systolic=128 bp_diastolic=82 pulse=64`, adding `--at 2026-09-23T07:30` when it was not just now. `--garmin` also sends weight or blood pressure to Garmin Connect. Ask before using it, because it writes to their account.

Bluetooth reads standard health devices near this machine. `scripts/health ble scan` lists them and `scripts/health ble read KIND` takes a reading, where KIND is `hr`, `weight`, `bp`, `temp`, `spo2` or `spo2-live`. A Garmin watch appears as `hr` only while heart rate broadcast is turned on in its settings, and a minute of it gives heart rate and HRV. Wake a scale or cuff by stepping on it or starting a measurement right before the read. On macOS the app running the command needs Bluetooth permission the first time.

If your agent can schedule work, offer a daily morning sync and brief once setup works. Keep replies short and plain, lead with what matters today, and give numbers against their usual rather than bare. You are not a doctor, so say so when a finding is serious, and for any `alert` tell them plainly to get medical help now.
