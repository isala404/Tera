---
name: health
description: Track the owner's health from Garmin Connect and Bluetooth devices, with trends and advice.
---

Run `scripts/health status` first. Whichever line says `missing` or `failed` decides what you do next. Everything lands in one SQLite database under `.runtime/health/`, so history survives restarts and every answer can lean on months of data.

If `dependencies` is missing, run `scripts/health setup`. It installs the pinned Garmin and Bluetooth packages into a private virtualenv, and needs uv or Python 3.12. The brief, trends and manual log work without it.

If `garmin-credentials` is missing, ask with the `request_secret` tool for `GARMIN_EMAIL` and then `GARMIN_PASSWORD`, with the reason `read your Garmin data`. The values go into tera's secret store and you will not see them. Then run `scripts/health login`. If it says Garmin sent a code, ask the owner for the code from their email and run `scripts/health login --mfa CODE`. The code dies after about ten minutes, and a wrong one can simply be sent again. The saved login then lasts for months.

Run `scripts/health sync` before answering anything about recent data. The first run backfills 30 days and later runs catch up, and `--days 90` reaches further back. Garmin rate limits this unofficial API hard, so never loop it, and if sync reports rate limiting stop for an hour.

For how am I doing, run `scripts/health brief`. It shows the day against the owner's own 28 day normal, then recommendations with the most urgent first. `scripts/health trends` compares the last week with the four weeks before it, `scripts/health history METRIC` lists one metric a day, and `scripts/health metrics` lists the names. Add `--json` to brief or trends when you want to reason over the numbers. For anything the commands do not answer, query the database with sqlite3, after reading `references/data.md` for the tables.

When the owner tells you a reading, record it, like `scripts/health log weight_kg=72.4` or `scripts/health log bp_systolic=128 bp_diastolic=82 pulse=64`, adding `--at 2026-09-23T07:30` when it was not just now. `--garmin` also sends weight or blood pressure to Garmin Connect. Ask before using it, because it writes to their account.

Bluetooth reads standard health devices near this machine. `scripts/health ble scan` lists them and `scripts/health ble read KIND` takes a reading, where KIND is `hr`, `weight`, `bp`, `temp`, `spo2` or `spo2-live`. A Garmin watch appears as `hr` only while heart rate broadcast is turned on in its settings, and a minute of it gives heart rate and HRV. Wake a scale or cuff by stepping on it or starting a measurement right before the read. On macOS the app running tera needs Bluetooth permission the first time.

Once setup works, offer to schedule a daily morning sync and brief with the `schedule` tool. Keep replies short and plain, lead with what matters today, and give numbers against their usual rather than bare. You are not a doctor, so say so when a finding is serious, and for any `alert` tell them plainly to get medical help now.
