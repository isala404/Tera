# Health database

The database is `.runtime/health/health.sqlite3` in the workspace. Open it read only with `sqlite3 -readonly`, and change data only through `scripts/health log`, so every value keeps the same shape.

Every number is a row in `observations`, whatever produced it. The `source` column says where it came from, one of `garmin`, `ble` or `manual`. For Garmin daily summaries `at` is the day itself, and for point readings it is a local timestamp, so a day can hold several readings of one metric. Average per day before comparing days, the way the script does. Run `scripts/health metrics` for every metric name with its unit.

```sql
CREATE TABLE observations (
    metric TEXT NOT NULL,   -- steps, resting_hr, sleep_hours, hrv_night, weight_kg, bp_systolic, ...
    source TEXT NOT NULL,   -- garmin, ble, manual
    at TEXT NOT NULL,       -- 2026-09-23 or 2026-09-23T07:30:00, local time
    day TEXT NOT NULL,      -- first ten characters of at
    value REAL NOT NULL,
    device TEXT,            -- Bluetooth device name for ble readings
    PRIMARY KEY (metric, source, at)
);
CREATE TABLE activities (
    id TEXT PRIMARY KEY,    -- Garmin activity id
    day TEXT NOT NULL,
    started_at TEXT NOT NULL,
    type TEXT,              -- running, cycling, strength_training, ...
    name TEXT,
    duration_min REAL,
    distance_km REAL,
    avg_hr REAL,
    max_hr REAL,
    calories REAL,
    aerobic_te REAL,        -- Garmin training effect, 0 to 5
    anaerobic_te REAL,
    training_load REAL
);
CREATE TABLE raw (          -- every Garmin response as fetched, for anything not extracted
    source TEXT NOT NULL,
    kind TEXT NOT NULL,     -- summary, sleep, hrv, readiness, max_metrics, body_composition, blood_pressure
    day TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    body TEXT NOT NULL,     -- JSON
    PRIMARY KEY (source, kind, day)
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);  -- garmin_last_day, garmin_synced_at
```

The `raw` table holds more than the extracted metrics, such as sleep stage timelines and stress by the minute. Reach for `json_extract` on it when a question needs one of those.

```sql
-- Resting heart rate by week
SELECT strftime('%Y-%W', day) AS week, ROUND(AVG(value), 1)
FROM observations WHERE metric = 'resting_hr' GROUP BY week ORDER BY week;

-- Does a late run cost HRV the next night
SELECT a.day, a.started_at, a.training_load, h.value AS next_hrv
FROM activities a JOIN observations h
  ON h.metric = 'hrv_night' AND h.day = date(a.day, '+1 day')
ORDER BY a.day DESC LIMIT 20;

-- Garmin's sleep quality label for last night
SELECT json_extract(body, '$.dailySleepDTO.sleepScores.overall.qualifierKey') FROM raw
WHERE kind = 'sleep' ORDER BY day DESC LIMIT 1;
```
