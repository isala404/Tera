//! Turning a `timing` argument into the next instant a schedule should fire.
//!
//! Two rule forms, both deliberate:
//!
//! * a cron expression, evaluated in the **host's local time**. The tool used to
//!   accept an IANA `timezone` and then evaluate the rule in UTC anyway, so
//!   `"0 7 * * *"` for a morning brief fired at 12:30 for anyone east of UTC. The
//!   daemon runs on the owner's own machine, and their local time *is* that
//!   machine's, so the timezone argument was a lie without a tz database behind it.
//!   It is gone; local is the contract.
//! * `EVERY_<n>M` / `EVERY_<n>H` / `EVERY_<n>D`, a fixed interval from the last
//!   run. Not anchored to a wall-clock slot, and that is the point. "check every
//!   20 minutes" means from now, not on the hour.
//!
//! An unparseable rule is an error. It used to fall back to "repeat in one hour",
//! which turned a typo'd cron expression into an hourly task nobody asked for and
//! left nothing in the log to explain it.

use anyhow::{anyhow, Result};
use chrono::{Local, TimeZone};
use cron::Schedule as CronSchedule;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// When a schedule fires: once at an instant, or on a rule.
///
/// Parsed and checked in one place so the failure modes are explicit: the
/// original code stored whatever arrived, so a one-shot in the past was accepted
/// and then fired on the very next tick, "every minute for five minutes"
/// delivered five messages at once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ScheduleTiming {
    Once { at_ms: i64 },
    Recurring { rrule: String },
}

impl ScheduleTiming {
    /// Validate a `timing` tool argument. Anything accepted here has a next run
    /// after `now_ms`.
    pub fn parse(timing: &serde_json::Value, now_ms: i64) -> Result<Self> {
        match timing["type"].as_str().unwrap_or("once") {
            "once" => {
                let at = timing["at"].as_str().ok_or_else(|| {
                    anyhow!("a one-shot schedule requires timing.at as an RFC3339 timestamp")
                })?;
                let at_ms = chrono::DateTime::parse_from_rfc3339(at)
                    .map_err(|e| {
                        anyhow!("timing.at must be RFC3339 with a UTC offset, got {at:?}: {e}")
                    })?
                    .timestamp_millis();
                if at_ms <= now_ms {
                    return Err(anyhow!(
                        "timing.at is in the past ({} <= now {}). Recompute from the current time and retry.",
                        rfc3339(at_ms),
                        rfc3339(now_ms),
                    ));
                }
                Ok(Self::Once { at_ms })
            }
            "recurring" => {
                let rule = timing["rrule"].as_str().ok_or_else(|| {
                    anyhow!("a recurring schedule requires timing.rrule (a cron expression in local time, or EVERY_<n>M / EVERY_<n>H / EVERY_<n>D)")
                })?;
                if Recurrence::parse(rule)?.next_after(now_ms).is_none() {
                    return Err(anyhow!("timing.rrule {rule:?} never fires after now"));
                }
                Ok(Self::Recurring {
                    rrule: rule.to_string(),
                })
            }
            other => Err(anyhow!(
                "unknown schedule type {other:?}; use \"once\" or \"recurring\""
            )),
        }
    }

    /// The first time this fires strictly after `from_ms`, or `None` once it
    /// never will again.
    pub fn next_run(&self, from_ms: i64) -> Result<Option<i64>> {
        match self {
            Self::Once { at_ms } => Ok(Some(*at_ms).filter(|at| *at > from_ms)),
            Self::Recurring { rrule } => Ok(Recurrence::parse(rrule)?.next_after(from_ms)),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Once { .. } => "once",
            Self::Recurring { .. } => "recurring",
        }
    }

    pub fn rrule(&self) -> Option<&str> {
        match self {
            Self::Once { .. } => None,
            Self::Recurring { rrule } => Some(rrule),
        }
    }
}

fn rfc3339(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms)
        .map(|d| d.to_rfc3339())
        .unwrap_or_else(|| ms.to_string())
}

/// Render an instant the way a human reads a clock: this machine's local time.
pub fn local_time(ms: i64) -> String {
    match Local.timestamp_millis_opt(ms).single() {
        Some(dt) => dt.format("%Y-%m-%d %H:%M %Z").to_string(),
        None => ms.to_string(),
    }
}

/// Translate the five-field crontab everyone actually writes into the dialect the
/// `cron` crate speaks.
///
/// Three incompatibilities, all silent:
///
/// * the crate wants seconds as the first field, so `"30 7 * * *"`, the form in
///   every crontab and the form a model will produce, does not parse at all;
/// * its day-of-week is 1=Sunday..7=Saturday, where crontab is 0=Sunday..6=Saturday
///   (with 7 also Sunday). So `"0 9 * * 1"` for "Monday morning" fired on **Sunday**,
///   and `"0 9 * * 0"` for Sunday was rejected as invalid;
/// * with both day fields restricted, crontab fires when *either* matches and the
///   crate only when *both* do. That one is refused rather than translated.
///
/// Only five-field input is translated. Six fields is the crate's own form, so its
/// author meant the crate's numbering and gets it untouched.
fn normalize_cron(rule: &str) -> Result<String> {
    let fields: Vec<&str> = rule.split_whitespace().collect();
    if fields.len() != 5 {
        return Ok(rule.to_string());
    }
    let unrestricted = |field: &str| field == "*" || field == "?";
    if !unrestricted(fields[2]) && !unrestricted(fields[4]) {
        return Err(anyhow!(
            "timing.rrule {rule:?} restricts both day-of-month and day-of-week, which crontab \
             treats as either-or. Create one schedule for each instead."
        ));
    }
    Ok(format!(
        "0 {} {} {} {} {}",
        fields[0],
        fields[1],
        fields[2],
        fields[3],
        shift_day_of_week(fields[4])
    ))
}

/// Remap crontab day numbers to the crate's, leaving names and wildcards alone.
///
/// Numeric ranges are expanded to a list: crontab's `5-7` (Friday to Sunday)
/// wraps past the end of the crate's week, so it has no range form there.
fn shift_day_of_week(field: &str) -> String {
    field
        .split(',')
        .map(shift_day_item)
        .collect::<Vec<_>>()
        .join(",")
}

fn shift_day_item(item: &str) -> String {
    let (base, step) = match item.split_once('/') {
        Some((base, step)) => (base, Some(step)),
        None => (item, None),
    };
    let bounds = match base.split_once('-') {
        Some((start, end)) => start.parse::<u32>().ok().zip(end.parse::<u32>().ok()),
        // `n/step` runs from n to the end of the week.
        None => base
            .parse::<u32>()
            .ok()
            .map(|day| (day, if step.is_some() { 6 } else { day })),
    };
    let step = match step.map(str::parse::<usize>) {
        None => 1,
        Some(Ok(step)) if step > 0 => step,
        Some(_) => return item.to_string(),
    };
    let Some((start, end)) = bounds.filter(|(start, end)| start <= end && *end <= 7) else {
        return item.to_string();
    };

    // 0 and 7 both mean Sunday in crontab, and the crate calls it 1.
    let mut days: Vec<u32> = (start..=end).step_by(step).map(|day| day % 7 + 1).collect();
    days.sort_unstable();
    days.dedup();
    days.iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// A recurrence rule that parsed. Either form, resolved once.
enum Recurrence {
    /// Wall-clock slots, in local time. Boxed: a parsed `CronSchedule` is ~250
    /// bytes and would otherwise set the size of every interval rule too.
    Cron(Box<CronSchedule>),
    /// A fixed gap from the previous run.
    Every(i64),
}

impl Recurrence {
    fn parse(rule: &str) -> Result<Self> {
        if let Some(spec) = rule.strip_prefix("EVERY_") {
            return Self::parse_interval(rule, spec);
        }
        CronSchedule::from_str(&normalize_cron(rule)?)
            .map(|schedule| Recurrence::Cron(Box::new(schedule)))
            .map_err(|e| {
                anyhow!(
                    "timing.rrule {rule:?} is not a cron expression ({e}) and does not start with \
                     EVERY_. Use a 5- or 6-field cron in local time (\"30 7 * * *\"), or an \
                     interval like EVERY_15M / EVERY_2H / EVERY_3D."
                )
            })
    }

    fn parse_interval(rule: &str, spec: &str) -> Result<Self> {
        let malformed = || {
            anyhow!(
                "timing.rrule {rule:?} is not a valid interval. Use EVERY_<n>M, EVERY_<n>H or \
                 EVERY_<n>D with a positive whole number, e.g. EVERY_15M."
            )
        };

        // Split on the unit character rather than by byte offset: `EVERY_10€` is
        // not a char boundary and `split_at` would panic on it.
        let unit = spec.chars().next_back().ok_or_else(malformed)?;
        let digits = &spec[..spec.len() - unit.len_utf8()];

        let n: i64 = digits.parse().map_err(|_| malformed())?;
        if n <= 0 {
            return Err(malformed());
        }

        // A bad unit used to silently become an hour. Now it says so.
        let per_unit_ms = match unit {
            'M' => 60_000,
            'H' => 3_600_000,
            'D' => 86_400_000,
            _ => return Err(malformed()),
        };

        Ok(Recurrence::Every(n * per_unit_ms))
    }

    fn next_after(&self, from_ms: i64) -> Option<i64> {
        match self {
            // Local, not UTC: the fields mean what the person who wrote them
            // meant. `cron` is generic over the timezone of the instant it is
            // given, so this is the whole fix.
            Recurrence::Cron(schedule) => {
                let from = Local.timestamp_millis_opt(from_ms).single()?;
                schedule.after(&from).next().map(|dt| dt.timestamp_millis())
            }
            Recurrence::Every(gap_ms) => Some(from_ms + gap_ms),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, Timelike, Weekday};
    use serde_json::json;

    fn recurring(rrule: &str) -> ScheduleTiming {
        ScheduleTiming::Recurring {
            rrule: rrule.to_string(),
        }
    }

    #[test]
    fn test_recurrence_once() {
        let future = 2000000000000;
        let past = 1000000000000;
        let now = 1500000000000;

        assert_eq!(
            ScheduleTiming::Once { at_ms: future }
                .next_run(now)
                .unwrap(),
            Some(future)
        );
        assert_eq!(
            ScheduleTiming::Once { at_ms: past }.next_run(now).unwrap(),
            None
        );
    }

    #[test]
    fn test_recurrence_interval() {
        let now = 1000000;
        let next = recurring("EVERY_10M").next_run(now).unwrap();
        assert_eq!(next, Some(now + 600000));
    }

    #[test]
    fn test_day_intervals_are_supported() {
        let now = 1_700_000_000_000;
        let next = recurring("EVERY_3D").next_run(now).unwrap().unwrap();
        assert_eq!(next - now, 3 * 86_400_000);
    }

    /// The bug this module exists to fix: the tool advertised a timezone and then
    /// evaluated cron in UTC, so a 07:30 morning brief arrived at whatever 07:30
    /// UTC happens to be locally.
    #[test]
    fn test_cron_fires_at_the_local_wall_clock_time() {
        let now = Local
            .with_ymd_and_hms(2026, 8, 18, 3, 0, 0)
            .single()
            .expect("unambiguous local instant")
            .timestamp_millis();

        let next = recurring("30 7 * * *")
            .next_run(now)
            .unwrap()
            .expect("a daily rule always has a next run");

        let fired = Local.timestamp_millis_opt(next).single().unwrap();
        assert_eq!((fired.hour(), fired.minute()), (7, 30));
        assert_eq!(fired.day(), 18, "should be later the same local day");
    }

    /// Regression: five one-shot schedules were created for "one per minute for
    /// five minutes" and all five fired at once, because the agent computed the
    /// times in the past and nothing rejected them.
    #[test]
    fn test_one_shot_in_the_past_is_rejected() {
        let now = 1_700_000_000_000;
        let past = json!({"type": "once", "at": rfc3339(now - 60_000)});
        let err = ScheduleTiming::parse(&past, now).unwrap_err().to_string();
        assert!(err.contains("in the past"), "unhelpful error: {err}");
    }

    #[test]
    fn test_one_shot_in_the_future_is_accepted() {
        let now = 1_700_000_000_000;
        let future_ms = now + 300_000;
        let timing = json!({"type": "once", "at": rfc3339(future_ms)});
        let parsed = ScheduleTiming::parse(&timing, now).unwrap();
        assert_eq!(parsed, ScheduleTiming::Once { at_ms: future_ms });
        assert_eq!(parsed.next_run(now).unwrap(), Some(future_ms));
    }

    /// Regression: a recurring schedule stored next_run_at_ms = None, and the
    /// runner only selects rows with a next run, so it never fired at all.
    #[test]
    fn test_recurring_gets_a_first_run() {
        let now = 1_700_000_000_000;
        let timing = json!({"type": "recurring", "rrule": "EVERY_1M"});
        let parsed = ScheduleTiming::parse(&timing, now).unwrap();
        assert_eq!(parsed.next_run(now).unwrap(), Some(now + 60_000));
    }

    #[test]
    fn test_missing_timing_details_are_rejected() {
        let now = 1_700_000_000_000;
        assert!(ScheduleTiming::parse(&json!({"type": "once"}), now).is_err());
        assert!(ScheduleTiming::parse(&json!({"type": "recurring"}), now).is_err());
        assert!(ScheduleTiming::parse(&json!({"type": "weekly"}), now).is_err());
    }

    #[test]
    fn test_non_rfc3339_timestamp_is_rejected() {
        let now = 1_700_000_000_000;
        let timing = json!({"type": "once", "at": "2026-08-17 14:18:00"});
        let err = ScheduleTiming::parse(&timing, now).unwrap_err().to_string();
        assert!(err.contains("RFC3339"), "unhelpful error: {err}");
    }

    /// A rule that does not parse used to become "repeat in one hour", so a
    /// mistyped cron expression became an hourly task with nothing in the log to
    /// say why.
    #[test]
    fn test_an_unparseable_rule_is_rejected_not_turned_into_an_hourly_task() {
        let now = 1_700_000_000_000;
        for bad in [
            "every morning",
            "0 7 * *",
            "EVERY_10X",
            "EVERY_0M",
            "EVERY_",
            "EVERY_-5M",
        ] {
            let timing = json!({"type": "recurring", "rrule": bad});
            let err = ScheduleTiming::parse(&timing, now).unwrap_err().to_string();
            assert!(
                err.contains("EVERY_") || err.contains("cron"),
                "{bad:?} gave an unhelpful error: {err}"
            );
        }
    }

    /// Six-field cron (with seconds) is what the `cron` crate natively takes, and
    /// five-field is what everyone writes. Both have to work.
    #[test]
    fn test_both_five_and_six_field_cron_parse() {
        let now = 1_700_000_000_000;
        for rule in ["30 7 * * *", "0 30 7 * * *", "0 9 * * 1", "0 9 * * 0"] {
            let timing = json!({"type": "recurring", "rrule": rule});
            assert!(
                ScheduleTiming::parse(&timing, now).is_ok(),
                "{rule:?} should parse"
            );
        }
    }

    /// The `cron` crate numbers days 1=Sunday, crontab numbers them 0=Sunday. So
    /// "0 9 * * 1", Monday morning to anyone who has written a crontab, fired on
    /// Sunday, and "0 9 * * 0" was rejected as invalid rather than meaning Sunday.
    #[test]
    fn test_five_field_cron_uses_crontab_day_numbering() {
        // A Tuesday, so every weekday in the week ahead is a distinct next-run.
        let now = Local
            .with_ymd_and_hms(2026, 8, 18, 3, 0, 0)
            .single()
            .unwrap()
            .timestamp_millis();

        let expected = [
            ("0 9 * * 0", Weekday::Sun),
            ("0 9 * * 1", Weekday::Mon),
            ("0 9 * * 3", Weekday::Wed),
            ("0 9 * * 6", Weekday::Sat),
            ("0 9 * * 7", Weekday::Sun),
            // Names were never ambiguous; they must keep working.
            ("0 9 * * MON", Weekday::Mon),
            ("0 9 * * FRI", Weekday::Fri),
        ];

        for (rule, day) in expected {
            let next = recurring(rule)
                .next_run(now)
                .unwrap()
                .unwrap_or_else(|| panic!("{rule:?} produced no next run"));
            let fired = Local.timestamp_millis_opt(next).single().unwrap();
            assert_eq!(fired.weekday(), day, "{rule:?} fired on {fired}");
            assert_eq!(fired.hour(), 9, "{rule:?} fired at the wrong hour");
        }
    }

    /// A six-field expression is the crate's own dialect, so its author meant the
    /// crate's numbering and must get it unchanged.
    #[test]
    fn test_six_field_cron_day_numbering_is_left_alone() {
        assert_eq!(normalize_cron("0 0 9 * * 1").unwrap(), "0 0 9 * * 1");
    }

    /// The digits after a slash are a step, not a day. Remapping them would change
    /// the interval instead of the day.
    #[test]
    fn test_step_and_range_day_fields_survive_translation() {
        assert_eq!(shift_day_of_week("*/2"), "*/2");
        assert_eq!(shift_day_of_week("*"), "*");
        assert_eq!(shift_day_of_week("MON-FRI"), "MON-FRI");
        // 1-5 (Mon-Fri in crontab) is 2..6 in the crate's numbering.
        assert_eq!(shift_day_of_week("1-5"), "2,3,4,5,6");
        assert_eq!(shift_day_of_week("1,3,5"), "2,4,6");
        assert_eq!(shift_day_of_week("1-5/2"), "2,4,6");
    }

    /// A range ending in 7 wraps past the end of the crate's week: `5-7` is
    /// Friday to Sunday, and `0-7` every day.
    #[test]
    fn test_day_ranges_ending_on_sunday_keep_every_day() {
        assert_eq!(shift_day_of_week("5-7"), "1,6,7");
        assert_eq!(shift_day_of_week("6-7"), "1,7");
        assert_eq!(shift_day_of_week("0-7"), "1,2,3,4,5,6,7");

        let now = Local
            .with_ymd_and_hms(2026, 8, 18, 3, 0, 0)
            .single()
            .unwrap()
            .timestamp_millis();
        let next = recurring("0 9 * * 5-7").next_run(now).unwrap().unwrap();
        let fired = Local.timestamp_millis_opt(next).single().unwrap();
        assert_eq!(fired.weekday(), Weekday::Fri, "fired on {fired}");
    }

    /// Crontab fires when either day field matches; the crate would need both.
    #[test]
    fn test_both_day_fields_restricted_is_refused() {
        let timing = json!({"type": "recurring", "rrule": "0 9 1 * 1"});
        let err = ScheduleTiming::parse(&timing, 1_700_000_000_000)
            .unwrap_err()
            .to_string();
        assert!(err.contains("either-or"), "{err}");
    }
}
