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
use std::str::FromStr;

/// A validated `timing` argument from the `schedule` tool.
///
/// Parsed and checked in one place so the failure modes are explicit: the
/// original code stored whatever arrived, so a one-shot in the past was accepted
/// and then fired on the very next tick, "every minute for five minutes"
/// delivered five messages at once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleTiming {
    pub schedule_type: String,
    pub one_shot_at_ms: Option<i64>,
    pub rrule: Option<String>,
    /// When this schedule should first fire.
    pub first_run_ms: i64,
}

impl ScheduleTiming {
    pub fn parse(timing: &serde_json::Value, now_ms: i64) -> Result<Self> {
        let schedule_type = timing["type"].as_str().unwrap_or("once").to_string();
        let rrule = timing["rrule"].as_str().map(str::to_string);

        let one_shot_at_ms = match timing["at"].as_str() {
            Some(at) => Some(
                chrono::DateTime::parse_from_rfc3339(at)
                    .map_err(|e| {
                        anyhow!("timing.at must be RFC3339 with a UTC offset, got {at:?}: {e}")
                    })?
                    .timestamp_millis(),
            ),
            None => None,
        };

        match schedule_type.as_str() {
            "once" => {
                let at_ms = one_shot_at_ms.ok_or_else(|| {
                    anyhow!("a one-shot schedule requires timing.at as an RFC3339 timestamp")
                })?;
                if at_ms <= now_ms {
                    return Err(anyhow!(
                        "timing.at is in the past ({} <= now {}). Recompute from the current time and retry.",
                        rfc3339(at_ms),
                        rfc3339(now_ms),
                    ));
                }
            }
            "recurring" => {
                let rule = rrule.as_deref().ok_or_else(|| {
                    anyhow!("a recurring schedule requires timing.rrule (a cron expression in local time, or EVERY_<n>M / EVERY_<n>H / EVERY_<n>D)")
                })?;
                // Rejected here rather than at fire time: the agent is holding the
                // conversation and can read the error and correct itself, which
                // nothing can do five hours later inside the scheduler loop.
                Recurrence::parse(rule)?;
            }
            other => return Err(anyhow!("unknown schedule type {other:?}; use \"once\" or \"recurring\"")),
        }

        // A recurring schedule has no `at`, so its first run must be derived from
        // the rule; storing None left it permanently not-due, because the runner
        // only selects rows that have a next run.
        let first_run_ms = RecurrenceEngine::compute_next_run(
            &schedule_type,
            one_shot_at_ms,
            rrule.as_deref(),
            now_ms,
        )?
        .ok_or_else(|| anyhow!("could not derive a first run time from the given timing"))?;

        Ok(Self {
            schedule_type,
            one_shot_at_ms,
            rrule,
            first_run_ms,
        })
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
/// Two incompatibilities, both silent:
///
/// * the crate wants seconds as the first field, so `"30 7 * * *"`, the form in
///   every crontab, and the form a model will produce. Does not parse at all;
/// * its day-of-week is 1=Sunday..7=Saturday, where crontab is 0=Sunday..6=Saturday
///   (with 7 also Sunday). So `"0 9 * * 1"` for "Monday morning" fired on **Sunday**,
///   and `"0 9 * * 0"` for Sunday was rejected as invalid.
///
/// Only five-field input is translated. Six fields is the crate's own form, so its
/// author meant the crate's numbering and gets it untouched.
fn normalize_cron(rule: &str) -> String {
    let fields: Vec<&str> = rule.split_whitespace().collect();
    if fields.len() != 5 {
        return rule.to_string();
    }
    format!(
        "0 {} {} {} {} {}",
        fields[0],
        fields[1],
        fields[2],
        fields[3],
        shift_day_of_week(fields[4])
    )
}

/// Remap crontab day numbers to the crate's, leaving names, wildcards and step
/// values alone.
///
/// The digits after a `/` are a step (`*/2` is "every second day"), not a day, so
/// they must survive unchanged, remapping them would quietly change the interval.
fn shift_day_of_week(field: &str) -> String {
    let mut out = String::with_capacity(field.len() + 2);
    let mut digits = String::new();
    let mut after_slash = false;

    for ch in field.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            continue;
        }
        flush_day(&mut out, &mut digits, after_slash);
        after_slash = ch == '/';
        out.push(ch);
    }
    flush_day(&mut out, &mut digits, after_slash);
    out
}

fn flush_day(out: &mut String, digits: &mut String, is_step: bool) {
    if digits.is_empty() {
        return;
    }
    match (is_step, digits.parse::<u32>()) {
        // 0 and 7 both mean Sunday in crontab, and the crate calls it 1.
        (false, Ok(day)) if day <= 7 => out.push_str(&((day % 7) + 1).to_string()),
        _ => out.push_str(digits),
    }
    digits.clear();
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
        CronSchedule::from_str(&normalize_cron(rule))
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

pub struct RecurrenceEngine;

impl RecurrenceEngine {
    pub fn compute_next_run(
        schedule_type: &str,
        one_shot_at_ms: Option<i64>,
        rrule: Option<&str>,
        from_ms: i64,
    ) -> Result<Option<i64>> {
        match schedule_type {
            "once" => Ok(one_shot_at_ms.filter(|ts| *ts > from_ms)),
            "recurring" => match rrule {
                Some(rule) => Ok(Recurrence::parse(rule)?.next_after(from_ms)),
                None => Ok(None),
            },
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, Timelike, Weekday};
    use serde_json::json;

    #[test]
    fn test_recurrence_once() {
        let future = 2000000000000;
        let past = 1000000000000;
        let now = 1500000000000;

        assert_eq!(
            RecurrenceEngine::compute_next_run("once", Some(future), None, now).unwrap(),
            Some(future)
        );
        assert_eq!(
            RecurrenceEngine::compute_next_run("once", Some(past), None, now).unwrap(),
            None
        );
    }

    #[test]
    fn test_recurrence_interval() {
        let now = 1000000;
        let next = RecurrenceEngine::compute_next_run("recurring", None, Some("EVERY_10M"), now).unwrap();
        assert_eq!(next, Some(now + 600000));
    }

    #[test]
    fn test_day_intervals_are_supported() {
        let now = 1_700_000_000_000;
        let next = RecurrenceEngine::compute_next_run("recurring", None, Some("EVERY_3D"), now)
            .unwrap()
            .unwrap();
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

        let next = RecurrenceEngine::compute_next_run("recurring", None, Some("30 7 * * *"), now)
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
        assert_eq!(parsed.first_run_ms, future_ms);
        assert_eq!(parsed.one_shot_at_ms, Some(future_ms));
    }

    /// Regression: a recurring schedule stored next_run_at_ms = None, and the
    /// runner only selects rows with a next run, so it never fired at all.
    #[test]
    fn test_recurring_gets_a_first_run() {
        let now = 1_700_000_000_000;
        let timing = json!({"type": "recurring", "rrule": "EVERY_1M"});
        let parsed = ScheduleTiming::parse(&timing, now).unwrap();
        assert_eq!(parsed.first_run_ms, now + 60_000);
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
        for bad in ["every morning", "0 7 * *", "EVERY_10X", "EVERY_0M", "EVERY_", "EVERY_-5M"] {
            let timing = json!({"type": "recurring", "rrule": bad});
            let err = ScheduleTiming::parse(&timing, now)
                .map(|t| t.first_run_ms)
                .unwrap_err()
                .to_string();
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
            let next = RecurrenceEngine::compute_next_run("recurring", None, Some(rule), now)
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
        assert_eq!(normalize_cron("0 0 9 * * 1"), "0 0 9 * * 1");
    }

    /// The digits after a slash are a step, not a day. Remapping them would change
    /// the interval instead of the day.
    #[test]
    fn test_step_and_range_day_fields_survive_translation() {
        assert_eq!(shift_day_of_week("*/2"), "*/2");
        assert_eq!(shift_day_of_week("*"), "*");
        assert_eq!(shift_day_of_week("MON-FRI"), "MON-FRI");
        // 1-5 (Mon-Fri in crontab) becomes 2-6 in the crate's numbering.
        assert_eq!(shift_day_of_week("1-5"), "2-6");
        assert_eq!(shift_day_of_week("1,3,5"), "2,4,6");
        // Range with a step: endpoints shift, the step does not.
        assert_eq!(shift_day_of_week("1-5/2"), "2-6/2");
    }
}

