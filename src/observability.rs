//! Tracing setup.
//!
//! The console copy goes to stderr, as it always has. The second copy goes to a
//! file in the workspace, because the agent is expected to diagnose itself and
//! it cannot read the terminal the daemon was started in. Under systemd there
//! is not even a terminal. Instructions for reading it are in the root
//! `AGENTS.md` (`data/workspace/AGENTS.md`).

use chrono::{Duration, Local, NaiveDate};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Log files older than this are deleted at startup. Matches memory generation
/// retention: long enough to explain a problem noticed days later, short enough
/// that nothing has to prune it by hand.
const RETAIN_DAYS: i64 = 14;

/// Start tracing. `log_dir` adds the file copy; one-shot commands with no
/// workspace pass `None` and log to stderr only.
pub fn init_tracing(log_dir: Option<&Path>) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,tera=debug"));

    let file_layer = log_dir.and_then(|dir| match fs::create_dir_all(dir) {
        Ok(()) => {
            prune_old_logs(dir, RETAIN_DAYS);
            // No ANSI: the agent reads this with rg and jq, and escape codes
            // turn every colourised level into a needless match problem.
            Some(fmt::layer().with_ansi(false).with_writer(DailyLog::new(dir)))
        }
        Err(e) => {
            eprintln!("Could not create log directory {dir:?}: {e}; logging to stderr only");
            None
        }
    });

    let _ = tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(file_layer)
        .with(filter)
        .try_init();
}

/// Name of the log file covering a given day.
pub fn log_file_name(date: NaiveDate) -> String {
    format!("tera-{}.log", date.format("%Y-%m-%d"))
}

/// The file the daemon is writing to right now.
pub fn current_log_path(log_dir: &Path) -> PathBuf {
    log_dir.join(log_file_name(Local::now().date_naive()))
}

/// Delete logs older than `retain_days`.
///
/// Parsed out of the filename rather than read from file metadata: a copied or
/// restored workspace has whatever mtimes the copy gave it, and the day a log
/// covers is a fact about its contents.
fn prune_old_logs(dir: &Path, retain_days: i64) {
    let cutoff = Local::now().date_naive() - Duration::days(retain_days);
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(date) = name
            .strip_prefix("tera-")
            .and_then(|rest| rest.strip_suffix(".log"))
            .and_then(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d").ok())
        else {
            continue;
        };

        if date < cutoff {
            if let Err(e) = fs::remove_file(entry.path()) {
                eprintln!("Could not remove old log {:?}: {e}", entry.path());
            }
        }
    }
}

/// Appends to one file per local day, rolling over when the date changes.
///
/// Hand-rolled rather than pulling in `tracing-appender` for it: the whole
/// behaviour is "append to today's file", and rotation that follows the local
/// day is what makes `logs/tera-2026-08-17.log` mean what a reader expects.
struct DailyLog {
    dir: PathBuf,
    /// The file currently open, and the day it covers.
    open: Mutex<Option<(NaiveDate, File)>>,
}

impl DailyLog {
    fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            open: Mutex::new(None),
        }
    }
}

struct DailyLogWriter<'a> {
    dir: &'a Path,
    // Poisoning is ignored: a panic mid-log-line leaves the file usable, and
    // refusing to log after one is a worse failure than a torn line.
    open: std::sync::MutexGuard<'a, Option<(NaiveDate, File)>>,
}

impl Write for DailyLogWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let today = Local::now().date_naive();

        let stale = !matches!(&*self.open, Some((day, _)) if *day == today);
        if stale {
            let path = self.dir.join(log_file_name(today));
            let file = OpenOptions::new().create(true).append(true).open(path)?;
            *self.open = Some((today, file));
        }

        // Unwrap is sound: the branch above guarantees an open file.
        self.open.as_mut().unwrap().1.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        match self.open.as_mut() {
            Some((_, file)) => file.flush(),
            None => Ok(()),
        }
    }
}

impl<'a> MakeWriter<'a> for DailyLog {
    type Writer = DailyLogWriter<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        DailyLogWriter {
            dir: &self.dir,
            open: self.open.lock().unwrap_or_else(|e| e.into_inner()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writes_land_in_todays_file() {
        let tmp = tempfile::tempdir().unwrap();
        let log = DailyLog::new(tmp.path());

        log.make_writer().write_all(b"first line\n").unwrap();
        log.make_writer().write_all(b"second line\n").unwrap();

        let path = current_log_path(tmp.path());
        assert_eq!(
            fs::read_to_string(path).unwrap(),
            "first line\nsecond line\n"
        );
    }

    /// A day's file is opened once and appended to, not reopened per line.
    #[test]
    fn test_the_file_is_held_open_across_writes() {
        let tmp = tempfile::tempdir().unwrap();
        let log = DailyLog::new(tmp.path());

        log.make_writer().write_all(b"one\n").unwrap();
        let opened_first = log.open.lock().unwrap().as_ref().unwrap().0;
        log.make_writer().write_all(b"two\n").unwrap();
        let opened_second = log.open.lock().unwrap().as_ref().unwrap().0;

        assert_eq!(opened_first, opened_second);
    }

    #[test]
    fn test_old_logs_are_pruned_and_recent_ones_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let today = Local::now().date_naive();
        let old = today - Duration::days(RETAIN_DAYS + 1);
        let recent = today - Duration::days(RETAIN_DAYS - 1);

        for date in [old, recent, today] {
            fs::write(tmp.path().join(log_file_name(date)), "x").unwrap();
        }
        // Not ours; must survive whatever we do to the directory.
        fs::write(tmp.path().join("notes.txt"), "keep me").unwrap();

        prune_old_logs(tmp.path(), RETAIN_DAYS);

        assert!(!tmp.path().join(log_file_name(old)).exists());
        assert!(tmp.path().join(log_file_name(recent)).exists());
        assert!(tmp.path().join(log_file_name(today)).exists());
        assert!(tmp.path().join("notes.txt").exists());
    }
}
