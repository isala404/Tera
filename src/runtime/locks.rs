use anyhow::{anyhow, Result};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

pub struct DaemonLock {
    _file: File,
}

impl DaemonLock {
    pub fn acquire(lock_path: &Path) -> Result<Self> {
        if let Some(parent) = lock_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)?;

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            let res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
            if res != 0 {
                return Err(anyhow!(
                    "Failed to acquire daemon lock on {:?}. Is another tera process running?",
                    lock_path
                ));
            }
        }

        let pid = std::process::id();
        let _ = file.set_len(0);
        let _ = writeln!(file, "{}", pid);

        Ok(Self { _file: file })
    }
}
