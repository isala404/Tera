//! Writing a file so a crash cannot leave half of it behind.

use anyhow::{Context, Result};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// Replace `path` with `contents`, so a reader ever sees the old file or the new
/// one and never a mix of both.
///
/// Three steps, all of which matter. Write a sibling temp file and `fsync` it,
/// or the rename can land while the data is still in the page cache. Rename over
/// the target, which is atomic within a filesystem. Then `fsync` the directory,
/// or the rename itself can be lost by a crash a moment later.
///
/// `mode` is explicit because getting it wrong is silent: the secret store is
/// `0o600` and a default-permission version of that file is readable by every
/// process on the machine.
pub fn write_atomic(path: &Path, contents: &[u8], mode: u32) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{path:?} has no parent directory"))?;
    fs::create_dir_all(parent)?;

    // Named after the process so two daemons racing cannot corrupt each other's
    // half-written copy, and cleaned up on failure so a crashed write does not
    // leave litter next to the real file.
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));

    let write = || -> Result<()> {
        // A same-pid leftover from a crashed write would otherwise be reopened
        // with its old permissions, since `mode` only applies on creation.
        let _ = fs::remove_file(&temporary);
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(mode)
            .open(&temporary)
            .with_context(|| format!("Cannot write {temporary:?}"))?;
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);

        fs::rename(&temporary, path).with_context(|| format!("Cannot replace {path:?}"))?;
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    };

    write().inspect_err(|_| {
        let _ = fs::remove_file(&temporary);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn test_an_existing_file_is_replaced_and_nothing_is_left_beside_it() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("state.json");
        write_atomic(&path, b"first", 0o644).unwrap();
        write_atomic(&path, b"second", 0o644).unwrap();

        assert_eq!(fs::read_to_string(&path).unwrap(), "second");
        let leftovers: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| name != "state.json")
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    /// A secret written with default permissions is readable by every process on
    /// the machine, and nothing would ever report that.
    #[test]
    fn test_the_requested_mode_reaches_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("secrets.json");
        write_atomic(&path, b"{}", 0o600).unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn test_a_missing_parent_directory_is_created() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nested").join("deeper").join("file");
        write_atomic(&path, b"x", 0o644).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "x");
    }
}
