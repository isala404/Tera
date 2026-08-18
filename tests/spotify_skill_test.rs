#[cfg(target_os = "linux")]
#[test]
fn spotify_controller_dispatches_safe_playerctl_commands() {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::process::Command;

    let temp = tempfile::tempdir().unwrap();
    let fake_playerctl = temp.path().join("playerctl");
    let log = temp.path().join("playerctl.log");
    fs::write(
        &fake_playerctl,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$FAKE_PLAYERCTL_LOG\"\n",
    )
    .unwrap();
    fs::set_permissions(&fake_playerctl, fs::Permissions::from_mode(0o755)).unwrap();

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data/skills/spotify/scripts/spotify-control");
    let path = format!(
        "{}:{}",
        temp.path().display(),
        std::env::var("PATH").unwrap()
    );

    for (action, expected) in [
        ("status", "--player=spotify\nstatus\n"),
        ("play", "--player=spotify\nplay\n"),
        ("pause", "--player=spotify\npause\n"),
        ("toggle", "--player=spotify\nplay-pause\n"),
        ("next", "--player=spotify\nnext\n"),
        ("previous", "--player=spotify\nprevious\n"),
        (
            "open",
            "--player=spotify\nopen\nspotify:track:4uLU6hMCjMI75M1A2tKUQC\n",
        ),
    ] {
        let mut command = Command::new(&script);
        command
            .arg(action)
            .env("PATH", &path)
            .env("FAKE_PLAYERCTL_LOG", &log);
        if action == "open" {
            command.arg("spotify:track:4uLU6hMCjMI75M1A2tKUQC");
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{action} failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(fs::read_to_string(&log).unwrap(), expected);
    }

    for args in [
        &["unknown"][..],
        &["open"][..],
        &["open", "https://example.com"][..],
    ] {
        let result = Command::new(&script)
            .args(args)
            .env("PATH", &path)
            .env("FAKE_PLAYERCTL_LOG", &log)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(2), "args {args:?} should be rejected");
    }
}
