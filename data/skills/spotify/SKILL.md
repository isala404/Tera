---
name: spotify
description: Control the active Spotify client for playback status, play, pause, toggle, next, previous, and Spotify URI requests. Use when the user asks to control Spotify already open on a local device.
---

Use this skill for the active Spotify client. It does not search Spotify or choose a remote device.

## Control playback

Run the bundled controller from the workspace root:

```bash
.agents/skills/spotify/scripts/spotify-control status
.agents/skills/spotify/scripts/spotify-control play
.agents/skills/spotify/scripts/spotify-control pause
.agents/skills/spotify/scripts/spotify-control toggle
.agents/skills/spotify/scripts/spotify-control next
.agents/skills/spotify/scripts/spotify-control previous
.agents/skills/spotify/scripts/spotify-control open "spotify:track:TRACK_ID"
```

Use `open` only with a Spotify URI supplied by the user or already known from a prior result. Preserve the URI as one argument. Report the controller output after a successful command.

If the script reports that Spotify or its controller is unavailable, give the short error and ask only for the missing action. Do not claim playback when the command failed.

For a named song without a Spotify URI, ask for a URI or explain that this starter controls the active client and does not search. Do not guess a track.
