---
name: spotify
description: Control Spotify Connect playback with PKCE login and remote devices.
---

Run `scripts/spotify-prereq` first. It reports whether this machine can use the skill and the exact missing setup.

Use `scripts/spotify-auth start` to begin PKCE login. Send the generated Spotify URL to the user. The redirect goes through the local relay on port 8790. Never ask the user to paste the callback URL or OAuth code into WhatsApp.

Use `scripts/spotify-control` for `status`, `devices`, `search QUERY`, `track NAME`, `connect NAME`, `play`, `pause`, `toggle`, `next`, `previous`, and `open spotify:track:ID`. Report success only after the command succeeds.
