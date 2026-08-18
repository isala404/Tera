---
name: spotify
description: Control Spotify Connect through the Web API with WhatsApp assisted PKCE login.
---

Run `scripts/spotify auth status` first. This skill controls existing Spotify clients and Connect devices through the Web API. It does not stream audio or create a player. Playback control requires Spotify Premium.

If the client ID is missing, ask the user to create a Spotify developer app, allow exactly `http://127.0.0.1:8989/login`, and send its client ID. Never request a client secret or install anything. Run `scripts/spotify auth start CLIENT_ID`, then send its authorization URL through WhatsApp. Do not open a GUI or start a callback server.

After approval, the phone browser will fail to open the loopback address. That is expected. Ask the user to copy the complete address from the browser and send it through WhatsApp. On that next message, pass the address unchanged and quoted to `scripts/spotify auth finish 'REDIRECT_URL'`. The code is a one time credential. Consume it immediately, never repeat it, and never reuse it.

Use `scripts/spotify` for `status`, `devices`, `search QUERY`, `track NAME`, `connect NAME`, `play`, `pause`, `toggle`, `next`, `previous`, and `open spotify:track:ID`. Report success only after the command succeeds. If Spotify says no device is active, ask the user to open Spotify on a phone, computer, or speaker first.
