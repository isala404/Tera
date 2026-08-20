---
name: spotify
description: Control Spotify Connect through the Web API, with a WhatsApp assisted PKCE login.
---

Run `scripts/spotify auth status` first. It prints three lines, and whichever one says `missing` decides what you do next. This skill drives Spotify apps that are already running, on a phone, a computer or a speaker. It does not stream audio and it does not turn this machine into a player. Playback control needs Spotify Premium.

If `client-id` is missing, the owner has to make a Spotify app once. Ask for it with the `request_secret` tool, under the name `SPOTIFY_CLIENT_ID`, with the reason `control your Spotify`. Then tell them in your own words, in a few short messages, to open developer.spotify.com/dashboard, create an app, put exactly `http://127.0.0.1:8989/login` in Redirect URIs, tick Web API, save, then copy the Client ID from the settings page and send it. Never ask for the client secret. This skill has no use for one, and a message carrying it would be a real leak where the client ID is not. The value goes into tera's secret store and you will not see it. That is fine, because nothing here needs you to. You will get a note when it arrives.

If `authorization` is missing, run `scripts/spotify auth start` and send the URL it prints. There is no argument to pass, the script reads the client ID itself.

After the owner approves, the browser tries to open a `127.0.0.1:8989` address and fails. That is expected, and the address bar now holds a single use code. Ask them to copy the whole address and send it back, then pass it unchanged and quoted to `scripts/spotify auth finish 'REDIRECT_URL'`. Run that once. Never repeat the code and never reuse it. It dies after about ten minutes, so if the reply comes back much later, run `auth start` again and send a fresh URL.

Then use `scripts/spotify` for `status`, `devices`, `search QUERY`, `track NAME`, `connect NAME`, `play`, `pause`, `toggle`, `next`, `previous`, and `open spotify:track:ID`. Report success only after a command actually succeeds. If Spotify says no device is active, ask the owner to open Spotify somewhere first, because a device only appears once the app has been opened on it.
