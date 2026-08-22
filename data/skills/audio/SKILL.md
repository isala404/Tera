---
name: audio
description: Transcribe voice notes, audio files and video soundtracks locally with parakeet.cpp.
---

Run `scripts/transcribe status` first. It prints four lines, and whichever one says `missing` decides what you do next. Everything here runs on this machine, so nothing is uploaded and there is no per minute cost.

If `parakeet-cli` or `model` is missing, run `scripts/transcribe setup`. It fetches a pinned parakeet.cpp build and the parakeet tdt 0.6b v3 weights into `.runtime/parakeet/`, verifies both against their checksums, and refuses to install anything that does not match. The model is about 0.9GB and takes a few minutes on a normal connection, so tell the owner it is downloading before you start and confirm when it lands. Setup is needed once per machine, never again. If `ffmpeg` is missing, say so and stop, because every input is decoded through it.

Then run `scripts/transcribe <path>` and it prints the transcript to stdout. Any container ffmpeg reads works, so WhatsApp opus voice notes, mp3, m4a, wav, and mp4 or mov video all go in unchanged. Video transcribes as its soundtrack. Attachment paths come from the transcript as `[Attachment audio: ...]` and resolve relative to `history/jsonl/`.

Add `--json` when you need per word timing or confidence rather than prose, for example to quote a moment in a recording or to find where a topic starts. It prints one object with `text` and a `words` array carrying `start`, `end` and `conf` in seconds.

The model handles 25 European languages and picks the language itself, with punctuation and casing. It has no Sinhala, Tamil or Arabic, so a voice note in one of those comes back as nonsense rather than as an error. If the transcript reads like gibberish, suspect the language before you suspect the audio.

Audio longer than five minutes is cut into five minute pieces and rejoined, which is faster and leaner than one pass. A word landing exactly on a cut can come out garbled, so do not quote a single word from a long transcript as though it were exact.

You still hear short voice notes directly as part of the turn. Reach for this skill when you need the exact words, when the file is long, when it is a video, or when it is a file on disk rather than something the owner just sent.
