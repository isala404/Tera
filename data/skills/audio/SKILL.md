---
name: audio
description: Transcribe audio locally, synthesize speech, and send WhatsApp voice notes with Aloud Q8 models.
---

Use this for exact transcription, audio or video files on disk, text to speech, or a spoken WhatsApp reply. Short voice notes can still be understood directly in the turn. Reach for the local tool when the exact wording matters or when you need to create audio.

Everything runs locally with [Aloud](https://github.com/isala404/aloud). The wrapper uses only the published Q8 GGUF models from `isala404/aloud`. Those are Audio8 ASR 0.1B for transcription and Audio8 TTS 0.6B for speech. Model files resolve through Aloud's `hf://` support, are downloaded atomically into the normal Hugging Face cache on first use, and work offline after that.

Start with this.

```bash
.codex-home/skills/tera/audio/scripts/transcribe status
```

If `aloud` is missing, run setup. It installs the pinned Aloud ARM64 Linux, Apple Silicon, or x86 64 Linux release under `.runtime/aloud/` after checking the release archive's SHA 256. It does not install a system package. `ffmpeg` must already be on `PATH` because incoming media needs conversion and WhatsApp voice notes need Opus.

```bash
.codex-home/skills/tera/audio/scripts/transcribe setup
```

Transcribe any audio or video container that ffmpeg can read. Video is treated as its soundtrack. Attachment paths in the transcript resolve relative to `history/jsonl/`.

```bash
.codex-home/skills/tera/audio/scripts/transcribe path/to/note.ogg
```

Synthesize a WAV with either embedded voice.

```bash
.codex-home/skills/tera/audio/scripts/transcribe tts "Text to speak" -o .runtime/reply.wav --voice sky
```

For a real WhatsApp voice note, ask the wrapper for Opus and send the resulting file through `voice_note_path`. This sets WhatsApp's push to talk flag. `audio_path` sends an ordinary audio attachment instead.

```bash
.codex-home/skills/tera/audio/scripts/transcribe tts "Text to speak" -o .runtime/reply.ogg --voice aiden --voice-note
```

Then call `send_message` with `voice_note_path` set to `.runtime/reply.ogg`. Text is optional. Do not claim a WAV sent through `audio_path` is a voice note.

Aloud ASR expects mono 16 kHz PCM and TTS emits mono 44.1 kHz PCM. The wrapper handles those conversions. Use `--device cpu` implicitly because Aloud currently selects CPU for `auto` and the release binaries are CPU builds.
