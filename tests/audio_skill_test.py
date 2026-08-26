import hashlib
import importlib.machinery
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True

SCRIPT = Path(__file__).parents[1] / "data" / "skills" / "audio" / "scripts" / "transcribe"
LOADER = importlib.machinery.SourceFileLoader("tera_audio_skill", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
AUDIO = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(AUDIO)


class AudioSkillTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "aloud"
        self.environment = mock.patch.dict(
            os.environ, {"TERA_ALOUD_ROOT": str(self.root)}, clear=False
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.tempdir.cleanup()

    def install_fake(self):
        self.root.mkdir(parents=True, exist_ok=True)
        AUDIO.cli_path().write_text("binary")

    def test_status_reports_aloud_and_q8_models(self):
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            AUDIO.status()
        text = output.getvalue()
        self.assertIn("aloud: missing", text)
        self.assertIn("q8_0.gguf", text)
        self.assertNotIn("parakeet", text.lower())

    def test_release_archives_are_pinned_for_arm_linux(self):
        archive, digest = AUDIO.RELEASES[("Linux", "aarch64")]
        self.assertIn("aarch64-unknown-linux-gnu", archive)
        self.assertEqual(len(digest), 64)
        for _, checksum in AUDIO.RELEASES.values():
            self.assertEqual(len(checksum), 64)

    def test_a_download_that_fails_its_checksum_installs_nothing(self):
        destination = self.root / "payload"
        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = [b"tampered", b""]
        with mock.patch.object(AUDIO.urllib.request, "urlopen", return_value=response):
            with self.assertRaises(AUDIO.AudioError):
                AUDIO.download("https://example.invalid/payload", destination, "0" * 64)
        self.assertFalse(destination.exists())

    def test_a_matching_download_lands_atomically(self):
        payload = b"real bytes"
        destination = self.root / "payload"
        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = [payload, b""]
        with mock.patch.object(AUDIO.urllib.request, "urlopen", return_value=response):
            AUDIO.download(
                "https://example.invalid/payload",
                destination,
                hashlib.sha256(payload).hexdigest(),
            )
        self.assertEqual(destination.read_bytes(), payload)

    def test_transcription_decodes_then_runs_aloud_q8(self):
        self.install_fake()
        source = Path(self.tempdir.name) / "note.ogg"
        source.write_bytes(b"audio")
        with mock.patch.object(AUDIO, "decode") as decode:
            with mock.patch.object(AUDIO, "run", return_value="hello") as run:
                self.assertEqual(AUDIO.transcribe(source), "hello")
        decode.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[1], "transcribe")
        self.assertIn(AUDIO.ASR_MODEL, command)

    def test_voice_note_synthesizes_wav_then_encodes_opus(self):
        self.install_fake()
        output = Path(self.tempdir.name) / "reply.ogg"
        with mock.patch.object(AUDIO, "run", return_value="") as run:
            self.assertEqual(AUDIO.synthesize("hello", output, "aiden", True), output)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][1], "tts")
        self.assertIn(AUDIO.TTS_MODEL, commands[0])
        self.assertIn("libopus", commands[1])
        self.assertIn("voip", commands[1])

    def test_plain_tts_does_not_reencode_the_wav(self):
        self.install_fake()
        output = Path(self.tempdir.name) / "reply.wav"
        with mock.patch.object(AUDIO, "run", return_value="") as run:
            AUDIO.synthesize("hello", output, "sky", False)
        self.assertEqual(run.call_count, 1)
        self.assertIn(str(output), run.call_args.args[0])

    def test_a_bare_path_is_the_transcribe_command(self):
        with mock.patch.object(AUDIO, "transcribe", return_value="text") as transcribe:
            with mock.patch.object(AUDIO.sys, "argv", ["transcribe", "note.ogg"]):
                with mock.patch("sys.stdout", io.StringIO()):
                    AUDIO.main()
        transcribe.assert_called_once_with("note.ogg")


if __name__ == "__main__":
    unittest.main()
