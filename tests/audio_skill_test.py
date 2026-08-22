import importlib.machinery
import importlib.util
import io
import json
import os
import tempfile
import unittest
import sys
import wave
from pathlib import Path
from unittest import mock

# The script has no .py suffix, so loading it writes a __pycache__ into the skill
# directory. Anything left there ships in the binary and reads as a user edit.
sys.dont_write_bytecode = True


SCRIPT = Path(__file__).parents[1] / "data" / "skills" / "audio" / "scripts" / "transcribe"
LOADER = importlib.machinery.SourceFileLoader("tera_audio_skill", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
AUDIO = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(AUDIO)


def write_wav(path, seconds, rate=AUDIO.SAMPLE_RATE):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * int(seconds * rate))


class AudioSkillTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "parakeet"
        self.environment = mock.patch.dict(
            os.environ, {"TERA_PARAKEET_ROOT": str(self.root)}, clear=False
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.tempdir.cleanup()

    def install_fakes(self):
        self.root.mkdir(parents=True, exist_ok=True)
        AUDIO.cli_path().write_text("#!/bin/sh\n")
        AUDIO.model_path().write_text("weights")

    def test_status_reports_each_missing_piece(self):
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            AUDIO.status()
        self.assertIn("parakeet-cli: missing", output.getvalue())
        self.assertIn("model: missing", output.getvalue())
        self.assertIn(str(self.root), output.getvalue())

    def test_status_reports_an_installed_pair(self):
        self.install_fakes()
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            AUDIO.status()
        self.assertIn("parakeet-cli: installed", output.getvalue())
        self.assertIn("model: installed", output.getvalue())

    def test_every_pinned_release_carries_a_full_sha256(self):
        for flavour, digest in AUDIO.RELEASES.values():
            self.assertEqual(len(digest), 64, flavour)
            self.assertEqual(digest, digest.lower(), flavour)
        self.assertEqual(len(AUDIO.MODEL_SHA256), 64)
        self.assertIn(AUDIO.MODEL_REVISION, AUDIO.MODEL_URL)

    def test_an_unsupported_platform_names_what_is_supported(self):
        with mock.patch.object(AUDIO.platform, "system", return_value="Plan9"):
            with mock.patch.object(AUDIO.platform, "machine", return_value="risc"):
                with self.assertRaises(AUDIO.AudioError) as caught:
                    AUDIO.platform_release()
        self.assertIn("Plan9 risc", str(caught.exception))
        self.assertIn("Darwin arm64", str(caught.exception))

    def test_a_download_that_fails_its_checksum_installs_nothing(self):
        """A binary that does not match its pin must never reach the disk.

        Setup downloads an executable and then runs it with the same privileges as
        everything else here, so this is the check that matters most in the file.
        """
        destination = self.root / "payload"
        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = [b"tampered", b""]
        with mock.patch.object(AUDIO.urllib.request, "urlopen", return_value=response):
            with self.assertRaises(AUDIO.AudioError) as caught:
                AUDIO.download("https://example.invalid/payload", destination, "0" * 64)
        self.assertIn("pinned checksum", str(caught.exception))
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".*")), [])

    def test_a_matching_download_lands_at_its_destination(self):
        payload = b"real bytes"
        digest = AUDIO.hashlib.sha256(payload).hexdigest()
        destination = self.root / "payload"
        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = [payload, b""]
        with mock.patch.object(AUDIO.urllib.request, "urlopen", return_value=response):
            AUDIO.download("https://example.invalid/payload", destination, digest)
        self.assertEqual(destination.read_bytes(), payload)

    def test_short_audio_is_transcribed_in_one_piece(self):
        source = Path(self.tempdir.name) / "short.wav"
        write_wav(source, 5)
        pieces = AUDIO.split_wav(source, Path(self.tempdir.name))
        self.assertEqual(pieces, [(source, 0.0)])

    def test_long_audio_splits_into_whole_pieces_that_lose_no_frames(self):
        source = Path(self.tempdir.name) / "long.wav"
        seconds = AUDIO.PIECE_SECONDS * 2 + 30
        write_wav(source, seconds)
        scratch = Path(self.tempdir.name) / "pieces"
        scratch.mkdir()
        pieces = AUDIO.split_wav(source, scratch)

        self.assertEqual(len(pieces), 3)
        self.assertEqual(
            [offset for _, offset in pieces],
            [0.0, float(AUDIO.PIECE_SECONDS), float(AUDIO.PIECE_SECONDS * 2)],
        )
        total = 0
        for path, _ in pieces:
            with wave.open(str(path), "rb") as handle:
                self.assertEqual(handle.getframerate(), AUDIO.SAMPLE_RATE)
                self.assertEqual(handle.getnchannels(), 1)
                total += handle.getnframes()
        self.assertEqual(total, seconds * AUDIO.SAMPLE_RATE)

    def test_transcribing_before_setup_points_at_setup(self):
        source = Path(self.tempdir.name) / "short.wav"
        write_wav(source, 2)
        with self.assertRaises(AUDIO.AudioError) as caught:
            AUDIO.transcribe(source, False)
        self.assertIn("setup", str(caught.exception))

    def test_a_missing_file_is_named(self):
        self.install_fakes()
        with self.assertRaises(AUDIO.AudioError) as caught:
            AUDIO.transcribe(Path(self.tempdir.name) / "nope.wav", False)
        self.assertIn("nope.wav", str(caught.exception))

    def test_pieces_are_joined_into_one_transcript(self):
        self.install_fakes()
        source = Path(self.tempdir.name) / "long.wav"
        write_wav(source, AUDIO.PIECE_SECONDS * 2)
        with mock.patch.object(AUDIO, "decode_to_wav", side_effect=lambda s, d: write_wav(d, AUDIO.PIECE_SECONDS * 2)):
            with mock.patch.object(AUDIO, "run_cli", side_effect=["first half.", "second half."]):
                self.assertEqual(AUDIO.transcribe(source, False), "first half. second half.")

    def test_json_word_times_are_shifted_by_the_piece_offset(self):
        """Word timings have to refer to the whole recording, not to a piece.

        Without the shift, every piece restarts at zero and a quoted moment points
        at the wrong part of the file, wrongly and silently.
        """
        self.install_fakes()
        source = Path(self.tempdir.name) / "long.wav"
        write_wav(source, AUDIO.PIECE_SECONDS * 2)
        pieces = [
            json.dumps({"text": "one", "words": [{"w": "one", "start": 1.0, "end": 1.5}]}),
            json.dumps({"text": "two", "words": [{"w": "two", "start": 2.0, "end": 2.5}]}),
        ]
        with mock.patch.object(AUDIO, "decode_to_wav", side_effect=lambda s, d: write_wav(d, AUDIO.PIECE_SECONDS * 2)):
            with mock.patch.object(AUDIO, "run_cli", side_effect=pieces):
                result = json.loads(AUDIO.transcribe(source, True))

        self.assertEqual(result["text"], "one two")
        self.assertEqual(
            [(word["start"], word["end"]) for word in result["words"]],
            [(1.0, 1.5), (AUDIO.PIECE_SECONDS + 2.0, AUDIO.PIECE_SECONDS + 2.5)],
        )

    def test_a_bare_path_is_treated_as_the_run_command(self):
        with mock.patch.object(AUDIO, "transcribe", return_value="text") as transcribe:
            with mock.patch.object(AUDIO.sys, "argv", ["transcribe", "note.ogg", "--json"]):
                with mock.patch("sys.stdout", io.StringIO()):
                    AUDIO.main()
        transcribe.assert_called_once_with("note.ogg", True)

    def test_the_subcommands_are_not_read_as_paths(self):
        with mock.patch.object(AUDIO, "status") as status:
            with mock.patch.object(AUDIO.sys, "argv", ["transcribe", "status"]):
                AUDIO.main()
        status.assert_called_once()


if __name__ == "__main__":
    unittest.main()
