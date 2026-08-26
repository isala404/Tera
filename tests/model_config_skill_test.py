import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "data/skills/model-config/scripts/verify"


class VerifyScriptTest(unittest.TestCase):
    def run_verify(self, term: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            bin_dir = root / "bin"
            workspace.mkdir()
            bin_dir.mkdir()

            codex = bin_dir / "codex"
            codex.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$TERM\" \"$CODEX_HOME\" \"$*\"\n"
            )
            codex.chmod(0o755)

            environment = os.environ.copy()
            environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
            environment["TERM"] = term
            result = subprocess.run(
                [VERIFY, workspace],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            return result.stdout.splitlines()

    def test_replaces_dumb_terminal_for_headless_codex(self):
        term, codex_home, _ = self.run_verify("dumb")[:3]

        self.assertEqual(term, "xterm-256color")
        self.assertTrue(codex_home.endswith("/workspace/.codex-home"))

    def test_preserves_a_usable_terminal(self):
        term = self.run_verify("screen-256color")[0]

        self.assertEqual(term, "screen-256color")

    def test_a_real_inference_is_the_only_check(self):
        """A config that only parses is what shipped the GLM 404.

        `codex doctor` also grades the terminal and the desktop app, so it fails
        headless for reasons a model change cannot fix.
        """
        lines = self.run_verify("dumb")

        self.assertNotIn("doctor", " ".join(lines))
        exec_line = next(line for line in lines if line.startswith("exec "))
        self.assertIn("--strict-config", exec_line)
        self.assertIn("--ephemeral", exec_line)
        self.assertIn("--sandbox read-only", exec_line)


if __name__ == "__main__":
    unittest.main()
