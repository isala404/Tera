import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse


SCRIPT = Path(__file__).parents[1] / "data" / "skills" / "spotify" / "scripts" / "spotify"
LOADER = importlib.machinery.SourceFileLoader("tera_spotify_skill", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
SPOTIFY = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(SPOTIFY)


class SpotifySkillTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.environment = mock.patch.dict(
            os.environ,
            {"SPOTIFY_CLIENT_ID": "", "TERA_SPOTIFY_ROOT": self.tempdir.name},
            clear=False,
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.tempdir.cleanup()

    def read_state(self, name):
        return json.loads((Path(self.tempdir.name) / name).read_text())

    def write_pending(self, **overrides):
        pending = {
            "client_id": "client-id",
            "created_at": int(time.time()),
            "redirect_uri": SPOTIFY.REDIRECT_URI,
            "state": "expected-state",
            "verifier": "verifier",
        }
        pending.update(overrides)
        SPOTIFY.write_json("pending.json", pending)

    def test_auth_start_generates_pkce_url_and_private_state(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            SPOTIFY.start_auth("client-id")

        parsed = urlparse(output.getvalue().strip())
        query = parse_qs(parsed.query)
        pending = self.read_state("pending.json")

        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "accounts.spotify.com")
        self.assertEqual(query["client_id"], ["client-id"])
        self.assertEqual(query["redirect_uri"], [SPOTIFY.REDIRECT_URI])
        self.assertEqual(query["state"], [pending["state"]])
        self.assertEqual(query["code_challenge_method"], ["S256"])
        self.assertEqual(
            query["code_challenge"],
            [SPOTIFY.pkce_challenge(pending["verifier"])],
        )
        self.assertEqual(set(query["scope"][0].split()), set(SPOTIFY.SCOPES))

        root_mode = stat.S_IMODE(Path(self.tempdir.name).stat().st_mode)
        pending_mode = stat.S_IMODE((Path(self.tempdir.name) / "pending.json").stat().st_mode)
        self.assertEqual(root_mode, 0o700)
        self.assertEqual(pending_mode, 0o600)
        self.assertFalse((Path(self.tempdir.name) / "config.json").exists())

    def test_auth_finish_validates_redirect_and_saves_tokens(self):
        self.write_pending()
        token = {
            "access_token": "access-token",
            "expires_in": 3600,
            "refresh_token": "refresh-token",
            "scope": " ".join(SPOTIFY.SCOPES),
            "token_type": "Bearer",
        }

        with mock.patch.object(SPOTIFY, "token_request", return_value=token) as exchange:
            with contextlib.redirect_stdout(io.StringIO()):
                SPOTIFY.finish_auth(
                    f"{SPOTIFY.REDIRECT_URI}?code=one-time-code&state=expected-state"
                )

        exchange.assert_called_once_with(
            {
                "client_id": "client-id",
                "code": "one-time-code",
                "code_verifier": "verifier",
                "grant_type": "authorization_code",
                "redirect_uri": SPOTIFY.REDIRECT_URI,
            }
        )
        saved = self.read_state("token.json")
        self.assertEqual(saved["access_token"], "access-token")
        self.assertEqual(saved["refresh_token"], "refresh-token")
        self.assertFalse((Path(self.tempdir.name) / "pending.json").exists())

    def test_auth_finish_rejects_wrong_state_before_exchange(self):
        self.write_pending()

        with mock.patch.object(SPOTIFY, "token_request") as exchange:
            with self.assertRaisesRegex(SPOTIFY.SpotifyError, "state does not match"):
                SPOTIFY.finish_auth(
                    f"{SPOTIFY.REDIRECT_URI}?code=one-time-code&state=wrong-state"
                )
        exchange.assert_not_called()

    def test_auth_finish_rejects_ambiguous_or_expired_redirects(self):
        callbacks = (
            f"{SPOTIFY.REDIRECT_URI}?code=one&code=two&state=expected-state",
            f"{SPOTIFY.REDIRECT_URI}?code=one&state=expected-state&state=expected-state",
            "https://example.com/login?code=one&state=expected-state",
        )
        with mock.patch.object(SPOTIFY, "token_request") as exchange:
            for callback in callbacks:
                with self.subTest(callback=callback):
                    self.write_pending()
                    with self.assertRaises(SPOTIFY.SpotifyError):
                        SPOTIFY.finish_auth(callback)

            self.write_pending(created_at=int(time.time()) - SPOTIFY.PENDING_LIFETIME_SECONDS - 1)
            with self.assertRaisesRegex(SPOTIFY.SpotifyError, "expired"):
                SPOTIFY.finish_auth(
                    f"{SPOTIFY.REDIRECT_URI}?code=one&state=expected-state"
                )
        exchange.assert_not_called()

    def test_authorization_error_still_requires_matching_state(self):
        self.write_pending()
        with self.assertRaisesRegex(SPOTIFY.SpotifyError, "state does not match"):
            SPOTIFY.finish_auth(
                f"{SPOTIFY.REDIRECT_URI}?error=access_denied&state=wrong-state"
            )

    def test_refresh_keeps_existing_refresh_token(self):
        token = {
            "access_token": "expired",
            "client_id": "client-id",
            "expires_at": 0,
            "refresh_token": "refresh-token",
            "scope": " ".join(SPOTIFY.SCOPES),
            "token_type": "Bearer",
        }
        SPOTIFY.write_json("token.json", token)

        with mock.patch.object(
            SPOTIFY,
            "token_request",
            return_value={"access_token": "fresh", "expires_in": 3600},
        ):
            refreshed = SPOTIFY.refresh_access_token(token)

        self.assertEqual(refreshed["access_token"], "fresh")
        self.assertEqual(refreshed["refresh_token"], "refresh-token")

    def test_api_retries_once_after_unauthorized_response(self):
        unauthorized = SPOTIFY.SpotifyError("expired", status=401)
        with mock.patch.object(
            SPOTIFY,
            "request_json",
            side_effect=[unauthorized, {"is_playing": True}],
        ), mock.patch.object(
            SPOTIFY,
            "access_token",
            side_effect=["old-token", "fresh-token"],
        ) as token:
            result = SPOTIFY.api_request("/me/player")

        self.assertEqual(result, {"is_playing": True})
        self.assertEqual(token.call_args_list, [mock.call(force_refresh=False), mock.call(force_refresh=True)])

    def test_track_uri_accepts_only_spotify_tracks(self):
        self.assertEqual(SPOTIFY.track_uri("spotify:track:abc123"), "spotify:track:abc123")
        self.assertEqual(
            SPOTIFY.track_uri("https://open.spotify.com/track/abc123?si=ignored"),
            "spotify:track:abc123",
        )
        with self.assertRaises(SPOTIFY.SpotifyError):
            SPOTIFY.track_uri("https://example.com/track/abc123")

    def test_playback_commands_use_spotify_web_api(self):
        expected = {
            "next": ("/me/player/next", "POST"),
            "pause": ("/me/player/pause", "PUT"),
            "play": ("/me/player/play", "PUT"),
            "previous": ("/me/player/previous", "POST"),
        }
        with mock.patch.object(SPOTIFY, "api_request") as request:
            with contextlib.redirect_stdout(io.StringIO()):
                for command in expected:
                    SPOTIFY.playback_command(command)

        self.assertEqual(
            request.call_args_list,
            [mock.call(path, method=method) for path, method in expected.values()],
        )

    def test_connect_prefers_an_exact_unrestricted_device(self):
        devices = [
            {"id": "one", "name": "Kitchen", "is_restricted": False},
            {"id": "two", "name": "Kitchen speaker", "is_restricted": False},
            {"id": "three", "name": "Bedroom", "is_restricted": True},
        ]
        with mock.patch.object(SPOTIFY, "get_devices", return_value=devices), mock.patch.object(
            SPOTIFY, "api_request"
        ) as request, contextlib.redirect_stdout(io.StringIO()):
            SPOTIFY.choose_device("kitchen")

        request.assert_called_once_with(
            "/me/player",
            method="PUT",
            body={"device_ids": ["one"], "play": False},
        )


if __name__ == "__main__":
    unittest.main()
