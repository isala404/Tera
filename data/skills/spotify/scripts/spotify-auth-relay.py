#!/usr/bin/env python3
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


LOCAL_CALLBACK = "http://127.0.0.1:8989/login"
HOST = os.environ.get("TERA_SPOTIFY_RELAY_HOST", "0.0.0.0")
PORT = int(os.environ.get("TERA_SPOTIFY_RELAY_PORT", "8790"))


class RelayHandler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        return

    def send_page(self, status, body):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path != "/":
            self.send_page(404, "Not found")
            return
        self.send_page(
            200,
            """<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spotify login relay</title>
<h2>Spotify login relay</h2>
<p>After approving Spotify, copy the complete redirected address from the browser and paste it here.</p>
<form method="post" action="/relay">
<input name="redirect_url" type="url" required style="width:100%;padding:10px" placeholder="http://127.0.0.1:8989/login?code=...&state=...">
<button type="submit" style="margin-top:12px;padding:10px">Finish login</button>
</form>""",
        )

    def do_POST(self):
        if self.path != "/relay":
            self.send_page(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            form = parse_qs(self.rfile.read(length).decode("utf-8"), strict_parsing=True)
            redirect_url = form["redirect_url"][0]
            parsed = urlparse(redirect_url)
            query = parse_qs(parsed.query, strict_parsing=True)
            if parsed.scheme != "http" or parsed.netloc != "127.0.0.1:8989" or parsed.path != "/login":
                raise ValueError("unexpected callback address")
            if not query.get("code") or not query.get("state"):
                raise ValueError("callback is missing authorization data")
            callback = LOCAL_CALLBACK + "?" + urlencode({"code": query["code"][0], "state": query["state"][0]})
            urlopen(Request(callback, method="GET"), timeout=10).read()
        except Exception:
            self.send_page(400, "That callback was not accepted. Keep the browser address intact and try again.")
            return
        self.send_page(200, "Spotify authorization was sent to the Pi. You can close this page.")


if __name__ == "__main__":
    HTTPServer((HOST, PORT), RelayHandler).serve_forever()
