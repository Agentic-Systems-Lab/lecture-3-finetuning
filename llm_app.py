"""Browser UI for real generative LLM fine-tuning: `uv run llm_app.py --open`."""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
LAB = None


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/api/status":
            return self.send_json(LAB.status())
        if self.path == "/api/examples":
            return self.send_json({"examples": LAB.training_rows})
        files = {"/": ("llm.html", "text/html"), "/llm.js": ("llm.js", "text/javascript"), "/style.css": ("style.css", "text/css"), "/llm.css": ("llm.css", "text/css")}
        if self.path in files:
            name, mime = files[self.path]
            payload = (ROOT / "web" / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{mime}; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            return self.wfile.write(payload)
        self.send_error(404)

    def do_POST(self):
        if self.path not in ("/api/generate", "/api/train"):
            return self.send_error(404)
        length = int(self.headers.get("Content-Length", "0"))
        if length > 20000:
            return self.send_json({"error": "Request too large"}, 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self.send_json({"error": "Invalid JSON"}, 400)
        try:
            if self.path == "/api/generate":
                prompt = body.get("prompt", "")
                if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 300:
                    return self.send_json({"error": "Enter a prompt of at most 300 characters."}, 400)
                variant = body.get("model", "base")
                if variant not in ("base", "tuned"):
                    return self.send_json({"error": "Choose the base or fine-tuned model."}, 400)
                return self.send_json(LAB.generate(prompt.strip(), variant))
            return self.send_json({"started": LAB.start_training(), **LAB.status()})
        except RuntimeError as error:
            return self.send_json({"error": str(error)}, 409)


def main():
    parser = argparse.ArgumentParser(description="Real generative LLM fine-tuning lab")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open browser when ready")
    args = parser.parse_args()
    try:
        from llm_lab import GenerativeLab
    except ImportError as error:
        raise SystemExit("Install the real-model dependencies first: uv sync (or pip install -r requirements-llm.txt)") from error
    print("Loading SmolLM2-135M-Instruct. The first run downloads the model...", flush=True)
    global LAB
    LAB = GenerativeLab()
    address = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Generative fine-tuning lab: {address}\nPress Ctrl+C to stop.", flush=True)
    if args.open:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
