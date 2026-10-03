"""Run the classroom demo with `python3 app.py` (or `py app.py` on Windows)."""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from model import LABELS, REPLIES, build_base

ROOT = Path(__file__).parent


class Experiment:
    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            base, self.examples, self.eval_examples = build_base()
            self.base = base
            self.current = base.copy()
            self.history = []
            self.training = False
            self.trained = False

    def status(self):
        with self.lock:
            return {
                "training": self.training,
                "trained": self.trained,
                "history": self.history.copy(),
                "base_accuracy": round(self.base.accuracy(self.eval_examples), 3),
                "current_accuracy": round(self.current.accuracy(self.eval_examples), 3),
                "training_examples": len(self.examples),
                "held_out_examples": len(self.eval_examples),
                "parameters": self.current.parameter_count,
                "labels": LABELS,
                "replies": REPLIES,
            }

    def chat(self, prompt: str):
        with self.lock:
            return {"base": self.base.predict(prompt), "current": self.current.predict(prompt)}

    def start_training(self):
        with self.lock:
            if self.training:
                return False
            self.current = self.base.copy()
            self.history = []
            self.training = True
            self.trained = False
        threading.Thread(target=self._train, daemon=True).start()
        return True

    def _train(self):
        try:
            for epoch in range(1, 41):
                with self.lock:
                    loss = self.current.epoch(self.examples, 0.035, 1000 + epoch)
                    accuracy = self.current.accuracy(self.eval_examples)
                    self.history.append({"epoch": epoch, "loss": round(loss, 4), "accuracy": round(accuracy, 3)})
                time.sleep(0.07)  # makes the live update visible; computation is much faster
        finally:
            with self.lock:
                self.training = False
                self.trained = True


EXPERIMENT = Experiment()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, obj, status=200):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/api/status":
            return self.send_json(EXPERIMENT.status())
        if self.path in ("/", "/app.js", "/style.css"):
            name = "index.html" if self.path == "/" else self.path[1:]
            content_type = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[name]
            payload = (ROOT / "web" / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            return self.wfile.write(payload)
        self.send_error(404)

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/train", "/api/reset"):
            return self.send_error(404)
        length = int(self.headers.get("Content-Length", "0"))
        if length > 10000:
            return self.send_json({"error": "Request too large"}, 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self.send_json({"error": "Invalid JSON"}, 400)
        if self.path == "/api/chat":
            prompt = body.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 500:
                return self.send_json({"error": "Enter a prompt of at most 500 characters."}, 400)
            return self.send_json(EXPERIMENT.chat(prompt.strip()))
        if self.path == "/api/train":
            started = EXPERIMENT.start_training()
            return self.send_json({"started": started, **EXPERIMENT.status()})
        if EXPERIMENT.status()["training"]:
            return self.send_json({"error": "Wait for training to finish."}, 409)
        EXPERIMENT.reset()
        return self.send_json(EXPERIMENT.status())


def main():
    parser = argparse.ArgumentParser(description="Local fine-tuning classroom demo")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open the browser automatically")
    args = parser.parse_args()
    address = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Fine-tuning lab: {address}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
