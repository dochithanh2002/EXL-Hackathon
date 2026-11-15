"""A scripted OpenAI-compatible chat-completions server. Each request takes the next scripted reply and is recorded,
so tests can drive the real agent service and inspect everything it sent to the model."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def text(content: str) -> dict:
    return {"text": content}


def tool(name: str, args: dict) -> dict:
    return {"tool": name, "args": args}


def status(code: int) -> dict:
    return {"status": code}


def hang(seconds: float) -> dict:
    return {"hang": seconds}


class FakeLLM:
    def __init__(self):
        self.script: list[dict] = []
        self.requests: list[dict] = []
        self._calls = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                fake.requests.append({"path": self.path, "body": body,
                                      "headers": {k.lower(): v for k, v in self.headers.items()}})
                reply = fake.script.pop(0) if fake.script else text("(script exhausted)")
                if "status" in reply:
                    self.send_response(reply["status"])
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": {"message": f"scripted {reply['status']}"}}).encode())
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                if "hang" in reply:
                    self._chunk({"role": "assistant", "content": "thinking"})
                    time.sleep(reply["hang"])
                    return
                fake._calls += 1
                if "tool" in reply:
                    self._chunk({"role": "assistant", "tool_calls": [{
                        "index": 0, "id": f"call_{fake._calls}", "type": "function",
                        "function": {"name": reply["tool"], "arguments": json.dumps(reply["args"])}}]})
                    self._chunk({}, "tool_calls")
                else:
                    self._chunk({"role": "assistant", "content": reply["text"]})
                    self._chunk({}, "stop")
                self.wfile.write(b"data: [DONE]\n\n")

            def _chunk(self, delta: dict, finish: str | None = None):
                chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "fake",
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if finish:
                    chunk["usage"] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def sent(self, requests: list[dict] | None = None) -> str:
        """Every string the agent service sent to the model, one per line."""
        out: list[str] = []

        def walk(node):
            if isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
            elif isinstance(node, str):
                out.append(node)
        walk([r["body"] for r in (self.requests if requests is None else requests)])
        return "\n".join(out)

    def close(self):
        self.server.shutdown()
