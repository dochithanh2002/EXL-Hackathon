"""Client for the Node agent service (`platform/agents`)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable


class AgentServiceError(Exception):
    pass


class AgentService:
    def __init__(self, url: str, timeout_s: float = 3600):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s

    def run(self, request: dict, on_event: Callable[[dict], None]) -> dict:
        """Stream one role run. Returns the final event: `done` (text, messages), `error` (message) or `aborted`."""
        try:
            with self._post("/runs", request, self.timeout_s) as r:
                for line in r:
                    event = json.loads(line)
                    if event["type"] in ("done", "error", "aborted"):
                        return event
                    on_event(event)
        except (urllib.error.URLError, OSError) as e:
            raise AgentServiceError(f"agent service at {self.url} failed: {e}") from e
        raise AgentServiceError("the agent service closed the run without a result")

    def abort(self, run_id: str) -> bool:
        try:
            with self._post(f"/runs/{run_id}/abort", {}, 10):
                return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    def test(self, model_ref: dict, timeout_s: float = 120) -> dict:
        try:
            with self._post("/test", {"model": model_ref}, timeout_s) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError) as e:
            return {"ok": False, "error": f"agent service at {self.url} failed: {e}"}

    def _post(self, path: str, body: dict, timeout: float):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)
