"""Step 2 end to end: the real agent service and Pi, a scripted model server, and the control plane on a real port
so the Analyst's tool calls come back over HTTP."""
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import httpx2
import pytest
import uvicorn

from controlplane.api.app import create_app
from controlplane.settings import Settings
from controlplane.tracking import Tracker
from tests.fake_llm import FakeLLM, hang, status, text, tool
from tests.test_baseline import CLAIMS, poisson_spec

AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"
DATA = "\n".join(f"{line},{'policy' if i == 0 else f'secret-{i}'}" for i, line in enumerate(CLAIMS.splitlines()))
CODE = "# claims frequency\n" + poisson_spec().baseline_r + "summary(fit)\n"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, proc: subprocess.Popen | None = None, log: Path | None = None) -> None:
    for _ in range(200):
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except OSError:
            if proc is not None and proc.poll() is not None:
                pytest.fail(log.read_text() if log else "process exited")
            time.sleep(0.05)
    pytest.fail(f"{url} did not come up")


@pytest.fixture(scope="session")
def agent_service(tmp_path_factory):
    node = shutil.which("node")
    if node is None or not (AGENTS_DIR / "node_modules").is_dir():
        pytest.skip("the agent service is not installed; run platform/bootstrap.sh")
    home, port = tmp_path_factory.mktemp("agents"), free_port()
    log = home / "agents.log"
    proc = subprocess.Popen([node, "server.ts"], cwd=AGENTS_DIR, stdout=log.open("wb"), stderr=subprocess.STDOUT,
                            env={"PATH": os.environ["PATH"], "HOME": str(home), "AGENTS_PORT": str(port)})
    url = f"http://127.0.0.1:{port}"
    wait_for(f"{url}/health", proc, log)
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.fixture
def llm():
    fake = FakeLLM()
    yield fake
    fake.close()


@pytest.fixture
def cp(tmp_path, agent_service, mlflow_uri):
    port = free_port()
    settings = Settings(data_dir=tmp_path / "var", agents_url=agent_service, url=f"http://127.0.0.1:{port}",
                        mlflow_uri=mlflow_uri)
    server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_for(f"{settings.url}/api/health")
    with httpx2.Client(base_url=settings.url, timeout=30) as c:
        c.settings = settings
        yield c
    server.should_exit = True
    thread.join(10)


def configure(c, llm, key="sk-test") -> str:
    r = c.post("/api/settings/endpoints", json={"name": "fake", "base_url": llm.url, "api": "openai-completions",
                                                 "models": ["fake-1"], "api_key": key})
    assert r.status_code == 201, r.text
    eid = r.json()["id"]
    assert c.put("/api/settings/roles/analyst", json={"endpoint_id": eid, "model": "fake-1", "thinking": "off"}
                 ).status_code == 200
    return eid


def upload(c) -> str:
    r = c.post("/api/projects", data={"name": "claims"},
               files=[("dataset", ("claims.csv", DATA.encode())), ("r_code", ("model.R", CODE.encode()))])
    assert r.status_code == 201, r.text
    return r.json()["id"]


def settle(c, pid, until=lambda s: s["status"] != "working", timeout=60) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = c.get(f"/api/projects/{pid}").json()
        if body["running_job"] is None and until(body["state"]):
            return body["state"]
        time.sleep(0.05)
    pytest.fail(f"project did not settle: {body}")


def test_analyst_drafts_the_spec_answers_the_ds_and_the_confirmed_spec_runs_the_baseline(cp, llm, envs, tmp_path):
    key = f"!touch {tmp_path / 'pwned'}"                        # Pi would run a `!command` given as a provider key
    configure(cp, llm, key)
    bad = poisson_spec(baseline_r=poisson_spec().baseline_r.replace("inputs/claims.csv", "claims.csv"))
    llm.script = [tool("dataset_profile", {}), tool("read_file", {"name": "model.R"}),
                  tool("submit_model_spec", bad.model_dump(mode="json")),
                  tool("submit_model_spec", poisson_spec().model_dump(mode="json")),
                  text("A Poisson frequency GLM with a log exposure offset.")]
    pid = upload(cp)
    s = settle(cp, pid)
    assert (s["step"], s["status"], s["reason"], s["drafted_version"]) == (2, "awaiting_ds", "spec_drafted", 1), s

    sent = llm.sent()
    assert "inputs/claims.csv" in llm.sent(llm.requests[:1])                               # project summary
    assert '"rows": 200' in sent and '"levels": [' in sent and "summary(fit)" in sent      # profile and code
    assert "secret-" not in sent and "1,3,0.6,B,33" not in sent                           # never a data row
    assert "baseline_r must read the dataset from inputs/claims.csv" in llm.sent(llm.requests[3:4])
    assert {r["headers"]["authorization"] for r in llm.requests} == {f"Bearer {key}"}
    assert not (tmp_path / "pwned").exists()

    chat = cp.get(f"/api/projects/{pid}/chat").json()
    assert [(m["author"], m["spec_version"]) for m in chat] == [("analyst", 1)]
    tracker = Tracker(cp.settings.mlflow_uri)
    [run] = tracker.client.search_runs([tracker.experiment(pid)], "tags.kind = 'analyst'")
    assert run.info.status == "FINISHED" and run.data.params["model"] == "fake / fake-1"
    assert run.data.metrics == {"tool_calls": 4.0, "invalid_submissions": 1.0}
    spec = cp.get(f"/api/projects/{pid}/specs/1").json()
    assert spec["spec"]["family"] == "poisson" and spec["support"] == {"tier": "exact", "proven": False, "known": True}

    # the next turn replays the whole transcript, tool results included
    llm.script = [text("Claims are counts per unit of exposure.")]
    assert cp.post(f"/api/projects/{pid}/chat", json={"text": "Why Poisson?"}).status_code == 202
    s = settle(cp, pid)
    assert (s["status"], s["reason"]) == ("awaiting_ds", "analyst_replied")
    last = llm.sent(llm.requests[-1:])
    assert "Why Poisson?" in last and '"rows": 200' in last and "log exposure offset" in last
    assert [m["author"] for m in cp.get(f"/api/projects/{pid}/chat").json()] == ["analyst", "ds", "analyst"]

    assert cp.post(f"/api/projects/{pid}/specs/1/confirm", json={}).status_code == 409    # parity risk not accepted
    assert cp.post(f"/api/projects/{pid}/specs/1/confirm", json={"parity_risk_accepted": True}).status_code == 200
    s = settle(cp, pid, until=lambda s: s["step"] == 4, timeout=120)
    assert s["baseline_id"] and s["spec_version"] == 1


def test_a_failed_baseline_goes_back_to_the_analyst_for_a_corrected_spec(cp, llm, envs):
    configure(cp, llm)
    broken = poisson_spec(baseline_r=poisson_spec().baseline_r.replace("fit <- glm", "fit <- glmm"))
    llm.script = [tool("submit_model_spec", broken.model_dump(mode="json")), text("Drafted.")]
    pid = upload(cp)
    settle(cp, pid)
    cp.post(f"/api/projects/{pid}/specs/1/confirm", json={"parity_risk_accepted": True})
    s = settle(cp, pid)
    assert (s["step"], s["reason"]) == (3, "step_failed") and 'could not find function "glmm"' in s["detail"]
    assert cp.post(f"/api/projects/{pid}/retry").status_code == 200       # retry is still possible...
    settle(cp, pid)

    llm.script = [tool("submit_model_spec", poisson_spec().model_dump(mode="json")), text("Fixed the call.")]
    assert cp.post(f"/api/projects/{pid}/chat", json={"text": "Please fix it."}).status_code == 202   # ...or fix
    s = settle(cp, pid)
    assert (s["step"], s["reason"], s["drafted_version"], s["spec_version"]) == (2, "spec_drafted", 2, 1)
    turn = llm.sent(llm.requests[-2:-1])
    assert "The R baseline for spec v1 failed" in turn and "glmm" in turn and "Please fix it." in turn
    assert [m["author"] for m in cp.get(f"/api/projects/{pid}/chat").json()] == [
        "analyst", "platform", "platform", "ds", "analyst"]

    cp.post(f"/api/projects/{pid}/specs/2/confirm", json={"parity_risk_accepted": True})
    s = settle(cp, pid, until=lambda s: s["step"] == 4, timeout=120)
    assert s["spec_version"] == 2 and s["baseline_id"]


def test_cancel_aborts_the_analyst_run(cp, llm):
    configure(cp, llm)
    llm.script = [hang(30)]
    pid = upload(cp)
    deadline = time.monotonic() + 30
    while not llm.requests and time.monotonic() < deadline:
        time.sleep(0.05)
    jid = cp.get(f"/api/projects/{pid}").json()["running_job"]["id"]
    events = []

    def listen():
        with cp.stream("GET", f"/api/jobs/{jid}/events") as r:
            events.extend(json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: "))
    listener = threading.Thread(target=listen)
    listener.start()
    time.sleep(0.5)
    started = time.monotonic()
    assert cp.post(f"/api/projects/{pid}/cancel").status_code == 202
    s = settle(cp, pid)
    assert (s["status"], s["reason"]) == ("awaiting_ds", "cancelled") and time.monotonic() - started < 10
    assert cp.get(f"/api/projects/{pid}/chat").json() == []
    listener.join(10)
    assert events[0] == {"type": "job", "job_id": jid, "kind": "analyst"}
    assert {"type": "text", "delta": "thinking"} in events
    assert cp.get(f"/api/jobs/{jid}/events").status_code == 404

    llm.script = [text("Nothing to draft yet.")]                 # retry runs the opening turn again
    assert cp.post(f"/api/projects/{pid}/retry").status_code == 200
    assert settle(cp, pid)["reason"] == "analyst_replied"
    assert "Project files:" in llm.sent(llm.requests[-1:])


def test_model_errors_and_a_missing_role_model_fail_the_turn_with_the_reason(cp, llm):
    pid = upload(cp)
    s = settle(cp, pid)
    assert s["reason"] == "step_failed" and "no model is assigned to the analyst role" in s["detail"]

    configure(cp, llm)
    llm.script = [status(401)]
    cp.post(f"/api/projects/{pid}/retry")
    s = settle(cp, pid)
    assert s["reason"] == "step_failed" and "scripted 401" in s["detail"]
    assert cp.get(f"/api/projects/{pid}/chat").json() == []


def test_endpoint_keys_are_encrypted_and_never_returned(cp, llm):
    eid = configure(cp, llm, key="sk-very-secret")
    listed = cp.get("/api/settings/endpoints").text
    assert "sk-very-secret" not in listed and json.loads(listed)[0]["has_key"] is True
    db = cp.settings.db_path.read_bytes()
    assert b"sk-very-secret" not in db
    assert oct((cp.settings.data_dir / "secret.key").stat().st_mode & 0o777) == "0o600"

    llm.script = [text("OK")]
    assert cp.post(f"/api/settings/endpoints/{eid}/test", json={"model": "fake-1"}).json() == {"ok": True, "text": "OK"}
    assert llm.requests[-1]["headers"]["authorization"] == "Bearer sk-very-secret"
    llm.script = [status(401)]
    r = cp.post(f"/api/settings/endpoints/{eid}/test", json={"model": "fake-1"}).json()
    assert r["ok"] is False and "401" in r["error"]

    # updating without a key keeps the stored one
    cp.put(f"/api/settings/endpoints/{eid}", json={"name": "renamed", "base_url": llm.url,
                                                   "api": "openai-completions", "models": ["fake-1"]})
    llm.script = [text("OK")]
    cp.post(f"/api/settings/endpoints/{eid}/test", json={"model": "fake-1"})
    assert llm.requests[-1]["headers"]["authorization"] == "Bearer sk-very-secret"


def test_tool_calls_need_a_live_run_token(cp):
    r = cp.post("/internal/tools/read_file", json={"name": "model.R"}, headers={"authorization": "Bearer nope"})
    assert r.status_code == 401
