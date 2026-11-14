"""Robustness end to end (PLAN.md section 13): kill the control plane mid-step and resume, cancel running steps
through the API and retry them, and an approval that cannot be undone by a spec change."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx2
import pytest

from controlplane.settings import Settings
from controlplane.tracking import Tracker
from controlplane.workflow import Store
from tests.fake_llm import hang, text, tool
from tests.test_analyst import (agent_service, configure, cp, free_port, llm, settle, upload,  # noqa: F401
                                wait_for)
from tests.test_baseline import CLAIMS, POISSON_R, poisson_spec, project_at_baseline
from tests.test_report import assign_reporter, bindings, report
from tests.test_repair import state
from tests.test_translation import POISSON_PY
from tests.test_translator import SPEC, submit, translate

BACKEND = Path(__file__).resolve().parents[1]
SLOW = POISSON_PY.replace("def fit(train):\n", "def fit(train):\n    while True:\n        pass\n")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def control_plane(settings: Settings, port: int, log: Path) -> subprocess.Popen:
    env = {"PATH": os.environ["PATH"], "CONTROLPLANE_DATA_DIR": str(settings.data_dir),
           "CONTROLPLANE_MLFLOW_URI": settings.mlflow_uri, "CONTROLPLANE_AGENTS_URL": "http://127.0.0.1:9",
           "CONTROLPLANE_URL": f"http://127.0.0.1:{port}", "CONTROLPLANE_PORT": str(port)}
    proc = subprocess.Popen([sys.executable, "-m", "controlplane"], cwd=BACKEND, env=env, stdout=log.open("ab"),
                            stderr=subprocess.STDOUT)
    wait_for(f"http://127.0.0.1:{port}/api/health", proc, log)
    return proc


def wait_until(cond, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := cond():
            return value
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for {what}")


def test_a_killed_control_plane_leaves_nothing_running_and_the_step_reruns(envs, mlflow_uri, tmp_path):
    marker = tmp_path / "r.pid"
    # The first baseline run records R's pid and sleeps; the rerun finds the marker and fits straight away.
    slow_once = (f'if (!file.exists("{marker}")) {{ writeLines(as.character(Sys.getpid()), "{marker}"); '
                 'Sys.sleep(300) }\n')
    settings = Settings(data_dir=tmp_path / "var", mlflow_uri=mlflow_uri)
    settings.data_dir.mkdir()
    pid = project_at_baseline(Store(settings.db_path), settings, poisson_spec(baseline_r=slow_once + POISSON_R),
                              {"claims.csv": CLAIMS.encode(), "m.R": b""})
    port, log = free_port(), tmp_path / "controlplane.log"

    plane = control_plane(settings, port, log)                     # autostart picks up the confirmed spec
    try:
        r_pid = int(wait_until(lambda: marker.is_file() and marker.read_text().strip(), 60, "the R baseline"))
        with httpx2.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as c:
            job = c.get(f"/api/projects/{pid}").json()["running_job"]
        assert job["kind"] == "r_baseline" and _alive(r_pid)
    finally:
        plane.send_signal(signal.SIGKILL)                          # no shutdown handler runs
        plane.wait()
    wait_until(lambda: not _alive(r_pid), 5, "R to exit after the control plane died")

    plane = control_plane(settings, port, log)
    try:
        with httpx2.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as c:
            body = c.get(f"/api/projects/{pid}").json()
            s = body["state"]
            assert (s["step"], s["status"], s["reason"]) == (3, "awaiting_ds", "interrupted"), s
            assert body["running_job"] is None and s["detail"] == "interrupted by a restart"
            assert [(j["id"], j["status"]) for j in c.get(f"/api/projects/{pid}/jobs").json()] == [
                (job["id"], "interrupted")]
            tracker = Tracker(mlflow_uri)
            [run] = tracker.client.search_runs([tracker.experiment(pid)], "tags.kind = 'baseline'")
            assert (run.info.status, run.data.tags["error"]) == ("KILLED", "interrupted by a restart")

            assert c.post(f"/api/projects/{pid}/retry").status_code == 200
            s = settle(c, pid, until=lambda s: s["step"] != 3 and s["status"] != "working", timeout=120)
            assert s["baseline_id"] and s["step"] == 4               # the Translator has no model: step 4 waits
            assert s["detail"] == "no model is assigned to the translator role; assign one in settings"
            jobs = c.get(f"/api/projects/{pid}/jobs").json()
            assert [(j["kind"], j["status"]) for j in jobs] == [("r_baseline", "interrupted"),
                                                               ("r_baseline", "succeeded"), ("translation", "failed")]
            assert "interrupted" not in c.get(f"/api/projects/{pid}/jobs/{jobs[1]['id']}/log").text
            assert c.get(f"/api/projects/{pid}/jobs/{jobs[2]['id']}/log").status_code == 404
            coefs = c.get(f"/api/projects/{pid}/evidence/{s['baseline_id']}/coefficients").json()
            assert not coefs["python"] and [t["term"] for t in coefs["terms"]] == ["(Intercept)", "areaB", "areaC",
                                                                                    "age"]
            assert "R version" in c.get(f"/api/projects/{pid}/evidence/{s['baseline_id']}/files/sessionInfo.txt").text
            assert c.get(f"/api/projects/{pid}/evidence/{s['baseline_id']}/files/..%2Fspec.json").status_code == 404
    finally:
        plane.terminate()
        plane.wait(10)


def test_cancelling_a_running_candidate_kills_it_and_the_retry_passes(cp, llm):
    eid = configure(cp, llm)
    assert cp.put("/api/settings/roles/translator", json={"endpoint_id": eid, "model": "fake-1", "thinking": "off"}
                  ).status_code == 200
    llm.script = [tool("submit_model_spec", SPEC), text("Drafted."), submit(SLOW), text("Never returns.")]
    pid = upload(cp)
    settle(cp, pid)
    assert cp.post(f"/api/projects/{pid}/specs/1/confirm", json={"parity_risk_accepted": True}).status_code == 200
    wait_until(lambda: cp.get(f"/api/projects/{pid}/evidence", params={"kind": "candidate"}).json(), 180,
               "the candidate")
    job = cp.get(f"/api/projects/{pid}").json()["running_job"]
    assert job["kind"] == "translation"
    time.sleep(1)                                                   # the candidate is now spinning in fit()
    started = time.monotonic()
    assert cp.post(f"/api/projects/{pid}/cancel").json() == {"job_id": job["id"]}
    s = settle(cp, pid, timeout=30)
    assert (s["step"], s["status"], s["reason"]) == (4, "awaiting_ds", "cancelled") and time.monotonic() - started < 15
    assert cp.post(f"/api/projects/{pid}/cancel").status_code == 409

    tracker = Tracker(cp.settings.mlflow_uri)
    [run] = tracker.client.search_runs([tracker.experiment(pid)], "tags.kind = 'validation'")
    assert run.info.status == "KILLED"
    assert cp.get(f"/api/projects/{pid}/evidence", params={"kind": "validation"}).json() == []

    llm.script = [submit(POISSON_PY), text("Done.")]
    assert cp.post(f"/api/projects/{pid}/retry").status_code == 200
    s = settle(cp, pid, timeout=120)
    assert (s["step"], s["reason"]) == (4, "candidate_passed"), s
    coefs = cp.get(f"/api/projects/{pid}/evidence/{s['validation_id']}/coefficients").json()
    assert coefs["python"] and len(coefs["terms"]) == 4
    assert all(t["estimate_rel"] is not None and t["estimate_rel"] < 1e-6 for t in coefs["terms"])
    assert [j["status"] for j in cp.get(f"/api/projects/{pid}/jobs").json()] == [
        "succeeded", "succeeded", "cancelled", "succeeded"]


def test_after_approval_the_spec_cannot_change_and_a_cancelled_report_reruns(cp, llm):
    pid = translate(cp, llm, [submit(POISSON_PY), text("Done.")])
    s = state(cp, pid)
    assign_reporter(cp)
    llm.script = [hang(60)]
    assert cp.post(f"/api/projects/{pid}/approve", json=bindings(s)).status_code == 200
    wait_until(lambda: (cp.get(f"/api/projects/{pid}").json()["running_job"] or {}).get("kind") == "report", 30,
               "the Reporter")
    r = cp.post(f"/api/projects/{pid}/chat", json={"text": "Change the tolerances."})
    assert r.status_code == 409 and "requires step 2" in r.json()["detail"]
    assert cp.post(f"/api/projects/{pid}/specs/1/confirm", json={"parity_risk_accepted": True}).status_code == 409
    assert cp.post(f"/api/projects/{pid}/cancel").status_code == 202
    s = settle(cp, pid, timeout=20)
    assert (s["step"], s["reason"]) == (6, "cancelled") and s["approval"] == bindings(s)
    assert cp.post(f"/api/projects/{pid}/chat", json={"text": "Change the tolerances."}).status_code == 409
    [d] = cp.get(f"/api/projects/{pid}").json()["decisions"]
    assert d["decision"] == "approved" and not d["void"]

    [decision] = [f"decision-{t['seq']}" for t in cp.get(f"/api/projects/{pid}/transitions").json()
                  if t["event"]["type"] == "candidate_approved"]
    llm.script = [report(("Result", [("The validation passed and the DS approved it.",
                                      [s["validation_id"], decision])])), text("Reported.")]
    assert cp.post(f"/api/projects/{pid}/retry").status_code == 200
    s = settle(cp, pid, timeout=120)
    assert (s["step"], s["status"]) == (7, "done"), s
