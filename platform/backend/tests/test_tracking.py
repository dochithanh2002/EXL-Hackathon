"""MLflow is a mirror: jobs, the report and the bundle never wait for it, and its queue can be retried."""
import sqlite3
import threading

import httpx2
import pytest
import uvicorn

from controlplane.api.app import create_app
from controlplane.settings import Settings
from controlplane.tracking import Mirror, Run, Tracker
from controlplane.workflow import Status, Step, Store
from controlplane.workflow.machine import RetryStep
from tests.conftest import make_jobs
from tests.fake_llm import text
from tests.test_analyst import agent_service, free_port, llm, settle, synced, wait_for  # noqa: F401
from tests.test_recovery import wait_until
from tests.test_baseline import CLAIMS, poisson_spec, project_at_baseline
from tests.test_report import assign_reporter, bindings, report
from tests.test_translation import POISSON_PY
from tests.test_translator import submit, translate


@pytest.fixture
def dead_uri(monkeypatch):
    """An MLflow URI nothing listens on, failing without the client's retries."""
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    return f"http://127.0.0.1:{free_port()}"


def test_failed_enqueue_cannot_commit_a_baseline_without_its_tracking_intent(envs, tmp_path):
    jobs = make_jobs(tmp_path, envs, Tracker("http://127.0.0.1:9"))
    store = jobs.store
    pid = project_at_baseline(store, jobs.settings, poisson_spec(), {"claims.csv": CLAIMS.encode(), "m.R": b""})
    with sqlite3.connect(store.path) as db:
        db.execute("CREATE TRIGGER fail_enqueue BEFORE INSERT ON tracking WHEN NEW.evidence_id IS NOT NULL "
                   "BEGIN SELECT RAISE(ABORT, 'tracking insert failed'); END")

    jid = jobs.baseline(pid)
    state = store.state(pid)
    assert state.step == Step.BASELINE and state.status == Status.AWAITING_DS
    assert state.baseline_id is None and store.evidence_list(pid, "baseline") == []
    assert store.get_job(jid)["status"] == "failed"

    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER fail_enqueue")
    store.apply(pid, RetryStep())
    retry = jobs.baseline(pid)
    state = store.state(pid)
    assert state.step == Step.TRANSLATION and state.baseline_id
    [row] = [r for r in store.tracking(pid) if r["job_id"] == retry]
    assert (row["evidence_id"], row["status"]) == (state.baseline_id, "pending")


def test_a_job_completes_with_mlflow_unreachable_and_its_mirror_is_retried(envs, tmp_path, dead_uri, mlflow_uri):
    jobs = make_jobs(tmp_path, envs, Tracker(dead_uri))
    pid = project_at_baseline(jobs.store, jobs.settings, poisson_spec(), {"claims.csv": CLAIMS.encode(), "m.R": b""})
    jobs.baseline(pid)
    s = jobs.store.state(pid)
    assert (s.step, s.status) == (Step.TRANSLATION, Status.WORKING) and s.baseline_id

    assert jobs.mirror.sync() == 1
    [row] = jobs.store.tracking(pid)
    assert (row["kind"], row["evidence_id"], row["status"], row["attempts"]) == ("baseline", s.baseline_id,
                                                                                  "failed", 1)
    assert row["last_error"] and row["mlflow_run_id"] is None
    assert jobs.mirror.sync() == 0                                 # failed rows wait for a retry

    jobs.mirror.tracker = tracker = Tracker(mlflow_uri)
    assert jobs.store.retry_tracking(pid) == [row["id"]]
    assert jobs.mirror.sync() == 1
    [row] = jobs.store.tracking(pid)
    assert (row["status"], row["attempts"], row["last_error"]) == ("done", 2, "")
    run = tracker.client.get_run(row["mlflow_run_id"])
    assert run.info.status == "FINISHED" and run.data.tags["mlflow.parentRunId"] == tracker.workflow_run(pid)
    assert "capture" in {a.path for a in tracker.client.list_artifacts(run.info.run_id)}
    with pytest.raises(ValueError, match="not failed"):
        jobs.store.retry_tracking(pid, row["id"])


def test_a_retry_replaces_a_half_made_run_and_large_files_are_skipped(tmp_path, mlflow_uri):
    store = Store(tmp_path / "db.sqlite")
    pid = store.create_project("p")
    out = tmp_path / "var" / "out"
    out.mkdir(parents=True)
    (out / "small.csv").write_text("a\n1\n")
    (out / "big.csv").write_bytes(b"0" * 300_000)
    run = Run("baseline", "baseline v1")
    run.params(spec_version=1)
    run.artifact(out, "capture")
    run.artifact(out.parent / "later.csv")
    rid = store.track(pid, "baseline", run.payload(tmp_path / "var"))
    child = Run("model_fit", "python fit", parent=rid)
    child.links(reference_run_id=rid)
    cid = store.track(pid, "model_fit", child.payload(tmp_path / "var"))
    tracker = Tracker(mlflow_uri)
    mirror = Mirror(store, tracker, tmp_path / "var", max_file_mb=0.1)

    assert mirror.sync() == 1                                      # the child waits for its parent
    first, waiting = store.tracking(pid)
    assert first["status"] == "failed" and first["mlflow_run_id"] and waiting["status"] == "pending"

    (out.parent / "later.csv").write_text("b\n2\n")
    store.retry_tracking(pid, rid)
    assert mirror.sync() == 2
    parent, fit = store.tracking(pid)
    assert parent["status"] == fit["status"] == "done" and parent["mlflow_run_id"] != first["mlflow_run_id"]
    old = tracker.client.get_run(first["mlflow_run_id"])
    assert old.info.lifecycle_stage == "deleted" and old.info.status == "FAILED"
    [active] = tracker.client.search_runs([tracker.experiment(pid)], "tags.kind = 'baseline'")
    assert active.info.run_id == parent["mlflow_run_id"]
    assert active.data.tags["skipped_artifacts"] == "capture/big.csv (0.3 MB)"
    assert {a.path for a in tracker.client.list_artifacts(active.info.run_id, "capture")} == {"capture/small.csv"}
    child_run = tracker.client.get_run(fit["mlflow_run_id"]).data.tags
    assert child_run["mlflow.parentRunId"] == child_run["reference_run_id"] == active.info.run_id


@pytest.fixture
def offline(tmp_path, agent_service, dead_uri):
    """The control plane with MLflow unreachable; the test may point its mirror at a live server later."""
    port = free_port()
    settings = Settings(data_dir=tmp_path / "var", agents_url=agent_service, url=f"http://127.0.0.1:{port}",
                        mlflow_uri=dead_uri)
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_for(f"{settings.url}/api/health")
    with httpx2.Client(base_url=settings.url, timeout=30) as c:
        c.settings, c.app = settings, app
        yield c
    server.should_exit = True
    thread.join(10)


def test_the_workflow_delivers_without_mlflow_and_mirrors_after_a_retry(offline, llm, mlflow_uri):
    cp = offline
    pid = translate(cp, llm, [submit(POISSON_PY), text("Done.")])
    s = settle(cp, pid)
    assert (s["step"], s["reason"]) == (4, "candidate_passed"), s
    assign_reporter(cp)
    decision = f"decision-{len(cp.get(f'/api/projects/{pid}/transitions').json()) + 1}"
    llm.script = [report(s["validation_id"], decision), text("Reported.")]
    assert cp.post(f"/api/projects/{pid}/approve", json=bindings(s)).status_code == 200
    s = settle(cp, pid, timeout=120)
    assert (s["step"], s["status"]) == (7, "done"), s

    md = cp.get(f"/api/projects/{pid}/report").json()["body"]["markdown"]
    assert "![" in md and "mlflow" not in md.lower()
    bundle = cp.get(f"/api/projects/{pid}/deliverables").json()
    assert bundle["mlflow_run_id"] is None and "plots/parity.svg" in bundle["manifest"]["files"]
    assert cp.get(f"/api/projects/{pid}/deliverables/bundle.zip").status_code == 200
    assert cp.get(f"/api/projects/{pid}/evidence/{s['validation_id']}").json()["body"]["mlflow_run_id"] is None
    assert cp.get(f"/api/projects/{pid}/mlflow").status_code == 503

    steps = {"analyst", "baseline", "translation", "report", "deliverables"}
    rows = wait_until(lambda: (rows := cp.get(f"/api/projects/{pid}/tracking").json()) and steps == {
        r["kind"] for r in rows if r["status"] == "failed"} and rows, 60, "the step rows to fail")
    assert "payload" not in rows[0] and all(r["last_error"] for r in rows if r["status"] == "failed")
    # the validation (linked to the translation), its nested runs and the decision wait for the runs they name
    assert {r["kind"] for r in rows if r["status"] == "pending"} == {"validation", "model_fit", "cv_fold",
                                                                    "challenger", "decision"}

    cp.app.state.mirror.tracker = tracker = Tracker(mlflow_uri)
    queued = cp.post(f"/api/projects/{pid}/tracking/retry").json()["queued"]
    assert len(queued) == len(steps)
    assert {r["status"] for r in synced(cp, pid)} == {"done"}
    assert cp.post(f"/api/projects/{pid}/tracking/retry", json={"id": queued[0]}).status_code == 409

    exp = tracker.experiment(pid)
    [b_run] = tracker.client.search_runs([exp], "tags.kind = 'baseline'")
    [t_run] = tracker.client.search_runs([exp], "tags.kind = 'translation'")
    [v_run] = tracker.client.search_runs([exp], "tags.kind = 'validation'")
    [fit] = tracker.client.search_runs([exp], "tags.kind = 'model_fit'")
    folds = tracker.client.search_runs([exp], "tags.kind = 'cv_fold'")
    [challenger] = tracker.client.search_runs([exp], "tags.kind = 'challenger'")
    workflow = tracker.workflow_run(pid)
    assert b_run.data.tags["mlflow.parentRunId"] == v_run.data.tags["mlflow.parentRunId"] == workflow
    assert fit.data.tags["mlflow.parentRunId"] == challenger.data.tags["mlflow.parentRunId"] == v_run.info.run_id
    assert len(folds) == 3 and {f.data.tags["mlflow.parentRunId"] for f in folds} == {fit.info.run_id}
    assert fit.data.tags["reference_run_id"] == v_run.data.tags["reference_run_id"] == b_run.info.run_id
    assert fit.data.tags["candidate_run_id"] == t_run.info.run_id
    assert fit.data.tags["validation_run_id"] == v_run.info.run_id
    assert v_run.data.tags["decision"] == t_run.data.tags["decision"] == "approved"
    assert "plots" in {a.path for a in tracker.client.list_artifacts(v_run.info.run_id)}
    v = cp.get(f"/api/projects/{pid}/evidence/{s['validation_id']}").json()["body"]
    assert v["mlflow_run_id"] == v_run.info.run_id
    jobs = cp.get(f"/api/projects/{pid}/jobs").json()
    assert [j["mlflow_run_ids"] for j in jobs if j["kind"] == "translation"] == [[t_run.info.run_id,
                                                                                 v_run.info.run_id]]
