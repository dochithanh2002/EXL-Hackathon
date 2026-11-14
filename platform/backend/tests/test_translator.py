"""Step 4 end to end: the real agent service and Pi with a scripted model, from a confirmed spec through the R
baseline, the Translator, the candidate run and validation, to the DS's approval or rejection."""
import hashlib

from controlplane.tracking import Tracker
from tests.fake_llm import text, tool
from tests.test_analyst import agent_service, configure, cp, llm, settle, upload  # noqa: F401
from tests.test_baseline import poisson_spec
from tests.test_translation import POISSON_PY

SPEC = poisson_spec().model_dump(mode="json")
DRIFTED = POISSON_PY.replace('offset=np.log(train["exposure"])', 'offset=np.log(train["exposure"]) * 0.99')


def translate(cp, llm, translator_script: list, eid: str | None = None) -> str:
    """Draft and confirm the Poisson spec; the baseline then runs and the Translator takes the scripted turn."""
    eid = eid or configure(cp, llm)
    assert cp.put("/api/settings/roles/translator", json={"endpoint_id": eid, "model": "fake-1", "thinking": "off"}
                  ).status_code == 200
    llm.script = [tool("submit_model_spec", SPEC), text("Drafted.")] + translator_script
    pid = upload(cp)
    settle(cp, pid)
    assert cp.post(f"/api/projects/{pid}/specs/1/confirm", json={"parity_risk_accepted": True}).status_code == 200
    settle(cp, pid, until=lambda s: s["step"] != 3 and s["status"] != "working", timeout=180)
    return pid


def submit(module: str) -> dict:
    return tool("submit_candidate", {"module": module, "notes": ["area is treatment-coded with A as reference"]})


def test_translator_candidate_passes_validation_and_the_ds_approves_it(cp, llm):
    pid = translate(cp, llm, [tool("dataset_profile", {}), tool("read_rulebook", {}),
                              submit("import os\n" + POISSON_PY), submit(POISSON_PY),
                              text("A Poisson GLM with R's starting values.")])
    s = cp.get(f"/api/projects/{pid}").json()["state"]
    assert (s["step"], s["status"], s["reason"]) == (4, "awaiting_ds", "candidate_passed"), s
    assert s["candidate_hash"] == hashlib.sha256(POISSON_PY.encode()).hexdigest()

    turn = llm.requests[2:]
    sent = llm.sent(turn)
    assert "You are the Translator" in sent and '"(Intercept)", "areaB", "areaC", "age"' in llm.sent(turn[:1])
    assert "R-EST-02" in sent and '"rows": 200' in sent                     # rulebook and profile were read
    assert "guardrail: line 1: module 'os' is not available" in sent
    assert "secret-" not in sent                                           # never a data row
    baseline = cp.get(f"/api/projects/{pid}/evidence/{s['baseline_id']}").json()["body"]
    coefs = (cp.settings.data_dir / baseline["dir"] / "coefficients.csv").read_text().splitlines()[1:]
    assert not any(line.split(",")[1] in sent for line in coefs)           # never R's estimates

    [cand] = cp.get(f"/api/projects/{pid}/evidence", params={"kind": "candidate"}).json()
    [val] = cp.get(f"/api/projects/{pid}/evidence", params={"kind": "validation"}).json()
    assert val["id"] == s["validation_id"]
    v = cp.get(f"/api/projects/{pid}/evidence/{val['id']}").json()["body"]
    assert v["passed"] and v["candidate_id"] == cand["id"] and v["stage"] == "parity"
    c = cp.get(f"/api/projects/{pid}/evidence/{cand['id']}").json()["body"]
    assert c["module"] == POISSON_PY and c["model"] == "fake / fake-1"

    tracker = Tracker(cp.settings.mlflow_uri)
    exp = tracker.experiment(pid)
    [t_run] = tracker.client.search_runs([exp], "tags.kind = 'translation'")
    [v_run] = tracker.client.search_runs([exp], "tags.kind = 'validation'")
    assert t_run.info.status == v_run.info.status == "FINISHED"
    assert t_run.data.metrics == {"tool_calls": 4.0, "invalid_submissions": 1.0}
    assert v_run.data.tags["passed"] == "True" and v_run.data.metrics["coefficient_estimates"] < 1e-3
    assert {a.path for a in tracker.client.list_artifacts(v_run.info.run_id)} >= {"code", "output", "step.log",
                                                                                     "validation.json"}

    bindings = {k: s[k] for k in ("spec_version", "baseline_id", "candidate_hash", "validation_id")}
    stale = bindings | {"validation_id": "validation-0000000000"}
    assert cp.post(f"/api/projects/{pid}/approve", json=stale).status_code == 409
    r = cp.post(f"/api/projects/{pid}/approve", json=bindings)
    assert r.status_code == 200 and (r.json()["state"]["step"], r.json()["state"]["status"]) == (6, "working")
    [d] = cp.get(f"/api/projects/{pid}").json()["decisions"]
    assert d["decision"] == "approved" and not d["void"]
    assert tracker.client.get_run(v_run.info.run_id).data.tags["decision"] == "approved"
    assert tracker.client.get_run(t_run.info.run_id).data.tags["decision"] == "approved"


def test_a_failing_candidate_goes_to_repair_and_a_rejected_one_carries_the_comment(cp, llm):
    eid = configure(cp, llm)
    pid = translate(cp, llm, [submit(DRIFTED), text("Done.")], eid)
    s = cp.get(f"/api/projects/{pid}").json()["state"]
    assert (s["step"], s["status"], s["validation_passed"]) == (5, "awaiting_ds", False), s
    assert s["detail"] == "no model is assigned to the diagnostician role; assign one in settings"
    v = cp.get(f"/api/projects/{pid}/evidence/{s['validation_id']}").json()["body"]
    assert not v["passed"] and "coefficient estimates" in {c["name"] for c in v["checks"] if not c["passed"]}
    bindings = {k: s[k] for k in ("spec_version", "baseline_id", "candidate_hash", "validation_id")}
    assert cp.post(f"/api/projects/{pid}/approve", json=bindings).status_code == 409

    pid = translate(cp, llm, [submit(POISSON_PY), text("Done.")], eid)
    s = cp.get(f"/api/projects/{pid}").json()["state"]
    bindings = {k: s[k] for k in ("spec_version", "baseline_id", "candidate_hash", "validation_id")}
    assert cp.post(f"/api/projects/{pid}/reject", json={"bindings": bindings, "comment": ""}).status_code == 422
    r = cp.post(f"/api/projects/{pid}/reject", json={"bindings": bindings, "comment": "Use exposure in years."})
    s = r.json()["state"]
    assert (s["step"], s["status"], s["ds_comment"]) == (5, "working", "Use exposure in years.")
    settle(cp, pid)
    [d] = cp.get(f"/api/projects/{pid}").json()["decisions"]
    assert d["decision"] == "rejected" and d["comment"] == "Use exposure in years."
    tracker = Tracker(cp.settings.mlflow_uri)
    [v_run] = tracker.client.search_runs([tracker.experiment(pid)], "tags.kind = 'validation'")
    assert v_run.data.tags["decision"] == "rejected" and v_run.data.tags["ds_comment"] == "Use exposure in years."


def test_a_turn_without_a_candidate_fails_and_can_be_retried(cp, llm):
    pid = translate(cp, llm, [text("I could not translate this.")])
    s = cp.get(f"/api/projects/{pid}").json()["state"]
    assert (s["step"], s["reason"]) == (4, "step_failed")
    assert "without an accepted candidate: I could not translate this." in s["detail"]
    tracker = Tracker(cp.settings.mlflow_uri)
    assert [r.info.status for r in tracker.client.search_runs([tracker.experiment(pid)],
                                                              "tags.kind = 'translation'")] == ["FAILED"]

    llm.script = [submit(POISSON_PY), text("Done.")]
    assert cp.post(f"/api/projects/{pid}/retry").status_code == 200
    s = settle(cp, pid, timeout=120)
    assert (s["step"], s["reason"]) == (4, "candidate_passed")
