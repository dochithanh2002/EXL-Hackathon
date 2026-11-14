"""Step 5 end to end: the real agent service and Pi with a scripted model. A failed or rejected candidate is
diagnosed, repaired and validated again, up to the retry limit; a diagnosis that needs a spec change goes to the DS,
who can return to the Analyst."""
import csv

import pytest

from controlplane.tracking import Tracker
from tests.fake_llm import text, tool
from tests.test_analyst import agent_service, configure, cp, llm, settle  # noqa: F401
from tests.test_baseline import poisson_spec
from tests.test_translation import POISSON_PY
from tests.test_translator import DRIFTED, submit, translate

DRIFTED_AGAIN = POISSON_PY.replace('offset=np.log(train["exposure"])', 'offset=np.log(train["exposure"]) * 0.98')
LEAKY = POISSON_PY.replace("df.index = pd.RangeIndex(1, len(df) + 1)",
                           'df.index = pd.RangeIndex(1, len(df) + 1)\n    df["policy"] = df["policy"].astype(float)')


def diagnosis(**change) -> dict:
    return tool("submit_diagnosis", {
        "root_cause": "The offset is scaled by a constant.", "evidence": ["every coefficient moved"],
        "repair_instructions": ["Pass offset=np.log(exposure) without scaling."], "spec_change_needed": False,
        "category": "specification" if change.get("spec_change_needed") else "migration_defect"} | change)


def repaired(module: str, changes: str = "Removed the offset scaling.") -> dict:
    return tool("submit_candidate", {"module": module, "notes": [], "changes": changes})


def turns(llm, role: str) -> list[dict]:
    """The model requests made by the role's runs, in order (one per model call)."""
    return [r for r in llm.requests if f"You are the {role}" in llm.sent([r])]


def numbers(path) -> list[str]:
    with open(path, newline="") as f:
        return [r[k] for r in csv.DictReader(f) for k in ("estimate", "std_error")]


def start(cp, llm, script: list) -> str:
    eid = configure(cp, llm)
    assert cp.put("/api/settings/roles/diagnostician", json={"endpoint_id": eid, "model": "fake-1",
                                                             "thinking": "off"}).status_code == 200
    return translate(cp, llm, script, eid)


def state(cp, pid) -> dict:
    return cp.get(f"/api/projects/{pid}").json()["state"]


def body(cp, pid, eid) -> dict:
    return cp.get(f"/api/projects/{pid}/evidence/{eid}").json()["body"]


def test_a_failed_candidate_is_diagnosed_and_the_repair_passes(cp, llm):
    pid = start(cp, llm, [
        submit(DRIFTED), text("Done."),
        tool("dataset_profile", {}), diagnosis(repair_instructions=[]), diagnosis(), text("The offset is scaled."),
        submit(DRIFTED), repaired(POISSON_PY, changes=""), repaired(POISSON_PY), text("Fixed the offset.")])
    s = state(cp, pid)
    assert (s["step"], s["status"], s["reason"], s["repairs"]) == (4, "awaiting_ds", "candidate_passed", 1), s

    [first_val, _] = cp.get(f"/api/projects/{pid}/evidence", params={"kind": "validation"}).json()
    v = body(cp, pid, first_val["id"])
    data = cp.settings.data_dir
    r_numbers = numbers(data / body(cp, pid, s["baseline_id"])["dir"] / "coefficients.csv")
    py_numbers = numbers(cp.settings.project_dir(pid) / "runs" / v["job_id"] / "output" / "coefficients.csv")

    sent = llm.sent(turns(llm, "Diagnostician"))
    assert "* 0.99" in llm.sent(turns(llm, "Diagnostician")[:1])
    assert '"coefficient estimates"' in sent and '"term": "areaB"' in sent
    assert '"reference_levels"' in sent and '"sample_suitability"' in sent        # design and assessment summaries
    assert "repair_instructions must not be empty" in sent
    assert not any(n in sent for n in r_numbers + py_numbers) and "secret-" not in sent

    repair = llm.sent(turns(llm, "Translator")[-3:])
    assert "repair attempt 1 of 3" in repair and "Pass offset=np.log(exposure) without scaling." in repair
    assert "identical to the previous candidate" in repair and "changes: describe what changed" in repair
    assert not any(n in repair for n in r_numbers)

    [d] = cp.get(f"/api/projects/{pid}/evidence", params={"kind": "diagnosis"}).json()
    assert d["id"] == s["diagnosis_id"]
    db = body(cp, pid, d["id"])
    assert "coefficient estimates" in db["failed_checks"] and db["category"] == "migration_defect"
    assert db["candidate_hash"] != s["candidate_hash"]
    cands = cp.get(f"/api/projects/{pid}/evidence", params={"kind": "candidate"}).json()
    c = body(cp, pid, cands[-1]["id"])
    assert (c["attempt"], c["diagnosis_id"], c["changes"]) == (1, d["id"], "Removed the offset scaling.")

    tracker = Tracker(cp.settings.mlflow_uri)
    [run] = tracker.client.search_runs([tracker.experiment(pid)], "tags.kind = 'diagnosis'")
    assert run.info.status == "FINISHED" and run.data.tags["spec_change_needed"] == "False"
    assert run.data.metrics == {"tool_calls": 3.0, "invalid_submissions": 1.0}
    assert run.data.params["validation_id"] == first_val["id"]


def test_a_rejected_candidate_is_diagnosed_with_the_ds_comment(cp, llm):
    pid = start(cp, llm, [submit(POISSON_PY), text("Done.")])
    s = state(cp, pid)
    bindings = {k: s[k] for k in ("spec_version", "baseline_id", "candidate_hash", "validation_id")}
    llm.script = [diagnosis(root_cause="The DS wants the exposure column named.",
                            repair_instructions=["Name the exposure column in a constant."]), text("Noted."),
                  repaired(POISSON_PY.replace('RESPONSE = "claims"', 'RESPONSE = "claims"\nEXPOSURE = "exposure"'),
                           "Named the exposure column.")]
    cp.post(f"/api/projects/{pid}/reject", json={"bindings": bindings, "comment": "Name the exposure column."})
    s = settle(cp, pid, timeout=120)
    assert (s["step"], s["reason"], s["repairs"]) == (4, "candidate_passed", 1), s
    d_sent = llm.sent(turns(llm, "Diagnostician"))
    assert "the DS rejected the candidate with this comment:\nName the exposure column." in d_sent
    assert "The DS rejected it with this comment:\nName the exposure column." in llm.sent(
        turns(llm, "Translator")[-1:])


def test_a_runtime_error_reaches_the_diagnostician_without_data_and_a_spec_change_goes_back_to_the_analyst(cp, llm):
    pid = start(cp, llm, [
        submit(LEAKY), text("Done."),
        diagnosis(spec_change_needed=True, proposed_spec_change="Loosen the coefficient rtol to 1e-5.", repair_instructions=[]),
        text("This needs a spec change.")])
    s = state(cp, pid)
    assert (s["step"], s["status"], s["reason"]) == (5, "awaiting_ds", "spec_change"), s
    sent = llm.sent(turns(llm, "Diagnostician"))
    assert "could not convert string to float: <redacted>" in sent and '"function": "prepare"' in sent
    assert "secret-" not in sent
    note = cp.get(f"/api/projects/{pid}/chat").json()[-1]
    assert note["author"] == "platform" and "Proposed change: Loosen the coefficient rtol to 1e-5." in note["text"]
    assert "the spec needs to change before the migration can match R" in note["text"]

    v2 = poisson_spec(tolerances={"coefficients": {"atol": 1e-8, "rtol": 1e-5}}).model_dump(mode="json")
    llm.script = [tool("submit_model_spec", v2), text("Loosened the coefficient rtol.")]
    assert cp.post(f"/api/projects/{pid}/chat", json={"text": "Take the proposal."}).status_code == 202
    s = settle(cp, pid)
    assert (s["step"], s["reason"], s["drafted_version"]) == (2, "spec_drafted", 2)
    analyst = llm.sent(turns(llm, "Analyst")[-2:-1])
    assert "Proposed change: Loosen the coefficient rtol to 1e-5." in analyst and "Take the proposal." in analyst

    baseline_id = s["baseline_id"]
    llm.script = [submit(POISSON_PY), text("Done.")]
    assert cp.post(f"/api/projects/{pid}/specs/2/confirm", json={"parity_risk_accepted": True}).status_code == 200
    s = settle(cp, pid, timeout=120)
    assert (s["step"], s["reason"], s["spec_version"], s["repairs"]) == (4, "candidate_passed", 2, 0), s
    assert s["baseline_id"] == baseline_id                                  # tolerances only: baseline reused
    assert "repair attempt" not in llm.sent(turns(llm, "Translator")[-2:])


@pytest.fixture
def limit_one(monkeypatch):
    monkeypatch.setenv("CONTROLPLANE_RETRY_LIMIT", "1")


def test_the_retry_limit_escalates_to_the_ds_who_can_raise_it(limit_one, cp, llm):
    pid = start(cp, llm, [submit(DRIFTED), text("Done."), diagnosis(), text("Scaled offset."),
                          repaired(DRIFTED_AGAIN), text("Done.")])
    s = state(cp, pid)
    assert (s["step"], s["status"], s["reason"], s["repairs"]) == (5, "awaiting_ds", "retry_limit", 1), s
    assert len(cp.get(f"/api/projects/{pid}/evidence", params={"kind": "diagnosis"}).json()) == 1   # none at the limit
    assert cp.post(f"/api/projects/{pid}/resume", json={}).status_code == 409

    llm.script = [diagnosis(root_cause="The offset is still scaled."), text("Still scaled."),
                  repaired(POISSON_PY), text("Done.")]
    r = cp.post(f"/api/projects/{pid}/resume", json={"additional_attempts": 1})
    assert r.status_code == 200 and r.json()["state"]["retry_limit"] == 2
    s = settle(cp, pid, timeout=120)
    assert (s["step"], s["reason"], s["repairs"]) == (4, "candidate_passed", 2), s
    second = llm.sent(turns(llm, "Diagnostician")[-2:])
    assert "Earlier attempts for this spec" in second and "The offset is scaled by a constant." in second
    assert "Repairs used: 1 of 2." in second
    assert cp.post(f"/api/projects/{pid}/resume", json={}).status_code == 409     # nothing to resume now
