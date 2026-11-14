"""Steps 6 and 7 end to end: the real agent service and Pi with a scripted model. After the DS approves, the Reporter
writes a narrative that cites recorded evidence, code fills in its numbers and adds the evidence tables, and the
deliverables bundle is built and reruns against the R reference on its own."""
import hashlib
import io
import json
import shutil
import subprocess
import zipfile

import pytest
from pydantic import ValidationError

from controlplane.report import _words, known
from controlplane.schemas import canonical_hash
from controlplane.schemas.agents import Claim
from controlplane.tracking import Tracker
from tests.fake_llm import text, tool
from tests.test_analyst import agent_service, configure, cp, llm, settle  # noqa: F401
from tests.test_repair import diagnosis, repaired, start, state, turns
from tests.test_translation import POISSON_PY
from tests.test_translator import DRIFTED, submit, translate

REPAIRED = POISSON_PY.replace("RPoisson", "Poisson2")


QUESTIONS = ("sample_suitability", "stability", "calibration", "decision_performance")


def report(validation: str, decision: str, spec: str = "spec-v1", **sections) -> dict:
    """A complete narrative citing what each section must; `sections` replaces any section's claims."""
    def c(*claims):
        return [{"text": t, "evidence_ids": ids} for t, ids in claims]
    body = {
        "scope": c(("The migration covers `{{" + spec + ".r_function}}`.", [spec])),
        "model_and_data_semantics": c(("The response is {{" + spec + ".response_type}}, predicted as "
                                       "{{" + spec + ".prediction.units}}.", [spec])),
        "parity_results": c(("The validation passed.", [validation])),
        "financial_assessment": {q: c(("This question is {{assessment." + q + ".status}}.", ["assessment"]))
                                 for q in QUESTIONS},
        "skipped_checks": c(("{{" + validation + ".checks_unavailable}} checks were unavailable.", [validation])),
        "limitations": c(("The spec records the limits of the migration.", [spec])),
        "acceptance_decision": c(("The DS accepted the migration.", [decision])),
    }
    for name, claims in sections.items():
        if name in QUESTIONS:
            body["financial_assessment"][name] = c(*claims)
        else:
            body[name] = c(*claims)
    return tool("submit_report", body)


def assign_reporter(cp) -> None:
    [ep] = cp.get("/api/settings/endpoints").json()
    assert cp.put("/api/settings/roles/reporter", json={"endpoint_id": ep["id"], "model": "fake-1",
                                                        "thinking": "off"}).status_code == 200


def bindings(s: dict) -> dict:
    return {k: s[k] for k in ("spec_version", "baseline_id", "candidate_hash", "validation_id")}


def test_the_report_cites_recorded_evidence_and_the_bundle_reruns(cp, llm, envs, tmp_path):
    pid = start(cp, llm, [submit(DRIFTED), text("Done."), diagnosis(), text("Scaled offset."),
                          repaired(REPAIRED), text("Fixed.")])
    s = state(cp, pid)
    assert (s["step"], s["reason"], s["repairs"]) == (4, "candidate_passed", 1), s
    assign_reporter(cp)
    v1, v2 = [v["id"] for v in cp.get(f"/api/projects/{pid}/evidence", params={"kind": "validation"}).json()]
    b, d = s["baseline_id"], s["diagnosis_id"]
    decision = f"decision-{len(cp.get(f'/api/projects/{pid}/transitions').json()) + 1}"
    good = report(v2, decision, scope=[
        ("The R reference in `baseline.R` fit `{{spec-v1.r_function}}` with a {{spec-v1.family}} family on {{" + b
         + ".n_fit}} training rows.", ["spec-v1", b]),
        ("The first candidate failed the {{" + v1 + ".failed_checks}} check; {{" + d + ".root_cause}}", [v1, d])],
        parity_results=[
        ("The repaired candidate, with its `Poisson2` family, passed {{" + v2 + ".checks_passed}} of {{" + v2
         + ".checks_required}} required checks; the largest coefficient difference was {{" + v2
         + ".coefficient_estimates.max_abs_diff}} against an rtol of {{" + v2 + ".coefficient_estimates.rtol}}.",
         [v2])])
    llm.script = [
        report(v2, decision, parity_results=[("All 19 checks passed.", [v2])]),
        report(v2, decision, parity_results=[
            ("See {{" + v2 + ".nonexistent}}, {{spec-v1.version}}, `x9`, `InventedEstimator`, `９９９` and "
             "`Poisson2.age`.", [v2, "evidence-x"])],
            stability=[("Stable.", [v2])], acceptance_decision=[("Accepted.", ["spec-v1"])]),
        good, text("Reported.")]
    assert cp.post(f"/api/projects/{pid}/approve", json=bindings(s)).status_code == 200
    s = settle(cp, pid, timeout=120)
    assert (s["step"], s["status"]) == (7, "done"), s

    sent = llm.sent(turns(llm, "Reporter"))
    assert "numbers must be {{evidence_id.field}} placeholders" in sent
    for error in ("unknown evidence id 'evidence-x'", f"{v2} has no field 'nonexistent'",
                  "cites spec-v1, which is not in the claim's evidence_ids", "`x9` is not a known project name",
                  "`InventedEstimator` is not a known project name", "`９９９` is not a known project name",
                  "`Poisson2.age` is not a known project name",
                  "financial_assessment.stability: cite assessment, the evidence this section reports",
                  f"acceptance_decision: cite {decision}", f"the report must cite {decision}"):
        assert error in sent, error
    assert '"coefficient_estimates.rtol"' in sent and '"sample_suitability.status"' in sent and "secret-" not in sent

    rep = cp.get(f"/api/projects/{pid}/report").json()
    assert rep["id"] == s["report_id"]
    md = rep["body"]["markdown"]
    v = cp.get(f"/api/projects/{pid}/evidence/{v2}").json()["body"]
    n_checks = len(v["checks"])
    assert "{{" not in md and f"passed {n_checks} of {n_checks} required checks" in md
    assert "with a poisson family on 140 training rows. [spec-v1, " + b + "]" in md
    assert "against an rtol of 1.000e-06." in md and f"[{decision}]" in md
    assert "### Coefficients" in md and "| `areaB` |" in md
    assert "The offset is scaled by a constant." in md                            # the attempts table
    assert "| 0 |" in md and "| 1 |" in md and "accepted" in md
    # the fixed template, with parity kept apart from business fitness
    headings = [line for line in md.splitlines() if line.startswith("## ")]
    assert headings == ["## Scope", "## Model and data semantics", "## Parity results",
                        "## Financial model assessment", "## Skipped or unavailable checks", "## Limitations",
                        "## Acceptance decision", "## Appendix: reproducibility"]
    assert "### Decision performance\n\n**Not assessed.** Needs: a binary event" in md     # a count model
    assert "### Stability\n\n**Partially assessed.** Needs: a PSI threshold." in md
    assert "This question is partially assessed. [assessment]" in md
    assert "It is not a statement that the model is fit for business use" in md
    assert "| Prediction | scalar, count: claims per policy; aligned on row |" in md

    # ---- deliverables
    bundle = cp.get(f"/api/projects/{pid}/deliverables").json()
    files = bundle["manifest"]["files"]
    assert {"model.py", "report.md", "verify.py", "parity.py", "harness.py", "README.md", "environment/uv.lock",
            "environment/renv.lock", "environment/pyproject.toml", "reference/coefficients.csv"} <= files.keys()
    assert bundle["manifest"]["bindings"] == {**bindings(s)}
    assert cp.get(f"/api/projects/{pid}/deliverables/files/model.py").text == REPAIRED
    assert cp.get(f"/api/projects/{pid}/deliverables/files/report.md").text == md
    assert cp.get(f"/api/projects/{pid}/deliverables/files/manifest.json").status_code == 200
    assert cp.get(f"/api/projects/{pid}/deliverables/files/inputs/claims.csv").status_code == 404
    archive = cp.get(f"/api/projects/{pid}/deliverables/bundle.zip")
    assert hashlib.sha256(archive.content).hexdigest() == bundle["archive_sha256"]
    with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
        z.extractall(tmp_path / "unzipped")
    [root] = (tmp_path / "unzipped").iterdir()
    for name, sha in files.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    assert canonical_hash({name: files[f"environment/{name}"] for name in envs.lockfiles}) == (
        bundle["manifest"]["environment_version"])

    dataset = cp.settings.project_dir(pid) / "inputs" / "claims.csv"
    rerun = subprocess.run([envs.py(), "verify.py", dataset], cwd=root, capture_output=True, text=True)
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert "pass  coefficient estimates  max |diff|" in rerun.stdout and "FAIL" not in rerun.stdout
    assert [line.split("  ")[1] for line in rerun.stdout.splitlines() if line.startswith("pass  ")] == (
        [check["name"] for check in v["checks"]])
    other = tmp_path / "claims.csv"
    shutil.copy(dataset, other)
    other.chmod(0o644)
    other.write_text(other.read_text() + "\n")
    rerun = subprocess.run([envs.py(), "verify.py", other], cwd=root, capture_output=True, text=True)
    assert rerun.returncode == 1 and "is not the dataset the model was validated on" in rerun.stdout

    tracker = Tracker(cp.settings.mlflow_uri)
    exp = tracker.experiment(pid)
    [r_run] = tracker.client.search_runs([exp], "tags.kind = 'report'")
    [d_run] = tracker.client.search_runs([exp], "tags.kind = 'deliverables'")
    assert r_run.info.status == d_run.info.status == "FINISHED"
    assert r_run.data.metrics == {"tool_calls": 3.0, "invalid_submissions": 2.0}
    assert {a.path for a in tracker.client.list_artifacts(r_run.info.run_id)} >= {"report.md", "narrative.json"}
    assert {a.path for a in tracker.client.list_artifacts(d_run.info.run_id)} >= {"bundle", f"{bundle['job_id']}.zip"}
    assert cp.post(f"/api/projects/{pid}/retry").status_code == 409


def test_a_report_without_a_reporter_model_fails_and_can_be_retried(cp, llm):
    pid = translate(cp, llm, [submit(POISSON_PY), text("Done.")])
    s = state(cp, pid)
    assert cp.get(f"/api/projects/{pid}/report").status_code == 404
    assert cp.post(f"/api/projects/{pid}/approve", json=bindings(s)).status_code == 200
    s = settle(cp, pid)
    assert (s["step"], s["reason"]) == (6, "step_failed")
    assert s["detail"] == "no model is assigned to the reporter role; assign one in settings"

    assign_reporter(cp)
    decision = f"decision-{len(cp.get(f'/api/projects/{pid}/transitions').json()) - 1}"
    llm.script = [report(s["validation_id"], decision), text("Reported.")]
    assert cp.post(f"/api/projects/{pid}/retry").status_code == 200
    s = settle(cp, pid, timeout=120)
    assert (s["step"], s["status"]) == (7, "done"), s
    assert cp.get(f"/api/projects/{pid}/deliverables").status_code == 200


def test_claims_allow_digits_only_in_placeholders_and_code_spans():
    Claim(text="`NumberOfTime30-59DaysPastDueNotWorse` has {{validation-a.metric.test.auc.r}}.", evidence_ids=["x"])
    for bad in ("It took 3 attempts.", "Version v2 passed.", "{{validation-a}} is 1", "Error ９９９.", "Error ²."):
        with pytest.raises(ValidationError):
            Claim(text=bad, evidence_ids=["x"])
    assert json.dumps(Claim.model_json_schema())                              # still a plain JSON Schema


def test_code_spans_may_hold_names_with_digits_but_not_numbers():
    words = _words('{"columns": ["NumberOfTime30-59DaysPastDueNotWorse", "SeriousDlqin2yrs", "x"], "rtol": 1e-06}',
                   "np.log1p(df.x)", "NumberOfTime30.59DaysPastDueNotWorse", "stats::glm", "cs-training.csv")
    assert all(known(t, words) for t in ("NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime30.59DaysPastDueNotWorse",
                                        "SeriousDlqin2yrs", "log1p", "np.log1p", "stats::glm", "cs-training.csv", "x"))
    assert not any(known(t, words) for t in ("1", "06", "1e", "1.5", "x9", "９９９", "²", "InventedEstimator",
                                            "NumberOfTime30+59DaysPastDueNotWorse", "x.log1p", "x log1p",
                                            "NumberOfTime30", "59DaysPastDueNotWorse", "e-06"))
