"""Steps 6 and 7 end to end: the real agent service and Pi with a scripted model. After the DS approves, code renders
the report from the recorded evidence, the Reporter adds key findings, limitations and recommendations that cite it
(code fills in their numbers), and the deliverables bundle is built and reruns against the R reference on its own."""
import hashlib
import io
import json
import shutil
import subprocess
import zipfile

import pytest
from pydantic import ValidationError

from controlplane.report import _evaluation_tables, _verdict, _words, evaluation_facts, known
from controlplane.schemas import canonical_hash
from controlplane.schemas.agents import Claim
from controlplane.tracking import Tracker
from tests.fake_llm import text, tool
from tests.test_analyst import agent_service, configure, cp, llm, settle, synced  # noqa: F401
from tests.test_repair import diagnosis, repaired, start, state, turns
from tests.test_translation import POISSON_PY
from tests.test_translator import DRIFTED, submit, translate

REPAIRED = POISSON_PY.replace("RPoisson", "Poisson2")


def report(validation: str, decision: str, **lists) -> dict:
    """A narrative whose key findings cite what they must; `lists` replaces any list's claims, by its name."""
    def c(*claims):
        return [{"text": t, "evidence_ids": ids} for t, ids in claims]
    body = {
        "key_findings": c(("The DS accepted the migration: the validation passed.", [validation, decision]),
                          ("Business fitness is {{evaluation.business_fitness.verdict}}.", ["evaluation"])),
        "limitations": [],
        "recommendations": c(("State {{evaluation.missing_evidence_count}} missing items.", ["evaluation"])),
    }
    return tool("submit_report", body | {name: c(*claims) for name, claims in lists.items()})


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
    good = report(v2, decision, key_findings=[
        ("The R reference in `baseline.R` fit `{{spec-v1.r_function}}` with a {{spec-v1.family}} family on {{" + b
         + ".n_fit}} training rows.", ["spec-v1", b]),
        ("The repaired candidate, with its `Poisson2` family, passed {{" + v2 + ".checks_passed}} of {{" + v2
         + ".checks_required}} required checks; the largest coefficient difference was {{" + v2
         + ".coefficient_estimates.max_abs_diff}} against an rtol of {{" + v2 + ".coefficient_estimates.rtol}}, "
         "and the DS accepted it.", [v2, decision]),
        ("The Python model's test deviance per unit exposure was {{evaluation.baseline_comparison.sets.test."
         "mean_deviance.python}} against {{evaluation.baseline_comparison.sets.test.mean_deviance.constant}} for "
         "the constant comparator; business fitness is {{evaluation.business_fitness.verdict}}.", ["evaluation"])],
        limitations=[("The first candidate failed the {{" + v1 + ".failed_checks}} check; {{" + d
                      + ".root_cause}}", [v1, d])])
    llm.script = [
        report(v2, decision, recommendations=[("Recheck all 19 checks in 2019 for the team.", [v2])]),
        report(v2, decision, key_findings=[
            ("See {{" + v2 + ".nonexistent}}, {{spec-v1.version}}, `x9`, `InventedEstimator`, `９９９` and "
             "`Poisson2.age`.", [v2, "evidence-x"])]),
        good, text("Reported.")]
    assert cp.post(f"/api/projects/{pid}/approve", json=bindings(s)).status_code == 200
    s = settle(cp, pid, timeout=120)
    assert (s["step"], s["status"]) == (7, "done"), s

    sent = llm.sent(turns(llm, "Reporter"))
    assert 'recommendations.0.text: Value error, literal number in "Recheck all 19 checks in"…' in sent
    for error in ("unknown evidence id 'evidence-x'", f"{v2} has no field 'nonexistent'",
                  "cites spec-v1, which is not in the claim's evidence_ids",
                  '`x9` in …"`x9`,"… is not a known project name',
                  "`InventedEstimator` in", "`９９９` in", "`Poisson2.age` in",
                  f"key_findings: cite {decision}", "key_findings: cite evaluation"):
        assert error in sent, error
    assert '"coefficient_estimates.rtol"' in sent and '"sample_suitability.status"' in sent and "secret-" not in sent
    assert '"baseline_comparison.sets.test.mean_deviance.python"' in sent and "business fitness verdict" in sent
    assert "```python" not in sent and "baseline.R, the R code" not in sent       # no candidate module or R code

    rep = cp.get(f"/api/projects/{pid}/report").json()
    assert rep["id"] == s["report_id"]
    md = rep["body"]["markdown"]
    v = cp.get(f"/api/projects/{pid}/evidence/{v2}").json()["body"]
    n_checks = len(v["checks"])
    assert "{{" not in md and f"passed {n_checks} of {n_checks} required checks" in md
    assert "with a poisson family on 140 training rows. [spec-v1, " + b + "]" in md
    assert "against an rtol of 1.000e-06, and the DS accepted it." in md and f"{decision}]" in md
    assert "#### Coefficients" in md and "| `areaB` |" in md
    assert "The offset is scaled by a constant." in md                            # the attempts table
    assert "| 0 |" in md and "| 1 |" in md and "accepted" in md
    # migration acceptance and business fitness are separate parts with separate conclusions; the Reporter's lists
    # sit after the scope and at the end
    headings = [line for line in md.splitlines() if line.startswith("## ")]
    assert headings == ["## Scope", "## Key findings", "## Model and data semantics", "## Migration acceptance",
                        "## Model validation", "## Missing evidence", "## Business fitness", "## Limitations",
                        "## Recommendations", "## Appendix: reproducibility"]
    assert [line for line in md.splitlines() if line.startswith("### ")][:13] == [
        "### Validation plan", "### Parity results", "### Skipped or unavailable checks", "### Acceptance decision",
        "### Baseline comparison", "### Calibration", "### Subgroup performance", "### Stability",
        "### Sensitivity", "### Model assumptions", "### Sample suitability", "### Decision performance",
        "### Attempts"]
    assert "### Decision performance\n\n**Not applicable.** Defined for a binary event" in md     # a count model
    assert ("Criterion: **no criterion**. The specification states no min_events_per_parameter. Computed on train: "
            "per parameter") in md
    assert "| Sample suitability | always |" in md and "| Decision performance | not applicable |" in md
    assert "The DS accepted the migration: the validation passed." not in md      # replaced by the good findings
    assert "Business fitness is not established. [evaluation]" not in md
    assert "business fitness is not established. [evaluation]" in md
    assert "The Reporter identified" not in md and "The Reporter made no recommendations" not in md
    assert "It is not a statement that the model is fit for business use" in md
    assert "| Business fitness | not established |" in md and "**Not established.**" in md
    assert "- Sample suitability: no criterion for min_events_per_parameter" in md
    assert "| Prediction | scalar, count: claims per policy; aligned on row |" in md
    assert "| test | mean deviance |" in md and "for the constant comparator" in md
    assert "- Sensitivity: sensitivity of area (the column is not numeric)" in md          # missing evidence
    assert "| Subgroup performance | yes | claims frequency by area and age band |" in md
    assert "| train | development |" in md and "`claims.csv`" in md                     # scope
    assert "mlflow" not in md.lower()

    # ---- plots: copied from the validation's work directory, served from the report's copies
    figures = rep["body"]["figures"]
    assert len(figures) >= 8 and "mlflow" not in md.lower()
    for f in figures:
        assert f"![{f['title']}]({f['path']})" in md
        assert f["sha256"] == hashlib.sha256((cp.settings.data_dir / v["dir"] / f["path"]).read_bytes()).hexdigest()
        served = cp.get(f"/api/projects/{pid}/report/files/{f['path']}")
        assert served.status_code == 200 and served.headers["content-type"] == "image/svg+xml"
        assert hashlib.sha256(served.content).hexdigest() == f["sha256"]
    assert md.index("plots/parity.svg") < md.index("## Model validation") < md.index("plots/calibration-test.svg")
    assert cp.get(f"/api/projects/{pid}/report/files/..%2F..%2Fdb.sqlite").status_code == 404

    # ---- deliverables
    bundle = cp.get(f"/api/projects/{pid}/deliverables").json()
    files = bundle["manifest"]["files"]
    assert {"model.py", "report.md", "verify.py", "parity.py", "harness.py", "README.md", "environment/uv.lock",
            "environment/renv.lock", "environment/pyproject.toml", "reference/coefficients.csv",
            "reference/segments.csv", *(f["path"] for f in figures)} <= files.keys()
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
        assert b"mlflow" not in (root / name).read_bytes().lower() or name.startswith("environment/"), name
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

    synced(cp, pid)
    tracker = Tracker(cp.settings.mlflow_uri)
    exp = tracker.experiment(pid)
    [r_run] = tracker.client.search_runs([exp], "tags.kind = 'report'")
    [d_run] = tracker.client.search_runs([exp], "tags.kind = 'deliverables'")
    assert r_run.info.status == d_run.info.status == "FINISHED"
    assert r_run.data.metrics == {"tool_calls": 3.0, "invalid_submissions": 2.0}
    assert {a.path for a in tracker.client.list_artifacts(r_run.info.run_id)} >= {"report", "narrative.json"}
    assert {a.path for a in tracker.client.list_artifacts(r_run.info.run_id, "report/plots")} == {
        f"report/{f['path']}" for f in figures}
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
    llm.script = [report(s["validation_id"], decision, recommendations=[]), text("Reported.")]
    assert cp.post(f"/api/projects/{pid}/retry").status_code == 200
    s = settle(cp, pid, timeout=120)
    assert (s["step"], s["status"]) == (7, "done"), s
    md = cp.get(f"/api/projects/{pid}/report").json()["body"]["markdown"]
    assert "## Recommendations\n\nThe Reporter made no recommendations." in md
    assert "The Reporter identified no further limitations." in md
    assert cp.get(f"/api/projects/{pid}/deliverables").status_code == 200


def test_claims_allow_digits_only_in_placeholders_and_code_spans():
    Claim(text="`NumberOfTime30-59DaysPastDueNotWorse` has {{validation-a.metric.test.auc.r}}.", evidence_ids=["x"])
    for bad in ("It took 3 attempts.", "Version v2 passed.", "{{validation-a}} is 1", "Error ９９９.", "Error ²."):
        with pytest.raises(ValidationError):
            Claim(text=bad, evidence_ids=["x"])
    assert json.dumps(Claim.model_json_schema())                              # still a plain JSON Schema


def test_report_reads_legacy_stability_results_without_changing_the_evidence():
    psi = {"psi_vs_development": {"test": 0.15}, "max_psi": 0.15, "by_period": {
        "later": {"psi_vs_first": 0.12, "units": 50}}}
    results = {"psi": psi}
    evaluation = {"checks": {"stability": {"status": "executed", "results": results}}}
    validation = {"evaluation": evaluation, "assessment": {"python": {"stability": {"results": psi}}}}
    digest = canonical_hash(validation)

    assert evaluation_facts(evaluation)["stability"]["psi"]["python"]["vs_development"] == {"test": 0.15}
    assert _evaluation_tables("stability", results)[0] == (
        ["Compared", "PSI, Python"], [["test vs development", "0.15"], ["period later", "0.12"]])
    verdict, basis = _verdict(validation)
    assert verdict == "not established" and "predates" in basis[0]
    assert canonical_hash(validation) == digest


def test_code_spans_may_hold_names_with_digits_but_not_numbers():
    words = _words('{"columns": ["NumberOfTime30-59DaysPastDueNotWorse", "SeriousDlqin2yrs", "x"], "rtol": 1e-06}',
                   "np.log1p(df.x)", "NumberOfTime30.59DaysPastDueNotWorse", "stats::glm", "cs-training.csv")
    assert all(known(t, words) for t in ("NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime30.59DaysPastDueNotWorse",
                                        "SeriousDlqin2yrs", "log1p", "np.log1p", "stats::glm", "cs-training.csv", "x"))
    assert not any(known(t, words) for t in ("1", "06", "1e", "1.5", "x9", "９９９", "²", "InventedEstimator",
                                            "NumberOfTime30+59DaysPastDueNotWorse", "x.log1p", "x log1p",
                                            "NumberOfTime30", "59DaysPastDueNotWorse", "e-06"))
