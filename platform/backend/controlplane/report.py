"""Step 6: the evidence the final report is built from, the Reporter's inputs and the checks on its narrative.

`gather` collects what was recorded for the approved translation: the confirmed spec, the frozen baseline, every
attempt under that spec and baseline (candidate, validation, diagnosis) and the DS's decisions. `catalog` turns it
into entries of citable facts. The Reporter writes prose only: each claim cites catalog ids, and every number in it
is a `{{id.field}}` placeholder that code fills in from the catalog. The tables after the narrative are built by
code from the same evidence, so the report's numbers never pass through the model.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .diagnosis import failed_checks
from .schemas import ModelSpec, ReportNarrative, canonical_hash
from .schemas.agents import CODE_SPAN, PLACEHOLDER
from .settings import Settings
from .assessment import QUESTIONS
from .runner.parity import _coefs, score
from .workflow import Store
from .workflow.machine import Bindings, ProjectState

Fact = str | int | float | bool | None
TOKEN = re.compile(r"(?<![\w.:-])(?:[^\W\d]|\.(?=[^\W\d]))[\w.:-]*")
NUMBER = re.compile(r"[+-]?(?:\d[\d_]*(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


@dataclass
class Attempt:
    """One translation attempt; each part is an evidence row (id, kind, hash, body, created_at)."""
    candidate: dict
    validation: dict | None = None
    diagnosis: dict | None = None


@dataclass
class Record:
    project: str
    state: ProjectState
    spec: ModelSpec
    spec_hash: str
    baseline: dict
    baseline_dir: Path
    attempts: list[Attempt]
    decisions: list[dict]                     # id, decision, at, comment, bindings
    output_dir: Path                          # the approved candidate's run output
    environment_version: str
    approved: Attempt = field(init=False)
    approval_id: str = field(init=False)

    def __post_init__(self):
        b = self.state.approval
        self.approved = next(a for a in self.attempts if a.validation and a.validation["id"] == b.validation_id)
        self.approval_id = next(d["id"] for d in reversed(self.decisions)
                                if d["decision"] == "approved" and d["bindings"] == b)


def gather(store: Store, settings: Settings, pid: str) -> Record:
    """Everything recorded for the project's current, approved translation."""
    s = store.state(pid)
    if not s.approval_valid():
        raise ValueError("the project has no valid approval")
    key = (s.spec_version, s.baseline_id)
    rows = [store.evidence(pid, e["id"]) for e in store.evidence_list(pid)]
    current = [r for r in rows if (r["body"].get("spec_version"), r["body"].get("baseline_id")) == key]
    attempts = {r["id"]: Attempt(r) for r in current if r["kind"] == "candidate"}
    for r in current:
        if r["kind"] in ("validation", "diagnosis") and r["body"]["candidate_id"] in attempts:
            setattr(attempts[r["body"]["candidate_id"]], r["kind"], r)       # a later row replaces an earlier one
    decisions = []
    for t in store.transitions(pid):
        e = t["event"]
        if e["type"] in ("candidate_approved", "candidate_rejected"):
            b = Bindings.model_validate(e["bindings"])
            if (b.spec_version, b.baseline_id) == key:
                decisions.append({"id": f"decision-{t['seq']}", "decision": e["type"].removeprefix("candidate_"),
                                  "at": t["at"], "comment": e.get("comment"), "bindings": b})
    baseline = store.evidence(pid, s.baseline_id)
    spec = store.spec(pid, s.spec_version)
    validation = store.evidence(pid, s.validation_id)["body"]
    return Record(
        project=store.name(pid), state=s, spec=spec, spec_hash=canonical_hash(spec), baseline=baseline,
        baseline_dir=settings.data_dir / baseline["body"]["dir"], attempts=list(attempts.values()),
        decisions=decisions, output_dir=settings.project_dir(pid) / "runs" / validation["job_id"] / "output",
        environment_version=validation["environment_version"])


# ---------------------------------------------------------------- catalog
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _scalars(prefix: str, d: dict) -> dict[str, Fact]:
    return {f"{prefix}{k}": v for k, v in d.items() if isinstance(v, (str, int, float, bool)) or v is None}


def _flatten(prefix: str, value, out: dict[str, Fact], depth: int = 6) -> dict[str, Fact]:
    """Nested scalars as dotted fields; lists of scalars joined; lists of records (calibration bins) left to the
    tables. Path parts that are data labels (periods, classes) are slugged so placeholders can name them."""
    if isinstance(value, dict) and depth:
        for k, v in value.items():
            part = str(k) if re.fullmatch(r"[A-Za-z0-9_-]+", str(k)) else slug(str(k)) or "_"
            _flatten(f"{prefix}.{part}" if prefix else part, v, out, depth - 1)
    elif isinstance(value, list):
        if all(isinstance(x, (str, int, float, bool)) for x in value):
            out[prefix] = "; ".join(map(str, value))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value
    return out


def _validation_facts(v: dict) -> dict[str, Fact]:
    required = [c for c in v["checks"] if c.get("required", True)]
    facts: dict[str, Fact] = {
        "candidate_id": v["candidate_id"], "passed": v["passed"], "stage": v["stage"],
        "checks_total": len(v["checks"]), "checks_required": len(required),
        "checks_passed": sum(c["passed"] for c in v["checks"]),
        "checks_failed": sum(c.get("status") == "failed" for c in v["checks"]),
        "checks_unavailable": sum(c.get("status") == "unavailable" for c in v["checks"]),
        "unavailable_checks": "; ".join(f"{c['name']} ({c.get('reason') or 'unavailable'})" for c in v["checks"]
                                        if c.get("status") == "unavailable"),
        "failed_checks": "; ".join(failed_checks(v)), "notes": "; ".join(v["notes"]),
        "environment_version": v["environment_version"]}
    if v.get("error"):
        facts |= {"error_stage": v["error"]["stage"], "error_type": v["error"]["type"]}
    for c in v["checks"]:
        name = slug(c["name"])
        facts |= {name: c["value"], f"{name}.tolerance": c["tolerance"], f"{name}.status": c.get("status")}
        for k in ("atol", "rtol", "max_abs_diff"):
            if c.get(k) is not None:
                facts[f"{name}.{k}"] = c[k]
    for s, metrics in v.get("metrics", {}).items():
        for name, sides in metrics.items():
            facts |= {f"metric.{s}.{name}.r": sides["r"], f"metric.{s}.{name}.python": sides["python"]}
    return facts


def _spec_facts(sp: ModelSpec) -> dict[str, Fact]:
    facts = _scalars("", sp.model_dump(include={"target_model", "r_function", "python_implementation", "family",
                                                "link", "formula", "response", "response_type", "offset",
                                                "limitation"}))
    for name in ("event", "grouped_binomial", "weights", "prediction", "convergence"):
        if (part := getattr(sp, name)) is not None:
            _flatten(name, part.model_dump(mode="json"), facts)
    a = sp.assessment
    _flatten("assessment", a.model_dump(mode="json", exclude={"evaluation_sets"}), facts)
    facts["assessment.evaluation_sets"] = "; ".join(f"{e.set} ({e.role}): {e.description}" for e in a.evaluation_sets)
    _flatten("tolerance", sp.tolerances.model_dump(mode="json"), facts)
    _flatten("optional_metrics", sp.optional_metrics, facts)
    facts |= {"metrics": ", ".join(sp.metrics), "split_method": sp.split.method, "seed": sp.split.seed,
              "preprocessing_steps": len(sp.preprocessing), "r_packages": ", ".join(sp.r_packages),
              "unsupported_items": "; ".join(sp.unsupported_items),
              "substitutions": "; ".join(f"{x.original} -> {x.proposed}" for x in sp.substitutions)}
    return facts


def catalog(r: Record) -> dict[str, dict]:
    """Citable entries: id -> {"kind", "facts"}. Ids are evidence ids, `intake`, `spec-v<N>`, `assessment` (the
    approved candidate's financial assessment) and `decision-<seq>`."""
    sp, s, meta = r.spec, r.state, r.baseline["body"]["meta"]
    files = r.state.files
    v = r.approved.validation["body"]
    entries = {
        "intake": {"kind": "intake", "facts": {
            "project": r.project, "dataset": next(f.name for f in files if f.role == "dataset"),
            "dataset_sha256": next(f.sha256 for f in files if f.role == "dataset"),
            "data": ", ".join(f.name for f in files if f.role == "data"),
            "r_code": ", ".join(f.name for f in files if f.role == "r_code"),
            "documents": ", ".join(f.name for f in files if f.role == "document")}},
        f"spec-v{s.spec_version}": {"kind": "spec", "facts": {
            "version": s.spec_version, "hash": r.spec_hash, "tier": s.spec_tier, "proven": s.spec_proven,
            **_spec_facts(sp)}},
        r.baseline["id"]: {"kind": "baseline", "facts": {
            "hash": r.baseline["body"]["hash"], "environment_version": r.baseline["body"]["environment_version"],
            "r_version": meta["r_version"], "n_fit": meta["n_fit"], "converged": meta["converged"],
            "iterations": meta.get("iterations"), "coefficients": len(meta["design_columns"]),
            **{f"n.{k}": v for k, v in meta["n"].items()}, **_scalars("fit.", r.baseline["body"]["fit"])}},
        "assessment": {"kind": "assessment", "facts": _flatten("", (v.get("assessment") or {}).get("python", {}),
                                                               {"validation_id": r.approved.validation["id"]})},
    }
    for a in r.attempts:
        c = a.candidate["body"]
        entries[a.candidate["id"]] = {"kind": "candidate", "facts": {
            "attempt": c["attempt"], "hash": c["hash"], "llm": c["model"], "notes": "; ".join(c["notes"]),
            "limitations": "; ".join(c.get("limitations", [])), "changes": c["changes"],
            "approved": a is r.approved}}
        if a.validation:
            entries[a.validation["id"]] = {"kind": "validation", "facts": _validation_facts(a.validation["body"])}
        if a.diagnosis:
            d = a.diagnosis["body"]
            entries[a.diagnosis["id"]] = {"kind": "diagnosis", "facts": {
                "candidate_id": d["candidate_id"], "category": d.get("category"), "root_cause": d["root_cause"],
                "repair_instructions": "; ".join(d["repair_instructions"]),
                "assessment_findings": "; ".join(d.get("assessment_findings", [])),
                "spec_change_needed": d["spec_change_needed"], "proposed_spec_change": d["proposed_spec_change"],
                "limitation": d.get("limitation"), "failed_checks": "; ".join(d["failed_checks"]),
                "ds_comment": d["ds_comment"], "llm": d["model"]}}
    for d in r.decisions:
        entries[d["id"]] = {"kind": "decision", "facts": {
            "decision": "accepted" if d["decision"] == "approved" else d["decision"], "at": d["at"],
            "comment": d["comment"], "validation_id": d["bindings"].validation_id,
            "candidate_hash": d["bindings"].candidate_hash}}
    return entries


def fmt(v: Fact) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        return "0" if v == 0 else f"{v:.3e}" if abs(v) < 1e-3 or abs(v) >= 1e6 else f"{v:.6g}"
    return str(v)


def _words(*sources: str) -> set[str]:
    """Source names and their qualified components; never compose a new name from separate known parts."""
    words = set()
    for t in TOKEN.findall("\n".join(sources)):
        t = t.rstrip(".:")
        parts = re.split(r"[.:]+", t)
        # Qualified identifiers may be cited by their components, but never split filenames or R's names that
        # contain punctuation followed by digits (for example NumberOfTime30.59DaysPastDueNotWorse).
        for w in [t, *(parts if all(p.isidentifier() for p in parts) else [])]:
            if any(c.isalpha() for c in w) and not NUMBER.fullmatch(w):
                words.add(w)
    return words


def known(token: str, words: set[str]) -> bool:
    """The entire span must be a known name, even when it has no digits or contains Unicode characters."""
    return token in words and any(c.isalpha() for c in token) and not NUMBER.fullmatch(token)


# ---------------------------------------------------------------- Reporter
TITLES = {
    "scope": "Scope", "model_and_data_semantics": "Model and data semantics", "parity_results": "Parity results",
    "financial_assessment.sample_suitability": "Sample suitability", "financial_assessment.stability": "Stability",
    "financial_assessment.calibration": "Calibration",
    "financial_assessment.decision_performance": "Decision performance",
    "skipped_checks": "Skipped or unavailable checks", "limitations": "Limitations",
    "acceptance_decision": "Acceptance decision"}
STATUS = {"assessed": "Assessed", "partially assessed": "Partially assessed", "not assessed": "Not assessed"}


class Reporter:
    def __init__(self, record: Record):
        self.r = record
        self.entries = catalog(record)
        meta = record.baseline["body"]["meta"]
        self.words = _words(record.spec.model_dump_json(), record.project, *(f.name for f in record.state.files),
                            *(a.candidate["body"]["module"] for a in record.attempts),
                            *meta["design_columns"], *meta.get("column_classes", {}), *_coefs(
                                record.baseline_dir / "coefficients.csv"))
        # Preserve exact names that tokenisation cannot represent (spaces, formula terms, numeric filenames).
        self.words.update(["baseline.R", record.project, *(f.name for f in record.state.files), *meta["design_columns"],
                           *meta.get("column_classes", {}), record.spec.response,
                           *[name for step in record.spec.preprocessing for name in step.columns]])
        validation, approval, spec = record.approved.validation["id"], record.approval_id, f"spec-v{record.state.spec_version}"
        self.required = [validation, approval]
        # each section must rest on the evidence it reports
        self.section_cites = {"model_and_data_semantics": spec, "parity_results": validation,
                              "skipped_checks": validation, "acceptance_decision": approval,
                              **{f"financial_assessment.{q}": "assessment" for q in QUESTIONS}}

    def check(self, raw: dict) -> tuple[ReportNarrative | None, list[str]]:
        try:
            n = ReportNarrative.model_validate(raw)
        except ValidationError as e:
            return None, [f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors()]
        errors, cited = [], set()
        for path, claims in n.sections():
            section_ids = set()
            for j, claim in enumerate(claims):
                at = f"{path}.{j}"
                ids = set(claim.evidence_ids)
                cited |= ids
                section_ids |= ids
                errors += [f"{at}.evidence_ids: unknown evidence id {x!r}" for x in sorted(ids - self.entries.keys())]
                for m in PLACEHOLDER.finditer(claim.text):
                    eid, name = m.groups()
                    if eid not in ids:
                        errors.append(f"{at}.text: {m.group(0)} cites {eid}, which is not in the claim's evidence_ids")
                    elif eid in self.entries and name not in self.entries[eid]["facts"]:
                        errors.append(f"{at}.text: {eid} has no field {name!r}")
                errors += [f"{at}.text: {e}" for e in self._spans(claim.text)]
            need = self.section_cites.get(path)
            if need and need not in section_ids:
                errors.append(f"{path}: cite {need}, the evidence this section reports")
        errors += [f"the report must cite {x}" for x in self.required if x not in cited]
        return (None if errors else n), errors

    def _spans(self, text: str) -> list[str]:
        without_placeholders = PLACEHOLDER.sub("", text)
        errors = (["malformed placeholder; write {{evidence_id.field}}"]
                  if "{{" in without_placeholders or "}}" in without_placeholders else [])
        for span in CODE_SPAN.findall(text):
            name = span[1:-1]
            if not PLACEHOLDER.fullmatch(name) and not known(name, self.words):
                errors.append(f"{span} is not a known project name; use a complete known name or write numbers as "
                              "{{evidence_id.field}} placeholders")
        if "`" in CODE_SPAN.sub("", text):
            errors.append("use single backticks around one known project name or one placeholder")
        return errors

    def kickoff(self) -> str:
        a = self.r.approved
        statuses = {q: (a.validation["body"].get("assessment") or {}).get("python", {}).get(q, {}).get("status")
                    for q in QUESTIONS}
        return "\n".join([
            f"Project {self.r.project}: the DS accepted the migration {a.candidate['id']} and its validation "
            f"{a.validation['id']} ({self.required[1]}). Write the migration validation report.", "",
            "Evidence catalog: each entry's id, kind and the facts a claim may cite as {{id.field}}:",
            json.dumps(self.entries, indent=1), "",
            "Financial assessment status by question (computed by the platform): " + json.dumps(statuses), "",
            "The approved candidate:", "```python", a.candidate["body"]["module"].rstrip(), "```", "",
            "baseline.R, the R code the reference ran:", "```r", self.r.spec.baseline_r.rstrip(), "```", "",
            "Code adds every heading, each assessment question's status and what it needs, and tables of the "
            "semantics, checks, evaluation metrics, assessment results, coefficients, attempts, inputs and "
            "evidence.", "",
            "Cite " + ", ".join(dict.fromkeys(self.section_cites.values())) + " in their sections, then submit the "
            "report."])

    # ------------------------------------------------------------ rendering
    def render(self, n: ReportNarrative) -> str:
        r, s, a = self.r, self.r.state, self.r.approved
        v = a.validation["body"]
        sp = r.spec
        facts = {eid: e["facts"] for eid, e in self.entries.items()}
        approval = next(d for d in r.decisions if d["id"] == r.approval_id)
        assessment = (v.get("assessment") or {}).get("python", {})
        required = [c for c in v["checks"] if c.get("required", True)]
        unavailable = [c for c in v["checks"] if c.get("status") == "unavailable"]

        def claims(path: str) -> list[str]:
            lines = []
            for c in dict(n.sections())[path]:
                text = PLACEHOLDER.sub(lambda m: fmt(facts[m.group(1)][m.group(2)]), c.text)
                lines += [f"{text} [{', '.join(c.evidence_ids)}]", ""]
            return lines

        out = [f"# {r.project}: Migration Validation Report", ""]
        if s.spec_tier == "approximate":
            out += ["**Approximate migration.** The Python implementation fits this model with a different algorithm "
                    "from R; the tolerances below were agreed for it.", ""]
        out += _table(["", ""], [
            ["Statistical model", f"`{sp.r_function}` → `{sp.python_implementation}`"],
            ["Parity tier", f"{s.spec_tier}{'' if s.spec_proven else ' (no passing fixture yet)'}"],
            ["Model specification", f"v{s.spec_version}"],
            ["Parity checks", f"{sum(c['passed'] for c in required)} of {len(required)} required checks passed"
                              + (f"; {len(unavailable)} unavailable" if unavailable else "")],
            ["Migration accepted", approval["at"]]]) + [""]

        out += ["## Scope", ""] + claims("scope")

        out += ["## Model and data semantics", ""] + _table(["", ""], _semantics_rows(sp)) + [""]
        out += claims("model_and_data_semantics")

        out += ["## Parity results", "", "Each value is compared as |Python − R| ≤ atol + rtol·|R|; the ratio column "
                "is the worst element's |Python − R| / (atol + rtol·|R|).", ""] + claims("parity_results")
        out += _table(["Check", "Status", "Max \\|Δ\\|", "atol", "rtol", "Ratio to tolerance"],
                      [[c["name"] + ("" if c.get("required", True) else " (optional)"), _status(c), *_check_cells(c)]
                       for c in v["checks"]]) + [""]
        metrics = [[st, k, fmt(x["r"]), fmt(x["python"])] for st, ks in v.get("metrics", {}).items()
                   for k, x in ks.items() if (x["r"], x["python"]) != (None, None)]
        if metrics:
            out += ["### Evaluation metrics", ""] + _table(["Set", "Metric", "R", "Python"], metrics) + [""]
        rc = _coefs(r.baseline_dir / "coefficients.csv")
        if rc:
            pc = _coefs(r.output_dir / "coefficients.csv")
            ct, st = sp.tolerances.coefficients, sp.tolerances.std_errors
            out += ["### Coefficients", ""] + _table(
                ["Term", "R estimate", "Python estimate", "Ratio", "R std error", "Python std error", "Ratio"],
                [[f"`{t}`", _g(e), _g(pc[t][0]), fmt(score(e, pc[t][0], ct.atol, ct.rtol)), _g(se), _g(pc[t][1]),
                  fmt(score(se, pc[t][1], st.atol, st.rtol))] for t, (e, se) in rc.items() if t in pc]) + [""]

        out += ["## Financial model assessment", "",
                "This section asks whether the migrated model suits its business use, from summaries the platform "
                "computed on the Python predictions. It is separate from migration parity: a faithful migration can "
                "still be unfit for use, and a question without the data, periods, thresholds or business "
                "assumptions it needs is reported as not assessed.", ""]
        for q in QUESTIONS:
            ans = assessment.get(q) or {"status": "not assessed", "needs": ["the assessment was not recorded"]}
            title = TITLES[f"financial_assessment.{q}"]
            verdict = STATUS[ans["status"]] + ("" if ans.get("meets") is None else
                                               f": criterion {ans['criterion']} {'met' if ans['meets'] else 'not met'}")
            out += [f"### {title}", "", f"**{verdict}.**" + (" Needs: " + "; ".join(ans["needs"]) + "."
                                                             if ans.get("needs") else ""), ""]
            rows = _assessment_rows(q, ans.get("results") or {})
            if rows:
                out += _table(rows[0], rows[1:]) + [""]
            out += claims(f"financial_assessment.{q}")

        out += ["## Skipped or unavailable checks", ""]
        out += [f"- {c['name']}: unavailable{'' if c.get('required', True) else ' (optional)'}"
                + (f", {c['reason']}" if c.get("reason") else "") for c in unavailable]
        out += [f"- {x}" for x in v["notes"]] + ([""] if unavailable or v["notes"] else [])
        out += claims("skipped_checks")

        out += ["## Limitations", ""]
        limits = ([f"Approximate parity tier: {sp.python_implementation} does not use R's algorithm."]
                  if s.spec_tier == "approximate" else [])
        limits += [f"Not translated: {x}" for x in sp.unsupported_items]
        limits += [f"Substituted: {x.original} → {x.proposed} ({x.rationale})" for x in sp.substitutions]
        limits += [f"Reported by the Translator: {x}" for x in a.candidate["body"].get("limitations", [])]
        limits += [f"Still to provide: {x}" for x in sp.assessment.missing_information]
        out += [f"- {x}" for x in limits] + ([""] if limits else []) + claims("limitations")

        out += ["## Acceptance decision", "",
                f"The data scientist accepted the migration at {approval['at']}: {sum(c['passed'] for c in required)} "
                f"of {len(required)} required parity checks passed. This acceptance attests that the Python code "
                "reproduces the R reference within the agreed tolerances. It is not a statement that the model is fit "
                "for business use; the financial model assessment above is: "
                + "; ".join(f"{TITLES['financial_assessment.' + q].lower()} "
                            f"{STATUS[(assessment.get(q) or {}).get('status', 'not assessed')].lower()}"
                            for q in QUESTIONS) + ".", ""] + claims("acceptance_decision")

        out += ["## Appendix: reproducibility", ""]
        out += _table(["", ""], [
            ["Spec hash", f"`{r.spec_hash[:12]}`"],
            ["R reference", f"{r.baseline['id']}, {r.baseline['body']['meta']['r_version']}"],
            ["Candidate", f"{a.candidate['id']} `{a.candidate['body']['hash'][:12]}`, code generated by "
                          f"{a.candidate['body']['model']}"],
            ["Validation", a.validation["id"]],
            ["Environment", f"`{r.environment_version[:12]}`"]]) + [""]
        by_candidate = {d["bindings"].candidate_hash: d for d in r.decisions}
        out += ["### Attempts", ""] + _table(["Attempt", "Candidate", "Validation", "Failed checks", "Diagnosis",
                                              "DS"], [
            [str(x.candidate["body"]["attempt"]), x.candidate["id"],
             x.validation["id"] if x.validation else "did not run",
             "; ".join(failed_checks(x.validation["body"])) if x.validation else "",
             x.diagnosis["body"]["root_cause"] if x.diagnosis else "",
             _decision(by_candidate.get(x.candidate["body"]["hash"]))] for x in r.attempts]) + [""]
        out += ["### Inputs", ""] + _table(["File", "Role", "SHA-256"],
                                           [[f.name, f.role, f"`{f.sha256}`"] for f in s.files]) + [""]
        rows = [[r.baseline["id"], "R reference", f"`{r.baseline['hash'][:12]}`", r.baseline["body"]["mlflow_run_id"]]]
        for x in r.attempts:
            rows += [[e["id"], e["kind"], f"`{e['hash'][:12]}`", e["body"]["mlflow_run_id"]]
                     for e in (x.candidate, x.validation, x.diagnosis) if e]
        out += ["### Evidence", ""] + _table(["Id", "Kind", "Hash", "MLflow run"], rows)
        return "\n".join(out) + "\n"


def _check_cells(c: dict) -> list[str]:
    """Max |Δ|, atol, rtol and the ratio for a tolerance check; the observed and expected value otherwise."""
    if c.get("status") == "unavailable":
        return ["", "", "", c.get("reason") or "unavailable"]
    if c.get("kind") == "tolerance":
        return [fmt(c.get("max_abs_diff")), fmt(c.get("atol")), fmt(c.get("rtol")), fmt(c["value"])]
    if c.get("kind") == "flag":
        return ["", "", "", "yes" if c["value"] else "no"]
    at_least = "at least " if c.get("kind") == "minimum" else ""
    return ["", "", "", f"{fmt(c['value'])} (expected {at_least}{fmt(c['tolerance'])})"]


def _status(c: dict) -> str:
    return {"passed": "pass", "failed": "FAIL", "unavailable": "unavailable"}.get(c.get("status"), "FAIL")


def _semantics_rows(sp: ModelSpec) -> list[list[str]]:
    p, a = sp.prediction, sp.assessment
    rows = [["Target", sp.target_model], ["Response", f"`{sp.response}` ({sp.response_type})"]]
    rows += [["R convergence check", sp.convergence.r_check or sp.convergence.mode],
             ["Convergence rationale", sp.convergence.rationale or "Standard estimator status"],
             ["Optional metrics", "; ".join(f"{s}: {', '.join(ms)}" for s, ms in sp.optional_metrics.items()) or "None"]]
    if sp.event:
        rows.append(["Event", f"{sp.event.meaning}, coded `{sp.event.coding}`; otherwise {sp.event.complement}"])
    if g := sp.grouped_binomial:
        rows.append(["Grouped binomial", f"{g.form}: trials {g.trials}" + (f", successes {g.successes}" if g.successes
                                                                            else "")])
    if w := sp.weights:
        rows.append(["Weights", f"{w.expression}: {w.type} weights, normalization {w.normalization}"])
    if sp.offset:
        rows.append(["Offset", sp.offset])
    rows.append(["Prediction", f"{p.kind}, {p.quantity}: {p.units}" + (f" per {p.exposure}" if p.exposure else "")
                 + (f"; keys {', '.join(p.keys)}" if p.keys else "") + f"; aligned on {', '.join(p.alignment_keys)}"])
    rows += [["Business use", a.business_use], ["Intended population", a.intended_population],
             ["Sample", a.sample_population],
             ["Evaluation sets", "; ".join(f"{e.set} ({e.role.replace('_', ' ')}): {e.description}"
                                           for e in a.evaluation_sets)]]
    if a.decision:
        rows.append(["Decision", f"{a.decision.action} at or above {fmt(a.decision.threshold)}"])
    return rows


def _assessment_rows(q: str, res: dict) -> list[list[str]]:
    """A compact table of one question's results; the first row is the header."""
    if q == "sample_suitability" and res.get("sets"):
        rows = [["Set", "Role", "Rows", "Valid", "Events", "Event rate / mean"]]
        for s, e in res["sets"].items():
            rows.append([s, e["role"].replace("_", " "), fmt(e["rows"]), fmt(e["valid"]), fmt(e.get("events")),
                         fmt(e.get("event_rate", e.get("mean_observed")))])
        rows.append(["per parameter", res.get("per_parameter_basis", ""), fmt(res.get("parameters")), "", "",
                     fmt(res.get("per_parameter"))])
        return rows
    if q == "stability" and (res.get("psi_vs_development") or res.get("by_period")):
        rows = [["Compared", "PSI", "Mean prediction", "Mean observed"]]
        rows += [[f"{s} vs development", fmt(v), "", ""] for s, v in res.get("psi_vs_development", {}).items()]
        rows += [[f"period {k}", fmt(v["psi_vs_first"]), fmt(v["mean_prediction"]), fmt(v["mean_observed"])]
                 for k, v in res.get("by_period", {}).items()]
        return rows
    if q == "calibration" and res.get("sets"):
        rows = [["Set", "Output", "Observed / expected", "Largest decile gap"]]
        for s, e in res["sets"].items():
            if "coverage" in e:
                rows += [[s, f"quantile {k}", f"coverage {fmt(c['observed'])} vs {fmt(c['nominal'])}", ""]
                         for k, c in e["coverage"].items()]
            else:
                rows += [[s, k, fmt(c["observed_expected"]), fmt(c["max_bin_gap"])] for k, c in e.items()]
        return rows
    if q == "decision_performance" and res.get("discrimination"):
        rows = [["Set", "AUC", "Gini", "KS", "Flagged share", "Event rate flagged", "Events captured"]]
        for s, d in res["discrimination"].items():
            t = res.get("at_threshold", {}).get(s, {})
            rows.append([s, fmt(d["auc"]), fmt(d["gini"]), fmt(d["ks"]), fmt(t.get("flagged_share")),
                         fmt(t.get("event_rate_flagged")), fmt(t.get("events_captured"))])
        return rows
    return []


def _decision(d: dict | None) -> str:
    if d is None:
        return ""
    label = "accepted" if d["decision"] == "approved" else d["decision"]
    return f"{label}: {d['comment']}" if d["comment"] else label


def _g(x: float) -> str:
    return "NA" if math.isnan(x) else f"{x:.10g}"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    def cell(x: str) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + [
        "| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
