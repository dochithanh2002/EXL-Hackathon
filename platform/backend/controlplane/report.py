"""Step 6: the evidence the final report is built from, the Reporter's inputs and the checks on its narrative.

`gather` collects what was recorded for the approved translation: the confirmed spec, the frozen baseline, every
attempt under that spec and baseline (candidate, validation, diagnosis) and the DS's decisions. `catalog` turns it
into entries of citable facts. The Reporter writes prose only: each claim cites catalog ids, and every number in it
is a `{{id.field}}` placeholder that code fills in from the catalog. The tables after the narrative are built by
code from the same evidence, so the report's numbers never pass through the model.

The report's plots are the approved validation run's MLflow artifacts: `retrieve_plots` downloads them into the
report's own directory, which the bundle copies, so the report stays readable without the MLflow server. Each plot
links to the run it came from. Migration acceptance (parity) and business fitness (the validation plan's evaluation
and the assessment) are separate parts of the report, with separate conclusions.
"""
from __future__ import annotations

import hashlib
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
from .tracking import Tracker
from .assessment import QUESTIONS
from .evaluation import CHECKS, subgroup_prompt_results
from .evaluation import TITLES as CHECK_TITLES
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
        "evaluation": {"kind": "evaluation", "facts": _flatten("", evaluation_facts(v.get("evaluation") or {}),
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


def evaluation_facts(ev: dict) -> dict:
    """The validation plan's results as a nested dict of citable scalars: lists of records (groups) become dicts
    keyed by their label, and per-bin and per-fold detail is left to the tables and plots."""
    checks = ev.get("checks") or {}
    out: dict = {"missing_evidence": ev.get("missing_evidence", []),
                 "missing_evidence_count": len(ev.get("missing_evidence", [])),
                 "primary_metric": ev.get("primary_metric")}
    for name in CHECKS:
        c = checks.get(name) or {}
        out[name] = {"status": c.get("status", "not recorded"), "rationale": c.get("rationale"),
                     "needs": c.get("needs", []), "notes": c.get("notes", [])}
    res = {k: (checks.get(k) or {}).get("results") or {} for k in CHECKS}
    out["baseline_comparison"] |= {"sets": res["baseline_comparison"].get("sets", {})}
    out["calibration"] |= {"sets": {s: {k: {side: {f: d.get(f) for f in ("observed_expected", "slope", "intercept",
                                                                         "max_bin_gap", "coverage") if f in d}
                                            for side, d in sides.items()} for k, sides in keys.items()}
                                    for s, keys in (res["calibration"].get("sets") or {}).items()}}
    subgroups = subgroup_prompt_results(res["subgroup_performance"])
    out["subgroup_performance"] |= {"min_rows": subgroups.get("min_rows"), "columns": {
        col: {"status": c["status"], "cells_withheld": c["cells_withheld"],
              "groups": {g["group"]: g["sets"] for g in c["groups"]}}
        for col, c in subgroups.get("columns", {}).items()}}
    st = res["stability"]
    if cv := st.get("cross_validation"):
        worst = cv["coefficients"][0] if cv["coefficients"] else {}
        out["stability"] |= {"cross_validation": {"folds": len(cv["folds"]), **cv["summary"],
                                                  "terms_with_sign_changes": cv["terms_with_sign_changes"],
                                                  "most_variable_term": worst.get("term"),
                                                  "most_variable_relative_sd": worst.get("relative_sd")}}
    if tc := st.get("training_curve"):
        out["stability"] |= {"training_curve": {"metric": tc["metric"], "iterations": tc["iterations"],
                                                "final": tc["values"][-1], "r_iterations": tc.get("r_iterations")}}
    if psi := st.get("psi"):
        out["stability"] |= {"psi": {"max_psi": psi.get("max_psi"), "vs_development": psi.get("psi_vs_development")}}
    out["sensitivity"] |= {"shift_sd": res["sensitivity"].get("shift_sd"), "columns": {
        col: c if c.get("status") != "executed" else {"status": c["status"], "shift": c["shift"], **c["sets"]}
        for col, c in (res["sensitivity"].get("columns") or {}).items()}}
    out["assumptions"] |= {"items": {
        a: {"status": x["status"], "reason": x.get("reason"), "reference": x.get("reference"),
            **{k: v for k, v in (x.get("results") or {}).items() if k != "columns"}}
        for a, x in (res["assumptions"].get("assumptions") or {}).items()}}
    return out


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


# ---------------------------------------------------------------- plots
def retrieve_plots(tracker: Tracker, validation: dict, report_dir: Path, public_url: str) -> list[dict]:
    """Download the validation run's plots from MLflow into `report_dir`/plots. Returns each figure with the run it
    came from and a link to that run."""
    report_dir.mkdir(parents=True, exist_ok=True)
    plots = validation.get("plots") or []
    if not plots:
        return []
    run_id = validation["mlflow_run_id"]
    tracker.client.download_artifacts(run_id, "plots", str(report_dir))
    eid = tracker.client.get_run(run_id).info.experiment_id
    url = f"{public_url.rstrip('/')}/#/experiments/{eid}/runs/{run_id}"
    out = []
    for p in plots:
        path = report_dir / "plots" / p["name"]
        if path.is_file():
            out.append(p | {"path": f"plots/{p['name']}", "run_id": run_id, "url": url,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return out


# ---------------------------------------------------------------- Reporter
TITLES = {
    "scope": "Scope", "model_and_data_semantics": "Model and data semantics",
    "migration.parity_results": "Parity results", "migration.skipped_checks": "Skipped or unavailable checks",
    "migration.acceptance_decision": "Acceptance decision", "validation.performance": "Performance",
    **{f"validation.{c}": CHECK_TITLES[c] for c in CHECKS if c != "baseline_comparison"},
    "validation.sample_suitability": "Sample suitability", "validation.decision_performance": "Decision performance",
    "missing_evidence": "Missing evidence", "limitations": "Limitations", "business_fitness": "Business fitness"}
STATUS = {"assessed": "Assessed", "partially assessed": "Partially assessed", "not assessed": "Not assessed"}
CHECK_STATUS = {"executed": "Executed", "partially executed": "Partially executed", "not executed": "Not executed",
                "not planned": "Not planned"}
# report section -> the validation plan check and the assessment question it shows
VALIDATION = {"performance": ("baseline_comparison", None), "calibration": ("calibration", "calibration"),
              "subgroup_performance": ("subgroup_performance", None), "stability": ("stability", "stability"),
              "sensitivity": ("sensitivity", None), "assumptions": ("assumptions", None),
              "sample_suitability": (None, "sample_suitability"),
              "decision_performance": (None, "decision_performance")}


def business_fitness(evaluation: dict, assessment: dict) -> tuple[str, list[str]]:
    """The code's verdict on business fitness and what it rests on: established only when every planned check ran
    and every assessment question was assessed; then supported or not by the criteria the spec states."""
    gaps = [CHECK_TITLES[k] for k, c in (evaluation.get("checks") or {}).items()
            if c["status"] in ("partially executed", "not executed")]
    for q in QUESTIONS:
        answer = assessment.get(q) or {}
        if answer.get("status") != "assessed" or not isinstance(answer.get("meets"), bool):
            gaps.append(q.replace("_", " "))
    if gaps or not evaluation.get("checks"):
        return "not established", gaps
    unmet = [q.replace("_", " ") for q in QUESTIONS if assessment[q].get("meets") is False]
    return ("criteria not met", unmet) if unmet else ("criteria met", [])


class Reporter:
    def __init__(self, record: Record, figures: list[dict] = ()):
        self.r = record
        self.figures = list(figures)
        self.entries = catalog(record)
        meta = record.baseline["body"]["meta"]
        self.words = _words(record.spec.model_dump_json(), record.project, *(f.name for f in record.state.files),
                            *(a.candidate["body"]["module"] for a in record.attempts),
                            *meta["design_columns"], *meta.get("column_classes", {}), *_coefs(
                                record.baseline_dir / "coefficients.csv"))
        # Preserve exact names that tokenisation cannot represent (spaces, formula terms, numeric filenames).
        self.words.update(["baseline.R", record.project, *(f.name for f in record.state.files), *meta["design_columns"],
                           *meta.get("column_classes", {}), record.spec.response,
                           *[name for step in record.spec.preprocessing for name in step.columns],
                           *record.spec.validation_plan.subgroup_performance.columns,
                           *record.spec.validation_plan.sensitivity.columns])
        validation, approval, spec = record.approved.validation["id"], record.approval_id, f"spec-v{record.state.spec_version}"
        self.required = [validation, approval]
        # each section must rest on the evidence it reports
        self.section_cites = {"model_and_data_semantics": spec, "migration.parity_results": validation,
                              "migration.skipped_checks": validation, "migration.acceptance_decision": approval,
                              **{f"validation.{k}": "evaluation" if check else "assessment"
                                 for k, (check, _) in VALIDATION.items()},
                              "missing_evidence": "evaluation", "business_fitness": "evaluation"}

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
        v = a.validation["body"]
        assessment = (v.get("assessment") or {}).get("python", {})
        evaluation = v.get("evaluation") or {}
        statuses = {"assessment": {q: assessment.get(q, {}).get("status") for q in QUESTIONS},
                    "evaluation": {k: c["status"] for k, c in (evaluation.get("checks") or {}).items()}}
        verdict, basis = business_fitness(evaluation, assessment)
        return "\n".join([
            f"Project {self.r.project}: the DS accepted the migration {a.candidate['id']} and its validation "
            f"{a.validation['id']} ({self.required[1]}). Write the model validation report.", "",
            "Evidence catalog: each entry's id, kind and the facts a claim may cite as {{id.field}}:",
            json.dumps(self.entries, indent=1), "",
            "Status of each assessment question and validation plan check (computed by the platform): "
            + json.dumps(statuses), "",
            f"The platform's business fitness verdict: {verdict}" + (f" ({'; '.join(basis)})" if basis else "")
            + ". Interpret it; do not overrule it.", "",
            "Plots the platform embeds, by section: " + json.dumps(
                [{"title": f["title"], "section": f["section"]} for f in self.figures]), "",
            "The approved candidate:", "```python", a.candidate["body"]["module"].rstrip(), "```", "",
            "baseline.R, the R code the reference ran:", "```r", self.r.spec.baseline_r.rstrip(), "```", "",
            "Code adds every heading, each status and what it needs, the plots, and tables of the semantics, "
            "validation plan, checks, metrics, evaluation and assessment results, coefficients, attempts, inputs "
            "and evidence.", "",
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
        evaluation = v.get("evaluation") or {}
        checks = evaluation.get("checks") or {}
        required = [c for c in v["checks"] if c.get("required", True)]
        unavailable = [c for c in v["checks"] if c.get("status") == "unavailable"]
        verdict, basis = business_fitness(evaluation, assessment)
        sections = dict(n.sections())

        def claims(path: str) -> list[str]:
            lines = []
            for c in sections[path]:
                text = PLACEHOLDER.sub(lambda m: fmt(facts[m.group(1)][m.group(2)]), c.text)
                lines += [f"{text} [{', '.join(c.evidence_ids)}]", ""]
            return lines

        def figures(section: str) -> list[str]:
            lines = []
            for f in self.figures:
                if f["section"] == section:
                    lines += [f"![{f['title']}]({f['path']})", "",
                              f"[MLflow run `{f['run_id'][:12]}`]({f['url']}) · `{f['path']}`", ""]
            return lines

        passed = f"{sum(c['passed'] for c in required)} of {len(required)} required checks passed"
        out = [f"# {r.project}: Model Validation Report", ""]
        if s.spec_tier == "approximate":
            out += ["**Approximate migration.** The Python implementation fits this model with a different algorithm "
                    "from R; the tolerances below were agreed for it.", ""]
        out += _table(["", ""], [
            ["Statistical model", f"`{sp.r_function}` → `{sp.python_implementation}`"],
            ["Parity tier", f"{s.spec_tier}{'' if s.spec_proven else ' (no passing fixture yet)'}"],
            ["Model specification", f"v{s.spec_version}"],
            ["Migration", f"accepted {approval['at']}; {passed}"
                          + (f"; {len(unavailable)} unavailable" if unavailable else "")],
            ["Business fitness", verdict + (f": {'; '.join(basis)}" if basis else "")]]) + [""]

        out += ["## Scope", ""] + claims("scope")

        out += ["## Model and data semantics", ""] + _table(["", ""], _semantics_rows(sp)) + [""]
        out += ["### Validation plan", ""] + _table(["Check", "Planned", "Rationale"], [
            [CHECK_TITLES[k], "yes" if c.run else "no", c.rationale] for k, c in sp.validation_plan.checks()]) + [""]
        out += claims("model_and_data_semantics")

        out += ["## Migration acceptance", "", "Does the Python code reproduce the R reference? Each value is compared "
                "as |Python − R| ≤ atol + rtol·|R|; the ratio is the worst element's |Python − R| / "
                "(atol + rtol·|R|).", "", "### Parity results", ""] + figures("migration.parity_results")
        out += claims("migration.parity_results")
        out += _table(["Check", "Status", "Max \\|Δ\\|", "atol", "rtol", "Ratio to tolerance"],
                      [[c["name"] + ("" if c.get("required", True) else " (optional)"), _status(c), *_check_cells(c)]
                       for c in v["checks"]]) + [""]
        metrics = [[st, k, fmt(x["r"]), fmt(x["python"])] for st, ks in v.get("metrics", {}).items()
                   for k, x in ks.items() if (x["r"], x["python"]) != (None, None)]
        if metrics:
            out += ["#### Evaluation metrics compared", ""] + _table(["Set", "Metric", "R", "Python"], metrics) + [""]
        rc = _coefs(r.baseline_dir / "coefficients.csv")
        if rc:
            pc = _coefs(r.output_dir / "coefficients.csv")
            ct, st = sp.tolerances.coefficients, sp.tolerances.std_errors
            out += ["#### Coefficients", ""] + _table(
                ["Term", "R estimate", "Python estimate", "Ratio", "R std error", "Python std error", "Ratio"],
                [[f"`{t}`", _g(e), _g(pc[t][0]), fmt(score(e, pc[t][0], ct.atol, ct.rtol)), _g(se), _g(pc[t][1]),
                  fmt(score(se, pc[t][1], st.atol, st.rtol))] for t, (e, se) in rc.items() if t in pc]) + [""]
        out += ["### Skipped or unavailable checks", ""]
        out += [f"- {c['name']}: unavailable{'' if c.get('required', True) else ' (optional)'}"
                + (f", {c['reason']}" if c.get("reason") else "") for c in unavailable]
        out += [f"- {x}" for x in v["notes"]] + ([""] if unavailable or v["notes"] else [])
        out += claims("migration.skipped_checks")
        out += ["### Acceptance decision", "",
                f"The data scientist accepted the migration at {approval['at']}: {passed}. This acceptance attests "
                "that the Python code reproduces the R reference within the agreed tolerances. It is not a statement "
                "that the model is fit for business use, which the model validation below addresses.", ""]
        out += claims("migration.acceptance_decision")

        out += ["## Model validation", "",
                "Is the model fit for its business use? The platform ran the validation plan's checks on the R "
                "reference's and the Python candidate's predictions; the assessment questions use the Python "
                "predictions. A faithful migration can still be unfit for use, and evidence that could not be "
                "gathered is reported as such.", ""]
        for key, (check, question) in VALIDATION.items():
            out += [f"### {TITLES['validation.' + key]}", ""]
            if check:
                c = checks.get(check) or {"status": "not executed", "needs": ["the evaluation was not recorded"],
                                          "rationale": ""}
                line = f"**{CHECK_STATUS[c['status']]}.**" + (f" {_sentence(c['rationale'])}" if c.get("rationale")
                                                               else "")
                out += [line + (" Needs: " + "; ".join(c["needs"]) + "." if c.get("needs") else ""), ""]
                out += [f"- {x}" for x in c.get("notes", [])] + ([""] if c.get("notes") else [])
            if question:
                ans = assessment.get(question) or {"status": "not assessed",
                                                   "needs": ["the assessment was not recorded"]}
                verdict_q = STATUS[ans["status"]] + ("" if ans.get("meets") is None else
                                                     f": criterion {ans['criterion']} "
                                                     f"{'met' if ans['meets'] else 'not met'}")
                out += [f"Assessment: **{verdict_q}.**" + (" Needs: " + "; ".join(ans["needs"]) + "."
                                                           if ans.get("needs") else ""), ""]
            out += figures(f"validation.{key}")
            if check:
                head, rows = _evaluation_rows(check, (checks.get(check) or {}).get("results") or {})
                if rows:
                    out += _table(head, rows) + [""]
            # the evaluation's calibration table already holds the assessment's observed/expected and gaps
            results = (assessment.get(question) or {}).get("results") or {} if question else {}
            if question and question != "calibration" and (rows := _assessment_rows(question, results)):
                out += _table(rows[0], rows[1:]) + [""]
            out += claims(f"validation.{key}")

        out += ["## Missing evidence", ""]
        missing = evaluation.get("missing_evidence") or [f"Still to provide: {x}"
                                                         for x in sp.assessment.missing_information]
        out += [f"- {x}" for x in missing] + ([""] if missing else []) + claims("missing_evidence")

        out += ["## Limitations", ""]
        limits = ([f"Approximate parity tier: {sp.python_implementation} does not use R's algorithm."]
                  if s.spec_tier == "approximate" else [])
        limits += [f"Not translated: {x}" for x in sp.unsupported_items]
        limits += [f"Substituted: {x.original} → {x.proposed} ({x.rationale})" for x in sp.substitutions]
        limits += [f"Reported by the Translator: {x}" for x in a.candidate["body"].get("limitations", [])]
        limits += [f"Evaluation: {x}" for x in evaluation.get("notes", [])]
        out += [f"- {x}" for x in limits] + ([""] if limits else []) + claims("limitations")

        out += ["## Business fitness", "", f"**{verdict.capitalize()}**" + (f": {'; '.join(basis)}." if basis else ".")
                + " Business fitness is established only when every planned check ran and every assessment question "
                "was assessed against criteria the specification states; it is judged separately from the migration's "
                "acceptance.", ""] + claims("business_fitness")

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
        rows += [["", name.replace("_", " "), "", run] for name, run in (v.get("mlflow_runs") or {}).items()]
        out += ["### Evidence", ""] + _table(["Id", "Kind", "Hash", "MLflow run"], rows) + [""]
        if self.figures:
            out += ["### Figures", "", "Copied from the validation run's MLflow artifacts into `plots/`.", ""]
            out += _table(["Figure", "File", "SHA-256"],
                          [[f["title"], f"`{f['path']}`", f"`{f['sha256'][:12]}`"] for f in self.figures])
        return "\n".join(out).rstrip("\n") + "\n"


def _sentence(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] + ("" if text.endswith((".", "!", "?")) else ".")


def _range(group: dict) -> str:
    """A numeric subgroup's bounds: below its upper edge, from its lower edge, or between them."""
    if "lower" not in group:
        return ""
    lo, hi = group.get("lower"), group.get("upper")
    if lo is None:
        return f" (below {fmt(hi)})" if hi is not None else ""
    return f" ({fmt(lo)} and above)" if hi is None else f" ({fmt(lo)} to below {fmt(hi)})"


def _evaluation_rows(check: str, res: dict) -> tuple[list[str], list[list[str]]]:
    """A compact table of one evaluation check's results: (header, rows)."""
    if check == "baseline_comparison" and res.get("sets"):
        return (["Set", "Metric", "R", "Python", "Constant", "Python vs constant"],
                [[s, m.replace("_", " "), fmt(x.get("r")), fmt(x.get("python")), fmt(x.get("constant")),
                  fmt(x.get("python_vs_constant"))] for s, e in res["sets"].items() for m, x in e.items()])
    if check == "calibration" and res.get("sets"):
        rows = []
        for s, keys in res["sets"].items():
            for k, sides in keys.items():
                for side, d in sides.items():
                    if "coverage" in d:
                        rows.append([s, f"quantile {k}", side, f"coverage {fmt(d['coverage'])} vs {fmt(d['nominal'])}",
                                     "", "", ""])
                    else:
                        rows.append([s, k, side, fmt(d["observed_expected"]), fmt(d["slope"]), fmt(d["intercept"]),
                                     fmt(d["max_bin_gap"])])
        return ["Set", "Output", "Model", "Observed / expected", "Slope", "Intercept", "Largest decile gap"], rows
    if check == "subgroup_performance" and res.get("columns"):
        rows = []
        for col, c in res["columns"].items():
            if c["status"] != "executed":
                rows.append([f"`{col}`", c.get("reason", c["status"]), "", "", "", "", ""])
                continue
            for g in c["groups"]:
                for s, e in g["sets"].items():
                    py = e.get("python") or e.get("r") or {}
                    rng = _range(g)
                    rows.append([f"`{col}`", g["group"] + rng, s, fmt(e["rows"]) + (" (small)" if e["small"] else ""),
                                 fmt(py.get("observed_expected")), fmt(py.get(res["metric"])), fmt(py.get("auc"))])
        return ["Column", "Group", "Set", "Rows", "Observed / expected", res["metric"].replace("_", " "), "AUC"], rows
    if check == "stability":
        rows = []
        cv = res.get("cross_validation") or {}
        for m, x in (cv.get("summary") or {}).items():
            rows.append([f"{m.replace('_', ' ')}, {len(cv['folds'])} folds", fmt(x["mean"]), fmt(x["sd"]),
                         f"{fmt(x['min'])} – {fmt(x['max'])}", fmt(x["in_sample"])])
        for t in (cv.get("coefficients") or [])[:5]:
            rows.append([f"`{t['term']}`", fmt(t["estimate"]), fmt(t["fold_sd"]), f"relative {fmt(t['relative_sd'])}",
                         f"{t['sign_changes']} sign changes"])
        if tc := res.get("training_curve"):
            rows.append([f"{tc['metric']} by iteration", fmt(tc["values"][-1]), "",
                         f"{tc['iterations']} recorded", f"R: {fmt(tc.get('r_iterations'))} iterations"])
        return ["Quantity", "Mean / estimate", "SD across folds", "Range", "Reference"], rows
    if check == "sensitivity" and res.get("columns"):
        rows = []
        for col, c in res["columns"].items():
            if c.get("status") != "executed":
                rows.append([f"`{col}`", "", "", c.get("reason") or c.get("type") or c.get("status"), ""])
                continue
            for s, keys in c["sets"].items():
                for k, d in keys.items():
                    rows.append([f"`{col}`" + ("" if k == "value" else f", {k}"), s, fmt(c["shift"]),
                                 fmt((d.get("down") or {}).get("mean_change")),
                                 fmt((d.get("up") or {}).get("mean_change"))])
        return ["Column", "Set", "Shift", "Mean change, down", "Mean change, up"], rows
    if check == "assumptions" and res.get("assumptions"):
        rows = []
        for name, x in res["assumptions"].items():
            if x["status"] != "executed":
                rows.append([name.replace("_", " "), "", x.get("reason", x["status"])])
                continue
            for side, vals in x["results"].items():
                if side == "columns":
                    continue
                shown = vals if isinstance(vals, dict) else {side: vals}
                rows.append([name.replace("_", " "), side if isinstance(vals, dict) else "",
                             "; ".join(f"{k.replace('_', ' ')} {fmt(v) if not isinstance(v, dict) else ''}".strip()
                                       for k, v in shown.items() if not isinstance(v, (dict, list)))])
        return ["Assumption", "Model", "Results"], rows
    return [], []


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
