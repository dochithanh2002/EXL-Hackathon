"""Step 6: the model validation report, rendered by code from recorded evidence, with the Reporter's interpretation.

`gather` collects what was recorded for the approved translation: the confirmed spec, the frozen baseline, every
attempt under that spec and baseline (candidate, validation, diagnosis) and the DS's decisions. `catalog` turns it
into entries of citable facts. Code renders every factual part of the report: scope, semantics, the validation
plan, parity results, the acceptance decision, each validation check with its status, criterion outcome, tables
and plots, the missing evidence, the business fitness verdict and the appendix. The Reporter writes only three
lists of claims (key findings, limitations, recommendations); each claim cites catalog ids, and every number in it
is a `{{id.field}}` placeholder that code fills in, so the report's numbers never pass through the model.

`copy_plots` copies the approved validation's plots from its work directory into the report's own directory,
which the bundle copies, so the report reads on its own. Migration acceptance (parity) and business fitness (the
model validation checks) are separate parts of the report, with separate conclusions.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .diagnosis import failed_checks
from .evaluation import CHECKS, TITLES as CHECK_TITLES, subgroup_prompt_results, unstated
from .runner.parity import _coefs, score
from .schemas import ModelSpec, ReportNarrative, canonical_hash
from .schemas.agents import CODE_SPAN, PLACEHOLDER, quote
from .settings import Settings
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
    """Citable entries: id -> {"kind", "facts"}. Ids are evidence ids, `intake`, `spec-v<N>`, `evaluation` (the
    approved validation's model validation checks and verdict) and `decision-<seq>`."""
    sp, s, meta = r.spec, r.state, r.baseline["body"]["meta"]
    files = r.state.files
    v = r.approved.validation["body"]
    verdict, basis = _verdict(v)
    evaluation = (v.get("evaluation") or {}) | {"business_fitness": {"verdict": verdict, "basis": basis}}
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
        "evaluation": {"kind": "evaluation", "facts": _flatten("", evaluation_facts(evaluation),
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


def _stability_psi(results: dict) -> dict:
    """Read pre-consolidation validation records without changing their evidence or approval bindings."""
    psi = results.get("psi") or {}
    if "psi_vs_development" in psi:
        return {"python": {"vs_development": psi["psi_vs_development"], "by_period": psi["by_period"],
                           "max_psi": psi["max_psi"]}}
    return psi


def evaluation_facts(ev: dict) -> dict:
    """The model validation as a nested dict of citable scalars: each check's status, criterion and headline
    results (the Python candidate's, and R's where the comparison matters); lists of records (groups) become dicts
    keyed by their label, and per-bin and per-fold detail is left to the tables and plots."""
    checks = ev.get("checks") or {}
    fitness = ev.get("business_fitness") or {}
    out: dict = {"missing_evidence": ev.get("missing_evidence", []),
                 "missing_evidence_count": len(ev.get("missing_evidence", [])),
                 "primary_metric": ev.get("primary_metric"),
                 "business_fitness": {"verdict": fitness.get("verdict", "not recorded"),
                                      "basis": fitness.get("basis", [])}}
    for name in CHECKS:
        c = checks.get(name) or {}
        out[name] = {"status": c.get("status", "not recorded"), "rationale": c.get("rationale"),
                     "needs": c.get("needs", []), "notes": c.get("notes", [])}
        if cr := c.get("criterion"):
            out[name]["criterion"] = {k: cr.get(k) for k in ("rule", "outcome", "limit", "value", "judged_on")}
    res = {k: (checks.get(k) or {}).get("results") or {} for k in CHECKS}
    out["baseline_comparison"] |= {"sets": res["baseline_comparison"].get("sets", {})}
    out["calibration"] |= {"max_deviation": res["calibration"].get("max_deviation"), "sets": {
        s: {k: {side: {f: d.get(f) for f in ("observed_expected", "slope", "intercept", "max_bin_gap", "coverage")
                       if f in d} for side, d in sides.items()} for k, sides in keys.items()}
        for s, keys in (res["calibration"].get("sets") or {}).items()}}
    subgroups = subgroup_prompt_results(res["subgroup_performance"])
    out["subgroup_performance"] |= {"min_rows": subgroups.get("min_rows"), "columns": {
        col: {"status": c["status"], "cells_withheld": c["cells_withheld"],
              "groups": {g["group"]: g["sets"] for g in c["groups"]}}
        for col, c in subgroups.get("columns", {}).items()}}
    fair = subgroup_prompt_results(res["fairness"])
    out["fairness"] |= {k: res["fairness"].get(k) for k in ("impact_ratio", "max_group_deviation", "threshold",
                                                            "favourable_outcome") if k in res["fairness"]}
    if fair:
        out["fairness"] |= {"min_rows": fair.get("min_rows"), "columns": {
            col: {"status": c["status"], "cells_withheld": c["cells_withheld"], "impact_ratio": c.get("impact_ratio"),
                  "groups": {g["group"]: g["sets"] for g in c["groups"]}}
            for col, c in fair.get("columns", {}).items()}}
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
    if psi := _stability_psi(st):
        out["stability"] |= {"psi": {side: {"max_psi": x["max_psi"], "vs_development": x["vs_development"],
                                            "by_period": {k: p["psi_vs_first"] for k, p in x["by_period"].items()}}
                                     for side, x in psi.items()}}
    out["sensitivity"] |= {"shift_sd": res["sensitivity"].get("shift_sd"), "columns": {
        col: c if c.get("status") != "executed" else {"status": c["status"], "shift": c["shift"], **c["sets"]}
        for col, c in (res["sensitivity"].get("columns") or {}).items()}}
    out["assumptions"] |= {"items": {
        a: {"status": x["status"], "reason": x.get("reason"), "reference": x.get("reference"),
            **{k: v for k, v in (x.get("results") or {}).items() if k != "columns"}}
        for a, x in (res["assumptions"].get("assumptions") or {}).items()}}
    sample = res["sample_suitability"]
    out["sample_suitability"] |= {k: sample.get(k) for k in ("per_parameter", "per_parameter_basis", "parameters",
                                                             "fitted_rows", "excluded_from_fit") if k in sample}
    if sample.get("sets"):
        out["sample_suitability"]["sets"] = {s: {"role": e["role"], **{
            k: v for k, v in (e.get("python") or e.get("r") or {}).items() if k != "class_counts"}}
            for s, e in sample["sets"].items()}
    dec = res["decision_performance"]
    if dec.get("discrimination"):
        out["decision_performance"] |= {"threshold": dec.get("threshold"), "action": dec.get("action"), "sets": {
            s: {**(sides.get("python") or {}), **((dec.get("at_threshold") or {}).get(s, {}).get("python") or {})}
            for s, sides in dec["discrimination"].items()}}
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
def copy_plots(validation: dict, run_dir: Path, report_dir: Path) -> list[dict]:
    """Copy the validation's plots from its work directory `run_dir` into `report_dir`/plots. Returns each figure
    with its path in the report directory and its hash."""
    report_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in validation.get("plots") or []:
        source = run_dir / "plots" / p["name"]
        if source.is_file():
            path = report_dir / "plots" / p["name"]
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, path)
            out.append(p | {"path": f"plots/{p['name']}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return out


# ---------------------------------------------------------------- Reporter
SECTIONS = {"key_findings": "Key findings", "limitations": "Limitations", "recommendations": "Recommendations"}


def _checks(v: dict) -> dict:
    return (v.get("evaluation") or {}).get("checks") or {}


def _verdict(v: dict) -> tuple[str, list[str]]:
    f = (v.get("evaluation") or {}).get("business_fitness") or {}
    if not f and v.get("assessment"):
        return "not established", ["this validation predates the current fitness criteria; rerun model validation "
                                   "to establish the verdict"]
    return f.get("verdict", "not established"), f.get("basis", ["the model validation was not recorded"])


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
                           *record.spec.validation_plan.sensitivity.columns,
                           *(record.spec.validation_plan.fairness.columns if record.spec.validation_plan.fairness
                             else [])])
        # the key findings rest on the accepted parity checks, the acceptance and the model validation
        self.required = [record.approved.validation["id"], record.approval_id, "evaluation"]

    def check(self, raw: dict) -> tuple[ReportNarrative | None, list[str]]:
        try:
            n = ReportNarrative.model_validate(raw)
        except ValidationError as e:
            return None, [f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors()]
        errors, findings = [], set()
        for path, claims in n.sections():
            for j, claim in enumerate(claims):
                at = f"{path}.{j}"
                ids = set(claim.evidence_ids)
                if path == "key_findings":
                    findings |= ids
                errors += [f"{at}.evidence_ids: unknown evidence id {x!r}" for x in sorted(ids - self.entries.keys())]
                for m in PLACEHOLDER.finditer(claim.text):
                    eid, name = m.groups()
                    if eid not in ids:
                        errors.append(f"{at}.text: {m.group(0)} cites {eid}, which is not in the claim's evidence_ids")
                    elif eid in self.entries and name not in self.entries[eid]["facts"]:
                        errors.append(f"{at}.text: {eid} has no field {name!r}")
                errors += [f"{at}.text: {e}" for e in self._spans(claim.text)]
        errors += [f"key_findings: cite {x}" for x in self.required if x not in findings]
        return (None if errors else n), errors

    def _spans(self, text: str) -> list[str]:
        without_placeholders = PLACEHOLDER.sub("", text)
        errors = (["malformed placeholder; write {{evidence_id.field}}"]
                  if "{{" in without_placeholders or "}}" in without_placeholders else [])
        for m in CODE_SPAN.finditer(text):
            name = m.group(0)[1:-1]
            if not PLACEHOLDER.fullmatch(name) and not known(name, self.words):
                errors.append(f"{m.group(0)} in {quote(text, m.start(), m.end())} is not a known project name; "
                              "use a complete known name or write numbers as {{evidence_id.field}} placeholders")
        if "`" in CODE_SPAN.sub("", text):
            errors.append("use single backticks around one known project name or one placeholder")
        return errors

    def kickoff(self) -> str:
        a = self.r.approved
        v = a.validation["body"]
        statuses = {k: {"status": c["status"], **({"criterion": c["criterion"]["rule"],
                                                   "outcome": c["criterion"]["outcome"]} if c.get("criterion") else {})}
                    for k, c in _checks(v).items()}
        verdict, basis = _verdict(v)
        c = a.candidate["body"]
        return "\n".join([
            f"Project {self.r.project}: the DS accepted the migration {a.candidate['id']} and its validation "
            f"{a.validation['id']} ({self.r.approval_id}). Write the key findings, limitations and recommendations "
            "of the model validation report.", "",
            "Evidence catalog: each entry's id, kind and the facts a claim may cite as {{id.field}}:",
            json.dumps(self.entries, separators=(",", ":")), "",
            "Model validation checks (status: how far each ran; outcome: the specification's criterion on the "
            "Python candidate): " + json.dumps(statuses, separators=(",", ":")), "",
            f"The platform's business fitness verdict: {verdict}" + (f" ({'; '.join(basis)})" if basis else "")
            + ". Interpret it; do not overrule it.", "",
            "The Translator's notes on the approved candidate: " + json.dumps(c["notes"]),
            "Limitations it reported: " + json.dumps(c.get("limitations", [])), "",
            "The platform renders the scope, semantics, validation plan, parity results, acceptance decision, every "
            "check with its status, criterion, tables and plots, the missing evidence and the verdict.", "",
            f"Cite {', '.join(self.required)} in key_findings, then submit the report."])

    # ------------------------------------------------------------ rendering
    def render(self, n: ReportNarrative) -> str:
        r, s, a = self.r, self.r.state, self.r.approved
        v = a.validation["body"]
        sp = r.spec
        facts = {eid: e["facts"] for eid, e in self.entries.items()}
        approval = next(d for d in r.decisions if d["id"] == r.approval_id)
        evaluation = v.get("evaluation") or {}
        checks = evaluation.get("checks") or {}
        required = [c for c in v["checks"] if c.get("required", True)]
        unavailable = [c for c in v["checks"] if c.get("status") == "unavailable"]
        verdict, basis = _verdict(v)
        sections = dict(n.sections())
        meta = r.baseline["body"]["meta"]

        def claims(path: str) -> list[str]:
            lines = []
            for c in sections[path]:
                text = PLACEHOLDER.sub(lambda m: fmt(facts[m.group(1)][m.group(2)]), c.text)
                lines.append(f"- {text} [{', '.join(c.evidence_ids)}]")
            return lines + ([""] if lines else [])

        def figures(section: str) -> list[str]:
            return [line for f in self.figures if f["section"] == section for line in (f"![{f['title']}]({f['path']})",
                                                                                         "")]

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
            ["Business fitness", verdict]]) + [""]

        # ---- scope
        files = s.files
        by_role = {role: [f.name for f in files if f.role == role] for role in ("dataset", "data", "r_code",
                                                                                "document")}
        dataset = next(f for f in files if f.role == "dataset")
        out += ["## Scope", "",
                f"{_sentence(sp.target_model)} The R code in {_names(by_role['r_code'])} fits `{sp.r_function}`; "
                f"the migration reproduces it in Python with `{sp.python_implementation}`.",
                "",
                f"The model is fitted on {_names([dataset.name])} (SHA-256 `{dataset.sha256[:12]}`)"
                + (f", with {_names(by_role['data'])}" if by_role["data"] else "")
                + (f"; documents: {_names(by_role['document'])}" if by_role["document"] else "") + ". "
                f"It predicts `{sp.response}`, a {sp.response_type.replace('_', ' ')} response, as "
                f"{sp.prediction.units}, for {sp.assessment.business_use}.", ""]
        out += _table(["Set", "Role", "Rows", "Description"], [
            [e.set, e.role.replace("_", " "), fmt((meta.get("n") or {}).get(e.set)), e.description]
            for e in sp.assessment.evaluation_sets]) + [""]

        out += ["## Key findings", ""] + claims("key_findings")

        # ---- semantics and plan
        out += ["## Model and data semantics", ""] + _table(["", ""], _semantics_rows(sp)) + [""]
        plan = dict(sp.validation_plan.checks())
        out += ["### Validation plan", ""] + _table(["Check", "Planned", "Rationale"], [
            [CHECK_TITLES[k], "yes" if plan[k].run else "no", plan[k].rationale] if k in plan else
            [CHECK_TITLES[k], "no", (checks.get(k) or {}).get("rationale", "")] if k == "fairness" else
            [CHECK_TITLES[k], "not applicable" if (checks.get(k) or {}).get("status") == "not applicable" else "always",
             (checks.get(k) or {}).get("rationale", "")] for k in CHECKS]) + [""]

        # ---- migration acceptance
        out += ["## Migration acceptance", "", "Does the Python code reproduce the R reference? Each value is compared "
                "as |Python − R| ≤ atol + rtol·|R|; the ratio is the worst element's |Python − R| / "
                "(atol + rtol·|R|).", "", "### Parity results", "",
                f"{_sentence(passed)}" + (f" {len(unavailable)} could not be computed." if unavailable else ""), ""]
        out += figures("migration.parity_results")
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
        skipped = [f"- {c['name']}: unavailable{'' if c.get('required', True) else ' (optional)'}"
                   + (f", {c['reason']}" if c.get("reason") else "") for c in unavailable]
        skipped += [f"- {x}" for x in v["notes"]]
        out += (skipped or ["Every parity check was computed."]) + [""]
        out += ["### Acceptance decision", "",
                f"The data scientist accepted the migration at {approval['at']}: {passed}."
                + (f" Their comment: “{approval['comment']}”" if approval["comment"] else "")
                + " This acceptance attests that the Python code reproduces the R reference within the agreed "
                "tolerances. It is not a statement that the model is fit for business use, which the model "
                "validation below addresses.", ""]

        # ---- model validation
        out += ["## Model validation", "",
                "Is the model fit for its business use? The platform ran each check on the R reference's and the "
                "Python candidate's predictions and judged the specification's criteria on the Python candidate. "
                "Each check states how far it ran and, separately, what its criterion concluded.", ""]
        for key in CHECKS:
            c = checks.get(key) or {"status": "not executed", "needs": ["the model validation was not recorded"],
                                    "rationale": "", "notes": [], "results": {}, "criterion": None}
            line = f"**{c['status'].capitalize()}.**" + (f" {_sentence(c['rationale'])}" if c.get("rationale") else "")
            out += [f"### {CHECK_TITLES[key]}", "",
                    line + (" Needs " + "; ".join(c["needs"]) + "." if c.get("needs") else ""), ""]
            if cr := c.get("criterion"):
                out += [_criterion(key, cr), ""]
            out += [f"- {x}" for x in c.get("notes", [])] + ([""] if c.get("notes") else [])
            out += figures(f"validation.{key}")
            for head, rows in _evaluation_tables(key, c.get("results") or {}):
                if rows:
                    out += _table(head, rows) + [""]

        out += ["## Missing evidence", ""]
        missing = evaluation.get("missing_evidence") or [f"Still to provide: {x}"
                                                         for x in sp.assessment.missing_information]
        out += ([f"- {x}" for x in missing] or ["None."]) + [""]

        out += ["## Business fitness", "", f"**{verdict.capitalize()}.** The verdict reads the criterion outcomes on "
                "the Python candidate and the checks' execution: a criterion not met decides it; otherwise every "
                "stated criterion must be met and every check that applies must have run. It is judged separately "
                "from the migration's acceptance.", ""]
        out += [f"- {x}" for x in basis] + ([""] if basis else [])

        out += ["## Limitations", ""]
        limits = ([f"Approximate parity tier: {sp.python_implementation} does not use R's algorithm."]
                  if s.spec_tier == "approximate" else [])
        limits += [f"Not translated: {x}" for x in sp.unsupported_items]
        limits += [f"Substituted: {x.original} → {x.proposed} ({x.rationale})" for x in sp.substitutions]
        limits += [f"Reported by the Translator: {x}" for x in a.candidate["body"].get("limitations", [])]
        limits += [f"Evaluation: {x}" for x in evaluation.get("notes", [])]
        out += [f"- {x}" for x in limits] + ([""] if limits else [])
        out += claims("limitations") or ["The Reporter identified no further limitations.", ""]

        out += ["## Recommendations", ""]
        out += claims("recommendations") or ["The Reporter made no recommendations.", ""]

        # ---- appendix
        out += ["## Appendix: reproducibility", ""]
        out += _table(["", ""], [
            ["Spec hash", f"`{r.spec_hash[:12]}`"],
            ["R reference", f"{r.baseline['id']}, {meta['r_version']}"],
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
        rows = [[r.baseline["id"], "R reference", f"`{r.baseline['hash'][:12]}`"]]
        for x in r.attempts:
            rows += [[e["id"], e["kind"], f"`{e['hash'][:12]}`"] for e in (x.candidate, x.validation, x.diagnosis) if e]
        out += ["### Evidence", ""] + _table(["Id", "Kind", "Hash"], rows) + [""]
        if self.figures:
            out += ["### Figures", ""] + _table(["Figure", "File", "SHA-256"], [
                [f["title"], f"`{f['path']}`", f"`{f['sha256'][:12]}`"] for f in self.figures])
        return "\n".join(out).rstrip("\n") + "\n"


def _names(names: list[str]) -> str:
    return ", ".join(f"`{x}`" for x in names) or "the uploaded files"


def _sentence(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] + ("" if text.endswith((".", "!", "?")) else ".")


def _criterion(check: str, cr: dict) -> str:
    """One sentence: the criterion, its outcome and the statistic it was judged on."""
    on = f" on {', '.join(cr['judged_on'])}" if cr.get("judged_on") else ""
    values = "; ".join(f"{k.replace('_', ' ')} {fmt(cr['value'].get(k))} against {fmt(lim)}"
                       for k, lim in cr["limit"].items())
    if cr["outcome"] in ("met", "not met"):
        return f"Criterion `{cr['rule']}`: **{cr['outcome']}**{on} ({values})."
    if cr["outcome"] == "not judged":
        return f"Criterion `{cr['rule']}`: **not judged**; its statistic could not be computed{on}."
    computed = "; ".join(f"{k.replace('_', ' ')} {fmt(x)}" for k, x in cr["value"].items() if x is not None)
    return (f"Criterion: **no criterion**. The specification states no {', '.join(unstated(check, cr))}"
            + (f"; the rule it does state, `{cr['rule']}`, gives {values}" if cr["rule"] else "")
            + (f". Computed{on}: {computed}." if computed else "."))


def _range(group: dict) -> str:
    """A numeric subgroup's bounds: below its upper edge, from its lower edge, or between them."""
    if "lower" not in group:
        return ""
    lo, hi = group.get("lower"), group.get("upper")
    if lo is None:
        return f" (below {fmt(hi)})" if hi is not None else ""
    return f" ({fmt(lo)} and above)" if hi is None else f" ({fmt(lo)} to below {fmt(hi)})"


def _evaluation_tables(check: str, res: dict) -> list[tuple[list[str], list[list[str]]]]:
    """Compact tables of one check's results: (header, rows) each."""
    if check == "baseline_comparison" and res.get("sets"):
        return [(["Set", "Metric", "R", "Python", "Constant", "Python vs constant"],
                 [[s, m.replace("_", " "), fmt(x.get("r")), fmt(x.get("python")), fmt(x.get("constant")),
                   fmt(x.get("python_vs_constant"))] for s, e in res["sets"].items() for m, x in e.items()])]
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
        return [(["Set", "Output", "Model", "Observed / expected", "Slope", "Intercept", "Largest decile gap"], rows)]
    if check == "subgroup_performance" and res.get("columns"):
        rows = []
        for col, c in res["columns"].items():
            if c["status"] != "executed":
                rows.append([f"`{col}`", c.get("reason", c["status"]), "", "", "", "", ""])
                continue
            for g in c["groups"]:
                for s, e in g["sets"].items():
                    py = e.get("python") or e.get("r") or {}
                    rows.append([f"`{col}`", g["group"] + _range(g), s,
                                 fmt(e["rows"]) + (" (small)" if e["small"] else ""),
                                 fmt(py.get("observed_expected")), fmt(py.get(res["metric"])), fmt(py.get("auc"))])
        return [(["Column", "Group", "Set", "Rows", "Observed / expected", res["metric"].replace("_", " "), "AUC"],
                 rows)]
    if check == "fairness" and res.get("columns"):
        rows = []
        for col, c in res["columns"].items():
            if c["status"] != "executed":
                rows.append([f"`{col}`", c.get("reason", c["status"]), "", "", "", ""])
                continue
            for g in c["groups"]:
                for s, e in g["sets"].items():
                    py = e.get("python") or e.get("r") or {}
                    rows.append([f"`{col}`", g["group"] + _range(g), s,
                                 fmt(e["rows"]) + (" (small)" if e["small"] else ""),
                                 fmt(py.get("observed_expected")), fmt(py.get("favourable_share"))])
        ratios = [[f"`{col}`", s, fmt(x)] for col, c in res["columns"].items()
                  for s, x in (c.get("impact_ratio") or {}).items()]
        return [(["Column", "Group", "Set", "Rows", "Observed / expected", "Favourable share"], rows),
                (["Column", "Set", "Impact ratio"], ratios)]
    if check == "stability":
        out = []
        if psi := _stability_psi(res):
            sides = list(psi)
            compared = list(dict.fromkeys([f"{s} vs development" for x in psi.values() for s in x["vs_development"]]
                                          + [f"period {k}" for x in psi.values() for k in x["by_period"]]))

            def value(x: dict, label: str):
                if label.startswith("period "):
                    return (x["by_period"].get(label.removeprefix("period ")) or {}).get("psi_vs_first")
                return x["vs_development"].get(label.removesuffix(" vs development"))
            out.append((["Compared", *[f"PSI, {SIDE[s]}" for s in sides]],
                        [[label, *[fmt(value(psi[s], label)) for s in sides]] for label in compared]))
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
        return out + [(["Quantity", "Mean / estimate", "SD across folds", "Range", "Reference"], rows)]
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
        return [(["Column", "Set", "Shift", "Mean change, down", "Mean change, up"], rows)]
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
        return [(["Assumption", "Model", "Results"], rows)]
    if check == "sample_suitability" and res.get("sets"):
        rows = []
        for s, e in res["sets"].items():
            x = e.get("python") or e.get("r") or {}
            rows.append([s, e["role"].replace("_", " "), fmt(e["units"]), fmt(x.get("valid")), fmt(x.get("events")),
                         fmt(x.get("event_rate", x.get("mean_observed")))])
        rows.append(["per parameter", res.get("per_parameter_basis", ""), fmt(res.get("parameters")), "", "",
                     fmt(res.get("per_parameter"))])
        return [(["Set", "Role", "Rows", "Valid", "Events", "Event rate / mean"], rows)]
    if check == "decision_performance" and res.get("discrimination"):
        at = res.get("at_threshold") or {}
        rows = [[s, SIDE[side], fmt(d["auc"]), fmt(d["gini"]), fmt(d["ks"]),
                 *[fmt(at.get(s, {}).get(side, {}).get(k)) for k in ("flagged_share", "event_rate_flagged",
                                                                     "events_captured")]]
                for s, sides in res["discrimination"].items() for side, d in sides.items()]
        return [(["Set", "Model", "AUC", "Gini", "KS", "Flagged share", "Event rate flagged", "Events captured"],
                 rows)]
    return []


SIDE = {"r": "R", "python": "Python"}


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
    rows = [["Target", sp.target_model], ["Response", f"`{sp.response}` ({sp.response_type})"],
            ["Formula", f"`{sp.formula}`"]]
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
             ["Sample", a.sample_population]]
    if a.decision:
        rows.append(["Decision", f"{a.decision.action} at or above {fmt(a.decision.threshold)}"])
    stated = {k: v for k, v in a.criteria.model_dump().items() if v is not None}
    rows.append(["Criteria", "; ".join(f"{k} {fmt(v)}" for k, v in stated.items()) or "None stated"])
    return rows


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
