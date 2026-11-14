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
from .runner.parity import _coefs, _rel
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


def _validation_facts(v: dict) -> dict[str, Fact]:
    facts: dict[str, Fact] = {"candidate_id": v["candidate_id"], "passed": v["passed"], "stage": v["stage"],
                              "checks_total": len(v["checks"]), "checks_passed": sum(c["passed"] for c in v["checks"]),
                              "failed_checks": "; ".join(failed_checks(v)), "notes": "; ".join(v["notes"]),
                              "environment_version": v["environment_version"]}
    if v.get("error"):
        facts |= {"error_stage": v["error"]["stage"], "error_type": v["error"]["type"]}
    for c in v["checks"]:
        facts |= {slug(c["name"]): c["value"], f"{slug(c['name'])}.tolerance": c["tolerance"]}
    for s, kpis in v["kpis"].items():
        for name, sides in kpis.items():
            facts |= {f"kpi.{s}.{name}.r": sides["r"], f"kpi.{s}.{name}.python": sides["python"]}
    return facts


def catalog(r: Record) -> dict[str, dict]:
    """Citable entries: id -> {"kind", "facts"}. Ids are evidence ids, `intake`, `spec-v<N>` and `decision-<seq>`."""
    sp, s, meta = r.spec, r.state, r.baseline["body"]["meta"]
    files = r.state.files
    entries = {
        "intake": {"kind": "intake", "facts": {
            "project": r.project, "dataset": next(f.name for f in files if f.role == "dataset"),
            "dataset_sha256": next(f.sha256 for f in files if f.role == "dataset"),
            "data": ", ".join(f.name for f in files if f.role == "data"),
            "r_code": ", ".join(f.name for f in files if f.role == "r_code"),
            "documents": ", ".join(f.name for f in files if f.role == "document")}},
        f"spec-v{s.spec_version}": {"kind": "spec", "facts": {
            "version": s.spec_version, "hash": r.spec_hash, "tier": s.spec_tier, "proven": s.spec_proven,
            **_scalars("", sp.model_dump(include={"target_model", "r_function", "statsmodels_counterpart", "family",
                                                  "link", "formula", "response", "offset", "weights"})),
            **_scalars("tolerance.", sp.tolerances.model_dump()),
            "kpis": ", ".join(sp.kpis), "split_method": sp.split.method, "seed": sp.split.seed,
            "preprocessing_steps": len(sp.preprocessing), "r_packages": ", ".join(sp.r_packages),
            "unsupported_items": "; ".join(sp.unsupported_items),
            "substitutions": "; ".join(f"{x.original} -> {x.proposed}" for x in sp.substitutions)}},
        r.baseline["id"]: {"kind": "baseline", "facts": {
            "hash": r.baseline["body"]["hash"], "environment_version": r.baseline["body"]["environment_version"],
            "r_version": meta["r_version"], "n_fit": meta["n_fit"], "converged": meta["converged"],
            "iterations": meta.get("iterations"), "coefficients": len(meta["design_columns"]),
            **{f"n.{k}": v for k, v in meta["n"].items()}, **_scalars("fit.", r.baseline["body"]["fit"])}},
    }
    for a in r.attempts:
        c = a.candidate["body"]
        entries[a.candidate["id"]] = {"kind": "candidate", "facts": {
            "attempt": c["attempt"], "hash": c["hash"], "model": c["model"], "notes": "; ".join(c["notes"]),
            "changes": c["changes"], "approved": a is r.approved}}
        if a.validation:
            entries[a.validation["id"]] = {"kind": "validation", "facts": _validation_facts(a.validation["body"])}
        if a.diagnosis:
            d = a.diagnosis["body"]
            entries[a.diagnosis["id"]] = {"kind": "diagnosis", "facts": {
                "candidate_id": d["candidate_id"], "root_cause": d["root_cause"],
                "repair_instructions": "; ".join(d["repair_instructions"]), "spec_change_needed": d["spec_change_needed"],
                "proposed_spec_change": d["proposed_spec_change"], "failed_checks": "; ".join(d["failed_checks"]),
                "ds_comment": d["ds_comment"], "model": d["model"]}}
    for d in r.decisions:
        entries[d["id"]] = {"kind": "decision", "facts": {
            "decision": d["decision"], "at": d["at"], "comment": d["comment"],
            "validation_id": d["bindings"].validation_id, "candidate_hash": d["bindings"].candidate_hash}}
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
        self.required = [record.approved.validation["id"], record.approval_id]

    def check(self, raw: dict) -> tuple[ReportNarrative | None, list[str]]:
        try:
            n = ReportNarrative.model_validate(raw)
        except ValidationError as e:
            return None, [f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors()]
        errors, cited = [], set()
        for i, sec in enumerate(n.sections):
            errors += [f"sections.{i}.heading: {e}" for e in self._spans(sec.heading)]
            for j, claim in enumerate(sec.claims):
                at = f"sections.{i}.claims.{j}"
                ids = set(claim.evidence_ids)
                cited |= ids
                errors += [f"{at}.evidence_ids: unknown evidence id {x!r}" for x in sorted(ids - self.entries.keys())]
                for m in PLACEHOLDER.finditer(claim.text):
                    eid, name = m.groups()
                    if eid not in ids:
                        errors.append(f"{at}.text: {m.group(0)} cites {eid}, which is not in the claim's evidence_ids")
                    elif eid in self.entries and name not in self.entries[eid]["facts"]:
                        errors.append(f"{at}.text: {eid} has no field {name!r}")
                errors += [f"{at}.text: {e}" for e in self._spans(claim.text)]
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
        return "\n".join([
            f"Project {self.r.project}: the DS approved the candidate {a.candidate['id']} and its validation "
            f"{a.validation['id']} ({self.required[1]}). Write the final report of this translation.", "",
            "Evidence catalog: each entry's id, kind and the facts a claim may cite as {{id.field}}:",
            json.dumps(self.entries, indent=1), "",
            "The approved candidate:", "```python", a.candidate["body"]["module"].rstrip(), "```", "",
            "baseline.R, the R code the baseline ran:", "```r", self.r.spec.baseline_r.rstrip(), "```", "",
            "After your narrative, code adds tables of the approved validation's checks and KPIs, every coefficient "
            "in R and Python, the attempts, the input files and the evidence index.", "",
            f"Cite {self.required[0]} and {self.required[1]}, then submit the report."])

    # ------------------------------------------------------------ rendering
    def render(self, n: ReportNarrative) -> str:
        r, s, a = self.r, self.r.state, self.r.approved
        v = a.validation["body"]
        facts = {eid: e["facts"] for eid, e in self.entries.items()}
        approval = next(d for d in r.decisions if d["id"] == r.approval_id)
        out = [f"# {r.project}", ""]
        if s.spec_tier == "approximate":
            out += ["**Approximate result.** statsmodels fits this model with a different algorithm from R; the "
                    "tolerances below were agreed for it.", ""]
        out += _table(["", ""], [
            ["Model", f"`{r.spec.r_function}` → `{r.spec.statsmodels_counterpart}`"],
            ["Tier", f"{s.spec_tier}{'' if s.spec_proven else ' (no passing fixture yet)'}"],
            ["Spec", f"v{s.spec_version} `{r.spec_hash[:12]}`"],
            ["R baseline", f"{r.baseline['id']}, {r.baseline['body']['meta']['r_version']}"],
            ["Candidate", f"{a.candidate['id']} `{a.candidate['body']['hash'][:12]}`"],
            ["Validation", f"{a.validation['id']}, {sum(c['passed'] for c in v['checks'])} of {len(v['checks'])} "
                           "checks passed"],
            ["Approved", approval["at"]],
            ["Environment", f"`{r.environment_version[:12]}`"]]) + [""]

        for sec in n.sections:
            out += [f"## {sec.heading}", ""]
            for c in sec.claims:
                text = PLACEHOLDER.sub(lambda m: fmt(facts[m.group(1)][m.group(2)]), c.text)
                out += [f"{text} [{', '.join(c.evidence_ids)}]", ""]

        out += ["## Validation", ""] + _table(["Check", "Value", "Tolerance", ""], [
            [c["name"], fmt(c["value"]), fmt(c["tolerance"]), "pass" if c["passed"] else "FAIL"] for c in v["checks"]])
        out += [""] + [f"- {x}" for x in v["notes"]] + ([""] if v["notes"] else [])
        kpis = [[st, k, fmt(x["r"]), fmt(x["python"])] for st, ks in v["kpis"].items() for k, x in ks.items()
                if (x["r"], x["python"]) != (None, None)]                # undefined ones are in the notes
        if kpis:
            out += ["## KPIs", ""] + _table(["Set", "KPI", "R", "Python"], kpis) + [""]

        rc, pc = _coefs(r.baseline_dir / "coefficients.csv"), _coefs(r.output_dir / "coefficients.csv")
        out += ["## Coefficients", ""] + _table(
            ["Term", "R estimate", "Python estimate", "Rel diff", "R std error", "Python std error", "Rel diff"],
            [[f"`{t}`", _g(e), _g(pc[t][0]), fmt(_rel(e, pc[t][0])), _g(se), _g(pc[t][1]), fmt(_rel(se, pc[t][1]))]
             for t, (e, se) in rc.items()]) + [""]

        by_candidate = {d["bindings"].candidate_hash: d for d in r.decisions}
        out += ["## Attempts", ""] + _table(["Attempt", "Candidate", "Validation", "Failed checks", "Diagnosis", "DS"], [
            [str(x.candidate["body"]["attempt"]), x.candidate["id"],
             x.validation["id"] if x.validation else "did not run",
             "; ".join(failed_checks(x.validation["body"])) if x.validation else "",
             x.diagnosis["body"]["root_cause"] if x.diagnosis else "",
             _decision(by_candidate.get(x.candidate["body"]["hash"]))] for x in r.attempts]) + [""]

        out += ["## Inputs", ""] + _table(["File", "Role", "SHA-256"],
                                          [[f.name, f.role, f"`{f.sha256}`"] for f in s.files]) + [""]
        rows = [[r.baseline["id"], "baseline", f"`{r.baseline['hash'][:12]}`", r.baseline["body"]["mlflow_run_id"]]]
        for x in r.attempts:
            rows += [[e["id"], e["kind"], f"`{e['hash'][:12]}`", e["body"]["mlflow_run_id"]]
                     for e in (x.candidate, x.validation, x.diagnosis) if e]
        out += ["## Evidence", ""] + _table(["Id", "Kind", "Hash", "MLflow run"], rows)
        return "\n".join(out) + "\n"


def _decision(d: dict | None) -> str:
    if d is None:
        return ""
    return f"{d['decision']}: {d['comment']}" if d["comment"] else d["decision"]


def _g(x: float) -> str:
    return "NA" if math.isnan(x) else f"{x:.10g}"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    def cell(x: str) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + [
        "| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
