"""Step 5: what the Diagnostician sees and the checks on its submissions.

The Diagnostician sees what the Translator saw (spec, baseline.R, baseline metadata, R's coefficient names), the
candidate under diagnosis, the DS's comment if they rejected it, earlier attempts for the same spec, and a filtered
view of the validation. The view keeps every check, per-term differences in tolerance units, row ids, evaluation
metrics, the design comparison (fitted rows, design-matrix column statistics, reference levels, offsets and
weights in R and Python) and Python's side of the model validation checks, and withholds three things:
- R's estimates and fit statistics, and the candidate's: its repair instructions go to the Translator, which must
  never be able to copy R's numbers into a candidate;
- row-level predictions and responses;
- data values in runtime error messages (`redact`). The step log, which holds the full traceback, is never sent.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from . import candidate as contract
from .evaluation import evaluation_view
from .schemas import Diagnosis, ModelSpec
from .translation import baseline_context, coefficient_names

REDACTED = "<redacted>"
QUOTED = re.compile(r"(?<!\w)'([^'\n]*)'(?!\w)|(?<!\w)\"([^\"\n]*)\"(?!\w)")
DECIMAL = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])")
TYPE_NAMES = {"str", "int", "float", "bool", "object", "list", "dict", "tuple", "set", "NoneType", "bytes", "complex",
              "category", "string", "int8", "int16", "int32", "int64", "uint8", "float32", "float64", "boolean",
              "datetime64[ns]", "Series", "DataFrame", "Index", "ndarray"}
# validation `differences` keys that hold only names and row ids, passed on as they are
NAMES_AND_ROWS = ("missing_terms", "extra_terms", "rows_only_in_r", "rows_only_in_python", "response_differs_rows",
                  "prediction_na_differs_rows")
CHECK_FIELDS = ("name", "quantity", "kind", "status", "required", "reason", "value", "tolerance", "atol", "rtol",
                "max_abs_diff", "compared", "missing_both", "missing_one_side", "nonfinite_mismatch", "total")


def _sig(x: float | None) -> float | None:
    return None if x is None else float(f"{x:.3g}")


def _direction(r: float | None, py: float | None) -> str:
    """How the candidate's value sits relative to R's, without either value. Missing values arrive as None."""
    if r is None or py is None:
        return {(True, True): "NA on both sides", (True, False): "NA in R only",
                (False, True): "NA in Python only"}[(r is None, py is None)]
    if r == py:
        return "equal"
    if r * py < 0:
        return "opposite sign"
    return "larger in magnitude" if abs(py) > abs(r) else "smaller in magnitude"


def redact(message: str, vocabulary: set[str], source: str) -> str:
    """Remove data values from an error message: quoted text unless it is a known name (column, level, term, type or
    a literal in the candidate's own code), and decimal numbers unless they appear in the candidate's code.
    Integers stay, since they are mostly row ids, counts and shapes."""
    literals = _literals(source)
    names = vocabulary | TYPE_NAMES | literals["names"]

    def quoted(m: re.Match) -> str:
        text = m.group(1) if m.group(1) is not None else m.group(2)
        return m.group(0) if text in names else REDACTED
    message = QUOTED.sub(quoted, message)
    return DECIMAL.sub(lambda m: m.group(0) if m.group(0).lstrip("+-") in literals["numbers"] else REDACTED, message)


def _literals(source: str) -> dict[str, set[str]]:
    names, numbers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source)), set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"names": names, "numbers": numbers}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, float):
            numbers |= {repr(node.value), ast.get_source_segment(source, node) or ""}
    return {"names": names, "numbers": numbers - {""}}


def vocabulary(profile: dict, terms: list[str], spec: ModelSpec) -> set[str]:
    """Names an error message may quote: what the agents already see in the profile, the spec and R's terms."""
    words = set(terms) | {spec.response, "train", "test"}
    for col in profile.get("columns", []):
        words |= {col["name"], col["r_name"]} | {str(lv["value"]) for lv in col.get("levels", [])}
    return words


def validation_view(v: dict, words: set[str], source: str) -> dict:
    """The validation as the Diagnostician sees it (module docstring)."""
    view = {"passed": v["passed"], "stage": v["stage"],
            "checks": [{k: c[k] for k in CHECK_FIELDS if c.get(k) not in (None, "")} for c in v["checks"]],
            "metrics": v.get("metrics", {}), "notes": v["notes"]}
    if v.get("error"):
        e = v["error"]
        view["error"] = {"stage": e["stage"], "type": e["type"], "message": redact(e["message"], words, source),
                         "candidate_frames": e.get("candidate_frames", [])}
    d = v.get("differences", {})
    diffs = {k: d[k] for k in NAMES_AND_ROWS if k in d}
    if "worst_terms" in d:
        diffs["worst_terms"] = [
            {"term": t["term"],
             "estimate_rel_diff": _sig(t["estimate_rel"]),
             "estimate_tolerance_units": _sig(t["estimate_score"]),
             "estimate": _direction(t["estimate_r"], t["estimate_python"]),
             "std_error_rel_diff": _sig(t["std_error_rel"]),
             "std_error_tolerance_units": _sig(t["std_error_score"]),
             "std_error": _direction(t["std_error_r"], t["std_error_python"])} for t in d["worst_terms"]]
    if "worst_predictions" in d:
        diffs["worst_predictions"] = [{"set": p["set"], "row": p["row"], "key": p["key"],
                                       "abs_diff": _sig(p["abs_diff"]), "tolerance_units": _sig(p["score"])}
                                      for p in d["worst_predictions"]]
    if "fit_statistics" in d:
        diffs["fit_statistics"] = {k: _direction(s["r"], s["python"]) for k, s in d["fit_statistics"].items()}
    view["differences"] = diffs
    if v.get("design"):
        view["design"] = v["design"]
    if v.get("evaluation"):
        view["evaluation"] = evaluation_view(v["evaluation"])
    return view


def failed_checks(v: dict) -> list[str]:
    """Required checks that did not pass; an unavailable one is named as such."""
    if v.get("error"):
        return [f"runtime error in {v['error']['stage']}"]
    return [c["name"] + (" (unavailable)" if c.get("status") == "unavailable" else "")
            for c in v["checks"] if not c["passed"] and c.get("required", True)]


class Diagnostician:
    """One diagnosis's inputs and checks. `candidate`, `validation` and `earlier` (the diagnoses already made for
    this spec, oldest first) are evidence bodies."""

    def __init__(self, spec: ModelSpec, baseline_dir: Path, profile: Callable[[], dict], candidate: dict,
                 validation: dict, ds_comment: str | None, earlier: list[dict], repairs: int, retry_limit: int):
        self.spec, self.baseline_dir, self.profile = spec, baseline_dir, profile
        self.candidate, self.validation, self.ds_comment = candidate, validation, ds_comment
        self.earlier, self.repairs, self.retry_limit = earlier, repairs, retry_limit

    def check(self, raw: dict) -> tuple[Diagnosis | None, list[str]]:
        try:
            return Diagnosis.model_validate(raw), []
        except ValidationError as e:
            return None, [f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors()]

    def view(self) -> dict:
        words = vocabulary(self.profile(), coefficient_names(self.baseline_dir), self.spec)
        return validation_view(self.validation, words, self.candidate["module"])

    def kickoff(self) -> str:
        c = self.candidate
        lines = baseline_context(self.spec, self.baseline_dir) + [
            "", contract.INTERFACE, "",
            "The candidate under diagnosis:", "```python", c["module"].rstrip(), "```",
            "The Translator's notes: " + json.dumps(c["notes"])]
        if c.get("changes"):
            lines.append("What it changed from the attempt before: " + c["changes"])
        if c.get("limitations"):
            lines.append("Limitations the Translator reported: " + json.dumps(c["limitations"]))
        lines += ["", "Its validation against the R baseline (R's estimates, row-level values and data values are "
                      "withheld). A tolerance check's value is the worst |python - r| / (atol + rtol * |r|); above 1 "
                      "fails. `design` compares what each side fitted; `evaluation` gives the model validation "
                      "checks on the candidate (baseline comparison, calibration, subgroups, stability, "
                      "sensitivity, assumptions, sample suitability, decision performance), each with how far it "
                      "ran (status), what it needs, and the outcome of the spec's criterion, plus the platform's "
                      "business fitness verdict:", json.dumps(self.view(), indent=1), ""]
        if self.ds_comment:
            lines += ["The validation passed, but the DS rejected the candidate with this comment:", self.ds_comment,
                      ""]
        if self.earlier:
            lines += ["Earlier attempts for this spec, oldest first:", json.dumps(
                [{k: e.get(k) for k in ("category", "failed_checks", "root_cause", "repair_instructions")}
                 for e in self.earlier], indent=1), ""]
        lines += [f"Repairs used: {self.repairs} of {self.retry_limit}.", "",
                  "Read the dataset profile and, where useful, the rulebook, then submit the diagnosis."]
        return "\n".join(lines)
