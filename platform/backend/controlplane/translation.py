"""Step 4: what the Translator sees, the checks on its submissions, and the candidate run.

The Translator sees the confirmed spec with its statistical semantics, baseline.R, the baseline's metadata and R's
coefficient names (never R's estimates, which a candidate could otherwise copy), the dataset profile and the
rulebook. A submission is a
Candidate only if it validates and passes the static guardrails. The candidate then runs in the stable Python
environment against R's sets, and the validator compares its output with the frozen baseline."""
from __future__ import annotations

import csv
import json
import shutil
import tomllib
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from . import candidate as contract
from .baseline import copy_verified
from .environments import Environments
from .runner import Limits, Process, StepCancelled
from .schemas import Candidate, ModelSpec
from .evaluation import diagnostics_config
from .statistics import Inputs
from .validation import Validation, compare, failed_run, with_evaluation
from .workflow.machine import InputFile

HARNESS = Path(__file__).parent / "runner" / "candidate_harness.py"
RULEBOOK = Path(__file__).parent / "rulebook.yaml"
META_KEYS = ("r_version", "class", "formula", "family", "link", "response", "response_type", "event_coding",
             "response_levels", "prediction", "design_columns", "aliased", "xlevels", "contrasts", "offset", "weights",
             "n", "n_fit", "converged", "closed_form", "iterations", "control", "column_classes")


def environment_packages(envs: Environments) -> dict[str, str]:
    """Pinned versions of the packages a candidate may import."""
    lock = tomllib.loads(envs.python_lockfile.read_text())
    return {p["name"]: p["version"] for p in lock["package"] if p["name"] in contract.ENV_PACKAGES}


def baseline_context(spec: ModelSpec, baseline_dir: Path) -> list[str]:
    """The confirmed spec, baseline.R, the baseline's metadata and R's coefficient names: what the Translator and
    the Diagnostician both start from. R's estimates are never part of it."""
    meta = json.loads((baseline_dir / "meta.json").read_text())
    return [
        "Confirmed ModelSpec:", json.dumps(spec.model_dump(mode="json", exclude={"baseline_r"}), indent=1), "",
        "baseline.R, the R code the baseline ran:", "```r", spec.baseline_r.rstrip(), "```", "",
        "R baseline metadata:", json.dumps({k: meta.get(k) for k in META_KEYS}, indent=1), "",
        "R's coefficient names, in order: " + json.dumps(coefficient_names(baseline_dir))]


def semantics(spec: ModelSpec) -> list[str]:
    """The statistical semantics the candidate must carry through fitting and prediction, in plain sentences."""
    p = spec.prediction
    lines = [f"Response '{spec.response}' ({spec.response_type})."]
    if spec.event:
        lines.append(f"Event: {spec.event.meaning}, coded '{spec.event.coding}' (1 in RESPONSE); otherwise "
                     f"{spec.event.complement} (0).")
    if g := spec.grouped_binomial:
        lines.append(f"Grouped binomial ({g.form}): trials = {g.trials}" + (f", successes = {g.successes}" if
                                                                             g.successes else "")
                     + "; RESPONSE holds the proportion of successes and the trials enter the fit as R's weights.")
    if w := spec.weights:
        lines.append(f"Weights {w.expression}: {w.type} weights, normalization {w.normalization}. In statsmodels "
                     "GLM, frequency weights are freq_weights and precision weights are var_weights; sampling "
                     "weights reproduce R's point estimates as R's prior weights do.")
    if spec.offset:
        lines.append(f"Offset: {spec.offset}.")
    keys = f" Keys, in order: {json.dumps(p.keys)}." if p.keys else ""
    lines.append(f"Prediction: {p.kind}, {p.quantity} in {p.units}" + (f", per {p.exposure}" if p.exposure else "")
                 + f". R computes it as {p.r_predict}.{keys} Alignment keys: {p.alignment_keys}.")
    return ["Statistical semantics:", *(f"- {x}" for x in lines)]


def coefficient_names(baseline_dir: Path) -> list[str]:
    with open(baseline_dir / "coefficients.csv", newline="") as f:
        return [r["term"] for r in csv.DictReader(f)]


def rulebook() -> str:
    return RULEBOOK.read_text()


class Translator:
    """One translation attempt's inputs and checks. A repair attempt also gets the previous candidate and the
    Diagnostician's findings (`repair`: attempt, limit, module, diagnosis)."""

    def __init__(self, spec: ModelSpec, baseline_dir: Path, dataset: str, envs: Environments,
                 profile: Callable[[str | None], dict], data: list[str] = (), repair: dict | None = None):
        self.spec, self.baseline_dir, self.dataset, self.envs = spec, baseline_dir, dataset, envs
        self.profile, self.data, self.repair = profile, list(data), repair

    def check(self, raw: dict) -> tuple[Candidate | None, list[str]]:
        try:
            cand = Candidate.model_validate(raw)
        except ValidationError as e:
            return None, [f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors()]
        errors = [f"guardrail: {v}" for v in contract.check(cand.module, data=bool(self.data))]
        if self.repair:
            if cand.module.strip() == self.repair["module"].strip():
                errors.append("module: identical to the previous candidate; apply the repair instructions")
            if not cand.changes.strip():
                errors.append("changes: describe what changed from the previous candidate and why")
        return (None if errors else cand), errors

    # ------------------------------------------------------------ prompt
    def kickoff(self) -> str:
        packages = ", ".join(f"{n}=={v}" for n, v in sorted(environment_packages(self.envs).items()))
        lines = baseline_context(self.spec, self.baseline_dir) + [
            "",
            f"prepare() is called with the path of {self.dataset}. RESPONSE is '{self.spec.response}'.", "",
            *semantics(self.spec), "",
            contract.INTERFACE, "",
            *([contract.DATA_INTERFACE + " The data files: " + ", ".join(self.data) + ". dataset_profile with a "
               "file's name gives its profile.", ""] if self.data else []),
            f"Python environment: {packages}. Standard-library imports allowed: "
            + ", ".join(sorted(contract.STDLIB - {"__future__"})) + ".",
            "The module must not read any file other than the project's data files, write files, use the network, "
            "run shell commands, use randomness, or use eval, exec, getattr or dunder attributes.", ""]
        if not self.repair:
            return "\n".join(lines + ["Read the dataset profile and, where useful, the rulebook, then submit the "
                                       "candidate."])
        r, d = self.repair, self.repair["diagnosis"]
        lines += [f"This is repair attempt {r['attempt']} of {r['limit']}. The previous candidate:",
                  "```python", r["module"].rstrip(), "```", ""]
        if d.get("ds_comment"):
            lines += ["The DS rejected it with this comment:", d["ds_comment"], ""]
        lines += ["The Diagnostician's findings:", json.dumps(
            {k: d[k] for k in ("root_cause", "evidence", "repair_instructions")}, indent=1), ""]
        if d["spec_change_needed"]:
            lines += [f"The Diagnostician proposed a spec change ({d['proposed_spec_change']}); the DS kept the "
                      "confirmed spec, so repair the code within it.", ""]
        return "\n".join(lines + ["Apply the repair instructions and submit the complete module, with `changes` "
                                   "saying what changed and why."])


# ---------------------------------------------------------------- run
def run_candidate(workdir: Path, module: str, spec: ModelSpec, baseline_dir: Path, files: tuple[InputFile, ...],
                  inputs_dir: Path, envs: Environments, limits: Limits = Limits(),
                  on_start: Callable[[Process], None] = lambda p: None) -> Validation:
    """Run the candidate under the harness in a fresh working directory holding only the data files and R's sets, then
    compare its output with the baseline. A candidate that fails to run is a failed validation, not an error. A
    passing candidate then runs the validation plan's Python diagnostics in a second harness process, whose failure
    is recorded as missing evidence, never as a failed validation."""
    workdir.mkdir(parents=True)
    inputs = workdir / "inputs"
    inputs.mkdir()
    dataset = next(f for f in files if f.role == "dataset")
    data = [f for f in files if f.role == "data"]
    for f in [dataset, *data]:
        copy_verified(inputs_dir / f.name, inputs / f.name, f.sha256)
    shutil.copy(baseline_dir / "split.csv", inputs / "split.csv")
    (inputs / "split.csv").chmod(0o444)
    inputs.chmod(0o555)
    (workdir / "candidate.py").write_text(module)
    shutil.copy(HARNESS, workdir / "harness.py")
    config = {"candidate": "candidate.py", "dataset": f"inputs/{dataset.name}",
              "data": {f.name: f"inputs/{f.name}" for f in data}, "split": "inputs/split.csv", "out": "output",
              **spec.capture_config()}
    (workdir / "config.json").write_text(json.dumps(config))

    def harness(config_file: str):
        proc = Process([str(envs.py()), "-I", "harness.py", config_file], workdir, {"PYTHONDONTWRITEBYTECODE": "1"},
                       limits)
        on_start(proc)
        outcome = proc.start().wait()
        if outcome.cancelled:
            raise StepCancelled()
        return outcome

    outcome = harness("config.json")
    out = workdir / "output"
    if not outcome.ok:
        error = out / "error.json"
        if error.is_file() and not outcome.timed_out:
            return failed_run(json.loads(error.read_text()))
        return failed_run({"stage": "the candidate harness", "type": "Stopped",
                           "message": "the candidate " + outcome.describe(lines=0)})
    inputs = Inputs(baseline_dir, out, None, spec.model_dump(mode="json"))
    v = compare(baseline_dir, out, spec, inputs)
    diagnostics = None
    if v.passed and (plan := diagnostics_config(spec.model_dump(mode="json"))):
        diagnostics = workdir / "diagnostics"
        (workdir / "diagnostics-config.json").write_text(json.dumps(config | {"out": "diagnostics",
                                                                               "diagnostics": plan}))
        outcome = harness("diagnostics-config.json")
        if not outcome.ok and not (diagnostics / "error.json").is_file():
            diagnostics.mkdir(exist_ok=True)
            (diagnostics / "error.json").write_text(json.dumps({"stage": "the diagnostics harness",
                                                                "type": "Stopped"}))
    inputs.diagnostics = diagnostics
    return with_evaluation(v, inputs)
