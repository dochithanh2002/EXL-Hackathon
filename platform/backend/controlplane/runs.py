"""The MLflow layout of the models a validation compares, as comparable runs with explicit lineage.

    workflow
    ├─ baseline vN              kind=baseline, model=r: the R reference fit
    ├─ translation vN           kind=translation: the Translator's candidate
    └─ validation vN            kind=validation: r/…, python/… and parity/… metrics, plots, tables
       ├─ python fit            kind=model_fit, model=python: the candidate fitted on R's train rows
       │  └─ cv fold k          kind=cv_fold: a refit without fold k, scored on fold k
       └─ challenger constant   kind=challenger: a constant fitted on the development set

Every model run (the R fit, the Python fit, the challenger) logs the same keys: `model/<set>/<metric>` for its
actual performance, the spec and the data split as parameters, and the splits as dataset inputs, so MLflow's run
comparison lines them up. CV folds log `model/held_out/<metric>`. Lineage is in tags: `reference_run_id` (R's
fit), `candidate_run_id` (the translation), `validation_run_id`, `fit_run_id`, plus the evidence ids. Parity
differences live only on the validation run, under `parity/`."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .evaluation import Inputs, evaluated_metrics, scores
from .schemas import ModelSpec
from .tracking import StepRun, Tracker

FIT_STATS = {"r": ("deviance", "null_deviance", "log_lik", "aic", "bic", "dispersion", "pearson_chi2", "nobs"),
             "python": ("deviance", "null_deviance", "llf", "aic", "bic", "scale", "pearson_chi2", "nobs")}


def spec_params(spec: ModelSpec) -> dict:
    """The model specification as run parameters."""
    p = {"spec.r_function": spec.r_function, "spec.python_implementation": spec.python_implementation,
         "spec.tier": spec.support_tier, "spec.family": spec.family, "spec.link": spec.link,
         "spec.formula": spec.formula, "spec.response": spec.response, "spec.response_type": spec.response_type,
         "spec.prediction": f"{spec.prediction.kind}, {spec.prediction.quantity}: {spec.prediction.units}",
         "spec.offset": spec.offset, "spec.weights": spec.weights.expression if spec.weights else None,
         "spec.metrics": ", ".join(spec.metrics), "spec.split_method": spec.split.method,
         "spec.seed": spec.split.seed}
    return {k: v for k, v in p.items() if v is not None}


def _split(baseline_dir: Path) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = defaultdict(list)
    with open(baseline_dir / "split.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["set"]].append(int(r["row"]))
    return rows


def split_params(spec: ModelSpec, baseline_dir: Path) -> dict:
    meta = json.loads((baseline_dir / "meta.json").read_text())
    roles = {e.set: e.role for e in spec.assessment.evaluation_sets}
    out = {f"split.{s}.rows": len(rows) for s, rows in _split(baseline_dir).items()}
    out |= {f"split.{s}.role": role for s, role in roles.items()}
    return out | {"split.fitted_rows": meta.get("n_fit")}


def split_inputs(spec: ModelSpec, baseline_dir: Path, dataset: str, sha256: str) -> list[tuple[str, str, str, str]]:
    """Each set as a dataset input: its digest is the hash of its data-row numbers in the dataset file."""
    roles = {e.set: e.role for e in spec.assessment.evaluation_sets}
    out = []
    for s, rows in _split(baseline_dir).items():
        digest = hashlib.sha256(",".join(map(str, rows)).encode()).hexdigest()[:16]
        out.append((f"{dataset}:{s}", digest, f"{dataset} (sha256 {sha256[:12]})",
                    "training" if s == "train" else roles.get(s, "evaluation")))
    return out


def model_metrics(by_set: dict[str, dict[str, float | None]], prefix: str = "model") -> dict[str, float]:
    return {f"{prefix}/{s}/{m}": v for s, ms in by_set.items() for m, v in ms.items() if v is not None}


def fit_metrics(fit: dict, side: str) -> dict[str, float]:
    return {f"fit/{k}": fit[k] for k in FIT_STATS[side] if isinstance(fit.get(k), (int, float))
            and not isinstance(fit.get(k), bool)}


def log_reference(run: StepRun, spec: ModelSpec, baseline_dir: Path, dataset: str, sha256: str) -> None:
    """The R fit's model run: the same keys a Python fit logs."""
    body = spec.model_dump(mode="json")
    by_model, _ = scores(Inputs(baseline_dir, None, None, body))
    fit = json.loads((baseline_dir / "fit.json").read_text())
    run.params(**spec_params(spec), **split_params(spec, baseline_dir))
    run.metrics(**model_metrics(by_model["r"]), **fit_metrics(fit, "r"))
    run.tags(model="r")
    run.datasets(split_inputs(spec, baseline_dir, dataset, sha256))
    run.artifact_text(spec.model_dump_json(indent=2), "spec.json")


def log_comparison(tracker: Tracker, pid: str, validation: StepRun, *, spec: ModelSpec, baseline_dir: Path,
                   output_dir: Path, evaluation: dict, dataset: str, sha256: str, lineage: dict) -> dict[str, str]:
    """The Python fit, its CV folds and the constant challenger under the validation run. Returns their run ids."""
    body = spec.model_dump(mode="json")
    by_model, constants = scores(Inputs(baseline_dir, output_dir, None, body))
    lineage = lineage | {"validation_run_id": validation.run_id}
    params = spec_params(spec) | split_params(spec, baseline_dir)
    ids = {}
    job = {"job_id": lineage["job_id"]}
    tags = {k: v for k, v in lineage.items() if k != "job_id"}
    results = evaluation.get("checks", {}).get("stability", {}).get("results") or {}
    cv, history = results.get("cross_validation"), results.get("training_curve")
    primary = evaluated_metrics(body)[0]

    with tracker.step(pid, "model_fit", "python fit", parent=validation.run_id) as fit:
        ids["python_fit"] = fit.run_id
        fit.params(**job, **params)
        fit.metrics(**model_metrics(by_model.get("python", {})),
                    **fit_metrics(json.loads((output_dir / "fit.json").read_text()), "python"))
        fit.tags(model="python", **tags)
        fit.datasets(split_inputs(spec, baseline_dir, dataset, sha256))
        if history:                                               # step = IRLS iteration
            fit.series(f"train/{history['metric']}", history["values"], list(range(1, len(history["values"]) + 1)))
        if cv:                                                    # step = fold
            done = [f for f in cv["folds"] if f.get(primary) is not None]
            fit.series(f"cv/held_out/{primary}", [f[primary] for f in done], [f["fold"] for f in done])
        for name in ("coefficients.csv", "fit.json", "fit_history.json"):
            if (output_dir / name).is_file():
                fit.artifact(output_dir / name)

    if cv:
        names = list(cv["summary"])
        for f in cv["folds"]:
            with tracker.step(pid, "cv_fold", f"cv fold {f['fold']}", parent=ids["python_fit"]) as run:
                ids[f"cv_fold_{f['fold']}"] = run.run_id
                run.params(**job, fold=f["fold"], folds=len(cv["folds"]), fit_rows=f["fit_rows"],
                           held_out_rows=f["held_out_rows"], fold_assignment="hash of the data-row number")
                run.metrics(**{f"model/held_out/{m}": f[m] for m in names if f.get(m) is not None})
                run.tags(model="python", fit_run_id=ids["python_fit"], converged=f["converged"], **tags)

    if by_model.get("constant"):
        with tracker.step(pid, "challenger", "challenger constant", parent=validation.run_id) as run:
            ids["challenger_constant"] = run.run_id
            run.params(**job, **split_params(spec, baseline_dir), comparator="constant fitted on the development set",
                       **{f"constant.{k or 'value'}": v for k, v in constants.items()})
            run.metrics(**model_metrics(by_model["constant"]))
            run.tags(model="constant", **tags)
    return ids
