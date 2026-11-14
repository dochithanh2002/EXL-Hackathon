"""Model evaluation beyond migration parity: the confirmed validation plan's checks, executed by code.

The Analyst chooses which checks apply and why (`ModelSpec.validation_plan`); agents interpret the results. This
module computes them, on R's reference predictions and the Python candidate's, with the same metric code as the
parity checks:
- baseline comparison: each side's metrics against a constant comparator fitted on the development set;
- calibration: observed against predicted overall, by decile, and as a recalibration slope and intercept;
- subgroup performance: rows, observed, predicted, observed/expected and the primary loss per subgroup of the
  plan's columns (values exported by capture.R);
- stability: the assessment's PSI, cross-validated metrics and coefficients of Python refits, and the fit's
  iteration history;
- sensitivity: the change in Python's predictions when one predictor moves, all else fixed;
- assumptions: screens of the model's assumptions (overdispersion, zero inflation, multicollinearity, residual
  distribution, heteroscedasticity, separation).

Each check is executed, partially executed, not executed or not planned, and `needs` says what it could not
establish; `missing_evidence` collects all of it. Results are aggregates: no row or row-level value appears. None of
this decides the migration's acceptance, which rests on the parity checks alone."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .assessment import BINS, _bin, _quantile_edges
from .runner.parity import BINARY, _coefs, _predictions, metric, metric_weighting, units_by_set

EXECUTED, PARTIAL, NOT_EXECUTED, NOT_PLANNED = "executed", "partially executed", "not executed", "not planned"
CHECKS = ("baseline_comparison", "calibration", "subgroup_performance", "stability", "sensitivity", "assumptions")
TITLES = {"baseline_comparison": "Baseline comparison", "calibration": "Calibration",
          "subgroup_performance": "Subgroup performance", "stability": "Stability", "sensitivity": "Sensitivity",
          "assumptions": "Model assumptions"}
LOWER_IS_BETTER = {"rmse", "mae", "deviance", "mean_deviance", "log_loss", "brier", "pinball_loss"}
SCORES = {"auc", "gini", "ks"}
MAX_LEVELS = 12
MIN_AGENT_GROUP_ROWS = 10
QUINTILES = 5
SIDES = ("r", "python")


def _r(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(f"{x:.6g}")


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _check(status: str, rationale: str, results: dict | None = None, needs: list[str] | None = None,
           notes: list[str] | None = None) -> dict:
    return {"status": status, "rationale": rationale, "needs": needs or [], "notes": notes or [],
            "results": results or {}}


# ---------------------------------------------------------------- metrics
def primary_metric(spec: dict) -> str:
    """The proper loss each prediction kind is judged by."""
    p = spec["prediction"]
    if p["kind"] == "quantile":
        return "pinball_loss"
    if p["kind"] == "class_probability" or spec["response_type"] in BINARY:
        return "log_loss"
    if (spec.get("family") or "").removeprefix("quasi") in ("poisson", "binomial", "Gamma", "gaussian",
                                                            "inverse.gaussian"):
        return "mean_deviance"
    return "rmse"


def evaluated_metrics(spec: dict) -> list[str]:
    """The primary loss and the spec's metrics that a set of predictions defines (AIC needs a fit)."""
    names = [primary_metric(spec)] + [m for m in spec["metrics"] if m != "aic"]
    return list(dict.fromkeys(names))


def score(name: str, units: list, spec: dict, weighted: bool) -> float | None:
    kind = spec["prediction"]["kind"]
    if name == "mean_deviance":
        d = metric("deviance", units, spec.get("family"), kind, weighted)
        total = sum(w for *_, w in units)
        return None if d is None or not total else d / total
    return metric(name, units, spec.get("family"), kind, weighted)


def comparison(name: str, model: float | None, constant: float | None) -> float | None:
    """Skill (1 - model / constant) for a loss, the difference for a ranking score, None for a ratio."""
    if model is None or constant is None:
        return None
    if name in LOWER_IS_BETTER:
        return 1 - model / constant if constant > 0 else None
    if name in SCORES:
        return model - constant
    return None


# ---------------------------------------------------------------- inputs
class Inputs:
    """Both sides' predictions as valid units per set, with R's prior weights on both."""

    def __init__(self, baseline: Path, candidate: Path | None, diagnostics: Path | None, spec: dict):
        self.baseline, self.candidate, self.diagnostics, self.spec = baseline, candidate, diagnostics, spec
        r = _predictions(baseline / "predictions.csv")
        self.weights = {k: w for k, (_, _, w) in r.items()}
        found = candidate is not None and (candidate / "predictions.csv").is_file()
        py = _predictions(candidate / "predictions.csv") if found else {}
        self.units = {"r": units_by_set(r), "python": units_by_set(py, self.weights) if py else {}}
        self.weighted, why = metric_weighting(spec)
        self.notes = [why] if why else []
        self.roles = {e["set"]: e["role"] for e in spec["assessment"]["evaluation_sets"]}
        self.dev = next((s for s, role in self.roles.items() if role == "development"), "train")
        self.independent = [s for s, role in self.roles.items() if role != "development"
                            and self.units["r"].get(s)]
        self.judged = self.independent or [self.dev]
        self.meta = _json(baseline / "meta.json")
        self.metrics = evaluated_metrics(spec)
        self.keys = spec["prediction"]["keys"] or [""]

    def wt(self, w: float) -> float:
        return w if self.weighted else 1.0

    def side_units(self, side: str, s: str) -> list:
        return self.units[side].get(s, [])

    def sides(self) -> list[str]:
        return [side for side in SIDES if self.units[side]]


# ---------------------------------------------------------------- baseline comparison
def _constant_units(inp: Inputs, units: list) -> tuple[list, dict]:
    """The units with every prediction replaced by the development set's constant for its key."""
    kind = inp.spec["prediction"]["kind"]
    dev = inp.side_units("r", inp.dev)
    constants: dict[str, float] = {}
    for key in inp.keys:
        g = [(y, inp.wt(w)) for _, k, y, _, w in dev if k == key]
        total = sum(w for _, w in g)
        if not total:
            continue
        if kind == "quantile":
            tau, acc = float(key), 0.0
            for y, w in sorted(g):
                acc += w
                if acc >= tau * total:
                    constants[key] = y
                    break
        else:
            constants[key] = sum(w * y for y, w in g) / total
    return [(row, k, y, constants[k], w) for row, k, y, _, w in units if k in constants], constants


def scores(inp: Inputs) -> tuple[dict[str, dict[str, dict[str, float | None]]], dict[str, float]]:
    """model ("r", "python", "constant") -> set -> metric, and the constant comparator's values by key. The models
    are scored on the same rows with the same weights, so their runs compare directly."""
    out: dict[str, dict] = {m: {} for m in (*inp.sides(), "constant")}
    constants: dict[str, float] = {}
    for s in inp.roles:
        r_units = inp.side_units("r", s)
        if not r_units:
            continue
        const_units, constants = _constant_units(inp, r_units)
        out["constant"][s] = {m: _r(score(m, const_units, inp.spec, inp.weighted)) for m in inp.metrics}
        for side in inp.sides():
            out[side][s] = {m: _r(score(m, inp.side_units(side, s), inp.spec, inp.weighted)) for m in inp.metrics}
    return out, constants


def baseline_comparison(inp: Inputs, rationale: str) -> dict:
    by_model, constants = scores(inp)
    sets = {}
    for s, const in by_model["constant"].items():
        entry = {}
        for name, c in const.items():
            row = {"constant": c}
            for side in inp.sides():
                v = by_model[side].get(s, {}).get(name)
                row[side] = v
                row[f"{side}_vs_constant"] = _r(comparison(name, v, c))
            entry[name] = row
        sets[s] = entry
    notes = []
    if inp.spec.get("offset") or inp.spec["prediction"].get("exposure"):
        notes.append("the constant comparator ignores the offset and exposure: it is a no-information benchmark, "
                     "not an exposure-only model")
    if not sets:
        return _check(NOT_EXECUTED, rationale, needs=["valid R predictions on an evaluation set"])
    return _check(EXECUTED, rationale, {"comparator": "constant fitted on the development set",
                                        "constants": {k or "value": _r(v) for k, v in constants.items()},
                                        "primary_metric": primary_metric(inp.spec), "sets": sets}, notes=notes)


# ---------------------------------------------------------------- calibration
def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def logistic_recalibration(g: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    """(intercept, slope) of y ~ a + b logit(p) by Newton-Raphson on the weighted binomial likelihood."""
    xs = [(_logit(p), y, w) for y, p, w in g]
    a, b = 0.0, 1.0
    for _ in range(50):
        ga = gb = haa = hab = hbb = 0.0
        for x, y, w in xs:
            eta = max(min(a + b * x, 35.0), -35.0)
            mu = 1 / (1 + math.exp(-eta))
            v = w * mu * (1 - mu)
            ga += w * (y - mu)
            gb += w * (y - mu) * x
            haa, hab, hbb = haa + v, hab + v * x, hbb + v * x * x
        det = haa * hbb - hab * hab
        if det <= 1e-12 * max(haa * hbb, 1e-300):
            return None
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-10:
            return a, b
    return None


def linear_recalibration(g: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    """(intercept, slope) of the weighted least-squares fit of y on the prediction."""
    tw = sum(w for *_, w in g)
    if not tw:
        return None
    mx, my = sum(w * p for _, p, w in g) / tw, sum(w * y for y, _, w in g) / tw
    sxx = sum(w * (p - mx) ** 2 for _, p, w in g)
    if sxx <= 0:
        return None
    b = sum(w * (p - mx) * (y - my) for y, p, w in g) / sxx
    return my - b * mx, b


def _calibration_side(inp: Inputs, g: list[tuple[float, float, float]], probability: bool) -> dict:
    tw = sum(w for *_, w in g)
    obs, exp = sum(w * y for y, _, w in g), sum(w * p for _, p, w in g)
    edges = _quantile_edges([p for _, p, _ in g], BINS)
    bins: dict[int, list] = defaultdict(list)
    for y, p, w in g:
        bins[_bin(p, edges)].append((y, p, w))
    table = []
    for i in sorted(bins):
        b, bw = bins[i], sum(w for *_, w in bins[i])
        if bw:
            table.append({"bin": i + 1, "units": len(b), "mean_predicted": _r(sum(w * p for _, p, w in b) / bw),
                          "mean_observed": _r(sum(w * y for y, _, w in b) / bw)})
    gap = max((abs(t["mean_observed"] - t["mean_predicted"]) for t in table
               if None not in (t["mean_observed"], t["mean_predicted"])), default=None)
    fit = (logistic_recalibration if probability and all(0 <= y <= 1 for y, *_ in g) else linear_recalibration)(g)
    return {"units": len(g), "weight": _r(tw), "observed_expected": _r(obs / exp) if exp > 0 else None,
            "intercept": _r(fit[0]) if fit else None, "slope": _r(fit[1]) if fit else None,
            "max_bin_gap": _r(gap), "bins": table}


def calibration(inp: Inputs, rationale: str) -> dict:
    p = inp.spec["prediction"]
    probability = p["quantity"] == "probability" or p["kind"] == "class_probability"
    sets, needs = {}, []
    for s in inp.roles:
        entry = {}
        for key in inp.keys:
            by_side = {}
            for side in inp.sides():
                g = [(y, pr, inp.wt(w)) for _, k, y, pr, w in inp.side_units(side, s) if k == key]
                if not g:
                    continue
                if p["kind"] == "quantile":
                    tw = sum(w for *_, w in g)
                    by_side[side] = {"units": len(g), "nominal": float(key),
                                     "coverage": _r(sum(w for y, q, w in g if y <= q) / tw) if tw else None}
                else:
                    by_side[side] = _calibration_side(inp, g, probability)
            if by_side:
                entry[key or "value"] = by_side
        if entry:
            sets[s] = entry
        else:
            needs.append(f"valid predictions on {s}")
    if not sets:
        return _check(NOT_EXECUTED, rationale, needs=needs or ["valid predictions"])
    if not inp.independent:
        needs.append("an independent evaluation set: calibration is measured on the development set only")
    method = ("coverage of each quantile level" if p["kind"] == "quantile" else
              "logistic recalibration, logit(observed) ~ intercept + slope × logit(predicted)" if probability else
              "least squares of observed on predicted")
    return _check(PARTIAL if needs else EXECUTED, rationale, {"method": method, "judged_on": inp.judged,
                                                              "sets": sets}, needs)


# ---------------------------------------------------------------- subgroups
def _segments(baseline: Path) -> dict[str, dict[tuple[str, int], str]]:
    out: dict[str, dict[tuple[str, int], str]] = defaultdict(dict)
    for r in _rows(baseline / "segments.csv"):
        out[r["column"]][(r["set"], int(r["row"]))] = r["value"]
    return out


def _groups(inp: Inputs, values: dict[tuple[str, int], str], kind: str) -> tuple[dict[tuple[str, int], str], list]:
    """Each (set, row)'s group label and the groups in order with their description."""
    labels: dict[tuple[str, int], str] = {}
    if kind == "numeric":
        dev = [float(v) for (s, _), v in values.items() if s == inp.dev and v != ""]
        edges = _quantile_edges(dev, QUINTILES)
        names = [f"Q{i + 1}" for i in range(len(edges) + 1)]
        bounds = [-math.inf, *edges, math.inf]
        groups = [{"group": n, "lower": _r(bounds[i]) if i else None,
                   "upper": _r(bounds[i + 1]) if i < len(edges) else None} for i, n in enumerate(names)]
        for k, v in values.items():
            labels[k] = "(missing)" if v == "" else names[_bin(float(v), edges)]
        if any(v == "" for v in values.values()):
            groups.append({"group": "(missing)"})
        return labels, groups
    counts: dict[str, int] = defaultdict(int)
    for (s, _), v in values.items():
        if s == inp.dev:
            counts[v or "(missing)"] += 1
    ranked = sorted(counts, key=lambda v: (-counts[v], v))
    top = set(ranked[:MAX_LEVELS])
    for k, v in values.items():
        v = v or "(missing)"
        labels[k] = v if v in top else "Other"
    groups = [{"group": v} for v in ranked[:MAX_LEVELS]] + ([{"group": "Other"}] if len(ranked) > MAX_LEVELS else [])
    return labels, groups


def _group_stats(inp: Inputs, units: list, binary: bool) -> dict:
    primary = {inp.metrics[0]: _r(score(inp.metrics[0], units, inp.spec, inp.weighted))}
    if inp.keys != [""]:                            # class probabilities or quantiles: the loss is the summary
        return primary
    tw = sum(inp.wt(w) for *_, w in units)
    obs = sum(inp.wt(w) * y for _, _, y, _, w in units)
    exp = sum(inp.wt(w) * p for _, _, _, p, w in units)
    out = {"mean_observed": _r(obs / tw) if tw else None, "mean_predicted": _r(exp / tw) if tw else None,
           "observed_expected": _r(obs / exp) if exp > 0 else None, **primary}
    if binary:
        out["auc"] = _r(score("auc", units, inp.spec, inp.weighted))
    return out


def subgroup_performance(inp: Inputs, plan: dict) -> dict:
    segments, types = _segments(inp.baseline), inp.meta.get("segments") or {}
    binary = inp.spec["response_type"] in BINARY and inp.spec["prediction"]["kind"] in ("scalar", "time_indexed")
    columns, needs = {}, []
    for col in plan["columns"]:
        kind = types.get(col, "missing")
        if kind == "missing" or col not in segments:
            columns[col] = {"status": NOT_EXECUTED, "reason": "the column is not in R's set frames"}
            needs.append(f"subgroup column {col} in R's set frames")
            continue
        labels, groups = _groups(inp, segments[col], kind)
        buckets: dict[tuple[str, str, str | None], list] = defaultdict(list)
        for side in inp.sides():
            for s in inp.roles:
                for x in inp.side_units(side, s):
                    buckets[(side, s, labels.get((s, x[0])))].append(x)
        for g in groups:
            g["sets"] = {}
            for s in inp.roles:
                rows = len({x[0] for x in buckets.get(("r", s, g["group"]), [])})
                if rows:
                    entry = {"rows": rows, "small": rows < plan["min_rows"], "scored_rows": {}}
                    for side in inp.sides():
                        if units := buckets.get((side, s, g["group"])):
                            entry[side] = _group_stats(inp, units, binary)
                            # Count contributing observations per output and side, not rows assigned by R.
                            by_key: dict[str, set[int]] = defaultdict(set)
                            for row, key, _, _, weight in units:
                                if weight > 0:
                                    by_key[key].add(row)
                            entry["scored_rows"][side] = min(len(by_key[k]) for k in inp.keys)
                    g["sets"][s] = entry
        columns[col] = {"status": EXECUTED, "type": kind, "groups": [g for g in groups if g["sets"]]}
        small = [g["group"] for g in groups if any(v["small"] for v in g["sets"].values())]
        if small:
            columns[col]["small_groups"] = small
    executed = [c for c in columns.values() if c["status"] == EXECUTED]
    status = NOT_EXECUTED if not executed else PARTIAL if needs else EXECUTED
    return _check(status, plan["rationale"], {"metric": inp.metrics[0], "min_rows": plan["min_rows"],
                                              "judged_on": inp.judged, "columns": columns}, needs,
                  notes=["numeric columns are cut at quintiles of the development set; categorical columns keep "
                         f"their {MAX_LEVELS} largest levels and fold the rest into Other"])


# ---------------------------------------------------------------- stability
def _cv(inp: Inputs, report: dict) -> tuple[dict | None, list[str]]:
    cv = report.get("cross_validation") or {}
    if cv.get("status") != EXECUTED:
        return None, [f"cross-validation refits ({_failed(cv)})"]
    folds: dict[int, list] = defaultdict(list)
    for r in _rows(inp.diagnostics / "cv_predictions.csv"):
        y, p = float(r["y"]) if r["y"] else math.nan, float(r["prediction"]) if r["prediction"] else math.nan
        w = inp.weights.get(("train", int(r["row"]), r["key"]), math.nan)
        if math.isfinite(y) and math.isfinite(p) and math.isfinite(w):
            folds[int(r["fold"])].append((int(r["row"]), r["key"], y, p, w))
    names = [m for m in inp.metrics if m != "deviance"]
    per_fold = []
    for k in sorted(folds):
        fit = next((f for f in cv["fits"] if f["fold"] == k), {})
        per_fold.append({"fold": k, "fit_rows": fit.get("fit_rows"), "held_out_rows": fit.get("held_out_rows"),
                         "converged": fit.get("converged"),
                         **{m: _r(score(m, folds[k], inp.spec, inp.weighted)) for m in names}})
    full = {m: _r(score(m, inp.side_units("python", "train"), inp.spec, inp.weighted)) for m in names}
    summary = {}
    for m in names:
        v = [f[m] for f in per_fold if f[m] is not None]
        if v:
            mean = sum(v) / len(v)
            sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else None
            summary[m] = {"mean": _r(mean), "sd": _r(sd), "min": _r(min(v)), "max": _r(max(v)), "in_sample": full[m]}
    full_coefs = _coefs(inp.candidate / "coefficients.csv")
    by_term: dict[str, list[float]] = defaultdict(list)
    for r in _rows(inp.diagnostics / "cv_coefficients.csv"):
        if r["estimate"]:
            by_term[r["term"]].append(float(r["estimate"]))
    terms = []
    for t, v in by_term.items():
        est = full_coefs.get(t, (math.nan, math.nan))[0]
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        terms.append({"term": t, "estimate": _r(est), "fold_sd": _r(sd),
                      "relative_sd": _r(sd / abs(est)) if math.isfinite(est) and est else None,
                      "sign_changes": sum(1 for x in v if math.isfinite(est) and x * est < 0)})
    terms.sort(key=lambda t: -(t["relative_sd"] or 0))
    unconverged = [f["fold"] for f in per_fold if f["converged"] is False]
    needs = [f"converged refits on folds {unconverged}"] if unconverged else []
    return {"folds": per_fold, "summary": summary, "coefficients": terms,
            "terms_with_sign_changes": sum(1 for t in terms if t["sign_changes"])}, needs


def _failed(part: dict) -> str:
    if part.get("status") == "failed":
        return f"failed in {part.get('stage')} ({part.get('type')})"
    return part.get("reason") or "not run"


def stability(inp: Inputs, plan: dict, assessment: dict, report: dict | None) -> dict:
    results, needs = {}, []
    psi = (assessment.get("python") or assessment.get("r") or {}).get("stability") or {}
    if psi.get("results"):
        results["psi"] = psi["results"]
    else:
        needs += psi.get("needs") or ["a holdout or out-of-time set, or periods, to compare score distributions"]
    history = _json(inp.candidate / "fit_history.json")
    if history:
        name = "deviance" if "deviance" in history else next(iter(history))
        results["training_curve"] = {"metric": name, "values": [_r(v) for v in history[name]],
                                     "iterations": len(history[name]), "r_iterations": inp.meta.get("iterations")}
    if report is None:
        needs.append("cross-validation refits (the Python diagnostics did not run)")
    else:
        cv, more = _cv(inp, report)
        needs += more
        if cv:
            results["cross_validation"] = cv
    status = NOT_EXECUTED if not results else PARTIAL if needs else EXECUTED
    notes = ["folds are assigned by a hash of the data-row number; each refit reuses prepare()'s frames, so "
             "preprocessing fitted on all rows is not refitted per fold"] if "cross_validation" in results else []
    return _check(status, plan["rationale"], results, needs, notes)


# ---------------------------------------------------------------- sensitivity
def sensitivity(plan: dict, report: dict | None) -> dict:
    if report is None:
        return _check(NOT_EXECUTED, plan["rationale"], needs=["sensitivity predictions (the Python diagnostics "
                                                              "did not run)"])
    part = report.get("sensitivity") or {}
    if part.get("status") != EXECUTED:
        return _check(NOT_EXECUTED, plan["rationale"], needs=[f"sensitivity predictions ({_failed(part)})"])
    columns = part["columns"]
    needs = [f"sensitivity of {c} ({_failed(v)})" for c, v in columns.items() if v["status"] != EXECUTED]
    executed = any(v["status"] == EXECUTED for v in columns.values())
    return _check(NOT_EXECUTED if not executed else PARTIAL if needs else EXECUTED, plan["rationale"],
                  {"shift_sd": plan["shift_sd"], "columns": columns}, needs,
                  ["each column moves by shift_sd training standard deviations in the frames the Python model "
                   "reads; changes are unweighted means over rows"])


# ---------------------------------------------------------------- assumptions
def _moments(residuals: list[float]) -> dict:
    n = len(residuals)
    if n < 3:
        return {}
    mean = sum(residuals) / n
    m2 = sum((x - mean) ** 2 for x in residuals) / n
    if m2 <= 0:
        return {}
    skew = sum((x - mean) ** 3 for x in residuals) / n / m2 ** 1.5
    kurt = sum((x - mean) ** 4 for x in residuals) / n / m2 ** 2 - 3
    return {"rows": n, "skewness": _r(skew), "excess_kurtosis": _r(kurt),
            "jarque_bera": _r(n / 6 * (skew ** 2 + kurt ** 2 / 4))}


def _assumption(inp: Inputs, name: str, report: dict | None) -> dict:
    train = {side: inp.side_units(side, "train") for side in inp.sides()}
    family = (inp.spec.get("family") or "").removeprefix("quasi")
    if name == "overdispersion":
        fits = {"r": _json(inp.baseline / "fit.json"), "python": _json(inp.candidate / "fit.json")}
        keys = {"r": ("pearson_chi2", "df_residual", "deviance"), "python": ("pearson_chi2", "df_resid", "deviance")}
        out = {}
        for side, (chi2, df, dev) in keys.items():
            f = fits[side]
            if isinstance(f.get(chi2), (int, float)) and f.get(df):
                out[side] = {"pearson_dispersion": _r(f[chi2] / f[df]),
                             "deviance_dispersion": _r(f[dev] / f[df]) if isinstance(f.get(dev), (int, float))
                             else None}
        if not out:
            return {"status": NOT_EXECUTED, "reason": "the fits report no Pearson statistic"}
        return {"status": EXECUTED, "results": out, "reference": "1 for a correctly specified dispersion"}
    if name == "zero_inflation":
        if family != "poisson":
            return {"status": NOT_EXECUTED, "reason": "expected zeros are computed under a Poisson model only"}
        out = {}
        for side, u in train.items():
            tw = sum(inp.wt(w) for *_, w in u)
            if tw:
                observed = sum(inp.wt(w) for _, _, y, _, w in u if y == 0) / tw
                expected = sum(inp.wt(w) * math.exp(-p) for _, _, _, p, w in u) / tw
                out[side] = {"observed_zero_share": _r(observed), "expected_zero_share": _r(expected),
                             "ratio": _r(observed / expected) if expected > 0 else None}
        return {"status": EXECUTED if out else NOT_EXECUTED, "results": out}
    if name == "multicollinearity":
        part = (report or {}).get("vif") or {}
        if part.get("status") != EXECUTED:
            return {"status": NOT_EXECUTED, "reason": "variance inflation factors " + (
                _failed(part) if report is not None else "need the Python diagnostics, which did not run")}
        cols = sorted(part["columns"], key=lambda c: -c["vif"])
        return {"status": EXECUTED, "results": {
            "python": {"max_vif": _r(cols[0]["vif"]) if cols else None,
                       "condition_number": _r(part.get("condition_number"))},
            "columns": [{"name": c["name"], "vif": _r(c["vif"])} for c in cols[:15]]}}
    if name == "residual_distribution":
        out = {side: _moments([y - p for _, _, y, p, _ in u]) for side, u in train.items()}
        out = {k: v for k, v in out.items() if v}
        return {"status": EXECUTED if out else NOT_EXECUTED, "results": out,
                "reference": "skewness and excess kurtosis 0 for normal residuals"}
    if name == "heteroscedasticity":
        out = {}
        for side, u in train.items():
            edges = _quantile_edges([p for *_, p, _ in u], QUINTILES)
            groups: dict[int, list[float]] = defaultdict(list)
            for _, _, y, p, _ in u:
                groups[_bin(p, edges)].append(y - p)
            var = {i + 1: sum(x * x for x in g) / len(g) for i, g in sorted(groups.items()) if len(g) > 1}
            if len(var) > 1 and min(var.values()) > 0:
                out[side] = {"residual_variance_by_quintile": {str(k): _r(v) for k, v in var.items()},
                             "max_min_ratio": _r(max(var.values()) / min(var.values()))}
        return {"status": EXECUTED if out else NOT_EXECUTED, "results": out,
                "reference": "1 for constant residual variance across predicted quintiles"}
    if name == "separation":
        out = {}
        for side, u in train.items():
            if u:
                extreme = sum(1 for *_, p, _ in u if p < 1e-8 or p > 1 - 1e-8)
                out[side] = {"extreme_prediction_share": _r(extreme / len(u))}
        coefs = _coefs(inp.candidate / "coefficients.csv")
        if coefs:
            out.setdefault("python", {})["max_abs_estimate"] = _r(max(abs(e) for e, _ in coefs.values()
                                                                      if math.isfinite(e)))
        out.setdefault("r", {})["fitted_probabilities_warning"] = bool(inp.meta.get("separation_warning"))
        return {"status": EXECUTED, "results": out,
                "reference": "no extreme predictions and no R warning that fitted probabilities are 0 or 1"}
    raise ValueError(f"unknown assumption {name}")


def assumptions(inp: Inputs, plan: dict, report: dict | None) -> dict:
    items = {name: _assumption(inp, name, report) for name in plan["assumptions"]}
    needs = [f"{name.replace('_', ' ')} ({v.get('reason', 'not executed')})" for name, v in items.items()
             if v["status"] != EXECUTED]
    executed = any(v["status"] == EXECUTED for v in items.values())
    return _check(NOT_EXECUTED if not executed else PARTIAL if needs else EXECUTED, plan["rationale"],
                  {"assumptions": items}, needs)


# ---------------------------------------------------------------- the plan
def diagnostics_config(spec: dict) -> dict | None:
    """What the Python diagnostics harness must run for the plan, or None if nothing."""
    vp = spec["validation_plan"]
    config = {"folds": vp["stability"]["folds"] if vp["stability"]["run"] else None,
              "sensitivity": {"columns": vp["sensitivity"]["columns"], "shift_sd": vp["sensitivity"]["shift_sd"]}
              if vp["sensitivity"]["run"] else None,
              "vif": vp["assumptions"]["run"] and "multicollinearity" in vp["assumptions"]["assumptions"]}
    return config if any(config.values()) else None


def evaluate(baseline: Path, candidate: Path, diagnostics: Path | None, spec: dict, assessment: dict) -> dict:
    """Every check of the spec's validation plan. `diagnostics` is the Python diagnostics output directory, or None
    when they did not run (a failed parity check, or no check needing them)."""
    inp = Inputs(baseline, candidate, diagnostics, spec)
    report = _json(diagnostics / "diagnostics.json") if diagnostics is not None else None
    if diagnostics is not None and (diagnostics / "error.json").is_file():
        e = _json(diagnostics / "error.json")
        failed = {"status": "failed", "stage": e.get("stage"), "type": e.get("type")}
        report = {"cross_validation": failed, "sensitivity": failed, "vif": failed}
    vp = spec["validation_plan"]
    run = {
        "baseline_comparison": lambda c: baseline_comparison(inp, c["rationale"]),
        "calibration": lambda c: calibration(inp, c["rationale"]),
        "subgroup_performance": lambda c: subgroup_performance(inp, c),
        "stability": lambda c: stability(inp, c, assessment, report),
        "sensitivity": lambda c: sensitivity(c, report),
        "assumptions": lambda c: assumptions(inp, c, report),
    }
    checks = {name: run[name](vp[name]) if vp[name]["run"] else _check(NOT_PLANNED, vp[name]["rationale"])
              for name in CHECKS}
    return {"checks": checks, "metrics": inp.metrics, "primary_metric": inp.metrics[0], "notes": inp.notes,
            "missing_evidence": missing_evidence(checks, assessment, spec)}


def missing_evidence(checks: dict, assessment: dict, spec: dict) -> list[str]:
    """Everything the validation could not establish, in one list: checks left out of the plan, what the executed
    ones lacked, the assessment's needs and the information the spec says is still missing."""
    out = []
    for name, c in checks.items():
        if c["status"] == NOT_PLANNED:
            out.append(f"{TITLES[name]}: not in the validation plan ({c['rationale']})")
        out += [f"{TITLES[name]}: {n}" for n in c["needs"]]
    for q, a in (assessment.get("python") or assessment.get("r") or {}).items():
        out += [f"{q.replace('_', ' ').capitalize()}: {n}" for n in a.get("needs", [])]
    out += [f"Still to provide: {x}" for x in spec["assessment"]["missing_information"]]
    return list(dict.fromkeys(out))


def evaluation_view(ev: dict) -> dict:
    """The evaluation as the Diagnostician sees it: statuses, needs and Python's aggregate results, without R's
    values or coefficient estimates (its instructions reach the Translator)."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("r", "r_vs_constant", "estimate", "bins")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    checks = ev.get("checks", {})
    out = {k: strip(v) for k, v in checks.items()}
    if "subgroup_performance" in out:
        out["subgroup_performance"]["results"] = subgroup_prompt_results(
            checks["subgroup_performance"].get("results", {}), sides=("python",))
    return {"checks": out, "missing_evidence": ev.get("missing_evidence", [])}


def subgroup_prompt_results(results: dict, sides: tuple[str, ...] = SIDES) -> dict:
    """Keep only adequately populated subgroup cells in agent prompts. Omit small cell labels and data cut points."""
    if not results:
        return {}
    minimum = max(MIN_AGENT_GROUP_ROWS, results.get("min_rows", 0))
    columns = {}
    for name, col in results.get("columns", {}).items():
        groups, withheld = [], 0
        for group in col.get("groups", []):
            sets = {}
            for s, cell in group["sets"].items():
                visible = {side: cell[side] for side in sides
                           if side in cell and cell.get("scored_rows", {}).get(side, 0) >= minimum}
                withheld += any(side in cell and side not in visible for side in sides)
                if visible:
                    sets[s] = {"rows": cell["rows"], **visible}
            if sets:
                groups.append({"group": group["group"], "sets": sets})
        columns[name] = {"status": col["status"], "groups": groups, "cells_withheld": withheld}
        if "reason" in col:
            columns[name]["reason"] = col["reason"]
    return {"metric": results.get("metric"), "min_rows": minimum, "judged_on": results.get("judged_on", []),
            "columns": columns}


# ---------------------------------------------------------------- tables
def tables(v: dict) -> dict[str, tuple[list[str], list[list]]]:
    """The diagnostic tables logged with the validation run: file name -> (header, rows)."""
    out: dict[str, tuple[list[str], list[list]]] = {}
    out["parity_checks.csv"] = (["check", "quantity", "status", "required", "ratio", "max_abs_diff", "atol", "rtol",
                                 "reason"],
                                [[c["name"], c.get("quantity"), c.get("status"), c.get("required"), c.get("value"),
                                  c.get("max_abs_diff"), c.get("atol"), c.get("rtol"), c.get("reason")]
                                 for c in v["checks"]])
    ev = v.get("evaluation") or {}
    checks = ev.get("checks") or {}
    out["missing_evidence.csv"] = (["item"], [[x] for x in ev.get("missing_evidence", [])])
    res = lambda name: (checks.get(name) or {}).get("results") or {}  # noqa: E731
    if sets := res("baseline_comparison").get("sets"):
        out["performance.csv"] = (["set", "metric", "r", "python", "constant", "r_vs_constant",
                                   "python_vs_constant"],
                                  [[s, m, x.get("r"), x.get("python"), x.get("constant"), x.get("r_vs_constant"),
                                    x.get("python_vs_constant")] for s, e in sets.items() for m, x in e.items()])
    if sets := res("calibration").get("sets"):
        rows = []
        for s, keys in sets.items():
            for key, sides in keys.items():
                for side, d in sides.items():
                    rows += [[s, key, side, b["bin"], b["units"], b["mean_predicted"], b["mean_observed"]]
                             for b in d.get("bins", [])]
        out["calibration_bins.csv"] = (["set", "output", "side", "bin", "units", "mean_predicted", "mean_observed"],
                                       rows)
    if cols := res("subgroup_performance").get("columns"):
        rows = []
        for col, c in cols.items():
            for g in c.get("groups", []):
                for s, e in g["sets"].items():
                    for side in SIDES:
                        if side in e:
                            rows.append([col, g["group"], s, side, e["rows"], e["small"], *[
                                e[side].get(k) for k in ("mean_observed", "mean_predicted", "observed_expected",
                                                         res("subgroup_performance")["metric"], "auc")]])
        out["subgroups.csv"] = (["column", "group", "set", "side", "rows", "small", "mean_observed", "mean_predicted",
                                 "observed_expected", res("subgroup_performance").get("metric"), "auc"], rows)
    if cv := res("stability").get("cross_validation"):
        names = list(cv["summary"])
        out["cv_folds.csv"] = (["fold", "fit_rows", "held_out_rows", "converged", *names],
                               [[f["fold"], f["fit_rows"], f["held_out_rows"], f["converged"],
                                 *[f.get(m) for m in names]] for f in cv["folds"]])
        out["cv_coefficients.csv"] = (["term", "estimate", "fold_sd", "relative_sd", "sign_changes"],
                                      [[t["term"], t["estimate"], t["fold_sd"], t["relative_sd"], t["sign_changes"]]
                                       for t in cv["coefficients"]])
    if cols := res("sensitivity").get("columns"):
        out["sensitivity.csv"] = (["column", "set", "output", "direction", "rows", "mean_prediction", "mean_change",
                                   "mean_abs_change", "max_abs_change"],
                                  [[c, s, k, d, x["rows"], x["mean_prediction"], x["mean_change"],
                                    x["mean_abs_change"], x["max_abs_change"]]
                                   for c, e in cols.items() if e.get("status") == EXECUTED
                                   for s, keys in e["sets"].items() for k, dirs in keys.items()
                                   for d, x in dirs.items()])
    items = res("assumptions").get("assumptions") or {}
    if (m := items.get("multicollinearity") or {}).get("status") == EXECUTED:
        out["vif.csv"] = (["column", "vif"], [[c["name"], c["vif"]] for c in m["results"]["columns"]])
    return out


def write_tables(v: dict, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (header, rows) in tables(v).items():
        with open(dest / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(["" if x is None else x for x in row] for row in rows)
        written.append(name)
    return written
