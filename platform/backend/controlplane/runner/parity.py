"""Shared approval and standalone-bundle comparison of a candidate with the frozen R baseline, evolved from
`legacy/agent/tools/parity.py`. Standard library only: this file is also shipped unchanged in deliverables bundles.

Every numeric comparison is `abs(python - r) <= atol + rtol * abs(r)`, with tolerances set per quantity in the spec.
Missing and non-finite values are compared, not skipped: missing on both sides, or the same infinity, is a match;
missing or non-finite on one side only is a mismatch. A check is passed, failed or unavailable (it could not be
computed); an unavailable required check fails the validation and is never shown as passed. Metrics are computed
here, by the same code for both sides, from each side's observed values and predictions."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

EPS = 1e-15
EXACT_STATS = {"nobs", "df_residual"}
# R fit.json key -> how to read it from the candidate's results (fit.json written by the harness)
LM_STATS = {"nobs": "nobs", "df_residual": "df_resid", "rss": "ssr", "sigma": "sigma", "r_squared": "rsquared",
            "adj_r_squared": "rsquared_adj", "log_lik": "llf", "aic": "aic", "bic": "bic"}
GLM_STATS = {"nobs": "nobs", "df_residual": "df_resid", "deviance": "deviance", "null_deviance": "null_deviance",
             "dispersion": "scale", "pearson_chi2": "pearson_chi2"}
OTHER_STATS = {"nobs": "nobs", "log_lik": "llf"}
LIKELIHOOD = {"log_lik": "llf", "aic": "aic", "bic": "bic"}
# families whose statsmodels log-likelihood is defined as R's logLik.glm (other families estimate the scale
# differently, so their likelihood statistics are not compared)
SAME_LIKELIHOOD = {("binomial", None), ("poisson", None), ("gaussian", "identity")}
ESTIMATED_SCALE = {"gaussian", "Gamma", "inverse.gaussian"}
BINARY = {"binary", "grouped_binomial"}
METRIC_WEIGHTS = {"frequency", "sampling"}


# ---------------------------------------------------------------- reading
def _num(s: str | None) -> float:
    return float(s) if s not in (None, "", "NA", "NaN") else math.nan


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _coefs(path: Path) -> dict[str, tuple[float, float]]:
    if not path.is_file():
        return {}
    return {r["term"]: (_num(r["estimate"]), _num(r["std_error"])) for r in _rows(path)}


Key = tuple[str, int, str]                       # set, data row, prediction key ("" for one value per row)


def _predictions(path: Path) -> dict[Key, tuple[float, float, float]]:
    """(set, row, key) -> (observed, prediction, prior weight). Only R's capture has weights; they are data."""
    return {(r["set"], int(r["row"]), r.get("key") or ""): (_num(r["y"]), _num(r["prediction"]),
                                                             _num(r.get("weight", "1")))
            for r in _rows(path)}


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def _rel(r: float, py: float) -> float:
    if _same(r, py):
        return 0.0
    if math.isnan(r) or math.isnan(py) or r == 0:
        return math.inf
    return abs(r - py) / abs(r)


def score(r: float, py: float, atol: float, rtol: float) -> float:
    """|py - r| in units of the allowed difference atol + rtol * |r|: at most 1 passes. Missing on both sides and
    equal infinities are 0; missing or non-finite on one side is infinite."""
    if _same(r, py):
        return 0.0
    if not (math.isfinite(r) and math.isfinite(py)):
        return math.inf
    bound = atol + rtol * abs(r)
    return abs(py - r) / bound if bound > 0 else math.inf


# ---------------------------------------------------------------- checks
def _check(name: str, quantity: str, kind: str, value: float | None, tolerance: float, passed: bool | None,
           required: bool = True, reason: str = "", **extra) -> dict:
    status = "unavailable" if passed is None else "passed" if passed else "failed"
    return dict(name=name, quantity=quantity, kind=kind, value=value, tolerance=tolerance, status=status,
                passed=status == "passed", required=required, reason=reason, **extra)


def count_check(name: str, quantity: str, n: int) -> dict:
    """A count that must be zero."""
    return _check(name, quantity, "count", n, 0, n == 0)


def flag_check(name: str, quantity: str, flag: bool | None, reason: str = "") -> dict:
    return _check(name, quantity, "flag", None if flag is None else float(flag), 1, flag, reason=reason)


def tolerance_check(name: str, quantity: str, pairs: list[tuple[float, float]], tol: dict, required: bool = True,
                    reason: str = "") -> dict:
    """Elementwise within atol + rtol * |r|; unavailable when there is nothing to compare."""
    atol, rtol = tol["atol"], tol["rtol"]
    scores = [score(r, py, atol, rtol) for r, py in pairs]
    finite = [abs(py - r) for r, py in pairs if math.isfinite(r) and math.isfinite(py)]
    one_sided = sum(1 for r, py in pairs if math.isnan(r) != math.isnan(py))
    nonfinite = sum(1 for r, py in pairs if not _same(r, py) and not math.isnan(r) and not math.isnan(py)
                    and not (math.isfinite(r) and math.isfinite(py)))
    value = max(scores) if pairs else None
    return _check(name, quantity, "tolerance", value, 1.0, None if not pairs else value <= 1, required,
                  reason or ("nothing to compare" if not pairs else ""), atol=atol, rtol=rtol,
                  max_abs_diff=max(finite, default=None), compared=len(pairs),
                  missing_both=sum(1 for r, py in pairs if math.isnan(r) and math.isnan(py)),
                  missing_one_side=one_sided, nonfinite_mismatch=nonfinite)


def unavailable(name: str, quantity: str, reason: str, required: bool = True) -> dict:
    return _check(name, quantity, "tolerance", None, 1.0, None, required, reason)


# ---------------------------------------------------------------- metrics
Unit = tuple[int, str, float, float, float]       # row, key, observed, prediction, weight


def _auc_ks(pairs: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    """Weighted Mann-Whitney AUC (ties count half) and KS for observed values in [0, 1]: a row contributes
    weight * y events and weight * (1 - y) non-events at its score, which also covers grouped binomial rows."""
    if not pairs or any(not 0 <= y <= 1 for y, _, _ in pairs):
        return None
    by_score: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for y, p, w in pairs:
        by_score[p][0] += w * y
        by_score[p][1] += w * (1 - y)
    pos, neg = sum(v[0] for v in by_score.values()), sum(v[1] for v in by_score.values())
    if pos <= 0 or neg <= 0:
        return None
    auc, below_neg, below_pos, ks = 0.0, 0.0, 0.0, 0.0
    for s in sorted(by_score):
        p, n = by_score[s]
        auc += p * (below_neg + n / 2)
        below_neg, below_pos = below_neg + n, below_pos + p
        ks = max(ks, abs(below_neg / neg - below_pos / pos))
    return auc / (pos * neg), ks


def _ylogy(y: float, mu: float) -> float:
    return y * math.log(y / mu) if y > 0 else 0.0


def _deviance(family: str | None, pairs: list[tuple[float, float, float]]) -> float | None:
    base = {"quasibinomial": "binomial", "quasipoisson": "poisson"}.get(family, family)
    try:
        if base == "binomial":
            return 2 * sum(w * (_ylogy(y, m) + _ylogy(1 - y, 1 - m))
                           for y, mu, w in pairs for m in [min(max(mu, EPS), 1 - EPS)])
        if base == "poisson":
            return 2 * sum(w * (_ylogy(y, m) - (y - m)) for y, mu, w in pairs for m in [max(mu, EPS)])
        if base == "Gamma":
            return -2 * sum(w * (math.log(y / m) - (y - m) / m) for y, mu, w in pairs for m in [max(mu, EPS)])
        if base == "gaussian":
            return sum(w * (y - mu) ** 2 for y, mu, w in pairs)
        if base == "inverse.gaussian":
            return sum(w * (y - mu) ** 2 / (y * mu ** 2) for y, mu, w in pairs)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def metric(name: str, units: list[Unit], family: str | None = None, kind: str = "scalar",
           weighted: bool = True) -> float | None:
    """One metric over a set's valid units (finite observed value and prediction); None where it is not defined.
    `weighted` applies the prior weights to every metric except deviance, which always uses them, as R does."""
    if not units:
        return None
    wt = (lambda w: w) if weighted else (lambda w: 1.0)
    if kind == "class_probability":
        rows: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        for row, _, y, p, w in units:
            rows[row].append((y, p, w))
        total = sum(wt(v[0][2]) for v in rows.values())
        if name == "log_loss":
            true = [(next((p for y, p, _ in v if y == 1), None), wt(v[0][2])) for v in rows.values()]
            if any(p is None for p, _ in true) or not total:
                return None
            return -sum(w * math.log(min(max(p, EPS), 1)) for p, w in true) / total
        if name == "brier":
            return sum(wt(v[0][2]) * sum((y - p) ** 2 for y, p, _ in v) for v in rows.values()) / total if total else None
        return None
    if kind == "quantile":
        if name != "pinball_loss":
            return None
        by_key: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for _, key, y, q, w in units:
            by_key[key].append((y, q, wt(w)))
        losses = []
        for key, v in by_key.items():
            tau, total = float(key), sum(w for _, _, w in v)
            losses.append(sum(w * max(tau * (y - q), (tau - 1) * (y - q)) for y, q, w in v) / total if total else 0)
        return sum(losses) / len(losses)
    pairs = [(y, p, w) for _, _, y, p, w in units]
    weighted_pairs = [(y, p, wt(w)) for y, p, w in pairs]
    total = sum(w for _, _, w in weighted_pairs)
    if name in ("auc", "gini", "ks"):
        r = _auc_ks(weighted_pairs)
        return None if r is None else {"auc": r[0], "gini": 2 * r[0] - 1, "ks": r[1]}[name]
    if name == "actual_expected":
        expected = sum(w * p for _, p, w in weighted_pairs)
        return sum(w * y for y, _, w in weighted_pairs) / expected if expected else None
    if not total:
        return None
    if name == "rmse":
        return math.sqrt(sum(w * (y - p) ** 2 for y, p, w in weighted_pairs) / total)
    if name == "mae":
        return sum(w * abs(y - p) for y, p, w in weighted_pairs) / total
    if name == "brier":
        return sum(w * (y * (1 - p) ** 2 + (1 - y) * p ** 2) for y, p, w in weighted_pairs) / total \
            if all(0 <= y <= 1 for y, _, _ in pairs) else None
    if name == "log_loss":
        if any(not 0 <= y <= 1 for y, _, _ in pairs):
            return None
        return -sum(w * (y * math.log(m) + (1 - y) * math.log(1 - m))
                    for y, p, w in weighted_pairs for m in [min(max(p, EPS), 1 - EPS)]) / total
    if name == "deviance":
        return _deviance(family, pairs)
    if name == "pinball_loss":
        return None
    raise ValueError(f"unknown metric {name}")


def metric_weighting(spec: dict) -> tuple[bool, str]:
    """Whether metrics apply the prior weights, and why: frequency and sampling weights and grouped-binomial trials
    count observations; precision weights describe variance, so metrics are unweighted."""
    if spec["response_type"] == "grouped_binomial":
        return True, "metrics are weighted by the number of trials"
    w = spec.get("weights")
    if not w:
        return False, ""
    if w["type"] in METRIC_WEIGHTS:
        return True, f"metrics are weighted by the {w['type']} weights ({w['expression']})"
    return False, f"metrics are unweighted: {w['expression']} are precision weights (deviance still uses them)"


def units_by_set(preds: dict[Key, tuple[float, float, float]], weights: dict[Key, float] | None = None
                 ) -> dict[str, list[Unit]]:
    """Valid units per set, in R's set order. `weights` (R's prior weights) replace the side's own."""
    out: dict[str, list[Unit]] = {}
    for (s, row, key), (y, p, w) in preds.items():
        out.setdefault(s, [])
        if weights is not None:
            w = weights.get((s, row, key), math.nan)
        if math.isfinite(y) and math.isfinite(p) and math.isfinite(w):
            out[s].append((row, key, y, p, w))
    return out


# ---------------------------------------------------------------- comparison
def python_fit_stats(raw: dict, rank: int, r_class: list[str], family: str | None) -> dict[str, float | None]:
    """The harness's results statistics, plus AIC and BIC counted the way R counts parameters."""
    stats = dict(raw)
    llf, nobs, scale = raw.get("llf"), raw.get("nobs"), raw.get("scale")
    if "lm" in r_class or "glm" in r_class:
        k = rank + (1 if "glm" not in r_class or family in ESTIMATED_SCALE else 0)
        stats["aic"] = None if llf is None else -2 * llf + 2 * k
        stats["bic"] = None if llf is None or not nobs else -2 * llf + math.log(nobs) * k
    stats["sigma"] = None if scale is None or scale < 0 else math.sqrt(scale)
    return stats


def compare(baseline: Path, candidate: Path, spec: dict) -> dict:
    tol = spec["tolerances"]
    meta = json.loads((baseline / "meta.json").read_text())
    family, link, r_class = meta.get("family"), meta.get("link"), meta.get("class") or []
    kind = spec["prediction"]["kind"]
    checks: list[dict] = []
    diffs: dict = {}
    notes: list[str] = []

    # ---- convergence
    py_raw = json.loads((candidate / "fit.json").read_text())
    convergence = spec.get("convergence", {})
    if convergence.get("mode") == "not_applicable":
        checks.append(unavailable("R reference fit converged", "convergence", convergence["rationale"], required=False))
    else:
        checks.append(flag_check("R reference fit converged", "convergence", meta.get("converged"),
                                 "" if meta.get("converged") is not None else "R reports no convergence status"))
    if meta.get("closed_form") and py_raw.get("converged") is None:
        notes.append("the R model is fitted in closed form, so no Python convergence flag is needed")
    else:
        checks.append(flag_check("Python fit converged", "convergence", py_raw.get("converged"),
                                 "" if py_raw.get("converged") is not None else
                                 "the Python results report no convergence status"))

    # ---- coefficients
    rc, pc = _coefs(baseline / "coefficients.csv"), _coefs(candidate / "coefficients.csv")
    if not rc:
        notes.append("R reports no coefficients for this model; coefficients are not compared")
    else:
        missing, extra = [t for t in rc if t not in pc], [t for t in pc if t not in rc]
        checks.append(count_check("coefficient names missing or extra", "coefficients", len(missing) + len(extra)))
        if missing or extra:
            diffs["missing_terms"], diffs["extra_terms"] = missing, extra
            checks.append(unavailable("coefficient estimates", "coefficients", "coefficient names differ"))
            checks.append(unavailable("coefficient standard errors", "std_errors", "coefficient names differ"))
        else:
            ct, st = tol["coefficients"], tol["std_errors"]
            checks.append(tolerance_check("coefficient estimates", "coefficients",
                                          [(rc[t][0], pc[t][0]) for t in rc], ct))
            if any(not math.isnan(rc[t][1]) for t in rc):
                checks.append(tolerance_check("coefficient standard errors", "std_errors",
                                              [(rc[t][1], pc[t][1]) for t in rc], st))
            else:
                notes.append("R reports no standard errors for this model; they are not compared")
            sc = {t: (score(rc[t][0], pc[t][0], ct["atol"], ct["rtol"]),
                      score(rc[t][1], pc[t][1], st["atol"], st["rtol"])) for t in rc}
            worst = sorted(rc, key=lambda t: max(sc[t]), reverse=True)[:8]
            diffs["worst_terms"] = [{"term": t, "estimate_r": rc[t][0], "estimate_python": pc[t][0],
                                     "estimate_rel": _rel(rc[t][0], pc[t][0]), "estimate_score": sc[t][0],
                                     "std_error_r": rc[t][1], "std_error_python": pc[t][1],
                                     "std_error_rel": _rel(rc[t][1], pc[t][1]), "std_error_score": sc[t][1]}
                                    for t in worst]

    # ---- alignment, observed values and predictions
    rp, pp = _predictions(baseline / "predictions.csv"), _predictions(candidate / "predictions.csv")
    only_r, only_py = rp.keys() - pp.keys(), pp.keys() - rp.keys()
    checks.append(count_check("alignment keys only on one side", "alignment", len(only_r) + len(only_py)))
    if only_r or only_py:
        diffs["rows_only_in_r"] = sorted(only_r)[:10]
        diffs["rows_only_in_python"] = sorted(only_py)[:10]
    common = sorted(rp.keys() & pp.keys())
    y_bad = [k for k in common if not _same(rp[k][0], pp[k][0])]
    checks.append(count_check("observed values differing", "observed", len(y_bad)))
    if y_bad:
        diffs["response_differs_rows"] = y_bad[:10]
    na_bad = [k for k in common if math.isnan(rp[k][1]) != math.isnan(pp[k][1])]
    checks.append(count_check("prediction missingness differing", "predictions", len(na_bad)))
    if na_bad:
        diffs["prediction_na_differs_rows"] = [{"set": s, "row": r, "key": k, "r_is_na": math.isnan(rp[(s, r, k)][1])}
                                               for s, r, k in na_bad[:10]]
    both = [k for k in common if math.isnan(rp[k][1]) == math.isnan(pp[k][1])]
    checks.append(tolerance_check("predictions", "predictions", [(rp[k][1], pp[k][1]) for k in both],
                                  tol["predictions"]))
    pt = tol["predictions"]
    ranked = sorted(((score(rp[k][1], pp[k][1], pt["atol"], pt["rtol"]), k) for k in both), reverse=True)[:5]
    diffs["worst_predictions"] = [{"set": k[0], "row": k[1], "key": k[2], "r": rp[k][1], "python": pp[k][1],
                                   "abs_diff": abs(rp[k][1] - pp[k][1]), "score": s} for s, k in ranked]
    sets = list(dict.fromkeys(k[0] for k in rp))
    for s in sets:
        keys = [k for k in rp if k[0] == s]
        valid = sum(1 for k in keys if k in pp and math.isfinite(rp[k][1]) and math.isfinite(pp[k][1]))
        checks.append(_check(f"valid predictions on {s}", "predictions", "minimum", valid, 1, valid >= 1,
                             reason="" if valid else "no prediction on the set is finite on both sides",
                             total=len(keys)))
        if 0 < valid < len(keys):
            notes.append(f"{len(keys) - valid} of {len(keys)} predictions on {s} are missing or non-finite")

    # ---- fit statistics
    r_fit = json.loads((baseline / "fit.json").read_text())
    rank = sum(1 for _, (e, _) in pc.items() if not math.isnan(e))
    py_fit = python_fit_stats(py_raw, rank, r_class, family)
    if "glm" in r_class:
        mapping = dict(GLM_STATS)
        if (family, None) in SAME_LIKELIHOOD or (family, link) in SAME_LIKELIHOOD:
            mapping |= LIKELIHOOD
        else:
            notes.append(f"log-likelihood, AIC and BIC are not compared: statsmodels estimates the {family} "
                         "likelihood's scale differently from R")
    elif "lm" in r_class:
        mapping = LM_STATS
    else:
        mapping = OTHER_STATS
        notes.append("AIC and BIC are not compared for this model class: R and Python may count parameters "
                     "differently; the log-likelihood is compared")
    diffs["fit_statistics"] = {}
    ft = tol["fit_statistics"]
    for r_key, py_key in mapping.items():
        r_val, py_val = r_fit.get(r_key), py_fit.get(py_key)
        if not isinstance(r_val, (int, float)) or isinstance(r_val, bool):
            notes.append(f"R reports no {r_key}; not compared")
            continue
        diffs["fit_statistics"][r_key] = {"r": r_val, "python": py_val}
        if py_val is None:
            checks.append(unavailable(f"fit {r_key}", "fit_statistics", "the Python results do not report it"))
        elif r_key in EXACT_STATS:
            checks.append(count_check(f"fit {r_key} differs", "fit_statistics", int(py_val != r_val)))
        else:
            checks.append(tolerance_check(f"fit {r_key}", "fit_statistics", [(float(r_val), float(py_val))], ft))

    # ---- evaluation metrics
    weighted, why = metric_weighting(spec)
    if why:
        notes.append(why)
    weights = {k: w for k, (_, _, w) in rp.items()}
    r_units, py_units = units_by_set(rp), units_by_set(pp, weights)
    metrics: dict[str, dict[str, dict[str, float | None]]] = {}
    for name in spec["metrics"]:
        mt = tol["per_metric"].get(name, tol["metrics"])
        if name == "aic":
            r_val, py_val = r_fit.get("aic"), py_fit.get("aic")
            metrics.setdefault("train", {})["aic"] = {"r": r_val, "python": py_val}
            if not isinstance(r_val, (int, float)):
                checks.append(unavailable("aic on train", "metrics", "R reports no AIC",
                                          required=name not in spec.get("optional_metrics", {}).get("train", [])))
            elif py_val is None:
                checks.append(unavailable("aic on train", "metrics", "Python reports no AIC",
                                          required=name not in spec.get("optional_metrics", {}).get("train", [])))
            else:
                checks.append(tolerance_check("aic on train", "metrics",
                                              [(float(r_val), math.nan if py_val is None else py_val)], mt))
            continue
        for s in sets:
            sides = {"r": metric(name, r_units.get(s, []), family, kind, weighted),
                     "python": metric(name, py_units.get(s, []), family, kind, weighted)}
            metrics.setdefault(s, {})[name] = sides
            if sides["r"] is None and sides["python"] is None:
                checks.append(unavailable(f"{name} on {s}", "metrics", f"{name} is not defined for the {s} set's "
                                          "observed values", required=name not in
                                          spec.get("optional_metrics", {}).get(s, [])))
            else:
                checks.append(tolerance_check(f"{name} on {s}", "metrics",
                                              [tuple(math.nan if v is None else v for v in sides.values())], mt))

    required = [c for c in checks if c["required"]]
    return dict(passed=all(c["passed"] for c in required), stage="parity", checks=checks, metrics=metrics,
                differences=diffs, notes=notes)
