"""Shared approval and standalone-bundle comparison of a candidate with the frozen R baseline, evolved from
`legacy/agent/tools/parity.py`. Standard library only: this file is also shipped unchanged in deliverables bundles.

Missing values are compared, not skipped: a value missing on one side only is a mismatch, missing on both a match.
KPIs are computed here, by the same code for both sides, from each side's response and predictions."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

EPS = 1e-15
EXACT_STATS = {"nobs", "df_residual"}
# R fit.json key -> how to read it from the candidate's statsmodels results (fit.json written by the harness)
LM_STATS = {"nobs": "nobs", "df_residual": "df_resid", "rss": "ssr", "sigma": "sigma", "r_squared": "rsquared",
            "adj_r_squared": "rsquared_adj", "log_lik": "llf", "aic": "aic", "bic": "bic"}
GLM_STATS = {"nobs": "nobs", "df_residual": "df_resid", "deviance": "deviance", "null_deviance": "null_deviance",
             "dispersion": "scale", "pearson_chi2": "pearson_chi2"}
LIKELIHOOD = {"log_lik": "llf", "aic": "aic", "bic": "bic"}
# families whose statsmodels log-likelihood is defined as R's logLik.glm (other families estimate the scale
# differently, so their likelihood statistics are not compared)
SAME_LIKELIHOOD = {("binomial", None), ("poisson", None), ("gaussian", "identity")}
ESTIMATED_SCALE = {"gaussian", "Gamma", "inverse.gaussian"}


# ---------------------------------------------------------------- reading
def _num(s: str | None) -> float:
    return float(s) if s not in (None, "", "NA") else math.nan


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _coefs(path: Path) -> dict[str, tuple[float, float]]:
    return {r["term"]: (_num(r["estimate"]), _num(r["std_error"])) for r in _rows(path)}


def _predictions(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    return {(r["set"], int(r["row"])): (_num(r["y"]), _num(r["response"])) for r in _rows(path)}


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def _rel(r: float, py: float) -> float:
    if _same(r, py):
        return 0.0
    if math.isnan(r) or math.isnan(py) or r == 0:
        return math.inf
    return abs(r - py) / abs(r)


# ---------------------------------------------------------------- KPIs
def _auc(pairs: list[tuple[float, float]]) -> float | None:
    """Mann-Whitney: the probability a positive scores above a negative, ties counting half."""
    if not pairs or any(y not in (0.0, 1.0) for y, _ in pairs):
        return None
    n1 = sum(1 for y, _ in pairs if y == 1.0)
    n0 = len(pairs) - n1
    if n1 == 0 or n0 == 0:
        return None
    ordered = sorted(pairs, key=lambda p: p[1])
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        mid = (i + j) / 2 + 1                                       # average 1-based rank of the tie group
        rank_sum += mid * sum(1 for k in range(i, j + 1) if ordered[k][0] == 1.0)
        i = j + 1
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def _ylogy(y: float, mu: float) -> float:
    return y * math.log(y / mu) if y > 0 else 0.0


def _deviance(family: str, pairs: list[tuple[float, float]]) -> float | None:
    base = {"quasibinomial": "binomial", "quasipoisson": "poisson"}.get(family, family)
    try:
        if base == "binomial":
            return 2 * sum(_ylogy(y, m) + _ylogy(1 - y, 1 - m)
                           for y, mu in pairs for m in [min(max(mu, EPS), 1 - EPS)])
        if base == "poisson":
            return 2 * sum(_ylogy(y, m) - (y - m) for y, mu in pairs for m in [max(mu, EPS)])
        if base == "Gamma":
            return -2 * sum(math.log(y / m) - (y - m) / m for y, mu in pairs for m in [max(mu, EPS)])
        if base == "gaussian":
            return sum((y - mu) ** 2 for y, mu in pairs)
        if base == "inverse.gaussian":
            return sum((y - mu) ** 2 / (y * mu ** 2) for y, mu in pairs)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def kpi(name: str, family: str, pairs: list[tuple[float, float]]) -> float | None:
    """One KPI over (response, prediction) pairs; None where it is not defined for this response."""
    if not pairs:
        return None
    n = len(pairs)
    if name in ("auc", "gini"):
        auc = _auc(pairs)
        return None if auc is None else auc if name == "auc" else 2 * auc - 1
    if name == "actual_expected":
        expected = sum(mu for _, mu in pairs)
        return sum(y for y, _ in pairs) / expected if expected else None
    if name == "rmse":
        return math.sqrt(sum((y - mu) ** 2 for y, mu in pairs) / n)
    if name == "mae":
        return sum(abs(y - mu) for y, mu in pairs) / n
    if name == "log_loss":
        if any(y not in (0.0, 1.0) for y, _ in pairs):
            return None
        return -sum(math.log(m) if y == 1.0 else math.log(1 - m)
                    for y, mu in pairs for m in [min(max(mu, EPS), 1 - EPS)]) / n
    if name == "deviance":
        return _deviance(family, pairs)
    raise ValueError(f"unknown KPI {name}")


# ---------------------------------------------------------------- comparison
def python_fit_stats(raw: dict, rank: int, r_class: list[str], family: str) -> dict[str, float | None]:
    """The harness's statsmodels statistics, plus AIC and BIC counted the way R counts parameters."""
    stats = dict(raw)
    llf, nobs, scale = raw.get("llf"), raw.get("nobs"), raw.get("scale")
    k = rank + (1 if "glm" not in r_class or family in ESTIMATED_SCALE else 0)
    stats["aic"] = None if llf is None else -2 * llf + 2 * k
    stats["bic"] = None if llf is None or not nobs else -2 * llf + math.log(nobs) * k
    stats["sigma"] = None if scale is None or scale < 0 else math.sqrt(scale)
    return stats


def compare(baseline: Path, candidate: Path, spec: dict) -> dict:
    tol = spec["tolerances"]
    meta = json.loads((baseline / "meta.json").read_text())
    family, link, r_class = meta["family"], meta["link"], meta["class"]
    checks: list[dict] = []
    diffs: dict = {}
    notes: list[str] = []

    def add(name: str, value: float | None, tolerance: float, exact: bool = False) -> None:
        ok = value is not None and (value == tolerance if exact else value <= tolerance)   # NaN never passes
        checks.append(dict(name=name, value=value, tolerance=tolerance, passed=bool(ok)))

    # ---- coefficients
    rc, pc = _coefs(baseline / "coefficients.csv"), _coefs(candidate / "coefficients.csv")
    missing, extra = [t for t in rc if t not in pc], [t for t in pc if t not in rc]
    add("coefficient names missing or extra", len(missing) + len(extra), 0, exact=True)
    if missing or extra:
        diffs["missing_terms"], diffs["extra_terms"] = missing, extra
    else:
        rel = {t: (_rel(rc[t][0], pc[t][0]), _rel(rc[t][1], pc[t][1])) for t in rc}
        add("coefficient max rel diff", max(r for r, _ in rel.values()), tol["coef_rel"])
        add("std error max rel diff", max(s for _, s in rel.values()), tol["se_rel"])
        worst = sorted(rc, key=lambda t: max(rel[t]), reverse=True)[:8]
        diffs["worst_terms"] = [{"term": t, "estimate_r": rc[t][0], "estimate_python": pc[t][0],
                                 "estimate_rel": rel[t][0],
                                 "std_error_r": rc[t][1], "std_error_python": pc[t][1], "std_error_rel": rel[t][1]}
                                for t in worst]

    # ---- rows, response and predictions
    rp, pp = _predictions(baseline / "predictions.csv"), _predictions(candidate / "predictions.csv")
    only_r, only_py = rp.keys() - pp.keys(), pp.keys() - rp.keys()
    add("rows only on one side", len(only_r) + len(only_py), 0, exact=True)
    if only_r or only_py:
        diffs["rows_only_in_r"] = sorted(only_r)[:10]
        diffs["rows_only_in_python"] = sorted(only_py)[:10]
    common = sorted(rp.keys() & pp.keys())
    y_bad = [k for k in common if not _same(rp[k][0], pp[k][0])]
    add("response values differing", len(y_bad), 0, exact=True)
    if y_bad:
        diffs["response_differs_rows"] = y_bad[:10]
    na_bad = [k for k in common if math.isnan(rp[k][1]) != math.isnan(pp[k][1])]
    add("prediction NA pattern differing", len(na_bad), 0, exact=True)
    if na_bad:
        diffs["prediction_na_differs_rows"] = [{"set": s, "row": r, "r_is_na": math.isnan(rp[(s, r)][1])}
                                               for s, r in na_bad[:10]]
    both = [(abs(rp[k][1] - pp[k][1]), k) for k in common if not (math.isnan(rp[k][1]) or math.isnan(pp[k][1]))]
    add("prediction max abs diff", max((d for d, _ in both), default=0.0), tol["pred_abs"])
    diffs["worst_predictions"] = [{"set": k[0], "row": k[1], "r": rp[k][1], "python": pp[k][1], "abs_diff": d}
                                  for d, k in sorted(both, reverse=True)[:5]]

    # ---- fit statistics
    r_fit = json.loads((baseline / "fit.json").read_text())
    rank = sum(1 for _, (e, _) in pc.items() if not math.isnan(e))
    py_fit = python_fit_stats(json.loads((candidate / "fit.json").read_text()), rank, r_class, family)
    if "glm" in r_class:
        mapping = dict(GLM_STATS)
        if (family, None) in SAME_LIKELIHOOD or (family, link) in SAME_LIKELIHOOD:
            mapping |= LIKELIHOOD
        else:
            notes.append(f"log-likelihood, AIC and BIC are not compared: statsmodels estimates the {family} "
                         "likelihood's scale differently from R")
    else:
        mapping = LM_STATS
    diffs["fit_statistics"] = {}
    for r_key, py_key in mapping.items():
        r_val, py_val = r_fit.get(r_key), py_fit.get(py_key)
        if not isinstance(r_val, (int, float)) or isinstance(r_val, bool):
            notes.append(f"R reports no {r_key}; not compared")
            continue
        diffs["fit_statistics"][r_key] = {"r": r_val, "python": py_val}
        if r_key in EXACT_STATS:
            add(f"fit {r_key} differs", None if py_val is None else float(py_val != r_val), 0, exact=True)
        else:
            add(f"fit {r_key} rel diff", None if py_val is None else _rel(float(r_val), float(py_val)),
                tol["fit_stat_rel"])

    # ---- KPIs
    kpis: dict[str, dict[str, dict[str, float | None]]] = {}
    for name in spec["kpis"]:
        if name == "aic":
            r_val, py_val = r_fit.get("aic"), py_fit.get("aic")
            kpis.setdefault("train", {})["aic"] = {"r": r_val, "python": py_val}
            if isinstance(r_val, (int, float)):
                add("kpi aic train abs diff", None if py_val is None else abs(r_val - py_val), tol["kpi_abs"])
            continue
        for s in dict.fromkeys(k[0] for k in rp):
            sides = {}
            for side, preds in (("r", rp), ("python", pp)):
                pairs = [(y, mu) for (st, _), (y, mu) in preds.items()
                         if st == s and not (math.isnan(y) or math.isnan(mu))]
                sides[side] = kpi(name, family, pairs)
            kpis.setdefault(s, {})[name] = sides
            if sides["r"] is None and sides["python"] is None:
                notes.append(f"{name} is not defined for the {s} set's response; not compared")
            else:
                add(f"kpi {name} {s} abs diff", None if None in sides.values() else abs(sides["r"] - sides["python"]),
                    tol["kpi_abs"])

    return dict(passed=all(c["passed"] for c in checks), stage="parity", checks=checks, kpis=kpis,
                differences=diffs, notes=notes)
