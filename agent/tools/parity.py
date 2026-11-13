"""Deterministic parity check between the R reference capture and a Python candidate run.

Missing values are compared, not skipped: a row that is NA on one side only is a mismatch, and rows that are NA
on both sides match. The deviance check uses the model family recorded in the R capture (meta.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-15


def _xlogy_ratio(y, mu):
    """y * log(y / mu) with the convention 0 * log(0) = 0."""
    out = np.zeros_like(y)
    pos = y > 0
    out[pos] = y[pos] * np.log(y[pos] / mu[pos])
    return out


def _binomial(y, mu):
    mu = np.clip(mu, EPS, 1 - EPS)
    return 2 * np.sum(_xlogy_ratio(y, mu) + _xlogy_ratio(1 - y, 1 - mu))


def _poisson(y, mu):
    mu = np.clip(mu, EPS, None)
    return 2 * np.sum(_xlogy_ratio(y, mu) - (y - mu))


def _gamma(y, mu):
    mu = np.clip(mu, EPS, None)
    return -2 * np.sum(np.log(y / mu) - (y - mu) / mu)


def _gaussian(y, mu):
    return np.sum((y - mu) ** 2)


def _inverse_gaussian(y, mu):
    return np.sum((y - mu) ** 2 / (y * mu ** 2))


# R family name -> unit-deviance sum (quasi families share the deviance of their base family)
DEVIANCE = {"binomial": _binomial, "quasibinomial": _binomial, "poisson": _poisson, "quasipoisson": _poisson,
            "Gamma": _gamma, "gaussian": _gaussian, "inverse.gaussian": _inverse_gaussian}


def _family(ref_dir: Path) -> str:
    meta = ref_dir / "meta.json"
    if not meta.exists():                       # references captured before meta.json existed were binomial
        return "binomial"
    return json.loads(meta.read_text()).get("family", "gaussian")


def compare(ref_dir: Path, cand_dir: Path, tol: dict) -> dict:
    rc = pd.read_csv(ref_dir / "coefficients.csv")
    pc = pd.read_csv(cand_dir / "coefficients.csv")
    rp = pd.read_csv(ref_dir / "predictions.csv").set_index("row").sort_index()
    pp = pd.read_csv(cand_dir / "predictions.csv").set_index("row").sort_index()
    checks, evidence = [], {}

    def add(name, value, limit, kind="max"):
        ok = bool(value <= limit) if kind == "max" else bool(value == limit)   # NaN never passes
        checks.append({"check": name, "value": float(value), "tolerance": float(limit), "pass": ok})

    # G-05 feature fidelity: identical coefficient names (order-insensitive)
    missing, extra = sorted(set(rc.term) - set(pc.term)), sorted(set(pc.term) - set(rc.term))
    add("G-05 coefficient names: missing + extra", len(missing) + len(extra), 0, "eq")
    if missing or extra:
        evidence["missing_terms"], evidence["extra_terms"] = missing, extra
    else:
        m = rc.merge(pc, on="term", suffixes=("_r", "_py"))
        absd = (m.estimate_r - m.estimate_py).abs()
        rel = absd / m.estimate_r.abs().clip(lower=1e-300)
        serel = (m.std_error_r - m.std_error_py).abs() / m.std_error_r.abs().clip(lower=1e-300)
        add("coefficient max rel diff", rel.max(skipna=False), tol["coef_rel"])
        add("std error max rel diff", serel.max(skipna=False), tol["se_rel"])
        m["rel_diff"] = rel
        evidence["worst_terms"] = (m.sort_values("rel_diff", ascending=False)
                                   .head(8)[["term", "estimate_r", "estimate_py", "rel_diff"]].to_dict("records"))

    same_rows = rp.index.equals(pp.index)
    add("rows: identical set", 0 if same_rows else 1, 0, "eq")
    if not same_rows:
        evidence["rows_only_in_r"] = sorted(set(rp.index) - set(pp.index))[:10]
        evidence["rows_only_in_py"] = sorted(set(pp.index) - set(rp.index))[:10]
        return {"pass": False, "checks": checks, "evidence": evidence}

    # response: numeric comparison (R writes the model's numeric response scale), NA == NA
    y_r = pd.to_numeric(rp["y"], errors="coerce")
    y_p = pd.to_numeric(pp["y"], errors="coerce")
    y_bad = ~((y_r == y_p) | (y_r.isna() & y_p.isna()))
    add("response values: rows differing", int(y_bad.sum()), 0, "eq")
    if y_bad.any():
        evidence["response_mismatch_rows"] = pd.DataFrame({"y_r": rp["y"], "y_py": pp["y"]})[y_bad].head(5).astype(str).to_dict("index")
    add("split membership: rows differing", int((rp["set"] != pp["set"]).sum()), 0, "eq")

    # predictions: NA pattern must match exactly, then max abs diff on rows predicted by both
    p_r, p_p = rp["pd"].astype(float), pp["pd"].astype(float)
    na_bad = p_r.isna() != p_p.isna()
    add("prediction NA pattern: rows differing", int(na_bad.sum()), 0, "eq")
    if na_bad.any():
        evidence["prediction_na_mismatch_rows"] = pd.DataFrame({"pd_r": p_r, "pd_py": p_p})[na_bad].head(5).to_dict("index")
    both = p_r.notna() & p_p.notna()
    d = (p_r[both] - p_p[both]).abs()
    add("prediction max abs diff (rows predicted by both)", d.max() if len(d) else 0.0, tol["pred_abs"])
    if len(d) and d.max() > tol["pred_abs"]:
        evidence["worst_rows"] = d.sort_values(ascending=False).head(5).round(12).to_dict()

    # deviance of R's vs Python's predictions against the same response, under the model's family
    family = _family(ref_dir)
    dev_fn = DEVIANCE.get(family)
    ok = both & y_r.notna()
    evidence["deviance_family"], evidence["deviance_rows"] = family, int(ok.sum())
    if dev_fn is None:
        evidence["deviance_skipped"] = f"no deviance formula for family '{family}'"
    else:
        y = y_r[ok].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            for nm, p in (("r", p_r[ok].to_numpy(float)), ("py", p_p[ok].to_numpy(float))):
                evidence[f"deviance_{nm}"] = float(dev_fn(y, p))
        add(f"{family} deviance abs diff", abs(evidence["deviance_r"] - evidence["deviance_py"]), tol["deviance_abs"])
    return {"pass": all(c["pass"] for c in checks), "checks": checks, "evidence": evidence}
