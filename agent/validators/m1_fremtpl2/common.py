"""Shared helpers for M1 (Poisson claim-frequency GLM) validation: champion, Poisson metrics, chart style."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from agent.validators.champion import ROOT, Champion

CH = Champion("m1_fremtpl2", id_column="IDpol")
M = CH.m                                     # paths + config + thresholds for M1
OUT = M.reports
TARGET, EXPO = "ClaimNb", "Exposure"
AGE_BANDS = {"1": "18-20", "2": "21-25", "3": "26-30", "4": "31-40", "5": "41-50", "6": "51-70", "7": "71+"}
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_scored():
    """Agent-preprocessed data with frozen split, champion fitted on the learning rows, predictions for all rows."""
    df = CH.load()
    tr = df[df["set"] == "train"]
    fit = CH.fit(tr)
    df["mu"] = fit.predict_df(df)
    df["freq_hat"] = df["mu"] / df[EXPO]
    return df, fit


def poisson_loss(y, mu) -> float:
    """Average Poisson deviance x 100 (the case study's x10^-2 convention)."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    ylog = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(100 * 2 * np.mean(ylog - (y - mu)))


def lorenz_gini(y, mu, expo) -> float:
    """Concentration (Lorenz) Gini: policies ordered by predicted frequency; 0 = no discrimination."""
    o = np.argsort(np.asarray(mu) / np.asarray(expo), kind="mergesort")
    x = np.r_[0, np.cumsum(np.asarray(expo)[o]) / np.sum(expo)]
    c = np.r_[0, np.cumsum(np.asarray(y)[o]) / np.sum(y)]
    return float(1 - 2 * np.trapezoid(c, x))


def ae_table(d: pd.DataFrame, by: str) -> pd.DataFrame:
    t = d.groupby(by, observed=True).agg(policies=(TARGET, "size"), exposure=(EXPO, "sum"),
                                         actual=(TARGET, "sum"), expected=("mu", "sum"))
    t["ae"] = t["actual"] / t["expected"]
    t["ci_lo"] = stats.chi2.ppf(0.025, 2 * t["actual"]) / 2 / t["expected"]
    t["ci_hi"] = stats.chi2.ppf(0.975, 2 * (t["actual"] + 1)) / 2 / t["expected"]
    t["p_value"] = [2 * min(stats.poisson.cdf(a, e), stats.poisson.sf(a - 1, e), 0.5) for a, e in zip(t["actual"], t["expected"])]
    t["flag"] = np.where(t["p_value"] < 0.05, np.where(t["ae"] > 1, "UNDER-predicts", "over-predicts"), "ok")
    return t.reset_index()


def psi(expected, actual, edges) -> float:
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def cat_psi(dev: pd.Series, new: pd.Series) -> float:
    cats = sorted(set(dev.dropna().unique()) | set(new.dropna().unique()))
    e = np.clip(dev.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(), 1e-6, None)
    a = np.clip(new.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def quantile_edges(x, q=10) -> np.ndarray:
    e = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    e[0], e[-1] = -np.inf, np.inf
    return e


NUM_INPUTS = ["BonusMalusGLM", "DensityGLM", "AreaGLM"]
CAT_INPUTS = ["VehPowerGLM", "VehAgeGLM", "DrivAgeGLM", "VehBrand", "VehGas", "Region"]


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9); ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.set_facecolor(SURFACE)


def log_mlflow(run_name: str, tags: dict, metrics: dict, artifacts_dir=None) -> None:
    import os
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from agent.tools.mlflow_setup import init_tracking
    init_tracking(CH.cfg["tracking"]["experiment"])
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"model_id": "M1", **{k: str(v) for k, v in tags.items()}})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        if artifacts_dir is not None:
            mlflow.log_artifacts(str(artifacts_dir), artifacts_dir.name)
