"""Shared helpers for M2 validation and fairness: data, model fitting, metrics, chart style.

The champion is the agent-generated Python module (agent/validators/m2_taiwan/champion.py). Challenger / ablation variants
(different term sets) are refitted here with statsmodels on the champion's own design matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from agent.validators.m2_taiwan import champion

ROOT = champion.ROOT
TARGET = champion.module().TARGET
FINAL_TERMS = list(champion.module().TERMS)
M = champion.M
SEED_SENSITIVITY = M.reports / "validation" / "seed_sensitivity.json"

# Chart style (validated reference palette, light mode).
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

SEX = {1: "Male", 2: "Female"}
EDU = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other/unknown"}
MAR = {1: "Married", 2: "Single", 3: "Other"}
AGE_BINS = [0, 25, 35, 45, 55, 200]
AGE_LABELS = ["21-25", "26-35", "36-45", "46-55", "56+"]


def load() -> pd.DataFrame:
    """Data preprocessed by the agent module, joined to the frozen R split, with readable group columns."""
    df = champion.load()
    df["sex_g"] = df["sex"].map(SEX)
    df["edu_g"] = df["education"].map(EDU)
    df["mar_g"] = df["marriage"].map(MAR)
    df["age_g"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS)
    return df


def design(df: pd.DataFrame, terms: list[str] = FINAL_TERMS) -> pd.DataFrame:
    return champion.module().design_matrix(df)[terms]


def fit_glm(train: pd.DataFrame, terms: list[str] = FINAL_TERMS):
    """Champion term set -> the agent module's own fit; any other term set -> statsmodels challenger."""
    if list(terms) == FINAL_TERMS:
        return champion.fit(train)
    return sm.GLM(train[TARGET].astype(float), design(train, terms),
                  family=sm.families.Binomial()).fit(method="IRLS", tol=1e-12)


def score(m, df: pd.DataFrame, terms: list[str] = FINAL_TERMS) -> pd.Series:
    if isinstance(m, champion.Fit):
        return m.predict_df(df)
    return m.predict(design(df, terms))


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def disc(y, p) -> dict:
    a = roc_auc_score(y, p)
    return {"auc": float(a), "gini": float(2 * a - 1), "ks": ks(y, p)}


def bootstrap_ci(y, p, fn, n=1000, seed=7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if y[ix].min() != y[ix].max():
            vals.append(fn(y[ix], p[ix]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def psi(expected, actual, bins) -> float:
    e = np.histogram(expected, bins)[0] / len(expected)
    a = np.histogram(actual, bins)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def quantile_edges(x, q=10) -> np.ndarray:
    edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def hosmer_lemeshow(y, p, g=10) -> dict:
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"], g, labels=False, duplicates="drop")
    t = d.groupby("bin").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    hl = float((((t.obs - t.exp) ** 2) / (t.exp * (1 - t.exp / t.n))).sum())
    df_ = len(t) - 2
    return {"hl_stat": hl, "df": df_, "p_value": float(1 - stats.chi2.cdf(hl, df_))}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    return float(c - h), float(c + h)


def style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
