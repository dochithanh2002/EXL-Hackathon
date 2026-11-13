"""M2 fairness assessment by sex, age, education (and marital status).

Favourable outcome = credit APPROVED (PD below cut-off). Two cut-offs:
  * policy PD 0.50
  * approve 80% (cut-off = 80th percentile of training PD)
Per group (test set): approval rate, disparate-impact ratio (4/5ths rule), statistical parity
difference, equal opportunity (approval among good payers), approval among defaulters,
AUC, within-group calibration.

Ablation: refit without protected inputs (A: baseline, B: -sex -marriage, C: -sex -marriage -education)
to show the accuracy / fairness / calibration trade-off, plus a proxy test (can the remaining
inputs predict sex or age?).

Outputs -> output/m2_taiwan/reports/fairness/ + MLflow run. Run by the Validation agent.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from agent.validators.m2_taiwan import common as c

OUT = c.M.reports / "fairness"
ATTRS = {"sex_g": "Sex", "age_g": "Age band", "edu_g": "Education", "mar_g": "Marital status"}
ORDER = {"sex_g": list(c.SEX.values()), "age_g": c.AGE_LABELS, "edu_g": list(c.EDU.values()),
         "mar_g": list(c.MAR.values())}
_FT = c.M.thresholds("fairness")              # agent/config/thresholds.yaml -> fairness.binary_pd
SMALL_N = _FT["small_group_n"]
SHORT = {"Graduate school": "Grad", "University": "Univ", "High school": "High sch", "Other/unknown": "Other/unk"}
DIR_MIN, CAL_GAP_PP = _FT["dir_min"], _FT["cal_gap_pp"]

VARIANTS = {
    "A_baseline": c.FINAL_TERMS,
    "B_no_sex_marriage": [t for t in c.FINAL_TERMS if not t.startswith(("sex", "marriage"))],
    "C_no_sex_marriage_education": [t for t in c.FINAL_TERMS if not t.startswith(("sex", "marriage", "education"))],
}
NON_PROTECTED = ["limit_bal", "pay_0", "pay_2", "pay_3", "pay_5", "pay_6", "pay_amt1", "pay_amt2"]


def _safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def group_table(te: pd.DataFrame, attr: str, cut: float) -> pd.DataFrame:
    y = te[c.TARGET].values
    approve = (te["pd"] < cut).astype(int).values
    metrics = {
        "n": lambda yt, yp: len(yt),
        "default_rate": lambda yt, yp: yt.mean(),
        "approval_rate": lambda yt, yp: yp.mean(),
        "approval_good_payers": lambda yt, yp: yp[yt == 0].mean(),     # equal opportunity
        "approval_defaulters": lambda yt, yp: yp[yt == 1].mean(),      # false approval rate
        "bad_rate_approved": lambda yt, yp: yt[yp == 1].mean() if yp.sum() else np.nan,
    }
    mf = MetricFrame(metrics=metrics, y_true=y, y_pred=approve, sensitive_features=te[attr].astype(str))
    t = mf.by_group
    cal = te.groupby(te[attr].astype(str), observed=True).apply(
        lambda d: pd.Series({"mean_pd": d["pd"].mean(), "auc": _safe_auc(d[c.TARGET], d["pd"])}), include_groups=False)
    t = t.join(cal)
    t["cal_gap_pp"] = 100 * (t["mean_pd"] - t["default_rate"])
    ci = [c.wilson(int(round(a * n)), int(n)) for a, n in zip(t["approval_rate"], t["n"])]
    t["approval_ci_lo"], t["approval_ci_hi"] = [x[0] for x in ci], [x[1] for x in ci]
    t["dir_vs_best"] = t["approval_rate"] / t["approval_rate"].max()
    t["small_group"] = t["n"] < SMALL_N
    t = t.reindex([g for g in ORDER[attr] if g in t.index])
    t.index.name = "group"
    return t


def summarise(t: pd.DataFrame) -> dict:
    big = t[~t["small_group"]]
    return {
        "disparate_impact_ratio": float(t["approval_rate"].min() / t["approval_rate"].max()),
        "disparate_impact_ratio_excl_small": float(big["approval_rate"].min() / big["approval_rate"].max()),
        "statistical_parity_diff": float(t["approval_rate"].max() - t["approval_rate"].min()),
        "equal_opportunity_diff": float(t["approval_good_payers"].max() - t["approval_good_payers"].min()),
        "equalised_odds_diff": float(max(t["approval_good_payers"].max() - t["approval_good_payers"].min(),
                                         t["approval_defaulters"].max() - t["approval_defaulters"].min())),
        "auc_range": float(t["auc"].max() - t["auc"].min()),
        "max_abs_cal_gap_pp": float(t["cal_gap_pp"].abs().max()),
        "max_abs_cal_gap_pp_excl_small": float(big["cal_gap_pp"].abs().max()),
        "lowest_approval_group": str(t["approval_rate"].idxmin()),
        "highest_approval_group": str(t["approval_rate"].idxmax()),
        "worst_calibrated_group": str(t["cal_gap_pp"].abs().idxmax()),
        "small_groups": [str(g) for g in t.index[t["small_group"]]],
    }


def flags(s: dict) -> list[str]:
    out = []
    if s["disparate_impact_ratio_excl_small"] < DIR_MIN:
        out.append(f"DIR {s['disparate_impact_ratio_excl_small']:.3f} < {DIR_MIN}")
    if s["max_abs_cal_gap_pp_excl_small"] > CAL_GAP_PP:
        out.append(f"calibration gap {s['max_abs_cal_gap_pp_excl_small']:.1f}pp > {CAL_GAP_PP}pp ({s['worst_calibrated_group']})")
    return out


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    df = c.load()
    tr, te = df[df["set"] == "train"].copy(), df[df["set"] == "test"].copy()
    R = {"model_id": "M2", "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "favourable_outcome": "approved (PD < cut-off)", "champion": c.champion.provenance(), "variants": {}}

    for vname, terms in VARIANTS.items():
        m = c.fit_glm(tr, terms)
        tr_v, te_v = tr.copy(), te.copy()
        tr_v["pd"], te_v["pd"] = c.score(m, tr_v, terms), c.score(m, te_v, terms)
        cuts = {"pd_0.50": _FT["pd_cutoff"], "approve_80pct": float(np.quantile(tr_v["pd"], _FT["approval_rate_policy"]))}
        V = {"terms": terms, "test_auc": float(roc_auc_score(te_v[c.TARGET], te_v["pd"])), "cutoffs": cuts, "attrs": {}}
        for cname, cut in cuts.items():
            for attr in ATTRS:
                t = group_table(te_v, attr, cut)
                t.to_csv(OUT / f"{vname}__{cname}__{attr}.csv")
                s = summarise(t)
                V["attrs"].setdefault(attr, {})[cname] = {"summary": s, "flags": flags(s),
                                                         "table": t.reset_index().to_dict(orient="records")}
        R["variants"][vname] = V
        if vname == "A_baseline":
            te_base = te_v

    # Proxy test: can non-protected model inputs predict the protected attribute?
    proxy = {}
    for label, target in (("sex=female", (df["sex"] == 2).astype(int)),
                          ("age<=25", (df["age"] <= 25).astype(int)),
                          ("age>=56", (df["age"] >= 56).astype(int)),
                          ("marital=single", (df["marriage"] == 2).astype(int))):
        g = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        g.fit(tr[NON_PROTECTED], target[df["set"] == "train"])
        proxy[label] = float(roc_auc_score(target[df["set"] == "test"], g.predict_proba(te[NON_PROTECTED])[:, 1]))
    R["proxy_auc"] = proxy

    # Headline: baseline at the 80%-approval cut-off.
    R["headline"] = {a: {"summary": R["variants"]["A_baseline"]["attrs"][a]["approve_80pct"]["summary"],
                         "flags": R["variants"]["A_baseline"]["attrs"][a]["approve_80pct"]["flags"]} for a in ATTRS}
    R["statistical_rag"] = ("RED" if any(R["headline"][a]["flags"] for a in ("sex_g", "age_g", "edu_g")) else "GREEN")
    # Direct use of a protected characteristic is a legal/policy issue regardless of outcome metrics
    # (UK Equality Act 2010 s.13 direct discrimination; FCA Consumer Duty fair-value outcome).
    direct = [t for t in c.FINAL_TERMS if t.startswith(("sex", "marriage"))]
    R["legal_rag"] = "RED" if direct else "GREEN"
    R["direct_protected_inputs"] = direct
    A, B = R["variants"]["A_baseline"], R["variants"]["B_no_sex_marriage"]
    R["recommendation"] = {
        "variant": "B_no_sex_marriage",
        "auc_change": B["test_auc"] - A["test_auc"],
        "sex_dir_change": (B["attrs"]["sex_g"]["approve_80pct"]["summary"]["disparate_impact_ratio"]
                           - A["attrs"]["sex_g"]["approve_80pct"]["summary"]["disparate_impact_ratio"]),
        "new_flags": {a: B["attrs"][a]["approve_80pct"]["flags"] for a in ATTRS if B["attrs"][a]["approve_80pct"]["flags"]},
    }
    R["overall_rag"] = "RED" if "RED" in (R["statistical_rag"], R["legal_rag"]) else "GREEN"

    charts(R)
    (OUT / "fairness_results.json").write_text(json.dumps(R, indent=2, default=float))
    log_mlflow(R)
    return R


def charts(R: dict) -> None:
    base = R["variants"]["A_baseline"]["attrs"]
    # 1. Approval rate by group (small multiples, 80% approval cut-off) with 4/5ths reference line
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.8), facecolor=c.SURFACE, layout="constrained",
                             gridspec_kw={"width_ratios": [2, 5, 4, 3]})
    for ax, attr in zip(axes, ATTRS):
        t = pd.DataFrame(base[attr]["approve_80pct"]["table"])
        x = np.arange(len(t))
        err = np.vstack([100 * (t["approval_rate"] - t["approval_ci_lo"]), 100 * (t["approval_ci_hi"] - t["approval_rate"])])
        ax.bar(x, 100 * t["approval_rate"], color=[c.GRID if s else c.C1 for s in t["small_group"]], width=0.7)
        ax.errorbar(x, 100 * t["approval_rate"], yerr=err, fmt="none", ecolor=c.INK2, lw=1, capsize=2)
        ref = 100 * DIR_MIN * t["approval_rate"].max()
        ax.axhline(ref, color=c.C2, lw=1.2, ls="--")
        for xi, v, small in zip(x, t["approval_rate"], t["small_group"]):
            ax.text(xi, 100 * v / 2, f"{100 * v:.0f}", ha="center", color=c.INK if small else "white",
                    fontsize=8, fontweight="bold")
        ax.set_xticks(x, [SHORT.get(g, g) for g in t["group"]], fontsize=7)
        ax.set_ylim(0, 105)
        c.style(ax, ATTRS[attr], "", "Approval rate (%)" if attr == "sex_g" else "")
    fig.supxlabel("Dashed line: 4/5ths of best group (disparate-impact threshold). Grey bars: n < 200. Whiskers: 95% CI.",
                  fontsize=7, color=c.INK2, x=0.99, ha="right")
    fig.suptitle("Approval rate by group (80% overall approval cut-off, test set)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=c.INK)
    fig.savefig(OUT / "approval_by_group.png", dpi=160); plt.close(fig)

    # 2. Within-group calibration gap: baseline vs ablation C
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.0), facecolor=c.SURFACE, layout="constrained",
                             gridspec_kw={"width_ratios": [2, 5, 4]})
    for ax, attr in zip(axes, ["sex_g", "age_g", "edu_g"]):
        ta = pd.DataFrame(R["variants"]["A_baseline"]["attrs"][attr]["approve_80pct"]["table"])
        tc = pd.DataFrame(R["variants"]["C_no_sex_marriage_education"]["attrs"][attr]["approve_80pct"]["table"])
        x = np.arange(len(ta))
        ax.bar(x - 0.2, ta["cal_gap_pp"], width=0.38, color=c.C1, label="Baseline (uses sex, marriage, education)")
        ax.bar(x + 0.2, tc["cal_gap_pp"], width=0.38, color=c.C2, label="Without protected inputs")
        ax.axhline(0, color=c.INK2, lw=0.8)
        for b in (CAL_GAP_PP, -CAL_GAP_PP):
            ax.axhline(b, color=c.INK2, lw=0.8, ls=":")
        ax.set_xticks(x, [SHORT.get(g, g) for g in ta["group"]], fontsize=7)
        c.style(ax, ATTRS[attr], "", "Mean PD − observed rate (pp)" if attr == "sex_g" else "")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="outside lower left",
               ncol=2, labelcolor=c.INK2)
    fig.suptitle("Within-group calibration (positive = over-predicts risk). Dotted: ±2pp tolerance",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=c.INK)
    fig.savefig(OUT / "calibration_by_group.png", dpi=160); plt.close(fig)


def log_mlflow(R: dict) -> None:
    import mlflow
    from agent.tools.mlflow_setup import init_tracking
    init_tracking(c.M.experiment())
    with mlflow.start_run(run_name="fairness"):
        mlflow.set_tags({"model_id": "M2", "stage": "fairness", "overall_rag": R["overall_rag"],
                         "statistical_rag": R["statistical_rag"], "legal_rag": R["legal_rag"]})
        for v, V in R["variants"].items():
            mlflow.log_metric(f"{v}__test_auc", V["test_auc"])
            for a in ATTRS:
                s = V["attrs"][a]["approve_80pct"]["summary"]
                mlflow.log_metric(f"{v}__{a}__dir", s["disparate_impact_ratio"])
                mlflow.log_metric(f"{v}__{a}__max_cal_gap_pp", s["max_abs_cal_gap_pp"])
        for k, v in R["proxy_auc"].items():
            mlflow.log_metric(f"proxy_auc__{k.replace('=', '_').replace('<', 'lt').replace('>', 'gt')}", v)
        mlflow.log_artifacts(str(OUT), "fairness")


if __name__ == "__main__":
    r = main()
    for v, V in r["variants"].items():
        print(f"\n== {v}  test AUC {V['test_auc']:.4f}")
        for a in ATTRS:
            for cn in ("pd_0.50", "approve_80pct"):
                s = V["attrs"][a][cn]["summary"]
                print(f"  {a:6} {cn:14} DIR {s['disparate_impact_ratio']:.3f} (excl small {s['disparate_impact_ratio_excl_small']:.3f}) "
                      f"SPD {s['statistical_parity_diff']:.3f} EOd {s['equal_opportunity_diff']:.3f} "
                      f"AUCrange {s['auc_range']:.3f} calgap {s['max_abs_cal_gap_pp_excl_small']:.2f}pp ({s['worst_calibrated_group']}) "
                      f"flags {V['attrs'][a][cn]['flags']}")
    print("\nproxy AUC:", r["proxy_auc"], "\nstatistical:", r["statistical_rag"], "legal:", r["legal_rag"], "overall:", r["overall_rag"],
          "\nrecommendation:", r["recommendation"])
