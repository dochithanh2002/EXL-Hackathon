"""M1 independent validation of the agent-generated Poisson claim-frequency GLM.

Areas (pricing-model standard, simulated bank/insurer MRM thresholds):
  1 Predictive power  Poisson deviance loss (learn/test), deviance explained, Lorenz Gini, repeated-split loss
  2 Calibration       A/E overall and by predicted-frequency decile (exact Poisson tests), calibration slope
  3 Stability         PSI of predicted frequency learn->test, CSI per rating factor, coefficient stability across splits
  4 Specification     LR tests for dropping AreaGLM / VehBrand / Region, Poisson GBM challenger
  5 Fairness (age)    A/E and premium relativity by driver-age band; proxy test for young drivers
Outputs output/m1_fremtpl2/reports/validation/validation_results.json, reports/fairness/fairness_results.json, charts, MLflow.
Run by the Validation agent.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from agent.validators.m1_fremtpl2 import common as c

OUT, FOUT = c.OUT / "validation", c.OUT / "fairness"
RAW = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region", "Area"]
N_SPLITS = c.M.thresholds("validation")["repeated_splits"]


# Traffic lights: agent/config/thresholds.yaml -> validation / fairness .poisson_frequency
_T, _FT = c.M.thresholds("validation"), c.M.thresholds("fairness")


def rag_ae(ae):
    d = abs(ae - 1)
    return "GREEN" if d < _T["ae_dev"]["green"] else "AMBER" if d < _T["ae_dev"]["amber"] else "RED"


def refit_loss(df, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df)); n = int(round(0.9 * len(df)))
    tr, te = df.iloc[np.sort(idx[:n])], df.iloc[np.sort(idx[n:])]
    f = c.CH.fit(tr)
    return c.poisson_loss(te[c.TARGET], f.predict_df(te)), f.params


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True); FOUT.mkdir(parents=True, exist_ok=True)
    df, fit = c.load_scored()
    m = c.CH.module
    tr, te = df[df["set"] == "train"].copy(), df[df["set"] == "test"].copy()
    R = {"model_id": "M1", "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "champion": c.CH.provenance()}

    # 1 predictive power ----------------------------------------------------------------------------------
    null_te = tr[c.TARGET].sum() / tr[c.EXPO].sum() * te[c.EXPO]
    loss_tr, loss_te = c.poisson_loss(tr[c.TARGET], tr["mu"]), c.poisson_loss(te[c.TARGET], te["mu"])
    loss_null = c.poisson_loss(te[c.TARGET], null_te)
    reps = [refit_loss(df.drop(columns=["mu", "freq_hat"]), s) for s in range(N_SPLITS)]
    rep_loss = np.array([r[0] for r in reps]); rep_coef = pd.DataFrame([r[1] for r in reps])
    gini_te = c.lorenz_gini(te[c.TARGET], te["mu"], te[c.EXPO])
    R["predictive_power"] = {
        "loss_learn_x100": loss_tr, "loss_test_x100": loss_te, "loss_test_null_x100": loss_null,
        "deviance_explained_test": 1 - loss_te / loss_null, "gini_test": gini_te,
        "gini_learn": c.lorenz_gini(tr[c.TARGET], tr["mu"], tr[c.EXPO]),
        "repeated_split_loss_x100": {"n": N_SPLITS, "mean": float(rep_loss.mean()), "sd": float(rep_loss.std(ddof=1)),
                                     "min": float(rep_loss.min()), "max": float(rep_loss.max())},
        "published_case_study_glm1": {"loss_learn_x100": 31.267, "loss_test_x100": 32.171, "note": "different split (R sample() changed in 3.6)"},
        "rag": "GREEN" if loss_te < loss_null and gini_te > _T["gini_min"] else "AMBER" if loss_te < loss_null else "RED"}

    # 2 calibration ---------------------------------------------------------------------------------------
    edges = c.quantile_edges(tr["freq_hat"], 10)
    te["band"] = pd.cut(te["freq_hat"], edges, labels=False) + 1
    dec = c.ae_table(te, "band"); dec.to_csv(OUT / "ae_by_decile_test.csv", index=False)
    # calibration slope of the risk ranking: y ~ a + b log(freq_hat), offset log(Exposure)  (b = 1 is ideal)
    slope = sm.GLM(te[c.TARGET].astype(float), sm.add_constant(np.log(te["freq_hat"].values)), family=sm.families.Poisson(),
                   offset=np.log(te[c.EXPO].values)).fit()
    # offset assumption: is the claim count proportional to exposure? free the log-exposure coefficient on learn
    off_fit = sm.GLM(tr[c.TARGET].astype(float), sm.add_constant(np.column_stack([np.log(tr["freq_hat"].values), np.log(tr[c.EXPO].values)])),
                     family=sm.families.Poisson()).fit()
    ae_all = te[c.TARGET].sum() / te["mu"].sum()
    flagged = int((dec["flag"] != "ok").sum())
    R["calibration"] = {"ae_test": float(ae_all), "ae_learn": float(tr[c.TARGET].sum() / tr["mu"].sum()),
                        "calibration_slope": float(slope.params.iloc[1]), "calibration_slope_se": float(slope.bse.iloc[1]),
                        "exposure_elasticity": float(off_fit.params.iloc[2]), "exposure_elasticity_se": float(off_fit.bse.iloc[2]),
                        "exposure_note": "1.0 means claims are proportional to exposure, as the log-exposure offset assumes",
                        "deciles_flagged": flagged, "worst_decile_ae": float(dec.loc[(dec["ae"] - 1).abs().idxmax(), "ae"]),
                        "rag": "GREEN" if rag_ae(ae_all) == "GREEN" and flagged <= _T["max_deciles_flagged"] else "AMBER" if rag_ae(ae_all) != "RED" else "RED"}

    # 3 stability -----------------------------------------------------------------------------------------
    csi = {v: c.psi(tr[v], te[v], c.quantile_edges(tr[v], 10)) for v in c.NUM_INPUTS}
    csi.update({v: c.cat_psi(tr[v], te[v]) for v in c.CAT_INPUTS})
    se_ratio = (rep_coef.std(ddof=1) / fit.bse).replace([np.inf], np.nan)
    psi_score = c.psi(tr["freq_hat"], te["freq_hat"], edges)
    R["stability"] = {"psi_freq_learn_vs_test": psi_score, "csi_learn_vs_test": csi, "max_csi": max(csi.values()),
                      "coef_split_sd_over_model_se": {"median": float(se_ratio.median()), "max": float(se_ratio.max()),
                                                      "max_term": str(se_ratio.idxmax())},
                      "note": "No time dimension in freMTPL2: only learn/test and repeated random splits; out-of-time stability is untested.",
                      "rag": "AMBER" if max(psi_score, max(csi.values())) < _T["stability_psi_amber"] else "RED"}

    # 4 specification -------------------------------------------------------------------------------------
    Xtr = m.design_matrix(tr); off = np.log(tr[c.EXPO].to_numpy()); y = tr[c.TARGET].astype(float)
    def dev_without(prefix):
        cols = [k for k in Xtr.columns if not k.startswith(prefix)]
        return sm.GLM(y, Xtr[cols], family=sm.families.Poisson(), offset=off).fit(method="IRLS", tol=1e-10).deviance, len(Xtr.columns) - len(cols)
    lr = {}
    for v in ("AreaGLM", "VehBrand", "Region", "DrivAgeGLM"):
        d0, k = dev_without(v)
        lr[v] = {"LR": float(d0 - fit.deviance), "df": k, "p": float(stats.chi2.sf(d0 - fit.deviance, k))}
    Xg_tr, Xg_te = tr[RAW].copy(), te[RAW].copy()
    for col in ("VehBrand", "VehGas", "Region", "Area"):
        cats = {v: i for i, v in enumerate(sorted(tr[col].unique()))}
        Xg_tr[col], Xg_te[col] = tr[col].map(cats), te[col].map(cats)
    gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                        categorical_features=[RAW.index(k) for k in ("VehBrand", "VehGas", "Region", "Area")],
                                        random_state=0)
    gbm.fit(Xg_tr, tr[c.TARGET] / tr[c.EXPO], sample_weight=tr[c.EXPO])
    mu_gbm = gbm.predict(Xg_te) * te[c.EXPO]
    loss_gbm = c.poisson_loss(te[c.TARGET], mu_gbm)
    R["specification"] = {"lr_drop_tests": lr, "challenger_gbm_loss_test_x100": loss_gbm,
                          "challenger_gbm_gini_test": c.lorenz_gini(te[c.TARGET], mu_gbm, te[c.EXPO]),
                          "champion_minus_challenger_loss": loss_te - loss_gbm,
                          "n_nonsignificant_terms": int((2 * stats.norm.sf(np.abs(fit.params / fit.bse)) > _T["term_significance"]).sum()),
                          "rag": "AMBER" if (loss_te - loss_gbm) > _T["challenger_loss_gap"] or lr["AreaGLM"]["p"] > _T["lr_test_p"] else "GREEN"}

    # 5 fairness: driver age -------------------------------------------------------------------------------
    te["age_band"] = te["DrivAgeGLM"].map(c.AGE_BANDS)
    age = c.ae_table(te, "age_band")
    ref_freq = te.loc[te["DrivAgeGLM"] == "5", "freq_hat"].mean()
    rel = te.groupby("age_band")["freq_hat"].mean() / ref_freq
    age["premium_relativity_vs_41_50"] = age["age_band"].map(rel)
    age.to_csv(FOUT / "ae_by_driver_age_test.csv", index=False)
    big = age[age["exposure"] >= _FT["min_exposure"]]
    young = (df["DrivAge"] <= _FT["young_driver_max_age"]).astype(int)
    feats = ["BonusMalus", "VehPower", "VehAge", "Density"]
    pc = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(tr[feats], young[df["set"] == "train"])
    proxy_auc = float(roc_auc_score(young[df["set"] == "test"], pc.predict_proba(te[feats])[:, 1]))
    worst = big.loc[(big["ae"] - 1).abs().idxmax()]
    # any band whose A/E differs significantly from 1 (exact Poisson test) is flagged, small bands included
    sig = age[age["flag"] != "ok"]
    stat_rag = "GREEN" if sig.empty and ((big["ci_lo"] <= _FT["ae_band"][1]) & (big["ci_hi"] >= _FT["ae_band"][0])).all() else "AMBER"
    F = {"model_id": "M1", "timestamp_utc": R["timestamp_utc"], "champion": R["champion"],
         "protected_characteristic": "age (driver age bands are a rating factor)",
         "ae_by_age_band": age.to_dict(orient="records"),
         "bands_mispriced": [{"band": r.age_band, "ae": float(r.ae), "p_value": float(r.p_value), "relativity": float(r.premium_relativity_vs_41_50),
                              "exposure": float(r.exposure), "direction": r.flag} for r in sig.itertuples()], "worst_band": {"band": worst["age_band"], "ae": float(worst["ae"])},
         "max_premium_relativity": float(rel.max()), "max_relativity_band": str(rel.idxmax()),
         "lr_test_driver_age": lr["DrivAgeGLM"], "proxy_auc_young_driver": proxy_auc,
         "statistical_rag": stat_rag,
         "legal_rag": "AMBER",
         "legal_note": ("Age is a permitted rating factor in UK insurance only where it is based on relevant and reliable actuarial "
                        "data (Equality Act 2010 Sch.3 para 20A). The age bands are statistically significant and calibrated; the "
                        "owner must document that evidence and the FCA Consumer Duty fair-value assessment."),
         "recommendation": {"variant": "keep driver age with documented actuarial justification; monitor A/E by age band"}}
    F["overall_rag"] = "RED" if "RED" in (stat_rag, F["legal_rag"]) else "AMBER" if "AMBER" in (stat_rag, F["legal_rag"]) else "GREEN"
    R["fairness"] = {k: F[k] for k in ("statistical_rag", "legal_rag", "overall_rag", "worst_band", "proxy_auc_young_driver")}

    rags = [R[k]["rag"] for k in ("predictive_power", "calibration", "stability", "specification")] + [F["overall_rag"]]
    R["overall_rag"] = "RED" if "RED" in rags else "AMBER" if "AMBER" in rags else "GREEN"
    R["headline_metric"] = {"name": "test Poisson deviance loss x100", "value": loss_te}
    charts(dec, age, te, mu_gbm)
    (OUT / "validation_results.json").write_text(json.dumps(R, indent=2, default=float))
    (FOUT / "fairness_results.json").write_text(json.dumps(F, indent=2, default=float))
    c.log_mlflow("validation", {"stage": "validation", "overall_rag": R["overall_rag"], "champion_sha256": R["champion"]["module_sha256"]},
                 {"loss_test_x100": loss_te, "gini_test": gini_te, "ae_test": ae_all, "psi_freq": psi_score,
                  "challenger_loss_test_x100": loss_gbm, "proxy_auc_young": proxy_auc}, OUT)
    return R


def charts(dec, age, te, mu_gbm):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=c.SURFACE)
    x = dec["band"].values
    ax.bar(x - 0.2, dec["expected"] / dec["exposure"], width=0.38, color=c.C1, label="Predicted frequency")
    ax.bar(x + 0.2, dec["actual"] / dec["exposure"], width=0.38, color=c.C2, label="Observed frequency")
    c.style(ax, "Back-test by predicted-frequency decile (test)", "Decile (1 = lowest risk)", "Claims per policy-year")
    ax.set_xticks(x); ax.legend(frameon=False, fontsize=8, labelcolor=c.INK2)
    fig.tight_layout(); fig.savefig(OUT / "ae_deciles.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=c.SURFACE)
    xs = np.arange(len(age))
    ax.errorbar(xs, age["ae"], yerr=[age["ae"] - age["ci_lo"], age["ci_hi"] - age["ae"]], fmt="o", color=c.C1, ecolor=c.INK2, capsize=3)
    ax.axhline(1, color=c.INK2, lw=0.8); ax.axhspan(*_FT["ae_band"], color=c.GRID, alpha=0.5)
    ax.set_xticks(xs, age["age_band"]); c.style(ax, "Actual / expected claims by driver age (test, 95% CI)", "Driver age band", "A/E")
    fig.tight_layout(); fig.savefig(FOUT / "ae_by_driver_age.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=c.SURFACE)
    for mu, col, lab in ((te["mu"].values, c.C1, "GLM (champion)"), (np.asarray(mu_gbm), c.C3, "GBM challenger")):
        o = np.argsort(mu / te[c.EXPO].values)
        xx = np.r_[0, np.cumsum(te[c.EXPO].values[o]) / te[c.EXPO].sum()]
        yy = np.r_[0, np.cumsum(te[c.TARGET].values[o]) / te[c.TARGET].sum()]
        ax.plot(xx, yy, color=col, lw=2, label=f"{lab}  Gini {c.lorenz_gini(te[c.TARGET], mu, te[c.EXPO]):.3f}")
    ax.plot([0, 1], [0, 1], color=c.GRID, ls="--")
    c.style(ax, "Lorenz curve (test)", "Share of exposure (low → high predicted risk)", "Share of claims")
    ax.legend(frameon=False, fontsize=8, labelcolor=c.INK2)
    fig.tight_layout(); fig.savefig(OUT / "lorenz.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    r = main()
    print({k: r[k]["rag"] for k in ("predictive_power", "calibration", "stability", "specification")},
          "fairness:", r["fairness"]["overall_rag"], "overall:", r["overall_rag"])
    print("test loss x100 %.4f  null %.4f  GBM %.4f  Gini %.3f  A/E %.4f" % (
        r["predictive_power"]["loss_test_x100"], r["predictive_power"]["loss_test_null_x100"],
        r["specification"]["challenger_gbm_loss_test_x100"], r["predictive_power"]["gini_test"], r["calibration"]["ae_test"]))
