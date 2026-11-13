"""M2 independent parity re-check: the validator re-runs the AGENT-GENERATED module against the frozen R reference.

The Migration Agent already proved parity inside its own harness. The validator does not take that on trust:
it loads the packaged module (agent/validators/m2_taiwan/champion.py), checks its SHA-256 against the agent's manifest, refits it
on the frozen R split, and compares with the R reference outputs using its own code and tolerances.
A third implementation (scikit-learn, different optimiser) confirms the maximum-likelihood solution.

Writes output/m2_taiwan/reports/parity/parity_results.json + parity_report_M2.md + coefficient_comparison.csv, logs to MLflow.
Run by the Validation agent: agent/bin/mrf validate m2_taiwan
"""
from __future__ import annotations

from agent.tools.paths import thresholds
import json
import sys
from datetime import datetime, timezone

from agent.validators.champion import fitted_stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from agent.validators.m2_taiwan import champion as ch

ROOT = ch.ROOT
OUT = ch.M.reports / "parity"

_T = thresholds()["validator_parity"]          # agent/config/thresholds.yaml
TOL = {**{k: _T[k] for k in ("coef_abs", "coef_rel", "se_rel", "pred_abs", "deviance_abs")},
       "sklearn_coef_rel": _T["crosscheck_coef_rel"], "sklearn_pred_abs": _T["crosscheck_pred_abs"]}


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def rel_diff(a, b):
    return np.abs(a - b) / np.maximum(np.abs(a), 1e-300)


def check(name, value, tol) -> dict:
    return {"check": name, "value": float(value), "tolerance": tol, "pass": bool(value <= tol)}


def sklearn_crosscheck(df: pd.DataFrame, terms: list[str]) -> tuple[pd.Series, np.ndarray]:
    """Unpenalised sklearn fit on standardised X, back-transformed to the raw scale."""
    m = ch.module()
    tr = df[df["set"] == "train"]
    X = m.design_matrix(tr).drop(columns="(Intercept)")
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=1000)
    lr.fit(sc.transform(X), tr[m.TARGET])
    beta = lr.coef_[0] / sc.scale_
    coefs = pd.Series(np.r_[lr.intercept_[0] - np.sum(beta * sc.mean_), beta], index=terms)
    return coefs, 1 / (1 + np.exp(-(m.design_matrix(df).values @ coefs.values)))


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    m, ref = ch.module(), ch.reference_dir()
    prov = ch.provenance()
    rc = pd.read_csv(ref / "coefficients.csv").set_index("term")
    rp = pd.read_csv(ref / "predictions.csv").set_index("row").sort_index()
    meta = json.loads((ref / "meta.json").read_text())

    df = ch.load()
    tr = df[df["set"] == "train"]
    res = m.fit(tr)
    fst = fitted_stats(m, res, tr, ch._C.family)          # deviance recomputed by the validator
    pc = m.coefficients(res).set_index("term")
    pp = pd.Series(m.predict(res, df), index=np.arange(1, len(df) + 1))
    pc = pc.loc[rc.index]

    k = len(rc)
    checks = [
        check("artefact: module SHA-256 differs from agent manifest", int(prov["module_sha256"] != prov["manifest_module_sha256"]), 0),
        check("term names/order differ from R model.matrix", int(list(rc.index) != list(m.coefficients(res)["term"])), 0),
        check("split: rows with different train/test membership", int((rp["set"].values != df["set"].values).sum()), 0),
        check("target: rows with different y", int((rp["y"].values != df[m.TARGET].values).sum()), 0),
        check("coef max abs diff", np.max(np.abs(rc["estimate"] - pc["estimate"])), TOL["coef_abs"]),
        check("coef max rel diff", np.max(rel_diff(rc["estimate"].values, pc["estimate"].values)), TOL["coef_rel"]),
        check("std error max rel diff", np.max(rel_diff(rc["std_error"].values, pc["std_error"].values)), TOL["se_rel"]),
        check("PD max abs diff (all 30,000 rows)", np.max(np.abs(rp["pd"].values - pp.values)), TOL["pred_abs"]),
        check("residual deviance abs diff", abs(meta["deviance"] - fst["deviance"]), TOL["deviance_abs"]),
        check("AIC abs diff", abs(meta["aic"] - (fst["deviance"] + 2 * k)), TOL["deviance_abs"]),
    ]
    te = rp["set"] == "test"
    yt, pr, pp_t = rp.loc[te, "y"].values, rp.loc[te, "pd"].values, pp[te.values].values
    auc_r, auc_p, ks_r, ks_p = roc_auc_score(yt, pr), roc_auc_score(yt, pp_t), ks(yt, pr), ks(yt, pp_t)
    checks += [check("test AUC abs diff", abs(auc_r - auc_p), 1e-12), check("test KS abs diff", abs(ks_r - ks_p), 1e-12)]
    boundary = {}
    cut80 = float(np.quantile(rp.loc[~te, "pd"], 0.80))
    for label, cut in (("PD 0.50", 0.5), ("80% approval", cut80)):
        near = (np.abs(pr - cut) <= TOL["pred_abs"]) | (np.abs(pp_t - cut) <= TOL["pred_abs"])
        diff = ((pr >= cut) != (pp_t >= cut)) & ~near
        boundary[f"{label} {cut:.6f}"] = int(near.sum())
        checks.append(check(f"decisions differing at {label} cut-off {cut:.4f} (excl. {int(near.sum())} exact-tie rows)", int(diff.sum()), 0))

    sk_coef, sk_pd = sklearn_crosscheck(df, list(rc.index))
    sk = [check("sklearn coef max rel diff vs R", np.max(rel_diff(rc["estimate"].values, sk_coef.values)), TOL["sklearn_coef_rel"]),
          check("sklearn PD max abs diff vs R", np.max(np.abs(rp["pd"].values - sk_pd)), TOL["sklearn_pred_abs"])]

    state = json.loads((ch.M.mig_dir / "state.json").read_text())
    agent_checks = state["iterations"][-1].get("parity", {}).get("checks", [])

    coef_table = pd.DataFrame({"R estimate": rc["estimate"], "Python estimate": pc["estimate"],
                               "abs diff": (rc["estimate"] - pc["estimate"]).abs(),
                               "rel diff": rel_diff(rc["estimate"].values, pc["estimate"].values),
                               "R SE": rc["std_error"], "Python SE": pc["std_error"]})
    coef_table.to_csv(OUT / "coefficient_comparison.csv", float_format="%.17g")
    results = {
        "model_id": "M2", "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "champion": prov,
        "reference": {"language": "R", "version": meta["r_version"], "engine": "stats::glm (IRLS)", "seed": meta["seed"],
                      "script": ch.cfg()["legacy"]["script"]},
        "rebuild": {"language": "Python", "engine": "agent-generated IRLS mirroring glm.fit", "module": prov["module"],
                    "iterations": fst["iterations"], "converged": fst["converged"]},
        "metrics": {"auc_test_r": auc_r, "auc_test_py": auc_p, "ks_test_r": ks_r, "ks_test_py": ks_p,
                    "deviance_r": meta["deviance"], "deviance_py": fst["deviance"]},
        "tolerances": TOL, "boundary_tie_rows": boundary, "checks": checks,
        "independent_crosscheck_sklearn": sk, "agent_reported_checks": agent_checks,
        "parity_pass": all(c["pass"] for c in checks),
    }
    (OUT / "parity_results.json").write_text(json.dumps(results, indent=2))
    write_report(results, coef_table)
    log_mlflow(results)
    return results


def write_report(r: dict, coef_table: pd.DataFrame) -> None:
    fmt = lambda v: f"{v:.3e}" if isinstance(v, float) and v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4) else f"{v:g}"
    tbl = lambda rows: ["| Check | Observed | Tolerance | Pass |", "|---|---|---|---|"] + \
        [f"| {c['check']} | {fmt(c['value'])} | {fmt(c['tolerance'])} | {'✅' if c['pass'] else '❌'} |" for c in rows]
    p, m = r["champion"], r["metrics"]
    lines = [
        "# M2 Parity Re-check: R `glm` → agent-generated Python module", "",
        f"_Generated {r['timestamp_utc']} by `agent/validators/m2_taiwan/parity.py` (validator, independent of the agent's own harness)_", "",
        f"**Result: {'✅ PASS' if r['parity_pass'] else '❌ FAIL'}**: {sum(c['pass'] for c in r['checks'])}/{len(r['checks'])} checks within tolerance.", "",
        f"- Validated artefact: `{p['module']}` (sha256 `{p['module_sha256'][:16]}`, manifest {'matches' if p['module_sha256'] == p['manifest_module_sha256'] else 'DOES NOT MATCH'})",
        f"- Generated by: Migration Agent ({p.get('migrated_by') or 'agent'}); migration status {p['migration_status']}",
        f"- Reference: {r['reference']['version']}, legacy script `{r['reference']['script']}` run unmodified, split seed {r['reference']['seed']}",
        f"- Refit by the validator: {r['rebuild']['iterations']} IRLS iterations, converged {r['rebuild']['converged']}", "",
        "## Validator checks", "", *tbl(r["checks"]), "",
        "## Independent cross-check (scikit-learn, different optimiser)", "", *tbl(r["independent_crosscheck_sklearn"]), "",
        "## Agent-reported parity (for comparison)", "", *tbl(r["agent_reported_checks"]), "",
        "## Performance (test set)", "", "| | R | Python |", "|---|---|---|",
        f"| AUC | {m['auc_test_r']:.10f} | {m['auc_test_py']:.10f} |", f"| KS | {m['ks_test_r']:.10f} | {m['ks_test_py']:.10f} |",
        f"| Residual deviance | {m['deviance_r']:.8f} | {m['deviance_py']:.8f} |", "",
        "## Coefficients", "", "| Term | R | Python | abs diff | rel diff |", "|---|---|---|---|---|",
        *[f"| {t} | {row['R estimate']:.12g} | {row['Python estimate']:.12g} | {row['abs diff']:.2e} | {row['rel diff']:.2e} |"
          for t, row in coef_table.iterrows()], "",
        "## Notes", "",
        "- The validator re-runs the packaged module itself; it does not reuse the agent's parity code or outputs.",
        "- Rows within 1e-10 of a cut-off are counted as exact ties and excluded from the decision check.",
        "- The original notebook recorded no seed; the capture harness fixed seed 2026 and froze the split. "
        "Split-to-split variation is quantified in `validation/seed_sensitivity.json`.",
    ]
    (OUT / "parity_report_M2.md").write_text("\n".join(lines) + "\n")


def log_mlflow(r: dict) -> None:
    import mlflow
    from agent.tools.mlflow_setup import init_tracking
    init_tracking(ch.M.experiment())
    with mlflow.start_run(run_name="parity_recheck"):
        mlflow.set_tags({"model_id": "M2", "stage": "parity_recheck", "parity_pass": str(r["parity_pass"]),
                         "champion_sha256": r["champion"]["module_sha256"]})
        mlflow.log_params({"reference_seed": r["reference"]["seed"], **{f"tol_{k}": v for k, v in r["tolerances"].items()}})
        mlflow.log_metrics(r["metrics"])
        mlflow.log_artifacts(str(OUT), "parity")


if __name__ == "__main__":
    res = main()
    for c in res["checks"] + res["independent_crosscheck_sklearn"]:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']:<70} {c['value']:.3e}  (tol {c['tolerance']:.0e})")
    print("PARITY:", "PASS" if res["parity_pass"] else "FAIL")
    sys.exit(0 if res["parity_pass"] else 1)
