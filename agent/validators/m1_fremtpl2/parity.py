"""M1 independent parity re-check: the validator re-runs the agent-generated Poisson module against the frozen R reference.

Checks the module hash against the agent manifest, refits on the frozen learning rows with the validator's own code,
and compares coefficients, standard errors, every prediction and the Poisson deviance with R. statsmodels (a different
IRLS implementation, fully converged) confirms the maximum-likelihood solution independently.
Run by the Validation agent: agent/bin/mrf validate m1_fremtpl2
"""
from __future__ import annotations

from agent.tools.paths import thresholds
import json
import sys
from datetime import datetime, timezone

from agent.validators.champion import fitted_stats
import numpy as np
import pandas as pd
import statsmodels.api as sm

from agent.validators.m1_fremtpl2 import common as c

OUT = c.OUT / "parity"
_T = thresholds()["validator_parity"]          # agent/config/thresholds.yaml
TOL = {**{k: _T[k] for k in ("coef_abs", "coef_rel", "se_rel", "pred_abs", "deviance_abs")},
       "statsmodels_coef_rel": _T["crosscheck_coef_rel"], "statsmodels_pred_abs": _T["crosscheck_pred_abs"]}


def check(name, value, tol):
    return {"check": name, "value": float(value), "tolerance": tol, "pass": bool(value <= tol)}


def rel(a, b):
    return np.abs(a - b) / np.maximum(np.abs(a), 1e-300)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    m, ref, prov = c.CH.module, c.CH.reference_dir, c.CH.provenance()
    rc = pd.read_csv(ref / "coefficients.csv").set_index("term")
    rp = pd.read_csv(ref / "predictions.csv").set_index("row").sort_index()
    meta = json.loads((ref / "meta.json").read_text())
    df = c.CH.load()
    tr = df[df["set"] == "train"]
    res = m.fit(tr)
    fst = fitted_stats(m, res, tr, c.CH.family)          # deviance recomputed by the validator
    pc = m.coefficients(res).set_index("term")
    pp = pd.Series(m.predict(res, df), index=np.asarray(df.index) + 1)
    checks = [
        check("artefact: module SHA-256 differs from agent manifest", int(prov["module_sha256"] != prov["manifest_module_sha256"]), 0),
        check("term names/order differ from R model.matrix", int(list(rc.index) != list(pc.index)), 0),
        check("split: rows with different learn/test membership", int((rp["set"].values != df["set"].values).sum()), 0),
        check("response: rows with different ClaimNb", int((rp["y"].values != df[c.TARGET].values).sum()), 0),
        check("coef max abs diff", np.max(np.abs(rc["estimate"] - pc.loc[rc.index, "estimate"])), TOL["coef_abs"]),
        check("coef max rel diff", np.max(rel(rc["estimate"].values, pc.loc[rc.index, "estimate"].values)), TOL["coef_rel"]),
        check("std error max rel diff", np.max(rel(rc["std_error"].values, pc.loc[rc.index, "std_error"].values)), TOL["se_rel"]),
        check("expected claims max abs diff (all 678,013 rows)", np.max(np.abs(rp["pd"].values - pp.values)), TOL["pred_abs"]),
        check("residual deviance abs diff", abs(meta["deviance"] - fst["deviance"]), TOL["deviance_abs"]),
        check("AIC-equivalent: deviance + 2k abs diff", abs((meta["deviance"] + 2 * len(rc)) - (fst["deviance"] + 2 * len(pc))), TOL["deviance_abs"]),
    ]
    te = rp["set"] == "test"
    loss_r, loss_p = c.poisson_loss(rp.loc[te, "y"], rp.loc[te, "pd"]), c.poisson_loss(rp.loc[te, "y"], pp[te.values])
    checks.append(check("test Poisson loss abs diff", abs(loss_r - loss_p), 1e-10))

    X = m.design_matrix(tr)
    sm_fit = sm.GLM(tr[c.TARGET].astype(float), X, family=sm.families.Poisson(),
                    offset=np.log(tr[c.EXPO].astype(float))).fit(method="IRLS", tol=1e-12)
    sm_pred = np.exp(m.design_matrix(df).to_numpy() @ sm_fit.params.to_numpy() + np.log(df[c.EXPO].to_numpy()))
    xc = [check("statsmodels coef max rel diff vs R", np.max(rel(rc["estimate"].values, sm_fit.params[rc.index].values)), TOL["statsmodels_coef_rel"]),
          check("statsmodels expected claims max abs diff vs R", np.max(np.abs(rp["pd"].values - sm_pred)), TOL["statsmodels_pred_abs"])]
    state = json.loads((c.M.mig_dir / "state.json").read_text())
    agent_checks = state["iterations"][-1].get("parity", {}).get("checks", [])
    ct = pd.DataFrame({"R estimate": rc["estimate"], "Python estimate": pc.loc[rc.index, "estimate"],
                       "rel diff": rel(rc["estimate"].values, pc.loc[rc.index, "estimate"].values),
                       "R SE": rc["std_error"], "Python SE": pc.loc[rc.index, "std_error"]})
    ct.to_csv(OUT / "coefficient_comparison.csv", float_format="%.17g")
    R = {"model_id": "M1", "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "champion": prov,
         "reference": {"version": meta["r_version"], "seed": meta["seed"], "script": c.CH.cfg["legacy"]["script"]},
         "rebuild": {"iterations": fst["iterations"], "converged": fst["converged"]},
         "metrics": {"test_loss_r": loss_r, "test_loss_py": loss_p, "deviance_r": meta["deviance"], "deviance_py": fst["deviance"]},
         "tolerances": TOL, "checks": checks, "independent_crosscheck_statsmodels": xc, "agent_reported_checks": agent_checks,
         "migration_iterations": prov["migration_iterations"], "parity_pass": all(x["pass"] for x in checks)}
    (OUT / "parity_results.json").write_text(json.dumps(R, indent=2))
    fmt = lambda v: f"{v:.3e}" if isinstance(v, float) and v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4) else f"{v:g}"
    tbl = lambda rows: ["| Check | Observed | Tolerance | Pass |", "|---|---|---|---|"] + \
        [f"| {x['check']} | {fmt(x['value'])} | {fmt(x['tolerance'])} | {'✅' if x['pass'] else '❌'} |" for x in rows]
    md = ["# M1 Parity Re-check: R `glm` (Poisson, offset) → agent-generated Python module", "",
          f"_Generated {R['timestamp_utc']} by `agent/validators/m1_fremtpl2/parity.py`_", "",
          f"**Result: {'✅ PASS' if R['parity_pass'] else '❌ FAIL'}** ({sum(x['pass'] for x in checks)}/{len(checks)}). "
          f"Module `{prov['module']}` sha256 `{prov['module_sha256'][:16]}`; the agent reached parity in {prov['migration_iterations']} iterations.", "",
          "## Validator checks", "", *tbl(checks), "", "## Independent cross-check (statsmodels, fully converged)", "", *tbl(xc), "",
          "## Agent-reported parity", "", *tbl(agent_checks), "",
          "## Coefficients", "", "| Term | R | Python | rel diff |", "|---|---|---|---|",
          *[f"| {t} | {r['R estimate']:.12g} | {r['Python estimate']:.12g} | {r['rel diff']:.2e} |" for t, r in ct.iterrows()]]
    (OUT / "parity_report_M1.md").write_text("\n".join(md) + "\n")
    c.log_mlflow("parity_recheck", {"stage": "parity_recheck", "parity_pass": R["parity_pass"], "champion_sha256": prov["module_sha256"]},
                 R["metrics"], OUT)
    return R


if __name__ == "__main__":
    r = main()
    for x in r["checks"] + r["independent_crosscheck_statsmodels"]:
        print(f"{'PASS' if x['pass'] else 'FAIL'}  {x['check']:<55} {x['value']:.3e}  (tol {x['tolerance']:.0e})")
    print("PARITY:", "PASS" if r["parity_pass"] else "FAIL")
    sys.exit(0 if r["parity_pass"] else 1)
