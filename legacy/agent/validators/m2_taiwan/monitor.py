"""M2 ongoing monitoring: score each batch with the validated champion, compare with the development baseline,
rate every metric against the re-validation triggers in the validation report (§10), and raise a
re-validation trigger in the model inventory when any metric is red.

Batches
  default (no --batch): two batches built from the data we have, labelled as simulated:
    * "2005-09 cohort"  : the held-out test clients scored as at Sep-2005, with their observed default outcome
    * "2005-08 re-score": the same clients re-scored one month earlier (inputs shifted one month back); the
                          default outcome for that month is not observed, so performance checks are skipped
  --batch FILE.csv [--name NAME]: a new scoring batch in the raw input format (target column optional)

Outputs: output/m2_taiwan/reports/monitoring/{monitoring_results.json, MONITORING_REPORT.md, REVALIDATION_TRIGGER.md if red},
         output/inventory/inventory_index.json (monitoring status), one MLflow run per batch.
Usage: agent/bin/mrf monitor m2_taiwan [--batch FILE --name NAME]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from agent.validators.m2_taiwan import champion, common as c

OUT = c.M.reports / "monitoring"
from agent.tools.paths import INVENTORY, rel
INDEX = INVENTORY / "inventory_index.json"
INPUTS_NUM = ["limit_bal", "pay_0", "pay_2", "pay_3", "pay_5", "pay_6", "pay_amt1", "pay_amt2"]
INPUTS_CAT = ["sex", "education", "marriage"]
GROUPS = {"sex_g": "sex", "age_g": "age", "edu_g": "education", "mar_g": "marital status"}
# (amber, red) re-validation triggers: agent/config/thresholds.yaml -> monitoring.binary_pd
_MT = c.M.thresholds("monitoring")
MIN_GROUP = _MT["min_group_n"]
T = {k: (v["amber"], v["red"]) for k, v in _MT.items() if isinstance(v, dict)}
LOWER_IS_WORSE = {k for k, v in _MT.items() if isinstance(v, dict) and v.get("direction") == "lower_is_worse"}
_APPROVE = c.M.thresholds("fairness")["approval_rate_policy"]


def rag(metric: str, v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    a, r = T[metric]
    if metric in LOWER_IS_WORSE:
        return "RED" if v < r else "AMBER" if v < a else "GREEN"
    return "RED" if v >= r else "AMBER" if v >= a else "GREEN"


def cat_psi(dev: pd.Series, new: pd.Series) -> float:
    cats = sorted(set(dev.unique()) | set(new.unique()))
    e = np.clip(dev.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(), 1e-6, None)
    a = np.clip(new.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def baseline():
    df = c.load()
    dev = df[df["set"] == "train"].copy()
    m = c.fit_glm(dev)
    dev["pd"] = c.score(m, dev)
    V = json.loads((c.M.reports / "validation" / "validation_results.json").read_text())
    return df, dev, m, {"validated_test_auc": V["discrimination"]["test"]["auc"], "score_edges": c.quantile_edges(dev["pd"], 10),
                        "cutoff_80": float(np.quantile(dev["pd"], _APPROVE)), "validation_timestamp": V["timestamp_utc"]}


def assess(name: str, batch: pd.DataFrame, has_outcome: bool, dev: pd.DataFrame, m, base: dict, note: str) -> dict:
    b = batch.copy()
    b["pd"] = c.score(m, b)
    csi = {v: c.psi(dev[v], b[v], c.quantile_edges(dev[v], 10)) for v in INPUTS_NUM}
    csi.update({v: cat_psi(dev[v], b[v]) for v in INPUTS_CAT})
    approved = b["pd"] < base["cutoff_80"]
    dirs, gaps = {}, {}
    for g in GROUPS:
        t = b.assign(appr=approved).groupby(g, observed=True).agg(n=("appr", "size"), appr=("appr", "mean"),
                                                                 pd=("pd", "mean"), **({"obs": (c.TARGET, "mean")} if has_outcome else {}))
        t = t[t["n"] >= MIN_GROUP]
        dirs[g] = float(t["appr"].min() / t["appr"].max())
        if has_outcome:
            gaps[g] = float((100 * (t["pd"] - t["obs"])).abs().max())
    M = {"n": int(len(b)), "mean_pd": float(b["pd"].mean()), "approval_rate_at_dev_cutoff": float(approved.mean()),
         "score_psi": c.psi(dev["pd"], b["pd"], base["score_edges"]), "csi": csi, "max_csi": max(csi.values()),
         "max_csi_variable": max(csi, key=csi.get), "dir_by_group": dirs, "min_dir": min(dirs.values())}
    if has_outcome:
        y = b[c.TARGET].to_numpy()
        auc = float(roc_auc_score(y, b["pd"]))
        M.update({"observed_default_rate": float(y.mean()), "auc": auc, "auc_drop": base["validated_test_auc"] - auc,
                  "citl_pp": float(abs(100 * (b["pd"].mean() - y.mean()))), "group_cal_gap_pp": gaps,
                  "max_group_cal_gap_pp": max(gaps.values())})
    else:
        M.update({"auc": None, "auc_drop": None, "citl_pp": None, "max_group_cal_gap_pp": None})
    ratings = {k: rag(k, M.get(k)) for k in T}
    breaches = [f"{k} = {M[k]:.4f} ({r}; amber/red at {T[k][0]}/{T[k][1]})" + (f", driven by {M['max_csi_variable']}" if k == "max_csi" else "")
                for k, r in ratings.items() if r in ("RED", "AMBER")]
    overall = "RED" if "RED" in ratings.values() else "AMBER" if "AMBER" in ratings.values() else "GREEN"
    return {"batch": name, "note": note, "has_outcome": has_outcome, "metrics": M, "ratings": ratings,
            "overall": overall, "breaches": breaches, "revalidation_trigger": overall == "RED"}


def demo_batches(df: pd.DataFrame):
    from agent.validators.m2_taiwan.validate import shifted_month
    te = df[df["set"] == "test"].copy()
    yield "2005-09 cohort", te, True, "Simulated batch: held-out test clients scored as at Sep-2005, observed default outcome."
    sh = shifted_month(te)
    yield "2005-08 re-score", sh, False, ("Simulated batch: the same clients re-scored with inputs shifted one month back "
                                          "(Mar-2005 repayment status is unobserved and carried). Default outcome not yet observed.")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch"); ap.add_argument("--name")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    df, dev, m, base = baseline()
    if a.batch:
        mod = champion.module()
        b = mod.load_and_preprocess(a.batch)
        for g, col, mp in (("sex_g", "sex", c.SEX), ("edu_g", "education", c.EDU), ("mar_g", "marriage", c.MAR)):
            b[g] = b[col].map(mp)
        b["age_g"] = pd.cut(b["age"], c.AGE_BINS, labels=c.AGE_LABELS)
        batches = [(a.name or a.batch, b, c.TARGET in b.columns, f"Batch file {a.batch}")]
    else:
        batches = list(demo_batches(df))
    res = [assess(n, b, y, dev, m, base, note) for n, b, y, note in batches]
    R = {"model_id": "M2", "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "champion": champion.provenance(), "baseline": {k: v for k, v in base.items() if k != "score_edges"},
         "thresholds": T, "min_group_size": MIN_GROUP, "batches": res,
         "revalidation_required": any(r["revalidation_trigger"] for r in res)}
    (OUT / "monitoring_results.json").write_text(json.dumps(R, indent=2, default=float))
    write_report(R)
    update_inventory(R)
    log_mlflow(R)
    return R


def write_report(R: dict) -> None:
    f = lambda v, d=4: "n/a" if v is None else f"{v:.{d}f}"
    icon = {"GREEN": "🟢", "AMBER": "🟠", "RED": "🔴", "N/A": "⚪"}
    rows = []
    for k, (am, rd) in T.items():
        rows.append(f"| {k} | {am} / {rd} | " + " | ".join(f"{icon[b['ratings'][k]]} {f(b['metrics'].get(k))}" for b in R["batches"]) + " |")
    head = "| Metric | Amber / red | " + " | ".join(b["batch"] for b in R["batches"]) + " |"
    csi_vars = list(R["batches"][0]["metrics"]["csi"])
    csi_rows = [f"| {v} | " + " | ".join(f(b["metrics"]["csi"][v]) for b in R["batches"]) + " |" for v in csi_vars]
    md = [f"# Monitoring Report: M2", "",
          f"_Generated {R['timestamp_utc']} by `agent/validators/m2_taiwan/monitor.py`. Champion `{R['champion']['module']}` "
          f"(sha256 `{R['champion']['module_sha256'][:12]}`). Baseline = development (train) scores; validated test AUC "
          f"{R['baseline']['validated_test_auc']:.4f}._", "",
          f"**Re-validation required: {'YES' if R['revalidation_required'] else 'no'}**", "",
          head, "|---|---|" + "---|" * len(R["batches"]), *rows, "",
          "| Overall | | " + " | ".join(f"{icon[b['overall']]} **{b['overall']}**" for b in R["batches"]) + " |", "",
          "## Characteristic stability (CSI vs development)", "", "| Input | " + " | ".join(b["batch"] for b in R["batches"]) + " |",
          "|---|" + "---|" * len(R["batches"]), *csi_rows, "", "## Batches", ""]
    for b in R["batches"]:
        md += [f"### {b['batch']}: {b['overall']}", "", b["note"], "",
               *([f"- {x}" for x in b["breaches"]] or ["- No breaches."]), ""]
    md += ["## Notes", "",
           "- DIR is computed at the development 80%-approval cut-off for groups with n ≥ 200.",
           "- Performance metrics (AUC drop, calibration) need observed outcomes; batches without them are rated on stability and outcome fairness only.",
           "- A red rating means re-validation is required. It does not diagnose the cause: the validator triages it "
           "(for example, validation finding F3 explains a pay_0 shift as a data-coding artefact)."]
    (OUT / "MONITORING_REPORT.md").write_text("\n".join(md) + "\n")
    trig = OUT / "REVALIDATION_TRIGGER.md"
    if R["revalidation_required"]:
        trig.write_text("# Re-validation trigger: M2\n\n" + f"_Raised {R['timestamp_utc']} by monitoring._\n\n" +
                        "\n".join(f"- **{b['batch']}**: " + "; ".join(x for x in b["breaches"] if "RED" in x)
                                  for b in R["batches"] if b["revalidation_trigger"]) +
                        "\n\nOwner action: triage the breach, then re-run validation (`/mrf:run m2_taiwan`, stage validate).\n")
    elif trig.exists():
        trig.unlink()


def update_inventory(R: dict) -> None:
    idx = json.loads(INDEX.read_text()) if INDEX.exists() else {"models": {}}
    e = idx["models"].setdefault("M2", {})
    e["monitoring"] = {"last_run_utc": R["timestamp_utc"], "batches": {b["batch"]: b["overall"] for b in R["batches"]},
                       "revalidation_required": R["revalidation_required"],
                       "reasons": [x for b in R["batches"] for x in b["breaches"] if "RED" in x],
                       "report": rel(OUT / "MONITORING_REPORT.md")}
    INDEX.write_text(json.dumps(idx, indent=2))


def log_mlflow(R: dict) -> None:
    import mlflow
    from agent.tools.mlflow_setup import init_tracking
    init_tracking(c.M.experiment())
    for b in R["batches"]:
        with mlflow.start_run(run_name=f"monitoring: {b['batch']}"):
            mlflow.set_tags({"model_id": "M2", "stage": "monitoring", "batch": b["batch"], "overall_rag": b["overall"],
                             "revalidation_trigger": str(b["revalidation_trigger"]), "champion_sha256": R["champion"]["module_sha256"]})
            mlflow.log_metrics({k: v for k, v in b["metrics"].items() if isinstance(v, (int, float)) and v is not None})
            mlflow.log_metrics({f"csi_{k}": v for k, v in b["metrics"]["csi"].items()})
            mlflow.log_artifacts(str(OUT), "monitoring")


if __name__ == "__main__":
    r = main()
    for b in r["batches"]:
        print(f"{b['batch']:<18} {b['overall']:<6} " + "; ".join(b["breaches"]))
    print("RE-VALIDATION REQUIRED" if r["revalidation_required"] else "no re-validation trigger")
