"""M1 ongoing monitoring: score batches with the validated champion, compare with the learning-sample baseline,
rate against the re-validation triggers in the validation report (§10), raise a trigger in the inventory when red.

Default batches (simulated from data we have, labelled as such):
  * "test cohort"           : the 67,801 held-out policies with observed claims
  * "young-driver growth"   : the same policies re-weighted so drivers aged 18-25 make up 3x their share
                              (a plausible portfolio-mix shift after a marketing campaign)
--batch FILE.csv [--name NAME]: a real batch in the raw freMTPL2 format (ClaimNb optional).
Usage: agent/bin/mrf monitor m1_fremtpl2
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from agent.validators.m1_fremtpl2 import common as c

OUT = c.OUT / "monitoring"
from agent.tools.paths import INVENTORY, rel
INDEX = INVENTORY / "inventory_index.json"
# (amber, red) re-validation triggers: agent/config/thresholds.yaml -> monitoring.poisson_frequency
_MT = c.M.thresholds("monitoring")
T = {k: (v["amber"], v["red"]) for k, v in _MT.items() if isinstance(v, dict)}
MIN_EXPOSURE = _MT["min_exposure"]


def rag(k, v):
    if v is None:
        return "N/A"
    a, r = T[k]
    return "RED" if v >= r else "AMBER" if v >= a else "GREEN"


def assess(name, b, has_y, dev, base, note):
    csi = {v: c.psi(dev[v], b[v], c.quantile_edges(dev[v], 10)) for v in c.NUM_INPUTS}
    csi.update({v: c.cat_psi(dev[v], b[v]) for v in c.CAT_INPUTS})
    M = {"policies": int(len(b)), "exposure": float(b[c.EXPO].sum()), "mean_pred_freq": float(b["freq_hat"].mean()),
         "freq_psi": c.psi(dev["freq_hat"], b["freq_hat"], base["edges"]), "csi": csi, "max_csi": max(csi.values()),
         "max_csi_variable": max(csi, key=csi.get), "young_driver_share": float((b["DrivAge"] <= 25).mean())}
    if has_y:
        ae = b[c.TARGET].sum() / b["mu"].sum()
        loss = c.poisson_loss(b[c.TARGET], b["mu"])
        b2 = b.assign(age_band=b["DrivAgeGLM"].map(c.AGE_BANDS))
        t = c.ae_table(b2, "age_band"); t = t[t["exposure"] >= MIN_EXPOSURE]
        M.update({"ae": float(ae), "ae_dev": float(abs(ae - 1)), "loss_x100": loss,
                  "loss_increase_rel": float(max(0.0, loss / base["validated_loss"] - 1)),
                  "age_band_ae": dict(zip(t["age_band"], t["ae"].astype(float))),
                  "max_age_band_ae_dev": float((t["ae"] - 1).abs().max())})
    else:
        M.update({"ae_dev": None, "loss_increase_rel": None, "max_age_band_ae_dev": None})
    ratings = {k: rag(k, M.get(k)) for k in T}
    breaches = [f"{k} = {M[k]:.4f} ({r}; amber/red at {T[k][0]}/{T[k][1]})" + (f", driven by {M['max_csi_variable']}" if k == "max_csi" else "")
                for k, r in ratings.items() if r in ("AMBER", "RED")]
    overall = "RED" if "RED" in ratings.values() else "AMBER" if "AMBER" in ratings.values() else "GREEN"
    return {"batch": name, "note": note, "has_outcome": has_y, "metrics": M, "ratings": ratings, "overall": overall,
            "breaches": breaches, "revalidation_trigger": overall == "RED"}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--batch"); ap.add_argument("--name")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    df, fit = c.load_scored()
    dev, te = df[df["set"] == "train"], df[df["set"] == "test"]
    V = json.loads((c.OUT / "validation" / "validation_results.json").read_text())
    base = {"edges": c.quantile_edges(dev["freq_hat"], 10), "validated_loss": V["predictive_power"]["loss_test_x100"]}
    if a.batch:
        b = c.CH.module.load_and_preprocess(a.batch)
        b["mu"] = fit.predict_df(b); b["freq_hat"] = b["mu"] / b[c.EXPO]
        batches = [(a.name or a.batch, b, c.TARGET in b.columns, f"Batch file {a.batch}")]
    else:
        young = te[te["DrivAge"] <= 25]
        shifted = pd.concat([te, young, young], ignore_index=True)
        batches = [("test cohort", te, True, "Simulated batch: held-out policies with observed claims."),
                   ("young-driver growth", shifted, True, "Simulated scenario: test policies with drivers aged 18-25 tripled "
                                                          "(portfolio-mix shift); observed claims of those policies reused.")]
    res = [assess(n, b, y, dev, base, note) for n, b, y, note in batches]
    R = {"model_id": "M1", "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "champion": c.CH.provenance(),
         "baseline": {"validated_loss_x100": base["validated_loss"]}, "thresholds": T, "batches": res,
         "revalidation_required": any(r["revalidation_trigger"] for r in res)}
    (OUT / "monitoring_results.json").write_text(json.dumps(R, indent=2, default=float))
    icon = {"GREEN": "🟢", "AMBER": "🟠", "RED": "🔴", "N/A": "⚪"}
    f = lambda v: "n/a" if v is None else f"{v:.4f}"
    md = ["# Monitoring Report: M1", "", f"_Generated {R['timestamp_utc']}. Champion `{R['champion']['module']}` "
          f"(sha256 `{R['champion']['module_sha256'][:12]}`). Baseline = learning sample; validated test loss {base['validated_loss']:.4f}._", "",
          f"**Re-validation required: {'YES' if R['revalidation_required'] else 'no'}**", "",
          "| Metric | Amber / red | " + " | ".join(b["batch"] for b in res) + " |", "|---|---|" + "---|" * len(res),
          *[f"| {k} | {t[0]} / {t[1]} | " + " | ".join(f"{icon[b['ratings'][k]]} {f(b['metrics'].get(k))}" for b in res) + " |" for k, t in T.items()],
          "| Overall | | " + " | ".join(f"{icon[b['overall']]} **{b['overall']}**" for b in res) + " |", ""]
    for b in res:
        md += [f"## {b['batch']}: {b['overall']}", "", b["note"], "", *([f"- {x}" for x in b["breaches"]] or ["- No breaches."]), ""]
    (OUT / "MONITORING_REPORT.md").write_text("\n".join(md) + "\n")
    trig = OUT / "REVALIDATION_TRIGGER.md"
    if R["revalidation_required"]:
        trig.write_text("# Re-validation trigger: M1\n\n" + "\n".join(f"- **{b['batch']}**: " + "; ".join(x for x in b["breaches"] if "RED" in x)
                                                                  for b in res if b["revalidation_trigger"]) + "\n")
    elif trig.exists():
        trig.unlink()
    idx = json.loads(INDEX.read_text()) if INDEX.exists() else {"models": {}}
    idx["models"].setdefault("M1", {})["monitoring"] = {"last_run_utc": R["timestamp_utc"], "batches": {b["batch"]: b["overall"] for b in res},
                                                        "revalidation_required": R["revalidation_required"],
                                                        "reasons": [x for b in res for x in b["breaches"] if "RED" in x],
                                                        "report": rel(OUT / "MONITORING_REPORT.md")}
    INDEX.write_text(json.dumps(idx, indent=2))
    for b in res:
        c.log_mlflow(f"monitoring: {b['batch']}", {"stage": "monitoring", "batch": b["batch"], "overall_rag": b["overall"]},
                     {k: v for k, v in b["metrics"].items() if isinstance(v, (int, float))})
    return R


if __name__ == "__main__":
    r = main()
    for b in r["batches"]:
        print(f"{b['batch']:<22} {b['overall']:<6} " + "; ".join(b["breaches"]))
    print("RE-VALIDATION REQUIRED" if r["revalidation_required"] else "no re-validation trigger")
