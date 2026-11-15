"""Deterministic tools for the Validation and Monitoring agents, plus status, human approval and the run summary.

    mrf list                         models in input/ and where each one stands
    mrf status    <model>            stage-by-stage status (JSON)
    mrf validate  <model>            runs model.yaml `validation.steps` from agent/validators/<suite>/
    mrf monitor   <model> [--batch FILE.csv --name NAME]
    mrf summary   <model>            output/<model>/reports/RUN_SUMMARY.md, inventory index, MLflow
    mrf approve   <model> --target migration|documentation --reviewer NAME --decision approve|reject [--comment]
    mrf reset     <model> --stage intake|documentation|migration     (archives the stage's logs, never deletes)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from agent.tools.paths import AGENT, INVENTORY, ROOT, Model, Stage, list_models, now, rel, sha

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def jload(p) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _run(m: Model, st: Stage, cmd: list[str]) -> int:
    t0 = time.time()
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONPATH": str(ROOT)}
    print(f"  $ {' '.join(c.replace(str(ROOT) + '/', '') for c in cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-8:]
    for line in tail:
        print(f"    {line}")
    rec = {"cmd": " ".join(cmd).replace(sys.executable, "python").replace(str(ROOT) + "/", ""), "exit": r.returncode,
           "seconds": round(time.time() - t0, 1), "tail": tail}
    st.state.setdefault("steps", []).append(rec)
    st.save()
    st.audit("step", **{k: rec[k] for k in ("cmd", "exit", "seconds")})
    return r.returncode


def _step_cmd(m: Model, step: str, extra: list[str] | None = None) -> list[str]:
    suite = m.cfg["validation"]["suite"]
    d = AGENT / "validators" / suite
    if (d / f"{step}.R").exists():                         # R steps get <data> <reports/validation>
        return ["Rscript", str(d / f"{step}.R"), str(m.data), str(m.reports / "validation")]
    if not (d / f"{step}.py").exists():
        raise SystemExit(f"validation step '{step}' not found in {rel(d)}")
    return [sys.executable, "-m", f"agent.validators.{suite}.{step}", *(extra or [])]


def validate(m: Model) -> int:
    mig = Stage(m, "migration")
    if not mig.done("package"):
        raise SystemExit("no packaged module: the Migration agent must finish first (mrf package)")
    st = Stage(m, "validation")
    st.state = {"stages": {}, "started_utc": now(), "steps": []}
    for step in m.cfg["validation"]["steps"]:
        if rc := _run(m, st, _step_cmd(m, step)):
            st.state["status"] = f"FAILED at {step}"
            st.save()
            return rc
    st.state["status"] = "DONE"
    st.mark("validate", steps=m.cfg["validation"]["steps"])
    print(json.dumps(_results(m)["validation"], indent=1, default=str))
    return 0


def monitor(m: Model, batch: str | None = None, name: str | None = None) -> int:
    if not (m.reports / "validation").exists():
        raise SystemExit("validate the model first (mrf validate)")
    st = Stage(m, "monitoring")
    st.state = {"stages": {}, "started_utc": now(), "steps": []}
    extra = (["--batch", batch] if batch else []) + (["--name", name] if name else [])
    for step in m.cfg["validation"].get("monitoring_steps", ["monitor"]):
        if rc := _run(m, st, _step_cmd(m, step, extra)):
            return rc
    st.state["status"] = "DONE"
    st.mark("monitor")
    print(json.dumps(_results(m)["monitoring"], indent=1, default=str))
    return 0


def _results(m: Model) -> dict:
    R = m.reports
    P, V = jload(R / "parity" / "parity_results.json"), jload(R / "validation" / "validation_results.json")
    F, M = jload(R / "fairness" / "fairness_results.json"), jload(R / "monitoring" / "monitoring_results.json")
    cur = sha(m.module_path) if m.module_path.exists() else None
    stale = sorted(n for n, d in (("parity", P), ("validation", V), ("fairness", F), ("monitoring", M))
                   if d and d.get("champion", {}).get("module_sha256") not in (None, cur))
    return {
        "stale_results_for_older_module": stale,
        "parity_recheck": {"pass": P.get("parity_pass"), "checks_passed": sum(c["pass"] for c in P.get("checks", [])),
                           "checks": len(P.get("checks", [])), "report": rel(R / "parity" / f"parity_report_{m.id}.md")},
        "validation": {"overall_rag": V.get("overall_rag"), "date_utc": V.get("timestamp_utc"),
                       "areas": {k: v["rag"] for k, v in V.items() if isinstance(v, dict) and "rag" in v},
                       "headline": V.get("headline_metric") or ({"name": "test AUC", "value": V["discrimination"]["test"]["auc"]}
                                                               if "discrimination" in V else None),
                       "report": rel(R / f"validation_report_{m.id}.html")},
        "fairness": {"overall_rag": F.get("overall_rag"), "statistical_rag": F.get("statistical_rag"), "legal_rag": F.get("legal_rag"),
                     "recommended_variant": F.get("recommendation", {}).get("variant")},
        "monitoring": {"revalidation_required": M.get("revalidation_required"),
                       "batches": {b["batch"]: b["overall"] for b in M.get("batches", [])},
                       "report": rel(R / "monitoring" / "MONITORING_REPORT.md")},
    }


def status(m: Model) -> dict:
    out = {"model": m.slug, "model_id": m.id, "name": m.cfg["model_name"], "input": rel(m.input), "output": rel(m.out),
           "inputs_changed_since_intake": m.changed_inputs()}
    for name in ("intake", "documentation", "migration", "validation", "monitoring"):
        s = jload(m.logs / name / "state.json")
        out[name] = {"status": s.get("status") or ("not started" if not s else "in progress"),
                     "done": [k for k, v in s.get("stages", {}).items() if v.get("status") == "done"],
                     **({"iterations": [i.get("verdict") for i in s["iterations"]]} if s.get("iterations") else {}),
                     **({"escalation": s["escalation"]} if s.get("escalation") else {}),
                     **({"human_review": s["human_review"]} if s.get("human_review") else {})}
    out["results"] = _results(m)
    return out


def list_all() -> list[dict]:
    rows = []
    for slug in list_models():
        s = status(Model(slug))
        rows.append({"model": slug, "id": s["model_id"], "name": s["name"],
                     **{k: s[k]["status"] for k in ("documentation", "migration", "validation", "monitoring")},
                     "validation_rag": s["results"]["validation"]["overall_rag"]})
    return rows


def approve(m: Model, target: str, reviewer: str, decision: str, comment: str) -> int:
    st = Stage(m, target)
    if st.state.get("status") != "AWAITING_HUMAN_APPROVAL":
        raise SystemExit(f"cannot {decision} {target}: status is {st.state.get('status')}")
    files = m.model_dir if target == "migration" else m.reports / "documentation"
    rec = {"reviewer": reviewer, "decision": decision, "comment": comment, "ts_utc": now(),
           "artefact_sha256": {rel(p): sha(p) for p in sorted(files.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}}
    st.state["human_review"] = rec
    st.state["status"] = "APPROVED" if decision == "approve" else "REJECTED"
    st.save()
    st.audit("human_review", **{k: v for k, v in rec.items() if k != "artefact_sha256"})
    from agent.tools import tracking
    tracking.tag_review(m, st, st.state.get("mlflow_run_id"), rec)
    print(f"{target}: {st.state['status']} by {reviewer}")
    return 0


def reset(m: Model, stage: str) -> int:
    d = m.logs / stage
    if not d.exists():
        print(f"nothing to reset: {rel(d)} does not exist")
        return 0
    arch = m.logs / "_archive" / f"{stage}_{time.strftime('%Y%m%dT%H%M%S')}"
    arch.parent.mkdir(parents=True, exist_ok=True)
    d.rename(arch)
    with open(m.logs / "audit.jsonl", "a") as f:          # logged under the stage that was reset
        f.write(json.dumps({"ts_utc": now(), "stage": stage, "event": "reset", "archived_to": rel(arch)}) + "\n")
    print(f"archived {rel(d)} -> {rel(arch)}")
    return 0


def summary(m: Model) -> dict:
    s = status(m)
    res = s["results"]
    doc, mig = s["documentation"], s["migration"]
    S = {"model_id": m.id, "model": m.slug, "generated_utc": now(), "documentation": doc, "migration": mig, **res,
         "awaiting_human": [x for x, st in (("model card", doc["status"]), ("migrated module", mig["status"]))
                            if st == "AWAITING_HUMAN_APPROVAL"],
         "validation_steps": jload(m.logs / "validation" / "state.json").get("steps", []),
         "monitoring_steps": jload(m.logs / "monitoring" / "state.json").get("steps", [])}
    (m.logs / "run_summary.json").write_text(json.dumps(S, indent=2, default=str))
    (m.reports / "RUN_SUMMARY.md").write_text(render(m, S))
    idx_f = INVENTORY / "inventory_index.json"
    INVENTORY.mkdir(parents=True, exist_ok=True)
    idx = jload(idx_f) or {"models": {}}
    e = idx["models"].setdefault(m.id, {})
    e.update({"input": rel(m.input), "output": rel(m.out), "documentation_status": doc["status"],
              "migration_status": mig["status"], "python_module": rel(m.module_path),
              "parity_recheck_pass": res["parity_recheck"]["pass"],
              "validation": {"overall_rag": res["validation"]["overall_rag"], "fairness_rag": res["fairness"]["overall_rag"],
                             "report": res["validation"]["report"], "date_utc": res["validation"]["date_utc"]},
              "last_run_summary": rel(m.reports / "RUN_SUMMARY.md")})
    idx_f.write_text(json.dumps(idx, indent=2))
    _portfolio(idx)
    _log_mlflow(m, S)
    print(f"-> {rel(m.reports / 'RUN_SUMMARY.md')}; waiting on a human: {', '.join(S['awaiting_human']) or 'nothing'}")
    return S


def render(m: Model, S: dict) -> str:
    v, f, mo, p = S["validation"], S["fairness"], S["monitoring"], S["parity_recheck"]
    mig_it = S["migration"].get("iterations", [])
    rows = [
        ("Document", S["documentation"]["status"], rel(m.reports / "documentation" / f"MODEL_CARD_{m.id}.md")),
        ("Migrate", f"{mig_it[-1] if mig_it else '-'} in {len(mig_it)} iteration(s); {S['migration']['status']}", rel(m.module_path)),
        ("Validate: parity re-check", f"{'PASS' if p['pass'] else 'FAIL' if p['pass'] is False else 'not run'} {p['checks_passed']}/{p['checks']}", p["report"]),
        ("Validate: performance", (f"overall {v['overall_rag']} (" + ", ".join(f"{k} {x}" for k, x in v["areas"].items()) + ")"
                                   + (f"; {v['headline']['name']} {v['headline']['value']:.4f}" if v["headline"] else "")) if v["overall_rag"] else "not run", v["report"]),
        ("Validate: fairness", f"overall {f['overall_rag']} (statistical {f['statistical_rag']}, legal {f['legal_rag']}); recommended: {f['recommended_variant']}",
         rel(m.reports / "fairness" / "fairness_results.json")),
        ("Monitor", f"re-validation required: {mo['revalidation_required']}; " + ", ".join(f"{k} {x}" for k, x in mo["batches"].items()), mo["report"]),
    ]
    steps = "\n".join(f"| `{s['cmd']}` | {s['exit']} | {s['seconds']} |" for s in S["validation_steps"] + S["monitoring_steps"])
    return f"""# Run summary: {m.id} {m.cfg['model_name']}

_Generated {S['generated_utc']} by `mrf summary {m.slug}`. Input `{rel(m.input)}`, output `{rel(m.out)}`.
Thresholds: `agent/config/thresholds.yaml`. Full audit trail: `{rel(m.logs / 'audit.jsonl')}`._

| Stage | Result | Artefact |
|---|---|---|
""" + "\n".join(f"| {a} | {b} | `{c}` |" for a, b, c in rows) + f"""

**Waiting on a human:** {', '.join(S['awaiting_human']) or 'nothing'}.

## Deterministic steps run by the Validation and Monitoring agents
| Command | Exit | Seconds |
|---|---|---|
{steps}
"""


def _portfolio(idx: dict) -> None:
    rows = [[mid, e.get("model_name", ""), f"`{e.get('input', '')}`", e.get("documentation_status") or e.get("status", ""),
             e.get("migration_status", ""), (e.get("validation") or {}).get("overall_rag", ""),
             (e.get("validation") or {}).get("fairness_rag", ""),
             ("RE-VALIDATE" if (e.get("monitoring") or {}).get("revalidation_required") else "ok" if e.get("monitoring") else "")]
            for mid, e in sorted(idx["models"].items())]
    head = ["Id", "Model", "Input", "Documentation", "Migration", "Validation", "Fairness", "Monitoring"]
    md = "# Model inventory\n\n_Generated " + now() + " from `inventory_index.json`._\n\n" + "\n".join(
        ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"
    (INVENTORY / "INVENTORY.md").write_text(md)


def _log_mlflow(m: Model, S: dict) -> None:
    try:
        import mlflow

        from agent.tools.mlflow_setup import init_tracking
        init_tracking(m.experiment())
        with mlflow.start_run(run_name="run_summary"):
            mlflow.set_tags({"model_id": m.id, "stage": "summary", "migration_status": S["migration"]["status"],
                             "documentation_status": S["documentation"]["status"], "validation_rag": S["validation"]["overall_rag"],
                             "fairness_rag": S["fairness"]["overall_rag"], "revalidation_required": S["monitoring"]["revalidation_required"]})
            mlflow.log_metrics({"parity_recheck_pass": float(bool(S["parity_recheck"]["pass"])),
                                **({"headline": S["validation"]["headline"]["value"]} if S["validation"]["headline"] else {})})
            mlflow.log_artifact(str(m.reports / "RUN_SUMMARY.md"), "summary")
    except Exception as e:
        print(f"  (MLflow logging skipped: {e})")
