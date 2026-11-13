"""Deterministic tools for the INTAKE step and the Migration agent (agent/agents/migration-agent.md).

The agent (Claude Code / Codex) does the reasoning: it reads the R code, writes analysis.json and plan.json,
writes Python candidates and repairs them. These tools do everything that must NOT be left to an LLM:
capture the R reference, scan, validate the agent's files against typed schemas, run candidates in a sandbox,
check parity against frozen tolerances, count iterations, escalate, package and hash.

    mrf intake          <model>               G-12 hashes, R capture, static scan -> brief for the agents
    mrf check-analysis  <model>               G-10 schema + terms == R terms
    mrf check-plan      <model>               G-10 schema + G-13 rule coverage + allowed packages
    mrf test-candidate  <model> <file.py>     G-01..04 static, G-06/07 sandbox run, G-05 parity, G-08 limit
    mrf package         <model>               G-12 re-check, module + tests + manifest + reports, MLflow
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

import pandas as pd
import yaml
from pydantic import ValidationError

from agent.tools import guardrails, parity
from agent.tools.paths import AGENT, RULEBOOK, THRESHOLDS, Model, Stage, now, rel, sha
from agent.tools.schemas import Analysis, Plan

TOOLS = Path(__file__).resolve().parent
EXIT_OK, EXIT_FAIL, EXIT_ESCALATED = 0, 1, 4


def rulebook() -> tuple[dict, str]:
    rb = yaml.safe_load(RULEBOOK.read_text())
    return {r["id"]: r for r in rb["rules"]}, str(rb["version"])


# ============================================================================================ INTAKE
SCAN_PATTERNS = {
    "R-PKG-01": r"\blibrary\(\s*(\w+)", "R-IO-01": r"\bread\.csv\(", "R-IO-02": r"tolower\(\s*colnames",
    "R-IDX-01": r"\[\[\s*\w+\s*\]\]|\[\s*,\s*\d+\s*\]|seq\(\s*\d+", "R-IDX-02": r"which\(.*\)\s*,\s*\]",
    "R-REC-01": r"%in%", "R-FAC-01": r"\b(as\.)?factor\(", "R-RNG-01": r"set\.seed|sample\(|createDataPartition",
    "R-EST-01": r"\bglm\([^)]*binomial", "R-EST-02": r"\bglm\(", "R-EST-03": r"poisson|Gamma|offset\s*=",
    "R-EST-04": r"\blm\(", "R-PRED-01": r"\bpredict\(",
}


def intake(m: Model, force: bool = False) -> int:
    """Hash inputs, run the legacy R script under the capture harness, scan it. Shared by both agents."""
    m.ensure()
    st = Stage(m, "intake")
    hashes = m.input_hashes()
    same = st.state.get("input_hashes") == hashes and (m.reference / "meta.json").exists()
    if same and not force:
        print(f"intake unchanged since {st.state['stages']['scan']['ts_utc']} (use --force to re-capture)")
        _brief(m, st)
        return EXIT_OK
    st.state = {"stages": {}, "input_hashes": hashes}
    st.mark("hash", hashes=hashes)

    L = m.cfg["legacy"]
    cmd = ["Rscript", str(TOOLS / "r_capture.R"), str(m.script), L["model_object"], L["train_object"],
           L["test_object"], str(L["seed"]), str(m.reference)]
    print(f"[capture] running {rel(m.script)} under the R capture harness ...", flush=True)
    r = subprocess.run(cmd, cwd=m.input, capture_output=True, text=True, timeout=m.limits["r_capture_timeout_s"])
    if r.returncode != 0:
        st.audit("capture_failed", stderr=r.stderr[-3000:])
        raise SystemExit(f"R capture failed:\n{r.stderr[-3000:]}")
    st.mark("capture", reference_dir=rel(m.reference), r_stdout=r.stdout.strip()[-300:])

    src = m.script.read_text()
    flat = re.sub(r"\s+", " ", "\n".join(l.split("#")[0] for l in src.splitlines()))
    meta = json.loads((m.reference / "meta.json").read_text())
    triggered = {}
    for rid, p in SCAN_PATTERNS.items():
        hits = re.findall(p, flat)
        if hits:
            triggered[rid] = f"pattern /{p}/ matched {len(hits)}x" + (f": {sorted(set(hits))}" if rid == "R-PKG-01" else "")
    if meta.get("xlevels"):
        triggered["R-FAC-02"] = f"factors in formula: {list(meta['xlevels'])}"
    if meta.get("column_classes_after_preprocessing", {}).get(meta["response"]) == "factor":
        triggered["R-FAC-03"] = f"factor response '{meta['response']}'"
    for always in ("R-IO-03", "R-FAC-04", "R-REC-02", "GOV-01"):
        triggered[always] = "always applies"
    if re.search(r"\b(glm|lm)\(", flat):
        triggered["R-NA-01"] = "glm/lm default na.action = na.omit"
    raw = pd.read_csv(m.data)
    prof = {"n_rows": int(len(raw)), "columns": [
        {"name": c, "dtype": str(raw[c].dtype), "n_unique": int(raw[c].nunique()), "n_missing": int(raw[c].isna().sum()),
         **({"levels": sorted(raw[c].dropna().unique().tolist())} if raw[c].nunique() <= 12 else {})}
        for c in raw.columns]}
    na_cols = [c for c in raw.columns if raw[c].isna().any()]
    if na_cols or re.search(r"na\.omit|na\.action|complete\.cases|is\.na\(", flat):
        triggered["R-NA-01"] = f"missing values in raw columns {na_cols[:20]}" if na_cols else "NA handling in R code"
    rules, _ = rulebook()
    scan = {"triggered_rules": {k: {"severity": rules[k]["severity"], "why": v} for k, v in triggered.items()},
            "data_profile": prof, "protected_attributes": m.cfg["data"].get("protected_attributes", [])}
    (st.dir / "scan.json").write_text(json.dumps(scan, indent=2, default=str))
    st.mark("scan", n_triggered=len(triggered))
    st.state["status"] = "DONE"
    st.save()
    _brief(m, st)
    return EXIT_OK


def _brief(m: Model, st: Stage) -> None:
    """Short, human-readable pointer list for the agents (what to read, where to write)."""
    scan = json.loads((st.dir / "scan.json").read_text())
    meta = json.loads((m.reference / "meta.json").read_text())
    must = sorted(k for k, v in scan["triggered_rules"].items() if v["severity"] == "MUST")
    md = f"""# Intake brief: {m.id} {m.cfg['model_name']}

_Generated {now()} by `mrf intake`. Deterministic; no LLM involved._

| | |
|---|---|
| Legacy script | `{rel(m.script)}` ({m.cfg['legacy']['language']}) |
| R model | `{meta['formula']}`; {meta['family']}/{meta['link']}; {len(meta['terms'])} coefficients; converged {meta['converged']} |
| Train / test rows | {meta['n_train']} / {meta['n_test']} (split frozen in `{rel(m.reference / 'split.csv')}`) |
| Data | `{rel(m.data)}`: {scan['data_profile']['n_rows']} rows, schema in `{rel(st.dir / 'scan.json')}` (never read data rows, G-11) |
| Protected attributes | {', '.join(scan['protected_attributes']) or 'none declared'} |
| Allowed Python packages | {', '.join(m.cfg['target_python']['allowed_packages'])} |
| Target module | `{m.module_name}.py` |

**Triggered MUST rules** (the plan must cite every one, G-13): {', '.join(must)}

**Files**
- R capture (facts): `{rel(m.reference / 'meta.json')}`, `coefficients.csv`
- Rulebook: `{rel(RULEBOOK)}` · thresholds: `{rel(THRESHOLDS)}` (read-only for agents)
- Migration agent writes: `{rel(m.mig_dir / 'analysis.json')}`, `{rel(m.mig_dir / 'plan.json')}`, `{rel(m.mig_dir / 'work')}/candidate.py`
- Documentation agent writes: `{rel(m.doc_dir / 'extract.json')}`
"""
    (m.logs / "INTAKE_BRIEF.md").write_text(md)
    print(md)


# ============================================================================================ MIGRATION
def _load_checked(path: Path, schema, st: Stage, step: str):
    if not path.exists():
        raise SystemExit(f"{rel(path)} not found: the Migration agent must write it first")
    try:
        obj = schema.model_validate_json(path.read_text())
    except ValidationError as e:
        st.audit("schema_rejected", step=step, file=rel(path), errors=e.errors()[:10])
        print(json.dumps({"verdict": "FAIL", "guardrail": "G-10", "errors": json.loads(e.json())[:10]}, indent=1, default=str))
        raise SystemExit(EXIT_FAIL)
    return obj


def _escalate(st: Stage, step: str, reasons: list[str]) -> int:
    st.escalate(step, reasons, f"Migration escalation: {st.m.id}")
    print(json.dumps({"verdict": "ESCALATED", "step": step, "reasons": reasons, "file": rel(st.dir / "ESCALATION.md")}, indent=1))
    return EXIT_ESCALATED


def check_analysis(m: Model) -> int:
    Stage(m, "intake").require("scan")
    st = Stage(m, "migration")
    a = _load_checked(m.mig_dir / "analysis.json", Analysis, st, "analyse")
    meta = json.loads((m.reference / "meta.json").read_text())
    if set(a.model.terms) != set(meta["terms"]):
        st.audit("check_failed", step="analyse", reason="terms differ")
        print(json.dumps({"verdict": "FAIL", "reason": "analysis terms != R terms",
                          "missing": sorted(set(meta["terms"]) - set(a.model.terms)),
                          "extra": sorted(set(a.model.terms) - set(meta["terms"]))}, indent=1))
        return EXIT_FAIL
    st.state.setdefault("hashes", {})["analysis"] = sha(m.mig_dir / "analysis.json")
    st.mark("analyse", escalations=a.escalations)
    if a.escalations:
        return _escalate(st, "analyse", a.escalations)
    print(json.dumps({"verdict": "PASS", "step": "analyse", "n_preprocessing_steps": len(a.preprocessing),
                      "protected_attributes_used": a.protected_attributes_used}, indent=1))
    return EXIT_OK


def check_plan(m: Model) -> int:
    st = Stage(m, "migration")
    st.require("analyse")
    pl = _load_checked(m.mig_dir / "plan.json", Plan, st, "plan")
    scan = json.loads((m.logs / "intake" / "scan.json").read_text())
    must = {k: v["why"] for k, v in scan["triggered_rules"].items() if v["severity"] == "MUST"}
    cov = guardrails.plan_coverage({r for s in pl.steps for r in s.rule_ids}, must)
    allowed = m.cfg["target_python"]["allowed_packages"]
    bad = sorted(set(pl.packages) - set(allowed))
    problems = [v["message"] for v in cov.violations] + [f"package not allowed: {p}" for p in bad]
    if problems:
        st.audit("check_failed", step="plan", problems=problems)
        print(json.dumps({"verdict": "FAIL", "problems": problems}, indent=1))
        return EXIT_FAIL
    st.state.setdefault("hashes", {})["plan"] = sha(m.mig_dir / "plan.json")
    st.mark("plan", coverage="pass", escalations=pl.escalations + pl.unmapped)
    if pl.escalations or pl.unmapped:
        return _escalate(st, "plan", pl.escalations + [f"unmapped: {u}" for u in pl.unmapped])
    print(json.dumps({"verdict": "PASS", "step": "plan", "must_rules_covered": sorted(must)}, indent=1))
    return EXIT_OK


def test_candidate(m: Model, file: Path, summary: str = "") -> int:
    st = Stage(m, "migration")
    st.require("plan")
    if st.done("test_and_repair"):
        print("a candidate already passed; run `mrf package` (or `mrf reset <model> --stage migration` to start over)")
        return EXIT_OK
    hist = st.state.setdefault("iterations", [])
    it = len(hist) + 1
    max_it = m.limits["max_repair_iterations"]
    cdir = m.mig_dir / "candidates"
    cdir.mkdir(exist_ok=True)
    path = cdir / f"candidate_{it}.py"                      # immutable copy: what was tested is what is recorded
    shutil.copyfile(file, path)
    code = path.read_text()
    allowed = m.cfg["target_python"]["allowed_packages"]
    rec = {"iteration": it, "file": rel(path), "sha256": sha(path), "change_summary": summary}
    g = guardrails.static_check(code, allowed)
    rec["static_guardrails"] = g.__dict__
    if g.passed:
        out = cdir / f"run_{it}"
        env = {k: v for k, v in os.environ.items() if not re.search(r"KEY|TOKEN|SECRET|PASSWORD|AZURE|ANTHROPIC|OPENAI", k)}
        try:
            subprocess.run([sys.executable, "-I", str(TOOLS / "candidate_runner.py"), str(path), str(m.data),
                            str(m.reference / "split.csv"), str(out)], cwd=cdir, env=env, capture_output=True, text=True,
                           timeout=m.limits["exec_timeout_s"])
            status = json.loads((out / "run_status.json").read_text())
        except subprocess.TimeoutExpired:
            status = {"ok": False, "error_type": "Timeout", "error": "G-06 execution timeout"}
        rec["run"] = {k: v for k, v in status.items() if k != "traceback"}
        if status["ok"]:
            pr = parity.compare(m.reference, out, m.tolerances)
            rec["parity"] = pr
            if pr["pass"]:
                rec["verdict"] = "PASS"
                hist.append(rec)
                shutil.copyfile(path, m.mig_dir / "final_candidate.py")
                st.audit("iteration", iteration=it, sha256=rec["sha256"], verdict="PASS")
                st.mark("test_and_repair", iterations=it, final=rec["file"])
                print(json.dumps({"verdict": "PASS", "iteration": it, "checks": pr["checks"]}, indent=1))
                return EXIT_OK
            rec["failure"] = {"stage": "parity", "checks": [c for c in pr["checks"] if not c["pass"]], "evidence": pr["evidence"]}
        else:
            rec["failure"] = {"stage": "runtime", **{k: v for k, v in status.items()}}
    else:
        rec["failure"] = {"stage": "static_guardrails", "violations": g.violations}
    rec["verdict"] = "FAIL"
    hist.append(rec)
    st.save()
    st.audit("iteration", iteration=it, sha256=rec["sha256"], verdict="FAIL", failure_stage=rec["failure"]["stage"])
    if it >= max_it:
        return _escalate(st, "test_and_repair", [f"no passing candidate after {it} iterations (G-08, limit {max_it})",
                                                 json.dumps(rec["failure"], default=str)[:1500]])
    print(json.dumps({"verdict": "FAIL", "iteration": it, "iterations_left": max_it - it, "failure": rec["failure"]},
                     indent=1, default=str)[:6000])
    return EXIT_FAIL


TEST_TEMPLATE = '''"""Generated parity regression test: re-runs the migrated module and compares it with the frozen R reference.
Run from the repository root:  .venv/bin/python -m pytest {model_dir}/tests
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # output/<model>/model
ROOT = HERE.parents[2]
TOL = json.loads({tol!r})


def test_parity(tmp_path):
    sys.path.insert(0, str(ROOT))
    from agent.tools import parity
    runner = ROOT / "agent" / "tools" / "candidate_runner.py"
    subprocess.run([sys.executable, str(runner), str(HERE / "{module}.py"), str(ROOT / {data!r}),
                    str(HERE / "reference" / "split.csv"), str(tmp_path)], capture_output=True, text=True)
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["ok"], status
    res = parity.compare(HERE / "reference", tmp_path, TOL)
    assert res["pass"], [c for c in res["checks"] if not c["pass"]]
'''


def package(m: Model, migrated_by: str = "") -> int:
    st = Stage(m, "migration")
    st.require("test_and_repair")
    if changed := m.changed_inputs():
        return _escalate(st, "package", [f"G-12 inputs changed since intake: {changed}"])
    out = m.model_dir
    (out / "tests").mkdir(parents=True, exist_ok=True)
    mod = m.module_path
    shutil.copyfile(m.mig_dir / "final_candidate.py", mod)
    used = {(n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0]
            for n in ast.walk(ast.parse(mod.read_text())) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in (n.names if isinstance(n, ast.Import) else [n])}
    allowed = m.cfg["target_python"]["allowed_packages"]
    (out / "requirements.txt").write_text("\n".join(f"{p}=={pkg_version(p)}" for p in allowed if p in used) + "\n")
    (out / "tests" / "test_parity.py").write_text(TEST_TEMPLATE.format(
        model_dir=rel(out), module=m.module_name, data=rel(m.data), tol=json.dumps(m.tolerances)))
    from agent.tools.report import migration_report
    rep_dir = m.reports / "migration"
    rep_dir.mkdir(parents=True, exist_ok=True)
    migration_report(m, st, rep_dir)
    migrated_by = migrated_by or os.getenv("MRF_AGENT", "Migration agent (agent/agents/migration-agent.md)")
    manifest = {"model_id": m.id, "status": "AWAITING_HUMAN_APPROVAL", "packaged_utc": now(), "migrated_by": migrated_by,
                "input_hashes": m.input_hashes(),
                "files": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                          if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    st.state["status"] = "AWAITING_HUMAN_APPROVAL"
    st.mark("package", output=rel(out))
    from agent.tools import tracking
    final = st.state["iterations"][-1]
    _, rb_version = rulebook()
    st.state["mlflow_run_id"] = tracking.log_run(
        m, st, "agent_migration",
        tags={"model_id": m.id, "stage": "migration", "status": st.state["status"], "rulebook_version": rb_version,
              "migrated_by": migrated_by, "module_sha256": sha(mod)},
        params={**{f"tol_{k}": v for k, v in m.tolerances.items()}, "allowed_packages": allowed,
                "max_repair_iterations": m.limits["max_repair_iterations"], "legacy_script": rel(m.script)},
        metrics={"iterations": len(st.state["iterations"]), "parity_pass": float(final["verdict"] == "PASS"),
                 **{f"parity.{c['check']}": c["value"] for c in final.get("parity", {}).get("checks", [])}},
        artifacts={"model": out, "report": rep_dir, "audit": m.logs / "audit.jsonl"})
    st.save()
    print(f"PACKAGED -> {rel(out)}  status AWAITING_HUMAN_APPROVAL  (review: {rel(rep_dir / 'HUMAN_REVIEW.md')})")
    return EXIT_OK
