"""`mrf`: the deterministic tool belt the agents call. Humans use the slash commands (/mrf:run, ...) instead.

Run from the repository root:  agent/bin/mrf <command> ...   (wrapper for .venv/bin/python -m agent.tools)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.tools import document, migrate, pipeline
from agent.tools.paths import Model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mrf", description="Model Risk Factory tools (called by the agents)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, help_, model=True):
        p = sub.add_parser(name, help=help_)
        if model:
            p.add_argument("model", help="input folder name (e.g. m2_taiwan) or model id (e.g. M2)")
        return p

    cmd("list", "models in input/ and their status", model=False)
    cmd("schema", "print the JSON schema an agent work file must match", model=False).add_argument(
        "name", choices=["analysis", "plan", "doc-extract"])
    cmd("status", "stage-by-stage status of one model")
    cmd("intake", "hash inputs, capture the R reference, scan").add_argument("--force", action="store_true")
    cmd("doc-facts", "Documentation: deterministic facts + required disclosures")
    cmd("doc-verify", "Documentation: verify logs/documentation/extract.json (V-01..V-06)")
    cmd("doc-render", "Documentation: render model card + inventory record").add_argument("--by", default="")
    cmd("check-analysis", "Migration: validate logs/migration/analysis.json")
    cmd("check-plan", "Migration: validate logs/migration/plan.json")
    p = cmd("test-candidate", "Migration: guardrails + sandbox run + parity for one candidate")
    p.add_argument("file", type=Path); p.add_argument("--summary", default="", help="what changed and why (repairs)")
    cmd("package", "Migration: package the passing candidate").add_argument("--by", default="")
    cmd("validate", "Validation: run the model's validation suite")
    p = cmd("monitor", "Monitoring: score batches, rate against triggers")
    p.add_argument("--batch"); p.add_argument("--name")
    cmd("summary", "write RUN_SUMMARY.md, update the inventory, log to MLflow")
    p = cmd("approve", "record a human decision")
    p.add_argument("--target", choices=["migration", "documentation"], required=True)
    p.add_argument("--reviewer", required=True); p.add_argument("--decision", choices=["approve", "reject"], required=True)
    p.add_argument("--comment", default="")
    cmd("reset", "archive one stage's logs so it can re-run").add_argument(
        "--stage", choices=["intake", "documentation", "migration", "validation", "monitoring"], required=True)
    a = ap.parse_args(argv)

    if a.cmd == "list":
        print(json.dumps(pipeline.list_all(), indent=1))
        return 0
    if a.cmd == "schema":
        from agent.tools import schemas
        cls = {"analysis": schemas.Analysis, "plan": schemas.Plan, "doc-extract": schemas.DocExtract}[a.name]
        print(json.dumps(cls.model_json_schema(), indent=1))
        return 0
    m = Model(a.model)
    match a.cmd:
        case "status":
            print(json.dumps(pipeline.status(m), indent=1, default=str)); return 0
        case "intake":
            return migrate.intake(m, force=a.force)
        case "doc-facts":
            return document.doc_facts(m)
        case "doc-verify":
            return document.doc_verify(m)
        case "doc-render":
            return document.doc_render(m, a.by)
        case "check-analysis":
            return migrate.check_analysis(m)
        case "check-plan":
            return migrate.check_plan(m)
        case "test-candidate":
            return migrate.test_candidate(m, a.file, a.summary)
        case "package":
            return migrate.package(m, a.by)
        case "validate":
            return pipeline.validate(m)
        case "monitor":
            return pipeline.monitor(m, a.batch, a.name)
        case "summary":
            pipeline.summary(m); return 0
        case "approve":
            return pipeline.approve(m, a.target, a.reviewer, a.decision, a.comment)
        case "reset":
            return pipeline.reset(m, a.stage)
    return 1


if __name__ == "__main__":
    sys.exit(main())
