"""End-to-end wiring checks: layout, single thresholds file, and every downstream stage using the agent artefacts.
Run after `/mrf:run <model>`:  .venv/bin/python -m pytest -q agent/tests
"""
import hashlib
import json

import pytest
import yaml

from agent.tools.paths import AGENT, INPUT, INVENTORY, OUTPUT, ROOT, THRESHOLDS, Model

MODELS = [s for s in ("m2_taiwan", "m1_fremtpl2") if (OUTPUT / s / "logs" / "run_summary.json").exists()]


def j(p):
    return json.loads(p.read_text())


def test_three_folder_layout():
    for d in (AGENT, INPUT, OUTPUT):
        assert d.is_dir(), d
    for old in ("src", "runs", "reports", "legacy_models", "r2py_agent", "data", "ref_code"):
        assert not (ROOT / old).exists(), f"legacy folder {old}/ still present"
    for a in ("migration-agent", "documentation-agent", "validation-agent", "monitoring-agent"):
        assert (AGENT / "agents" / f"{a}.md").exists(), a
    for s in ("run", "new-model", "status", "approve", "monitor"):
        assert (AGENT / "skills" / s / "SKILL.md").exists(), s


def test_no_thresholds_in_model_yaml():
    for p in INPUT.glob("*/model.yaml"):
        cfg = yaml.safe_load(p.read_text())
        assert not {"tolerances", "limits", "thresholds"} & set(cfg), p
        assert cfg["model_type"] in yaml.safe_load(THRESHOLDS.read_text())["validation"], p


@pytest.mark.parametrize("model", MODELS)
def test_downstream_uses_the_agent_module(model):
    m = Model(model)
    h = hashlib.sha256(m.module_path.read_bytes()).hexdigest()
    assert j(m.model_dir / "manifest.json")["files"][m.module_path.name] == h
    for p in ("parity/parity_results.json", "validation/validation_results.json",
              "fairness/fairness_results.json", "monitoring/monitoring_results.json"):
        assert j(m.reports / p)["champion"]["module_sha256"] == h, p
    assert j(m.reports / "parity" / "parity_results.json")["parity_pass"] is True


@pytest.mark.parametrize("model", MODELS)
def test_report_cites_agent_card_and_module(model):
    m = Model(model)
    md = (m.reports / f"validation_report_{m.id}.md").read_text()
    assert f"output/{model}/reports/documentation/MODEL_CARD_{m.id}.md" in md
    assert f"output/{model}/model/{m.module_name}.py" in md


@pytest.mark.parametrize("model", MODELS)
def test_inventory_index_has_every_stage(model):
    m = Model(model)
    e = j(INVENTORY / "inventory_index.json")["models"][m.id]
    for k in ("model_card", "migration_status", "python_module", "parity_recheck_pass", "validation", "monitoring"):
        assert k in e, k
    assert e["parity_recheck_pass"] is True


@pytest.mark.parametrize("model", MODELS)
def test_one_audit_trail_with_every_stage(model):
    events = [json.loads(l) for l in (Model(model).logs / "audit.jsonl").read_text().splitlines()]
    stages = {e["stage"] for e in events}
    assert {"intake", "migration", "documentation", "validation", "monitoring"} <= stages, stages
    assert sum(e["event"] == "mlflow_logged" for e in events) >= 2


def test_monitor_thresholds_come_from_config():
    from agent.validators.m2_taiwan.monitor import rag
    t = yaml.safe_load(THRESHOLDS.read_text())["monitoring"]["binary_pd"]
    assert rag("score_psi", t["score_psi"]["amber"] / 2) == "GREEN"
    assert rag("score_psi", t["score_psi"]["amber"]) == "AMBER" and rag("score_psi", t["score_psi"]["red"]) == "RED"
    assert rag("min_dir", 0.9) == "GREEN" and rag("min_dir", 0.82) == "AMBER" and rag("min_dir", 0.7) == "RED"
    assert rag("auc_drop", None) == "N/A"
