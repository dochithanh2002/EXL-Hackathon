"""Single source of truth for where things live.

    agent/   the agents (md), their deterministic tools (this package), validators, config
    input/   one folder per legacy model: model.R, model.yaml, data/, docs/
    output/  one folder per model: logs/ (state, audit trail, agent work files), model/ (migrated module,
             frozen R reference, tests, manifest), reports/ (model card, migration, validation, monitoring)
             plus output/inventory/ (portfolio index) and output/mlflow/ (tracking store)
"""
from __future__ import annotations

import hashlib
import json
import time
from functools import cached_property
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "agent"
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
CONFIG = AGENT / "config"
THRESHOLDS = CONFIG / "thresholds.yaml"
RULEBOOK = CONFIG / "rulebook.yaml"
INVENTORY = OUTPUT / "inventory"
MLFLOW = OUTPUT / "mlflow"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(p: Path | str) -> str:
    b = Path(p).read_bytes() if isinstance(p, Path) else p.encode()
    return hashlib.sha256(b).hexdigest()


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def thresholds() -> dict:
    return yaml.safe_load(THRESHOLDS.read_text())


def list_models() -> list[str]:
    return sorted(p.parent.name for p in INPUT.glob("*/model.yaml") if not p.parent.name.startswith("_"))


class Model:
    """Paths and config for one model, addressed by its input folder name (e.g. `m2_taiwan`) or its id (`M2`)."""

    def __init__(self, name: str):
        slug = name
        if not (INPUT / slug / "model.yaml").exists():
            by_id = {yaml.safe_load((INPUT / s / "model.yaml").read_text())["model_id"].lower(): s for s in list_models()}
            slug = by_id.get(name.lower())
            if not slug:
                raise SystemExit(f"unknown model '{name}'. Models in input/: {', '.join(list_models()) or 'none'}")
        self.slug = slug
        self.input = INPUT / slug
        self.cfg_path = self.input / "model.yaml"
        self.cfg = yaml.safe_load(self.cfg_path.read_text())
        self.id = self.cfg["model_id"]
        self.out = OUTPUT / slug
        self.logs = self.out / "logs"
        self.model_dir = self.out / "model"
        self.reports = self.out / "reports"
        self.reference = self.model_dir / "reference"
        self.mig_dir = self.logs / "migration"
        self.doc_dir = self.logs / "documentation"

    # inputs ---------------------------------------------------------------------------------------
    @property
    def script(self) -> Path:
        return self.input / self.cfg["legacy"]["script"]

    @property
    def data(self) -> Path:
        return self.input / self.cfg["data"]["path"]

    @property
    def docs(self) -> list[Path]:
        return [self.input / d for d in self.cfg.get("supporting_docs", [])]

    @property
    def module_name(self) -> str:
        return self.cfg["target_python"]["module_name"]

    @property
    def module_path(self) -> Path:
        return self.model_dir / f"{self.module_name}.py"

    @cached_property
    def limits(self) -> dict:
        return thresholds()["agent_limits"]

    @cached_property
    def tolerances(self) -> dict:
        return thresholds()["parity_tolerances"]

    def thresholds(self, section: str) -> dict:
        """Thresholds for this model's type, e.g. thresholds('validation') -> validation.binary_pd."""
        return thresholds()[section][self.cfg["model_type"]]

    def experiment(self) -> str:
        return self.cfg.get("tracking", {}).get("experiment", f"{self.id.lower()}-model-factory")

    def input_hashes(self) -> dict:
        """G-12: everything the run depends on. Recorded at intake, re-checked before packaging / rendering."""
        h = {"model_yaml": sha(self.cfg_path), "legacy_script": sha(self.script), "data": sha(self.data),
             "thresholds": sha(THRESHOLDS), "rulebook": sha(RULEBOOK)}
        h.update({f"doc:{d.relative_to(self.input)}": sha(d) for d in self.docs})
        return h

    def changed_inputs(self) -> list[str]:
        f = self.logs / "intake" / "state.json"
        if not f.exists():
            return ["(no intake)"]
        then = json.loads(f.read_text()).get("input_hashes", {})
        return [k for k, v in self.input_hashes().items() if then.get(k) != v]

    def ensure(self) -> "Model":
        for d in (self.mig_dir, self.doc_dir, self.model_dir, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        return self


class Stage:
    """Resumable JSON state for one agent stage (intake, migration, documentation) under output/<model>/logs/<name>/.

    Every stage writes to ONE append-only audit trail per model, output/<model>/logs/audit.jsonl, so a reviewer
    reads a single timeline of what each agent and tool did.
    """

    def __init__(self, m: Model, name: str):
        self.m, self.name = m, name
        self.dir = m.logs / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / "state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {"stages": {}}

    def audit(self, event: str, **kw):
        with open(self.m.logs / "audit.jsonl", "a") as f:
            f.write(json.dumps({"ts_utc": now(), "stage": self.name, "event": event, **kw}, default=str) + "\n")

    def save(self):
        self.state_file.write_text(json.dumps(self.state, indent=2, default=str))

    def done(self, stage: str) -> bool:
        return self.state["stages"].get(stage, {}).get("status") == "done"

    def mark(self, stage: str, **kw):
        self.state["stages"][stage] = {"status": "done", "ts_utc": now(), **kw}
        self.save()
        self.audit(f"stage_done:{stage}", **kw)

    def require(self, *stages: str):
        missing = [s for s in stages if not self.done(s)]
        if missing:
            raise SystemExit(f"run these steps first: {', '.join(missing)}")
        if self.state.get("status") == "ESCALATED":
            raise SystemExit(f"stage is ESCALATED: {self.state.get('escalation')}. A human must resolve it "
                             f"(then `mrf reset <model> --stage ...`).")

    def escalate(self, stage: str, reasons: list[str], title: str):
        self.state.update(status="ESCALATED", escalation={"stage": stage, "reasons": reasons})
        self.save()
        self.audit("escalated", stage=stage, reasons=reasons)
        (self.dir / "ESCALATION.md").write_text(f"# {title} at `{stage}`\n\n" + "\n".join(f"- {r}" for r in reasons)
                                                + "\n\nA human must resolve these before re-running.\n")
