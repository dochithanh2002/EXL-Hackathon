"""Multi-seed parity: the migrated module must match the UNMODIFIED legacy R script on several splits.

The legacy script sets no seed, so its split is random. For each seed below the harness
(r_capture.R) seeds R, sources the legacy script as-is, and freezes the split it drew; the
Python candidate is then fitted on exactly those rows and compared with parity.compare().
Passing on many splits rules out a conversion that only agrees on one lucky partition.

Needs Rscript (with the legacy script's packages) and the approved M2 module; skipped otherwise.
Run:  .venv/bin/python -m pytest -q agent/tests/test_multiseed_parity.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from agent.tools import parity
from agent.tools.paths import Model

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "agent" / "tools"
M2 = Model("m2_taiwan")
CFG = M2.cfg
CANDIDATE = M2.module_path
SEEDS = [1, 42, 777, 12345, 99999, CFG["legacy"]["seed"]]

pytestmark = [
    pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed"),
    pytest.mark.skipif(not CANDIDATE.exists(), reason="no packaged M2 module in output/m2_taiwan/model"),
]


@pytest.mark.parametrize("seed", SEEDS)
def test_parity_across_seeds(seed, tmp_path):
    L = CFG["legacy"]
    ref, cand = tmp_path / "r", tmp_path / "py"

    r = subprocess.run(
        ["Rscript", str(PKG / "r_capture.R"), str(M2.script), L["model_object"],
         L["train_object"], L["test_object"], str(seed), str(ref)],
        cwd=M2.input, capture_output=True, text=True, timeout=M2.limits["exec_timeout_s"])
    assert r.returncode == 0, f"R capture failed for seed {seed}:\n{r.stderr[-2000:]}"

    p = subprocess.run(
        [sys.executable, "-I", str(PKG / "candidate_runner.py"), str(CANDIDATE),
         str(M2.data), str(ref / "split.csv"), str(cand)],
        capture_output=True, text=True, timeout=M2.limits["exec_timeout_s"])
    status = json.loads((cand / "run_status.json").read_text())
    assert status["ok"], f"candidate failed for seed {seed}: {status}"

    res = parity.compare(ref, cand, M2.tolerances)
    failed = [c for c in res["checks"] if not c["pass"]]
    assert res["pass"], f"seed {seed}: {failed}\nevidence: {res['evidence']}"
