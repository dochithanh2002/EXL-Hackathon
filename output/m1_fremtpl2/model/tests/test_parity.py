"""Generated parity regression test: re-runs the migrated module and compares it with the frozen R reference.
Run from the repository root:  .venv/bin/python -m pytest output/m1_fremtpl2/model/tests
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # output/<model>/model
ROOT = HERE.parents[2]
TOL = json.loads('{"coef_rel": 1e-06, "se_rel": 1e-06, "pred_abs": 1e-08, "deviance_abs": 1e-06}')


def test_parity(tmp_path):
    sys.path.insert(0, str(ROOT))
    from agent.tools import parity
    runner = ROOT / "agent" / "tools" / "candidate_runner.py"
    subprocess.run([sys.executable, str(runner), str(HERE / "m1_fremtpl2_freq.py"), str(ROOT / 'input/m1_fremtpl2/data/freMTPL2freq.csv'),
                    str(HERE / "reference" / "split.csv"), str(tmp_path)], capture_output=True, text=True)
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["ok"], status
    res = parity.compare(HERE / "reference", tmp_path, TOL)
    assert res["pass"], [c for c in res["checks"] if not c["pass"]]
