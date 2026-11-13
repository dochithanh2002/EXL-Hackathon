"""Red-team tests: every plausible migration mistake must be caught by a guardrail or by parity.

Static cases need nothing else. Dynamic cases use the M2 run's frozen R reference
(output/m2_taiwan/model/reference) and mutate the approved module; each mutant must FAIL parity.
Run:  .venv/bin/python -m pytest -q agent/tests
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent.tools import guardrails, parity
from agent.tools.paths import Model

M2 = Model("m2_taiwan")
ROOT = Path(__file__).resolve().parents[2]
# Frozen known-good M2 candidate (fixture), so the mutants do not depend on whatever code the agent writes next.
GOOD = Path(__file__).with_name("fixtures") / "m2_known_good_candidate.py"
PACKAGED = M2.module_path if M2.module_path.exists() else None
ALLOWED = M2.cfg["target_python"]["allowed_packages"]
TOL = M2.tolerances                              # agent/config/thresholds.yaml
CONTRACT_STUB = ("\nTARGET='y'\nTERMS=['(Intercept)']\ndef load_and_preprocess(p): ...\ndef fit(t): ...\n"
                 "def predict(r, d): ...\ndef coefficients(r): ...\ndef design_matrix(d): ...\n")

STATIC_CASES = {
    "sklearn penalised logistic": ("from sklearn.linear_model import LogisticRegression" + CONTRACT_STUB, "G-02"),
    "re-randomised split": ("import pandas as pd\ndef x(df): return df.sample(frac=0.7)" + CONTRACT_STUB, "G-02"),
    "numpy RNG": ("import numpy as np\nnp.random.seed(1)" + CONTRACT_STUB, "G-02"),
    "get_dummies level order": ("import pandas as pd\nX = pd.get_dummies(None)" + CONTRACT_STUB, "G-02"),
    "shell access": ("import os\nos.system('ls')" + CONTRACT_STUB, "G-02"),
    "file write": ("f = open('x.csv', 'w')" + CONTRACT_STUB, "G-02"),
    "eval": ("eval('1+1')" + CONTRACT_STUB, "G-02"),
    "unvetted package": ("import xgboost" + CONTRACT_STUB, "G-01"),
    "missing contract": ("import pandas as pd\nTARGET='y'\ndef fit(t): ...", "G-03"),
    "missing design_matrix/TERMS": (CONTRACT_STUB.replace("TERMS=", "X=").replace("def design_matrix", "def dm"), "G-03"),
    "syntax error": ("def broken(:\n  pass", "G-04"),
}


@pytest.mark.parametrize("name", list(STATIC_CASES))
def test_static_guardrail_blocks(name):
    code, gid = STATIC_CASES[name]
    rep = guardrails.static_check(code, ALLOWED)
    assert not rep.passed and any(v["guardrail"] == gid for v in rep.violations), rep.violations


def test_known_good_candidate_passes_static():
    assert guardrails.static_check(GOOD.read_text(), ALLOWED).passed


@pytest.mark.skipif(PACKAGED is None, reason="run the M2 migration first")
def test_packaged_module_passes_static():
    assert guardrails.static_check(PACKAGED.read_text(), ALLOWED).passed


def _run_mutant(code: str, tmp: Path) -> dict:
    cand = tmp / "mutant.py"
    cand.write_text(code)
    out = tmp / "out"
    subprocess.run([sys.executable, "-I", str(ROOT / "agent" / "tools" / "candidate_runner.py"), str(cand),
                    str(M2.data), str(M2.reference / "split.csv"), str(out)],
                   capture_output=True, text=True, timeout=300)
    status = json.loads((out / "run_status.json").read_text())
    if not status["ok"]:
        return {"pass": False, "stage": "runtime", "error": status["error"]}
    return parity.compare(M2.reference, out, TOL)


MUTANTS = {
    # subtle mistakes a translator (human or LLM) plausibly makes
    "wrong reference level (education=4 as base)": (
        'FACTOR_LEVELS = {"sex": [1, 2], "education": [1, 2, 3, 4], "marriage": [1, 2, 3]}',
        'FACTOR_LEVELS = {"sex": [1, 2], "education": [4, 1, 2, 3], "marriage": [1, 2, 3]}'),
    "positional loop off by one (0-based slice)": ("pay_cols = list(df.columns[5:11])\n    assert",
                                                    "pay_cols = list(df.columns[6:12])\n    pay_cols = pay_cols  # assert removed\n    if False: assert"),
    # (dropping 4 from isin([0,4,5,6]) -> 4 is an EQUIVALENT mutant, 4->4 is a no-op, so it is not listed)
    "undocumented marriage code merged into wrong level (54 rows)": (
        'df.loc[df["marriage"] == 0, "marriage"] = 3', 'df.loc[df["marriage"] == 0, "marriage"] = 1'),
    "pay codes -2/-1 not merged": ("df.loc[df[c].isin([-2, -1, 0]), c] = 0", "df.loc[df[c].isin([0]), c] = 0"),
    "looser convergence than R": ("GLM_EPSILON, GLM_MAXIT = 1e-8, 25", "GLM_EPSILON, GLM_MAXIT = 1e-4, 25"),
    "dropped a feature (marriage)": ('FACTOR_LEVELS = {"sex": [1, 2], "education": [1, 2, 3, 4], "marriage": [1, 2, 3]}',
                                     'FACTOR_LEVELS = {"sex": [1, 2], "education": [1, 2, 3, 4], "marriage": [1]}'),
}


@pytest.mark.skipif(not M2.reference.exists(), reason="run M2 intake first")
@pytest.mark.parametrize("name", list(MUTANTS))
def test_parity_catches_mutant(name, tmp_path):
    old, new = MUTANTS[name]
    src = GOOD.read_text()
    assert old in src, f"mutation anchor not found for {name}"
    res = _run_mutant(src.replace(old, new, 1), tmp_path)
    assert not res["pass"], f"mutant '{name}' was NOT caught"


@pytest.mark.skipif(not M2.reference.exists(), reason="run M2 intake first")
def test_known_good_candidate_passes_parity(tmp_path):
    """Sanity check for the mutant tests: the unmutated fixture passes, including the runtime G-03 contract check."""
    res = _run_mutant(GOOD.read_text(), tmp_path)
    assert res["pass"], res
