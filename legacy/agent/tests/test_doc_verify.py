"""Red-team tests for the Documentation agent: every plausible documentation error must be caught by verify().

Uses the approved M2 extract (output/m2_taiwan/logs/documentation/extract_final.json) and mutates it. Each mutant must fail
the named check; the unmutated extract must pass.
Run:  .venv/bin/python -m pytest -q agent/tests/test_doc_verify.py
"""
import copy
import json
from pathlib import Path

import pytest

from agent.tools import document
from agent.tools.paths import Model

M2 = Model("m2_taiwan")
DOC = M2.doc_dir
SRC = M2.script.read_text()
READY = (DOC / "extract_final.json").exists()


def _load():
    ex = json.loads((DOC / "extract_final.json").read_text())
    facts = json.loads((DOC / "facts.json").read_text())
    docs = "\n".join(document.strip_console_output(d.read_text()) for d in M2.docs)
    return ex, facts, SRC + "\n" + docs


def _var(ex, name):
    return next(v for v in ex["variables"] if v["name"] == name)


def _lim(ex, did):
    return next(l for l in ex["limitations"] if did in l["disclosure_ids"])


def m_drop_variable(ex):
    ex["variables"] = [v for v in ex["variables"] if v["name"] != "pay_6"]


def m_invent_variable(ex):
    ex["variables"].append({**_var(ex, "limit_bal"), "name": "age", "source_lines": [38]})


def m_wrong_reference_level(ex):
    _var(ex, "education")["levels"] = ["4", "1", "2", "3"]


def m_factor_as_numeric(ex):
    _var(ex, "sex")["kind"] = "numeric"


def m_line_out_of_range(ex):
    ex["preprocessing"][0]["source_lines"] = [999]


def m_blank_line(ex):
    ex["preprocessing"][1]["source_lines"] = [9]


def m_ungrounded_variable(ex):
    _var(ex, "limit_bal")["source_lines"] = [10]


def m_missing_protected_disclosure(ex):
    ex["limitations"] = [l for l in ex["limitations"] if "D-01:sex" not in l["disclosure_ids"]]


def m_disclosure_cited_but_not_named(ex):
    _lim(ex, "D-02")["text"] = "The split has some issues."


def m_recode_disclosure_wrong_lines(ex):
    _lim(ex, "D-04")["source_lines"] = [38]


def m_hallucinated_metric(ex):
    ex["methodology"] += " The model reaches an AUC of 0.81 on the test set."


def m_hallucinated_rate(ex):
    ex["target_definition"] += " The default rate is 22.1%."


MUTANTS = {
    "dropped formula variable": (m_drop_variable, "V-01"),
    "invented input variable": (m_invent_variable, "V-01"),
    "wrong factor reference level": (m_wrong_reference_level, "V-02"),
    "factor documented as numeric": (m_factor_as_numeric, "V-02"),
    "citation out of range": (m_line_out_of_range, "V-03"),
    "citation to blank line": (m_blank_line, "V-03"),
    "variable cites lines without its name": (m_ungrounded_variable, "V-03"),
    "protected attribute disclosure missing": (m_missing_protected_disclosure, "V-04"),
    "disclosure id cited but issue not named": (m_disclosure_cited_but_not_named, "V-04"),
    "recode disclosure cites wrong lines": (m_recode_disclosure_wrong_lines, "V-04"),
    "hallucinated AUC": (m_hallucinated_metric, "V-05"),
    "hallucinated default rate": (m_hallucinated_rate, "V-05"),
}


@pytest.mark.skipif(not READY, reason="run the M2 documentation first (/mrf:run m2_taiwan)")
def test_approved_extract_passes():
    ex, facts, evidence = _load()
    assert document.verify(ex, facts, SRC, evidence) == []


@pytest.mark.skipif(not READY, reason="run the M2 documentation first")
@pytest.mark.parametrize("name", list(MUTANTS))
def test_verify_catches_mutant(name):
    ex, facts, evidence = _load()
    mutate, check = MUTANTS[name]
    bad = copy.deepcopy(ex)
    mutate(bad)
    viol = document.verify(bad, facts, SRC, evidence)
    assert any(v["check"] == check for v in viol), f"'{name}' not caught by {check}: {viol}"


def test_console_output_is_stripped():
    txt = "prose line\n## [1] 30000 25\n##   1  20000  2  2  1  24\ncode(x)\n"
    assert document.strip_console_output(txt) == "prose line\ncode(x)"


def test_formula_variables_expands_dot():
    resp, vars_ = document.formula_variables("y ~ .", ["a", "b", "y"])
    assert resp == "y" and vars_ == ["a", "b"]
    assert document.formula_variables("y ~ a + log(b) + a:c", [])[1] == ["a", "b", "c"]
