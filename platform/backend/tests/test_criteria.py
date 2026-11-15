"""Default acceptance criteria: valid spec criteria by response type, offered to the Analyst, never applied by code."""
import pytest
import yaml

from controlplane import criteria
from controlplane.analyst import Analyst
from controlplane import support
from controlplane.schemas import sha256_file
from controlplane.workflow.machine import InputFile
from tests.test_baseline import CLAIMS
from tests.test_schemas import spec as make_spec


def test_defaults_are_spec_criteria_for_known_response_types():
    d = criteria.load()
    assert d.for_response("binary") == {"min_auc": 0.65, "max_psi": 0.25, "min_impact_ratio": 0.8}
    assert d.for_response("count") == {"max_psi": 0.25, "max_calibration_deviation": 0.1,
                                       "max_group_calibration_deviation": 0.1}
    assert d.for_response("ordinal") == {}


def test_an_unknown_criterion_or_response_type_is_rejected(tmp_path):
    bad = tmp_path / "c.yaml"
    bad.write_text(yaml.safe_dump({"version": 1, "defaults": {"binary": {"min_gini": 0.3}}}))
    with pytest.raises(ValueError, match="min_gini"):
        criteria.load(bad)
    bad.write_text(yaml.safe_dump({"version": 1, "defaults": {"survival": {"max_psi": 0.2}}}))
    with pytest.raises(ValueError, match="survival"):
        criteria.load(bad)


def test_a_spec_without_criteria_stays_without_them():
    assert make_spec().assessment.criteria.model_dump(exclude_none=True) == {}


def test_the_analyst_is_offered_the_defaults(tmp_path, envs):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "claims.csv").write_text(CLAIMS)
    (inputs / "m.R").write_text("# model\n")
    files = tuple(InputFile(role=r, name=n, sha256=sha256_file(inputs / n))
                  for r, n in [("dataset", "claims.csv"), ("r_code", "m.R")])
    prompt = Analyst(files, inputs, tmp_path / "cache", support.load(), envs).kickoff()
    assert "Default acceptance criteria by response type" in prompt
    assert "- binary: max_psi 0.25, min_auc 0.65, min_impact_ratio 0.8" in prompt
