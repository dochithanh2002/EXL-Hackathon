import json

import pytest

from controlplane import support
from controlplane.analyst import Analyst, ToolError, tool_schema
from controlplane.environments import Environments
from controlplane.profile import ProfileFailed, profile
from controlplane.schemas import ModelSpec, sha256_file
from controlplane.workflow.machine import InputFile
from tests.test_baseline import poisson_spec

TRICKY = "\n".join([
    "a b,1x,a.b,flag,count,ratio,word,empty,mixed,big,if",
    "1,2,x,TRUE,1,0.5,u,NA,1,3000000000,T",
    "2,,y,FALSE,NA,1e3,,NA,a,1,F",
    "3,4,x,T,3,-2,w,,2,2,T",
]) + "\n"


def run_profile(tmp_path, envs, content: str) -> dict:
    path = tmp_path / "d.csv"
    path.write_text(content)
    return profile(path, tmp_path / "work", envs)


def test_profile_reads_names_classes_and_missing_values_as_read_csv_does(tmp_path, envs):
    p = run_profile(tmp_path, envs, TRICKY)
    assert p["rows"] == 3 and not p["row_names_from_file"]
    assert [c["r_name"] for c in p["columns"]] == ["a.b.1", "X1x", "a.b", "flag", "count", "ratio", "word", "empty",
                                                   "mixed", "big", "if."]
    assert [c["class"] for c in p["columns"]] == ["integer", "integer", "character", "logical", "integer", "numeric",
                                                  "character", "logical", "character", "numeric", "logical"]
    assert [c["missing"] for c in p["columns"]] == [0, 1, 0, 0, 1, 0, 0, 3, 0, 0, 0]


def test_profile_follows_row_names_in_the_file_and_counts_nan_as_missing(tmp_path, envs):
    p = run_profile(tmp_path, envs, "a,b\nr1,1,x\nr2,NaN,y\nr3,3,z\nr4,Inf,x\n")   # one header field fewer
    assert p["row_names_from_file"] and p["rows"] == 4
    a, b = p["columns"]
    assert (a["name"], a["class"], a["missing"], a["non_finite"], a["max"]) == ("a", "numeric", 1, 1, 3)
    assert (b["name"], b["class"], b["distinct"]) == ("b", "character", 3)       # the last field is not dropped


def test_profile_shows_levels_only_where_they_summarise(tmp_path, envs):
    p = run_profile(tmp_path, envs, "id,grp,y\n" + "".join(f"id{i},{'ab'[i % 2]},{i}\n" for i in range(50)))
    cols = {c["name"]: c for c in p["columns"]}
    assert "levels" not in cols["id"] and cols["id"]["distinct"] == 50
    assert cols["grp"]["levels"] == [{"value": "a", "n": 25}, {"value": "b", "n": 25}]
    assert (cols["y"]["min"], cols["y"]["max"], cols["y"]["mean"]) == (0, 49, 24.5)
    assert "id7" not in json.dumps(p)
    tiny = run_profile(tmp_path, envs, "k,v\nu,1\nw,2\nu,3\n")                    # 3 rows: no level lists
    assert all("levels" not in c for c in tiny["columns"])


def test_profile_failure_reports_r_error_not_the_log(tmp_path, envs):
    with pytest.raises(ProfileFailed, match="R could not read the dataset with read.csv") as e:
        run_profile(tmp_path, envs, "a,b\n1,2,3,4\n")
    assert len(str(e.value)) < 1200


@pytest.fixture
def analyst(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "claims.csv").write_text("claims,exposure\n1,1\n")
    (inputs / "m.R").write_text("fit <- glm(claims ~ 1)\n")
    (inputs / "notes.pdf").write_bytes(b"%PDF")
    files = tuple(InputFile(role=r, name=n, sha256=sha256_file(inputs / n))
                  for r, n in (("dataset", "claims.csv"), ("r_code", "m.R"), ("document", "notes.pdf")))
    return Analyst(files, inputs, tmp_path / "cache", support.load(), Environments())


def test_the_analyst_can_read_code_but_not_the_dataset(analyst):
    assert analyst.read_file("m.R").startswith("fit <-")
    for name, message in (("claims.csv", "cannot be read"), ("notes.pdf", "not a text file"), ("x.R", "no uploaded")):
        with pytest.raises(ToolError, match=message):
            analyst.read_file(name)
    (analyst.inputs_dir / "m.R").chmod(0o644)
    (analyst.inputs_dir / "m.R").write_text("changed\n")
    with pytest.raises(ToolError, match="intake hash"):
        analyst.read_file("m.R")


def errors(analyst, **overrides) -> list[str]:
    spec, errs = analyst.check(poisson_spec(**overrides).model_dump(mode="json"))
    assert (spec is None) == bool(errs)
    return errs


def test_spec_checks(analyst):
    assert errors(analyst) == []
    r = poisson_spec().baseline_r
    assert errors(analyst, baseline_r=r.replace("inputs/claims.csv", "claims.csv")) == [
        "baseline_r must read the dataset from inputs/claims.csv"]
    assert errors(analyst, baseline_r="library(MASS)\n" + r) == [
        "baseline_r loads ['MASS'], which r_packages does not list"]
    assert errors(analyst, baseline_r="x <- dplyr::filter(df)\n" + r, r_packages=["dplyr"]) == []
    assert "stable R environment has no ['randomForest']" in errors(analyst, r_packages=["randomForest"])[0]
    assert errors(analyst, baseline_objects={"model": "model", "train": "train", "test": None}) == [
        "baseline_objects.model 'model' does not appear in baseline_r"]
    assert errors(analyst, family="binomialish")[0].startswith("support_tier is 'exact'")
    rf = dict(r_function="randomForest::randomForest", support_tier="unsupported", r_packages=["randomForest"],
              baseline_r="library(randomForest)\n" + r)
    assert errors(analyst, **rf) == []                                 # an unsupported model can still be drafted
    spec, errs = analyst.check({"formula": 1})
    assert spec is None and "target_model: Field required" in errs


def test_tool_schema_has_no_references():
    schema = tool_schema(ModelSpec)
    assert "$ref" not in json.dumps(schema) and "$defs" not in schema
    assert schema["properties"]["split"]["properties"]["seed"]
