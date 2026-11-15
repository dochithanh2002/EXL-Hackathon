"""Projects with other data files next to the dataset (M1 joins claim counts from a second CSV): the Analyst sees
their profiles, baseline.R and the candidate read them, and the bundle's verify.py checks them."""
import hashlib
import json
import shutil
import subprocess

import pytest

from controlplane import candidate, support
from controlplane.analyst import Analyst, ToolError
from controlplane.baseline import CAPTURED
from controlplane.deliverables import PARITY, VERIFY
from controlplane.schemas import canonical_hash, sha256_file
from controlplane.translation import HARNESS, Translator, run_candidate
from controlplane.workflow import GateError, Step
from controlplane.workflow.machine import FilesUploaded, InputFile, ProjectState, SpecConfirmed, apply
from tests.test_baseline import CLAIMS, POISSON_R, jobs, poisson_spec  # noqa: F401
from tests.test_translation import POISSON_PY

AREAS = "area,region\nA,north\nB,south\nC,north\n"
JOINED_R = POISSON_R.replace('df$area <- factor(df$area)\n', (
    'areas <- read.csv("inputs/areas.csv")\n'
    'df <- merge(x = df, y = areas, by = "area", all.x = TRUE)\n'      # like M1: merge() rebuilds row names
    'df$region <- factor(df$region)\n')).replace("area + age", "region + age")
JOINED_PY = POISSON_PY.replace('''def prepare(path, sets):
    df = pd.read_csv(path)
''', '''def prepare(path, sets, data):
    df = pd.read_csv(path)
    areas = pd.read_csv(data["areas.csv"])
    df["region"] = df["area"].map(dict(zip(areas["area"], areas["region"])))
''').replace('''    for level in AREAS[1:]:
        X["area" + level] = (frame["area"] == level).astype(float)''', '''    X["regionsouth"] = (frame["region"] == "south").astype(float)''')


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_other_data_files_sit_beside_exactly_one_dataset():
    files = (InputFile(role="dataset", name="freq.csv", sha256=sha("a")),
             InputFile(role="data", name="sev.csv", sha256=sha("b")), InputFile(role="r_code", name="m.R", sha256=sha("c")))
    assert apply(ProjectState(), FilesUploaded(files=files)).step == Step.ALIGNMENT
    with pytest.raises(GateError, match="exactly one dataset"):
        apply(ProjectState(), FilesUploaded(files=files[1:]))


def test_guardrails_allow_reading_data_files_only_through_prepares_third_argument():
    assert candidate.check(JOINED_PY, data=True) == []
    assert any("'prepare' must take 3" in e for e in candidate.check(POISSON_PY, data=True))
    assert any("only the data files may be read" in e for e in candidate.check(JOINED_PY))
    assert any("only the data files may be read" in e for e in candidate.check(
        JOINED_PY.replace('pd.read_csv(data["areas.csv"])', 'pd.read_csv("inputs/areas.csv")'), data=True))


def test_a_joined_data_file_runs_through_baseline_candidate_and_bundle(jobs, tmp_path):
    st, settings = jobs.store, jobs.settings
    pid = st.create_project("joined")
    inputs = settings.project_dir(pid) / "inputs"
    inputs.mkdir(parents=True)
    files = []
    for role, name, content in [("dataset", "claims.csv", CLAIMS), ("data", "areas.csv", AREAS),
                                ("r_code", "m.R", "# model\n")]:
        (inputs / name).write_text(content)
        files.append(InputFile(role=role, name=name, sha256=sha256_file(inputs / name)))
    st.apply(pid, FilesUploaded(files=tuple(files)))

    analyst = Analyst(tuple(files), inputs, settings.project_dir(pid) / "cache", support.load(), jobs.envs)
    assert [c["name"] for c in analyst.profile("areas.csv")["columns"]] == ["area", "region"]
    assert "areas.csv as inputs/areas.csv" in analyst.kickoff() and "areas.csv: other data, 3 rows" in analyst.kickoff()
    assert {c["name"] for c in analyst.columns()["columns"]} >= {"claims", "region"}
    for bad in ("areas.csv", "claims.csv"):
        with pytest.raises(ToolError, match="is data and cannot be read"):
            analyst.read_file(bad)
    with pytest.raises(ToolError, match="the data files are"):
        analyst.profile("m.R")

    spec = poisson_spec(baseline_r=JOINED_R, formula="claims ~ region + age + offset(log(exposure))")
    v = st.add_spec(pid, spec, support.load())
    s = st.state(pid)
    st.apply(pid, SpecConfirmed(version=v, tier=s.drafted_tier, proven=s.drafted_proven,
                                baseline_key=canonical_hash(spec.baseline_key()), parity_risk_accepted=True))
    jobs.baseline(pid)
    s = st.state(pid)
    assert s.step == Step.TRANSLATION, s.detail
    baseline_dir = settings.data_dir / st.evidence(pid, s.baseline_id)["body"]["dir"]

    translator = Translator(spec, baseline_dir, "claims.csv", jobs.envs, analyst.profile, ["areas.csv"])
    assert 'data["<file name>"]' in translator.kickoff() and "The data files: areas.csv" in translator.kickoff()
    assert translator.check({"module": JOINED_PY, "notes": []})[1] == []
    result = run_candidate(tmp_path / "cand", JOINED_PY, spec, baseline_dir, s.files, inputs, jobs.envs)
    assert result.passed, [c for c in result.checks if not c.passed] or result.error

    # the bundle's verifier finds the data file next to the dataset and checks its hash
    root = tmp_path / "bundle"
    (root / "reference").mkdir(parents=True)
    for name in CAPTURED:
        shutil.copy(baseline_dir / name, root / "reference" / name)
    for src, name in [(VERIFY, "verify.py"), (PARITY, "parity.py"), (HARNESS, "harness.py")]:
        shutil.copy(src, root / name)
    (root / "model.py").write_text(JOINED_PY)
    (root / "spec.json").write_text(spec.model_dump_json())
    (root / "verify.json").write_text(json.dumps({"dataset_sha256": sha(CLAIMS), "data_sha256": {"areas.csv": sha(AREAS)}}))
    rerun = subprocess.run([jobs.envs.py(), "verify.py", inputs / "claims.csv"], cwd=root, capture_output=True, text=True)
    assert rerun.returncode == 0 and "FAIL" not in rerun.stdout, rerun.stdout + rerun.stderr

    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(inputs / "claims.csv", alone / "claims.csv")
    rerun = subprocess.run([jobs.envs.py(), "verify.py", alone / "claims.csv"], cwd=root, capture_output=True, text=True)
    assert rerun.returncode == 1 and "areas.csv is not next to the dataset" in rerun.stdout
    (alone / "areas.csv").write_text(AREAS.replace("south", "east"))
    rerun = subprocess.run([jobs.envs.py(), "verify.py", alone / "claims.csv"], cwd=root, capture_output=True, text=True)
    assert rerun.returncode == 1 and "is not the data file the model was validated on" in rerun.stdout
