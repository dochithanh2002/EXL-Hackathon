import csv
import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from controlplane import support
from controlplane.runner import run
from controlplane.schemas import ModelSpec, canonical_hash, sha256_file
from controlplane.workflow import Reason, Status, Step
from controlplane.workflow.machine import FilesUploaded, InputFile, SpecConfirmed
from tests.conftest import make_jobs
from tests.test_schemas import spec as make_spec

FIXTURES = Path(__file__).parent / "fixtures"
M4_DIR = Path(os.environ.get("M4_DIR", Path.home() / "Desktop/EXL-Hackathon-26/downloads/M4_givemesomecredit"))

CLAIMS = "\n".join(
    ["id,claims,exposure,area,age"]
    + [f"{i},{(i * 7) % 4},{0.5 + (i % 5) / 10},{'ABC'[i % 3]},{20 + (i * 13) % 50}" for i in range(1, 201)])
POISSON_R = """df <- read.csv("inputs/claims.csv")
df <- df[nrow(df):1, ]
df$area <- factor(df$area)
idx <- sample(nrow(df), 140)
train <- df[idx, ]
test <- df[-idx, ]
fit <- glm(claims ~ area + age + offset(log(exposure)), family = poisson, data = train)
"""


def poisson_spec(**overrides) -> ModelSpec:
    return make_spec(**{
        "family": "poisson", "link": "log", "formula": "claims ~ area + age + offset(log(exposure))",
        "response": "claims", "offset": "log(exposure)", "baseline_r": POISSON_R,
        "split": {"method": "sample", "r_code": "idx <- sample(nrow(df), 140)", "seed": 42},
        "baseline_objects": {"model": "fit", "train": "train", "test": "test"}} | overrides)


def project_at_baseline(store, settings, spec, files: dict[str, Path | bytes]) -> str:
    pid = store.create_project("p")
    inputs = settings.project_dir(pid) / "inputs"
    inputs.mkdir(parents=True)
    recorded = []
    for i, (name, content) in enumerate(files.items()):
        dest = inputs / name
        if isinstance(content, Path):
            shutil.copy(content, dest)
        else:
            dest.write_bytes(content)
        recorded.append(InputFile(role="dataset" if i == 0 else "r_code", name=name, sha256=sha256_file(dest)))
    store.apply(pid, FilesUploaded(files=tuple(recorded)))
    v = store.add_spec(pid, spec, support.load())
    s = store.state(pid)
    store.apply(pid, SpecConfirmed(version=v, tier=s.drafted_tier, proven=s.drafted_proven,
                                   baseline_key=canonical_hash(spec.baseline_key()), parity_risk_accepted=True))
    return pid


@pytest.fixture
def jobs(tmp_path, envs, tracker):
    return make_jobs(tmp_path, envs, tracker)


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def test_baseline_is_captured_frozen_tracked_and_reproducible(jobs):
    files = {"claims.csv": CLAIMS.encode(), "model.R": b"# original upload\n"}
    pid = project_at_baseline(jobs.store, jobs.settings, poisson_spec(), files)
    jid = jobs.baseline(pid)

    s = jobs.store.state(pid)
    assert (s.step, s.status) == (Step.TRANSLATION, Status.WORKING), s.detail
    ev = jobs.store.evidence(pid, s.baseline_id)["body"]
    out = jobs.settings.data_dir / ev["dir"]
    assert ev["job_id"] == jid and ev["environment_version"] == jobs.envs.version()

    coefs = read_csv(out / "coefficients.csv")
    assert [c["term"] for c in coefs] == ["(Intercept)", "areaB", "areaC", "age"]
    split = read_csv(out / "split.csv")
    assert sum(r["set"] == "train" for r in split) == 140 and len({r["row"] for r in split}) == 200
    raw = {str(i): line.split(",")[1] for i, line in enumerate(CLAIMS.splitlines()[1:], 1)}
    preds = read_csv(out / "predictions.csv")
    assert len(preds) == 200 and all(float(p["y"]) == float(raw[p["row"]]) for p in preds)   # ids are source rows
    meta = json.loads((out / "meta.json").read_text())
    assert meta["offset"] and meta["row_identity_check"] == "passed" and meta["converged"]
    assert meta["design_columns"] == ["(Intercept)", "areaB", "areaC", "age"]
    assert meta["packages"]["jsonlite"] and "R_LIBS" not in meta
    assert oct((out / "coefficients.csv").stat().st_mode & 0o777) == "0o444"

    run = jobs.tracker.client.get_run(ev["mlflow_run_id"])
    assert run.info.status == "FINISHED" and run.data.params["spec_version"] == "1"
    assert run.data.tags["mlflow.parentRunId"] == jobs.tracker.workflow_run(pid)
    assert float(run.data.metrics["deviance"]) == ev["fit"]["deviance"]
    logged = {a.path for a in jobs.tracker.client.list_artifacts(run.info.run_id)}
    assert {"code", "capture", "environment", "step.log"} <= logged

    # a second project with the same inputs and spec reproduces the same baseline, bit for bit
    pid2 = project_at_baseline(jobs.store, jobs.settings, poisson_spec(), files)
    jobs.baseline(pid2)
    ev2 = jobs.store.evidence(pid2, jobs.store.state(pid2).baseline_id)["body"]
    assert ev2["hash"] == ev["hash"] and ev2["mlflow_run_id"] != ev["mlflow_run_id"]


def test_r_error_fails_the_step_with_the_r_message(jobs):
    spec = poisson_spec(baseline_r=POISSON_R + 'print(head(df))\nstop("boom from baseline")\n')
    pid = project_at_baseline(jobs.store, jobs.settings, spec, {"claims.csv": CLAIMS.encode(), "m.R": b""})
    jobs.baseline(pid)
    s = jobs.store.state(pid)
    assert (s.step, s.status, s.reason) == (Step.BASELINE, Status.AWAITING_DS, Reason.STEP_FAILED)
    assert "exited with status 1" in s.detail and "boom from baseline" in s.detail
    assert "exposure" in s.detail                                  # the DS sees the log, printed rows included
    [note] = jobs.store.chat(pid)
    assert note["author"] == "platform" and note["spec_version"] == 1
    assert "R stopped in baseline.R: Error" in note["text"] and "boom from baseline" in note["text"]
    assert "exposure" not in note["text"]                          # the Analyst gets R's message only
    runs = jobs.tracker.client.search_runs([jobs.tracker.experiment(pid)], "tags.kind = 'baseline'")
    assert [r.info.status for r in runs] == ["FAILED"]


@pytest.mark.parametrize("breakage, message", [
    ("rownames(train) <- NULL\n", "train set's rows cannot be traced"),
    ("test <- test[order(test$age), ]\nrownames(test) <- paste0('r', seq_len(nrow(test)))\n", "test set's rows"),
    ("rm(test)\n", "did not create the test object 'test'"),
])
def test_lost_row_identity_or_missing_objects_fail(jobs, breakage, message):
    spec = poisson_spec(baseline_r=POISSON_R + breakage)
    pid = project_at_baseline(jobs.store, jobs.settings, spec, {"claims.csv": CLAIMS.encode(), "m.R": b""})
    jobs.baseline(pid)
    assert message in jobs.store.state(pid).detail


def test_reset_row_names_cannot_pass_as_source_rows(jobs):
    """Rows 1 and 5 have the same response; swapping them and renumbering 1..n used to pass a response check."""
    swapped = POISSON_R.replace('df <- df[nrow(df):1, ]\n',
                                'df <- df[c(5, 2:4, 1, 6:nrow(df)), ]\nrownames(df) <- NULL\n')
    assert swapped != POISSON_R
    pid = project_at_baseline(jobs.store, jobs.settings, poisson_spec(baseline_r=swapped),
                              {"claims.csv": CLAIMS.encode(), "m.R": b""})
    jobs.baseline(pid)
    assert "rows cannot be traced to rows of the dataset file" in jobs.store.state(pid).detail


LOOKUP = 'lk <- data.frame(area = c("A", "B", "C"), region = c("n", "s", "n"))\n'


def joined(join: str) -> str:
    return POISSON_R.replace("df <- df[nrow(df):1, ]\n", "df <- df[nrow(df):1, ]\n" + LOOKUP + join)


@pytest.mark.parametrize("join", [
    'df <- merge(df, lk, by = "area", all.x = TRUE)\n',
    'df <- merge(lk, df, by = "area")\n',                                           # the dataset on the y side
    # input columns named like the wrapper's key survive untouched, on either side and on both
    'df$.capture_row <- df$age\ndf <- merge(df, lk, by = "area")\nstopifnot(identical(df$.capture_row, df$age))\n',
    'lk$.capture_row <- 7\ndf <- merge(df, lk, by = "area")\n'
    'stopifnot(all(df$.capture_row == 7), !any(c(".capture_row.x", ".capture_row.y") %in% names(df)))\n',
    'df$.capture_row <- df$age\nlk$.capture_row <- 7\ndf <- merge(df, lk, by = "area")\n'
    'stopifnot(identical(df$.capture_row.x, df$age), all(df$.capture_row.y == 7))\n',
])
def test_merge_keeps_each_row_traced_to_its_dataset_row(jobs, join):
    pid = project_at_baseline(jobs.store, jobs.settings, poisson_spec(baseline_r=joined(join)),
                              {"claims.csv": CLAIMS.encode(), "m.R": b""})
    jobs.baseline(pid)
    s = jobs.store.state(pid)
    assert s.step == Step.TRANSLATION, s.detail
    out = jobs.settings.data_dir / jobs.store.evidence(pid, s.baseline_id)["body"]["dir"]
    predictions = read_csv(out / "predictions.csv")
    assert len(predictions) == 200
    assert all(float(r["y"]) == (int(r["row"]) * 7) % 4 for r in predictions)      # claims of that data row


@pytest.mark.parametrize("join", [
    'lk <- rbind(lk, data.frame(area = "A", region = "x"))\ndf <- merge(df, lk, by = "area")\n',   # duplicates rows
    'lk <- rbind(lk, data.frame(area = "D", region = "x"))\ndf <- merge(df, lk, by = "area", all = TRUE)\n',
])
def test_a_join_that_duplicates_or_adds_rows_fails(jobs, join):
    pid = project_at_baseline(jobs.store, jobs.settings, poisson_spec(baseline_r=joined(join)),
                              {"claims.csv": CLAIMS.encode(), "m.R": b""})
    jobs.baseline(pid)
    assert "rows cannot be traced to rows of the dataset file" in jobs.store.state(pid).detail


def test_dataset_read_without_the_tagged_readers_fails(jobs):
    spec = poisson_spec(baseline_r=POISSON_R.replace("read.csv(", "utils::read.csv("))
    pid = project_at_baseline(jobs.store, jobs.settings, spec, {"claims.csv": CLAIMS.encode(), "m.R": b""})
    jobs.baseline(pid)
    assert "row identity with the dataset file cannot be established" in jobs.store.state(pid).detail


def test_lm_is_captured(jobs):
    lm_r = POISSON_R.replace("fit <- glm(claims ~ area + age + offset(log(exposure)), family = poisson, data = train)",
                             "fit <- lm(claims ~ area + age, data = train)")
    spec = poisson_spec(r_function="stats::lm", statsmodels_counterpart="statsmodels.api.OLS", family=None,
                        link=None, offset=None, formula="claims ~ area + age", baseline_r=lm_r)
    pid = project_at_baseline(jobs.store, jobs.settings, spec, {"claims.csv": CLAIMS.encode(), "m.R": b""})
    jobs.baseline(pid)
    s = jobs.store.state(pid)
    assert s.step == Step.TRANSLATION, s.detail
    ev = jobs.store.evidence(pid, s.baseline_id)["body"]
    assert ev["meta"]["class"] == ["lm"] and ev["meta"]["link"] == "identity" and not ev["meta"]["offset"]
    assert {"r_squared", "sigma", "rss"} <= set(ev["fit"])
    preds = read_csv(jobs.settings.data_dir / ev["dir"] / "predictions.csv")
    assert len(preds) == 200 and all(p["link"] == p["response"] for p in preds)


def test_modified_upload_is_refused(jobs):
    pid = project_at_baseline(jobs.store, jobs.settings, poisson_spec(), {"claims.csv": CLAIMS.encode(), "m.R": b""})
    p = jobs.settings.project_dir(pid) / "inputs" / "claims.csv"
    p.write_text(CLAIMS + "\n201,0,1,A,30")
    jobs.baseline(pid)
    assert "no longer matches its intake hash" in jobs.store.state(pid).detail


def test_cancel_kills_r_and_closes_the_job(jobs):
    spec = poisson_spec(baseline_r=POISSON_R + "Sys.sleep(60)\n")
    pid = project_at_baseline(jobs.store, jobs.settings, spec, {"claims.csv": CLAIMS.encode(), "m.R": b""})
    t = threading.Timer(1.0, lambda: jobs.cancel(jobs.store.running_job(pid)["id"]))
    t.start()
    jobs.baseline(pid)
    s = jobs.store.state(pid)
    assert (s.status, s.reason) == (Status.AWAITING_DS, Reason.CANCELLED)
    runs = jobs.tracker.client.search_runs([jobs.tracker.experiment(pid)], "tags.kind = 'baseline'")
    assert [r.info.status for r in runs] == ["KILLED"]


@pytest.mark.skipif(not (M4_DIR / "data" / "cs-training.csv").exists(), reason=f"M4 not found at {M4_DIR}")
def test_m4_baseline_matches_the_unmodified_notebook(jobs, tmp_path):
    spec = ModelSpec.model_validate_json((FIXTURES / "m4" / "spec.json").read_text())
    original = M4_DIR / "code" / "givemesomecredit_logistic_notebook.R"
    pid = project_at_baseline(jobs.store, jobs.settings, spec, {
        "cs-training.csv": M4_DIR / "data" / "cs-training.csv", original.name: original})
    jobs.baseline(pid)
    s = jobs.store.state(pid)
    assert s.step == Step.TRANSLATION, s.detail
    ev = jobs.store.evidence(pid, s.baseline_id)["body"]
    out = jobs.settings.data_dir / ev["dir"]
    meta = ev["meta"]
    assert meta["n"] == {"train": 150000} and meta["row_identity_check"] == "passed" and meta["converged"]
    assert meta["response_levels"] == ["0", "1"] and len(meta["design_columns"]) == 9

    # the notebook as downloaded, run in the layout it expects (../input/), in the same R environment
    nb = tmp_path / "nb"
    (nb / "code").mkdir(parents=True)
    (nb / "input").mkdir()
    shutil.copy(M4_DIR / "data" / "cs-training.csv", nb / "input")
    shutil.copy(M4_DIR / "data" / "cs-test.csv", nb / "input")
    shutil.copy(original, nb / "code" / "notebook.R")
    # same execution conditions as the baseline: the runner pins BLAS to one thread, because the last bits of a
    # fit depend on the thread count (a different R or BLAS build can still change them)
    lib = str(jobs.envs.r_library)
    out_nb = run([jobs.envs.r(), "--vanilla", "-e",
                  'sys.source("notebook.R", e <- new.env()); '
                  'write.csv(data.frame(t = names(coef(e$RegModel)), v = sprintf("%.17g", coef(e$RegModel))), '
                  '"coef.csv", row.names = FALSE)'],
                 nb / "code", {"R_LIBS": lib, "R_LIBS_USER": lib, "R_LIBS_SITE": lib})
    assert out_nb.ok, out_nb.describe()
    want = {row["t"]: row["v"] for row in read_csv(nb / "code" / "coef.csv")}
    got = {row["term"]: row["estimate"] for row in read_csv(out / "coefficients.csv")}
    assert got == want                                  # identical to the last bit
