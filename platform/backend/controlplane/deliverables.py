"""Step 7: the deliverables bundle, built from recorded evidence only.

The bundle holds the approved module, the harness and a standalone `verify.py` that reruns it against the frozen R
reference, baseline.R, the confirmed spec, both lockfiles, the approved validation, the final report and rerun
instructions. `manifest.json` gives every file's SHA-256 and what the approval was bound to; the bundle is also one
zip archive."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from .baseline import CAPTURED
from .environments import Environments
from .report import Record
from .schemas import canonical_hash, sha256_file
from .translation import HARNESS

VERIFY = Path(__file__).parent / "runner" / "bundle_verify.py"
PARITY = VERIFY.with_name("parity.py")

README = """# {project}

`{r_function}` in R, reproduced in Python with `{counterpart}` ({tier} tier) and approved by the data scientist.

| File | |
|---|---|
| `model.py` | The model: `prepare`, `fit` and `predict` |
| `report.md` | Final report |
| `validation.json` | The approved validation against the R baseline |
| `spec.json` | The confirmed model spec, v{spec_version} |
| `baseline.R` | The R code the baseline ran |
| `reference/` | The R baseline's capture: coefficients, predictions, R's train and test rows, fit statistics |
| `environment/` | `pyproject.toml` and `uv.lock` for Python, `renv.lock` for R |
| `verify.py`, `parity.py`, `harness.py`, `verify.json` | Rerun and compare with R |
| `manifest.json` | SHA-256 of every file |

## Rerun

```sh
uv sync --frozen --project environment
uv run --frozen --project environment python verify.py path/to/{dataset}
```
{data}
`verify.py` checks the SHA-256 of every data file, fits `model.py` on R's train rows, predicts on every set R formed and
repeats the approval checks against `reference/`: coefficient names, estimates and standard errors, row identities,
response values, prediction missingness and values, fit statistics and the requested KPIs within the spec's
tolerances. Statistics excluded during approval remain excluded and are listed in the output.

The R baseline used R {r_version} with the packages in `environment/renv.lock`
(`renv::restore(lockfile = "environment/renv.lock")`).
"""


def build_bundle(dest: Path, record: Record, report_md: str, envs: Environments) -> dict:
    """Write the bundle to `dest` and its archive to `dest`.zip. Returns the manifest."""
    r, a = record, record.approved
    # Capture bytes once: the files hashed here are exactly the ones written to the bundle, even if the shared
    # environments are updated during packaging. An older approval must never silently acquire newer lockfiles.
    environment = {p.name: p.read_bytes() for p in (
        envs.python_lockfile.with_name("pyproject.toml"), *envs.lockfiles.values())}
    version = canonical_hash({name: hashlib.sha256(environment[name]).hexdigest() for name in envs.lockfiles})
    if version != r.environment_version:
        raise ValueError("the environment lockfiles no longer match the approved validation; restore the approved "
                         "lockfiles before retrying deliverables")
    dest.mkdir(parents=True)
    dataset = next(f for f in r.state.files if f.role == "dataset")
    data = {f.name: f.sha256 for f in r.state.files if f.role == "data"}
    (dest / "model.py").write_text(a.candidate["body"]["module"])
    (dest / "report.md").write_text(report_md)
    (dest / "validation.json").write_text(json.dumps(a.validation["body"], indent=1, sort_keys=True))
    (dest / "spec.json").write_text(r.spec.model_dump_json(indent=2))
    (dest / "baseline.R").write_text(r.spec.baseline_r)
    shutil.copy(HARNESS, dest / "harness.py")
    shutil.copy(VERIFY, dest / "verify.py")
    shutil.copy(PARITY, dest / "parity.py")
    (dest / "verify.json").write_text(json.dumps({"dataset_sha256": dataset.sha256, "data_sha256": data}, indent=1))
    (dest / "reference").mkdir()
    for name in CAPTURED:
        shutil.copy(r.baseline_dir / name, dest / "reference" / name)
    (dest / "environment").mkdir()
    for name, content in environment.items():
        (dest / "environment" / name).write_bytes(content)
    (dest / "README.md").write_text(README.format(
        project=r.project, r_function=r.spec.r_function, counterpart=r.spec.statsmodels_counterpart,
        tier=r.state.spec_tier, spec_version=r.state.spec_version, dataset=dataset.name,
        data="" if not data else "\n" + " ".join(f"`{n}`" for n in data) + " must be in the same directory as the "
        "dataset.\n",
        r_version=r.baseline["body"]["meta"]["r_version"].removeprefix("R version ")))

    manifest = {
        "project": r.project, "bindings": r.state.approval.model_dump(), "spec_hash": r.spec_hash,
        "baseline_hash": r.baseline["body"]["hash"], "environment_version": r.environment_version,
        "dataset": {"name": dataset.name, "sha256": dataset.sha256}, "data": data,
        "files": {str(p.relative_to(dest)): sha256_file(p) for p in sorted(dest.rglob("*")) if p.is_file()}}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    with zipfile.ZipFile(dest.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob("*")):
            if p.is_file():
                z.write(p, Path(dest.name) / p.relative_to(dest))
    return manifest
