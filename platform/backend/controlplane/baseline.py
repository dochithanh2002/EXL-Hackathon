"""Step 3: run the confirmed spec's baseline.R under the capture harness and freeze what it exports."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .environments import Environments
from .runner import Limits, Process, StepCancelled
from .schemas import ModelSpec, canonical_hash, sha256_file
from .workflow.machine import InputFile

CAPTURE_R = Path(__file__).parent / "runner" / "capture.R"
CAPTURED = ("coefficients.csv", "split.csv", "predictions.csv", "fit.json", "meta.json", "sessionInfo.txt")


class BaselineFailed(Exception):
    """`str()` is the full detail for the DS; `reason` is safe to show the Analyst (no log lines)."""

    def __init__(self, detail: str, reason: str | None = None):
        super().__init__(detail)
        self.reason = reason or detail


@dataclass(frozen=True)
class Baseline:
    dir: Path
    files: dict[str, str]                 # captured file -> sha256
    environment_version: str
    fit: dict
    meta: dict
    runtime_s: float

    @property
    def hash(self) -> str:
        """Identifies the baseline's content; equal across reruns of the same spec in the same environment."""
        return canonical_hash({"files": self.files, "environment_version": self.environment_version})


def copy_verified(src: Path, dest: Path, sha256: str) -> None:
    """Copy an uploaded file into a step's working directory, checking it still has its intake hash."""
    h = hashlib.sha256()
    with open(src, "rb") as f, open(dest, "wb") as out:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            out.write(chunk)
    if h.hexdigest() != sha256:
        dest.unlink()
        raise BaselineFailed(f"uploaded file {src.name} no longer matches its intake hash")
    dest.chmod(0o444)


def run_baseline(workdir: Path, spec: ModelSpec, files: tuple[InputFile, ...], inputs_dir: Path, envs: Environments,
                 limits: Limits = Limits(), on_start: Callable[[Process], None] = lambda p: None) -> Baseline:
    workdir.mkdir(parents=True)
    (workdir / "inputs").mkdir()
    for f in files:
        if f.role != "document":
            copy_verified(inputs_dir / f.name, workdir / "inputs" / f.name, f.sha256)
    (workdir / "inputs").chmod(0o555)
    (workdir / "baseline.R").write_text(spec.baseline_r)
    shutil.copy(CAPTURE_R, workdir / "capture.R")
    dataset = next(f.name for f in files if f.role == "dataset")
    (workdir / "config.json").write_text(json.dumps({
        "script": "baseline.R", "dataset": f"inputs/{dataset}", "response": spec.response,
        "objects": spec.baseline_objects.model_dump(), "seed": spec.split.seed, "out": "capture"}))

    proc = Process([str(envs.r()), "--vanilla", "capture.R", "config.json"], workdir, envs.r_env(), limits)
    on_start(proc)
    outcome = proc.start().wait()
    if outcome.cancelled:
        raise StepCancelled()
    if not outcome.ok:
        error = workdir / "capture" / "error.json"
        if error.is_file():
            e = json.loads(error.read_text())
            reason = f"R stopped in {e['stage']}: {e['message']}"
        else:
            reason = "R " + outcome.describe(lines=0)
        raise BaselineFailed("R baseline " + outcome.describe(), reason)

    out = workdir / "capture"
    missing = [n for n in CAPTURED if not (out / n).is_file()]
    if missing:
        raise BaselineFailed(f"the capture harness did not write {missing}")
    for p in out.iterdir():
        p.chmod(0o444)
    out.chmod(0o555)
    return Baseline(
        dir=out, files={n: sha256_file(out / n) for n in CAPTURED}, environment_version=envs.version(),
        fit=json.loads((out / "fit.json").read_text()), meta=json.loads((out / "meta.json").read_text()),
        runtime_s=json.loads((out / "timing.json").read_text())["runtime_s"])
