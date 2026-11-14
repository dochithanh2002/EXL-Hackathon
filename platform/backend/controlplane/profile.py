"""What the Analyst sees of the dataset: column names, R classes, level sets and summary statistics, never rows.

The profile is computed by `runner/profile.R` in the stable R environment with `read.csv` defaults, so it describes
the data frame exactly as baseline.R reads it (row names taken from the file, NaN counted as missing, and so on)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .environments import Environments
from .runner import Limits, Process

PROFILE_R = Path(__file__).parent / "runner" / "profile.R"
MAX_LEVELS = 20


class ProfileFailed(Exception):
    pass


def profile(dataset: Path, workdir: Path, envs: Environments, limits: Limits = Limits()) -> dict:
    """Run the R profiler in a fresh working directory. On failure the message is R's error only, never the log,
    which could hold printed rows."""
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    out = workdir / "profile.json"
    outcome = Process([str(envs.r()), "--vanilla", str(PROFILE_R), str(dataset), str(out), str(MAX_LEVELS)],
                      workdir, envs.r_env(), limits).start().wait()
    if not outcome.ok:
        error = workdir / "profile.json.error"
        reason = error.read_text().strip() if error.is_file() else outcome.describe(lines=0)
        raise ProfileFailed(f"R could not read the dataset with read.csv: {reason}")
    return json.loads(out.read_text())
