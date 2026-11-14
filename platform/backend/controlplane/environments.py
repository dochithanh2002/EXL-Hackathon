"""The two stable execution environments under `platform/envs`, built by `bootstrap.sh`.

`bootstrap.sh` writes the SHA-256 of the lockfile it restored next to each environment; an environment whose
marker does not match its lockfile is stale and steps refuse to run in it."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .schemas import canonical_hash, sha256_file

ENVS_DIR = Path(__file__).resolve().parents[2] / "envs"
MARKER = ".lock-sha256"
R_BASE = frozenset({"base", "compiler", "datasets", "graphics", "grDevices", "grid", "methods", "parallel", "splines",
                    "stats", "stats4", "tcltk", "tools", "utils"})


class EnvironmentNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class Environments:
    root: Path = ENVS_DIR
    rscript: Path | None = None

    @property
    def r_lockfile(self) -> Path:
        return self.root / "r" / "renv.lock"

    @property
    def r_library(self) -> Path:
        return self.root / "r" / "library"

    @property
    def python_lockfile(self) -> Path:
        return self.root / "python" / "uv.lock"

    @property
    def python(self) -> Path:
        return self.root / "python" / ".venv" / "bin" / "python"

    @property
    def lockfiles(self) -> dict[str, Path]:
        return {"renv.lock": self.r_lockfile, "uv.lock": self.python_lockfile}

    def version(self) -> str:
        """Hash of the two lockfiles; recorded on every run."""
        return canonical_hash({name: sha256_file(p) for name, p in self.lockfiles.items()})

    def r_packages(self) -> frozenset[str]:
        """Every package baseline.R can load: R's base packages and the locked ones."""
        return R_BASE | json.loads(self.r_lockfile.read_text())["Packages"].keys()

    def r(self) -> Path:
        rscript = self.rscript or os.environ.get("CONTROLPLANE_RSCRIPT") or shutil.which("Rscript")
        if not rscript:
            raise EnvironmentNotReady("Rscript not found; set CONTROLPLANE_RSCRIPT")
        self._check(self.r_library, self.r_lockfile)
        return Path(rscript)

    def r_env(self) -> dict[str, str]:
        """Environment for an Rscript step: only the stable library is visible."""
        lib = str(self.r_library)
        return {"PATH": f"{self.r().parent}:/usr/bin:/bin", "R_LIBS": lib, "R_LIBS_USER": lib, "R_LIBS_SITE": lib}

    def py(self) -> Path:
        self._check(self.python.parents[1], self.python_lockfile)
        return self.python

    @staticmethod
    def _check(target: Path, lockfile: Path) -> None:
        marker = target / MARKER
        if not marker.is_file():
            raise EnvironmentNotReady(f"{target} is not built; run platform/bootstrap.sh")
        if marker.read_text().strip() != sha256_file(lockfile):
            raise EnvironmentNotReady(f"{target} was built from an older {lockfile.name}; run platform/bootstrap.sh")
