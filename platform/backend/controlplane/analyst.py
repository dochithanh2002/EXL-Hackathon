"""Step 2: what the Analyst sees, its read-only tools, and the deterministic checks on its submissions.

The Analyst reads R code and text documents and profiles of the dataset and any other data files; no data file is
ever read into its context. A submission is a ModelSpec only if it validates and passes every check here; otherwise the errors go
back to the Analyst."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from pydantic import ValidationError

from .environments import R_BASE, Environments
from .profile import ProfileFailed, profile
from .runner import Limits
from .schemas import ModelSpec, sha256_file
from .support import SupportMatrix
from .workflow.machine import InputFile

TEXT = {".r", ".rmd", ".qmd", ".md", ".txt"}
MAX_CHARS = 400_000
MAX_INVALID = 3
PACKAGE_USE = re.compile(r"\b(?:library|require|requireNamespace)\(\s*['\"]?([A-Za-z][\w.]*)|([A-Za-z][\w.]*):::?")


class ToolError(Exception):
    pass


def tool_schema(model: type[ModelSpec]) -> dict:
    """The model's JSON Schema with every `$ref` inlined, which every provider's tool API accepts."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].rsplit("/", 1)[1]])
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node
    return inline(schema)


class Analyst:
    """One project's Analyst inputs and checks."""

    def __init__(self, files: tuple[InputFile, ...], inputs_dir: Path, cache_dir: Path, matrix: SupportMatrix,
                 envs: Environments, limits: Limits = Limits()):
        self.files = {f.name: f for f in files}
        self.dataset = next(f for f in files if f.role == "dataset")
        self.data = [f for f in files if f.role == "data"]
        self.inputs_dir, self.cache_dir, self.matrix = inputs_dir, cache_dir, matrix
        self.envs, self.limits, self.r_packages = envs, limits, envs.r_packages()

    # ------------------------------------------------------------ tools
    def read_file(self, name: str) -> str:
        f = self.files.get(name)
        if f is None:
            raise ToolError(f"no uploaded file is named {name!r}; the files are {sorted(self.files)}")
        if f.role in ("dataset", "data"):
            raise ToolError(f"{name} is data and cannot be read; use dataset_profile")
        if Path(name).suffix.lower() not in TEXT:
            raise ToolError(f"{name} is not a text file")
        text = self._verified(f).read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_CHARS:
            return text[:MAX_CHARS] + f"\n[truncated: the file has {len(text)} characters]"
        return text

    def profile(self, name: str | None = None) -> dict:
        """The profile of the dataset, or of the data file `name`; computed once per file and environment version."""
        f = self.dataset if name in (None, "", self.dataset.name) else self.files.get(name)
        if f is None or f.role not in ("dataset", "data"):
            data = [d.name for d in self.data]
            raise ToolError(f"no data file is named {name!r}" + (f"; the data files are {data}" if data else
                                                                 "; omit the name for the dataset"))
        cached = self.cache_dir / f"profile-{f.sha256[:16]}-{self.envs.version()[:16]}.json"
        if not cached.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(dir=self.cache_dir))
            try:
                p = profile(self._verified(f), work / "run", self.envs, self.limits)
            except ProfileFailed as e:
                raise ToolError(str(e)) from e
            finally:
                shutil.rmtree(work, ignore_errors=True)
            tmp = cached.with_suffix(".tmp")
            tmp.write_text(json.dumps(p))
            os.replace(tmp, cached)
        return json.loads(cached.read_text())

    def columns(self) -> dict:
        """Every data file's columns in one profile-shaped dict: the names error-message redaction may keep."""
        return {"columns": [c for f in [self.dataset, *self.data] for c in self.profile(f.name)["columns"]]}

    def _verified(self, f: InputFile) -> Path:
        path = self.inputs_dir / f.name
        if sha256_file(path) != f.sha256:
            raise ToolError(f"uploaded file {f.name} no longer matches its intake hash")
        return path

    # ------------------------------------------------------------ prompt
    def kickoff(self) -> str:
        p = self.profile()
        lines = ["Project files:"]
        for f in self.files.values():
            if f.role == "dataset":
                rn = ", the first field of each row holds row names" if p["row_names_from_file"] else ""
                lines.append(f"- {f.name}: dataset, {p['rows']} rows, {len(p['columns'])} columns{rn}")
            elif f.role == "data":
                q = self.profile(f.name)
                lines.append(f"- {f.name}: other data, {q['rows']} rows, {len(q['columns'])} columns")
            elif f.role == "r_code":
                n = self.read_file(f.name).count("\n") + 1
                lines.append(f"- {f.name}: R code, {n} lines")
            else:
                lines.append(f"- {f.name}: document")
        lines += ["", f"baseline.R runs in a fresh directory and reads the dataset as inputs/{self.dataset.name}"
                  + "".join(f", and {d.name} as inputs/{d.name}" for d in self.data) + ". The model's rows come from "
                  "the dataset; the other data files may only be joined or aggregated into it.", "",
                  "Support matrix (R function [families]: statsmodels counterparts, tier). Anything else is "
                  "unsupported:"]
        for pair in self.matrix.pairs:
            fam = f" [{', '.join(pair.families)}]" if pair.families else ""
            lines.append(f"- {pair.r}{fam}: {', '.join(pair.statsmodels)} ({pair.tier})")
        lines += ["", "R packages baseline.R can load: " + ", ".join(sorted(self.r_packages, key=str.lower)), "",
                  "Read the files and the data profiles (dataset_profile, with a data file's name for the others), "
                  "then submit the ModelSpec."]
        return "\n".join(lines)

    # ------------------------------------------------------------ checks
    def check(self, raw: dict) -> tuple[ModelSpec | None, list[str]]:
        try:
            spec = ModelSpec.model_validate(raw)
        except ValidationError as e:
            return None, [f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors()]
        errors = self.matrix.check(spec)
        path = f"inputs/{self.dataset.name}"
        if path not in spec.baseline_r:
            errors.append(f"baseline_r must read the dataset from {path}")
        used = {a or b for a, b in PACKAGE_USE.findall(spec.baseline_r)} - R_BASE
        if missing := sorted(used - set(spec.r_packages)):
            errors.append(f"baseline_r loads {missing}, which r_packages does not list")
        if spec.support_tier != "unsupported" and (outside := sorted(set(spec.r_packages) - self.r_packages)):
            errors.append(f"the stable R environment has no {outside}: drop them from baseline_r if the target model "
                          f"does not need them; otherwise set support_tier to unsupported and list them in "
                          f"unsupported_items")
        objects = spec.baseline_objects.model_dump()
        for field, name in objects.items():
            if name is not None and not re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.])", spec.baseline_r):
                errors.append(f"baseline_objects.{field} '{name}' does not appear in baseline_r")
        return (None if errors else spec), errors
