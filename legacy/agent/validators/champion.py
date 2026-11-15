"""Generic champion loader: the model under validation is the module the Migration Agent generated.

`Champion("m2_taiwan")` reads input/<model>/model.yaml, loads output/<model>/model/<module>.py, the frozen R reference,
and exposes: module, load() (agent preprocessing + raw id + frozen split), fit(), provenance() (module SHA-256,
manifest hash, migration status) so every validation result is tied to the exact validated artefact.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from agent.tools.paths import ROOT, Model  # noqa: F401


def fitted_stats(module, res, train: pd.DataFrame, family: str) -> dict:
    """What the validator needs from a fit, using ONLY the interface contract (predict + TARGET).

    Deviance is recomputed by the validator from the module's own training predictions, so validation never depends
    on the shape of the object `fit()` returns. Iteration count / convergence are read if the result exposes them.
    """
    from agent.tools.parity import DEVIANCE
    mu = np.asarray(module.predict(res, train), dtype=float)
    y = train[module.TARGET].to_numpy(dtype=float)
    dev = DEVIANCE.get(family)
    get = (lambda k: res.get(k)) if isinstance(res, dict) else (lambda k: getattr(res, k, None))
    return {"deviance": float(dev(y, mu)) if dev else None,
            "iterations": get("iterations") or get("iter"), "converged": get("converged")}


class Fit:
    """Adapter over the agent module's fitted result: params, bse, deviance, llf (binomial), predict_df."""

    def __init__(self, champ: "Champion", res, train: pd.DataFrame):
        m = champ.module
        self.champ, self.res = champ, res
        co = m.coefficients(res).set_index("term")
        self.params, self.bse = co["estimate"], co["std_error"]
        st = fitted_stats(m, res, train, champ.family)
        self.deviance, self.iterations, self.converged = st["deviance"], st["iterations"], st["converged"]
        fam = champ.family
        if fam == "binomial":
            p = np.clip(m.predict(res, train), 1e-15, 1 - 1e-15)
            y = train[m.TARGET].to_numpy(dtype=float)
            self.llf = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        else:
            self.llf = None

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.champ.module.predict(self.res, df), index=df.index)


class Champion:
    def __init__(self, model: str, id_column: str | None = None):
        self.m = Model(model)
        self.cfg = self.m.cfg
        self.id_column = id_column or self.cfg.get("data", {}).get("id_column")

    @property
    def output_dir(self) -> Path:
        return self.m.model_dir

    @property
    def module_path(self) -> Path:
        return self.m.module_path

    @property
    def reference_dir(self) -> Path:
        return self.m.reference

    @property
    def data_path(self) -> Path:
        return self.m.data

    @property
    def family(self) -> str:
        return json.loads((self.reference_dir / "meta.json").read_text()).get("family", "gaussian")

    @cached_property
    def module(self):
        p = self.module_path
        if not p.exists():
            raise FileNotFoundError(f"{p} not found: run the migration agent first "
                                    f"(/mrf:run {self.m.slug})")
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[p.stem] = mod
        spec.loader.exec_module(mod)
        return mod

    def provenance(self) -> dict:
        man = json.loads((self.output_dir / "manifest.json").read_text())
        state = json.loads((self.m.mig_dir / "state.json").read_text())
        return {"module": str(self.module_path.relative_to(ROOT)),
                "module_sha256": hashlib.sha256(self.module_path.read_bytes()).hexdigest(),
                "manifest_module_sha256": man["files"].get(self.module_path.name),
                "migration_status": state.get("status"), "migration_iterations": len(state.get("iterations", [])),
                "human_review": state.get("human_review"),
                "migrated_by": man.get("migrated_by"),
                "reference_dir": str(self.reference_dir.relative_to(ROOT))}

    def load(self) -> pd.DataFrame:
        """Data preprocessed by the agent module (all rows, original order), plus raw id and frozen split."""
        df = self.module.load_and_preprocess(str(self.data_path))
        if self.id_column:
            df["id"] = pd.read_csv(self.data_path, usecols=[self.id_column])[self.id_column].to_numpy()[df.index]
        split = pd.read_csv(self.reference_dir / "split.csv").set_index("row")["set"]
        df["set"] = split.reindex(np.asarray(df.index) + 1).to_numpy()
        if df["set"].isna().any():
            raise ValueError("frozen split does not cover every row")
        return df

    def fit(self, train: pd.DataFrame) -> Fit:
        return Fit(self, self.module.fit(train), train)
