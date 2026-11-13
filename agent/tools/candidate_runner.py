"""Executes ONE generated candidate in an isolated interpreter (guardrail G-06).

Invoked by the agent as:  python -I candidate_runner.py <candidate.py> <data.csv> <split.csv> <out_dir>
It imports the candidate, enforces row identity (G-07), fits on the R-captured training rows
(R-RNG-01), predicts every row and writes coefficients + predictions for the parity check.
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def main(candidate: str, data: str, split: str, out: str) -> int:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {"ok": False}
    try:
        spec = importlib.util.spec_from_file_location("candidate", candidate)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["candidate"] = mod      # required for dataclasses / typing introspection
        spec.loader.exec_module(mod)

        n_raw = len(pd.read_csv(data, usecols=[0]))          # parsed, so quoted newlines are counted correctly
        df = mod.load_and_preprocess(data)
        # G-07 row identity: the index must be the raw file's 0-based row position. Rows may be dropped
        # (when R drops them) but never reordered, duplicated or relabelled.
        idx = df.index
        if not pd.api.types.is_integer_dtype(idx):
            raise AssertionError(f"G-07 row identity: index must be integer raw-row positions, got dtype {idx.dtype}")
        if not idx.is_unique or not idx.is_monotonic_increasing:
            raise AssertionError("G-07 row identity: index must be unique and in original row order (do not sort or reset_index)")
        if len(idx) and (idx.min() < 0 or idx.max() >= n_raw):
            raise AssertionError(f"G-07 row identity: index outside raw row range 0..{n_raw - 1}")
        df = df.copy()
        df["__row__"] = np.asarray(idx) + 1                 # R rownames = original 1-based row number

        sp = pd.read_csv(split)
        missing = sorted(set(sp["row"]) - set(df["__row__"]))
        if missing:
            raise AssertionError(f"G-07 row identity: {len(missing)} rows used by R (train/test) are missing from "
                                 f"load_and_preprocess, e.g. raw rows {missing[:10]} (1-based)")
        df = df.merge(sp, left_on="__row__", right_on="row", how="inner", validate="one_to_one")
        train, test = df[df["set"] == "train"], df[df["set"] == "test"]

        res = mod.fit(train)
        coefs = mod.coefficients(res)
        # G-03 at runtime: the public design matrix the independent validator uses must match TERMS and the fit
        dm_cols = list(mod.design_matrix(train.drop(columns=["__row__", "row", "set"])).columns)
        if dm_cols != list(mod.TERMS) or list(mod.TERMS) != list(coefs["term"]):
            raise AssertionError(f"G-03 contract: design_matrix columns {dm_cols[:6]}..., TERMS {list(mod.TERMS)[:6]}... and "
                                 f"coefficients() terms {list(coefs['term'])[:6]}... must be identical and in the same order")
        if not {"term", "estimate", "std_error"} <= set(coefs.columns):
            raise AssertionError("coefficients() must return columns term, estimate, std_error")
        coefs.to_csv(out_dir / "coefficients.csv", index=False, float_format="%.17g")

        rows = []
        for part, name in ((train, "train"), (test, "test")):
            p = np.asarray(mod.predict(res, part), dtype=float)
            if p.shape[0] != len(part):
                raise AssertionError("predict() must return one value per row")
            rows.append(pd.DataFrame({"row": part["__row__"].values, "set": name,
                                      "y": part[mod.TARGET].values, "pd": p}))
        pd.concat(rows).sort_values("row").to_csv(out_dir / "predictions.csv", index=False, float_format="%.17g")
        status = {"ok": True, "n_rows": int(len(df)), "n_train": int(len(train)), "n_test": int(len(test))}
    except Exception as e:  # report every failure back to the repair loop
        status = {"ok": False, "error_type": type(e).__name__, "error": str(e)[:2000],
                  "traceback": traceback.format_exc()[-4000:]}
    (out_dir / "run_status.json").write_text(json.dumps(status, indent=2))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
