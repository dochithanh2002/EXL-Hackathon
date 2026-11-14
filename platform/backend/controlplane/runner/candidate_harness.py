"""Python candidate harness, the counterpart of capture.R. Runs in the stable Python environment:
    python -I candidate_harness.py <config.json>
    config: candidate, dataset, split (R's split.csv), response, out
Imports the candidate, hands it R's sets, checks it returns exactly R's rows, fits, and exports the coefficients,
predictions and fit statistics the validator compares with R's capture. A failure writes the stage, exception type,
message and the candidate's own frames (line and function) to error.json, like capture.R; the full traceback goes
only to the step log."""
import csv
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.base.model import Results
from statsmodels.base.wrapper import ResultsWrapper

FIT_STATS = ("nobs", "df_resid", "llf", "deviance", "null_deviance", "scale", "pearson_chi2", "ssr", "rsquared",
             "rsquared_adj")

cfg = json.loads(Path(sys.argv[1]).read_text())
out = Path(cfg["out"])
out.mkdir()
stage = "the candidate harness"


def number(x) -> str:
    x = float(x)
    return "" if math.isnan(x) else repr(x)


def write_csv(name: str, header: list[str], rows) -> None:
    with open(out / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def stat(result, name: str):
    try:
        v = float(getattr(result, name))
    except Exception:                       # absent on this results class, or it failed to compute
        return None
    return v if math.isfinite(v) else None


try:
    split = pd.read_csv(cfg["split"])
    sets = {s: pd.Index(g["row"].to_numpy(), name="row") for s, g in split.groupby("set", sort=False)}

    # Check the resolved path too: a candidate could rebind prepare's path argument.
    dataset_path = Path(cfg["dataset"]).resolve()

    def dataset_reader(reader):
        def read(filepath_or_buffer, *args, **kwargs):
            if not isinstance(filepath_or_buffer, (str, Path)) or Path(filepath_or_buffer).resolve() != dataset_path:
                raise ValueError("only the dataset may be read")
            return reader(filepath_or_buffer, *args, **kwargs)
        return read

    pd.read_csv = dataset_reader(pd.read_csv)
    pd.read_table = dataset_reader(pd.read_table)

    stage = "importing the candidate"
    spec = importlib.util.spec_from_file_location("candidate", cfg["candidate"])
    candidate = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = candidate
    spec.loader.exec_module(candidate)

    stage = "prepare()"
    frames = candidate.prepare(cfg["dataset"], {k: v.copy() for k, v in sets.items()})
    stage = "checking prepare()'s frames"
    if not isinstance(frames, dict) or set(frames) != set(sets):
        raise ValueError(f"prepare() must return a dict with keys {sorted(sets)}")
    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"prepare()['{name}'] is not a pandas DataFrame")
        if not frame.index.is_unique:
            raise ValueError(f"prepare()['{name}'] has duplicate index labels")
        missing, extra = sets[name].difference(frame.index), frame.index.difference(sets[name])
        if len(missing) or len(extra):
            raise ValueError(f"prepare()['{name}'] must be indexed by exactly the {len(sets[name])} rows R put in the "
                             f"{name} set; {len(missing)} are missing and {len(extra)} are not in it")
        if cfg["response"] not in frame.columns:
            raise ValueError(f"prepare()['{name}'] has no response column '{cfg['response']}'")
        y = frame[cfg["response"]]
        if not (pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y)):
            raise TypeError(f"the response column '{cfg['response']}' must be numeric, not {y.dtype}")

    stage = "fit()"
    t0 = time.monotonic()
    result = candidate.fit(frames["train"])
    runtime = time.monotonic() - t0
    stage = "reading the fitted results"
    if not isinstance(result, (Results, ResultsWrapper)):
        raise TypeError("fit() must return a statsmodels results object")
    params, bse = result.params, result.bse
    terms = list(params.index) if isinstance(params, pd.Series) else list(result.model.exog_names)
    params, bse = np.asarray(params, dtype=float), np.asarray(bse, dtype=float)
    if not (len(terms) == len(params) == len(bse)):
        raise ValueError("the results' params and bse do not have one value per coefficient")
    write_csv("coefficients.csv", ["term", "estimate", "std_error"],
              ((str(t), number(e), number(s)) for t, e, s in zip(terms, params, bse)))
    (out / "fit.json").write_text(json.dumps({k: stat(result, k) for k in FIT_STATS}, indent=1))

    rows = []
    for name, frame in frames.items():
        stage = f"predict() on the {name} set"
        p = candidate.predict(result, frame)
        stage = f"checking predict()'s output on the {name} set"
        if isinstance(p, pd.Series):
            if not (p.index.is_unique and len(p) == len(frame) and p.index.isin(frame.index).all()):
                raise ValueError(f"predict() on the {name} set must return one value per row, indexed like the frame")
            p = p.reindex(frame.index)
        else:
            p = pd.Series(np.asarray(p, dtype=float).ravel(), index=frame.index) if np.ndim(p) else None
            if p is None or len(p) != len(frame):
                raise ValueError(f"predict() on the {name} set must return one value per row")
        y = frame[cfg["response"]].to_numpy(dtype=float, na_value=np.nan)
        rows += [(int(r), name, number(a), number(b)) for r, a, b in zip(frame.index, y, p.to_numpy(dtype=float))]
    write_csv("predictions.csv", ["row", "set", "y", "response"], rows)
    (out / "timing.json").write_text(json.dumps({"fit_s": runtime}))
except Exception as e:
    traceback.print_exc()
    source = Path(cfg["candidate"]).resolve()
    frames = [{"line": f.lineno, "function": f.name} for f in traceback.extract_tb(e.__traceback__)
              if Path(f.filename).resolve() == source]
    (out / "error.json").write_text(json.dumps({"stage": stage, "type": type(e).__name__,
                                                "message": str(e)[:1000], "candidate_frames": frames}))
    sys.exit(1)
print(f"exported {len(rows)} predictions")
