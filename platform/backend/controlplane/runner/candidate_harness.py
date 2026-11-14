"""Python candidate harness, the counterpart of capture.R. Runs in the stable Python environment:
    python -I candidate_harness.py <config.json>
    config: candidate, dataset, data (other data file name -> path; may be empty), split (R's split.csv), response,
    response_type, prediction {kind, keys}, out
Imports the candidate, hands it R's sets, checks it returns exactly R's rows, fits, and exports the coefficients,
predictions (one row per set, data row and key, like capture.R), fit statistics and a design summary the validator
compares with R's capture. A failure writes the stage, exception type,
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
FIT_STATS = ("nobs", "df_resid", "llf", "aic", "bic", "deviance", "null_deviance", "scale", "pearson_chi2", "ssr", "rsquared",
             "rsquared_adj")

cfg = json.loads(Path(sys.argv[1]).read_text())
kind = cfg["prediction"]["kind"]
keys = [str(k) for k in cfg["prediction"]["keys"]]
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


def summary(x):
    """Aggregates of a per-observation array, like capture.R's summary_of; None if there is none."""
    if x is None or not np.ndim(x):
        return None
    x = np.asarray(x, dtype=float).ravel()
    if not x.size:
        return None
    return {"n": int(x.size), "sum": float(x.sum()), "mean": float(x.mean()), "min": float(x.min()),
            "max": float(x.max()), "n_zero": int((x == 0).sum())}


def design(result) -> dict:
    """What a statsmodels fit consumed: design-matrix column statistics, prior weights and offset. Never a row."""
    model = getattr(result, "model", None)
    exog, names = getattr(model, "exog", None), getattr(model, "exog_names", None)
    columns = None
    if exog is not None and names is not None and np.ndim(exog) == 2:
        exog = np.asarray(exog, dtype=float)
        columns = [{"name": str(n), "mean": float(exog[:, j].mean()),
                    "sd": float(exog[:, j].std(ddof=1)) if len(exog) > 1 else None,
                    "min": float(exog[:, j].min()), "max": float(exog[:, j].max())} for j, n in enumerate(names)]
    if hasattr(model, "freq_weights") or hasattr(model, "var_weights"):     # GLM: model.weights are IRLS weights
        w = np.ones(int(getattr(result, "nobs", 0) or 0))
        for attr in ("freq_weights", "var_weights"):
            if np.ndim(getattr(model, attr, None)):
                w = w * np.asarray(getattr(model, attr), dtype=float)
    else:
        w = getattr(model, "weights", None)
    weights = w if np.ndim(w) and not np.all(np.asarray(w) == 1) else None
    offset = None
    for attr in ("offset", "exposure"):            # statsmodels keeps the log of an exposure, like R's offset
        o = getattr(model, attr, None)
        if o is not None and np.ndim(o):
            offset = np.asarray(o, dtype=float) + (0 if offset is None else offset)
    nobs = getattr(result, "nobs", None)
    return {"n_fit": None if nobs is None else int(nobs), "columns": columns, "weights": summary(weights),
            "offset": summary(offset)}


def converged(result):
    for v in (getattr(result, "converged", None), (getattr(result, "mle_retvals", None) or {}).get("converged")):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
    return None


def coefficients(candidate, result) -> list[tuple[str, float, float]]:
    if hasattr(candidate, "coefficients"):
        table = candidate.coefficients(result)
        if not isinstance(table, pd.DataFrame) or not {"estimate", "std_error"} <= set(table.columns):
            raise TypeError("coefficients() must return a DataFrame with columns estimate and std_error")
        return [(str(t), float(e), float(s)) for t, e, s in zip(table.index, table["estimate"], table["std_error"])]
    params, bse = getattr(result, "params", None), getattr(result, "bse", None)
    if params is None:
        return []
    terms = list(params.index) if isinstance(params, pd.Series) else list(result.model.exog_names)
    params = np.asarray(params, dtype=float).ravel()
    bse = np.full(len(params), np.nan) if bse is None else np.asarray(bse, dtype=float).ravel()
    if not (len(terms) == len(params) == len(bse)):
        raise ValueError("the results' params and bse do not have one value per coefficient")
    return list(zip(map(str, terms), params, bse))


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
    data = cfg.get("data", {})
    readable = {Path(p).resolve() for p in [cfg["dataset"], *data.values()]}

    def dataset_reader(reader):
        def read(filepath_or_buffer, *args, **kwargs):
            if not isinstance(filepath_or_buffer, (str, Path)) or Path(filepath_or_buffer).resolve() not in readable:
                raise ValueError("only the project's data files may be read")
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
    frames = candidate.prepare(cfg["dataset"], {k: v.copy() for k, v in sets.items()}, *([dict(data)] if data else []))
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
        if kind != "class_probability" and not (pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y)):
            raise TypeError(f"the response column '{cfg['response']}' must be numeric, not {y.dtype}")

    stage = "fit()"
    t0 = time.monotonic()
    result = candidate.fit(frames["train"])
    runtime = time.monotonic() - t0
    stage = "reading the fitted results"
    write_csv("coefficients.csv", ["term", "estimate", "std_error"],
              ((t, number(e), number(se)) for t, e, se in coefficients(candidate, result)))
    (out / "fit.json").write_text(json.dumps({k: stat(result, k) for k in FIT_STATS}
                                             | {"converged": converged(result)}, indent=1))
    (out / "design.json").write_text(json.dumps(design(result), indent=1))

    rows = []
    for name, frame in frames.items():
        stage = f"predict() on the {name} set"
        p = candidate.predict(result, frame)
        stage = f"checking predict()'s output on the {name} set"
        if keys:
            if not isinstance(p, pd.DataFrame) or [str(c) for c in p.columns] != keys:
                raise ValueError(f"predict() on the {name} set must return a DataFrame with the columns {keys}")
            p = p.set_axis(keys, axis=1)
        elif isinstance(p, pd.DataFrame):
            raise ValueError(f"predict() on the {name} set must return one value per row, not a DataFrame")
        elif not isinstance(p, pd.Series):
            p = pd.Series(np.asarray(p, dtype=float).ravel(), index=frame.index) if np.ndim(p) else None
            if p is None or len(p) != len(frame):
                raise ValueError(f"predict() on the {name} set must return one value per row")
        if not (p.index.is_unique and len(p) == len(frame) and p.index.isin(frame.index).all()):
            raise ValueError(f"predict() on the {name} set must return one value per row, indexed like the frame")
        p = p.reindex(frame.index)
        y = frame[cfg["response"]]
        for key in keys or [""]:
            values = (p[key] if keys else p).to_numpy(dtype=float, na_value=np.nan)
            observed = (np.where(y.isna(), np.nan, (y.astype(str) == key).astype(float)) if kind == "class_probability"
                        else y.to_numpy(dtype=float, na_value=np.nan))
            rows += [(int(r), name, key, number(a), number(b)) for r, a, b in zip(frame.index, observed, values)]
    write_csv("predictions.csv", ["row", "set", "key", "y", "prediction"], rows)
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
