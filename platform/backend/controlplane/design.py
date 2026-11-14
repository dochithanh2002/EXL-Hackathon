"""The Diagnostician's design comparison: R's design summary (capture.R's design.json) beside the candidate's (the
harness's), with their differences. Aggregates only."""
from __future__ import annotations

import json
from pathlib import Path

from .statistics import rounded as _r


def _summary_diff(r: dict | None, py: dict | None) -> dict:
    out = {"r": r, "python": py}
    if r and py:
        out["differences"] = {k: _r(py[k] - r[k]) for k in ("n", "sum", "mean", "min", "max", "n_zero")
                              if isinstance(r.get(k), (int, float)) and isinstance(py.get(k), (int, float))}
    return out


def design_comparison(baseline: Path, candidate: Path, limit: int = 60) -> dict:
    """Effective fitting sample, design-matrix columns and their statistics, reference levels, offsets and weights,
    in R and in Python, with their differences. Python's reference level of a factor is the level of R's xlevels
    that has no design column of its own."""
    r = json.loads((baseline / "design.json").read_text()) if (baseline / "design.json").is_file() else {}
    meta = json.loads((baseline / "meta.json").read_text())
    path = candidate / "design.json"
    py = json.loads(path.read_text()) if path.is_file() else {}
    rcols = {c["name"]: c for c in r.get("columns") or []}
    pcols = {c["name"]: c for c in py.get("columns") or []}
    columns = []
    for name, rc in rcols.items():
        pc = pcols.get(name)
        row = {"name": name, "r": {k: _r(rc.get(k)) for k in ("mean", "sd", "min", "max")}}
        if pc:
            row["python"] = {k: _r(pc.get(k)) for k in ("mean", "sd", "min", "max")}
            row["mean_abs_diff"] = _r(abs((pc.get("mean") or 0) - (rc.get("mean") or 0)))
            row["sd_abs_diff"] = _r(abs((pc.get("sd") or 0) - (rc.get("sd") or 0)))
        columns.append(row)
    columns.sort(key=lambda c: -(c.get("mean_abs_diff") or 0) - (c.get("sd_abs_diff") or 0))
    references = {}
    for factor, levels in (meta.get("xlevels") or {}).items():
        levels = levels if isinstance(levels, list) else [levels]
        without = [lv for lv in levels if f"{factor}{lv}" not in pcols]
        references[factor] = {"r": (r.get("reference_levels") or {}).get(factor),
                              "python": without[0] if len(without) == 1 else without if pcols else None}
    return {"fitted_rows": {"r": r.get("n_fit"), "python": py.get("n_fit")},
            "columns_only_in_r": [c for c in rcols if pcols and c not in pcols],
            "columns_only_in_python": [c for c in pcols if c not in rcols],
            "python_reports_design": bool(pcols),
            "columns": columns[:limit], "columns_shown": min(limit, len(columns)), "columns_total": len(columns),
            "reference_levels": references, "contrasts": r.get("contrasts"),
            "weights": _summary_diff(r.get("weights"), py.get("weights")),
            "offset": _summary_diff(r.get("offset"), py.get("offset"))}
