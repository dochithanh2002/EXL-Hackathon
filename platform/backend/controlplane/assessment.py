"""Deterministic summaries for the financial model assessment and the Diagnostician's design comparison.

`assess` answers four questions for one side's predictions (R's reference or the Python candidate): sample
suitability, stability, calibration and decision performance. Each answer has a status:
- assessed: the data and the spec's criterion are there, and `meets` says whether the criterion holds;
- partially assessed: the summaries are computed but a criterion, threshold or independent set is missing;
- not assessed: the data the question needs is missing.
`needs` says what is missing. Agents interpret these summaries; they never compute them. Aggregates only: no row
or row-level prediction appears in the output.

`design_comparison` sets R's design summary (capture.R's design.json) beside the candidate's (the harness's)."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .runner.parity import BINARY, _coefs, _predictions, metric, metric_weighting, units_by_set

ASSESSED, PARTIAL, NOT_ASSESSED = "assessed", "partially assessed", "not assessed"
QUESTIONS = ("sample_suitability", "stability", "calibration", "decision_performance")
BINS = 10
PSI_FLOOR = 1e-4


def _r(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(f"{x:.6g}")


def _answer(results: dict, needs: list[str], meets: bool | None, criterion: str | None,
            data_missing: bool = False) -> dict:
    status = NOT_ASSESSED if data_missing else PARTIAL if needs else ASSESSED
    return {"status": status, "needs": needs, "criterion": criterion,
            "meets": meets if status == ASSESSED else None, "results": results}


def _quantile_edges(values: list[float], n: int) -> list[float]:
    """Inner edges of up to n equal-count bins; tied values share a bin."""
    s = sorted(values)
    if not s:
        return []
    return sorted({s[min(len(s) - 1, int(len(s) * i / n))] for i in range(1, n)})


def _bin(x: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def psi(expected: list[float], actual: list[float], edges: list[float]) -> float | None:
    """Population stability index of `actual` against `expected`, on bins with the given inner edges."""
    if not expected or not actual:
        return None
    k = len(edges) + 1
    e, a = [0] * k, [0] * k
    for x in expected:
        e[_bin(x, edges)] += 1
    for x in actual:
        a[_bin(x, edges)] += 1
    total = 0.0
    for ei, ai in zip(e, a):
        pe, pa = max(ei / len(expected), PSI_FLOOR), max(ai / len(actual), PSI_FLOOR)
        total += (pa - pe) * math.log(pa / pe)
    return total


def _periods(split: Path) -> dict[tuple[str, int], str]:
    with open(split, newline="") as f:
        return {(r["set"], int(r["row"])): r["period"] for r in csv.DictReader(f) if r.get("period") not in (None, "")}


def assess(predictions: Path, split: Path, spec: dict, n_params: int, n_fit: int, n_train: int,
           weights_from: Path | None = None) -> dict:
    """The four questions for one side. `weights_from` is R's predictions.csv, whose prior weights both sides use."""
    preds = _predictions(predictions)
    weights = None if weights_from is None else {k: w for k, (_, _, w) in _predictions(weights_from).items()}
    units = units_by_set(preds, weights)
    weighted, _ = metric_weighting(spec)
    a, p = spec["assessment"], spec["prediction"]
    roles = {e["set"]: e["role"] for e in a["evaluation_sets"]}
    crit = a["criteria"]
    binary = spec["response_type"] in BINARY and p["kind"] in ("scalar", "time_indexed")
    wt = (lambda w: w) if weighted else (lambda w: 1.0)
    totals = {s: len([k for k in preds if k[0] == s]) for s in roles}
    if not any(units.get(s) for s in roles):
        missing = {"status": NOT_ASSESSED, "needs": ["valid predictions on at least one evaluation set"],
                   "criterion": None, "meets": None, "results": {}}
        return {q: dict(missing) for q in QUESTIONS}
    dev = next((s for s, r in roles.items() if r == "development"), "train")
    independent = [s for s, r in roles.items() if r != "development" and units.get(s)]

    # ---- sample suitability
    per_set = {}
    for s in roles:
        u = units.get(s, [])
        rows = {row: w for row, *_, w in u}
        entry = {"role": roles[s], "units": totals[s], "valid": len(u), "rows": len(rows),
                 "weight_sum": _r(sum(wt(w) for w in rows.values()))}
        if binary:
            events = sum(wt(w) * y for _, _, y, _, w in u)
            total = sum(wt(w) for *_, w in u)
            entry |= {"events": _r(events), "non_events": _r(total - events),
                      "event_rate": _r(events / total) if total else None}
        elif p["kind"] == "class_probability":
            counts = defaultdict(float)
            for _, key, y, _, w in u:
                counts[key] += wt(w) * y
            entry["class_counts"] = {k: _r(v) for k, v in counts.items()}
        else:
            total = sum(wt(w) for *_, w in u)
            entry["mean_observed"] = _r(sum(wt(w) * y for _, _, y, _, w in u) / total) if total else None
        per_set[s] = entry
    d = per_set.get(dev, {})
    if binary:
        limiting = min(d.get("events") or 0, d.get("non_events") or 0)
        per_param, basis = (limiting / n_params if n_params else None), "the smaller of events and non-events"
    elif p["kind"] == "class_probability":
        limiting = min((d.get("class_counts") or {"": 0}).values())
        per_param, basis = (limiting / n_params if n_params else None), "the rarest class"
    else:
        per_param, basis = (n_fit / n_params if n_params else None), "fitted observations"
    results = {"sets": per_set, "development_set": dev, "parameters": n_params, "fitted_rows": n_fit,
               "train_rows": n_train, "excluded_from_fit": n_train - n_fit,
               "per_parameter": _r(per_param), "per_parameter_basis": basis}
    needs = [] if crit.get("min_events_per_parameter") else ["a minimum number of events (or observations) per "
                                                              "parameter"]
    meets = None if per_param is None or not crit.get("min_events_per_parameter") else \
        per_param >= crit["min_events_per_parameter"]
    suitability = _answer(results, needs, meets, None if not crit.get("min_events_per_parameter") else
                          f"per_parameter >= {crit['min_events_per_parameter']:g}")

    # ---- stability: score distribution against the development set, by set and by period
    def scores(u, key=None):
        return [pr for _, k, _, pr, _ in u if key is None or k == key]
    pkeys = p["keys"] or [None]
    by_set = {}
    for s in independent:
        values = []
        for k in pkeys:
            ref = scores(units.get(dev, []), k)
            values.append(psi(ref, scores(units[s], k), _quantile_edges(ref, BINS)) if ref else None)
        by_set[s] = _r(max((v for v in values if v is not None), default=None))
    by_period = {}
    periods = _periods(split) if a.get("period_column") else {}
    if periods:
        grouped: dict[str, list] = defaultdict(list)
        for s in roles:
            for unit in units.get(s, []):
                if (per := periods.get((s, unit[0]))) is not None:
                    grouped[per].append(unit)
        order = sorted(grouped)
        first = order[0] if order else None
        for per in order:
            u = grouped[per]
            values = []
            for k in pkeys:
                ref = scores(grouped[first], k)
                values.append(psi(ref, scores(u, k), _quantile_edges(ref, BINS)) if ref else None)
            total = sum(wt(w) for *_, w in u)
            by_period[per] = {"units": len(u), "psi_vs_first": _r(max((v for v in values if v is not None),
                                                                     default=None)),
                              "mean_prediction": _r(sum(wt(w) * pr for *_, pr, w in u) / total) if total else None,
                              "mean_observed": _r(sum(wt(w) * y for _, _, y, _, w in u) / total) if total else None}
    if not by_set and len(by_period) < 2:
        stability = _answer({}, ["a holdout or out-of-time evaluation set, or a period column with at least two "
                                 "periods"], None, None, data_missing=True)
    else:
        all_psi = [v for v in by_set.values() if v is not None] + [
            v["psi_vs_first"] for v in by_period.values() if v["psi_vs_first"] is not None]
        worst = max(all_psi, default=None)
        needs = [] if crit.get("max_psi") else ["a PSI threshold"]
        stability = _answer({"psi_vs_development": by_set, "by_period": by_period, "max_psi": _r(worst)}, needs,
                            None if worst is None or not crit.get("max_psi") else worst <= crit["max_psi"],
                            None if not crit.get("max_psi") else f"max_psi <= {crit['max_psi']:g}")

    # ---- calibration: observed against predicted, overall and by prediction decile
    cal = {}
    calibration_missing = []
    for s in roles:
        u = units.get(s, [])
        if not u:
            calibration_missing.append(f"valid calibration observations on {s}")
            continue
        if p["kind"] == "quantile":
            cover = {}
            for k in p["keys"]:
                kk = [(y, q, wt(w)) for _, key, y, q, w in u if key == k]
                tw = sum(w for *_, w in kk)
                if tw <= 0:
                    calibration_missing.append(f"valid calibration observations for {s}, quantile {k}")
                cover[k] = {"nominal": float(k), "observed": _r(sum(w for y, q, w in kk if y <= q) / tw) if tw else None}
            dev_max = max((abs(c["observed"] - c["nominal"]) for c in cover.values() if c["observed"] is not None),
                          default=None)
            cal[s] = {"coverage": cover, "max_deviation": _r(dev_max)}
            continue
        groups = {k: [(y, pr, wt(w)) for _, key, y, pr, w in u if key == k] for k in (p["keys"] or [""])}
        entry = {}
        for k, g in groups.items():
            if not g or sum(w for _, _, w in g) <= 0:
                calibration_missing.append(f"valid calibration observations for {s}, output {k or 'value'}")
                entry[k or "value"] = {"status": NOT_ASSESSED, "observed": None, "expected": None,
                                       "observed_expected": None, "max_bin_gap": None, "bins": []}
                continue
            obs, exp = sum(w * y for y, _, w in g), sum(w * pr for _, pr, w in g)
            edges = _quantile_edges([pr for _, pr, _ in g], BINS)
            bins: dict[int, list] = defaultdict(list)
            for y, pr, w in g:
                bins[_bin(pr, edges)].append((y, pr, w))
            table = []
            for i in sorted(bins):
                b, tw = bins[i], sum(w for *_, w in bins[i])
                if tw:
                    table.append({"bin": i + 1, "units": len(b), "mean_predicted": _r(sum(w * pr for _, pr, w in b) / tw),
                                  "mean_observed": _r(sum(w * y for y, _, w in b) / tw)})
            gap = max((abs(t["mean_observed"] - t["mean_predicted"]) for t in table
                       if None not in (t["mean_observed"], t["mean_predicted"])), default=None)
            oe = obs / exp if exp > 0 else None
            entry[k or "value"] = {"observed": _r(obs), "expected": _r(exp), "observed_expected": _r(oe),
                                   "max_bin_gap": _r(gap), "bins": table}
        cal[s] = entry
    if not cal:
        calibration = _answer({}, ["valid predictions"], None, None, data_missing=True)
    else:
        needs = list(calibration_missing)
        if not independent:
            needs.append("an independent evaluation set: calibration is measured on the development set only")
        if not crit.get("max_calibration_deviation"):
            needs.append("a calibration tolerance (largest acceptable |observed / expected - 1|)")
        judged = [s for s in independent] or [dev]
        deviations = []
        for s in judged:
            for v in (cal.get(s) or {}).values():
                if isinstance(v, dict) and v.get("observed_expected") is not None:
                    deviations.append(abs(v["observed_expected"] - 1))
            if p["kind"] == "quantile" and (cal.get(s) or {}).get("max_deviation") is not None:
                deviations.append(cal[s]["max_deviation"])
        worst = max(deviations, default=None)
        if worst is None:
            needs.append("a computable calibration statistic on the evaluation sets")
        limit = crit.get("max_calibration_deviation")
        calibration = _answer({"sets": cal, "judged_on": judged, "max_deviation": _r(worst)}, needs,
                              None if worst is None or not limit else worst <= limit,
                              None if not limit else f"max_deviation <= {limit:g}")

    # ---- decision performance: discrimination always, decisions only at the spec's threshold
    if not binary:
        decision = _answer({}, ["a binary event with a predicted probability; decision performance is not defined "
                                f"for a {spec['response_type']} response with {p['kind']} outputs"], None, None,
                           data_missing=True)
    else:
        disc, dec = {}, {}
        for s in roles:
            u = units.get(s, [])
            disc[s] = {m: _r(metric(m, u, spec.get("family"), "scalar", weighted)) for m in ("auc", "gini", "ks")}
            t = (a.get("decision") or {}).get("threshold")
            if t is not None and u:
                total = sum(wt(w) for *_, w in u)
                events = sum(wt(w) * y for _, _, y, _, w in u)
                flagged = [(y, wt(w)) for _, _, y, pr, w in u if pr >= t]
                fw = sum(w for _, w in flagged)
                fe = sum(w * y for y, w in flagged)
                rest_w, rest_e = total - fw, events - fe
                dec[s] = {"flagged_share": _r(fw / total) if total else None,
                          "event_rate_flagged": _r(fe / fw) if fw else None,
                          "event_rate_not_flagged": _r(rest_e / rest_w) if rest_w else None,
                          "events_captured": _r(fe / events) if events else None,
                          "non_events_flagged": _r((fw - fe) / (total - events)) if total - events else None}
        needs = [] if a.get("decision") else ["the decision threshold and the action it triggers"]
        judged = independent or [dev]
        if not independent:
            needs.append("an independent evaluation set: decision performance is measured on development data only")
        for s in roles:
            if roles[s] != "development" and not units.get(s):
                needs.append(f"valid observations on the independent {s} set")
        aucs = [disc[s]["auc"] for s in judged if disc.get(s, {}).get("auc") is not None]
        if len(aucs) != len(judged):
            needs.append("a computable AUC on every evaluation set (both events and non-events)")
        limit = crit.get("min_auc")
        if not limit:
            needs.append("a minimum AUC")
        recall_limit = crit.get("min_events_captured")
        if recall_limit is None:
            needs.append("a minimum event recall at the decision threshold")
        recalls = [dec[s]["events_captured"] for s in judged if dec.get(s, {}).get("events_captured") is not None]
        if len(recalls) != len(judged):
            needs.append("computable event recall on every evaluation set")
        criterion = " and ".join(([f"auc >= {limit:g}"] if limit else []) +
                                 ([f"events_captured >= {recall_limit:g}"] if recall_limit is not None else []))
        decision = _answer({"discrimination": disc, "at_threshold": dec, "judged_on": judged,
                            "threshold": (a.get("decision") or {}).get("threshold"),
                            "action": (a.get("decision") or {}).get("action")}, needs,
                           None if needs else min(aucs) >= limit and min(recalls) >= recall_limit,
                           criterion or None)
    return {"sample_suitability": suitability, "stability": stability, "calibration": calibration,
            "decision_performance": decision}


def assess_both(baseline: Path, candidate: Path, spec: dict) -> dict:
    """The assessment of R's reference and of the Python candidate, with the same prior weights."""
    meta = json.loads((baseline / "meta.json").read_text())
    design = json.loads((baseline / "design.json").read_text())
    n_params = sum(1 for e, _ in _coefs(baseline / "coefficients.csv").values() if math.isfinite(e))
    args = dict(spec=spec, n_params=n_params, n_fit=design.get("n_fit") or meta.get("n_fit") or 0,
                n_train=(meta.get("n") or {}).get("train") or 0)
    out = {"r": assess(baseline / "predictions.csv", baseline / "split.csv", **args)}
    if (candidate / "predictions.csv").is_file():
        out["python"] = assess(candidate / "predictions.csv", baseline / "split.csv",
                               weights_from=baseline / "predictions.csv", **args)
    return out


# ---------------------------------------------------------------- design comparison
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
