"""Model validation beyond migration parity: nine checks, executed by code on both sides' predictions.

The Analyst plans seven of them and says why (`ModelSpec.validation_plan`); sample suitability and decision
performance run for every model they apply to. The statistics come from `statistics.Inputs`, computed once per side
and set; agents interpret the results.
- baseline comparison: each side's metrics against a constant comparator fitted on the development set;
- calibration: observed against predicted overall, by decile, and as a recalibration slope and intercept;
- subgroup performance: rows, observed, predicted, observed/expected and the primary loss per subgroup of the
  plan's columns (values exported by capture.R);
- stability: PSI by set and period, cross-validated metrics and coefficients of Python refits, and the fit's
  iteration history;
- sensitivity: the change in Python's predictions when one predictor moves, all else fixed;
- assumptions: screens of the model's assumptions (overdispersion, zero inflation, multicollinearity, residual
  distribution, heteroscedasticity, separation);
- fairness: per protected group, rows, observed/expected and, for a binary decision, the favourable-outcome rate
  and the impact ratio between groups (values exported by capture.R, like the subgroups);
- sample suitability: rows, events and events per estimated parameter;
- decision performance: discrimination, and the decisions at the business threshold.

Each check has two independent answers. `status` says how far it ran: executed, partially executed, not executed,
not planned or not applicable, with `needs` naming what prevented full execution. `criterion` says what the spec's
acceptance criterion concluded on the Python candidate: met, not met, no criterion (the spec states none) or not
judged (the statistic could not be computed). `business_fitness` reads only those two answers, and
`missing_evidence` collects every gap in one list. Results are aggregates: no row or row-level value appears. None
of this decides the migration's acceptance, which rests on the parity checks alone."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from .runner.parity import _coefs
from .statistics import (SIDES, Inputs, comparison, evaluated_metrics, primary_metric, quantile_edges, bin_of,
                         read_json, rounded, score, scores)

__all__ = ["Inputs", "evaluate", "evaluated_metrics", "scores", "business_fitness", "missing_evidence",
           "evaluation_view", "tables", "write_tables"]

EXECUTED, PARTIAL, NOT_EXECUTED = "executed", "partially executed", "not executed"
NOT_PLANNED, NOT_APPLICABLE = "not planned", "not applicable"
MET, NOT_MET, NO_CRITERION, NOT_JUDGED = "met", "not met", "no criterion", "not judged"
PLANNED = ("baseline_comparison", "calibration", "subgroup_performance", "stability", "sensitivity", "assumptions",
           "fairness")
CHECKS = (*PLANNED, "sample_suitability", "decision_performance")
TITLES = {"baseline_comparison": "Baseline comparison", "calibration": "Calibration",
          "subgroup_performance": "Subgroup performance", "stability": "Stability", "sensitivity": "Sensitivity",
          "assumptions": "Model assumptions", "fairness": "Fairness", "sample_suitability": "Sample suitability",
          "decision_performance": "Decision performance"}
# check -> statistic -> (the spec's criterion, how the statistic must compare with it)
CRITERIA = {"sample_suitability": {"per_parameter": ("min_events_per_parameter", ">=")},
            "calibration": {"max_deviation": ("max_calibration_deviation", "<=")},
            "stability": {"max_psi": ("max_psi", "<=")},
            "decision_performance": {"auc": ("min_auc", ">="), "events_captured": ("min_events_captured", ">=")},
            "fairness": {"impact_ratio": ("min_impact_ratio", ">="),
                         "max_group_deviation": ("max_group_calibration_deviation", "<=")}}
NO_FAIRNESS_PLAN = "No fairness review was planned: the specification names no protected attributes."
MAX_LEVELS = 12
MIN_AGENT_GROUP_ROWS = 10
QUINTILES = 5

_r = rounded


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _check(status: str, rationale: str, results: dict | None = None, needs: list[str] | None = None,
           notes: list[str] | None = None, criterion: dict | None = None) -> dict:
    return {"status": status, "rationale": rationale, "needs": list(dict.fromkeys(needs or [])),
            "notes": notes or [], "results": results or {}, "criterion": criterion}


def _status(results: bool, needs: list[str]) -> str:
    return NOT_EXECUTED if not results else PARTIAL if needs else EXECUTED


def judge(check: str, spec: dict, values: dict[str, float | None], judged_on: list[str],
          applies: list[str] | None = None) -> dict:
    """The spec's criterion for a check, judged on the Python candidate's statistics. A stated limit that fails is
    not met; one whose statistic is missing is not judged; a limit the spec does not state leaves no criterion.
    `applies` restricts the check's criteria to those defined for this model (all by default)."""
    stated = spec["assessment"]["criteria"]
    defined = {stat: x for stat, x in CRITERIA[check].items() if applies is None or stat in applies}
    limits = {stat: stated[key] for stat, (key, _) in defined.items() if stated.get(key) is not None}
    ops = {stat: op for stat, (_, op) in defined.items()}
    failing = [stat for stat, limit in limits.items() if values.get(stat) is not None
               and not (values[stat] >= limit if ops[stat] == ">=" else values[stat] <= limit)]
    outcome = (NOT_MET if failing else NOT_JUDGED if any(values.get(stat) is None for stat in limits)
               else NO_CRITERION if len(limits) < len(defined) else MET)
    out = {"rule": " and ".join(f"{stat} {ops[stat]} {limit:g}" for stat, limit in limits.items()) or None,
           "limit": limits, "value": {k: _r(v) for k, v in values.items() if k in defined}, "outcome": outcome,
           "judged_on": judged_on}
    if applies is not None:
        out["applies"] = list(defined)
    return out


def unstated(check: str, criterion: dict) -> list[str]:
    """The spec's criteria a check needs but the spec does not state."""
    applies = criterion.get("applies")
    return [key for stat, (key, _) in CRITERIA[check].items()
            if stat not in criterion["limit"] and (applies is None or stat in applies)]


# ---------------------------------------------------------------- baseline comparison
def baseline_comparison(inp: Inputs, rationale: str) -> dict:
    by_model, constants = scores(inp)
    sets = {}
    for s, const in by_model["constant"].items():
        entry = {}
        for name, c in const.items():
            row = {"constant": c}
            for side in inp.sides():
                v = by_model[side].get(s, {}).get(name)
                row[side] = v
                row[f"{side}_vs_constant"] = _r(comparison(name, v, c))
            entry[name] = row
        sets[s] = entry
    notes = []
    if inp.spec.get("offset") or inp.spec["prediction"].get("exposure"):
        notes.append("the constant comparator ignores the offset and exposure: it is a no-information benchmark, "
                     "not an exposure-only model")
    if not sets:
        return _check(NOT_EXECUTED, rationale, needs=["valid R predictions on an evaluation set"])
    return _check(EXECUTED, rationale, {"comparator": "constant fitted on the development set",
                                        "constants": {k or "value": _r(v) for k, v in constants.items()},
                                        "primary_metric": primary_metric(inp.spec), "sets": sets}, notes=notes)


# ---------------------------------------------------------------- calibration
def calibration(inp: Inputs, rationale: str) -> dict:
    p = inp.spec["prediction"]
    sets, needs = {}, []
    for s in inp.roles:
        entry: dict[str, dict] = {}
        for side in inp.sides():
            cal = inp.calibration(side, s)
            for key in inp.keys:
                if (key or "value") in cal:
                    entry.setdefault(key or "value", {})[side] = cal[key or "value"]
                elif inp.side_units(side, s):
                    needs.append(f"valid calibration observations for {s}, output {key or 'value'}")
        if entry:
            sets[s] = {k: entry[k] for k in (key or "value" for key in inp.keys) if k in entry}
        else:
            needs.append(f"valid predictions on {s}")
    if not inp.independent:
        needs.append("an independent evaluation set: calibration is measured on the development set only")
    deviations = []
    for s in inp.judged:
        for d in (sets.get(s) or {}).values():
            py = d.get("python") or {}
            if py.get("observed_expected") is not None:
                deviations.append(abs(py["observed_expected"] - 1))
            elif py.get("coverage") is not None:
                deviations.append(abs(py["coverage"] - py["nominal"]))
    worst = max(deviations, default=None)
    if sets and worst is None:
        needs.append("a computable observed / expected (or coverage) of the Python predictions on "
                     + ", ".join(inp.judged))
    criterion = judge("calibration", inp.spec, {"max_deviation": worst}, inp.judged)
    if not sets:
        return _check(NOT_EXECUTED, rationale, needs=needs, criterion=criterion)
    method = ("coverage of each quantile level" if p["kind"] == "quantile" else
              "logistic recalibration, logit(observed) ~ intercept + slope × logit(predicted)" if inp.probability else
              "least squares of observed on predicted")
    deviation = ("|coverage - nominal level|" if p["kind"] == "quantile" else "|observed / expected - 1|")
    return _check(_status(True, needs), rationale, {"method": method, "deviation": deviation,
                                                    "judged_on": inp.judged, "max_deviation": _r(worst),
                                                    "sets": sets}, needs, criterion=criterion)


# ---------------------------------------------------------------- subgroups
def _segments(baseline: Path) -> dict[str, dict[tuple[str, int], str]]:
    out: dict[str, dict[tuple[str, int], str]] = defaultdict(dict)
    for r in _rows(baseline / "segments.csv"):
        out[r["column"]][(r["set"], int(r["row"]))] = r["value"]
    return out


def _groups(inp: Inputs, values: dict[tuple[str, int], str], kind: str) -> tuple[dict[tuple[str, int], str], list]:
    """Each (set, row)'s group label and the groups in order with their description."""
    labels: dict[tuple[str, int], str] = {}
    if kind == "numeric":
        dev = [float(v) for (s, _), v in values.items() if s == inp.dev and v != ""]
        edges = quantile_edges(dev, QUINTILES)
        names = [f"Q{i + 1}" for i in range(len(edges) + 1)]
        bounds = [-math.inf, *edges, math.inf]
        groups = [{"group": n, "lower": _r(bounds[i]) if i else None,
                   "upper": _r(bounds[i + 1]) if i < len(edges) else None} for i, n in enumerate(names)]
        for k, v in values.items():
            labels[k] = "(missing)" if v == "" else names[bin_of(float(v), edges)]
        if any(v == "" for v in values.values()):
            groups.append({"group": "(missing)"})
        return labels, groups
    counts: dict[str, int] = defaultdict(int)
    for (s, _), v in values.items():
        if s == inp.dev:
            counts[v or "(missing)"] += 1
    ranked = sorted(counts, key=lambda v: (-counts[v], v))
    top = set(ranked[:MAX_LEVELS])
    for k, v in values.items():
        v = v or "(missing)"
        labels[k] = v if v in top else "Other"
    groups = [{"group": v} for v in ranked[:MAX_LEVELS]] + ([{"group": "Other"}] if len(ranked) > MAX_LEVELS else [])
    return labels, groups


def _group_stats(inp: Inputs, units: list) -> dict:
    primary = {inp.metrics[0]: _r(score(inp.metrics[0], units, inp.spec, inp.weighted))}
    if inp.keys != [""]:                            # class probabilities or quantiles: the loss is the summary
        return primary
    tw = sum(inp.wt(w) for *_, w in units)
    obs = sum(inp.wt(w) * y for _, _, y, _, w in units)
    exp = sum(inp.wt(w) * p for _, _, _, p, w in units)
    out = {"mean_observed": _r(obs / tw) if tw else None, "mean_predicted": _r(exp / tw) if tw else None,
           "observed_expected": _r(obs / exp) if exp > 0 else None, **primary}
    if inp.binary:
        out["auc"] = _r(score("auc", units, inp.spec, inp.weighted))
    return out


def subgroup_performance(inp: Inputs, plan: dict) -> dict:
    segments, types = _segments(inp.baseline), inp.meta.get("segments") or {}
    columns, needs = {}, []
    for col in plan["columns"]:
        kind = types.get(col, "missing")
        if kind == "missing" or col not in segments:
            columns[col] = {"status": NOT_EXECUTED, "reason": "the column is not in R's set frames"}
            needs.append(f"subgroup column {col} in R's set frames")
            continue
        labels, groups = _groups(inp, segments[col], kind)
        buckets: dict[tuple[str, str, str | None], list] = defaultdict(list)
        for side in inp.sides():
            for s in inp.roles:
                for x in inp.side_units(side, s):
                    buckets[(side, s, labels.get((s, x[0])))].append(x)
        for g in groups:
            g["sets"] = {}
            for s in inp.roles:
                rows = len({x[0] for x in buckets.get(("r", s, g["group"]), [])})
                if rows:
                    entry = {"rows": rows, "small": rows < plan["min_rows"], "scored_rows": {}}
                    for side in inp.sides():
                        if units := buckets.get((side, s, g["group"])):
                            entry[side] = _group_stats(inp, units)
                            # Count contributing observations per output and side, not rows assigned by R.
                            by_key: dict[str, set[int]] = defaultdict(set)
                            for row, key, _, _, weight in units:
                                if weight > 0:
                                    by_key[key].add(row)
                            entry["scored_rows"][side] = min(len(by_key[k]) for k in inp.keys)
                    g["sets"][s] = entry
        columns[col] = {"status": EXECUTED, "type": kind, "groups": [g for g in groups if g["sets"]]}
        small = [g["group"] for g in groups if any(v["small"] for v in g["sets"].values())]
        if small:
            columns[col]["small_groups"] = small
    executed = any(c["status"] == EXECUTED for c in columns.values())
    return _check(_status(executed, needs), plan["rationale"], {
        "metric": inp.metrics[0], "min_rows": plan["min_rows"], "judged_on": inp.judged, "columns": columns}, needs,
        notes=["numeric columns are cut at quintiles of the development set; categorical columns keep their "
               f"{MAX_LEVELS} largest levels and fold the rest into Other"])


# ---------------------------------------------------------------- fairness
def _favourable_share(inp: Inputs, units: list, threshold: float, favourable: str) -> float | None:
    total = sum(inp.wt(w) for *_, w in units)
    if not total:
        return None
    below = sum(inp.wt(w) for _, _, _, p, w in units if p < threshold)
    return (below if favourable == "below_threshold" else total - below) / total


def fairness(inp: Inputs, plan: dict) -> dict:
    """Per protected group on every evaluation set: rows, observed/expected and, for a binary decision, the share of
    favourable outcomes at the threshold. Judged on the Python candidate over groups of at least `min_rows` rows on
    the judged sets: the impact ratio (lowest favourable share over the highest, per column and set) and the largest
    |observed / expected - 1| of a group."""
    segments, types = _segments(inp.baseline), inp.meta.get("segments") or {}
    decision = inp.spec["assessment"].get("decision") or {}
    threshold, favourable = decision.get("threshold"), plan.get("favourable_outcome")
    decides = inp.binary and threshold is not None and favourable is not None
    applies = ["max_group_deviation", *(["impact_ratio"] if inp.binary else [])]
    columns, needs, ratios, deviations = {}, [], [], []
    if not plan["columns"]:
        needs.append("the protected attributes to compare")
    if inp.binary and not decides:
        needs.append("the decision threshold and which side of it is favourable, for the impact ratio")
    for col in plan["columns"]:
        kind = types.get(col, "missing")
        if kind == "missing" or col not in segments:
            columns[col] = {"status": NOT_EXECUTED, "reason": "the column is not in R's set frames"}
            needs.append(f"protected attribute {col} in R's set frames")
            continue
        labels, groups = _groups(inp, segments[col], kind)
        buckets: dict[tuple[str, str, str | None], list] = defaultdict(list)
        for side in inp.sides():
            for s in inp.roles:
                for x in inp.side_units(side, s):
                    buckets[(side, s, labels.get((s, x[0])))].append(x)
        judged_shares: dict[str, list[float]] = defaultdict(list)
        for g in groups:
            g["sets"] = {}
            for s in inp.roles:
                rows = len({x[0] for x in buckets.get(("r", s, g["group"]), [])})
                if not rows:
                    continue
                entry = {"rows": rows, "small": rows < plan["min_rows"]}
                for side in inp.sides():
                    if units := buckets.get((side, s, g["group"])):
                        stats = _group_stats(inp, units)
                        if decides:
                            stats["favourable_share"] = _r(_favourable_share(inp, units, threshold, favourable))
                        entry[side] = stats
                g["sets"][s] = entry
                py = entry.get("python") or {}
                if s in inp.judged and not entry["small"] and py:
                    if py.get("observed_expected") is not None:
                        deviations.append(abs(py["observed_expected"] - 1))
                    if py.get("favourable_share") is not None:
                        judged_shares[s].append(py["favourable_share"])
        impact = {s: _r(min(v) / max(v)) if len(v) > 1 and max(v) > 0 else None for s, v in judged_shares.items()}
        ratios += [x for x in impact.values() if x is not None]
        columns[col] = {"status": EXECUTED, "type": kind, "groups": [g for g in groups if g["sets"]],
                        "impact_ratio": impact}
        if small := [g["group"] for g in groups if any(v["small"] for v in g["sets"].values())]:
            columns[col]["small_groups"] = small
    if columns and not deviations and not ratios:
        needs.append(f"protected groups of at least {plan['min_rows']} rows with valid Python predictions on "
                     + ", ".join(inp.judged))
    values = {"impact_ratio": min(ratios) if ratios else None,
              "max_group_deviation": max(deviations) if deviations else None}
    criterion = judge("fairness", inp.spec, values, inp.judged, applies)
    executed = any(c["status"] == EXECUTED for c in columns.values())
    notes = ["groups below min_rows are reported but not judged; numeric columns are cut at quintiles of the "
             f"development set, categorical columns keep their {MAX_LEVELS} largest levels"]
    if decides:
        notes.append(f"favourable outcome: prediction {'below' if favourable == 'below_threshold' else 'at or above'} "
                     f"{threshold:g}")
    return _check(_status(executed, needs), plan["rationale"], {
        "min_rows": plan["min_rows"], "judged_on": inp.judged, "threshold": threshold if decides else None,
        "favourable_outcome": favourable if decides else None, "impact_ratio": _r(values["impact_ratio"]),
        "max_group_deviation": _r(values["max_group_deviation"]), "columns": columns}, needs, notes, criterion)


# ---------------------------------------------------------------- stability
def _failed(part: dict) -> str:
    if part.get("status") == "failed":
        return f"failed in {part.get('stage')} ({part.get('type')})"
    return part.get("reason") or "not run"


def _cv(inp: Inputs, report: dict) -> tuple[dict | None, list[str]]:
    cv = report.get("cross_validation") or {}
    if cv.get("status") != EXECUTED:
        return None, [f"cross-validation refits ({_failed(cv)})"]
    folds: dict[int, list] = defaultdict(list)
    for r in _rows(inp.diagnostics / "cv_predictions.csv"):
        y, p = float(r["y"]) if r["y"] else math.nan, float(r["prediction"]) if r["prediction"] else math.nan
        w = inp.weights.get(("train", int(r["row"]), r["key"]), math.nan)
        if math.isfinite(y) and math.isfinite(p) and math.isfinite(w):
            folds[int(r["fold"])].append((int(r["row"]), r["key"], y, p, w))
    names = [m for m in inp.metrics if m != "deviance"]
    per_fold = []
    for k in sorted(folds):
        fit = next((f for f in cv["fits"] if f["fold"] == k), {})
        per_fold.append({"fold": k, "fit_rows": fit.get("fit_rows"), "held_out_rows": fit.get("held_out_rows"),
                         "converged": fit.get("converged"),
                         **{m: _r(score(m, folds[k], inp.spec, inp.weighted)) for m in names}})
    summary = {}
    for m in names:
        v = [f[m] for f in per_fold if f[m] is not None]
        if v:
            mean = sum(v) / len(v)
            sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else None
            summary[m] = {"mean": _r(mean), "sd": _r(sd), "min": _r(min(v)), "max": _r(max(v)),
                          "in_sample": inp.metric("python", "train", m)}
    full_coefs = _coefs(inp.candidate / "coefficients.csv")
    by_term: dict[str, list[float]] = defaultdict(list)
    for r in _rows(inp.diagnostics / "cv_coefficients.csv"):
        if r["estimate"]:
            by_term[r["term"]].append(float(r["estimate"]))
    terms = []
    for t, v in by_term.items():
        est = full_coefs.get(t, (math.nan, math.nan))[0]
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        terms.append({"term": t, "estimate": _r(est), "fold_sd": _r(sd),
                      "relative_sd": _r(sd / abs(est)) if math.isfinite(est) and est else None,
                      "sign_changes": sum(1 for x in v if math.isfinite(est) and x * est < 0)})
    terms.sort(key=lambda t: -(t["relative_sd"] or 0))
    unconverged = [f["fold"] for f in per_fold if f["converged"] is False]
    needs = [f"converged refits on folds {unconverged}"] if unconverged else []
    return {"folds": per_fold, "summary": summary, "coefficients": terms,
            "terms_with_sign_changes": sum(1 for t in terms if t["sign_changes"])}, needs


def stability(inp: Inputs, plan: dict, report: dict | None, planned: bool = True) -> dict:
    """PSI, the training curve and, when planned, the cross-validation refits."""
    results, needs = {}, []
    if psi := {side: x for side in inp.sides() if (x := inp.stability(side))}:
        results["psi"] = psi
    else:
        needs.append("a holdout or out-of-time evaluation set, or a period column with at least two periods")
    history = read_json(inp.candidate / "fit_history.json") if inp.candidate else {}
    if history:
        name = "deviance" if "deviance" in history else next(iter(history))
        results["training_curve"] = {"metric": name, "values": [_r(v) for v in history[name]],
                                     "iterations": len(history[name]), "r_iterations": inp.meta.get("iterations")}
    if planned and report is None:
        needs.append("cross-validation refits (the Python diagnostics did not run)")
    elif planned:
        cv, more = _cv(inp, report)
        needs += more
        if cv:
            results["cross_validation"] = cv
    py = psi.get("python") or {}
    judged_on = list(py.get("vs_development", {})) + (["periods"] if len(py.get("by_period", {})) > 1 else [])
    criterion = judge("stability", inp.spec, {"max_psi": py.get("max_psi")}, judged_on)
    notes = ["folds are assigned by a hash of the data-row number; each refit reuses prepare()'s frames, so "
             "preprocessing fitted on all rows is not refitted per fold"] if "cross_validation" in results else []
    return _check(_status(bool(results), needs), plan["rationale"], results, needs, notes, criterion)


# ---------------------------------------------------------------- sensitivity
def sensitivity(plan: dict, report: dict | None) -> dict:
    if report is None:
        return _check(NOT_EXECUTED, plan["rationale"], needs=["sensitivity predictions (the Python diagnostics "
                                                              "did not run)"])
    part = report.get("sensitivity") or {}
    if part.get("status") != EXECUTED:
        return _check(NOT_EXECUTED, plan["rationale"], needs=[f"sensitivity predictions ({_failed(part)})"])
    columns = part["columns"]
    needs = [f"sensitivity of {c} ({_failed(v)})" for c, v in columns.items() if v["status"] != EXECUTED]
    executed = any(v["status"] == EXECUTED for v in columns.values())
    return _check(_status(executed, needs), plan["rationale"], {"shift_sd": plan["shift_sd"], "columns": columns},
                  needs, ["each column moves by shift_sd training standard deviations in the frames the Python "
                          "model reads; changes are unweighted means over rows"])


# ---------------------------------------------------------------- assumptions
def _moments(residuals: list[float]) -> dict:
    n = len(residuals)
    if n < 3:
        return {}
    mean = sum(residuals) / n
    m2 = sum((x - mean) ** 2 for x in residuals) / n
    if m2 <= 0:
        return {}
    skew = sum((x - mean) ** 3 for x in residuals) / n / m2 ** 1.5
    kurt = sum((x - mean) ** 4 for x in residuals) / n / m2 ** 2 - 3
    return {"rows": n, "skewness": _r(skew), "excess_kurtosis": _r(kurt),
            "jarque_bera": _r(n / 6 * (skew ** 2 + kurt ** 2 / 4))}


def _assumption(inp: Inputs, name: str, report: dict | None) -> dict:
    train = {side: inp.side_units(side, "train") for side in inp.sides()}
    family = (inp.spec.get("family") or "").removeprefix("quasi")
    if name == "overdispersion":
        fits = {"r": read_json(inp.baseline / "fit.json"), "python": read_json(inp.candidate / "fit.json")}
        keys = {"r": ("pearson_chi2", "df_residual", "deviance"), "python": ("pearson_chi2", "df_resid", "deviance")}
        out = {}
        for side, (chi2, df, dev) in keys.items():
            f = fits[side]
            if isinstance(f.get(chi2), (int, float)) and f.get(df):
                out[side] = {"pearson_dispersion": _r(f[chi2] / f[df]),
                             "deviance_dispersion": _r(f[dev] / f[df]) if isinstance(f.get(dev), (int, float))
                             else None}
        if not out:
            return {"status": NOT_EXECUTED, "reason": "the fits report no Pearson statistic"}
        return {"status": EXECUTED, "results": out, "reference": "1 for a correctly specified dispersion"}
    if name == "zero_inflation":
        if family != "poisson":
            return {"status": NOT_EXECUTED, "reason": "expected zeros are computed under a Poisson model only"}
        out = {}
        for side, u in train.items():
            tw = sum(inp.wt(w) for *_, w in u)
            if tw:
                observed = sum(inp.wt(w) for _, _, y, _, w in u if y == 0) / tw
                expected = sum(inp.wt(w) * math.exp(-p) for _, _, _, p, w in u) / tw
                out[side] = {"observed_zero_share": _r(observed), "expected_zero_share": _r(expected),
                             "ratio": _r(observed / expected) if expected > 0 else None}
        return {"status": EXECUTED if out else NOT_EXECUTED, "results": out}
    if name == "multicollinearity":
        part = (report or {}).get("vif") or {}
        if part.get("status") != EXECUTED:
            return {"status": NOT_EXECUTED, "reason": "variance inflation factors " + (
                _failed(part) if report is not None else "need the Python diagnostics, which did not run")}
        cols = sorted(part["columns"], key=lambda c: -c["vif"])
        return {"status": EXECUTED, "results": {
            "python": {"max_vif": _r(cols[0]["vif"]) if cols else None,
                       "condition_number": _r(part.get("condition_number"))},
            "columns": [{"name": c["name"], "vif": _r(c["vif"])} for c in cols[:15]]}}
    if name == "residual_distribution":
        out = {side: _moments([y - p for _, _, y, p, _ in u]) for side, u in train.items()}
        out = {k: v for k, v in out.items() if v}
        return {"status": EXECUTED if out else NOT_EXECUTED, "results": out,
                "reference": "skewness and excess kurtosis 0 for normal residuals"}
    if name == "heteroscedasticity":
        out = {}
        for side, u in train.items():
            edges = quantile_edges([p for *_, p, _ in u], QUINTILES)
            groups: dict[int, list[float]] = defaultdict(list)
            for _, _, y, p, _ in u:
                groups[bin_of(p, edges)].append(y - p)
            var = {i + 1: sum(x * x for x in g) / len(g) for i, g in sorted(groups.items()) if len(g) > 1}
            if len(var) > 1 and min(var.values()) > 0:
                out[side] = {"residual_variance_by_quintile": {str(k): _r(v) for k, v in var.items()},
                             "max_min_ratio": _r(max(var.values()) / min(var.values()))}
        return {"status": EXECUTED if out else NOT_EXECUTED, "results": out,
                "reference": "1 for constant residual variance across predicted quintiles"}
    if name == "separation":
        out = {}
        for side, u in train.items():
            if u:
                extreme = sum(1 for *_, p, _ in u if p < 1e-8 or p > 1 - 1e-8)
                out[side] = {"extreme_prediction_share": _r(extreme / len(u))}
        coefs = _coefs(inp.candidate / "coefficients.csv")
        if coefs:
            out.setdefault("python", {})["max_abs_estimate"] = _r(max(abs(e) for e, _ in coefs.values()
                                                                      if math.isfinite(e)))
        out.setdefault("r", {})["fitted_probabilities_warning"] = bool(inp.meta.get("separation_warning"))
        return {"status": EXECUTED, "results": out,
                "reference": "no extreme predictions and no R warning that fitted probabilities are 0 or 1"}
    raise ValueError(f"unknown assumption {name}")


def assumptions(inp: Inputs, plan: dict, report: dict | None) -> dict:
    items = {name: _assumption(inp, name, report) for name in plan["assumptions"]}
    needs = [f"{name.replace('_', ' ')} ({v.get('reason', 'not executed')})" for name, v in items.items()
             if v["status"] != EXECUTED]
    executed = any(v["status"] == EXECUTED for v in items.values())
    return _check(_status(executed, needs), plan["rationale"], {"assumptions": items}, needs)


# ---------------------------------------------------------------- sample suitability
def sample_suitability(inp: Inputs) -> dict:
    rationale = "Enough events, or observations, for each parameter the model estimates."
    side = "python" if inp.units["python"] else "r"
    if not any(inp.side_units(side, s) for s in inp.roles):
        return _check(NOT_EXECUTED, rationale, needs=["valid predictions on at least one evaluation set"],
                      criterion=judge("sample_suitability", inp.spec, {"per_parameter": None}, [inp.dev]))
    sets = {s: {"role": role, "units": inp.totals[s], **{sd: inp.counts(sd, s) for sd in inp.sides()}}
            for s, role in inp.roles.items()}
    dev, k = inp.counts(side, inp.dev), inp.parameters
    if inp.binary:
        limiting, basis = min(dev.get("events") or 0, dev.get("non_events") or 0), "the smaller of events and " \
                                                                                     "non-events"
    elif inp.spec["prediction"]["kind"] == "class_probability":
        limiting, basis = min((dev.get("class_counts") or {"": 0}).values()), "the rarest class"
    else:
        limiting, basis = inp.fitted_rows, "fitted observations"
    per_parameter = limiting / k if k else None
    train = (inp.meta.get("n") or {}).get("train") or 0
    results = {"sets": sets, "development_set": inp.dev, "parameters": k, "fitted_rows": inp.fitted_rows,
               "train_rows": train, "excluded_from_fit": train - inp.fitted_rows,
               "per_parameter": _r(per_parameter), "per_parameter_basis": basis}
    needs = [] if k else ["a parameter count: R reports no finite coefficients"]
    return _check(_status(True, needs), rationale, results, needs,
                  criterion=judge("sample_suitability", inp.spec, {"per_parameter": per_parameter}, [inp.dev]))


# ---------------------------------------------------------------- decision performance
def decision_performance(inp: Inputs) -> dict:
    spec, p = inp.spec, inp.spec["prediction"]
    if not inp.binary:
        return _check(NOT_APPLICABLE, f"Defined for a binary event with a predicted probability; this model has a "
                                      f"{spec['response_type']} response with {p['kind']} outputs.")
    rationale = "How well the model ranks events, and the decisions it makes at the business threshold."
    decision = spec["assessment"].get("decision") or {}
    t = decision.get("threshold")
    sides = inp.sides()
    disc = {s: {side: inp.discrimination(side, s) for side in sides if inp.side_units(side, s)} for s in inp.roles}
    disc = {s: d for s, d in disc.items() if d}
    at = {s: {side: inp.at_threshold(side, s, t) for side in sides if inp.side_units(side, s)}
          for s in inp.roles} if t is not None else {}
    at = {s: d for s, d in at.items() if d}
    needs = [] if t is not None else ["the decision threshold and the action it triggers"]
    if not inp.independent:
        needs.append("an independent evaluation set: decision performance is measured on development data only")
    for s, role in inp.roles.items():
        if role != "development" and not inp.side_units("python", s):
            needs.append(f"valid Python predictions on the independent {s} set")
    aucs = [disc.get(s, {}).get("python", {}).get("auc") for s in inp.judged]
    if any(a is None for a in aucs):
        needs.append("a computable AUC on every judged set (both events and non-events)")
    recalls = [at.get(s, {}).get("python", {}).get("events_captured") for s in inp.judged] if t is not None else [None]
    if t is not None and any(x is None for x in recalls):
        needs.append("computable event recall at the threshold on every judged set")
    values = {"auc": None if any(a is None for a in aucs) else min(aucs),
              "events_captured": None if any(x is None for x in recalls) else min(recalls)}
    criterion = judge("decision_performance", spec, values, inp.judged)
    return _check(_status(bool(disc), needs), rationale, {
        "discrimination": disc, "at_threshold": at, "judged_on": inp.judged, "threshold": t,
        "action": decision.get("action")}, needs, criterion=criterion)


# ---------------------------------------------------------------- the plan
def diagnostics_config(spec: dict) -> dict | None:
    """What the Python diagnostics harness must run for the plan, or None if nothing."""
    vp = spec["validation_plan"]
    config = {"folds": vp["stability"]["folds"] if vp["stability"]["run"] else None,
              "sensitivity": {"columns": vp["sensitivity"]["columns"], "shift_sd": vp["sensitivity"]["shift_sd"]}
              if vp["sensitivity"]["run"] else None,
              "vif": vp["assumptions"]["run"] and "multicollinearity" in vp["assumptions"]["assumptions"]}
    return config if any(config.values()) else None


def evaluate(inp: Inputs) -> dict:
    """Every check, the missing evidence and the business fitness verdict. `inp.diagnostics` is the Python
    diagnostics output directory, or None when they did not run (a failed parity check, or no check needing them).
    A check the plan leaves out still runs when the spec sets a criterion for it, so that criterion is judged."""
    spec, diagnostics = inp.spec, inp.diagnostics
    report = read_json(diagnostics / "diagnostics.json") if diagnostics is not None else None
    if diagnostics is not None and (diagnostics / "error.json").is_file():
        e = read_json(diagnostics / "error.json")
        failed = {"status": "failed", "stage": e.get("stage"), "type": e.get("type")}
        report = {"cross_validation": failed, "sensitivity": failed, "vif": failed}
    vp, stated = spec["validation_plan"], spec["assessment"]["criteria"]
    run = {
        "baseline_comparison": lambda c, planned: baseline_comparison(inp, c["rationale"]),
        "calibration": lambda c, planned: calibration(inp, c["rationale"]),
        "subgroup_performance": lambda c, planned: subgroup_performance(inp, c),
        "stability": lambda c, planned: stability(inp, c, report, planned),
        "sensitivity": lambda c, planned: sensitivity(c, report),
        "assumptions": lambda c, planned: assumptions(inp, c, report),
        "fairness": lambda c, planned: fairness(inp, c),
    }
    checks = {}
    for name in PLANNED:
        c = vp.get(name) or {"run": False, "rationale": NO_FAIRNESS_PLAN, "columns": [], "min_rows": 100}
        keys = [key for key, _ in CRITERIA.get(name, {}).values() if stated.get(key) is not None]
        if c["run"]:
            checks[name] = run[name](c, True)
        elif keys:
            checks[name] = run[name](c, False)
            checks[name]["notes"].insert(0, f"not in the validation plan ({c['rationale']}); executed because the "
                                            f"specification sets {', '.join(keys)}")
        else:
            checks[name] = _check(NOT_PLANNED, c["rationale"])
    checks["sample_suitability"] = sample_suitability(inp)
    checks["decision_performance"] = decision_performance(inp)
    return {"checks": checks, "metrics": inp.metrics, "primary_metric": inp.metrics[0], "notes": list(inp.notes),
            "missing_evidence": missing_evidence(checks, spec), "business_fitness": business_fitness(checks)}


def _value(criterion: dict) -> str:
    return ", ".join(f"{k} {'NA' if v is None else f'{v:.4g}'}" for k, v in criterion["value"].items()
                     if k in criterion["limit"])


def business_fitness(checks: dict) -> dict:
    """The platform's verdict on fitness for the business use, from the criterion outcomes and the execution gaps
    only: any criterion not met; otherwise not established while a criterion is missing or not judged, or a check
    ran only partly or not at all; otherwise every criterion met. Not planned and not applicable checks are not
    gaps."""
    if not checks:
        return {"verdict": "not established", "basis": ["the model validation was not recorded"]}
    with_criterion = {k: c for k, c in checks.items() if c.get("criterion")}
    unmet = [f"{TITLES[k]}: {c['criterion']['rule']} not met ({_value(c['criterion'])})"
             for k, c in with_criterion.items() if c["criterion"]["outcome"] == NOT_MET]
    if unmet:
        return {"verdict": "criteria not met", "basis": unmet}
    gaps = []
    for k, c in checks.items():
        if c["status"] in (PARTIAL, NOT_EXECUTED):
            gaps.append(f"{TITLES[k]}: {c['status']}" + (f" (needs {'; '.join(c['needs'])})" if c["needs"] else ""))
        cr = c.get("criterion")
        if cr and cr["outcome"] == NO_CRITERION:
            gaps.append(f"{TITLES[k]}: no criterion for {', '.join(unstated(k, cr))}")
        elif cr and cr["outcome"] == NOT_JUDGED:
            gaps.append(f"{TITLES[k]}: {cr['rule']} not judged")
    if gaps:
        return {"verdict": "not established", "basis": gaps}
    return {"verdict": "criteria met", "basis": [f"{TITLES[k]}: {c['criterion']['rule']} met "
                                                 f"({_value(c['criterion'])})" for k, c in with_criterion.items()]}


def missing_evidence(checks: dict, spec: dict) -> list[str]:
    """Everything the validation could not establish, in one list: checks left out of the plan, what the executed
    ones lacked, criteria the spec does not state or that could not be judged, and the information the spec says
    is still missing."""
    out = []
    for name, c in checks.items():
        if c["status"] == NOT_PLANNED:
            out.append(f"{TITLES[name]}: not in the validation plan ({c['rationale']})")
        out += [f"{TITLES[name]}: {n}" for n in c["needs"]]
        cr = c.get("criterion")
        if cr and cr["outcome"] == NO_CRITERION:
            out.append(f"{TITLES[name]}: no acceptance criterion ({', '.join(unstated(name, cr))}) in the "
                       "specification")
        elif cr and cr["outcome"] == NOT_JUDGED:
            out.append(f"{TITLES[name]}: the criterion {cr['rule']} could not be judged")
    out += [f"Still to provide: {x}" for x in spec["assessment"]["missing_information"]]
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- agent views
def evaluation_view(ev: dict) -> dict:
    """The evaluation as the Diagnostician sees it: statuses, criteria, needs and Python's aggregate results,
    without R's values or coefficient estimates (its instructions reach the Translator)."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("r", "r_vs_constant", "estimate", "bins")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    checks = ev.get("checks", {})
    out = {k: strip(v) for k, v in checks.items()}
    for name in ("subgroup_performance", "fairness"):
        if name in out:
            out[name]["results"] = subgroup_prompt_results(checks[name].get("results", {}), sides=("python",))
    return {"checks": out, "missing_evidence": ev.get("missing_evidence", []),
            "business_fitness": ev.get("business_fitness")}


def subgroup_prompt_results(results: dict, sides: tuple[str, ...] = SIDES) -> dict:
    """Keep only adequately populated subgroup cells in agent prompts. Omit small cell labels and data cut points."""
    if not results:
        return {}
    minimum = max(MIN_AGENT_GROUP_ROWS, results.get("min_rows", 0))
    columns = {}
    for name, col in results.get("columns", {}).items():
        groups, withheld = [], 0
        for group in col.get("groups", []):
            sets = {}
            for s, cell in group["sets"].items():
                visible = {side: cell[side] for side in sides
                           if side in cell and cell.get("scored_rows", {}).get(side, 0) >= minimum}
                withheld += any(side in cell and side not in visible for side in sides)
                if visible:
                    sets[s] = {"rows": cell["rows"], **visible}
            if sets:
                groups.append({"group": group["group"], "sets": sets})
        columns[name] = {"status": col["status"], "groups": groups, "cells_withheld": withheld}
        if "reason" in col:
            columns[name]["reason"] = col["reason"]
        if "impact_ratio" in col:
            columns[name]["impact_ratio"] = col["impact_ratio"]
    out = {"metric": results.get("metric"), "min_rows": minimum, "judged_on": results.get("judged_on", []),
           "columns": columns}
    for key in ("impact_ratio", "max_group_deviation", "threshold", "favourable_outcome"):
        if key in results:
            out[key] = results[key]
    return out


# ---------------------------------------------------------------- tables
def tables(v: dict) -> dict[str, tuple[list[str], list[list]]]:
    """The diagnostic tables logged with the validation run: file name -> (header, rows)."""
    out: dict[str, tuple[list[str], list[list]]] = {}
    out["parity_checks.csv"] = (["check", "quantity", "status", "required", "ratio", "max_abs_diff", "atol", "rtol",
                                 "reason"],
                                [[c["name"], c.get("quantity"), c.get("status"), c.get("required"), c.get("value"),
                                  c.get("max_abs_diff"), c.get("atol"), c.get("rtol"), c.get("reason")]
                                 for c in v["checks"]])
    ev = v.get("evaluation") or {}
    checks = ev.get("checks") or {}
    out["validation_checks.csv"] = (["check", "status", "criterion", "outcome", "needs"], [
        [k, c["status"], (c.get("criterion") or {}).get("rule"), (c.get("criterion") or {}).get("outcome"),
         "; ".join(c.get("needs", []))] for k, c in checks.items()])
    out["missing_evidence.csv"] = (["item"], [[x] for x in ev.get("missing_evidence", [])])
    res = lambda name: (checks.get(name) or {}).get("results") or {}  # noqa: E731
    if sets := res("baseline_comparison").get("sets"):
        out["performance.csv"] = (["set", "metric", "r", "python", "constant", "r_vs_constant",
                                   "python_vs_constant"],
                                  [[s, m, x.get("r"), x.get("python"), x.get("constant"), x.get("r_vs_constant"),
                                    x.get("python_vs_constant")] for s, e in sets.items() for m, x in e.items()])
    if sets := res("calibration").get("sets"):
        rows = []
        for s, keys in sets.items():
            for key, sides in keys.items():
                for side, d in sides.items():
                    rows += [[s, key, side, b["bin"], b["units"], b["mean_predicted"], b["mean_observed"]]
                             for b in d.get("bins", [])]
        out["calibration_bins.csv"] = (["set", "output", "side", "bin", "units", "mean_predicted", "mean_observed"],
                                       rows)
    if cols := res("subgroup_performance").get("columns"):
        rows = []
        for col, c in cols.items():
            for g in c.get("groups", []):
                for s, e in g["sets"].items():
                    for side in SIDES:
                        if side in e:
                            rows.append([col, g["group"], s, side, e["rows"], e["small"], *[
                                e[side].get(k) for k in ("mean_observed", "mean_predicted", "observed_expected",
                                                         res("subgroup_performance")["metric"], "auc")]])
        out["subgroups.csv"] = (["column", "group", "set", "side", "rows", "small", "mean_observed", "mean_predicted",
                                 "observed_expected", res("subgroup_performance").get("metric"), "auc"], rows)
    if psi := res("stability").get("psi"):
        out["psi.csv"] = (["side", "compared", "units", "psi", "mean_prediction", "mean_observed"], [
            row for side, x in psi.items() for row in
            [[side, f"{s} vs development", "", v, "", ""] for s, v in x["vs_development"].items()]
            + [[side, f"period {k}", p["units"], p["psi_vs_first"], p["mean_prediction"], p["mean_observed"]]
               for k, p in x["by_period"].items()]])
    if cv := res("stability").get("cross_validation"):
        names = list(cv["summary"])
        out["cv_folds.csv"] = (["fold", "fit_rows", "held_out_rows", "converged", *names],
                               [[f["fold"], f["fit_rows"], f["held_out_rows"], f["converged"],
                                 *[f.get(m) for m in names]] for f in cv["folds"]])
        out["cv_coefficients.csv"] = (["term", "estimate", "fold_sd", "relative_sd", "sign_changes"],
                                      [[t["term"], t["estimate"], t["fold_sd"], t["relative_sd"], t["sign_changes"]]
                                       for t in cv["coefficients"]])
    if cols := res("sensitivity").get("columns"):
        out["sensitivity.csv"] = (["column", "set", "output", "direction", "rows", "mean_prediction", "mean_change",
                                   "mean_abs_change", "max_abs_change"],
                                  [[c, s, k, d, x["rows"], x["mean_prediction"], x["mean_change"],
                                    x["mean_abs_change"], x["max_abs_change"]]
                                   for c, e in cols.items() if e.get("status") == EXECUTED
                                   for s, keys in e["sets"].items() for k, dirs in keys.items()
                                   for d, x in dirs.items()])
    if cols := res("fairness").get("columns"):
        out["fairness.csv"] = (["column", "group", "set", "side", "rows", "small", "mean_observed", "mean_predicted",
                                "observed_expected", "favourable_share"],
                               [[col, g["group"], s, side, e["rows"], e["small"],
                                 *[e[side].get(k) for k in ("mean_observed", "mean_predicted", "observed_expected",
                                                            "favourable_share")]]
                                for col, c in cols.items() for g in c.get("groups", [])
                                for s, e in g["sets"].items() for side in SIDES if side in e])
    items = res("assumptions").get("assumptions") or {}
    if (m := items.get("multicollinearity") or {}).get("status") == EXECUTED:
        out["vif.csv"] = (["column", "vif"], [[c["name"], c["vif"]] for c in m["results"]["columns"]])
    if sets := res("sample_suitability").get("sets"):
        fields = ["valid", "rows", "weight_sum", "events", "non_events", "event_rate", "mean_observed"]
        out["sample.csv"] = (["set", "role", "side", *fields], [
            [s, e["role"], side, *[e[side].get(f) for f in fields]] for s, e in sets.items() for side in SIDES
            if side in e])
    if disc := res("decision_performance").get("discrimination"):
        at = res("decision_performance").get("at_threshold") or {}
        fields = ["flagged_share", "event_rate_flagged", "event_rate_not_flagged", "events_captured",
                  "non_events_flagged"]
        out["decision.csv"] = (["set", "side", "auc", "gini", "ks", *fields], [
            [s, side, d["auc"], d["gini"], d["ks"], *[at.get(s, {}).get(side, {}).get(f) for f in fields]]
            for s, sides in disc.items() for side, d in sides.items()])
    return out


def write_tables(v: dict, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (header, rows) in tables(v).items():
        with open(dest / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(["" if x is None else x for x in row] for row in rows)
        written.append(name)
    return written
