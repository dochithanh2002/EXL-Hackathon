"""The statistics model validation rests on, computed once per side and set from both sides' predictions.

`Inputs` reads R's and the Python candidate's predictions.csv once each (the parity comparison reuses them) and
keeps each side's valid units per set, with R's prior weights on both. Every statistic the checks share is computed
here on first use and cached per model and set: metrics (the constant comparator's too), calibration and quantile
coverage, discrimination, decisions at the business threshold, the population stability index by set and period,
and sample counts. Results are aggregates: no row or row-level value leaves this module."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from functools import cached_property
from pathlib import Path

from .runner.parity import BINARY, _auc_ks, _coefs, _predictions, metric, metric_weighting, units_by_set

BINS = 10
PSI_FLOOR = 1e-4
LOWER_IS_BETTER = {"rmse", "mae", "deviance", "mean_deviance", "log_loss", "brier", "pinball_loss"}
SCORES = ("auc", "gini", "ks")
SIDES = ("r", "python")


def rounded(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(f"{x:.6g}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


# ---------------------------------------------------------------- binning
def quantile_edges(values: list[float], n: int) -> list[float]:
    """Inner edges of up to n equal-count bins; tied values share a bin."""
    s = sorted(values)
    if not s:
        return []
    return sorted({s[min(len(s) - 1, int(len(s) * i / n))] for i in range(1, n)})


def bin_of(x: float, edges: list[float]) -> int:
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
        e[bin_of(x, edges)] += 1
    for x in actual:
        a[bin_of(x, edges)] += 1
    total = 0.0
    for ei, ai in zip(e, a):
        pe, pa = max(ei / len(expected), PSI_FLOOR), max(ai / len(actual), PSI_FLOOR)
        total += (pa - pe) * math.log(pa / pe)
    return total


# ---------------------------------------------------------------- metrics
def primary_metric(spec: dict) -> str:
    """The proper loss each prediction kind is judged by."""
    p = spec["prediction"]
    if p["kind"] == "quantile":
        return "pinball_loss"
    if p["kind"] == "class_probability" or spec["response_type"] in BINARY:
        return "log_loss"
    if (spec.get("family") or "").removeprefix("quasi") in ("poisson", "binomial", "Gamma", "gaussian",
                                                            "inverse.gaussian"):
        return "mean_deviance"
    return "rmse"


def evaluated_metrics(spec: dict) -> list[str]:
    """The primary loss and the spec's metrics that a set of predictions defines (AIC needs a fit)."""
    names = [primary_metric(spec)] + [m for m in spec["metrics"] if m != "aic"]
    return list(dict.fromkeys(names))


def score(name: str, units: list, spec: dict, weighted: bool) -> float | None:
    kind = spec["prediction"]["kind"]
    if name == "mean_deviance":
        d = metric("deviance", units, spec.get("family"), kind, weighted)
        total = sum(w for *_, w in units)
        return None if d is None or not total else d / total
    return metric(name, units, spec.get("family"), kind, weighted)


def comparison(name: str, model: float | None, constant: float | None) -> float | None:
    """Skill (1 - model / constant) for a loss, the difference for a ranking score, None for a ratio."""
    if model is None or constant is None:
        return None
    if name in LOWER_IS_BETTER:
        return 1 - model / constant if constant > 0 else None
    if name in SCORES:
        return model - constant
    return None


# ---------------------------------------------------------------- calibration
def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def logistic_recalibration(g: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    """(intercept, slope) of y ~ a + b logit(p) by Newton-Raphson on the weighted binomial likelihood."""
    xs = [(_logit(p), y, w) for y, p, w in g]
    a, b = 0.0, 1.0
    for _ in range(50):
        ga = gb = haa = hab = hbb = 0.0
        for x, y, w in xs:
            eta = max(min(a + b * x, 35.0), -35.0)
            mu = 1 / (1 + math.exp(-eta))
            v = w * mu * (1 - mu)
            ga += w * (y - mu)
            gb += w * (y - mu) * x
            haa, hab, hbb = haa + v, hab + v * x, hbb + v * x * x
        det = haa * hbb - hab * hab
        if det <= 1e-12 * max(haa * hbb, 1e-300):
            return None
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-10:
            return a, b
    return None


def linear_recalibration(g: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    """(intercept, slope) of the weighted least-squares fit of y on the prediction."""
    tw = sum(w for *_, w in g)
    if not tw:
        return None
    mx, my = sum(w * p for _, p, w in g) / tw, sum(w * y for y, _, w in g) / tw
    sxx = sum(w * (p - mx) ** 2 for _, p, w in g)
    if sxx <= 0:
        return None
    b = sum(w * (p - mx) * (y - my) for y, p, w in g) / sxx
    return my - b * mx, b


def calibration_stats(g: list[tuple[float, float, float]], probability: bool) -> dict:
    """Observed against predicted for (observed, prediction, weight) triples: totals, observed / expected, the
    recalibration fit and the decile table."""
    tw = sum(w for *_, w in g)
    obs, exp = sum(w * y for y, _, w in g), sum(w * p for _, p, w in g)
    edges = quantile_edges([p for _, p, _ in g], BINS)
    bins: dict[int, list] = defaultdict(list)
    for y, p, w in g:
        bins[bin_of(p, edges)].append((y, p, w))
    table = []
    for i in sorted(bins):
        b, bw = bins[i], sum(w for *_, w in bins[i])
        if bw:
            table.append({"bin": i + 1, "units": len(b), "mean_predicted": rounded(sum(w * p for _, p, w in b) / bw),
                          "mean_observed": rounded(sum(w * y for y, _, w in b) / bw)})
    gap = max((abs(t["mean_observed"] - t["mean_predicted"]) for t in table
               if None not in (t["mean_observed"], t["mean_predicted"])), default=None)
    fit = (logistic_recalibration if probability and all(0 <= y <= 1 for y, *_ in g) else linear_recalibration)(g)
    return {"units": len(g), "weight": rounded(tw), "observed": rounded(obs), "expected": rounded(exp),
            "observed_expected": rounded(obs / exp) if exp > 0 else None,
            "intercept": rounded(fit[0]) if fit else None, "slope": rounded(fit[1]) if fit else None,
            "max_bin_gap": rounded(gap), "bins": table}


# ---------------------------------------------------------------- inputs
class Inputs:
    """Both sides' predictions, read once, as valid units per set with R's prior weights on both, and the shared
    statistics computed from them. `diagnostics` is the Python diagnostics output directory, if they ran."""

    def __init__(self, baseline: Path, candidate: Path | None, diagnostics: Path | None, spec: dict):
        self.baseline, self.candidate, self.diagnostics, self.spec = baseline, candidate, diagnostics, spec
        r = _predictions(baseline / "predictions.csv")
        found = candidate is not None and (candidate / "predictions.csv").is_file()
        py = _predictions(candidate / "predictions.csv") if found else {}
        self.predictions = {"r": r, "python": py}
        self.weights = {k: w for k, (_, _, w) in r.items()}
        self.units = {"r": units_by_set(r), "python": units_by_set(py, self.weights) if py else {}}
        self.totals = Counter(s for s, _, _ in r)
        self.weighted, why = metric_weighting(spec)
        self.notes = [why] if why else []
        a, p = spec["assessment"], spec["prediction"]
        self.roles = {e["set"]: e["role"] for e in a["evaluation_sets"]}
        self.dev = next((s for s, role in self.roles.items() if role == "development"), "train")
        self.independent = [s for s, role in self.roles.items() if role != "development"
                            and self.units["r"].get(s)]
        self.judged = self.independent or [self.dev]
        self.meta = read_json(baseline / "meta.json")
        self.metrics = evaluated_metrics(spec)
        self.keys = p["keys"] or [""]
        self.binary = spec["response_type"] in BINARY and p["kind"] in ("scalar", "time_indexed")
        self.probability = p["quantity"] == "probability" or p["kind"] == "class_probability"
        self._cache: dict[tuple, object] = {}

    def wt(self, w: float) -> float:
        return w if self.weighted else 1.0

    def side_units(self, side: str, s: str) -> list:
        return self.units[side].get(s, [])

    def sides(self) -> list[str]:
        return [side for side in SIDES if self.units[side]]

    def _cached(self, key: tuple, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    # ------------------------------------------------------------ the fit
    @cached_property
    def parameters(self) -> int:
        """R's estimated parameters: its finite coefficients."""
        return sum(1 for e, _ in _coefs(self.baseline / "coefficients.csv").values() if math.isfinite(e))

    @cached_property
    def fitted_rows(self) -> int:
        return read_json(self.baseline / "design.json").get("n_fit") or self.meta.get("n_fit") or 0

    @cached_property
    def periods(self) -> dict[tuple[str, int], str]:
        """(set, row) -> period, when the spec names a period column."""
        split = self.baseline / "split.csv"
        if not self.spec["assessment"].get("period_column") or not split.is_file():
            return {}
        with open(split, newline="") as f:
            return {(r["set"], int(r["row"])): r["period"] for r in csv.DictReader(f)
                    if r.get("period") not in (None, "")}

    # ------------------------------------------------------------ metrics
    @cached_property
    def constants(self) -> dict[str, float]:
        """The constant comparator: the development set's weighted mean (or quantile) of R's observed values, by
        key."""
        dev = self.side_units("r", self.dev)
        out: dict[str, float] = {}
        for key in self.keys:
            g = [(y, self.wt(w)) for _, k, y, _, w in dev if k == key]
            total = sum(w for _, w in g)
            if not total:
                continue
            if self.spec["prediction"]["kind"] == "quantile":
                tau, acc = float(key), 0.0
                for y, w in sorted(g):
                    acc += w
                    if acc >= tau * total:
                        out[key] = y
                        break
            else:
                out[key] = sum(w * y for y, w in g) / total
        return out

    def model_units(self, model: str, s: str) -> list:
        """A side's units, or for "constant" R's units with every prediction replaced by the comparator."""
        if model != "constant":
            return self.side_units(model, s)
        c = self.constants
        return [(row, k, y, c[k], w) for row, k, y, _, w in self.side_units("r", s) if k in c]

    def metric(self, model: str, s: str, name: str) -> float | None:
        """One metric of "r", "python" or "constant" on a set, rounded; the ranking scores share one pass."""
        if name not in SCORES:
            return self._cached(("metric", model, s, name), lambda: rounded(
                score(name, self.model_units(model, s), self.spec, self.weighted)))
        return self.discrimination(model, s)[name]

    def discrimination(self, model: str, s: str) -> dict[str, float | None]:
        def make():
            units = self.model_units(model, s)
            res = None
            if self.spec["prediction"]["kind"] not in ("class_probability", "quantile") and units:
                res = _auc_ks([(y, p, self.wt(w)) for _, _, y, p, w in units])
            auc, ks = res if res else (None, None)
            return {"auc": rounded(auc), "gini": rounded(None if auc is None else 2 * auc - 1), "ks": rounded(ks)}
        return self._cached(("discrimination", model, s), make)

    # ------------------------------------------------------------ calibration
    def calibration(self, side: str, s: str) -> dict[str, dict]:
        """Key ("value" for one output) -> observed against predicted on a set, or quantile coverage. A key without
        weighted observations is absent."""
        def make():
            out = {}
            quantile = self.spec["prediction"]["kind"] == "quantile"
            for key in self.keys:
                g = [(y, pr, self.wt(w)) for _, k, y, pr, w in self.side_units(side, s) if k == key]
                tw = sum(w for *_, w in g)
                if not g or tw <= 0:
                    continue
                if quantile:
                    cover = sum(w for y, q, w in g if y <= q) / tw
                    out[key] = {"units": len(g), "nominal": float(key), "coverage": rounded(cover)}
                else:
                    out[key or "value"] = calibration_stats(g, self.probability)
            return out
        return self._cached(("calibration", side, s), make)

    # ------------------------------------------------------------ samples and decisions
    def counts(self, side: str, s: str) -> dict:
        """Valid units, data rows, total weight and events (or class counts, or the mean observed value)."""
        def make():
            u = self.side_units(side, s)
            rows = {row: w for row, *_, w in u}
            out = {"valid": len(u), "rows": len(rows), "weight_sum": rounded(sum(self.wt(w) for w in rows.values()))}
            total = sum(self.wt(w) for *_, w in u)
            if self.binary:
                events = sum(self.wt(w) * y for _, _, y, _, w in u)
                out |= {"events": rounded(events), "non_events": rounded(total - events),
                        "event_rate": rounded(events / total) if total else None}
            elif self.spec["prediction"]["kind"] == "class_probability":
                classes: dict[str, float] = defaultdict(float)
                for _, key, y, _, w in u:
                    classes[key] += self.wt(w) * y
                out["class_counts"] = {k: rounded(v) for k, v in classes.items()}
            else:
                out["mean_observed"] = rounded(sum(self.wt(w) * y for _, _, y, _, w in u) / total) if total else None
            return out
        return self._cached(("counts", side, s), make)

    def at_threshold(self, side: str, s: str, threshold: float) -> dict:
        """The decisions a threshold on the prediction makes on a set."""
        def make():
            u = self.side_units(side, s)
            total = sum(self.wt(w) for *_, w in u)
            events = sum(self.wt(w) * y for _, _, y, _, w in u)
            flagged = [(y, self.wt(w)) for _, _, y, pr, w in u if pr >= threshold]
            fw, fe = sum(w for _, w in flagged), sum(w * y for y, w in flagged)
            rest_w, rest_e = total - fw, events - fe
            return {"flagged_share": rounded(fw / total) if total else None,
                    "event_rate_flagged": rounded(fe / fw) if fw else None,
                    "event_rate_not_flagged": rounded(rest_e / rest_w) if rest_w else None,
                    "events_captured": rounded(fe / events) if events else None,
                    "non_events_flagged": rounded((fw - fe) / (total - events)) if total - events else None}
        return self._cached(("threshold", side, s, threshold), make)

    # ------------------------------------------------------------ stability
    def stability(self, side: str) -> dict | None:
        """The score distribution's PSI against the development set for each independent set, and by period
        against the first period, with each period's mean prediction and observed value; None when there is
        neither an independent set nor two periods."""
        def make():
            def scores(u, key):
                return [pr for _, k, _, pr, _ in u if k == key]

            def worst(ref_units, units):
                values = []
                for k in self.keys:
                    ref = scores(ref_units, k)
                    values.append(psi(ref, scores(units, k), quantile_edges(ref, BINS)) if ref else None)
                return rounded(max((v for v in values if v is not None), default=None))
            dev = self.side_units(side, self.dev)
            by_set = {s: worst(dev, self.side_units(side, s)) for s in self.independent if self.side_units(side, s)}
            grouped: dict[str, list] = defaultdict(list)
            if self.periods:
                for s in self.roles:
                    for unit in self.side_units(side, s):
                        if (per := self.periods.get((s, unit[0]))) is not None:
                            grouped[per].append(unit)
            order = sorted(grouped)
            by_period = {}
            for per in order:
                u = grouped[per]
                total = sum(self.wt(w) for *_, w in u)
                by_period[per] = {
                    "units": len(u), "psi_vs_first": worst(grouped[order[0]], u),
                    "mean_prediction": rounded(sum(self.wt(w) * pr for *_, pr, w in u) / total) if total else None,
                    "mean_observed": rounded(sum(self.wt(w) * y for _, _, y, _, w in u) / total) if total else None}
            if not by_set and len(by_period) < 2:
                return None
            values = [v for v in by_set.values() if v is not None] + [
                v["psi_vs_first"] for v in by_period.values() if v["psi_vs_first"] is not None]
            return {"vs_development": by_set, "by_period": by_period, "max_psi": max(values, default=None)}
        return self._cached(("stability", side), make)


def scores(inp: Inputs) -> tuple[dict[str, dict[str, dict[str, float | None]]], dict[str, float]]:
    """model ("r", "python", "constant") -> set -> metric, and the constant comparator's values by key. The models
    are scored on the same rows with the same weights, so their runs compare directly."""
    out: dict[str, dict] = {m: {} for m in (*inp.sides(), "constant")}
    for s in inp.roles:
        if not inp.side_units("r", s):
            continue
        for model in out:
            out[model][s] = {m: inp.metric(model, s, m) for m in inp.metrics}
    return out, inp.constants
