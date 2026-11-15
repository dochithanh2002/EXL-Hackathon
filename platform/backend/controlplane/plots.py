"""Diagnostic plots as SVG files, drawn from recorded aggregates by `charts` (matplotlib).

Each file is light-themed and self-contained, with text as paths, and has a legend whenever it shows two or more
series. The report puts each plot beside its table, which is the plot's accessible view. `render` writes every
plot the validation and evaluation support and returns what it wrote."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .charts import Series, bars, lines
from .charts import fmt as _fmt

log = logging.getLogger(__name__)


@dataclass
class Plot:
    name: str                                     # file name
    title: str
    section: str                                  # where the report shows it
    svg: str


# ---------------------------------------------------------------- the report's plots
SIDE_NAMES = {"r": "R", "python": "Python"}


def _parity(v: dict) -> Plot | None:
    rows = [c for c in v["checks"] if c.get("kind") == "tolerance" and c.get("value") is not None]
    if not rows:
        return None
    floor = 1e-6
    svg = bars("Parity: worst difference in tolerance units", "1 is the agreed tolerance",
               [("", [c["name"] for c in rows], [Series("ratio", [max(c["value"], floor) for c in rows])])],
               dots=True, log=True, ref=1.0, x_label=f"|Python − R| / (atol + rtol·|R|), floored at {_fmt(floor)}")
    return svg and Plot("parity.svg", "Parity checks", "migration.parity_results", svg)


def _performance(ev: dict) -> Plot | None:
    c = ev["checks"]["baseline_comparison"]
    sets = (c.get("results") or {}).get("sets") or {}
    if not sets:
        return None
    names = list(dict.fromkeys(m for e in sets.values() for m in e))[:4]
    sides = [k for k in ("r", "python", "constant") if any(k in e.get(m, {}) for e in sets.values() for m in names)]
    panels = [(m.replace("_", " "), list(sets), [Series(SIDE_NAMES.get(k, "Constant comparator"),
                                                        [sets[s].get(m, {}).get(k) for s in sets]) for k in sides])
              for m in names]
    svg = bars("Model performance against a constant comparator", "", panels)
    return svg and Plot("performance.svg", "Model performance", "validation.baseline_comparison", svg)


def _calibration(ev: dict) -> list[Plot]:
    res = ev["checks"]["calibration"].get("results") or {}
    out = []
    for s, keys in (res.get("sets") or {}).items():
        for key, sides in keys.items():
            series = [Series(SIDE_NAMES[k], [b["mean_observed"] for b in d["bins"]],
                             [b["mean_predicted"] for b in d["bins"]]) for k, d in sides.items() if d.get("bins")]
            name = f"calibration-{s}" + ("" if key == "value" else f"-{_slug(key)}") + ".svg"
            svg = lines(f"Calibration on {s}" + ("" if key == "value" else f", {key}"), "by decile of prediction",
                        series, x_label="mean predicted", y_label="mean observed", diagonal=True)
            if svg:
                out.append(Plot(name, f"Calibration on {s}" + ("" if key == "value" else f", {key}"),
                                "validation.calibration", svg))
    return out


def _subgroups(ev: dict, check: str = "subgroup_performance", prefix: str = "subgroups") -> list[Plot]:
    """Observed / expected (or the primary metric, or the favourable share at the decision threshold) by group."""
    res = (ev["checks"].get(check) or {}).get("results") or {}
    out = []
    for col, c in (res.get("columns") or {}).items():
        if c.get("status") != "executed":
            continue
        for s in res.get("judged_on", []):
            groups = [g for g in c["groups"] if s in g["sets"]]
            cells = [g["sets"][s].get("python", g["sets"][s].get("r", {})) for g in groups]
            stat = ("favourable_share" if any("favourable_share" in x for x in cells) else
                    "observed_expected" if any("observed_expected" in x for x in cells) else res.get("metric"))
            if stat is None:
                continue
            series = [Series(SIDE_NAMES[k], [g["sets"][s].get(k, {}).get(stat) for g in groups])
                      for k in ("r", "python") if any(k in g["sets"][s] for g in groups)]
            label = "observed / expected" if stat == "observed_expected" else stat.replace("_", " ")
            svg = bars(f"{col} on {s}", "", [("", [g["group"] for g in groups], series)],
                       dots=True, ref=1.0 if stat == "observed_expected" else None, x_label=label)
            if svg:
                out.append(Plot(f"{prefix}-{_slug(col)}-{s}.svg", f"{col} on {s}", f"validation.{check}", svg))
    return out


def _stability(ev: dict) -> list[Plot]:
    res = ev["checks"]["stability"].get("results") or {}
    out = []
    if tc := res.get("training_curve"):
        svg = lines("Python fit by iteration", "", [Series(tc["metric"], tc["values"],
                                                           list(range(1, len(tc["values"]) + 1)))],
                    x_label="iteration", y_label=tc["metric"])
        if svg:
            out.append(Plot("training-curve.svg", "Python fit by iteration", "validation.stability", svg))
    if cv := res.get("cross_validation"):
        names = list(cv["summary"])[:4]
        panels = [(m.replace("_", " "), [f"fold {f['fold']}" for f in cv["folds"]] + ["in sample"],
                   [Series("Python", [f[m] for f in cv["folds"]] + [cv["summary"][m]["in_sample"]])])
                  for m in names]
        svg = bars("Cross-validated performance", "held-out fold of the train rows", panels, dots=True)
        if svg:
            out.append(Plot("cross-validation.svg", "Cross-validated performance", "validation.stability", svg))
    return out


def _sensitivity(ev: dict) -> list[Plot]:
    res = ev["checks"]["sensitivity"].get("results") or {}
    cols = {c: v for c, v in (res.get("columns") or {}).items() if v.get("status") == "executed"}
    if not cols:
        return []
    out = []
    for s in dict.fromkeys(s for v in cols.values() for s in v["sets"]):
        keys = list(dict.fromkeys(k for v in cols.values() for k in v["sets"].get(s, {})))
        panels = [("" if k == "value" else k, list(cols), [
            Series(f"shift {d}", [cols[c]["sets"].get(s, {}).get(k, {}).get(d, {}).get("mean_change") for c in cols])
            for d in ("down", "up")]) for k in keys]
        svg = bars(f"Sensitivity on {s}", f"mean change in prediction, ±{res['shift_sd']:g} sd", panels,
                   x_label="mean change in prediction")
        if svg:
            out.append(Plot(f"sensitivity-{s}.svg", f"Sensitivity on {s}", "validation.sensitivity", svg))
    return out


def _assumptions(ev: dict) -> list[Plot]:
    items = ((ev["checks"]["assumptions"].get("results") or {}).get("assumptions") or {})
    m = items.get("multicollinearity") or {}
    if m.get("status") != "executed" or not m["results"]["columns"]:
        return []
    cols = m["results"]["columns"]
    svg = bars("Variance inflation factors", "Python design, largest first",
               [("", [c["name"] for c in cols], [Series("VIF", [c["vif"] for c in cols])])], ref=1.0, x_label="VIF")
    return [Plot("vif.svg", "Variance inflation factors", "validation.assumptions", svg)] if svg else []


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in s.lower()).strip("-") or "x"


def render(validation: dict, dest: Path) -> list[dict]:
    """Write every plot the validation and its evaluation support into `dest`; returns name, title and section."""
    ev = validation.get("evaluation") or {}
    makers = [lambda: [_parity(validation)]]
    if ev.get("checks"):
        makers += [lambda: [_performance(ev)], *(lambda f=f: f(ev) for f in (_calibration, _subgroups, _stability,
                                                                            _sensitivity, _assumptions)),
                   lambda: _subgroups(ev, "fairness", "fairness")]
    plots = []
    for make in makers:
        try:
            plots += [p for p in make() if p]
        except Exception:                      # a plot is presentation: it never fails the validation it shows
            log.exception("could not draw a plot")
    dest.mkdir(parents=True, exist_ok=True)
    for p in plots:
        (dest / p.name).write_text(p.svg)
    return [{"name": p.name, "title": p.title, "section": p.section} for p in plots]
