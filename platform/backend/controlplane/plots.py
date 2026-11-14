"""Diagnostic plots as self-contained SVG files, drawn from recorded aggregates by code (no plotting dependency).

Each file carries its own light and dark styles, a legend whenever it shows two or more series, and a `<title>` on
every mark for hover. The report puts each plot beside its table, which is the plot's accessible view. `render`
writes every plot the validation and evaluation support and returns what it wrote."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

log = logging.getLogger(__name__)
W = 680
PAD_L, PAD_R, PAD_T = 196, 24, 56
ROW, PANEL_GAP = 22, 34
STYLE = """<style>
.viz{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){.viz{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
.bg{fill:var(--surface)}.t{fill:var(--ink);font-size:14px;font-weight:600}.st{fill:var(--ink2);font-size:12px}
.lab{fill:var(--ink2);font-size:12px}.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.ref{stroke:var(--muted);stroke-width:1}.s1{fill:var(--s1);stroke:var(--s1)}.s2{fill:var(--s2);stroke:var(--s2)}
.s3{fill:var(--s3);stroke:var(--s3)}.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{stroke:var(--surface);stroke-width:2}.bar{stroke:none}
</style>"""


@dataclass
class Series:
    name: str
    values: list[float | None]                    # one per row (bars, dots) or per point (lines)
    x: list[float] = field(default_factory=list)  # lines only


@dataclass
class Plot:
    name: str                                     # file name
    title: str
    section: str                                  # where the report shows it
    svg: str


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        return f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-")
    return f"{v:,.4g}" if a >= 1000 else f"{v:.4g}"


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + (abs(lo) or 1)
    step = 10 ** math.floor(math.log10((hi - lo) / n))
    for m in (1, 2, 2.5, 5, 10):
        if (hi - lo) / (step * m) <= n:
            step *= m
            break
    start = math.floor(lo / step) * step
    return [round(start + i * step, 12) for i in range(int(math.ceil((hi - start) / step)) + 1)]


class Scale:
    def __init__(self, lo: float, hi: float, a: float, b: float, log: bool = False, zero: bool = True):
        self.log = log
        if log:
            lo, hi = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
            if hi == lo:
                hi += 1
            self.ticks = [10.0 ** e for e in range(int(lo), int(hi) + 1)]
        else:
            if zero:
                lo, hi = min(lo, 0), max(hi, 0)
            self.ticks = _ticks(lo, hi)
            lo, hi = min(self.ticks[0], lo), max(self.ticks[-1], hi)
        self.lo, self.hi, self.a, self.b = lo, hi, a, b

    def __call__(self, v: float) -> float:
        x = math.log10(v) if self.log else v
        return self.a + (x - self.lo) / ((self.hi - self.lo) or 1) * (self.b - self.a)


def _svg(height: float, title: str, subtitle: str, body: list[str], legend: list[str]) -> str:
    keys, x = [], PAD_L
    if len(legend) > 1:
        for i, name in enumerate(legend):
            keys.append(f'<rect x="{x}" y="38" width="10" height="10" rx="2" class="s{i + 1} bar"/>'
                        f'<text x="{x + 15}" y="47" class="lab">{escape(name)}</text>')
            x += 30 + 7 * len(name)
    head = [f'<text x="16" y="22" class="t">{escape(title)}</text>']
    if subtitle:
        head.append(f'<text x="{x + 8 if keys else 16}" y="47" class="st">{escape(subtitle)}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height:.0f}" width="{W}" '
            f'height="{height:.0f}" role="img" aria-label="{escape(title, {chr(34): "&quot;"})}" class="viz">'
            f'{STYLE}<rect class="bg" width="{W}" height="{height:.0f}" rx="8"/>'
            + "".join(head + keys + body) + "</svg>\n")


def _x_axis(sc: Scale, y0: float, y1: float, label: str) -> list[str]:
    out = []
    for t in sc.ticks:
        x = sc(t)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{y1 + 14}" text-anchor="middle" class="tick">{_fmt(t)}</text>')
    if label:
        out.append(f'<text x="{(sc.a + sc.b) / 2:.1f}" y="{y1 + 30}" text-anchor="middle" class="lab">'
                   f'{escape(label)}</text>')
    return out


def bars(title: str, subtitle: str, panels: list[tuple[str, list[str], list[Series]]], *, dots: bool = False,
         log: bool = False, ref: float | None = None, x_label: str = "") -> str | None:
    """Horizontal grouped bars (or dots) in stacked panels; each panel has its own scale, never a second axis on
    one panel. A panel is (heading, row labels, series with one value per row)."""
    values = [v for _, _, ss in panels for s in ss for v in s.values if v is not None and math.isfinite(v)]
    if log:
        values = [v for v in values if v > 0]
    if not values:
        return None
    body, y = [], PAD_T + 8
    legend = [s.name for s in panels[0][2]]
    for heading, rows, series in panels:
        vals = [v for s in series for v in s.values if v is not None and math.isfinite(v) and (v > 0 or not log)]
        if not vals:
            continue
        if heading:
            body.append(f'<text x="16" y="{y + 12}" class="lab" font-weight="600">{escape(heading)}</text>')
            y += 20
        band = ROW * max(1, len(series)) if not dots else ROW
        top, bottom = y, y + band * len(rows)
        lo, hi = min(vals + ([ref] if ref is not None else [])), max(vals + ([ref] if ref is not None else []))
        sc = Scale(lo, hi, PAD_L, W - PAD_R, log=log, zero=not dots)
        body += _x_axis(sc, top, bottom, x_label if heading == panels[-1][0] else "")
        base = sc.a if log or dots else sc(0)
        if not log and not dots:
            body.append(f'<line x1="{base:.1f}" x2="{base:.1f}" y1="{top}" y2="{bottom}" class="axis"/>')
        if ref is not None:
            body.append(f'<line x1="{sc(ref):.1f}" x2="{sc(ref):.1f}" y1="{top}" y2="{bottom}" class="ref"/>')
        for i, row in enumerate(rows):
            y0 = top + i * band
            body.append(f'<text x="{PAD_L - 8}" y="{y0 + band / 2 + 4:.1f}" text-anchor="end" class="lab">'
                        f'{escape(_clip(row))}</text>')
            for j, s in enumerate(series):
                v = s.values[i]
                if v is None or not math.isfinite(v) or (log and v <= 0):
                    continue
                tip = f"<title>{escape(row)} · {escape(s.name)}: {_fmt(v)}</title>"
                if dots:
                    body.append(f'<circle cx="{sc(v):.1f}" cy="{y0 + band / 2:.1f}" r="{_radius(j, series)}" '
                                f'class="s{j + 1} dot">{tip}</circle>')
                    continue
                h = min(16, ROW - 6)
                yb = y0 + j * ROW + (ROW - h) / 2
                x0, x1 = sorted((base, sc(v)))
                body.append(f'<path d="{_bar_path(x0, x1, yb, h, right=sc(v) >= base)}" class="s{j + 1} bar">'
                            f'{tip}</path>')
        y = bottom + PANEL_GAP
    return _svg(y + 6, title, subtitle, body, legend)


def _bar_path(x0: float, x1: float, y: float, h: float, right: bool) -> str:
    """A bar square at the baseline with a 4px rounded data end."""
    r = min(4, (x1 - x0) / 2, h / 2)
    if right:
        return (f"M{x0:.1f},{y:.1f}H{x1 - r:.1f}Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f}V{y + h - r:.1f}"
                f"Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f}H{x0:.1f}Z")
    return (f"M{x1:.1f},{y:.1f}H{x0 + r:.1f}Q{x0:.1f},{y:.1f} {x0:.1f},{y + r:.1f}V{y + h - r:.1f}"
            f"Q{x0:.1f},{y + h:.1f} {x0 + r:.1f},{y + h:.1f}H{x1:.1f}Z")


def _radius(j: int, series: list) -> int:
    """Earlier series draw larger markers, so a later one lying on top (R and Python agreeing) leaves both seen."""
    return 4 + 4 * (len(series) - 1 - j)


def _clip(s: str, n: int = 29) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def lines(title: str, subtitle: str, series: list[Series], *, x_label: str, y_label: str,
          diagonal: bool = False, markers: bool = True) -> str | None:
    pts = [(x, v) for s in series for x, v in zip(s.x, s.values) if v is not None and math.isfinite(v)]
    if len(pts) < 2:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if diagonal:
        lo, hi = min(xs + ys), max(xs + ys)
        xs, ys = [lo, hi], [lo, hi]
    h = 320
    top, bottom = PAD_T + 12, h - 46
    sx = Scale(min(xs), max(xs), 64, W - PAD_R, zero=False)
    sy = Scale(min(ys), max(ys), bottom, top, zero=False)
    body = _x_axis(sx, top, bottom, x_label)
    for t in sy.ticks:
        y = sy(t)
        if top - 1 <= y <= bottom + 1:
            body.append(f'<line x1="64" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                        f'<text x="58" y="{y + 4:.1f}" text-anchor="end" class="tick">{_fmt(t)}</text>')
    body.append(f'<text x="16" y="{(top + bottom) / 2:.0f}" class="lab" transform="rotate(-90 16 '
                f'{(top + bottom) / 2:.0f})" text-anchor="middle">{escape(y_label)}</text>')
    if diagonal:
        body.append(f'<line x1="{sx(sx.lo):.1f}" y1="{sy(sx.lo):.1f}" x2="{sx(sx.hi):.1f}" y2="{sy(sx.hi):.1f}" '
                    'class="ref"><title>observed = predicted</title></line>')
    for j, s in enumerate(series):
        p = [(sx(x), sy(v), x, v) for x, v in zip(s.x, s.values) if v is not None and math.isfinite(v)]
        if len(p) > 1:
            body.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b, *_ in p)}" '
                        f'class="s{j + 1} line"/>')
        if markers:
            body += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{_radius(j, series)}" class="s{j + 1} dot"><title>'
                     f'{escape(s.name)}: {_fmt(x)}, {_fmt(v)}</title></circle>' for a, b, x, v in p]
    return _svg(h, title, subtitle, body, [s.name for s in series])


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
    return svg and Plot("performance.svg", "Model performance", "validation.performance", svg)


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


def _subgroups(ev: dict) -> list[Plot]:
    res = ev["checks"]["subgroup_performance"].get("results") or {}
    out = []
    for col, c in (res.get("columns") or {}).items():
        if c.get("status") != "executed":
            continue
        for s in res.get("judged_on", []):
            groups = [g for g in c["groups"] if s in g["sets"]]
            stat = "observed_expected" if any("observed_expected" in g["sets"][s].get("python", g["sets"][s].get(
                "r", {})) for g in groups) else res["metric"]
            series = [Series(SIDE_NAMES[k], [g["sets"][s].get(k, {}).get(stat) for g in groups])
                      for k in ("r", "python") if any(k in g["sets"][s] for g in groups)]
            label = "observed / expected" if stat == "observed_expected" else stat.replace("_", " ")
            svg = bars(f"{col} on {s}", "", [("", [g["group"] for g in groups], series)],
                       dots=True, ref=1.0 if stat == "observed_expected" else None, x_label=label)
            if svg:
                out.append(Plot(f"subgroups-{_slug(col)}-{s}.svg", f"{col} on {s}", "validation.subgroup_performance",
                                svg))
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
                                                                            _sensitivity, _assumptions))]
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
