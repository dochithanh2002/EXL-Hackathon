"""Chart drawing for the report's plots: horizontal bars or dots in stacked panels, and lines, as SVG text.

Drawn with matplotlib's object-oriented API on its own figure and canvas, never pyplot. Output is reproducible
byte for byte: fixed rcParams from matplotlib's defaults, fixed size and hash salt, no date, and text drawn as
paths in the bundled DejaVu Sans, so a viewer needs no fonts or styles. Layout is in pixels: the figure is `W`
wide at 72 dpi, so one point is one SVG unit, and the height grows with the rows."""
from __future__ import annotations

import io
import math
import re
import threading
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

import matplotlib
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.textpath import text_to_path
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

W = 680
DPI = 72
PAD, GAP = 16, 8
ROW, BAR, GROUP_GAP = 22, 15, 0.5             # row and bar heights in pixels; gap in rows
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ROLES = {"R": "#2a78d6", "Python": "#eb6834", "Constant comparator": "#1baf7a"}
PALETTE = list(ROLES.values())
SIZE_TITLE, SIZE_LABEL, SIZE_TICK = 14, 12, 11
MAX_LABEL = 40                                     # characters before a row label is clipped
MAX_LEFT = 0.4                                     # of the width, for row labels

RC = {k: v for k, v in matplotlib.rcParamsDefault.items() if k != "backend"} | {
    "font.family": "DejaVu Sans", "svg.fonttype": "path", "svg.hashsalt": "controlplane", "svg.id": None,
    "text.parse_math": False, "path.simplify": False, "figure.dpi": DPI, "savefig.dpi": DPI,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.grid": False, "axes.axisbelow": True, "axes.unicode_minus": True,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelcolor": MUTED, "ytick.labelcolor": INK2,
    "xtick.labelsize": SIZE_TICK, "ytick.labelsize": SIZE_LABEL, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.minor.size": 0, "ytick.minor.size": 0, "xtick.major.pad": 5, "ytick.major.pad": GAP,
    "legend.frameon": False, "legend.fontsize": SIZE_LABEL, "legend.labelcolor": INK2,
}
_lock = threading.Lock()                           # rcParams are process-global; one figure is drawn at a time


@dataclass
class Series:
    name: str
    values: list[float | None]                    # one per row (bars, dots) or per point (lines)
    x: list[float] = field(default_factory=list)  # lines only


def fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        return f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-")
    return f"{v:,.4g}" if a >= 1000 else f"{v:.4g}"


def _ticks() -> FuncFormatter:
    return FuncFormatter(lambda v, _: fmt(v).replace("-", "−"))


def _ok(v: float | None, log: bool = False) -> bool:
    return v is not None and math.isfinite(v) and (v > 0 or not log)


def _colors(series: list[Series]) -> list[str]:
    """Each role keeps its hue in every plot; other series take the hues no role here uses, in order."""
    free = [c for c in PALETTE if c not in {ROLES.get(s.name) for s in series}]
    return [ROLES.get(s.name) or (free.pop(0) if free else MUTED) for s in series]


def _width(s: str, size: float) -> float:
    return text_to_path.get_text_width_height_descent(s, FontProperties(family="DejaVu Sans", size=size),
                                                      ismath=False)[0]


def _clip(s: str, room: float, size: float = SIZE_LABEL) -> str:
    s = s if len(s) <= MAX_LABEL else s[:MAX_LABEL - 1] + "…"
    while len(s) > 1 and _width(s, size) > room:
        s = s[:-2] + "…"
    return s


def _header(title: str, subtitle: str) -> list[tuple[float, str, dict]]:
    """Title, then subtitle, as (baseline from the top, text, style)."""
    out = [(PAD + SIZE_TITLE, title, {"fontsize": SIZE_TITLE, "fontweight": "bold", "color": INK})]
    if subtitle:
        out.append((out[0][0] + 19, subtitle, {"fontsize": SIZE_LABEL, "color": INK2}))
    return out


def _legend(fig: Figure, names: list[str], colors: list[str], top: float, height: float, kind: str) -> None:
    if len(names) < 2:
        return
    if kind == "bar":
        handles = [Patch(facecolor=c, edgecolor="none") for c in colors]
    else:
        handles = [Line2D([], [], color=c, marker=None if kind == "plain" else "o", markersize=8,
                          markeredgecolor=SURFACE, linestyle="none" if kind == "dot" else "-", linewidth=2)
                   for c in colors]
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(PAD / W, 1 - top / height),
               ncols=len(names), handlelength=1.2 if kind == "bar" else 1.6, handleheight=0.8, handletextpad=0.5,
               columnspacing=1.4, borderaxespad=0, borderpad=0)


def _stride(n: int, dots: bool) -> float:
    """Rows of height `ROW` one category takes: a dot row, or a bar per series and a gap between groups."""
    return 1 if dots or n < 2 else n + GROUP_GAP


def _x_axis(ax, lo: float, hi: float, *, log: bool, zero: bool, label: str) -> None:
    if log:
        a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
        b += a == b
        ax.set_xscale("log")
        ax.set_xlim(10.0 ** a, 10.0 ** b)
        ax.xaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(a, b + 1)]))
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        if zero:
            lo, hi = min(lo, 0), max(hi, 0)
        pad = (hi - lo) * (0.04 if zero else 0.07) or abs(lo) * 0.1 or 1
        ax.set_xlim(lo if zero and lo == 0 else lo - pad, hi if zero and hi == 0 else hi + pad)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(_ticks())
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    if label:
        ax.set_xlabel(label, fontsize=SIZE_LABEL, color=INK2, labelpad=6)


def _plain(ax) -> None:
    for side in ax.spines.values():
        side.set_visible(False)


def _svg(fig: Figure, title: str) -> str:
    buf = io.StringIO()
    FigureCanvasSVG(fig).print_svg(buf, metadata={"Date": None})
    svg = buf.getvalue()
    head = re.search(r"<svg [^>]*>", svg)
    w, h = (re.search(rf'{a}="([\d.]+)pt"', head.group(0)).group(1) for a in ("width", "height"))
    tag = (head.group(0).replace(f'width="{w}pt"', f'width="{float(w):g}"')
           .replace(f'height="{h}pt"', f'height="{float(h):g}"')
           .replace("<svg ", f'<svg role="img" aria-label="{escape(title, {chr(34): "&quot;"})}" ', 1))
    return tag + f"\n <title>{escape(title)}</title>" + svg[head.end():]


def bars(title: str, subtitle: str, panels: list[tuple[str, list[str], list[Series]]], *, dots: bool = False,
         log: bool = False, ref: float | None = None, x_label: str = "") -> str | None:
    """Horizontal grouped bars (or dots) in stacked panels; each panel has its own scale, never a second axis on
    one panel. A panel is (heading, row labels, series with one value per row)."""
    drawn = [(h, rows, ss) for h, rows, ss in panels if any(_ok(v, log) for s in ss for v in s.values)]
    if not drawn:
        return None
    names = [s.name for s in panels[0][2]]
    colors = _colors(panels[0][2])
    room = W * MAX_LEFT - PAD - GAP
    labels = [[_clip(r, room) for r in rows] for _, rows, _ in drawn]
    left = PAD + GAP + max(_width(r, SIZE_LABEL) for ls in labels for r in ls)

    header = _header(title, subtitle)
    y = header[-1][0] + 12
    legend_top = y
    y += 26 if len(names) > 1 else 4
    boxes = []                                    # (heading baseline, axes top, axes height) in pixels
    for i, (heading, rows, series) in enumerate(drawn):
        hb = None
        if heading:
            hb, y = y + SIZE_LABEL, y + SIZE_LABEL + 10
        h = ROW * _stride(len(series), dots) * len(rows)
        boxes.append((hb, y, h))
        y += h + 22 + (20 if x_label and i == len(drawn) - 1 else 0) + (14 if i < len(drawn) - 1
                                                                                       else 0)
    height = math.ceil(y + PAD)

    with _lock, matplotlib.rc_context(RC):
        fig = Figure(figsize=(W / DPI, height / DPI), dpi=DPI)
        for b, text, style in header:
            fig.text(PAD / W, 1 - b / height, text, va="baseline", ha="left", **style)
        _legend(fig, names, colors, legend_top, height, "dot" if dots else "bar")
        for i, ((heading, rows, series), (hb, top, h), labs) in enumerate(zip(drawn, boxes, labels)):
            if heading:
                fig.text(PAD / W, 1 - hb / height, heading, va="baseline", ha="left", fontsize=SIZE_LABEL,
                         fontweight="bold", color=INK)
            ax = fig.add_axes((left / W, 1 - (top + h) / height, (W - left - PAD - 6) / W, h / height))
            _plain(ax)
            vals = [v for s in series for v in s.values if _ok(v, log)] + ([ref] if ref is not None else [])
            _x_axis(ax, min(vals), max(vals), log=log, zero=not dots and not log,
                    label=x_label if i == len(drawn) - 1 else "")
            n = _stride(len(series), dots)
            ax.set_ylim(len(rows) * n, 0)
            ax.set_yticks([k * n + n / 2 for k in range(len(rows))], labs)
            if not log and not dots:
                ax.axvline(0, color=AXIS, linewidth=1, zorder=2)
            if ref is not None:
                ax.axvline(ref, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
            for j, (s, c) in enumerate(zip(series, _colors(series))):
                ks = [k for k, v in enumerate(s.values) if _ok(v, log)]
                vs = [s.values[k] for k in ks]
                if dots:
                    ax.plot(vs, [k + 0.5 for k in ks], linestyle="none", marker="o", color=c,
                            markersize=2 * (4 + 3 * (len(series) - 1 - j)), markeredgecolor=SURFACE,
                            markeredgewidth=1.5, zorder=3 + j, clip_on=False)
                else:
                    ax.barh([k * n + (n - len(series)) / 2 + j + 0.5 for k in ks], vs, height=BAR / ROW, color=c,
                            linewidth=0, left=ax.get_xlim()[0] if log else 0, zorder=3)
        return _svg(fig, title)


def lines(title: str, subtitle: str, series: list[Series], *, x_label: str, y_label: str,
          diagonal: bool = False, markers: bool = True) -> str | None:
    pts = [(x, v) for s in series for x, v in zip(s.x, s.values) if _ok(v) and _ok(x)]
    if len(pts) < 2:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if diagonal:
        xs = ys = [min(xs + ys), max(xs + ys)]
    colors = _colors(series)
    header = _header(title, subtitle)
    legend_top = header[-1][0] + 12
    top = legend_top + (30 if len(series) > 1 else 10)
    height, bottom = 340, 50

    with _lock, matplotlib.rc_context(RC):
        fig = Figure(figsize=(W / DPI, height / DPI), dpi=DPI)
        for b, text, style in header:
            fig.text(PAD / W, 1 - b / height, text, va="baseline", ha="left", **style)
        _legend(fig, [s.name for s in series], colors, legend_top, height, "line" if markers else "plain")
        ax = fig.add_axes((0, 0, 1, 1))
        _plain(ax)
        _x_axis(ax, min(xs), max(xs), log=False, zero=False, label=x_label)
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.07 or abs(lo) * 0.1 or 1
        ax.set_ylim(ax.get_xlim() if diagonal else (lo - pad, hi + pad))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(_ticks())
        ax.tick_params(axis="y", labelsize=SIZE_TICK, labelcolor=MUTED, pad=6)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_ylabel(y_label, fontsize=SIZE_LABEL, color=INK2, labelpad=8)
        ticks = [fmt(t) for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
        left = PAD + SIZE_LABEL + 8 + 6 + max(_width(t, SIZE_TICK) for t in ticks) + 2
        ax.set_position((left / W, bottom / height, (W - left - PAD - 6) / W, (height - top - bottom) / height))
        if diagonal:
            ax.plot(ax.get_xlim(), ax.get_xlim(), color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        for j, (s, c) in enumerate(zip(series, colors)):
            p = [(x, v) for x, v in zip(s.x, s.values) if _ok(v) and _ok(x)]
            if not p:
                continue
            ax.plot([a for a, _ in p], [b for _, b in p], color=c, linewidth=2, solid_joinstyle="round",
                    solid_capstyle="round", marker="o" if markers else None,
                    markersize=2 * (4 + 3 * (len(series) - 1 - j)), markeredgecolor=SURFACE, markeredgewidth=1.5,
                    zorder=3 + j, clip_on=False)
        return _svg(fig, title)
