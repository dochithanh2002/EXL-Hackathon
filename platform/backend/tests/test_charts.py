"""The chart drawing behind the report's plots: reproducible, self-contained SVG that fits its labels."""
import math
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

from controlplane.charts import MAX_LEFT, PAD, ROW, SIZE_LABEL, W, Series, _clip, _width, bars, lines

SVG = "{http://www.w3.org/2000/svg}"


def size(svg: str) -> tuple[float, float]:
    root = ET.fromstring(svg)
    w, h = float(root.get("width")), float(root.get("height"))
    assert root.get("viewBox") == f"0 0 {w:g} {h:g}"
    return w, h


def dots(labels: list[str], *series: Series, **kw) -> str | None:
    return bars("t", "", [("", labels, list(series))], dots=True, **kw)


def test_charts_are_reproducible_self_contained_svg():
    a = bars("Performance", "on test", [("auc", ["train", "test"], [Series("R", [0.8, 0.7]),
                                                                    Series("Python", [0.8, 0.7])])])
    curve = [Series("R", [0.1, 0.5, 0.8], [0.1, 0.4, 0.9]), Series("Python", [0.1, 0.5, 0.8], [0.1, 0.4, 0.9])]
    with ThreadPoolExecutor(4) as pool:            # drawn from several threads, as the control plane does
        again = list(pool.map(lambda _: bars("Performance", "on test", [
            ("auc", ["train", "test"], [Series("R", [0.8, 0.7]), Series("Python", [0.8, 0.7])])]), range(4)))
        curves = list(pool.map(lambda _: lines("Calibration", "", curve, x_label="p", y_label="o", diagonal=True),
                                range(4)))
    assert again == [a] * 4 and len(set(curves)) == 1
    assert a.startswith("<svg") and size(a)[0] == W
    root = ET.fromstring(a)
    assert root.get("role") == "img" and root.get("aria-label") == "Performance"
    assert root.find(f"{SVG}title").text == "Performance"
    # text is drawn as paths: no fonts, styles or dates for a viewer or converter to resolve
    assert not list(root.iter(f"{SVG}text"))
    assert "font-family" not in a and "var(--" not in a and "@media" not in a and "dc:date" not in a


def test_too_little_finite_data_draws_nothing():
    assert bars("t", "", [("", ["a", "b"], [Series("R", [None, math.nan])])]) is None
    assert bars("t", "", [("", ["a"], [Series("R", [math.inf])])], dots=True) is None
    assert lines("t", "", [Series("R", [1.0], [1.0])], x_label="x", y_label="y") is None
    assert lines("t", "", [Series("R", [1.0, math.nan, None], [1.0, 2.0, 3.0])], x_label="x", y_label="y") is None
    assert lines("t", "", [Series("R", [1.0, 2.0], [1.0, 2.0])], x_label="x", y_label="y") is not None


def test_a_log_scale_leaves_out_non_positive_values():
    assert dots(["a", "b"], Series("ratio", [0.0, -1.0]), log=True, ref=1.0) is None
    shown = dots(["a", "b", "c"], Series("ratio", [0.0, 0.01, 3.0]), log=True, ref=1.0)
    assert shown == dots(["a", "b", "c"], Series("ratio", [None, 0.01, 3.0]), log=True, ref=1.0)
    assert shown != dots(["a", "b", "c"], Series("ratio", [0.001, 0.01, 3.0]), log=True, ref=1.0)


def test_a_legend_only_with_two_or_more_series():
    one, two = Series("VIF", [3.0, 1.5]), Series("Python", [2.0, 1.0])
    assert "legend" not in bars("t", "", [("", ["a", "b"], [one])], ref=1.0)
    assert "legend" in bars("t", "", [("", ["a", "b"], [one, two])])
    assert "legend" in dots(["a", "b"], one, two)
    assert "legend" not in lines("t", "", [Series("deviance", [5.0, 4.0], [1, 2])], x_label="i", y_label="d")
    assert "legend" in lines("t", "", [Series("R", [5.0, 4.0], [1, 2]), Series("Python", [5.0, 4.0], [1, 2])],
                             x_label="i", y_label="d")


def test_long_labels_fit():
    level = "Postcode district {} – Greater Manchester and surrounding boroughs"
    short, long = ([dots([level.format(i) for i in range(n)], Series("R", [1.0] * n), Series("Python", [1.0] * n))
                    for n in (10, 40)])
    (w, h10), (_, h40) = size(short), size(long)
    assert w == W and h40 - h10 == 30 * ROW          # the height grows by a row per label
    room = W * MAX_LEFT - 3 * PAD
    clipped = _clip(level.format(12), room)
    assert clipped.endswith("…") and _width(clipped, SIZE_LABEL) <= room and clipped.startswith("Postcode district 12")
    assert _clip("x", room) == "x"
    bars_tall = bars("t", "", [("", [f"C(region)[T.{i}]" for i in range(25)], [Series("VIF", [float(i + 1)
                                                                                            for i in range(25)])])])
    assert size(bars_tall)[1] > 25 * ROW
    assert re.search(r'height="\d+"', bars_tall)
