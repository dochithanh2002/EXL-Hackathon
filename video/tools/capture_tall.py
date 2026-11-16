"""Tall captures of the M4 steps: the viewport grows to the full height of the page's scroll area, so the
camera can pan through everything. Also records where key text sits (CSS px) for highlight boxes.

Credentials come from R2PY_USER / R2PY_PASS; nothing is written to disk but screenshots and layout.
"""
import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "https://r-pythonizer.iline.work"
OUT = Path(__file__).resolve().parent.parent / "assets" / "screens"
PID = json.loads((OUT / "project.json").read_text())["id"]
NEEDLES = {
    1: ["cs-training.csv · 150,000 rows", "1bd46da486a5708c", "givemesomecredit_logistic_notebook.R"],
    2: ["Model Specification v1", "binomial (logit)", "Tolerances", "baseline.R", "Analyst", "Preprocessing"],
    3: ["glm · binomial · logit", "Fitted rows", "Coefficients", "Converged"],
    4: ["25 of 25 passed", "coefficient estimates", "predictions", "baseline.R", "model.py",
        "Business fitness not established", "Evaluation metrics", "Coefficients", "Missing evidence"],
    6: ["Key findings", "Business fitness", "Migration acceptance", "Parity results", "validation-a0d5336048"],
    7: ["Package (.zip)", "verify.py", "Rerun", "Package SHA-256"],
}
FIND = """(needles) => {
  const out = {};
  const all = [...document.querySelectorAll('body *')].filter(e => e.offsetParent !== null);
  for (const n of needles) {
    const hits = all.filter(e => (e.innerText || '').includes(n));
    const leaves = hits.filter(e => ![...e.children].some(c => (c.innerText || '').includes(n)));
    out[n] = leaves.slice(0, 4).map(e => { const r = e.getBoundingClientRect();
      return {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), tag: e.tagName}; });
  }
  return out;
}"""
EXTRA = """() => Math.max(0, ...[...document.querySelectorAll('*')].filter(e => {
  const s = getComputedStyle(e); return /(auto|scroll)/.test(s.overflowY) && e.scrollHeight > e.clientHeight + 4;
}).map(e => e.scrollHeight - e.clientHeight))"""


def main():
    user, password = os.environ.get("R2PY_USER"), os.environ.get("R2PY_PASS")
    if not user or not password:
        sys.exit("Set R2PY_USER and R2PY_PASS first.")
    layout = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=2,
                                  http_credentials={"username": user, "password": password})
        page = ctx.new_page()
        for step in NEEDLES:
            page.set_viewport_size({"width": 1920, "height": 1080})
            page.goto(f"{BASE}/#/projects/{PID}/{step}")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1500)
            h = min(1080 + page.evaluate(EXTRA), 7000)
            page.set_viewport_size({"width": 1920, "height": h})
            page.wait_for_timeout(1200)
            name = f"{step:02d}_tall"
            page.screenshot(path=OUT / f"{name}.png")
            layout[name] = {"h": h, "rects": page.evaluate(FIND, NEEDLES[step])}
            print("captured", name, h)
        browser.close()
    (OUT / "layout.json").write_text(json.dumps(layout, indent=1))


if __name__ == "__main__":
    main()
