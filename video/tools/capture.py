"""Capture the M4 project from the published platform: stills of every step plus each page's text.

Credentials come from R2PY_USER / R2PY_PASS in the caller's shell and are never written to disk.
Usage: uv run python capture.py [project-name-substring]   (default "M4")
"""
import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "https://r-pythonizer.iline.work"
OUT = Path(__file__).resolve().parent.parent / "assets" / "screens"
NEEDLE = (sys.argv[1] if len(sys.argv) > 1 else "M4").lower()


def settle(page):
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)


def shoot(page, name):
    settle(page)
    page.screenshot(path=OUT / f"{name}.png")
    page.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
    page.wait_for_timeout(400)
    page.screenshot(path=OUT / f"{name}_full.png", full_page=True)
    (OUT / f"{name}.txt").write_text(page.evaluate("document.body.innerText"))
    print("captured", name)


def main():
    user, password = os.environ.get("R2PY_USER"), os.environ.get("R2PY_PASS")
    if not user or not password:
        sys.exit("Set R2PY_USER and R2PY_PASS first.")
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1920, "height": 1080}, device_scale_factor=2,
            http_credentials={"username": user, "password": password},
        )
        page = ctx.new_page()
        page.goto(f"{BASE}/#/")
        shoot(page, "00_projects")
        projects = page.evaluate("fetch('/api/projects').then(r => r.json())")
        (OUT / "projects.json").write_text(json.dumps(projects, indent=2))
        items = projects if isinstance(projects, list) else projects.get("projects", [])
        match = [p for p in items if NEEDLE in json.dumps(p).lower()]
        if not match:
            sys.exit(f"No project matching {NEEDLE!r}; see assets/screens/projects.json")
        pid = match[0]["id"]
        print("project", pid, match[0].get("name"))
        detail = page.evaluate(f"fetch('/api/projects/{pid}').then(r => r.json())")
        (OUT / "project.json").write_text(json.dumps(detail, indent=2))
        for step in range(1, 8):
            page.goto(f"{BASE}/#/projects/{pid}/{step}")
            shoot(page, f"{step:02d}_step")
            tabs = page.locator("[role=tab]")
            for i in range(tabs.count()):
                tabs.nth(i).click()
                shoot(page, f"{step:02d}_step_tab{i}")
        browser.close()


if __name__ == "__main__":
    main()
