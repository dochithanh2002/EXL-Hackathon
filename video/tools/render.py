"""Render scenes from src/scenes.html.

  uv run python render.py still A2 3 9.5 20      # PNG stills at given seconds -> export/stills/
  uv run python render.py clip A1 A2 ...         # silent 30 fps clips -> assets/scenes/<id>.mp4
"""
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
FPS = 30


def open_scene(browser, sid):
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    demo = (SRC / "demo.json").exists() and sid in json.loads((SRC / "demo.json").read_text())
    page.goto(f"{(SRC / ('demo.html' if demo else 'scenes.html')).as_uri()}?s={sid}")
    info = page.evaluate("window.sceneReady")
    return page, info["dur"]


def stills(browser, sid, times):
    out = ROOT / "export" / "stills"
    out.mkdir(parents=True, exist_ok=True)
    page, _ = open_scene(browser, sid)
    for t in times:
        page.evaluate(f"window.seek({t})")
        page.screenshot(path=out / f"{sid}_{t:05.1f}.png")
    page.close()


def clip(browser, sid):
    page, dur = open_scene(browser, sid)
    dst = ROOT / "assets" / "scenes" / f"{sid}.mp4"
    ff = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
         "-r", str(FPS), str(dst)],
        stdin=subprocess.PIPE,
    )
    for i in range(round(dur * FPS)):
        page.evaluate(f"window.seek({i / FPS})")
        ff.stdin.write(page.screenshot(type="png"))
    ff.stdin.close()
    ff.wait()
    page.close()
    print("clip", sid, dur)


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--allow-file-access-from-files"])
        if mode == "still":
            stills(browser, args[0], [float(t) for t in args[1:]])
        else:
            for sid in args:
                clip(browser, sid)
        browser.close()


main()
