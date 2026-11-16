# Demo video

[`export/R-Pythonizer_demo.mp4`](export/R-Pythonizer_demo.mp4) · 2:51 · 1920×1080, 30 fps, H.264 + AAC, burned-in
captions ([`.srt`](export/R-Pythonizer_demo.srt)). EXL × Trinity Data & AI Hackathon 2026, Challenge 4.

| Part | Scenes | Content |
|---|---|---|
| Story | A1–A8 | R models on a laptop, why move to Python, why R and not SAS, the parity problem, R-Pythonizer, the proof, beyond parity, platform and AI-model benchmark |
| Demo | B1–B5 | Project M4 (GiveMeSomeCredit) on the published platform, every step from data to package |
| Close | B6, TEAM | Tagline, The Fantastic 4 |

The script, with sources for every number on screen, is [`SCRIPT.md`](SCRIPT.md).

## How it is built

Everything is code, so any scene can be changed and the video rebuilt.

| Piece | Source | Tool |
|---|---|---|
| Story scenes | `src/scenes.html`, one `<section>` per scene | Playwright renders each frame, ffmpeg encodes |
| Demo shots | `src/demo.json`: camera moves and highlights over real M4 screenshots in `assets/screens/` | same |
| Voice-over | `> ` lines of `SCRIPT.md`; two voices take turns | `tools/vo.py` (Microsoft neural TTS via edge-tts) |
| Music | `assets/audio/music.flac` | ducked under the voice |
| Team card | `team.json` | |
| Order and voice timing | `src/timeline.json` | `tools/assemble.py` (captions, mix, loudness −16 LUFS) |

Colours follow the EXL deck: navy `#12324B`, orange `#FF5000`, slate `#4E6173`, mist `#D6DFE8`, cream `#FBF4EF`.
Type is Inter and JetBrains Mono.

## Rebuild

Needs uv; Chromium is installed by Playwright.

```sh
cd video/tools
uv sync && uv run playwright install chromium
uv run python vo.py                                   # voice-over from SCRIPT.md (all scenes, or name some: vo.py A3 B4)
uv run python render.py clip INTRO A1 A2 A3 A4 A5 A6 A7 A8 B0 B1a B1b B2 B3 B4 B5a B5b B6 TEAM
uv run python assemble.py                             # export/R-Pythonizer_demo.mp4 and .srt
uv run python render.py still A8 12                   # one frame, for checking a layout
```

New platform screenshots: `./capture.sh M4` then `./capture.sh M4 capture_tall.py`. Both ask for the platform login
in the terminal and keep it only in that process.
