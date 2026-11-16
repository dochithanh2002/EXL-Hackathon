"""Voice-over: one MP3 + one SRT per scene from the `> ` lines of SCRIPT.md, using Microsoft neural TTS."""
import asyncio
import json
import re
import sys
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "audio" / "vo"
RATE = "+8%"
# Two presenters take turns, like a team pitching together.
VOICES = {"A": ("en-US-EmmaNeural", "en-US-BrianNeural")}
DEMO_VOICE, CLOSE_VOICE = "en-US-EmmaNeural", "en-US-BrianNeural"


def voice(sid):
    if sid == "B6":
        return CLOSE_VOICE
    if sid.startswith("B"):
        return DEMO_VOICE
    return VOICES["A"][(int(sid[1]) - 1) % 2]
# Spoken form only; captions keep the written form.
SAY = {
    "R-Pythonizer": "R Pythonizer",
    "SS1/23": "S S one twenty-three",
    "re-runs": "reruns",
    "re-run": "rerun",
    "GPT-6.1 terra": "G P T six point one Terra",
    "scikit-learn": "sci-kit learn",
    '"Looks the same"': "Looks the same",
}


def scenes():
    out, sid = {}, None
    for line in (ROOT / "SCRIPT.md").read_text().splitlines():
        if m := re.match(r"### (\w\d) ", line):
            sid = m.group(1)
        elif line.startswith("> ") and sid:
            out[sid] = (out.get(sid, "") + " " + line[2:]).strip()
    return out


async def render(sid, text):
    spoken = text
    for k, v in SAY.items():
        spoken = spoken.replace(k, v)
    comm = edge_tts.Communicate(spoken, voice(sid), rate=RATE, boundary="SentenceBoundary")
    sub = edge_tts.SubMaker()
    with open(OUT / f"{sid}.mp3", "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                sub.feed(chunk)
    srt = sub.get_srt()
    for k, v in SAY.items():  # restore written form in captions
        srt = srt.replace(v, k.strip('"'))
    (OUT / f"{sid}.srt").write_text(srt)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = scenes()
    (OUT / "text.json").write_text(json.dumps(s, indent=2))
    only = set(sys.argv[1:]) or set(s)
    for k, v in s.items():
        if k not in only:
            continue
        for attempt in range(4):
            try:
                await render(k, v)
                break
            except edge_tts.exceptions.NoAudioReceived:
                await asyncio.sleep(2 + attempt * 3)
        else:
            raise SystemExit(f"TTS failed for {k}")
    print(", ".join(s))


asyncio.run(main())
