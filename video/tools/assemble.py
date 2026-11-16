"""Assemble the final video from src/timeline.json: scene clips + voice-over + music + burned-in captions."""
import json
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
FF = imageio_ffmpeg.get_ffmpeg_exe()
CLIPS, VO, OUT = ROOT / "assets" / "scenes", ROOT / "assets" / "audio" / "vo", ROOT / "export"
NAME = "R-Pythonizer_demo"
MUSIC = ROOT / "assets" / "audio" / "music.flac"
MUSIC_GAIN, DUCK_RATIO = 1.6, 6


def duration(path):
    err = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def srt_cues(path):
    for block in path.read_text().strip().split("\n\n"):
        lines = block.splitlines()
        a, b = [sum(float(x) * f for x, f in zip(t.replace(",", ".").split(":"), (3600, 60, 1))) for t in lines[1].split(" --> ")]
        yield a, b, " ".join(lines[2:])


def ass_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def srt_time(t):
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def split_cue(a, b, text, width=72):
    """Break a sentence into single-line chunks, timed in proportion to their length."""
    chunks, cur = [], ""
    for w in text.split():
        if cur and len(cur) + len(w) + 1 > width:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    chunks.append(cur)
    if len(chunks) > 1 and len(chunks[-1]) < 18:  # avoid a dangling short tail
        chunks[-2:] = [" ".join(chunks[-2:])]
    total, t = sum(len(c) for c in chunks), a
    for c in chunks:
        d = (b - a) * len(c) / total
        yield t, t + d, c
        t += d


def wrap(text, width=64):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return r"\N".join(lines)


def main():
    timeline = json.loads((ROOT / "src" / "timeline.json").read_text())
    OUT.mkdir(exist_ok=True)
    t, cues, vos = 0.0, [], []
    for item in timeline:
        clip = CLIPS / f"{item['clip']}.mp4"
        if "vo" in item:
            start = t + item["at"]
            vos.append((VO / f"{item['vo']}.mp3", start))
            for a, b, txt in srt_cues(VO / f"{item['vo']}.srt"):
                cues += [(start + x, start + y, line) for x, y, line in split_cue(a, b, txt)]
        t += duration(clip)
    total = t

    (OUT / "concat.txt").write_text("".join(f"file '{CLIPS / (i['clip'] + '.mp4')}'\n" for i in timeline))
    silent = OUT / "video_silent.mp4"
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(OUT / "concat.txt"), "-c", "copy", str(silent)], check=True)

    srt = "".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n\n" for i, (a, b, txt) in enumerate(cues, 1))
    (OUT / f"{NAME}.srt").write_text(srt)
    ass = OUT / "captions.ass"
    ass.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 2\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Cap,Helvetica Neue,34,&H00FFFFFF,&H00FFFFFF,&H1A4B3212,&H1A4B3212,0,0,0,0,100,100,0,0,3,14,0,2,200,200,34,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "".join(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,{wrap(txt, 90)}\n" for a, b, txt in cues)
    )

    args = [FF, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(MUSIC)]
    for path, _ in vos:
        args += ["-i", str(path)]
    parts = [f"[{i + 2}:a]aresample=48000,adelay={int(s * 1000)}:all=1[v{i}]" for i, (_, s) in enumerate(vos)]
    mix = "".join(f"[v{i}]" for i in range(len(vos)))
    fc = ";".join(parts) + (
        f";{mix}amix=inputs={len(vos)}:normalize=0,apad,atrim=0:{total:.3f},asplit[vo][key]"
        f";[1:a]atrim=0:{total:.3f},volume={MUSIC_GAIN},afade=t=out:st={total - 3:.3f}:d=3[mus]"
        f";[mus][key]sidechaincompress=threshold=0.02:ratio={DUCK_RATIO}:attack=20:release=600[duck]"
        ";[vo][duck]amix=inputs=2:normalize=0,loudnorm=I=-16:TP=-1.5[aout]"
        f";[0:v]subtitles='{ass}'[vout]"
    )
    final = OUT / f"{NAME}.mp4"
    subprocess.run(args + ["-filter_complex", fc, "-map", "[vout]", "-map", "[aout]", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(final)], check=True)
    print(f"{final} {total:.1f}s")


main()
