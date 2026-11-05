"""Normalise all clips (30 fps, 1280x720, yuv420p) and concatenate them into media/AeroSynapse_Einstein_showcase.mp4.
Also writes media/workbench_walkthrough.mp4 and media/simulations_and_graphs.mp4 (without the UI part).
Audio: a quiet synthesised ambient pad (generated here with ffmpeg sine sources; no third-party audio)."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "media" / "clips"
NORM = ROOT / "media" / "_cache" / "norm"
MEDIA = ROOT / "media"


def sh(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def normalise(src: Path) -> Path:
    NORM.mkdir(parents=True, exist_ok=True)
    out = NORM / (src.stem + ".mp4")
    sh(
        "-i",
        str(src),
        "-vf",
        "fps=30,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=#04060c,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "21",
        "-an",
        str(out),
    )
    return out


def concat(parts: list[Path], out: Path, music: bool = True) -> None:
    lst = NORM / f"{out.stem}.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    tmp = NORM / f"{out.stem}_silent.mp4"
    sh("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(tmp))
    if not music:
        tmp.replace(out)
        return
    dur = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(tmp)],
            capture_output=True,
            text=True,
        ).stdout
    )
    pad = (
        f"sine=f=110:d={dur}[a];sine=f=164.8:d={dur}[b];sine=f=220:d={dur}[c];sine=f=329.6:d={dur}[e];"
        f"[a][b][c][e]amix=inputs=4,volume=0.12,tremolo=f=0.15:d=0.6,afade=t=in:d=3,afade=t=out:st={max(0.0, dur - 4)}:d=4[out]"
    )
    sh(
        "-i",
        str(tmp),
        "-filter_complex",
        pad,
        "-map",
        "0:v",
        "-map",
        "[out]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-shortest",
        str(out),
    )


def main() -> None:
    clips = sorted(p for p in CLIPS.glob("*.mp4"))
    norm = {p.stem: normalise(p) for p in clips}
    ui = MEDIA / "_cache" / "ui_raw.webm"
    if ui.exists():
        norm["41_ui"] = normalise(ui)  # -> media/_cache/norm/ui_raw.mp4
    order = sorted(norm)
    concat([norm[k] for k in order], MEDIA / "AeroSynapse_Einstein_showcase.mp4")
    concat([norm[k] for k in order if not k.startswith(("39", "40", "41"))], MEDIA / "simulations_and_graphs.mp4")
    if ui.exists():
        concat([norm[k] for k in order if k.startswith(("39", "40", "41"))], MEDIA / "workbench_walkthrough.mp4")
    for f in sorted(MEDIA.glob("*.mp4")):
        d = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"{f.name}: {float(d):.1f} s, {f.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
