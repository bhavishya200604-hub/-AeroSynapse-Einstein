"""Convert animations between formats.

    python -m aerosynapse.thermal.viz.anim.convert docs/animations/cavity.gif --to mp4 webp html
    python -m aerosynapse.thermal.viz.anim.convert docs/animations/*.gif --to pptx --out docs/animations/all.pptx   (one deck)

Inputs: an animated GIF / WebP / APNG, or a folder of PNG frames. Outputs: gif, webp, apng, html, frames (Pillow, no
extra software); mp4 and pptx through Microsoft PowerPoint (Windows; the slide shows the GIF and PowerPoint exports
the video), checked at call time with a clear message when PowerPoint is not available.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def read_frames(src: Path, work: Path) -> tuple[list[Path], int]:
    """Frames of an animation as PNG files in `work`, and its frame rate."""
    from PIL import Image, ImageSequence

    work.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        frames = sorted(src.glob("*.png"))
        return frames, 15
    im = Image.open(src)
    out, durs = [], []
    for k, fr in enumerate(ImageSequence.Iterator(im)):
        fr.load()  # WebP / APNG set a frame's duration only once it is loaded (Pillow >= 13; 12 kept the previous one)
        durs.append(fr.info.get("duration") or 0)
        p = work / f"f{k:05d}.png"
        fr.convert("RGB").save(p)
        out.append(p)
    dur = float(np.median([d for d in durs if d > 0])) if any(d > 0 for d in durs) else 66.0
    return out, max(1, round(1000 / dur))


def _ps(script: str, timeout: int = 900) -> str:
    """Run a PowerShell script (PowerPoint automation); raises with PowerShell's message on failure."""
    if sys.platform != "win32":
        raise RuntimeError("mp4 / pptx conversion needs Microsoft PowerPoint on Windows")
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8-sig") as fh:
        fh.write(script)
        path = fh.name
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    Path(path).unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"PowerPoint automation failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def _bgr(rgb: int) -> int:
    """PowerPoint colours are BGR integers."""
    return ((rgb & 0xFF) << 16) | (rgb & 0xFF00) | (rgb >> 16)


def _deck_script(
    gifs: list[Path],
    titles: list[str],
    durations: list[float],
    pptx: Path | None,
    mp4: Path | None,
    height: int = 720,
    slide: tuple[float, float] = (960, 540),
    captions: list[str] | None = None,
    font: str = "Georgia",
) -> str:
    """PowerShell: one slide per image / GIF (title strip on top, optional caption below, picture fitted between),
    saved as pptx and / or exported as mp4 (each slide shown for its duration). slide = (width, height) in points:
    (960, 540) is 16:9, (540, 540) square, (540, 675) 4:5; the video is `height` pixels high at that aspect."""
    SW, SH = slide
    u = SH / 540  # text sizes scale with the slide
    lines = [
        '$ErrorActionPreference = "Stop"',
        "$pp = New-Object -ComObject PowerPoint.Application",
        "try {",
        "  $p = $pp.Presentations.Add(0)",
        f"  $p.PageSetup.SlideWidth = {SW}; $p.PageSetup.SlideHeight = {SH}",
    ]
    from PIL import Image

    caps = captions or [""] * len(gifs)
    for k, (g, t, d, c) in enumerate(zip(gifs, titles, durations, caps, strict=True), start=1):
        w, h = Image.open(g).size
        top = (24 + 22 * u * (t.count("\n") + 1)) if t else 0
        bot = (18 + 15 * u * (c.count("\n") + 1)) if c else 0
        sc = min(SW / w, (SH - top - bot) / h)
        W, H = w * sc, h * sc
        L, T = (SW - W) / 2, top + (SH - top - bot - H) / 2
        lines += [
            f"  $s = $p.Slides.Add({k}, 12)",
            f"  $null = $s.Shapes.AddPicture('{g.resolve()}', 0, -1, {L:.1f}, {T:.1f}, {W:.1f}, {H:.1f})",
            f"  $s.SlideShowTransition.AdvanceOnTime = -1; $s.SlideShowTransition.AdvanceTime = {max(d, 1.0):.2f}",
        ]
        for text, y, hh, size, color in (
            (t, 8, top - 10, 20 * u, 0x111111),
            (c, SH - bot + 2, bot - 6, 12.5 * u, 0x444444),
        ):
            if not text:
                continue
            # PowerShell double-quoted string: escape backtick, quote and dollar; a line break is `r
            tt = text.replace("`", "``").replace('"', '`"').replace("$", "`$").replace("\n", "`r")
            lines += [
                f"  $b = $s.Shapes.AddTextbox(1, {0.04 * SW:.1f}, {y:.1f}, {0.92 * SW:.1f}, {max(hh, 10):.1f})",
                "  $b.TextFrame.WordWrap = -1",
                f'  $b.TextFrame.TextRange.Text = "{tt}"',
                f"  $b.TextFrame.TextRange.Font.Size = {size:.1f}; $b.TextFrame.TextRange.Font.Name = '{font}'",
                f"  $b.TextFrame.TextRange.Font.Color.RGB = {_bgr(color)}",
            ]
    if pptx:
        lines.append(f"  $p.SaveAs('{pptx.resolve()}', 24)")  # ppSaveAsOpenXMLPresentation
    if mp4:
        lines += [
            f"  $p.CreateVideo('{mp4.resolve()}', -1, 5, {height}, 30, 90)",
            "  $t0 = Get-Date",
            "  while ($p.CreateVideoStatus -eq 1 -or $p.CreateVideoStatus -eq 2) {",
            "    Start-Sleep -Milliseconds 500",
            "    if (((Get-Date) - $t0).TotalSeconds -gt 840) { throw 'video export timed out' }",
            "  }",
            "  if ($p.CreateVideoStatus -ne 3) { throw ('video export failed, status ' + $p.CreateVideoStatus) }",
        ]
    lines += ["  $p.Close()", "} finally { $pp.Quit() }", "'ok'"]
    return "\n".join(lines) + "\n"


def to_mp4(gif: Path, out: Path, duration: float) -> Path:
    if out.exists():
        out.unlink()
    _ps(_deck_script([gif], [""], [duration], None, out))
    if not out.exists():
        raise RuntimeError(f"PowerPoint did not write {out}")
    return out


def to_pptx(gifs: list[Path], out: Path, titles: list[str], durations: list[float]) -> Path:
    if out.exists():
        out.unlink()
    _ps(_deck_script(gifs, titles, durations, out, None))
    return out


def deck_video(gifs: list[Path], out: Path, titles: list[str], durations: list[float]) -> Path:
    """One video of several animations in a row (titled slides)."""
    if out.exists():
        out.unlink()
    _ps(_deck_script(gifs, titles, durations, None, out), timeout=3600)
    return out


def convert(src: Path, to: list[str], out_dir: Path) -> dict[str, Path]:
    from aerosynapse.thermal.viz.anim import core

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem.replace(".apng", "") if src.is_file() else src.name.removesuffix("_frames")
    done: dict[str, Path] = {}
    with tempfile.TemporaryDirectory() as tmp:
        frames, fps = read_frames(src, Path(tmp) / "frames")
        dur = len(frames) / fps
        gif = None
        for f in to:
            if f == "gif":
                done[f] = gif = core.write_gif(frames, out_dir / f"{stem}.gif", fps)
            elif f == "webp":
                done[f] = core.write_webp(frames, out_dir / f"{stem}.webp", fps)
            elif f == "apng":
                done[f] = core.write_apng(frames, out_dir / f"{stem}.apng.png", fps)
            elif f == "html":
                done[f] = core.write_html(frames, out_dir / f"{stem}.html", fps, stem, "", str(src))
            elif f == "frames":
                d = out_dir / f"{stem}_frames"
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir()
                for p in frames:
                    shutil.copy(p, d / p.name)
                done[f] = d
            elif f in ("mp4", "pptx"):
                if gif is None:
                    gif = (
                        src if src.suffix.lower() == ".gif" else core.write_gif(frames, Path(tmp) / f"{stem}.gif", fps)
                    )
                if f == "mp4":
                    done[f] = to_mp4(gif, out_dir / f"{stem}.mp4", dur)
                else:
                    done[f] = to_pptx([gif], out_dir / f"{stem}.pptx", [stem], [dur])
            else:
                raise ValueError(f"unknown target format {f!r}; choose from {core.FORMATS}")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="animated GIF / WebP / APNG files or PNG-frame folders")
    ap.add_argument("--to", nargs="+", required=True, help="target formats: gif webp apng html frames mp4 pptx")
    ap.add_argument(
        "--out", default=None, help="output folder; with --to pptx/mp4 and several inputs: one combined file"
    )
    a = ap.parse_args(argv)
    srcs = [Path(p) for p in a.inputs]
    if a.out and Path(a.out).suffix in (".pptx", ".mp4") and len(srcs) > 1:
        gifs, durs = [], []
        tmp = Path(tempfile.mkdtemp())
        for s in srcs:
            g = s if s.suffix.lower() == ".gif" else convert(s, ["gif"], tmp)["gif"]
            from PIL import Image

            im = Image.open(g)
            gifs.append(g)
            durs.append(getattr(im, "n_frames", 1) * (im.info.get("duration", 66) or 66) / 1000)
        out = Path(a.out)
        titles = [s.stem.replace("_", " ") for s in srcs]
        res = to_pptx(gifs, out, titles, durs) if out.suffix == ".pptx" else deck_video(gifs, out, titles, durs)
        print(f"wrote {res}")
        return 0
    for s in srcs:
        res = convert(s, a.to, Path(a.out) if a.out else s.parent)
        for k, v in res.items():
            print(f"{s.name} -> {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
