"""Encode the frames AeroSynapse Studio records from its 3D view (frame_00000.ppm ...) into an MP4 video.

usage: python scripts/frames_to_mp4.py <frames folder> [--fps 30] [--keep]
The video is written next to the frames as animation.mp4; the frames are deleted afterwards unless --keep is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--keep", action="store_true", help="keep the frame files")
    a = ap.parse_args(argv)
    d = Path(a.folder)
    frames = sorted(d.glob("frame_*.ppm"))
    if not frames:
        print(f"no frames in {d}", file=sys.stderr)
        return 1
    import imageio.v2 as imageio  # imageio-ffmpeg supplies the encoder

    out = d / "animation.mp4"
    with Image.open(frames[0]) as im0:
        size = im0.size
    w, h = size[0] // 2 * 2, size[1] // 2 * 2  # H.264 needs even sizes
    with imageio.get_writer(out, fps=a.fps, codec="libx264", quality=8, macro_block_size=2) as wr:
        for f in frames:
            with Image.open(f) as src:  # closed at once: Windows cannot delete an open file
                im = src.convert("RGB")
            if im.size != size:
                im = im.resize(size)
            wr.append_data(np.asarray(im)[:h, :w])
    if not a.keep:
        for f in frames:
            f.unlink()
    print(f"wrote {out} ({len(frames)} frames, {len(frames) / a.fps:.1f} s at {a.fps} fps, {w} x {h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
