"""Draw the AeroSynapse logo: an SVG for documents and the web, a multi-size Windows .ico for the Studio.

The mark is a swept delta planform with a canard, inside a ring broken by three flow lines. It is drawn from
geometry here (no external assets), rasterised with supersampling for the icon sizes.

    python scripts/make_logo.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAVY, CYAN, WHITE = (16, 42, 84), (0, 174, 239), (255, 255, 255)

# unit square coordinates (0..1, y down)
PLANFORM = [(0.50, 0.14), (0.58, 0.40), (0.84, 0.66), (0.84, 0.72), (0.60, 0.70), (0.56, 0.82), (0.66, 0.88), (0.66, 0.91), (0.50, 0.88),
            (0.34, 0.91), (0.34, 0.88), (0.44, 0.82), (0.40, 0.70), (0.16, 0.72), (0.16, 0.66), (0.42, 0.40)]
CANARD = [(0.50, 0.34), (0.62, 0.44), (0.62, 0.47), (0.50, 0.44), (0.38, 0.47), (0.38, 0.44)]


def svg() -> str:
    pts = lambda P, s=512: " ".join(f"{x * s:.1f},{y * s:.1f}" for x, y in P)  # noqa: E731
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <title>AeroSynapse</title>
  <circle cx="256" cy="256" r="244" fill="rgb{NAVY}"/>
  <circle cx="256" cy="256" r="214" fill="none" stroke="rgb{CYAN}" stroke-width="14" stroke-dasharray="390 58"/>
  <path d="M60 300 C 160 270, 200 330, 300 300 S 430 280, 470 300" fill="none" stroke="rgb{CYAN}" stroke-width="7" opacity="0.55"/>
  <path d="M50 340 C 150 312, 210 370, 310 340 S 430 322, 466 340" fill="none" stroke="rgb{CYAN}" stroke-width="5" opacity="0.4"/>
  <polygon points="{pts(PLANFORM)}" fill="rgb{WHITE}"/>
  <polygon points="{pts(CANARD)}" fill="rgb{CYAN}"/>
</svg>
"""


def _inside(px: np.ndarray, py: np.ndarray, poly: list[tuple[float, float]]) -> np.ndarray:
    ins = np.zeros(px.shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        c = ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1)
        ins ^= c
    return ins


def raster(size: int, ss: int = 4) -> np.ndarray:
    n = size * ss
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    r = np.hypot(x - 0.5, y - 0.5)
    img = np.zeros((n, n, 4))
    disc = r <= 0.477
    img[disc, :3], img[disc, 3] = NAVY, 255
    ring = (np.abs(r - 0.418) < 0.014) & (np.degrees(np.arctan2(y - 0.5, x - 0.5)) % 120 > 14)
    img[ring, :3] = CYAN
    w = _inside(x, y, PLANFORM) & disc
    img[w, :3] = WHITE
    c = _inside(x, y, CANARD)
    img[c, :3] = CYAN
    return img.reshape(size, ss, size, ss, 4).mean((1, 3)).round().astype(np.uint8)


def ico(sizes: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)) -> bytes:
    """ICO with 32-bit BMP images (bottom-up BGRA plus an AND mask), readable by windres and Windows."""
    blobs = []
    for s in sizes:
        im = raster(s)
        bgra = im[::-1, :, [2, 1, 0, 3]].tobytes()
        mask_row = ((s + 31) // 32) * 4
        mask = bytes(mask_row * s)
        hdr = struct.pack("<IiiHHIIiiII", 40, s, 2 * s, 1, 32, 0, len(bgra) + len(mask), 0, 0, 0, 0)
        blobs.append((s, hdr + bgra + mask))
    out = struct.pack("<HHH", 0, 1, len(blobs))
    off = 6 + 16 * len(blobs)
    for s, b in blobs:
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(b), off)
        off += len(b)
    return out + b"".join(b for _, b in blobs)


def main() -> None:
    (ROOT / "docs" / "assets").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "assets" / "aerosynapse_logo.svg").write_text(svg(), encoding="utf-8")
    (ROOT / "cpp" / "studio").mkdir(parents=True, exist_ok=True)
    (ROOT / "cpp" / "studio" / "logo.ico").write_bytes(ico())
    try:
        import matplotlib.pyplot as plt

        plt.imsave(ROOT / "docs" / "assets" / "aerosynapse_logo.png", raster(256))
    except Exception:  # matplotlib is optional
        pass
    print("logo written")


if __name__ == "__main__":
    main()
