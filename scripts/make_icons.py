"""Draw AeroSynapse Studio's icon set: one icon per tool and per scheme, in the scheme's colour.

    python scripts/make_icons.py

Reads cpp/studio/tools.inc (schemes with their colours, tools with their icon names) and writes
assets/icons/<name>.png (64 x 64, transparent): a glossy gradient tile with a soft drop shadow and a white glyph.
Glyphs are drawn at 256 px and downsampled, so edges stay clean at any size Studio draws them.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "icons"
FLAT = ROOT / "assets" / "icons_flat"  # glyph-only icons for the ribbon
S = 256
W = (255, 255, 255, 255)
SOFT = (255, 255, 255, 150)
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"
FONT_S = "C:/Windows/Fonts/seguisym.ttf"


def font(size: int, sym: bool = False) -> ImageFont.FreeTypeFont:
    for f in ((FONT_S, FONT_B) if sym else (FONT_B,)):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


class C:
    """Glyph canvas in unit coordinates (0..1, y down) inside a 256 px tile."""

    def __init__(self) -> None:
        self.im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    @staticmethod
    def p(x: float, y: float) -> tuple[float, float]:
        return (28 + x * 200, 28 + y * 200)

    def line(self, pts, w=14, col=W):
        q = [self.p(*a) for a in pts]
        self.d.line(q, fill=col, width=w, joint="curve")
        r = w / 2 - 1
        for x, y in (q[0], q[-1]):
            self.d.ellipse((x - r, y - r, x + r, y + r), fill=col)

    def poly(self, pts, fill=None, outline=W, w=12):
        q = [self.p(*a) for a in pts]
        if fill:
            self.d.polygon(q, fill=fill)
        if outline:
            self.d.line(q + [q[0]], fill=outline, width=w, joint="curve")

    def ell(self, x0, y0, x1, y1, fill=None, outline=W, w=12):
        self.d.ellipse((*self.p(x0, y0), *self.p(x1, y1)), fill=fill, outline=outline, width=w)

    def arc(self, x0, y0, x1, y1, a0, a1, w=12, col=W):
        self.d.arc((*self.p(x0, y0), *self.p(x1, y1)), a0, a1, fill=col, width=w)

    def rect(self, x0, y0, x1, y1, fill=None, outline=W, w=12, r=10):
        self.d.rounded_rectangle((*self.p(x0, y0), *self.p(x1, y1)), radius=r, fill=fill, outline=outline, width=w)

    def text(self, s, x, y, size=90, col=W, sym=False):
        f = font(size, sym)
        self.d.text(self.p(x, y), s, font=f, fill=col, anchor="mm")

    def arrow(self, a, b, w=12, head=0.12, col=W):
        self.line([a, b], w, col)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        for s_ in (-1, 1):
            t = ang + math.pi + s_ * 0.5
            self.line([b, (b[0] + head * math.cos(t), b[1] + head * math.sin(t))], w, col)

    def curve(self, f, t0, t1, n=60, w=12, col=W):
        self.line([f(t0 + (t1 - t0) * i / n) for i in range(n + 1)], w, col)


def airfoil(c: C, x0=0.08, x1=0.92, yc=0.55, t=0.16, w=12, fill=None):
    n = 40
    up, lo = [], []
    for i in range(n + 1):
        u = (1 - math.cos(math.pi * i / n)) / 2
        h = 5 * t * (0.2969 * math.sqrt(u) - 0.126 * u - 0.3516 * u * u + 0.2843 * u**3 - 0.1036 * u**4)
        x = x0 + (x1 - x0) * u
        up.append((x, yc - h - 0.04 * math.sin(math.pi * u)))
        lo.append((x, yc + h - 0.04 * math.sin(math.pi * u)))
    c.poly(up + lo[::-1], fill=fill, w=w)


def cube(c: C, faces=""):
    A, B, Cc, D = (0.5, 0.12), (0.88, 0.32), (0.5, 0.52), (0.12, 0.32)
    E, F, G = (0.12, 0.72), (0.5, 0.92), (0.88, 0.72)
    if "t" in faces:
        c.poly([A, B, Cc, D], fill=SOFT, outline=None)
    if "f" in faces:
        c.poly([D, Cc, F, E], fill=SOFT, outline=None)
    if "s" in faces:
        c.poly([Cc, B, G, F], fill=SOFT, outline=None)
    c.poly([A, B, G, F, E, D])
    c.line([D, Cc, B])
    c.line([Cc, F])


def doc(c: C, label: str, size=58):
    c.poly([(0.2, 0.06), (0.66, 0.06), (0.82, 0.22), (0.82, 0.94), (0.2, 0.94)])
    c.line([(0.66, 0.06), (0.66, 0.22), (0.82, 0.22)], 10)
    c.rect(0.08, 0.5, 0.92, 0.78, fill=(255, 255, 255, 235), outline=None, r=12)
    c.text(label, 0.5, 0.645, size, col=(20, 30, 50, 255))


def jet(c: C, s=1.0, ox=0.0, oy=0.0, fill=W):
    pts = [(0.5, 0.06), (0.56, 0.3), (0.56, 0.44), (0.92, 0.66), (0.92, 0.72), (0.56, 0.62), (0.55, 0.8), (0.68, 0.9), (0.68, 0.94), (0.5, 0.9),
           (0.32, 0.94), (0.32, 0.9), (0.45, 0.8), (0.44, 0.62), (0.08, 0.72), (0.08, 0.66), (0.44, 0.44), (0.44, 0.3)]
    c.poly([(ox + 0.5 + (x - 0.5) * s, oy + 0.5 + (y - 0.5) * s) for x, y in pts], fill=fill, outline=None)


def streamlines(c: C):
    for k, dy in enumerate((-0.3, -0.18, 0.2, 0.32)):
        c.curve(lambda t, dy=dy: (t, 0.55 + dy - (0.1 if dy < 0 else -0.06) * math.exp(-((t - 0.45) / 0.22) ** 2)), 0.0, 1.0, w=7, col=SOFT if k % 2 else W)
    airfoil(c, 0.18, 0.78, 0.56, 0.18, w=10, fill=W)


def network(c: C, w=8):
    layers = [[0.3, 0.7], [0.18, 0.5, 0.82], [0.3, 0.7]]
    xs = [0.16, 0.5, 0.84]
    for i in range(2):
        for a in layers[i]:
            for b in layers[i + 1]:
                c.line([(xs[i], a), (xs[i + 1], b)], w - 3, SOFT)
    for x, ys in zip(xs, layers, strict=True):
        for y in ys:
            c.ell(x - 0.08, y - 0.08, x + 0.08, y + 0.08, fill=W, outline=None)


def number(c: C, n: str):
    c.ell(0.1, 0.1, 0.9, 0.9, w=12)
    c.text(n, 0.5, 0.48, 120)


G: dict = {}


def g(name):
    def deco(f):
        G[name] = f
        return f
    return deco


@g("sketch_revolve")
def _(c):
    c.line([(0.1, 0.78), (0.9, 0.78)], 6, SOFT)
    c.curve(lambda t: (0.12 + 0.76 * t, 0.78 - 0.5 * math.sin(math.pi * t) ** 0.7), 0, 1)
    c.arc(0.02, 0.5, 0.36, 1.06, 200, 340, 10)
    c.arrow((0.31, 0.62), (0.34, 0.56), 10, 0.08)


@g("sketch_extrude")
def _(c):
    pts = [(0.15, 0.85), (0.6, 0.85), (0.75, 0.6), (0.4, 0.5), (0.12, 0.62)]
    c.poly(pts, fill=SOFT)
    c.arrow((0.45, 0.68), (0.45, 0.08), 12, 0.14)


@g("sketch_loft")
def _(c):
    for y, r in ((0.15, 0.18), (0.5, 0.32), (0.85, 0.24)):
        c.ell(0.5 - r, y - 0.07, 0.5 + r, y + 0.07, w=9)
    c.line([(0.32, 0.15), (0.18, 0.5), (0.26, 0.85)], 8)
    c.line([(0.68, 0.15), (0.82, 0.5), (0.74, 0.85)], 8)


@g("fuselage")
def _(c):
    c.curve(lambda t: (0.04 + 0.92 * t, 0.55 - 0.2 * math.sin(math.pi * t) ** 0.6), 0, 1, w=11)
    c.curve(lambda t: (0.04 + 0.92 * t, 0.55 + 0.2 * math.sin(math.pi * t) ** 0.6), 0, 1, w=11)
    c.ell(0.22, 0.27, 0.48, 0.45, fill=SOFT, outline=None)


@g("sphere")
def _(c):
    c.ell(0.1, 0.1, 0.9, 0.9, fill=SOFT)
    c.arc(0.22, 0.22, 0.6, 0.6, 190, 260, 10)


@g("ellipsoid")
def _(c):
    c.ell(0.04, 0.25, 0.96, 0.75, fill=SOFT)
    c.arc(0.04, 0.42, 0.96, 0.58, 0, 180, 7)


@g("box")
def _(c):
    cube(c, "t")


@g("cylinder")
def _(c):
    c.ell(0.18, 0.08, 0.82, 0.3, fill=SOFT)
    c.line([(0.18, 0.19), (0.18, 0.8)])
    c.line([(0.82, 0.19), (0.82, 0.8)])
    c.arc(0.18, 0.69, 0.82, 0.91, 0, 180, 12)


@g("cone")
def _(c):
    c.line([(0.5, 0.08), (0.14, 0.8)])
    c.line([(0.5, 0.08), (0.86, 0.8)])
    c.ell(0.14, 0.68, 0.86, 0.92, fill=SOFT)


@g("torus")
def _(c):
    c.ell(0.04, 0.22, 0.96, 0.78, w=16)
    c.ell(0.3, 0.4, 0.7, 0.6, w=10)


@g("capsule")
def _(c):
    c.rect(0.06, 0.3, 0.94, 0.7, fill=SOFT, r=52)


@g("cut_slot")
def _(c):
    cube(c)
    c.rect(0.38, 0.46, 0.62, 0.84, fill=(255, 255, 255, 230), outline=None, r=4)


@g("cut_hole")
def _(c):
    c.rect(0.08, 0.08, 0.92, 0.92, fill=SOFT, r=14)
    c.ell(0.3, 0.3, 0.7, 0.7, fill=(0, 0, 0, 0), w=12)


@g("cut_cavity")
def _(c):
    c.ell(0.06, 0.12, 0.94, 0.88, fill=SOFT)
    c.ell(0.28, 0.34, 0.72, 0.66, w=8)


@g("revolve")
def _(c):
    c.curve(lambda t: (0.5 - 0.3 * math.sin(math.pi * t) ** 0.8, 0.08 + 0.84 * t), 0, 1)
    c.curve(lambda t: (0.5 + 0.3 * math.sin(math.pi * t) ** 0.8, 0.08 + 0.84 * t), 0, 1)
    c.arc(0.14, 0.44, 0.86, 0.62, 10, 170, 8)


@g("extrude")
def _(c):
    top = [(0.2, 0.22), (0.55, 0.12), (0.8, 0.3), (0.62, 0.45), (0.25, 0.42)]
    bot = [(x, y + 0.42) for x, y in top]
    c.poly(top, fill=SOFT)
    for a, b in zip(top, bot, strict=True):
        if a[0] in (0.2, 0.8, 0.62, 0.25):
            c.line([a, b], 8)
    c.line([bot[0], bot[4], bot[3], bot[2]], 10)


@g("loft")
def _(c):
    c.ell(0.06, 0.4, 0.3, 0.6, w=9)
    c.ell(0.4, 0.28, 0.66, 0.72, w=9)
    c.ell(0.76, 0.36, 0.94, 0.64, w=9)
    c.line([(0.18, 0.4), (0.53, 0.28), (0.85, 0.36)], 7)
    c.line([(0.18, 0.6), (0.53, 0.72), (0.85, 0.64)], 7)


@g("sheet")
def _(c):
    for k in range(4):
        c.curve(lambda t, k=k: (0.06 + 0.88 * t, 0.2 + 0.2 * k + 0.08 * math.sin(2 * math.pi * t + k * 0.6)), 0, 1, w=7)
    for x in (0.06, 0.5, 0.94):
        c.line([(x, 0.2 + 0.08 * math.sin(2 * math.pi * (x - 0.06) / 0.88)), (x, 0.8 + 0.08 * math.sin(2 * math.pi * (x - 0.06) / 0.88 + 1.8))], 6, SOFT)


@g("duct")
def _(c):
    c.curve(lambda t: (0.06 + 0.88 * t, 0.3 + 0.25 * math.sin(math.pi * t)), 0, 1, w=11)
    c.curve(lambda t: (0.06 + 0.88 * t, 0.55 + 0.25 * math.sin(math.pi * t)), 0, 1, w=11)
    c.ell(0.0, 0.28, 0.12, 0.57, w=8)


@g("implicit")
def _(c):
    c.text("f(x)", 0.5, 0.38, 92)
    c.text("\u2264 0", 0.5, 0.78, 72, sym=True)


@g("superellipsoid")
def _(c):
    n = 4
    c.poly([(0.5 + 0.42 * math.copysign(abs(math.cos(t)) ** (2 / n), math.cos(t)), 0.5 + 0.34 * math.copysign(abs(math.sin(t)) ** (2 / n), math.sin(t)))
            for t in [2 * math.pi * i / 80 for i in range(80)]], fill=SOFT)


@g("sweep")
def _(c):
    c.curve(lambda t: (0.08 + 0.84 * t, 0.5 + 0.3 * math.sin(6 * math.pi * t)), 0, 1, n=140, w=10)
    c.curve(lambda t: (0.08 + 0.84 * t + 0.03, 0.5 + 0.3 * math.sin(6 * math.pi * t)), 0, 1, n=140, w=4, col=SOFT)


@g("lattice")
def _(c):
    for k in range(3):
        c.curve(lambda t, k=k: (0.08 + 0.84 * t, 0.2 + 0.3 * k + 0.1 * math.sin(2 * math.pi * 2 * t)), 0, 1, w=8)
        c.curve(lambda t, k=k: (0.2 + 0.3 * k + 0.1 * math.sin(2 * math.pi * 2 * t), 0.08 + 0.84 * t), 0, 1, w=8, col=SOFT)


@g("halfspace")
def _(c):
    c.poly([(0.06, 0.66), (0.62, 0.5), (0.94, 0.66), (0.38, 0.84)], fill=SOFT)
    c.arrow((0.5, 0.66), (0.5, 0.12), 11)


@g("wing")
def _(c):
    c.poly([(0.08, 0.86), (0.08, 0.44), (0.86, 0.08), (0.94, 0.16), (0.66, 0.86)], fill=SOFT)


@g("canard")
def _(c):
    c.line([(0.06, 0.5), (0.94, 0.5)], 14)
    c.poly([(0.3, 0.5), (0.52, 0.18), (0.6, 0.5)], fill=W, outline=None)
    c.poly([(0.3, 0.5), (0.52, 0.82), (0.6, 0.5)], fill=W, outline=None)


@g("tail")
def _(c):
    c.line([(0.06, 0.5), (0.94, 0.5)], 14)
    c.poly([(0.58, 0.5), (0.8, 0.22), (0.9, 0.22), (0.92, 0.5)], fill=SOFT)
    c.poly([(0.58, 0.5), (0.8, 0.78), (0.9, 0.78), (0.92, 0.5)], fill=SOFT)


@g("fin")
def _(c):
    c.line([(0.06, 0.88), (0.94, 0.88)], 12)
    c.poly([(0.2, 0.88), (0.6, 0.14), (0.8, 0.14), (0.86, 0.88)], fill=SOFT)


@g("strake")
def _(c):
    c.line([(0.06, 0.7), (0.94, 0.7)], 12)
    c.poly([(0.08, 0.7), (0.92, 0.7), (0.92, 0.26)], fill=SOFT)


@g("blade")
def _(c):
    c.poly([(0.06, 0.8), (0.94, 0.8), (0.8, 0.64), (0.2, 0.64)], fill=SOFT, w=9)
    c.poly([(0.42, 0.72), (0.5, 0.14), (0.62, 0.14), (0.6, 0.72)], fill=W)


@g("airfoil")
def _(c):
    airfoil(c, 0.06, 0.94, 0.55, 0.22, fill=SOFT)
    c.line([(0.06, 0.55), (0.94, 0.55)], 4, SOFT)


@g("mirror")
def _(c):
    c.line([(0.5, 0.04), (0.5, 0.96)], 6)
    c.poly([(0.42, 0.2), (0.12, 0.5), (0.42, 0.8)], fill=W)
    c.poly([(0.58, 0.2), (0.88, 0.5), (0.58, 0.8)], fill=None, w=9)


@g("pattern")
def _(c):
    for k in range(3):
        c.rect(0.06 + 0.28 * k, 0.6 - 0.24 * k, 0.32 + 0.28 * k, 0.86 - 0.24 * k, fill=SOFT if k else W, r=8)


@g("shell")
def _(c):
    c.ell(0.08, 0.08, 0.92, 0.92, w=26)
    c.line([(0.5, 0.5), (0.92, 0.5)], 8, SOFT)


@g("blend")
def _(c):
    c.line([(0.12, 0.08), (0.12, 0.6)], 14)
    c.arc(0.12, 0.28, 0.76, 0.92, 90, 180, 14)
    c.line([(0.44, 0.92), (0.92, 0.92)], 14)


@g("twist")
def _(c):
    c.curve(lambda t: (0.08 + 0.84 * t, 0.5 - 0.3 * math.cos(math.pi * t)), 0, 1, w=12)
    c.curve(lambda t: (0.08 + 0.84 * t, 0.5 + 0.3 * math.cos(math.pi * t)), 0, 1, w=12, col=SOFT)


@g("taper")
def _(c):
    c.poly([(0.06, 0.3), (0.94, 0.12), (0.94, 0.88), (0.06, 0.7)], fill=SOFT)


@g("bend")
def _(c):
    c.arc(0.1, 0.3, 0.9, 1.3, 200, 340, 34)


@g("shear")
def _(c):
    c.poly([(0.06, 0.82), (0.36, 0.18), (0.94, 0.18), (0.64, 0.82)], fill=SOFT)


@g("stretch")
def _(c):
    c.rect(0.32, 0.32, 0.68, 0.68, fill=SOFT, r=8)
    c.arrow((0.3, 0.5), (0.04, 0.5), 10, 0.1)
    c.arrow((0.7, 0.5), (0.96, 0.5), 10, 0.1)


@g("pivot")
def _(c):
    c.line([(0.2, 0.8), (0.86, 0.3)], 18)
    c.ell(0.12, 0.72, 0.28, 0.88, fill=W, outline=None)
    c.arc(0.0, 0.3, 0.8, 1.3, 270, 320, 8)


@g("infill")
def _(c):
    c.ell(0.06, 0.06, 0.94, 0.94, w=12)
    for k in range(3):
        c.curve(lambda t, k=k: (0.2 + 0.6 * t, 0.32 + 0.18 * k + 0.07 * math.sin(4 * math.pi * t)), 0, 1, w=6, col=SOFT)


@g("clear")
def _(c):
    c.poly([(0.1, 0.62), (0.5, 0.22), (0.88, 0.6), (0.48, 1.0)], fill=SOFT)
    c.line([(0.3, 0.42), (0.68, 0.8)], 8)


@g("duplicate")
def _(c):
    c.rect(0.08, 0.08, 0.6, 0.6, fill=SOFT, r=10)
    c.rect(0.4, 0.4, 0.92, 0.92, fill=None, r=10)


@g("booleans")
def _(c):
    c.ell(0.06, 0.2, 0.62, 0.76, fill=SOFT)
    c.ell(0.38, 0.24, 0.94, 0.8, w=12)


@g("delete")
def _(c):
    c.line([(0.14, 0.2), (0.86, 0.2)], 12)
    c.rect(0.38, 0.06, 0.62, 0.2, r=4, w=8)
    c.poly([(0.2, 0.28), (0.8, 0.28), (0.72, 0.94), (0.28, 0.94)], fill=SOFT)


@g("undo")
def _(c):
    c.arc(0.16, 0.2, 0.86, 0.9, 200, 450, 14)
    c.arrow((0.3, 0.32), (0.14, 0.42), 12, 0.0)
    c.poly([(0.06, 0.4), (0.3, 0.22), (0.3, 0.52)], fill=W, outline=None)


@g("redo")
def _(c):
    c.arc(0.14, 0.2, 0.84, 0.9, 90, 340, 14)
    c.poly([(0.94, 0.4), (0.7, 0.22), (0.7, 0.52)], fill=W, outline=None)


@g("view_iso")
def _(c):
    cube(c, "tfs")


@g("view_front")
def _(c):
    cube(c, "f")


@g("view_side")
def _(c):
    cube(c, "s")


@g("view_top")
def _(c):
    cube(c, "t")


@g("view_fit")
def _(c):
    for x0, y0, dx, dy in ((0.06, 0.06, 1, 1), (0.94, 0.06, -1, 1), (0.06, 0.94, 1, -1), (0.94, 0.94, -1, -1)):
        c.line([(x0, y0 + 0.24 * dy), (x0, y0), (x0 + 0.24 * dx, y0)], 12)
    jet(c, 0.5)


@g("view_wire")
def _(c):
    cube(c)
    c.line([(0.12, 0.32), (0.5, 0.92)], 5, SOFT)
    c.line([(0.5, 0.12), (0.88, 0.72)], 5, SOFT)


def cutplane(letter):
    def f(c):
        c.poly([(0.06, 0.62), (0.52, 0.42), (0.94, 0.62), (0.48, 0.86)], fill=SOFT)
        c.text(letter, 0.5, 0.26, 100)
    return f


for _L in "xyz":
    G[f"cut_{_L}"] = cutplane(_L.upper())


@g("resolution")
def _(c):
    for k in range(5):
        c.line([(0.08 + 0.21 * k, 0.08), (0.08 + 0.21 * k, 0.92)], 6)
        c.line([(0.08, 0.08 + 0.21 * k), (0.92, 0.08 + 0.21 * k)], 6)


@g("measure")
def _(c):
    c.poly([(0.22, 0.32), (0.78, 0.32), (0.92, 0.92), (0.08, 0.92)], fill=SOFT)
    c.ell(0.38, 0.06, 0.62, 0.3, w=9)
    c.text("kg", 0.5, 0.64, 70)


@g("check")
def _(c):
    c.ell(0.06, 0.06, 0.94, 0.94, w=12)
    c.line([(0.26, 0.52), (0.44, 0.7), (0.76, 0.32)], 16)


@g("quick_aero")
def _(c):
    c.poly([(0.1, 0.62), (0.9, 0.62), (0.76, 0.42), (0.24, 0.42)], fill=SOFT, w=9)
    for x in (0.24, 0.5, 0.76):
        c.line([(x, 0.42), (x, 0.96)], 5)
    c.line([(0.24, 0.42), (0.76, 0.42)], 9)
    c.text("\u0393", 0.5, 0.18, 70, sym=True)


def cfd(label):
    def f(c):
        streamlines(c)
        c.rect(0.0, 0.0, 0.62, 0.26, fill=(255, 255, 255, 235), outline=None, r=12)
        c.text(label, 0.31, 0.13, 56, col=(40, 20, 60, 255))
    return f


for _k, _lab in (("cfd_cruise", "M.85"), ("cfd_transonic", "M.95"), ("cfd_super", "M1.6"), ("cfd_alpha", "25\u00b0")):
    G[_k] = cfd(_lab)


@g("flow_field")
def _(c):
    cols = [(34, 211, 238, 255), (163, 230, 53, 255), (251, 191, 36, 255), (244, 114, 182, 255)]
    for k, col in enumerate(cols):
        c.curve(lambda t, k=k: (0.06 + 0.88 * t, 0.2 + 0.2 * k + 0.08 * math.sin(2 * math.pi * t)), 0, 1, w=22, col=col)
    airfoil(c, 0.2, 0.75, 0.5, 0.2, w=8, fill=W)


@g("particles")
def _(c):
    for _k, (x, y) in enumerate(((0.7, 0.2), (0.86, 0.46), (0.62, 0.62), (0.8, 0.84), (0.42, 0.36))):
        c.line([(x - 0.4, y + 0.04), (x, y)], 5, SOFT)
        c.ell(x - 0.07, y - 0.07, x + 0.07, y + 0.07, fill=W, outline=None)


@g("analytic")
def _(c):
    c.text("\u222b", 0.32, 0.46, 170, sym=True)
    c.text("dx", 0.68, 0.6, 76)


@g("pinn")
def _(c):
    network(c)


@g("structure")
def _(c):
    pts = [(0.06, 0.8), (0.32, 0.3), (0.58, 0.8), (0.84, 0.3)]
    c.line(pts, 10)
    c.line([(0.06, 0.8), (0.94, 0.8)], 12)
    c.line([(0.32, 0.3), (0.94, 0.3)], 12)
    c.line([(0.84, 0.3), (0.94, 0.8)], 8)


def colourbar(label, cols):
    def f(c):
        for k, col in enumerate(cols):
            c.rect(0.06, 0.08 + 0.21 * k, 0.32, 0.29 + 0.21 * k, fill=col, outline=None, r=4)
        c.text(label, 0.66, 0.5, 92)
    return f


G["colour_cp"] = colourbar("Cp", [(239, 68, 68, 255), (250, 204, 21, 255), (74, 222, 128, 255), (59, 130, 246, 255)])
G["colour_cf"] = colourbar("Cf", [(244, 114, 182, 255), (167, 139, 250, 255), (96, 165, 250, 255), (34, 211, 238, 255)])


@g("colour_t")
def _(c):
    c.rect(0.4, 0.06, 0.6, 0.66, r=40, w=11)
    c.ell(0.3, 0.6, 0.7, 0.96, fill=(248, 113, 113, 255), outline=W, w=10)
    c.line([(0.5, 0.3), (0.5, 0.7)], 12, (248, 113, 113, 255))


@g("results")
def _(c):
    for k, h in enumerate((0.3, 0.55, 0.42, 0.75)):
        c.rect(0.08 + 0.22 * k, 0.92 - h, 0.24 + 0.22 * k, 0.92, fill=SOFT, outline=None, r=4)
    c.line([(0.12, 0.6), (0.38, 0.34), (0.6, 0.44), (0.86, 0.12)], 10)


@g("pinn_lab")
def _(c):
    c.poly([(0.38, 0.06), (0.62, 0.06), (0.62, 0.36), (0.92, 0.9), (0.08, 0.9), (0.38, 0.36)], fill=SOFT)
    c.ell(0.3, 0.6, 0.44, 0.74, fill=W, outline=None)
    c.ell(0.56, 0.66, 0.68, 0.78, fill=W, outline=None)


@g("pinn_2d")
def _(c):
    for i in range(4):
        for j in range(4):
            a = int(80 + 170 * (i + j) / 6)
            c.rect(0.06 + 0.22 * i, 0.06 + 0.22 * j, 0.26 + 0.22 * i, 0.26 + 0.22 * j, fill=(255, 255, 255, a), outline=None, r=4)


@g("pinn_inverse")
def _(c):
    network(c, 7)
    c.ell(0.44, 0.44, 0.84, 0.84, w=11)
    c.line([(0.8, 0.8), (0.96, 0.96)], 14)


@g("verify")
def _(c):
    c.poly([(0.5, 0.04), (0.9, 0.18), (0.84, 0.6), (0.5, 0.96), (0.16, 0.6), (0.1, 0.18)], fill=SOFT)
    c.line([(0.3, 0.5), (0.46, 0.66), (0.72, 0.34)], 14)


@g("opt_centre")
def _(c):
    for r, w in ((0.44, 10), (0.28, 10), (0.1, 0)):
        c.ell(0.5 - r, 0.5 - r, 0.5 + r, 0.5 + r, fill=W if w == 0 else None, w=w or 1)
    c.arrow((0.96, 0.04), (0.56, 0.44), 9, 0.1)


@g("opt_airfoil")
def _(c):
    airfoil(c, 0.06, 0.94, 0.6, 0.2, fill=SOFT)
    c.arrow((0.4, 0.4), (0.4, 0.06), 9, 0.08)
    c.arrow((0.62, 0.46), (0.62, 0.16), 9, 0.08)


@g("opt_xfoil")
def _(c):
    c.text("XF", 0.5, 0.42, 110)
    airfoil(c, 0.1, 0.9, 0.82, 0.16, w=8)


@g("opt_topology")
def _(c):
    c.ell(0.04, 0.16, 0.96, 0.84, fill=SOFT)
    for x, y, r in ((0.3, 0.42, 0.09), (0.55, 0.58, 0.11), (0.74, 0.38, 0.07)):
        c.ell(x - r, y - r, x + r, y + r, fill=(0, 0, 0, 0), w=8)


@g("opt_fighter")
def _(c):
    jet(c, 1.0)


@g("opt_aircraft")
def _(c):
    c.line([(0.5, 0.04), (0.5, 0.96)], 18)
    c.poly([(0.5, 0.36), (0.94, 0.58), (0.94, 0.64), (0.5, 0.52), (0.06, 0.64), (0.06, 0.58)], fill=W, outline=None)
    c.poly([(0.5, 0.82), (0.7, 0.92), (0.3, 0.92)], fill=W, outline=None)


@g("opt_surrogate")
def _(c):
    network(c, 7)
    c.poly([(0.62, 0.0), (0.36, 0.52), (0.52, 0.52), (0.4, 1.0), (0.72, 0.42), (0.56, 0.42)], fill=(251, 191, 36, 255), outline=None)


@g("opt_structure")
def _(c):
    c.rect(0.08, 0.08, 0.92, 0.24, fill=W, outline=None, r=4)
    c.rect(0.08, 0.76, 0.92, 0.92, fill=W, outline=None, r=4)
    c.rect(0.42, 0.2, 0.58, 0.8, fill=SOFT, outline=None, r=2)


@g("wf_window")
def _(c):
    for x, y in ((0.06, 0.08), (0.06, 0.66), (0.62, 0.37)):
        c.rect(x, y, x + 0.32, y + 0.26, fill=SOFT, r=8)
    c.line([(0.38, 0.21), (0.5, 0.21), (0.5, 0.5), (0.62, 0.5)], 8)
    c.line([(0.38, 0.79), (0.5, 0.79), (0.5, 0.5)], 8)


@g("wf_new")
def _(c):
    c.poly([(0.06, 0.24), (0.36, 0.24), (0.44, 0.34), (0.94, 0.34), (0.94, 0.9), (0.06, 0.9)], fill=SOFT)
    c.line([(0.5, 0.46), (0.5, 0.8)], 12)
    c.line([(0.33, 0.63), (0.67, 0.63)], 12)


@g("wf_open")
def _(c):
    c.poly([(0.06, 0.2), (0.36, 0.2), (0.44, 0.3), (0.84, 0.3), (0.84, 0.44), (0.06, 0.44)], fill=SOFT)
    c.poly([(0.06, 0.9), (0.2, 0.44), (0.96, 0.44), (0.82, 0.9)], fill=W, outline=None)


@g("wf_run")
def _(c):
    c.poly([(0.08, 0.14), (0.5, 0.5), (0.08, 0.86)], fill=W, outline=None)
    c.poly([(0.46, 0.14), (0.9, 0.5), (0.46, 0.86)], fill=SOFT, outline=None)


for _i in range(1, 9):
    G[f"stage{_i}"] = (lambda n: (lambda c: number(c, n)))(str(_i))


@g("save")
def _(c):
    c.rect(0.08, 0.08, 0.92, 0.92, fill=SOFT, r=10)
    c.rect(0.24, 0.08, 0.72, 0.36, fill=W, outline=None, r=4)
    c.rect(0.22, 0.56, 0.78, 0.92, fill=None, r=4, w=9)


@g("save_as")
def _(c):
    G["save"](c)
    c.line([(0.6, 0.98), (0.98, 0.6)], 14, (251, 191, 36, 255))


for _k, _lab in (("stl", "STL"), ("obj", "OBJ"), ("svg", "SVG"), ("report_pdf", "PDF"), ("report_docx", "DOC"), ("report_txt", "TXT"), ("report_html", "WEB")):
    G[_k] = (lambda lab: (lambda c: doc(c, lab)))(_lab)


@g("package")
def _(c):
    c.poly([(0.5, 0.06), (0.92, 0.26), (0.92, 0.74), (0.5, 0.94), (0.08, 0.74), (0.08, 0.26)], fill=SOFT)
    c.line([(0.08, 0.26), (0.5, 0.46), (0.92, 0.26)], 9)
    c.line([(0.5, 0.46), (0.5, 0.94)], 9)
    c.line([(0.29, 0.16), (0.71, 0.36)], 9)


# scheme icons
@g("sch_sketch")
def _(c):
    c.poly([(0.14, 0.86), (0.2, 0.62), (0.72, 0.1), (0.9, 0.28), (0.38, 0.8)], fill=SOFT)
    c.line([(0.2, 0.62), (0.38, 0.8)], 8)
    c.curve(lambda t: (0.06 + 0.5 * t, 0.94 - 0.04 * math.sin(4 * math.pi * t)), 0, 1, w=6)


@g("sch_curves")
def _(c):
    c.curve(lambda t: ((1 - t) ** 3 * 0.06 + 3 * (1 - t) ** 2 * t * 0.2 + 3 * (1 - t) * t * t * 0.8 + t**3 * 0.94,
                       (1 - t) ** 3 * 0.86 + 3 * (1 - t) ** 2 * t * 0.06 + 3 * (1 - t) * t * t * 0.94 + t**3 * 0.14), 0, 1, w=12)
    for x, y in ((0.2, 0.06), (0.8, 0.94)):
        c.rect(x - 0.07, y - 0.07, x + 0.07, y + 0.07, fill=W, outline=None, r=3)
    c.line([(0.06, 0.86), (0.2, 0.06)], 4, SOFT)
    c.line([(0.94, 0.14), (0.8, 0.94)], 4, SOFT)


G["sch_aero"] = G["opt_fighter"]


@g("sch_xform")
def _(c):
    for a, b in (((0.5, 0.5), (0.5, 0.04)), ((0.5, 0.5), (0.5, 0.96)), ((0.5, 0.5), (0.04, 0.5)), ((0.5, 0.5), (0.96, 0.5))):
        c.arrow(a, b, 11, 0.12)


@g("sch_view")
def _(c):
    c.poly([(0.04, 0.5)] + [(0.04 + 0.92 * t, 0.5 - 0.32 * math.sin(math.pi * t)) for t in [i / 30 for i in range(1, 31)]] +
           [(0.04 + 0.92 * t, 0.5 + 0.32 * math.sin(math.pi * t)) for t in [i / 30 for i in range(29, 0, -1)]], fill=SOFT)
    c.ell(0.34, 0.34, 0.66, 0.66, fill=W, outline=None)


G["sch_sim"] = G["flow_field"]
G["sch_opt"] = G["opt_centre"]
G["sch_wf"] = G["wf_window"]
G["sch_export"] = G["package"]



# ---------------------------------------------------------------- workbenches and part design
def plane_glyph(letter_u, letter_v, fill_face):
    def f(c):
        c.poly([(0.06, 0.62), (0.5, 0.36), (0.94, 0.62), (0.5, 0.88)] if fill_face == "h" else
               [(0.2, 0.1), (0.8, 0.22), (0.8, 0.9), (0.2, 0.78)] if fill_face == "v" else [(0.12, 0.2), (0.62, 0.1), (0.62, 0.8), (0.12, 0.9)], fill=SOFT)
        c.text(letter_u + letter_v, 0.5, 0.2 if fill_face == "h" else 0.5, 64)
    return f


G["plane_xy"] = plane_glyph("X", "Y", "h")
G["plane_xz"] = plane_glyph("X", "Z", "v")
G["plane_yz"] = plane_glyph("Y", "Z", "f")


@g("sketch_line")
def _(c):
    c.line([(0.08, 0.84), (0.3, 0.3), (0.62, 0.62), (0.92, 0.14)], 10)
    for x, y in ((0.08, 0.84), (0.3, 0.3), (0.62, 0.62), (0.92, 0.14)):
        c.ell(x - 0.06, y - 0.06, x + 0.06, y + 0.06, fill=W, outline=None)


@g("sketch_rect")
def _(c):
    c.rect(0.1, 0.22, 0.9, 0.78, r=4, w=11)
    for x, y in ((0.1, 0.22), (0.9, 0.78)):
        c.ell(x - 0.07, y - 0.07, x + 0.07, y + 0.07, fill=W, outline=None)


@g("sketch_circle")
def _(c):
    c.ell(0.1, 0.1, 0.9, 0.9, w=11)
    c.line([(0.5, 0.5), (0.9, 0.5)], 6, SOFT)
    c.ell(0.44, 0.44, 0.56, 0.56, fill=W, outline=None)


@g("sketch_slot")
def _(c):
    c.rect(0.06, 0.32, 0.94, 0.68, r=46, w=11)
    c.line([(0.24, 0.5), (0.76, 0.5)], 5, SOFT)


@g("pad")
def _(c):
    c.poly([(0.1, 0.66), (0.5, 0.48), (0.9, 0.66), (0.5, 0.84)], fill=SOFT)
    c.poly([(0.22, 0.4), (0.5, 0.27), (0.78, 0.4), (0.5, 0.53)], fill=W, outline=None)
    c.line([(0.22, 0.4), (0.22, 0.6)], 8)
    c.line([(0.78, 0.4), (0.78, 0.6)], 8)
    c.arrow((0.5, 0.25), (0.5, 0.02), 9, 0.08)


@g("pocket")
def _(c):
    c.poly([(0.06, 0.5), (0.5, 0.28), (0.94, 0.5), (0.5, 0.72)], fill=SOFT)
    c.poly([(0.3, 0.5), (0.5, 0.4), (0.7, 0.5), (0.5, 0.6)], fill=(0, 0, 0, 120), outline=W, w=7)
    c.arrow((0.5, 0.08), (0.5, 0.38), 9, 0.08)


@g("shaft")
def _(c):
    c.line([(0.04, 0.5), (0.96, 0.5)], 5, SOFT)
    c.curve(lambda t: (0.1 + 0.8 * t, 0.5 - 0.3 * math.sin(math.pi * t) ** 0.6), 0, 1)
    c.curve(lambda t: (0.1 + 0.8 * t, 0.5 + 0.3 * math.sin(math.pi * t) ** 0.6), 0, 1, col=SOFT)
    c.arc(0.78, 0.2, 1.0, 0.8, 300, 60, 8)


@g("groove")
def _(c):
    c.rect(0.08, 0.26, 0.92, 0.74, fill=SOFT, r=10)
    c.rect(0.4, 0.26, 0.6, 0.74, fill=(0, 0, 0, 110), outline=W, r=2, w=7)
    c.line([(0.02, 0.5), (0.98, 0.5)], 4, SOFT)


@g("rib")
def _(c):
    c.curve(lambda t: (0.1 + 0.8 * t, 0.7 - 0.45 * math.sin(math.pi * t)), 0, 1, w=22, col=SOFT)
    c.curve(lambda t: (0.1 + 0.8 * t, 0.7 - 0.45 * math.sin(math.pi * t)), 0, 1, w=6)
    c.ell(0.02, 0.58, 0.18, 0.82, w=8)


@g("chamfer")
def _(c):
    c.poly([(0.1, 0.9), (0.1, 0.38), (0.38, 0.1), (0.9, 0.1), (0.9, 0.9)], fill=SOFT)
    c.line([(0.1, 0.38), (0.38, 0.1)], 14, (251, 191, 36, 255))


@g("draft")
def _(c):
    c.poly([(0.08, 0.9), (0.26, 0.12), (0.74, 0.12), (0.92, 0.9)], fill=SOFT)
    c.line([(0.26, 0.12), (0.26, 0.9)], 5, (251, 191, 36, 255))
    c.text("5\u00b0", 0.5, 0.56, 64)


G["zebra"] = None


def _zebra(c):
    for k in range(5):
        c.curve(lambda t, k=k: (0.06 + 0.88 * t, 0.12 + 0.19 * k + 0.08 * math.sin(math.pi * t)), 0, 1, w=14, col=W if k % 2 == 0 else (30, 30, 40, 255))


G["zebra"] = _zebra


@g("curvature")
def _(c):
    c.curve(lambda t: (0.06 + 0.88 * t, 0.75 - 0.4 * math.sin(math.pi * t)), 0, 1, w=10)
    for k in range(1, 8):
        t = k / 8
        x, y = 0.06 + 0.88 * t, 0.75 - 0.4 * math.sin(math.pi * t)
        L = 0.06 + 0.16 * math.sin(math.pi * t)
        c.line([(x, y), (x, y - L)], 5, (251, 191, 36, 255))


@g("bentplate")
def _(c):
    c.line([(0.06, 0.82), (0.5, 0.82), (0.5, 0.3), (0.94, 0.3)], 18)
    c.line([(0.06, 0.82), (0.5, 0.82), (0.5, 0.3), (0.94, 0.3)], 6, (30, 30, 50, 160))


@g("unfold")
def _(c):
    c.line([(0.06, 0.4), (0.4, 0.4), (0.4, 0.12)], 12, SOFT)
    c.line([(0.06, 0.82), (0.94, 0.82)], 16)
    c.arrow((0.62, 0.28), (0.8, 0.62), 8, 0.08)


@g("laminate")
def _(c):
    for k, ang in enumerate((0, 45, -45, 90)):
        y = 0.18 + 0.2 * k
        c.rect(0.08, y, 0.92, y + 0.15, fill=SOFT if k % 2 else W, outline=None, r=3)
        for j in range(5):
            x = 0.15 + 0.17 * j
            dx = 0.06 * math.cos(math.radians(ang))
            c.line([(x - dx, y + 0.075 - 0.05 * math.sin(math.radians(ang))), (x + dx, y + 0.075 + 0.05 * math.sin(math.radians(ang)))], 3, (20, 30, 60, 255))


@g("clash")
def _(c):
    c.rect(0.06, 0.2, 0.58, 0.72, fill=SOFT, r=8)
    c.rect(0.42, 0.36, 0.94, 0.88, fill=None, r=8, w=10)
    c.rect(0.42, 0.36, 0.58, 0.72, fill=(248, 113, 113, 255), outline=None, r=2)


@g("reserve")
def _(c):
    cube(c)
    c.text("R", 0.5, 0.56, 70, col=(251, 191, 36, 255))


@g("kinematics")
def _(c):
    c.line([(0.2, 0.8), (0.82, 0.32)], 14)
    c.line([(0.2, 0.8), (0.86, 0.66)], 6, SOFT)
    c.ell(0.12, 0.72, 0.28, 0.88, fill=W, outline=None)
    c.arc(0.0, 0.2, 1.0, 1.2, 290, 345, 8)
    c.arc(0.0, 0.2, 1.0, 1.2, 345, 360, 4)


@g("pipe")
def _(c):
    c.curve(lambda t: (0.06 + 0.88 * t, 0.7 - 0.4 * t * t), 0, 1, w=20)
    c.curve(lambda t: (0.06 + 0.88 * t, 0.7 - 0.4 * t * t), 0, 1, w=6, col=(30, 40, 60, 200))


@g("harness")
def _(c):
    for k, dy in enumerate((-0.06, 0.0, 0.06)):
        c.curve(lambda t, dy=dy: (0.06 + 0.88 * t, 0.5 + dy + 0.18 * math.sin(2 * math.pi * t)), 0, 1, w=7,
                col=[(248, 113, 113, 255), (250, 204, 21, 255), (96, 165, 250, 255)][k])


@g("full_view")
def _(c):
    for x0, y0 in ((0.06, 0.06), (0.94, 0.06), (0.06, 0.94), (0.94, 0.94)):
        c.arrow((0.5 + (x0 - 0.5) * 0.35, 0.5 + (y0 - 0.5) * 0.35), (x0, y0), 9, 0.12)


@g("panel_tree")
def _(c):
    c.rect(0.08, 0.08, 0.92, 0.92, r=8, w=9)
    c.rect(0.08, 0.08, 0.36, 0.92, fill=SOFT, outline=None, r=6)
    for y in (0.25, 0.42, 0.59, 0.76):
        c.line([(0.14, y), (0.3, y)], 5)


@g("panel_props")
def _(c):
    c.rect(0.08, 0.08, 0.92, 0.92, r=8, w=9)
    c.rect(0.64, 0.08, 0.92, 0.92, fill=SOFT, outline=None, r=6)
    for y in (0.25, 0.42, 0.59, 0.76):
        c.line([(0.7, y), (0.86, y)], 5)


@g("annotate")
def _(c):
    jet(c, 0.6, 0, 0.08)
    c.arrow((0.5, 0.92), (0.06, 0.92), 6, 0.06)
    c.arrow((0.5, 0.92), (0.94, 0.92), 6, 0.06)
    c.text("mm", 0.5, 0.1, 50)


@g("cfd_sideslip")
def _(c):
    jet(c, 0.7, 0.02, 0.05)
    for k in range(3):
        c.arrow((0.02, 0.2 + 0.22 * k), (0.3, 0.3 + 0.22 * k), 6, 0.07, SOFT)
    c.text("\u03b2", 0.85, 0.15, 60, sym=True)


@g("cfd_gust")
def _(c):
    c.curve(lambda t: (0.06 + 0.88 * t, 0.3 + 0.12 * math.sin(4 * math.pi * t)), 0, 1, w=10)
    airfoil(c, 0.18, 0.82, 0.72, 0.18, w=8, fill=SOFT)
    c.arrow((0.5, 0.42), (0.5, 0.6), 7, 0.06)


@g("sim_setup")
def _(c):
    for k in range(3):  # three sliders: the conditions of a run
        y = 0.24 + 0.26 * k
        c.line([(0.12, y), (0.88, y)], 7)
        x = (0.3, 0.68, 0.46)[k]
        c.rect(x - 0.08, y - 0.09, x + 0.08, y + 0.09, fill=W, outline=None, r=3)


@g("turbulence")
def _(c):
    for k in range(3):  # eddies shed into a wake
        cx, cy, r = 0.26 + 0.25 * k, 0.5 + (0.12 if k % 2 else -0.12), 0.15 - 0.02 * k
        c.curve(lambda t, cx=cx, cy=cy, r=r: (cx + r * (1 - 0.6 * t) * math.cos(5 * t), cy + r * (1 - 0.6 * t) * math.sin(5 * t)), 0, 1, w=7)
    c.line([(0.04, 0.5), (0.12, 0.5)], 7)


@g("optimise")
def _(c):
    c.curve(lambda t: (0.1 + 0.8 * t, 0.78 - 0.5 * math.exp(-((t - 0.62) / 0.2) ** 2)), 0, 1, w=7)  # an objective with its best point
    for x in (0.18, 0.34, 0.5, 0.82):
        y = 0.78 - 0.5 * math.exp(-(((x - 0.1) / 0.8 - 0.62) / 0.2) ** 2)
        c.ell(x - 0.035, y - 0.035, x + 0.035, y + 0.035, fill=W, w=4)
    c.line([(0.6, 0.12), (0.6, 0.2)], 6)  # the optimum


@g("sch_apps")
def _(c):
    for i in range(3):
        for j in range(3):
            x, y = 0.08 + i * 0.3, 0.08 + j * 0.3
            c.rect(x, y, x + 0.24, y + 0.24, fill=W, w=6, r=8)


@g("opt_wing_apply")
def _(c):
    c.poly([(0.08, 0.62), (0.62, 0.62), (0.92, 0.86), (0.42, 0.86)], fill=W, w=6)
    c.arrow((0.5, 0.06), (0.5, 0.5), 12, 0.14)


@g("zoom_in")
def _(c):
    c.ell(0.08, 0.08, 0.66, 0.66, w=12)
    c.line([(0.6, 0.6), (0.92, 0.92)], 16)
    c.line([(0.37, 0.22), (0.37, 0.52)], 10)
    c.line([(0.22, 0.37), (0.52, 0.37)], 10)


@g("zoom_out")
def _(c):
    c.ell(0.08, 0.08, 0.66, 0.66, w=12)
    c.line([(0.6, 0.6), (0.92, 0.92)], 16)
    c.line([(0.22, 0.37), (0.52, 0.37)], 10)


@g("dims_part")
def _(c):
    c.poly([(0.1, 0.55), (0.55, 0.55), (0.9, 0.82), (0.45, 0.82)], fill=W, w=6)
    c.arrow((0.5, 0.3), (0.08, 0.3), 6, 0.08)
    c.arrow((0.5, 0.3), (0.92, 0.3), 6, 0.08)
    c.line([(0.08, 0.2), (0.08, 0.5)], 5)
    c.line([(0.92, 0.2), (0.92, 0.75)], 5)


@g("appearance")
def _(c):
    c.ell(0.06, 0.06, 0.94, 0.94, w=10)
    for k, (x, y) in enumerate(((0.32, 0.3), (0.62, 0.28), (0.74, 0.55), (0.3, 0.58))):
        c.ell(x - 0.09, y - 0.09, x + 0.09, y + 0.09, fill=W, w=4)
        _ = k


@g("prefs")
def _(c):
    c.ell(0.28, 0.28, 0.72, 0.72, w=10)
    for k in range(8):
        a = k * math.pi / 4
        c.line([(0.5 + 0.24 * math.cos(a), 0.5 + 0.24 * math.sin(a)), (0.5 + 0.38 * math.cos(a), 0.5 + 0.38 * math.sin(a))], 12)
    c.ell(0.42, 0.42, 0.58, 0.58, fill=W, outline=None)


@g("cfd_quality")
def _(c):
    for k in range(4):
        c.rect(0.08 + 0.22 * k, 0.86 - 0.18 * (k + 1), 0.24 + 0.22 * k, 0.86, fill=W if k < 3 else SOFT, outline=None, r=3)
    c.text("Q", 0.3, 0.2, 60)


@g("live")
def _(c):
    c.ell(0.12, 0.12, 0.88, 0.88, w=10)
    c.poly([(0.38, 0.28), (0.74, 0.5), (0.38, 0.72)], fill=W, outline=None)
    c.ell(0.74, 0.06, 0.94, 0.26, fill=(248, 113, 113, 255), outline=None)


@g("streamlines")
def _(c):
    for k in range(4):
        c.curve(lambda t, k=k: (0.04 + 0.92 * t, 0.2 + 0.2 * k - 0.12 * math.exp(-((t - 0.5) / 0.2) ** 2) * (1 if k < 2 else -1)), 0, 1, w=8)
    c.ell(0.44, 0.44, 0.56, 0.56, fill=(251, 191, 36, 255), outline=None)


@g("vortex")
def _(c):
    c.curve(lambda t: (0.5 + 0.38 * t * math.cos(10 * t), 0.5 + 0.38 * t * math.sin(10 * t)), 0, 1, n=120, w=10)


@g("slice")
def _(c):
    cube(c)
    c.poly([(0.31, 0.22), (0.69, 0.42), (0.69, 0.82), (0.31, 0.62)], fill=(236, 72, 153, 200), outline=W, w=6)


@g("clear_results")
def _(c):
    G["clear"](c)


@g("rules")
def _(c):
    c.rect(0.12, 0.06, 0.88, 0.94, fill=SOFT, r=8)
    for k, ok in enumerate((True, True, False)):
        y = 0.24 + 0.24 * k
        if ok:
            c.line([(0.2, y), (0.28, y + 0.07), (0.4, y - 0.07)], 7)
        else:
            c.line([(0.2, y - 0.06), (0.36, y + 0.08)], 7, (248, 113, 113, 255))
            c.line([(0.2, y + 0.08), (0.36, y - 0.06)], 7, (248, 113, 113, 255))
        c.line([(0.48, y), (0.8, y)], 6)


@g("skeleton")
def _(c):
    pts = [(0.5, 0.1), (0.5, 0.55), (0.12, 0.62), (0.88, 0.62), (0.5, 0.92)]
    c.line([pts[0], pts[1], pts[4]], 7)
    c.line([pts[2], pts[1], pts[3]], 7)
    for x, y in pts:
        c.ell(x - 0.07, y - 0.07, x + 0.07, y + 0.07, fill=W, outline=None)


@g("rflp")
def _(c):
    for k, L in enumerate("RFLP"):
        x = 0.14 + 0.24 * k
        c.ell(x - 0.11, 0.39, x + 0.11, 0.61, fill=SOFT if k % 2 else W, outline=None)
        c.text(L, x, 0.5, 46, col=(20, 30, 60, 255))
        if k < 3:
            c.line([(x + 0.11, 0.5), (x + 0.13, 0.5)], 5)


@g("start_page")
def _(c):
    c.poly([(0.5, 0.08), (0.92, 0.44), (0.82, 0.44), (0.82, 0.9), (0.18, 0.9), (0.18, 0.44), (0.08, 0.44)], fill=SOFT)
    c.rect(0.42, 0.6, 0.58, 0.9, fill=W, outline=None, r=3)


@g("new_blank")
def _(c):
    c.poly([(0.2, 0.06), (0.66, 0.06), (0.82, 0.22), (0.82, 0.94), (0.2, 0.94)], fill=SOFT)
    c.line([(0.51, 0.36), (0.51, 0.76)], 12)
    c.line([(0.31, 0.56), (0.71, 0.56)], 12)


G["sch_start"] = G["start_page"]
G["sch_part"] = G["pad"]
G["sch_sheet"] = G["bentplate"]
G["sch_asm"] = G["clash"]
G["sch_know"] = G["rules"]


def hexrgb(r, g_, b, f):
    return tuple(int(max(0, min(255, v + (255 - v) * f if f > 0 else v * (1 + f)))) for v in (r, g_, b))


def tile(rgb, glyph) -> Image.Image:
    pad = 24
    im = Image.new("RGBA", (S + 2 * pad, S + 2 * pad), (0, 0, 0, 0))
    # soft drop shadow for depth
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((pad + 6, pad + 14, pad + S - 6, pad + S + 6), radius=58, fill=(0, 0, 0, 150))
    im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(10)))
    # gradient body
    body = Image.new("RGBA", (S, S))
    top, bot = hexrgb(*rgb, 0.35), hexrgb(*rgb, -0.35)
    for y in range(S):
        t = y / (S - 1)
        col = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)) + (255,)
        ImageDraw.Draw(body).line([(0, y), (S, y)], fill=col)
    # diagonal sheen and gloss
    gl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(gl).ellipse((-S * 0.3, -S * 0.75, S * 1.3, S * 0.42), fill=(255, 255, 255, 56))
    body = Image.alpha_composite(body, gl)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, S - 1, S - 1), radius=58, fill=255)
    im.paste(body, (pad, pad), mask)
    ImageDraw.Draw(im).rounded_rectangle((pad + 2, pad + 2, pad + S - 3, pad + S - 3), radius=56, outline=hexrgb(*rgb, 0.55) + (170,), width=4)
    # glyph with a faint shadow
    c = C()
    glyph(c)
    gs = Image.new("RGBA", c.im.size, (0, 0, 0, 0))
    gs.putalpha(c.im.getchannel("A").point(lambda a: a * 0.35))
    im.alpha_composite(gs.filter(ImageFilter.GaussianBlur(4)), (pad + 3, pad + 5))
    im.alpha_composite(c.im, (pad, pad))
    return im.resize((64 + 12, 64 + 12), Image.LANCZOS).crop((6, 6, 70, 70))


def flat(rgb, glyph) -> Image.Image:
    """The ribbon's icon: the glyph alone, in the workbench colour (darkened for contrast on light grey), on
    transparency, with a thin darker outline so light colours still read on white."""
    c = C()
    glyph(c)
    a = c.im.getchannel("A")
    ink = hexrgb(*rgb, -0.25)
    edge = hexrgb(*rgb, -0.6)
    out = Image.new("RGBA", c.im.size, (0, 0, 0, 0))
    halo = Image.new("RGBA", c.im.size, edge + (255,))
    halo.putalpha(a.filter(ImageFilter.MaxFilter(7)).point(lambda v: v * 0.55))
    out = Image.alpha_composite(out, halo)
    body = Image.new("RGBA", c.im.size, ink + (255,))
    body.putalpha(a)
    out = Image.alpha_composite(out, body)
    return out.crop((14, 14, S - 14, S - 14)).resize((64, 64), Image.LANCZOS)


def parse() -> tuple[dict, list]:
    t = (ROOT / "cpp" / "studio" / "tools.inc").read_text(encoding="utf-8")
    schemes = {m[0]: (int(m[1]), int(m[2]), int(m[3]), m[4]) for m in re.findall(r'\{"(\w+)", "[^"]*", (\d+), (\d+), (\d+), "(\w+)"\}', t)}
    tools = re.findall(r'\{"(\w+)", "(\w+)", "[^"]*", "[^"]*",', t)
    return schemes, tools


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FLAT.mkdir(parents=True, exist_ok=True)
    schemes, tools = parse()
    n, missing = 0, []
    for _key, (r, g_, b, icon) in schemes.items():
        tile((r, g_, b), G[icon]).save(OUT / f"{icon}.png")
        n += 1
    for sch, icon in tools:
        if icon not in G:
            missing.append(icon)
            continue
        r, g_, b, _ = schemes[sch]
        tile((r, g_, b), G[icon]).save(OUT / f"{icon}.png")
        flat((r, g_, b), G[icon]).save(FLAT / f"{icon}.png")
        n += 1
    # contact sheet for the documentation
    names = [i for _, (_, _, _, i) in schemes.items()] + [i for _, i in tools if i in G]
    cols = 12
    sheet = Image.new("RGBA", (cols * 72 + 8, ((len(names) + cols - 1) // cols) * 72 + 8), (14, 24, 42, 255))
    for k, nm in enumerate(names):
        sheet.alpha_composite(Image.open(OUT / f"{nm}.png"), (8 + (k % cols) * 72, 8 + (k // cols) * 72))
    sheet.save(ROOT / "docs" / "assets" / "studio_icons.png")
    print(f"{n} icons in {OUT} (and flat ribbon icons in {FLAT}); missing glyphs: {missing or 'none'}")


if __name__ == "__main__":
    main()
