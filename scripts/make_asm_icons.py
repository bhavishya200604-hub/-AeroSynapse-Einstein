"""Flat 64 x 64 icons for the Assembly tools of AeroSynapse Studio (assets/icons_flat/asm_*.png).

Drawn at 4x and reduced, in the style of the other flat icons: simple filled shapes, a few colours, no outlines.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "assets" / "icons_flat"
S = 256  # drawing size
STEEL, STEEL_D, ORANGE, TEAL, BLUE, GREEN, INK = (150, 158, 170), (96, 104, 118), (232, 120, 40), (20, 150, 140), (37, 99, 235), (34, 160, 90), (55, 62, 75)


def hexagon(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill, rot: float = 0.0) -> None:
    d.polygon([(cx + r * math.cos(rot + k * math.pi / 3), cy + r * math.sin(rot + k * math.pi / 3)) for k in range(6)], fill=fill)


def block(d, x0, y0, x1, y1, fill, top=None):
    d.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=fill)
    if top:
        d.rounded_rectangle((x0, y0, x1, y0 + 26), radius=12, fill=top)


def arrow(d, x0, y0, x1, y1, fill, w=16):
    d.line((x0, y0, x1, y1), fill=fill, width=w)
    a = math.atan2(y1 - y0, x1 - x0)
    h = 34
    d.polygon([(x1 + 10 * math.cos(a), y1 + 10 * math.sin(a)), (x1 - h * math.cos(a - 0.55), y1 - h * math.sin(a - 0.55)), (x1 - h * math.cos(a + 0.55), y1 - h * math.sin(a + 0.55))], fill=fill)


def bolt(d, cx, top, length, width=46, head=86):
    hexagon(d, cx, top + 26, head / 2, STEEL_D, math.pi / 6)
    d.rectangle((cx - head / 2 * 0.87, top, cx + head / 2 * 0.87, top + 52), fill=STEEL_D)
    d.rectangle((cx - width / 2, top + 50, cx + width / 2, top + 50 + length), fill=STEEL)
    for k in range(int(length / 18)):
        y = top + 64 + k * 18
        d.line((cx - width / 2, y, cx + width / 2, y + 10), fill=STEEL_D, width=5)


def make(name: str, draw) -> None:
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    draw(d)
    im.resize((64, 64), Image.LANCZOS).save(OUT / f"{name}.png")


def gear(d, cx, cy, r, z, fill, hole=True):
    """A toothed wheel: z square-ish teeth on a disc of radius r."""
    pts = []
    for k in range(z):
        a0 = 2 * math.pi * k / z
        for da, rr in ((-0.32, r), (-0.2, r * 1.22), (0.2, r * 1.22), (0.32, r)):
            a = a0 + da * 2 * math.pi / z
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d.polygon(pts, fill=fill)
    if hole:
        d.ellipse((cx - r * 0.28, cy - r * 0.28, cx + r * 0.28, cy + r * 0.28), fill=(0, 0, 0, 0))


def icons() -> dict:
    return {
        "env_setup": lambda d: (d.rectangle((20, 150, 236, 236), fill=(96, 165, 250)), [d.arc((20 + 54 * k, 126, 74 + 54 * k, 176), 200, 340, fill=BLUE, width=12) for k in range(4)],
                                d.rounded_rectangle((150, 20, 186, 150), radius=18, fill=(220, 38, 38)), d.ellipse((140, 120, 196, 176), fill=(220, 38, 38)),
                                arrow(d, 70, 120, 70, 30, BLUE, 12)),
        "env_clear": lambda d: (d.rectangle((20, 150, 236, 236), fill=(147, 197, 253)), d.line((40, 40, 216, 216), fill=(220, 38, 38), width=22), d.line((216, 40, 40, 216), fill=(220, 38, 38), width=22)),
        "asm_gear": lambda d: gear(d, 128, 128, 92, 12, ORANGE),
        "asm_bearing": lambda d: (d.ellipse((16, 16, 240, 240), fill=STEEL_D), d.ellipse((46, 46, 210, 210), fill=(0, 0, 0, 0)), [d.ellipse((128 + 66 * math.cos(k * 0.785) - 18, 128 + 66 * math.sin(k * 0.785) - 18, 128 + 66 * math.cos(k * 0.785) + 18, 128 + 66 * math.sin(k * 0.785) + 18), fill=STEEL) for k in range(8)], d.ellipse((84, 84, 172, 172), fill=STEEL_D), d.ellipse((104, 104, 152, 152), fill=(0, 0, 0, 0))),
        "asm_bushing": lambda d: (d.rectangle((70, 40, 186, 220), fill=(196, 140, 60)), d.rectangle((40, 40, 216, 76), fill=(170, 115, 45)), d.rectangle((104, 40, 152, 220), fill=(0, 0, 0, 0))),
        "asm_keyshaft": lambda d: (d.rounded_rectangle((16, 98, 240, 158), radius=14, fill=STEEL), d.rectangle((80, 98, 176, 116), fill=STEEL_D)),
        "asm_key": lambda d: d.rounded_rectangle((40, 96, 216, 160), radius=30, fill=STEEL_D),
        "asm_coupling": lambda d: (d.rectangle((70, 70, 186, 186), fill=STEEL_D), d.rectangle((10, 110, 70, 146), fill=STEEL), d.rectangle((186, 110, 246, 146), fill=STEEL)),
        "asm_roller": lambda d: (d.rectangle((60, 40, 196, 216), fill=STEEL), d.ellipse((40, 40, 80, 216), fill=STEEL_D), d.ellipse((176, 40, 216, 216), fill=STEEL)),
        "asm_roll_mate": lambda d: (d.ellipse((14, 60, 134, 180), fill=STEEL_D), d.ellipse((126, 84, 222, 180), fill=ORANGE), arrow(d, 74, 40, 120, 40, BLUE, 10)),
        "asm_pulley": lambda d: (d.ellipse((16, 16, 240, 240), fill=STEEL), d.ellipse((44, 44, 212, 212), fill=STEEL_D), d.ellipse((60, 60, 196, 196), fill=STEEL), d.ellipse((100, 100, 156, 156), fill=(0, 0, 0, 0))),
        "asm_sprocket": lambda d: (gear(d, 128, 128, 90, 14, STEEL_D), [d.ellipse((128 + 96 * math.cos((k + 0.5) * 2 * math.pi / 14) - 13, 128 + 96 * math.sin((k + 0.5) * 2 * math.pi / 14) - 13, 128 + 96 * math.cos((k + 0.5) * 2 * math.pi / 14) + 13, 128 + 96 * math.sin((k + 0.5) * 2 * math.pi / 14) + 13), fill=(0, 0, 0, 0)) for k in range(14)]),
        "asm_belt": lambda d: (d.ellipse((14, 74, 122, 182), fill=STEEL), d.ellipse((120, 30, 246, 156), fill=STEEL), d.line((68, 74, 183, 30), fill=INK, width=16), d.line((68, 182, 183, 156), fill=INK, width=16),
                               d.arc((14, 74, 122, 182), 90, 270, fill=INK, width=16), d.arc((120, 30, 246, 156), 270, 90, fill=INK, width=16)),
        "asm_chain": lambda d: (gear(d, 68, 128, 46, 9, STEEL_D), gear(d, 186, 128, 54, 11, STEEL_D), [d.ellipse((60 + 18 * k - 9, 62 - 9, 60 + 18 * k + 9, 62 + 9), fill=ORANGE) for k in range(8)],
                                [d.ellipse((60 + 18 * k - 9, 194 - 9, 60 + 18 * k + 9, 194 + 9), fill=ORANGE) for k in range(8)]),
        "asm_belt_demo": lambda d: (d.rectangle((14, 196, 242, 236), fill=(120, 160, 205)), d.ellipse((20, 60, 100, 140), fill=STEEL), d.ellipse((120, 30, 236, 146), fill=STEEL),
                                    d.line((60, 60, 178, 30), fill=INK, width=12), d.line((60, 140, 178, 146), fill=INK, width=12), d.polygon([(30, 150), (30, 190), (64, 170)], fill=GREEN)),
        "asm_worm": lambda d: (d.rounded_rectangle((20, 92, 236, 164), radius=30, fill=STEEL), [d.line((40 + 26 * k, 92, 58 + 26 * k, 164), fill=STEEL_D, width=12) for k in range(7)]),
        "asm_worm_wheel": lambda d: (gear(d, 128, 128, 88, 16, (196, 140, 60)), d.arc((40, 40, 216, 216), 200, 340, fill=(150, 100, 40), width=10)),
        "asm_worm_demo": lambda d: (gear(d, 128, 92, 66, 14, (196, 140, 60)), d.rounded_rectangle((20, 168, 236, 220), radius=24, fill=STEEL), [d.line((40 + 26 * k, 168, 54 + 26 * k, 220), fill=STEEL_D, width=10) for k in range(7)]),
        "asm_cam": lambda d: (d.ellipse((30, 70, 190, 230), fill=STEEL_D), d.ellipse((70, 20, 150, 120), fill=STEEL_D), d.polygon([(70, 70), (150, 70), (190, 150), (30, 150)], fill=STEEL_D), d.ellipse((96, 136, 124, 164), fill=(0, 0, 0, 0))),
        "asm_follower": lambda d: (d.rectangle((116, 20, 140, 170), fill=STEEL), d.ellipse((88, 150, 168, 230), fill=STEEL_D), d.ellipse((116, 178, 140, 202), fill=STEEL)),
        "asm_cam_mate": lambda d: (d.ellipse((20, 110, 150, 240), fill=STEEL_D), d.ellipse((50, 70, 120, 140), fill=STEEL_D), d.rectangle((150, 10, 172, 120), fill=ORANGE), d.ellipse((132, 100, 190, 158), fill=ORANGE)),
        "asm_cam_demo": lambda d: (d.rectangle((14, 196, 242, 236), fill=(120, 160, 205)), d.ellipse((50, 100, 150, 200), fill=STEEL_D), d.ellipse((80, 70, 120, 120), fill=STEEL_D), d.rectangle((90, 10, 110, 80), fill=ORANGE), d.polygon([(170, 150), (170, 190), (204, 170)], fill=GREEN)),
        "asm_flow": lambda d: ([d.arc((20 + 0, 40 + 50 * k, 236, 120 + 50 * k), 200, 340, fill=BLUE, width=14) for k in range(3)], d.ellipse((96, 96, 160, 160), fill=STEEL_D)),
        "asm_flow_stop": lambda d: (d.rounded_rectangle((50, 50, 206, 206), radius=24, fill=(220, 38, 38))),
        "asm_flow_view": lambda d: (d.polygon([(30, 200), (110, 40), (230, 40), (150, 200)], fill=(96, 165, 250)), d.ellipse((90, 90, 130, 130), fill=(220, 38, 38)), d.ellipse((140, 110, 180, 150), fill=BLUE)),
        "asm_flow_cyl": lambda d: (d.ellipse((24, 100, 80, 156), fill=STEEL_D), [d.ellipse((96 + 50 * k, 70 + (60 if k % 2 else 0), 136 + 50 * k, 110 + (60 if k % 2 else 0)), outline=(220, 38, 38) if k % 2 else BLUE, width=10) for k in range(3)]),
        "asm_flow_tank": lambda d: (d.rectangle((40, 30, 216, 230), outline=STEEL_D, width=14), d.rectangle((54, 80, 202, 222), fill=(147, 197, 253)), d.rectangle((122, 10, 134, 170), fill=STEEL), d.rectangle((86, 160, 170, 176), fill=STEEL_D), d.arc((70, 110, 186, 200), 180, 360, fill=BLUE, width=8)),
        "asm_flow_mode": lambda d: (d.rectangle((30, 30, 226, 226), fill=(96, 165, 250)), d.rectangle((60, 60, 196, 196), fill=(34, 160, 90)), d.rectangle((96, 96, 160, 160), fill=(220, 38, 38))),
        "asm_flow_gears": lambda d: (d.rectangle((14, 60, 242, 242), fill=(250, 204, 21)), gear(d, 92, 150, 52, 9, STEEL_D), gear(d, 176, 150, 36, 7, (220, 38, 38))),
        "asm_flow_belt": lambda d: (d.rectangle((14, 60, 242, 242), fill=(147, 197, 253)), d.ellipse((30, 110, 110, 190), fill=STEEL), d.ellipse((140, 90, 236, 186), fill=STEEL), d.line((70, 110, 188, 90), fill=INK, width=12), d.line((70, 190, 188, 186), fill=INK, width=12)),
        "asm_flow_splash": lambda d: (d.rectangle((14, 150, 242, 242), fill=(220, 38, 38)), gear(d, 90, 120, 56, 10, STEEL_D), gear(d, 180, 130, 44, 8, STEEL), [d.ellipse((60 + 30 * k, 40 + 12 * (k % 2), 76 + 30 * k, 56 + 12 * (k % 2)), fill=(250, 204, 21)) for k in range(5)]),
        "asm_flow_surf": lambda d: (d.ellipse((30, 30, 226, 226), fill=(34, 160, 90)), d.pieslice((30, 30, 226, 226), 200, 320, fill=(220, 38, 38)), d.pieslice((30, 30, 226, 226), 20, 120, fill=(37, 99, 235))),
        "asm_flow_walls": lambda d: (d.polygon([(40, 60), (150, 20), (150, 180), (40, 220)], fill=(37, 99, 235)), d.polygon([(150, 20), (230, 60), (230, 220), (150, 180)], fill=(220, 38, 38)), d.polygon([(40, 220), (150, 180), (230, 220), (120, 250)], fill=(34, 160, 90)), [d.line((70, 90 + 30 * k, 115, 75 + 30 * k), fill=(255, 255, 255), width=6) for k in range(4)]),
        "asm_check": lambda d: (block(d, 20, 100, 160, 220, (120, 160, 205)), block(d, 96, 30, 236, 150, ORANGE), d.rectangle((96, 100, 160, 150), fill=(220, 38, 38))),
        "asm_power": lambda d: (d.polygon([(150, 14), (60, 140), (124, 140), (100, 242), (200, 104), (134, 104)], fill=(234, 179, 8))),
        "asm_drive_demo": lambda d: (d.rectangle((14, 196, 242, 236), fill=(120, 160, 205)), gear(d, 84, 116, 52, 9, ORANGE), d.ellipse((140, 90, 236, 186), fill=STEEL), d.polygon([(186, 20), (186, 70), (226, 45)], fill=GREEN)),
        "asm_gear_helical": lambda d: (gear(d, 128, 128, 92, 12, ORANGE), [d.line((128 + 70 * math.cos(k * 0.52), 128 + 70 * math.sin(k * 0.52), 128 + 40 * math.cos(k * 0.52 + 0.3), 128 + 40 * math.sin(k * 0.52 + 0.3)), fill=(255, 220, 180), width=8) for k in range(12)]),
        "asm_gear_bevel": lambda d: (d.polygon([(30, 200), (226, 200), (176, 70), (80, 70)], fill=ORANGE), [d.line((40 + 20 * k, 196, 86 + 9 * k, 74), fill=(255, 220, 180), width=6) for k in range(10)]),
        "asm_rack": lambda d: (d.rectangle((14, 150, 242, 210), fill=ORANGE), [d.polygon([(20 + 38 * k, 152), (30 + 38 * k, 110), (44 + 38 * k, 110), (54 + 38 * k, 152)], fill=ORANGE) for k in range(6)], gear(d, 128, 60, 34, 8, BLUE)),
        "asm_gear_demo2": lambda d: (gear(d, 90, 150, 62, 9, ORANGE), d.polygon([(178, 150), (178, 236), (244, 193)], fill=GREEN), d.polygon([(150, 20), (240, 20), (215, 90), (175, 90)], fill=BLUE)),
        "asm_gear_mate": lambda d: (gear(d, 86, 150, 62, 9, ORANGE), gear(d, 182, 76, 44, 7, BLUE)),
        "asm_gear_demo": lambda d: (gear(d, 100, 140, 80, 10, ORANGE), d.polygon([(178, 150), (178, 236), (244, 193)], fill=GREEN)),
        "asm_new": lambda d: (block(d, 30, 120, 170, 220, (120, 160, 205), (150, 185, 225)), block(d, 96, 40, 226, 140, ORANGE, (245, 160, 90))),
        "asm_from_part": lambda d: (block(d, 26, 70, 150, 200, (120, 160, 205), (150, 185, 225)), arrow(d, 160, 135, 236, 135, ORANGE)),
        "asm_insert": lambda d: (block(d, 30, 130, 226, 226, (120, 160, 205), (150, 185, 225)), block(d, 84, 20, 172, 96, ORANGE), arrow(d, 128, 92, 128, 128, INK, 12)),
        "asm_demo": lambda d: (block(d, 20, 170, 236, 226, (120, 160, 205)), d.rounded_rectangle((70, 90, 230, 130), radius=18, fill=ORANGE),
                               d.ellipse((50, 80, 110, 140), fill=STEEL_D), d.rectangle((70, 100, 90, 200), fill=STEEL)),
        "asm_bolt": lambda d: bolt(d, 128, 22, 160),
        "asm_screw": lambda d: (d.rounded_rectangle((80, 22, 176, 86), radius=10, fill=STEEL_D), hexagon(d, 128, 52, 24, (60, 66, 78)),
                                d.rectangle((105, 84, 151, 234), fill=STEEL), [d.line((105, 98 + k * 18, 151, 108 + k * 18), fill=STEEL_D, width=5) for k in range(7)]),
        "asm_nut": lambda d: (hexagon(d, 128, 128, 104, STEEL_D), hexagon(d, 128, 128, 86, STEEL), d.ellipse((84, 84, 172, 172), fill=(0, 0, 0, 0))),
        "asm_washer": lambda d: (d.ellipse((24, 24, 232, 232), fill=STEEL), d.ellipse((84, 84, 172, 172), fill=(0, 0, 0, 0))),
        "asm_pin": lambda d: d.rounded_rectangle((100, 18, 156, 238), radius=26, fill=STEEL),
        "asm_shaft": lambda d: (d.rectangle((40, 104, 216, 152), fill=STEEL), d.ellipse((24, 104, 56, 152), fill=STEEL_D), d.rectangle((150, 96, 190, 112), fill=STEEL_D)),
        "asm_mate": lambda d: (block(d, 20, 130, 150, 230, (120, 160, 205), (150, 185, 225)), block(d, 106, 24, 236, 124, ORANGE, (245, 160, 90)),
                               d.ellipse((104, 100, 152, 148), fill=GREEN)),
        "asm_insert_hole": lambda d: (block(d, 20, 150, 236, 232, (120, 160, 205), (150, 185, 225)), d.rectangle((108, 150, 148, 232), fill=(70, 90, 120)), bolt(d, 128, 14, 110, 36, 70)),
        "asm_play": lambda d: (d.ellipse((16, 16, 240, 240), fill=GREEN), d.polygon([(98, 70), (98, 186), (192, 128)], fill=(255, 255, 255))),
        "asm_motor": lambda d: (d.ellipse((40, 40, 216, 216), outline=ORANGE, width=26), d.rectangle((150, 20, 236, 128), fill=(0, 0, 0, 0)),
                                d.polygon([(150, 26), (236, 26), (193, 92)], fill=ORANGE), d.ellipse((104, 104, 152, 152), fill=STEEL_D)),
        "asm_key": lambda d: (d.line((20, 128, 236, 128), fill=INK, width=10), d.polygon([(128, 50), (196, 128), (128, 206), (60, 128)], fill=ORANGE)),
        "asm_trace": lambda d: (d.line([(24 + k * 4, 170 - 90 * math.sin(k * 0.06)) for k in range(52)], fill=ORANGE, width=14), d.ellipse((206, 30, 246, 70), fill=BLUE),
                                d.ellipse((10, 150, 50, 190), fill=TEAL)),
        "asm_reset": lambda d: (d.arc((40, 40, 216, 216), 40, 320, fill=BLUE, width=26), d.polygon([(170, 22), (236, 70), (166, 100)], fill=BLUE)),
        "asm_explode": lambda d: (block(d, 88, 88, 168, 168, (120, 160, 205)), arrow(d, 80, 80, 26, 26, ORANGE, 14), arrow(d, 176, 80, 230, 26, ORANGE, 14),
                                  arrow(d, 80, 176, 26, 230, ORANGE, 14), arrow(d, 176, 176, 230, 230, ORANGE, 14)),
    }


PALETTE = OUT.parent / "icons"  # the tool palette's tiles: the workbench colour with the glyph in white
SCHEME_RGB = {"asm": (249, 115, 22), "sim": (236, 72, 153)}


def tile(name: str, scheme: str) -> None:
    """A palette tile from a flat icon: a rounded square shading to the workbench colour, the glyph in white."""
    r, g, b = SCHEME_RGB[scheme]
    t = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    grad = Image.new("RGBA", (S, S))
    for y in range(S):
        f = 0.45 * (1 - y / S)
        grad.paste((int(r + (255 - r) * f), int(g + (255 - g) * f), int(b + (255 - b) * f), 255), (0, y, S, y + 1))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((6, 6, S - 6, S - 6), radius=44, fill=255)
    t.paste(grad, (0, 0), mask)
    glyph = Image.open(OUT / f"{name}.png").convert("RGBA").resize((168, 168), Image.LANCZOS)
    white = Image.new("RGBA", glyph.size, (255, 255, 255, 255))
    white.putalpha(glyph.getchannel("A"))
    t.alpha_composite(white, (44, 44))
    t.resize((64, 64), Image.LANCZOS).save(PALETTE / f"{name}.png")


if __name__ == "__main__":
    for n, fn in icons().items():
        make(n, fn)
        tile(n, "asm")
    for n in ("cfd_flowlines", "cfd_sweep", "cfd_record"):  # their flat icons are drawn by hand (kept), tiles from them
        if (OUT / f"{n}.png").exists():
            tile(n, "sim")
    print("wrote", len(icons()), "icons to", OUT, "and palette tiles to", PALETTE)
