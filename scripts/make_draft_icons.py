"""Flat icons and palette tiles for the 2D drawing, sheet-metal and fin-PINN tools of AeroSynapse Studio.

Same style as ``make_asm_icons.py`` (drawn at 4x, reduced to 64 x 64; tiles in the workbench's colour).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_asm_icons as A  # noqa: E402

A.SCHEME_RGB.update(
    {
        "view": (100, 116, 139),
        "sketch": (0, 188, 235),
        "sheet": (168, 85, 247),
        "export": (234, 179, 8),
        "sim": (236, 72, 153),
    }
)
A.SCHEME_RGB["opt"] = (34, 197, 94)
INK, BLUE, ORANGE, TEAL, GREEN, RED, STEEL = A.INK, A.BLUE, A.ORANGE, A.TEAL, A.GREEN, (220, 38, 38), A.STEEL
W = 14


def sheet(d, pts, fill=STEEL, w=26):
    d.line(pts, fill=fill, width=w, joint="curve")


def draft(d):  # a drawing board: a sheet with a part outline and a dimension
    d.rounded_rectangle((26, 40, 230, 216), radius=14, fill=(240, 244, 250))
    d.rectangle((44, 120, 150, 190), outline=INK, width=W)
    d.ellipse((160, 70, 214, 124), outline=INK, width=W)
    d.line((44, 92, 150, 92), fill=BLUE, width=8)
    for x in (44, 150):
        d.line((x, 80, x, 104), fill=BLUE, width=8)
    d.line((26, 216, 230, 216), fill=ORANGE, width=16)


def draft_views(d):  # three views: front, top and side
    for x0, y0, x1, y1 in ((30, 40, 120, 120), (30, 140, 120, 220), (140, 40, 220, 120)):
        d.rectangle((x0, y0, x1, y1), outline=INK, width=W)
    d.line((60, 140, 60, 220), fill=BLUE, width=8)
    for k in range(4):
        d.line((150 + k * 18, 80, 160 + k * 18, 80), fill=BLUE, width=8)
    d.polygon([(150, 150), (200, 140), (226, 170), (176, 182)], fill=ORANGE)


def draft_dxf(d):  # a file with DXF on it
    d.polygon([(52, 24), (166, 24), (210, 68), (210, 232), (52, 232)], fill=(240, 244, 250))
    d.polygon([(166, 24), (210, 68), (166, 68)], fill=STEEL)
    d.rounded_rectangle((36, 120, 190, 184), radius=10, fill=BLUE)
    for k, ch in enumerate("DXF"):
        x = 56 + 44 * k
        if ch == "D":
            d.line((x, 132, x, 172), fill="white", width=10)
            d.arc((x - 20, 132, x + 22, 172), -90, 90, fill="white", width=10)
        elif ch == "X":
            d.line((x - 4, 132, x + 24, 172), fill="white", width=10)
            d.line((x + 24, 132, x - 4, 172), fill="white", width=10)
        else:
            d.line((x, 132, x, 172), fill="white", width=10)
            d.line((x, 136, x + 26, 136), fill="white", width=10)
            d.line((x, 152, x + 20, 152), fill="white", width=10)


def sheet_sketch(d):  # a pencil drawing a bend line that becomes a bent sheet
    sheet(d, [(40, 200), (40, 120), (150, 120), (150, 60)], STEEL)
    d.line((40, 200, 40, 120, 150, 120, 150, 60), fill=BLUE, width=6)
    d.polygon([(170, 150), (228, 92), (246, 110), (188, 168)], fill=ORANGE)
    d.polygon([(170, 150), (188, 168), (160, 178)], fill=INK)


def sheet_flat(d):  # a flat blank with dashed bend lines and an arrow from the bent part
    d.rectangle((24, 120, 232, 210), fill=STEEL)
    for x in (92, 164):
        for y in range(126, 206, 20):
            d.line((x, y, x, y + 10), fill=RED, width=8)
    sheet(d, [(60, 96), (60, 40), (196, 40), (196, 96)], A.STEEL_D, 18)
    A.arrow(d, 128, 52, 128, 108, ORANGE, 12)


def sheet_channel(d):
    sheet(d, [(50, 60), (50, 196), (206, 196), (206, 60)])


def sheet_z(d):
    sheet(d, [(30, 196), (120, 196), (120, 60), (226, 60)])


def sheet_hat(d):
    sheet(d, [(20, 196), (70, 196), (70, 70), (186, 70), (186, 196), (236, 196)])


def sheet_l(d):
    sheet(d, [(60, 40), (60, 196), (216, 196)])


def fin_pinn(d):  # a finned heat sink coloured hot to cold, with a network node pattern
    d.rectangle((30, 180, 226, 214), fill=(120, 30, 30))
    cols = [(220, 38, 38), (234, 88, 12), (245, 158, 11), (250, 204, 21), (132, 204, 22)]
    for k, c in enumerate(cols):
        x = 40 + k * 38
        d.rectangle((x, 60 + k * 14, x + 22, 182), fill=c)
    pts = [(60, 36), (128, 24), (196, 36)]
    for a in pts:
        for b in pts:
            if a < b:
                d.line((*a, *b), fill=TEAL, width=6)
    for x, y in pts:
        d.ellipse((x - 12, y - 12, x + 12, y + 12), fill=TEAL)


def hq_image(d):  # a sharp photo frame with a 4K badge and a sparkle
    d.rounded_rectangle((20, 50, 236, 206), radius=18, fill=(30, 41, 59))
    d.polygon([(40, 186), (100, 110), (140, 160), (170, 128), (216, 186)], fill=(34, 211, 238))
    d.ellipse((168, 70, 200, 102), fill=(250, 204, 21))
    d.rounded_rectangle((150, 18, 246, 66), radius=14, fill=(236, 72, 153))
    d.text((163, 24), "4K", fill="white", font_size=40)


def _window(d, accent):  # an app window: title bar, sidebar, panel
    d.rounded_rectangle((18, 30, 238, 226), radius=16, fill=(241, 245, 249))
    d.rounded_rectangle((18, 30, 238, 70), radius=16, fill=accent)
    d.rectangle((18, 54, 238, 70), fill=accent)
    d.rectangle((30, 82, 84, 214), fill=(203, 213, 225))


def labs(d):
    _window(d, (234, 88, 12))
    d.polygon([(150, 96), (178, 96), (178, 140), (214, 200), (114, 200), (150, 140)], fill=(37, 99, 235))
    d.rectangle((144, 90, 184, 100), fill=(30, 41, 59))


def labs_app(d):
    _window(d, (220, 38, 38))
    for k, c in enumerate([(220, 38, 38), (234, 88, 12), (250, 204, 21)]):
        d.rectangle((104 + k * 40, 200 - 30 * (k + 1), 132 + k * 40, 210), fill=c)


def labs_designer(d):
    _window(d, (37, 99, 235))
    d.polygon([(160, 86), (174, 140), (226, 160), (226, 172), (174, 164), (168, 196), (186, 210), (134, 210), (152, 196), (146, 164), (96, 172), (96, 160), (146, 140)], fill=(30, 41, 59))


def labs_wb(d):
    _window(d, (20, 150, 140))
    d.line((100, 200, 130, 150, 160, 170, 196, 104, 226, 120), fill=(20, 150, 140), width=12, joint="curve")


def labs_airfrans(d):  # an airfoil in a coloured flow field
    d.rounded_rectangle((14, 40, 242, 216), radius=16, fill=(30, 58, 138))
    for k, c in enumerate([(59, 130, 246), (34, 211, 238), (250, 204, 21), (249, 115, 22)]):
        d.line((24, 70 + k * 36, 232, 60 + k * 36), fill=c, width=10)
    d.polygon([(54, 128), (90, 106), (150, 104), (214, 124), (150, 138), (90, 142)], fill=(241, 245, 249))


ICONS = {
    "draft": ("export", draft),
    "draft_views": ("export", draft_views),
    "draft_dxf": ("export", draft_dxf),
    "sheet_sketch": ("sheet", sheet_sketch),
    "sheet_flat": ("sheet", sheet_flat),
    "sheet_channel": ("sheet", sheet_channel),
    "sheet_z": ("sheet", sheet_z),
    "sheet_hat": ("sheet", sheet_hat),
    "sheet_l": ("sheet", sheet_l),
    "fin_pinn": ("sim", fin_pinn),
    "hq_image": ("view", hq_image),
    "labs": ("sim", labs),
    "labs_app": ("sim", labs_app),
    "labs_designer": ("opt", labs_designer),
    "labs_wb": ("sim", labs_wb),
    "labs_airfrans": ("sim", labs_airfrans),
}

if __name__ == "__main__":
    for name, (scheme, fn) in ICONS.items():
        A.make(name, fn)
        A.tile(name, scheme)
    print("wrote", len(ICONS), "icons and palette tiles")
