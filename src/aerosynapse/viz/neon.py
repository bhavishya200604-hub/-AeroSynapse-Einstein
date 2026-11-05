"""Futuristic dark ("neon") theme for animations: depth-gradient colormaps, glow lines, captions, mp4 writer.

Colour choices stay perceptually ordered (sequential maps increase monotonically in lightness) so the
visual style does not distort the data; diverging data use a symmetric diverging map centred at zero.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import LightSource, LinearSegmentedColormap  # noqa: E402

BG = "#04060c"
PANEL = "#0a0f1c"
GRID = "#1b2840"
TEXT = "#dbe7ff"
CYAN, MAGENTA, LIME, ORANGE, VIOLET, GOLD = "#38f2ff", "#ff3ec9", "#9dff4a", "#ff9a3c", "#9b6bff", "#ffd84a"
NEON = [CYAN, MAGENTA, LIME, ORANGE, VIOLET, GOLD]
W, H, DPI = 12.8, 7.2, 100

AURORA = LinearSegmentedColormap.from_list(
    "aurora", ["#02030a", "#120c3a", "#3a0f7a", "#8a1fa8", "#d6338f", "#ff6b4a", "#ffb43c", "#fff3b0"]
)
ICE = LinearSegmentedColormap.from_list("ice", ["#010208", "#06203f", "#0b4f7d", "#1196b8", "#38f2ff", "#e9fdff"])
NEON_DIVERGING = LinearSegmentedColormap.from_list(
    "neon_div", ["#38f2ff", "#1474b8", "#0a1a3a", "#04060c", "#3a0a2a", "#b8146f", "#ff3ec9"]
)

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": GRID,
        "axes.grid": True,
        "grid.alpha": 0.6,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "savefig.facecolor": BG,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
    }
)


def figure(nrows: int = 1, ncols: int = 1, **kw: Any) -> tuple[Any, Any]:
    fig, ax = plt.subplots(nrows, ncols, figsize=(W, H), dpi=DPI, **kw)
    return fig, ax


def glow(ax: Any, x: np.ndarray, y: np.ndarray, color: str, lw: float = 2.0, label: str | None = None) -> list[Any]:
    """Neon glow: one sharp line on top of three wide translucent ones. Returns the artists (for animation)."""
    arts = [
        ax.plot(x, y, color=color, lw=lw * k, alpha=a, solid_capstyle="round")[0]
        for k, a in ((6, 0.06), (3.5, 0.12), (2, 0.25))
    ]
    arts.append(ax.plot(x, y, color=color, lw=lw, label=label)[0])
    return arts


def set_glow(arts: list[Any], x: np.ndarray, y: np.ndarray) -> None:
    for a in arts:
        a.set_data(x, y)


def header(fig: Any, title: str, subtitle: str = "", source: str = "") -> None:
    fig.text(0.02, 0.955, title, fontsize=20, fontweight="bold", color=CYAN)
    if subtitle:
        fig.text(0.02, 0.915, subtitle, fontsize=12, color=TEXT, alpha=0.85)
    if source:
        fig.text(0.02, 0.012, source, fontsize=9, color=TEXT, alpha=0.6)
    fig.text(0.98, 0.012, "AeroSynapse-Einstein | research use only", fontsize=9, color=MAGENTA, alpha=0.7, ha="right")


def shaded(Z: np.ndarray, cmap: Any, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Depth-gradient facecolours: colormap by value + hill-shade lighting (gives 3D depth)."""
    ls = LightSource(azdeg=315, altdeg=35)
    vmin = float(np.nanmin(Z)) if vmin is None else vmin
    vmax = float(np.nanmax(Z)) if vmax is None else vmax
    rgb = ls.shade(np.nan_to_num(Z, nan=vmin), cmap=cmap, vert_exag=0.6, blend_mode="soft", vmin=vmin, vmax=vmax)
    return np.asarray(rgb)


def style_3d(ax: Any) -> None:
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.04, 0.06, 0.11, 1.0))
        axis._axinfo["grid"]["color"] = (0.11, 0.16, 0.25, 0.8)  # noqa: SLF001
        axis.label.set_color(TEXT)
    ax.tick_params(colors=TEXT, labelsize=8)


def write(fig: Any, update: Callable[[int], Any], frames: int, path: str | Path, fps: int = 30) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=4500, extra_args=["-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20"]
    )
    anim.save(str(path), writer=writer, dpi=DPI)
    plt.close(fig)
    return path


def ease(t: float) -> float:
    """Smooth ease-in-out on [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)
