"""Publication style of the project (Phase 7, Step 7.1): the single source of every figure's look.

Replaces the three styles that grew in Phases 1-6B (the Phase 1 `apply_style`, the Phase 5-6 `inverse_figs.style`
and the papers' interim `paper/common/figstyle.py`, FIGSTYLE_INTERIM); those names now delegate here, so every figure
function and experiment keeps its call. No scientific content depends on the style.

    from aerosynapse.thermal.viz.style import apply_style, save_fig, label, PALETTE, SEQ_CMAP, DIV_CMAP
    apply_style()                         # once, before building figures
    ax.set_xlabel(label("Pe"))            # "Péclet number $\\mathrm{Pe}$"
    save_fig(fig, "p7_example", odir)     # docs/figures/p7_example.{pdf,png} (+ odir copy)

Rules: colour-blind-safe categorical colours in a fixed order (Okabe-Ito); perceptually uniform sequential (viridis)
and diverging (RdBu_r, centred at zero) colormaps; dimensionless symbols and units in axis labels; vector PDF plus
300-dpi PNG, byte-reproducible (no timestamps); white background; text in ink colours, never in a series colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
FIG_DIR = ROOT / "docs" / "figures"

# ---------------------------------------------------------------------------- colours
# Okabe-Ito, fixed order (blue first: the series that is compared against comes first)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]  # legacy order
SEQ_CMAP = "viridis"  # perceptually uniform, sequential (fields, magnitudes)
DIV_CMAP = "RdBu_r"  # diverging (signed errors; centre the norm at 0)
CYC_CMAP = "twilight"  # cyclic (phases, angles)
# role colours used across the project
REF = "#000000"  # analytic / reference solution
SOLVER = "#0072B2"  # verified numerical solver
PINN = "#D55E00"  # physics-informed network
DATA = "#009E73"  # data, measurements, snapshots
# neutral ink and surfaces (text never wears a series colour)
INK, INK2, MUTED = "#111111", "#4a4a4a", "#8a8a8a"
GRID, AXIS, SURF = "#e2e2e2", "#9a9a9a", "#ffffff"

# ---------------------------------------------------------------------------- sizes (Elsevier)
COL_W = 3.5  # single-column width [in] (90 mm)
PAGE_W = 7.2  # double-column width [in] (190 mm)

# ---------------------------------------------------------------------------- labels: symbols and units
SYMBOLS = {
    "Re": ("Reynolds number", r"$\mathrm{Re}$"),
    "Pe": ("Péclet number", r"$\mathrm{Pe}$"),
    "Pr": ("Prandtl number", r"$\mathrm{Pr}$"),
    "Bi": ("Biot number", r"$\mathrm{Bi}$"),
    "Ra": ("Rayleigh number", r"$\mathrm{Ra}$"),
    "Ri": ("Richardson number", r"$\mathrm{Ri}$"),
    "Nu": ("Nusselt number", r"$\mathrm{Nu}$"),
    "x": ("position", r"$x$"),
    "y": ("position", r"$y$"),
    "t": ("time", r"$t$"),
    "theta": ("dimensionless temperature", r"$\theta$"),
    "u": ("velocity", r"$u$"),
    "v": ("velocity", r"$v$"),
    "p": ("pressure", r"$p$"),
    "psi": ("stream function", r"$\psi$"),
    "rel_l2": ("relative error", r"$\|e\|_2/\|u\|_2$"),
    "h": ("grid spacing", r"$h$"),
    "N": ("number of points", r"$N$"),
}


def label(name: str, unit: str | None = None, long: bool = True) -> str:
    """Axis label: '<name> <symbol> [unit]' for known quantities (dimensionless unless a unit is given)."""
    words, sym = SYMBOLS.get(name, (name, ""))
    text = f"{words} {sym}".strip() if long else (sym or words)
    return f"{text} [{unit}]" if unit else text


# ---------------------------------------------------------------------------- style
def rc(context: str = "paper") -> dict:
    """rcParams of the project style. context 'paper': serif text matching LaTeX, small fonts for column widths;
    'report': the same look with larger fonts for screens and the docs gallery."""
    size = {"paper": 8.0, "report": 9.0}[context]
    return {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": size,
        "axes.titlelocation": "left",
        "axes.titleweight": "normal",
        "legend.fontsize": size - 1,
        "xtick.labelsize": size - 1,
        "ytick.labelsize": size - 1,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "image.cmap": SEQ_CMAP,
        "axes.formatter.limits": (-3, 4),  # 7.5e-4 is written 7.5 with a x10^-4 offset, not 0.00075
        "axes.formatter.use_mathtext": True,
        "figure.dpi": 110,
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,  # embedded TrueType: text stays editable and searchable
        "ps.fonttype": 42,
    }


def apply_style(context: str = "paper") -> None:
    """Apply the project style (call once before building figures)."""
    mpl.rcParams.update(rc(context))


def colorbar(fig: plt.Figure, mappable, ax, label: str = "", **kw):
    """Colorbar whose power-of-ten scale goes into the label ('signed error [x 10^-4]') instead of a floating offset
    text above the bar, which collides with panel titles."""
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    cb = fig.colorbar(mappable, ax=ax, **kw)
    lo, hi = mappable.get_clim()
    big = max(abs(lo), abs(hi))
    if big > 0 and (big < 1e-3 or big >= 1e4):
        e = int(np.floor(np.log10(big)))
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 10**e:g}".replace("-", "−")))
        label = f"{label} [$\\times 10^{{{e}}}$]".strip()
    cb.set_label(label)
    return cb


def _labels_overlap(axis, renderer) -> tuple[bool, int]:
    """(do the visible tick labels of one axis overlap, how many are visible)."""
    horiz = axis.axis_name == "x"
    frame = axis.axes.get_window_extent(renderer)
    boxes = []
    for t in axis.get_ticklabels(which="both"):
        if not (t.get_visible() and t.get_text()):
            continue
        b = t.get_window_extent(renderer)
        mid = 0.5 * (b.x0 + b.x1) if horiz else 0.5 * (b.y0 + b.y1)
        if (frame.x0 - 1 <= mid <= frame.x1 + 1) if horiz else (frame.y0 - 1 <= mid <= frame.y1 + 1):
            boxes.append(b)  # only labels that are drawn: inside the axis' extent
    boxes.sort(key=lambda b: b.x0 if horiz else b.y0)
    for a, b in zip(boxes, boxes[1:], strict=False):
        if (b.x0 < a.x1 - 0.5) if horiz else (b.y0 < a.y1 - 0.5):
            return True, len(boxes)
    return False, len(boxes)


def tidy_log_ticks(fig: plt.Figure) -> None:
    """Phase 7.3: on log axes spanning less than about a decade matplotlib labels minor ticks, and the labels collide
    ('2x10^-3 3x10^-3 ...'). Where labels overlap, drop the minor labels; if fewer than two labels remain, label the
    1-2-5 ticks, or only the decades if those still collide."""
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter

    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() != "log":
                continue
            fig.canvas.draw()
            bad, _ = _labels_overlap(axis, renderer)
            if not bad:
                continue
            axis.set_minor_formatter(NullFormatter())
            fig.canvas.draw()
            bad, n = _labels_overlap(axis, renderer)
            if not bad and n >= 2:
                continue
            for subs in ((1.0, 2.0, 5.0), (1.0,)):
                axis.set_major_locator(LogLocator(base=10, subs=subs))
                axis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
                fig.canvas.draw()
                bad, n = _labels_overlap(axis, renderer)
                if not bad:
                    break


def save_fig(fig: plt.Figure, name: str, also_to: Path | None = None, docs: bool = True) -> list[Path]:
    """Save vector PDF + 300-dpi PNG into docs/figures (unless docs=False) and optionally a results folder.
    PDFs carry no creation date, so an unchanged figure is byte-identical when regenerated. Overlapping log-axis
    tick labels are thinned first (tidy_log_ticks)."""
    tidy_log_ticks(fig)
    paths = []
    for d in ([FIG_DIR] if docs else []) + ([also_to] if also_to else []):
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            p = d / f"{name}.{ext}"
            meta = {"CreationDate": None} if ext == "pdf" else {}
            fig.savefig(p, bbox_inches="tight", metadata=meta)
            paths.append(p)
    plt.close(fig)
    return paths
