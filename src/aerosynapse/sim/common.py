"""Shared plotting style and output helpers of the simulations.

The chart style is a user preference (``settings.json`` in the project folder, written by Studio's Preferences
window, or the ``AEROSYNAPSE_PLOT_STYLE`` environment variable):

* ``plain`` (default): white paper, thin dark lines, muted colours, light grid; reads like a careful hand-made chart.
* ``dark``: the dark-blue viewer style with bright colours.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def settings() -> dict[str, Any]:
    try:
        return dict(json.loads((ROOT / "settings.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


STYLE = os.environ.get("AEROSYNAPSE_PLOT_STYLE") or str(settings().get("plot_style", "plain"))
if STYLE == "dark":
    BG, PANEL, GRID, TEXT = "#0a1424", "#0f1d33", "#1f3354", "#d6e4f5"
    NEON = ["#22d3ee", "#f472b6", "#a3e635", "#fbbf24", "#a78bfa", "#fb7185"]
    INK = "#0a1424"
else:  # the thermal project's publication style: white paper, serif text, Okabe-Ito colours
    BG, PANEL, GRID, TEXT = "#ffffff", "#ffffff", "#e2e2e2", "#111111"
    NEON = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
    INK = "#ffffff"


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    from cycler import cycler

    common = {"figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG, "axes.edgecolor": TEXT if STYLE != "dark" else GRID,
              "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT, "grid.color": GRID, "axes.grid": True,
              "axes.titlesize": 10, "font.size": 9, "legend.edgecolor": GRID, "axes.prop_cycle": cycler(color=NEON), "figure.dpi": 110,
              "savefig.bbox": "tight", "figure.constrained_layout.use": True}
    if STYLE == "dark":
        common.update({"grid.linewidth": 0.6, "axes.titleweight": "bold", "legend.facecolor": PANEL, "lines.linewidth": 1.8})
        p.rcParams.update(common)
    else:
        from aerosynapse.thermal.viz import style as TS

        rc = TS.rc("report")
        rc.update({"savefig.dpi": 150, "axes.prop_cycle": cycler(color=NEON), "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5})
        p.rcParams.update(rc)
    return p


def save(fig: Any, out: Path, name: str) -> str:
    f = out / f"{name}.png"
    fig.savefig(f)
    plt().close(fig)
    return f.name


def finish(out: Path, title: str, lines: list[str], res: dict[str, Any]) -> dict[str, Any]:
    (out / "summary.txt").write_text(title + "\n" + "=" * len(title) + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")

    def clean(o: Any) -> Any:
        if isinstance(o, float):
            return o if o == o and abs(o) != float("inf") else None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, complex):
            return [o.real, o.imag]
        if hasattr(o, "tolist"):
            return clean(o.tolist())
        return o

    (out / "result.json").write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
    return res
