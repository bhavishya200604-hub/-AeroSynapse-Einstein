"""Paper figure style: the project publication style of src/viz/style.py (Phase 7, Step 7.1).

Was the INTERIM style of Phase 6B (patch P4, FIGSTYLE_INTERIM); since Step 7.1 it only re-exports the project style, so
the paper figures and the project figures share one look. No scientific content depends on it.
"""

from __future__ import annotations

from aerosynapse.thermal.viz.style import COL_W, DIV_CMAP, PAGE_W, PALETTE, SEQ_CMAP, apply_style, save_fig  # noqa: F401

SEQ, DIV = SEQ_CMAP, DIV_CMAP


def apply() -> None:
    apply_style("paper")


def save(fig, path):
    """Vector PDF (for LaTeX) and a 300-dpi PNG preview, byte-reproducible."""
    from pathlib import Path

    path = Path(path)
    return save_fig(fig, path.stem, also_to=path.parent, docs=False)
