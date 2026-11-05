"""Shared figure style: serif fonts matching LaTeX, Okabe–Ito colour-blind-safe palette, 85/175 mm widths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
SINGLE, DOUBLE = 85 / 25.4, 175 / 25.4
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.prop_cycle": matplotlib.cycler(color=OKABE),
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "svg.hashsalt": "as",
    }
)


def load(exp: str, name: str = "metrics.json") -> Any | None:
    p = RES / exp / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig: Any, name: str) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (
        ("pdf", {"metadata": {"CreationDate": None}}),
        ("svg", {"metadata": {"Date": None}}),
        ("png", {"dpi": 600}),
    ):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, **kw)
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


def save_html(fig: Any, name: str) -> str:
    p = OUT / f"{name}.html"
    fig.write_html(str(p), include_plotlyjs="cdn", full_html=True)
    return str(p.relative_to(ROOT))
