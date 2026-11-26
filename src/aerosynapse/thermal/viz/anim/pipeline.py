"""The project as a system: an animated map of the phases and how results flow between them.

Every number on the map is counted from the repository when the animation is rendered (results folders, figures,
unit tests, decision-log entries, the verification gate), so the map updates itself as the project grows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from aerosynapse.thermal.viz import style as S
from aerosynapse.thermal.viz.anim.core import Scene

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
# (key, title, what it does, result-folder / figure prefixes, position)
STAGES = [
    ("0", "Setup and audit", "repository, configuration, reproducibility rules", ("p0",), (0.08, 0.76)),
    (
        "1",
        "Verified solvers",
        "finite differences, spectral, finite volume, Navier-Stokes; manufactured solutions",
        ("p1",),
        (0.20, 0.70),
    ),
    ("2", "Data generation", "synthetic data sets, sensors and noise from the solvers", ("p2",), (0.48, 0.76)),
    ("3", "Baseline PINN", "physics-informed networks for the 1D problems", ("p3",), (0.68, 0.76)),
    (
        "4",
        "PINN methods and 2D",
        "architectures, loss weights, sampling, parametric 2D, hybrid data, ablation",
        ("p4",),
        (0.62, 0.70),
    ),
    (
        "5",
        "Inverse identification",
        "identifiability, least squares versus PINN, robustness, uncertainty",
        ("p5",),
        (0.34, 0.30),
    ),
    (
        "6",
        "Analysis",
        "scaling, uncertainty, extrapolation, sensitivity, diagnostics, validation, cost",
        ("p6_", "p6b"),
        (0.76, 0.70),
    ),
    ("6B", "Two research papers", "pre-registered gap runs, Word / PDF / LaTeX papers", ("p6b",), (0.68, 0.28)),
    ("7", "Visualisation", "one publication style, figure catalogue, animations", ("p7",), (0.9, 0.28)),
]
EDGES = [
    ("0", "1"),
    ("1", "2"),
    ("2", "3"),
    ("3", "4"),
    ("4", "6"),
    ("1", "5"),
    ("2", "5"),
    ("5", "6"),
    ("6", "6B"),
    ("6", "7"),
    ("6B", "7"),
]
PATH = ["0", "1", "2", "3", "4", "6", "1", "5", "6", "6B", "7"]  # the order in which the animation walks the map
# box centres (axes units): top row = the forward path to the methods, bottom row = inverse, analysis, papers, figures
POS = {
    "0": (0.08, 0.76),
    "1": (0.28, 0.76),
    "2": (0.48, 0.76),
    "3": (0.68, 0.76),
    "4": (0.89, 0.76),
    "5": (0.18, 0.28),
    "6": (0.44, 0.28),
    "6B": (0.68, 0.28),
    "7": (0.90, 0.28),
}


def counts() -> dict[str, dict]:
    """Per stage: experiments (results folders) and figures; global: tests, decisions, verification gate."""
    res = [p.name for p in (ROOT / "results").iterdir() if p.is_dir()] if (ROOT / "results").exists() else []
    figs = [p.stem for p in (ROOT / "docs/figures").glob("*.png")]
    out = {}
    for key, _, _, prefixes, _ in STAGES:
        if key == "6":
            r = [n for n in res if n.startswith("p6_")]
            f = [n for n in figs if n.startswith("p6_")]
        elif key == "6B":
            r = [n for n in res if n.startswith("p6b")]
            f = [n for n in figs if n.startswith("p6b")]
        else:
            r = [n for n in res if any(n.startswith(px + "_") for px in prefixes)]
            f = [n for n in figs if any(n.startswith(px + "_") for px in prefixes)]
        out[key] = {"experiments": len(r), "figures": len(f)}
    tests = sum(
        len(re.findall(r"^def test_", p.read_text(encoding="utf-8"), re.M)) for p in __import__("aerosynapse.thermal._home", fromlist=["HOME"]).THERMAL_TESTS.rglob("test_*.py")
    )
    dec = len(re.findall(r"^\| D\d+ ", (ROOT / "docs/DECISIONS.md").read_text(encoding="utf-8"), re.M))
    gate = {}
    f = ROOT / "results/p1_7_verification_smoke/metrics.json"
    if f.exists():
        m = json.loads(f.read_text(encoding="utf-8"))
        gate = {"pass": m.get("gate_pass"), "total": m.get("gate_total")}
    papers = [p.name for p in (ROOT / "paper/dist").glob("paper*.docx")] if (ROOT / "paper/dist").exists() else []
    out["_global"] = {
        "tests": tests,
        "decisions": dec,
        "verification": gate,
        "papers": papers,
        "experiments": len([n for n in res if re.match(r"p\d", n)]),
        "figures": len(figs),
    }
    return out


def scene_pipeline(p: dict, cache: Path) -> Scene:
    c = counts()
    g = c["_global"]
    pos = dict(POS)
    info = {k: (t, d) for k, t, d, _, _ in STAGES}
    legs = list(zip(PATH[:-1], PATH[1:], strict=True))
    n = int(p["frames"])
    per_leg = n // (len(legs) + 1)

    W, H = 0.16, 0.26  # box size in axes units

    def border(c0, c1):
        """Point where the segment from box centre c0 towards c1 leaves box c0."""
        (x0, y0), (x1, y1) = c0, c1
        dx, dy = x1 - x0, y1 - y0
        t = min(W / 2 / abs(dx) if dx else np.inf, H / 2 / abs(dy) if dy else np.inf)
        return x0 + dx * t, y0 + dy * t

    def make():
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        fig = plt.figure(figsize=(11.0, 6.2))
        ax = fig.add_axes([0.01, 0.24, 0.98, 0.68])
        ax.set(xlim=(-0.01, 1.0), ylim=(0.08, 0.96))
        ax.axis("off")
        boxes = {}
        for k, (x, y) in pos.items():
            b = FancyBboxPatch(
                (x - W / 2, y - H / 2),
                W,
                H,
                boxstyle="round,pad=0.006,rounding_size=0.018",
                fc="white",
                ec=S.AXIS,
                lw=1.0,
                zorder=3,
            )
            ax.add_patch(b)
            ax.text(x, y + 0.075, f"Phase {k}", ha="center", va="center", fontsize=8, color=S.INK2, zorder=4)
            ax.text(
                x,
                y + 0.012,
                info[k][0].replace(" and ", "\nand ") if len(info[k][0]) > 18 else info[k][0],
                ha="center",
                va="center",
                fontsize=8.6,
                color=S.INK,
                zorder=4,
                linespacing=1.1,
            )
            ax.text(
                x,
                y - 0.078,
                f"{c[k]['experiments']} runs  ·  {c[k]['figures']} figures",
                ha="center",
                va="center",
                fontsize=7,
                color=S.MUTED,
                zorder=4,
            )
            boxes[k] = b
        ends = {}
        for a, b in EDGES:
            p0, p1 = border(pos[a], pos[b]), border(pos[b], pos[a])
            ends[(a, b)] = (p0, p1)
            cs = "arc3"
            # an arrow that would pass behind a box in the same row is drawn as an arc underneath it
            between = [
                k
                for k, (x, y) in pos.items()
                if k not in (a, b)
                and abs(y - pos[a][1]) < 1e-9
                and min(pos[a][0], pos[b][0]) < x < max(pos[a][0], pos[b][0])
            ]
            if between and abs(pos[a][1] - pos[b][1]) < 1e-9:
                p0, p1 = (pos[a][0], pos[a][1] - H / 2), (pos[b][0], pos[b][1] - H / 2)
                cs = "arc3,rad=0.16"
            ax.add_patch(
                FancyArrowPatch(
                    p0,
                    p1,
                    arrowstyle="-|>",
                    mutation_scale=10,
                    lw=1.0,
                    color="#bdbdbd",
                    shrinkA=2,
                    shrinkB=2,
                    zorder=1,
                    connectionstyle=cs,
                    clip_on=False,
                )
            )
        (trail,) = ax.plot([], [], "-", lw=3, color=S.PINN, alpha=0.35, zorder=2, solid_capstyle="round")
        (tok,) = ax.plot([], [], "o", ms=10, color=S.PINN, mec="white", mew=1.5, zorder=2.5)
        tx = fig.add_axes([0.04, 0.02, 0.92, 0.18])
        tx.axis("off")
        head = tx.text(0, 0.80, "", fontsize=12.5, color=S.INK, va="center")
        body = tx.text(0, 0.45, "", fontsize=10, color=S.INK2, va="center")
        tx.text(
            0,
            0.04,
            (
                f"Whole project: {g['experiments']} experiments, {g['figures']} figures, "
                f"{g['tests']} unit tests, {g['decisions']} logged decisions"
                + (
                    f"; solver verification {g['verification']['pass']} / {g['verification']['total']} checks"
                    if g.get("verification")
                    else ""
                )
                + (f"; papers: {', '.join(g['papers'])}" if g.get("papers") else "")
            ),
            fontsize=8,
            color=S.MUTED,
            va="bottom",
        )
        fig.suptitle(
            "How the project works: from verified solvers to physics-informed networks, inverse problems and papers",
            fontsize=11.5,
        )

        def update(i):
            k = min(i // per_leg, len(legs))
            if k >= len(legs):
                a = b = PATH[-1]
                f = 1.0
            else:
                a, b = legs[k]
                f = (i % per_leg) / per_leg
            if a == b:
                tok.set_data([], [])
                trail.set_data([], [])
            else:
                if (a, b) in ends:
                    (x0, y0), (x1, y1) = ends[(a, b)]
                elif (b, a) in ends:
                    (x0, y0), (x1, y1) = ends[(b, a)][::-1]
                else:  # a jump back along the path (no dependency arrow): drawn dashed
                    (x0, y0), (x1, y1) = border(pos[a], pos[b]), border(pos[b], pos[a])
                trail.set_linestyle("-" if ((a, b) in ends or (b, a) in ends) else (0, (2, 2)))
                s_ = 0.5 - 0.5 * np.cos(np.pi * f)  # ease in / out
                tok.set_data([x0 + (x1 - x0) * s_], [y0 + (y1 - y0) * s_])
                trail.set_data([x0, x0 + (x1 - x0) * s_], [y0, y0 + (y1 - y0) * s_])
            here = b if f > 0.5 else a
            for kk, bx in boxes.items():
                on = kk == here
                bx.set_edgecolor(S.PINN if on else S.AXIS)
                bx.set_linewidth(2.2 if on else 1.0)
                bx.set_facecolor("#fff4ec" if on else "white")
            head.set_text(f"Phase {here}: {info[here][0]}")
            body.set_text(info[here][1])

        return fig, update

    return Scene(
        "pipeline",
        "How the project works",
        "An animated map of the project's phases and how results flow between them: verified solvers feed "
        "the data, the networks and the classical inverse estimator; the analyses feed the papers and the "
        "visualisations. The counts are read from the repository when the animation is rendered.",
        n,
        p["fps"],
        make,
        "repository (results/, docs/figures/, tests/, docs/DECISIONS.md)",
        category="project",
        meta={"counts": c},
    )


SCENES = {"pipeline": scene_pipeline}
