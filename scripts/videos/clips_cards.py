"""Title / section cards (animated particle field) and a CLI clip that replays REAL captured terminal output."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from _base import CACHE, CLIPS, ROOT

from aerosynapse.viz import neon as N

FPS = 30


def card(name: str, title: str, lines: list[str], seconds: float = 4.0, accent: str = N.CYAN) -> str:
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    n = 420
    P = rng.random((n, 3))
    V = (rng.random((n, 2)) - 0.5) * 0.004
    fig = N.plt.figure(figsize=(N.W, N.H), dpi=N.DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    sc = ax.scatter(P[:, 0], P[:, 1], s=2 + 30 * P[:, 2] ** 3, c=P[:, 2], cmap=N.ICE, alpha=0.7)
    t1 = ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=40, fontweight="bold", color=accent, alpha=0)
    ts = [
        ax.text(0.5, 0.45 - 0.06 * i, s, ha="center", fontsize=16, color=N.TEXT, alpha=0) for i, s in enumerate(lines)
    ]
    bar = ax.plot([0.5, 0.5], [0.52, 0.52], color=N.MAGENTA, lw=3)[0]
    frames = int(seconds * FPS)

    def frame(k: int) -> None:
        P[:, :2] = (P[:, :2] + V) % 1.0
        sc.set_offsets(P[:, :2])
        a = N.ease(k / 25)
        t1.set_alpha(a)
        w = 0.32 * N.ease((k - 10) / 25)
        bar.set_data([0.5 - w, 0.5 + w], [0.52, 0.52])
        for i, t in enumerate(ts):
            t.set_alpha(N.ease((k - 25 - 8 * i) / 20))

    return str(N.write(fig, frame, frames, CLIPS / f"{name}.mp4", FPS))


def cards() -> list[str]:
    return [
        card(
            "00_title",
            "AeroSynapse–Einstein",
            [
                "Physics-aware AI for computational engineering",
                "verified solvers · PINNs · neural operators · UQ · optimisation",
                "RESEARCH USE ONLY — validation tier 1/5 · every number from results/",
            ],
            6.0,
        ),
        card(
            "09_sec_sim",
            "1 · Simulations",
            ["classical solvers computed live", "heat · convection–diffusion · Burgers · wave · Navier–Stokes"],
        ),
        card(
            "19_sec_ml",
            "2 · Scientific machine learning",
            ["PINN training · neural operator · design optimisation"],
            3.5,
            N.MAGENTA,
        ),
        card(
            "29_sec_graphs",
            "3 · Results, animated",
            ["verification · core study · generalisation · Pareto · trust · validation"],
            3.5,
            N.LIME,
        ),
        card(
            "39_sec_cli", "4 · Running the software", ["command line, then the five-workspace workbench"], 3.5, N.GOLD
        ),
        card(
            "99_end",
            "Honest summary",
            [
                "solvers verified & validated against published benchmarks",
                "adaptive collocation: no significant gain (negative result)",
                "surrogates fast in-range, unreliable outside → trust engine falls back",
                "github.com/bhavishya200604-hub/-AeroSynapse-Einstein",
            ],
            7.0,
            N.GOLD,
        ),
    ]


def capture_cli() -> list[tuple[str, str]]:
    cmds = [
        ["aerosynapse", "--help"],
        ["aerosynapse", "doctor"],
        ["aerosynapse", "run", "configs/workflow_quick.yaml", "--results", str(CACHE / "cli_results")],
    ]
    out = []
    for c in cmds:
        r = subprocess.run(c, cwd=ROOT, capture_output=True, text=True, timeout=900)
        txt = r.stdout + r.stderr
        if c[1] == "run":  # keep the head and the verdict lines of the long JSON
            lines = txt.splitlines()
            keep = (
                lines[:6]
                + ["  ..."]
                + [
                    ln
                    for ln in lines
                    if any(
                        k in ln
                        for k in (
                            '"verdict"',
                            '"rel_l2"',
                            '"mass_drift"',
                            '"decision"',
                            "report_html",
                            '"trusted_feasible"',
                        )
                    )
                ]
            )
            txt = "\n".join(keep)
        out.append((" ".join(c[:3]) if c[1] == "run" else " ".join(c), txt))
    return out


def cli() -> str:
    sessions = capture_cli()
    script: list[tuple[str, str]] = []
    for cmd, txt in sessions:
        script.append(("cmd", "$ " + cmd))
        for ln in txt.splitlines()[:26]:
            script.append(("out", ln[:118]))
        script.append(("out", ""))
    fig = N.plt.figure(figsize=(N.W, N.H), dpi=N.DPI)
    N.header(
        fig,
        "Running the software from the command line",
        "real commands, real captured output (this machine)",
        "aerosynapse CLI · output captured live while rendering this clip",
    )
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.83])
    ax.set_facecolor("#02040a"), ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_xticks([]), ax.set_yticks([])
    texts = [ax.text(0.01, 0.97 - 0.034 * i, "", family="DejaVu Sans Mono", fontsize=10, va="top") for i in range(28)]
    timeline: list[tuple[int, int]] = []  # (line index, chars shown)
    for i, (kind, s) in enumerate(script):
        if kind == "cmd":
            timeline += [(i, c) for c in range(0, len(s) + 1, 2)] + [(i, len(s))] * 12
        else:
            timeline.append((i, len(s)))
    timeline += [(len(script) - 1, len(script[-1][1]))] * 45

    def frame(k: int) -> None:
        li, nch = timeline[k]
        start = max(0, li - 27)
        for j, t in enumerate(texts):
            idx = start + j
            if idx > li:
                t.set_text("")
                continue
            kind, s = script[idx]
            t.set_text(s[:nch] + ("█" if idx == li and kind == "cmd" else "") if idx == li else s)
            t.set_color(
                N.LIME if kind == "cmd" else (N.MAGENTA if "RESEARCH USE ONLY" in s or "ACCEPT" in s else N.TEXT)
            )

    return str(N.write(fig, frame, len(timeline), CLIPS / "40_cli.mp4", FPS))


if __name__ == "__main__":
    for c in cards():
        print(c)
    print(cli())
    sys.exit(0)
