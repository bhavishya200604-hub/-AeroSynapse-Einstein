"""Phase 7, Step 7.1: specimen of the project publication style (src/viz/style.py) — palette, role colours, line and
marker conventions, sequential / diverging colormaps on a field, labels with symbols and units. Analytic data only.

    python -m aerosynapse.thermal.experiments.p7_1_style            -> docs/figures/p7_1_style_specimen.{pdf,png}
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from aerosynapse.thermal.viz import style as S


def specimen() -> plt.Figure:
    fig, axs = plt.subplots(2, 2, figsize=(S.PAGE_W, 5.2))
    # (a) categorical palette in its fixed order
    ax = axs[0, 0]
    x = np.linspace(0, 1, 50)
    for i, c in enumerate(S.PALETTE):
        ax.plot(x, np.sin(np.pi * x) * (1 - 0.1 * i), color=c, label=f"series {i + 1}")
    ax.set(title="(a) categorical palette (Okabe-Ito, fixed order)", xlabel=S.label("x"), ylabel=S.label("theta"))
    ax.legend(ncols=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    # (b) role colours: reference, solver, PINN, data
    ax = axs[0, 1]
    pe = 10.0
    xs = np.linspace(0, 1, 200)
    exact = (np.exp(pe * xs) - 1) / (np.exp(pe) - 1)
    xg = np.linspace(0, 1, 11)
    ax.plot(xs, exact, color=S.REF, lw=1.0, label="analytic")
    ax.plot(xg, (np.exp(pe * xg) - 1) / (np.exp(pe) - 1), "o", color=S.SOLVER, label="solver")
    ax.plot(xs, exact + 0.02 * np.sin(6 * np.pi * xs), "--", color=S.PINN, label="PINN")
    ax.plot(xg[1::3], (np.exp(pe * xg[1::3]) - 1) / (np.exp(pe) - 1), "s", mfc="none", color=S.DATA, label="data")
    ax.set(title=f"(b) role colours (steady problem, Pe = {pe:g})", xlabel=S.label("x"), ylabel=S.label("theta"))
    ax.legend()
    # (c) sequential field
    yy, xx = np.mgrid[0:1:120j, 0:1:120j]
    psi = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    ax = axs[1, 0]
    im = ax.pcolormesh(xx, yy, psi, cmap=S.SEQ_CMAP, shading="auto")
    ax.contour(xx, yy, psi, levels=6, colors="white", linewidths=0.4)
    S.colorbar(fig, im, ax, S.label("psi"))
    ax.set(title="(c) sequential colormap (field)", xlabel=S.label("x"), ylabel=S.label("y"), aspect="equal")
    ax.grid(False)
    # (d) diverging, centred at zero
    err = np.sin(2 * np.pi * xx) * np.cos(np.pi * yy) * 1e-3
    ax = axs[1, 1]
    lim = np.abs(err).max()
    im = ax.pcolormesh(xx, yy, err, cmap=S.DIV_CMAP, vmin=-lim, vmax=lim, shading="auto")
    S.colorbar(fig, im, ax, "signed error")
    ax.set(
        title="(d) diverging colormap (signed error, centred at 0)",
        xlabel=S.label("x"),
        ylabel=S.label("y"),
        aspect="equal",
    )
    ax.grid(False)
    return fig


def main() -> None:
    S.apply_style()
    paths = S.save_fig(specimen(), "p7_1_style_specimen")
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
