"""Phase 6 gate figures: one overview panel per step (from the saved metrics), a 3D map of where a PINN pays off
(error x speed-up x break-even), the traceability matrix (steps x gate checks) and a 3D completeness chart
(steps x artefact kinds)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from aerosynapse.thermal.viz.inverse_figs import AXIS, GRID, INK, INK2, SURF, style  # noqa: F401  (style re-exported)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2", "#b04a1e", "#d4a017"]
PASS, FAIL, NA = "#1baf7a", "#d1495b", "#c8c8c8"


def _off(ax, msg: str = "no data"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=INK2, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def overview(m: dict) -> plt.Figure:
    """2 x 4 panels, one per step: 6.1 equal-error cost, 6.2 reliability, 6.3 / 6.6b error vs distance, 6.4 Sobol ST,
    6.4b predicted cost vs machine, 6.5 / 6.6b gradient conflict, 6.6 solver vs references, 6.7 break-even."""
    fig, axs = plt.subplots(2, 4, figsize=(20.0, 9.0))
    ax = axs[0][0]
    if m.get("6.1"):
        k = 0
        for prob, ee in m["6.1"]["equal_error"].items():
            r = [e for e in ee if e.get("pinn") and e.get("fd")]
            ax.loglog(
                [e["target"] for e in r],
                [e["pinn"]["wall_s"] / e["fd"]["wall_s"] for e in r],
                "-o",
                color=C[k],
                mec=SURF,
                label=f"{prob}: PINN / FD",
            )
            r = [e for e in ee if e.get("pinn") and e.get("spectral")]
            ax.loglog(
                [e["target"] for e in r],
                [e["pinn"]["wall_s"] / e["spectral"]["wall_s"] for e in r],
                "--s",
                color=C[k],
                mec=SURF,
                label=f"{prob}: PINN / Chebyshev",
            )
            k += 1
        ax.invert_xaxis()
        ax.set(xlabel="target relative L2 error", ylabel="PINN cost / solver cost")
        ax.legend(fontsize=6.5)
    else:
        _off(ax)
    ax.set_title("6.1 equal-error cost (solver cheaper above 1)", fontsize=9)
    ax = axs[0][1]
    if m.get("6.2"):
        ax.plot([0, 1], [0, 1], color=INK, lw=0.8)
        k = 0
        for prob, an in m["6.2"]["analysis"].items():
            ens = max(
                (k for k in an["methods"] if k.startswith("ensemble_M")),
                key=lambda k: int(k.split("M")[1]),
                default=None,
            )
            mcd = max((k for k in an["methods"] if k.startswith("mcd_")), default=None)
            for meth, ls in ((ens, "-"), (mcd, ":")):
                if meth is None:
                    continue
                cv = an["methods"].get(meth, {}).get("coverage")
                if cv:
                    lv = sorted(float(x) for x in cv)
                    ax.plot(
                        lv,
                        [cv[str(x)] if str(x) in cv else cv[f"{x:g}"] for x in lv],
                        ls,
                        marker="o",
                        ms=4,
                        color=C[k],
                        label=f"{prob} {meth.split('_')[0]}",
                    )
            k += 1
        ax.set(xlabel="nominal coverage", ylabel="empirical coverage", xlim=(0.4, 1), ylim=(0, 1.02))
        ax.legend(fontsize=6.5)
    else:
        _off(ax)
    ax.set_title("6.2 calibration: ensemble vs MC-dropout", fontsize=9)
    ax = axs[0][2]
    k = 0
    if m.get("6.3"):
        for prob, b in m["6.3"]["parametric"].items():
            d, e = np.asarray(b["distance"]), np.asarray(b["values"]["rel_l2"])
            ax.semilogy(d, e, "o", ms=3, color=C[k], alpha=0.7, label=f"{prob} (1D, 6.3)")
            k += 1
    if m.get("6.6b"):
        x = (m["6.6b"]["analysis"].get("D_graetz") or {}).get("extrapolation")
        if x:
            ax.semilogy(
                [b["d"] for b in x["by_distance"]],
                [b["rel_err_median"] for b in x["by_distance"]],
                "-s",
                color=C[k],
                mec=SURF,
                label="Graetz 2D median (6.6b)",
            )
    if k or m.get("6.6b"):
        ax.set(xlabel="distance outside the training range [decades]", ylabel="relative error")
        ax.legend(fontsize=6.5)
    else:
        _off(ax)
    ax.set_title("6.3 / 6.6b: error outside the range", fontsize=9)
    ax = axs[0][3]
    if m.get("6.4"):
        names = ["Re", "Pr", "Bi"]
        x = np.arange(3)
        cases = list(m["6.4"]["cases"])
        w = 0.8 / (2 * len(cases))
        for i, c in enumerate(cases):
            d = m["6.4"]["cases"][c]
            ax.bar(x + (2 * i) * w - 0.4 + w / 2, d["solver"]["ST"], w, color=C[i], label=f"{c} solver")
            ax.bar(
                x + (2 * i + 1) * w - 0.4 + w / 2,
                d["pinn"]["ensemble"]["ST"],
                w,
                color=C[i],
                alpha=0.45,
                hatch="//",
                label=f"{c} PINN",
            )
        ax.set_xticks(x, names)
        ax.set(ylabel="total Sobol index ST")
        ax.legend(fontsize=6.5)
    else:
        _off(ax)
    ax.set_title("6.4 Sobol: solver vs PINN surrogate", fontsize=9)
    ax = axs[1][0]
    if m.get("6.4b"):
        g = m["6.4b"]["grid"]
        ram = (g.get("machine") or {}).get("total_memory")
        cs = [c for c in g if isinstance(g[c], dict) and "prediction" in g[c]]
        mem = [g[c]["prediction"]["memory_bytes"][0] / 2**30 for c in cs]
        ax.bar(range(len(cs)), mem, color=C[: len(cs)])
        if ram:
            ax.axhline(ram / 2**30, color=FAIL, lw=1.2, ls=(0, (4, 2)), label=f"this machine {ram / 2**30:.1f} GiB")
        for i, c in enumerate(cs):
            ax.text(
                i,
                mem[i],
                f"{mem[i]:.1f} GiB\n{g[c]['prediction']['time_per_solve_s'][0] / 3600:.2f} h/solve",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_xticks(range(len(cs)), [f"{c}\n(target {g[c]['prediction']['target']})" for c in cs], fontsize=7)
        ax.set(yscale="log", ylabel="predicted memory [GiB]")
        ax.legend(fontsize=6.5)
    else:
        _off(ax)
    ax.set_title("6.4b: predicted 1024 grid vs the machine", fontsize=9)
    ax = axs[1][1]
    labs, cos, cols = [], [], []
    if m.get("6.5"):
        for r in m["6.5"].get("conflict", []):
            labs.append(f"{r['case']}/{r['variant']}")
            cos.append(r["final"]["cosine"])
            cols.append(C[0])
    if m.get("6.6b"):
        for c, d in m["6.6b"]["cases"].items():
            for r in d.get("diagnostics") or []:
                labs.append(f"{c}/{r['member']}")
                cos.append(r["final"]["conflict"]["unweighted"]["cosine"])
                cols.append(C[1])
    if labs:
        ax.barh(range(len(labs)), cos, color=cols)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(range(len(labs)), labs, fontsize=5.5)
        ax.set(xlabel="cosine(g_PDE, g_BC) at the final weights", xlim=(-1.05, 1.05))
        ax.invert_yaxis()
    else:
        _off(ax)
    ax.set_title("6.5 (1D, blue) / 6.6b (2D, orange): gradient conflict", fontsize=9)
    ax = axs[1][2]
    if m.get("6.6"):
        a = m["6.6"]["analysis"]
        gr = sorted(a["graetz"], key=lambda e: e["Pe"])
        ax.semilogx(
            [e["Pe"] for e in gr],
            [100 * e["dev_local_median"] for e in gr],
            "-o",
            color=C[0],
            mec=SURF,
            label="Graetz: |solver − series| local Nu (vs Pe)",
        )
        nat = [e for e in a["natural"].values() if abs(e["Pr"] - 0.71) < 1e-9]
        nat = sorted(nat, key=lambda e: e["Ra"])
        ax.semilogx(
            [e["Ra"] for e in nat],
            [100 * abs(e["dev_BP"]) for e in nat],
            "-^",
            color=C[1],
            mec=SURF,
            label="cavity: |solver − Berkovsky–Polevikov| (vs Ra)",
        )
        dv = [e for e in nat if "dev_dVD" in e]
        ax.semilogx(
            [e["Ra"] for e in dv],
            [100 * abs(e["dev_dVD"]) for e in dv],
            "s",
            color=C[2],
            ms=8,
            mec=SURF,
            label="cavity: |solver − de Vahl Davis|",
        )
        ax.set(xlabel="Pe (Graetz) / Ra (cavity)", ylabel="deviation [%]", yscale="log")
        ax.legend(fontsize=6.5)
    else:
        _off(ax)
    ax.set_title("6.6: solver vs series, benchmark, correlation", fontsize=9)
    ax = axs[1][3]
    if m.get("6.7"):
        a = {k: v for k, v in m["6.7"]["analysis"].items() if not k.startswith("_")}
        cs = list(a)
        be = [a[c].get("break_even_ensemble") for c in cs]
        bq = [a[c].get("break_even_equal_accuracy") for c in cs]
        x = np.arange(len(cs))
        top = max([v for v in be + bq if v] or [10])
        for i in range(len(cs)):
            for j, (v, col, lab) in enumerate(((be[i], C[1], "case grid"), (bq[i], C[2], "equal accuracy"))):
                xx = x[i] + (j - 0.5) * 0.38
                if v:
                    ax.bar(xx, v, 0.36, color=col, label=lab if i == 0 else None)
                else:
                    ax.text(xx, 1.5, "never", rotation=90, ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x, cs, fontsize=7)
        ax.set(yscale="log", ylabel="break-even queries N*", ylim=(1, top * 5))
        ax.legend(fontsize=6.5)
    else:
        _off(ax)
    ax.set_title("6.7: break-even vs the solver", fontsize=9)
    return fig


def landscape3d(m67: dict) -> plt.Figure:
    """3D: every case as (log10 PINN error, log10 speed-up per query, log10 break-even N*); stems to the floor; the
    plane N* = 1e3 marks 'pays off only for many queries'. Cases with no break-even sit at the top as open markers."""
    a = {k: v for k, v in m67["analysis"].items() if not k.startswith("_") and v.get("speedup_per_query")}
    fig = plt.figure(figsize=(8.5, 6.8))
    ax = fig.add_subplot(projection="3d")
    zs = [np.log10(v["break_even_ensemble"]) for v in a.values() if v.get("break_even_ensemble")]
    ztop = (max(zs) if zs else 3) + 1.5
    for i, (c, v) in enumerate(a.items()):
        x, y = np.log10(v["pinn_error"]), np.log10(v["speedup_per_query"])
        nb = v.get("break_even_ensemble")
        z = np.log10(nb) if nb else ztop
        ax.plot([x, x], [y, y], [0, z], color=C[i % len(C)], lw=1.0)
        ax.scatter(
            [x],
            [y],
            [z],
            s=90,
            color=C[i % len(C)],
            edgecolor=INK,
            depthshade=False,
            marker="o" if nb else "v",
            label=f"{c}: " + (f"N* {nb:,.0f}" if nb else "never"),
        )
        ax.text(x, y, z + 0.15, c, fontsize=8)
    X, Y = np.meshgrid(np.linspace(-3, 0.5, 2), np.linspace(-1, 3, 2))
    ax.plot_surface(X, Y, np.full_like(X, 3.0), color=NA, alpha=0.25)
    ax.set(xlabel="log10 PINN error (median)", ylabel="log10 speed-up per query", zlabel="log10 break-even N*")
    ax.set_title("Where a PINN surrogate pays off (grey plane: 1 000 queries; ▼ = never)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.view_init(elev=22, azim=-135)
    return fig


CHECKS = ["run ok", "table", "figures exist", "registry IDs", "decisions", "unit tests", "audit"]


def _matrix(g: dict) -> np.ndarray:
    inv = g["inventory"]
    aud = g["audit"]
    M = np.zeros((len(inv), len(CHECKS)))
    for i, r in enumerate(inv):
        au = [a for a in aud if a["step"] == r["step"] and a["status"] != "NO DATA"]
        vals = [
            r["found"] and r["status"] == "ok",
            r["table_exists"],
            r["n_figures"] > 0 and not r["figures_missing"],
            bool(r["registry_ids"]),
            bool(r["decisions"]),
            bool(r["tests"]),
            (all(a["status"] == "FOUND" for a in au)) if au else None,
        ]
        M[i] = [np.nan if v is None else float(v) for v in vals]
    return M


def traceability(g: dict) -> plt.Figure:
    """Steps x gate checks: pass (green) / fail (red) / no audit item (grey)."""
    from matplotlib.colors import ListedColormap

    M = _matrix(g)
    steps = [r["step"] for r in g["inventory"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.imshow(np.nan_to_num(M, nan=0.5), cmap=ListedColormap([FAIL, NA, PASS]), vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        txt = "—" if np.isnan(v) else ("✓" if v else "✗")
        ax.text(j, i, txt, ha="center", va="center", color=SURF if not np.isnan(v) else INK, fontsize=11)
    ax.set_xticks(range(len(CHECKS)), CHECKS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(steps)), [f"Step {s}" for s in steps], fontsize=8)
    ax.grid(False)
    ax.set_title("Phase 6 gate: traceability matrix", fontsize=9)
    return fig


def completeness3d(g: dict) -> plt.Figure:
    """3D bars: steps x artefact kinds (figures, registry IDs, decisions, audited numbers) — how much evidence each
    step left behind."""
    inv = g["inventory"]
    kinds = ["audited numbers", "decisions", "registry IDs", "figures"]  # the tallest (figures) at the back
    val = np.array(
        [
            [
                sum(a["step"] == r["step"] and a["status"] == "FOUND" for a in g["audit"]),
                len(r["decisions"]),
                len(r["registry_ids"]),
                r["n_figures"],
            ]
            for r in inv
        ],
        float,
    )
    fig = plt.figure(figsize=(10.0, 6.5))
    ax = fig.add_subplot(projection="3d")
    for j in range(len(kinds)):
        xs = np.arange(len(inv))
        ax.bar3d(
            xs,
            np.full_like(xs, j, dtype=float),
            0,
            0.5,
            0.5,
            np.log10(1 + val[:, j]),
            color=C[j],
            alpha=0.85,
            shade=True,
        )
    ax.set_xticks(np.arange(len(inv)) + 0.25, [r["step"] for r in inv], fontsize=7)
    ax.set_yticks(np.arange(len(kinds)) + 0.25, kinds, fontsize=7)
    ax.set_zlabel("log10 (1 + count)")
    ax.set_title("Phase 6 evidence per step", fontsize=9)
    ax.view_init(elev=28, azim=-60)
    return fig
