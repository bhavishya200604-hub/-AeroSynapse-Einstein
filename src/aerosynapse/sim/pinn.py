r"""Simulation with the parametric PINN: the skin-heating network as a fast solver, checked as it runs.

The network (inputs :math:`x, t, \ln Bi`; :mod:`aerosynapse.thermal.bridge`) answers the convectively heated slab for
any Biot number in 0.2 to 20 in one forward pass. Here it is run as a simulator:

1. temperature histories of both faces at three Biot numbers, against the exact eigen-series and the project's
   finite-difference (Crank-Nicolson) solver, with the error of each;
2. the full temperature field :math:`\theta(x,t)` from the PINN and its error field against the exact solution;
3. query speed: PINN against the eigen-series and the finite-difference solver;
4. the design's own skin zones through the hot flight states, with the materials and gauges the design carries
   (from the workflow's PINN stage when present), every answer labelled PINN or analytic (outside the trained box).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.sim.common import NEON, finish, plt, save

ROOT = Path(__file__).resolve().parents[3]


def network() -> tuple[Any, Any, str]:
    from aerosynapse.thermal import bridge as B

    for c in (ROOT / "results" / "thermal_skin" / "skin_pinn.pt",):
        if c.exists():
            n, p = B.load(c)
            return n, p, str(c)
    raise FileNotFoundError("no trained skin PINN: run the workflow's thermal stage or `aerosynapse thermal` first")


def viz_zones(design: dict[str, Any], zones: dict[str, Any], out: Path) -> None:
    """3D view: the airframe coloured by each skin zone's peak temperature (zones placed by position on the body)."""
    import json

    from aerosynapse.cad import kernel
    from aerosynapse.structures.wing import main_wing

    inf = kernel.info(design, h=0.1, nx=40)
    lo, hi = inf["bbox_lo"], inf["bbox_hi"]
    L = hi[0] - lo[0]
    try:
        y0 = float(main_wing(design).get("translate", [0, 0, 0])[1]) + 0.05
    except Exception:
        y0 = 0.15 * (hi[1] - lo[1])
    zc = 0.5 * (lo[2] + hi[2])
    big = 1e3

    def box(x0: float, x1: float, ya: float = -big, yb: float = big, za: float = -big, zb: float = big) -> list[float]:
        return [lo[0] + x0 * L, lo[0] + x1 * L, ya, yb, za, zb]

    order = [("nozzle", [box(0.93, 1.01, -y0, y0)]), ("radome", [box(-0.01, 0.07, -y0, y0)]),
             ("control", [box(0.72, 1.01, -big, -y0), box(0.72, 1.01, y0, big), box(0.6, 1.01, -y0, y0, zc + 0.35 * (hi[2] - zc), big)]),
             ("wing_box", [box(0.0, 1.01, -big, -y0), box(0.0, 1.01, y0, big)]), ("fwd_fuselage", [box(0.07, 0.35)]), ("centre_fuselage", [box(0.35, 0.68)]),
             ("aft_fuselage", [box(0.68, 0.93)])]
    zl = [{"name": z, "box": b, "value": zones[z]["T_max_K"]} for z, bs in order if z in zones for b in bs]
    viz = {"title": "Skin heating: peak temperature of each zone (parametric PINN with exact-series fallback)",
           "colour": {"kind": "zones", "label": "peak skin temperature, K (zones by position)", "zones": zl}}
    (out / "viz3d.json").write_text(json.dumps(viz), encoding="utf-8")


def run(design: dict[str, Any], out: str | Path, setup: dict[str, Any] | None = None) -> dict[str, Any]:
    from aerosynapse.aircraft import materials as mt
    from aerosynapse.sim import setup as SU
    from aerosynapse.thermal import bridge as B
    from aerosynapse.thermal import skin_design as SD

    S = SU.merged("pinn", setup)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pl = plt()
    net, prob, ck = network()
    lines = ["Setup:", *SU.describe("pinn", S), "", f"Network: {ck}", f"Trained box: Bi {B.BI_RANGE[0]} to {B.BI_RANGE[1]}, Fourier time 0 to {B.HORIZON}", ""]
    res: dict[str, Any] = {"network": ck, "figures": []}
    fo = np.linspace(0.02, B.HORIZON, 60)
    x = np.array([0.0, 1.0])
    fig, ax = pl.subplots(1, 3, figsize=(12, 3.3))
    errs = {}
    for k, bi in enumerate((float(S["biot_1"]), float(S["biot_2"]), float(S["biot_3"]))):
        ex = B.theta_exact(bi, fo, x)
        pn = B.theta_pinn(net, prob, bi, fo, x)
        fd = B.theta_fd(bi, fo, x)
        for j, (face, ls) in enumerate((("inner face", "-"), ("heated face", "--"))):
            ax[k].plot(fo, ex[:, j], ls, color=NEON[0], lw=2.6, alpha=0.6, label=f"exact series, {face}" if k == 0 else None)
            ax[k].plot(fo, pn[:, j], ls, color=NEON[1], lw=1.4, label=f"PINN, {face}" if k == 0 else None)
            ax[k].plot(fo, fd[:, j], ls, color=NEON[2], lw=0.9, label=f"finite differences, {face}" if k == 0 else None)
        errs[f"Bi={bi}"] = {"pinn_max_abs": float(np.abs(pn - ex).max()), "fd_max_abs": float(np.abs(fd - ex).max())}
        ax[k].set(xlabel="Fourier time", ylabel="theta" if k == 0 else "", title=f"Bi = {bi}: PINN error {errs[f'Bi={bi}']['pinn_max_abs']:.3f}")
    ax[0].legend(fontsize=6.5)
    res["figures"].append(save(fig, out, "1_pinn_vs_exact_vs_fd"))
    res["errors"] = errs
    lines.append("1. PINN against the exact eigen-series and the finite-difference solver (theta = scaled temperature, 1 at the start)")
    lines += [f"   {k}: PINN max error {v['pinn_max_abs']:.4f}, finite differences {v['fd_max_abs']:.2e}" for k, v in errs.items()]
    lines.append("")
    # field and error field
    xs = np.linspace(0, 1, 41)
    fos = np.linspace(0.02, B.HORIZON, 80)
    bf = float(S["field_biot"])
    th_p = B.theta_pinn(net, prob, bf, fos, xs)
    th_e = B.theta_exact(bf, fos, xs)
    fig, ax = pl.subplots(1, 2, figsize=(11, 3.4))
    im = ax[0].pcolormesh(xs, fos, th_p, cmap="turbo", shading="auto")
    fig.colorbar(im, ax=ax[0], label="theta")
    ax[0].set(xlabel="depth x / thickness (0 inner, 1 heated)", ylabel="Fourier time", title=f"PINN temperature field, Bi = {bf:g}")
    im = ax[1].pcolormesh(xs, fos, np.abs(th_p - th_e), cmap="magma", shading="auto")
    fig.colorbar(im, ax=ax[1], label="|PINN - exact|")
    ax[1].set(xlabel="depth x / thickness", title="Error field against the exact series")
    res["figures"].append(save(fig, out, "2_pinn_field"))
    # timing
    t0 = time.perf_counter()
    for _ in range(50):
        B.theta_pinn(net, prob, 2.0, fo, x)
    tp = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter()
    for _ in range(5):
        B.theta_exact(2.0, fo, x)
    te = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    B.theta_fd(2.0, fo, x)
    tf = time.perf_counter() - t0
    res["query_seconds"] = {"pinn": tp, "exact_series": te, "finite_differences": tf}
    lines += ["2. Query time for 60 instants at both faces", f"   PINN {1e3 * tp:.1f} ms, exact series {1e3 * te:.1f} ms ({te / tp:.0f}x slower), finite differences {1e3 * tf:.0f} ms ({tf / tp:.0f}x slower)", ""]
    # the design's zones
    skin = design.get("skin") or {}
    zones = {}
    for zone, (t_nom, _, _) in SD.PANELS.items():
        z = dict(skin.get(zone) or {})
        if S.get(f"zone_{zone}_material", "design") != "design":
            z["material"] = mt.LIBRARY[str(S[f"zone_{zone}_material"])].name
        if float(S.get(f"zone_{zone}_thickness", 0) or 0) > 0:
            z["thickness_mm"] = float(S[f"zone_{zone}_thickness"])
        m = next((mm for mm in mt.LIBRARY.values() if mm.name == z.get("material")), None)
        if m is None:
            sel = next((c for c in mt.select(SD.zone_temperatures()) if c.zone == zone), None)
            m = mt.LIBRARY[sel.material] if sel else mt.LIBRARY["al7075"]
        t = float(z.get("thickness_mm", t_nom * 1e3)) * 1e-3
        w = SD.panel_worst(net, prob, zone, m, t)
        zones[zone] = {"material": m.name, "thickness_mm": t * 1e3, **w}
    res["zones"] = zones
    fig, ax = pl.subplots(figsize=(9, 3.2))
    names = list(zones)
    ax.bar(names, [zones[n]["T_max_K"] for n in names], color=NEON[0], alpha=0.85, label="peak temperature")
    ax.plot(names, [next(mm.T_max for mm in mt.LIBRARY.values() if mm.name == zones[n]["material"]) for n in names], "D", color=NEON[1], label="material service limit")
    ax.set(ylabel="K", title="The design's skin zones through the hot flight states (PINN)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=7.5)
    res["figures"].append(save(fig, out, "3_design_zones"))
    lines.append("3. The design's skin zones (worst of dash, cold high, max-q and hot-day dash)")
    lines += [f"   {z:16s} {v['material'][:28]:28s} {v['thickness_mm']:5.2f} mm  peak {v['T_max_K']:5.0f} K  dT {v['dT_max_K']:5.1f} K  stress ratio {v['stress_ratio']:.3f}  "
              f"answers {v['source']}" for z, v in zones.items()]
    viz_zones(design, zones, out)
    tot = {k: sum(v["source"].get(k, 0) for v in zones.values()) for k in ("pinn", "analytic", "initial")}
    share = tot["pinn"] / max(1, tot["pinn"] + tot["analytic"])
    res["answer_share_pinn"] = share
    lines.append(f"   {100 * share:.0f} % of these answers came from the PINN; the rest fall outside its trained box (thin skins have Biot numbers below "
                 f"{B.BI_RANGE[0]}, or the state outlasts the Fourier horizon) and were answered by the exact series")
    return finish(out, "Simulation with the parametric PINN", lines, res)
