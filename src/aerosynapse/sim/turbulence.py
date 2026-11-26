"""Turbulence and flow instability of a design.

1. Free-stream turbulence and laminar-turbulent transition on the wing section: panel-method edge velocity,
   Thwaites' laminar boundary layer, the e^N envelope method (Drela and Giles 1987) with Mack's critical N from the
   free-stream turbulence intensity; transition position and friction drag against Tu.
2. Wake instability: the inviscid Rayleigh equation solved as a Chebyshev-collocation eigenproblem on the wake
   profiles of the 3D flow (or on momentum-matched Gaussian wakes when there is no flow run), giving the growth rate,
   frequency and Strouhal number of the sinuous (vortex-street) and varicose modes. Checked on the tanh mixing layer
   (Michalke 1964, U = tanh z: maximum growth 0.1897 at wavenumber 0.4446).
3. Turbulence in the 3D field (time-accurate runs): turbulent kinetic energy, r.m.s. velocity and vorticity, and
   spectra at the probes with the Kolmogorov -5/3 reference, the measured inflow intensity and integral scales.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.sim import setup as SU
from aerosynapse.sim.common import NEON, finish, plt, save


# ------------------------------------------------------------------ 1 transition
def mack_ncrit(tu_percent: float) -> float:
    """Critical amplification exponent from the free-stream turbulence intensity (Mack 1977)."""
    return float(max(0.5, -8.43 - 2.4 * np.log(max(tu_percent, 1e-3) / 100.0)))


def thwaites_H_l(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape factor H and shear function l of Thwaites' method (Cebeci and Bradshaw fits)."""
    lam = np.clip(lam, -0.09, 0.25)
    H = np.where(lam >= 0, 2.61 - 3.75 * lam + 5.24 * lam**2, 2.088 + 0.0731 / (lam + 0.14))
    ell = np.where(lam >= 0, 0.22 + 1.57 * lam - 1.8 * lam**2, 0.22 + 1.402 * lam + 0.018 * lam / (lam + 0.107))
    return H, ell


def envelope_rate(H: np.ndarray, Rt: np.ndarray, th: np.ndarray) -> np.ndarray:
    """dN/ds of the e^N envelope method (Drela and Giles 1987), zero below the critical Re_theta."""
    Hm1 = np.maximum(H - 1.0, 0.05)
    log_rt0 = (1.415 / Hm1 - 0.489) * np.tanh(20.0 / Hm1 - 12.9) + 3.295 / Hm1 + 0.44
    dn_drt = 0.01 * np.sqrt((2.4 * H - 3.7 + 2.5 * np.tanh(1.5 * H - 4.65)) ** 2 + 0.25)
    lH = (6.54 * H - 14.07) / H**2
    mH = (0.058 * (H - 4.0) ** 2 / Hm1 - 0.068) / lH
    rate = dn_drt * 0.5 * (mH + 1.0) * lH / np.maximum(th, 1e-12)
    return np.where(np.log10(np.maximum(Rt, 1.0)) > log_rt0, np.maximum(rate, 0.0), 0.0)


def boundary_layer(s: np.ndarray, ue: np.ndarray, nu: float, n_crit: float) -> dict[str, Any]:
    """Laminar boundary layer along one surface from the stagnation point; s in m, ue in m/s (ue > 0)."""
    from scipy.signal import savgol_filter

    ue = np.maximum(ue, 1e-3 * ue.max())
    if len(ue) > 15:  # panel velocities carry discretisation noise that the gradient would amplify
        ue = np.r_[ue[:3], np.maximum(savgol_filter(ue, 9, 2)[3:], 1e-3 * ue.max())]
    due = np.gradient(ue, s)
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (ue[1:] ** 5 + ue[:-1] ** 5) * np.diff(s))])
    th2 = 0.45 * nu / ue**6 * integ
    if due[0] > 0:  # stagnation point (Hiemenz flow); a flat-plate leading edge starts from zero
        th2[0] = 0.075 * nu / due[0]
    th = np.sqrt(th2)
    lam = th2 * due / nu
    H, ell = thwaites_H_l(lam)
    Rt = ue * th / nu
    rate = envelope_rate(H, Rt, th)
    amp = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(s))])
    sep = np.where(lam < -0.09)[0]
    tr = np.where(amp >= n_crit)[0]
    i_sep = int(sep[0]) if len(sep) else None
    i_tr = int(tr[0]) if len(tr) else None
    mech = "none (laminar to the trailing edge)"
    i = None
    if i_tr is not None and (i_sep is None or i_tr <= i_sep):
        i, mech = i_tr, "natural (Tollmien-Schlichting waves reach N crit)"
    elif i_sep is not None:
        i, mech = i_sep, "laminar separation (a separation bubble trips the layer)"
    return {"theta": th, "H": H, "Re_theta": Rt, "N": amp, "lambda": lam, "i_transition": i, "mechanism": mech}


def section_transition(design: dict[str, Any], S: dict[str, Any], air: Any, U: float, tus: np.ndarray) -> dict[str, Any]:
    from aerosynapse.numerics.panel import solve_panel
    from aerosynapse.shapeopt.geometry import to_panel_order
    from aerosynapse.sim.analytic import section_xy
    from aerosynapse.structures.wing import main_wing

    f = main_wing(design)
    eta = float(S["section_span"])
    chord = float(f["root_chord"]) * (1 - (1 - float(f.get("taper", 0.5))) * eta)
    xy = section_xy(f, 161)
    X, Y = to_panel_order(xy)
    r = solve_panel(X, Y, float(S["alpha"]), mach=min(float(S["mach"]), 0.6))
    ue = np.abs(r.vt)
    i0 = int(np.argmin(ue))  # stagnation point
    seg = np.hypot(np.diff(r.xc), np.diff(r.yc))
    sides = {}
    for name, idx in (("upper", np.arange(i0, len(ue))), ("lower", np.arange(i0, -1, -1))):
        if len(idx) < 5:
            continue
        ds = np.r_[0.0, seg[np.minimum(idx[1:], idx[:-1])]]
        sides[name] = {"s": np.cumsum(ds) * chord, "ue": ue[idx] * U, "x_c": r.xc[idx]}
    nu = air.mu / air.rho
    offset = {"smooth": 0.0, "painted": -1.0, "rough": -3.0}[str(S["roughness"])]
    n_main = float(S["n_crit"]) if float(S["n_crit"]) > 0 else mack_ncrit(float(S["turbulence_intensity"])) + offset
    out: dict[str, Any] = {"wing": f.get("id"), "chord_m": chord, "eta": eta, "n_crit": n_main, "Re_chord": U * chord / nu, "sides": {}, "sweep": []}
    for name, sd in sides.items():
        bl = boundary_layer(sd["s"], sd["ue"], nu, n_main)
        i = bl["i_transition"]
        out["sides"][name] = {**bl, "x_c": sd["x_c"], "s": sd["s"], "x_tr_c": float(sd["x_c"][i]) if i is not None else 1.0}
    for tu in tus:
        n = max(0.5, mack_ncrit(float(tu)) + offset)
        row = {"Tu_percent": float(tu), "N_crit": n}
        for name, sd in sides.items():
            bl = boundary_layer(sd["s"], sd["ue"], nu, n)
            i = bl["i_transition"]
            row[f"x_tr_{name}"] = float(sd["x_c"][i]) if i is not None else 1.0
        out["sweep"].append(row)
    return out


def friction_cd(re_c: float, xt: float) -> float:
    """Flat-plate friction coefficient of one side with transition at x/c = xt (laminar Blasius, then turbulent from a virtual origin)."""
    xt = float(np.clip(xt, 0.0, 1.0))
    cf_t = 0.074 / re_c**0.2
    if xt <= 0:
        return cf_t
    re_t = max(re_c * xt, 1.0)
    return float(cf_t - xt * (0.074 / re_t**0.2 - 1.328 / np.sqrt(re_t)))


# ------------------------------------------------------------------ 2 Rayleigh stability
def cheb(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Chebyshev points on [-1, 1] and the differentiation matrix (Trefethen, Spectral Methods in MATLAB)."""
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.r_[2.0, np.ones(n - 1), 2.0] * (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(n + 1))
    D -= np.diag(D.sum(1))
    return x, D


def rayleigh(Uf: Any, d2Uf: Any, half: float, k: float, n: int = 80) -> list[dict[str, Any]]:
    """Temporal inviscid modes of U(z) on [-half, half] at wavenumber k: c = c_r + i c_i, with the mode's parity."""
    from scipy.linalg import eig

    x, D = cheb(n)
    z = half * x
    D2 = (D @ D) / half**2
    I = np.eye(n + 1)  # noqa: E741
    U, U2 = np.diag(Uf(z)), np.diag(d2Uf(z))
    A = U @ (D2 - k**2 * I) - U2
    B = D2 - k**2 * I
    A, B = A[1:-1, 1:-1], B[1:-1, 1:-1]  # phi = 0 far from the shear layer
    w, v = eig(A, B)
    out = []
    umin, umax = float(Uf(z).min()), float(Uf(z).max())
    for c, ph in zip(w, v.T, strict=True):
        if not np.isfinite(c) or c.imag <= 1e-6 or not (umin - 1e-6 <= c.real <= umax + 1e-6) or abs(c.imag) > (umax - umin):
            continue
        ph = ph / ph[np.argmax(np.abs(ph))]
        sym = float(np.linalg.norm(ph - ph[::-1]) / max(np.linalg.norm(ph), 1e-12))
        out.append({"c": complex(c), "growth": float(k * c.imag), "omega_r": float(k * c.real), "mode": "sinuous" if sym < 0.5 else "varicose"})
    return out


def most_unstable(Uf: Any, d2Uf: Any, half: float, ks: np.ndarray, n: int) -> dict[str, Any]:
    best: dict[str, dict[str, Any]] = {}
    curve: dict[str, list[float]] = {"sinuous": [], "varicose": []}
    for k in ks:
        modes = rayleigh(Uf, d2Uf, half, float(k), n)
        for kind in curve:
            m = max((mm for mm in modes if mm["mode"] == kind), key=lambda mm: mm["growth"], default=None)
            g = m["growth"] if m else 0.0
            curve[kind].append(g)
            if m and g > best.get(kind, {"growth": 0})["growth"]:
                best[kind] = {**m, "k": float(k), "c": [m["c"].real, m["c"].imag]}
    return {"best": best, "curve": curve}


def gauss_wake(d: float, b: float = 1.0) -> tuple[Any, Any]:
    a = np.log(2.0) / b**2
    return (lambda z: 1 - d * np.exp(-a * z**2)), (lambda z: -d * np.exp(-a * z**2) * (4 * a**2 * z**2 - 2 * a))


# ------------------------------------------------------------------ 3 spectra
def welch_psd(x: np.ndarray, dt: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    x = x - x.mean()
    nper = max(16, len(x) // max(1, segments))
    return welch(x, fs=1.0 / dt, nperseg=min(nper, len(x)))


def integral_time(x: np.ndarray, dt: float) -> float:
    x = x - x.mean()
    if x.std() == 0:
        return 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1:] / (x.var() * len(x))
    neg = np.where(ac <= 0)[0]
    end = int(neg[0]) if len(neg) else len(ac)
    return float(np.trapezoid(ac[:end], dx=dt))


# ------------------------------------------------------------------ the analysis
def run(design: dict[str, Any], out: str | Path, setup: dict[str, Any] | None = None) -> dict[str, Any]:
    from aerosynapse.aircraft.atmosphere import isa

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    S = SU.merged("turbulence", setup)
    pl = plt()
    air = isa(float(S["altitude"]))
    U = float(S["mach"]) * air.a
    res: dict[str, Any] = {"setup": S, "figures": [], "flight": {"U_m_s": U, "T_K": air.T, "rho": air.rho, "nu": air.mu / air.rho}}
    lines = ["Setup:", *SU.describe("turbulence", S), ""]

    # 1 transition
    tus = np.geomspace(0.01, 5.0, 25)
    tr = section_transition(design, S, air, U, tus)
    res["transition"] = {k: v for k, v in tr.items() if k != "sides"}
    res["transition"]["sides"] = {k: {"x_tr_c": v["x_tr_c"], "mechanism": v["mechanism"]} for k, v in tr["sides"].items()}
    fig, ax = pl.subplots(1, 3, figsize=(13, 3.6))
    for j, (name, sd) in enumerate(tr["sides"].items()):
        i = sd["i_transition"]
        e = (i + 1) if i is not None else len(sd["x_c"])  # the laminar method stops at transition
        ax[0].plot(sd["x_c"][:e], sd["N"][:e], color=NEON[j], label=f"{name} surface")
        ax[1].plot(sd["x_c"][:e], sd["H"][:e], color=NEON[j], label=f"H, {name}")
        if i is not None:
            ax[0].plot([sd["x_c"][i]], [sd["N"][i]], "o", color=NEON[j], ms=5)
            ax[1].plot([sd["x_c"][i]], [sd["H"][i]], "o", color=NEON[j], ms=5)
    ax[0].set_ylim(0, max(2.0 * tr["n_crit"], 10.0))
    ax[0].axhline(tr["n_crit"], color="0.4", ls="--", lw=1, label=f"N crit {tr['n_crit']:.1f} (Tu {S['turbulence_intensity']:g} %)")
    ax[0].set(xlabel="x / c", ylabel="amplification N = ln(A/A0)", title="Growth of Tollmien-Schlichting waves (e^N envelope)", xlim=(0, 1))
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel="x / c", ylabel="shape factor H", title="Laminar boundary layer up to transition (dot)", xlim=(0, 1))
    ax[1].legend(fontsize=7)
    sw = tr["sweep"]
    for j, name in enumerate(tr["sides"]):
        ax[2].semilogx([r_["Tu_percent"] for r_ in sw], [r_[f"x_tr_{name}"] for r_ in sw], "o-", ms=3, color=NEON[j], label=f"{name} surface")
    ax[2].set(xlabel="free-stream turbulence Tu, %", ylabel="transition x / c", title="Transition moves forward as the air gets rougher", ylim=(0, 1.02))
    ax[2].legend(fontsize=7)
    res["figures"].append(save(fig, out, "1_transition"))
    re_c = tr["Re_chord"]
    xt_u = tr["sides"].get("upper", {}).get("x_tr_c", 0.0)
    xt_l = tr["sides"].get("lower", {}).get("x_tr_c", 0.0)
    cf_tr = friction_cd(re_c, xt_u) + friction_cd(re_c, xt_l)
    cf_turb = 2 * friction_cd(re_c, 0.0)
    res["transition"]["cd_friction_section"] = cf_tr
    res["transition"]["cd_friction_fully_turbulent"] = cf_turb
    lines += ["1. Free-stream turbulence and transition on the wing section",
              f"   section of '{tr['wing']}' at {100 * tr['eta']:.0f} % semi-span, chord {tr['chord_m']:.2f} m, Re {re_c:.3g}, alpha {S['alpha']:g} deg",
              f"   critical N {tr['n_crit']:.2f} (Mack, Tu {S['turbulence_intensity']:g} %, surface {S['roughness']})"]
    for name, sd in tr["sides"].items():
        lines.append(f"   {name} surface: transition at x/c {sd['x_tr_c']:.4f}, {sd['mechanism']}"
                     + (", right behind the leading-edge suction peak" if sd['x_tr_c'] < 0.02 and "separation" in sd['mechanism'] else ""))
    lines += [f"   section friction drag {cf_tr:.5f} with this transition, {cf_turb:.5f} fully turbulent ({100 * (cf_tr / cf_turb - 1):+.1f} %)",
              f"   Tu {sw[0]['Tu_percent']:.2f} % moves upper transition to x/c {sw[0]['x_tr_upper']:.4f}; Tu {sw[-1]['Tu_percent']:.1f} % to {sw[-1]['x_tr_upper']:.4f}",
              "   limits: incompressible Thwaites layer on the panel-method velocity (Prandtl-Glauert Mach capped at 0.6 in the panel solution); no crossflow instability of the swept wing", ""]

    # 2 wake instability
    flow = Path(S["flow_folder"]) if S["flow_folder"] else None
    V = None
    if flow and (flow / "volume.bin").exists():
        from aerosynapse.sim.flow3d import Volume

        V = Volume(flow)
    stations = [float(x) for x in str(S["wake_stations"]).replace(";", ",").split(",") if x.strip()]
    ks = np.linspace(0.05, float(S["k_max"]), 30)
    n = int(S["rayleigh_points"])
    wakes = []
    if V is not None:
        from scipy.optimize import curve_fit

        jy = int(np.argmin(np.abs(V.y)))
        uinf = V.Uinf
        for xs in stations:
            x = V.body_hi[0] + xs * V.L
            if x > V.x[-1]:
                continue
            i = int(np.searchsorted(V.x, x))
            u = np.where(V.solid[i, jy], np.nan, V.u[i, jy]) / uinf
            ok = np.isfinite(u)
            z = V.z[ok]
            uu = u[ok]
            try:
                p0 = [max(0.02, 1 - uu.min()), float(z[np.argmin(uu)]), 0.1 * V.L]
                (d, z0, b), _ = curve_fit(lambda zz, d, z0, b: 1 - d * np.exp(-np.log(2) * ((zz - z0) / b) ** 2), z, uu, p0=p0, maxfev=4000)
                b = abs(b)
            except Exception:
                continue
            if not (0.005 < d < 0.95 and b > 0):
                continue
            wakes.append({"x_over_L": xs, "x_m": float(x), "deficit": float(d), "half_width_m": float(b), "z_centre_m": float(z0), "profile_z": z, "profile_u": uu, "source": "3D flow"})
    if not wakes:  # momentum-matched Gaussian wakes of the wing section's friction drag
        theta = 0.5 * cf_tr * tr["chord_m"]
        for xs, d in zip(stations, np.linspace(0.45, 0.12, len(stations)), strict=False):
            b = theta / (np.sqrt(np.pi / np.log(2)) * (d - d**2 / np.sqrt(2)))
            wakes.append({"x_over_L": xs, "deficit": float(d), "half_width_m": float(b), "source": "model (momentum of the section drag)"})
    fig, ax = pl.subplots(1, 2, figsize=(11, 3.6))
    for j, w in enumerate(wakes):
        Uf, d2Uf = gauss_wake(w["deficit"])
        st = most_unstable(Uf, d2Uf, 7.0, ks, n)
        w["stability"] = st["best"]
        ax[0].plot(ks, st["curve"]["sinuous"], color=NEON[j % len(NEON)], label=f"x/L {w['x_over_L']:g}: deficit {w['deficit']:.2f}, sinuous")
        ax[0].plot(ks, st["curve"]["varicose"], ":", color=NEON[j % len(NEON)], lw=1)
        sin = st["best"].get("sinuous")
        if sin:
            f_hz = sin["omega_r"] * U / (2 * np.pi * w["half_width_m"])
            w["frequency_hz"] = float(f_hz)
            w["strouhal_b"] = float(sin["omega_r"] / (2 * np.pi))
            w["growth_per_s"] = float(sin["growth"] * U / w["half_width_m"])
        if "profile_z" in w:
            ax[1].plot(w["profile_u"], (w["profile_z"] - w["z_centre_m"]) / w["half_width_m"], "o", ms=2.5, color=NEON[j % len(NEON)])
        zz = np.linspace(-4, 4, 200)
        ax[1].plot(Uf(zz), zz, color=NEON[j % len(NEON)], lw=1, label=f"x/L {w['x_over_L']:g} ({w['source'].split(' ')[0]})")
    ax[0].set(xlabel="wavenumber k b (b = wake half-width)", ylabel="growth rate omega_i b / U", title="Rayleigh stability of the wake (solid sinuous, dotted varicose)")
    ax[0].legend(fontsize=6.5)
    ax[1].set(xlabel="u / U", ylabel="z / b", title="Wake profiles (points: 3D flow; lines: Gaussian fit)", xlim=(0, 1.05), ylim=(-6, 6))
    ax[1].legend(fontsize=6.5)
    res["figures"].append(save(fig, out, "2_wake_instability"))
    # check of the eigen-solver on the tanh mixing layer
    chk = most_unstable(np.tanh, lambda z: -2 * np.tanh(z) / np.cosh(z) ** 2, 12.0, np.linspace(0.40, 0.49, 46), max(n, 80))
    best = max(chk["best"].values(), key=lambda m: m["growth"])
    res["rayleigh_check"] = {"growth": best["growth"], "k": best["k"], "reference_growth": 0.1897, "reference_k": 0.4446}
    res["wakes"] = [{k: v for k, v in w.items() if not k.startswith("profile")} for w in wakes]
    lines += ["2. Wake instability (inviscid Rayleigh equation, temporal, Chebyshev collocation)",
              f"   solver check, tanh mixing layer: maximum growth {best['growth']:.4f} at k {best['k']:.3f} (Michalke 1964: 0.1897 at 0.4446)"]
    for w in wakes:
        s_ = w.get("stability", {}).get("sinuous")
        v_ = w.get("stability", {}).get("varicose")
        txt = f"   x/L {w['x_over_L']:g} ({w['source']}): deficit {w['deficit']:.2f}, half-width {w['half_width_m']:.3f} m"
        if s_:
            txt += (f"; sinuous mode grows at {s_['growth']:.3f} U/b (e-folding {w['half_width_m'] / U / max(s_['growth'], 1e-9) * 1e3:.2f} ms), "
                    f"{w['frequency_hz']:.1f} Hz, St_b {w['strouhal_b']:.3f}")
        if v_:
            txt += f"; varicose {v_['growth']:.3f} U/b"
        lines.append(txt)
    lines += ["   the sinuous mode is the one that rolls the wake into a vortex street; the Gaussian fit smooths the coarse-grid profile", ""]

    # 3 turbulence in the field and spectra
    lines.append("3. Turbulence in the 3D flow")
    viz: dict[str, Any] = {"title": "Turbulence and instability", "lines": [], "meshes": []}
    if V is not None and V.stats:
        k = V.stats.get("tke")
        sl = V.crop(0.3, 1.0, 0.35)
        kk = np.where(V.solid, np.nan, k)
        res["field"] = {"tke_max": float(np.nanmax(kk[sl])), "u_rms_max": float(np.nanmax(np.where(V.solid, np.nan, V.stats.get("u_rms", kk))[sl])),
                        "samples": V.samples}
        jy = int(np.argmin(np.abs(V.y)))
        fig, ax = pl.subplots(1, 2, figsize=(11, 3.4))
        im = ax[0].pcolormesh(V.x[sl[0]], V.z[sl[2]], kk[sl[0], jy, sl[2]].T, cmap="viridis", shading="auto")
        fig.colorbar(im, ax=ax[0], label="k / U^2")
        ax[0].set(xlabel="x, m", ylabel="z, m", title="Turbulent kinetic energy, centre plane", aspect="equal")
        ix = min(len(V.x) - 1, int(np.searchsorted(V.x, V.body_hi[0] + 0.15 * V.L)))
        im = ax[1].pcolormesh(V.y[sl[1]], V.z[sl[2]], kk[ix, sl[1], sl[2]].T, cmap="viridis", shading="auto")
        fig.colorbar(im, ax=ax[1], label="k / U^2")
        ax[1].set(xlabel="y, m", ylabel="z, m", title=f"Turbulent kinetic energy behind the tail (x {V.x[ix]:.1f} m)", aspect="equal")
        res["figures"].append(save(fig, out, "3_turbulent_energy"))
        lines.append(f"   statistics over {V.samples} samples: largest turbulent kinetic energy {res['field']['tke_max']:.4f} U^2, "
                     f"largest u r.m.s. {res['field']['u_rms_max']:.3f} U")
        from aerosynapse.sim.flow3d import isosurface

        lev = float(np.nanpercentile(kk[sl][np.isfinite(kk[sl])], 98.5))
        coords = (V.x[sl[0]], V.y[sl[1]], V.z[sl[2]])
        if lev > 0 and isosurface(np.nan_to_num(kk[sl]), lev, coords, out / "iso_tke.obj"):
            viz["meshes"].append({"obj": "iso_tke.obj", "color": [0.55, 0.62, 0.72, 0.35], "label": f"turbulent kinetic energy {lev:.4f} U^2"})
    else:
        lines.append("   no time-accurate 3D flow given: run the 3D flow with turbulence intensity above 0 (or a gust) and Save the 3D field for field statistics")
    pr = flow / "probes.csv" if flow else None
    if pr and pr.exists():
        meta = json.loads((flow / "probes.json").read_text(encoding="utf-8"))
        data = np.genfromtxt(pr, delimiter=",", names=True)
        t = np.asarray(data["time"], float) * float(meta.get("time_unit_s", 1.0))
        half = len(t) // 2
        t = t[half:]
        dt = float(np.median(np.diff(t))) if len(t) > 4 else 0.0
        a_inf = 1.0 / float(meta.get("time_unit_s", 1.0))
        rows = []
        fig, ax = pl.subplots(figsize=(8, 3.6))
        for q, pm in enumerate(meta["probes"]):
            u = np.asarray(data[f"u{q}"], float)[half:] * a_inf
            if dt <= 0 or len(u) < 32:
                continue
            fr, P = welch_psd(u, dt, int(S["spectrum_segments"]))
            tu_m = float(u.std() / max(abs(u.mean()), 1e-9))
            T_int = integral_time(u, dt)
            pk = int(np.argmax(P[1:])) + 1 if len(P) > 3 else 0
            resolved = pk > 1  # a peak in the lowest bin only says the record is too short
            rows.append({"probe": pm["name"], "at": pm["at"], "Tu_percent": 100 * tu_m, "integral_time_s": T_int, "integral_length_m": T_int * abs(u.mean()),
                         "peak_hz": float(fr[pk]) if resolved else None, "lowest_hz": float(fr[1]) if len(fr) > 1 else None,
                         "strouhal_L": float(fr[pk] * float(meta.get("length", 1)) / max(abs(u.mean()), 1e-9)) if resolved else None})
            ax.loglog(fr[1:], P[1:], color=NEON[q % len(NEON)], lw=1.1, label=pm["name"])
        if rows:
            f0 = fr[1:]
            ref = P[1:].max() * (f0 / f0[0]) ** (-5 / 3)
            ax.loglog(f0, ref, "--", color="0.45", lw=1, label="-5/3 (Kolmogorov)")
            ax.set(xlabel="frequency, Hz", ylabel="PSD of u, (m/s)^2 / Hz", title="Velocity spectra at the probes")
            ax.legend(fontsize=6.5)
            res["figures"].append(save(fig, out, "4_spectra"))
            res["probes"] = rows
            lines.append(f"   probes ({len(rows)}), second half of the run, {len(t)} samples at {dt * 1e3:.2f} ms:")
            lines += [f"     {r_['probe']:16s} Tu {r_['Tu_percent']:6.2f} %  integral length {r_['integral_length_m']:.2f} m  "
                      + (f"spectral peak {r_['peak_hz']:.1f} Hz (St_L {r_['strouhal_L']:.3f})" if r_["peak_hz"] else
                         f"no resolved peak (record too short: lowest frequency {r_['lowest_hz']:.0f} Hz; run more steps)") for r_ in rows]
            up = next((r_ for r_ in rows if r_["probe"] == "upstream"), None)
            if up and flow:
                cs = json.loads((flow / "case.json").read_text(encoding="utf-8"))
                req = 100 * float(cs.get("flow", {}).get("turbulence_intensity", 0))
                gust = 100 * float(cs.get("flow", {}).get("gust_amplitude", 0))
                lines.append(f"   inflow check: requested Tu {req:.2f} %" + (f" and a {gust:.0f} % gust" if gust else "") + f"; measured upstream {up['Tu_percent']:.2f} % "
                             "(the grid damps eddies smaller than a few cells; gusts and the body's pressure field add to it)")
        else:
            pl.close(fig)
    lines += ["", "Limits: the 3D flow is a coarse large-eddy simulation, so only eddies several cells across are resolved; the stability analysis is linear, "
              "inviscid and parallel-flow (no absolute/convective distinction)."]
    (out / "viz3d.json").write_text(json.dumps(viz), encoding="utf-8")
    return finish(out, "Turbulence and flow instability", lines, res)
