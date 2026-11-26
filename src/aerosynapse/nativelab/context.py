"""The design and the assembly as inputs of the labs.

Studio writes ``context.json``: the selected (or largest) part with its material (density, conductivity, heat
capacity), its size and turning speed, every component, the Environment (fluid and its properties, temperatures,
current speed), the drive train's losses and the feature tree of the design. From that come the numbers the labs ask
for, each with how it was obtained:

* Biot number Bi = h Lc / k_part (Lc = half the part's smallest size; h from a flat-plate correlation, forced
  convection with the current or the part's surface speed, else natural convection), the part's diffusivity;
* Peclet Pe = U L / alpha_fluid, Reynolds Re = U L / nu, Prandtl Pr of the fluid, Richardson Ri = g beta dT L / U^2;
* the volumetric heat source q_vol = the part's share of the drive-train losses / its volume;
* for the aircraft designer: body length, wing area, aspect ratio, taper and sweep from the fuselage and wing features.

A label of a lab's input that matches one of these gets that value as its starting value (shown as "from the design").
"""

from __future__ import annotations

import math
from typing import Any

G = 9.81


def _num(d: dict, k: str, default: float = 0.0) -> float:
    try:
        return float(d.get(k, default))
    except (TypeError, ValueError):
        return default


def derived(ctx: dict[str, Any]) -> dict[str, tuple[float, str]]:
    """Named quantities from the design, each with a note of where it came from."""
    out: dict[str, tuple[float, str]] = {}
    part = ctx.get("part") or {}
    env = ctx.get("environment") or {}
    fl = env.get("fluid") or {}
    mat = part.get("material") or {}
    size = [float(s) for s in part.get("size_m", [])[:3] if float(s) > 0]
    pname = f"{part.get('id', 'the part')} ({mat.get('label', 'material?')})"
    if size:
        Lmax, Lc = max(size), min(size) / 2
        rho_f, mu_f, k_f, cp_f = (
            _num(fl, "rho", 1.2),
            _num(fl, "mu", 1.8e-5),
            _num(fl, "k", 0.026),
            _num(fl, "cp", 1006),
        )
        nu_f, a_f, Pr = mu_f / max(rho_f, 1e-12), k_f / max(rho_f * cp_f, 1e-12), _num(fl, "Pr", 0.71)
        U = _num(env, "U_m_s")
        how = f"the current {U:g} m/s"
        if U <= 0 and _num(part, "rpm") > 0:
            U = _num(part, "rpm") * 2 * math.pi / 60 * Lmax / 2
            how = f"the surface speed of {part.get('id')} at {_num(part, 'rpm'):g} rpm"
        dT = abs(_num(env, "T_part_C", 60) - _num(env, "T_fluid_C", 15))
        beta = _num(fl, "beta", 1 / 288.0)
        if U > 0:
            Re = U * Lmax / nu_f
            Nu = 0.664 * Re**0.5 * Pr ** (1 / 3) if Re < 5e5 else 0.037 * Re**0.8 * Pr ** (1 / 3)
            hwhy = f"forced convection, flat plate, Re {Re:.3g} ({how})"
            out["Re"] = (Re, f"U L / nu with {how}, L = {Lmax * 1000:.0f} mm, {env.get('medium', 'fluid')}")
            out["Pe"] = (U * Lmax / a_f, f"U L / alpha of {env.get('medium', 'the fluid')} ({how})")
            if dT > 0:
                out["Ri"] = (G * beta * dT * Lmax / U**2, f"g beta dT L / U^2, dT = {dT:g} K")
        else:
            Ra = G * beta * max(dT, 1e-3) * Lmax**3 / (nu_f * a_f)
            Nu = 0.54 * Ra**0.25 if Ra < 1e7 else 0.15 * Ra ** (1 / 3)
            hwhy = f"natural convection, Ra {Ra:.3g}"
        h = Nu * k_f / Lmax
        out["Pr"] = (Pr, f"{env.get('medium', 'the fluid')} at {_num(env, 'T_fluid_C', 15):g} C")
        out["h"] = (h, hwhy)
        k_s = _num(mat, "k")
        if k_s > 0:
            out["Bi"] = (
                h * Lc / k_s,
                f"h Lc / k: h {h:.3g} W/m2K ({hwhy}), Lc {Lc * 1000:.1f} mm, k {k_s:g} W/mK of {pname}",
            )
            out["alpha"] = (k_s / max(_num(mat, "rho", 1) * _num(mat, "cp", 1), 1e-12), f"k / (rho cp) of {pname}")
            out["k"] = (k_s, f"conductivity of {pname}")
        vol = _num(part, "volume_m3") or (size[0] * size[1] * size[2] if len(size) == 3 else 0)
        loss = _num(part, "loss_W") or _num(ctx, "power_loss_W")
        if loss > 0 and vol > 0:
            out["q_vol"] = (loss / vol, f"{loss:.3g} W of drive-train loss in {vol * 1e6:.3g} cm3 of {part.get('id')}")
        out["L"] = (Lmax, f"the largest size of {part.get('id')}")
    # the aircraft: fuselage and wings of the feature tree
    feats = ((ctx.get("design") or {}).get("features")) or []
    S = span_tot = 0.0
    for f in feats:
        if not isinstance(f, dict) or f.get("enabled") is False:
            continue
        t = f.get("type")
        if t == "fuselage" and f.get("stations"):
            xs = [float(s[0]) for s in f["stations"] if isinstance(s, list) and s]
            if xs:
                out["body_length"] = (max(xs) - min(xs), f"fuselage '{f.get('id', '')}' stations")
                ws = [float(s[1]) for s in f["stations"] if isinstance(s, list) and len(s) > 2]
                hs = [float(s[2]) for s in f["stations"] if isinstance(s, list) and len(s) > 2]
                if ws:
                    out["body_halfwidth"] = (max(ws), "fuselage stations")
                if hs:
                    out["body_halfheight"] = (max(hs), "fuselage stations")
        if t == "wing":
            b, cr, tp = _num(f, "span"), _num(f, "root_chord"), _num(f, "taper", 1)
            mirror = f.get("mirror", True) is not False
            area = b * cr * (1 + tp) / 2 * (2 if mirror else 1)
            S += area
            span_tot = max(span_tot, b * (2 if mirror else 1))
            out["taper"] = (tp, f"wing '{f.get('id', '')}'")
            out["sweep"] = (_num(f, "sweep"), f"wing '{f.get('id', '')}' leading-edge sweep")
    if S > 0:
        out["wing_area"] = (S, "the wing features (both halves)")
        out["aspect_ratio"] = (span_tot**2 / S, "span^2 / S of the wing features")
    return out


# a lab input's label -> the quantity that answers it
LABELS = {
    r"^\s*Bi\b(?!\s*(from|to)\b)|Biot": "Bi",  # (a sweep's "from / to" range keeps its own ends)
    r"^\s*Pe\b(?!\s*(from|to)\b)|P[eé]clet number": "Pe",
    r"^\s*Re\b(?!\s*(from|to)\b)|Reynolds": "Re",
    r"^\s*Pr\b|Prandtl": "Pr",
    r"^\s*Ri\b|Richardson": "Ri",
    r"volumetric heat source": "q_vol",  # the THERMAL LAB inverse page (not the PINN app's experiment source)
    r"Body length": "body_length",
    r"Body half width": "body_halfwidth",
    r"Body half height": "body_halfheight",
    r"Wing area": "wing_area",
    r"Aspect ratio": "aspect_ratio",
    r"Taper ratio": "taper",
    r"Sweep, degrees": "sweep",
}


def label_defaults(ctx: dict[str, Any]) -> dict[str, tuple[float, str]]:
    """regex over an input's label -> (value, why), for the quantities this design gives."""
    d = derived(ctx)
    return {pat: (float(f"{d[name][0]:.6g}"), d[name][1]) for pat, name in LABELS.items() if name in d}


def summary(ctx: dict[str, Any]) -> str:
    d = derived(ctx)
    if not d:
        return "No design input: open a part or an assembly (and set its Environment) to fill the labs from it."
    lines = ["Inputs taken from the design and the assembly:"]
    for k, (v, why) in d.items():
        lines.append(f"  {k:16s} {v:12.5g}   {why}")
    return "\n".join(lines)
