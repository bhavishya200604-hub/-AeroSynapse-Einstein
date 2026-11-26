"""Quick aerodynamics of any feature-tree design (what the Studio's Analysis menu runs).

No optimiser variables are needed: geometry comes from the kernel (volume, wetted and frontal areas, area
distribution), lift and stability from the vortex lattice of the design's wing-type features, friction from
the boundary-layer model, and supersonic wave drag from Mach-plane cuts of the 3D shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.aircraft.analysis import wave_drag_area
from aerosynapse.aircraft.atmosphere import isa
from aerosynapse.aircraft.layers import friction
from aerosynapse.cad import kernel
from aerosynapse.fighter import vlm


def _mach_plane(design: dict[str, Any], info: dict[str, Any], mach: float, n_x: int = 32, n_roll: int = 4, n_y: int = 40, n_z: int = 60) -> float:
    lo, hi = np.array(info["bbox_lo"]), np.array(info["bbox_hi"])
    mu = np.arcsin(1 / mach)
    half = max(abs(lo[1]), abs(hi[1])) + 0.3
    zc, zh = 0.5 * (lo[2] + hi[2]), 0.5 * (hi[2] - lo[2]) + 0.2
    u, w = np.linspace(-half, half, n_y), np.linspace(-zh, zh, n_z)
    du, dw = u[1] - u[0], w[1] - w[0]
    U, Wz = np.meshgrid(u, w, indexing="ij")
    pts, keys = [], []
    for k in range(n_roll):
        th = np.pi * k / n_roll
        n = np.array([np.cos(mu), -np.sin(mu) * np.cos(th), -np.sin(mu) * np.sin(th)])
        e1 = np.cross(n, [0.0, 0.0, 1.0])
        e1 /= max(np.linalg.norm(e1), 1e-9)
        e2 = np.cross(n, e1)
        e2 *= np.sign(e2[2] or 1.0)
        ext = max(half, zh) / np.tan(mu)
        for x0 in np.linspace(lo[0] - ext, hi[0] + ext, n_x):
            pts.append((np.array([x0, 0.0, zc]) + U[..., None] * e1 + Wz[..., None] * e2).reshape(-1, 3))
            keys.append((k, x0))
    d = kernel.distance(design, np.vstack(pts)).reshape(len(keys), -1)
    area = np.clip(0.5 - d / dw, 0, 1).sum(1) * du * dw * np.cos(mu)
    out = []
    for k in range(n_roll):
        idx = [i for i, kk in enumerate(keys) if kk[0] == k]
        x, a = np.array([keys[i][1] for i in idx]), area[idx]
        nz = np.nonzero(a > 1e-6)[0]
        if len(nz) >= 3:
            sl = slice(max(nz[0] - 1, 0), min(nz[-1] + 2, len(a)))
            out.append(wave_drag_area(x[sl], a[sl]))
    return float(np.mean(out)) if out else 0.0


def analyse(design: dict[str, Any], h: float = 0.08) -> dict[str, Any]:
    info = kernel.info(design, h=h, nx=60)
    surf = vlm.surfaces(design["features"])
    main = next((f for f in surf if f.get("type") == "wing"), surf[0] if surf else None)
    out: dict[str, Any] = {"geometry": {k: info[k] for k in ("volume", "wetted_area", "frontal_area", "planform_area", "length", "centroid")}}
    if main is None:
        out["note"] = "no wing-type feature: lift and stability are not computed"
        S, c, b = max(info["frontal_area"], 1e-3), info["length"], 2 * max(abs(info["bbox_lo"][1]), abs(info["bbox_hi"][1]))
    else:
        span = float(main["span"]) + float(main.get("translate", [0, 0, 0])[1])
        b = 2 * span if main.get("mirror", True) else span
        lam = float(main.get("taper", 0.5))
        cr = float(main["root_chord"])
        S = 0.5 * (1 + lam) * cr * b
        c = 2 / 3 * cr * (1 + lam + lam**2) / (1 + lam)
    x_ref = info["centroid"][0]
    res: dict[str, Any] = {}
    for M in (0.3, 0.85):
        if main is None:
            break
        d = vlm.derivatives(design["features"], M, S, c, b, x_ref)
        res[f"M{M}"] = {"CL_alpha_per_rad": d.CL_alpha, "induced_K": d.K, "span_efficiency": 1 / (np.pi * b * b / S * d.K) if d.K > 0 else None, "neutral_point_x": d.x_np,
                        "static_margin_about_centroid": (d.x_np - x_ref) / c, "Cn_beta_per_rad": d.Cn_beta, "Cl_beta_per_rad": d.Cl_beta, "Kv_vortex_lift": d.Kv}
    air = isa(11000.0)
    cf = friction(air, 0.85, info["length"], 0.05)
    out["reference"] = {"S": S, "c": c, "b": b, "x_ref": x_ref}
    out["lattice"] = res
    out["friction_CD0_estimate"] = cf * 1.25 * info["wetted_area"] / S
    out["wave_drag_D_over_q_m2"] = {f"M{m}": _mach_plane(design, info, m) for m in (1.2, 1.6, 2.0)}
    out["wave_drag_CD"] = {k: v / S for k, v in out["wave_drag_D_over_q_m2"].items()}
    return out


def report(a: dict[str, Any]) -> str:
    g = a["geometry"]
    lines = ["Quick aerodynamics (lattice, area rule, friction)", "", f"Volume {g['volume']:.3f} m3, wetted area {g['wetted_area']:.2f} m2, frontal {g['frontal_area']:.3f} m2, length {g['length']:.2f} m",
             f"Reference area {a['reference']['S']:.2f} m2, mean chord {a['reference']['c']:.2f} m, span {a['reference']['b']:.2f} m", ""]
    for k, d in a.get("lattice", {}).items():
        lines.append(f"{k}: CL_alpha {d['CL_alpha_per_rad']:.3f}/rad, span efficiency {d['span_efficiency'] or float('nan'):.3f}, neutral point x {d['neutral_point_x']:.2f} m "
                     f"(static margin about the centroid {100 * d['static_margin_about_centroid']:.1f} % MAC), Cn_beta {d['Cn_beta_per_rad']:.3f}, Cl_beta {d['Cl_beta_per_rad']:.3f}")
    lines += ["", f"Friction CD0 estimate at M0.85, 11 km: {a['friction_CD0_estimate']:.4f}"]
    lines += ["Supersonic wave drag (Mach-plane area rule): " + ", ".join(f"{k} CD {v:.4f}" for k, v in a["wave_drag_CD"].items())]
    if "note" in a:
        lines += ["", a["note"]]
    lines += ["", "Lattice: thin surfaces, attached flow. Area rule: linear theory. Use the 3D flow solver to check."]
    return "\n".join(lines)


def main(path: str, out: str | None = None) -> str:
    a = analyse(json.loads(Path(path).read_text(encoding="utf-8")))
    txt = report(a)
    if out:
        Path(out).write_text(txt, encoding="utf-8")
        Path(out).with_suffix(".json").write_text(json.dumps(a, indent=1, default=float), encoding="utf-8")
    return txt
