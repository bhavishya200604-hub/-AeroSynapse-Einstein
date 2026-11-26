"""Build the whole-aircraft comparison page from results/aircraft/study.json."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
from _common import RESULTS

from aerosynapse.aircraft import analysis as an
from aerosynapse.aircraft import datahub as dh
from aerosynapse.aircraft import layers as Y
from aerosynapse.aircraft import optimise as op
from aerosynapse.aircraft import panel3d as p3
from aerosynapse.aircraft import vehicle as vh

SCRIPTS = RESULTS.parent / "scripts"


def outline(d: dict) -> dict:
    """Top-view outlines of a design: body and wing, in metres."""
    v = vh.build(d)
    x, a = v.x_st, v.a
    body = np.r_[np.c_[x, a], np.c_[x[::-1], -a[::-1]]]
    s = v.strips
    k = [j for j in range(len(s.y)) if not s.is_body[j] and abs(v.mesh.phi[j]) < 0.3]
    # strips are stored at their centres, so extend to the true root and tip edges for the drawing
    ds = v.mesh.ds[k]
    ye = np.r_[s.y[k][0] - ds[0] / 2, s.y[k], s.y[k][-1] + ds[-1] / 2]
    xl = np.interp(ye, s.y[k], s.x_le[k])
    xl[0] -= (s.x_le[k][1] - s.x_le[k][0]) / max(s.y[k][1] - s.y[k][0], 1e-6) * ds[0] / 2
    xl[-1] += (s.x_le[k][-1] - s.x_le[k][-2]) / max(s.y[k][-1] - s.y[k][-2], 1e-6) * ds[-1] / 2
    ce = np.interp(ye, s.y[k], s.c[k])
    le = np.c_[xl, ye]
    te = np.c_[xl + ce, ye]
    wing = np.r_[le, te[::-1]]
    polys = [body, wing, wing * np.array([1.0, -1.0])]
    # drawn with the nose to the left, span up the page
    pts = np.vstack(polys)
    return {"polys": [np.round(q, 3).tolist() for q in polys], "x0": float(pts[:, 0].min()) - 1, "x1": float(pts[:, 0].max()) + 1, "y0": float(pts[:, 1].min()) - 1, "y1": float(pts[:, 1].max()) + 1}


def fam(f: dict) -> dict:
    Js = f["seeds_J"]
    return {"s": f["summary"], "d": f["design"], "J": f["J"], "Js": Js, "spread": 100 * (max(Js) - min(Js)) / max(min(Js), 1e-9), "off": f["off_design"], "on": f["off_names"], "unmet": sum(x > 1e-3 for x in f["constraints"].values()), "ok": all(x <= 1e-3 for x in f["constraints"].values()), "o": outline(f["design"]), "v": f["verify"]}


def sensitivity(a: dict) -> list[dict]:
    """Re-score the final designs with the flat-body mass penalty doubled and tripled.

    The penalty is the least certain number in the study, and the ranking of unconventional
    bodies rests on it, so the ranking is tested against it rather than assumed.
    """
    st = op.Study(an.Mission(), vh.PRESETS["tube_and_wing"], 41000.0)
    rows = []
    for k, f in a["families"].items():
        v = vh.build(f["design"])
        fuels = [an.evaluate(v, replace(st.m, k_flat=q)).fuel for q in (0.10, 0.20, 0.30, 0.45)]
        rows.append({"name": k, "fuel_t": [x / 1000 for x in fuels]})
    return rows


def validation() -> list[list[str]]:
    from aerosynapse.aircraft.atmosphere import isa

    t = np.linspace(0, np.pi, 34)
    ph = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    rows = []
    for fin, exact in ((1.0, 1.5), (6.0, p3.ellipsoid_vmax(6.0))):
        G = np.stack([np.outer(-fin * np.cos(t), np.ones_like(ph)), np.outer(np.sin(t), np.cos(ph)), np.outer(np.sin(t), np.sin(ph))], -1)
        s = p3.solve(p3.from_grid(G))
        rows.append(["3D panels", f"peak speed, {'sphere' if fin == 1 else '6:1 ellipsoid'}", f"{np.sqrt((s.V**2).sum(1)).max():.4f}", f"{exact:.4f} exact"])
    L, A = 30.0, 4.0
    x = np.linspace(0, L, 600)
    S = A * np.clip(4 * (x / L) * (1 - x / L), 0, None) ** 1.5
    rows.append(["Wave drag", "Sears-Haack D/q", f"{an.wave_drag_area(x, S):.4f}", f"{9 * np.pi * A**2 / (2 * L**2):.4f} exact"])
    rows.append(["Atmosphere", "density at 11 km", f"{isa(11000).rho:.4f}", "0.3639 ISA table"])
    rows.append(["Thermal", "total temperature, Mach 2.02, 16 km, ISA+5", f"{Y.t_total(isa(16000, 5.0).T, 2.02) - 273.15:.0f} °C", "127 °C Concorde limit"])
    rows.append(["Geometry", "pseudosphere Gaussian curvature", "matches -1/a² within 2 %", "exact theory"])
    rows.append(["Geometry", "Moebius map keeps hyperbolic distance", "to 10 digits", "exact theory"])
    rows.append(["Cycle", "ideal turbojet specific thrust", "within 4 % (fuel mass)", "closed form"])
    rows.append(["Drag level", "airliner zero-lift drag", "about 0.018", "0.018 to 0.021, weak public data"])
    return rows


def main() -> None:
    s = json.loads((RESULTS / "aircraft" / "study.json").read_text(encoding="utf-8"))
    a, b = s["missions"]["airliner"], s["missions"]["supersonic"]
    fa = {k: fam(f) for k, f in a["families"].items()}
    fs = {k: fam(f) for k, f in b["families"].items()}
    best_ss = min((v for v in fs.values() if v["ok"]), key=lambda v: v["s"]["fuel_t"], default=next(iter(fs.values())))
    m = an.SUPERSONIC
    D = {
        "air": {"base": a["baseline"], "baseO": outline(a["baseline_design"]), "baseV": a["baseline_verify"], "fam": fa},
        "ss": {"base": b["baseline"], "baseO": outline(b["baseline_design"]), "fam": fs, "best": best_ss, "mach": m["mach"], "alt": m["altitude"]},
        "thermal": b["baseline_thermal"],
        "mat_limit_C": Y.MATERIALS[m["material"]].T_max - 273.15,
        "val": validation(),
        "sens": sensitivity(a),
        "limits": [
            "<b>No 3D RANS or LES was run.</b> Whole-aircraft CFD does not fit on this computer. Friction uses flat-plate laws and form factors, checked by panels and integral boundary layers on the body only.",
            "<b>Mass is calibrated, not predicted.</b> The baseline is set to 41 t empty. Flat pressurised bodies carry a stated mass penalty, and the ranking of unconventional bodies depends on it.",
            "<b>Flutter is comparative.</b> The baseline margin of 1.3 is assumed. Stability derivatives come from simple estimates.",
            "<b>Engines are not integrated.</b> There are no inlet, nacelle or boundary-layer ingestion effects.",
            "<b>Slender-body wave drag is weak for wide flat shapes</b>, so the supersonic free-form result is less reliable than the slender one.",
            "<b>The cabin check counts floor area with headroom.</b> It does not lay out seats, doors or evacuation.",
        ],
        "missing": dh.reference()["missing"],
    }
    t = (SCRIPTS / "shapeopt_report_template.html").read_text(encoding="utf-8")
    css = t[t.index("<style>") + 7 : t.index("</style>")]
    js = t[t.index("const $=i=>") : t.index("function render(){")]
    page = (SCRIPTS / "aircraft_report_template.html").read_text(encoding="utf-8")
    page = page.replace("/*__CSS__*/", css).replace("/*__HELPERS__*/", js).replace("/*__DATA__*/null", json.dumps(D, separators=(",", ":"), default=float))
    out = RESULTS / "aircraft" / "report.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out, round(len(page) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
