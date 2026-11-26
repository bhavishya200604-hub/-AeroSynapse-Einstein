"""Build the start-page templates of AeroSynapse Studio: design files and rendered thumbnails.

    python scripts/make_templates.py

Writes assets/templates/<key>.json, <key>.png (3D render) and index.json (key, group, title, description).
Groups: Start (blank), Jet fighters (the five design-space families, the optimised falcon), Airplanes (light
airplane, airliner, glider, flying wing, blended wing body, UAV), Airfoil wings (NACA 0012, 2412, 4412, biconvex,
diamond, the shape optimiser's NACA 2412, the falcon's optimised fighter section) and Showcase.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "assets" / "templates"


def wing(id_, span, cr, taper, sweep, x, y0, z=0.0, tc=0.12, camber=0.0, dihedral=0.0, twist=0.0, profile="naca", mirror=True, **kw):
    f = {"id": id_, "type": kw.pop("type", "wing"), "op": "add", "span": span, "root_chord": cr, "taper": taper, "sweep": sweep, "dihedral": dihedral, "twist": twist,
         "tc": tc, "camber": camber, "camber_pos": 0.4, "profile": profile, "translate": [x, y0, z], "mirror": mirror}
    f.update(kw)
    return f


def fus(id_, L, w, h, nose=0.18, tail=0.25, n=2.0, z=0.0, droop=0.0):
    st = [[0, 0.02, 0.02, z + droop, 2], [nose * L * 0.4, 0.55 * w, 0.55 * h, z + 0.6 * droop, 2], [nose * L, 0.92 * w, 0.92 * h, z, n], [0.45 * L, w, h, z, n],
          [0.75 * L, 0.75 * w, 0.8 * h, z + 0.05 * h, n], [L, tail * w, tail * h, z + 0.3 * h, 2]]
    return {"id": id_, "type": "fuselage", "op": "add", "stations": [[round(v, 4) for v in s] for s in st]}


def nacelle(id_, x, y, z, L=3.6, r=0.85):
    p = [[0, 0.62 * r], [0.25 * L, r], [0.8 * L, r], [L, 0.75 * r]]
    return {"id": id_, "type": "revolve", "op": "add", "profile": [[0, 0]] + p + [[L, 0]], "smooth": True, "translate": [x, y, z], "mirror": True}


def airfoil_table(dat: Path) -> dict:
    xy = np.loadtxt(dat, skiprows=1)
    le = int(np.argmin(xy[:, 0]))
    up, lo = xy[: le + 1][::-1], xy[le:]
    u = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 61)))
    yu, yl = np.interp(u, up[:, 0], up[:, 1]), np.interp(u, lo[:, 0], lo[:, 1])
    return {"u": np.round(u, 6).tolist(), "t": np.round(yu - yl, 6).tolist(), "zc": np.round(0.5 * (yu + yl), 6).tolist()}


def templates() -> list[tuple[str, str, str, str, dict]]:
    from aerosynapse.fighter import design as D

    out = [("blank", "Start", "Blank design", "An empty design: draw a body in the sketcher, or add parts from the tool palette.", {"name": "new design", "units": "m", "features": []})]
    base = json.loads((ROOT / "results" / "fighter" / "baseline_design.json").read_text())["variables"]
    blurbs = {"conventional": "Wing, strakes, horizontal tail and one fin: the reference fighter.", "canard_delta": "Delta wing with a close-coupled canard, no tailplane.",
              "twin_tail_lerx": "Leading-edge extensions and canted twin fins.", "tailless_blended": "Blended tailless body with blades and a notched wing.",
              "freeform": "Every option of the design space switched on by size: the freest fighter body."}
    for fam, blurb in blurbs.items():
        f = ROOT / "results" / "fighter" / f"{fam}.design.json"
        d = json.loads(f.read_text()) if f.exists() else D.build(D.complete(fam, {k: float(v) for k, v in base.items() if k not in D.FAMILIES[fam]["fixed"]}))
        d["name"] = fam.replace("_", " ")
        out.append((f"jet_{fam}", "Jet fighters", fam.replace("_", " ").capitalize(), blurb, d))
    fal = ROOT / "projects" / "falcon" / "08_export" / "design.json"
    if fal.exists():
        out.append(("jet_falcon", "Jet fighters", "Falcon (optimised)", "The workflow's result: optimised design, airfoil, area-ruled body, sized structure and skin.", json.loads(fal.read_text())))
    out.append(("plane_light", "Airplanes", "Light airplane", "High wing, nose spinner, conventional tail: a four-seat trainer.", {"name": "light airplane", "units": "m", "features": [
        fus("body", 8.2, 0.6, 0.72, nose=0.2, tail=0.18), {"id": "spinner", "type": "cone", "op": "add", "r1": 0.05, "r2": 0.32, "length": 0.55, "translate": [-0.45, 0, 0]},
        wing("wing", 5.0, 1.6, 0.7, 2.0, 2.4, 0.35, z=0.55, tc=0.15, camber=0.02, dihedral=2.0, twist=-2.0),
        wing("tail", 1.6, 0.9, 0.6, 8.0, 7.0, 0.15, z=0.25, tc=0.1, type="tail"), wing("fin", 1.4, 1.1, 0.5, 30.0, 6.8, 0.0, z=0.4, tc=0.1, dihedral=90, mirror=False, type="fin")]}))
    out.append(("plane_airliner", "Airplanes", "Airliner", "Tube and swept wing with two underwing engines.", {"name": "airliner", "units": "m", "features": [
        fus("body", 38.0, 1.95, 2.05, nose=0.1, tail=0.15, droop=-0.2), wing("wing", 15.5, 7.0, 0.25, 27.0, 13.5, 1.6, z=-1.2, tc=0.13, camber=0.02, dihedral=5.0, twist=-3.0),
        nacelle("engines", 12.0, 5.8, -2.4), wing("tail", 6.2, 4.0, 0.35, 32.0, 32.0, 0.6, z=0.6, tc=0.1, type="tail"),
        wing("fin", 6.5, 5.2, 0.35, 38.0, 30.5, 0.0, z=1.6, tc=0.1, dihedral=90, mirror=False, type="fin")]}))
    out.append(("plane_glider", "Airplanes", "Glider", "Slender pod, very high aspect-ratio wing, T-tail.", {"name": "glider", "units": "m", "features": [
        fus("body", 7.0, 0.32, 0.45, nose=0.25, tail=0.1), wing("wing", 7.4, 0.95, 0.35, 1.0, 2.2, 0.2, z=0.3, tc=0.14, camber=0.035, dihedral=3.0, twist=-2.5),
        wing("fin", 1.2, 0.9, 0.55, 25.0, 6.1, 0.0, z=0.2, tc=0.1, dihedral=90, mirror=False, type="fin"),
        wing("tail", 1.3, 0.55, 0.6, 5.0, 6.75, 0.0, z=1.32, tc=0.09, type="tail")]}))
    out.append(("plane_flying_wing", "Airplanes", "Flying wing", "No fuselage and no tail: a swept, washed-out wing with winglets.", {"name": "flying wing", "units": "m", "features": [
        wing("wing", 9.0, 7.5, 0.18, 35.0, 0.0, 0.0, tc=0.12, camber=0.01, twist=-4.0), wing("winglets", 1.2, 1.4, 0.4, 40.0, 7.0, 8.9, z=0.1, tc=0.08, dihedral=75, type="fin")]}))
    out.append(("plane_bwb", "Airplanes", "Blended wing body", "A lofted lifting centre body flowing into the outer wings.", {"name": "blended wing body", "units": "m", "features": [
        {"id": "centre_body", "type": "loft", "op": "add", "smooth": True, "translate": [0, 0, 0], "sections": [
            {"x": 0, "points": [[r * 0.4 * math.cos(a), r * 0.25 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 16, endpoint=False) for r in [1]]},
            {"x": 8, "points": [[4.5 * math.cos(a), 1.4 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]},
            {"x": 20, "points": [[2.2 * math.cos(a), 0.5 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]}]},
        wing("wing", 16.0, 9.0, 0.22, 38.0, 8.0, 3.5, tc=0.12, camber=0.015, twist=-3.0), wing("fins", 2.5, 2.4, 0.45, 40.0, 17.0, 2.0, z=0.4, tc=0.08, dihedral=70, type="fin")]}))
    out.append(("plane_uav", "Airplanes", "UAV (twin boom)", "Pusher pod, straight wing, twin booms and an inverted-V tail.", {"name": "uav", "units": "m", "features": [
        fus("pod", 3.2, 0.32, 0.36, nose=0.3, tail=0.45), wing("wing", 3.4, 0.7, 0.6, 2.0, 1.2, 0.25, z=0.2, tc=0.14, camber=0.03),
        {"id": "booms", "type": "cylinder", "op": "add", "r": 0.06, "length": 3.2, "axis": "x", "translate": [3.0, 1.2, 0.2], "mirror": True},
        wing("vtail", 1.1, 0.5, 0.6, 20.0, 4.3, 1.2, z=0.2, tc=0.1, dihedral=-40, type="tail")]}))
    for key, title, prof, tc, cam, blurb in (("naca0012", "NACA 0012", "naca", 0.12, 0.0, "Symmetric 12 % section: tails, fins, rotor blades."),
                                            ("naca2412", "NACA 2412", "naca", 0.12, 0.02, "2 % camber, 12 % thick: light aircraft wings."),
                                            ("naca4412", "NACA 4412", "naca", 0.12, 0.04, "4 % camber: high-lift, slow flight."),
                                            ("biconvex", "Biconvex 4 %", "biconvex", 0.04, 0.0, "Sharp-edged supersonic section."),
                                            ("diamond", "Diamond 4 %", "diamond", 0.04, 0.0, "Double-wedge supersonic section: lowest wave drag for its thickness.")):
        out.append((f"airfoil_{key}", "Airfoil wings", title, blurb, {"name": title, "units": "m", "features": [wing("wing", 6.0, 2.0, 0.5, 5.0, 0.0, 0.0, tc=tc, camber=cam, profile=prof)]}))
    so = ROOT / "results" / "shapeopt" / "naca2412" / "optimised_section.dat"
    if so.exists():
        w = wing("wing", 6.0, 2.0, 0.5, 5.0, 0.0, 0.0, profile="table")
        w["section"] = airfoil_table(so)
        out.append(("airfoil_optimised", "Airfoil wings", "Optimised NACA 2412", "The shape optimiser's section, after the search over all flow states.", {"name": "optimised airfoil", "units": "m", "features": [w]}))
    if fal.exists():
        fw = next(f for f in json.loads(fal.read_text())["features"] if f["id"] == "wing")
        out.append(("airfoil_falcon", "Airfoil wings", "Fighter section (falcon)", "The workflow's optimised thin section, in its own swept wing.", {"name": "falcon wing", "units": "m", "features": [{**fw, "translate": [0, 0, 0]}]}))
    sc = ROOT / "projects" / "tools_showcase.json"
    if sc.exists():
        out.append(("showcase_tools", "Showcase", "Curve tools", "Revolved pods, a sheet fairing and an airfoil-table wing on a fighter.", json.loads(sc.read_text())))
    out.append(("showcase_math", "Showcase", "Math features", "Equation surface, superellipsoid, helix sweep, gyroid lattice and a sheet-metal bracket.", {"name": "math features", "units": "m", "features": [
        {"id": "pod", "type": "implicit", "op": "add", "expr": "y^2 + z^2 - (0.35 + 0.1*sin(3*x))^2", "size": [5, 1.2, 1.2], "translate": [2.5, 0, 0]},
        {"id": "block", "type": "superellipsoid", "op": "add", "radii": [1.0, 0.6, 0.45], "n1": 0.35, "n2": 0.35, "translate": [7.0, 0, 0]},
        {"id": "spring", "type": "sweep", "op": "add", "x": "0.25*t", "y": "0.5*cos(t)", "z": "0.5*sin(t)", "t0": 0, "t1": 25.13, "samples": 240, "radius": 0.06, "translate": [0.5, 0, 1.6]},
        {"id": "lattice", "type": "lattice", "op": "add", "size": [1.2, 0.8, 0.6], "cell": 0.2, "wall": 0.02, "kind": "gyroid", "translate": [9.5, 0, 0]},
        {"id": "bracket", "type": "bentplate", "op": "add", "profile": [[0, 0], [0.5, 0], [0.5, 0.4], [0.9, 0.4]], "thickness": 0.01, "width": 0.4, "bend_radius": 0.03,
         "translate": [6.5, 1.3, -0.5]}]}))
    return out


def render_airfoil(design: dict, path: Path, title: str) -> None:
    """The section itself: what an airfoil template is about."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aerosynapse.structures.wing import Airfoil

    af = Airfoil.of(design["features"][0])
    u = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 160)))
    yu, yl = af.camber_line(u) + af.half(u), af.camber_line(u) - af.half(u)
    fig = plt.figure(figsize=(4.8, 3.0), dpi=100)
    fig.patch.set_facecolor("#0b1a36")
    ax = fig.add_axes((0.04, 0.1, 0.92, 0.8))
    ax.set_facecolor("#0b1a36")
    ax.fill_between(u, yl, yu, color="#22d3ee", alpha=0.35)
    ax.plot(np.r_[u, u[::-1]], np.r_[yu, yl[::-1]], color="#67e8f9", lw=2)
    ax.plot(u, af.camber_line(u), "--", color="#f472b6", lw=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, -0.22, title, color="#cfe6ff", ha="center", fontsize=11)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def render(design: dict, path: Path) -> None:
    import pyvista as pv

    from aerosynapse.cad import kernel

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(480, 300))
    p.set_background("#07122a", top="#1d4078")
    if design["features"]:
        obj = OUT / "_tmp.obj"
        info = kernel.info(design, h=0.1, nx=30)
        L = max(np.subtract(info["bbox_hi"], info["bbox_lo"]))
        kernel.mesh(design, obj, h=max(0.02, L / 160))
        m = pv.read(str(obj))
        p.add_mesh(m, color="#dbe7f5", smooth_shading=True, specular=0.5, specular_power=20)
        c = np.array(m.center)
        p.camera.position = tuple(c + np.array([-0.9, -1.1, 0.6]) * L)
        p.camera.focal_point = tuple(c)
        p.camera.up = (0, 0, 1)
        obj.unlink(missing_ok=True)
    else:
        p.add_text("+", position=(205, 95), font_size=60, color="#7dd3fc")
        p.add_text("blank", position=(195, 60), font_size=14, color="#cfe6ff")
    p.screenshot(str(path))
    p.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for key, group, title, blurb, d in templates():
        (OUT / f"{key}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        if key.startswith("airfoil_"):
            render_airfoil(d, OUT / f"{key}.png", title)
        else:
            render(d, OUT / f"{key}.png")
        index.append({"key": key, "group": group, "title": title, "description": blurb})
        print("template", key)
    (OUT / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{len(index)} templates in {OUT}")


if __name__ == "__main__":
    main()
