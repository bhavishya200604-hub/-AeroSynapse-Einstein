"""Export and documentation: neutral CAD files, drawings, bill of materials, mass properties, report and package.

Files (in ``08_export``):

* ``<name>.stl`` (binary) and ``<name>.obj``: the surface mesh at the profile's resolution.
* ``<name>.glb``: binary glTF 2.0 (Y up, metres) with a metallic material, for viewers, game engines and the web.
* ``<name>.step``: ISO 10303-21, AP214 (automotive_design schema) with a FACETED_BREP of a coarser mesh. It is a
  faceted solid, not NURBS surfaces: good for packaging, clearance and CFD meshing in other tools.
* ``<name>_3view.dxf``: top, front and side outlines in millimetres (DXF R12, layers per view).
* ``<name>_drawing.pdf/.svg/.png``: an A3 three-view general-arrangement drawing with dimensions and a title block;
  outlines are traced on the exact distance field (sphere tracing), so thin edges stay sharp.
* ``bom.csv``: the feature tree as a bill of materials (type, operation, volume, wetted area, assigned zone,
  skin material and gauge from the thermal stage, skin mass).
* ``mass_properties.json``, ``design.json`` (final revision), ``report.html``, ``report.pdf``, ``report.docx`` and
  ``report.txt`` (the whole workflow, the same content in each format).
* ``../<name>_package.zip`` with ``manifest.json``: every project file with its size and SHA-256.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import struct
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.cad import kernel
from aerosynapse.workflow.project import KEYS, STAGES, Project, now, sha256


# ------------------------------------------------------------------ meshes
def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    V, F = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V, float), np.array(F, np.int64)


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], n)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def write_glb(path: Path, V: np.ndarray, F: np.ndarray, name: str) -> None:
    Vy = np.c_[V[:, 0], V[:, 2], -V[:, 1]].astype(np.float32)  # glTF is Y up
    N = vertex_normals(Vy.astype(float), F).astype(np.float32)
    idx = F.astype(np.uint32).ravel()
    blobs = [Vy.tobytes(), N.tobytes(), idx.tobytes()]
    offs, buf = [], b""
    for b in blobs:
        offs.append(len(buf))
        buf += b + b"\0" * ((4 - len(b) % 4) % 4)
    doc = {
        "asset": {"version": "2.0", "generator": "AeroSynapse design workflow"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"name": "airframe", "pbrMetallicRoughness": {"baseColorFactor": [0.74, 0.79, 0.86, 1.0], "metallicFactor": 0.6, "roughnessFactor": 0.42}}],
        "buffers": [{"byteLength": len(buf)}],
        "bufferViews": [{"buffer": 0, "byteOffset": offs[0], "byteLength": len(blobs[0]), "target": 34962},
                        {"buffer": 0, "byteOffset": offs[1], "byteLength": len(blobs[1]), "target": 34962},
                        {"buffer": 0, "byteOffset": offs[2], "byteLength": len(blobs[2]), "target": 34963}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(Vy), "type": "VEC3", "min": Vy.min(0).tolist(), "max": Vy.max(0).tolist()},
                      {"bufferView": 1, "componentType": 5126, "count": len(N), "type": "VEC3"},
                      {"bufferView": 2, "componentType": 5125, "count": len(idx), "type": "SCALAR"}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(buf)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(buf), b"BIN\0") + buf)


def write_step(path: Path, V: np.ndarray, F: np.ndarray, name: str) -> None:
    """ISO 10303-21 file, AP214, one FACETED_BREP solid."""
    lines = []
    k = [0]

    def ent(s: str) -> int:
        k[0] += 1
        lines.append(f"#{k[0]}={s};")
        return k[0]

    ctx = ent("APPLICATION_CONTEXT('automotive design')")
    ent(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{ctx})")
    pc = ent(f"PRODUCT_CONTEXT('',#{ctx},'mechanical')")
    prod = ent(f"PRODUCT('{name}','{name}','airframe outer mould line',(#{pc}))")
    ent(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{prod}))")
    pdf = ent(f"PRODUCT_DEFINITION_FORMATION('1','',#{prod})")
    pdc = ent(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{ctx},'design')")
    pd = ent(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdc})")
    pds = ent(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
    lu = ent("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))")
    au = ent("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    su = ent("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = ent(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#{lu},'distance_accuracy_value','')")
    gctx = ent(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))GLOBAL_UNIT_ASSIGNED_CONTEXT((#{lu},#{au},#{su}))REPRESENTATION_CONTEXT('',''))")
    o = ent("CARTESIAN_POINT('',(0.,0.,0.))")
    dz = ent("DIRECTION('',(0.,0.,1.))")
    dx = ent("DIRECTION('',(1.,0.,0.))")
    ax = ent(f"AXIS2_PLACEMENT_3D('',#{o},#{dz},#{dx})")
    pid = [ent(f"CARTESIAN_POINT('',({x:.6f},{y:.6f},{z:.6f}))") for x, y, z in V]
    faces = []
    for a, b, c in F:
        loop = ent(f"POLY_LOOP('',(#{pid[a]},#{pid[b]},#{pid[c]}))")
        bound = ent(f"FACE_OUTER_BOUND('',#{loop},.T.)")
        faces.append(ent(f"FACE('',(#{bound}))"))
    shell = ent("CLOSED_SHELL('',(" + ",".join(f"#{i}" for i in faces) + "))")
    brep = ent(f"FACETED_BREP('{name}',#{shell})")
    rep = ent(f"FACETED_BREP_SHAPE_REPRESENTATION('',(#{ax},#{brep}),#{gctx})")
    ent(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    head = ["ISO-10303-21;", "HEADER;", "FILE_DESCRIPTION(('AeroSynapse airframe, faceted B-rep'),'2;1');",
            f"FILE_NAME('{path.name}','{time.strftime('%Y-%m-%dT%H:%M:%S')}',('AeroSynapse'),('AeroSynapse'),'AeroSynapse design workflow','AeroSynapse','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;"]
    path.write_text("\n".join(head + lines + ["ENDSEC;", "END-ISO-10303-21;", ""]), encoding="ascii")


# ------------------------------------------------------------------ drawings
VIEWS = {"TOP": (0, 1), "SIDE": (0, 2), "FRONT": (1, 2)}  # plan (x, y), side (x, z), front (y, z) looking aft


def silhouette(V: np.ndarray, F: np.ndarray, axes: tuple[int, int], res: float = 0.015) -> list[np.ndarray]:
    """Outline polylines of the projection of a mesh onto two axes (metres)."""
    import contourpy
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure

    P = V[:, list(axes)]
    lo, hi = P.min(0) - 3 * res, P.max(0) + 3 * res
    nx, ny = (np.ceil((hi - lo) / res)).astype(int)
    fig = Figure(figsize=(nx / 100, ny / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.axis("off")
    ax.add_collection(PolyCollection(P[F], facecolors="k", edgecolors="k", linewidths=0.4, antialiased=False))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[::-1, :, 0] < 128
    xs = np.linspace(lo[0], hi[0], img.shape[1])
    ys = np.linspace(lo[1], hi[1], img.shape[0])
    lines = contourpy.contour_generator(xs, ys, img.astype(float)).lines(0.5)
    return [ln for ln in lines if len(ln) > 8]


def silhouette_sdf(design: dict[str, Any], axes: tuple[int, int], lo: np.ndarray, hi: np.ndarray, res: float = 0.02, max_iter: int = 600) -> list[np.ndarray]:
    """Exact outline of the projection of the design along the third axis, by sphere tracing the distance field:
    for every column the smallest distance met along the ray (negative once the ray enters the body). Unlike the
    mesh silhouette this keeps razor-thin trailing edges, which surface nets lose at drawing resolution."""
    import contourpy

    a, b = axes
    c = 3 - a - b
    ua = np.arange(lo[a] - 3 * res, hi[a] + 3 * res, res)
    ub = np.arange(lo[b] - 3 * res, hi[b] + 3 * res, res)
    UA, UB = np.meshgrid(ua, ub, indexing="xy")
    n = UA.size
    t = np.full(n, lo[c] - 0.05)
    best = np.full(n, np.inf)
    active = np.ones(n, bool)
    for _ in range(max_iter):
        idx = np.nonzero(active)[0]
        if not len(idx):
            break
        P = np.zeros((len(idx), 3))
        P[:, a], P[:, b], P[:, c] = UA.ravel()[idx], UB.ravel()[idx], t[idx]
        d = kernel.distance(design, P)
        best[idx] = np.minimum(best[idx], d)
        t[idx] += np.maximum(d, 0.4 * res)
        active[idx] = (d > 0) & (t[idx] < hi[c] + 0.05)
    f = (best - 0.25 * res).reshape(UA.shape)
    lines = contourpy.contour_generator(ua, ub, f).lines(0.0)
    return [ln for ln in lines if len(ln) > 8]


def write_dxf(path: Path, views: dict[str, tuple[list[np.ndarray], np.ndarray]], texts: list[tuple[str, float, float, float, str]]) -> None:
    """DXF R12 entities: each view's polylines (mm, offset placed), and text items (layer, x, y, height, text)."""
    out = ["0", "SECTION", "2", "ENTITIES"]
    for layer, (polys, off) in views.items():
        for pl_ in polys:
            out += ["0", "POLYLINE", "8", layer, "66", "1", "70", "1" if np.allclose(pl_[0], pl_[-1]) else "0"]
            for x, y in pl_ * 1000 + off:
                out += ["0", "VERTEX", "8", layer, "10", f"{x:.2f}", "20", f"{y:.2f}"]
            out += ["0", "SEQEND"]
    for layer, x, y, h, t in texts:
        out += ["0", "TEXT", "8", layer, "10", f"{x:.2f}", "20", f"{y:.2f}", "40", f"{h:.2f}", "1", t]
    out += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(out) + "\n", encoding="ascii", errors="replace")


def drawing_sheet(base: Path, V: np.ndarray, F: np.ndarray, title: dict[str, str], design: dict[str, Any] | None = None) -> dict[str, Any]:
    """A3 landscape general-arrangement drawing: plan, side and front views at a common scale, dimensions, title block."""
    from aerosynapse.workflow.stages import NAVY, plt

    pl = plt()
    lo, hi = V.min(0), V.max(0)
    if design is not None:  # extents and outlines from the exact distance field (the mesh loses thin fins and edges)
        bb = kernel.info(design, h=0.06, nx=40)
        lo, hi = np.minimum(lo, bb["bbox_lo"]), np.maximum(hi, bb["bbox_hi"])
        outlines = {k: silhouette_sdf(design, a, lo, hi, res=max(0.012, float((hi - lo).max()) / 900)) for k, a in VIEWS.items()}
    else:
        outlines = {k: silhouette(V, F, a) for k, a in VIEWS.items()}
    L, B, H = hi - lo
    # common scale so that plan + front fit across and plan + side fit down (sheet units mm)
    scale = next(s for s in (20, 25, 40, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 1000) if (L + B) * 1000 / s < 330 and (B + H) * 1000 / s < 200)
    W_, H_ = 420.0, 297.0
    fig = pl.figure(figsize=(W_ / 25.4, H_ / 25.4), layout="none")  # the sheet places its own axes
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W_)
    ax.set_ylim(0, H_)
    ax.axis("off")
    ax.add_patch(pl.Rectangle((10, 10), W_ - 20, H_ - 20, fill=False, lw=1.4, ec="k"))
    ax.add_patch(pl.Rectangle((5, 5), W_ - 10, H_ - 10, fill=False, lw=0.4, ec="k"))
    s = 1000 / scale
    # placement (sheet mm): plan view upper left, side view below it, front view to the right of the side view
    origin = {"TOP": np.array([25 - lo[0] * s, 275 - B * s - lo[1] * s - 5]), "SIDE": np.array([25 - lo[0] * s, 95 - lo[2] * s]),
              "FRONT": np.array([25 + L * s + 25 - lo[1] * s, 95 - lo[2] * s])}
    for k, polys in outlines.items():
        for ln in polys:
            q = ln * s + origin[k]
            ax.plot(q[:, 0], q[:, 1], color="k", lw=0.7)
        ctr = origin[k] + np.array([(lo[VIEWS[k][0]] + hi[VIEWS[k][0]]) / 2, lo[VIEWS[k][1]]]) * s
        ax.text(ctr[0], ctr[1] - 7, {"TOP": "PLAN VIEW", "SIDE": "SIDE VIEW", "FRONT": "FRONT VIEW (looking aft)"}[k], ha="center", fontsize=7, weight="bold")
    # centre lines
    yt = origin["TOP"][1]
    ax.plot([origin["TOP"][0] + lo[0] * s - 4, origin["TOP"][0] + hi[0] * s + 4], [yt, yt], ls=(0, (8, 2, 2, 2)), color="k", lw=0.4)
    xf = origin["FRONT"][0]
    ax.plot([xf, xf], [origin["FRONT"][1] + lo[2] * s - 4, origin["FRONT"][1] + hi[2] * s + 4], ls=(0, (8, 2, 2, 2)), color="k", lw=0.4)

    def dim(p0: np.ndarray, p1: np.ndarray, off: np.ndarray, text: str) -> None:
        a, b = p0 + off, p1 + off
        ax.annotate("", a, b, arrowprops={"arrowstyle": "<|-|>", "lw": 0.5, "color": "k", "shrinkA": 0, "shrinkB": 0, "mutation_scale": 6})
        for p_, q_ in ((p0, a), (p1, b)):
            ax.plot([p_[0], q_[0]], [p_[1], q_[1]], color="k", lw=0.3)
        m = 0.5 * (a + b)
        vertical = abs(b[0] - a[0]) < abs(b[1] - a[1])
        ax.text(m[0] + (2 if vertical else 0), m[1] + (0 if vertical else 1.5), text, fontsize=6.5, ha="left" if vertical else "center", rotation=90 if vertical else 0, va="center" if vertical else "bottom")

    o = origin["SIDE"]
    dim(o + np.array([lo[0], lo[2]]) * s, o + np.array([hi[0], lo[2]]) * s, np.array([0, -12]), f"{L * 1000:.0f}")
    o = origin["TOP"]
    dim(o + np.array([hi[0], lo[1]]) * s, o + np.array([hi[0], hi[1]]) * s, np.array([10, 0]), f"{B * 1000:.0f}")
    o = origin["FRONT"]
    dim(o + np.array([hi[1], lo[2]]) * s, o + np.array([hi[1], hi[2]]) * s, np.array([10, 0]), f"{H * 1000:.0f}")
    # title block
    x0, y0, w, h = W_ - 10 - 180, 10, 180, 56
    ax.add_patch(pl.Rectangle((x0, y0), w, h, fill=False, lw=1.0, ec="k"))
    rows = [("TITLE", title.get("title", "")), ("PROJECT", title.get("project", "")), ("DRAWING NO.", title.get("number", "")), ("REVISION", title.get("revision", "")),
            ("DATE", title.get("date", "")), ("SCALE", f"1:{scale}  (A3)"), ("UNITS", "millimetres; third-angle projection"), ("MATERIAL", title.get("material", "")),
            ("MASS", title.get("mass", "")), ("SOFTWARE", "AeroSynapse Studio design workflow")]
    rh = h / len(rows)
    for i, (k, v) in enumerate(rows):
        yy = y0 + h - (i + 1) * rh
        ax.plot([x0, x0 + w], [yy, yy], color="k", lw=0.3)
        ax.text(x0 + 2, yy + rh / 2, k, fontsize=5.5, va="center", color="#444")
        ax.text(x0 + 32, yy + rh / 2, v, fontsize=6.5, va="center", weight="bold" if k == "TITLE" else "normal", color=NAVY if k == "TITLE" else "k")
    ax.plot([x0 + 30, x0 + 30], [y0, y0 + h], color="k", lw=0.3)
    ax.text(20, 16, "General arrangement. Outlines traced on the exact geometry (distance field); dimensions are overall extents.", fontsize=6, color="#333")
    out = {}
    for ext in ("pdf", "svg", "png"):
        f = base.with_name(base.name + f"_drawing.{ext}")
        with pl.rc_context({"figure.constrained_layout.use": False}):  # the sheet places its own axes
            fig.savefig(f, dpi=200 if ext == "png" else None)
        out[ext] = f.name
    pl.close(fig)
    out["scale"] = f"1:{scale}"
    out["outlines"] = {k: v for k, v in outlines.items()}
    out["origin_mm"] = {k: v.tolist() for k, v in origin.items()}
    out["s"] = s
    return out


# ------------------------------------------------------------------ bill of materials
ZONE_OF = {"fuselage": "centre_fuselage", "wing": "wing_box", "canard": "control", "tail": "control", "fin": "control", "strake": "wing_le", "blade": "control",
           "ellipsoid": "fwd_fuselage", "revolve": "aft_fuselage", "sheet": "control", "loft": "centre_fuselage", "extrude": "control", "lattice": "centre_fuselage"}


def bom(p: Project, design: dict[str, Any], h: float) -> list[dict[str, Any]]:
    skin = design.get("skin", {})
    from aerosynapse.aircraft import materials as mt

    names = {m.name: m for m in mt.LIBRARY.values()}
    rows = []
    for i, f in enumerate(design["features"]):
        one = {"features": [{**f, "op": "add"}]}
        try:
            inf = kernel.info(one, h=h, nx=40)
            vol, wet = inf["volume"], inf["wetted_area"]
        except Exception:
            vol = wet = float("nan")
        zone = ZONE_OF.get(f.get("type", ""), "centre_fuselage")
        sk = skin.get(zone) or {}
        m = names.get(sk.get("material", ""))
        skin_mass = (m.rho * sk["thickness_mm"] * 1e-3 * wet) if (m and f.get("op", "add") == "add" and np.isfinite(wet)) else 0.0
        rows.append({"item": i + 1, "id": f.get("id", ""), "type": f.get("type", ""), "operation": f.get("op", "add"), "mirrored": bool(f.get("mirror", False)),
                     "volume_m3": round(vol, 4), "wetted_area_m2": round(wet, 3), "zone": zone, "skin_material": sk.get("material", ""),
                     "skin_mm": sk.get("thickness_mm", ""), "skin_mass_kg": round(skin_mass, 1), "enabled": f.get("enabled", True)})
    return rows


# ------------------------------------------------------------------ report
def _img(path: Path) -> str:
    return f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="{html.escape(path.stem)}">' if path.exists() else ""


def _n(v: Any, spec: str = ".5f") -> str:
    """A number, or 'not converged' for a value the solver did not reach (stored as null)."""
    return format(v, spec) if isinstance(v, (int, float)) and v == v else "not converged"


def _fmt(v: Any, nd: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return html.escape(str(v))


def report(p: Project, exports: dict[str, Any]) -> str:
    st = {k: p.result(k) for k in KEYS}
    sec = []
    rows = "".join(f"<tr><td>{i + 1}</td><td>{html.escape(t)}</td><td class='s-{p.status(k)}'>{p.status(k)}</td><td>{_fmt(p.meta['stages'][k].get('seconds', ''), 3)}</td>"
                   f"<td>{html.escape(str(p.meta['stages'][k].get('profile', '')))}</td></tr>" for i, (k, t) in enumerate(STAGES))
    sec.append(f"<h2>Stages</h2><table><tr><th>#</th><th>stage</th><th>status</th><th>seconds</th><th>profile</th></tr>{rows}</table>")
    # scorecard progression
    cards = [("concept", st["concept"].get("scorecard")), ("design", st["design"].get("scorecard_after")), ("wing", st["wing"].get("scorecard_after")),
             ("body", st["body"].get("scorecard_after"))]
    cards = [(k, c) for k, c in cards if c]
    if cards:
        keys = list(cards[0][1])
        head = "".join(f"<th>{k}</th>" for k, _ in cards)
        body = "".join("<tr><td>" + html.escape(m) + "</td>" + "".join(f"<td>{_fmt(c.get(m))}</td>" for _, c in cards) + "</tr>" for m in keys)
        sec.append(f"<h2>Scorecard from stage to stage</h2><p>The same quick analysis (geometry, vortex lattice, friction, Mach-plane area rule) of each stage's design.</p>"
                   f"<table><tr><th>quantity</th>{head}</tr>{body}</table>")
    req = p.meta["requirements"]
    sec.append("<h2>Requirements</h2><pre>" + html.escape(json.dumps(req, indent=1)) + "</pre>")
    c = st["concept"]
    if c:
        perf = c.get("performance", {})
        sec.append(f"<h2>1 Basic design</h2><p>Origin: {html.escape(c.get('origin', ''))}. {len(c.get('features', []))} features.</p>"
                   + (f"<p>Full performance model: take-off mass {perf['W0_kg']:.0f} kg, empty {perf['empty_kg']:.0f} kg, feasible {perf['feasible']}.</p>" if perf else ""))
    d = st["design"]
    if d:
        extra = ""
        if "performance_after" in d:
            a, b = d["performance_before"], d["performance_after"]
            extra = f"<p>Take-off mass {a['W0_kg']:.0f} &rarr; {b['W0_kg']:.0f} kg; feasible {a['feasible']} &rarr; {b['feasible']}.</p>"
        sec.append(f"<h2>2 Design optimisation</h2><p>{html.escape(d.get('method', ''))}</p><p>Objective {_fmt(d.get('J_before'))} &rarr; {_fmt(d.get('J_after'))} "
                   f"after {d.get('evaluations')} evaluations.</p>{extra}{_img(p.stage_dir('design') / 'convergence.png')}")
    w = st["wing"]
    if w:
        trs = "".join(f"<tr><td>{html.escape(s['state'])}</td><td>{s['mach_normal']:.3f}</td><td>{s['cl_normal']:.3f}</td><td>{1e4 * s['cd_before']:.1f}</td><td>{1e4 * s['cd_after']:.1f}</td>"
                      f"<td>{100 * (s['cd_after'] / s['cd_before'] - 1):+.1f} %</td></tr>" for s in w["states"])
        xfs = w.get("xfoil", {})
        if "error" in xfs or not xfs:
            xft = f"<p>XFOIL check: {html.escape(str(xfs.get('error', 'not run')))}</p>"
        else:
            xft = "<p>XFOIL check (independent of NeuralFoil): " + "; ".join(
                f"{html.escape(k)} (M {v['mach']:.2f}, c<sub>l</sub> {v['cl']:.2f}): c<sub>d</sub> {_n(v['before']['cd'])} &rarr; {_n(v['after']['cd'])}" for k, v in xfs.items()) + ".</p>"
        sec.append(f"<h2>3 Wing optimisation</h2><p>{html.escape(w['method'])}.</p><p>Objective {w['J_before']:.4f} &rarr; {w['J_after']:.4f}; supersonic wave drag of the section "
                   f"x{w['supersonic_wave_ratio']:.3f}; thickness {w['thickness_before']:.4f} &rarr; {w['thickness_after']:.4f}.</p>"
                   f"<table><tr><th>state</th><th>normal Mach</th><th>normal c<sub>l</sub></th><th>c<sub>d</sub> before, counts</th><th>after</th><th>change</th></tr>{trs}</table>{xft}"
                   f"{_img(p.stage_dir('wing') / 'section.png')}")
    b = st["body"]
    if b and "before" in b:
        bb, ba = b["before"], b["after"]
        sec.append(f"<h2>4 Body optimisation</h2><p>{html.escape(b['method'])}.</p><table><tr><th></th><th>before</th><th>after</th></tr>"
                   f"<tr><td>wave D/q at M1.2, m&sup2;</td><td>{bb['w12']:.4f}</td><td>{ba['w12']:.4f}</td></tr><tr><td>wave D/q at M1.6, m&sup2;</td><td>{bb['w16']:.4f}</td><td>{ba['w16']:.4f}</td></tr>"
                   f"<tr><td>volume, m&sup3;</td><td>{bb['volume']:.3f}</td><td>{ba['volume']:.3f}</td></tr><tr><td>wetted area, m&sup2;</td><td>{bb['wetted']:.2f}</td><td>{ba['wetted']:.2f}</td></tr>"
                   f"<tr><td>envelope violation, m</td><td>{bb['envelope_violation_m']:.3f}</td><td>{ba['envelope_violation_m']:.3f}</td></tr></table>{_img(p.stage_dir('body') / 'area_rule.png')}")
    s = st["structure"]
    if s:
        wg = s["wing"]
        ms = "".join(f"<tr><td>{html.escape(k)}</td><td class='{'ok' if v >= -1e-6 else 'bad'}'>{v:+.3f}</td></tr>" for k, v in wg["margins_min"].items())
        tr = "".join(f"<tr><td>{html.escape(t['material'])}</td><td>{_fmt(t.get('box_mass_kg', '-'))}</td><td>{_fmt(t.get('strength_sized_kg', '-'))}</td><td>{_fmt(t.get('stiffness_scale', '-'))}</td>"
                     f"<td>{html.escape(t.get('note', ''))}</td></tr>" for t in s["material_trade"])
        sec.append(f"<h2>5 Structure and design codes</h2><p>Code: {html.escape(s['code'])}; limit n = {wg['n_limit']}, ultimate {wg['n_ultimate']}; take-off mass {s['W0_kg']:.0f} kg. "
                   f"Wing box in {html.escape(wg['material'])} at {wg['temperature_K']:.0f} K: root bending moment {wg['root_bending_moment_kNm']:.0f} kNm per side, box mass {wg['box_mass_both_sides_kg']:.0f} kg "
                   f"(strength-sized {wg['strength_sized_box_mass_kg']:.0f} kg, stiffness factor {wg['stiffness_scale']:.2f} for flutter and divergence), tip deflection {wg['tip_deflection_pct_semispan']:.1f} % "
                   f"of the semispan, first bending {wg['first_bending_hz']:.1f} Hz and torsion {wg['first_torsion_hz']:.1f} Hz, flutter {wg['flutter']['V_flutter_eas_ms']:.0f} m/s EAS against the required "
                   f"{wg['flutter']['required_ms']:.0f}.</p><table><tr><th>check</th><th>minimum margin of safety</th></tr>{ms}</table>"
                   f"<h3>Material trade (same loads)</h3><table><tr><th>material</th><th>box kg</th><th>strength-sized kg</th><th>stiffness factor</th><th>note</th></tr>{tr}</table>"
                   f"{_img(p.stage_dir('structure') / 'structure.png')}")
    t = st["thermal"]
    if t:
        lv = "".join(f"<tr><td>{r['adam_epochs']}</td><td>{r['max_abs']:.3e}</td><td>{r['rel_l2']:.3e}</td><td>{r['wall_s']:.0f}</td></tr>" for r in t["convergence"]["levels"])
        fit = t["convergence"].get("fit", {})
        zs = "".join(f"<tr><td>{html.escape(z)}</td><td>{html.escape(v['optimum']['material'])}</td><td>{v['optimum']['thickness_mm']:.2f}</td><td>{v['optimum']['areal_kg_m2']:.2f}</td>"
                     f"<td>{v['optimum']['T_max_K']:.0f}</td><td>{v['optimum']['stress_ratio']:.3f}</td><td>{html.escape(json.dumps(v['optimum']['source']))}</td></tr>"
                     for z, v in t["design"]["zones"].items() if v["optimum"])
        sec.append(f"<h2>6 Skin heating with the parametric PINN</h2><h3>Convergence of the PINN</h3><table><tr><th>Adam epochs</th><th>max error</th><th>relative L2</th><th>s</th></tr>{lv}</table>"
                   f"<p>Fit ({html.escape(str(fit.get('model', '')))}): {html.escape(json.dumps({k: v for k, v in fit.items() if isinstance(v, (int, float, str))}))}</p>"
                   f"<p>Network used for the design: {html.escape(Path(t['network_used']).parent.name)} (max error {t['network_max_error']:.2e}).</p>"
                   f"<h3>PINN-optimised skin</h3><table><tr><th>zone</th><th>material</th><th>mm</th><th>kg/m&sup2;</th><th>T max, K</th><th>stress ratio</th><th>answers (PINN / analytic)</th></tr>{zs}</table>"
                   f"{_img(p.stage_dir('thermal') / 'thermal.png')}")
    m = st["simulation"]
    if m:
        gr = "".join(f"<tr><td>{g['h']}</td><td>{g['cells']}</td><td>{g['CL']:.4f}</td><td>{g['CD']:.5f}</td><td>{g['seconds']:.0f}</td></tr>" for g in m["grid_study"])
        rc = m["richardson"]
        stt = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['mach']}</td><td>{v['alpha']}</td><td>{v['CL']:.4f}</td><td>{v['CD']:.5f}</td><td>{_fmt(v.get('separated_fraction'))}</td></tr>" for k, v in m["states"].items())
        sec.append(f"<h2>7 3D flow simulation</h2><h3>Grid convergence at cruise</h3><table><tr><th>h, m</th><th>cells</th><th>CL</th><th>CD</th><th>s</th></tr>{gr}</table>"
                   f"<p>CL {rc['CL']['extrapolated']:.4f} &plusmn; {100 * rc['CL']['gci_fine']:.1f} % ({html.escape(rc['CL'].get('kind', ''))}); "
                   f"CD {rc['CD']['extrapolated']:.5f} &plusmn; {100 * rc['CD']['gci_fine']:.1f} % ({html.escape(rc['CD'].get('kind', ''))}).</p>"
                   f"<table><tr><th>state</th><th>Mach</th><th>&alpha;</th><th>CL</th><th>CD</th><th>separated fraction</th></tr>{stt}</table>"
                   f"{_img(p.stage_dir('simulation') / 'grid_convergence.png')}{_img(p.stage_dir('simulation') / 'field.png')}")
    fl = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['bytes']:,}</td><td><code>{v['sha256'][:16]}</code></td></tr>" for k, v in exports.get("files", {}).items())
    sec.append(f"<h2>8 Export</h2><p>Drawing scale {html.escape(str(exports.get('drawing_scale', '')))}.</p><table><tr><th>file</th><th>bytes</th><th>SHA-256 (start)</th></tr>{fl}</table>"
               f"{_img(p.stage_dir('export') / (p.name + '_drawing.png'))}")
    revs = "".join(f"<tr><td>{r['rev']}</td><td>{html.escape(r['stage'])}</td><td>{html.escape(r['note'])}</td><td>{html.escape(r['time'])}</td><td><code>{r['sha256'][:16]}</code></td></tr>"
                   for r in p.meta["revisions"])
    sec.append(f"<h2>Revision history</h2><table><tr><th>rev</th><th>stage</th><th>note</th><th>time</th><th>SHA-256 (start)</th></tr>{revs}</table>")
    css = """body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f7fb;color:#162640}header{background:linear-gradient(90deg,#16325c,#0b4f86);color:#fff;padding:22px 36px}
header h1{margin:0;font-size:24px}header p{margin:4px 0 0;color:#bcd3ee}main{max-width:1100px;margin:0 auto;padding:12px 36px 40px}h2{color:#16325c;border-bottom:2px solid #0096d6;padding-bottom:4px;margin-top:34px}
table{border-collapse:collapse;margin:10px 0;font-size:13px;background:#fff}td,th{border:1px solid #d5deea;padding:4px 9px;text-align:left}th{background:#e8edf4}
img{max-width:100%;border:1px solid #d5deea;background:#fff;margin:8px 0}pre{background:#fff;border:1px solid #d5deea;padding:10px;font-size:12px;overflow:auto}
.ok{color:#137333;font-weight:600}.bad{color:#b3261e;font-weight:600}.s-done{color:#137333}.s-failed{color:#b3261e}.s-stale{color:#a15c00}footer{color:#6b7a90;font-size:12px;margin-top:30px}"""
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(p.name)} design report</title><style>{css}</style></head><body>"
            f"<header><h1>{html.escape(p.name)}: design workflow report</h1><p>AeroSynapse Studio &middot; generated {now()} &middot; {html.escape(p.meta.get('origin', ''))}</p></header>"
            f"<main>{''.join(sec)}<footer>Research use only: preliminary-design methods with their stated assumptions; not certification analysis.</footer></main></body></html>")


# ------------------------------------------------------------------ any design, without a project
def export_design(design: dict[str, Any], out: str | Path, h: float = 0.05, name: str = "design") -> dict[str, Any]:
    """STL, OBJ, glTF, STEP, DXF three-view and the A3 drawing of one design file (what Studio's Export > CAD package runs)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kernel.mesh(design, out / f"{name}.stl", h=h)
    kernel.mesh(design, out / f"{name}.obj", h=h)
    V, F = read_obj(out / f"{name}.obj")
    write_glb(out / f"{name}.glb", V, F, name)
    coarse = out / "_coarse.obj"
    kernel.mesh(design, coarse, h=max(2.5 * h, 0.08))
    Vc, Fc = read_obj(coarse)
    write_step(out / f"{name}.step", Vc, Fc, name)
    coarse.unlink(missing_ok=True)
    title = design.get("name", name)
    sheet = drawing_sheet(out / name, V, F, {"title": f"{str(title).upper()} GENERAL ARRANGEMENT", "project": str(title), "number": f"AS-{str(title).upper()[:8]}-GA-001",
                                             "revision": str(design.get("revision", 1)), "date": time.strftime("%Y-%m-%d"), "material": "", "mass": ""}, design=design)
    lo, hi = V.min(0), V.max(0)
    y_side = (lo[1] - hi[2]) * 1000 - 3000.0
    offs = {"TOP": np.array([0.0, 0.0]), "SIDE": np.array([0.0, y_side]), "FRONT": np.array([(hi[0] - lo[1]) * 1000 + 3000.0, y_side])}
    write_dxf(out / f"{name}_3view.dxf", {k: (sheet["outlines"][k], offs[k]) for k in VIEWS}, [("TITLE", 0.0, y_side + lo[2] * 1000 - 1500, 250.0, f"{title} general arrangement, 1:1 mm")])
    return {"files": sorted(f.name for f in out.iterdir() if f.is_file()), "triangles": int(len(F)), "step_faces": int(len(Fc)), "drawing_scale": sheet["scale"]}


# ------------------------------------------------------------------ the stage
def run(p: Project, design: dict[str, Any], prof: dict[str, Any]) -> dict[str, Any]:
    out = p.stage_dir("export")
    name = p.name
    p.log("export: surface mesh (STL, OBJ)")
    kernel.mesh(design, out / f"{name}.stl", h=prof["mesh_h"])
    kernel.mesh(design, out / f"{name}.obj", h=prof["mesh_h"])
    V, F = read_obj(out / f"{name}.obj")
    p.log(f"export: glTF ({len(F)} triangles)")
    write_glb(out / f"{name}.glb", V, F, name)
    coarse = out / "_coarse.obj"
    kernel.mesh(design, coarse, h=max(2.5 * prof["mesh_h"], 0.08))
    Vc, Fc = read_obj(coarse)
    p.log(f"export: STEP AP214 faceted B-rep ({len(Fc)} faces)")
    write_step(out / f"{name}.step", Vc, Fc, name)
    p.log("export: drawings")
    inf = kernel.info(design, h=prof["mesh_h"])
    perf = p.result("structure")
    mass = f"{perf['W0_kg']:.0f} kg take-off" if perf else ""
    mat = (perf.get("wing", {}).get("material", "") + " wing box") if perf else ""
    rev = str(len(p.meta["revisions"]))
    sheet = drawing_sheet(out / name, V, F, {"title": f"{name.upper()} GENERAL ARRANGEMENT", "project": name, "number": f"AS-{name.upper()[:8]}-GA-001", "revision": rev,
                                             "date": time.strftime("%Y-%m-%d"), "material": mat, "mass": mass}, design=design)
    lo, hi = V.min(0), V.max(0)
    gap = 3000.0  # mm between views in model space (the DXF is 1:1, millimetres)
    y_side = (lo[1] - hi[2]) * 1000 - gap
    offs = {"TOP": np.array([0.0, 0.0]), "SIDE": np.array([0.0, y_side]), "FRONT": np.array([(hi[0] - lo[1]) * 1000 + gap, y_side])}
    write_dxf(out / f"{name}_3view.dxf", {k: (sheet["outlines"][k], offs[k]) for k in VIEWS},
              [("TITLE", 0.0, y_side + (lo[2] * 1000) - 1500, 250.0, f"{name} general arrangement, 1:1 millimetres, revision {rev}")]
              + [(k, offs[k][0] + lo[VIEWS[k][0]] * 1000, offs[k][1] + hi[VIEWS[k][1]] * 1000 + 400, 200.0, f"{k} VIEW") for k in VIEWS])
    p.log("export: bill of materials and mass properties")
    rows = bom(p, design, max(prof["mesh_h"], 0.06))
    with open(out / "bom.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    mp = {k: inf[k] for k in ("volume", "wetted_area", "frontal_area", "planform_area", "side_area", "length", "centroid", "bbox_lo", "bbox_hi", "inertia_unit_density")}
    mp["skin_mass_kg"] = float(sum(r["skin_mass_kg"] for r in rows))
    if perf:
        mp["W0_kg"] = perf["W0_kg"]
        mp["wing_box_kg"] = perf["wing"]["box_mass_both_sides_kg"]
    (out / "mass_properties.json").write_text(json.dumps(mp, indent=1), encoding="utf-8")
    (out / "design.json").write_text(json.dumps(design, indent=1), encoding="utf-8")
    coarse.unlink(missing_ok=True)
    files = {}
    for f in sorted(out.iterdir()):
        if f.is_file() and f.suffix != ".zip" and f.name != "result.json":
            files[f.name] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
    res = {"files": files, "drawing_scale": sheet["scale"], "triangles": int(len(F)), "step_faces": int(len(Fc)), "mass_properties": mp, "bom_rows": len(rows)}
    p.log("export: report")
    (out / "report.html").write_text(report(p, res), encoding="utf-8")
    files["report.html"] = {"bytes": (out / "report.html").stat().st_size, "sha256": sha256(out / "report.html")}
    from aerosynapse.workflow import report_formats as RF

    p.log("export: report as PDF, Word and text")
    for f in RF.write_all(p).values():
        files[f.name] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
    # package: every project file with its checksum
    pkg = p.dir / f"{name}_package.zip"
    manifest = {"project": name, "created": now(), "revisions": p.meta["revisions"], "files": {}}
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(p.dir.rglob("*")):
            if f.is_file() and f != pkg and not f.name.endswith(".bin") and not f.name.endswith(".zip"):
                rel = f.relative_to(p.dir).as_posix()
                manifest["files"][rel] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
                z.write(f, rel)
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
    res["package"] = {"file": pkg.name, "bytes": pkg.stat().st_size, "files": len(manifest["files"]), "sha256": sha256(pkg)}
    p.log(f"export: package {pkg.name} ({pkg.stat().st_size / 1e6:.1f} MB, {len(manifest['files'])} files)")
    return res

