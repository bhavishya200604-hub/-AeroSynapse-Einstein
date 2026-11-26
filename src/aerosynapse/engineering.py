r"""Engineering workbenches of AeroSynapse Studio, each writing ``summary.txt`` (and figures) to a folder for the viewer.

* **Clash analysis** (assembly / DMU): the interference volume of every pair of solid features, and of every feature with
  the *space reservations* (features marked ``"reserve": true``: keep-out volumes for equipment, crew, gear, engines).
* **Design rules** (knowledgeware): rules written as expressions on the design's measured properties and its feature
  parameters (``length <= 20``, ``wing.tc >= 0.03``, ``static_margin > -0.05``), each reported pass or fail with its value.
* **Composite laminate** (classical lamination theory): ply stiffness, the A, B and D matrices of a layup such as
  ``[0/45/-45/90]s`` and its effective moduli, with the in-plane stiffness against direction.
* **Skeleton parameters** (top-down design): a design built from design-space variables is rebuilt from an edited table.
* **RFLP** (systems engineering): requirements traced to functions, logical elements (methods, stages) and physical
  features, with the verification status each workflow stage reported.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.cad import kernel
from aerosynapse.sim.common import INK, NEON, finish, plt, save


# ------------------------------------------------------------------ clash analysis
def clash(design: dict[str, Any], out: str | Path, h: float = 0.05) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    feats = [f for f in design["features"] if f.get("op", "add") == "add" and (f.get("enabled", True) or f.get("reserve"))]
    rows = []
    for a, b in itertools.combinations(feats, 2):
        fa, fb = {**a, "op": "add", "enabled": True}, {**b, "op": "intersect", "enabled": True, "blend": 0, "chamfer": 0}
        try:
            v = kernel.info({"features": [fa, fb]}, h=h, nx=30)["volume"] if _boxes_overlap(fa, fb) else 0.0
        except Exception:
            v = 0.0
        if v > 1e-5:
            kind = "reservation violated" if (a.get("reserve") or b.get("reserve")) else "interference (intended where parts join)"
            rows.append({"a": a.get("id"), "b": b.get("id"), "volume_m3": v, "kind": kind})
    rows.sort(key=lambda r: -r["volume_m3"])
    lines = [f"{len(feats)} solid features and reservations, {len(feats) * (len(feats) - 1) // 2} pairs checked", ""]
    viol = [r for r in rows if r["kind"].startswith("reservation")]
    lines += [f"RESERVATION VIOLATED  {r['a']} / {r['b']}: {1e3 * r['volume_m3']:.1f} litres" for r in viol]
    lines += [f"interference          {r['a']} / {r['b']}: {1e3 * r['volume_m3']:.1f} litres" for r in rows if r not in viol]
    if not rows:
        lines.append("no interference")
    return finish(out, "Clash analysis", lines, {"pairs": rows, "violations": len(viol)})


def _boxes_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    try:
        ia = kernel.info({"features": [a]}, h=0.2, nx=10)
        ib = kernel.info({"features": [b]}, h=0.2, nx=10)
        return all(ia["bbox_lo"][k] <= ib["bbox_hi"][k] and ib["bbox_lo"][k] <= ia["bbox_hi"][k] for k in range(3))
    except Exception:
        return True


# ------------------------------------------------------------------ design rules
DEFAULT_RULES = [
    {"name": "fits a hardened shelter (length)", "expr": "length <= 20"},
    {"name": "fits a hardened shelter (span)", "expr": "span <= 14"},
    {"name": "internal volume for fuel and systems", "expr": "volume >= 12"},
    {"name": "wetted area not excessive", "expr": "wetted_area / ref_area <= 6.5"},
    {"name": "longitudinally stable (about the centroid)", "expr": "static_margin >= -0.05"},
    {"name": "not over-stable", "expr": "static_margin <= 0.25"},
    {"name": "supersonic wave drag (Mach-plane area rule)", "expr": "wave_CD_M16 <= 0.035"},
    {"name": "wing thick enough to build and hold fuel", "expr": "wing.tc >= 0.03"},
    {"name": "wing sweep in the usual range", "expr": "wing.sweep <= 65"},
]
_OK = (ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Name, ast.Load, ast.Constant, ast.Attribute, ast.Call, ast.And, ast.Or, ast.Not,
       ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Mod)


class _NS(dict):
    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:
            raise NameError(k) from e


def safe_eval(expr: str, env: dict[str, Any]) -> Any:
    tree = ast.parse(expr, mode="eval")
    for n in ast.walk(tree):
        if not isinstance(n, _OK):
            raise ValueError(f"not allowed in a rule: {type(n).__name__}")
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in ("min", "max", "abs", "sqrt")):
            raise ValueError("only min, max, abs and sqrt may be called")
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            raise ValueError(f"not allowed in a rule: private attribute {n.attr}")
    return eval(compile(tree, "<rule>", "eval"), {"__builtins__": {}, "min": min, "max": max, "abs": abs, "sqrt": math.sqrt}, env)  # noqa: S307 (AST checked above)


def rules(design: dict[str, Any], out: str | Path) -> dict[str, Any]:
    from aerosynapse.workflow.stages import scorecard

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sc = scorecard(design, h=0.08)
    env: dict[str, Any] = {"volume": sc["volume_m3"], "wetted_area": sc["wetted_area_m2"], "frontal_area": sc["frontal_area_m2"], "length": sc["length_m"],
                           "span": sc["span_m"], "ref_area": sc["ref_area_m2"], "static_margin": sc["static_margin_centroid"] or 0.0, "wave_CD_M16": sc["wave_CD_M1.6"] or 0.0,
                           "wave_CD_M12": sc["wave_CD_M1.2"] or 0.0, "friction_CD0": sc["friction_CD0"], "CL_alpha": sc["CL_alpha_per_rad"] or 0.0}
    for f in design["features"]:
        if f.get("id"):
            env[re.sub(r"\W", "_", str(f["id"]))] = _NS({k: v for k, v in f.items() if isinstance(v, (int, float))})
    rs = design.get("rules") or DEFAULT_RULES
    res, lines = [], []
    for r in rs:
        try:
            ok = bool(safe_eval(r["expr"], env))
            lhs = r["expr"].split("<")[0].split(">")[0].split("=")[0].strip()
            val = safe_eval(lhs, env) if lhs else None
            res.append({**r, "pass": ok, "value": val})
            lines.append(f"{'PASS' if ok else 'FAIL'}  {r['name']:44s} {r['expr']:34s} value {val if not isinstance(val, float) else round(val, 4)}")
        except Exception as e:
            res.append({**r, "pass": None, "error": str(e)})
            lines.append(f"ERROR {r['name']:44s} {r['expr']:34s} {e}")
    n_ok = sum(1 for r in res if r["pass"])
    lines.insert(0, f"{n_ok} of {len(res)} rules pass" + ("" if design.get("rules") else " (the default rule set; put your own under \"rules\" in the design file)"))
    lines.insert(1, "")
    return finish(out, "Design rules (knowledgeware)", lines, {"rules": res, "measured": env | {k: None for k in []}})


# ------------------------------------------------------------------ composites
PLY = {  # unidirectional carbon / epoxy ply (IM7/8552 typical): E1, E2, G12 (Pa), nu12, thickness (m), density
    "cfrp_im7": (165e9, 8.8e9, 5.1e9, 0.30, 0.125e-3, 1570.0),
    "cfrp_bmi": (155e9, 8.5e9, 4.9e9, 0.31, 0.125e-3, 1600.0),
    "gfrp": (45e9, 12e9, 4.5e9, 0.28, 0.2e-3, 1900.0),
}


def parse_layup(s: str) -> list[float]:
    t = s.strip().replace("[", "").replace("]", "")
    sym = t.endswith("s")
    t = t.rstrip("s")
    ang = [float(a.replace("+", "")) for a in t.split("/") if a.strip()]
    return ang + ang[::-1] if sym else ang


def laminate(layup: str = "[0/45/-45/90]s", material: str = "cfrp_im7") -> dict[str, Any]:
    E1, E2, G12, n12, tp, rho = PLY[material]
    n21 = n12 * E2 / E1
    Q = np.array([[E1 / (1 - n12 * n21), n12 * E2 / (1 - n12 * n21), 0], [n12 * E2 / (1 - n12 * n21), E2 / (1 - n12 * n21), 0], [0, 0, G12]])
    ang = parse_layup(layup)
    h = tp * len(ang)
    z = np.linspace(-h / 2, h / 2, len(ang) + 1)
    A, B, D = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    for k, a in enumerate(ang):
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        T = np.array([[c * c, s * s, 2 * c * s], [s * s, c * c, -2 * c * s], [-c * s, c * s, c * c - s * s]])
        R = np.diag([1, 1, 2])
        Qb = np.linalg.inv(T) @ Q @ R @ T @ np.linalg.inv(R)
        A += Qb * (z[k + 1] - z[k])
        B += Qb * (z[k + 1] ** 2 - z[k] ** 2) / 2
        D += Qb * (z[k + 1] ** 3 - z[k] ** 3) / 3
    a = np.linalg.inv(A)
    return {"layup": layup, "plies": ang, "thickness_mm": h * 1e3, "areal_kg_m2": rho * h, "A": A, "B": B, "D": D,
            "Ex_GPa": 1 / (a[0, 0] * h) / 1e9, "Ey_GPa": 1 / (a[1, 1] * h) / 1e9, "Gxy_GPa": 1 / (a[2, 2] * h) / 1e9, "nuxy": -a[0, 1] / a[0, 0],
            "symmetric": bool(np.abs(B).max() < 1e-6 * np.abs(A).max())}


def composites(layup: str, material: str, out: str | Path) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    r = laminate(layup, material)
    th = np.radians(np.arange(0, 360.1, 2))
    a = np.linalg.inv(r["A"])
    h = r["thickness_mm"] * 1e-3
    Eth = []
    for t in th:
        c, s = math.cos(t), math.sin(t)
        Eth.append(1 / (h * (a[0, 0] * c**4 + (2 * a[0, 1] + a[2, 2]) * c * c * s * s + a[1, 1] * s**4)) / 1e9)
    pl = plt()
    fig = pl.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    ax.plot(th, Eth, color=NEON[0])
    ax.fill(th, Eth, color=NEON[0], alpha=0.2)
    ax.set_title(f"In-plane stiffness E(direction), GPa: {layup}", fontsize=9)
    ax2 = fig.add_subplot(1, 2, 2)
    for k, ang in enumerate(r["plies"]):
        ax2.barh(k, 1, color=NEON[k % len(NEON)] if ang == 0 else NEON[(int(abs(ang)) // 45 + 1) % len(NEON)], edgecolor=INK)
        ax2.text(0.5, k, f"{ang:+.0f} deg", ha="center", va="center", fontsize=8, color=INK)
    ax2.set(yticks=[], xticks=[], title=f"Ply stack ({len(r['plies'])} plies, {r['thickness_mm']:.2f} mm)")
    figs = [save(fig, out, "1_laminate")]
    lines = [f"layup {layup} in {material}: {len(r['plies'])} plies, {r['thickness_mm']:.3f} mm, {r['areal_kg_m2']:.2f} kg/m2, symmetric: {r['symmetric']}",
             f"effective moduli: Ex {r['Ex_GPa']:.1f} GPa, Ey {r['Ey_GPa']:.1f} GPa, Gxy {r['Gxy_GPa']:.1f} GPa, nu_xy {r['nuxy']:.3f}",
             "A (N/m), B (N), D (N m) matrices:"]
    for k in ("A", "B", "D"):
        lines += [f"  {k}: " + "  ".join(f"{v:11.4g}" for v in row) for row in r[k]]
    lines.append("Draping over doubly curved skins shears the plies; check the drape of curved panels in a dedicated tool before layup.")
    return finish(out, "Composite laminate (classical lamination theory)", lines, {k: v for k, v in r.items()} | {"figures": figs})


# ------------------------------------------------------------------ skeleton parameters
def skeleton(design: dict[str, Any], variables: dict[str, float] | None, family: str = "freeform") -> dict[str, Any]:
    from aerosynapse.fighter import design as D

    v = dict(design.get("variables") or {})
    v.update(variables or {})
    if not v:
        raise ValueError("this design has no design-space variables (it was not built from a skeleton)")
    new = D.build(D.complete(family, {k: float(x) for k, x in v.items()}), name=design.get("name", "design"))
    keep = [f for f in design["features"] if f.get("id") not in {g["id"] for g in new["features"]}]
    new["features"] += [f for f in keep if f.get("user", True) and f.get("id", "").startswith(("sketch", "user", "pod", "fairing"))]
    return new


# ------------------------------------------------------------------ RFLP
def rflp(project_dir: str | Path, out: str | Path) -> dict[str, Any]:
    from aerosynapse.workflow import Project

    p = Project(project_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    r = {k: p.result(k) for k in ("concept", "design", "wing", "body", "structure", "thermal", "simulation", "export")}
    req = p.meta["requirements"]
    st = r["structure"].get("wing", {})
    perf = r["design"].get("performance_after") or r["concept"].get("performance") or {}
    rows = [
        ("R1 carry the mission (radius, payload)", "F1 generate lift, F2 store fuel", "L: design space + performance model (stage 2)", "wing, body", perf.get("feasible")),
        ("R2 low drag in cruise and dash", "F3 minimise drag", "L: wing shape optimiser (3), area rule (4), 3D flow (7)", "wing section, fuselage stations",
         bool(r["wing"]) and bool(r["body"]) and r["wing"].get("J_after", 9) <= r["wing"].get("J_before", 0) + 1e-9),
        (f"R3 strength to {req['code']}", "F4 carry flight loads", "L: wing box sizing, FEM (5)", "wing box covers and spars",
         None if not st else bool(min(v for k, v in st.get("margins_min", {}).items() if k.startswith(("cover", "web"))) >= -1e-6)),
        ("R4 free of flutter and divergence to 1.15 V_D", "F5 stay aeroelastically stable", "L: typical-section flutter, divergence (5)", "wing box stiffness",
         None if not st else bool(st["margins_min"]["flutter"] >= -1e-6 and st["margins_min"]["divergence"] >= -1e-6)),
        ("R5 survive skin heating at Mach 1.8", "F6 resist heat", "L: parametric PINN skin design (6)", "skin materials and gauges",
         None if not r["thermal"] else all(z["optimum"] is not None for z in r["thermal"]["design"]["zones"].values())),
        ("R6 documented and exchangeable geometry", "F7 hand over the design", "L: export (8)", "STEP, glTF, drawings, BOM", bool(r["export"])),
    ]
    lines = ["Requirement -> Function -> Logical -> Physical -> verified", ""]
    lines += [f"{a}\n    -> {b}\n    -> {c}\n    -> {d}\n    => {'VERIFIED' if v else 'open' if v is None else 'NOT MET'}" for a, b, c, d, v in rows]
    pl = plt()
    fig, ax = pl.subplots(figsize=(12, 4.6))
    ax.axis("off")
    cols = ["Requirements", "Functions", "Logical", "Physical"]
    for j, c in enumerate(cols):
        ax.text(0.12 + j * 0.25, 0.97, c, ha="center", fontsize=11, weight="bold", color=NEON[j])
    for i, row in enumerate(rows):
        y = 0.86 - i * 0.145
        for j in range(4):
            ax.text(0.12 + j * 0.25, y, row[j][:44], ha="center", va="center", fontsize=7.5, bbox={"boxstyle": "round,pad=0.4", "fc": "#13284a", "ec": NEON[j], "lw": 1})
            if j < 3:
                ax.annotate("", (0.12 + (j + 1) * 0.25 - 0.115, y), (0.12 + j * 0.25 + 0.115, y), arrowprops={"arrowstyle": "->", "color": "#8aa0bd"})
        v = row[4]
        ax.text(0.995, y, "verified" if v else "open" if v is None else "not met", ha="right", va="center", fontsize=8, weight="bold",
                color="#4ade80" if v else "#fbbf24" if v is None else "#f87171")
    figs = [save(fig, out, "1_rflp")]
    return finish(out, "Systems engineering: RFLP traceability", lines, {"rows": [list(r_) for r_ in rows], "figures": figs})


def main_cli(kind: str, path: str, out: str, extra: dict[str, Any]) -> str:
    if kind == "rflp":
        rflp(path, out)
    else:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if kind == "clash":
            clash(d, out, h=float(extra.get("spacing") or 0.05))
        elif kind == "rules":
            rules(d, out)
        elif kind == "composites":
            composites(extra.get("layup") or d.get("layup") or "[0/45/-45/90]s", extra.get("material") or "cfrp_im7", out)
        elif kind == "skeleton":
            new = skeleton(d, extra.get("variables"), extra.get("family") or "freeform")
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "design.json").write_text(json.dumps(new, indent=1), encoding="utf-8")
            (Path(out) / "summary.txt").write_text(f"Rebuilt from {len(new['variables'])} skeleton parameters: {len(new['features'])} features\n", encoding="utf-8")
    return (Path(out) / "summary.txt").read_text(encoding="utf-8")
