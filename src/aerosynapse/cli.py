"""Command-line interface: ``aerosynapse run | train | optimize | shapeopt | topology | xfoil | aircraft | fighter | studio | cad-analyse | cfd | cfd-verify | thermal | thermal-lab | surrogate | surrogate-predict | workflow | structure | simulate | export-design | engineer | designer | validate | report | reproduce | figures | doctor``.

Every command prints the research-use banner and the validation tier first (spec Section 16).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from aerosynapse.device import hardware_report
from aerosynapse.safety import banner

ROOT = Path(__file__).resolve().parents[2]


def _doctor(_: argparse.Namespace) -> int:
    hw = hardware_report()
    print(json.dumps(hw, indent=2))
    checks = {"numpy": hw.get("numpy"), "scipy": hw.get("scipy"), "torch (optional, [torch])": hw.get("torch")}
    for name, ver in checks.items():
        print(f"[{'ok' if ver else '--'}] {name}: {ver or 'not installed'}")
    try:
        import streamlit  # noqa: F401

        print("[ok] streamlit (optional, [ui])")
    except ImportError:
        print("[--] streamlit (optional, [ui]): not installed")
    return 0


def _script(name: str, extra: list[str]) -> int:
    path = ROOT / "scripts" / name
    if not path.is_file():
        print(f"script {path} not found (run from a source checkout)", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, str(path), *extra])


def _designer(_: argparse.Namespace) -> int:
    app = Path(__file__).resolve().parent / "ui" / "designer.py"
    # headless: Streamlit's first-run e-mail question would otherwise wait on a console that Studio does not have
    import threading
    import webbrowser

    port = 8501
    threading.Timer(4.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.address", "127.0.0.1", "--server.port", str(port),
                            "--server.headless", "true", "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal"])


def _studio(ns: argparse.Namespace) -> int:
    from aerosynapse.cad import kernel

    for name in ("aerokernel", "aeroflow", "AeroSynapseStudio"):
        kernel.build(name)
    args = [str(kernel.exe("AeroSynapseStudio"))] + ([str(Path(ns.design).resolve())] if ns.design else [])
    subprocess.Popen(args)
    return 0


def _cad_analyse(ns: argparse.Namespace) -> int:
    from aerosynapse.cad import analyse

    print(analyse.main(ns.design, ns.out))
    return 0


def _cfd(ns: argparse.Namespace) -> int:
    from aerosynapse.cfd import runner

    design = json.loads(Path(ns.design).read_text(encoding="utf-8"))
    from aerosynapse.cad import kernel

    info = kernel.info(design, nx=8)
    L, span = info["length"], info["bbox_hi"][1] - info["bbox_lo"][1]
    ref = {"length": L, "area": ns.area or 0.6 * info["planform_area"], "chord": ns.chord or 0.25 * L, "span": span, "moment_point": [info["bbox_lo"][0] + 0.55 * L, 0.0, 0.0]}
    fl = runner.Flight(mach=ns.mach, alpha=ns.alpha, beta=ns.beta, altitude=ns.altitude, delta_T=ns.dT, gust_amplitude=ns.gust, turbulence_intensity=ns.tu)
    num = runner.Numerics(cells_per_length=ns.cells, steps=ns.steps, save_volume=ns.save_volume)
    res = runner.run(design, fl, ref, ns.out, num)
    print(json.dumps({k: res[k] for k in ("coefficients", "converged", "steps", "seconds", "cells", "separated_fraction", "T_wall_max_K") if k in res}, indent=1))
    return 0


def _surrogate_predict(ns: argparse.Namespace) -> int:
    from aerosynapse.store.predict import report

    print(report(ns.design))
    return 0


def _workflow(ns: argparse.Namespace) -> int:
    from aerosynapse.workflow import STAGES, Project
    from aerosynapse.workflow import stages as S

    if ns.action == "new":
        p = Project.create(ns.name, ns.source, ns.family, ns.code, ns.W0, where=Path(ns.where) if ns.where else None, overwrite=ns.overwrite)
        print(f"created {p.dir} ({p.meta['origin']})")
        return 0
    p = Project(ns.name)
    if ns.action == "status":
        for i, (k, t) in enumerate(STAGES):
            st = p.meta["stages"][k]
            print(f"{i + 1}. {t:30s} {st.get('status', 'pending'):8s} {st.get('seconds', 0) or 0:8.0f} s  {st.get('error') or ''}")
        print(f"revisions: {len(p.meta['revisions'])}; folder: {p.dir}")
        return 0
    if ns.action == "report":
        from aerosynapse.workflow import report_formats as RF

        for f in RF.write_all(p, ("pdf", "docx", "txt") if ns.format == "all" else (ns.format,)).values():
            print(f)
        return 0
    if ns.stage:
        S.run_stage(p, ns.stage, ns.profile)
    else:
        S.run_all(p, ns.profile, ns.start, ns.stop)
    return 0


def _structure(ns: argparse.Namespace) -> int:
    import json as _json
    from pathlib import Path as _Path

    from aerosynapse.structures import wing as SW

    d = _json.loads(_Path(ns.design).read_text(encoding="utf-8"))
    r = SW.analyse(d, ns.W0, ns.code, ns.material, ns.T, v_d_eas=ns.vd, fuel_kg=ns.fuel)
    w = r["wing"]
    print(f"{r['code']}: limit n {w['n_limit']}, ultimate {w['n_ultimate']}; wing box in {w['material']} at {w['temperature_K']:.0f} K")
    print(f"root bending moment {w['root_bending_moment_kNm']:.0f} kNm, shear {w['root_shear_kN']:.0f} kN per side (limit)")
    print(f"box mass {w['box_mass_both_sides_kg']:.0f} kg (strength-sized {w['strength_sized_box_mass_kg']:.0f} kg, stiffness factor {w['stiffness_scale']:.2f})")
    print(f"tip deflection {w['tip_deflection_pct_semispan']:.1f} % of the semispan, twist {w['tip_twist_deg']:.2f} deg; modes {', '.join(f'{f:.1f}' for f in w['modes_hz'][:4])} Hz")
    print(f"flutter {w['flutter']['V_flutter_eas_ms']:.0f} m/s EAS (required {w['flutter']['required_ms']:.0f}); divergence {w['divergence']['V_div_eas_ms']:.0f} m/s EAS")
    print("minimum margins of safety: " + ", ".join(f"{k} {v:+.3f}" for k, v in w["margins_min"].items()))
    f = r["fuselage"]
    if "max_moment_kNm" in f:
        print(f"fuselage: {f['max_moment_kNm']:.0f} kNm at x {f['at_x_m']:.1f} m (ultimate), skin {f['skin_required_mm']:.2f} mm")
    if ns.out:
        _Path(ns.out).write_text(_json.dumps(r, indent=1, default=float), encoding="utf-8")
    return 0


def _simulate(ns: argparse.Namespace) -> int:
    import json as _json
    from pathlib import Path as _Path

    if ns.kind == "schema":  # the setup form of every simulation, for Studio
        from aerosynapse.sim import setup as SU

        print(SU.write_schema(ns.design if ns.design != "-" else None))
        return 0
    if ns.kind == "flow3d":  # post-process a finished 3D flow folder
        from aerosynapse.sim import flow3d as F

        F.postprocess(ns.design)
        print((_Path(ns.design) / "summary.txt").read_text(encoding="utf-8"))
        return 0
    d = _json.loads(_Path(ns.design).read_text(encoding="utf-8"))
    from aerosynapse.sim import setup as SU

    setup = SU.load(ns.setup)
    if ns.kind == "cfd":
        from aerosynapse.sim import flow3d as F

        F.run(d, ns.out, ns.case, ns.quality, setup=setup)
        print((_Path(ns.out) / "summary.txt").read_text(encoding="utf-8"))
        return 0
    if ns.kind == "optimise":  # CFD-driven Bayesian optimisation of the main wing
        from aerosynapse.sim import optimise as O

        O.run(d, setup, ns.out)
        print((_Path(ns.out) / "summary.txt").read_text(encoding="utf-8"))
        return 0
    if ns.kind == "turbulence":
        from aerosynapse.sim import turbulence as T

        T.run(d, ns.out, setup)
        print((_Path(ns.out) / "summary.txt").read_text(encoding="utf-8"))
        return 0
    kinds = ["analytic", "pinn", "structure", "turbulence"] if ns.kind == "all" else [ns.kind]
    for k in kinds:
        out = _Path(ns.out) / k if ns.kind == "all" else _Path(ns.out)
        if k == "analytic":
            from aerosynapse.sim import analytic as M
        elif k == "pinn":
            from aerosynapse.sim import pinn as M
        elif k == "turbulence":
            from aerosynapse.sim import turbulence as M
        else:
            from aerosynapse.sim import structure as M
        if setup is not None and ns.kind != "all":
            M.run(d, out, setup=setup)
        elif k == "structure":
            M.run(d, out, W0_kg=ns.W0 or 12000.0, code=ns.code, material=ns.material or "al7075")
        else:
            M.run(d, out)
        print((out / "summary.txt").read_text(encoding="utf-8"))
    return 0


def _export_design(ns: argparse.Namespace) -> int:
    import json as _json
    from pathlib import Path as _Path

    from aerosynapse.workflow.export import export_design

    r = export_design(_json.loads(_Path(ns.design).read_text(encoding="utf-8")), ns.out, ns.h, ns.name)
    print(f"{len(r['files'])} files in {ns.out}: {', '.join(r['files'])}")
    return 0


def _engineer(ns: argparse.Namespace) -> int:
    import json as _json

    from aerosynapse.engineering import main_cli

    extra = {"layup": ns.layup, "material": ns.material, "family": ns.family, "variables": _json.loads(ns.variables) if ns.variables else None}
    if ns.setup:  # a setup file from Studio's form (kinds composites and clash)
        from pathlib import Path as _Path

        extra.update({k: v for k, v in _json.loads(_Path(ns.setup).read_text(encoding="utf-8")).items() if v not in (None, "")})
    print(main_cli(ns.kind, ns.path, ns.out, extra))
    return 0


def _run(ns: argparse.Namespace) -> int:
    from aerosynapse.experiments.runner import run_config

    out = run_config(ns.config, profile=ns.profile, root=ns.results)
    print(json.dumps(out, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aerosynapse", description="AeroSynapse-Einstein research platform")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check environment and report hardware").set_defaults(fn=_doctor)
    for name, script, hlp in (
        ("train", "run_phase2.py", "train PINN / baseline studies"),
        ("optimize", "run_phase5.py", "run optimisation and inverse-problem studies"),
        ("validate", "run_phase7.py", "build trust reports and fallback checks"),
        ("report", "make_report.py", "generate HTML/Markdown reports from results/"),
        ("reproduce", "reproduce_all.py", "reproduce every reported result"),
        ("figures", "make_figures.py", "generate paper figures from results/"),
    ):
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("--profile", default="quick", choices=["quick", "standard", "full"])
        sp.set_defaults(fn=lambda ns, s=script: _script(s, ["--profile", ns.profile]))
    so = sub.add_parser("shapeopt", help="optimise a shape across laminar, turbulent and extreme flow states")
    so.add_argument("--shape", default="naca2412", help="nacaXXXX, circle, square, ellipse:r or a coordinate file")
    so.add_argument("--profile", default="quick", choices=["quick", "standard", "full"])
    so.set_defaults(fn=lambda ns: _script("run_shapeopt.py", ["--shape", ns.shape, "--profile", ns.profile]))
    ac = sub.add_parser("aircraft", help="optimise whole aircraft bodies for each mission")
    ac.add_argument("--profile", default="quick", choices=["quick", "standard", "full"])
    ac.set_defaults(fn=lambda ns: _script("run_aircraft.py", ["--profile", ns.profile]))
    fg = sub.add_parser("fighter", help="optimise fighter-class configurations against every flight situation")
    fg.add_argument("--profile", default="standard", choices=["quick", "standard", "full"])
    fg.set_defaults(fn=lambda ns: _script("run_fighter.py", ["--profile", ns.profile]))
    st = sub.add_parser("studio", help="open AeroSynapse Studio, the feature-based design editor (C++)")
    st.add_argument("design", nargs="?", default="")
    st.set_defaults(fn=_studio)
    ca = sub.add_parser("cad-analyse", help="quick aerodynamics of a design file (lattice, area rule, friction)")
    ca.add_argument("design")
    ca.add_argument("--out", default=None)
    ca.set_defaults(fn=_cad_analyse)
    cf = sub.add_parser("cfd", help="3D flow simulation of a design file")
    cf.add_argument("design")
    cf.add_argument("--mach", type=float, default=0.85)
    cf.add_argument("--alpha", type=float, default=2.0)
    cf.add_argument("--beta", type=float, default=0.0)
    cf.add_argument("--altitude", type=float, default=11000.0)
    cf.add_argument("--dT", type=float, default=0.0)
    cf.add_argument("--gust", type=float, default=0.0, help="vertical gust speed / flight speed")
    cf.add_argument("--tu", type=float, default=0.0, help="inflow turbulence intensity")
    cf.add_argument("--cells", type=float, default=40.0, help="cells per body length near the body")
    cf.add_argument("--steps", type=int, default=2500)
    cf.add_argument("--area", type=float, default=0.0)
    cf.add_argument("--chord", type=float, default=0.0)
    cf.add_argument("--save-volume", action="store_true")
    cf.add_argument("--out", default="results/cfd/run")
    cf.set_defaults(fn=_cfd)
    cv = sub.add_parser("cfd-verify", help="verify the 3D solver against exact solutions and benchmarks")
    cv.add_argument("--profile", default="standard", choices=["quick", "standard", "full"])
    cv.set_defaults(fn=lambda ns: _script("cfd_verify.py", ["--profile", ns.profile]))
    th = sub.add_parser("thermal", help="skin heating with the parametric PINN and the verified solvers")
    th.add_argument("--profile", default="standard", choices=["quick", "standard", "full"])
    th.set_defaults(fn=lambda ns: _script("run_thermal_skin.py", ["--profile", ns.profile]))
    sg = sub.add_parser("surrogate", help="train surrogates from the on-disk data store (streamed, bounded memory)")
    sg.add_argument("--profile", default="standard", choices=["quick", "standard", "full"])
    sg.set_defaults(fn=lambda ns: _script("train_surrogate.py", ["--profile", ns.profile]))
    sp2 = sub.add_parser("surrogate-predict", help="instant performance estimate of a fighter design file from the trained surrogate")
    sp2.add_argument("design")
    sp2.set_defaults(fn=_surrogate_predict)
    tp = sub.add_parser("topology", help="topology optimisation of a 2D body (density method, Brinkman Navier-Stokes, adjoint)")
    tp.add_argument("--shape", default="square")
    tp.add_argument("--profile", default="quick", choices=["quick", "standard", "full"])
    tp.set_defaults(fn=lambda ns: _script("shapeopt_topology.py", ["--shape", ns.shape, "--profile", ns.profile]))
    xf = sub.add_parser("xfoil", help="check an optimised airfoil with XFOIL over the hold-out flow states")
    xf.add_argument("--shape", default="naca2412")
    xf.set_defaults(fn=lambda ns: _script("shapeopt_xfoil.py", ["--shape", ns.shape]))
    tl = sub.add_parser("thermal-lab", help="open THERMAL LAB, the merged convection-diffusion PINN project, in the browser")
    tl.add_argument("--port", type=int, default=8766)
    tl.set_defaults(fn=lambda ns: subprocess.call([sys.executable, "-m", "aerosynapse.thermal.web.server", "--port", str(ns.port)]))
    wf = sub.add_parser("workflow", help="the staged design workflow: basic design, design, wing and body optimisation, structure, PINN skin, CFD, export")
    wf.add_argument("action", choices=["new", "run", "status", "report"])
    wf.add_argument("name", help="project name (folder under projects/) or a project folder")
    wf.add_argument("--from", dest="source", help="new: an input design file (any feature tree) instead of a family baseline")
    wf.add_argument("--family", default=None, help="new: fighter design-space family of the baseline (conventional, canard_delta, twin_tail_lerx, tailless_blended, freeform)")
    wf.add_argument("--code", default="mil_fighter", help="new: structural design code (mil_fighter, cs25, cs23_utility, aerobatic)")
    wf.add_argument("--W0", type=float, default=None, help="new: take-off mass, kg, for designs without fighter variables")
    wf.add_argument("--overwrite", action="store_true")
    wf.add_argument("--where", default=None, help="new: the parent folder of the new project folder (default projects/)")
    wf.add_argument("--stage", choices=["concept", "design", "wing", "body", "structure", "thermal", "simulation", "export"], help="run: one stage only")
    wf.add_argument("--start", default=None, help="run: rerun from this stage onward")
    wf.add_argument("--stop", default=None, help="run: stop after this stage")
    wf.add_argument("--profile", default="quick", choices=["quick", "standard"])
    wf.add_argument("--format", default="all", choices=["all", "pdf", "docx", "txt"], help="report: which formats to write")
    wf.set_defaults(fn=_workflow)
    sr = sub.add_parser("structure", help="wing box sizing and design-code checks of a design file (strength, buckling, flutter, divergence)")
    sr.add_argument("design")
    sr.add_argument("--W0", type=float, default=12000.0, help="take-off mass, kg")
    sr.add_argument("--code", default="mil_fighter", choices=["mil_fighter", "cs25", "cs23_utility", "aerobatic"])
    sr.add_argument("--material", default="al7075")
    sr.add_argument("--T", type=float, default=293.0, help="structure temperature, K")
    sr.add_argument("--vd", type=float, default=400.0, help="design dive speed, EAS m/s")
    sr.add_argument("--fuel", type=float, default=0.0, help="fuel in the wing, kg")
    sr.add_argument("--out", default=None)
    sr.set_defaults(fn=_structure)
    sm = sub.add_parser("simulate", help="simulate a design file: analytic (closed-form), pinn (parametric PINN), structure (FEM, flutter) or all")
    sm.add_argument("kind", choices=["analytic", "pinn", "structure", "turbulence", "all", "cfd", "flow3d", "optimise", "schema"])
    sm.add_argument("design")
    sm.add_argument("--out", default="results/studio/sim")
    sm.add_argument("--W0", type=float, default=12000.0)
    sm.add_argument("--code", default="mil_fighter", choices=["mil_fighter", "cs25", "cs23_utility", "aerobatic"])
    sm.add_argument("--material", default="al7075")
    sm.add_argument("--case", default="cruise", choices=["cruise", "transonic", "supersonic", "high_alpha", "sideslip", "gust"], help="cfd: flight case")
    sm.add_argument("--quality", default="standard", choices=["live", "draft", "standard", "fine"], help="cfd: grid and steps")
    sm.add_argument("--setup", default=None, help="a setup file (JSON) with any of the inputs listed by 'simulate schema -' (Studio writes it)")
    sm.set_defaults(fn=_simulate)
    ed = sub.add_parser("export-design", help="STEP, glTF, STL, OBJ, DXF three-view and A3 drawing of a design file")
    ed.add_argument("design")
    ed.add_argument("--out", default="results/studio/cad_package")
    ed.add_argument("--h", type=float, default=0.05, help="mesh spacing, m")
    ed.add_argument("--name", default="design")
    ed.set_defaults(fn=_export_design)
    en = sub.add_parser("engineer", help="clash analysis, design rules, composite laminate, skeleton rebuild, RFLP traceability")
    en.add_argument("kind", choices=["clash", "rules", "composites", "skeleton", "rflp"])
    en.add_argument("path", help="design file (rflp: project folder)")
    en.add_argument("--out", default="results/studio/engineer")
    en.add_argument("--layup", default=None, help="composites: e.g. [0/45/-45/90]s")
    en.add_argument("--material", default=None, help="composites: cfrp_im7, cfrp_bmi or gfrp")
    en.add_argument("--setup", default=None, help="a setup file (JSON) from Studio: layup, material, spacing")
    en.add_argument("--family", default=None, help="skeleton: design-space family")
    en.add_argument("--variables", default=None, help="skeleton: JSON object of changed variables")
    en.set_defaults(fn=_engineer)
    sub.add_parser("designer", help="open the Aircraft Designer app").set_defaults(fn=_designer)
    rp = sub.add_parser("run", help="run a single experiment from a YAML config")
    rp.add_argument("config")
    rp.add_argument("--profile", default=None, choices=["quick", "standard", "full"])
    rp.add_argument("--results", default="results")
    rp.set_defaults(fn=_run)
    ns = p.parse_args(argv)
    print(banner())
    return int(ns.fn(ns))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
