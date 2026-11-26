"""The Applications tab of AeroSynapse Studio: every runnable part of AeroSynapse-Einstein in one catalogue.

    python scripts/make_app_catalog.py

Writes ``assets/app_catalog.json``: groups of entries, each with a label, the command line after ``python`` (run
from the project root with ``PYTHONPATH=src``; ``{design}`` stands for the design open in Studio), a description
taken from the module's own docstring, an icon, and ``web: true`` for the local apps that open in the browser.
Studio shows one button per group; a click lists the group's entries, and picking one opens it in the task window
(command, description, Run, live log, results).

Sources, so nothing is left out by hand:
* the command-line interface (``aerosynapse.cli``): every subcommand, with arguments that need nothing else;
* every research script in ``scripts/`` that is an experiment, a study or a report (helpers are skipped);
* every experiment of the thermal PINN project (``aerosynapse.thermal.experiments``), in phase order;
* the local web apps (Designer, THERMAL LAB, the thermal results app).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLI = [  # (group, label, args, icon) - the subcommands of aerosynapse.cli, ready to run
    ("Aircraft design", "Whole design workflow (8 stages)", "-m aerosynapse.cli workflow run falcon --profile quick", "wf_window"),
    ("Aircraft design", "Whole-aircraft study (airliner, SST)", "-m aerosynapse.cli aircraft --profile quick", "opt_aircraft"),
    ("Aircraft design", "Fighter study (five families)", "-m aerosynapse.cli fighter --profile quick", "opt_fighter"),
    ("Aircraft design", "Quick aerodynamics of this design", "-m aerosynapse.cli cad-analyse {design}", "quick_aero"),
    ("Aircraft design", "Surrogate: train from the data store", "-m aerosynapse.cli surrogate --profile quick", "opt_surrogate"),
    ("Aircraft design", "Surrogate: predict this design", "-m aerosynapse.cli surrogate-predict {design}", "opt_surrogate"),
    ("Aerodynamics and CFD", "3D flow of this design", "-m aerosynapse.cli cfd {design} --cells 30 --steps 2400 --save-volume --out results/studio/cfd_app", "cfd_cruise"),
    ("Aerodynamics and CFD", "3D flow verification suite", "-m aerosynapse.cli cfd-verify --profile quick", "verify"),
    ("Aerodynamics and CFD", "Turbulence and instability of this design", "-m aerosynapse.cli simulate turbulence {design} --out results/studio/turbulence_app", "turbulence"),
    ("Aerodynamics and CFD", "Analytic solutions of this design", "-m aerosynapse.cli simulate analytic {design} --out results/studio/analytic_app", "analytic"),
    ("Shape and wing optimisation", "Wing optimisation with CFD (Bayesian)", "-m aerosynapse.cli simulate optimise {design} --out results/optimise/studio_app", "optimise"),
    ("Shape and wing optimisation", "Airfoil and wing shape (CMA-ES, quick)", "-m aerosynapse.cli shapeopt --shape naca2412 --profile quick", "opt_airfoil"),
    ("Shape and wing optimisation", "Airfoil and wing shape (CMA-ES, standard)", "-m aerosynapse.cli shapeopt --shape naca2412 --profile standard", "opt_airfoil"),
    ("Shape and wing optimisation", "Topology optimisation of a 2D body", "-m aerosynapse.cli topology --shape square --profile quick", "opt_topology"),
    ("Shape and wing optimisation", "XFOIL check of the optimised airfoil", "-m aerosynapse.cli xfoil --shape naca2412", "opt_xfoil"),
    ("Structures", "Wing box sizing and design codes", "-m aerosynapse.cli structure {design} --W0 12000 --code mil_fighter --material al7075 --vd 400", "opt_structure"),
    ("Structures", "Structure simulation (FEM, flutter)", "-m aerosynapse.cli simulate structure {design} --out results/studio/structure_app", "structure"),
    ("Structures", "Composite laminate", "-m aerosynapse.cli engineer composites {design} --out results/studio/laminate_app", "laminate"),
    ("Structures", "Clash analysis", "-m aerosynapse.cli engineer clash {design} --out results/studio/clash_app", "clash"),
    ("Thermal and PINN", "Skin heating with the PINN (aircraft)", "-m aerosynapse.cli thermal --profile quick", "pinn"),
    ("Thermal and PINN", "PINN simulation of this design", "-m aerosynapse.cli simulate pinn {design} --out results/studio/pinn_app", "pinn"),
    ("Research pipeline", "Environment and hardware check", "-m aerosynapse.cli doctor", "check"),
    ("Research pipeline", "Train PINN and baseline studies", "-m aerosynapse.cli train --profile quick", "pinn_2d"),
    ("Research pipeline", "Optimisation and inverse studies", "-m aerosynapse.cli optimize --profile quick", "opt_centre"),
    ("Research pipeline", "Trust reports and fallback checks", "-m aerosynapse.cli validate --profile quick", "check"),
    ("Research pipeline", "Reports from the results", "-m aerosynapse.cli report --profile quick", "report_html"),
    ("Research pipeline", "Paper figures", "-m aerosynapse.cli figures --profile quick", "results"),
    ("Research pipeline", "Reproduce every reported result", "-m aerosynapse.cli reproduce --profile quick", "wf_run"),
    ("Export", "CAD package of this design (STEP, glTF, STL, DXF)", "-m aerosynapse.cli export-design {design} --out results/studio/cad_app", "package"),
]
WEB = [
    ("Local apps", "Aircraft Designer (browser)", "-m aerosynapse.cli designer", "opt_centre", "Editor, studies, 3D flow, materials, skin heating, data store and the optimisation centre, in the browser."),
    ("Local apps", "THERMAL LAB (browser)", "-m aerosynapse.thermal.web.server --port 8766", "pinn_lab", "The whole convection-diffusion PINN project: verified solvers, parametric PINNs, inverse identification, real data."),
    ("Local apps", "Thermal results app (browser)", "-m streamlit run src/aerosynapse/thermal/app/main.py --server.headless true --browser.gatherUsageStats false",
     "results", "Saved PINN ensembles against the verified solvers, inverse identification, grid convergence, uncertainty and export."),
]
SCRIPT_GROUPS = {  # research scripts worth running from Studio, by file-name pattern
    r"^(cfd_|fighter_cfd)": ("Aerodynamics and CFD", "cfd_cruise"),
    r"^shapeopt_(search|verify|trust_split|xfoil|topology|rerun_ns)": ("Shape and wing optimisation", "opt_airfoil"),
    r"^run_(phase\d|phase3_extras|phase4_genmap|airfrans|quality_gate)": ("Research pipeline", "wf_run"),
    r"^(make_(report|figures|aircraft_report|fighter_report|shapeopt_report|reproducibility|architecture)|reproduce_all|audit_claims|check_paper)": ("Reports and audits", "report_html"),
}
PHASE_NAMES = {"1": "verified solvers", "2": "data", "3": "1D PINN", "4": "2D PINN", "5": "inverse problems", "6": "analysis",
               "6b": "gap runs", "7": "visualisation", "8": "interface", "9": "finalisation", "10": "real data"}


def doc_first(path: Path) -> str:
    try:
        d = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    except (SyntaxError, UnicodeDecodeError):
        return ""
    text = " ".join(d.split("\n\n")[0].split())
    return text[:400]


def has_option(path: Path, opt: str) -> str | None:
    t = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'add_argument\("' + re.escape(opt) + r'"[^)]*default="?([\w.-]+)', t)
    return m.group(1) if m else ("" if f'"{opt}"' in t else None)


def main() -> None:
    groups: dict[str, list[dict]] = {}

    def add(group: str, label: str, args: str, icon: str, about: str, web: bool = False) -> None:
        groups.setdefault(group, []).append({"label": label, "args": args, "icon": icon, "about": about, "web": web})

    for g, label, args, icon in CLI:
        add(g, label, args, icon, "")
    for g, label, args, icon, about in WEB:
        add(g, label, args, icon, about, web=True)
    for p in sorted((ROOT / "scripts").glob("*.py")):
        for pat, (g, icon) in SCRIPT_GROUPS.items():
            if re.search(pat, p.stem):
                prof = has_option(p, "--profile")
                args = f"scripts/{p.name}" + (" --profile quick" if prof is not None and prof in ("quick", "standard", "") else "")
                add(g, p.stem.replace("_", " "), args, icon, doc_first(p))
                break
    exp = ROOT / "src" / "aerosynapse" / "thermal" / "experiments"

    def phase_key(p: Path) -> tuple:
        m = re.match(r"p(\d+)(b?)_", p.stem)
        return (int(m.group(1)), m.group(2), p.stem) if m else (99, "", p.stem)

    for p in sorted(exp.glob("p*.py"), key=phase_key):
        m = re.match(r"p(\d+)(b?)_", p.stem)
        if not m:
            continue
        ph = m.group(1) + m.group(2)
        g = f"Thermal PINN research: phase {m.group(1)}" if int(m.group(1)) <= 5 else "Thermal PINN research: phases 6 to 10"
        prof = has_option(p, "--profile")
        args = f"-m aerosynapse.thermal.experiments.{p.stem}" + (" --profile smoke" if prof is not None else "")
        about = doc_first(p)
        add(g, f"{p.stem} ({PHASE_NAMES.get(ph, 'phase ' + ph)})", args, "pinn_2d", about + ("" if prof is not None else " Runs with its own fixed settings."))
    for f in ("pipeline.py",):
        if (exp / f).exists():
            add("Thermal PINN research: phases 6 to 10", "pipeline (end to end)", f"-m aerosynapse.thermal.experiments.{f[:-3]}", "wf_run", doc_first(exp / f))
    order = ["Aircraft design", "Aerodynamics and CFD", "Shape and wing optimisation", "Structures", "Thermal and PINN", "Local apps",
             "Thermal PINN research: phase 1", "Thermal PINN research: phase 2", "Thermal PINN research: phase 3", "Thermal PINN research: phase 4",
             "Thermal PINN research: phase 5", "Thermal PINN research: phases 6 to 10", "Research pipeline", "Reports and audits", "Export"]
    out = [{"title": g, "items": groups[g]} for g in order if g in groups] + [{"title": g, "items": v} for g, v in groups.items() if g not in order]
    n = sum(len(g["items"]) for g in out)
    (ROOT / "assets" / "app_catalog.json").write_text(json.dumps({"groups": out, "entries": n}, indent=1), encoding="utf-8")
    print(f"{n} entries in {len(out)} groups -> assets/app_catalog.json")
    for g in out:
        print(f"  {g['title']}: {len(g['items'])}")


if __name__ == "__main__":
    main()
