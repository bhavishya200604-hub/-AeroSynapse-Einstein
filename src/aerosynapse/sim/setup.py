"""Every input of every simulation: the setup schema that the Studio setup window shows and the CLI accepts.

A setup is a flat JSON object ``{key: value}``; missing keys take the defaults below. ``python -m aerosynapse.cli
simulate schema`` writes the schema (``assets/sim_setup_schema.json``) that the Studio form is built from, so the
form, the CLI and the solvers always agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MATERIALS = ["al7075", "al2024", "al_li", "ti64", "ti6242", "in718", "cfrp_im7", "cfrp_bmi", "cmc_csic", "steel_300m", "glass_epoxy"]
ZONES = ["radome", "fwd_fuselage", "centre_fuselage", "wing_box", "wing_le", "control", "aft_fuselage", "nozzle"]
PRESETS: dict[str, dict[str, Any]] = {
    "cruise": {"mach": 0.85, "alpha": 2.0, "beta": 0.0, "altitude": 11000.0, "gust_amplitude": 0.0, "turbulence_intensity": 0.0},
    "transonic": {"mach": 0.95, "alpha": 6.0, "beta": 0.0, "altitude": 5000.0, "gust_amplitude": 0.0, "turbulence_intensity": 0.0},
    "supersonic": {"mach": 1.6, "alpha": 2.0, "beta": 0.0, "altitude": 12000.0, "gust_amplitude": 0.0, "turbulence_intensity": 0.0},
    "high_alpha": {"mach": 0.3, "alpha": 25.0, "beta": 0.0, "altitude": 6000.0, "gust_amplitude": 0.0, "turbulence_intensity": 0.0},
    "sideslip": {"mach": 0.6, "alpha": 4.0, "beta": 5.0, "altitude": 6000.0, "gust_amplitude": 0.0, "turbulence_intensity": 0.0},
    "gust": {"mach": 0.7, "alpha": 3.0, "beta": 0.0, "altitude": 3000.0, "gust_amplitude": 5.0, "gust_wavelength": 20.0, "turbulence_intensity": 0.0},
    "turbulent": {"mach": 0.5, "alpha": 4.0, "beta": 0.0, "altitude": 3000.0, "gust_amplitude": 0.0, "turbulence_intensity": 5.0, "turbulence_length": 3.0},
}
QUALITY = {"live": (18.0, 260), "draft": (30.0, 900), "standard": (48.0, 2500), "fine": (70.0, 3500)}
# steady runs that stop on settled forces get room to settle (the stop usually comes well before): on the Falcon
# cruise, draft forces after 900 steps were still far off (C_D 0.030 against a settled 0.0107); they settle by ~1400
STEADY_STEPS = {"live": 260, "draft": 2400, "standard": 4000, "fine": 5000}


def F(key: str, label: str, group: str, kind: str, default: Any, unit: str = "", lo: float | None = None, hi: float | None = None,
      choices: list[str] | None = None, help: str = "") -> dict[str, Any]:  # noqa: A002 (the schema's own word)
    d: dict[str, Any] = {"key": key, "label": label, "group": group, "type": kind, "default": default, "unit": unit, "help": help}
    if lo is not None:
        d["min"] = lo
    if hi is not None:
        d["max"] = hi
    if choices:
        d["choices"] = choices
    return d


FLIGHT = "Flight condition"
TURB = "Free-stream turbulence and gusts"
MODEL = "Physical model"
GRID = "Grid and time stepping"
OUT = "Outputs and probes"

SCHEMA: dict[str, dict[str, Any]] = {
    "cfd": {"title": "3D flow (CFD)", "presets": PRESETS, "fields": [
        F("preset", "Preset", FLIGHT, "choice", "cruise", choices=["custom", *PRESETS], help="Fills the flight condition; change any value afterwards"),
        F("mach", "Mach number", FLIGHT, "num", 0.85, "", 0.05, 4.0, help="Flight Mach number"),
        F("alpha", "Angle of attack", FLIGHT, "num", 2.0, "deg", -20, 60, help="Incidence of the flight path to the body x axis"),
        F("beta", "Sideslip", FLIGHT, "num", 0.0, "deg", -30, 30, help="Sideslip angle; any value other than 0 needs the full model"),
        F("altitude", "Altitude", FLIGHT, "num", 11000.0, "m", 0, 25000, help="ISA altitude: sets temperature, pressure, density and Reynolds number"),
        F("delta_T", "Temperature offset", FLIGHT, "num", 0.0, "K", -40, 40, help="Hot (+) or cold (-) day relative to ISA"),
        F("turbulence_intensity", "Turbulence intensity Tu", TURB, "num", 0.0, "%", 0, 20,
          help="r.m.s. velocity fluctuation of the oncoming air as a percentage of the flight speed (synthetic Fourier-mode inflow turbulence). 0 = calm air"),
        F("turbulence_length", "Turbulence length scale", TURB, "num", 3.0, "m", 0.1, 100, help="Integral length of the inflow eddies (energy-containing size)"),
        F("gust_amplitude", "Vertical gust", TURB, "num", 0.0, "% of speed", 0, 30, help="1-cosine-like sinusoidal vertical gust amplitude"),
        F("gust_wavelength", "Gust wavelength", TURB, "num", 30.0, "m", 1, 1000),
        F("viscous", "Viscous flow", MODEL, "bool", True, help="Off: inviscid Euler flow (no friction, no separation from smooth walls)"),
        F("wall_model", "Wall model", MODEL, "bool", True, help="Equilibrium law-of-the-wall friction on the walls"),
        F("sgs", "Turbulence model (LES)", MODEL, "choice", "wale", choices=["wale", "smagorinsky", "none"], help="Sub-grid-scale model of the large-eddy simulation"),
        F("low_mach_fix", "Low-Mach correction", MODEL, "choice", "auto", choices=["auto", "on", "off"], help="Thornber correction: less numerical dissipation of eddies; auto = on for time-accurate runs"),
        F("thin_surfaces", "Thin-surface treatment", MODEL, "bool", True, help="Two-sided walls for wings and fins thinner than a cell"),
        F("quality", "Quality", GRID, "choice", "draft", choices=["live", "draft", "standard", "fine", "custom"], help="Sets cells per body length and steps; custom uses the two values below"),
        F("cells_per_length", "Cells per body length", GRID, "num", 30.0, "", 8, 200, help="Used when quality is custom"),
        F("steps", "Time steps", GRID, "int", 900, "", 50, 100000, help="Used when quality is custom"),
        F("full_model", "Full model (both sides)", GRID, "bool", True, help="Off: half model with a symmetry plane (half the cells, no sideslip)"),
        F("unsteady", "Time-accurate", GRID, "choice", "auto", choices=["auto", "on", "off"], help="auto: time-accurate when there is turbulence or a gust"),
        F("average_from", "Average from", GRID, "num", 50.0, "% of steps", 0, 95, help="Statistics (mean, r.m.s., turbulent kinetic energy) are collected after this point of a time-accurate run"),
        F("cfl", "CFL number", GRID, "num", 1.2, "", 0.2, 2.0),
        F("residual_drop", "Residual target", GRID, "num", 1e-5, "", 1e-9, 1e-1, help="Steady runs stop when the residual falls this far and the lift has settled"),
        F("max_cells", "Cell limit", GRID, "num", 2.4e6, "", 1e4, 2.5e6, help="32-bit build limit is about 2.5 million"),
        F("stretch", "Grid stretching", GRID, "num", 1.15, "", 1.0, 1.4),
        F("domain_up", "Domain upstream", GRID, "num", 0.0, "body lengths", 0, 10, help="0 = automatic (1.5, or 0.4 supersonic)"),
        F("domain_down", "Domain downstream", GRID, "num", 0.0, "body lengths", 0, 20, help="0 = automatic (2.5, or 0.6 supersonic)"),
        F("domain_side", "Domain sideways", GRID, "num", 0.0, "body lengths", 0, 10, help="0 = automatic"),
        F("domain_vert", "Domain vertical", GRID, "num", 0.0, "body lengths", 0, 10, help="0 = automatic"),
        F("start", "Start from", GRID, "choice", "free stream", choices=["free stream", "coarse pass first", "previous run"],
          help="coarse pass first: a quick run on a grid 0.55 times as fine, and the full grid starts from its field (grid sequencing); "
               "previous run: the folder below. On the Falcon cruise the slow force swings set the run time either way, so the gain was small"),
        F("start_folder", "Previous run folder", GRID, "text", "", help="With 'previous run': a run folder of the same flight with a saved 3D field"),
        F("force_band", "Stop when forces settle", GRID, "num", 0.5, "%", 0.0, 10.0,
          help="Steady runs stop once the mean CL and CD over the last 600 steps agree within this band with the 600 steps before; "
               "the result is then that mean. 0 = run all steps or to the residual target"),
        F("save_volume", "Save the 3D field", OUT, "bool", True, help="Needed for 3D streamlines, vortex cores, slices and the instability analysis"),
        F("flow_pictures", "Flow pictures", OUT, "choice", "stills and movies", choices=["stills and movies", "stills", "none"],
          help="Two-panel plane cuts on white paper: temperature rise with tracer particles over vorticity; movies show the tracers streaming"),
        F("flow_section_span", "Wing section for the pictures", OUT, "num", 55.0, "% semi-span", 5.0, 95.0, help="Where the wing-section cut is taken"),
        F("probe_1", "Probe 1", OUT, "text", "", "x, y, z m", help="A point where u, v, w, p are recorded every step; empty = automatic probes (wake, tip wakes, over the wing, upstream)"),
        F("probe_2", "Probe 2", OUT, "text", "", "x, y, z m"),
        F("probe_3", "Probe 3", OUT, "text", "", "x, y, z m"),
    ]},
    "optimise": {"title": "CFD design optimisation", "fields": [
        F("objective", "Objective", "Goal", "choice", "max L/D", choices=["max L/D", "min drag", "max lift"]),
        F("cl_min", "Lowest CL allowed", "Goal", "num", 0.0, "", 0.0, 2.0, help="A soft constraint: wings below this lift coefficient are penalised; 0 = none"),
        F("budget", "CFD runs", "Search", "int", 14, "", 4, 80, help="Every run is one 3D flow solution; the first is the baseline design"),
        F("initial_runs", "Space-filling runs", "Search", "int", 6, "", 2, 30, help="Latin-hypercube wings before the Gaussian process starts choosing"),
        F("seed", "Random seed", "Search", "int", 1, "", 0, 9999),
        F("fidelity", "Grid of each run", "Search", "choice", "draft", choices=["draft", "standard", "live"],
          help="draft is the smallest grid whose wings are thicker than a cell; live is for trying the study out only"),
        F("force_band", "Stop when forces settle", "Search", "num", 0.5, "%", 0.05, 5.0),
        F("mach", "Mach number", FLIGHT, "num", 0.85, "", 0.1, 2.5),
        F("alpha", "Angle of attack", FLIGHT, "num", 2.0, "deg", -5.0, 20.0),
        F("altitude", "Altitude", FLIGHT, "num", 11000.0, "m", 0.0, 25000.0),
        F("keep_area", "Keep the wing area", "Wing variables", "bool", True, help="The root chord follows span and taper so the wing area stays the same"),
        F("vary_sweep", "Vary sweep", "Wing variables", "bool", True),
        F("sweep_min", "Sweep from", "Wing variables", "num", 20.0, "deg", 0.0, 70.0),
        F("sweep_max", "Sweep to", "Wing variables", "num", 50.0, "deg", 0.0, 70.0),
        F("vary_taper", "Vary taper", "Wing variables", "bool", True),
        F("taper_min", "Taper from", "Wing variables", "num", 0.15, "", 0.05, 1.0),
        F("taper_max", "Taper to", "Wing variables", "num", 0.6, "", 0.05, 1.0),
        F("vary_twist", "Vary tip twist", "Wing variables", "bool", True),
        F("twist_min", "Twist from", "Wing variables", "num", -6.0, "deg", -12.0, 6.0),
        F("twist_max", "Twist to", "Wing variables", "num", 1.0, "deg", -12.0, 6.0),
        F("vary_dihedral", "Vary dihedral", "Wing variables", "bool", False),
        F("dihedral_min", "Dihedral from", "Wing variables", "num", -5.0, "deg", -15.0, 15.0),
        F("dihedral_max", "Dihedral to", "Wing variables", "num", 6.0, "deg", -15.0, 15.0),
        F("vary_span", "Vary span", "Wing variables", "bool", False, help="As a factor on the baseline span; with the area kept, this is the aspect ratio"),
        F("span_min", "Span from", "Wing variables", "num", 0.85, "x baseline", 0.5, 2.0),
        F("span_max", "Span to", "Wing variables", "num", 1.2, "x baseline", 0.5, 2.0),
    ]},
    "turbulence": {"title": "Turbulence and flow instability", "fields": [
        F("flow_folder", "3D flow results", "Source", "text", "", "folder", help="A finished 3D flow run (Studio fills in the last one). Empty: model wake and no field statistics"),
        F("mach", "Mach number", FLIGHT, "num", 0.85, "", 0.05, 3.0),
        F("altitude", "Altitude", FLIGHT, "num", 11000.0, "m", 0, 25000),
        F("alpha", "Angle of attack", FLIGHT, "num", 2.0, "deg", -10, 20, help="For the wing-section boundary layer"),
        F("turbulence_intensity", "Free-stream Tu", "Transition", "num", 0.1, "%", 0.01, 5, help="Sets the critical amplification N (Mack: N = -8.43 - 2.4 ln Tu)"),
        F("n_crit", "Critical N (override)", "Transition", "num", 0.0, "", 0, 20, help="0 = from Tu; 9 is the classic flight value"),
        F("section_span", "Section station", "Transition", "num", 0.35, "of semi-span", 0.0, 0.95, help="Spanwise station of the wing section analysed"),
        F("roughness", "Surface", "Transition", "choice", "smooth", choices=["smooth", "painted", "rough"], help="Rough surfaces lower N: painted -1, rough -3"),
        F("wake_stations", "Wake stations", "Wake instability", "text", "0.15, 0.4, 0.8", "x / L behind the tail", help="Where the wake profile is taken for the Rayleigh stability analysis"),
        F("rayleigh_points", "Collocation points", "Wake instability", "int", 80, "", 30, 200, help="Chebyshev points of the Rayleigh eigenproblem"),
        F("k_max", "Largest wavenumber", "Wake instability", "num", 3.0, "1 / wake half-width", 0.5, 10),
        F("spectrum_segments", "Spectrum segments", "Spectra", "int", 4, "", 1, 32, help="Welch averaging segments for the probe and force spectra"),
    ]},
    "analytic": {"title": "Analytic solutions", "fields": [
        F("alpha_min", "Angle from", "Panel method", "num", -2.0, "deg", -20, 20),
        F("alpha_max", "Angle to", "Panel method", "num", 8.0, "deg", -10, 25),
        F("alpha_step", "Angle step", "Panel method", "num", 1.0, "deg", 0.1, 5),
        F("ll_alpha", "Lifting-line angle", "Lifting line", "num", 4.0, "deg", 0.5, 15),
        F("nose_mach", "Nose and inlet Mach", "Shocks", "num", 1.6, "", 1.05, 5),
        F("ackeret_mach", "Section theory Mach", "Shocks", "num", 1.8, "", 1.05, 5),
        F("dash_mach", "Dash Mach", "Skin heating", "num", 1.8, "", 0.5, 5),
        F("dash_altitude", "Dash altitude", "Skin heating", "num", 12000.0, "m", 0, 25000),
        F("dash_seconds", "Dash duration", "Skin heating", "num", 600.0, "s", 10, 7200),
        F("skin_start_offset", "Skin start above ambient", "Skin heating", "num", 5.0, "K", -20, 100),
    ]},
    "pinn": {"title": "PINN skin heating", "fields": [
        F("biot_1", "Check at Biot", "Accuracy check", "num", 0.5, "", 0.01, 50),
        F("biot_2", "and Biot", "Accuracy check", "num", 2.0, "", 0.01, 50),
        F("biot_3", "and Biot", "Accuracy check", "num", 10.0, "", 0.01, 50),
        F("field_biot", "Field plot Biot", "Accuracy check", "num", 2.0, "", 0.01, 50),
        *[f for z in ZONES for f in (
            F(f"zone_{z}_material", f"{z.replace('_', ' ')}: material", "Skin zones", "choice", "design", choices=["design", *MATERIALS],
              help="design = the material in the design file (or the selection by temperature)"),
            F(f"zone_{z}_thickness", f"{z.replace('_', ' ')}: thickness", "Skin zones", "num", 0.0, "mm", 0, 30, help="0 = the design's thickness"))],
    ]},
    "structure": {"title": "Structure (FEM, loads, flutter)", "fields": [
        F("W0_kg", "Take-off mass", "Loads", "num", 12000.0, "kg", 200, 600000),
        F("fuel_fraction", "Fuel in the wing", "Loads", "num", 25.0, "% of mass", 0, 60),
        F("code", "Design code", "Loads", "choice", "mil_fighter", choices=["mil_fighter", "cs25", "cs23_utility", "aerobatic"]),
        F("v_d_eas", "Dive speed", "Loads", "num", 400.0, "m/s EAS", 30, 700),
        F("cl_max", "Maximum lift coefficient", "Loads", "num", 1.2, "", 0.5, 3.0),
        F("material", "Wing box material", "Material", "choice", "al7075", choices=MATERIALS),
        F("T_K", "Structure temperature", "Material", "num", 293.0, "K", 150, 900, help="Hot structure lowers the allowable stresses"),
    ]},
    "composites": {"title": "Composite laminate", "engineer": True, "fields": [
        F("layup", "Ply layup", "Laminate", "text", "[0/45/-45/90]s", "deg", help="Ply angles from the outer face, '/' between plies, 's' for a mirrored (symmetric) stack, e.g. [0/45/-45/90]s or [0/90/0]"),
        F("material", "Ply material", "Laminate", "choice", "cfrp_im7", choices=["cfrp_im7", "cfrp_bmi", "gfrp"], help="Unidirectional ply: carbon IM7/8552, carbon/BMI or glass/epoxy"),
    ]},
    "clash": {"title": "Clash analysis", "engineer": True, "fields": [
        F("spacing", "Analysis spacing", "Resolution", "num", 0.05, "m", 0.005, 0.5, help="Grid spacing of the intersection volumes: smaller finds thinner overlaps but takes longer"),
    ]},
}


def defaults(kind: str) -> dict[str, Any]:
    return {f["key"]: f["default"] for f in SCHEMA[kind]["fields"]}


def merged(kind: str, setup: dict[str, Any] | None) -> dict[str, Any]:
    """Defaults overlaid with the user's setup, values checked against their ranges and choices."""
    s = defaults(kind)
    for f in SCHEMA[kind]["fields"]:
        k = f["key"]
        if not setup or k not in setup or setup[k] is None:
            continue
        v = setup[k]
        if f["type"] in ("num", "int"):
            v = float(v)
            if "min" in f and v < f["min"] or "max" in f and v > f["max"]:
                raise ValueError(f"{f['label']} = {v:g} is outside {f.get('min')} to {f.get('max')} {f['unit']}")
            v = int(round(v)) if f["type"] == "int" else v
        elif f["type"] == "bool":
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        elif f["type"] == "choice" and str(v) not in f["choices"]:
            raise ValueError(f"{f['label']}: '{v}' is not one of {', '.join(f['choices'])}")
        s[k] = v
    return s


def load(path: str | Path | None) -> dict[str, Any] | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def describe(kind: str, s: dict[str, Any]) -> list[str]:
    """The setup as report lines, grouped as in the form."""
    lines, grp = [], None
    for f in SCHEMA[kind]["fields"]:
        if f["group"] != grp:
            grp = f["group"]
            lines.append(f"  {grp}:")
        v = s.get(f["key"])
        if v in ("", None):
            continue
        vs = f"{v:g}" if isinstance(v, float) else str(v)
        lines.append(f"    {f['label']}: {vs} {f['unit']}".rstrip())
    return lines


def write_schema(path: str | Path | None = None) -> Path:
    p = Path(path) if path else Path(__file__).resolve().parents[3] / "assets" / "sim_setup_schema.json"
    p.write_text(json.dumps({"kinds": list(SCHEMA), **SCHEMA}, indent=1), encoding="utf-8")
    return p
