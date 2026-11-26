"""Phase 8 gate: the interface works and agrees with the project's earlier results.

    python -m aerosynapse.thermal.experiments.p8_gate [--tag default] [--skip-launch]

Checks (written to results/p8_gate_<tag>/metrics.json and docs/tables/phase8_gate.md):
  G1 the app launches: `streamlit run src/app/main.py` answers its health check and serves the page
  G2 headless AppTest: every section of every case runs without an exception or an error message
  G3 walkthrough: the screenshots of experiments/p8_walkthrough.py exist and none timed out or showed an exception
  G4 numbers through the interface's backend agree with earlier phases (inverse vs the Phase 5 gate table; solver
     orders vs Phase 1; PINN errors inside / outside the training range vs Step 6.3; Graetz Nusselt error vs Step 6.7)
  G5 the validity guard: extrapolation is flagged 'stop' (PINN range, cavity above the critical Re) and the CLI's
     --strict exits with code 2 on it
Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.request

import numpy as np

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

SECTIONS = [
    "Overview",
    "Solver",
    "PINN",
    "Comparison",
    "Inverse",
    "Convergence",
    "Training monitor",
    "Uncertainty & validity",
    "Export",
]
BUTTONS = {
    ("D_graetz", "Solver"): "run_solver",
    ("cavity", "Solver"): "run_solver",
    ("D_graetz", "PINN"): "run_pinn",
    ("D_graetz", "Comparison"): "run_cmp",
}


def g1_launch(port: int = 8502, timeout: float = 90.0) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(__import__("aerosynapse.thermal._home", fromlist=["HOME"]).CODE / "app/main.py"),
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    out = {"port": port, "healthy": False, "page_status": None}
    try:
        while time.perf_counter() - t0 < timeout:
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2).read() == b"ok":
                    out["healthy"] = True
                    out["seconds_to_healthy"] = round(time.perf_counter() - t0, 2)
                    out["page_status"] = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).status
                    break
            except OSError:
                time.sleep(0.5)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    out["pass"] = bool(out["healthy"] and out["page_status"] == 200)
    return out


def g2_apptest() -> dict:
    from streamlit.testing.v1 import AppTest

    from aerosynapse.thermal.app import backend as B

    rows = []
    for case in B.CASES:
        for sec in SECTIONS:
            t0 = time.perf_counter()
            at = AppTest.from_file(str(__import__("aerosynapse.thermal._home", fromlist=["HOME"]).CODE / "app/main.py"), default_timeout=900)
            at.query_params["case"] = case
            at.query_params["section"] = sec
            at.run()
            if (case, sec) in BUTTONS:
                at.button(key=BUTTONS[(case, sec)]).click().run()
            rows.append(
                {
                    "case": case,
                    "section": sec,
                    "exceptions": len(at.exception),
                    "errors": len(at.error),
                    "error_text": [e.value[:160] for e in at.error],
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            )
            print(f"  AppTest {case:9s} {sec:24s} exc {len(at.exception)} err {len(at.error)}", flush=True)
    # an error message is expected only where the guard must speak; on default parameters there is none
    bad = [r for r in rows if r["exceptions"] or r["errors"]]
    return {"runs": rows, "n": len(rows), "failed": len(bad), "pass": not bad}


def g3_walkthrough() -> dict:
    f = ROOT / "results/p8_walkthrough_default/metrics.json"
    if not f.exists():
        return {"pass": False, "reason": "run python -m aerosynapse.thermal.experiments.p8_walkthrough first"}
    m = json.loads(f.read_text(encoding="utf-8"))
    shots = m["shots"]
    missing = [s["file"] for s in shots if not (ROOT / s["file"]).exists()]
    return {
        "shots": len(shots),
        "failures": m["failures"],
        "missing": missing,
        "pass": bool(shots) and not m["failures"] and not missing,
    }


def g4_numbers() -> dict:
    from aerosynapse.thermal.app import backend as B

    out: dict = {}
    # inverse: same data as the Phase 5 gate (default experiment, 2 % noise, seed 0); the table prints exp(e) - 1
    gate = json.loads((ROOT / "results/p5_gate/metrics.json").read_text(encoding="utf-8"))
    ls = next(r for r in gate["rows"] if r["name"].startswith("least squares (analytic), parametric"))
    ref = {k: 100 * (math.exp(ls["err"][k]) - 1) for k in ("alpha", "k", "h")}
    inv = B.inverse()
    got = {r["parameter"]: r["error_pct"] for r in inv["rows"]}
    out["inverse"] = {
        "app_error_pct": got,
        "phase5_gate_error_pct": ref,
        "max_abs_diff_pp": max(abs(got[k] - ref[k]) for k in ref),
        "covers_truth": all(r["ci95"][0] < r["truth"] < r["ci95"][1] for r in inv["rows"]),
    }
    nos = B.inverse(q_vol=0.0)
    out["inverse_no_source"] = {
        "identifiable": nos["identifiable"],
        "rank": nos["rank"],
        "null_direction": nos["null_directions"],
    }
    # verification: observed orders
    cvA = B.convergence("A_steady", {"Pe": 10.0}, [10, 20, 40, 80, 160, 320])
    out["order_steady"] = {k: v["observed_order"] for k, v in cvA["schemes"].items()}
    out["order_slab"] = {
        f"Bi={bi:g}": B.convergence("B_slab", {"Bi": bi}, [10, 20, 40, 80, 160])["schemes"]["central"]["observed_order"]
        for bi in (0.1, 1.0, 10.0)
    }
    # PINN inside / outside the training range (Step 6.3: error at most 1.2 %, up to 55 % outside, slab)
    out["pinn"] = {}
    if B.pinn_available("B_slab")[0]:
        for bi in (1.0, 30.0):
            c = B.compare("B_slab", {"Bi": bi})
            out["pinn"][f"B_slab Bi={bi:g}"] = {
                "rel_l2_vs_solver": c["rel_l2"],
                "solver_s": c["solver_s"],
                "pinn_s": c["pinn_s"],
                "guard": [v["level"] for v in c["validity"]],
            }
        c = B.compare("A_steady", {"Pe": 10.0})
        out["pinn"]["A_steady Pe=10"] = {
            "rel_l2_vs_solver": c["rel_l2"],
            "solver_s": c["solver_s"],
            "pinn_s": c["pinn_s"],
        }
        sw = B.sweep_1d("B_slab", np.array([1.0, 30.0]))
        out["pinn"]["B_slab sweep (exact reference)"] = {
            "rel_l2": sw["rel_l2"].tolist(),
            "spread": sw["rms_sd"].tolist(),
        }
    if B.pinn_available("D_graetz")[0]:
        c = B.compare("D_graetz", {"Re": 30.0, "Pr": 1.0, "Bi": 5.0})
        p67 = json.loads((ROOT / "results/p6_7_cost_default/metrics.json").read_text(encoding="utf-8"))
        out["graetz"] = {
            "rel_l2_vs_solver": c["rel_l2"],
            "Nu_solver": c["Nu_solver"],
            "Nu_pinn": c["Nu_pinn"],
            "Nu_rel_err": c["Nu_rel_err"],
            "step_6_7_pinn_error": p67["analysis"]["D_graetz"]["pinn_error"],
            "solver_s": c["solver_s"],
            "pinn_s": c["pinn_s"],
        }
    cav = {}
    for Re, N in ((100.0, 32), (1000.0, 48)):
        r = B.solve("cavity", {"Re": Re, "Pr": 0.71}, {"N": N})
        cav[f"Re={Re:g} N={N}"] = {
            "psi_min": r["psi_min"],
            "converged": r["converged"],
            "Nu_lid": r["Nu_lid"],
            "wall_s": r["wall_s"],
        }
    out["cavity"] = cav
    ok = (
        out["inverse"]["max_abs_diff_pp"] < 0.1
        and out["inverse"]["covers_truth"]
        and not out["inverse_no_source"]["identifiable"]
        and abs(out["order_steady"]["central"] - 2) < 0.1
        and abs(out["order_steady"]["upwind1"] - 1) < 0.15
        and all(abs(v - 2) < 0.1 for v in out["order_slab"].values())
        and all(v["converged"] for v in cav.values())
    )
    out["pass"] = bool(ok)
    return out


def g5_guard() -> dict:
    from aerosynapse.thermal.app import backend as B
    from aerosynapse.thermal.app import cli

    cases = {
        "B_slab Bi=30": ("B_slab", {"Bi": 30.0}),
        "A_steady Pe=100": ("A_steady", {"Pe": 100.0}),
        "D_graetz Re=200": ("D_graetz", {"Re": 200.0, "Pr": 1.0, "Bi": 5.0}),
        "cavity Re=12000": ("cavity", {"Re": 12000.0, "Pr": 0.71}),
        "B_slab Bi=1": ("B_slab", {"Bi": 1.0}),
        "cavity Re=400": ("cavity", {"Re": 400.0, "Pr": 0.71}),
    }
    levels = {k: [m["level"] for m in B.validity(c, p)] for k, (c, p) in cases.items()}
    expect_stop = ["B_slab Bi=30", "A_steady Pe=100", "D_graetz Re=200", "cavity Re=12000"]
    expect_ok = ["B_slab Bi=1", "cavity Re=400"]
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        code_stop = cli.main(["validity", "--case", "B_slab", "--set", "Bi=30", "--strict"])
        code_ok = cli.main(["validity", "--case", "B_slab", "--set", "Bi=1", "--strict"])
    ok = (
        all("stop" in levels[k] for k in expect_stop)
        and all("stop" not in levels[k] for k in expect_ok)
        and code_stop == 2
        and code_ok == 0
    )
    return {"levels": levels, "cli_strict_exit": {"Bi=30": code_stop, "Bi=1": code_ok}, "pass": bool(ok)}


def report(m: dict) -> str:
    g = m["gate"]
    lines = [
        "# Phase 8 gate — interface (auto-generated by experiments/p8_gate.py)",
        "",
        f"git {m['git'][:7] if m.get('git') else '?'}; nothing trained; verification only (no measured data).",
        "",
        "| check | status | evidence |",
        "|---|---|---|",
    ]
    g1, g2, g3, g4, g5 = (g[k] for k in ("G1", "G2", "G3", "G4", "G5"))
    lines.append(
        f"| G1 app launches | **{'PASS' if g1.get('pass') else 'FAIL'}** | "
        + (
            f"healthy after {g1.get('seconds_to_healthy')} s, page HTTP {g1.get('page_status')} |"
            if not g1.get("skipped")
            else "skipped |"
        )
    )
    lines.append(
        f"| G2 headless AppTest | **{'PASS' if g2['pass'] else 'FAIL'}** | {g2['n']} runs (4 cases x 9 "
        f"sections, run buttons pressed), {g2['failed']} with an exception or error |"
    )
    lines.append(
        f"| G3 walkthrough screenshots | **{'PASS' if g3['pass'] else 'FAIL'}** | {g3.get('shots', 0)} shots "
        f"in docs/figures/app, failures {g3.get('failures')} |"
    )
    inv = g4["inverse"]
    lines.append(
        f"| G4 numbers agree with earlier phases | **{'PASS' if g4['pass'] else 'FAIL'}** | inverse errors "
        + ", ".join(f"{k} {v:+.3g} %" for k, v in inv["app_error_pct"].items())
        + f" (Phase 5 gate table: max difference {inv['max_abs_diff_pp']:.2g} pp; intervals cover the truth: "
        f"{inv['covers_truth']}); no source: identifiable {g4['inverse_no_source']['identifiable']} "
        f"(rank {g4['inverse_no_source']['rank']}); observed orders steady central "
        f"{g4['order_steady']['central']:.3f} / upwind1 {g4['order_steady']['upwind1']:.3f}, slab "
        + ", ".join(f"{k} {v:.3f}" for k, v in g4["order_slab"].items())
        + " |"
    )
    lines.append(
        f"| G5 validity guard | **{'PASS' if g5['pass'] else 'FAIL'}** | "
        + "; ".join(f"{k}: {'/'.join(v)}" for k, v in g5["levels"].items())
        + f"; CLI --strict exit codes {g5['cli_strict_exit']} |"
    )
    lines += ["", "## Interface numbers (backend of the app and the CLI)", "", "| quantity | value |", "|---|---|"]
    for k, v in g4.get("pinn", {}).items():
        if "rel_l2_vs_solver" in v:
            lines.append(
                f"| PINN vs solver, {k} | rel. L2 {100 * v['rel_l2_vs_solver']:.3g} %; query "
                f"{v['pinn_s']:.3g} s vs solver {v['solver_s']:.3g} s |"
            )
        else:
            lines.append(
                f"| PINN vs exact, {k} | rel. L2 {', '.join(f'{100 * x:.3g} %' for x in v['rel_l2'])}; "
                f"spread {', '.join(f'{100 * x:.3g} %' for x in v['spread'])} |"
            )
    if "graetz" in g4:
        gz = g4["graetz"]
        lines.append(
            f"| Graetz Re 30, Pr 1, Bi 5 | field rel. L2 {100 * gz['rel_l2_vs_solver']:.3g} %; Nu solver "
            f"{gz['Nu_solver']:.4g}, PINN {gz['Nu_pinn']:.4g} ({100 * gz['Nu_rel_err']:.2g} %; Step 6.7 "
            f"{100 * gz['step_6_7_pinn_error']:.2g} %) |"
        )
    for k, v in g4["cavity"].items():
        lines.append(
            f"| cavity {k} (steady Newton) | psi_min {v['psi_min']:.5f}, Nu at the lid {v['Nu_lid']:.4g}, "
            f"{v['wall_s']:.2g} s |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--skip-launch", action="store_true")
    a = ap.parse_args()
    with Experiment(f"p8_gate_{a.tag}", {"seed": 0}) as ex:
        t0 = time.perf_counter()
        g = {"G1": {"skipped": True, "pass": False} if a.skip_launch else g1_launch()}
        print("G1", g["G1"], flush=True)
        g["G3"] = g3_walkthrough()
        print("G3", g["G3"], flush=True)
        g["G5"] = g5_guard()
        print("G5", g["G5"]["pass"], flush=True)
        g["G4"] = g4_numbers()
        print("G4", g["G4"]["pass"], flush=True)
        g["G2"] = g2_apptest()
        print("G2", g["G2"]["pass"], g["G2"]["failed"], flush=True)
        ex.metrics["gate"] = g
        ex.metrics["pass"] = all(v.get("pass") for v in g.values())
        ex.metrics["wall_s"] = round(time.perf_counter() - t0, 1)
        ex.metrics["git"] = git_hash()
        table = ROOT / "docs/tables/phase8_gate.md"
        table.write_text(report(ex.metrics), encoding="utf-8")
        ex.metrics["table"] = str(table.relative_to(ROOT))
    print(f"gate {'PASS' if ex.metrics['pass'] else 'FAIL'}; {table}")


if __name__ == "__main__":
    main()
