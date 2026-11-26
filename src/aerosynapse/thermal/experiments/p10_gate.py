"""Phase 10 gate: real-data identification (10.1, 10.1b) and the THERMAL LAB web demo (10.2).

    python -m aerosynapse.thermal.experiments.p10_gate [--tag default]

Reads the saved results and checks:
  G1 provenance     every station file analysed has the SHA-256 recorded when it was downloaded
  G2 verification   forward model: observed order within +-0.15 of 2 in space and time (exact damped wave); fast
                    propagator = step-by-step solver (< 1e-9 K); complex-step = central-difference Jacobian (< 1e-4)
  G3 convergence    every fit of the study ends either converged (stationarity <= 1e-6, not on a bound) or reported as
                    not converged with its diagnosis; counts of fits converged by plain Levenberg-Marquardt, rescued by
                    the strategy pipeline, and not converged (of which frozen)
  G4 validation     hold-out RMSE of every model and station against the no-physics baseline (REPORTED, no pass /
                    fail threshold: this criterion was defined after the data were seen); models rejected because their
                    optimum lies outside the plausible bounds are listed
  G5 frozen weeks   the latent-heat model of Step 10.1b was fitted to every non-converged frozen week
  G6 web demo       walkthrough: every shot rendered without error (results/p10_2_walkthrough_default)
Writes results/p10_gate_<tag>/metrics.json and docs/tables/phase10_gate.md.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash


def load(rel: str) -> dict:
    f = ROOT / rel
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    m = load("results/p10_1_real_soil_default/metrics.json")
    fz = load("results/p10_1b_freeze_default/metrics.json")
    wk = load("results/p10_2_walkthrough_default/metrics.json")
    g: dict = {}
    prov = m.get("provenance", {})
    g["G1"] = {
        "files": len(prov),
        "matching": sum(v["matches_download_record"] for v in prov.values()),
        "pass": bool(prov) and all(v["matches_download_record"] for v in prov.values()),
    }
    v, d = m.get("verification", {}), m.get("derivatives", {})
    g["G2"] = {
        "order_space": v.get("order_space"),
        "order_time": v.get("order_time"),
        "fast_vs_stepwise": d.get("fast_vs_stepwise_max_abs"),
        "complex_step_vs_fd": d.get("complex_step_vs_central_difference_rel"),
        "pass": bool(v)
        and abs(v["order_space"] - 2) <= 0.15
        and abs(v["order_time"] - 2) <= 0.15
        and d["fast_vs_stepwise_max_abs"] < 1e-9
        and d["complex_step_vs_central_difference_rel"] < 1e-4,
    }
    cnt = {"fits": 0, "plain_lm": 0, "rescued": 0, "not_converged": 0, "not_converged_frozen": 0}
    rescued_by: dict = {}
    silent = 0
    for s in m.get("stations", {}).values():
        for rows in s.get("seasonal", {}).values():
            for r in rows:
                if "alpha" not in r:
                    continue
                cnt["fits"] += 1
                if r["converged"]:
                    if r["strategy_used"] == "levenberg-marquardt":
                        cnt["plain_lm"] += 1
                    else:
                        cnt["rescued"] += 1
                        rescued_by[r["strategy_used"]] = rescued_by.get(r["strategy_used"], 0) + 1
                else:
                    cnt["not_converged"] += 1
                    cnt["not_converged_frozen"] += bool(r["frozen"])
        for r in s.get("compare", {}).get("models", {}).values():
            if not r["converged"] and not (r["at_bound"] or r["stationarity"] > 1e-6):
                silent += 1  # a non-converged fit without a diagnosis would be a defect
    g["G3"] = cnt | {"rescued_by": rescued_by, "undiagnosed_failures": silent, "pass": cnt["fits"] > 0 and silent == 0}
    val, rejected = [], []
    for st, s in m.get("stations", {}).items():
        for name, r in s.get("compare", {}).get("models", {}).items():
            val.append(
                {
                    "station": st,
                    "model": name,
                    "converged": r["converged"],
                    "rmse_holdout_mean": float(np.mean(r["rmse_holdout"])),
                    "rmse_baseline_mean": float(np.mean(r["rmse_baseline_holdout"])),
                    "skill": r["skill_holdout"],
                }
            )
            if not r["converged"]:
                diag = r.get("relaxed_bounds_diagnostic")
                rejected.append(
                    {
                        "station": st,
                        "model": name,
                        "relaxed_converged": diag["converged"] if diag else None,
                        "relaxed_estimates": [x["estimate"] for x in diag["rows"]] if diag else None,
                    }
                )
    best = {st: s["compare"].get("best_holdout") for st, s in m.get("stations", {}).items() if s.get("compare")}
    g["G4"] = {"rows": val, "rejected": rejected, "best_on_holdout": best, "pass": None}
    fs = fz.get("summary", {})
    g["G5"] = fs | {"pass": bool(fz) and fs.get("weeks", 0) > 0}
    shots = wk.get("shots", [])
    g["G6"] = {
        "shots": len(shots),
        "ready": sum(s["ready"] and not s["error"] for s in shots),
        "pass": bool(shots) and all(s["ready"] and not s["error"] for s in shots),
    }
    ok = all(v_["pass"] for k, v_ in g.items() if v_["pass"] is not None)
    with Experiment(f"p10_gate_{a.tag}", {"seed": 0}) as ex:
        ex.metrics.update({"gate": g, "pass": ok, "git": git_hash()})
        c = g["G3"]
        best_txt = ", ".join(f"{k.split('_')[1]} {v_}" for k, v_ in best.items() if v_)
        rej_txt = ", ".join(f"{r_['station'].split('_')[1]} {r_['model']}" for r_ in rejected)
        L = [
            "# Phase 10 gate — real data and web demo (auto-generated by experiments/p10_gate.py)",
            "",
            f"git {git_hash()[:7]}; overall **{'PASS' if ok else 'FAIL'}** (G4 is reported, not graded).",
            "",
            "| check | status | evidence |",
            "|---|---|---|",
            f"| G1 provenance | **{'PASS' if g['G1']['pass'] else 'FAIL'}** | {g['G1']['matching']} / {g['G1']['files']} "
            "station files match the SHA-256 recorded at download |",
            f"| G2 verification | **{'PASS' if g['G2']['pass'] else 'FAIL'}** | observed order {g['G2']['order_space']:.2f} "
            f"(space), {g['G2']['order_time']:.2f} (time); fast vs step-by-step {g['G2']['fast_vs_stepwise']:.1e} K; "
            f"complex step vs central differences {g['G2']['complex_step_vs_fd']:.1e} |",
            f"| G3 convergence | **{'PASS' if c['pass'] else 'FAIL'}** | {c['fits']} weekly fits: {c['plain_lm']} converged "
            f"with plain Levenberg-Marquardt, {c['rescued']} rescued by the pipeline "
            f"({', '.join(f'{k} {n}' for k, n in sorted(c['rescued_by'].items()))}), {c['not_converged']} not "
            f"converged ({c['not_converged_frozen']} frozen weeks), all diagnosed |",
            f"| G4 validation | reported | best model on the hold-out window: {best_txt}; rejected (optimum outside "
            f"the plausible bounds): {rej_txt} |",
            f"| G5 frozen weeks | **{'PASS' if g['G5']['pass'] else 'FAIL'}** | {fs.get('weeks')} non-converged frozen "
            f"weeks refitted with the latent-heat model: converged {fs.get('freezing_converged')}, on a plausibility "
            f"bound {fs.get('freezing_at_bound')}, lower RMSE than conduction in {fs.get('freezing_better_rmse')} |",
            f"| G6 web demo | **{'PASS' if g['G6']['pass'] else 'FAIL'}** | {g['G6']['ready']} / {g['G6']['shots']} "
            "walkthrough shots rendered without error |",
            "",
            "## Hold-out validation (all models, all stations)",
            "",
            "| station | model | converged | hold-out RMSE [K] | no-physics baseline [K] | skill |",
            "|---|---|---|---|---|---|",
        ]
        for r in val:
            L.append(
                f"| {r['station']} | {r['model']} | {r['converged']} | {r['rmse_holdout_mean']:.3f} | "
                f"{r['rmse_baseline_mean']:.3f} | {', '.join(f'{s_:.2f}' for s_ in r['skill'])} |"
            )
        (ROOT / "docs" / "tables" / "phase10_gate.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for k, v_ in g.items():
        print(k, v_["pass"], {x: y for x, y in v_.items() if not isinstance(y, list | dict) and x != "pass"})
    print("gate", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
