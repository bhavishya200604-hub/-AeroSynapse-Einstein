"""Step 5.2 diagnostics: why the inverse PINNs miss the parameters.

(1) Physics-residual size of every saved Step 5.2 run: final PDE-residual RMS (in t_hat units of that run) relative to
    the RMS of d theta / d t_hat of the TRUE solution on the same domain (analytic series, finite differences in time).
    A ratio near 1 means the PDE is not enforced at all.
(2) Forward test: the same PINN with the parameters FROZEN at the truth (mode 'forward'), physics only and physics +
    data: can the network represent the true solution at this budget? theta error vs the analytic series on a grid.
(3) Budget test: the joint_w100 variant with 4x the Adam and L-BFGS budget (seed 0).

    python experiments/p5_2_diagnostics.py [--run p5_2_pinn_inverse_default]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import yaml

from aerosynapse.thermal.experiments import p5_2_pinn_inverse as P52
from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import pinn as PI
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks

MOD = "aerosynapse.thermal.experiments.p5_2_diagnostics"


def true_dtheta_rms(spec: dict, t_ref: float, n_r: int = 41, n_t: int = 121) -> float:
    """RMS of d theta / d t_hat of the true solution over r_hat in [0, 1], t_hat in (0, t_max / t_ref]."""
    body = IS.body_of(spec)
    p = {k: float(spec["parameters"][k]) for k in B.PARAMS}
    g = body.groups(p)
    ts = IS.measurement_times(spec)
    t_hat = np.linspace(ts.max() / t_ref / n_t, ts.max() / t_ref, n_t)
    tstar = t_hat * t_ref / g["t_scale"]
    h = 1e-6 * tstar
    r = np.linspace(0.0, 1.0, n_r)
    nt = max(B.truncation_terms(float((tstar - h).min())), 20)
    d = (
        B.theta_analytic(r, tstar + h, g["Bi"], g["Q"], nt, body.n)
        - B.theta_analytic(r, tstar - h, g["Bi"], g["Q"], nt, body.n)
    ) / (2 * h[:, None])
    return float(np.sqrt(np.mean((d * t_ref / g["t_scale"]) ** 2)))


def run_case(spec: dict, mode: str, overrides: dict, seed: int) -> dict:
    s = IS.load_spec(overrides=IS.deep_merge(spec_overrides(spec), overrides))
    data = D.load(s)
    true = {k: float(s["parameters"][k]) for k in B.PARAMS}
    init = true if mode == "forward" else {k: float(s["initial_guess"][k]) for k in s["unknown"]}
    ts, rs = IS.measurement_times(s), np.linspace(0.0, float(s["body"]["L"]), 21)
    var = {"weights": overrides.get("pinn", {}).get("weights", {})}
    r = PI.train(
        IS.pinn_settings(s, var), IS.body_of(s), data, list(s["unknown"]), init, true, mode, seed, {"r": rs, "t": ts}
    )
    exact = (B.temperatures(true, IS.body_of(s), rs, ts, "analytic", IS.n_terms(s)) - s["body"]["T_inf"]) / (
        s["body"]["T0"] - s["body"]["T_inf"]
    )
    r["theta_rel_l2"] = float(np.linalg.norm(r["theta_grid"] - exact) / np.linalg.norm(exact))
    r["theta_max_abs_K"] = float(np.abs(r["theta_grid"] - exact).max() * abs(s["body"]["T0"] - s["body"]["T_inf"]))
    r.pop("theta_grid")
    r["dtheta_rms_true"] = true_dtheta_rms(s, r["t_ref"])
    r.pop("history")
    return r


def spec_overrides(spec: dict) -> dict:
    """The saved run's settings as overrides (everything except the study list)."""
    return {k: v for k, v in spec.items() if k not in ("studies",)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="p5_2_pinn_inverse_default")
    ap.add_argument("--out", default=None, help="results folder name (default p5_2_diagnostics_<tag of --run>)")
    ap.add_argument("--budget-test", action="store_true", help="also the joint_w100 run at 4x the Adam / L-BFGS budget")
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures from the saved diagnostics metrics")
    a = ap.parse_args()
    if a.figures_only:
        out = a.out or "p5_2_diagnostics_" + a.run.replace("p5_2_pinn_inverse_", "")
        odir = ROOT / "results" / out
        figures(json.loads((odir / "metrics.json").read_text(encoding="utf-8")), odir, out)
        return
    rdir = ROOT / "results" / a.run
    met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
    spec = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
    out = a.out or "p5_2_diagnostics_" + a.run.replace("p5_2_pinn_inverse_", "")
    coord_of = {
        v["name"]: v.get("time_coordinate", spec["pinn"].get("time_coordinate", "linear"))
        for v in spec["studies"]["pinn_inverse"]["variants"]
    }
    with Experiment(out, {"run": a.run, "budget_test": a.budget_test}) as ex:
        resid = []
        dcache: dict = {}

        def case_of(cs: str) -> tuple[dict, dict]:
            """(spec, data) of a study case: the configured problem, another body shape, or the optimised design."""
            if cs not in dcache:
                base_shape = spec["body"].get("shape", "slab")
                s = (
                    IS.load_spec(overrides=spec_overrides(spec))
                    if cs == base_shape
                    else P52.case_spec(None, spec_overrides(spec), cs)
                )
                dcache[cs] = (s, D.load(s))
            return dcache[cs]

        cache: dict = {}
        for r in met["records"]:
            key = (r["shape"], r["t_ref"])
            if key not in cache:
                cache[key] = true_dtheta_rms(case_of(r["shape"])[0], r["t_ref"])
            # comparable to dtheta/dt_hat only in linear network time (the sqrt residual is multiplied by 2 sqrt(t_hat))
            lin = coord_of.get(r["variant"], "linear") == "linear"
            ratio = float(np.sqrt(r["losses"]["pde"]) / cache[key]) if lin else None
            resid.append(
                {
                    "shape": r["shape"],
                    "variant": r["variant"],
                    "seed": r["seed"],
                    "pde_rms": float(np.sqrt(r["losses"]["pde"])),
                    "dtheta_rms_true": cache[key],
                    "ratio": ratio,
                }
            )
            ex.log(study="residual", shape=r["shape"], variant=r["variant"], seed=r["seed"], ratio=ratio)
            print(
                f"[residual] {r['shape']} {r['variant']:14s} s{r['seed']}: PDE RMS {np.sqrt(r['losses']['pde']):.3g} / true dθ/dt̂ RMS {cache[key]:.3g} = {ratio if ratio is None else round(ratio, 3)}",
                flush=True,
            )
        # consistency: the EXACT model at each PINN's estimated parameters against the same data. A PINN whose field
        # satisfies the physics with its own parameters would reach the same data misfit (~ sigma); a much larger misfit
        # shows the network fits the data with a field that is not a solution for its parameters.
        consistency = []
        for r in met["records"]:
            cs = r["shape"]
            s, data = case_of(cs)
            nt = max(
                B.truncation_terms(float(np.min(data["t"])) * r["estimates"]["alpha"] / float(s["body"]["L"]) ** 2), 20
            )
            Tm = D.model_at(r["estimates"], s, data["r"], data["t"], nt)
            body = IS.body_of(s)
            g = body.groups(r["estimates"])
            gt = body.groups({k: float(s["parameters"][k]) for k in B.PARAMS})
            rec = {
                "shape": cs,
                "variant": r["variant"],
                "seed": r["seed"],
                "model_rmse_K": float(np.sqrt(np.mean((Tm - data["T"]) ** 2))),
                "pinn_rmse_K": r["data_rmse_K"],
                "sigma_K": float(np.mean(data["sigma"])),
                "Bi": g["Bi"],
                "Q": g["Q"],
                "Bi_true": gt["Bi"],
                "Q_true": gt["Q"],
            }
            consistency.append(rec)
            ex.log(study="consistency", **{k: v for k, v in rec.items()})
            print(
                f"[consistency] {cs} {r['variant']:14s} s{r['seed']}: exact model at the PINN estimates RMSE {rec['model_rmse_K']:.3f} K "
                f"(PINN {rec['pinn_rmse_K']:.3f} K, σ {rec['sigma_K']:.3f}); Bi {g['Bi']:.4g} ({gt['Bi']:.4g}), Q {g['Q']:.3g} ({gt['Q']:.3g})",
                flush=True,
            )
        pn = spec["pinn"]
        cases = [
            ("forward_physics_only", "forward", {"pinn": {"weights": {"data": 0.0}}}),
            (
                "forward_physics_only_linear",
                "forward",
                {"pinn": {"weights": {"data": 0.0}, "time_coordinate": "linear"}},
            ),
            ("forward_with_data", "forward", {"pinn": {"weights": {"data": 1.0}}}),
        ]
        budget = [
            (
                "joint_w100_4x_budget",
                "joint",
                {
                    "pinn": {
                        "weights": {"data": 100.0},
                        "adam": {"epochs": 4 * int(pn["adam"]["epochs"])},
                        "lbfgs": {"iterations": 4 * int(pn["lbfgs"]["iterations"])},
                    }
                },
            ),
        ]
        cases += budget if a.budget_test else []
        outs = run_tasks(
            [(MOD, "run_case", dict(spec=spec, mode=m, overrides=o, seed=0)) for _, m, o in cases],
            spec["execution"],
            lambda i, r, sec: print(
                f"[{cases[i][0]}] θ rel L2 {r['theta_rel_l2']:.3g}, max |ΔT| {r['theta_max_abs_K']:.3g} K, "
                f"estimates {r['estimates']}, data RMSE {r['data_rmse_K']:.3f} K, {r['wall_s']:.0f}s",
                flush=True,
            ),
        )
        tests = []
        for (name, mode, over), r in zip(cases, outs, strict=True):
            tests.append(
                {
                    "name": name,
                    "mode": mode,
                    "overrides": over,
                    **{
                        k: r[k]
                        for k in (
                            "theta_rel_l2",
                            "theta_max_abs_K",
                            "estimates",
                            "data_rmse_K",
                            "wall_s",
                            "losses",
                            "residual",
                            "dtheta_rms_true",
                        )
                    },
                    "time_coordinate": (over.get("pinn") or {}).get(
                        "time_coordinate", spec["pinn"].get("time_coordinate", "linear")
                    ),
                }
            )
        ex.metrics.update({"residuals": resid, "consistency": consistency, "tests": tests, "git": git_hash()})
        ex.metrics["figures"] = figures(ex.metrics, ex.dir, out)
        print(f"wrote {ex.dir / 'metrics.json'}")


def figures(met: dict, odir, out: str) -> list[str]:
    """Derived groups (Bi, Q) and the model-consistency misfit, per variant of the default case."""
    from aerosynapse.thermal.viz import inverse_figs as VF  # noqa: PLC0415
    from aerosynapse.thermal.viz.style import save_fig  # noqa: PLC0415

    VF.style()
    cons = met.get("consistency") or []
    if not cons:
        return []
    case0 = cons[0]["shape"]
    rows = []
    for vn in dict.fromkeys(c["variant"] for c in cons if c["shape"] == case0):
        cs = [c for c in cons if c["shape"] == case0 and c["variant"] == vn]
        rows.append(
            {
                "variant": vn,
                "Bi_ratio": [c["Bi"] / c["Bi_true"] for c in cs],
                "Q_ratio": [c["Q"] / c["Q_true"] for c in cs],
                "pinn_rmse": [c["pinn_rmse_K"] for c in cs],
                "model_rmse": [c["model_rmse_K"] for c in cs],
                "sigma": cs[0]["sigma_K"],
            }
        )
    return [
        str(p)
        for p in save_fig(VF.group_consistency(rows), f"p5_2_consistency_{out.replace('p5_2_diagnostics_', '')}", odir)
    ]


if __name__ == "__main__":
    main()
