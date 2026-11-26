"""Phase 3 (Step 3.4): baseline 1D PINN validation on Cases A and B across user-defined parameter grids.

    python experiments/p3_pinn1d.py --profile smoke
    python experiments/p3_pinn1d.py --config configs/pinn/examples/steady_only.yaml --tag steady
    python experiments/p3_pinn1d.py --set training.adam.epochs=3000 --set problems.B_slab.parameter.values=[0.5,5,50]
    python experiments/p3_pinn1d.py --describe                      # print resolved settings, train nothing

Per (problem, parameter value): train (Adam -> L-BFGS), evaluate vs the analytic solution (rel L2, L-inf, rel H1), compare
with FD and Chebyshev-spectral solvers on the same truth, record per-component losses and gradient norms. Studies: loss
weights (imbalance), error spectrum (spectral bias). Outputs: results/p3_pinn1d_<tag>/, docs/tables/phase3_failure_modes.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from aerosynapse.thermal.experiments.p3_report import failure_table
from aerosynapse.thermal.numerics import fd1d
from aerosynapse.thermal.numerics import spectral as spc
from aerosynapse.thermal.numerics.mol import solve_mol
from aerosynapse.thermal.numerics.time_integrators import integrate
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import userdata as U
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.metrics import evaluate_model
from aerosynapse.thermal.pinn.networks import resolve_device
from aerosynapse.thermal.pinn.trainer import Trainer, component_grad_norms
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.viz.style import apply_style, save_fig

COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def rel_l2_on_grid(approx: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(approx - ref) / max(np.linalg.norm(ref), 1e-300))


def reference_solvers(prob: P.Problem, p: float, nx: int, nt: int, rs: dict) -> dict:
    """FD (central, CN) and Chebyshev (Radau) solutions of the same problem, errors vs the analytic reference."""
    x = np.linspace(0.0, 1.0, nx)
    t = (prob.T * np.arange(1, nt + 1) / nt) if prob.has_time else None
    ref = prob.reference(x, t, p)
    c = prob.cond
    out: dict = {}
    N_fd, N_sp, dt = rs["fd_N"], rs["spectral_N"], rs["fd_dt"]
    if prob.name == "A_steady":
        kw = dict(a=c["c"], D=1.0 / p, left=fd1d.dirichlet(c["theta_left"]), right=fd1d.dirichlet(c["theta_right"]))
        t0 = time.perf_counter()
        f = fd1d.ConvDiff1D(N_fd, scheme="central", **kw)
        u_fd = np.interp(x, f.x, f.nodal(f.steady(), 0.0))
        out["fd"] = {"rel_l2": rel_l2_on_grid(u_fd, ref), "wall_s": time.perf_counter() - t0}
        t0 = time.perf_counter()
        s = spc.ChebConvDiff1D(N_sp, **kw)
        u_sp = spc.interpolate(s.x, s.nodal(s.steady(), 0.0), x)
        out["spectral"] = {"rel_l2": rel_l2_on_grid(u_sp, ref), "wall_s": time.perf_counter() - t0}
        return out
    if prob.name == "A_pulse":
        g = lambda xb: lambda tt: float(prob.reference(np.array([xb]), np.array([tt]), p)[0, 0])  # noqa: E731
        kw = dict(a=c["pulse"]["c"], D=1.0 / p, left=fd1d.BC(1, 0, g(0.0)), right=fd1d.BC(1, 0, g(1.0)))
        ic = lambda xa: prob.reference(xa, np.array([0.0]), p)[0]  # noqa: E731
    else:
        kw = dict(
            a=0.0,
            D=1.0,
            s=c["m"] ** 2,
            source=lambda X, tt: c["Q"] + 0 * X,
            left=fd1d.neumann(0.0),
            right=fd1d.robin(p),
        )
        ic = lambda xa: c["theta0"] * np.ones_like(xa)  # noqa: E731
    t0 = time.perf_counter()
    f = fd1d.ConvDiff1D(N_fd, scheme="central", **kw)
    y = ic(f.x_active)
    rows, tprev = [], 0.0
    for tk in t:
        y = integrate(f, y, tprev, float(tk), dt, "crank_nicolson", solver=rs.get("fd_solver", "auto"))
        rows.append(np.interp(x, f.x, f.nodal(y, float(tk))))
        tprev = float(tk)
    out["fd"] = {"rel_l2": rel_l2_on_grid(np.array(rows), ref), "wall_s": time.perf_counter() - t0}
    t0 = time.perf_counter()
    s = spc.ChebConvDiff1D(N_sp, **kw)
    sol = solve_mol(s, ic(s.x_active), 0.0, float(t[-1]), "Radau", 1e-9, 1e-11, t_eval=t)
    rows = [spc.interpolate(s.x, s.nodal(sol.y[:, k], float(tk)), x) for k, tk in enumerate(t)]
    out["spectral"] = {"rel_l2": rel_l2_on_grid(np.array(rows), ref), "wall_s": time.perf_counter() - t0}
    return out


def error_spectrum(ev: dict, k_cut: float) -> dict:
    """Energy of the error / reference in the band k > k_cut (cycles per unit length) at the final time (last row)."""
    u, ref, x = ev["_u"], ev["_ref"], ev["_x"]
    if u.ndim == 2:
        u, ref = u[-1], ref[-1]
    e = u - ref
    k = np.fft.rfftfreq(x.size, d=x[1] - x[0])
    Ee, Er = np.abs(np.fft.rfft(e)) ** 2, np.abs(np.fft.rfft(ref - ref.mean())) ** 2
    hi = k > k_cut
    return {
        "k_cut": k_cut,
        "error_high_band_fraction": float(Ee[hi].sum() / max(Ee.sum(), 1e-300)),
        "reference_high_band_fraction": float(Er[hi].sum() / max(Er.sum(), 1e-300)),
        "error_to_reference_ratio_high": float(np.sqrt(Ee[hi].sum() / max(Er[hi].sum(), 1e-300))),
        "error_to_reference_ratio_low": float(np.sqrt(Ee[~hi].sum() / max(Er[~hi].sum(), 1e-300))),
    }


def train_one(
    spec: dict,
    prob: P.Problem,
    p_eval: list[float],
    out_dir: Path,
    tag: str,
    weights: dict | None = None,
    case: dict | None = None,
) -> dict:
    """Train one network; evaluate at each p in p_eval. Returns a record per evaluation value + shared training info.
    ``case`` may name user data (``case['data']``): training data and/or extra evaluation truth (aerosynapse.thermal.pinn.userdata)."""
    dev = resolve_device(spec["device"])
    dtype = torch.float64 if spec["dtype"] == "float64" else torch.float32
    torch.manual_seed(spec["seed"])
    net = build_network(spec["network"], prob.lo, prob.hi).to(device=dev, dtype=dtype)
    ev_cfg = spec["evaluation"]
    p_mid = p_eval[len(p_eval) // 2]
    eval_fn = lambda m: {k: v for k, v in evaluate_model(m, prob, p_mid, 201, 11).items() if not k.startswith("_")}  # noqa: E731
    terms, evals = U.attach(spec, case or {}, prob, dev, dtype, spec["seed"])
    tr = Trainer(net, prob, spec, out_dir, eval_fn=eval_fn, weights=weights, tag=tag, data_terms=terms)
    t0 = time.perf_counter()
    res = tr.run()
    wall = time.perf_counter() - t0
    gnorm = component_grad_norms(net, prob, tr.batch(max(spec["training"]["adam"]["epochs"] - 1, 0)))
    # truth-free validation loss: same loss terms on an INDEPENDENT random collocation set (never trained on)
    s = spec["sampling"]
    vb = prob.sample(s["n_pde"], s["n_ic"], s["n_bc"], "random", 7_777_777 + spec["seed"], dev, dtype)
    v_total, v_comp = tr.fixed_losses(vb, include_data=False)  # physics-only, fixed weights
    user = U.evaluate_all(prob, net, evals)
    recs = []
    for p in p_eval:
        ev = evaluate_model(net, prob, p, ev_cfg["nx"], ev_cfg["nt"])
        rec = {k: v for k, v in ev.items() if not k.startswith("_")}
        rec.update(
            {
                "problem": prob.name,
                "parameter": prob.param_name,
                "value": p,
                "status": res["status"],
                "meets_target": bool(rec["rel_l2"] < ev_cfg["target_rel_l2"]),
                "wall_s": wall,
                "n_params": net.n_params(),
                "grad_norms": gnorm,
                "val_loss": float(v_total.detach()),
                "val_components": {k: float(v.detach()) for k, v in v_comp.items()},
                "arch": spec["network"].get("arch", "mlp"),
                "final_losses": {
                    k: v for k, v in tr.history[-1].items() if k in ("total", "pde", "ic", "bc_left", "bc_right")
                }
                if tr.history
                else {},
                "weights": tr.weights,
                "adam_epochs": spec["training"]["adam"]["epochs"],
                "lbfgs_iterations": spec["training"]["lbfgs"]["iterations"],
                "parametric": prob.parametric,
                "user_data": user,
            }
        )
        rec["grad_imbalance_max_over_min"] = (
            float(max(gnorm.values()) / max(min(v for v in gnorm.values() if v > 0), 1e-300)) if gnorm else None
        )
        recs.append((rec, ev))
    return {"records": recs, "history": tr.history, "net": net}


def redraw(met: dict, saved: dict, rdir: Path, tag: str) -> list[str]:
    """Phase 7: the figures of a saved run, without training. Each network is loaded from its checkpoint and evaluated
    on the evaluation grid (inference only); records and histories (log.csv) are the saved ones."""
    from aerosynapse.thermal.utils import replay as RP

    apply_style()
    ev_cfg = saved["evaluation"]
    runs = []
    for name, pcfg in saved["problems"].items():
        if not pcfg.get("enabled", True):
            continue
        vals = SP.param_values(pcfg["parameter"])
        groups = [(None, vals)] if pcfg.get("parametric", False) else [(v, [v]) for v in vals]
        for v, p_eval in groups:
            prob = P.build_problem(name, pcfg, v)
            d = rdir / "runs" / f"{name}_{'param' if v is None else f'{v:g}'}"
            net = build_network(saved["network"], prob.lo, prob.hi).to(dtype=torch.float64)
            net.load_state_dict(torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)["model"])
            hist = RP.history(d / "log.csv")
            for p in p_eval:
                rec = next(r for r in met["runs"] if r["problem"] == name and r["value"] == p)
                runs.append(
                    {"rec": rec, "ev": evaluate_model(net, prob, p, ev_cfg["nx"], ev_cfg["nt"]), "history": hist}
                )
    return figures(runs, met["studies"], saved, tag, RP.target(rdir.name))


def figures(runs: list[dict], studies: dict, spec: dict, tag: str, ex: Experiment) -> list[str]:
    paths = []
    probs = list(dict.fromkeys(r["rec"]["problem"] for r in runs))
    # solution comparison
    fig, axs = plt.subplots(1, len(probs), figsize=(4.6 * len(probs), 3.4), squeeze=False)
    for ax, name in zip(axs[0], probs, strict=True):
        sel = [r for r in runs if r["rec"]["problem"] == name]
        for i, r in enumerate(sel):
            ev = r["ev"]
            u, ref = (ev["_u"], ev["_ref"]) if ev["_u"].ndim == 1 else (ev["_u"][-1], ev["_ref"][-1])
            ax.plot(
                ev["_x"], ref, "-", color=COLORS[i % 6], lw=1.2, label=f"{r['rec']['parameter']}={r['rec']['value']:g}"
            )
            ax.plot(ev["_x"], u, "--", color="k", lw=0.9)
        ax.set(xlabel="$x^*$", ylabel=r"$\theta$", title=f"{name}: analytic (colour) vs PINN (dashed), final time")
        ax.legend(fontsize=6)
    paths += save_fig(fig, f"p3_solutions_{tag}", ex.dir)
    # error vs parameter
    fig, axs = plt.subplots(1, len(probs), figsize=(4.6 * len(probs), 3.4), squeeze=False)
    for ax, name in zip(axs[0], probs, strict=True):
        sel = sorted((r["rec"] for r in runs if r["rec"]["problem"] == name), key=lambda q: q["value"])
        v = [q["value"] for q in sel]
        ax.loglog(v, [max(q["rel_l2"], 1e-16) for q in sel], "o-", label="PINN")
        for key, m in (("fd", "s--"), ("spectral", "^--")):
            if all(key in q["reference_solvers"] for q in sel):
                ax.loglog(
                    v,
                    [max(q["reference_solvers"][key]["rel_l2"], 1e-16) for q in sel],
                    m,
                    label=key.upper() if key == "fd" else "Chebyshev",
                )
        ax.axhline(spec["evaluation"]["target_rel_l2"], color="k", ls=":", lw=0.8, label="target")
        ax.set(xlabel=sel[0]["parameter"], ylabel="relative $L_2$ error", title=name)
        ax.legend(fontsize=6)
    paths += save_fig(fig, f"p3_error_vs_parameter_{tag}", ex.dir)
    # loss curves
    n = len(runs)
    cols = min(n, 4)
    fig, axs = plt.subplots(-(-n // cols), cols, figsize=(3.4 * cols, 2.7 * -(-n // cols)), squeeze=False)
    for ax, r in zip(axs.ravel(), runs, strict=False):
        h = r["history"]
        step = np.arange(len(h))
        for k, lab in (("pde", "PDE"), ("ic", "IC"), ("bc_left", "BC left"), ("bc_right", "BC right")):
            if k in h[0]:
                ax.semilogy(step, [max(row[k], 1e-30) for row in h], label=lab)
        ax.axvline(sum(1 for row in h if row["phase"] == "adam"), color="k", ls=":", lw=0.8)
        ax.set(
            title=f"{r['rec']['problem']} {r['rec']['parameter']}={r['rec']['value']:g}",
            xlabel="log row (dotted: Adam→L-BFGS)",
        )
        ax.legend(fontsize=5)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    paths += save_fig(fig, f"p3_loss_components_{tag}", ex.dir)
    # failure-mode figure: spectrum + gradient imbalance
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    xs = np.arange(len(runs))
    axs[0].bar(xs - 0.2, [r["rec"]["spectrum"]["reference_high_band_fraction"] for r in runs], 0.4, label="reference")
    axs[0].bar(xs + 0.2, [r["rec"]["spectrum"]["error_high_band_fraction"] for r in runs], 0.4, label="PINN error")
    axs[0].set(
        xticks=xs,
        xticklabels=[f"{r['rec']['problem'][:6]} {r['rec']['value']:g}" for r in runs],
        ylabel=f"energy fraction, k > {spec['studies']['error_spectrum']['k_cut']:g}",
        title="Spectral content: reference vs error",
    )
    axs[0].tick_params(axis="x", rotation=70, labelsize=6)
    axs[0].legend(fontsize=7)
    axs[1].loglog(
        [r["rec"]["grad_imbalance_max_over_min"] for r in runs], [max(r["rec"]["rel_l2"], 1e-16) for r in runs], "o"
    )
    axs[1].set(
        xlabel="gradient-norm imbalance max/min over loss terms",
        ylabel="relative $L_2$ error",
        title="Imbalance vs error",
    )
    paths += save_fig(fig, f"p3_failure_modes_{tag}", ex.dir)
    return [str(p) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--describe", action="store_true")
    ap.add_argument(
        "--table-only",
        action="store_true",
        help="rebuild the failure-mode table and the figures from the saved run of --tag (no training)",
    )
    a = ap.parse_args()
    if a.table_only:
        rdir = ROOT / "results" / f"p3_pinn1d_{a.tag or a.profile}"
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        saved = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
        out = (
            rdir / "failure_modes.md" if (a.config or a.sets) else ROOT / "docs" / "tables" / "phase3_failure_modes.md"
        )
        failure_table(met["runs"], met["studies"], saved, out, a.tag or a.profile, rdir / "runs")
        print(f"wrote {out}")
        for f in redraw(met, saved, rdir, a.tag or a.profile):
            print(f)
        return
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    if a.describe:
        print(SP.describe(spec))
        print(yaml.safe_dump(spec, sort_keys=False))
        return
    tag = a.tag or a.profile
    custom = bool(a.config or a.sets)
    apply_style()
    print(SP.describe(spec), flush=True)
    with Experiment(f"p3_pinn1d_{tag}", spec) as ex:
        runs: list[dict] = []
        for name, pcfg in spec["problems"].items():
            if not pcfg.get("enabled", True):
                continue
            vals = SP.param_values(pcfg["parameter"])
            groups = [(None, vals)] if pcfg.get("parametric", False) else [(v, [v]) for v in vals]
            for v, p_eval in groups:
                prob = P.build_problem(name, pcfg, v)
                label = f"{name}_{'param' if v is None else f'{v:g}'}"
                out = train_one(spec, prob, p_eval, ex.dir / "runs" / label, tag)
                for rec, evd in out["records"]:
                    rec["reference_solvers"] = reference_solvers(
                        prob,
                        rec["value"],
                        spec["evaluation"]["nx"],
                        spec["evaluation"]["nt"],
                        spec["evaluation"]["reference_solvers"],
                    )
                    rec["spectrum"] = (
                        error_spectrum(evd, spec["studies"]["error_spectrum"]["k_cut"])
                        if spec["studies"]["error_spectrum"].get("enabled", True)
                        else {}
                    )
                    runs.append({"rec": rec, "ev": evd, "history": out["history"]})
                    ex.log(
                        problem=name,
                        value=rec["value"],
                        rel_l2=rec["rel_l2"],
                        linf=rec["linf"],
                        rel_h1=rec["rel_h1"],
                        wall_s=rec["wall_s"],
                        status=rec["status"],
                    )
                    print(
                        f"[{name} {rec['parameter']}={rec['value']:g}] relL2={rec['rel_l2']:.3e} Linf={rec['linf']:.3e} H1={rec['rel_h1']:.3e} "
                        f"| FD {rec['reference_solvers']['fd']['rel_l2']:.2e} Cheb {rec['reference_solvers']['spectral']['rel_l2']:.2e} | {rec['wall_s']:.0f}s {rec['status']}",
                        flush=True,
                    )
        studies: dict = {}
        lw = spec["studies"]["loss_weights"]
        if lw.get("enabled", False):
            pcfg = spec["problems"][lw["problem"]]
            sruns = []
            for w in lw["bc_weights"]:
                prob = P.build_problem(lw["problem"], pcfg, lw["parameter_value"])
                base_w = dict(spec["loss"]["weights"])
                base_w["bc"] = float(w)
                out = train_one(
                    spec,
                    prob,
                    [lw["parameter_value"]],
                    ex.dir / "runs" / f"study_weights_{lw['problem']}_bc{w:g}",
                    tag,
                    weights=base_w,
                )
                rec = out["records"][0][0]
                sruns.append(
                    {
                        "bc_weight": float(w),
                        "rel_l2": rec["rel_l2"],
                        "linf": rec["linf"],
                        "rel_h1": rec["rel_h1"],
                        "grad_norms": rec["grad_norms"],
                        "final_losses": rec["final_losses"],
                    }
                )
                print(
                    f"[study bc weight {w:g}] relL2={rec['rel_l2']:.3e} grad_norms={ {k: float(f'{v:.2e}') for k, v in rec['grad_norms'].items()} }",
                    flush=True,
                )
            studies["loss_weights"] = {
                "problem": lw["problem"],
                "parameter": SP.PROBLEMS[lw["problem"]],
                "value": lw["parameter_value"],
                "runs": sruns,
            }
        recs = [r["rec"] for r in runs]
        ex.metrics.update({"runs": recs, "studies": studies, "spec_git": git_hash(), "torch": torch.__version__})
        ex.metrics["n_runs"] = len(recs)
        ex.metrics["n_meet_target"] = int(sum(r["meets_target"] for r in recs))
        ex.metrics["figures"] = figures(runs, studies, spec, tag, ex)
        table = (ex.dir / "failure_modes.md") if custom else (ROOT / "docs" / "tables" / "phase3_failure_modes.md")
        failure_table(recs, studies, spec, table, tag, ex.dir / "runs")
        ex.metrics["failure_table"] = str(table)
    print(
        f"{ex.metrics['n_meet_target']}/{ex.metrics['n_runs']} runs meet the target rel L2 < {spec['evaluation']['target_rel_l2']:g}"
    )


if __name__ == "__main__":
    main()
