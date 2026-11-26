"""End-to-end workflow runner: define → simulate → train → validate → optimise → report.

Used by ``aerosynapse run <config.yaml>`` and by the workbench (Research workspace), so the CLI and
the UI execute exactly the same library code.
"""

from __future__ import annotations

import html
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.config import ExperimentConfig, load_config
from aerosynapse.experiments.record import ExperimentRecord, ExperimentRun
from aerosynapse.safety import DISCLAIMER

Progress = Callable[[str, float], None]


def run_workflow(
    cfg: ExperimentConfig,
    root: str | Path = "results",
    progress: Progress | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    import torch

    from aerosynapse.ml.nets import MLP, set_seed
    from aerosynapse.ml.pinn import PINNConfig, predict, train_pinn
    from aerosynapse.ml.problems import ConvDiff1D
    from aerosynapse.numerics.convdiff import solve_convdiff_fv
    from aerosynapse.optim import demos as D
    from aerosynapse.optim.protocol import OptimizationStudy
    from aerosynapse.physics import library as L
    from aerosynapse.validation.metrics import error_distribution, field_metrics

    say = progress or (lambda msg, frac: None)
    stop = cancel or (lambda: False)
    pe = float(cfg.problem.get("pe", 10.0))
    rec = ExperimentRecord(
        experiment_id=cfg.experiment_id,
        research_question=cfg.research_question or "end-to-end workflow check",
        hypothesis=cfg.hypothesis or "the quick workflow completes and every stage writes evidence",
        mathematical_formulation=f"u_t + u_x = u_xx / {pe} (periodic); thermal rod design (optim.demos)",
        seed=cfg.seed,
        model_configuration=cfg.model.model_dump(),
        solver_configuration=cfg.solver.model_dump(),
        training_configuration=cfg.training.model_dump(),
        optimization_configuration=cfg.optimization,
        evaluation_metrics=cfg.metrics,
        dataset_description="synthetic",
        limitations=["quick profile budgets"],
        reproducibility=f"aerosynapse run <config> --profile {cfg.profile}",
    )
    out: dict[str, Any] = {}
    with ExperimentRun(rec, root, config=cfg) as run:
        say("define", 0.05)
        prob = L.convdiff_1d(pe=pe, t_final=0.5)
        out["define"] = {k: str(v) for k, v in prob.summary().items()}
        say("simulate", 0.15)
        t0 = time.perf_counter()
        assert prob.initial_condition is not None and prob.exact is not None
        r = solve_convdiff_fv(0, 1, cfg.solver.nx, 1.0, 1 / pe, prob.initial_condition, 0.5, scheme=cfg.solver.scheme)
        out["simulate"] = {
            "rel_l2_vs_exact": field_metrics(r.u, prob.exact(r.x, 0.5))["rel_l2"],
            "mass_drift": float(max(r.mass) - min(r.mass)),
            "time_s": time.perf_counter() - t0,
        }
        if stop():
            raise RuntimeError("cancelled by user")
        say("train", 0.3)
        set_seed(cfg.seed)
        pp = ConvDiff1D(pe=pe, sigma0=0.05, t_final=0.5)
        m = MLP(
            2,
            1,
            cfg.model.width,
            cfg.model.depth,
            lo=pp.lo,
            hi=pp.hi,
            periodic_dims=(0,),
            seed=cfg.seed,
            fourier_features=cfg.model.fourier_features,
            fourier_scale=cfg.model.fourier_scale,
        )
        tc = cfg.training
        res = train_pinn(
            pp,
            m,
            PINNConfig(
                iterations=tc.iterations,
                n_collocation=tc.n_collocation,
                lr=tc.lr,
                sampler=tc.sampler,
                resample_every=tc.resample_every,
                log_every=max(1, tc.iterations // 20),
            ),
            seed=cfg.seed,
            callback=lambda it, row: say(f"train it {it}", 0.3 + 0.4 * it / tc.iterations),
        )
        for row in res.history:
            run.log(**row)
        torch.save(m.state_dict(), run.dir / "pinn.pt")
        if stop():
            raise RuntimeError("cancelled by user")
        say("validate", 0.72)
        G = pp.eval_grid(101)
        pred, ref = predict(pp, m, G), pp.exact(G)
        val: dict[str, Any] = {**field_metrics(pred, ref), **error_distribution(pred, ref)}
        val["decision"] = "fallback to classical solver" if val["rel_l2"] > 0.05 else "surrogate acceptable"
        out["validate"] = val
        say("optimise", 0.8)
        opt = cfg.optimization or {}
        st = OptimizationStudy(
            "workflow_thermal",
            baseline_design={"m": 1.0, "x_s": 0.8, "material": "aluminium"},
            space=D.thermal_space(),
            objective="min theta_max/theta_ref + 0.1 (m/4)^2",
            constraints=["theta(1)/0.02 - 1 <= 0"],
            constraints_declared=True,
            budget=int(opt.get("budget", 200)),
            stopping="budget or stall",
            model=D.thermal_model,
            trusted=lambda d: D.thermal_model(d, n=1600),
        )
        rep = st.run(opt.get("method", "de"), opt.get("baseline", "ga"), seed=cfg.seed)
        out["optimise"] = {
            "verdict": rep["verdict"],
            "final": rep["method"]["final"],
            "baseline_method_final": rep["baseline_method"]["final"],
            "baseline": rep["baseline"],
        }
        say("report", 0.95)
        out["report_html"] = str(write_html_report(run.dir, cfg.experiment_id, out))
        run.record.results.update(out)
    say("done", 1.0)
    return out


def write_html_report(d: Path, title: str, results: dict[str, Any]) -> Path:
    body = html.escape(json.dumps({k: v for k, v in results.items() if k != "report_html"}, indent=1, default=str))
    page = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title></head><body>"
        f"<p style='background:#fde68a;padding:8px'><b>{html.escape(DISCLAIMER)}</b></p>"
        f"<h1>{html.escape(title)}</h1><pre>{body}</pre></body></html>"
    )
    p = d / "report.html"
    p.write_text(page, encoding="utf-8")
    return p


def run_config(path: str | Path, profile: str | None = None, root: str | Path = "results") -> dict[str, Any]:
    cfg = load_config(path)
    if profile:
        cfg = cfg.model_copy(update={"profile": profile})
    if cfg.profile != "quick":
        scale = {"standard": 3, "full": 10}[cfg.profile]
        cfg.training.iterations *= scale
    out = run_workflow(cfg, root)
    return {k: v for k, v in out.items() if k in ("simulate", "validate", "optimise", "report_html")}


def summarise(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "max": float(a.max())}
