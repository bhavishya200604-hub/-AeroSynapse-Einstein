"""Phase 1: convergence / verification study for all Level-1 solvers -> results/phase1_verification."""

from __future__ import annotations

import time

import numpy as np
from _common import RESULTS, parse_profile

from aerosynapse.experiments import ExperimentRecord, ExperimentRun
from aerosynapse.numerics import amr, burgers, convdiff, fem, heat, poisson, wave
from aerosynapse.numerics import verification as V
from aerosynapse.physics import Dirichlet, Neumann, Periodic, Robin
from aerosynapse.physics import library as L


def main() -> None:
    args = parse_profile(__doc__ or "")
    scale = {"quick": 1, "standard": 2, "full": 4}[args.profile]
    rec = ExperimentRecord(
        experiment_id="phase1_verification",
        research_question="Do the classical solvers reach their formal order of accuracy?",
        hypothesis="Observed orders match theory within 0.15 on the finest pair; conservative schemes conserve "
        "mass to round-off.",
        mathematical_formulation="heat, conv-diff, Burgers, wave, Poisson (FD/FV/FEM); see physics.library",
        seed=0,
        solver_configuration={"profile": args.profile},
        evaluation_metrics=["max-norm error", "observed order", "mass drift", "energy drift"],
        dataset_description="analytical / manufactured solutions (no external data)",
        reproducibility="python scripts/run_phase1.py --profile quick",
        limitations=["1D/2D uniform grids", "Burgers pre-asymptotic at coarse grids"],
    )
    R: dict[str, object] = {}
    with ExperimentRun(rec, RESULTS) as run:
        t0 = time.perf_counter()
        p = L.heat_1d()
        ns = [16 * scale, 32 * scale, 64 * scale, 128 * scale]

        def e_heat(n: int) -> float:
            x, s = heat.solve_heat_1d(p, n, nt=n)
            return float(np.abs(s.u - p.exact(x, p.t_final)).max())

        R["heat1d_cn"] = V.convergence_study(e_heat, ns).as_dict()
        p2 = L.heat_2d()

        def e_heat2(n: int) -> float:
            X, Y, T = heat.solve_heat_2d(p2, n, nt=n)
            return float(np.abs(T - p2.exact(X, Y, p2.t_final)).max())

        R["heat2d_cn"] = V.convergence_study(e_heat2, [8, 16, 32 * scale]).as_dict()
        c = L.convdiff_1d(pe=50.0)
        for sch in ("upwind", "central", "quick", "vanleer"):

            def e_cd(n: int, sch: str = sch) -> float:
                r = convdiff.solve_convdiff_fv(0, 1, n, 1.0, c.p("D"), c.initial_condition, 0.5, scheme=sch, cfl=0.2)
                return float(np.abs(r.u - c.exact(r.x, 0.5)).max())

            R[f"convdiff_fv_{sch}_pe50"] = V.convergence_study(e_cd, [64, 128, 256, 512]).as_dict()
        r = convdiff.solve_convdiff_fv(0, 1, 256, 1.0, c.p("D"), c.initial_condition, 0.5, scheme="vanleer")
        R["convdiff_mass_drift"] = float(max(r.mass) - min(r.mass))
        b = L.burgers_1d(nu=0.1)

        def e_b(n: int) -> float:
            rr = burgers.solve_burgers(0, 2 * np.pi, n, 0.1, b.initial_condition, 0.5, cfl=0.3)
            return float(np.abs(rr.u - b.exact(rr.x, 0.5)).max())

        R["burgers_fv"] = V.convergence_study(e_b, [64, 128, 256, 512], 2 * np.pi).as_dict()
        w = L.wave_1d(t_final=0.75)

        def e_w(n: int) -> float:
            x, u, _ = wave.solve_wave(1.0, n, 1.0, w.initial_condition, 0.75)
            return float(np.abs(u - w.exact(x, 0.75)).max())

        R["wave_leapfrog"] = V.convergence_study(e_w, [16, 32, 64, 128]).as_dict()
        _, _, E = wave.solve_wave(1.0, 128, 1.0, w.initial_condition, 2.0)
        R["wave_energy_rel_drift"] = float((max(E) - min(E)) / E[0])
        pp = L.poisson_2d()

        def e_p(n: int) -> float:
            X, Y, U = poisson.solve_poisson_2d(pp, n)
            return float(np.abs(U - pp.exact(X, Y)).max())

        R["poisson2d_fd"] = V.convergence_study(e_p, [8, 16, 32, 64]).as_dict()
        grad = lambda x, y: (  # noqa: E731
            np.pi * np.cos(np.pi * x) * np.sin(2 * np.pi * y),
            2 * np.pi * np.sin(np.pi * x) * np.cos(2 * np.pi * y),
        )
        errs = [fem.fem_errors(*fem.solve_poisson_fem(n, pp.source, lambda x, y: 0 * x), pp.exact, grad)
                for n in (8, 16, 32, 64)]
        hs = [1 / 8, 1 / 16, 1 / 32, 1 / 64]
        R["poisson2d_fem_L2"] = V.observed_order(hs, [e[0] for e in errs]).as_dict()
        R["poisson2d_fem_H1"] = V.observed_order(hs, [e[1] for e in errs]).as_dict()
        ue = lambda x: np.cos(np.pi * x) + x  # noqa: E731
        f = lambda x: np.pi**2 * np.cos(np.pi * x)  # noqa: E731
        for name, (lb, rb) in {
            "DD": (Dirichlet(1.0), Dirichlet(0.0)),
            "NR": (Neumann(-1.0), Robin(2.0, 1.0, 1.0)),
            "RD": (Robin(1.0, 1.0, 0.0), Dirichlet(0.0)),
        }.items():

            def e1(n: int, lb=lb, rb=rb) -> float:  # type: ignore[no-untyped-def]
                x, u = poisson.solve_poisson_1d(0, 1, n, f, lb, rb)
                return float(np.abs(u - ue(x)).max())

            R[f"poisson1d_{name}"] = V.convergence_study(e1, [16, 32, 64, 128]).as_dict()

        def e_per(n: int) -> float:
            x, u = poisson.solve_poisson_1d(0, 1, n, lambda x: 4 * np.pi**2 * np.cos(2 * np.pi * x), Periodic(), Periodic())
            return float(np.abs(u - np.cos(2 * np.pi * x)).max())

        R["poisson1d_PP"] = V.convergence_study(e_per, [16, 32, 64, 128]).as_dict()
        s = L.convdiff_steady(pe=100.0)
        x, u, hist = amr.adaptive_solve(1.0, 0.01, max_nodes=80)
        xu = np.linspace(0, 1, x.size)
        uu = amr.solve_nonuniform(xu, 1.0, 0.01, 0.0, 1.0)
        R["amr_pe100"] = {"nodes": int(x.size), "err_amr": float(np.abs(u - s.exact(x)).max()),
                          "err_uniform_same_nodes": float(np.abs(uu - s.exact(xu)).max()), "history": hist}
        R["wall_clock_s"] = time.perf_counter() - t0
        run.record.results.update(R)
    for k, v in R.items():
        if isinstance(v, dict) and "pairwise_orders" in v:
            print(f"{k:28s} orders={np.round(v['pairwise_orders'], 3).tolist()}")
    print({k: R[k] for k in ("convdiff_mass_drift", "wave_energy_rel_drift", "amr_pe100", "wall_clock_s")})


if __name__ == "__main__":
    main()
