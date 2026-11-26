# Executive summary — AeroSynapse–Einstein
<!-- sources: phase1_verification, phase3_adaptive_collocation, phase4_operators_uq, phase5_optimization, phase6_fluids_aero, phase7_validation_engine -->

**The problem.** Machine-learning surrogates such as physics-informed neural networks (PINNs) and neural
operators promise faster engineering simulation, but they are often presented without the checks an engineer
needs: are they accurate, where do they fail, and when should one fall back to a conventional solver?

**The approach.** I built an open-source Python platform that puts learned models next to *verified* classical
solvers and an *independent* validation engine. Every number is produced by a recorded experiment
(hypothesis written before the run, seed, code version, hardware) and is checked automatically against the
saved results.

**What was built** (runs on a CPU laptop; Windows 11 supported):
* Verified solvers for heat, convection–diffusion, Burgers, wave and Poisson equations. Each reaches its
  theoretical order of accuracy; for example, the second-order central conv-diff scheme measured 1.999.
* A 2D Navier–Stokes solver that reproduces published benchmarks: lid-driven cavity centre-lines within RMS
  0.0044, and cylinder drag 5.4636 against the reference 5.5795.
* PINNs, neural operators (FNO, DeepONet), uncertainty quantification, an optimisation engine with a 12-step
  protocol that re-checks every final design with the trusted solver, a trust/fallback engine, and a
  five-workspace graphical workbench.

**Key verified results.**
* *Core research question (negative result).* At equal compute, adaptive placement of PINN training points did
  **not** improve accuracy for convection-dominated flow: at Péclet number 1000, uniform sampling gave median
  error 0.1793 and the adaptive methods 0.1658 to 0.1888, with no significant difference. Hard initial
  conditions (0.0706) and causal training (0.1134) mattered more.
* *Surrogates pay off only in-distribution.* The FNO was 35.6× faster per query than the solver and broke even
  after 955 queries. Outside its training range its error rose from 0.0016 to 0.0885.
* *Trust engine.* It sent 100 % of out-of-range queries back to the classical solver, cutting the worst-case
  error from 0.23389 to 0.00101; it also falsely rejected 9 % of valid queries.

**Honest limitations.** Laminar 1D/2D problems only; no experimental data, so no claim beyond validation
against published numerical benchmarks; the research contribution is a careful benchmark, not a new method.
PINNs were slower and less accurate than classical solvers on every problem here. The AirfRANS dataset and
OpenFOAM integration are implemented but were not run (network/tooling unavailable).

**Future work.** Mechanism analysis of why adaptive sampling fails (residual localisation, NTK); community
benchmarks (PDEBench, AirfRANS); larger studies on GPU; experimental validation data.
