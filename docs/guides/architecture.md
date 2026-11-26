# Architecture guide

```
src/aerosynapse/
  physics/      Layer A  problem.py (PDEProblem, Parameter, dimensionless numbers), bc.py (BC framework),
                         library.py (Level-1 problems + exact solutions), regime.py (flow-regime guards), airfoil.py
  numerics/     Layer B  fd1d, timestep (θ-method, adaptive dt, SSP-RK3), heat, convdiff (FV), burgers, wave,
                         poisson, fem (P1), amr, ns2d (MAC projection), boundary_layer, panel, external (OpenFOAM),
                         verification (order, Richardson, GCI, von Neumann), guards, cards
  ml/           Layer C  nets, autodiff, problems (PINN residuals), pinn (trainer), sampling, weighting,
                         bc_strategies, baselines, datasets, fno, deeponet, rom, hybrid, training, io, airfrans
  uq/                    gp, ensembles, mc, sobol, calibration, cards
  optim/        Layer D  variables, problem, algorithms, pareto, kkt, robust, protocol (8G), surrogate, inverse, demos
  validation/   Layer E  metrics, trust (policy + fallback + reports), resources (estimator)
  ui/           Layer F  Streamlit app (5 workspaces) + background jobs
  experiments/           record (Section 13 fields), runner (end-to-end workflow), parallel, stats, notebooks, studies/
  plugins/               Level-4 demo plugin(s)
  config.py, registry.py, exceptions.py, safety.py, device.py, cli.py
```

Rules: the UI and CLI call only library functions; solvers raise instead of returning invalid fields;
every experiment writes `results/<id>/{record.json, config.yaml, metrics.json, history.csv, git_hash.txt}`;
optimisers evaluate only through `OptimizationProblem.evaluate` (recorded, budgeted); the
`OptimizationStudy` protocol re-verifies every final candidate with the trusted solver.
