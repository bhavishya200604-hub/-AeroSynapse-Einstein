# Phase 8 summary — research workbench

**Built:** a Streamlit app (`src/aerosynapse/ui/app.py`) with the five workspaces of spec Section 14:

* **Physics:** equations, domain, parameters with units, BCs with their definitions, IC plot, dimensionless numbers.
* **Simulation:** conv-diff FV / heat / Poisson / NS cavity / panel method, with fields, error fields,
  conservation drift, divergence; stability errors are shown as errors.
* **ML:** background PINN training with progress and cancel, loss curve, prediction and error fields
  against the exact solution, and the Phase 4 generalisation/calibration tables.
* **Optimization:** optimiser choice and budget, the 12-step protocol verdict, history with feasible and
  infeasible points, the best feasible candidate re-verified, recorded surrogate uncertainty, and the
  Pareto front from Phase 5.
* **Research:** experiment records (hypothesis, git commit, versions, reproduce command), JSON/CSV export,
  a one-click background end-to-end workflow, and an HTML report.

All logic is in the library; the UI only calls the public API (`experiments.runner.run_workflow` is shared
with `aerosynapse run`).

**Gate evidence** (`tests/test_ui_cli.py`, Streamlit `AppTest`): every workspace renders headless without
exceptions and shows the disclaimer. The simulation and optimisation actions run; the background ML job
completes. `aerosynapse doctor` and `aerosynapse run configs/workflow_quick.yaml` complete; the run writes
`report.html`. Real screenshots: `docs/screenshots/workbench_*.png` (captured with headless Chromium).

**Limitations:** single-user, in-process thread pool; long jobs do not survive a server restart. Checkpointing
is in the library, not wired to the UI.
