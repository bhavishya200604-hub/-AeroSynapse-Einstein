# Figure captions (draft)
<!-- sources: phase3_extras, phase4_generalisation_map, phase5_optimization, phase6_fluids_aero -->

All figures are produced by `python scripts/make_figures.py` from `results/` (deterministic). Each is
available as PDF, SVG and 600-dpi PNG; Figs. A and C–G also as interactive Plotly HTML. Colours use the Okabe–Ito
colour-blind-safe palette; fonts are serif to match the LaTeX body; widths are 85 mm (single column) or 175 mm
(double column).

* **Fig. A (schematic, not data) — layered architecture and research pipeline.** Isometric stack of layers A–F
  (physics definition, classical solver, scientific ML, optimisation, validation engine, workbench), with
  the data flow upward and the validation gate (trust / fallback) returning to the classical solver. The
  mathematics → engineering-decision pipeline (Fig. B) is merged into the left-hand labels.
* **Fig. C — space–time solutions u(x, t)** of u_t + u_x = Pe⁻¹u_xx for Pe = 1, 100 and 1000: exact reference,
  PINN trained with RAD sampling, and absolute error, with the final collocation points overlaid. At Pe = 1 the
  PINN collapses to a nearly constant field (initial-layer failure); relative L2 errors are 0.1767, 0.0341 and
  0.0786.
* **Fig. D — performance surface:** log₁₀ median relative L2 error over (log₁₀ Pe, log₁₀ collocation budget) for
  uniform and RAD sampling (3 seeds). The surfaces nearly coincide and are flat in the budget direction.
* **Fig. E — 3D Pareto front** for the cantilever problem (mass, tip deflection, maximum stress): infeasible
  designs (grey), feasible dominated designs (blue) and the feasible non-dominated set (vermillion), from NSGA-II.
* **Fig. F — 3D aerodynamic view:** NACA 2412 section at α = 4° extruded along the span and coloured by the
  inviscid panel-method pressure coefficient. This is a 2D section with no 3D effects.
* **Fig. G — generalisation map:** log₁₀ median FNO error over (log₁₀ Pe, input bandwidth k_max) with the
  training region (Pe ∈ [5, 50], k_max ≤ 4) outlined. Error rises steeply outside the Pe range in both
  directions and is nearly independent of bandwidth.
* **Supplementary — convergence (Phase 1)**, **cavity centre-lines vs Ghia et al. (1982)**, and **Phase 3 sampler
  comparison** with bootstrap intervals.
