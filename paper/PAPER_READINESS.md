# Paper readiness assessment (honest)
<!-- sources: phase3_adaptive_collocation, phase4_operators_uq -->

**Verdict: not ready for a methods journal; suitable at most as a workshop / benchmark note or thesis chapter,
plus a research-software description once CI runs publicly.**

## What is established (evidence in `results/`)
* Verified classical solvers (observed orders, MMS for all BC types) and validation against published laminar
  benchmarks (Ghia cavity, DFG 2D-1 cylinder). This is engineering quality, not novelty.
* A pre-registered, equal-budget, paired-seed comparison of 7 collocation samplers across 4 Péclet numbers with an
  exact reference. The main hypothesis (adaptivity helps at high Pe) was **not supported**: at Pe = 1000 the
  sampler medians range from 0.1658 to 0.1888 against 0.1793 for uniform, and no comparison survives Holm correction.
* Secondary findings: hard IC and causal weighting matter more than sampling at Pe = 1000; untuned Fourier
  features hurt; the Pe = 1 failure is an initial diffusion layer.
* Phase 4: FNO resolution transfer holds; out-of-range Pe extrapolation fails in both directions; a linear POD
  regressor matches the FNO in-distribution (median 0.0005); a learned CG initial guess *increased* iterations.

## What is engineering integration (useful, not novel)
Solvers + PINNs + operators + UQ + optimisation with a trusted-solver protocol, a trust/fallback engine, a
workbench and a claims audit, all in one reproducible package.

## What might be novel (not yet demonstrated)
* An explanation of *why* residual-based sampling gives no gain for smooth transported pulses at equal budget.
  Candidate mechanism: the residual is not spatially localised when the error is a phase/advection error. This
  needs a measured residual-spectrum or NTK analysis.
* The counter-intuitive CG result (a learned initial guess slows Krylov convergence when the right-hand side is
  spectrally sparse) could become a short note with a spectral analysis.

## Required before submission
1. More seeds (≥ 20) and more budgets/architectures; wall-time-equalised comparison.
2. Sharper initial data and non-periodic boundaries (boundary layers), where adaptivity is expected to matter.
3. Community benchmarks (PDEBench) in addition to self-generated data.
4. A mechanism analysis (residual localisation, NTK) to explain the negative result.
5. Compile the LaTeX (no TeX toolchain in the build environment; sources validated by `scripts/check_paper.py`).
6. Verify the remaining author lists in `references.bib` (marked "not verified").
7. Have human experts review it; nothing here has been peer reviewed.
