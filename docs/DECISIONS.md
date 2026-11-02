# Decision log (ADR format: context · options · decision · consequences)

## ADR-0001 Repository and branch
Context: the user created `bhavishya200604-hub/-AeroSynapse-Einstein`, and the session must develop on
branch `claude/ecstatic-turing-b9c02v`. Decision: all work goes on that branch; `main` keeps the
initial commit until the user merges. Consequence: the user merges (or asks for a PR).

## ADR-0002 Licence
Options: MIT, Apache-2.0. Decision: **Apache-2.0** (explicit patent grant, common in scientific
software). Text fetched from apache.org.

## ADR-0003 Starting research question
See planning document §C. Q1 (adaptive collocation for transient conv-diff across Pe) scored 37/45
and has the best feasibility and validation; its novelty is low. Decision: Q1 is the core study,
framed as a careful equal-budget benchmark, not a new method.

## ADR-0004 Solver set (smallest defensible)
Decision: node-centred FD (heat, Poisson, wave), cell-centred FV (conv-diff, Burgers), P1 FEM
(Poisson cross-check only), MAC projection for NS, vortex-panel method for airfoils. Rejected:
spectral methods (periodic only, would duplicate FV) and a general unstructured FEM (scope).

## ADR-0005 Higher-order conv-diff schemes
Decision: QUICK (linear, 3rd-order face interpolation) and van Leer MUSCL (TVD). The observed order
of both is 2 because cell averages are treated as point values in the flux. This is documented, not
hidden.

## ADR-0006 Wave verification time
The standing-wave error at t = 1 showed spurious 4th order, because the leading phase-error term is
proportional to sin(πct), which vanishes at t = 1. Decision: verify at t = 0.75. Consequence: the
measured order is 2.00, matching theory.

## ADR-0007 CLI framework
Decision: `argparse` from the standard library (no extra dependency). `typer` was considered.

## ADR-0008 Python / CI version
The recorded environment is Python 3.13, numpy 2.5, torch 2.5.1 (CPU wheel from PyPI;
download.pytorch.org is blocked from the build container). CI targets Python 3.13 on Ubuntu and
Windows with `constraints.txt`. CI was configured but not executed in this run.
