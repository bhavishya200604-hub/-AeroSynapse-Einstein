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
Windows with `constraints.txt`. CI executes on every push; the torch pin was later removed from CI constraints (no CPU wheel for every OS/Python) and run 14 passed on Ubuntu and Windows.

## ADR-0009 Phase 3 design choices
Periodic embedding (exact periodic BC) with a soft IC; σ₀ = 0.1, T = 1; Pe ∈ {1, 10, 100, 1000}. The budget
is equalised by collocation points and iterations, not wall time, because machine load varied. 8 seeds were
used, since that is the minimum for which a Holm-corrected Wilcoxon test can reach α = 0.05 with 6 comparisons.

## ADR-0010 Learning-rate annealing α
The first implementation used α = 0.1. The paper (fetched) recommends α = 0.9, so the default was changed
and Phase 2 was rerun. Both settings diverged; the reported numbers use α = 0.9.

## ADR-0011 GNN deferred
No problem in scope lives on an unstructured graph; a GNN would be added only for fashion (rule 12). The
interface (`registry.MODELS`) exists. Status DEFERRED, P3.

## ADR-0012 Cylinder force accounting
Direct-forcing momentum removal must be counted both after the predictor and after the projection, otherwise
pressure drag is lost (C_D far too low in the superseded run; current error in the Phase 6 summary). The fix was found while validating against
DFG 2D-1; the earlier Phase 6 run is superseded.

## ADR-0013 Invalid geometries in surrogate datasets
Random CST parameters can produce self-intersecting airfoils whose panel solution is meaningless (|C_p| orders of magnitude too large). They are rejected and counted, not learned.

## ADR-0014 AirfRANS
The user asked to download AirfRANS with kagglehub. www.kaggle.com, the official mirror and Hugging Face are
all denied by the environment's network policy. The raw data (many GB) cannot be stored in GitHub anyway (100 MB
file limit). Implemented instead: loader, derived-coefficient table writer (ODbL-1.0 attribution), surrogate
with an explicit convergence criterion, panel-vs-RANS discrepancy, and a Kaggle/Colab notebook that downloads the
data with kagglehub. **Not run**; no AirfRANS numbers are reported.

## ADR-0015 Claims audit scoping
A global audit against all ~1.3e5 artifact values matched almost any number by chance. Documents now declare
`<!-- sources: ... -->`, raw arrays (> 64 elements) are excluded, and a negative-control test proves fabricated
numbers are caught.

## ADR-0016 Resource-estimator model
A single calibration point with time ∝ iterations × points × parameters over-predicted the PINN run several-fold in a superseded run (it
ignored the fixed per-iteration overhead that dominates small networks on CPU). Replaced by an affine model
fitted from two calibration sizes; the measured error is now +6 %.

## ADR-0017 Airfoil validity constraint
"No surface crossing" is thickness > 0 at every interior node (CST uses identical x-nodes on both surfaces).
A first version demanding thickness ≥ 0.005 at every node was infeasible near the closed trailing edge; the
12-step protocol correctly returned REJECT, and the constraint was corrected. The sampled-x check used before
missed crossings near the leading edge and let a degenerate geometry (|C_p| orders of magnitude above physical values) into the C_p dataset.
