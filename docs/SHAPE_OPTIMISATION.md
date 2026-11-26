# Whole-design shape optimisation

<!-- sources: shapeopt -->

Give the platform a shape and it runs that shape through laminar, turbulent and extreme flow
states, redesigns it, and compares the new design with the original.

> **RESEARCH USE ONLY.** Validation tier 1 (research simulation). Nothing here is certified, and
> no result here replaces a wind-tunnel test, a stress analysis or a flutter clearance.

## Running it

```bash
pip install -e ".[shapeopt]"
```
```bash
aerosynapse shapeopt --shape naca2412 --profile standard
```
```bash
python scripts/run_shapeopt.py --shape my_airfoil.dat --plan 16.2,7.5,1,0,1,0
```
```bash
python scripts/run_shapeopt.py --shape square --stage topology
```

A shape is `nacaXXXX`, `circle`, `square`, `ellipse:0.5` or a coordinate file (Selig or Lednicer
`.dat`, or x,y `.csv`). Airfoil-like shapes go through the whole-design search; shapes that the
section parameterisation cannot represent go through topology optimisation.

Outputs land in `results/shapeopt/<shape>/`: `optimised_section.dat`, `optimised_wing.stl`,
`optimised_design.json`, `search.json`, `verify.json` and `ns.json`.

## What can change

| Your request | How it is done | Module |
|---|---|---|
| Change the basic shape | 18 CST section numbers | `geometry` |
| Extend or shrink | wing area, aspect ratio, taper | `wing` |
| Rotate | incidence is solved per flight state; the section's zero-lift angle moves with camber | `wing`, `geometry` |
| Twist | tip twist and twist distribution | `wing` |
| Add a shape, continuously | winglet height (0 means none) | `wing` |
| Add a shape, discretely | slat, slotted flap, or both | `multielement` |
| Topology | density field with an adjoint: holes, splits and new bodies are possible | `topology` |

## Flow states

Eleven named states (`scenarios.NAMED`) cover laminar flow at Re = 6e4, laminar separation
bubbles, transition, turbulent cruise, Re = 2e7, high free-stream turbulence (n_crit = 1.5),
leading-edge contamination, dirty high lift, upward and downward gusts, and near-transonic flow.
Random extreme states add to these. They are built from physical quantities (chord, density,
Mach), so impossible combinations do not occur. The wing has seven named flight states plus
random ones, including 2.5 g manoeuvres and rough air.

## Objective

For each state the score is drag relative to the baseline in that same state, so laminar and
turbulent states count equally. With $r$ the ratio over states,

$$\bar J=\tfrac12\left(\mathbb E[r]+\mathrm{CVaR}_{0.9}[r]\right),\qquad
J=\tfrac12\bar J_{2D}+\tfrac12\bar J_{3D}+\rho_1\sum_i\max(g_i,0)+\rho_2\sum_i\max(g_i,0)^2$$

CVaR is the mean of the worst 10 % of states, which stops the optimiser from sacrificing rare
extremes for common conditions. The 13 constraints $g_i$ are thickness, area, nose radius, rear
depth, waviness, surface crossing, stall speed, stall location, flutter, divergence, wing solver
convergence, surrogate confidence and pitching moment.

## Equations used

| Model | Equation or method |
|---|---|
| Section geometry | Kulfan CST with leading-edge modification |
| Search | CMA-ES (rank-one and rank-mu covariance update, cumulative step-size adaptation) with increasing-population restarts |
| Lift and induced drag | Non-planar lifting line, bordered Newton, Trefftz-plane induced drag (Munk) |
| Boundary layers | Thwaites; Drela-Giles e^N envelope; Head entrainment; Ludwieg-Tillmann; Squire-Young |
| Compressibility | Prandtl-Glauert, Karman-Tsien, Lock's fourth-power wave drag |
| Flutter | Theodorsen typical section, k method, checked against a Jones state-space model |
| Gust | Pratt-Truitt load factor, Kussner response |
| Dynamic stall | Kirchhoff lift with a first-order separation-point lag |
| Topology | Brinkman-penalised steady Navier-Stokes, discrete adjoint, density filter, Heaviside projection, projected gradient |
| Unsteady check | Incompressible Navier-Stokes, MAC projection |
| Maximum lift, multi-element | Valarezo-Chin pressure-difference idea, calibrated on the single element |

## Who judges what

The evaluator inside the loop must not be the only judge of the result, because an optimiser finds
the errors in whatever it is given.

| Role | Tool | Trained or tuned on the result? | Strength | Weakness |
|---|---|---|---|---|
| In-loop evaluator | NeuralFoil | yes | accurate attached flow | misses abrupt stall |
| Hold-out judge | NeuralFoil, largest network, unseen states | no | detects overfitting to states | same model family |
| Surrogate check | XFOIL 6.99 (`xfoil`) | no | full viscous-inviscid coupling; 4 % drag error on NASA data | NeuralFoil was trained on it; stops converging before stall |
| Independent judge | panel + integral boundary layer (`vii`) | no | no shared code or data; predicts stall | lift about 7 % high; weak coupling |
| Physics check | unsteady Navier-Stokes (`nsverify`) | no | no turbulence or transition model needed | laminar, 2D, Re in the hundreds |

To use XFOIL, put `xfoil.exe` in `.venv/tools/xfoil/` or set `XFOIL_EXE`, then run:

```bash
python scripts/shapeopt_xfoil.py --shape naca2412
```

## Limits

* **No RANS or LES was run.** Turbulent results come from a surrogate of an integral boundary-layer
  code and from integral methods. Nothing here resolves turbulence.
* **NeuralFoil does not see abrupt stall.** On NACA 0012 at Re = 6e6 the wind-tunnel lift collapses
  near 18 degrees while NeuralFoil keeps rising. Its c_l,max gains are therefore capped in the
  objective.
* **The Mach model is an estimate.** There is no transonic solver, and the suction peak is
  under-resolved at the first boundary-layer station.
* **The wing model is comparative.** Unswept, one section along the span, no fuselage; structural
  mass is scaled to an assumed baseline fraction; flutter uses a strip-theory typical section.
* **Multi-element results are trends.** Inviscid flow plus a calibrated stall rule; confluent
  boundary layers are not modelled.
* **Topology results are low-Re.** Steady laminar flow, Re of order 100. A shape that is optimal at
  Re = 100 is not optimal at Re = 1e6.
* **The dynamic stall time constants are typical magnitudes**, not fitted to data. Use it to compare
  designs only.
