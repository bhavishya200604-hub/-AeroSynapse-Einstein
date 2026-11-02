# Phase 6 summary — fluid mechanics and aerospace
<!-- sources: phase6_fluids_aero -->

Source: `results/phase6_fluids_aero/` (`python scripts/run_phase6.py --profile quick`, 184 s).

**Built:** 2D incompressible Navier–Stokes on a MAC grid (Chorin projection) with every fluid BC (no-slip/moving
wall, slip, symmetry, velocity inlet, pressure inlet, pressure outlet, periodic, far-field) and staircase
immersed bodies; a laminar boundary-layer marching solver with a computed Blasius reference; regime and Mach
guards; NACA 4-digit and CST airfoils; the Hess–Smith panel method with Prandtl–Glauert; an OpenFOAM adapter
(templates, pre-flight, import); and the Level-4 Hamiltonian oscillator plugin (experimental, P3).

## Verification (analytic flows)
Couette (periodic, moving wall): max error 4.16e-10. Poiseuille (periodic body force): relative error 0.0159,
0.00392, 0.000978 on 8, 16 and 32 cells, i.e. observed order 2.017 / 2.004. Pressure-driven, velocity-inlet,
slip/symmetry and far-field flows are checked in `tests/test_fluids_aero.py`. The discrete divergence is at round-off.

## Validation against published benchmarks
| Case | Grid | RMS u | RMS v | Reference |
|---|---|---|---|---|
| Cavity Re = 100 | 32² | 0.0013 | 0.0036 | Ghia et al. (1982) |
| Cavity Re = 100 | 64² | 0.0018 | 0.0043 | |
| Cavity Re = 400 | 32² | 0.0154 | 0.0202 | |
| Cavity Re = 400 | 64² | 0.0036 | 0.0044 | |

DFG 2D-1 cylinder (Re = 20), h = 0.01 (220 × 41 cells, staircase cylinder): C_D = 5.4636 vs reference 5.5795,
relative error −0.0208.

Laminar boundary layer at Re_x = 1e4: computed Blasius f''(0) = 0.33206. The marching solver's max deviation from
Blasius is 0.00976 / 0.0102 / 0.0103 for 50 / 100 / 200 points in y. It does not decrease, because the
first-order x-marching error dominates (KI-15).

Panel method: NACA 0012 C_L(4°) = 0.4860 / 0.4844 / 0.4836 on 41 / 81 / 161 nodes; lift slope 6.931 per rad
(thin-airfoil value 6.283; the excess is the expected thickness effect); NACA 2412 at α = 0: C_L = 0.2596,
C_m,c/4 = −0.0546.

**Hypotheses.** H6.1 supported (order ≈ 2). H6.2 supported on 64² (RMS ≤ 0.0044); Re = 400 on 32² narrowly misses
(RMS v 0.0202). H6.3 supported (−2.1 %). H6.4 *borderline*: the max deviation is about 1 %, not below 1 %.
H6.5 supported.

## Aerodynamic optimisation (protocol 8G; inviscid model, so C_L is maximised, not C_L/C_D)
Maximise C_L at α = 4° subject to t/c ≥ 0.12, C_p,min ≥ −2 (heuristic suction-peak proxy) and no surface
crossing, with 60 panel evaluations. GP trust-region surrogate: trusted C_L = 1.3777 (the baseline design had
0.4851); the DE baseline optimiser reached 1.0718. The design re-verified at 201 panel nodes is feasible;
model–trusted difference 3.6e-05. Verdict: ACCEPT. Before ADR-0017 fixed the crossing constraint, the protocol
correctly returned REJECT. Drag is not available from the inviscid model, so the optimum has no meaning for
L/D, stall or viscous effects.

## Surrogate from panel data
POD (rank 8, energy 0.99739) plus GP regression of C_p(x) on the 6 CST parameters, 300 training geometries:
held-out relative L2 median 0.0148, q90 0.0278, max 0.0906. Self-intersecting random geometries are rejected
(ADR-0017).

## Not run / limitations
OpenFOAM is not installed (pre-flight reports it; Mode D not exercised). Only laminar regimes are covered; the
regime guard refuses turbulent queries. Wings, compressible flow beyond Prandtl–Glauert and drag are out of
scope. AirfRANS (RANS drag data) is integrated but not downloaded (KI-13).
