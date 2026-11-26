# Design workflow: from a basic design to an exported aircraft

One project, eight stages, one app. AeroSynapse Studio runs the whole project in its **Design workflow** window
(menu *Workflow*, key **F9**); the browser app has the same page (*Design workflow*), and the command line runs it too.
Every stage reads the design the previous stage left, writes its own design revision (with a SHA-256 checksum) and
its results, so each step can be inspected, loaded into the editor, rerun or exported.

```bash
python -m aerosynapse.cli workflow new falcon                       # conventional fighter baseline
python -m aerosynapse.cli workflow new myjet --from my_design.json  # any feature-tree body ("input body")
python -m aerosynapse.cli workflow run falcon --profile quick       # every stage that is not done yet
python -m aerosynapse.cli workflow run falcon --start wing          # rerun from a stage onward
python -m aerosynapse.cli workflow status falcon
```

`AeroSynapse.bat --project falcon` opens Studio with the project's workflow window and its newest design loaded.

## The stages

| # | Stage | What it does | Methods (modules) |
|---|---|---|---|
| 1 | Basic design | the starting design: a fighter-family baseline (reference variables) or any input body; geometry, mass properties, quick aerodynamics; the full performance model when the design has design-space variables | `fighter.design`, `cad.kernel`, `cad.analyse`, `fighter.performance` |
| 2 | Design optimisation | CMA-ES over the whole configuration: the fighter design space (weights, drag, turn, climb, gust and extreme states, constraints) — or, for an input body without variables, the main wing planform (span, chord, taper, sweep, position) on the quick analysis | `fighter.optimise`, `shapeopt.cmaes` |
| 3 | Wing optimisation | the **shape optimiser** on the main wing's own airfoil: CST fit (18 numbers), NeuralFoil with Prandtl-Glauert, critical Mach and Lock wave drag at the normal Mach number (simple sweep theory), Ackeret supersonic wave drag; thickness, area, nose radius, rear-spar depth and waviness held at least the baseline's; XFOIL check; the section goes back into the 3D wing as a section table | `shapeopt.geometry`, `shapeopt.aero2d`, `shapeopt.xfoil` |
| 4 | Body optimisation | fuselage cross-sections (half-width and half-height of every interior station) for the **supersonic area rule**: Mach-plane wave drag of the whole configuration at M1.2 and M1.6 plus wetted area; volume ≥ 99 % of the start; the engine and cockpit envelopes must stay inside the body | `cad.analyse._mach_plane`, kernel distance field |
| 5 | Structure and design codes | design-code loads and V-n diagram (MIL-A-8861 fighter, CS-25, CS-23, aerobatic); wing box from the real airfoil sized to zero margin for strength and buckling, then stiffened until flutter and divergence meet the code's margin; 3D frame finite-element model (deflection, twist, modes); fuselage bending; material trade | `structures.codes`, `structures.wing`, `structures.fem`, `structures.panels`, `structures.aeroelastic` |
| 6 | Skin heating with the PINN | **convergence study** of the parametric PINN (Adam budget vs error against the exact series, fitted power law), then the **PINN-optimised skin**: lightest material and gauge per zone with the PINN in the loop, holding thermal stress, temperature limit and the structural minimum (converted to each material) | `thermal.bridge`, `thermal.skin_design`, `thermal.pinn.scaling` |
| 7 | 3D flow simulation | the 3D flow solver on three grids (ratio √2): Richardson extrapolation and the grid-convergence index (Roache, safety factor 1.25); then manoeuvre and supersonic states on the medium grid; the finest cruise surface is painted on the model in Studio | `cfd.runner`, `aeroflow` |
| 8 | Export and documentation | STL, OBJ, glTF (.glb), STEP AP214 (faceted B-rep), DXF three-view (1:1 mm), A3 general-arrangement drawing (PDF, SVG, PNG) with dimensions and title block, bill of materials, mass properties, the report, and a project package (zip) with a SHA-256 manifest | `workflow.export` |

A stage that is rerun marks every later stage *stale*. Profiles: **quick** (about 45 minutes in all, coarse grids)
and **standard** (several hours: larger budgets, finer grids, longer PINN training).

## Studio

* **Workflow > Design workflow (F9)**: the stage list (status colour, time), *Run this stage*, *Run all remaining*,
  *Rerun from here*, *Stop* (ends the whole process tree), *Load stage design* (or double-click a stage),
  *Open report*, *Open project folder*, the profile, and the live log. When a run finishes, the newest stage's
  design is loaded into the editor; after the simulation stage the cruise surface pressure is painted on it.
* **New project**: from the design open in Studio (any body you drew, an *input body*) or from a family baseline.
* **Modify** menu, transformations on the selected feature: taper along x, bend (exact circular arc), shear,
  non-uniform stretch, rotate about a pivot (variable sweep, folding), hollow with gyroid lattice infill, clear.
* **Insert > lattice**: a gyroid, Schwarz-P, diamond or strut lattice block (lightweight cores, heat exchangers).

## Kernel transformations (feature keys)

| key | meaning | check (tests/test_structures.py) |
|---|---|---|
| `stretch [sx, sy, sz]` | non-uniform scale | sphere to ellipsoid volume exact to 0.02 % |
| `shear [dy/dx, dz/dx]` | shears cross-sections along x (sweep a body) | unit cube keeps volume 1 to 0.03 % |
| `taper_rate` | cross-section scale 1 + rate x | tapered cylinder volume to 0.04 % |
| `bend` | curvature (1/m): the local x axis becomes a circular arc, length preserved | bent rod volume to 0.15 % |
| `pivot [x, y, z]` | `rotate` turns about this point | box swung 90° lands exactly on its pivot |
| `infill {type, cell, wall}` | fills a hollow (`shell`) feature with a lattice | gyroid density 3.09 wall / cell to 2 % |
| `type: lattice` | `size`, `cell`, `wall`, `kind` gyroid, schwarz, diamond, struts | as above |

Distances after shear, bend, taper and stretch are scaled bounds (the surface and the inside/outside sign are exact).

## Structural methods and checks

* **Frame finite elements** (12 dof Euler-Bernoulli with torsion, consistent mass): cantilever tip deflection
  and twist exact; first two bending frequencies to 0.1 %; a two-bar truss exact.
* **Flutter**: typical section, Theodorsen's function, k (V-g) method: Hodges & Pierce's case
  (mu = 20, a = -0.2, x_alpha = 0.1, r^2 = 0.24, sigma = 0.4) gives U_F/(b omega_alpha) = 2.18 against their 2.17.
* **Sizing**: covers and spar webs at every station from bisection to zero margin at ultimate load (factor of
  safety 1.5), cover buckling between stringers combined with torsion shear, web shear buckling between ribs;
  then a uniform stiffness factor until the flutter speed is 1.15 V_D (MIL-A-8870) and divergence is clear.
  On the free-form fighter of the earlier study (11.1 t, 9 g) the strength-sized box is 517 kg and the flutter
  requirement raises it to 945 kg: the typical-section model has no sweep relief, so this is conservative.

## Limits

Preliminary-design methods with stated validity ranges, not certification analysis: the section model is 2D with
simple sweep theory; the area rule is linear theory; the typical-section flutter model ignores sweep and the
three-dimensional mode shapes; the STEP file is a faceted solid, not NURBS surfaces.

## Results: project `falcon` (conventional fighter baseline, quick profile, 2026-11-09)

All eight stages ran in order; the folder `projects/falcon` holds every stage's design revision (16 revisions with
SHA-256 checksums), results and figures, and `08_export/report.html` is the full report.

| Stage | Result |
|---|---|
| 1 Basic design | reference fighter, 8 features, take-off mass 13,741 kg, feasible (5 s) |
| 2 Design optimisation | 60 evaluations of the full performance model: objective 1.000 → 0.978; take-off mass 13,741 → 13,506 kg; cruise L/D 8.89 → 9.26; cruise drag −5.6 %; take-off run 726 → 654 m; specific range +6 %; sustained turn −0.6 % (269 s) |
| 3 Wing optimisation | 4.8 % section re-shaped (thickness kept): section drag −5.2 % cruise, −2.6 % sustained turn, −6.0 % loiter; supersonic wave drag of the section −20.5 %; XFOIL agrees: cd −0.5 % at cruise (M capped at 0.6) and −6.4 % at loiter (89 s) |
| 4 Body optimisation | wave D/q −15.2 % at M1.2 and −12.7 % at M1.6, wetted area +0.4 %, volume +7 % (allowed ≥ 99 %), engine and cockpit envelopes inside (39 s) |
| 5 Structure | MIL-A-8861 (n +9 / −3, ultimate 13.5): seven materials sized under the same loads; lightest box carbon/BMI 397 kg (Al-Li 458, 7075 471, Ti-6Al-4V 562, Ti-6242 568, 2024 616 kg); root moment 1,122 kNm per side; tip deflection 11 % of the semispan; modes 11.3 Hz bending, 33.9 Hz torsion; flutter 512 m/s EAS against 460 required (+11 %); divergence 823 m/s; fuselage 2,238 kNm with a 157 kN trimming tail load, skin 4.1 mm (51 s) |
| 6 PINN skin | convergence: relative L2 error 7.9 % → 5.0 % from 250 to 2,000 Adam epochs, a power law with exponent −0.13 (R² 0.93): more Adam epochs give slowly diminishing returns; the design uses the longer-trained network (max error 0.11). PINN-optimised skin: carbon/BMI on radome 3.0 mm, forward fuselage 2.4 mm, centre fuselage 4.1 mm, wing box 6.3 mm, leading edges 1.6 mm, controls 2.5 mm; C/SiC on the aft fuselage (2.6 mm) and nozzle (1.2 mm) (160 s) |
| 7 3D flow | cruise M0.85 on 85k, 170k and 340k cells: CL 0.081 / 0.091 / 0.089, CD 0.0211 / 0.0297 / 0.0185. Convergence is **oscillatory** (700 steps per grid), so no Richardson extrapolation: CL 0.0885 ± 7 %, CD 0.0185 ± 38 %. Manoeuvre (M0.6, 10°): CL 0.43, CD 0.104, no separation; supersonic (M1.6, 2°): CL 0.075, CD 0.024 (fast model: friction 0.0116 + wave 0.0192). The 3D lift at cruise is below the lattice's 0.21 (thin-surface deficit, KI-27 and KI-40) (30 min) |
| 8 Export | STL, OBJ, glTF (142,536 triangles), STEP AP214 (20,080 faces), DXF three-view, A3 drawing at 1:75, bill of materials, mass properties, report; package 28.4 MB, 104 files with SHA-256 manifest (116 s) |

The standard profile (larger budgets, finer grids, 1,500 steps per CFD run) is what turns stage 7 into a monotone
grid study; the quick profile shows the uncertainty honestly instead.
