# Fighter design, 3D flow, materials and AeroSynapse Studio

This part of AeroSynapse designs **whole fighter-class configurations** and lets a person edit them like a CAD
model. One design file (a JSON *feature tree*) is shared by four programs:

| Program | What it does | Language |
|---|---|---|
| `aerokernel` | geometry kernel: signed distance of the feature tree, surface mesh, volume, areas, inertia, area distribution, sections | C++ |
| `AeroSynapseStudio` | desktop design editor with logo, feature tree, properties, 3D view, section cuts, undo, export, analysis | C++ (Win32, OpenGL) |
| `aeroflow` | 3D compressible flow solver: immersed boundary, turbulence (LES), wall model, gusts, inflow turbulence | C++ |
| `aerosynapse.fighter` | design space with topology options, every flight situation, materials, optimisation | Python |

Build: the programs compile on first use with MinGW g++ (`python -m aerosynapse.cli studio`), or by hand:

```bash
python scripts/make_logo.py
g++ -std=c++14 -O3 -march=native -mfpmath=sse -msse2 -static -o bin/aerokernel.exe cpp/kernel/aerokernel.cpp
g++ -std=c++14 -O3 -march=native -mfpmath=sse -msse2 -static -o bin/aeroflow.exe cpp/flow/aeroflow.cpp
windres cpp/studio/studio.rc -O coff -o cpp/studio/studio.res.o
g++ -std=c++14 -O2 -static -o bin/AeroSynapseStudio.exe cpp/studio/studio.cpp cpp/studio/studio.res.o -mwindows -lopengl32 -lglu32 -lgdi32 -lcomctl32 -lcomdlg32
```

The compiler on the development machine (32-bit MinGW g++ 6.3) has no `std::thread` and no pthread library
for OpenMP, so `cpp/common/par.hpp` runs loops on a persistent pool of Win32 threads.

## 1. The design: a feature tree on a signed distance field

Every feature is a signed distance function (negative inside). Features are combined in order: **add** (union,
optionally a smooth blend of radius *r*), **cut** (subtraction: slots, holes, notches, ducts, bays) or
**intersect**. Because the solid is a distance field, cuts, holes, hollow cavities and fillets cannot fail the
way mesh booleans do, and the flow solver uses the same field directly as its immersed boundary.

| Feature | Parameters |
|---|---|
| `fuselage` | stations `[x, half width, half height, z centre, superellipse n]`, joined by monotone cubics (n = 2 round, 4 boxy with chines) |
| `wing`, `canard`, `tail`, `fin`, `strake`, `blade` | span, root chord, taper, leading-edge sweep, dihedral (90 = vertical), tip twist, t/c, camber; profile NACA, biconvex, diamond or a table (any section, e.g. an optimised airfoil) |
| primitives | sphere, ellipsoid, box (rounded), cylinder, cone, torus, capsule, duct (swept tube of varying radius along a path), half-space |
| `revolve` | profile `[[x, r], ...]` turned about the local x axis (nacelles, pods, spinners, nose cones) |
| `extrude` | closed outline `[[x, y], ...]` in the local x-y plane, `height` along z (plates, blades, flat fins) |
| `loft` | `sections [{"x", "points": [[y, z], ...]}, ...]`, closed curves with the same point count, blended along x (intakes, transitions) |
| `sheet` | `control`: 4 x 4 Bezier net (16 points) and `thickness`: a curved surface (skins, fairings, winglets) |
| `smooth: true` | on revolve, extrude, loft and duct: the curves pass through Catmull-Rom splines |
| modifiers on any feature | translate, rotate, scale, **mirror** (y = 0), **shell** (hollow wall of a given thickness), **pattern** (linear / rotational copies), **twist_rate**, blend, enable/disable |

Kernel checks against exact shapes (`tests/test_cad_cfd.py`): sphere volume 0.016 % and area 0.007 % off at
h = 0.02 m; torus volume 0.06 %; box with a cylindrical hole: volume exact to the digits shown, area 0.08 %. Curve tools at h = 0.02 m: revolved cylinder 0.01 %, revolved cone 0.03 %, extrusion exact, square
loft 4.666 against 4.6667, flat sheet 0.08 % against the swept-ball volume. `projects/tools_showcase.json` uses
all of them (revolved pods, a sheet fairing and a wing whose section is the optimised NACA 2412 airfoil).

## 2. AeroSynapse Studio (desktop editor)

> **Design workflow.** *Workflow > Design workflow (F9)* runs a whole project in eight stages, from a basic design
> through design, wing and body optimisation, structure and design codes, PINN skin design and 3D flow, to export
> (STEP, glTF, STL, DXF, drawing, bill of materials, report). See [DESIGN_WORKFLOW.md](DESIGN_WORKFLOW.md).
> *Modify* adds taper, bend, shear, stretch, pivot rotation and gyroid lattice infill; *Insert > lattice* adds a
> lattice block; *Optimise > Structure* sizes the wing box of the open design to a design code.

`python -m aerosynapse.cli studio results/fighter/baseline_design.json` (or run `bin/AeroSynapseStudio.exe`).

* **Feature tree** (left): select, reorder, suppress, delete, duplicate, cycle add / cut / intersect.
* **Properties** (right): every parameter of the selected feature, plus its full JSON for stations and paths.
* **Insert menu**: every feature type, and ready-made cuts (slot, hole, hollow cavity).
* **Modify menu**: make hollow (shell), linear pattern, mirror, blend (fillet), twist.
* **View**: orbit / pan / zoom, standard views, wireframe, **section cut** (live contour on a plane, any axis,
  exportable as SVG), three mesh resolutions; the selected feature is outlined in orange.
* **Tools**: mass properties (volume, wetted / frontal / planform / side area, centroid, inertia), design check.
* **Analysis**: quick aerodynamics (vortex lattice, area rule, friction), and one-click 3D flow runs (cruise,
  transonic turn, supersonic, high incidence) whose surface pressure is then painted on the model.
* **File**: open / save the JSON design, export STL or OBJ.
* **Wing from airfoil file** (Insert menu): any Selig or Lednicer `.dat` (for example an airfoil from the 2D
  shape optimiser in `results/shapeopt`) becomes the section table of a new wing.
* **Colour by** (Analysis results): Cp, rms Cp (unsteady load), skin friction or wall temperature.
* **Optimise menu and Optimisation centre**: every optimisation and simulation in the project in one place,
  grouped Airfoil (XFOIL, PINN, adjoint and topology shape optimisation), Topology, Aircraft, 3D flow, Thermal,
  Prediction, Analysis and Apps. Pick a task, edit its arguments (`{design}` is the open design), Run; the log
  streams into the window and Stop ends the whole process tree. The browser app has the same centre as its first
  page.
* Colours: light steel-blue panels, navy buttons with cyan for the primary action, dark navy viewport.
* Undo / redo of every edit; the surface is rebuilt in a background thread so the window never freezes.

The same editing is available in the browser (Designer app, *Fighter editor* workspace).

## 3. 3D flow solver (`aeroflow`)

Compressible Navier-Stokes, finite volumes on a stretched Cartesian grid (uniform near the body, geometric
growth outward); HLLC fluxes with MUSCL (van Albada) reconstruction; SSP Runge-Kutta 3; local time steps for
steady states, a global step for time-accurate runs.

* **Body**: ghost cells inside the body take the state mirrored from an image point in the fluid (no flow
  through the wall, zero normal pressure gradient, adiabatic). Forces: wall pressure summed over the closed
  stair-step surface (a uniform pressure sums to exactly zero) plus wall shear.
* **Turbulence**: WALE subgrid viscosity (large-eddy simulation; Smagorinsky also available); time-averaged
  statistics give turbulent kinetic energy, pressure fluctuation (buffet) and force fluctuation.
* **Wall model**: Spalding's law of the wall gives the wall shear from the velocity at the image point; that
  shear is removed from the first fluid cells, so friction drag and its effect on the flow come out of the
  same solution.
* **Situations**: Mach, incidence, sideslip (full model), altitude and hot / cold day (ISA plus offset),
  vertical gusts (sinusoidal, at the inflow) and synthetic inflow turbulence (random divergence-free Fourier
  modes with a set intensity and length scale).
* **Outputs**: coefficient history, mean and fluctuation of the forces, surface table (Cp, Cp fluctuation,
  friction, wall temperature, y+, reversed-flow flag), field planes (symmetry plane, cross-flow planes),
  the full volume on request (training data), restart files.

Limits, stated in every result file: boundary layers are not resolved (the wall model supplies friction);
separation from smooth surfaces is only approximate on these grids, separation from sharp edges is captured;
the resolved turbulence is limited by the cell size; HLLC can show a carbuncle on strong normal shocks.

## 4. Fighter design space

`aerosynapse.fighter.design` turns one set of named variables into a feature tree. **Body**: length, width,
height, nose length, station of the largest section, tail size, section squareness (chines), nose droop.
**Wing**: area, aspect ratio, taper, sweep, t/c, twist, camber, station, vertical position, dihedral.
**Options** (each one a size; below its threshold the feature is removed, which is a topology change):
canard, strake (LERX), horizontal tail, twin canted or single fins, ventral fins, wing blades (fences),
leading-edge notch (dogtooth cut), through-flow inlet duct (a hollow channel through the body to the
engine), internal bay (hollow cavity). **Propulsion**: thrust-to-weight ratio.

Families: `conventional` (F-16-like layout), `canard_delta`, `twin_tail_lerx`, `tailless_blended`, and
`freeform` (every option switchable).

**Main-body constraints**: the engine (a 5 m by 1.08 m cylinder tapering at the nozzle) and the cockpit must
lie inside the outer body (checked with the kernel's distance field); the fuel the mission needs must fit
(55 % of the free internal volume); the centre of gravity must be placeable within 5 % of the length of the
neutral point by arranging internal equipment; span 14 m and length 20 m at most.

## 5. Performance in every situation

`aerosynapse.fighter.performance` evaluates one design in 15 named situations plus 60 random extreme states:
take-off (sea level and hot-and-high), approach, crosswind approach with 12 degrees of sideslip, subsonic
cruise, climb, sustained and instantaneous turns, supercruise on dry thrust, M1.8 dash, M1.6 cold day at
15 km, maximum dynamic pressure at M1.2 low level, 35 degrees incidence, a 15 m/s gust and severe low-level
turbulence (6 m/s rms).

* Lift, induced drag, neutral point, Cn_beta and Cl_beta from a **vortex lattice of all lifting surfaces
  together** (`fighter/vlm.py`: Trefftz-plane induced drag, Prandtl-Glauert, wing and canard carried to the
  centre line), checked against Helmbold's lift slope, the Prandtl-Glauert trend and the span efficiency of
  a rectangular wing.
* Friction with Eckert's reference temperature, transition and roughness (from the selected materials),
  Raymer form factors, transonic drag rise (Korn, Lock), **supersonic wave drag from Mach-plane cuts of the
  actual 3D shape** (Harris / Eminton-Lord, averaged over roll angles, thin surfaces counted by a soft
  fraction of the distance field), supersonic drag-due-to-lift and partial leading-edge suction.
* High incidence: Polhamus vortex lift per swept surface with vortex breakdown, attached-flow stall delayed
  by strakes and canards, dynamic directional stability with fin blanketing.
* Propulsion: afterburning low-bypass turbofan, Mattingly installed thrust lapse and fuel flow.
* Weights: Raymer fighter group weights scaled by the selected materials, closed on take-off weight with
  the mission fuel; calibrated once so the reference design weighs 8,570 kg empty (F-16 Block 30, public).
* Gust load factor (Pratt), turbulence load (rms), robustness (CVaR of drag over available thrust in random
  extreme states).

Empirical constants quoted from memory are marked in the code. The model ranks shapes consistently; the 3D
solver checks the chosen designs.

## 6. Materials and tolerances

`aerosynapse.aircraft.materials`: eleven materials (aluminium 7075, 2024, Al-Li, Ti-6Al-4V, Ti-6242, Inconel
718, carbon/epoxy IM7/8552, carbon/BMI, C/SiC, steel 300M, glass/epoxy) with temperature curves of strength
and stiffness, service limits, expansion, conductivity, typical thickness tolerance and roughness. Nine
airframe zones (radome, fuselage, wing box, leading edges, controls, aft fuselage, nozzle, gear) each get the
lightest material that survives the zone's hottest temperature with a 20 K margin, by Ashby indices
(strength, panel buckling or fatigue). A Monte Carlo over manufacturing tolerances (thickness, roughness,
steps and gaps, thermal mismatch at joints) gives the spread in mass, drag and stress margin. Two entries
(IM7/8552 and Inconel 718) were checked against data sheets; the rest are typical handbook values, labelled.

## 7. Skin heating with the parametric PINN project

The author's *Adaptive parametric PINNs for convection-diffusion heat transfer* repository is merged as
`aerosynapse.thermal` (all modules, experiments, tests, the THERMAL LAB web app; its configs, data, docs and
paper sources in `projects/thermal_pinn`). `aerosynapse.thermal.bridge` maps an airframe skin panel onto that
project's Case B slab (Biot number from the boundary-layer heat transfer), trains one parametric PINN over
Bi in [0.2, 20] with the project's trainer, checks it against the project's exact series and FD solver, and
gives each zone's inner-face lag and thermal stress. Outside its trained range it falls back to the exact
series and says so (the project's own finding: PINNs do not extrapolate).

## 8. Data on disk and surrogates

`python -m aerosynapse.cli surrogate-predict results/fighter/freeform.design.json` gives an instant estimate from
the network trained on the store and marks each output reliable (held-out R2 at least 0.7) or not.

`aerosynapse.store` writes training data to shards on disk (JSON lines or float32 arrays, manifest with
SHA-256 per shard) and streams it back through a bounded shuffle buffer, so a dataset never has to fit in the
5.9 GB of RAM. Every optimiser evaluation is stored; `scripts/train_surrogate.py` sweeps the freeform space,
trains a network from the shards (checkpoint per epoch, resumable, held-out shards) and compares it with a
nearest-neighbour baseline. CFD runs can save their full volume fields into the same store
(`--save-volume`) for later field surrogates. Set `AEROSYNAPSE_STORE` to put the store on another drive.

## Results

See the report (`results/fighter/report.html`) and `docs/KNOWN_ISSUES.md`. All numbers there come from files
under `results/`.

## Results (standard profile, 2026-11-07)

All numbers below are read from `results/` files; the report (`results/fighter/report.html`) has the full tables.

**Study** (`results/fighter/study.json`): five families, CMA-ES with 420 evaluations each, against the F-16-class
reference calibrated to 8,570 kg empty (objective 1.000):

| Family | Objective | Take-off mass, t | Cruise drag, kN | Dash drag, kN | Sustained turn, deg/s |
|---|---|---|---|---|---|
| Reference | 1.000 | 13.74 | 13.28 | 51.36 | 13.47 |
| Freeform (best) | 0.944 | 11.13 | 10.67 | 46.13 | 12.95 |
| Conventional | 0.958 | 12.25 | 11.18 | 52.32 | 13.60 |
| Twin tail, LERX | 0.992 | 12.83 | 12.05 | 51.12 | 12.61 |
| Canard delta | 1.002 | 13.01 | 12.29 | 54.84 | 11.84 |
| Tailless blended | 1.152 | 16.58 | 14.41 | 83.81 | 15.28 |

The best design is a three-surface layout (canard, strakes, horizontal tail, twin fins canted 19 deg, ventral fins,
a leading-edge notch, an inlet duct and an internal bay). Its gains come with costs the objective allowed: the
search lowered thrust-to-weight to 0.87, so climb rate falls and gust loads rise 18 %, and on 60 hold-out random
extreme states it can sustain 38 % against 47 % for the reference.

**Skin heating** (`results/thermal_skin/skin.json`): the parametric PINN has relative L2 error 2 to 4 % inside
Bi 0.2 to 20 and 7 to 8 % at the ends, and answers 70 times faster than the exact series. In a 10-minute
Mach 1.8 dash the hottest zone (carbon/epoxy control surfaces) keeps a 52 K margin to its service limit, 13 K on a
25 K hotter day; thermal stresses stay below 2 % of the allowables.

**Surrogate** (`results/surrogate/report.json`): trained from shards on disk; held-out R^2 0.97 for climb Ps, 0.94
for C_D0, 0.89 for dash drag, 0.86 for empty mass, 0.78 for take-off weight, poor for outputs that depend on maximum
lift (KI-37).

**3D flow of the designs** (`results/fighter/cfd/summary.json`, 20 cm cells): at the lift each design needs in
cruise, the drag polars from two incidences give 17.5 kN for the reference and 15.5 kN for the optimised design
(-11 %; the fast model said -20 %). At Mach 1.6 its drag coefficient is 29 % lower. But the 3D flow does not
confirm everything: at 25 deg the optimised design has 10 % less lift than the reference (the fast model said 13 %
more), its lift slope is lower, and in the gust its lift fluctuates 68 % less (the fast model said 18 % more).
See KI-40.

**Verification** (`results/cfd_verification/verification.json`): Sod L1 density error 0.0018; inviscid spheroid
C_D 0.0004 (exact 0) and C_p,min -0.150 against -0.178 to -0.187; 10 deg cone at Mach 2 C_p 0.097 against 0.104
(Taylor-Maccoll); wall-model friction 0.64 of Karman-Schoenherr; thin AR 6 wing C_L 0.223 against 0.308 (lattice);
70 deg delta wing C_L 0.350 and 0.686 at 10 and 20 deg against 0.396 and 0.890 (Polhamus); Taylor-Green peak
dissipation 0.0106 at t* 5.8 (with the low-Mach correction) against about 0.0127 at t* 9. Limitations: KI-27 to KI-31.

**Tests**: 1,018 passed, 8 failed, 11 skipped (full suite without slow tests). The failures are 7 layout-specific
tests of the merged project (KI-39) and the intermittent process-pool test (KI-22), which passes alone.
