# Aircraft Designer

<!-- sources: aircraft -->

Design a whole aircraft or jet, of any shape, and optimise it against a mission. The app covers
drag, thrust, boundary and thermal layers, temperature effects and stability together.

> **RESEARCH USE ONLY.** Validation tier 1 (research simulation). This is a conceptual-design tool.
> It does not replace CFD, a wind tunnel, a stress analysis or flutter clearance, and its results
> are for comparing designs on the same model.

## Starting it

```bash
pip install -e ".[designer]"
```
```bash
aerosynapse designer
```

The app opens in your browser and is reachable only from your own computer. To run a full study
from the command line:

```bash
aerosynapse aircraft --profile standard
```

### Computer requirements

The app needs about 1 GB of free memory. On a 6 GB machine, running four long jobs at the same time
left 0.4 GB free and the app crashed on start. Close long jobs first, or run studies with fewer
workers:

```bash
python scripts/run_aircraft.py --profile standard --workers 2
```

Finished jobs are kept on disk, so a stopped or crashed study continues from where it left off.

## The workspaces

| Workspace | What you do there |
|---|---|
| Design | Load a preset or move any control. The 3D view and performance update as you go. Save and open projects. |
| Mission | Set payload, range, Mach, altitude, cabin floor, span limit, structure and a hot or cold day. |
| Analysis | See drag by source, thrust, boundary and thermal layers, stability and requirements. Run 3D surface pressure. |
| Optimise | Choose how much shape freedom to allow, run a search, compare with the baseline, download an STL. |
| Study results | Compare every body family from a full study. |
| Shape library | Explore the section shapes and curved surfaces, including their Gaussian curvature. |
| Data | See which data slots are filled, upload files, and read the sources of the built-in values. |

## One design space

A tube-and-wing airliner, a blended wing body, a flying wing and a free-form lifting body are
regions of the same 23-variable space, not separate models. So the search is free to leave
conventional layouts behind, and a shape needs no name to be analysed.

| Part | Variables |
|---|---|
| Body length | length, nose and tail fractions and fullness, tractrix tail blend, tail upsweep |
| Cross section | width, height, superformula exponent, Fourier lobes, hyperbolic warp |
| Wing | area, aspect ratio, taper, sweep, thickness, twist, winglet, position |
| Tail | horizontal and vertical volume coefficients |

## Mathematical shapes

Air does not know which geometry a curve came from, so these are a source of shapes rather than a
source of drag reduction. Each study measures whether they helped.

| Geometry | Shapes | Used for |
|---|---|---|
| Euclidean | superellipse, superformula, Fourier outlines, Bezier, Euler spiral, Sears-Haack, von Karman, Joukowski | sections, area laws, blends |
| Hyperbolic | pseudosphere and tractrix (K = -1/a²), Moebius maps of the Poincare disk, kept left-right symmetric | tail closure, section deformation |
| Elliptic | spherical caps (K = +1/R²), inverse stereographic projection | library |
| Minimal surfaces | catenoid | library |
| Random | random Fourier outlines | exploring sections |

## What is modelled

| Effect | Method |
|---|---|
| Atmosphere | ISA to 47 km with hot and cold day offset, Sutherland viscosity |
| Friction drag | flat-plate laws per strip with transition, compressibility and wall temperature (Eckert), roughness (Schlichting) |
| Form drag | Raymer form factors; flat bodies judged by thickness; square bodies +35 %; steep tails penalised |
| Induced drag | non-planar lifting line, Trefftz plane, so winglets count; sweep penalty |
| Wave drag, subsonic | Korn drag divergence, Lock's fourth-power law |
| Wave drag, supersonic | slender-body theory on the real area distribution |
| Thrust | turbojet or turbofan Brayton cycle, polytropic efficiencies, corrected flow off-design |
| Boundary layer | thickness along the body, transition from Reynolds number, surface and sweep crossflow |
| Thermal layer | thickness from the Prandtl number; Reynolds analogy for heating |
| Skin temperature | convection balanced against radiation, nose stagnation (Fay-Riddell) |
| Temperature on the body | expansion, stiffness loss, thermal stress, material limit |
| Temperature in the surroundings | hot and cold day density, viscosity and thrust; wake heating |
| Stability | static margin, short-period damping, directional stability |
| Instability | gust load, flutter, divergence, tail separation margin, turbulent wetted share |
| 3D pressure, verification | source panels on the real body, then integral boundary layers on three lines |

## How much to trust each number

| Quantity | Checked against | Trust |
|---|---|---|
| Atmosphere | ISA tables | high |
| 3D panel speeds | exact sphere and ellipsoid, within 0.3 % | high, for attached inviscid flow |
| Slender-body wave drag | exact Sears-Haack value, within 0.1 % | high for slender bodies; poor for wide flat ones |
| Ideal cycle | closed-form turbojet | high for the cycle; no engine map |
| Total temperature | Concorde's 127 °C limit | high |
| Drag level | public 737-class range, which is itself weak | moderate |
| Differences between conventional and unconventional layouts | published blended wing body studies | moderate to low |
| Mass | calibrated on the baseline | comparative only |
| Flutter | baseline margin is assumed | comparative only |
| Stability derivatives | none | low |

## Limits

* **There is no 3D RANS or LES.** Whole-aircraft CFD does not fit on a laptop. The tool exports STL
  and has a slot to bring external CFD results back in.
* Wide flat bodies at supersonic speed are outside slender-body theory.
* The Mach range is roughly 0.3 to 3.5. Above Mach 5, real-gas effects make the thermal model wrong.
* Engines are not integrated with the airframe: there are no inlet, boundary-layer ingestion or
  nacelle interference effects.
* The cabin model counts floor area with headroom. It does not lay out seats, doors or evacuation.
* Pressurisation of flat bodies is a mass penalty, not a structural analysis.

## Data slots

These are open for upload in a later phase: drag polars, CFD surface results, engine decks, weight
statements, material allowables and measured temperatures. See the Data workspace for each format.
