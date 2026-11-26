# Changelog
All notable changes are documented here (Keep a Changelog; semantic versioning).

## [Unreleased]
### Added
- AirfRANS graph data (the Kaggle download, 1000 airfoil RANS simulations): safe reading without PyTorch Geometric,
  sampling, a point-wise MLP and a PointNet trained on 800 simulations and tested on 100 held-out airfoils at full mesh
  resolution (`aerosynapse.ml.airfrans_graph`); an AirfRANS app in the Labs window, a ribbon button and a centre task.
  A 50-epoch run lowers the PointNet's held-out full-mesh pressure error from 28.6 % to 19.8 %; the Labs pages use the
  better run.
- Labs window: THERMAL LAB, the PINN heat-transfer app, the Designer and the research workbench (39 pages) run inside
  Studio instead of the browser (`aerosynapse.nativelab`: a Streamlit stand-in, a Plotly-to-Matplotlib redraw, a
  worker); native inputs and reports; inputs from the design and assembly (Bi, Pe, Re, Pr, Ri, heat source, aircraft
  sizes); the optimisation centre, the app catalogue and the PINN-lab tool open it.
- Ultra-clear images of the 3D view: drawn off screen at up to 8K with 2 x 2 supersampling, text and lines scaled with
  the picture, PNG, JPEG, TIFF, BMP or PDF, the logo and the design's name, clean file names; analysis result folders use
  it for their 3D pictures; `--hq-image FILE`.
- 2D drawing: SVG and PDF vector output; pictures at three times the window in any format.
- Flow: the grid chosen from the finest gap (gear module); gaps one cell wide between parts bridged as moving wall;
  Refine grid (1.5 times the cells, a side-by-side grid comparison in the report); `--flow-cells N`.
- Fin PINN: a 2D and a 3D analysis for every optimisation setting, 300 dpi figures with the logo and a footer, PDF/SVG
  copies, the naming `FinPINN_<SETTING>_<figure>_<2D|3D>`, settings.json; the results viewer shows a button per setting.
### Fixed
- Flow: the velocity snapshot started at u = (1, 1, 1) lattice units, so cells a moving part had just uncovered showed
  19.96 m/s in the gear example; the views, surface fields and per-axis profiles were affected (not the forces).
- 2D drawing in the manner of AutoCAD: command line with AutoCAD's aliases and coordinate input (x,y, @dx,dy, @len<ang,
  direct distance), layers with hidden and centre line types, object snap, ortho, grid snap, lines, polylines,
  rectangles, circles, arcs, text, linear and radius dimensions, erase, move, copy, mirror, offset, undo/redo; the design
  in three views and an isometric on an A3 sheet with hidden lines (depth buffer) and a title block; sheet-metal flat
  patterns with bend lines; AutoCAD R12 DXF in and out; the largest outline to the 3D sketcher.
- Sheet metal: a bent plate from a bend line clicked in the sketcher; U channel, Z bracket, hat stiffener and L bracket
  presets; the flat pattern in the 2D drawing.
- Sketcher: channel, Z and hat sections; sheet-metal bend lines; chamfer, reverse, DXF import and export, send to the
  2D drawing.
- Fin PINN study (`aerosynapse.thermal.fin_pinn`, Simulate > Fin PINN): data extracted from nine optimisation settings
  on the cooling fin, a parametric PINN and a data-only network trained on it, and a 2D and 3D analysis with one style
  per optimisation setting (errors, residual, energy balance, PINN in the loop) and an interactive 3D page.
- Flow: smooth playback (blended snapshots at 40 frames a second, steady part motion), smoothed surface fields, the
  whole system's X, Y and Z axes with millimetre ticks on the box, per-axis profiles (profile_x/y/z.csv) and their
  summary, invalid-speed cells counted and excluded.
- Results viewer: CSV tables drawn as charts, interactive 3D pages, the summary as a card, thumbnails, chips, and Pop out
  into separate windows; flow, environment, power-flow and interference results save a picture of the 3D view, so
  the viewer is never empty.
- A desktop shortcut to AeroSynapse Studio; accents on the ribbon's active tab.
- Flow simulation: pressure and velocity (with arrows of the flow along the wall) on the cutaway box's walls, each wall
  field with its own colour bar; by default pressure on the x walls, velocity on the y walls and the main field on z.
- Flow simulation from the Environment for any fluid (air, water, oils) and any design (a part design becomes an
  assembly); automatic set-up (bath, current or closed box; hot parts; natural convection alone); x, y and z slices and a
  cutaway-box view; the heat-transfer coefficient on the parts; the overlay on a light panel and the loss-torque polar
  only when its plane faces the eye; `--flow-auto`; the Environment runs the 3D simulation of the whole volume by default,
  and the report sums the whole volume (temperature, speed, turbulence, pressure and liquid, with where the peaks are).
- Flow simulation: a partly filled bath with a free surface (free-surface lattice Boltzmann: liquid, interface, gas)
  and its liquid-fraction field; the no-load loss torque per tooth round a turning gear (flank and tip-circle zones) as
  a polar plot and in the report; the parts' surfaces coloured by friction (wall shear), pressure, temperature,
  turbulence or liquid fraction; the report compared with the Environment analysis' correlations (von Karman disc,
  Changenet and Velex dip lubrication); a gear-splash example.
- Sketcher: NACA 4-digit airfoil, L, T and I sections, keyhole and cross shapes; linear and circular patterns on finish.
- Flow simulation: the whole fluid drawn as a coloured volume (blue-green-red, with a colour bar) of speed, pressure,
  temperature, turbulence or vorticity; temperature carried by the flow from hot parts with buoyancy (natural
  convection) and each part's heat flow, h and Nusselt number; belts and chains move the fluid; examples of gears in oil
  and a belt and chain in water.
- Sketcher: ellipse, regular polygon, rounded rectangle, star, involute gear, arc slot and triangle shapes; Edit menu
  with mirror, offset, fillet, rotate, scale, exact sizes and typed points.
### Fixed
- Motion study: in the belt-and-chain and worm examples the driven wheels' shafts now turn with them.
- Worm gears: a `worm` kernel feature (ZA, starts, hand), a throated bronze worm wheel, the worm in the gear mate
  (ratio, sense, phasing), and worm efficiency, back-driving and self-locking in power flow.
- Cams: disc cams from motion laws (cycloidal, harmonic, 3-4-5, uniform) with pressure-angle, curvature and undercut
  checks; roller and flat-faced followers; a cam mate and the lift-velocity-acceleration panel in the view.
- Environment: tables of heat-transfer coefficients (Re, Ra, Nu, h, UA, Bi, tau) for every part and of friction
  (static and kinetic, handbook pairs, dry and lubricated), rolling resistance and thermal contact conductance for
  every mate.
- Flow simulation of assemblies: a lattice-Boltzmann LES solver (D3Q19, Smagorinsky) with turning parts, for parts in a
  current or a stirred vessel; vorticity slices, vortex cores and particles in the view; forces, Strouhal number,
  power number and turbulence growth in the report; a vortex-street and a mixing-tank example.
- Belts and chains (Mechanisms tab): SPZ V-belt pulleys and ISO 606 roller-chain sprockets (06B to 16B); belt and
  chain drive mates (ratio by pitch diameters or teeth; length, wrap, links and alignment checks), the belt or chain
  drawn running round its wheels; power flow through them (belt efficiency, creep and Euler-Eytelwein slip limit from
  the strand tension; chain efficiency, breaking-load safety and speed ripple); a belt-and-chain example.
- Parts are drawn in their material's colour (aluminium, steels, cast iron, brass, bronze, copper, titanium, polymers,
  rubber, wood...), metals glossy and plastics matte.
- Mechanisms tab: ball bearings (ISO 6000/6200), flanged bushings, keyed shafts and DIN 6885 keys, sleeve couplings,
  rollers, gear hubs and splined bores; a roller (friction drive) mate; an interference check (overlap volumes from
  the parts' exact shapes, contacts, and fit classes: loose, sliding, transition, press); power flow (speeds, torques
  and power through locks, gears, racks and rollers, with bearing, churning, mesh and creep losses, roller slip and
  efficiency, labelled in the view); hydraulic and gear oils; a drive-train example in an oil bath.
- The 3D view: a soft floor shadow (stencil), a metallic sheen and the view's name.
- Helical and straight bevel gears (`"helix"`, `"cone"` on the kernel's gear feature) and a rack feature
  (`"type": "rack"`), all within 0.03 % of their exact volumes; Studio tools for each; the gear mate now handles
  parallel, crossed (bevel) and rack-and-pinion pairs with the turning sense from the contact, tooth phasing and
  checks (module, helix hand, bevel axis angle, centre or pitch-line distance); a *Gear examples* assembly.
- Environment analysis in Studio (`cpp/studio/envphys.inc`, materials in `envmat.inc`): air, fresh water, sea water,
  oil or vacuum at a temperature, with a current, a liquid surface or the floating waterline, gravity and
  lubrication; per part weight, buoyancy (exact on the mesh), pressure, drag, convection and radiation, cooling time,
  steady temperature, thermal expansion and fits; friction per contact mate; motor torque and power; corrosion,
  galvanic, temperature and medium warnings; the liquid and force arrows in the 3D view; materials per component.
  Checked against hand calculations (floating draft, buoyancy, heat loss).
- Gears: an involute spur gear feature in the kernel (`"type": "gear"`: module, teeth, width, pressure angle;
  evaluated in the point's own tooth sector, about a second to mesh, volume within 0.02 % of the exact outline);
  Studio's *Spur gear* tool, a gear mate (ratio from the teeth, teeth phased to mesh, unwrapped over many turns,
  centre-distance and module warnings) and a *Gear train* example.
- Rigid sub-assemblies: an assembly file inserted as one component (any depth), with picking, mates, explode and
  motion as a unit; component files resolved relative to the assembly's folder.
- Assemblies in Studio (`cpp/studio/asm.inc`, *Assembly & DMU* tab): components placed by rigid poses and meshed once;
  mates (coincident, concentric, distance, angle, parallel, perpendicular, lock, insert) picked on faces that Studio
  recognises as planes, holes or shafts; an ISO metric fastener library (hex bolts with cosmetic threads, cap screws,
  nuts, washers, dowel pins, shafts; M3 to M20); insert into hole; Shift/Ctrl drag with the mates holding; an
  animated exploded view; a motion study with motors, key poses, path traces and a timeline; a worked example.
- Flow animations: 3D streamlines through the solver volume with pulses at the local flow speed, half models
  mirrored, visible Mach-coloured particles, a sweeping slice, and recording the view to MP4
  (`scripts/frames_to_mp4.py`). Icons: `scripts/make_asm_icons.py`.
- Changed: sampled surfaces are meshed by dual contouring with sharp features (`cpp/common/mesh.hpp`): hole rims,
  slots and edges are sharp instead of rounded; `AEROSYNAPSE_SURFACE_NETS=1` restores surface nets.
- Fixed: Studio drew the coarse mesh over the close-up mesh on the model's own outer faces (a top face and its
  polygonal hole rims showed through); the Mach legend was unreadable on the light background.
- Studio in the manner of Creo and other CAD programs: a ribbon (File menu, one tab per workbench, tools in named
  groups with large and small buttons, flat glyph icons in `assets/icons_flat`, quick access), a model tree drawn with
  type icons and colour swatches, a light 3D view with a floating view toolbar and an axis triad, and the property panel
  off by default (`cpp/studio/ribbon.inc`, `appearance.inc`, `dims.inc`).
- Parts in colour: by part, by material or single; every vertex of the sampled surfaces coloured by the part that owns
  it; dark edges on every part; Material and Appearance per part from its right-click menu (saved as `material`,
  `appearance`); section views fill each cut part with a darker shade of its colour.
- Click a part in the 3D view to select it; its driving dimensions appear on it, and a click on a value edits it in
  place (wings: semi-span, root and tip chord, sweep, dihedral; primitives: their sizes; any part: overall size).
  Double-click edits the part in the feature pad; right-click opens the part menu.
- Previews of the 3D flow setup (domain, refined core with its cells, free-stream direction, probes, and the grid the
  solver will build) and of the wing optimisation setup (planforms at the ends of each variable range)
  (`cpp/studio/preview.inc`).
- Applications tab: all 129 runnable parts of AeroSynapse-Einstein (command line, research scripts, every thermal
  PINN experiment, the local web apps) in 15 groups, from `assets/app_catalog.json` (`scripts/make_app_catalog.py`);
  an entry opens in the task window with its command and description before it runs.
- Optimise tab: Wing group with the CFD optimiser and *Apply best wing* (the newest `best_design.json`, with its
  study's reliability warning shown).
- Thickness study of 3D CFD lift against the vortex lattice (`scripts/cfd_thickness_study.py`); `--exe` on the mesh
  study and `--out` on `scripts/cfd_verify.py`, so a new solver build is checked without touching earlier results.
- Mesh-independence study of the 3D CFD forces with the grid-convergence index of Celik et al. (2008)
  (`scripts/cfd_mesh_study.py`, tests in `tests/test_mesh_study.py`).
- Flow pictures in the thermal project's paper style (`aerosynapse.sim.flowviz`): for every 3D run, a wing section
  and the centre plane as two panels, temperature rise with tracer particles (inferno) over vorticity (diverging),
  the exact section as the solid, plus MP4 movies of the tracers (drawn once, particles blitted: about 35 s for
  both). Setup: *Flow pictures* (stills and movies, stills, none) and *Wing section for the pictures*.
- Faster CFD: steady runs stop once the window-mean C_L and C_D of two consecutive 600-step windows agree within
  `force_band` (0.5 %) and report that mean; on the draft Falcon cruise 1.85x faster than every step with C_L and C_D
  within 0.1 % (`scripts/cfd_speed_benchmark.py`). Grid sequencing (`init_from`: start from an earlier, coarser
  run's field, trilinear over its fluid cells, mirrored from a half model) is available but gave no gain there
  (KI-70). Setup: *Start from* (free stream, coarse pass first, previous run) and *Stop when forces settle*.
- CFD design optimisation (`aerosynapse.sim.optimise`, `simulate optimise`, Studio tool *Optimise*): Bayesian
  optimisation of the main wing with a Gaussian-process surrogate and expected improvement, every evaluation a
  full-aircraft 3D flow run; sweep, taper, twist, dihedral and span (at constant area), objectives L/D, drag or lift
  with an optional lift floor; per-run wall-cell counts and a grid warning; history, per-variable, surrogate-map,
  planform and flow figures, and `best_design.json`.
- Research audit (`research/`): project audit, baseline results, hardware and storage report, roadmap and
  experiment registry; `C:\AeroSynapse_Data` storage root created (not yet used).
- `scripts/shapeopt_trust_split.py`: splits a shape-optimisation gain into states where NeuralFoil is and is not
  confident.
- The results window lists flow movies after the figures and plays the one picked.
- Studio self-test (`--selftest`): every tool of every workbench run on a design, with a JSON report.
- Demo video (`docs/demo/aerosynapse_demo.mp4`) and its recorder `scripts/demo_video.py`.
- Tools that need a project offer to create one from the design in the editor; flow-field, particle, slice and
  results tools use the newest results on disk instead of asking for a file; SVG section turns the cut on itself;
  Unfold finds the bent plate.
- Exact display surfaces (`cpp/common/tess.hpp`): wings and all wing-like parts, fuselages, revolves, spheres,
  ellipsoids, cylinders and cones tessellated from their definitions with true edge lines; the sampled mesh only for
  parts that cuts reach or that are deformed; automatic coarser sampling for big designs (32-bit memory).
- Feature pad: every tool opens a pad with all the feature's parameters, previewed live in the 3D view; OK keeps,
  Cancel restores the design; double-click a tree item to reopen it; place a feature at a point picked on the model.
- Hole tool: diameter, depth or through-all, entry point typed or picked on the model, direction along any axis or
  any vector (picking uses the surface normal); stored in the feature as `hole` and compiled to a cylinder cut.
- Sharp view: crease lines on every mesh; after zooming, the visible region is re-meshed at the screen's
  resolution (about 5 million cells at most); display-only minimum thickness for wings and thin parts
  (`display_h`); selection shown as a tint.
- Turbulence and flow instability analysis (`aerosynapse.sim.turbulence`, `simulate turbulence`): free-stream
  turbulence and laminar-turbulent transition on the wing section (Thwaites, e^N envelope, Mack's critical N),
  transition against Tu and its friction drag; Rayleigh stability of the wake (Chebyshev collocation, sinuous and
  varicose modes, checked against Michalke's tanh mixing layer); turbulent kinetic energy, r.m.s. velocity and
  vorticity in 3D; probe spectra with the -5/3 line, measured inflow intensity and integral scales.
- Flow solver: probes (u, v, w, p every step), turbulence statistics (k, p_rms, u_rms, vorticity) in the saved 3D
  volume, inflow turbulence seeded in the whole field and forced in a slab upstream of the aircraft.
- Setup forms for every analysis (`aerosynapse.sim.setup`, `assets/sim_setup_schema.json`, `--setup FILE` on
  `simulate` and `engineer`): 3D flow (flight condition, turbulence, gusts, physical model, grid, time stepping,
  outputs, probes), turbulence, analytic, PINN (per-zone materials and thicknesses), structure, laminate, clash;
  units, ranges, help, remembered values, save and load.
- Preferences (`settings.json`): plain or dark charts (plain by default: white paper, thin lines, muted colours),
  calm or vivid 3D display, fixed or auto-hide tool bar. The tool bar slides in and out.
- Property editor: every parameter shows its unit; choice parameters are drop-down lists; the info card explains the
  parameter being edited.
- Full-3D flow analysis (`aerosynapse.sim.flow3d`, `simulate cfd --case cruise|transonic|supersonic|high_alpha|sideslip|gust
  --quality live|draft|standard|fine`, `simulate flow3d FOLDER`): whole-aircraft runs (no symmetry plane) that save the
  3D volume; streamlines all around the body, Q-criterion vortex cores, the sonic surface, convergence, surface
  pressure, wing sections, field slices and 3D renders; force-settling measure. In Studio: volume slices (Slice axis,
  Ctrl+wheel), 3D particles, animated streamlines, vortex and sonic surfaces, turbo colour map, live simulation.
- Studio workspaces: 14 workbenches (Start, Sketcher, Part Design, Surfaces & Math, Wing & Aero, Sheet & Composite,
  Assembly & DMU, Transform, View & Analysis, Simulate, Optimise, Knowledge, Workflow, Drafting & Export) with 171
  icons; click the active workbench to hide its tools; F2/F3/F4/F11 and bar buttons hide the tree, properties,
  palette, or everything (full 3D view); info card with icon, description and *What to do* for every feature; hover
  cards on the palette.
- Start page with a blank design, a new project in a new folder (`workflow new --where`), and 22 templates: jet
  fighters, airplanes, airfoil wings, showcase designs (`scripts/make_templates.py`, `assets/templates`).
- Sketcher on the XY, XZ or YZ plane with an offset; polyline / spline, rectangle, circle and slot; Pad, Pocket,
  Shaft, Groove and Loft; live segment length and angle, Shift for horizontal / vertical.
- Kernel: `bentplate` (sheet metal with bend radius), `chamfer` (45-degree bevel at joins) and `draft` (extrude) modifiers.
- Engineering checks (`aerosynapse.engineering`, CLI `engineer clash|rules|composites|skeleton|rflp`): clash analysis
  and space reservations, knowledgeware rules with a safe evaluator, classical lamination theory, skeleton
  parameters, RFLP traceability. Studio: unfold, kinematics of pivot joints, pipe and harness routes, zebra and
  curvature views, 3D dimensions.
- Overlays in the 3D view for the analytic (lifting-line loads, Sears-Haack body), PINN (skin zones) and structural
  (deflected wing, cover stress) simulations.
- Studio redesign: splash screen; scheme bar with nine colour-coded schemes; tool palette of 117 generated icons
  (`scripts/make_icons.py`, table `cpp/studio/tools.inc`); GDI+ gradient buttons with depth; stage cards in the
  workflow window; deeper viewport gradient with a horizon glow and fading grid.
- Sketcher (revolve, extrude, loft from mouse-drawn profiles with snap and spline smoothing).
- Kernel math features: `implicit` (any f(x,y,z) <= 0, compiled expressions), `superellipsoid`, `sweep` (tube along a
  parametric curve).
- Simulation suite `aerosynapse.sim` and CLI `simulate analytic|pinn|structure|all`; results viewer in Studio; 3D flow
  field display (Mach planes) and animated particles in the 3D view.
- Reports as PDF, Word and text (`workflow report`, export stage, Studio buttons); CLI `export-design` (STEP, glTF,
  STL, OBJ, DXF, A3 drawing of any design).
- Design workflow (`aerosynapse.workflow`, CLI `workflow new|run|status`, Studio *Workflow* menu and window, web
  page): basic design, design optimisation, wing optimisation (shape optimiser on the wing's own airfoil, XFOIL
  check), body optimisation (area rule with volume and envelope constraints), structure and design codes, PINN skin
  design with a convergence study, 3D flow with a grid-convergence study, export (STL, OBJ, glTF, STEP AP214, DXF,
  A3 drawing, bill of materials, mass properties, report, checksummed package). Project folders with revisions and
  SHA-256 checksums; stale-stage tracking.
- Structures package (`aerosynapse.structures`): design codes (MIL-A-8861, CS-25, CS-23, aerobatic) and V-n
  diagram, 3D frame and truss finite elements with modes, panel buckling and interaction, Bredt-Batho sections,
  Theodorsen flutter (k method) and divergence, wing box sizing to zero margin with stiffness sizing, fuselage
  bending, material trade by sizing.
- Kernel transformations: stretch, shear, taper, exact circular bend, rotation about a pivot; lattice features
  (gyroid, Schwarz P, diamond, struts) and lattice infill of hollow features; Studio Modify-menu entries.
- `thermal.skin_design`: PINN convergence study and PINN-in-the-loop skin optimisation per zone.
- `AeroSynapse.bat` launcher; `scripts/package_release.py` builds a checksummed distributable zip.
- Curve and surface tools in the kernel: `revolve`, `extrude`, `loft`, `sheet` (Bezier patch with thickness) and
  spline smoothing; Studio Insert > Wing from airfoil file; colour by Cp, rms Cp, skin friction or temperature;
  new light/navy colour scheme; Optimise menu and Optimisation centre (18 tasks across airfoil, topology,
  aircraft, 3D flow, thermal, prediction) in Studio and as the first page of the web app; CLI `surrogate-predict`,
  `topology`, `xfoil`, `thermal-lab`; tests for the curve tools and surrogate reliability labels.
- Fighter design tools: C++ geometry kernel (signed-distance feature tree: add, cut, intersect, blend, mirror, hollow
  shells, patterns, twist), AeroSynapse Studio desktop editor (Win32 + OpenGL, logo, feature tree, properties,
  section cuts, undo, STL/OBJ/SVG export, mass properties, one-click analysis and CFD), `aeroflow` 3D compressible
  solver (HLLC-MUSCL, immersed boundaries with two-sided thin surfaces, WALE large-eddy model, wall model, gusts,
  inflow turbulence, low-Mach correction for time-accurate runs), fighter design space with topology options,
  multi-surface vortex lattice, area rule from 3D Mach-plane cuts, every flight situation, materials with
  temperature-dependent allowables and tolerance Monte Carlo, skin heating with the merged PINN project, sharded
  on-disk data store with streamed training; CLI `fighter`, `studio`, `cad-analyse`, `cfd`, `cfd-verify`,
  `thermal`, `surrogate`.
- The adaptive parametric PINN project for convection-diffusion heat transfer merged whole as `aerosynapse.thermal`
  (`scripts/merge_thermal_pinn.py`; home folder `projects/thermal_pinn`; tests in `tests/thermal`).
- Phase 0: package scaffold (`src/aerosynapse`), pydantic configs, experiment records (Section 13
  fields), plugin registry, exception hierarchy, device/hardware report, safety ladder, CLI `doctor`,
  CI/pre-commit/Docker/devcontainer, planning document A–N.
- Phase 1: physics definition API, BC framework (Dirichlet/Neumann/Robin/periodic/initial/interface),
  Level-1 problem library with exact solutions; FD heat 1D/2D, FV conv-diff (upwind/central/QUICK/
  van Leer), Burgers FV, wave leapfrog, Poisson FD 1D/2D, P1 FEM, adaptive dt, 1D AMR, verification
  toolkit (order, Richardson, GCI, von Neumann), solver cards.
- Phase 2: PINN framework (MLP, Fourier features, periodic embedding, hard/penalty BCs, causal weighting,
  annealing, L-BFGS), 7 collocation samplers, domain decomposition, SDF hard constraints, supervised baseline,
  validation metrics.
- Phase 3: pre-registered equal-budget sampler study (7 samplers × 4 Pe × 8 seeds), ablations, post-hoc Holm,
  budget sweep, field snapshots.
- Phase 4: dataset generator with cards and leakage checks; FNO 1D/2D, DeepONet, POD, CNN/MLP baselines;
  mass-conserving FNO; neural correction, CG initial guess, Kalman data assimilation; GP, deep ensembles, MC,
  Sobol, calibration; generalisation map.
- Phase 5: design space, recorded/budgeted problems, 11 optimisers + capability table, KKT diagnostics, robust
  measures, 12-step protocol, GP trust-region surrogate, discrete adjoint and PINN inverse routes, engineering demos.
- Phase 6: MAC projection NS solver with 8 fluid BC types and immersed bodies, Blasius/BL marching, regime and
  Mach guards, NACA/CST airfoils, Hess–Smith panel method, OpenFOAM adapter, oscillator plugin, airfoil optimisation.
- Phase 7: trust policy + fallback + trust reports, affine resource estimator, notebooks, dataset catalogue.
- Phase 8: Streamlit workbench (5 workspaces, background jobs), end-to-end workflow runner (`aerosynapse run`).
- Phase 9: figures A–G (PDF/SVG/600-dpi PNG/HTML), paper package (LaTeX, generated tables, verified bib, claims map,
  readiness), source-scoped claims audit with negative control, reproduce_all, quality gate, docs site.
- AirfRANS integration (loader, derived table, surrogate, Kaggle notebook) — not run (network policy).
- Whole-design shape optimisation (`aerosynapse.shapeopt`, `aerosynapse shapeopt`): robust CVaR search over section and
  wing with CMA-ES, non-planar lifting line with wing-box sizing, density topology optimisation with a discrete adjoint,
  multi-element high-lift stage, independent panel + boundary-layer solver validated on NASA TM 4074 data, flutter and
  dynamic stall models, unsteady Navier-Stokes verification. NeuralFoil is an optional dependency (`[shapeopt]`).
- XFOIL adapter (`aerosynapse.shapeopt.xfoil`, `scripts/shapeopt_xfoil.py`): headless runs with a time limit, used to judge optimised sections on hold-out states.
- Aircraft Designer (`aerosynapse.aircraft`, `aerosynapse designer`, `aerosynapse aircraft`): one design space for any body,
  from tube and wing to free-form; Euclidean and non-Euclidean shape library with curvature checks; ISA atmosphere; friction,
  form, induced and wave drag; turbofan cycle; boundary and thermal layers, skin temperature and material effects; stability
  and instability margins; 3D panel verification; data slots for later uploads; Streamlit app with projects and STL export.

### Fixed
- 3D flow (aeroflow), thin surfaces: sheets thinner than a quarter cell are no longer missed (KI-78); no slot lets air
  through a thin section between two cell centres (KI-79); a surface that passes between two fluid cell centres is a
  wall on the face between them instead of a sheet cell snapped by up to half a cell (KI-80; `AEROFLOW_NOFACEWALL`
  restores the old cells); wall-model friction on thin surfaces samples the velocity one cell out (KI-81). Falcon
  cruise C_L on 34, 48, 68 cells per length: 0.140, 0.144, 0.143 (was 0.098, 0.097, 0.083). Tests
  `test_wing_thinner_than_a_cell_is_walled`, `test_sheet_wing_lift_matches_vortex_lattice` (slow).
  On 96 cells per length the drag converges (GCI 4.4 %) but C_L falls to 0.117 (KI-84, open); the optimiser's
  reliability note no longer calls lift grid-independent.

- The mesh-study script reported a grid order and a GCI for diverging grid triples (the change growing on refinement); these are now classified as divergent, with no order (KI-76). The Falcon mesh study shows the 3D forces are not grid-converged up to 68 cells per length (KI-77) and that the half/full-model disagreement was coarse-grid error (KI-68).
- Shape optimisation: a quick run could overwrite the standard result (KI-67); infeasible rounds could be reported
  as best (KI-69). Per-profile records, feasibility-first selection, verification reports penalised J and violated
  constraints. The standard NACA 2412 record was rebuilt from its per-seed results.
- Draft steady CFD runs stopped at 900 steps before their forces settled (KI-71).
- The CFD optimiser ran half models, which disagree with the full aircraft on coarse grids (KI-68); it now runs the
  full aircraft, and the study of 2026-11-10 is marked not reliable.
- Crash when zooming in and out quickly: two meshing threads shared the worker pool at once (KI-66).
- Studio closed on any unhandled error in a command (for example a job folder that could not be written); errors are
  now shown and Studio keeps running. Job folders are created directly instead of through `cmd mkdir`.
- Selecting a half-space ran out of memory (its highlight was meshed without an end).
- The inverse-PINN task was started with an option the script does not accept (`--profile smoke`).
- The Designer app stopped at Streamlit's first-run e-mail question when started from Studio (no console); it now
  starts headless and opens the browser itself.
- The fighter 3D-flow comparison said nothing when all its states were already computed.
- Zoomed wing edges looked dirty (noisy crease fragments on the sampled mesh, a band where fine and coarse meshes
  met); the 38 m airliner ran out of memory at the 45 mm spacing (std::bad_alloc).
- Capsule features were clipped to a ball round their start point (the bounding box added itself instead of the end
  point): negative volume alone, nothing added to a design.
- Thin wing trailing edges and tips broke into a sawtooth in the editor; 4 mm sheet-metal parts and lattice walls
  vanished at the editor's mesh spacing.
- Half-space default cut away almost the whole design; extrude default sat inside the fuselage.
- Modifier tools: a nested control notification overwrote the running command, so their pad did not open.
- Inflow turbulence entered only through the far-field boundary and was damped by the coarse outer grid (5 %
  requested, 0.31 % measured upstream of the aircraft); now seeded and forced near the aircraft: 1.8 % reaches it on a draft grid (KI-56).
- Thwaites start: a leading edge without a velocity gradient set a momentum-thickness floor along the whole surface.
- 3D-flow post-processing failed on volumes with turbulence statistics (it assumed six variables).
- C++ JSON parser was quadratic on large files (msvcrt `strtod`); Studio hung on streamline files (KI-51).
- 3D-flow post-processing no longer overwrites the solver's `result.json` (KI-50).
- Studio property panel: rows, Apply and the JSON panel no longer overlap on short windows.
- Cylinder force accounting (pressure contribution), panel-method sign convention, CST validity check,
  airfoil crossing constraint, annealing α, resource-estimator model, 2D Poisson assembly vectorised.
