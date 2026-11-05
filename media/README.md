# Videos

All videos are produced by code in `scripts/videos/` (`render_clips.py` → `clips_cards.py` → `record_ui.js` →
`assemble.py`) with the neon theme in `src/aerosynapse/viz/neon.py`. Every frame is either computed live by the
platform's own solvers/models while rendering, or read from `results/<id>/metrics.json`; the source is printed on
each frame. Soundtrack: a quiet ambient pad synthesised with ffmpeg (no third-party audio).

| File | Length | Content |
|---|---|---|
| `AeroSynapse_Einstein_showcase.mp4` | ~7.9 min | everything below in one film |
| `simulations_and_graphs.mp4` | ~2.2 min | live simulations + animated result graphs |
| `workbench_walkthrough.mp4` | ~5.7 min | command line (real captured output) + the five-workspace workbench driven through every option |

**Simulations (computed live):** 2D heat diffusion (rotating depth-shaded surface); convection–diffusion at
Pe = 1/10/100/1000 vs the exact solution; Burgers shock formation and the wave equation; lid-driven cavity (Re = 400,
speed + streamlines); vortex street behind a cylinder at Re = 100.

**Scientific ML & optimisation:** a PINN learning convection–diffusion (prediction, exact, error, loss and the RAD
collocation points over training); the FNO inside vs outside its training range; the airfoil morphing along the real
optimisation history with C_p and the best-so-far trace.

**Animated graphs (from results/):** convergence orders, the core sampler study with CIs, the FNO generalisation map,
the PINN budget surface, the 3D Pareto front, PINN space–time surfaces, the trust/fallback engine, cavity vs Ghia et al.

**Honesty notes.** The solvers are laminar; nothing shows turbulence. The cylinder clip at Re = 100 is outside the
validated range (validated at Re = 20) and the regime guard classifies it as 'transitional' — it is labelled on
screen as an unvalidated laminar visualisation. The workbench clips show real computations at small budgets (e.g. a
PINN trained for only a few hundred iterations has a large error, shown as such).

Regenerate: `python scripts/videos/sim_cylinder.py && python scripts/videos/render_clips.py && python
scripts/videos/clips_cards.py`, start the workbench on port 8517, `node scripts/videos/record_ui.js media/_cache/uivid_dark`,
then `python scripts/videos/assemble.py`.
