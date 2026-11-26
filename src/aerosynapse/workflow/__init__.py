"""The design workflow: one project, eight stages, from a basic design to a documented, exported aircraft.

    1 concept     basic design (a family baseline or any input body) and its first analysis
    2 design      design optimisation of the whole configuration (fighter design space, or the wing planform of
                  an input body) with CMA-ES
    3 wing        wing optimisation: the airfoil of the main wing by the shape optimiser (CST, NeuralFoil with
                  compressibility, supersonic wave drag) checked with XFOIL, put back into the 3D wing
    4 body        body optimisation: fuselage cross-sections for the supersonic area rule (Mach-plane wave drag),
                  with the internal volume and the engine and cockpit envelopes as constraints
    5 structure   design-code loads, wing box sizing (strength, buckling, flutter, divergence), fuselage bending,
                  material trade
    6 thermal     skin heating with the parametric PINN: convergence study, then the lightest skin per zone
    7 simulation  3D flow solver: grid-convergence study (Richardson extrapolation, GCI) and the flight envelope
    8 export      STL, OBJ, glTF, STEP, DXF drawings, drawing sheet, bill of materials, mass properties,
                  report, and a checksummed project package

Every stage reads the design the previous stage left, writes its own design revision and results, and records a
scorecard, so the improvement from stage to stage is visible. ``aerosynapse workflow`` runs it from the command line
and AeroSynapse Studio runs it from its Design workflow window.
"""

from aerosynapse.workflow.project import STAGES, Project

__all__ = ["STAGES", "Project"]
