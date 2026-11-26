"""Simulations of a design, each writing ``summary.txt``, ``result.json`` and figures to a folder.

    python -m aerosynapse.cli simulate analytic  design.json --out results/studio/sim/analytic
    python -m aerosynapse.cli simulate pinn      design.json --out ...
    python -m aerosynapse.cli simulate structure design.json --out ...
    python -m aerosynapse.cli simulate all       design.json --out ...

* :mod:`analytic`  closed-form and classical solutions on the design itself: panel method on the wing's own
  section, Prandtl lifting line on its planform, Sears-Haack area rule, Taylor-Maccoll cone and normal-shock
  relations, Ackeret supersonic section theory, exact transient skin heating.
* :mod:`pinn`      the parametric PINN as a simulator: against the exact series and the finite-difference solver,
  its temperature field, its query speed, and the design's skin zones.
* :mod:`structure` design-code loads, wing box sizing, finite elements, V-g flutter diagram.

3D flow (the ``aeroflow`` solver) runs from Studio's Simulate tab or ``aerosynapse cfd``.
"""
