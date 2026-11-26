"""Structural analysis and design codes for the airframe.

* :mod:`codes`       design codes (MIL-A-8861 fighter, CS-25, CS-23, aerobatic): limit load factors, factors of
                     safety, fitting factor, gust velocities; the V-n diagram (manoeuvre and Pratt gust lines).
* :mod:`fem`         finite elements: 3D truss and 3D Euler-Bernoulli frame (12 dof per element), static
                     solution, reactions, member forces, consistent mass and natural modes.
* :mod:`panels`      plate buckling in compression and shear (with interaction), Euler-Johnson columns,
                     thin-walled closed sections (Bredt-Batho torsion).
* :mod:`aeroelastic` Theodorsen's function, typical-section flutter by the k (V-g) method, divergence.
* :mod:`wing`        the wing box of a feature-tree design: spanwise loads (Schrenk), box section from the actual
                     airfoil, sizing of skins and webs to zero margin, a frame model for deflection and modes,
                     flutter and divergence margins, and a fuselage bending check.

Research use only: these are preliminary-design methods with their stated assumptions, not certification analysis.
"""
