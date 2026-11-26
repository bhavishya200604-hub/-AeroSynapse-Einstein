# How to add a physics module (Level-4 extension interface)

Every component registers itself by name, so you never edit core code:

```python
# my_package/elasticity.py
from aerosynapse.registry import PHYSICS, SOLVERS
from aerosynapse.physics.problem import PDEProblem, Parameter
from aerosynapse.physics.bc import Dirichlet

@PHYSICS.register("bar_elasticity")
def bar(E: float = 200e9, L: float = 1.0) -> PDEProblem:
    """-(E A u')' = f on [0, L]; u(0) = 0."""
    return PDEProblem("bar_elasticity", "-(E u')' = f", ((0.0, L),),
                      parameters={"E": Parameter("E", E, "Pa", 0, 1e13)},
                      bcs={"left": Dirichlet(0.0), "right": Dirichlet(0.0)})

@SOLVERS.register("bar_fd")
def solve_bar(problem, n): ...
```

Checklist (enforced by review, see CONTRIBUTING.md):

1. State the governing equations, assumptions and units in the docstring.
2. Provide an analytical or manufactured solution and a verification test with an observed order.
3. Add a solver card (`aerosynapse.numerics.cards.SOLVER_CARDS`) with all eight attributes.
4. Mark maturity honestly: `experimental (P3)` until verified.
5. Add the requirement to `docs/TRACEABILITY_MATRIX.md`.

Worked example shipped with the platform: `aerosynapse.plugins.oscillator` (Hamiltonian harmonic
oscillator, registered as `harmonic_oscillator`, experimental P3). Its test checks that the
symplectic Störmer–Verlet scheme keeps the energy error bounded while explicit Euler drifts
(`tests/test_fluids_aero.py::test_external_adapter_and_plugin`).

Interfaces kept architecturally possible but **not implemented** (DEFERRED, P3): elasticity,
thermoelasticity, structural dynamics, electromagnetics, molecular dynamics, quantum mechanics,
relativistic dynamics. GNNs: interface = `aerosynapse.registry.MODELS`; deferred (ADR-0011).
