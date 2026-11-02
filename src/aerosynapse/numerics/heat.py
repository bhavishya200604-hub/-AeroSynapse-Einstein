r"""Heat equation solvers :math:`T_t = \alpha\nabla^2 T + s` (1D and 2D, node-centred FD, θ-method)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError, StabilityError
from aerosynapse.numerics.fd1d import assemble_1d, trapezoid_weights
from aerosynapse.numerics.guards import check_field
from aerosynapse.numerics.timestep import TimeSolution, theta_integrate
from aerosynapse.physics.problem import PDEProblem
from aerosynapse.registry import SOLVERS

FloatArr = NDArray[np.float64]


@SOLVERS.register("heat1d_fd")
def solve_heat_1d(
    problem: PDEProblem,
    nx: int,
    nt: int | None = None,
    theta: float = 0.5,
    adaptive: bool = False,
    tol: float = 1e-7,
) -> tuple[FloatArr, TimeSolution]:
    """Solve the 1D heat problem; returns ``(x, solution)``.

    Explicit Euler (θ=0) is checked against the stability limit ``α Δt/Δx² ≤ 1/2`` and raises
    :class:`StabilityError` instead of producing a growing field.
    """
    ((x0, x1),) = problem.domain
    alpha = problem.p("alpha")
    op = assemble_1d(x0, x1, nx, alpha, 0.0, problem.bcs.get("left"), problem.bcs.get("right"))
    if problem.t_final is None or problem.initial_condition is None:
        raise ConfigurationError("transient problem needs t_final and initial_condition")
    nt = nt or nx
    dt = problem.t_final / nt
    if theta < 0.5 and alpha * dt / op.h**2 > 0.5 / (1 - 2 * theta):
        raise StabilityError(
            f"theta={theta} unstable: r = alpha dt/dx^2 = {alpha * dt / op.h**2:.3f} > {0.5 / (1 - 2 * theta):.3f}"
        )
    src = problem.source
    source = (lambda xx, t: np.asarray(src(xx, t))) if src is not None else None
    sol = theta_integrate(
        op,
        problem.initial_condition(op.x),
        problem.t_final,
        dt,
        theta,
        source,
        adaptive=adaptive,
        tol=tol,
        weights=trapezoid_weights(op),
    )
    return op.x, sol


def laplacian_2d_dirichlet(nx: int, ny: int, hx: float, hy: float) -> sp.csr_matrix:
    """Five-point Laplacian on the (nx-1)×(ny-1) interior nodes (Dirichlet eliminated), x fastest."""

    def d2(n: int, h: float) -> sp.csr_matrix:
        m = n - 1
        return sp.diags([np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)], [-1, 0, 1]) / h**2

    Ix, Iy = sp.identity(nx - 1), sp.identity(ny - 1)
    return sp.csr_matrix(sp.kron(Iy, d2(nx, hx)) + sp.kron(d2(ny, hy), Ix))


def _boundary_rhs(
    g: Callable[[FloatArr, FloatArr], FloatArr], X: FloatArr, Y: FloatArr, hx: float, hy: float
) -> FloatArr:
    """Contribution of Dirichlet values to the interior five-point stencil."""
    G = g(X, Y)
    b = np.zeros((X.shape[0] - 2, X.shape[1] - 2))
    b[:, 0] += G[1:-1, 0] / hx**2
    b[:, -1] += G[1:-1, -1] / hx**2
    b[0, :] += G[0, 1:-1] / hy**2
    b[-1, :] += G[-1, 1:-1] / hy**2
    return b.ravel()


def _at_time(g: Callable[[FloatArr, FloatArr, float], FloatArr], t: float) -> Callable[[FloatArr, FloatArr], FloatArr]:
    def gt(x: FloatArr, y: FloatArr) -> FloatArr:
        return g(x, y, t)

    return gt


@SOLVERS.register("heat2d_fd")
def solve_heat_2d(
    problem: PDEProblem,
    n: int,
    nt: int | None = None,
    theta: float = 0.5,
    g: Callable[[FloatArr, FloatArr, float], FloatArr] | None = None,
) -> tuple[FloatArr, FloatArr, FloatArr]:
    """2D heat with Dirichlet data ``g(x, y, t)`` (default 0) by the θ-method; returns ``(X, Y, T)``."""
    (x0, x1), (y0, y1) = problem.domain
    if problem.t_final is None or problem.initial_condition is None:
        raise ConfigurationError("transient problem needs t_final and initial_condition")
    if any(b.kind != "dirichlet" for b in problem.bcs.values()):
        raise ConfigurationError("solve_heat_2d supports Dirichlet sides only; see poisson_2d for Neumann")
    alpha = problem.p("alpha")
    hx, hy = (x1 - x0) / n, (y1 - y0) / n
    xs, ys = np.linspace(x0, x1, n + 1), np.linspace(y0, y1, n + 1)
    X, Y = np.meshgrid(xs, ys)  # shape (ny+1, nx+1), row = y
    L = alpha * laplacian_2d_dirichlet(n, n, hx, hy)
    nt = nt or n
    dt = problem.t_final / nt
    gg = g if g is not None else (lambda x, y, t: np.zeros_like(x))
    U = problem.initial_condition(X, Y).astype(float)
    u = U[1:-1, 1:-1].ravel()
    I = sp.identity(L.shape[0], format="csc")
    solve = spla.factorized((I - theta * dt * L).tocsc())
    R = (I + (1 - theta) * dt * L).tocsr()
    t = 0.0
    for k in range(nt):
        b0 = alpha * _boundary_rhs(_at_time(gg, t), X, Y, hx, hy)
        b1 = alpha * _boundary_rhs(_at_time(gg, t + dt), X, Y, hx, hy)
        rhs = R @ u + dt * ((1 - theta) * b0 + theta * b1)
        if problem.source is not None:
            src = problem.source
            rhs += dt * ((1 - theta) * src(X, Y, t)[1:-1, 1:-1].ravel() + theta * src(X, Y, t + dt)[1:-1, 1:-1].ravel())
        u = solve(rhs)
        t += dt
        check_field(u, step=k)
    T = gg(X, Y, t).astype(float)
    T[1:-1, 1:-1] = u.reshape(n - 1, n - 1)
    return X, Y, T
