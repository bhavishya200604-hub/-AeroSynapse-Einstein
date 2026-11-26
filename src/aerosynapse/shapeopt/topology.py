r"""Density-based topology optimisation of a body in steady laminar flow, with an exact adjoint.

Every cell of a design box holds a solid fraction :math:`\gamma\in[0,1]`, so holes, splits and new
bodies can appear, which no fixed parameterisation can do. Solid acts as a Brinkman drag term in
the incompressible Navier-Stokes equations (Borrvall and Petersson; Kondoh et al. for external
flow):

.. math:: (u\cdot\nabla)u+\nabla p-\nu\nabla^2u+\alpha(\tilde\gamma)u=0,\qquad \nabla\cdot u=0

The body force gives the drag directly, :math:`F=\int\alpha(\tilde\gamma)\,u\,dV`. The discrete
residual R(w, gamma) = 0 is solved by Newton on a staggered MAC grid. The Jacobian is exact because
the convective term is written with sparse interpolation and difference operators. Sensitivities
need one extra solve with the transposed, already factorised Jacobian per flow state:

.. math:: J^T\lambda=\partial_wF,\qquad dF/d\gamma=\partial_\gamma F-\lambda^T\partial_\gamma R .

Several flow states (Reynolds number, incidence) are combined with a smooth maximum, so the design
is driven by whichever state is currently worst. Density filtering sets a minimum feature size, and
Heaviside projection with a rising sharpness gives a crisp solid/fluid result.

The area limit acts on the physical field that the flow sees. A limit on the raw design field is a
loophole: material can be spread thinly until filtering and projection erase it, leaving a low
drag and no body. So the projection is :math:`P=\tfrac12(1+\tanh\beta(\tilde x+c-\tfrac12))`, and
at every step the uniform shift c is solved from :math:`\sum_iP_i=A`. Any area is reachable this
way. Differentiating the area condition gives the exact gradient

.. math:: \frac{dJ}{d\tilde x_i}=(g_i-\bar g)\,P'_i,\qquad \bar g=\frac{\sum_jg_jP'_j}{\sum_jP'_j},

so the design moves material rather than creating or destroying it. Updates are gradient steps
inside the box 0 <= x <= 1.

Validity: steady laminar flow, Re of order 1-100. Above the shedding threshold a steady solution
still exists but may be unstable, so results must be re-checked with the unsteady solver.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, label

FloatArr = NDArray[np.float64]


@dataclass
class Grid:
    lx: float = 6.0
    ly: float = 4.0
    h: float = 1.0 / 16.0
    box: tuple[float, float, float, float] = (1.2, 3.2, 1.2, 2.8)  # x0, x1, y0, y1 of the design region

    @property
    def nx(self) -> int:
        return round(self.lx / self.h)

    @property
    def ny(self) -> int:
        return round(self.ly / self.h)

    @property
    def xc(self) -> FloatArr:
        return (np.arange(self.nx) + 0.5) * self.h

    @property
    def yc(self) -> FloatArr:
        return (np.arange(self.ny) + 0.5) * self.h

    def design_mask(self) -> NDArray[np.bool_]:
        X, Y = np.meshgrid(self.xc, self.yc, indexing="ij")
        x0, x1, y0, y1 = self.box
        return (x0 < X) & (x1 > X) & (y0 < Y) & (y1 > Y)


class BrinkmanNS:
    """Steady MAC discretisation. Unknowns: u on x-faces, v on y-faces, p in cells.

    Boundaries: prescribed free stream on the left, top and bottom (all components), and a
    zero-gradient outflow with p = 0 on the right. ``gamma`` lives in cells.
    """

    def __init__(self, g: Grid) -> None:
        self.g = g
        nx, ny, _h = g.nx, g.ny, g.h
        self.nu_x, self.nu_y = nx, ny  # u: i = 1..nx (x-faces, i = nx is the outflow face), j = 0..ny-1
        self.Nu = nx * ny
        self.Nv = nx * (ny - 1)  # v: i = 0..nx-1, interior y-faces j = 1..ny-1
        self.Np = nx * ny
        self.N = self.Nu + self.Nv + self.Np
        self._ops()

    def _ops(self) -> None:
        nx, ny, h = self.g.nx, self.g.ny, self.g.h
        I = lambda n: sp.identity(n, format="csr")  # noqa: E731
        # 1D building blocks
        ex = sp.diags([np.ones(nx), -np.ones(nx - 1)], [0, -1], shape=(nx, nx), format="csr")  # backward difference, x
        # u index (i, j) -> i * ny + j, with i = 0..nx-1 standing for faces 1..nx
        self.idx_u = lambda i, j: i * ny + j
        # cell-centred u: uc_i = (u_{i-1} + u_i) / 2, with u_{-1} the inlet value
        Ax = 0.5 * sp.diags([np.ones(nx), np.ones(nx - 1)], [0, -1], shape=(nx, nx), format="csr")
        self.Uc = sp.kron(Ax, I(ny), format="csr")  # (cells) <- u
        self.uc_in = np.zeros(self.Nu)
        self.uc_in[:ny] = 0.5  # multiplies the inlet value
        # v at cell centres: vc_j = (v_j + v_{j+1}) / 2, with boundary faces prescribed
        By = sp.lil_matrix((ny, ny - 1))
        for j in range(ny):
            if j > 0:
                By[j, j - 1] = 0.5
            if j < ny - 1:
                By[j, j] = 0.5
        self.By = By.tocsr()
        self.Vc = sp.kron(I(nx), self.By, format="csr")
        self.vc_bc = np.zeros(self.Np).reshape(nx, ny)
        self.vc_bc[:, 0] = 0.5
        self.vc_bc[:, -1] = 0.5
        self.vc_bc = self.vc_bc.ravel()
        # divergence (cells)
        self.Dx_u = sp.kron(ex, I(ny), format="csr") / h  # u_i - u_{i-1}
        self.dx_in = np.zeros(self.Np)
        self.dx_in[:ny] = -1.0 / h
        Dy = sp.lil_matrix((ny, ny - 1))
        for j in range(ny):
            if j < ny - 1:
                Dy[j, j] = 1.0
            if j > 0:
                Dy[j, j - 1] = -1.0
        self.Dy1 = Dy.tocsr() / h
        self.Dy_v = sp.kron(I(nx), self.Dy1, format="csr")
        self.dy_bc = np.zeros((nx, ny))
        self.dy_bc[:, 0] = -1.0 / h
        self.dy_bc[:, -1] = 1.0 / h
        self.dy_bc = self.dy_bc.ravel()
        # pressure gradient on u faces: (p_{i+1} - p_i) / h, last face uses p_out = 0 ghost
        Gx = sp.lil_matrix((nx, nx))
        for i in range(nx):
            Gx[i, i] = -1.0
            if i < nx - 1:
                Gx[i, i + 1] = 1.0
            else:
                Gx[i, i] = -2.0  # p_b = 0 at the outflow face, half a cell from the last centre
        self.Gx = sp.kron(Gx.tocsr(), I(ny), format="csr") / h
        Gy = sp.lil_matrix((ny - 1, ny))
        for j in range(ny - 1):
            Gy[j, j] = -1.0
            Gy[j, j + 1] = 1.0
        self.Gy = sp.kron(I(nx), Gy.tocsr(), format="csr") / h
        # corner interpolation: (nx + 1 columns of corners at x_i, ny + 1 rows)
        # u at corners (x-face i, y-node j), v at corners, for the cross terms
        # u -> y-nodes on each x-face: rows j = 0..ny; wall rows use the prescribed value
        Cy = sp.lil_matrix((ny + 1, ny))
        for j in range(ny + 1):
            if 0 < j < ny:
                Cy[j, j - 1] = 0.5
                Cy[j, j] = 0.5
        self.Cy = Cy.tocsr()
        self.Ucor = sp.kron(I(nx), self.Cy, format="csr")  # faces 1..nx, y-nodes 0..ny
        ucb = np.zeros((nx, ny + 1))
        ucb[:, 0] = 1.0
        ucb[:, -1] = 1.0
        self.ucor_bc = ucb.ravel()
        # v -> x-faces at y-nodes: faces 1..nx; rows j = 1..ny-1 interior, j = 0, ny prescribed
        Cx = sp.lil_matrix((nx, nx))
        for i in range(nx):
            Cx[i, i] = 0.5
            if i < nx - 1:
                Cx[i, i + 1] = 0.5
            else:
                Cx[i, i] = 1.0  # zero-gradient at the outflow
        self.Cx = Cx.tocsr()
        P = sp.lil_matrix((ny + 1, ny - 1))
        for j in range(1, ny):
            P[j, j - 1] = 1.0
        self.Pv = P.tocsr()
        self.Vcor = sp.kron(self.Cx, self.Pv, format="csr")  # (nx * (ny+1)) <- v
        vcb = np.zeros((nx, ny + 1))
        vcb[:, 0] = 1.0
        vcb[:, -1] = 1.0
        self.vcor_bc = vcb.ravel()
        # d/dy of corner flux onto u faces: (F_{j+1} - F_j) / h
        Ey = sp.lil_matrix((ny, ny + 1))
        for j in range(ny):
            Ey[j, j] = -1.0
            Ey[j, j + 1] = 1.0
        self.Dcy = sp.kron(I(nx), Ey.tocsr(), format="csr") / h
        # d/dx of cell flux onto u faces: (f_{i+1} - f_i) / h, with zero-gradient past the outflow
        Fx = sp.lil_matrix((nx, nx))
        for i in range(nx):
            if i < nx - 1:
                Fx[i, i] = -1.0
                Fx[i, i + 1] = 1.0
        self.Dcx = sp.kron(Fx.tocsr(), I(ny), format="csr") / h
        # for v: d/dy of cell flux onto y-faces, d/dx of corner flux onto cells' y-faces
        Gyc = sp.lil_matrix((ny - 1, ny))
        for j in range(ny - 1):
            Gyc[j, j] = -1.0
            Gyc[j, j + 1] = 1.0
        self.Dvy = sp.kron(I(nx), Gyc.tocsr(), format="csr") / h
        # corner flux at faces 0..nx: face 0 is the inlet column
        Ex = sp.lil_matrix((nx, nx))
        for i in range(nx):
            Ex[i, i] = 1.0
            if i > 0:
                Ex[i, i - 1] = -1.0
        Sel = sp.lil_matrix((ny - 1, ny + 1))
        for j in range(ny - 1):
            Sel[j, j + 1] = 1.0
        self.Dvx = sp.kron(Ex.tocsr(), Sel.tocsr(), format="csr") / h  # (Nv) <- corner flux on faces 1..nx
        self.sel_in = np.zeros(self.Nv).reshape(nx, ny - 1)
        self.sel_in[0, :] = -1.0 / h
        self.sel_in = self.sel_in.ravel()
        # Laplacians (Dirichlet ghosts at prescribed boundaries)
        Lx = sp.lil_matrix((nx, nx))
        for i in range(nx):
            Lx[i, i] = -2.0
            if i > 0:
                Lx[i, i - 1] = 1.0
            if i < nx - 1:
                Lx[i, i + 1] = 1.0
            else:
                Lx[i, i] = -1.0
        Ly = sp.lil_matrix((ny, ny))
        for j in range(ny):
            Ly[j, j] = -2.0
            if j > 0:
                Ly[j, j - 1] = 1.0
            else:
                Ly[j, j] = -3.0
            if j < ny - 1:
                Ly[j, j + 1] = 1.0
            else:
                Ly[j, j] = -3.0
        self.Lu = (sp.kron(Lx.tocsr(), I(ny)) + sp.kron(I(nx), Ly.tocsr())).tocsr() / h**2
        lub = np.zeros((nx, ny))
        lub[0, :] += 1.0  # inlet value, x direction
        lyb = np.zeros((nx, ny))
        lyb[:, 0] += 2.0
        lyb[:, -1] += 2.0
        self.lu_inx = lub.ravel() / h**2
        self.lu_iny = lyb.ravel() / h**2
        Lxv = sp.lil_matrix((nx, nx))
        for i in range(nx):
            Lxv[i, i] = -2.0
            if i > 0:
                Lxv[i, i - 1] = 1.0
            else:
                Lxv[i, i] = -3.0
            if i < nx - 1:
                Lxv[i, i + 1] = 1.0
            else:
                Lxv[i, i] = -1.0
        Lyv = sp.lil_matrix((ny - 1, ny - 1))
        for j in range(ny - 1):
            Lyv[j, j] = -2.0
            if j > 0:
                Lyv[j, j - 1] = 1.0
            if j < ny - 2:
                Lyv[j, j + 1] = 1.0
        self.Lv = (sp.kron(Lxv.tocsr(), I(ny - 1)) + sp.kron(I(nx), Lyv.tocsr())).tocsr() / h**2
        lvb = np.zeros((nx, ny - 1))
        lvb[0, :] = 2.0
        lvy = np.zeros((nx, ny - 1))
        lvy[:, 0] = 1.0
        lvy[:, -1] = 1.0
        self.lv_inx = lvb.ravel() / h**2
        self.lv_iny = lvy.ravel() / h**2
        # cell -> face averages for the Brinkman coefficient
        Au = sp.lil_matrix((nx, nx))
        for i in range(nx):
            Au[i, i] = 0.5
            if i < nx - 1:
                Au[i, i + 1] = 0.5
            else:
                Au[i, i] = 1.0
        self.Au = sp.kron(Au.tocsr(), I(ny), format="csr")
        Av = sp.lil_matrix((ny - 1, ny))
        for j in range(ny - 1):
            Av[j, j] = 0.5
            Av[j, j + 1] = 0.5
        self.Av = sp.kron(I(nx), Av.tocsr(), format="csr")
        self.Z_up = sp.csr_matrix((self.Nu, self.Np))

    def residual_jacobian(
        self, w: FloatArr, a_u: FloatArr, a_v: FloatArr, nu: float, U: tuple[float, float], convect: bool = True
    ) -> tuple[FloatArr, sp.csr_matrix]:
        Nu, Nv = self.Nu, self.Nv
        u, v, p = w[:Nu], w[Nu : Nu + Nv], w[Nu + Nv :]
        Ui, Vi = U
        Ru = -nu * (self.Lu @ u + self.lu_inx * Ui + self.lu_iny * Ui) + self.Gx @ p + a_u * u
        Rv = -nu * (self.Lv @ v + self.lv_inx * Vi + self.lv_iny * Vi) + self.Gy @ p + a_v * v
        Juu = -nu * self.Lu + sp.diags(a_u)
        Jvv = -nu * self.Lv + sp.diags(a_v)
        Juv = sp.csr_matrix((Nu, Nv))
        Jvu = sp.csr_matrix((Nv, Nu))
        if convect:
            uc = self.Uc @ u + self.uc_in * Ui
            vc = self.Vc @ v + self.vc_bc * Vi
            ucr = self.Ucor @ u + self.ucor_bc * Ui
            vcr = self.Vcor @ v + self.vcor_bc * Vi
            Ru = Ru + self.Dcx @ (uc * uc) + self.Dcy @ (ucr * vcr)
            Juu = Juu + self.Dcx @ sp.diags(2 * uc) @ self.Uc + self.Dcy @ sp.diags(vcr) @ self.Ucor
            Juv = self.Dcy @ sp.diags(ucr) @ self.Vcor
            Rv = Rv + self.Dvy @ (vc * vc) + self.Dvx @ (ucr * vcr) + self.sel_in * (Ui * Vi)
            Jvv = Jvv + self.Dvy @ sp.diags(2 * vc) @ self.Vc + self.Dvx @ sp.diags(ucr) @ self.Vcor
            Jvu = self.Dvx @ sp.diags(vcr) @ self.Ucor
        Rp = self.Dx_u @ u + self.Dy_v @ v + self.dx_in * Ui + self.dy_bc * Vi
        R = np.r_[Ru, Rv, Rp]
        J = sp.bmat([[Juu, Juv, self.Gx], [Jvu, Jvv, self.Gy], [self.Dx_u, self.Dy_v, None]], format="csc")
        return R, J

    def solve(
        self, w0: FloatArr, a_u: FloatArr, a_v: FloatArr, nu: float, U: tuple[float, float], tol: float = 1e-8, iters: int = 25
    ) -> tuple[FloatArr, spla.SuperLU, int, float]:
        w = w0.copy()
        r0 = np.inf
        damp = 1.0
        lu = None
        k = 0
        for k in range(iters):  # noqa: B007 (k is returned)
            R, J = self.residual_jacobian(w, a_u, a_v, nu, U)
            res = float(np.abs(R).max())
            lu = spla.splu(J, permc_spec="COLAMD")
            if res < tol:
                break
            damp = 1.0 if res < r0 else max(0.25, damp * 0.5)
            w = w - damp * lu.solve(R)
            r0 = res
        return w, lu, k, res

    def initial(self, U: tuple[float, float]) -> FloatArr:
        return np.r_[np.full(self.Nu, U[0]), np.full(self.Nv, U[1]), np.zeros(self.Np)]


@dataclass
class FlowState:
    Re: float
    incidence_deg: float = 0.0
    weight: float = 1.0
    name: str = ""


@dataclass
class TopologyResult:
    gamma: FloatArr
    history: list[dict[str, float]] = field(default_factory=list)
    frames: list[FloatArr] = field(default_factory=list)
    drags0: list[float] = field(default_factory=list)
    drags: list[float] = field(default_factory=list)


def _interp(g: FloatArr, a_max: float, q: float) -> tuple[FloatArr, FloatArr]:
    """RAMP Brinkman law: alpha(g) with g = 1 solid; returns alpha and d alpha / d g."""
    a = a_max * g / (1 + q * (1 - g))
    da = a_max * (1 + q) / (1 + q * (1 - g)) ** 2
    return a, da


def project(x: FloatArr, beta: float, shift: float = 0.0) -> tuple[FloatArr, FloatArr]:
    """Logistic projection and its derivative. The derivative in x and in the shift are equal."""
    t = np.tanh(beta * (x + shift - 0.5))
    return 0.5 * (1 + t), 0.5 * beta * (1 - t**2)


def drag_of(ns: BrinkmanNS, w: FloatArr, a_u: FloatArr, a_v: FloatArr, U: tuple[float, float]) -> tuple[float, FloatArr, FloatArr]:
    """Drag along the free stream from the Brinkman force. Returns F, dF/da_u, dF/da_v."""
    Nu, Nv, h2 = ns.Nu, ns.Nv, ns.g.h**2
    u, v = w[:Nu], w[Nu : Nu + Nv]
    F = h2 * (U[0] * float(a_u @ u) + U[1] * float(a_v @ v))
    return F, h2 * U[0] * u, h2 * U[1] * v


class Topology:
    """Filter, projection, flow states, and the gradient of the smooth-max drag."""

    def __init__(
        self, g: Grid, states: list[FlowState], a_max: float = 2.0e3, q: float = 2.0, r_filter: float = 0.09, area_cells: float | None = None
    ) -> None:
        self.g, self.states, self.a_max, self.q = g, states, a_max, q
        self.ns = BrinkmanNS(g)
        self.mask = g.design_mask()
        self.sig = r_filter / g.h
        self.warm: dict[int, FloatArr] = {}
        self.area_cells = area_cells if area_cells is not None else 0.2 * float(self.mask.sum())

    def field(self, x: FloatArr, beta: float) -> tuple[FloatArr, FloatArr]:
        """Design field -> physical density with exact area, and dP/df, both zero outside the box."""
        f = gaussian_filter(x, self.sig, mode="nearest")
        m = self.mask
        lo, hi = -3.0, 3.0
        for _ in range(70):  # physical area rises with the shift
            c = 0.5 * (lo + hi)
            if project(f[m], beta, c)[0].sum() > self.area_cells:
                hi = c
            else:
                lo = c
        P, dP = project(f, beta, 0.5 * (lo + hi))
        return P * m, dP * m

    def analyse(self, phys: FloatArr, s: FlowState, i: int) -> tuple[float, FloatArr, FloatArr, FloatArr, float]:
        """Drag coefficient (per unit reference) and gradient w.r.t. the physical density."""
        ns = self.ns
        al, dal = _interp(phys.ravel(), self.a_max, self.q)
        a_u, a_v = ns.Au @ al, ns.Av @ al
        th = np.deg2rad(s.incidence_deg)
        U = (float(np.cos(th)), float(np.sin(th)))
        nu = 1.0 / s.Re
        w0 = self.warm.get(i)
        if w0 is None:
            w0 = ns.initial(U)
            # continuation in Reynolds number from a viscous, easily solved flow
            for nu_c in (nu * 16, nu * 4):
                w0, *_ = ns.solve(w0, a_u, a_v, nu_c, U, tol=1e-5, iters=12)
        w, lu, k, res = ns.solve(w0, a_u, a_v, nu, U)
        self.warm[i] = w
        F, dFu, dFv = drag_of(ns, w, a_u, a_v, U)
        rhs = np.zeros(ns.N)
        rhs[: ns.Nu] = a_u * ns.g.h**2 * U[0]
        rhs[ns.Nu : ns.Nu + ns.Nv] = a_v * ns.g.h**2 * U[1]
        lam = lu.solve(rhs, trans="T")
        u, v = w[: ns.Nu], w[ns.Nu : ns.Nu + ns.Nv]
        # dF/da - lam^T dR/da on the faces, then back to cells
        gu = dFu - lam[: ns.Nu] * u
        gv = dFv - lam[ns.Nu : ns.Nu + ns.Nv] * v
        gcell = (ns.Au.T @ gu + ns.Av.T @ gv) * dal
        return 2 * F, 2 * gcell.reshape(phys.shape), w, a_u, res

    def objective(self, x: FloatArr, beta: float, kappa: float = 12.0) -> tuple[float, FloatArr, list[float], FloatArr]:
        phys, dproj = self.field(x, beta)
        Fs, Gs = [], []
        for i, s in enumerate(self.states):
            Fi, gc, *_ = self.analyse(phys, s, i)
            Fs.append(Fi * s.weight)
            Gs.append(gc * s.weight)
        Fa = np.array(Fs)
        m = float(Fa.max())
        e = np.exp(kappa * (Fa - m) / m)
        J = float(m + m / kappa * np.log(e.mean()))
        wts = e / e.sum()
        g = sum(wi * gi for wi, gi in zip(wts, Gs, strict=False))
        # the shift keeps the area exact, so only redistribution counts
        gbar = float((g * dproj).sum()) / float(dproj.sum())
        grad = gaussian_filter((g - gbar) * dproj, self.sig, mode="nearest")
        return J, grad, [float(f) for f in Fa], phys


def optimise(
    T: Topology,
    x0: FloatArr,
    iters: int = 120,
    step: float = 0.12,
    betas: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0),
    frame_every: int = 10,
    log: Callable[[str], None] | None = None,
) -> TopologyResult:
    x = np.where(T.mask, np.clip(x0, 0.0, 1.0), 0.0)
    per = max(1, iters // len(betas))
    res = TopologyResult(x)
    best_x, best_J = x.copy(), np.inf
    lr = step
    for it in range(iters):
        b = betas[min(it // per, len(betas) - 1)]
        if it % per == 0:
            best_J = np.inf  # the objective changes with beta
        J, g, F, phys = T.objective(x, b)
        if it == 0:
            res.drags0 = F
        if best_J > J:
            best_J, best_x = J, x.copy()
            lr = min(lr * 1.1, step)
        else:
            lr = max(lr * 0.6, 0.01)
            x = best_x.copy()  # a worse step is rejected rather than built on
        gm = g[T.mask]
        scale = float(np.abs(gm - gm.mean()).max()) or 1.0
        x = np.where(T.mask, np.clip(x - lr * g / scale, 0.0, 1.0), 0.0)
        grey = float(np.mean(4 * phys[T.mask] * (1 - phys[T.mask])))
        res.history.append({"iter": it, "J": J, "beta": b, "grey": grey, "step": lr, "max_state": float(np.max(F)), "area": float(phys.sum())})
        if it % frame_every == 0 or it == iters - 1:
            res.frames.append(phys.copy())
        if log and it % 10 == 0:
            log(f"topology it {it} beta {b:g} J {J:.4f} grey {grey:.3f} step {lr:.3f} area {phys.sum() * T.g.h**2:.3f} states {np.round(F, 3)}")
    res.gamma = best_x
    phys = T.field(best_x, betas[-1])[0]
    res.drags = T.objective(best_x, betas[-1])[2]
    res.frames.append(phys.copy())
    return res


def count_bodies(phys: FloatArr) -> tuple[int, int]:
    """Number of separate solid bodies, and of enclosed holes."""
    solid = phys > 0.5
    _, n = label(solid)
    fl, m = label(~solid)
    return int(n), int(m - 1)
