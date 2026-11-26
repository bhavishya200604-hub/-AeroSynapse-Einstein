r"""Finite elements for frames and trusses in 3D.

Frame element: two nodes, six degrees of freedom each (u, v, w, theta_x, theta_y, theta_z), Euler-Bernoulli
bending about both principal axes, St Venant torsion and axial stretch. Local stiffness from Przemieniecki,
rotated to global axes with a reference vector that fixes the element's local y axis. Consistent mass matrix
(translational plus torsional inertia). Truss element: axial bar only (3 dof per node used).

Checks (``tests/test_structures.py``): cantilever tip deflection :math:`PL^3/3EI` and twist :math:`TL/GJ`, the
first two bending frequencies of a cantilever :math:`(1.8751, 4.6941)^2\sqrt{EI/(mL^4)}`, and a statically
determinate truss against hand calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

FloatArr = NDArray[np.float64]


@dataclass
class Section:
    E: float
    G: float
    A: float
    Iy: float  # about local y
    Iz: float  # about local z
    J: float
    rho: float = 0.0  # material density for mass (kg/m^3); or use m_per_length
    m_per_length: float = 0.0  # extra non-structural mass per length (fuel, systems), kg/m
    Ip: float = 0.0  # polar mass moment per length about the axis, kg m (0 = rho (Iy + Iz))


@dataclass
class Frame:
    nodes: FloatArr  # (N, 3)
    elements: list[tuple[int, int, Section, FloatArr | None]] = field(default_factory=list)  # (i, j, section, reference vector)
    truss: list[tuple[int, int, float, float]] = field(default_factory=list)  # (i, j, E, A)

    @property
    def ndof(self) -> int:
        return 6 * len(self.nodes)

    def add(self, i: int, j: int, s: Section, ref: FloatArr | None = None) -> None:
        self.elements.append((i, j, s, ref))

    # ------------------------------------------------------------------ element matrices
    def _rotation(self, i: int, j: int, ref: FloatArr | None) -> tuple[FloatArr, float]:
        d = self.nodes[j] - self.nodes[i]
        L = float(np.linalg.norm(d))
        ex = d / L
        r = np.array([0.0, 0.0, 1.0]) if ref is None else np.asarray(ref, float)
        if abs(np.dot(r, ex)) > 0.99 * np.linalg.norm(r):
            r = np.array([0.0, 1.0, 0.0]) if abs(ex[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        ez = np.cross(ex, r)
        ez /= np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        lam = np.vstack([ex, ey, ez])
        T = np.zeros((12, 12))
        for k in range(4):
            T[3 * k : 3 * k + 3, 3 * k : 3 * k + 3] = lam
        return T, L

    @staticmethod
    def _k_local(s: Section, L: float) -> FloatArr:
        E, G, A, Iy, Iz, J = s.E, s.G, s.A, s.Iy, s.Iz, s.J
        k = np.zeros((12, 12))
        a = E * A / L
        k[0, 0] = k[6, 6] = a
        k[0, 6] = k[6, 0] = -a
        t = G * J / L
        k[3, 3] = k[9, 9] = t
        k[3, 9] = k[9, 3] = -t
        # bending in the x-y plane (about z): dofs v1, rz1, v2, rz2 = 1, 5, 7, 11
        for (v1, r1, v2, r2), I, sgn in (((1, 5, 7, 11), Iz, 1.0), ((2, 4, 8, 10), Iy, -1.0)):  # rotation about y is minus dw/dx
            c = E * I / L**3
            idx = [v1, r1, v2, r2]
            kb = c * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
            S = np.diag([1, sgn, 1, sgn])
            kb = S @ kb @ S
            for p in range(4):
                for q in range(4):
                    k[idx[p], idx[q]] += kb[p, q]
        return k

    @staticmethod
    def _m_local(s: Section, L: float) -> FloatArr:
        m = s.rho * s.A + s.m_per_length
        ip = s.Ip if s.Ip > 0 else s.rho * (s.Iy + s.Iz)
        M = np.zeros((12, 12))
        M[0, 0] = M[6, 6] = m * L / 3
        M[0, 6] = M[6, 0] = m * L / 6
        M[3, 3] = M[9, 9] = ip * L / 3
        M[3, 9] = M[9, 3] = ip * L / 6
        for (v1, r1, v2, r2), sgn in (((1, 5, 7, 11), 1.0), ((2, 4, 8, 10), -1.0)):
            mb = m * L / 420 * np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L * L, 13 * L, -3 * L * L], [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L * L, -22 * L, 4 * L * L]])
            S = np.diag([1, sgn, 1, sgn])
            mb = S @ mb @ S
            idx = [v1, r1, v2, r2]
            for p in range(4):
                for q in range(4):
                    M[idx[p], idx[q]] += mb[p, q]
        return M

    # ------------------------------------------------------------------ assembly
    def matrices(self) -> tuple[FloatArr, FloatArr]:
        n = self.ndof
        K = np.zeros((n, n))
        M = np.zeros((n, n))
        for i, j, s, ref in self.elements:
            T, L = self._rotation(i, j, ref)
            kg = T.T @ self._k_local(s, L) @ T
            mg = T.T @ self._m_local(s, L) @ T
            dofs = np.r_[6 * i : 6 * i + 6, 6 * j : 6 * j + 6]
            K[np.ix_(dofs, dofs)] += kg
            M[np.ix_(dofs, dofs)] += mg
        for i, j, E, A in self.truss:
            d = self.nodes[j] - self.nodes[i]
            L = float(np.linalg.norm(d))
            e = d / L
            kb = E * A / L * np.outer(e, e)
            for a, b, sg in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
                K[6 * a : 6 * a + 3, 6 * b : 6 * b + 3] += sg * kb
        return K, M

    def solve(self, loads: FloatArr, fixed: list[int]) -> dict[str, FloatArr]:
        """Static solution. ``loads`` (N, 6) forces and moments at nodes; ``fixed`` global dof indices held at zero."""
        K, _ = self.matrices()
        n = self.ndof
        free = np.setdiff1d(np.arange(n), fixed)
        # dofs with no stiffness at all (a truss node's rotations) are held too
        free = free[np.abs(np.diag(K)[free]) > 0]
        f = np.asarray(loads, float).ravel()
        u = np.zeros(n)
        u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
        return {"u": u.reshape(-1, 6), "reactions": (K @ u - f).reshape(-1, 6)}

    def member_forces(self, u: FloatArr) -> list[FloatArr]:
        """End forces of every frame element in its local axes: N, Vy, Vz, T, My, Mz at each end."""
        out = []
        uf = np.asarray(u).ravel()
        for i, j, s, ref in self.elements:
            T, L = self._rotation(i, j, ref)
            dofs = np.r_[6 * i : 6 * i + 6, 6 * j : 6 * j + 6]
            out.append(self._k_local(s, L) @ (T @ uf[dofs]))
        return out

    def truss_forces(self, u: FloatArr) -> list[float]:
        uu = np.asarray(u).reshape(-1, 6)
        res = []
        for i, j, E, A in self.truss:
            d = self.nodes[j] - self.nodes[i]
            L = float(np.linalg.norm(d))
            res.append(float(E * A / L * np.dot(d / L, uu[j, :3] - uu[i, :3])))
        return res

    def modes(self, fixed: list[int], n_modes: int = 6, point_masses: dict[int, float] | None = None) -> dict[str, FloatArr]:
        K, M = self.matrices()
        for node, m in (point_masses or {}).items():
            for k in range(3):
                M[6 * node + k, 6 * node + k] += m
        n = self.ndof
        free = np.setdiff1d(np.arange(n), fixed)
        free = free[np.abs(np.diag(K)[free]) > 0]
        w2, vec = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], subset_by_index=[0, min(n_modes, len(free)) - 1])
        shapes = np.zeros((len(w2), n))
        shapes[:, free] = vec.T
        return {"freq_hz": np.sqrt(np.maximum(w2, 0)) / (2 * np.pi), "shapes": shapes.reshape(len(w2), -1, 6)}
