"""Steady Navier–Stokes (+ Boussinesq energy) on the MAC grid by Newton's method.

Unknowns x = [u (active faces), v (active faces), p (fluid cells), theta (all cells, optional)].
Residual (same discrete operators as the time-dependent solver, so a converged x is an exact discrete
steady state of :class:`MACNavierStokes2D`):
    R_u = N_u(u, v, theta) - (G p)_u,  R_v = N_v - (G p)_v,  R_c = D u  (one row replaced by p_0 = 0 when the
    pressure is only defined up to a constant),
    R_theta = energy_rhs(u, v, theta)  (theta - theta_s in isothermal solids)
R is at most quadratic in x (convection u u, energy u theta; BC ghosts/mirrors are affine), hence the
central difference (R(x + w) - R(x - w)) / 2 = J w is EXACT (up to round-off) for any w. The Jacobian is
recovered with 25-colour probing per block (column spacing 5 in i and j; every row depends on columns within
+-2 cells, including mirror ghosts and outflow ghosts), then the sparse saddle-point system is solved
with SuperLU. Globalisation: backtracking line search on ||R||_2.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.thermal.numerics.ns_projection import MACNavierStokes2D

Array = NDArray[np.float64]


class SteadyNewtonNS:
    def __init__(self, ns: MACNavierStokes2D, t: float | None = None):
        self.ns = ns
        self.t = ns.t if t is None else t
        g = ns.g
        self.nx, self.ny = g.nx, g.ny
        self.iu = np.flatnonzero(ns.act_u.ravel())
        self.iv = np.flatnonzero(ns.act_v.ravel())
        self.ip = ns.fluid_idx
        self.energy = ns.energy is not None
        self.nu_, self.nv_, self.np_ = self.iu.size, self.iv.size, self.ip.size
        self.nt_ = g.n if self.energy else 0
        self.n = self.nu_ + self.nv_ + self.np_ + self.nt_
        o = np.cumsum([0, self.nu_, self.nv_, self.np_, self.nt_])
        self.sl = {
            "u": slice(o[0], o[1]),
            "v": slice(o[1], o[2]),
            "p": slice(o[2], o[3]),
            "t": slice(o[3], o[4]),
        }
        # doubled integer positions of every unknown/row: u (2i, 2j+1), v (2i+1, 2j), cell (2i+1, 2j+1)
        Ju, Iu = np.divmod(self.iu, g.nx + 1)
        Jv, Iv = np.divmod(self.iv, g.nx)
        Jc, Ic = np.divmod(np.arange(g.n), g.nx)
        Jp, Ip = Jc[self.ip], Ic[self.ip]
        self.pos = {
            "u": (2 * Iu, 2 * Ju + 1, Iu, Ju),
            "v": (2 * Iv + 1, 2 * Jv, Iv, Jv),
            "t": (2 * Ic + 1, 2 * Jc + 1, Ic, Jc),
        }
        rowX = [2 * Iu, 2 * Iv + 1, 2 * Ip + 1] + ([2 * Ic + 1] if self.energy else [])
        rowY = [2 * Ju + 1, 2 * Jv, 2 * Jp + 1] + ([2 * Jc + 1] if self.energy else [])
        self.rowX, self.rowY = np.concatenate(rowX), np.concatenate(rowY)
        self.gauge = ns.singular
        # exact linear blocks
        Gf = ns.G[:, ns.fluid_idx]
        nuf = ns.u.size
        self.Gu = sps.csr_matrix(Gf[self.iu])
        self.Gv = sps.csr_matrix(Gf[nuf + self.iv])
        solid_theta = ns.energy_spec.solid_theta if ns.energy_spec is not None else None
        self.fix_theta = ns.solid.ravel() if (self.energy and solid_theta is not None) else np.zeros(g.n, bool)
        self.solid_theta = solid_theta
        # block -> column index map in the unknown vector
        self.col_offset = {"u": o[0], "v": o[1], "t": o[3]}
        self.lookup = {}
        for b, shape, idx in (
            ("u", (g.ny, g.nx + 1), self.iu),
            ("v", (g.ny + 1, g.nx), self.iv),
            ("t", (g.ny, g.nx), np.arange(g.n) if self.energy else np.zeros(0, int)),
        ):
            m = -np.ones(shape[0] * shape[1], int)
            m[idx] = np.arange(idx.size)
            self.lookup[b] = (m.reshape(shape), shape)

    # --- packing ----------------------------------------------------------------------------
    def pack(self, u: Array, v: Array, p: Array | None, theta: Array | None) -> Array:
        x = np.zeros(self.n)
        x[self.sl["u"]] = u.ravel()[self.iu]
        x[self.sl["v"]] = v.ravel()[self.iv]
        if p is not None:
            x[self.sl["p"]] = np.nan_to_num(p.ravel()[self.ip])
        if self.energy and theta is not None:
            x[self.sl["t"]] = theta.ravel()
        return x

    def unpack(self, x: Array) -> tuple[Array, Array, Array, Array | None]:
        ns = self.ns
        u = np.zeros_like(ns.u).ravel()
        v = np.zeros_like(ns.v).ravel()
        u[self.iu] = x[self.sl["u"]]
        v[self.iv] = x[self.sl["v"]]
        u, v = u.reshape(ns.u.shape), v.reshape(ns.v.shape)
        ns._set_fixed(u, v, self.t)
        ns._update_mirrors(u, v)
        p = np.full(ns.g.n, np.nan)
        p[self.ip] = x[self.sl["p"]]
        th = x[self.sl["t"]].reshape(ns.g.ny, ns.g.nx).copy() if self.energy else None
        if th is not None and self.solid_theta is not None:
            th[ns.solid] = self.solid_theta
        return u, v, p.reshape(ns.g.ny, ns.g.nx), th

    def residual(self, x: Array) -> Array:
        ns = self.ns
        u, v, _, th = self.unpack(x)
        Nu, Nv = ns.momentum_rhs(u, v, th, self.t)
        pf = x[self.sl["p"]]
        Ru = Nu.ravel()[self.iu] - self.Gu @ pf
        Rv = Nv.ravel()[self.iv] - self.Gv @ pf
        Rc = ns.D[ns.fluid_idx] @ np.concatenate([u.ravel(), v.ravel()])
        if self.gauge:
            Rc = Rc.copy()
            Rc[0] = pf[0]
        parts = [Ru, Rv, Rc]
        if self.energy:
            Rt = ns.energy_rhs(u, v, th, self.t).ravel()
            if self.fix_theta.any():
                Rt = np.where(self.fix_theta, x[self.sl["t"]] - self.solid_theta, Rt)
            parts.append(Rt)
        return np.concatenate(parts)

    # --- Jacobian by exact coloured probing -------------------------------------------------
    def jacobian(self, x: Array) -> tuple[sps.csr_matrix, float]:
        rows, cols, vals = [], [], []
        max_dropped = 0.0
        blocks = ["u", "v"] + (["t"] if self.energy else [])
        for b in blocks:
            X, Y, I, J = self.pos[b]
            ox, oy = (0, 1) if b == "u" else ((1, 0) if b == "v" else (1, 1))
            m, shape = self.lookup[b]
            for a in range(5):
                for c in range(5):
                    sel = (I % 5 == a) & (J % 5 == c)
                    if not sel.any():
                        continue
                    w = np.zeros(self.n)
                    w[self.sl[b].start + np.flatnonzero(sel)] = 1.0
                    Jw = 0.5 * (self.residual(x + w) - self.residual(x - w))
                    nz = np.flatnonzero(np.abs(Jw) > 0)
                    if nz.size == 0:
                        continue
                    # unique column of colour (a, c) within +-2 cells of each row
                    ci = self._nearest(self.rowX[nz], ox, a)
                    cj = self._nearest(self.rowY[nz], oy, c)
                    ok = (ci >= 0) & (cj >= 0) & (ci < shape[1]) & (cj < shape[0])
                    colloc = np.full(nz.size, -1)
                    colloc[ok] = m[cj[ok], ci[ok]]
                    good = colloc >= 0
                    if (~good).any():
                        max_dropped = max(max_dropped, float(np.max(np.abs(Jw[nz[~good]]))))
                    rows.append(nz[good])
                    cols.append(self.col_offset[b] + colloc[good])
                    vals.append(Jw[nz[good]])
        # exact pressure columns: -G in momentum rows, gauge row
        po = self.sl["p"].start
        Gu, Gv = sps.coo_matrix(self.Gu), sps.coo_matrix(self.Gv)
        rows += [Gu.row, self.nu_ + Gv.row]
        cols += [po + Gu.col, po + Gv.col]
        vals += [-Gu.data, -Gv.data]
        if self.gauge:
            rows.append(np.array([po]))
            cols.append(np.array([po]))
            vals.append(np.array([1.0]))
        Jm = sps.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n)
        )
        Jm.sum_duplicates()
        return Jm, max_dropped

    @staticmethod
    def _nearest(pos: Array, off: int, colour: int) -> Array:
        """Index k with k = colour (mod 5) in the window |2k + off - pos| <= 4 (unique for spacing 5)."""
        centre = (pos - off) / 2.0
        lo = np.ceil(centre - 2.0 - 1e-9).astype(int)
        k = lo + np.mod(colour - lo, 5)
        return np.where(k <= centre + 2.0 + 1e-9, k, -(10**9))

    # --- Newton -----------------------------------------------------------------------------
    def solve(self, x0: Array, tol: float = 1e-10, max_iter: int = 30, verbose: bool = False) -> dict:
        x = x0.copy()
        F = self.residual(x)
        hist = [float(np.linalg.norm(F))]
        info = {"converged": False, "jac_max_dropped": 0.0, "times": []}
        for it in range(max_iter):
            if hist[-1] < tol:
                info["converged"] = True
                break
            t0 = time.perf_counter()
            J, dropped = self.jacobian(x)
            info["jac_max_dropped"] = max(info["jac_max_dropped"], dropped)
            dx = spla.splu(sps.csc_matrix(J)).solve(-F)
            alpha = 1.0
            while True:
                xn = x + alpha * dx
                Fn = self.residual(xn)
                if np.linalg.norm(Fn) < (1 - 1e-4 * alpha) * hist[-1] or alpha < 1e-4:
                    break
                alpha *= 0.5
            x, F = xn, Fn
            hist.append(float(np.linalg.norm(F)))
            info["times"].append(time.perf_counter() - t0)
            if verbose:
                print(f"  newton it {it + 1}: |R| = {hist[-1]:.3e}  alpha = {alpha:g}  ({info['times'][-1]:.1f}s)")
        info["converged"] = info["converged"] or hist[-1] < tol
        info["residual_history"] = hist
        info["iterations"] = len(hist) - 1
        self.x = x
        return info

    def store(self, x: Array | None = None) -> Array:
        """Write the solution into the underlying solver state; returns the pressure (zero-mean if gauged)."""
        u, v, p, th = self.unpack(self.x if x is None else x)
        self.ns.u, self.ns.v = u, v
        if th is not None:
            self.ns.theta = th
        if self.gauge:
            p = p - np.nanmean(p)
        return p
