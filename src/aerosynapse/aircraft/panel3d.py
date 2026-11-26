r"""3D source panel method for any closed body, the verification tier above the design model.

The surface is a grid of flat quadrilaterals with constant source strength :math:`\sigma_j`. Flow
tangency at panel centres gives

.. math:: \tfrac12\sigma_i+\sum_{j\ne i}\sigma_j\,\frac{1}{4\pi}\int_j\frac{\mathbf r\cdot\mathbf n_i}{r^3}dS=-\mathbf V_\infty\cdot\mathbf n_i .

Panels close to the control point are integrated on sub-points. Tests use the exact sphere and
ellipsoid solutions. Surface speed then feeds the integral boundary layer in
:mod:`aerosynapse.shapeopt.vii` along body meridians, which gives a transition and separation
estimate from the real pressure field of that shape.

Limits: inviscid, non-lifting, attached flow, incompressible, so it checks the pressure field of the
body at zero lift. Prandtl-Glauert stretching can extend it to high subsonic speed for slender bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aerosynapse.aircraft.vehicle import Vehicle
from aerosynapse.shapeopt import vii

FloatArr = NDArray[np.float64]


@dataclass
class PanelBody:
    c: FloatArr  # centres (N, 3)
    n: FloatArr  # outward normals
    A: FloatArr  # areas
    corners: FloatArr  # (N, 4, 3)
    shape: tuple[int, int]


def from_grid(G: FloatArr) -> PanelBody:
    """Panels from a grid (ni, nj, 3) with j wrapping round the body. Degenerate ends are kept."""
    Gw = np.concatenate([G, G[:, :1]], 1)
    p0, p1, p2, p3 = Gw[:-1, :-1], Gw[1:, :-1], Gw[1:, 1:], Gw[:-1, 1:]
    nrm = 0.5 * np.cross(p2 - p0, p3 - p1)
    A = np.linalg.norm(nrm, axis=-1)
    c = 0.25 * (p0 + p1 + p2 + p3)
    n = nrm / np.maximum(A[..., None], 1e-14)
    # outward: away from the local axis point
    axis = G.mean(1, keepdims=True)[:-1]
    if float(((c - axis) * n).sum()) < 0:
        n = -n
    return PanelBody(c.reshape(-1, 3), n.reshape(-1, 3), A.ravel(), np.stack([p0, p1, p2, p3], 2).reshape(-1, 4, 3), c.shape[:2])


def _influence(b: PanelBody) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Velocity at each centre per unit source on each panel: three (N, N) matrices."""
    N = len(b.A)
    R = b.c[:, None, :] - b.c[None, :, :]
    r2 = (R**2).sum(-1)
    size2 = b.A[None, :]
    far = 1.0 / (4 * np.pi * np.maximum(r2, 1e-20) ** 1.5) * b.A[None, :]
    K = R * far[..., None]
    near = (r2 < 12.0 * size2) & ~np.eye(N, dtype=bool)
    i, j = np.where(near)
    if len(i):
        u = (np.arange(4) + 0.5) / 4
        s, t = np.meshgrid(u, u, indexing="ij")
        s, t = s.ravel(), t.ravel()
        w = np.stack([(1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t], 1)  # bilinear weights (16, 4)
        P = np.einsum("qk,nkd->nqd", w, b.corners[j])
        D = b.c[i][:, None, :] - P
        d3 = np.maximum((D**2).sum(-1), 1e-20) ** 1.5
        K[i, j] = (D / d3[..., None]).mean(1) * (b.A[j][:, None] / (4 * np.pi))
    K[np.arange(N), np.arange(N)] = 0.0
    return K[..., 0], K[..., 1], K[..., 2]


@dataclass
class Flow:
    V: FloatArr  # surface velocity (N, 3), free stream 1
    cp: FloatArr
    sigma: FloatArr
    leak: float  # mass balance error, sum(sigma A) / sum(|sigma| A)


def solve(b: PanelBody, v_inf: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> Flow:
    Kx, Ky, Kz = _influence(b)
    n = b.n
    A = Kx * n[:, 0:1] + Ky * n[:, 1:2] + Kz * n[:, 2:3]
    A[np.arange(len(b.A)), np.arange(len(b.A))] = 0.5
    V0 = np.array(v_inf)
    sig = np.linalg.solve(A, -(n @ V0))
    V = np.stack([Kx @ sig, Ky @ sig, Kz @ sig], 1) + V0
    V = V - (V * n).sum(1, keepdims=True) * n  # keep the tangential part
    sp2 = (V**2).sum(1)
    return Flow(V, 1 - sp2, sig, float(abs((sig * b.A).sum()) / max(float((np.abs(sig) * b.A).sum()), 1e-14)))


def ellipsoid_vmax(fineness: float) -> float:
    """Exact peak surface speed on a prolate spheroid in axial flow."""
    e = np.sqrt(1 - 1 / fineness**2)
    a0 = 2 * (1 - e**2) / e**3 * (0.5 * np.log((1 + e) / (1 - e)) - e)
    return float(2 / (2 - a0))


def meridian_layer(b: PanelBody, f: Flow, Re: float, j: int = 0, ncrit: float = 9.0, xtr: float = 1.0) -> dict[str, float]:
    """Integral boundary layer along one body meridian, length-normalised."""
    ni, nj = b.shape
    idx = np.arange(ni) * nj + (j % nj)
    P = b.c[idx]
    x = P[:, 0]
    L = float(x.max() - x.min())
    x = (x - x.min()) / L
    ue = np.sqrt((f.V[idx] ** 2).sum(1))
    ds = np.linalg.norm(np.diff(P, axis=0), axis=1) / L
    k = int(np.argmax(x > 0.97)) or len(x)
    s = np.r_[0.0, np.cumsum(ds)][:k]
    bl = vii.march(s, ue[:k], x[:k], Re, ncrit, xtr)
    return {"x_transition": bl.x_tr, "x_separation": bl.x_sep, "theta_te": bl.theta_te, "H_te": bl.H_te, "cp_min": float(1 - ue.max() ** 2)}


def body_grid(v: Vehicle, nx: int = 40) -> FloatArr:
    """Panel grid of a vehicle's body."""
    idx = np.linspace(0, len(v.x_st) - 1, nx).astype(int)
    sy, sz = v.sec_y[::2], v.sec_z[::2]
    x, a, b, zc = v.x_st[idx], v.a[idx], v.b[idx], v.zc[idx]
    return np.stack([np.repeat(x[:, None], len(sy), 1), a[:, None] * sy[None, :], b[:, None] * sz[None, :] + zc[:, None]], -1)
