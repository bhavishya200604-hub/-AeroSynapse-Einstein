r"""Vortex-lattice method for every lifting surface of a design together.

Each wing-type feature of the design (wing, canard, tail, fins, strakes, blades) becomes a lattice of
horseshoe vortices: bound segment on the quarter-chord line of each panel, trailing legs to infinity
along the body axis, control point at three-quarter chord, where the flow must be tangent to the camber
surface:

.. math:: \Big(\mathbf V_\infty+\sum_j \Gamma_j\,\mathbf v_{ij}\Big)\cdot\mathbf n_i=0 .

Lift, side force and moments come from Kutta-Joukowski on every bound segment, so they include the
interference between surfaces (canard downwash on the wing, wing downwash on the tail, fin sidewash).
Induced drag is computed in the Trefftz plane from the trailing vortex pairs, which is more reliable than
the near-field force. Horizontal surfaces are extended to the centre line, the usual way of counting the
lift the body carries between the wing roots. Compressibility: Prandtl-Glauert, by stretching x by
:math:`1/\beta` and dividing the loads by :math:`\beta` (subsonic only).

Vortex lift from sharp, highly swept edges follows Polhamus' suction analogy:
:math:`C_L = K_p\sin\alpha\cos^2\alpha + K_v\sin^2\alpha\cos\alpha`, with
:math:`K_v=(K_p-K_p^2/(\pi A))/\cos\Lambda` per swept surface.

Limits: thin surfaces, attached flow apart from the suction analogy, no thickness or body volume (the body
is added outside with slender-body theory). Tests check the elliptic-wing induced drag, Helmbold's lift
slope for a rectangular wing and the symmetry of the lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]
WING_TYPES = ("wing", "canard", "tail", "fin", "strake", "blade")


@dataclass
class Lattice:
    A: FloatArr  # bound segment start (N, 3); A -> B runs in +y on horizontal surfaces
    B: FloatArr
    C: FloatArr  # control points
    n: FloatArr  # unit normals
    surf: NDArray[np.int64]
    names: list[str]
    beta: float


def _planform(f: dict[str, Any]) -> tuple[float, float, float, float, float, float, FloatArr, bool]:
    span, cr, lam = float(f.get("span", 1)), float(f.get("root_chord", 1)), float(f.get("taper", 0.5))
    sw, di = np.radians(float(f.get("sweep", 30))), np.radians(float(f.get("dihedral", 0)))
    t = np.array(f.get("translate", [0, 0, 0]), float)
    mirror = bool(f.get("mirror", float(f.get("dihedral", 0)) < 80))
    if mirror and f.get("type") in ("wing", "canard") and abs(float(f.get("dihedral", 0))) < 30 and t[1] > 0:  # body carry-over
        ct = lam * cr
        cr0 = cr + (cr - ct) * t[1] / max(span, 1e-9)
        t = t - np.array([t[1] * np.tan(sw), t[1], 0.0])
        span, cr, lam = span + float(f["translate"][1]), cr0, ct / cr0
    return span, cr, lam, sw, di, np.radians(float(f.get("twist", 0))), t, mirror


def surfaces(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in features if f.get("type") in WING_TYPES and f.get("enabled", True) and f.get("op", "add") == "add"]


def lattice(features: list[dict[str, Any]], mach: float = 0.0, nc: int = 4, ns: int = 10) -> Lattice:
    beta = float(np.sqrt(max(1 - min(mach, 0.95) ** 2, 0.1)))
    A, B, C, N, sf, names = [], [], [], [], [], []
    for k, f in enumerate(surfaces(features)):
        span, cr, lam, sw, di, tw, t, mirror = _planform(f)
        camber, cpos = float(f.get("camber", 0)), float(f.get("camber_pos", 0.4))
        small = f.get("type") in ("blade", "strake")
        mc, ms = (max(2, nc // 2), max(3, ns // 2)) if small else (nc, ns)
        eta = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ms + 1)))
        xi = np.linspace(0, 1, mc + 1)

        def pt(e: float, x: float, s: float) -> FloatArr:
            c = cr * (1 - (1 - lam) * e)
            xq = e * span * np.tan(sw) + 0.25 * c
            dx = e * span * np.tan(sw) + x * c - xq
            zc = 0.0
            if camber > 0:
                zc = (camber / cpos**2 * (2 * cpos * x - x * x) if x < cpos else camber / (1 - cpos) ** 2 * (1 - 2 * cpos + 2 * cpos * x - x * x)) * c
            th = tw * e
            X, Z, y = xq + dx * np.cos(th) + zc * np.sin(th), -dx * np.sin(th) + zc * np.cos(th), e * span
            P = np.array([X, s * (y * np.cos(di) - Z * np.sin(di)), y * np.sin(di) + Z * np.cos(di)]) + t * np.array([1, s, 1])
            P[0] /= beta
            return P

        for s in ((1.0, -1.0) if mirror else (1.0,)):
            for j in range(ms):
                for i in range(mc):
                    e0, e1, x0, x1 = eta[j], eta[j + 1], xi[i], xi[i + 1]
                    xa, xc, em = x0 + 0.25 * (x1 - x0), x0 + 0.75 * (x1 - x0), 0.5 * (e0 + e1)
                    a, b = pt(e0, xa, s), pt(e1, xa, s)
                    if s < 0:
                        a, b = b, a
                    nrm = np.cross(pt(em, x1, s) - pt(em, x0, s), (pt(e1, 0.5 * (x0 + x1), s) - pt(e0, 0.5 * (x0 + x1), s)) * s)
                    nrm /= max(float(np.linalg.norm(nrm)), 1e-12)
                    A.append(a); B.append(b); C.append(0.5 * (pt(e0, xc, s) + pt(e1, xc, s))); N.append(nrm); sf.append(k)  # noqa: E702
        names.append(str(f.get("id", f.get("type"))))
    return Lattice(np.array(A).reshape(-1, 3), np.array(B).reshape(-1, 3), np.array(C).reshape(-1, 3), np.array(N).reshape(-1, 3), np.array(sf, int), names, beta)


def _seg(P: FloatArr, A: FloatArr, B: FloatArr) -> FloatArr:
    """Velocity at points P (M,3) from unit-strength segments A->B (N,3): (M, N, 3)."""
    r1 = P[:, None, :] - A[None]
    r2 = P[:, None, :] - B[None]
    cr = np.cross(r1, r2)
    c2 = np.sum(cr * cr, -1)
    n1, n2 = np.linalg.norm(r1, axis=-1), np.linalg.norm(r2, axis=-1)
    r0 = (B - A)[None]
    k = np.sum(r0 * (r1 / np.maximum(n1, 1e-12)[..., None] - r2 / np.maximum(n2, 1e-12)[..., None]), -1)
    core = 1e-10 + 1e-6 * np.sum(r0 * r0, -1)
    f = np.where(c2 > core, k / (4 * np.pi * np.maximum(c2, 1e-30)), 0.0)
    return np.asarray(cr * f[..., None])


def _leg(P: FloatArr, A: FloatArr, sign: float) -> FloatArr:
    """Semi-infinite vortex from A along +x (sign +1 leaving A, -1 arriving at A)."""
    d = np.array([1.0, 0.0, 0.0])
    r = P[:, None, :] - A[None]
    nr = np.linalg.norm(r, axis=-1)
    cr = np.cross(d[None, None, :], r)
    c2 = np.sum(cr * cr, -1)
    f = np.where(c2 > 1e-10, (1 + r[..., 0] / np.maximum(nr, 1e-12)) / (4 * np.pi * np.maximum(c2, 1e-30)), 0.0)
    return np.asarray(sign * cr * f[..., None])


def _trefftz(L: Lattice) -> FloatArr:
    """Normal wash at each wake strip (Trefftz plane) per unit circulation of every horseshoe: (N, N)."""
    ya, yb = L.A[:, 1:], L.B[:, 1:]  # (y, z) of the trailing legs
    mid = 0.5 * (ya + yb)
    t = yb - ya
    ln = np.linalg.norm(t, axis=1)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1) / np.maximum(ln, 1e-12)[:, None]  # +z for a strip running in +y

    def pv(P: FloatArr, Q: FloatArr, s: float) -> FloatArr:  # 2D point vortex, circulation s about +x
        r = P[:, None, :] - Q[None]
        r2 = np.maximum(np.sum(r * r, -1), 1e-10)
        return np.asarray(s * np.stack([-r[..., 1], r[..., 0]], -1) / (2 * np.pi * r2[..., None]))

    v = pv(mid, yb, 1.0) + pv(mid, ya, -1.0)  # the leg at B leaves along +x, the one at A arrives
    return np.einsum("ijk,ik->ij", v, nrm) * ln[:, None]


@dataclass
class Loads:
    CL: float
    CD: float  # induced, Trefftz plane
    CY: float
    Cl: float  # standard axes: positive right wing down
    Cm: float  # positive nose up
    Cn: float  # positive nose right
    CL_surf: list[float]


class Solver:
    """Lattice with its influence matrices built once; solve() for any incidence and sideslip."""

    def __init__(self, features: list[dict[str, Any]], mach: float, nc: int = 4, ns: int = 10) -> None:
        self.L = lattice(features, mach, nc, ns)
        L = self.L
        mid = 0.5 * (L.A + L.B)
        self.mid = mid
        if len(L.A):
            full = _seg(L.C, L.A, L.B) + _leg(L.C, L.B, 1.0) + _leg(L.C, L.A, -1.0)
            self.M = np.einsum("ijk,ik->ij", full, L.n)
            self.Vt = _leg(mid, L.B, 1.0) + _leg(mid, L.A, -1.0)
            self.T = _trefftz(L)

    def solve(self, alpha_deg: float, beta_deg: float, S: float, c: float, b: float, x_ref: float, z_ref: float = 0.0) -> Loads:
        L = self.L
        if len(L.A) == 0:
            return Loads(0, 0, 0, 0, 0, 0, [])
        a, bb = np.radians(alpha_deg), np.radians(beta_deg)
        V = np.array([np.cos(a) * np.cos(bb), -np.sin(bb), np.sin(a) * np.cos(bb)])  # +x is aft: the air moves along +x
        g = np.linalg.solve(self.M, -L.n @ V)
        vin = np.einsum("ijk,j->ik", self.Vt, g)
        F = g[:, None] * np.cross(V[None] + vin, L.B - L.A)  # the stretched wing's larger area already carries one 1/beta
        q = 0.5
        Fy = F.sum(0)[1]
        lift_dir = np.array([-np.sin(a), 0.0, np.cos(a)])
        Lf = float(F.sum(0) @ lift_dir)
        Di = float(-0.5 * np.sum(g * (self.T @ g)))  # Trefftz plane: D = -1/2 sum Gamma w ds; the far wake is not compressed
        r = self.mid.copy()
        r[:, 0] = r[:, 0] * L.beta - x_ref
        r[:, 2] -= z_ref
        Mo = np.cross(r, F).sum(0)
        cls = [float((F[L.surf == k] @ lift_dir).sum() / (q * S)) for k in range(len(L.names))]
        # body axes here: x aft, y right, z up. Standard signs: roll + right wing down = -Mx, pitch + nose up = +My, yaw + nose right = -Mz
        return Loads(Lf / (q * S), Di / (q * S), float(Fy) / (q * S), float(-Mo[0]) / (q * S * b), float(Mo[1]) / (q * S * c), float(-Mo[2]) / (q * S * b), cls)


@dataclass
class Derivatives:
    CL_alpha: float  # per rad
    K: float  # induced drag factor CDi = K CL^2
    x_np: float  # neutral point (m) of the lifting surfaces
    Cn_beta: float  # per rad, standard sign (+ stable)
    Cl_beta: float  # per rad (- stable)
    CY_beta: float
    CL0: float
    Cm0: float
    Kv: float  # Polhamus vortex-lift constant, area-weighted over swept surfaces
    lam_le: float  # area-weighted leading-edge sweep of those surfaces, rad
    CL_surf_alpha: list[float]
    names: list[str]


def derivatives(features: list[dict[str, Any]], mach: float, S: float, c: float, b: float, x_ref: float, nc: int = 4, ns: int = 10) -> Derivatives:
    sv = Solver(features, mach, nc, ns)
    if len(sv.L.A) == 0:
        return Derivatives(0, 1, x_ref, 0, 0, 0, 0, 0, 0, 0, [], [])
    l0 = sv.solve(0.0, 0.0, S, c, b, x_ref)
    l1 = sv.solve(2.0, 0.0, S, c, b, x_ref)
    lb = sv.solve(2.0, 4.0, S, c, b, x_ref)
    da, db = np.radians(2.0), np.radians(4.0)
    CLa = (l1.CL - l0.CL) / da
    Cma = (l1.Cm - l0.Cm) / da
    lm = sv.solve(-4.0, 0.0, S, c, b, x_ref)
    cl, cd = np.array([lm.CL, l0.CL, l1.CL]), np.array([lm.CD, l0.CD, l1.CD])
    K = float(np.polyfit(cl, cd, 2)[0])  # CDi = CDmin + K (CL - CLmin)^2: washout and camber shift the minimum
    Kv = lam = wsum = 0.0
    for k, f in enumerate(surfaces(features)):
        if abs(float(f.get("dihedral", 0))) > 60:
            continue
        span, cr, lmb, sw, *_ = _planform(f)
        span *= 2 if f.get("mirror", True) else 1
        Sf = 0.5 * (1 + lmb) * cr * span
        ARf = span**2 / max(Sf, 1e-9)
        kp = (l1.CL_surf[k] - l0.CL_surf[k]) / da * S / max(Sf, 1e-9)
        Kv += max((kp - kp**2 / (np.pi * ARf)) / np.cos(sw), 0.0) * Sf / S
        lam += sw * Sf
        wsum += Sf
    return Derivatives(
        float(CLa), float(K), float(x_ref - Cma / max(CLa, 1e-9) * c), float((lb.Cn - l1.Cn) / db), float((lb.Cl - l1.Cl) / db), float((lb.CY - l1.CY) / db),
        float(l0.CL), float(l0.Cm), float(Kv), float(lam / max(wsum, 1e-9)), [float((x1 - x0) / da) for x0, x1 in zip(l0.CL_surf, l1.CL_surf, strict=True)], sv.L.names,
    )
