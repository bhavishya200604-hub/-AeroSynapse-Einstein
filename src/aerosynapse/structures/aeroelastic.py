r"""Typical-section flutter (Theodorsen aerodynamics, k / V-g method) and torsional divergence.

Two degrees of freedom, plunge h (positive down) and pitch :math:`\alpha` (nose up) about the elastic axis at
:math:`a` semichords aft of mid-chord; static unbalance :math:`x_\alpha`, radius of gyration :math:`r_\alpha`,
frequency ratio :math:`\sigma=\omega_h/\omega_\alpha`, mass ratio :math:`\mu=m/(\pi\rho b^2)`.
Theodorsen's unsteady lift and moment for harmonic motion at reduced frequency :math:`k=\omega b/U`:

.. math:: L=\pi\rho b^2(\ddot h+U\dot\alpha-ba\ddot\alpha)+2\pi\rho UbC(k)\left[\dot h+U\alpha+b(\tfrac12-a)\dot\alpha\right]

.. math:: M=\pi\rho b^2\left[ba\ddot h-Ub(\tfrac12-a)\dot\alpha-b^2(\tfrac18+a^2)\ddot\alpha\right]
          +2\pi\rho Ub^2(a+\tfrac12)C(k)\left[\dot h+U\alpha+b(\tfrac12-a)\dot\alpha\right]

With structural damping g on the stiffness, :math:`K(1+ig)q=\omega^2(M_s+A(k))q`, so for each k the eigenvalues
:math:`\lambda=(1+ig)/\omega^2` give :math:`\omega` and the damping g needed for neutral motion. Flutter is where g
of a branch crosses zero upward. Check: Hodges & Pierce's case (:math:`\mu=20, a=-0.2, x_\alpha=0.1,
r_\alpha^2=0.24, \sigma=0.4`) gives :math:`U_F/(b\omega_\alpha)\approx2.2`.

Divergence of a typical section: :math:`q_D=k_\alpha/(S\,e\,C_{L\alpha})` with e the distance of the aerodynamic
centre ahead of the elastic axis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import hankel2


def theodorsen(k: float) -> complex:
    h0, h1 = hankel2(0, k), hankel2(1, k)
    return complex(h1 / (h1 + 1j * h0))


def flutter(mu: float, a: float, x_a: float, r_a2: float, sigma: float, k_list: np.ndarray | None = None) -> dict[str, Any]:
    """Non-dimensional flutter speed U_F/(b omega_alpha) and frequency ratio omega_F/omega_alpha (b = 1, omega_alpha = 1,
    rho pi b^2 = 1 so that the plunge mass is mu)."""
    ks = np.geomspace(3.0, 0.02, 400) if k_list is None else np.asarray(k_list)
    b, rho = 1.0, 1.0 / np.pi
    m, S, I = mu, mu * x_a, mu * r_a2
    Ms = np.array([[m, S], [S, I]], dtype=complex)
    Ks = np.diag([m * sigma**2, I]).astype(complex)
    branches = []
    for k in ks:
        C = theodorsen(k)
        pr = np.pi * rho
        Lh = pr * b**2 * (-1) + 2 * pr * b**2 * C * 1j / k
        La = pr * b**2 * (1j * b / k + b * a) + 2 * pr * b**2 * C / k * (b / k + 1j * b * (0.5 - a))
        Mh = -pr * b**3 * a + 2 * pr * b**3 * (a + 0.5) * C * 1j / k
        Ma = pr * b**4 * (-1j * (0.5 - a) / k + (1 / 8 + a * a)) + 2 * pr * b**3 * (a + 0.5) * C / k * (b / k + 1j * b * (0.5 - a))
        A = np.array([[-Lh, -La], [Mh, Ma]])
        lam = np.linalg.eigvals(np.linalg.solve(Ks, Ms + A))
        lam = lam[np.argsort(-lam.real)]
        w = 1 / np.sqrt(np.maximum(lam.real, 1e-12))
        g = lam.imag / lam.real
        branches.append([(float(k), float(wi * b / k), float(wi), float(gi)) for wi, gi in zip(w, g, strict=True)])
    br = np.array(branches)  # (nk, 2, 4): k, U, omega, g
    best = None
    for j in range(br.shape[1]):
        g, U, w = br[:, j, 3], br[:, j, 1], br[:, j, 2]
        for i in range(1, len(g)):
            if g[i - 1] < 0 <= g[i]:
                f = -g[i - 1] / (g[i] - g[i - 1])
                Uf, wf = U[i - 1] + f * (U[i] - U[i - 1]), w[i - 1] + f * (w[i] - w[i - 1])
                if best is None or Uf < best[0]:
                    best = (float(Uf), float(wf))
                break
    return {"U_flutter": best[0] if best else float("inf"), "omega_ratio": best[1] if best else float("nan"),
            "vg": {"U": br[:, :, 1].tolist(), "g": br[:, :, 3].tolist(), "omega": br[:, :, 2].tolist()}}


def divergence_q(k_alpha: float, S: float, e: float, cl_alpha: float) -> float:
    """Dynamic pressure of torsional divergence; infinite if the aerodynamic centre is behind the elastic axis."""
    return float(k_alpha / (S * e * cl_alpha)) if e > 0 else float("inf")
