r"""Panel and column stability, and thin-walled closed sections.

Plate buckling (long plate, loaded edges of width b, thickness t, Poisson ratio nu):

.. math:: \sigma_{cr}=k_c\frac{\pi^2E}{12(1-\nu^2)}\left(\frac tb\right)^2,\qquad
          \tau_{cr}=k_s\frac{\pi^2E}{12(1-\nu^2)}\left(\frac tb\right)^2

with :math:`k_c=4.0` (all edges simply supported) and :math:`k_s=5.35+4(b/a)^2` (Timoshenko & Gere; Bruhn C5).
Combined compression and shear: :math:`R_c+R_s^2\le1`. Buckling above the proportional limit is capped by the
material allowable (no plastic reduction factor is applied, which is conservative only below yield).

Columns: Euler :math:`\sigma=\pi^2E/(L/\rho)^2` with the Johnson parabola below the transition slenderness.
Closed thin-walled section: Bredt-Batho :math:`J=4A_e^2/\oint ds/t`, shear flow :math:`q=T/(2A_e)`.
"""

from __future__ import annotations

import numpy as np

NU = 0.33


def plate_compression(E: float, t: float, b: float, k: float = 4.0, nu: float = NU) -> float:
    return k * np.pi**2 * E / (12 * (1 - nu**2)) * (t / b) ** 2


def plate_shear(E: float, t: float, b: float, a: float = 1e9, nu: float = NU) -> float:
    bb, aa = min(a, b), max(a, b)
    ks = 5.35 + 4.0 * (bb / aa) ** 2
    return ks * np.pi**2 * E / (12 * (1 - nu**2)) * (t / bb) ** 2


def interaction(sigma: float, sigma_cr: float, tau: float, tau_cr: float) -> float:
    """Buckling margin under combined compression and shear: R_c + R_s^2 = 1 at buckling. Returns MS."""
    rc, rs = max(sigma, 0) / max(sigma_cr, 1e-9), abs(tau) / max(tau_cr, 1e-9)
    if rs < 1e-12:
        return 1 / rc - 1 if rc > 0 else float("inf")
    # find the factor f with f rc + (f rs)^2 = 1
    f = (-rc + np.sqrt(rc * rc + 4 * rs * rs)) / (2 * rs * rs)
    return float(f - 1)


def column(E: float, sigma_y: float, slenderness: float) -> float:
    """Critical stress of a pin-ended column, Euler above and Johnson parabola below the transition."""
    s_tr = np.pi * np.sqrt(2 * E / sigma_y)
    if slenderness >= s_tr:
        return float(np.pi**2 * E / slenderness**2)
    return float(sigma_y * (1 - sigma_y * slenderness**2 / (4 * np.pi**2 * E)))


def closed_section(y: np.ndarray, z: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """Properties of a closed thin-walled polygon (y, z) with wall thickness t per segment (segment k from point k
    to k+1, closing back to point 0). Returns area, centroid, second moments about the centroid, J and A_enclosed."""
    y2, z2 = np.roll(y, -1), np.roll(z, -1)
    ds = np.hypot(y2 - y, z2 - z)
    ym, zm = 0.5 * (y + y2), 0.5 * (z + z2)
    A = float(np.sum(t * ds))
    yc, zc = float(np.sum(t * ds * ym) / A), float(np.sum(t * ds * zm) / A)
    # thin segment second moment: t ds (d_perp^2 terms) with the segment's own inertia ds^2/12
    dz, dy = z2 - z, y2 - y
    Iyy = float(np.sum(t * ds * ((zm - zc) ** 2 + dz**2 / 12)))  # about the y axis (bending in z)
    Izz = float(np.sum(t * ds * ((ym - yc) ** 2 + dy**2 / 12)))
    Ae = 0.5 * abs(float(np.sum(y * z2 - y2 * z)))
    J = 4 * Ae**2 / float(np.sum(ds / t))
    return {"area": A, "y_c": yc, "z_c": zc, "I_bending": Iyy, "I_chordwise": Izz, "J": J, "A_enclosed": Ae, "perimeter": float(ds.sum())}
