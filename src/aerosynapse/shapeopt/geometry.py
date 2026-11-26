r"""Shape input, parameterisation, constraints and export.

A section is 18 numbers: 8 upper + 8 lower CST weights, a leading-edge-modification weight and a
trailing-edge thickness (Kulfan CST with LEM),

.. math:: y(x)=\sqrt{x}(1-x)\sum_{i=0}^{7}w_iB_{i,7}(x)+w_{le}\,x(1-x)^{8.5}\pm x\,t_{te}/2 .

Because y is linear in those numbers, a whole population of designs is a single matrix product,
and any airfoil-like contour can be fitted by linear least squares.

Ordering: this module returns **Selig** order (TE -> upper -> LE -> lower -> TE), which is what
NeuralFoil expects. The panel code in :mod:`aerosynapse.numerics.panel` uses the opposite order,
so use :func:`to_panel_order` before calling it. Mixing them gives wrong numbers without an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError
from aerosynapse.physics.airfoil import cosine_spacing, naca4

FloatArr = NDArray[np.float64]
N_W = 8  # CST weights per side (fixed by the NeuralFoil input layout)
N_SEC = 2 * N_W + 2


def _basis(x: FloatArr) -> tuple[FloatArr, FloatArr]:
    """CST shape basis (nx, 8) and the LEM column (nx,)."""
    k = N_W - 1
    B = np.stack([comb(k, i) * x**i * (1 - x) ** (k - i) for i in range(N_W)], axis=1)
    B = B * (np.sqrt(x) * (1 - x))[:, None]
    return B, x * np.maximum(1 - x, 0.0) ** (N_W + 0.5)


def surfaces(W: FloatArr, x: FloatArr) -> tuple[FloatArr, FloatArr]:
    """Upper and lower surface heights for designs ``W`` (..., 18) at stations ``x``."""
    B, lem = _basis(x)
    W = np.asarray(W, float)
    le, te = W[..., 16:17], W[..., 17:18]
    yu = W[..., :8] @ B.T + le * lem + te * x / 2
    yl = W[..., 8:16] @ B.T + le * lem - te * x / 2
    return yu, yl


def coords(W: FloatArr, n: int = 121) -> FloatArr:
    """Selig-ordered (2n-1, 2) coordinates for one design."""
    x = cosine_spacing(n)
    yu, yl = surfaces(W, x)
    return np.c_[np.r_[x[::-1], x[1:]], np.r_[yu[::-1], yl[1:]]]


def to_panel_order(xy: FloatArr) -> tuple[FloatArr, FloatArr]:
    """Selig -> the TE-lower-LE-upper-TE order used by the Hess-Smith solver."""
    return np.ascontiguousarray(xy[::-1, 0]), np.ascontiguousarray(xy[::-1, 1])


def normalise(xy: FloatArr) -> FloatArr:
    """Translate, de-rotate and scale so that the chord line runs from (0, 0) to (1, 0)."""
    xy = np.asarray(xy, float)
    te = 0.5 * (xy[0] + xy[-1])
    i = int(np.argmax(np.hypot(*(xy - te).T)))
    le = xy[i]
    d = te - le
    c, a = float(np.hypot(*d)), float(np.arctan2(d[1], d[0]))
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return ((xy - le) @ R.T) / c


def fit_cst(xy: FloatArr, n_fit: int = 201) -> tuple[FloatArr, float]:
    """Least-squares CST fit of a Selig-ordered contour. Returns (weights, RMS error / chord).

    A large RMS means that the contour is not airfoil-like (for example a square): such a shape
    cannot be represented here and has to go through the topology stage first.
    """
    xy = normalise(xy)
    i = int(np.argmin(xy[:, 0]))
    up, lo = xy[: i + 1][::-1], xy[i:]
    x = cosine_spacing(n_fit)[1:-1]
    yu = np.interp(x, up[:, 0], up[:, 1])
    yl = np.interp(x, lo[:, 0], lo[:, 1])
    B, lem = _basis(x)
    Z = np.zeros_like(B)
    A = np.r_[np.c_[B, Z, lem, x / 2], np.c_[Z, B, lem, -x / 2]]
    w, *_ = np.linalg.lstsq(A, np.r_[yu, yl], rcond=None)
    return w, float(np.sqrt(np.mean((A @ w - np.r_[yu, yl]) ** 2)))


def superellipse(a: float = 0.5, b: float = 0.5, p: float = 2.0, n: int = 241, centre: tuple[float, float] = (0.0, 0.0)) -> FloatArr:
    """Closed contour |x/a|^p + |y/b|^p = 1 (p = 2 ellipse, large p tends to a rectangle)."""
    t = np.linspace(0, 2 * np.pi, n)
    c, s = np.cos(t), np.sin(t)
    return np.c_[centre[0] + a * np.sign(c) * np.abs(c) ** (2 / p), centre[1] + b * np.sign(s) * np.abs(s) ** (2 / p)]


def load_shape(spec: str) -> FloatArr:
    """Load a shape: ``naca2412``, ``circle``, ``square``, ``ellipse:0.5`` or a coordinate file.

    Coordinate files hold whitespace or comma separated x y rows (Selig ``.dat`` and ``.csv``).
    Lednicer files (upper and lower listed separately, both LE to TE) are detected and reordered.
    """
    s = spec.strip().lower()
    if s.startswith("naca") and s[4:].isdigit() and len(s) == 8:
        X, Y = naca4(s[4:], 121)
        return np.c_[X[::-1], Y[::-1]]
    if s == "circle":
        return superellipse()
    if s == "square":
        return superellipse(p=12.0)
    if s.startswith("ellipse:"):
        return superellipse(0.5, 0.5 * float(s.split(":")[1]))
    path = Path(spec)
    if not path.is_file():
        raise ConfigurationError(f"unknown shape {spec!r}: use nacaXXXX, circle, square, ellipse:r or a file path")
    rows = []
    for line in path.read_text().splitlines():
        parts = line.replace(",", " ").split()
        try:
            v = [float(p) for p in parts[:2]]
        except ValueError:
            continue
        if len(v) == 2:
            rows.append(v)
    xy = np.array(rows, float)
    if len(xy) and xy[0, 0] > 2 and xy[0, 1] > 2:  # Lednicer header row holds the point counts
        nu = int(xy[0, 0])
        up, lo = xy[1 : nu + 1], xy[nu + 1 :]
        xy = np.r_[up[::-1], lo[1:]]
    if len(xy) < 10:
        raise ConfigurationError(f"{spec}: fewer than 10 coordinate rows")
    return xy


def polygon_area(xy: FloatArr) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_mask(polys: list[FloatArr], xc: FloatArr, yc: FloatArr) -> NDArray[np.bool_]:
    """Cells of the grid (xc, yc) that lie inside any polygon. Shape (len(xc), len(yc))."""
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    pts = np.c_[Xc.ravel(), Yc.ravel()]
    m = np.zeros(len(pts), bool)
    for p in polys:
        m |= MplPath(p).contains_points(pts)
    return m.reshape(Xc.shape)


def rotate(xy: FloatArr, deg: float, about: tuple[float, float] = (0.25, 0.0)) -> FloatArr:
    """Rotate nose-up by ``deg`` about a pivot."""
    a = np.deg2rad(-deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = np.asarray(about)
    return (xy - p) @ R.T + p


# ------------------------------------------------------------------ section constraints
_XC = cosine_spacing(161)[1:-1]
_XD = _XC[(_XC > 0.05) & (_XC < 0.95)]


@dataclass
class SectionLimits:
    """Geometric limits, taken from the baseline so that the design stays buildable."""

    t_min: float  # minimum max-thickness / chord
    area_min: float  # minimum cross-section area / chord^2 (fuel and structure volume)
    r_le_min: float  # minimum leading-edge CST weight (radius scales with its square)
    t_aft_min: float  # minimum thickness at x/c = 0.8 (rear spar depth)
    tv_max: float  # maximum total variation of surface slope (waviness)
    local_min: float  # minimum scaled local thickness (surfaces must not cross)

    @classmethod
    def from_baseline(cls, W: FloatArr) -> SectionLimits:
        m = section_metrics(W[None])
        return cls(
            t_min=float(m["t"][0]),
            area_min=0.95 * float(m["area"][0]),
            r_le_min=0.75 * float(m["r_le"][0]),
            t_aft_min=0.60 * float(m["t_aft"][0]),
            tv_max=1.25 * float(m["tv"][0]),
            local_min=0.5 * float(m["t_min_local"][0]),
        )


def section_metrics(W: FloatArr) -> dict[str, FloatArr]:
    """Thickness, area, nose bluntness, rear depth and waviness for a batch (D, 18)."""
    W = np.atleast_2d(W)
    yu, yl = surfaces(W, _XC)
    th = yu - yl
    dyu, dyl = surfaces(W, _XD)
    su = np.diff(dyu, axis=1) / np.diff(_XD)
    sl = np.diff(dyl, axis=1) / np.diff(_XD)
    tv = np.abs(np.diff(su, axis=1)).sum(1) + np.abs(np.diff(sl, axis=1)).sum(1)
    return {
        "t": th.max(1),
        "x_t": _XC[np.argmax(th, axis=1)],
        "area": np.trapezoid(th, _XC, axis=1),
        "r_le": np.minimum(W[:, 0], -W[:, 8]),
        "t_aft": th[:, np.argmin(np.abs(_XC - 0.8))],
        "t_min_local": (th / np.sqrt(_XC) / (1 - _XC + 0.02)).min(1),
        "tv": tv,
    }


def section_violations(W: FloatArr, lim: SectionLimits) -> FloatArr:
    """Normalised constraint values, g <= 0 feasible. Shape (D, 6)."""
    m = section_metrics(W)
    return np.stack(
        [
            1 - m["t"] / lim.t_min,
            1 - m["area"] / lim.area_min,
            1 - m["r_le"] / lim.r_le_min,
            1 - m["t_aft"] / lim.t_aft_min,
            m["tv"] / lim.tv_max - 1,
            1 - m["t_min_local"] / lim.local_min,  # surfaces must not cross
        ],
        axis=1,
    )


# ------------------------------------------------------------------ export
def write_dat(path: Path, xy: FloatArr, name: str) -> None:
    body = "\n".join(f"{x:10.6f} {y:10.6f}" for x, y in xy)
    path.write_text(f"{name}\n{body}\n")


def write_stl(path: Path, verts: FloatArr, faces: NDArray[np.int64], name: str = "wing") -> None:
    """ASCII STL (metres)."""
    out = [f"solid {name}"]
    for a, b, c in faces:
        p, q, r = verts[a], verts[b], verts[c]
        nrm = np.cross(q - p, r - p)
        ln = float(np.linalg.norm(nrm)) or 1.0
        nrm = nrm / ln
        out.append(f"facet normal {nrm[0]:.6e} {nrm[1]:.6e} {nrm[2]:.6e}\n outer loop")
        out += [f"  vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}" for v in (p, q, r)]
        out.append(" endloop\nendfacet")
    out.append(f"endsolid {name}")
    path.write_text("\n".join(out) + "\n")
