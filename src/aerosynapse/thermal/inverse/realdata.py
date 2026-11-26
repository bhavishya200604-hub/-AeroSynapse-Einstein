"""Phase 10.1: parameter identification from REAL measured temperatures (identify before you fit).

Data
  load_uscrn       NOAA U.S. Climate Reference Network hourly product (soil temperature / moisture at 5-100 cm)
  load_table       any CSV / whitespace table: a time column and temperature columns at known positions (your data)
  window           a time window with short gaps interpolated and long gaps reported

Models
  Conduction1D     T_t = d/dz (alpha(z) dT/dz) between two sensors whose measured series are imposed (Dirichlet);
                   the inner sensors are the data. alpha homogeneous or piecewise constant in layers whose edges lie
                   on cell faces; optional constant calibration offset per inner sensor. Crank-Nicolson in time with
                   a Rannacher start (the first two steps as four implicit-Euler half-steps), central differences in
                   space; verified against the exact damped-wave solution (verify_wave).
  Lumped           Newton cooling dT/dt = -(T - T_amb(t))/tau + s, exact exponential integrator for piecewise-linear
                   T_amb; tau = rho c V/(h A) gives h when the user supplies rho, c and V/A (valid for Bi << 1).

Derivatives
  Complex-step: every forward model runs in complex arithmetic, so dr/dp = Im r(p + i h e_k)/h with h = 1e-20 is
  exact to round-off (no subtraction, no step-size choice) and is differentiated THROUGH the implicit solver.
  Cross-checked against central differences (tests/unit/test_realdata.py).

Estimation (fit)
  1. identifiability: singular values of the scaled Jacobian at the initial guess; rank < number of unknowns means
     the data cannot separate some combination, which is then reported (and fitted in the identifiable subspace);
  2. Levenberg-Marquardt / trust-region least squares in log-parameters (scipy least_squares) with the exact
     Jacobian;
  3. convergence is judged by the scale-free stationarity measure max_k |J_k . r| / (|J_k| |r|) (the cosine between
     the residual and each sensitivity), not only by the optimiser's status; if the fit does not converge, the
     strategies of the configuration are tried in order: identifiability subspace, multistart (Sobol points in the
     log-bounds), continuation (window homotopy: fit a short window, extend), regularisation (Tikhonov on layer
     contrasts, then released), robust loss. Every attempt is logged.
Uncertainty
  Gauss-Newton covariance s^2 (J^T J)^-1, and the Newey-West (HAC) sandwich covariance with Bartlett weights over
  the time lags, because real residuals are autocorrelated (diurnal model error); intervals are reported for both.
Classical cross-check
  harmonic: amplitude-damping and phase-lag estimates of alpha from the first diurnal harmonic between two depths
  (semi-infinite homogeneous medium with periodic surface forcing).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_banded
from scipy.optimize import least_squares
from scipy.stats import qmc

CS_H = 1e-20  # complex-step size
DAY = 86400.0


# ============================================================================ data
@dataclass
class Series:
    """Measured temperatures: times t [s from t_ref], positions z [m], T (n_t, n_z) in degC (NaN = missing)."""

    t: np.ndarray
    z: np.ndarray
    T: np.ndarray
    stamp: list[str] = field(default_factory=list)  # ISO time of each row
    moisture: np.ndarray | None = None  # (n_t, n_z) m^3/m^3 where available
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, float)
        self.z = np.asarray(self.z, float)
        self.T = np.asarray(self.T, float)


def _uscrn_columns(headers: Path | None) -> list[str]:
    if headers and Path(headers).exists():
        lines = Path(headers).read_text(encoding="utf-8").splitlines()
        return lines[1].split()
    return (
        "WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE T_CALC T_HR_AVG T_MAX T_MIN P_CALC "
        "SOLARAD SOLARAD_FLAG SOLARAD_MAX SOLARAD_MAX_FLAG SOLARAD_MIN SOLARAD_MIN_FLAG SUR_TEMP_TYPE SUR_TEMP "
        "SUR_TEMP_FLAG SUR_TEMP_MAX SUR_TEMP_MAX_FLAG SUR_TEMP_MIN SUR_TEMP_MIN_FLAG RH_HR_AVG RH_HR_AVG_FLAG "
        "SOIL_MOISTURE_5 SOIL_MOISTURE_10 SOIL_MOISTURE_20 SOIL_MOISTURE_50 SOIL_MOISTURE_100 SOIL_TEMP_5 SOIL_TEMP_10 "
        "SOIL_TEMP_20 SOIL_TEMP_50 SOIL_TEMP_100"
    ).split()


def load_uscrn(path: str | Path, headers: str | Path | None = None) -> Series:
    """One station-year of the USCRN hourly02 product. Time = middle of the averaging hour (the file stamps the END
    of the hour), in seconds from 1 January 00:00 UTC of the file's first year; -9999 / -99 are missing."""
    import pandas as pd

    cols = _uscrn_columns(Path(headers) if headers else Path(path).parents[1] / "headers.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols, dtype={"UTC_DATE": str, "UTC_TIME": str})
    stamp = pd.to_datetime(df["UTC_DATE"] + df["UTC_TIME"].str.zfill(4), format="%Y%m%d%H%M", utc=True)
    stamp = stamp - pd.Timedelta(minutes=30)
    t_ref = pd.Timestamp(year=int(stamp.iloc[0].year), month=1, day=1, tz="UTC")
    t = (stamp - t_ref).dt.total_seconds().to_numpy()
    depths = [5, 10, 20, 50, 100]
    T = df[[f"SOIL_TEMP_{d}" for d in depths]].to_numpy(float)
    M = df[[f"SOIL_MOISTURE_{d}" for d in depths]].to_numpy(float)
    T[T < -900] = np.nan
    M[(M < -9) | (M > 1)] = np.nan
    meta = {
        "source": "NOAA USCRN hourly02",
        "file": Path(path).name,
        "wban": str(df["WBANNO"].iloc[0]),
        "latitude": float(df["LATITUDE"].iloc[0]),
        "longitude": float(df["LONGITUDE"].iloc[0]),
        "t_ref": t_ref.isoformat(),
        "air_temperature": df["T_HR_AVG"].where(df["T_HR_AVG"] > -900).to_numpy(float),
    }
    return Series(t, np.array(depths) / 100.0, T, [s.isoformat() for s in stamp], M, meta)


def load_table(
    src: str | Path,
    time_col: str,
    positions: dict[str, float],
    time_unit: str = "auto",
    ambient_col: str | None = None,
) -> Series:
    """A CSV (or whitespace) table of your own: one time column (seconds, hours, or ISO date-times) and one column of
    temperatures per position (positions = {column name: position in m}); optional ambient-temperature column.
    src is a file (Path) or the table text itself (str). A str is never looked up on disk: an uploaded table can
    therefore never name a file on the server, and long text cannot hit the operating system's path-length limit."""
    import pandas as pd

    text = src.read_text(encoding="utf-8") if isinstance(src, Path) else str(src)
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    for c in [time_col, *positions] + ([ambient_col] if ambient_col else []):
        if c not in df.columns:
            raise KeyError(f"column {c!r} not in the table (columns: {list(df.columns)})")
    raw = df[time_col]
    if time_unit == "auto":
        time_unit = "s" if pd.api.types.is_numeric_dtype(raw) else "datetime"
    if time_unit == "datetime":
        st = pd.to_datetime(raw, utc=True)
        t = (st - st.iloc[0]).dt.total_seconds().to_numpy()
        stamp = [s.isoformat() for s in st]
    else:
        scale = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": DAY}[time_unit]
        t = raw.to_numpy(float) * scale
        stamp = [f"{v:g} s" for v in t]
    names = sorted(positions, key=lambda c: positions[c])
    T = df[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    meta = {"source": "user table", "columns": names}
    if ambient_col:
        meta["ambient"] = pd.to_numeric(df[ambient_col], errors="coerce").to_numpy(float)
    return Series(t, np.array([positions[c] for c in names]), T, stamp, None, meta)


def fill_gaps(y: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over runs of at most max_gap missing samples (interior runs only)."""
    y = y.copy()
    bad = np.isnan(y)
    if not bad.any():
        return y
    idx = np.arange(len(y))
    run_start = None
    for i in range(len(y) + 1):
        b = i < len(y) and bad[i]
        if b and run_start is None:
            run_start = i
        elif not b and run_start is not None:
            if i - run_start <= max_gap and run_start > 0 and i < len(y):
                y[run_start:i] = np.interp(idx[run_start:i], [run_start - 1, i], [y[run_start - 1], y[i]])
            run_start = None
    return y


def window(s: Series, start: float, days: float, max_gap_h: int = 3, cols: list[int] | None = None) -> dict:
    """Rows with start <= t < start + days, the chosen columns gap-filled; valid fraction and the filled count."""
    cols = list(range(len(s.z))) if cols is None else cols
    sel = (s.t >= start - 1e-6) & (s.t < start + days * DAY - 1e-6)
    t = s.t[sel]
    T = s.T[sel][:, cols]
    raw_missing = int(np.isnan(T).sum())
    dt = np.median(np.diff(t)) if len(t) > 1 else 3600.0
    Tf = np.stack([fill_gaps(T[:, j], int(round(max_gap_h * 3600 / dt))) for j in range(T.shape[1])], 1)
    out = {
        "t": t,
        "z": s.z[cols],
        "T": Tf,
        "valid_fraction": float(np.isfinite(Tf).all(1).mean()) if len(t) else 0.0,
        "filled": raw_missing - int(np.isnan(Tf).sum()),
        "stamp": [s.stamp[i] for i in np.flatnonzero(sel)] if s.stamp else [],
    }
    if s.moisture is not None:
        out["moisture"] = s.moisture[sel][:, cols]
    return out


def iso_to_seconds(s: Series, iso: str) -> float:
    """Seconds from the series' reference time to an ISO date (UTC)."""
    import pandas as pd

    ref = pd.Timestamp(s.meta.get("t_ref", s.stamp[0] if s.stamp else "1970-01-01"))
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    t = pd.Timestamp(iso)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return float((t - ref).total_seconds())


# ============================================================================ forward model: conduction
class Conduction1D:
    """Heat conduction between two sensors (measured Dirichlet data), predicting the inner sensors."""

    def __init__(
        self,
        t_obs: np.ndarray,
        z_obs: np.ndarray,
        T_obs: np.ndarray,
        top: float,
        bottom: float,
        inner: list[float],
        dz: float = 0.0025,
        dt: float = 600.0,
        layers: int = 1,
        spinup_s: float = DAY,
        layer_edges: list[float] | None = None,
    ):
        z_obs = np.asarray(z_obs, float)

        def col(z):
            k = int(np.argmin(np.abs(z_obs - z)))
            if abs(z_obs[k] - z) > 1e-9:
                raise ValueError(f"no sensor at {z} m (sensors: {z_obs.tolist()})")
            return k

        self.top, self.bottom, self.inner = float(top), float(bottom), [float(v) for v in inner]
        self.ktop, self.kbot, self.kin = col(top), col(bottom), [col(z) for z in inner]
        self.t_obs = np.asarray(t_obs, float) - float(t_obs[0])
        self.T_obs = np.asarray(T_obs, float)
        self.N = int(round((self.bottom - self.top) / dz))
        self.dz = (self.bottom - self.top) / self.N
        self.zn = self.top + self.dz * np.arange(self.N + 1)
        self.node_in = []
        for z in self.inner:
            j = (z - self.top) / self.dz
            if abs(j - round(j)) > 1e-6:
                raise ValueError(f"sensor {z} m is not on a grid node; choose dz dividing the sensor spacing")
            self.node_in.append(int(round(j)))
        obs_dt = float(np.median(np.diff(self.t_obs)))
        self.sub = max(1, int(round(obs_dt / dt)))
        self.dt = obs_dt / self.sub
        self.n_steps = (len(self.t_obs) - 1) * self.sub
        tm = np.arange(self.n_steps + 1) * self.dt
        self.g0 = np.interp(tm, self.t_obs, self.T_obs[:, self.ktop])
        self.gN = np.interp(tm, self.t_obs, self.T_obs[:, self.kbot])
        prof = [k for k in range(len(z_obs)) if self.top - 1e-9 <= z_obs[k] <= self.bottom + 1e-9]
        zz, TT = z_obs[prof], self.T_obs[0, prof]
        if np.isnan(TT).any():
            raise ValueError("missing sensor value at the start of the window (initial profile)")
        self.T0 = PchipInterpolator(zz, TT)(self.zn)
        # layers: explicit interior edges [m] or `layers` of equal thickness; a face belongs to the layer of its centre
        zf = self.top + (np.arange(self.N) + 0.5) * self.dz
        if layer_edges:
            edges = np.sort(np.asarray(layer_edges, float))
            if np.any(edges <= self.top) or np.any(edges >= self.bottom):
                raise ValueError(f"layer edges must lie inside ({self.top}, {self.bottom}) m")
        else:
            edges = np.linspace(self.top, self.bottom, int(layers) + 1)[1:-1]
        self.layer_edges = edges.tolist()
        self.layers = len(edges) + 1
        self.n_log = self.layers  # leading log-parameters of the unknown vector
        self.face_layer = np.searchsorted(edges, zf)
        self.uniform = bool(np.allclose(np.diff(self.t_obs), obs_dt, rtol=1e-6, atol=1e-6))
        self.scored = self.t_obs >= spinup_s - 1e-6
        self.data = self.T_obs[:, self.kin]

    def faces(self, alpha: np.ndarray) -> np.ndarray:
        return np.asarray(alpha)[self.face_layer]

    def at(self, alpha, positions) -> np.ndarray:
        """Model temperatures at arbitrary positions (m) at the observation times: cubic Lagrange interpolation of the
        nodal profile (smooth in the position, so complex-step derivatives with respect to positions are exact)."""
        prof = self.run(alpha, full=True)  # (n_obs, N + 1) including the boundary nodes
        out = []
        for z in positions:
            u = (z - self.top) / self.dz
            j = int(np.clip(np.floor(np.real(u)), 1, self.N - 2))
            x = u - j  # local coordinate, nodes at -1, 0, 1, 2
            w = [
                -x * (x - 1) * (x - 2) / 6,
                (x + 1) * (x - 1) * (x - 2) / 2,
                -(x + 1) * x * (x - 2) / 2,
                (x + 1) * x * (x - 1) / 6,
            ]
            out.append(sum(wk * prof[:, j - 1 + k] for k, wk in enumerate(w)))
        return np.stack(out, 1)

    def run(self, alpha, full: bool = False) -> np.ndarray:
        """Model temperatures at the inner sensors at the observation times (complex if alpha is complex); with
        full=True the whole nodal profile including the boundary nodes.

        The scheme is linear and time-invariant and the boundary data are linear in time between observations, so
        one observation interval of `sub` Crank-Nicolson steps is an exact affine map
            T_{k+1} = P T_k + a0 g0_k + a1 g0_{k+1} + b0 gN_k + b1 gN_{k+1},
        built once per parameter value; the Rannacher start-up is stepped explicitly. Identical to run_stepwise up to
        round-off (tests/unit/test_realdata.py), at a fraction of the cost."""
        if not self.uniform:
            return self.run_stepwise(alpha, full=full)
        f = self.faces(np.atleast_1d(alpha)) / self.dz**2
        dtype = np.result_type(f, float)
        M = self.N - 1
        A = np.zeros((M, M), dtype)
        idx = np.arange(M)
        A[idx, idx] = -(f[1:] + f[:-1])
        A[idx[:-1], idx[:-1] + 1] = f[1:-1]
        A[idx[1:], idx[1:] - 1] = f[1:-1]
        Id = np.eye(M)
        Acn = Id - 0.5 * self.dt * A
        rhs = np.concatenate([Id + 0.5 * self.dt * A, np.zeros((M, 2), dtype)], 1)
        rhs[0, M] = 0.5 * self.dt * f[0]
        rhs[-1, M + 1] = 0.5 * self.dt * f[-1]
        sol = np.linalg.solve(Acn, rhs)
        S, e0, eN = sol[:, :M], sol[:, M], sol[:, M + 1]
        P = np.eye(M, dtype=dtype)
        a0, a1, b0, b1 = (np.zeros(M, dtype) for _ in range(4))
        for s_ in range(self.sub):
            wn, wn1 = 1 - s_ / self.sub, 1 - (s_ + 1) / self.sub  # weights of the earlier observation
            P = S @ P
            a0, a1 = S @ a0 + e0 * (wn + wn1), S @ a1 + e0 * (2 - wn - wn1)
            b0, b1 = S @ b0 + eN * (wn + wn1), S @ b1 + eN * (2 - wn - wn1)
        k0 = -(-2 // self.sub)  # observation intervals covered by the two Rannacher start-up steps
        head = self.run_stepwise(alpha, n_steps=k0 * self.sub, return_state=True, full=full)
        out, T = head
        G0 = self.g0[:: self.sub]
        GN = self.gN[:: self.sub]
        rows = np.array(self.node_in) - 1
        for k in range(k0, len(self.t_obs) - 1):
            T = P @ T + a0 * G0[k] + a1 * G0[k + 1] + b0 * GN[k] + b1 * GN[k + 1]
            if full:
                out[k + 1, 1:-1] = T
            else:
                out[k + 1] = T[rows]
        return out

    def run_stepwise(self, alpha, n_steps: int | None = None, return_state: bool = False, full: bool = False):
        """Reference implementation: every Crank-Nicolson step solved as a tridiagonal system (also the fallback for
        irregular observation times). With n_steps it stops early and can return the state."""
        f = self.faces(np.atleast_1d(alpha)) / self.dz**2
        dtype = np.result_type(f, float)
        M = self.N - 1
        diag = -(f[1:] + f[:-1])
        up = f[1:-1]

        def ab(theta, h):
            a = np.zeros((3, M), dtype)
            a[0, 1:] = -theta * h * up
            a[1, :] = 1.0 - theta * h * diag
            a[2, :-1] = -theta * h * up
            return a

        def apply(T):
            r = diag * T
            r[:-1] += up * T[1:]
            r[1:] += up * T[:-1]
            return r

        def bvec(n):
            b = np.zeros(M, dtype)
            b[0] += f[0] * self.g0[n]
            b[-1] += f[-1] * self.gN[n]
            return b

        T = self.T0[1:-1].astype(dtype)
        if full:
            out = np.zeros((len(self.t_obs), self.N + 1), dtype)
            out[:, 0] = self.g0[:: self.sub][: len(self.t_obs)]
            out[:, -1] = self.gN[:: self.sub][: len(self.t_obs)]
            out[0] = self.T0
        else:
            out = np.empty((len(self.t_obs), len(self.node_in)), dtype)
            out[0] = self.T0[self.node_in]
        A_ie = ab(1.0, self.dt / 2)
        A_cn = ab(0.5, self.dt)
        n = 0
        for step in range(self.n_steps if n_steps is None else n_steps):
            if step < 2:  # Rannacher start: two implicit-Euler half-steps per step
                for half in (0.5, 1.0):
                    tb = n + half
                    b = (
                        f[0] * np.interp(tb, [n, n + 1], self.g0[n : n + 2]),
                        f[-1] * np.interp(tb, [n, n + 1], self.gN[n : n + 2]),
                    )
                    rhs = T.copy()
                    rhs[0] += self.dt / 2 * b[0]
                    rhs[-1] += self.dt / 2 * b[1]
                    T = solve_banded((1, 1), A_ie, rhs, check_finite=False)
            else:
                rhs = T + 0.5 * self.dt * (apply(T) + bvec(n) + bvec(n + 1))
                T = solve_banded((1, 1), A_cn, rhs, check_finite=False)
            n += 1
            if n % self.sub == 0:
                if full:
                    out[n // self.sub, 1:-1] = T
                else:
                    out[n // self.sub] = T[np.array(self.node_in) - 1]
        return (out, T) if return_state else out


def verify_wave(
    alpha: float, amplitude: float, period_s: float, top: float, bottom: float, dz: float, dt: float, days: float = 3.0
) -> float:
    """Max error at the inner nodes of Conduction1D against the exact damped wave
    T = A exp(-z/d) sin(w t - z/d), d = sqrt(2 alpha / w), which solves T_t = alpha T_zz exactly."""
    w = 2 * np.pi / period_s
    d = np.sqrt(2 * alpha / w)
    t = np.arange(0, days * DAY + 1, 3600.0)
    zs = np.array([top, (top + bottom) / 2, bottom])
    exact = amplitude * np.exp(-zs[None] / d) * np.sin(w * t[:, None] - zs[None] / d)
    m = Conduction1D(t, zs, exact, top, bottom, [zs[1]], dz=dz, dt=dt, layers=1, spinup_s=0.0)
    # exact initial profile on the grid (the PCHIP through three sensors would add an interpolation error)
    m.T0 = amplitude * np.exp(-m.zn / d) * np.sin(-m.zn / d)
    # exact boundary data at every model time (linear interpolation of hourly data would add a time error)
    tm = np.arange(m.n_steps + 1) * m.dt
    m.g0 = amplitude * np.exp(-top / d) * np.sin(w * tm - top / d)
    m.gN = amplitude * np.exp(-bottom / d) * np.sin(w * tm - bottom / d)
    # step by step: the exact boundary values at every model time (the fast propagator interpolates them linearly
    # between observations, which is what real hourly data need, but would add an interpolation floor here)
    model = m.run_stepwise(alpha).real[:, 0]
    return float(np.max(np.abs(model - exact[:, 1])))


class FreezingConduction1D(Conduction1D):
    """Conduction with freezing / thawing of soil water by the apparent-heat-capacity method:

        c(T) dT/dt = d/dz (alpha dT/dz),   c(T) = 1 + S phi(T),   phi = Gaussian of width w centred at T_f,

    where S [K] = (latent heat of the pore water per volume) / (volumetric heat capacity) is the "latent temperature"
    released over the freezing range (integral of c - 1 over T). S = 0 recovers Conduction1D exactly. Unknowns
    (log alpha, log S); T_f and w are configuration inputs. Linearly implicit Crank-Nicolson: the capacity is taken at
    a predicted mid-step temperature (one predictor-corrector pass), so each step is one tridiagonal solve per pass;
    Rannacher start as in Conduction1D. Smooth in T, so complex-step derivatives remain exact."""

    def __init__(self, *args, t_freeze: float = 0.0, width: float = 0.5, **kw):
        super().__init__(*args, **kw)
        if self.layers != 1:
            raise ValueError("the freezing model is homogeneous (one diffusivity)")
        self.t_freeze, self.width = float(t_freeze), float(width)
        self.n_log = 2

    def capacity(self, T, S, tf=None):
        tf = self.t_freeze if tf is None else tf
        return 1.0 + S * np.exp(-0.5 * ((T - tf) / self.width) ** 2) / (self.width * np.sqrt(2 * np.pi))

    def run(self, params, full: bool = False) -> np.ndarray:
        alpha, S = params[0], params[1]
        tf = params[2] if len(params) > 2 else self.t_freeze  # optional unknown freezing point [degC]
        f = alpha / self.dz**2 * np.ones(self.N, dtype=np.result_type(alpha, float))
        dtype = np.result_type(f, S, float)
        M = self.N - 1
        diag = -2.0 * f[1:]
        up = f[1:-1]

        from scipy.linalg import lapack

        gtsv = lapack.zgtsv if np.issubdtype(dtype, np.complexfloating) else lapack.dgtsv
        off = {(th, h): (-th * h * up).astype(dtype) for th, h in ((1.0, self.dt / 2), (0.5, self.dt))}

        def tri(c, theta, h, rhs):  # (diag(c) - theta h A) x = rhs, LAPACK gtsv (no banded-matrix copies)
            o = off[(theta, h)]
            x = gtsv(o.copy(), (c - theta * h * diag).astype(dtype), o.copy(), rhs.astype(dtype))
            if x[4] != 0:
                raise np.linalg.LinAlgError(f"gtsv info {x[4]}")
            return x[3]

        def apply(T):
            r = diag * T
            r[:-1] += up * T[1:]
            r[1:] += up * T[:-1]
            return r

        def gb(n):  # boundary values at (fractional) step n: linear between the two neighbouring steps
            k = min(int(n), self.n_steps - 1)
            w = n - k
            return (
                (1 - w) * self.g0[k] + w * self.g0[k + 1],
                (1 - w) * self.gN[k] + w * self.gN[k + 1],
            )

        def bvec(n):
            g0, gN = gb(n)
            b = np.zeros(M, dtype)
            b[0] += f[0] * g0
            b[-1] += f[-1] * gN
            return b

        T = self.T0[1:-1].astype(dtype)
        rows = np.array(self.node_in) - 1
        if full:
            out = np.zeros((len(self.t_obs), self.N + 1), dtype)
            out[:, 0] = self.g0[:: self.sub][: len(self.t_obs)]
            out[:, -1] = self.gN[:: self.sub][: len(self.t_obs)]
            out[0] = self.T0
        else:
            out = np.empty((len(self.t_obs), len(self.node_in)), dtype)
            out[0] = self.T0[self.node_in]
        n = 0
        for step in range(self.n_steps):
            if step < 2:  # Rannacher start-up: implicit-Euler half-steps
                for half in (0.5, 1.0):
                    c = self.capacity(T, S, tf)
                    rhs = c * T + self.dt / 2 * bvec(n + half)
                    T = tri(c, 1.0, self.dt / 2, rhs)
            else:
                c0 = self.capacity(T, S, tf)
                b = bvec(n) + bvec(n + 1)
                pred = tri(c0, 0.5, self.dt, c0 * T + 0.5 * self.dt * (apply(T) + b))
                c = self.capacity(0.5 * (T + pred), S, tf)  # corrector: capacity at the mid-step temperature
                T = tri(c, 0.5, self.dt, c * T + 0.5 * self.dt * (apply(T) + b))
            n += 1
            if n % self.sub == 0:
                if full:
                    out[n // self.sub, 1:-1] = T
                else:
                    out[n // self.sub] = T[rows]
        return out

    def run_stepwise(self, alpha, n_steps=None, return_state=False, full=False):  # one scheme only
        return self.run(alpha, full=full)


# ============================================================================ forward model: lumped cooling
class Lumped:
    """Newton cooling dT/dt = -(T - T_amb(t))/tau + s of one measured temperature series. Unknowns (log tau,
    optionally T_amb if no ambient series, T0, s); exact exponential integrator for piecewise-linear T_amb."""

    def __init__(self, t: np.ndarray, T: np.ndarray, ambient: np.ndarray | None = None, fit_source: bool = False):
        self.t = np.asarray(t, float) - float(t[0])
        self.data = np.asarray(T, float).reshape(-1, 1)
        self.ambient = None if ambient is None else np.asarray(ambient, float)
        self.fit_source = fit_source
        self.names = (
            ["log_tau"] + ([] if self.ambient is not None else ["T_amb"]) + ["T0"] + (["source"] if fit_source else [])
        )
        self.scored = np.ones(len(self.t), bool)

    def run(self, p) -> np.ndarray:
        p = list(p)
        tau = np.exp(p[0])
        k = 1
        if self.ambient is None:
            amb = p[k] + 0 * self.t
            k += 1
        else:
            amb = self.ambient
        T0 = p[k]
        s = p[k + 1] if self.fit_source else 0.0
        T = [T0 + 0 * tau]
        for i in range(1, len(self.t)):
            h = self.t[i] - self.t[i - 1]
            a0, a1 = amb[i - 1], amb[i]
            slope = (a1 - a0) / h
            e = np.exp(-h / tau)
            # exact solution over the step for T_amb linear in time and constant source s
            Tp_end = a1 - tau * slope + tau * s
            Tp_start = a0 - tau * slope + tau * s
            T.append(Tp_end + (T[-1] - Tp_start) * e)
        return np.array(T).reshape(-1, 1)


# ============================================================================ estimation
@dataclass
class Problem:
    """A model, the unknowns and how they map to the model's parameters."""

    model: object
    names: list[str]
    x0: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    offsets: bool = False
    tikhonov: float = 0.0
    prior: np.ndarray | None = None
    depths: bool = False  # unknown depth shifts of the inner sensors (m)

    def predict(self, x) -> np.ndarray:
        m = self.model
        if isinstance(m, Conduction1D):
            x = np.asarray(x)
            na, ni = m.n_log, len(m.inner)
            k = na
            if isinstance(m, FreezingConduction1D):
                # (log alpha, log S) and, if the freezing point is unknown, T_f in degC (linear)
                return m.run([*np.exp(x[:2]), *x[2:3]])
            if self.depths:
                out = m.at(np.exp(x[:na]), [z + d for z, d in zip(m.inner, x[k : k + ni], strict=True)])
                k += ni
            else:
                out = m.run(np.exp(x[:na]))
            if self.offsets:
                out = out + x[k : k + ni][None, :]
            return out
        return m.run(x)

    def residual(self, x) -> np.ndarray:
        m = self.model
        P = self.predict(x)
        r = (P - m.data)[m.scored]
        r = r[np.isfinite(m.data[m.scored])]
        if self.tikhonov and isinstance(m, Conduction1D) and m.layers > 1:
            la = np.asarray(x[: m.layers])
            r = np.concatenate([r, np.sqrt(self.tikhonov) * np.diff(la)])
        return r

    def jacobian(self, x) -> np.ndarray:
        """Exact Jacobian by the complex-step method (each column one complex forward solve)."""
        x = np.asarray(x, float)
        cols = []
        for k in range(len(x)):
            if self.offsets and isinstance(self.model, Conduction1D) and k >= len(x) - len(self.model.inner):
                # offsets enter linearly: derivative is an indicator, no solve needed
                m = self.model
                ind = np.zeros_like(m.data)
                ind[:, k - (len(x) - len(m.inner))] = 1.0
                c = ind[m.scored][np.isfinite(m.data[m.scored])]
                if self.tikhonov and m.layers > 1:
                    c = np.concatenate([c, np.zeros(m.layers - 1)])
                cols.append(c)
                continue
            xc = x.astype(complex)
            xc[k] += 1j * CS_H
            cols.append(np.imag(self.residual(xc)) / CS_H)
        return np.stack(cols, 1)


def stationarity(J: np.ndarray, r: np.ndarray) -> float:
    """max_k |J_k . r| / (|J_k| |r|): the cosine between the residual and each sensitivity; 0 at a stationary point
    of the least-squares cost, scale-free (independent of units and of the size of the residual)."""
    nr = np.linalg.norm(r)
    if nr == 0:
        return 0.0
    nj = np.linalg.norm(J, axis=0)
    nj[nj == 0] = np.inf
    return float(np.max(np.abs(J.T @ r) / (nj * nr)))


def identifiability(J: np.ndarray, names: list[str], tol: float = 1e-8) -> dict:
    """Rank and conditioning of the column-scaled Jacobian (Fisher information up to the noise level)."""
    sc = np.linalg.norm(J, axis=0)
    sc[sc == 0] = 1.0
    s, V = np.linalg.svd(J / sc, full_matrices=False)[1:]
    rank = int((s > tol * s[0]).sum()) if s.size and s[0] > 0 else 0
    return {
        "singular_values": s.tolist(),
        "rank": rank,
        "n": len(names),
        "condition": float(s[0] / s[-1]) if s.size and s[-1] > 0 else float("inf"),
        "null_directions": [dict(zip(names, map(float, V[i]), strict=True)) for i in range(rank, len(names))],
    }


def _lsq(prob: Problem, x0, tol, loss="linear", max_nfev=200):
    return least_squares(
        prob.residual,
        np.clip(x0, prob.lb + 1e-12, prob.ub - 1e-12),
        jac=prob.jacobian,
        bounds=(prob.lb, prob.ub),
        method="trf",
        x_scale="jac",
        ftol=1e-15,
        xtol=1e-15,
        gtol=tol,
        loss=loss,
        max_nfev=max_nfev,
    )


def assess(prob: Problem, res, stat_tol: float = 1e-6) -> dict:
    J = prob.jacobian(res.x)
    r = prob.residual(res.x)
    st = stationarity(J, r)
    span = prob.ub - prob.lb
    at_bound = bool(np.any((res.x - prob.lb < 1e-6 * span) | (prob.ub - res.x < 1e-6 * span)))
    return {
        "cost": float(0.5 * r @ r),
        "rmse": float(np.sqrt(np.mean(r**2))) if r.size else float("nan"),
        "status": int(res.status),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "stationarity": st,
        "at_bound": at_bound,
        "converged": bool(res.status > 0 and st <= stat_tol and not at_bound),
        "x": res.x.tolist(),
    }


def fit(prob: Problem, cfg: dict, plain: bool = False, continuation_model=None) -> dict:
    """Identify, fit, and if the fit has not converged try the configured strategies in order (unless plain)."""
    tol = float(cfg.get("tol", 1e-10))
    loss = cfg.get("robust_loss", "linear")
    log: list[dict] = []
    J0 = prob.jacobian(prob.x0)
    ident = identifiability(J0, prob.names)
    res = _lsq(prob, prob.x0, tol, loss)
    a = assess(prob, res)
    log.append({"strategy": "levenberg-marquardt", "init": prob.x0.tolist(), **a})
    best = (a, res)
    if not plain and not a["converged"]:
        for strat in cfg.get("strategies", []):
            if best[0]["converged"]:
                break
            if strat == "identifiability" and ident["rank"] < len(prob.names):
                # fit in the identifiable subspace: the null directions stay at the initial guess
                sc = np.linalg.norm(J0, axis=0)
                sc[sc == 0] = 1.0
                V = np.linalg.svd(J0 / sc, full_matrices=False)[2][: ident["rank"]].T / sc[:, None]
                sub = Problem(
                    prob.model,
                    [f"subspace_{i}" for i in range(ident["rank"])],
                    np.zeros(ident["rank"]),
                    -np.inf * np.ones(ident["rank"]),
                    np.inf * np.ones(ident["rank"]),
                    prob.offsets,
                    prob.tikhonov,
                    None,
                    prob.depths,
                )
                sub.residual = lambda c, P=prob, V=V: P.residual(P.x0 + V @ c)  # type: ignore[method-assign]
                sub.jacobian = lambda c, P=prob, V=V: P.jacobian(P.x0 + V @ c) @ V  # type: ignore[method-assign]
                rs = least_squares(sub.residual, sub.x0, jac=sub.jacobian, method="lm", xtol=tol, ftol=tol, gtol=tol)
                x = np.clip(prob.x0 + V @ rs.x, prob.lb, prob.ub)
                r2 = _lsq(prob, x, tol, loss)
                a2 = assess(prob, r2)
                log.append({"strategy": "identifiable subspace then full", **a2})
            elif strat == "multistart":
                n = int(cfg.get("multistart_n", 8))
                lb = np.where(np.isfinite(prob.lb), prob.lb, prob.x0 - 3)
                ub = np.where(np.isfinite(prob.ub), prob.ub, prob.x0 + 3)
                pts = qmc.scale(qmc.Sobol(len(prob.x0), seed=int(cfg.get("seed", 0))).random(n), lb, ub)
                for p0 in pts:
                    r2 = _lsq(prob, p0, tol, loss)
                    a2 = assess(prob, r2)
                    log.append({"strategy": "multistart", "init": p0.tolist(), **a2})
                    if a2["cost"] < best[0]["cost"] or (a2["converged"] and not best[0]["converged"]):
                        best = (a2, r2)
                continue
            elif strat == "continuation" and continuation_model is not None:
                x = best[1].x
                for short in continuation_model:  # problems on growing windows: homotopy in the data
                    rs = _lsq(short, x, tol, loss)
                    x = rs.x
                r2 = _lsq(prob, x, tol, loss)
                a2 = assess(prob, r2)
                log.append({"strategy": "continuation (window homotopy)", **a2})
            elif strat == "regularise":
                strong = Problem(
                    prob.model,
                    prob.names,
                    prob.x0,
                    prob.lb,
                    prob.ub,
                    prob.offsets,
                    max(prob.tikhonov, 10.0),
                    None,
                    prob.depths,
                )
                rs = _lsq(strong, best[1].x, tol, loss)
                r2 = _lsq(prob, rs.x, tol, loss)  # released: the regularised solution only provides the start
                a2 = assess(prob, r2)
                log.append({"strategy": "regularise then release", **a2})
            elif strat == "robust":
                r2 = _lsq(prob, best[1].x, tol, "soft_l1")
                a2 = assess(prob, r2)
                log.append({"strategy": "robust loss (soft_l1)", **a2})
            else:
                continue
            if a2["converged"] and (not best[0]["converged"] or a2["cost"] <= best[0]["cost"] * (1 + 1e-9)):
                best = (a2, r2)
            elif not best[0]["converged"] and a2["cost"] < best[0]["cost"]:
                best = (a2, r2)
    a, res = best
    return {"result": a, "x": res.x, "identifiability": ident, "attempts": log, "strategy_used": _used(log, a)}


def _used(log: list[dict], a: dict) -> str:
    for e in log:
        if e["x"] == a["x"]:
            return e["strategy"]
    return log[-1]["strategy"]


def covariance(prob: Problem, x: np.ndarray, hac_lag_steps: int = 24) -> dict:
    """Gauss-Newton and Newey-West (HAC) covariances of the unknowns. The scores are summed over the sensors of each
    time row, then autocorrelation over hac_lag_steps rows is accounted for with Bartlett weights."""
    J = prob.jacobian(x)
    r = prob.residual(x)
    n, p = J.shape
    m = prob.model
    n_data = int(np.isfinite(m.data[m.scored]).sum())
    Jd, rd = J[:n_data], r[:n_data]
    s2 = float(rd @ rd / max(n_data - p, 1))
    JtJ = J.T @ J
    try:
        Ainv = np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        Ainv = np.linalg.pinv(JtJ)
    gn = s2 * Ainv
    # per-time scores (rows of the observation matrix, NaNs removed consistently)
    D = m.data[m.scored]
    mask = np.isfinite(D)
    k = 0
    G = np.zeros((D.shape[0], p))
    for i in range(D.shape[0]):
        c = int(mask[i].sum())
        G[i] = (Jd[k : k + c] * rd[k : k + c, None]).sum(0)
        k += c
    S = G.T @ G
    for lag in range(1, int(hac_lag_steps) + 1):
        w = 1.0 - lag / (hac_lag_steps + 1.0)
        Gl = G[lag:].T @ G[:-lag]
        S += w * (Gl + Gl.T)
    hac = Ainv @ S @ Ainv
    return {
        "s": float(np.sqrt(s2)),
        "sd_gn": np.sqrt(np.clip(np.diag(gn), 0, None)).tolist(),
        "sd_hac": np.sqrt(np.clip(np.diag(hac), 0, None)).tolist(),
        "lag1_autocorrelation": float(np.corrcoef(G[1:, 0], G[:-1, 0])[0, 1])
        if len(G) > 2 and G[:, 0].std() > 0
        else 0.0,
    }


# ============================================================================ classical cross-check
def harmonic(t: np.ndarray, T: np.ndarray, z: np.ndarray, period_s: float = DAY) -> dict:
    """First-harmonic amplitude and phase at each depth (least squares with a linear trend and the second
    harmonic), and the classical estimates of alpha between consecutive depths:
    amplitude: alpha = w dz^2 / (2 ln^2(A1/A2)),  phase: alpha = w dz^2 / (2 (phi2 - phi1)^2)."""
    w = 2 * np.pi / period_s
    tt = t - t[0]
    X = np.stack(
        [np.ones_like(tt), tt / DAY, np.cos(w * tt), np.sin(w * tt), np.cos(2 * w * tt), np.sin(2 * w * tt)], 1
    )
    A, P = [], []
    for j in range(T.shape[1]):
        ok = np.isfinite(T[:, j])
        c = np.linalg.lstsq(X[ok], T[ok, j], rcond=None)[0]
        A.append(float(np.hypot(c[2], c[3])))
        P.append(float(np.arctan2(c[3], c[2])))
    pairs = []
    for j in range(len(z) - 1):
        dz = z[j + 1] - z[j]
        ratio = A[j] / A[j + 1] if A[j + 1] > 0 else np.inf
        dphi = (P[j + 1] - P[j]) % (2 * np.pi)
        pairs.append(
            {
                "z1": float(z[j]),
                "z2": float(z[j + 1]),
                "amplitude_ratio": float(ratio),
                "phase_lag_h": float(dphi / w / 3600),
                "alpha_amplitude": float(w * dz**2 / (2 * np.log(ratio) ** 2)) if ratio > 1 else float("nan"),
                "alpha_phase": float(w * dz**2 / (2 * dphi**2)) if dphi > 0 else float("nan"),
            }
        )
    return {"amplitude": A, "phase": P, "pairs": pairs}


# ============================================================================ end-to-end helpers
def conduction_problem(win: dict, mcfg: dict, fcfg: dict) -> Problem:
    m = Conduction1D(
        win["t"],
        win["z"],
        win["T"],
        mcfg["top_m"],
        mcfg["bottom_m"],
        mcfg["inner_m"],
        float(mcfg.get("dz_m", 0.0025)),
        float(mcfg.get("dt_s", 600)),
        int(fcfg.get("layers", 1)),
        float(mcfg.get("spinup_h", 24)) * 3600,
        fcfg.get("layer_edges_m"),
    )
    na = m.layers
    lo, hi = (np.log(float(v)) for v in fcfg.get("alpha_bounds", [1e-8, 1e-5]))
    names = [f"log_alpha_{i + 1}" for i in range(na)]
    x0 = [np.log(float(fcfg.get("alpha_init", 1e-6)))] * na
    lb, ub = [lo] * na, [hi] * na
    if fcfg.get("depths"):
        dmax = float(fcfg.get("depth_bound_m", 0.03))
        names += [f"depth_shift_{z:g}m" for z in mcfg["inner_m"]]
        x0 += [0.0] * len(mcfg["inner_m"])
        lb += [-dmax] * len(mcfg["inner_m"])
        ub += [dmax] * len(mcfg["inner_m"])
    if fcfg.get("offsets"):
        names += [f"offset_{z:g}m" for z in mcfg["inner_m"]]
        x0 += [0.0] * len(mcfg["inner_m"])
        lb += [-5.0] * len(mcfg["inner_m"])
        ub += [5.0] * len(mcfg["inner_m"])
    return Problem(
        m,
        names,
        np.array(x0),
        np.array(lb),
        np.array(ub),
        bool(fcfg.get("offsets")),
        float(fcfg.get("tikhonov", 0.0)),
        None,
        bool(fcfg.get("depths")),
    )


def fit_conduction(win: dict, mcfg: dict, fcfg: dict, plain: bool = False) -> dict:
    """Fit alpha (and offsets) on one window; uncertainty; harmonic cross-check; returns a JSON-friendly record."""
    prob = conduction_problem(win, mcfg, fcfg)
    days = (win["t"][-1] - win["t"][0]) / DAY
    cont = []
    for frac in (0.25, 0.5):
        n = int(len(win["t"]) * frac)
        if n * 3600 > float(mcfg.get("spinup_h", 24)) * 3600 + 2 * DAY:
            sub = {k: (v[:n] if isinstance(v, np.ndarray | list) else v) for k, v in win.items()}
            sub["z"] = win["z"]
            cont.append(conduction_problem(sub, mcfg, fcfg))
    out = fit(prob, fcfg, plain=plain, continuation_model=cont)
    x = out["x"]
    cov = covariance(prob, x, int(fcfg.get("hac_lag_h", 24)) * int(round(3600 / np.median(np.diff(win["t"])))))
    na = prob.model.layers  # alpha rows (a freezing model's latent temperature is reported after them)
    alpha = np.exp(x[:na])
    rows = []
    for i in range(na):
        rows.append(
            {
                "parameter": f"alpha_{i + 1}",
                "estimate": float(alpha[i]),
                "ci95_gn": [
                    float(alpha[i] * np.exp(-1.96 * cov["sd_gn"][i])),
                    float(alpha[i] * np.exp(1.96 * cov["sd_gn"][i])),
                ],
                "ci95_hac": [
                    float(alpha[i] * np.exp(-1.96 * cov["sd_hac"][i])),
                    float(alpha[i] * np.exp(1.96 * cov["sd_hac"][i])),
                ],
            }
        )
    for j in range(na, len(x)):
        rows.append(
            {
                "parameter": prob.names[j],
                "estimate": float(x[j]),
                "ci95_gn": [float(x[j] - 1.96 * cov["sd_gn"][j]), float(x[j] + 1.96 * cov["sd_gn"][j])],
                "ci95_hac": [float(x[j] - 1.96 * cov["sd_hac"][j]), float(x[j] + 1.96 * cov["sd_hac"][j])],
            }
        )
    pred = prob.predict(x).real
    m = prob.model
    rmse = np.sqrt(np.nanmean((pred - m.data)[m.scored] ** 2, axis=0)).tolist()
    zh = np.array([mcfg["top_m"], *mcfg["inner_m"], mcfg["bottom_m"]])
    cols = [int(np.argmin(np.abs(win["z"] - v))) for v in zh]
    return {
        "days": float(days),
        "rows": rows,
        "fit": out["result"],
        "strategy_used": out["strategy_used"],
        "attempts": out["attempts"],
        "identifiability": out["identifiability"],
        "covariance": cov,
        "rmse_per_sensor": rmse,
        "prediction": pred,
        "harmonic": harmonic(win["t"], win["T"][:, cols], zh),
        "x": x,
    }


def freezing_problem(win: dict, mcfg: dict, fcfg: dict) -> Problem:
    """The apparent-heat-capacity model on a window: unknowns (log alpha, log S)."""
    m = FreezingConduction1D(
        win["t"],
        win["z"],
        win["T"],
        mcfg["top_m"],
        mcfg["bottom_m"],
        mcfg["inner_m"],
        float(mcfg.get("dz_m", 0.0025)),
        float(mcfg.get("dt_s", 600)),
        1,
        float(mcfg.get("spinup_h", 24)) * 3600,
        None,
        t_freeze=float(fcfg.get("t_freeze_c", 0.0)),
        width=float(fcfg.get("freeze_width_k", 0.5)),
    )
    lo, hi = (np.log(float(v)) for v in fcfg.get("alpha_bounds", [1e-8, 1e-5]))
    slo, shi = (np.log(float(v)) for v in fcfg.get("latent_bounds_k", [1e-2, 300.0]))
    x0 = [np.log(float(fcfg.get("alpha_init", 1e-6))), np.log(float(fcfg.get("latent_init_k", 10.0)))]
    names, lb, ub = ["log_alpha", "log_latent_temperature_K"], [lo, slo], [hi, shi]
    if fcfg.get("fit_freezing_point", False):
        tlo, thi = (float(v) for v in fcfg.get("freezing_point_bounds_c", [-2.0, 0.5]))
        names.append("freezing_point_C")
        x0.append(float(fcfg.get("t_freeze_c", 0.0)))
        lb.append(tlo)
        ub.append(thi)
    return Problem(m, names, np.array(x0), np.array(lb), np.array(ub))


def fit_freezing(win: dict, mcfg: dict, fcfg: dict, plain: bool = False) -> dict:
    """Fit the freezing model on one window; Gauss-Newton and HAC intervals of alpha and S."""
    prob = freezing_problem(win, mcfg, fcfg)
    out = fit(prob, fcfg, plain=plain)
    x = out["x"]
    lag = int(fcfg.get("hac_lag_h", 24)) * int(round(3600 / np.median(np.diff(win["t"]))))
    cov = covariance(prob, x, lag)
    rows = []
    for j, name in enumerate(("alpha", "latent_temperature_K")):
        e = float(np.exp(x[j]))
        rows.append(
            {
                "parameter": name,
                "estimate": e,
                "ci95_gn": [e * float(np.exp(-1.96 * cov["sd_gn"][j])), e * float(np.exp(1.96 * cov["sd_gn"][j]))],
                "ci95_hac": [e * float(np.exp(-1.96 * cov["sd_hac"][j])), e * float(np.exp(1.96 * cov["sd_hac"][j]))],
            }
        )
    if len(x) > 2:
        rows.append(
            {
                "parameter": "freezing_point_C",
                "estimate": float(x[2]),
                "ci95_gn": [float(x[2] - 1.96 * cov["sd_gn"][2]), float(x[2] + 1.96 * cov["sd_gn"][2])],
                "ci95_hac": [float(x[2] - 1.96 * cov["sd_hac"][2]), float(x[2] + 1.96 * cov["sd_hac"][2])],
            }
        )
    m = prob.model
    pred = prob.predict(x).real
    rmse = np.sqrt(np.nanmean((pred - m.data)[m.scored] ** 2, axis=0)).tolist()
    return {
        "rows": rows,
        "fit": out["result"],
        "strategy_used": out["strategy_used"],
        "attempts": out["attempts"],
        "identifiability": out["identifiability"],
        "covariance": cov,
        "rmse_per_sensor": rmse,
        "prediction": pred,
        "x": x,
    }


def holdout(win_train: dict, win_test: dict, mcfg: dict, fcfg: dict, x: np.ndarray) -> dict:
    """Predict the inner sensors on the next window with the fitted parameters (no refit), against a baseline that
    interpolates linearly in depth between the boundary sensors at each time (no dynamics)."""
    prob = conduction_problem(win_test, mcfg, fcfg)
    m = prob.model
    pred = prob.predict(x).real
    zt, zb = mcfg["top_m"], mcfg["bottom_m"]
    w = np.array([(z - zt) / (zb - zt) for z in mcfg["inner_m"]])
    base = m.T_obs[:, [m.ktop]] * (1 - w) + m.T_obs[:, [m.kbot]] * w
    sc = m.scored
    rm = np.sqrt(np.nanmean((pred - m.data)[sc] ** 2, axis=0))
    rb = np.sqrt(np.nanmean((base - m.data)[sc] ** 2, axis=0))
    return {
        "rmse_model": rm.tolist(),
        "rmse_linear_baseline": rb.tolist(),
        "skill": (1 - rm / rb).tolist(),
        "prediction": pred,
        "baseline": base,
        "observed": m.data,
        "t": m.t_obs,
    }


def parse_sensors(spec: str) -> dict[str, float]:
    """'T5:0.05, T10:0.10' -> {'T5': 0.05, 'T10': 0.10} (column name : position in m)."""
    out = {}
    for part in re.split(r"[;,]\s*", spec.strip()):
        if part:
            name, pos = part.rsplit(":", 1)
            out[name.strip()] = float(pos)
    return out
