"""Run Drela's XFOIL as an independent judge of a section.

XFOIL couples a panel method to an integral boundary layer with e^N transition. NeuralFoil was
trained on XFOIL, so this checks how well the surrogate held up at the designs the optimiser
chose. That is where a surrogate is most likely to be wrong, because the optimiser seeks out its
errors. XFOIL is not independent of NeuralFoil's physics: for that, see :mod:`aerosynapse.shapeopt.vii`.

Each run happens in its own temporary folder with a time limit, because XFOIL can hang on a hard
case. Points that fail to converge are left out of the polar and never filled in.

Limits: 2D, attached or mildly separated flow; compressibility by Karman-Tsien only, so it is not
used here above Mach 0.6.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError

FloatArr = NDArray[np.float64]
MACH_LIMIT = 0.6


def executable() -> Path:
    """Locate xfoil.exe: the XFOIL_EXE variable, then the project's tools folder."""
    env = os.environ.get("XFOIL_EXE")
    p = Path(env) if env else Path(__file__).resolve().parents[3] / ".venv" / "tools" / "xfoil" / "xfoil.exe"
    if not p.is_file():
        raise ConfigurationError(f"XFOIL not found at {p}. Set XFOIL_EXE to the program's path.")
    return p


def parse_polar(text: str) -> dict[str, FloatArr]:
    """Read an XFOIL polar file. Repeated angles keep their last value."""
    rows: dict[float, list[float]] = {}
    started = False
    for line in text.splitlines():
        if line.strip().startswith("------"):
            started = True
            continue
        if not started:
            continue
        try:
            v = [float(q) for q in line.split()]
        except ValueError:
            continue
        if len(v) >= 7:
            rows[v[0]] = v
    a = np.array([rows[k] for k in sorted(rows)], float).reshape(-1, 7 if not rows else len(next(iter(rows.values()))))
    return {"alpha": a[:, 0], "CL": a[:, 1], "CD": a[:, 2], "CDp": a[:, 3], "CM": a[:, 4], "xtr_u": a[:, 5], "xtr_l": a[:, 6]}


def polar(
    xy: FloatArr, Re: float, mach: float = 0.0, ncrit: float = 9.0, xtr: float = 1.0, a_lo: float = -8.0, a_hi: float = 18.0, step: float = 0.5, timeout: float = 45.0
) -> dict[str, FloatArr]:
    """Polar of a Selig-ordered contour, swept outward from zero so each point starts from a good one."""
    if mach > MACH_LIMIT:
        raise ConfigurationError(f"Mach {mach} is above the {MACH_LIMIT} limit used for XFOIL here")
    cmds = ["PLOP", "G F", "", "LOAD foil.dat", "PANE", "OPER", f"VISC {Re:.6g}"]
    if mach > 0:
        cmds.append(f"MACH {mach:.4f}")
    cmds += ["ITER 120", "VPAR", f"N {ncrit:.3f}", f"XTR {xtr:.4f} {xtr:.4f}", "", "PACC", "polar.txt", ""]
    cmds += [f"ASEQ 0 {a_hi:g} {step:g}", "INIT", f"ASEQ {-step:g} {a_lo:g} {-step:g}", "PACC", "", "QUIT", ""]
    with tempfile.TemporaryDirectory() as d:
        body = "\n".join(f"{x:.7f} {y:.7f}" for x, y in xy)
        (Path(d) / "foil.dat").write_text(f"section\n{body}\n", encoding="ascii")
        f = Path(d) / "polar.txt"
        for _ in range(3):  # a low-memory machine can fail to launch it, which must not look like a physics result
            with contextlib.suppress(subprocess.TimeoutExpired):  # keep whatever was written before a hang
                subprocess.run([str(executable())], input="\n".join(cmds), text=True, cwd=d, capture_output=True, timeout=timeout, check=False)
            if f.exists() and len(parse_polar(f.read_text(encoding="ascii", errors="ignore"))["alpha"]) > 0:
                return parse_polar(f.read_text(encoding="ascii", errors="ignore"))
            f.unlink(missing_ok=True)
            time.sleep(1.0)
        return parse_polar("")


def at_cl(p: dict[str, FloatArr], cl: float) -> dict[str, float]:
    """Drag and angle at a target lift, on the branch below c_l,max. NaN if the lift is not reached."""
    nan = {"cd": float("nan"), "alpha": float("nan"), "cl_max": float("nan")}
    if len(p["alpha"]) < 4:
        return nan
    k = int(np.argmax(p["CL"]))
    a, c, d = p["alpha"][: k + 1], p["CL"][: k + 1], p["CD"][: k + 1]
    out = {"cl_max": float(p["CL"][k]), "cd": float("nan"), "alpha": float("nan")}
    if c.min() <= cl <= c.max():
        o = np.argsort(c)
        out["cd"] = float(np.interp(cl, c[o], d[o]))
        out["alpha"] = float(np.interp(cl, c[o], a[o]))
    return out
