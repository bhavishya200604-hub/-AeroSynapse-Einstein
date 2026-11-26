"""Measured cost of a computation and its extrapolation to larger problems (Step 6.4 grid study).

peak_memory_bytes()   peak resident memory of THIS process (Windows: PeakWorkingSetSize; elsewhere ru_maxrss), so a
                      measurement is clean only in a fresh process (one solve per process, src/utils/isolate.py)
fit_power(n, y)       y = a n^b by least squares on (ln n, ln y); returns a, b, their standard errors and the residual
                      standard deviation s of ln y
predict(fit, n)       point prediction and 95 % prediction interval in log space:
                      ln y* +- t_{0.975, m-2} s sqrt(1 + 1/m + (ln n - mean ln n)^2 / Sxx)
machine()             total and available physical memory (bytes) and logical CPUs
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import t as student_t


def _win_pmc():
    """PROCESS_MEMORY_COUNTERS of this process (typed ctypes call: untyped HANDLEs fail silently on 64-bit)."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    f = k32.K32GetProcessMemoryInfo
    f.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    f.restype = wintypes.BOOL
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    if not f(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("K32GetProcessMemoryInfo failed")
    return c


def peak_memory_bytes() -> int:
    if sys.platform == "win32":
        c = _win_pmc()
        return int(max(c.PeakWorkingSetSize, c.PeakPagefileUsage))
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def machine() -> dict:
    out = {"cpus": os.cpu_count()}
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        out.update(total_memory=int(m.ullTotalPhys), available_memory=int(m.ullAvailPhys))
    else:
        try:
            pg, n, av = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_AVPHYS_PAGES")
            out.update(total_memory=int(pg * n), available_memory=int(pg * av))
        except (ValueError, OSError, AttributeError):
            out.update(total_memory=None, available_memory=None)
    return out


def fit_power(n, y) -> dict:
    ln, ly = np.log(np.asarray(n, float)), np.log(np.asarray(y, float))
    A = np.column_stack([np.ones_like(ln), ln])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    r = ly - A @ coef
    m = len(ln)
    s = float(np.sqrt(r @ r / max(m - 2, 1)))
    cov = s**2 * np.linalg.inv(A.T @ A)
    return {
        "a": float(np.exp(coef[0])),
        "b": float(coef[1]),
        "b_se": float(np.sqrt(cov[1, 1])),
        "s_log": s,
        "m": m,
        "mean_ln": float(ln.mean()),
        "sxx": float(((ln - ln.mean()) ** 2).sum()),
    }


def predict(fit: dict, n) -> dict:
    """Prediction at n (scalar or array) with the 95 % prediction interval (log space)."""
    ln = np.log(np.asarray(n, float))
    mu = np.log(fit["a"]) + fit["b"] * ln
    tq = float(student_t.ppf(0.975, max(fit["m"] - 2, 1)))
    half = tq * fit["s_log"] * np.sqrt(1 + 1 / fit["m"] + (ln - fit["mean_ln"]) ** 2 / max(fit["sxx"], 1e-300))
    return {"value": np.exp(mu), "lo": np.exp(mu - half), "hi": np.exp(mu + half)}


def human_time(s: float) -> str:
    for unit, k in (("years", 365.25 * 86400), ("days", 86400), ("h", 3600), ("min", 60)):
        if s >= k:
            return f"{s / k:.3g} {unit}"
    return f"{s:.3g} s"


def current_memory_bytes() -> int:
    """Current resident memory of this process (Windows WorkingSetSize / private bytes; Linux /proc/self/statm)."""
    if sys.platform == "win32":
        c = _win_pmc()
        return int(max(c.WorkingSetSize, c.PagefileUsage))
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        return peak_memory_bytes()


class MemorySampler:
    """Context manager: samples the current memory every `dt` seconds in a thread; `.extra` = max - value at entry
    (the memory a computation needs on top of what the process already holds)."""

    def __init__(self, dt: float = 0.01):
        self.dt, self.max, self.start, self.extra = dt, 0, 0, 0

    def __enter__(self):
        import threading

        self.start = self.max = current_memory_bytes()
        self._stop = threading.Event()

        def run():
            while not self._stop.is_set():
                self.max = max(self.max, current_memory_bytes())
                self._stop.wait(self.dt)

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.max = max(self.max, current_memory_bytes())
        self.extra = max(self.max - self.start, 1)
        return False
