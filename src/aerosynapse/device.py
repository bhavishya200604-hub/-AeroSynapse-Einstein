"""Device abstraction and hardware report (execution modes A/B; spec Section 12)."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any


def torch_available() -> bool:
    """Return True if PyTorch can be imported."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def select_device(prefer: str = "auto") -> str:
    """Select ``"cuda"``, ``"mps"`` or ``"cpu"``.

    Parameters
    ----------
    prefer
        ``"auto"`` picks CUDA, then MPS, then CPU. Any explicit value is validated.
    """
    if not torch_available():
        return "cpu"
    import torch

    if prefer == "cpu":
        return "cpu"
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if prefer in ("auto", "mps") and mps is not None and mps.is_available():
        return "mps"
    if prefer not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"unknown device preference {prefer!r}")
    return "cpu"


def hardware_report() -> dict[str, Any]:
    """Collect hardware / software information for experiment records and ``doctor``."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    try:
        import numpy
        import scipy

        info["numpy"] = numpy.__version__
        info["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            info["mem_total_kb"] = int(fh.readline().split()[1])
    except OSError:
        info["mem_total_kb"] = None
    if torch_available():
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            info["gpu"] = torch.cuda.get_device_name(0)
    else:
        info["torch"] = None
        info["cuda_available"] = False
    info["device"] = select_device()
    return info
