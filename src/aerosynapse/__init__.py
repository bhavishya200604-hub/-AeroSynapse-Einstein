"""AeroSynapse–Einstein: physics-aware AI for computational engineering (research use only).

Public subpackages: ``physics``, ``numerics``, ``ml``, ``optim``, ``uq``, ``validation``,
``experiments``, ``viz``, ``cli``, ``ui``. Importing the top-level package is cheap; PyTorch is
only imported by ``aerosynapse.ml`` and ``aerosynapse.uq.ensembles``.
"""

from aerosynapse._version import __version__
from aerosynapse.safety import DISCLAIMER, PLATFORM_TIER, ValidationTier

__all__ = ["DISCLAIMER", "PLATFORM_TIER", "ValidationTier", "__version__"]
