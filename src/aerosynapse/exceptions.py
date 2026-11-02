"""Custom exception hierarchy.

Every failure that could otherwise return a plausible-looking but wrong field raises one of these
(spec Section 16: report failures, never return silently).
"""

from __future__ import annotations


class AeroSynapseError(Exception):
    """Base class for all package errors."""


class ConfigurationError(AeroSynapseError):
    """Invalid or inconsistent configuration / problem definition."""


class SolverDivergenceError(AeroSynapseError):
    """A numerical solver produced non-finite values or exceeded a growth bound."""


class StabilityError(AeroSynapseError):
    """A requested discretisation violates a known stability limit (e.g. CFL)."""


class ConstraintViolationError(AeroSynapseError):
    """A design / state violates a hard constraint."""


class ValidationFailure(AeroSynapseError):
    """A validation check failed (accuracy, conservation, BC satisfaction, calibration)."""


class RegimeError(AeroSynapseError):
    """A query lies outside the validated physical regime (e.g. turbulent Re for a laminar solver)."""


class ResourceLimitError(AeroSynapseError):
    """A memory / time guard was exceeded."""


class BudgetExhaustedError(AeroSynapseError):
    """An optimisation / training evaluation budget was exhausted."""
