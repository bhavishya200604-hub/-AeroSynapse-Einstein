"""Flow-regime classification and guards (spec Level 2: never claim laminar results are turbulent).

Default thresholds are conventional engineering rules of thumb (marked UNVERIFIED in
docs/LITERATURE_AND_NOVELTY.md); they are configurable. Separately, ``VALIDATED_RANGE`` lists the
Reynolds numbers at which this platform's solvers were actually validated (Phase 6 results); a
query outside it raises :class:`RegimeError` unless ``extrapolate=True`` is given, in which case
the result is labelled an extrapolation.
"""

from __future__ import annotations

from aerosynapse.exceptions import RegimeError

THRESHOLDS = {  # (laminar upper bound, turbulent lower bound) — UNVERIFIED rules of thumb
    "channel": (2300.0, 4000.0),
    "pipe": (2300.0, 4000.0),
    "flat_plate": (5e5, 3e6),
    "cylinder": (47.0, 200.0),  # steady-wake limit and onset of 3D wake transition (approximate)
    "cavity": (8000.0, 20000.0),
}
VALIDATED_RANGE = {"cavity": (1.0, 1000.0), "channel": (0.0, 100.0), "cylinder": (0.0, 20.0), "flat_plate": (1e3, 1e5)}


def classify(case: str, re: float) -> str:
    try:
        lam, turb = THRESHOLDS[case]
    except KeyError:
        raise RegimeError(f"no regime thresholds for case {case!r}") from None
    if re < 0:
        raise RegimeError("Reynolds number must be non-negative")
    return "laminar" if re < lam else ("transitional" if re < turb else "turbulent")


def guard(case: str, re: float, extrapolate: bool = False) -> str:
    """Return a label; raise if the query is not laminar or not inside the validated range."""
    regime = classify(case, re)
    lo, hi = VALIDATED_RANGE.get(case, (0.0, 0.0))
    if regime != "laminar":
        raise RegimeError(f"{case} at Re={re:g} is {regime}: the laminar solvers here are not valid")
    if not lo <= re <= hi:
        if not extrapolate:
            raise RegimeError(f"{case} at Re={re:g} is outside the validated range [{lo:g}, {hi:g}]")
        return f"EXTRAPOLATION: {case} Re={re:g} (validated {lo:g}-{hi:g})"
    return f"validated range: {case} Re={re:g}"


def mach_guard(mach: float, limit: float = 0.7) -> None:
    """Prandtl–Glauert is a linearised subsonic correction; refuse near/above transonic."""
    if not 0 <= mach < limit:
        raise RegimeError(f"Mach {mach} outside the Prandtl-Glauert validity range [0, {limit})")
