"""Safety layer: five-tier validation ladder (spec Section 16) and the research-use disclaimer."""

from __future__ import annotations

from enum import IntEnum


class ValidationTier(IntEnum):
    """Five-tier validation ladder. Higher tiers need strictly more evidence."""

    RESEARCH_SIMULATION = 1
    ENGINEERING_ANALYSIS = 2
    VERIFIED_SOFTWARE = 3
    EXPERIMENTALLY_VALIDATED = 4
    SAFETY_CERTIFIED = 5


TIER_DESCRIPTIONS: dict[ValidationTier, str] = {
    ValidationTier.RESEARCH_SIMULATION: "Research simulation: exploratory, no accuracy guarantee.",
    ValidationTier.ENGINEERING_ANALYSIS: "Engineering analysis: documented assumptions, sanity-checked.",
    ValidationTier.VERIFIED_SOFTWARE: "Verified software: code verification (MMS, convergence order) passed.",
    ValidationTier.EXPERIMENTALLY_VALIDATED: "Experimentally validated: compared with physical experiments.",
    ValidationTier.SAFETY_CERTIFIED: "Safety-certified: certified by an accredited authority.",
}

#: Tier of the platform as a whole. The classical solvers carry code-verification evidence
#: (tier 3 for those modules); ML surrogates are tier 1. Nothing is experimentally validated
#: or certified. The *platform* tier is the minimum over its components.
PLATFORM_TIER = ValidationTier.RESEARCH_SIMULATION

DISCLAIMER = (
    "RESEARCH USE ONLY. AeroSynapse-Einstein results are not certified for aircraft, pressure "
    "vessels, industrial machinery or any safety-critical decision. Platform validation tier: "
    f"{PLATFORM_TIER.value}/5 ({TIER_DESCRIPTIONS[PLATFORM_TIER]}) Classical solvers: tier 3 "
    "(code-verified). ML surrogates: tier 1."
)


def banner() -> str:
    """Return the disclaimer banner used by CLI, UI and every generated report."""
    line = "=" * 78
    return f"{line}\n{DISCLAIMER}\n{line}"
