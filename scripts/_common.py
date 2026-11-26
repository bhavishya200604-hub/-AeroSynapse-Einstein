"""Shared helpers for phase scripts (profile parsing, result root)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))


def parse_profile(desc: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--profile", default="quick", choices=["quick", "standard", "full"])
    return p.parse_args()
