"""Shared imports/paths for the video clip scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
CLIPS = ROOT / "media" / "clips"
CACHE = ROOT / "media" / "_cache"
RES = ROOT / "results"


def load(exp: str, name: str = "metrics.json") -> dict:
    return json.loads((RES / exp / name).read_text())
