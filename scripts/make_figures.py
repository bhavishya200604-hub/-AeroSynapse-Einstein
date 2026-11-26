"""Generate all figures from results/ (deterministic). Writes paper/figures/manifest.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "figure_scripts"))
sys.path.insert(0, str(ROOT / "src"))

import figures  # noqa: E402


def main() -> None:
    out = {}
    for fn in figures.ALL:
        try:
            out[fn.__name__] = fn()
        except Exception as exc:  # report, never hide
            out[fn.__name__] = {"error": f"{type(exc).__name__}: {exc}"}
        print(fn.__name__, out[fn.__name__])
    (ROOT / "paper" / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper" / "figures" / "manifest.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
