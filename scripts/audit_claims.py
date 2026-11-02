"""Claims audit: every number quoted in README, summaries, docs and the paper must match a value in a
results/<id>/metrics.json (or record.json) artifact, up to the precision quoted.

Rules: numbers are scanned in prose and tables. Exempt: integers < 10 (counts, list markers), years,
version numbers, section/equation/figure numbering, numbers inside code spans/blocks, and numbers on
lines tagged ``<!-- audit:ignore -->`` (used for definitions such as thresholds and parameter values
that are inputs, not results). A quoted number q matches an artifact value v if
|v - q| <= 0.5 * 10^(-d) * max(1, ...) at the quoted number of significant/decimal digits, or if it
matches 100*v (percent).
Exit code 1 if any unmatched claim remains; the report lists file:line for each.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    "README.md",
    "docs/EXECUTIVE_SUMMARY.md",
    "docs/PROJECT_SUMMARY.md",
    "docs/VV_REPORT.md",
    "docs/phases/PHASE_1_SUMMARY.md",
    "docs/phases/PHASE_2_SUMMARY.md",
    "docs/phases/PHASE_3_SUMMARY.md",
    "docs/phases/PHASE_4_SUMMARY.md",
    "docs/phases/PHASE_5_SUMMARY.md",
    "docs/phases/PHASE_6_SUMMARY.md",
    "docs/phases/PHASE_7_SUMMARY.md",
    "docs/phases/PHASE_8_SUMMARY.md",
    "paper/main.tex",
    "paper/PAPER_READINESS.md",
]
NUM = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")


def collect_values() -> list[float]:
    vals: list[float] = []

    def walk(o: object) -> None:
        if isinstance(o, bool):
            return
        if isinstance(o, int | float):
            vals.append(float(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for f in list(ROOT.glob("results/*/metrics.json")) + list(ROOT.glob("results/*/record.json")):
        walk(json.loads(f.read_text()))
    for f in ROOT.glob("src/aerosynapse/data/*.json"):
        walk(json.loads(f.read_text()))
    return vals


def tolerance(tok: str) -> float:
    t = tok.lower().replace("−", "-").lstrip("-")
    if "e" in t:
        mant, exp = t.split("e")
        dec = len(mant.split(".")[1]) if "." in mant else 0
        return 0.5 * 10 ** (-dec) * 10 ** int(exp)
    dec = len(t.split(".")[1]) if "." in t else 0
    return 0.5 * 10 ** (-dec)


def matches(q: float, tol: float, vals: list[float]) -> bool:
    for v in vals:
        for cand in (v, 100 * v, abs(v), 100 * abs(v)):
            if abs(cand - q) <= tol * 1.0000001:
                return True
    return False


def scan(path: Path, vals: list[float]) -> list[str]:
    bad = []
    in_code = False
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip().startswith("```") or "\\begin{verbatim}" in line or "\\end{verbatim}" in line:
            in_code = not in_code
            continue
        if in_code or "audit:ignore" in line or line.lstrip().startswith(("%", "#!")):
            continue
        text = re.sub(r"`[^`]*`", "", line)
        text = re.sub(r"\]\([^)]*\)|https?://\S+|\\(?:cite|ref|label|includegraphics|input|eqref)\{[^}]*\}", "", text)
        text = re.sub(
            r"\b(?:Phase|Section|Fig\.?|Figure|Table|Eq\.?|H\d|ADR-|KI-|Level|Stage|Step|tier)\s*[-\d.]+",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"\b(?:19|20)\d\d\b", "", text)
        text = re.sub(r"\b\d+(?:\.\d+){2,}\b", "", text)  # versions
        for m in NUM.finditer(text):
            tok = m.group(0)
            q = float(tok.replace("−", "-"))
            if "." not in tok and "e" not in tok.lower() and abs(q) < 10:
                continue
            if not matches(q, tolerance(tok), vals):
                bad.append(f"{path.relative_to(ROOT)}:{i}: {tok}")
    return bad


def main() -> int:
    vals = collect_values()
    bad: list[str] = []
    checked = 0
    for d in DOCS:
        p = ROOT / d
        if p.exists():
            checked += 1
            bad += scan(p, vals)
    print(f"audited {checked} documents against {len(vals)} artifact values")
    for b in bad:
        print("UNMATCHED", b)
    print(f"unmatched claims: {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
