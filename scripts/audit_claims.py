"""Claims audit: every number quoted in README, summaries, docs and the paper must match a value in a
results/<id>/metrics.json (or record.json) artifact, up to the precision quoted.

Rules: numbers are scanned in prose and tables. Exempt: bare integers (inputs and counts: grid sizes, budgets, seeds, ranges), years,
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
    *[
        f"paper/sections/{s}.tex"
        for s in (
            "abstract",
            "introduction",
            "gap",
            "design",
            "results",
            "error",
            "ablation",
            "limitations",
            "conclusion",
        )
    ],
]
NUM = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")


RAW_KEYS = {
    "history",
    "trace",
    "profile",
    "field",
    "beam_points",
    "beam_pareto",
    "naca2412_cp",
    "naca2412_cp_alpha4",
    "hardware",
    "pit_histogram",
    "dt_history",
    "mass_history",
    "reverification",
    "dd_interface_jumps",
    "values",
    "dataset_card",
    "software_versions",
}
SOURCES = re.compile(r"(?:<!--|%)\s*sources:\s*([\w,\s./-]+?)\s*(?:-->|$)")


def collect_values(experiments: list[str] | None = None) -> list[float]:
    """Scalar values from results/<exp>/{metrics,record}.json, excluding raw arrays/histories and lists > 64."""
    vals: list[float] = []

    def walk(o: object) -> None:
        if isinstance(o, bool):
            return
        if isinstance(o, int | float):
            vals.append(float(o))
        elif isinstance(o, dict):
            for k, v in o.items():
                if k not in RAW_KEYS:
                    walk(v)
        elif isinstance(o, list) and len(o) <= 64:
            for v in o:
                walk(v)

    dirs = [ROOT / "results" / e for e in experiments] if experiments else list((ROOT / "results").glob("*"))
    for d in dirs:
        for name in ("metrics.json", "record.json"):
            f = d / name
            if f.exists():
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
            if "." not in tok and "e" not in tok.lower():
                continue  # bare integers are inputs/counts (grid sizes, budgets, seeds); results are quoted as decimals
            if not matches(q, tolerance(tok), vals):
                bad.append(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}:{i}: {tok}")
    return bad


def doc_sources(p: Path) -> list[str] | None:
    m = SOURCES.search(p.read_text(encoding="utf-8"))
    return [x.strip() for x in m.group(1).split(",") if x.strip()] if m else None


def audit(paths: list[Path]) -> list[str]:
    bad: list[str] = []
    for p in paths:
        bad += scan(p, collect_values(doc_sources(p)))
    return bad


def main() -> int:
    paths = [ROOT / d for d in DOCS if (ROOT / d).exists()]
    bad = audit(paths)
    print(f"audited {len(paths)} documents (each against its declared result sources)")
    for b in bad:
        print("UNMATCHED", b)
    print(f"unmatched claims: {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__" and "--claims-map" not in sys.argv:
    sys.exit(main())


def claims_map(out: Path = ROOT / "paper" / "CLAIMS_AUDIT.md") -> int:
    """Write a claim → artifact-key table for every audited number in the paper sources."""

    def paths(exps: list[str] | None) -> list[tuple[str, float]]:
        acc: list[tuple[str, float]] = []

        def walk(o: object, pre: str) -> None:
            if isinstance(o, bool):
                return
            if isinstance(o, int | float):
                acc.append((pre, float(o)))
            elif isinstance(o, dict):
                for k, v in o.items():
                    if k not in RAW_KEYS:
                        walk(v, f"{pre}.{k}")
            elif isinstance(o, list) and len(o) <= 64:
                for i, v in enumerate(o):
                    walk(v, f"{pre}[{i}]")

        dirs = [ROOT / "results" / e for e in exps] if exps else list((ROOT / "results").glob("*"))
        for d in dirs:
            f = d / "metrics.json"
            if f.exists():
                walk(json.loads(f.read_text()), f"results/{d.name}/metrics.json:")
        return acc

    rows = ["# Claims audit — paper", "", "Generated by `python scripts/audit_claims.py --claims-map`. Every decimal number "
            "in the paper sources is mapped to the artifact value(s) it matches at the quoted precision.", "",
            "| location | quoted | artifact key (first matches) |", "|---|---|---|"]
    unmatched = 0
    files = [ROOT / "paper" / "main.tex", *sorted((ROOT / "paper" / "sections").glob("*.tex")),
             *sorted((ROOT / "paper" / "tables").glob("*.tex"))]
    for f in files:
        kv = paths(doc_sources(f))
        for item in scan(f, []):
            loc, tok = item.rsplit(": ", 1)
            q = float(tok.replace("−", "-"))
            tol = tolerance(tok)
            hits = [k for k, v in kv if any(abs(c - q) <= tol * 1.0000001 for c in (v, 100 * v, abs(v)))]
            if not hits:
                unmatched += 1
            rows.append(f"| {loc} | {tok} | {'; '.join(hits[:2]) if hits else '**UNMATCHED**'} |")
    rows += ["", f"Unmatched: {unmatched}"]
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"claims map: {len(rows) - 8} numbers, {unmatched} unmatched")
    return unmatched


if __name__ == "__main__" and "--claims-map" in sys.argv:
    sys.exit(1 if claims_map() else 0)
