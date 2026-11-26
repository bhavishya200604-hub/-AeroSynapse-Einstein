"""Validate the LaTeX sources without a TeX toolchain: \\input targets exist, \\cite keys are in references.bib,
\\ref targets have labels, \\includegraphics files exist, braces balance. Exit 1 on any problem."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"


def main() -> int:
    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    texts = {"main.tex": main_tex}
    problems: list[str] = []
    for inp in re.findall(r"\\input\{([^}]+)\}", main_tex):
        f = PAPER / (inp if inp.endswith(".tex") else inp + ".tex")
        if f.exists():
            texts[inp] = f.read_text(encoding="utf-8")
        else:
            problems.append(f"missing \\input file {f.relative_to(PAPER)}")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    all_tex = "\n".join(texts.values())
    for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", all_tex):
        for k in group.split(","):
            if k.strip() not in keys:
                problems.append(f"citation key not in bib: {k.strip()}")
    labels = set(re.findall(r"\\label\{([^}]+)\}", all_tex))
    for r in re.findall(r"\\ref\{([^}]+)\}", all_tex):
        if r not in labels:
            problems.append(f"\\ref without label: {r}")
    for g in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", all_tex):
        if not (PAPER / g).exists():
            problems.append(f"missing figure {g}")
    for name, t in texts.items():
        body = re.sub(r"\\[{}]", "", "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("%")))
        if body.count("{") != body.count("}"):
            problems.append(f"unbalanced braces in {name}")
    for p in problems:
        print("PROBLEM", p)
    print(f"checked {len(texts)} files, {len(keys)} bib entries: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
