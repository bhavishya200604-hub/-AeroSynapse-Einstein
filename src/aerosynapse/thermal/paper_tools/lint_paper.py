"""Paper lint (Phase 6B): macros, citations, claims, structure and UK spelling.

    python -m aerosynapse.thermal.paper_tools.lint_paper paper/paper2

Checks
  macros     every \\Name{} used in the text is defined (numbers.tex or main.tex)
  citations  every \\cite key is in refs.bib and every refs.bib entry is VERIFIED
             in docs/LITERATURE.md; bib entries that are never cited are listed
  claims     a literal number in the prose must be a generated macro, a
             conventional constant, or sit on a line marked "% claims-ok: <reason>"
  structure  required sections exist; every \\label is referenced and every
             \\ref / \\eqref resolves; every figure / table file exists
  spelling   US spellings that UK English writes differently

Exit status 1 if any check fails. The report is printed verbatim for the
step's "Lint / compile output".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
# numbers that need no provenance: small counts, conventional confidence levels,
# powers of ten in ranges, the Gaussian 90 % multiplier.
CONVENTIONAL = {str(i) for i in range(0, 11)} | {"90", "95", "1.64", "100"}

US_UK = [
    (r"\b(\w+?)iz(e|es|ed|ing|ation|ations)\b", "ise"),
    (r"\b(colou?r)\b", None), (r"\bcolor", "colour"), (r"\bcenter", "centre"),
    (r"\bbehavior", "behaviour"), (r"\bmodeling\b", "modelling"), (r"\bmodeled\b", "modelled"),
    (r"\banalyz", "analys"), (r"\bfavor", "favour"), (r"\bneighbor", "neighbour"),
    (r"\bgray\b", "grey"), (r"\bfiber", "fibre"), (r"\bliter\b", "litre"),
    (r"\bdefense\b", "defence"), (r"\blabeled\b", "labelled"), (r"\btraveled\b", "travelled"),
    (r"\bcatalog\b", "catalogue"), (r"\bpercent\b", "per cent / %"),
]
ISE_EXEMPT = {"size", "sizes", "sized", "seize", "seized", "capsize", "prize", "prizes",
              "citizen", "maize", "baize", "resize"}

REQUIRED_SECTIONS = ["introduction", "conclusions"]


def strip_comment(line: str) -> str:
    return re.sub(r"(?<!\\)%.*$", "", line)


def read_tex(paper: Path) -> dict[str, list[str]]:
    files = [paper / "main.tex"] + sorted((paper / "sections").glob("*.tex"))
    return {str(f.relative_to(paper)): f.read_text(encoding="utf-8").splitlines() for f in files}


def check_macros(paper, tex):
    defined = set()
    for f in [paper / "numbers.tex", paper / "main.tex"]:
        defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", f.read_text(encoding="utf-8")))
    bad = []
    for name, lines in tex.items():
        for i, ln in enumerate(lines, 1):
            for m in re.findall(r"\\([A-Za-z]+)\{\}", strip_comment(ln)):
                if m not in defined:
                    bad.append(f"{name}:{i}: undefined macro \\{m}")
            for m in re.findall(r"(?<![\\A-Za-z])([A-Z][A-Za-z]+)\{\}", strip_comment(ln)):
                if m in defined:
                    bad.append(f"{name}:{i}: macro {m}{{}} without backslash (would print as text)")
    used = {m for lines in tex.values() for ln in lines for m in re.findall(r"\\([A-Za-z]+)\b", strip_comment(ln))}
    nums = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (paper / "numbers.tex").read_text(encoding="utf-8")))
    return bad, sorted(nums - used)


def literature_status() -> dict[str, str]:
    out = {}
    for ln in (ROOT / "docs" / "LITERATURE.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in ln.split("|")]
        if len(cells) > 4 and cells[1] and not cells[1].startswith("-"):
            out[cells[1]] = cells[4]
    return out


def check_citations(paper, tex):
    bib = (paper / "refs.bib").read_text(encoding="utf-8")
    keys = re.findall(r"^@\w+\{([^,]+),", bib, flags=re.M)
    cited = set()
    for lines in tex.values():
        for ln in lines:
            for grp in re.findall(r"\\cite[tp]?\{([^}]*)\}", strip_comment(ln)):
                cited |= {k.strip() for k in grp.split(",")}
    lit = literature_status()
    bad = [f"cited but not in refs.bib: {k}" for k in sorted(cited - set(keys))]
    for k in keys:
        st = lit.get(k)
        if st is None:
            bad.append(f"refs.bib entry not in docs/LITERATURE.md: {k}")
        elif not st.lstrip("*").upper().startswith("VERIFIED"):
            bad.append(f"refs.bib entry not VERIFIED: {k} ({st[:60]})")
    unused = sorted(set(keys) - cited)
    return bad, unused, len(cited), len(keys)


NUM = re.compile(r"(?<![\\A-Za-z_{])(\d+(?:\.\d+)?)")


def prose(line: str) -> str:
    s = strip_comment(line)
    cmds = ("label|ref|eqref|cite[tp]?|input|includegraphics|begin|end|graphicspath|usepackage|documentclass"
            "|section|subsection|paragraph")
    s = re.sub(r"\\(" + cmds + r")(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\[[^\]]*(width|pt|cm)[^\]]*\]", " ", s)
    s = re.sub(r"\{[^}]*(linewidth|textwidth)[^}]*\}", " ", s)
    s = re.sub(r"\d+(\.\d+)?\\linewidth", " ", s)
    s = re.sub(r"\\texttt\{[^}]*\}", " ", s)
    s = re.sub(r"results/[\w\\./-]+", " ", s)
    s = re.sub(r"[A-Za-z]+\d+[A-Za-z\d]*", " ", s)  # identifiers such as R5.1k, p6
    return s


def check_claims(tex):
    bad = []
    for name, lines in tex.items():
        if name == "main.tex":
            body = False
        for i, ln in enumerate(lines, 1):
            if name == "main.tex":
                if r"\begin{document}" in ln:
                    body = True
                if not body:
                    continue
            if "claims-ok" in ln:
                continue
            nums = [n for n in NUM.findall(prose(ln)) if n not in CONVENTIONAL]
            if nums:
                bad.append(f"{name}:{i}: literal number(s) {', '.join(dict.fromkeys(nums))} not from a macro "
                           "(mark '% claims-ok: <reason>' if an input)")
    return bad


def check_structure(paper, tex):
    bad = []
    alltext = "\n".join(strip_comment(l) for lines in tex.values() for l in lines)
    labels = re.findall(r"\\label\{([^}]+)\}", alltext)
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", alltext))
    for d in sorted({l for l in labels if labels.count(l) > 1}):
        bad.append(f"duplicate label {d}")
    for r in sorted(refs - set(labels)):
        bad.append(f"unresolved reference {r}")
    for l in labels:
        if l.split(":")[0] in ("fig", "tab", "eq", "app") and l not in refs:
            bad.append(f"label never referenced: {l}")
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", alltext):
        if not any((paper / "figures" / f"{g}{ext}").exists() for ext in ("", ".pdf", ".png")):
            bad.append(f"missing figure file {g}")
    for t in re.findall(r"\\input\{(tables/[^}]+)\}", alltext):
        if not (paper / f"{t}.tex").exists():
            bad.append(f"missing table file {t}")
    secs = [s.lower() for s in re.findall(r"\\section\*?\{([^}]+)\}", alltext)]
    for req in REQUIRED_SECTIONS:
        if not any(req in s for s in secs):
            bad.append(f"missing section: {req}")
    for env in ("abstract", "keyword"):
        if f"\\begin{{{env}}}" not in alltext:
            bad.append(f"missing environment: {env}")
    for heading in ("Data availability", "CRediT", "Declaration of competing interest"):
        if heading.lower() not in alltext.lower():
            bad.append(f"missing statement: {heading}")
    return bad, len(labels), len(refs), secs


def check_syntax(paper):
    """Balanced braces, \\begin/\\end pairs and inline-math dollars per file (no TeX installed here)."""
    bad = []
    files = [paper / "main.tex", paper / "numbers.tex"] + sorted((paper / "sections").glob("*.tex")) \
        + sorted((paper / "tables").glob("*.tex"))
    for f in files:
        s = "\n".join(strip_comment(ln) for ln in f.read_text(encoding="utf-8").splitlines())
        s2 = s.replace("\\{", "").replace("\\}", "")
        if s2.count("{") != s2.count("}"):
            bad.append(f"{f.name}: unbalanced braces ({s2.count('{')} open, {s2.count('}')} close)")
        stack = []
        for kind, env in re.findall(r"\\(begin|end)\{(\w+\*?)\}", s):
            if kind == "begin":
                stack.append(env)
            elif not stack or stack.pop() != env:
                bad.append(f"{f.name}: \\end{{{env}}} does not match")
                break
        if stack:
            bad.append(f"{f.name}: unclosed environment(s) {stack}")
        if len(re.findall(r"(?<!\\)\$", s)) % 2:
            bad.append(f"{f.name}: odd number of $")
    return bad, len(files)


def check_spelling(tex):
    bad = []
    for name, lines in tex.items():
        for i, ln in enumerate(lines, 1):
            cmds = "cite[tp]?|label|ref|eqref|input|includegraphics|usepackage|texttt|url|begin|end"
            s = re.sub(r"\\(" + cmds + r")\{[^}]*\}", " ", strip_comment(ln))
            s = re.sub(r"\b\w+=", " ", s)  # elsarticle keys such as organization=
            s = re.sub(r"\\[A-Za-z]+", " ", s)
            for pat, uk in US_UK:
                if uk is None:
                    continue
                for m in re.finditer(pat, s, flags=re.I):
                    w = m.group(0)
                    if uk == "ise" and w.lower() in ISE_EXEMPT:
                        continue
                    bad.append(f"{name}:{i}: '{w}' (UK: {uk})")
    return bad


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    paper = (ROOT / (argv[0] if argv else "paper/paper2")).resolve()
    tex = read_tex(paper)
    fails = 0
    shown = paper.relative_to(ROOT) if paper.is_relative_to(ROOT) else paper
    print(f"lint_paper: {shown} ({len(tex)} .tex files)")

    bad, unused_nums = check_macros(paper, tex)
    print(f"[macros]    {'FAIL' if bad else 'ok'}: {len(bad)} undefined; {len(unused_nums)} generated macros unused")
    for b in bad:
        print("   ", b)
    if unused_nums:
        print("    unused (info):", " ".join(unused_nums))
    fails += bool(bad)

    bad, unused, n_cited, n_bib = check_citations(paper, tex)
    print(f"[citations] {'FAIL' if bad else 'ok'}: {n_cited} keys cited, {n_bib} in refs.bib, all VERIFIED: {not bad}")
    for b in bad:
        print("   ", b)
    if unused:
        print("    bib entries never cited (info):", " ".join(unused))
    fails += bool(bad)

    bad = check_claims(tex)
    print(f"[claims]    {'FAIL' if bad else 'ok'}: {len(bad)} line(s) with unsourced literal numbers")
    for b in bad:
        print("   ", b)
    fails += bool(bad)

    bad, n_lab, n_ref, secs = check_structure(paper, tex)
    print(f"[structure] {'FAIL' if bad else 'ok'}: {len(secs)} sections, {n_lab} labels, {n_ref} references")
    for b in bad:
        print("   ", b)
    fails += bool(bad)

    bad, n_files = check_syntax(paper)
    print(f"[syntax]    {'FAIL' if bad else 'ok'}: braces, environments and $ balanced in {n_files} files")
    for b in bad:
        print("   ", b)
    fails += bool(bad)

    bad = check_spelling(tex)
    print(f"[spelling]  {'FAIL' if bad else 'ok'}: {len(bad)} US spelling(s)")
    for b in bad:
        print("   ", b)
    fails += bool(bad)

    print(f"lint_paper: {'PASS' if not fails else f'FAIL ({fails} check(s))'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
