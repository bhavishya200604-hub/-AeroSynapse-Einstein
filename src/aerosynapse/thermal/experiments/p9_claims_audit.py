"""Phase 9 Step 9.5: claims audit — every number in the write-up traced to docs/RESULTS_REGISTRY.md.

    python -m aerosynapse.thermal.experiments.p9_claims_audit [--tag default]

Write-up = the two papers (paper/paper1, paper/paper2) and the Markdown reports (docs/FINAL_REPORT.md,
docs/MODEL_CARD.md).

Papers: a number in the prose must be a generated macro (enforced by paper/tools/lint_paper.py, run here); each macro
in numbers.tex carries the registry rows it comes from in its comment. The audit checks, for every macro the paper
uses, that it names at least one registry row, that every named row exists, whether its value appears in those rows
(verbatim / rounded, or derived from them), and whether it changed against the committed numbers.tex and against the
evidence freeze (tag evidence-freeze-p6) — a changed value is STALE until the prose is re-read (Step 6B.12).

Markdown: every number on a line must appear in a registry row cited on that line as [R...]. Identifiers (Phase 5,
Step 6.3, R6.2a, C2.8, D141, RQ1, 1D / 2D), counts 0-10 and the conventional 90 / 95 / 100 are not claims.
Writes results/p9_claims_audit_<tag>/metrics.json and docs/tables/claims_audit.md.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

REGISTRY = ROOT / "docs" / "RESULTS_REGISTRY.md"
PAPERS = ["paper1", "paper2"]
MARKDOWN = ["docs/FINAL_REPORT.md", "docs/MODEL_CARD.md", "README.md::Key findings"]  # file[::section]
FREEZE_TAG = "evidence-freeze-p6"
ID_RE = re.compile(r"\bR(?:\d|B)[\w.\-]*\w")
NUM_RE = re.compile(r"(?<![\w./])[-+−]?\d+(?:[.,]\d+)?(?:[eE][-+−]?\d+)?")  # also the far end of 'a-b' ranges
CONVENTIONAL = {str(i) for i in range(0, 11)} | {"90", "95", "100"}
# identifiers and input conditions ('Re = 5000', 'Bi 30') are not claims
IDENT_BEFORE = re.compile(
    r"(Phase|Phases|Step|Steps|step|phase|RQ|claim|claims|Part|patch|Section|Ra|Re|Pe|Bi|Pr)\s*=?\s*$"
)


def resolve(x: str, reg: dict[str, str]) -> list[str]:
    """Registry rows a citation names: an exact id (R6.2a), a letter range (R6.3a-b -> R6.3a, R6.3b) or a section /
    group prefix (R1.7 -> R1.7a, R1.7b, ...). Empty if nothing matches."""
    if x in reg:
        return [x]
    m = re.fullmatch(r"(.*?)([a-z])-([a-z])", x)
    if m:
        got = [m.group(1) + chr(c) for c in range(ord(m.group(2)), ord(m.group(3)) + 1)]
        got = [g for g in got if g in reg]
        if got:
            return got
    return [k for k in reg if k.startswith(x) and k != x]


CONFIG_RE = re.compile(r"\[config: ([^:\]]+): ([^\]]+)\]")


def config_values(text: str) -> str:
    """Values of the settings a block cites as [config: <yaml file>: <dotted.key>] (conditions, not results), read
    from that file, so a stated setting is checked like a registry number."""
    import yaml

    out = []
    for f, key in CONFIG_RE.findall(text):
        try:
            v = yaml.safe_load((ROOT / f.strip()).read_text(encoding="utf-8"))
            for k in key.strip().split("."):
                v = v[k]
            out.append(str(v))
        except (OSError, KeyError, TypeError):
            out.append("")
    return " ".join(out)


def cited(ids: list[str], reg: dict[str, str]) -> tuple[str, list[str]]:
    """(text of every row the citations name, citations that name no row)."""
    rows, missing = [], []
    for x in ids:
        r = resolve(x, reg)
        rows += r
        if not r:
            missing.append(x)
    return " ".join(reg[r] for r in dict.fromkeys(rows)), missing


def registry() -> dict[str, str]:
    """{row id: the row's text plus its section's setup text} — a section's prose (heading, 'Setup: ...') states the
    conditions its rows share (ranges, network sizes, member counts), so citing a row cites them too."""
    rows, section_rows, section_text = {}, [], []

    def close():
        extra = " ".join(section_text)
        for r in section_rows:
            rows[r] = rows[r] + " " + extra

    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            close()
            section_rows, section_text = [], [line]
            continue
        m = re.match(r"^\| (R[\w.\-]+) \|(.*)$", line)
        if m:
            rows[m.group(1)] = m.group(2)
            section_rows.append(m.group(1))
        elif not line.startswith("|"):
            section_text.append(line)
    close()
    return rows


def blocks(lines: list[str]) -> list[tuple[int, str]]:
    """Paragraphs, list items and table rows (with their continuation lines): the unit a citation belongs to."""
    out: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        starts = not s or s.startswith(("#", "- ", "* ", "|")) or re.match(r"^\d+\. ", s)
        if starts or not out:
            out.append((i, line))
        else:
            out[-1] = (out[-1][0], out[-1][1] + " " + s)
        if not s:
            out.append((i, ""))
    return [b for b in out if b[1].strip()]


def norm(s: str) -> str:
    return s.replace("−", "-").replace(" ", " ").replace(" ", " ").replace("\\,", "")


def numbers(text: str) -> list[float]:
    out = []
    for t in NUM_RE.findall(norm(text)):
        try:
            out.append(float(t.replace(",", "").replace("−", "-")))
        except ValueError:
            pass
    return out


def in_rows(value: float, token: str, rows_text: str) -> str:
    """'verbatim' if the token is in the rows, 'rounded' if a number there rounds to it, else ''."""
    t = token.lstrip("+-−")
    # whole numbers only: '0.002' must not count as found inside '0.00201'
    if re.search(r"(?<![\d.])" + re.escape(t) + r"(?![\d])", norm(rows_text)):
        return "verbatim"
    digits = len(t.split(".")[1]) if "." in t and "e" not in t.lower() else None
    for x in numbers(rows_text):
        for y in (x, abs(x), 100 * x, abs(100 * x)):
            if digits is not None and round(y, digits) == round(abs(value), digits):
                return "rounded"
            if y != 0 and abs(abs(value) - y) / abs(y) < 5e-3:
                return "rounded"
    return ""


# ---------------------------------------------------------------------------- Markdown
def audit_markdown(path: str, reg: dict[str, str]) -> dict:
    file, _, section = path.partition("::")
    lines = (ROOT / file).read_text(encoding="utf-8").replace("–", "-").splitlines()  # en-dash ranges: R6.1j-k
    if section:  # only that section of the file (up to the next heading of the same level)
        start = next(i for i, x in enumerate(lines) if x.strip() == f"## {section}")
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
        lines = [""] * start + lines[start:end]  # keep the line numbers of the file
    claims, missing_ids = [], set()
    for i, line in blocks(lines):
        if line.startswith("#") or "](" in line and not ID_RE.search(line):
            text = re.sub(r"\[[^\]]*\]\([^)]*\)", "", line)  # link targets are not claims
        else:
            text = line
        text = re.sub(r"`[^`]*`", "", text)  # code spans: commands, paths
        ids = sorted(set(ID_RE.findall(text)))
        clean = ID_RE.sub("", text)
        for m in NUM_RE.finditer(norm(clean)):
            tok = m.group(0)
            before = norm(clean)[: m.start()]
            after = norm(clean)[m.end() : m.end() + 2]
            core = tok.lstrip("+-−")
            if (
                core in CONVENTIONAL
                or IDENT_BEFORE.search(before)
                or after[:1].upper() in ("D",)
                and core in ("1", "2")
            ):
                continue
            if re.search(r"[A-Z][\w.]*$", before) and not before.endswith(" "):  # part of an identifier (C2.8, D141)
                continue
            try:
                val = float(core.replace(",", ""))
            except ValueError:
                continue
            rows_text, miss = cited(ids, reg)
            rows_text += " " + config_values(text)
            found = in_rows(val, core, rows_text) if ids else ""
            missing_ids.update(miss)
            claims.append({"line": i, "number": tok, "ids": ids, "found": found or ("no cited row" if not ids else "")})
    bad = [c for c in claims if not c["found"] or c["found"] == "no cited row"]
    return {
        "file": path,
        "numbers": len(claims),
        "traced": len(claims) - len(bad),
        "untraced": bad,
        "missing_ids": sorted(missing_ids),
        "pass": not bad and not missing_ids,
    }


# ---------------------------------------------------------------------------- papers
MACRO_RE = re.compile(r"^\\newcommand\{\\(\w+)\}\{(.*)\}%\s*(.*)$")


def macros(text: str) -> dict[str, tuple[str, str]]:
    out = {}
    for line in text.splitlines():
        m = MACRO_RE.match(line.strip())
        if m:
            out[m.group(1)] = (m.group(2), m.group(3))
    return out


def git_show(ref: str, rel: str) -> str | None:
    r = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


def latex_value(v: str) -> tuple[float | None, str]:
    s = norm(v).replace("\\ensuremath{-}", "-").replace("\\ensuremath", "").replace("{", "").replace("}", "")
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*\\times\s*10\^([-+]?\d+)", s)
    if m:
        tok = f"{m.group(1)}e{m.group(2)}"
        return float(m.group(1)) * 10 ** int(m.group(2)), tok
    m = re.search(r"[-+]?\d[\d ]*(?:\.\d+)?", s)
    if not m:
        return None, s
    tok = m.group(0).replace(" ", "")
    return float(tok), tok


def audit_paper(name: str, reg: dict[str, str], head_ref: str) -> dict:
    d = ROOT / "paper" / name
    cur = macros((d / "numbers.tex").read_text(encoding="utf-8"))
    head = macros(git_show(head_ref, f"paper/{name}/numbers.tex") or "")
    frozen = macros(git_show(FREEZE_TAG, f"paper/{name}/numbers.tex") or "")
    tex = "\n".join(p.read_text(encoding="utf-8") for p in d.rglob("*.tex") if p.name != "numbers.tex")
    used = sorted(n for n in cur if re.search(r"\\" + n + r"(?![A-Za-z])", tex))
    rows, missing = [], set()
    for n in used:
        value, comment = cur[n]
        ids = sorted(set(ID_RE.findall(comment)))
        rows_text, miss = cited(ids, reg)
        missing.update(miss)
        num, tok = latex_value(value)
        found = in_rows(num, tok, rows_text) if (num is not None and ids) else ""
        status = "SUPPORTED"
        if n in head and head[n][0] != value:
            status = "STALE (changed since the last commit)"
        elif n in frozen and frozen[n][0] != value:
            status = "CHANGED SINCE FREEZE (gap runs / later evidence)"
        rows.append(
            {
                "macro": n,
                "value": value,
                "ids": ids,
                "in_row": found or "derived",
                "status": status,
                "comment": comment[:120],
            }
        )
    no_id = [r["macro"] for r in rows if not r["ids"]]
    stale = [r["macro"] for r in rows if r["status"].startswith("STALE")]
    lint = subprocess.run(
        [sys.executable, "-m", "aerosynapse.thermal.paper_tools.lint_paper", f"paper/{name}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        "paper": name,
        "macros_defined": len(cur),
        "macros_used": len(used),
        "rows": rows,
        "no_registry_id": no_id,
        "missing_ids": sorted(missing),
        "stale": stale,
        "lint_exit": lint.returncode,
        "lint_tail": lint.stdout.strip().splitlines()[-6:],
        "pass": not no_id and not missing and lint.returncode == 0,
    }


def report(m: dict) -> str:
    L = [
        "# Claims audit (auto-generated by experiments/p9_claims_audit.py)",
        "",
        f"git {m['git'][:7] if m.get('git') else '?'}; every number in the write-up traced to "
        "docs/RESULTS_REGISTRY.md. Overall: **" + ("PASS" if m["pass"] else "FAIL") + "**.",
        "",
        "## Summary",
        "",
        "| document | numbers / macros | traced | problems | status |",
        "|---|---|---|---|---|",
    ]
    for p in m["papers"]:
        prob = len(p["no_registry_id"]) + len(p["missing_ids"]) + (p["lint_exit"] != 0)
        L.append(
            f"| paper/{p['paper']} | {p['macros_used']} macros used ({p['macros_defined']} defined) | "
            f"{p['macros_used'] - len(p['no_registry_id'])} with registry rows | {prob} (stale: {len(p['stale'])}) "
            f"| {'PASS' if p['pass'] else 'FAIL'} |"
        )
    for d in m["markdown"]:
        L.append(
            f"| {d['file']} | {d['numbers']} numbers | {d['traced']} | {len(d['untraced'])} untraced, "
            f"{len(d['missing_ids'])} unknown ids | {'PASS' if d['pass'] else 'FAIL'} |"
        )
    for d in m["markdown"]:
        if d["untraced"]:
            L += [
                "",
                f"## Untraced numbers in {d['file']}",
                "",
                "| line | number | cited rows | issue |",
                "|---|---|---|---|",
            ]
            L += [
                f"| {c['line']} | {c['number']} | {', '.join(c['ids']) or '—'} | {c['found'] or 'not in the cited rows'} |"
                for c in d["untraced"]
            ]
    for p in m["papers"]:
        L += [
            "",
            f"## paper/{p['paper']} — macros used",
            "",
            f"Lint (claims: every number in the prose is a macro): exit {p['lint_exit']}.",
            "",
            "| macro | value | registry rows | value in the rows | status |",
            "|---|---|---|---|---|",
        ]
        for r in p["rows"]:
            v = r["value"].replace("|", "\\|")
            L.append(f"| `{r['macro']}` | {v} | {', '.join(r['ids']) or '**none**'} | {r['in_row']} | {r['status']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--head", default="HEAD", help="commit the regenerated numbers are compared with")
    a = ap.parse_args()
    reg = registry()
    with Experiment(f"p9_claims_audit_{a.tag}", {"seed": 0, "head": a.head}) as ex:
        papers = [audit_paper(p, reg, a.head) for p in PAPERS]
        md = [audit_markdown(f, reg) for f in MARKDOWN]
        ex.metrics.update(
            {
                "registry_rows": len(reg),
                "papers": papers,
                "markdown": md,
                "git": git_hash(),
                "pass": all(p["pass"] for p in papers) and all(d["pass"] for d in md),
            }
        )
        out = ROOT / "docs" / "tables" / "claims_audit.md"
        out.write_text(report(ex.metrics), encoding="utf-8")
    for p in papers:
        print(
            f"paper/{p['paper']}: {p['macros_used']} macros used, no id {len(p['no_registry_id'])}, unknown ids "
            f"{p['missing_ids']}, stale {len(p['stale'])}, lint {p['lint_exit']}"
        )
    for d in md:
        print(f"{d['file']}: {d['numbers']} numbers, untraced {len(d['untraced'])}, unknown ids {d['missing_ids']}")
        for c in d["untraced"][:30]:
            print(f"   line {c['line']}: {c['number']} {c['ids']} {c['found']}")
    print("audit", "PASS" if ex.metrics["pass"] else "FAIL", out)


if __name__ == "__main__":
    main()
