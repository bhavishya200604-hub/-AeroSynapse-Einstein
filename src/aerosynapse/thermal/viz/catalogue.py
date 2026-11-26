"""Phase 7, Step 7.2: catalogue of every figure in docs/figures — which script and figure function make it, how it is
re-rendered, which saved results it reads, whether those results are present, and what a rerun costs.

    python -m aerosynapse.thermal.viz.catalogue        -> docs/figures/CATALOGUE.md and docs/figures/catalogue.json

Matching: every figure name passed to save_fig(...) in experiments/p*.py (a string, an f-string, or a variable assigned
an f-string) becomes a pattern in which each {...} matches name characters; the {tag} / {profile} placeholder is
captured as the run tag. A figure belongs to the script whose most specific literal prefix matches its file name. The
figure function is the call passed to save_fig as the figure; results are the script's Experiment(...) folders for
that tag; the rerun cost is the wall time recorded by that run.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
FIG = ROOT / "docs" / "figures"
SCRIPTS = sorted((__import__("aerosynapse.thermal._home", fromlist=["HOME"]).CODE / "experiments").glob("p*.py"))
PROFILES = ("smoke", "full")  # tags that are profiles (--profile); others are --tag values
RERENDER_FLAGS = ("--figures-only", "--table-only", "--analysis-only")


@dataclass
class Template:
    script: str
    literal: str  # the name as written, e.g. "p6_5_spectra_{tag}"
    regex: str
    prefix: str  # literal text before the first {...}
    function: str | None = None  # figure function passed to save_fig, when it can be read
    line: int = 0


@dataclass
class Entry:
    name: str
    files: list[str]
    script: str | None = None
    template: str | None = None
    function: str | None = None
    line: int = 0
    tag: str | None = None
    results: list[str] = field(default_factory=list)
    results_present: bool = False
    run_wall_s: float | None = None  # wall time of the run that wrote the results (a full rerun costs about this)
    rerender: str | None = None
    used_by_papers: list[str] = field(default_factory=list)
    status: str = ""


def _literal(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            parts.append(str(v.value) if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}")
        return "".join(parts)
    return None


def _regex(lit: str) -> str:
    parts, tagged = [], False
    for piece in re.split(r"(\{[^}]*\})", lit):
        if not piece:
            continue
        if piece.startswith("{"):
            if ("tag" in piece or "profile" in piece) and not tagged:
                parts.append("(?P<tag>[A-Za-z0-9_.+-]+)")
                tagged = True
            else:
                parts.append("[A-Za-z0-9_.+-]+?")
        else:
            parts.append(re.escape(piece))
    return "^" + "".join(parts) + "$"


def _templates(path: Path) -> list[Template]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(ROOT).as_posix()
    assigned: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = _literal(node.value)
            if v and re.match(r"p\d", v):
                assigned[node.targets[0].id] = v
    out: list[Template] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "save_fig"):
            continue
        if len(node.args) < 2:
            continue
        lit = _literal(node.args[1])
        if lit is None and isinstance(node.args[1], ast.Name):  # save_fig(fig, name, ...) with name = f"..."
            lit = assigned.get(node.args[1].id)
        if not lit or not re.match(r"p\d", lit):
            continue
        fn = ast.unparse(node.args[0].func) if isinstance(node.args[0], ast.Call) else None
        out.append(Template(rel, lit, _regex(lit), lit.split("{")[0], fn, node.lineno))
    return out


def _flags(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    return [f for f in RERENDER_FLAGS if f'"{f}"' in src]


def _experiment_ids(path: Path) -> list[str]:
    """Experiment(...) id templates of a script (results/<id>/), including both branches of a conditional id."""
    src = path.read_text(encoding="utf-8")
    ids = re.findall(r"Experiment\(\s*f?\"([^\"]+)\"", src)
    ids += re.findall(r"Experiment\([^\n]*?\belse\s+f?\"([^\"]+)\"", src)
    return list(dict.fromkeys(ids))


def _paper_uses() -> dict[str, list[str]]:
    """docs/figures names reused by the papers' make_figures.py."""
    out: dict[str, list[str]] = {}
    for mf in (ROOT / "paper").glob("paper*/make_figures.py"):
        for m in re.finditer(r"\"(p\d[A-Za-z0-9_]+)\"", mf.read_text(encoding="utf-8")):
            out.setdefault(m.group(1), []).append(mf.parent.name)
    return out


def _wall(run: str) -> float | None:
    f = ROOT / run / "metrics.json"
    if not f.exists():
        return None
    w = json.loads(f.read_text(encoding="utf-8")).get("_meta", {}).get("wall_time_s")
    return float(w) if w else None


def build() -> list[Entry]:
    temps = [t for s in SCRIPTS for t in _templates(s)]
    flags = {s.relative_to(ROOT).as_posix(): _flags(s) for s in SCRIPTS}
    exps = {s.relative_to(ROOT).as_posix(): _experiment_ids(s) for s in SCRIPTS}
    papers = _paper_uses()
    names = sorted({p.stem for p in FIG.glob("*.png")} | {p.stem for p in FIG.glob("*.pdf")})
    entries = []
    for n in names:
        files = sorted(p.name for p in FIG.glob(f"{n}.*") if p.suffix in (".png", ".pdf"))
        e = Entry(n, files, used_by_papers=sorted(set(papers.get(n, []))))
        cands = [(t, re.match(t.regex, n)) for t in temps]
        cands = [(t, m) for t, m in cands if m]
        if not cands:
            e.status = "generator not found"
            entries.append(e)
            continue
        t, m = max(cands, key=lambda c: (len(c[0].prefix), -c[0].literal.count("{")))
        e.script, e.template, e.function, e.line = t.script, t.literal, t.function, t.line
        e.tag = m.groupdict().get("tag")
        if e.tag and t.literal.rstrip().endswith("}"):
            # the tag is the trailing placeholder; with several placeholders the split is ambiguous ('C_cavity_smoke'):
            # take the shortest trailing run of name segments for which a results folder of this script exists
            segs = n.split("_")
            for k in range(1, len(segs)):
                cand = "_".join(segs[-k:])
                if any(
                    (ROOT / "results" / re.sub(r"\{[^}]*\}", cand, x) / "metrics.json").exists()
                    for x in exps[t.script]
                    if "{" in x
                ):
                    e.tag = cand
                    break
        for x in exps[t.script]:
            if "{" in x and not e.tag:
                continue
            rid = re.sub(r"\{[^}]*\}", e.tag or "", x)
            if (ROOT / "results" / rid / "metrics.json").exists():
                e.results.append(f"results/{rid}")
        stem = Path(t.script).stem  # fallback: results folders are named after the script (+ _<tag>)
        for rid in ([f"{stem}_{e.tag}"] if e.tag else []) + [stem]:
            if not e.results and (ROOT / "results" / rid / "metrics.json").exists():
                e.results.append(f"results/{rid}")
        if not e.results and exps[t.script]:  # expected but missing: report the first id
            x = exps[t.script][0]
            if e.tag or "{" not in x:
                e.results.append("results/" + re.sub(r"\{[^}]*\}", e.tag or "", x))
        e.results_present = any((ROOT / r / "metrics.json").exists() for r in e.results)
        walls = [w for w in (_wall(r) for r in e.results) if w]
        e.run_wall_s = max(walls) if walls else None
        mod = t.script[:-3].replace("/", ".")
        opt = ""
        if e.tag:
            src = (ROOT / t.script).read_text(encoding="utf-8")
            has_profile = '"--profile"' in src or "cli_profile(" in src
            has_tag = '"--tag"' in src
            if e.tag in PROFILES and has_profile:
                opt = f" --profile {e.tag}"
            elif has_tag:
                opt = f" --tag {e.tag}"
            elif has_profile:
                opt = f" --profile {e.tag}"
        fl = flags[t.script]
        if fl:
            e.rerender = f"python -m {mod}{opt} {fl[0]}"
            e.status = "re-renderable from saved results" if e.results_present else "re-render flag, results missing"
        else:
            e.rerender = f"python -m {mod}{opt}"
            e.status = "rerun of the script (no re-render flag)"
        entries.append(e)
    return entries


def write(entries: list[Entry]) -> tuple[Path, Path]:
    j = FIG / "catalogue.json"
    j.write_text(json.dumps([asdict(e) for e in entries], indent=1), encoding="utf-8")
    by: dict[str, int] = {}
    for e in entries:
        by[e.status] = by.get(e.status, 0) + 1
    L = [
        "# Figure catalogue (auto-generated by `python -m aerosynapse.thermal.viz.catalogue`; Phase 7, Step 7.2)",
        "",
        f"{len(entries)} figures in docs/figures. Status counts: "
        + "; ".join(f"{k}: {v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1]))
        + ". Rerun cost = wall time recorded by the run that wrote the results.",
        "",
        "| figure | script (line) | figure function | results | rerun cost [s] | re-render command | papers | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        res = "<br>".join(f"`{r}`" + ("" if (ROOT / r / "metrics.json").exists() else " (missing)") for r in e.results)
        L.append(
            f"| {e.name} | {e.script or '—'}{f' ({e.line})' if e.line else ''} | {e.function or '—'} | {res or '—'} | "
            f"{f'{e.run_wall_s:.0f}' if e.run_wall_s else '—'} | {f'`{e.rerender}`' if e.rerender else '—'} | "
            f"{', '.join(e.used_by_papers) or '—'} | {e.status} |"
        )
    m = FIG / "CATALOGUE.md"
    m.write_text("\n".join(L) + "\n", encoding="utf-8")
    return m, j


def main() -> None:
    entries = build()
    m, j = write(entries)
    by: dict[str, int] = {}
    for e in entries:
        by[e.status] = by.get(e.status, 0) + 1
    print(f"{len(entries)} figures -> {m.relative_to(ROOT)}, {j.relative_to(ROOT)}")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
