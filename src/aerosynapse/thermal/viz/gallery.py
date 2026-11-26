"""Phase 7 Step 7.5: the figure gallery of docs/figures, generated from the catalogue (docs/figures/catalogue.json).

    python -m aerosynapse.thermal.viz.gallery          # writes docs/figures/index.html and docs/figures/GALLERY.md

index.html: every figure as a card (image, the script and figure function that draw it, its results folder, the
command that re-renders it, the papers that use it), filters by phase and a search box; GALLERY.md: the same as
tables, readable on GitHub.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
FIG = ROOT / "docs" / "figures"
PHASES = {
    "p1": "Phase 1 · verified solvers",
    "p2": "Phase 2 · data",
    "p3": "Phase 3 · baseline PINN",
    "p4": "Phase 4 · PINN methods and 2D",
    "p5": "Phase 5 · inverse problems",
    "p6": "Phase 6 · analysis",
    "p6b": "Phase 6B · gap runs for the papers",
    "p7": "Phase 7 · visualisation",
}


def phase_of(name: str) -> str:
    if name.startswith("p6b"):
        return "p6b"
    m = re.match(r"p(\d)", name)
    return f"p{m.group(1)}" if m else "other"


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Figure gallery</title>
<style>
:root {{ --ink:#141414; --muted:#5b5b5b; --line:#dcdcdc; --bg:#fff; --card:#f7f7f5; --accent:#0072B2; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink:#ececec; --muted:#a9a9a9; --line:#3a3a3a; --bg:#151515; --card:#1f1f1f; --accent:#56B4E9; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:20px 16px 60px; font-family:Georgia,serif; color:var(--ink); background:var(--bg); }}
main {{ max-width:1240px; margin:0 auto; }}
h1 {{ font-size:1.6rem; margin:0 0 .3rem; }} h2 {{ font-size:1.15rem; margin:2rem 0 .7rem; }}
.muted {{ color:var(--muted); font-size:.9rem; }}
.bar {{ display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:1rem 0; position:sticky; top:0;
  background:var(--bg); padding:.5rem 0; z-index:2; }}
.bar button {{ font:inherit; font-size:.85rem; padding:.3rem .8rem; border-radius:999px; border:1px solid var(--line);
  background:transparent; color:var(--ink); cursor:pointer; }}
.bar button.on {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
.bar input {{ font:inherit; padding:.35rem .7rem; border:1px solid var(--line); border-radius:8px; background:var(--bg);
  color:var(--ink); min-width:220px; flex:1; max-width:360px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(min(100%, 340px), 1fr)); gap:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; display:flex;
  flex-direction:column; }}
.card img {{ width:100%; height:220px; object-fit:contain; background:#fff; border-bottom:1px solid var(--line); }}
.card .b {{ padding:10px 12px; font-size:.84rem; line-height:1.4; }}
.card .n {{ font-weight:bold; font-size:.9rem; overflow-wrap:anywhere; }}
code {{ font-size:.78rem; overflow-wrap:anywhere; }}
.tag {{ display:inline-block; font-size:.72rem; padding:.05rem .45rem; border-radius:999px;
  border:1px solid var(--line);
  margin-right:.25rem; }}
a {{ color:var(--accent); }}
</style></head><body><main>
<h1>Figure gallery</h1>
<p class="muted">{n} figures of the project, drawn in one publication style (src/viz/style.py) and re-rendered from
saved results by <code>python -m aerosynapse.thermal.experiments.p7_3_rerender</code>. Each card names the script and function that draw
the figure, the results it is drawn from and the command that re-renders it. Animations:
<a href="../animations/index.html">docs/animations</a>. Catalogue: <a href="CATALOGUE.md">CATALOGUE.md</a>.</p>
<div class="bar"><input id="q" type="search" placeholder="search name, script, function" aria-label="search">
{buttons}</div>
{sections}
</main><script>
const q = document.getElementById('q'); let phase = 'all';
function apply() {{
  const s = q.value.toLowerCase();
  document.querySelectorAll('.card').forEach(c => {{
    const ok = (phase === 'all' || c.dataset.phase === phase) && c.dataset.text.includes(s);
    c.style.display = ok ? '' : 'none'; }});
  document.querySelectorAll('section').forEach(sec => {{
    sec.style.display = [...sec.querySelectorAll('.card')].some(c => c.style.display !== 'none') ? '' : 'none'; }});
}}
document.querySelectorAll('.bar button').forEach(b => b.onclick = () => {{
  phase = b.dataset.phase; document.querySelectorAll('.bar button').forEach(x => x.classList.toggle('on', x === b));
  apply(); }});
q.oninput = apply;
</script></body></html>
"""


def build(out_dir: Path = FIG) -> dict:
    figs = json.loads((out_dir / "catalogue.json").read_text(encoding="utf-8"))
    groups: dict[str, list] = {}
    for f in figs:
        groups.setdefault(phase_of(f["name"]), []).append(f)
    e = html.escape
    order = [p for p in PHASES if p in groups] + [p for p in groups if p not in PHASES]
    sections, md = (
        [],
        [
            "# Figure gallery",
            "",
            f"{len(figs)} figures; generated by `python -m aerosynapse.thermal.viz.gallery` from "
            "the catalogue. Interactive version: [index.html](index.html).",
            "",
        ],
    )
    for ph in order:
        cards = []
        md += [f"## {PHASES.get(ph, ph)}", "", "| figure | drawn by | re-render |", "|---|---|---|"]
        for f in sorted(groups[ph], key=lambda g: g["name"]):
            png, pdf = f"{f['name']}.png", f"{f['name']}.pdf"
            fn = f"{f['script']}" + (f" · {f['function']}()" if f.get("function") else "")
            papers = " ".join(f'<span class="tag">{e(p)}</span>' for p in f.get("used_by_papers") or [])
            res = ", ".join(f.get("results") or []) or "—"
            text = " ".join([f["name"], f["script"], f.get("function") or "", res]).lower()
            cards.append(
                f'<article class="card" data-phase="{ph}" data-text="{e(text)}">'
                f'<a href="{e(png)}"><img loading="lazy" src="{e(png)}" alt="{e(f["name"])}"></a>'
                f'<div class="b"><div class="n">{e(f["name"])}</div>'
                f'<div>{e(fn)}</div><div class="muted">results: <code>{e(res)}</code></div>'
                f'<div class="muted">re-render: <code>{e(f["rerender"])}</code></div>'
                f'<div>{papers} <a href="{e(pdf)}">PDF</a> · <a href="{e(png)}">PNG</a></div></div></article>'
            )
            md.append(f"| ![{f['name']}]({png})<br>`{f['name']}` | `{fn}` | `{f['rerender']}` |")
        sections.append(
            f'<section><h2>{e(PHASES.get(ph, ph))} ({len(groups[ph])})</h2><div class="grid">'
            + "".join(cards)
            + "</div></section>"
        )
        md.append("")
    buttons = '<button class="on" data-phase="all">all</button>' + "".join(
        f'<button data-phase="{p}">{e(PHASES.get(p, p).split(" · ")[0])}</button>' for p in order
    )
    (out_dir / "index.html").write_text(
        PAGE.format(n=len(figs), buttons=buttons, sections="".join(sections)), encoding="utf-8"
    )
    (out_dir / "GALLERY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {
        "figures": len(figs),
        "phases": {p: len(groups[p]) for p in order},
        "index": str((out_dir / "index.html").relative_to(ROOT)),
        "markdown": str((out_dir / "GALLERY.md").relative_to(ROOT)),
    }


if __name__ == "__main__":
    print(build())
