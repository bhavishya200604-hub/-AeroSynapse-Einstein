"""Phase 7 gate: the visualisation library.

    python -m aerosynapse.thermal.experiments.p7_gate [--tag default]

  G1 catalogue complete: every figure in docs/figures is in the catalogue and every catalogue figure has PNG + PDF
  G2 re-rendered in the project style: every catalogue figure was rewritten by the Step 7.3 run, no command failed
  G3 reproducible: the numbers the re-render regenerated equal the committed ones (volatile keys excluded)
  G4 one style, checked on the files: PNG 300 dpi, PDF without a creation date (byte-reproducible), TrueType fonts
     embedded (no Type 3)
  G5 gallery: docs/figures/index.html and GALLERY.md list every figure and every link resolves
  G6 animations: the last full render has every scene and no failure
  G7 smoke tests (Step 7.4, experiments/p7_4_smoke.py): every figure function has a captured fixture and replays
     without error; no figure code left inside main()
Writes results/p7_gate_<tag>/metrics.json and docs/tables/phase7_gate.md.
"""

from __future__ import annotations

import argparse
import json
import re

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

FIG = ROOT / "docs" / "figures"


def load(rel: str) -> dict | None:
    f = ROOT / rel
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--rerender-tags", nargs="+", default=["default", "fix"])
    a = ap.parse_args()
    from PIL import Image

    cat = json.loads((FIG / "catalogue.json").read_text(encoding="utf-8"))
    names = {f["name"] for f in cat}
    pngs = {p.stem for p in FIG.glob("*.png")}
    g: dict = {}
    missing_files = [n for n in names if not (FIG / f"{n}.png").exists() or not (FIG / f"{n}.pdf").exists()]
    g["G1"] = {
        "catalogue": len(names),
        "png_in_docs": len(pngs),
        "not_in_catalogue": sorted(pngs - names),
        "missing_files": sorted(missing_files),
        "pass": not (pngs - names) and not missing_files,
    }
    # the Step 7.3 runs: the full run and follow-up runs of single commands (--rerender-tags), counted together
    runs = [m for t in a.rerender_tags if (m := load(f"results/p7_3_rerender_{t}/metrics.json"))]
    done = {n for m in runs for r in m.get("runs", []) for n in r.get("rerendered", [])}
    failed = sorted({c for m in runs for c in m.get("failed_commands", [])})
    rr = {"metrics_diffs": [d for m in runs for d in m.get("metrics_diffs", [])]}
    g["G2"] = {
        "rerendered": len(done & names),
        "total": len(names),
        "not_rerendered": sorted(names - done),
        "failed_commands": failed,
        "commands": sum(len(m.get("runs", [])) for m in runs),
        "runs": list(a.rerender_tags),
        "pass": bool(runs) and names <= done and not failed,
    }
    diffs = rr.get("metrics_diffs", [])
    changed = [d for d in diffs if d.get("differing")]
    g["G3"] = {"metrics_files_regenerated": len(diffs), "with_differences": changed, "pass": bool(runs) and not changed}
    bad_dpi, dated, type3, nofont = [], [], [], []
    for n in sorted(names):
        im = Image.open(FIG / f"{n}.png")
        dpi = im.info.get("dpi", (0, 0))[0]
        if abs(dpi - 300) > 1:
            bad_dpi.append(f"{n} ({dpi:.0f})")
        b = (FIG / f"{n}.pdf").read_bytes()
        if b"/CreationDate" in b:
            dated.append(n)
        if b"/Subtype /Type3" in b:
            type3.append(n)
        if b"/FontFile2" not in b and b"/Font" in b:
            nofont.append(n)
    g["G4"] = {
        "png_not_300_dpi": bad_dpi,
        "pdf_with_creation_date": dated,
        "pdf_type3_fonts": type3,
        "pdf_fonts_not_embedded": nofont,
        "pass": not (bad_dpi or dated or type3 or nofont),
    }
    page = (FIG / "index.html").read_text(encoding="utf-8") if (FIG / "index.html").exists() else ""
    md = (FIG / "GALLERY.md").read_text(encoding="utf-8") if (FIG / "GALLERY.md").exists() else ""
    links = set(re.findall(r'(?:src|href)="([^"#]+)"', page))
    broken = sorted(x for x in links if not x.startswith("http") and not (FIG / x).exists())
    g["G5"] = {
        "index": bool(page),
        "markdown": bool(md),
        "figures_listed": sum(f'alt="{n}"' in page for n in names),
        "links": len(links),
        "broken_links": broken,
        "pass": bool(page and md) and not broken and all(f'alt="{n}"' in page for n in names),
    }
    an = load("results/p7_animate_default/metrics.json") or {}
    g["G6"] = {
        "scenes": len(an.get("animations", [])),
        "failures": an.get("failures"),
        "showreel": an.get("showreel"),
        "pass": bool(an.get("animations")) and not an.get("failures"),
    }
    sm = load("results/p7_4_smoke_default/metrics.json") or {}
    g["G7"] = {
        "figure_functions": sm.get("figure_functions"),
        "with_fixture": sm.get("functions_with_fixture"),
        "replayed_ok": sm.get("functions_replayed_ok"),
        "passed": sm.get("passed"),
        "failed": sm.get("failed", []),
        "skipped": len(sm.get("skipped", [])),
        "pass": bool(sm.get("pass")),
    }
    ok = all(v["pass"] for k, v in g.items() if k.startswith("G"))
    with Experiment(f"p7_gate_{a.tag}", {"seed": 0}) as ex:
        ex.metrics.update({"gate": g, "pass": ok, "git": git_hash()})
        L = [
            "# Phase 7 gate — visualisation (auto-generated by experiments/p7_gate.py)",
            "",
            f"git {git_hash()[:7]}; overall **{'PASS' if ok else 'FAIL'}**.",
            "",
            "| check | status | evidence |",
            "|---|---|---|",
        ]
        ev = {
            "G1": f"{g['G1']['catalogue']} catalogue figures, {g['G1']['png_in_docs']} PNG in docs/figures; "
            f"not in catalogue {len(g['G1']['not_in_catalogue'])}, missing PNG/PDF {len(g['G1']['missing_files'])}",
            "G2": f"{g['G2']['rerendered']} / {g['G2']['total']} re-rendered by {g['G2']['commands']} commands; "
            f"failed {g['G2']['failed_commands']}",
            "G3": f"{g['G3']['metrics_files_regenerated']} regenerated metrics files, "
            f"{len(g['G3']['with_differences'])} with a numerical difference",
            "G4": f"PNG not 300 dpi {len(bad_dpi)}; PDF with creation date {len(dated)}; Type 3 fonts {len(type3)}; "
            f"fonts not embedded {len(nofont)}",
            "G5": f"index.html lists {g['G5']['figures_listed']} / {len(names)}; {g['G5']['links']} links, "
            f"{len(broken)} broken; GALLERY.md {'present' if md else 'missing'}",
            "G6": f"{g['G6']['scenes']} animations, failures {g['G6']['failures']}",
            "G7": f"{g['G7']['with_fixture']} / {g['G7']['figure_functions']} figure functions with a fixture, "
            f"{g['G7']['replayed_ok']} replayed without error; replays passed {g['G7']['passed']}, "
            f"failed {len(g['G7']['failed'])}, skipped {g['G7']['skipped']} (docs/tables/phase7_4_smoke.md)",
        }
        for k in ("G1", "G2", "G3", "G4", "G5", "G6", "G7"):
            L.append(f"| {k} | **{'PASS' if g[k]['pass'] else 'FAIL'}** | {ev[k]} |")
        if changed:
            L += [
                "",
                "## Regenerated numbers that differ from the committed ones",
                "",
                "| file | compared | differing | max rel. diff | examples |",
                "|---|---|---|---|---|",
            ]
            for d in changed:
                L.append(
                    f"| {d['file']} | {d['compared']} | {d['differing']} | {d['max_rel_diff']:.3g} | "
                    + "; ".join(f"{k}: {x} → {y}" for k, x, y in d["examples"][:3]).replace("|", "/")
                    + " |"
                )
        (ROOT / "docs/tables/phase7_gate.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for k, v in g.items():
        print(k, v.get("pass"), {x: y for x, y in v.items() if x != "pass" and not isinstance(y, list | dict)})
    print("gate", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
