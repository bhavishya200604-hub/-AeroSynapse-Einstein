"""Pack a paper directory into an Overleaf-ready zip (paper/dist/<name>_overleaf.zip).

    python -m aerosynapse.thermal.paper_tools.pack_overleaf paper/paper2

The zip holds main.tex, numbers.tex, refs.bib, sections/, tables/ and the PDF
figures (PNG previews and the generator scripts stay in the repository).
Upload it with Overleaf's "New project > Upload project"; the main document
is main.tex (pdfLaTeX + BibTeX).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
def pack(paper: Path) -> Path:
    out = ROOT / "paper" / "dist" / f"{paper.name}_overleaf.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [paper / "main.tex", paper / "numbers.tex", paper / "refs.bib"]
    files += sorted((paper / "sections").glob("*.tex")) + sorted((paper / "tables").glob("*.tex"))
    files += sorted((paper / "figures").glob("*.pdf"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(paper).as_posix())
    print(f"packed {len(files)} files -> {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KiB)")
    return out


if __name__ == "__main__":
    pack((ROOT / (sys.argv[1] if len(sys.argv) > 1 else "paper/paper2")).resolve())
