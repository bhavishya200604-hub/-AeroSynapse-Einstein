"""The project report as PDF, Word (.docx) and plain text, built from one structured document.

    python -m aerosynapse.cli workflow report falcon --format all      # report.pdf, report.docx, report.txt

The document is assembled from every stage's ``result.json`` and figures, so the three formats (and the HTML
report) always say the same thing. PDF pages are drawn with matplotlib (A4, title page with the logo band, text,
tables and figures); the Word file uses python-docx with real headings, tables and pictures.
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aerosynapse.workflow.project import KEYS, STAGES, Project

ROOT = Path(__file__).resolve().parents[3]
LOGO = ROOT / "docs" / "assets" / "aerosynapse_logo.png"


@dataclass
class Section:
    title: str
    paragraphs: list[str] = field(default_factory=list)
    tables: list[tuple[list[str], list[list[str]]]] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)


def _f(v: Any, spec: str = ".4g") -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return format(v, spec) if v == v else "n/a"
    return "n/a" if v is None else str(v)


def build(p: Project) -> list[Section]:
    r = {k: p.result(k) for k in KEYS}
    sd = {k: p.stage_dir(k) for k in KEYS}
    out: list[Section] = []
    s = Section("Project")
    s.paragraphs.append(f"Project {p.name}, created {p.meta.get('created', '')}, origin: {p.meta.get('origin', '')}. Design revisions: {len(p.meta['revisions'])}.")
    req = p.meta["requirements"]
    s.tables.append((["requirement", "value"], [[k, ", ".join(f"{a} {_f(b, '.6g')}" for a, b in v.items()) if isinstance(v, dict) else _f(v, ".6g")] for k, v in req.items()]))
    s.tables.append((["#", "stage", "status", "seconds"], [[str(i + 1), t, p.status(k), _f(p.meta["stages"][k].get("seconds"), ".0f")] for i, (k, t) in enumerate(STAGES)]))
    out.append(s)
    cards = [(k, c) for k, c in (("concept", r["concept"].get("scorecard")), ("design", r["design"].get("scorecard_after")), ("wing", r["wing"].get("scorecard_after")),
                                 ("body", r["body"].get("scorecard_after"))) if c]
    if cards:
        s = Section("Scorecard from stage to stage", ["The same quick analysis (geometry, vortex lattice, friction, Mach-plane area rule) of each stage's design."])
        s.tables.append((["quantity"] + [k for k, _ in cards], [[m] + [_f(c.get(m)) for _, c in cards] for m in cards[0][1]]))
        out.append(s)
    c = r["concept"]
    if c:
        s = Section("1 Basic design", [f"Origin: {c.get('origin', '')}; {len(c.get('features', []))} features."])
        if c.get("performance"):
            pf = c["performance"]
            s.paragraphs.append(f"Full performance model: take-off mass {pf['W0_kg']:.0f} kg, empty mass {pf['empty_kg']:.0f} kg, feasible: {_f(pf['feasible'])}.")
        out.append(s)
    d = r["design"]
    if d:
        s = Section("2 Design optimisation", [d.get("method", ""), f"Objective {_f(d.get('J_before'))} -> {_f(d.get('J_after'))} after {d.get('evaluations')} evaluations."])
        if "performance_after" in d:
            a, b = d["performance_before"], d["performance_after"]
            rows = [["take-off mass, kg", _f(a["W0_kg"], ".0f"), _f(b["W0_kg"], ".0f")]] + [[k, _f(a["metrics"].get(k)), _f(b["metrics"].get(k))] for k in b["metrics"]]
            s.tables.append((["quantity", "before", "after"], rows))
        s.images.append(sd["design"] / "convergence.png")
        out.append(s)
    w = r["wing"]
    if w:
        s = Section("3 Wing optimisation", [w["method"], f"Drag index {_f(w.get('drag_index_before'))} -> {_f(w.get('drag_index_after'))}; supersonic wave drag of the section x{_f(w['supersonic_wave_ratio'], '.3f')}; "
                                                         f"thickness {_f(w['thickness_before'], '.4f')} -> {_f(w['thickness_after'], '.4f')}."])
        s.tables.append((["state", "normal Mach", "normal cl", "cd before (counts)", "cd after (counts)"],
                         [[x["state"], _f(x["mach_normal"], ".3f"), _f(x["cl_normal"], ".3f"), _f(1e4 * x["cd_before"], ".1f"), _f(1e4 * x["cd_after"], ".1f")] for x in w["states"]]))
        xf = w.get("xfoil", {})
        if xf and "error" not in xf:
            s.tables.append((["XFOIL check", "Mach", "cl", "cd before", "cd after"], [[k, _f(v["mach"], ".2f"), _f(v["cl"], ".2f"), _f(v["before"]["cd"], ".5f"), _f(v["after"]["cd"], ".5f")] for k, v in xf.items()]))
        s.images.append(sd["wing"] / "section.png")
        out.append(s)
    b = r["body"]
    if b and "before" in b:
        s = Section("4 Body optimisation", [b["method"]])
        s.tables.append((["quantity", "before", "after"], [["wave D/q M1.2, m2", _f(b["before"]["w12"]), _f(b["after"]["w12"])], ["wave D/q M1.6, m2", _f(b["before"]["w16"]), _f(b["after"]["w16"])],
                                                            ["volume, m3", _f(b["before"]["volume"]), _f(b["after"]["volume"])], ["wetted area, m2", _f(b["before"]["wetted"]), _f(b["after"]["wetted"])]]))
        s.images.append(sd["body"] / "area_rule.png")
        out.append(s)
    st = r["structure"]
    if st:
        wg = st["wing"]
        s = Section("5 Structure and design codes", [f"Code: {st['code']}; limit n {wg['n_limit']}, ultimate {wg['n_ultimate']}; take-off mass {st['W0_kg']:.0f} kg. Wing box in {wg['material']} at {wg['temperature_K']:.0f} K "
                                                     f"(Ashby screening suggested {st.get('screening_choice', '-')}): box mass {wg['box_mass_both_sides_kg']:.0f} kg, root bending moment {wg['root_bending_moment_kNm']:.0f} kNm per side, "
                                                     f"tip deflection {wg['tip_deflection_pct_semispan']:.1f} % of the semispan, modes {wg['first_bending_hz']:.1f} Hz bending and {wg['first_torsion_hz']:.1f} Hz torsion, "
                                                     f"flutter {wg['flutter']['V_flutter_eas_ms']:.0f} m/s EAS against {wg['flutter']['required_ms']:.0f} required."])
        s.tables.append((["check", "minimum margin of safety"], [[k, _f(v, "+.3f")] for k, v in wg["margins_min"].items()]))
        s.tables.append((["material", "box kg", "tip deflection %", "note"], [[t["material"], _f(t.get("box_mass_kg"), ".0f"), _f(t.get("tip_deflection_pct"), ".1f"), t.get("note", "")] for t in st["material_trade"]]))
        s.images.append(sd["structure"] / "structure.png")
        out.append(s)
    t = r["thermal"]
    if t:
        s = Section("6 Skin heating with the parametric PINN", [f"Network used for the design: {Path(t['network_used']).parent.name} (max error {t['network_max_error']:.2e} against the exact series)."])
        s.tables.append((["Adam epochs", "max error", "relative L2", "seconds"], [[str(x["adam_epochs"]), _f(x["max_abs"], ".3e"), _f(x["rel_l2"], ".3e"), _f(x["wall_s"], ".0f")] for x in t["convergence"]["levels"]]))
        fit = t["convergence"].get("fit", {})
        if fit:
            s.paragraphs.append(f"Convergence fit ({fit.get('best', '')}): error ~ epochs^{_f(fit.get('exponent'), '.3f')}.")
        s.tables.append((["zone", "material", "mm", "kg/m2", "T max K", "stress ratio"],
                         [[z, v["optimum"]["material"], _f(v["optimum"]["thickness_mm"], ".2f"), _f(v["optimum"]["areal_kg_m2"], ".2f"), _f(v["optimum"]["T_max_K"], ".0f"), _f(v["optimum"]["stress_ratio"], ".3f")]
                          for z, v in t["design"]["zones"].items() if v["optimum"]]))
        s.images.append(sd["thermal"] / "thermal.png")
        out.append(s)
    m = r["simulation"]
    if m:
        rc = m["richardson"]
        s = Section("7 3D flow simulation", [f"CL {_f(rc['CL']['extrapolated'])} +/- {_f(100 * (rc['CL']['gci_fine'] or 0), '.1f')} % ({rc['CL'].get('kind', '')}); "
                                             f"CD {_f(rc['CD']['extrapolated'], '.5f')} +/- {_f(100 * (rc['CD']['gci_fine'] or 0), '.1f')} % ({rc['CD'].get('kind', '')})."])
        s.tables.append((["h, m", "cells", "CL", "CD", "seconds"], [[_f(g["h"]), str(g["cells"]), _f(g["CL"], ".4f"), _f(g["CD"], ".5f"), _f(g["seconds"], ".0f")] for g in m["grid_study"]]))
        s.tables.append((["state", "Mach", "alpha", "CL", "CD"], [[k, _f(v["mach"]), _f(v["alpha"]), _f(v["CL"], ".4f"), _f(v["CD"], ".5f")] for k, v in m["states"].items()]))
        s.images += [sd["simulation"] / "grid_convergence.png", sd["simulation"] / "field.png"]
        out.append(s)
    e = r["export"]
    if e:
        s = Section("8 Export", [f"Drawing scale {e.get('drawing_scale', '')}; {e.get('triangles', 0)} surface triangles; STEP with {e.get('step_faces', 0)} faces."])
        s.tables.append((["file", "bytes", "SHA-256 (start)"], [[k, f"{v['bytes']:,}", v["sha256"][:16]] for k, v in e.get("files", {}).items()]))
        s.images.append(sd["export"] / f"{p.name}_drawing.png")
        out.append(s)
    s = Section("Revision history")
    s.tables.append((["rev", "stage", "note", "time"], [[str(x["rev"]), x["stage"], x["note"], x["time"]] for x in p.meta["revisions"]]))
    s.paragraphs.append("Research use only: preliminary-design methods with their stated assumptions; not certification analysis.")
    out.append(s)
    for sec in out:
        sec.images = [i for i in sec.images if i.exists()]
    return out


# ------------------------------------------------------------------ plain text
def to_txt(p: Project, secs: list[Section]) -> str:
    lines = [f"{p.name.upper()}: DESIGN WORKFLOW REPORT", f"AeroSynapse Studio, generated {time.strftime('%Y-%m-%d %H:%M')}", "=" * 78, ""]
    for s in secs:
        lines += [s.title.upper(), "-" * len(s.title)]
        for para in s.paragraphs:
            lines += textwrap.wrap(para, 78) + [""]
        for head, rows in s.tables:
            cols = [head] + rows
            wd = [min(40, max(len(str(r[i])) for r in cols)) for i in range(len(head))]
            fmt = "  ".join(f"{{:<{w}}}" for w in wd)
            lines.append(fmt.format(*[str(h)[:40] for h in head]))
            lines.append("  ".join("-" * w for w in wd))
            lines += [fmt.format(*[str(c)[:40] for c in r]) for r in rows]
            lines.append("")
        for im in s.images:
            lines.append(f"[figure: {im.name}]")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ Word
def to_docx(p: Project, secs: list[Section], path: Path) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Cm(2.0)
    st = doc.styles["Normal"]
    st.font.name = "Segoe UI"
    st.font.size = Pt(10)
    if LOGO.exists():
        doc.add_picture(str(LOGO), width=Cm(3))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    t = doc.add_heading(f"{p.name}: design workflow report", 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(f"AeroSynapse Studio  |  generated {time.strftime('%Y-%m-%d %H:%M')}  |  {p.meta.get('origin', '')}")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for s in secs:
        h = doc.add_heading(s.title, 1)
        for run in h.runs:
            run.font.color.rgb = RGBColor(0x16, 0x32, 0x5C)
        for para in s.paragraphs:
            doc.add_paragraph(para)
        for head, rows in s.tables:
            tb = doc.add_table(rows=1, cols=len(head))
            tb.style = "Light Grid Accent 1"
            for i, hd in enumerate(head):
                tb.rows[0].cells[i].text = str(hd)
            for r in rows:
                cells = tb.add_row().cells
                for i, v in enumerate(r):
                    cells[i].text = str(v)
            for row in tb.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            run.font.size = Pt(8.5)
            doc.add_paragraph()
        for im in s.images:
            doc.add_picture(str(im), width=Cm(16.5))
            cap = doc.add_paragraph(im.stem.replace("_", " "))
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].italic = True
    # page numbers in the footer
    fp = doc.sections[0].footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.text = text
        run._r.append(el)
    doc.save(str(path))


# ------------------------------------------------------------------ PDF
def to_pdf(p: Project, secs: list[Section], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    W, H = 8.27, 11.69  # A4, inches
    NAVY, INK = "#16325c", "#1a2a44"
    page = {"fig": None, "y": 0.0, "n": 0}
    pdf = PdfPages(str(path))

    def new_page() -> None:
        if page["fig"] is not None:
            pdf.savefig(page["fig"])
            plt.close(page["fig"])
        fig = plt.figure(figsize=(W, H), layout="none")
        page["n"] += 1
        fig.patches.append(Rectangle((0, 0.965), 1, 0.035, transform=fig.transFigure, color=NAVY, zorder=0))
        fig.text(0.06, 0.976, f"AeroSynapse Studio  |  {p.name} design workflow report", color="white", fontsize=8.5, va="center")
        fig.text(0.94, 0.976, f"page {page['n']}", color="white", fontsize=8.5, va="center", ha="right")
        page["fig"], page["y"] = fig, 0.93

    def need(h: float) -> None:
        if page["y"] - h < 0.05:
            new_page()

    def text(s: str, size: float = 9.5, color: str = INK, weight: str = "normal", width: int = 105) -> None:
        for ln in textwrap.wrap(s, width) or [""]:
            need(0.018)
            page["fig"].text(0.07, page["y"], ln, fontsize=size, color=color, weight=weight, va="top")
            page["y"] -= 0.0165 * size / 9.5

    # title page
    fig = plt.figure(figsize=(W, H), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    import numpy as np

    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(np.vstack([grad] * 2), extent=(0, 1, 0.55, 1), aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list("g", ["#0b1a33", "#16325c", "#0096d6"]))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if LOGO.exists():
        ax.imshow(mpimg.imread(str(LOGO)), extent=(0.42, 0.58, 0.80, 0.80 + 0.16 * W / H), zorder=3)
    ax.text(0.5, 0.72, f"{p.name.upper()}", color="white", fontsize=30, ha="center", weight="bold")
    ax.text(0.5, 0.67, "Design workflow report", color="#cfe6ff", fontsize=16, ha="center")
    ax.text(0.5, 0.60, f"AeroSynapse Studio  |  {time.strftime('%d %B %Y')}", color="#cfe6ff", fontsize=10, ha="center")
    yy = 0.48
    for i, (k, t) in enumerate(STAGES):
        ax.add_patch(Rectangle((0.2, yy - 0.012), 0.6, 0.03, color="#eef3f9" if i % 2 else "#e2ebf5", zorder=1))
        ax.text(0.22, yy, f"{i + 1}  {t}", fontsize=10, color=INK, va="center", zorder=2)
        ax.text(0.78, yy, p.status(k), fontsize=10, color="#137333" if p.status(k) == "done" else "#a15c00", va="center", ha="right", zorder=2)
        yy -= 0.034
    ax.text(0.5, 0.06, "Research use only: preliminary-design methods with stated assumptions, not certification analysis.", fontsize=8, color="#6b7a90", ha="center")
    pdf.savefig(fig)
    plt.close(fig)
    page["n"] = 1
    new_page()
    for s in secs:
        need(0.06)
        page["y"] -= 0.01
        page["fig"].patches.append(Rectangle((0.06, page["y"] - 0.022), 0.88, 0.028, transform=page["fig"].transFigure, color="#e8f2fb", zorder=0))
        page["fig"].text(0.07, page["y"], s.title, fontsize=13, color=NAVY, weight="bold", va="top")
        page["y"] -= 0.04
        for para in s.paragraphs:
            text(para)
            page["y"] -= 0.006
        for head, rows in s.tables:
            n = len(rows) + 1
            h = 0.0175 * n
            need(min(h, 0.85) + 0.02)
            if h > 0.85:  # long tables continue on following pages
                rows_left = rows
                while rows_left:
                    k = max(1, int((page["y"] - 0.07) / 0.0175) - 1)
                    chunk, rows_left = rows_left[:k], rows_left[k:]
                    _table(page, head, chunk)
                    if rows_left:
                        new_page()
            else:
                _table(page, head, rows)
        for im in s.images:
            img = mpimg.imread(str(im))
            ih, iw = img.shape[:2]
            wfrac = 0.86
            hfrac = wfrac * W / H * ih / iw
            if hfrac > 0.6:
                wfrac *= 0.6 / hfrac
                hfrac = 0.6
            need(hfrac + 0.03)
            ax = page["fig"].add_axes((0.5 - wfrac / 2, page["y"] - hfrac, wfrac, hfrac))
            ax.imshow(img)
            ax.axis("off")
            page["y"] -= hfrac + 0.012
            page["fig"].text(0.5, page["y"], im.stem.replace("_", " "), fontsize=8, color="#6b7a90", ha="center", va="top", style="italic")
            page["y"] -= 0.025
    pdf.savefig(page["fig"])
    plt.close(page["fig"])
    pdf.close()


def _table(page: dict[str, Any], head: list[str], rows: list[list[str]]) -> None:
    n = len(rows) + 1
    h = 0.0175 * n
    ax = page["fig"].add_axes((0.07, page["y"] - h, 0.86, h))
    ax.axis("off")
    cells = [[str(c)[:48] for c in r] for r in rows]
    tb = ax.table(cellText=cells, colLabels=[str(x) for x in head], loc="upper left", cellLoc="left", colLoc="left", bbox=(0, 0, 1, 1))
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.5)
    for (r, _c), cell in tb.get_celld().items():
        cell.set_edgecolor("#c9d6e6")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#16325c")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f7fb")
    page["y"] -= h + 0.02


def write_all(p: Project, formats: tuple[str, ...] = ("pdf", "docx", "txt")) -> dict[str, Path]:
    secs = build(p)
    out = p.stage_dir("export")
    done: dict[str, Path] = {}
    if "txt" in formats:
        done["txt"] = out / "report.txt"
        done["txt"].write_text(to_txt(p, secs), encoding="utf-8")
    if "docx" in formats:
        done["docx"] = out / "report.docx"
        to_docx(p, secs, done["docx"])
    if "pdf" in formats:
        done["pdf"] = out / "report.pdf"
        to_pdf(p, secs, done["pdf"])
    return done
