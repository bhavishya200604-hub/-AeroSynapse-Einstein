"""LinkedIn kit: a progress post built from the project's animations, with every number read from a saved metrics
file (never typed in). Writes, for experiments/p7_linkedin.py:

  teaser.mp4            square 1080 x 1080 video: cover card, one titled slide per highlight animation, closing card
  carousel.pdf          4:5 document post (1080 x 1350 pages): cover, one page per highlight, closing page
  carousel/page_XX.png  the same pages as images (an image post takes up to 20)
  cover.png             1200 x 627 image (link preview / single-image post)
  post.txt, post_short.txt   ready-to-copy post texts (templates in configs/viz/linkedin.yaml)
  facts.json            every number used, with the file and key it was read from
  index.html            one page with all of it: previews, copy buttons, sources, a posting checklist
"""

from __future__ import annotations

import html
import json
import math
import string
from pathlib import Path

import numpy as np

from aerosynapse.thermal.viz import style as S

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
# ---------------------------------------------------------------------------- facts (numbers with their source)
def facts() -> dict[str, dict]:
    """Every number a post may use: {name: {value, text, source}}. Missing results are skipped (a template that
    needs them then fails with the missing name, instead of printing an invented value)."""
    F: dict[str, dict] = {}

    def put(name: str, value, text: str, source: str) -> None:
        F[name] = {"value": value, "text": text, "source": source}

    def load(rel: str) -> dict | None:
        f = ROOT / rel
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    from aerosynapse.thermal.viz.anim import pipeline

    g = pipeline.counts()["_global"]
    src = "repository count (src/viz/anim/pipeline.py: counts)"
    put("experiments", g["experiments"], f"{g['experiments']}", src + ": results/p*/ folders")
    put("figures", g["figures"], f"{g['figures']}", src + ": docs/figures/*.png")
    put("tests", g["tests"], f"{g['tests']}", src + ": test functions in tests/")
    put("decisions", g["decisions"], f"{g['decisions']}", src + ": docs/DECISIONS.md")
    if g.get("verification"):
        v = g["verification"]
        rel = "results/p1_7_verification_smoke/metrics.json"
        put("ver_pass", v["pass"], f"{v['pass']}", rel + ": gate_pass")
        put("ver_total", v["total"], f"{v['total']}", rel + ": gate_total")

    m = load("results/p5_gate/metrics.json")
    if m:
        rel = "results/p5_gate/metrics.json"
        rows = {r["name"]: r for r in m["rows"]}
        ls = m["rows"][0]  # least squares (FDM), Gauss-Newton: the classical baseline of the gate table

        def pct(log_err: float) -> float:  # the gate stores log errors; the table prints exp(e) - 1
            return 100.0 * (math.exp(log_err) - 1.0)

        errs = {k: abs(pct(ls["err"][k])) for k in ("alpha", "k", "h")}
        for k, e in errs.items():
            put(f"ls_err_{k}", e, f"{e:.3g}", f"{rel}: rows[0] ({ls['name']}).err.{k}")
        put("ls_err_max", max(errs.values()), f"{max(errs.values()):.3g}", f"{rel}: rows[0].err (largest of α, k, h)")
        cov = m["calibration"]["ls_coverage95"]
        lo, hi = 100 * min(cov.values()), 100 * max(cov.values())
        put("ls_cov_min", lo, f"{lo:.0f}", f"{rel}: calibration.ls_coverage95")
        put("ls_cov_max", hi, f"{hi:.0f}", f"{rel}: calibration.ls_coverage95")
        best = rows.get("PINN joint_w100 (3 seeds)")
        if best:
            for k in ("k", "h", "Bi"):
                e = abs(pct(best["err"][k]))
                put(
                    f"pinn_{k.lower()}_err",
                    e,
                    f"{e:,.0f}" if e >= 100 else f"{e:.2g}",
                    f"{rel}: rows['{best['name']}'].err.{k}",
                )
    try:
        import yaml

        inv = yaml.safe_load((ROOT / "configs/inverse/defaults.yaml").read_text(encoding="utf-8"))
        lev = float(inv["data"]["level"])
        put("noise", 100 * lev, f"{100 * lev:g}", "configs/inverse/defaults.yaml: data.level (the Phase 5 gate data)")
    except (OSError, KeyError, TypeError):
        pass

    m = load("results/p6_7_cost_default/metrics.json")
    if m:
        rel = "results/p6_7_cost_default/metrics.json"
        for case, tag in (("D_graetz", "graetz"), ("B_slab", "slab")):
            a = m["analysis"].get(case)
            if not a:
                continue
            if a.get("break_even_ensemble"):
                be = a["break_even_ensemble"]
                put(f"{tag}_break_even", be, f"{be:,.0f}", f"{rel}: analysis.{case}.break_even_ensemble")
            put(
                f"{tag}_speedup",
                a["speedup_per_query"],
                f"{a['speedup_per_query']:.3g}",
                f"{rel}: analysis.{case}.speedup_per_query",
            )
            put(
                f"{tag}_pinn_err",
                100 * a["pinn_error"],
                f"{100 * a['pinn_error']:.2g}",
                f"{rel}: analysis.{case}.pinn_error",
            )

    m = load("results/p6_3_extrapolation_default/metrics.json")
    if m and "B_slab" in m["parametric"]:
        rel = "results/p6_3_extrapolation_default/metrics.json"
        c = m["parametric"]["B_slab"]
        x, e = np.array(c["values"]["value"]), 100 * np.array(c["values"]["rel_l2"])
        lo, hi = c["range"]
        inside = (x >= lo) & (x <= hi)
        put(
            "slab_in_max",
            float(e[inside].max()),
            f"{e[inside].max():.2g}",
            f"{rel}: parametric.B_slab rel_l2, max over Bi in the training range [{lo:g}, {hi:g}]",
        )
        put(
            "slab_out_max",
            float(e[~inside].max()),
            f"{e[~inside].max():.2g}",
            f"{rel}: parametric.B_slab rel_l2, max outside the training range",
        )
    return F


def fill(template: str, F: dict[str, dict], extra: dict[str, str]) -> tuple[str, list[str]]:
    """Fill {name} fields from facts (or extra plain strings such as the link); returns the text and the names used.
    An unknown name raises KeyError: a post never contains a number that does not come from a results file."""
    used = [f for _, f, _, _ in string.Formatter().parse(template) if f]
    vals = {}
    for n in used:
        if n in extra:
            vals[n] = extra[n]
        elif n in F:
            vals[n] = F[n]["text"]
        else:
            raise KeyError(f"template field {{{n}}} is neither a fact {sorted(F)} nor one of {sorted(extra)}")
    return template.format(**vals), [n for n in used if n in F]


# ---------------------------------------------------------------------------- images
def key_frame(anim: Path, fraction: float):
    """One frame of an animation (WebP / GIF / APNG) at `fraction` of its length, as an RGB image."""
    from PIL import Image

    im = Image.open(anim)
    n = getattr(im, "n_frames", 1)
    im.seek(min(n - 1, max(0, int(round(fraction * (n - 1))))))
    return im.convert("RGB")


ACCENT = "#0072B2"  # Okabe-Ito blue: the project's solver colour


def _canvas(size: tuple[int, int], dpi: int):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    S.apply_style("report")
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    fig.set_layout_engine("none")
    fig.patch.set_facecolor("white")
    fig.add_artist(Rectangle((0, 1 - 14 / size[1]), 1, 14 / size[1], color=ACCENT, transform=fig.transFigure, lw=0))
    return fig, fig.canvas.get_renderer()


def _text(fig, r, x_px: float, y_px: float, text: str, size: float, max_w: float, **kw):
    """Text wrapped to max_w pixels (measured, not estimated), top-left at (x_px, y_px); returns its height in px."""
    W, H = fig.bbox.width, fig.bbox.height
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            trial = f"{cur} {word}".strip()
            t = fig.text(0, 0, trial, fontsize=size, **kw)
            wide = t.get_window_extent(r).width > max_w
            t.remove()
            if wide and cur:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
    t = fig.text(x_px / W, y_px / H, "\n".join(lines), fontsize=size, va="top", ha="left", **kw)
    return t.get_window_extent(r).height


def _image(fig, im, x_px: float, top_px: float, max_w: float, max_h: float) -> float:
    """Picture fitted into max_w x max_h pixels with its top at top_px, centred horizontally in the box; returns its
    height in px."""
    W, H = fig.bbox.width, fig.bbox.height
    sc = min(max_w / im.size[0], max_h / im.size[1])
    w, h = im.size[0] * sc, im.size[1] * sc
    ax = fig.add_axes([(x_px + (max_w - w) / 2) / W, (top_px - h) / H, w / W, h / H])
    ax.imshow(np.asarray(im), interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(S.AXIS)
        sp.set_linewidth(0.8)
    return h


def _save(fig, out: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": "standard"}):  # the project style crops to 'tight'; a card keeps its size
        fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out


def card(
    out: Path,
    size: tuple[int, int],
    title: str,
    subtitle: str = "",
    body: list[str] | None = None,
    images: list | None = None,
    footer: str = "",
    kicker: str = "",
    page_no: str = "",
    dpi: int = 100,
) -> Path:
    """A portrait or square card (carousel page, video cover / closing): kicker, headline, finding, bullet lines and
    pictures, laid out top-down with every text block measured; footer with page number and source."""
    fig, r = _canvas(size, dpi)
    W, H = size
    u = W / 1080
    m = 72 * u
    y = H - 64 * u
    if kicker:
        y -= _text(fig, r, m, y, kicker.upper(), 15 * u, W - 2 * m, color=ACCENT, weight="bold") + 22 * u
    y -= _text(fig, r, m, y, title, 40 * u, W - 2 * m, color=S.INK, weight="bold", linespacing=1.12) + 24 * u
    if subtitle:
        y -= _text(fig, r, m, y, subtitle, 23 * u, W - 2 * m, color=S.INK2, linespacing=1.3) + 34 * u
    for line in body or []:
        fig.text((m + 4 * u) / W, (y - 15 * u) / H, "●", fontsize=13 * u, color=ACCENT, va="center")
        y -= _text(fig, r, m + 34 * u, y, line, 22 * u, W - 2 * m - 34 * u, color=S.INK, linespacing=1.3) + 22 * u
    foot = 96 * u
    if images:
        y -= 6 * u
        room = y - foot - 20 * u
        each = (room - 18 * u * (len(images) - 1)) / len(images)
        # fitted heights first, then the block is centred vertically in the free space
        hs = [min((W - 2 * m) / im.size[0], each / im.size[1]) * im.size[1] for im in images]
        y -= max(0.0, (room - sum(hs) - 18 * u * (len(images) - 1)) / 2)
        for im in images:
            y -= _image(fig, im, m, y, W - 2 * m, each) + 18 * u
    from matplotlib.lines import Line2D

    fig.add_artist(Line2D([m / W, 1 - m / W], [foot / H, foot / H], color=S.AXIS, lw=0.8, transform=fig.transFigure))
    if page_no:
        fig.text(m / W, (foot - 22 * u) / H, page_no, fontsize=14 * u, color=S.MUTED, va="top", weight="bold")
    if footer:
        fig.text(1 - m / W, (foot - 22 * u) / H, footer, fontsize=13 * u, color=S.MUTED, va="top", ha="right")
    return _save(fig, out, dpi)


def cover(
    out: Path,
    size: tuple[int, int],
    title: str,
    subtitle: str,
    images: list,
    footer: str = "",
    kicker: str = "",
    dpi: int = 100,
) -> Path:
    """Landscape cover (1200 x 627, link preview): text on the left, up to three pictures stacked on the right."""
    fig, r = _canvas(size, dpi)
    W, H = size
    u = H / 627
    m = 52 * u
    col = W * 0.47
    y = H - 56 * u
    if kicker:
        y -= _text(fig, r, m, y, kicker.upper(), 13 * u, col - m, color=ACCENT, weight="bold") + 18 * u
    y -= _text(fig, r, m, y, title, 34 * u, col - m, color=S.INK, weight="bold", linespacing=1.12) + 20 * u
    _text(fig, r, m, y, subtitle, 18 * u, col - m, color=S.INK2, linespacing=1.3)
    if footer:
        fig.text(m / W, 40 * u / H, footer, fontsize=11.5 * u, color=S.MUTED, va="bottom")
    x0, top, room = col + 24 * u, H - 40 * u, H - 80 * u
    each = (room - 12 * u * (len(images) - 1)) / max(len(images), 1)
    for im in images:
        top -= _image(fig, im, x0, top, W - x0 - m, each) + 12 * u
    return _save(fig, out, dpi)


# ---------------------------------------------------------------------------- page
# the page template (str.format fields; literal braces doubled)
PAGE = Path(__file__).with_name("linkedin_page.html")


def page(
    out_dir: Path,
    post: str,
    post_short: str,
    F: dict,
    used: list[str],
    n_pages: int,
    video_s: float | None,
    title: str,
    git: str,
) -> Path:
    e = html.escape
    rows = "".join(
        f"<tr><td><code>{e(n)}</code></td><td>{e(F[n]['text'])}</td><td><code>{e(F[n]['source'])}</code></td></tr>"
        for n in used
    )
    pages = "".join(
        f'<a href="carousel/page_{k:02d}.png"><img src="carousel/page_{k:02d}.png" alt="page {k}" loading="lazy"></a>'
        for k in range(1, n_pages + 1)
    )
    video = (
        '<video src="teaser.mp4" controls muted loop playsinline poster="carousel/page_01.png"></video>'
        if video_s
        else '<p class="muted">No video: it needs Microsoft PowerPoint (Windows). The carousel works everywhere.</p>'
    )
    intro = (
        "A ready-made progress post about the project: text to paste, a square video and a document carousel made "
        "from the project's animations. Physics comes from the verified solvers and analytic solutions, learning "
        "curves are replayed from saved logs, and no network was retrained to make any of it."
    )
    out = out_dir / "index.html"
    out.write_text(
        PAGE.read_text(encoding="utf-8").format(
            intro=e(intro),
            post=e(post),
            post_short=e(post_short),
            n_long=len(post),
            n_short=len(post_short),
            video=video,
            video_s=f"{video_s:.0f}" if video_s else "-",
            n_pages=n_pages,
            pages=pages,
            rows=rows,
            title=e(title),
            git=e(git),
        ),
        encoding="utf-8",
    )
    return out
