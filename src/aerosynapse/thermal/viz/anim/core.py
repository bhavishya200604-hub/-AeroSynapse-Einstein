"""Animation engine: a Scene draws frame i of an animation; render() writes its frames once to disk (low memory) and
then every requested format from those frames.

Formats without extra software: gif, webp, apng (Pillow), html (self-contained player with play / pause / speed /
scrub, frames embedded), frames (PNG sequence). With Microsoft PowerPoint on Windows: mp4 and pptx
(src/viz/anim/convert.py). Data that a scene computes (solver runs) is cached as .npz on disk, keyed by the scene's
parameters, so a re-render or a new format costs no computation.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerosynapse.thermal.viz import style as S

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
FORMATS = ("gif", "webp", "apng", "html", "frames", "mp4", "pptx")


@dataclass
class Scene:
    """name: file stem; title / caption: shown in the gallery; n_frames; fps; make(): returns (figure, update) where
    update(i) redraws the figure for frame i; source: where the data comes from (shown in the gallery)."""

    name: str
    title: str
    caption: str
    n_frames: int
    fps: int
    make: Callable[[], tuple]
    source: str = ""
    category: str = "physics"
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------- data cache (HDD, not RAM)
def cache_key(name: str, params: dict) -> str:
    return hashlib.sha256(json.dumps({"name": name, **params}, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _code_version(fn: Callable) -> str:
    """Hash of the code that computes the data: the source of `fn` and, recursively, of every function of the
    animation package it calls by name (e.g. a scene's lambda -> compute_obstacle -> _run_ns -> _advect). Editing the
    computation invalidates its cache; editing captions or drawing code does not."""
    import functools
    import inspect
    import types

    seen: set = set()
    parts: list[str] = []

    def names(code: types.CodeType) -> set[str]:
        out = set(code.co_names)
        for c in code.co_consts:  # nested functions and lambdas
            if isinstance(c, types.CodeType):
                out |= names(c)
        return out

    def visit(f) -> None:
        if isinstance(f, functools.partial):
            f = f.func
        if not inspect.isfunction(f) or f in seen:
            return
        seen.add(f)
        try:
            parts.append(inspect.getsource(f))
        except (OSError, TypeError):
            parts.append(f.__qualname__)
        used = sorted(names(f.__code__))
        # names resolved in the function's module, or as attributes of an animation module it uses (physics.f)
        spaces = [f.__globals__] + [
            vars(m)
            for m in (f.__globals__.get(n) for n in used)
            if isinstance(m, types.ModuleType) and m.__name__.startswith("aerosynapse.thermal.viz.anim")
        ]
        for n in used:
            for sp in spaces:
                g = sp.get(n)
                if inspect.isfunction(g) and (g.__module__ or "").startswith("aerosynapse.thermal.viz.anim"):
                    visit(g)

    visit(fn)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12] if parts else ""


def cached(cache_dir: Path, name: str, params: dict, compute: Callable[[], dict]) -> tuple[dict, bool]:
    """Arrays of a computation, stored as <cache_dir>/<name>_<key>.npz; returns (data, hit). The key covers the
    parameters and the source of the module that computes them, so neither a new value nor a code change can return
    stale data."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{name}_{cache_key(name, {**params, '_code': _code_version(compute)})}.npz"
    if f.exists():
        with np.load(f, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}, True
    data = compute()
    np.savez_compressed(f, **{k: np.asarray(v) for k, v in data.items()})
    return data, False


# ---------------------------------------------------------------------------- rendering
def render_frames(scene: Scene, frame_dir: Path, dpi: int, quick: int | None = None) -> list[Path]:
    """Draw every frame (or `quick` evenly spaced frames) to PNG files; the figure is reused, so memory stays flat."""
    import matplotlib.pyplot as plt

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    fig, update = scene.make()
    idx = list(range(scene.n_frames))
    if quick:
        idx = sorted({int(round(v)) for v in np.linspace(0, scene.n_frames - 1, quick)})
    paths = []
    # lay the figure out once (on the first frame), then freeze it: recomputing the constrained layout on every frame
    # dominated the render time, and a fixed layout also stops panels from shifting while text changes length
    update(idx[0])
    fig.canvas.draw()
    fig.set_layout_engine("none")
    # every frame at the figure's own size: the project style saves with a 'tight' box, which follows the text and
    # changes the image size from frame to frame (video and WebP writers need equal frames)
    with plt.rc_context({"savefig.bbox": "standard"}):
        for k, i in enumerate(idx):
            update(i)
            p = frame_dir / f"f{k:05d}.png"
            fig.savefig(p, dpi=dpi, facecolor=fig.get_facecolor())
            paths.append(p)
    plt.close(fig)
    return paths


def _even(im):
    """Even width / height (video encoders need it)."""
    w, h = im.size
    return im if (w % 2 == 0 and h % 2 == 0) else im.crop((0, 0, w - w % 2, h - h % 2))


def write_gif(frames: list[Path], out: Path, fps: int) -> Path:
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in frames]
    pal = ims[len(ims) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=False)
    return out


def write_webp(frames: list[Path], out: Path, fps: int) -> Path:
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in frames]
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, quality=88, method=4)
    return out


def write_apng(frames: list[Path], out: Path, fps: int) -> Path:
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in frames]
    ims[0].save(out, format="PNG", save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
    return out


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --ink:#111; --muted:#5a5a5a; --line:#ddd; --bg:#fff; --accent:#0072B2; }}
@media (prefers-color-scheme: dark) {{ :root {{ --ink:#eee; --muted:#aaa; --line:#444; --bg:#161616; }} }}
body {{ margin:0; padding:16px; font-family: Georgia, serif; color:var(--ink); background:var(--bg); }}
main {{ max-width:1100px; margin:0 auto; }}
h1 {{ font-size:1.25rem; margin:0 0 .3rem; }} p {{ color:var(--muted); margin:.2rem 0 .8rem; line-height:1.45; }}
img {{ width:100%; height:auto; background:#fff; border:1px solid var(--line); border-radius:6px; }}
.bar {{ display:flex; gap:.6rem; align-items:center; flex-wrap:wrap; margin-top:.6rem; }}
button {{ font:inherit; padding:.35rem .9rem; border:1px solid var(--line); border-radius:6px; background:transparent;
  color:var(--ink); cursor:pointer; }}
input[type=range] {{ flex:1; min-width:160px; accent-color:var(--accent); }}
.small {{ font-size:.85rem; color:var(--muted); }}
</style></head><body><main>
<h1>{title}</h1><p>{caption}</p>
<img id="f" alt="{title}" src="">
<div class="bar"><button id="play">Pause</button><input id="s" type="range" min="0" max="{last}" value="0">
<span class="small">speed</span><select id="v"><option>0.25</option><option>0.5</option><option selected>1</option>
<option>2</option><option>4</option></select><span class="small" id="lab"></span></div>
<p class="small">Source: {source}. Keys: space = play / pause, arrows = step.</p>
</main><script>
const F = {frames}; const fps = {fps};
const img = document.getElementById('f'), s = document.getElementById('s'), lab = document.getElementById('lab');
const play = document.getElementById('play'), sp = document.getElementById('v');
let i = 0, on = true, last = 0;
function show(k) {{ i = (k + F.length) % F.length; img.src = F[i]; s.value = i;
  lab.textContent = (i + 1) + ' / ' + F.length; }}
function tick(t) {{ if (on && t - last > 1000 / (fps * parseFloat(sp.value))) {{ last = t; show(i + 1); }}
  requestAnimationFrame(tick); }}
play.onclick = () => {{ on = !on; play.textContent = on ? 'Pause' : 'Play'; }};
s.oninput = () => {{ on = false; play.textContent = 'Play'; show(parseInt(s.value)); }};
document.onkeydown = e => {{ if (e.code === 'Space') {{ e.preventDefault(); play.onclick(); }}
  if (e.key === 'ArrowRight') {{ on = false; show(i + 1); }}
  if (e.key === 'ArrowLeft') {{ on = false; show(i - 1); }} }};
show(0); requestAnimationFrame(tick);
</script></body></html>
"""


def write_html(frames: list[Path], out: Path, fps: int, title: str, caption: str, source: str) -> Path:
    """Self-contained player: frames embedded as WebP data URIs (no server, no external files)."""
    from PIL import Image

    uris = []
    for p in frames:
        b = io.BytesIO()
        Image.open(p).convert("RGB").save(b, format="WEBP", quality=85)
        uris.append("data:image/webp;base64," + base64.b64encode(b.getvalue()).decode())
    esc = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
    out.write_text(
        HTML.format(
            title="".join(esc.get(c, c) for c in title),
            caption="".join(esc.get(c, c) for c in caption),
            source="".join(esc.get(c, c) for c in source),
            frames=json.dumps(uris),
            fps=fps,
            last=len(uris) - 1,
        ),
        encoding="utf-8",
    )
    return out


def render(
    scene: Scene,
    out_dir: Path,
    formats: tuple[str, ...],
    dpi: int = 110,
    quick: int | None = None,
    work_dir: Path | None = None,
) -> dict:
    """Render a scene to every requested format; returns a manifest entry (files, sizes, frames, seconds)."""
    bad = [f for f in formats if f not in FORMATS]
    if bad:
        raise ValueError(f"unknown format(s) {bad}; choose from {FORMATS}")
    t0 = time.perf_counter()
    S.apply_style("report")
    out_dir.mkdir(parents=True, exist_ok=True)
    work = (work_dir or out_dir / "_frames") / scene.name
    frames = render_frames(scene, work, dpi, quick)
    files: dict[str, str] = {}
    for f in formats:
        if f == "gif":
            files[f] = str(write_gif(frames, out_dir / f"{scene.name}.gif", scene.fps))
        elif f == "webp":
            files[f] = str(write_webp(frames, out_dir / f"{scene.name}.webp", scene.fps))
        elif f == "apng":
            files[f] = str(write_apng(frames, out_dir / f"{scene.name}.apng.png", scene.fps))
        elif f == "html":
            files[f] = str(
                write_html(frames, out_dir / f"{scene.name}.html", scene.fps, scene.title, scene.caption, scene.source)
            )
        elif f == "frames":
            d = out_dir / f"{scene.name}_frames"
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(work, d)
            files[f] = str(d)
    if "mp4" in formats or "pptx" in formats:
        from aerosynapse.thermal.viz.anim import convert

        src = files.get("gif") or str(write_gif(frames, out_dir / f"{scene.name}.gif", scene.fps))
        duration = len(frames) / scene.fps
        if "mp4" in formats:
            files["mp4"] = str(convert.to_mp4(Path(src), out_dir / f"{scene.name}.mp4", duration))
        if "pptx" in formats:
            files["pptx"] = str(convert.to_pptx([Path(src)], out_dir / f"{scene.name}.pptx", [scene.title], [duration]))
    sizes = {
        k: (Path(v).stat().st_size if Path(v).is_file() else sum(p.stat().st_size for p in Path(v).glob("*")))
        for k, v in files.items()
    }
    from PIL import Image

    w, h = Image.open(frames[0]).size
    return {
        "name": scene.name,
        "title": scene.title,
        "caption": scene.caption,
        "category": scene.category,
        "source": scene.source,
        "frames": len(frames),
        "fps": scene.fps,
        "seconds": round(len(frames) / scene.fps, 2),
        "size_px": [w, h],
        "files": {k: str(Path(v).relative_to(ROOT)) if Path(v).is_relative_to(ROOT) else v for k, v in files.items()},
        "bytes": sizes,
        "render_s": round(time.perf_counter() - t0, 1),
        "meta": scene.meta,
    }
