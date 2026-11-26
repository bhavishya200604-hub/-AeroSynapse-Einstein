"""Phase 7: render the project's animations in every requested format, with a gallery and a showreel — one command.

    python -m aerosynapse.thermal.experiments.p7_animate                         # everything in configs/viz/animations.yaml
    python -m aerosynapse.thermal.experiments.p7_animate --only cavity graetz    # some scenes
    python -m aerosynapse.thermal.experiments.p7_animate --formats gif mp4       # other formats
    python -m aerosynapse.thermal.experiments.p7_animate --quick 6               # smoke test: 6 frames per scene
    python -m aerosynapse.thermal.experiments.p7_animate --list                  # what exists and what it shows
    python -m aerosynapse.thermal.experiments.p7_animate --gallery-only          # rebuild index.html from the last manifest

Physics scenes run the verified solvers (results cached on disk, so later renders recompute nothing); learning scenes
read saved training logs and results; no network is trained. Writes <output_dir>/<scene>.{gif,html,webp,mp4,...},
<output_dir>/index.html (gallery), <output_dir>/showreel.{mp4,pptx} and results/p7_animate_<tag>/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import yaml

from aerosynapse.thermal.data.spec import coerce_numbers, deep_merge, parse_set
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.viz.anim import core

DEFAULT = ROOT / "configs" / "viz" / "animations.yaml"


def registry() -> dict:
    from aerosynapse.thermal.viz.anim import learning, physics, pipeline, studies

    return {**pipeline.SCENES, **physics.SCENES, **learning.SCENES, **studies.SCENES}


def load_cfg(files: list[str], sets: list[str]) -> dict:
    cfg = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    for f in files:
        cfg = deep_merge(cfg, coerce_numbers(yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}))
    return deep_merge(cfg, parse_set(sets))


def build_gallery(entries: list[dict], out_dir: Path, extra: dict | None = None) -> Path:
    from aerosynapse.thermal.viz.anim import gallery

    return gallery.write(entries, out_dir, extra or {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--formats", nargs="+", default=None)
    ap.add_argument("--quick", type=int, default=None, help="render only this many evenly spaced frames per scene")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--gallery-only", action="store_true")
    ap.add_argument("--no-showreel", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.sets)
    reg = registry()
    out_dir = ROOT / cfg["output_dir"]
    cache = ROOT / cfg["cache_dir"]
    names = a.only or [n for n in cfg["order"] if n in cfg["scenes"]]
    unknown = [n for n in names if n not in reg]
    if unknown:
        raise SystemExit(f"unknown scene(s) {unknown}; available: {sorted(reg)}")
    if a.list:
        for n in names:  # no computation: name, module, and the scene's settings
            mod = reg[n].__module__.rsplit(".", 1)[-1]
            print(f"{n:22s} [{mod}] {cfg['scenes'][n]}")
        return
    rdir = ROOT / "results" / f"p7_animate_{a.tag}"
    if a.gallery_only:
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        print(build_gallery(met["animations"], out_dir, met.get("showreel")))
        return
    formats = tuple(a.formats or cfg["formats"])
    with Experiment(f"p7_animate_{a.tag}", cfg | {"seed": 0, "formats": list(formats), "quick": a.quick}) as ex:
        entries, failures = [], []
        for n in names:
            t0 = time.perf_counter()
            try:
                scene = reg[n](cfg["scenes"][n], cache)
                e = core.render(scene, out_dir, formats, dpi=int(cfg["dpi"]), quick=a.quick, work_dir=cache / "frames")
                e["prepare_s"] = round(time.perf_counter() - t0 - e["render_s"], 1)
                entries.append(e)
                print(
                    f"[{n}] {e['frames']} frames, {e['seconds']} s of animation, "
                    f"{', '.join(e['files'])} ({time.perf_counter() - t0:.0f} s)",
                    flush=True,
                )
            except Exception as err:  # noqa: BLE001  (one failing scene must not stop the others; reported below)
                failures.append({"scene": n, "error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()})
                print(f"[{n}] FAILED: {type(err).__name__}: {err}", flush=True)
        if a.only and not a.quick:
            entries = merge_previous(entries, rdir / "metrics.json", cfg["order"])
        reel = None
        if cfg.get("showreel") and not a.no_showreel and entries and not a.quick:
            reel = showreel(entries, out_dir)
        ex.metrics["animations"] = entries
        ex.metrics["failures"] = failures
        ex.metrics["showreel"] = reel
        ex.metrics["git"] = git_hash()
        print(build_gallery(entries, out_dir, reel))
    if failures:
        raise SystemExit(f"{len(failures)} scene(s) failed: {[f['scene'] for f in failures]}")


def merge_previous(entries: list[dict], metrics: Path, order: list[str]) -> list[dict]:
    """--only re-renders some scenes: keep the other scenes of the last run with the same tag (whose files still
    exist), so the gallery, the showreel and the manifest always cover every animation."""
    if not metrics.exists():
        return entries
    new = {e["name"] for e in entries}
    old = [
        e
        for e in json.loads(metrics.read_text(encoding="utf-8")).get("animations", [])
        if e["name"] not in new and all((ROOT / f).exists() for f in e["files"].values())
    ]
    rank = {n: k for k, n in enumerate(order)}
    return sorted(old + entries, key=lambda e: rank.get(e["name"], len(rank)))


def showreel(entries: list[dict], out_dir: Path) -> dict:
    """Every animation in a row: one PowerPoint deck and one video (titled slides). Needs PowerPoint (Windows)."""
    from aerosynapse.thermal.viz.anim import convert

    gifs = [ROOT / e["files"]["gif"] for e in entries if "gif" in e["files"]]
    titles = [e["title"] for e in entries if "gif" in e["files"]]
    durs = [e["seconds"] for e in entries if "gif" in e["files"]]
    out = {}
    try:
        out["pptx"] = str(convert.to_pptx(gifs, out_dir / "showreel.pptx", titles, durs).relative_to(ROOT))
        out["mp4"] = str(convert.deck_video(gifs, out_dir / "showreel.mp4", titles, durs).relative_to(ROOT))
    except RuntimeError as err:
        out["error"] = str(err)
    return out


if __name__ == "__main__":
    main()
