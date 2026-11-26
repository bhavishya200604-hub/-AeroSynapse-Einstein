"""Phase 7: a LinkedIn progress kit from the project's animations — post text, square video, document carousel, cover
image and one page with copy buttons and the source of every number (docs/linkedin/index.html).

    python -m aerosynapse.thermal.experiments.p7_linkedin                  # build the kit (configs/viz/linkedin.yaml)
    python -m aerosynapse.thermal.experiments.p7_linkedin --facts          # the numbers a post may use, with their sources
    python -m aerosynapse.thermal.experiments.p7_linkedin --no-video       # skip the PowerPoint video export
    python -m aerosynapse.thermal.experiments.p7_linkedin --set author="Your Name"

Numbers are never typed: the post templates use {field} names that are read from results files (src/viz/anim/
social.py: facts); an unknown field stops the build. Needs the rendered animations (python tasks.py animations).
Nothing is uploaded or posted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from aerosynapse.thermal.data.spec import deep_merge, parse_set
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.viz.anim import convert, social

DEFAULT = ROOT / "configs" / "viz" / "linkedin.yaml"


def load_cfg(files: list[str], sets: list[str]) -> dict:
    cfg = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    for f in files:
        cfg = deep_merge(cfg, yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {})
    return deep_merge(cfg, parse_set(sets))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--facts", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.sets)
    F = social.facts()
    if a.facts:
        for k, v in F.items():
            print(f"{{{k}}}".ljust(22), v["text"].rjust(10), "  ", v["source"])
        return

    out = ROOT / cfg["output_dir"]
    adir = ROOT / cfg["animations_dir"]
    missing = [h["scene"] for h in cfg["highlights"] if not (adir / f"{h['scene']}.webp").exists()]
    if missing:
        raise SystemExit(f"render these animations first (python tasks.py animations --only ...): {missing}")
    manifest = {}
    mf = ROOT / "results/p7_animate_default/metrics.json"
    if mf.exists():
        manifest = {e["name"]: e for e in json.loads(mf.read_text(encoding="utf-8"))["animations"]}

    extra = {
        "link": str(cfg.get("link") or ""),
        "hashtags": " ".join(f"#{h}" for h in cfg.get("hashtags", [])),
        "author": str(cfg.get("author") or ""),
    }
    used: list[str] = []

    def fill(t: str) -> str:
        s, u = social.fill(t, F, extra)
        used.extend(n for n in u if n not in used)
        return s.strip()

    post, post_short = fill(cfg["post"]), fill(cfg["post_short"])
    hl = [
        {**h, "headline": fill(h["headline"]), "finding": fill(h["finding"]), "anim": adir / f"{h['scene']}.webp"}
        for h in cfg["highlights"]
    ]
    closing = [fill(c) for c in cfg.get("closing", [])]
    footer = " · ".join(x for x in (extra["author"], extra["link"]) if x)

    with Experiment(f"p7_linkedin_{a.tag}", cfg | {"seed": 0}) as ex:
        out.mkdir(parents=True, exist_ok=True)
        pages_dir = out / "carousel"
        if pages_dir.exists():
            shutil.rmtree(pages_dir)
        P = (1080, 1350)
        pages = [
            social.card(
                pages_dir / "page_01.png",
                P,
                cfg["title"],
                cfg["subtitle"],
                images=[social.key_frame(hl[0]["anim"], hl[0].get("frame", 0.5))],
                body=[f"{len(hl)} highlights, swipe →"],
                footer=footer,
            )
        ]
        for k, h in enumerate(hl, start=2):
            src = manifest.get(h["scene"], {}).get("source", "")
            pages.append(
                social.card(
                    pages_dir / f"page_{k:02d}.png",
                    P,
                    h["headline"],
                    h["finding"],
                    images=[social.key_frame(h["anim"], h.get("frame", 0.5))],
                    footer=f"{k} / {len(hl) + 2}" + (f" · data: {src}" if src else ""),
                )
            )
        pages.append(
            social.card(
                pages_dir / f"page_{len(hl) + 2:02d}.png", P, "Where it stands", "", body=closing, footer=footer
            )
        )
        from PIL import Image

        ims = [Image.open(p).convert("RGB") for p in pages]
        ims[0].save(out / "carousel.pdf", save_all=True, append_images=ims[1:], resolution=144)
        del ims
        thumbs = [
            social.key_frame(adir / f"{s}.webp", 0.8)
            for s in ("obstacle", "rayleigh_benard", "identifiability")
            if (adir / f"{s}.webp").exists()
        ]
        social.card(out / "cover.png", (1200, 627), cfg["title"], cfg["subtitle"], images=thumbs, footer=footer)

        video_s = None
        if cfg.get("video") and not a.no_video:
            with tempfile.TemporaryDirectory() as tmp:
                sq = (1080, 1080)
                first = social.card(
                    Path(tmp) / "cover.png",
                    sq,
                    cfg["title"],
                    cfg["subtitle"],
                    images=[social.key_frame(hl[0]["anim"], hl[0].get("frame", 0.5))],
                    footer=footer,
                )
                last = social.card(Path(tmp) / "closing.png", sq, "Where it stands", "", body=closing, footer=footer)
                gifs, titles, caps, durs = [first], [""], [""], [float(cfg["cover_seconds"])]
                for h in hl:
                    g = adir / f"{h['scene']}.gif"
                    if not g.exists():  # the video needs the GIF; convert the WebP if only that exists
                        g = convert.convert(h["anim"], ["gif"], Path(tmp))["gif"]
                    secs = manifest.get(h["scene"], {}).get("seconds") or 6.0
                    gifs.append(g)
                    titles.append(h["headline"])
                    caps.append(h["finding"])
                    durs.append(min(float(secs), float(cfg["video_seconds_max"])))
                gifs.append(last)
                titles.append("")
                caps.append("")
                durs.append(float(cfg["closing_seconds"]))
                mp4 = out / "teaser.mp4"
                if mp4.exists():
                    mp4.unlink()
                try:
                    convert._ps(
                        convert._deck_script(
                            gifs, titles, durs, None, mp4, height=1080, slide=(540, 540), captions=caps
                        ),
                        timeout=3600,
                    )
                    video_s = sum(durs)
                except RuntimeError as err:
                    ex.metrics["video_error"] = str(err)
                    print("video skipped:", err)

        (out / "post.txt").write_text(post + "\n", encoding="utf-8")
        (out / "post_short.txt").write_text(post_short + "\n", encoding="utf-8")
        (out / "facts.json").write_text(
            json.dumps({n: F[n] for n in used}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        g = git_hash()
        index = social.page(out, post, post_short, F, used, len(pages), video_s, cfg["title"], g[:7] if g else "")
        files = sorted(str(p.relative_to(ROOT)) for p in out.rglob("*") if p.is_file())
        ex.metrics.update(
            {
                "facts_used": {n: F[n] for n in used},
                "post_chars": len(post),
                "post_short_chars": len(post_short),
                "pages": len(pages),
                "video_seconds": video_s,
                "files": files,
                "bytes": {f: (ROOT / f).stat().st_size for f in files},
                "git": g,
            }
        )
    print(index)
    print(f"post {len(post)} characters, short {len(post_short)}; {len(pages)} carousel pages; video {video_s} s")


if __name__ == "__main__":
    main()
