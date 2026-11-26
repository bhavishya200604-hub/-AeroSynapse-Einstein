"""Animation engine (src/viz/anim, Phase 7): render a scene to every format, convert between formats, cache solver
data, and the scene registry / config agree. MP4 / PPTX need Microsoft PowerPoint and are tested only when it exists."""

import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from PIL import Image

from aerosynapse.thermal.viz.anim import convert, core


def _scene(n=5, fps=10):
    def make():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(2.0, 1.6))
        x = np.linspace(0, 1, 50)
        (ln,) = ax.plot(x, 0 * x)
        ax.set_ylim(-1, 1)

        def update(i):
            ln.set_ydata(np.sin(2 * np.pi * (x - i / n)))

        return fig, update

    return core.Scene("wave", "A wave", "test & <caption>", n, fps, make, source="test")


def _n_frames(path: Path) -> int:
    with Image.open(path) as im:
        return getattr(im, "n_frames", 1)


def test_render_every_pillow_format(tmp_path):
    e = core.render(_scene(), tmp_path, ("gif", "webp", "apng", "html", "frames"), dpi=50, work_dir=tmp_path / "w")
    assert (
        e["frames"] == 5 and e["seconds"] == 0.5 and abs(e["size_px"][0] - 100) <= 1 and abs(e["size_px"][1] - 80) <= 1
    )
    for f in ("gif", "webp", "apng"):
        assert _n_frames(Path(e["files"][f])) == 5, f
    html = Path(e["files"]["html"]).read_text(encoding="utf-8")
    assert html.count("data:image/webp;base64,") == 5 and "&amp; &lt;caption&gt;" in html
    assert len(list(Path(e["files"]["frames"]).glob("*.png"))) == 5
    assert all(b > 0 for b in e["bytes"].values())


def test_quick_render_and_unknown_format(tmp_path):
    e = core.render(_scene(n=20), tmp_path, ("gif",), dpi=40, quick=3, work_dir=tmp_path / "w")
    assert e["frames"] == 3
    with pytest.raises(ValueError):
        core.render(_scene(), tmp_path, ("avi",))


def test_convert_round_trip(tmp_path):
    e = core.render(_scene(n=6, fps=12), tmp_path / "a", ("gif",), dpi=40, work_dir=tmp_path / "w")
    out = convert.convert(Path(e["files"]["gif"]), ["webp", "html", "frames"], tmp_path / "b")
    assert _n_frames(out["webp"]) == 6 and len(list(out["frames"].glob("*.png"))) == 6
    frames, fps = convert.read_frames(out["webp"], tmp_path / "c")
    assert len(frames) == 6 and abs(fps - 12) <= 1
    back = convert.convert(out["frames"], ["gif"], tmp_path / "d")  # a PNG folder converts too
    assert _n_frames(back["gif"]) == 6


def test_cache_computes_once(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"a": np.arange(3.0)}

    d1, hit1 = core.cached(tmp_path, "s", {"Re": 1.0}, compute)
    d2, hit2 = core.cached(tmp_path, "s", {"Re": 1.0}, compute)
    d3, hit3 = core.cached(tmp_path, "s", {"Re": 2.0}, compute)
    assert (hit1, hit2, hit3) == (False, True, False) and len(calls) == 2
    assert np.array_equal(d2["a"], d1["a"])


def test_cache_key_follows_the_code(tmp_path, monkeypatch):
    """Editing the module that computes a scene must invalidate its cached data."""
    calls = []

    def compute():
        calls.append(1)
        return {"a": np.zeros(1)}

    core.cached(tmp_path, "s", {}, compute)
    monkeypatch.setattr(core, "_code_version", lambda fn: "edited")
    core.cached(tmp_path, "s", {}, compute)
    assert len(calls) == 2


def test_registry_matches_config():
    from aerosynapse.thermal.experiments import p7_animate

    cfg = p7_animate.load_cfg([], ["scenes.pulse.Pe=[2.0]"])
    reg = p7_animate.registry()
    assert set(cfg["order"]) == set(cfg["scenes"]) <= set(reg)
    assert cfg["scenes"]["pulse"]["Pe"] == [2.0]  # --set overrides a single value; the rest is kept
    assert set(cfg["formats"]) <= set(core.FORMATS)


def test_pulse_scene_matches_analytic(tmp_path):
    """A physics scene draws the analytic solution it claims (no network, no fitting)."""
    from aerosynapse.thermal.physics import analytic
    from aerosynapse.thermal.viz.anim import physics

    p = {"fps": 10, "frames": 4, "T": 0.5, "Pe": [10.0], "sigma0": 0.05, "x0": 0.25}
    d = physics.compute_pulse(p)
    k = -1
    ref = analytic.gaussian_pulse(d["x"], d["t"][k], 10.0, 0.05, 0.25, 1.0)
    assert np.allclose(np.asarray(d["theta"])[0, k], ref)
    e = core.render(physics.scene_pulse(p, tmp_path), tmp_path / "o", ("gif",), dpi=40, work_dir=tmp_path / "w")
    assert e["frames"] == 4


def test_gallery_lists_every_scene(tmp_path):
    from aerosynapse.thermal.viz.anim import gallery

    e = core.render(_scene(), tmp_path, ("gif", "webp"), dpi=40, work_dir=tmp_path / "w")
    page = gallery.write([e], tmp_path, {}).read_text(encoding="utf-8")
    assert "A wave" in page and "wave.webp" in page and "wave.gif" in page


@pytest.mark.skipif(os.name != "nt" or shutil.which("powershell") is None, reason="needs Windows + PowerPoint")
@pytest.mark.slow
def test_mp4_via_powerpoint(tmp_path):
    e = core.render(_scene(n=10), tmp_path, ("gif", "mp4"), dpi=60, work_dir=tmp_path / "w")
    mp4 = Path(e["files"]["mp4"])
    assert mp4.stat().st_size > 1000 and mp4.read_bytes()[4:8] == b"ftyp"


def test_social_fill_uses_only_known_facts():
    """A LinkedIn template takes numbers only from results files: an unknown field stops the build."""
    from aerosynapse.thermal.viz.anim import social

    F = {"x": {"value": 3.0, "text": "3", "source": "results/a/metrics.json: x"}}
    text, used = social.fill("x = {x} % ({link})", F, {"link": "L"})
    assert text == "x = 3 % (L)" and used == ["x"]
    with pytest.raises(KeyError):
        social.fill("{made_up_number}", F, {})


def test_social_facts_match_the_gate_table():
    """The post's inverse numbers are the Phase 5 gate table's (exp(log error) - 1), with their sources."""
    from aerosynapse.thermal.viz.anim import social

    F = social.facts()
    if "ls_err_k" not in F:
        pytest.skip("results/p5_gate not present")
    table = (__import__("aerosynapse.thermal._home", fromlist=["HOME"]).HOME / "docs/tables/phase5_gate.md").read_text(encoding="utf-8")
    assert f"-{F['ls_err_k']['text']} [" in table and f"-{F['ls_err_alpha']['text']} [" in table
    assert all(v["source"] for v in F.values())


def test_social_card_keeps_its_size(tmp_path):
    from aerosynapse.thermal.viz.anim import social

    im = Image.new("RGB", (400, 200), "white")
    out = social.card(tmp_path / "c.png", (540, 675), "A headline that wraps over lines", "finding", images=[im])
    cov = social.cover(tmp_path / "v.png", (600, 314), "Title", "Subtitle", images=[im, im])
    assert Image.open(out).size == (540, 675) and Image.open(cov).size == (600, 314)
