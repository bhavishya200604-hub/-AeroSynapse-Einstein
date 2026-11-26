"""Project publication style (src/viz/style.py, Phase 7 Step 7.1): one style for every figure, reproducible files."""

import hashlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from aerosynapse.thermal.viz import inverse_figs, style


def test_palette_is_okabe_ito_and_unique():
    okabe = {"#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"}
    assert set(style.PALETTE) == okabe and len(style.PALETTE) == 8
    assert set(style.OKABE_ITO) == okabe


def test_apply_style_sets_publication_rc():
    style.apply_style()
    assert mpl.rcParams["savefig.dpi"] == 300 and mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["image.cmap"] == "viridis" and mpl.rcParams["axes.facecolor"] == "#ffffff"
    assert [c for c in mpl.rcParams["axes.prop_cycle"].by_key()["color"]] == style.PALETTE
    style.apply_style("report")
    assert mpl.rcParams["font.size"] > style.rc("paper")["font.size"]


def test_legacy_styles_delegate():
    from aerosynapse.thermal.paper_common import figstyle

    mpl.rcParams.update(mpl.rcParamsDefault)
    inverse_figs.style()
    a = {k: mpl.rcParams[k] for k in style.rc()}
    mpl.rcParams.update(mpl.rcParamsDefault)
    figstyle.apply()
    b = {k: mpl.rcParams[k] for k in style.rc()}
    assert a == b
    assert inverse_figs.PARAM_COLOR["alpha"] == style.PALETTE[0]


def test_label_has_symbol_and_unit():
    assert style.label("Pe") == "Péclet number $\\mathrm{Pe}$"
    assert style.label("t", "s") == "time $t$ [s]"
    assert style.label("Nu", long=False) == "$\\mathrm{Nu}$"
    assert style.label("anything") == "anything"


def test_save_fig_writes_pdf_png_and_is_reproducible(tmp_path):
    style.apply_style()

    def make():
        fig, ax = plt.subplots(figsize=(style.COL_W, 2))
        ax.plot(np.linspace(0, 1, 11), np.linspace(0, 1, 11) ** 2)
        return fig

    paths = style.save_fig(make(), "t", also_to=tmp_path, docs=False)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    h1 = hashlib.sha256((tmp_path / "t.pdf").read_bytes()).hexdigest()
    style.save_fig(make(), "t", also_to=tmp_path, docs=False)
    assert hashlib.sha256((tmp_path / "t.pdf").read_bytes()).hexdigest() == h1


def test_colorbar_moves_power_of_ten_into_label():
    style.apply_style()
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([[-7.5e-4, 7.5e-4]]), cmap=style.DIV_CMAP)
    cb = style.colorbar(fig, im, ax, "signed error")
    assert cb.ax.get_ylabel() == r"signed error [$\times 10^{-4}$]"
    im2 = ax.imshow(np.array([[0.0, 0.9]]))
    assert style.colorbar(fig, im2, ax, "psi").ax.get_ylabel() == "psi"
    plt.close(fig)


def test_log_tick_labels_do_not_overlap():
    """Phase 7.3: short log axes get readable, non-overlapping tick labels (at least two on each axis)."""
    style.apply_style("report")
    fig, axs = plt.subplots(1, 2, figsize=(6, 2.5))
    h = np.geomspace(8e-3, 6e-2, 5)
    axs[0].loglog(h, h**2, "o-")
    axs[1].loglog(np.geomspace(30, 300, 5), np.geomspace(1e-1, 1e-15, 5))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    assert any(style._labels_overlap(a.xaxis, r)[0] for a in axs)  # matplotlib's default collides here
    style.tidy_log_ticks(fig)
    for a in axs:
        for axis in (a.xaxis, a.yaxis):
            bad, n = style._labels_overlap(axis, r)
            assert not bad and n >= 2
    plt.close(fig)
