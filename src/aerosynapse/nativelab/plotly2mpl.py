"""Plotly figures drawn again with Matplotlib (PNG), so the apps' charts show in Studio without a browser.

Covers what the apps use: Scatter (lines, markers, fill), Bar, Heatmap, Contour, Scatter3d, Surface and Mesh3d, with
titles, axis titles, log axes, legends, colour bars and make_subplots grids (each trace's xaxis/yaxis pair).
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

import numpy as np

DASH = {
    "dash": (0, (5, 3)),
    "dot": (0, (1, 2)),
    "dashdot": (0, (5, 2, 1, 2)),
    "longdash": (0, (9, 3)),
    "longdashdot": (0, (9, 3, 1, 3)),
    "solid": "-",
}
MARK = {
    "circle": "o",
    "square": "s",
    "diamond": "D",
    "cross": "P",
    "x": "X",
    "triangle-up": "^",
    "triangle-down": "v",
    "star": "*",
    "circle-open": "o",
    "square-open": "s",
    "diamond-open": "D",
}


def _txt(t: Any) -> str:
    if t is None:
        return ""
    s = t.text if hasattr(t, "text") else (t.get("text") if isinstance(t, dict) else t)
    s = str(s or "")
    s = re.sub(r"<br\s*/?>", "\n", s)
    return re.sub(r"<[^>]+>", "", s)


def _arr(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    try:
        a = np.atleast_1d(np.asarray(v))
        return a if a.size else None
    except Exception:  # noqa: BLE001
        return None


def _cmap(cs: Any) -> Any:
    import matplotlib.colors as mc

    if isinstance(cs, str):
        name = cs.lower().rstrip("_r")
        table = {
            "viridis": "viridis",
            "inferno": "inferno",
            "magma": "magma",
            "plasma": "plasma",
            "rdbu": "RdBu",
            "jet": "jet",
            "turbo": "turbo",
            "hot": "hot",
            "blues": "Blues",
            "reds": "Reds",
            "greys": "Greys",
            "ylorrd": "YlOrRd",
            "rdylbu": "RdYlBu",
            "picnic": "coolwarm",
            "balance": "coolwarm",
            "icefire": "coolwarm",
            "thermal": "inferno",
            "cividis": "cividis",
        }
        m = table.get(name, "viridis")
        return m + ("_r" if cs.endswith("_r") else "")
    try:  # [[0, colour], [1, colour], ...]
        pts = [(float(p[0]), p[1]) for p in cs]
        return mc.LinearSegmentedColormap.from_list("plotly", [(t, _col(c)) for t, c in pts])
    except Exception:  # noqa: BLE001
        return "viridis"


def _col(c: Any) -> Any:
    if c is None:
        return None
    if isinstance(c, str):
        m = re.match(r"rgba?\(([^)]+)\)", c.replace(" ", ""))
        if m:
            v = [float(x) for x in m.group(1).split(",")]
            return tuple([v[0] / 255, v[1] / 255, v[2] / 255] + ([v[3]] if len(v) > 3 else []))
        return c
    return None


def render(fig: Any, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = fig.to_dict() if hasattr(fig, "to_dict") else dict(fig)
    data, lay = d.get("data", []), d.get("layout", {})
    three = any(t.get("type") in ("scatter3d", "surface", "mesh3d") for t in data)
    # the subplot grid from the axes the traces use
    axes_used = sorted(
        {
            (t.get("xaxis", "x"), t.get("yaxis", "y"))
            for t in data
            if t.get("type") not in ("scatter3d", "surface", "mesh3d")
        }
    )
    scenes = sorted({t.get("scene", "scene") for t in data if t.get("type") in ("scatter3d", "surface", "mesh3d")})
    panels: list[tuple[str, Any]] = [("2d", a) for a in axes_used] + [("3d", s) for s in scenes]
    n = max(1, len(panels))
    cols = 1 if n == 1 else 2 if n <= 4 else 3
    rows = int(np.ceil(n / cols))
    h = float(lay.get("height") or 460) / 100
    fig_m = plt.figure(figsize=(min(16, 6.4 * cols + (1.2 if n == 1 else 0)), max(3.6, min(h, 6.0)) * rows))
    ax_of: dict[Any, Any] = {}
    for i, (kind, key) in enumerate(panels):
        ax_of[key] = fig_m.add_subplot(rows, cols, i + 1, projection="3d" if kind == "3d" else None)
    if not panels:
        ax_of[("x", "y")] = fig_m.add_subplot(1, 1, 1)
    legend_axes = set()
    for t in data:
        typ = t.get("type", "scatter")
        if typ in ("scatter3d", "surface", "mesh3d"):
            ax = ax_of[t.get("scene", "scene")]
        else:
            ax = ax_of.get((t.get("xaxis", "x"), t.get("yaxis", "y"))) or next(iter(ax_of.values()))
        name = t.get("name")
        show = t.get("showlegend", True) is not False and name
        line = t.get("line") or {}
        mk = t.get("marker") or {}
        color = _col(line.get("color")) or _col(mk.get("color") if isinstance(mk.get("color"), str) else None)
        kw: dict[str, Any] = {}
        if show:
            kw["label"] = name
            legend_axes.add(ax)
        if typ in ("scatter", "scattergl"):
            x, y = _arr(t.get("x")), _arr(t.get("y"))
            if y is None:
                continue
            if x is None:
                x = np.arange(len(y))
            mode = t.get("mode", "lines")
            ls = DASH.get(line.get("dash", "solid"), "-") if "lines" in mode else "none"
            m = MARK.get(str(mk.get("symbol", "circle")), "o") if "markers" in mode else None
            try:
                ax.plot(
                    x,
                    y,
                    linestyle=ls,
                    marker=m,
                    color=color,
                    lw=float(line.get("width", 2)) * 0.75,
                    ms=float(mk.get("size", 6)) * 0.8 if isinstance(mk.get("size", 6), (int, float)) else 4,
                    **kw,
                )
            except Exception:  # noqa: BLE001  (dates and categories as text)
                ax.plot([str(v) for v in x], y, linestyle=ls, marker=m, color=color, **kw)
            if t.get("fill") in ("tonexty", "tozeroy") and len(ax.lines) >= 1:
                ax.fill_between(
                    np.asarray(x, float),
                    np.asarray(y, float),
                    0
                    if t.get("fill") == "tozeroy"
                    else np.asarray(ax.lines[-2].get_ydata(), float)
                    if len(ax.lines) >= 2
                    else 0,
                    color=color or "C0",
                    alpha=0.2,
                )
            ey = (t.get("error_y") or {}).get("array")
            if ey is not None:
                ax.errorbar(x, y, yerr=np.asarray(ey, float), fmt="none", ecolor=color or "k", alpha=0.6)
        elif typ == "bar":
            x, y = _arr(t.get("x")), _arr(t.get("y"))
            if t.get("orientation") == "h":
                ax.barh(
                    [str(v) for v in y],
                    np.asarray(x, float),
                    color=_col(mk.get("color")) if isinstance(mk.get("color"), str) else None,
                    **kw,
                )
            else:
                ax.bar(
                    [str(v) for v in x] if x is not None and x.dtype.kind not in "if" else x,
                    np.asarray(y, float),
                    color=_col(mk.get("color")) if isinstance(mk.get("color"), str) else None,
                    **kw,
                )
        elif typ in ("heatmap", "contour"):
            z = _arr(t.get("z"))
            if z is None:
                continue
            x, y = _arr(t.get("x")), _arr(t.get("y"))
            x = x if x is not None else np.arange(z.shape[1])
            y = y if y is not None else np.arange(z.shape[0])
            cm = _cmap(t.get("colorscale", "viridis"))
            zmin, zmax = t.get("zmin"), t.get("zmax")
            if t.get("zmid") is not None and zmin is None:
                r = float(np.nanmax(np.abs(z - t["zmid"])))
                zmin, zmax = t["zmid"] - r, t["zmid"] + r
            if typ == "heatmap":
                im = ax.pcolormesh(x, y, z, cmap=cm, vmin=zmin, vmax=zmax, shading="auto")
            else:
                im = ax.contourf(x, y, z, 24, cmap=cm, vmin=zmin, vmax=zmax)
            cb = fig_m.colorbar(im, ax=ax)
            cbt = _txt((t.get("colorbar") or {}).get("title"))
            if cbt:
                cb.set_label(cbt)
        elif typ == "scatter3d":
            x, y, z = (_arr(t.get(k)) for k in ("x", "y", "z"))
            if z is None:
                continue
            mode = t.get("mode", "lines")
            if "lines" in mode:
                ax.plot(x, y, z, color=color, **kw)
            if "markers" in mode:
                c = mk.get("color")
                ax.scatter(
                    x,
                    y,
                    z,
                    c=c if not isinstance(c, str) else _col(c),
                    s=float(mk.get("size", 4)) * 3 if isinstance(mk.get("size", 4), (int, float)) else 10,
                    cmap=_cmap(mk.get("colorscale", "viridis")),
                    **({} if "lines" in mode else kw),
                )
        elif typ == "surface":
            z = _arr(t.get("z"))
            if z is None:
                continue
            x, y = _arr(t.get("x")), _arr(t.get("y"))
            if x is None or x.ndim == 1:
                X, Y = np.meshgrid(
                    x if x is not None else np.arange(z.shape[1]), y if y is not None else np.arange(z.shape[0])
                )
            else:
                X, Y = x, y
            ax.plot_surface(
                X, Y, z, cmap=_cmap(t.get("colorscale", "viridis")), alpha=float(t.get("opacity", 1) or 1), linewidth=0
            )
        elif typ == "mesh3d":
            x, y, z = (_arr(t.get(k)) for k in ("x", "y", "z"))
            i, j, k = (_arr(t.get(q)) for q in ("i", "j", "k"))
            if x is None or i is None:
                continue
            ic = _arr(t.get("intensity"))
            tri = np.stack([i, j, k], 1).astype(int)
            col = None
            if ic is not None and len(ic) == len(x):
                import matplotlib.cm as mcm

                v = ic[tri].mean(1)
                col = mcm.get_cmap(_cmap(t.get("colorscale", "viridis")))((v - v.min()) / max(1e-12, np.ptp(v)))
            ax.plot_trisurf(
                x,
                y,
                z,
                triangles=tri,
                color=_col(t.get("color")) if col is None else None,
                linewidth=0,
                alpha=float(t.get("opacity", 1) or 1),
                shade=True,
            ) if col is None else ax.add_collection3d(
                __import__("mpl_toolkits.mplot3d.art3d", fromlist=["Poly3DCollection"]).Poly3DCollection(
                    np.stack([x, y, z], 1)[tri], facecolors=col, linewidths=0
                )
            )
            ax.auto_scale_xyz(x, y, z)
    # titles, axis titles, log axes
    for key, ax in ax_of.items():
        if isinstance(key, tuple):
            xa = lay.get("xaxis" + key[0][1:], {}) if key[0] != "x" else lay.get("xaxis", {})
            ya = lay.get("yaxis" + key[1][1:], {}) if key[1] != "y" else lay.get("yaxis", {})
            ax.set_xlabel(_txt(xa.get("title")))
            ax.set_ylabel(_txt(ya.get("title")))
            if xa.get("type") == "log":
                ax.set_xscale("log")
            if ya.get("type") == "log":
                ax.set_yscale("log")
            if xa.get("range") and xa.get("type") != "log":
                ax.set_xlim(*xa["range"])
            if ya.get("range") and ya.get("type") != "log":
                ax.set_ylim(*ya["range"])
            if ya.get("scaleanchor") == "x":
                ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.3)
        else:
            sc = lay.get(key, {})
            for a, f in (("xaxis", ax.set_xlabel), ("yaxis", ax.set_ylabel), ("zaxis", ax.set_zlabel)):
                f(_txt((sc.get(a) or {}).get("title")))
    for a in legend_axes:
        a.legend(fontsize=8)
    # subplot titles (annotations at the top of each panel) and the figure title
    ann = [x for x in lay.get("annotations", []) if x.get("yref") == "paper" and x.get("xref") == "paper"]
    for a_, ax in zip(ann, ax_of.values(), strict=False):
        ax.set_title(_txt(a_.get("text")), fontsize=10)
    t = _txt(lay.get("title"))
    if t:
        fig_m.suptitle(t, fontsize=12, fontweight="bold")
    with contextlib.suppress(RuntimeError):  # colour bars of a mixed 2D/3D figure: the saved bounding box does it
        fig_m.tight_layout()
    fig_m.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig_m)
    _ = three
