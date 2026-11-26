"""A stand-in for ``streamlit`` that runs the browser apps' own page code inside AeroSynapse Studio.

Every widget a page asks for is recorded (label, kind, default, options, range, help) so Studio can draw it as a native
form field, and is answered with the value Studio sends (or a value taken from the design and assembly, or the page's
default). Everything a page shows is recorded in order as blocks: headings, text, metrics, tables (CSV), figures (PNG:
Matplotlib figures as they are, Plotly figures redrawn with Matplotlib), images, code, JSON, alerts, downloads (files)
and progress. ``st.stop`` and ``st.rerun`` behave as in Streamlit; ``st.session_state`` lives on between runs of the
same app in one worker. Nothing is served to a browser.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import re
import sys
import types
from pathlib import Path
from typing import Any


class StopPage(Exception):
    """st.stop()"""


class RerunPage(Exception):
    """st.rerun()"""


class Recorder:
    def __init__(
        self,
        out: Path,
        values: dict[str, Any],
        press: str | None,
        defaults: dict[str, tuple[Any, str]],
        progress: Path | None,
    ):
        self.out, self.values, self.press, self.defaults, self.progress = out, values, press, defaults, progress
        self.widgets: list[dict[str, Any]] = []
        self.blocks: list[dict[str, Any]] = []
        self.seen: dict[str, int] = {}
        self.n = 0
        self.section: list[str] = []
        self.sidebar = 0
        out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- output
    def block(self, kind: str, **kw: Any) -> None:
        b = {"type": kind, **kw}
        if self.section:
            b["section"] = " / ".join(self.section)
        self.blocks.append(b)

    def file(self, stem: str, ext: str) -> Path:
        self.n += 1
        return self.out / f"{self.n:03d}_{stem}.{ext}"

    # ---------------------------------------------------------------- widgets
    def key(self, kind: str, label: str, key: str | None) -> str:
        if key:
            return str(key)
        k = f"{kind}:{label}"
        i = self.seen.get(k, 0)
        self.seen[k] = i + 1
        return k if i == 0 else f"{k}#{i}"

    def answer(self, kind: str, label: str, key: str | None, default: Any, **meta: Any) -> Any:
        k = self.key(kind, label, key)
        src = "default"
        val = default
        if k in self.values:
            val, src = self.values[k], "user"
        elif key and key in SESSION and kind not in ("button",):
            val, src = SESSION[key], "session"
        else:
            for pat, (v, why) in self.defaults.items():
                if re.search(pat, label, re.IGNORECASE) and kind in ("number_input", "slider"):
                    val, src = v, "design: " + why
                    break
        w = {
            "key": k,
            "kind": kind,
            "label": label,
            "default": _plain(default),
            "value": _plain(val),
            "source": src,
            "sidebar": self.sidebar > 0,
        }
        w.update({m: _plain(v) for m, v in meta.items() if v is not None})
        if self.section:
            w["section"] = " / ".join(self.section)
        self.widgets.append(w)
        if key:
            SESSION[key] = val
        return val


REC: Recorder | None = None
SESSION: Any = None  # the app's st.session_state (set by begin)
CACHE: dict[Any, Any] = {}


def _plain(v: Any) -> Any:
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _num(v: Any, ref: Any) -> Any:
    try:
        if isinstance(ref, bool):
            return bool(v)
        if isinstance(ref, int) and not isinstance(ref, bool):
            return int(round(float(v)))
        return float(v)
    except Exception:  # noqa: BLE001
        return ref


# ======================================================================== the module's functions
class _State(dict):
    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def __delattr__(self, k: str) -> None:
        self.pop(k, None)


def _out(kind: str):
    def f(body: Any = "", *a: Any, **kw: Any) -> None:
        REC.block(kind, text=str(body) if not isinstance(body, str) else body)

    return f


title, header, subheader, caption, text, latex, markdown = (
    _out(k) for k in ("title", "header", "subheader", "caption", "text", "latex", "markdown")
)
info, success, warning, error = (_out(k) for k in ("info", "success", "warning", "error"))


def exception(e: BaseException) -> None:
    REC.block("error", text=f"{type(e).__name__}: {e}")


def code(body: str, language: str | None = None, **kw: Any) -> None:
    REC.block("code", text=str(body))


def divider() -> None:
    REC.block("divider")


def html(body: str, **kw: Any) -> None:
    REC.block("markdown", text=re.sub(r"<[^>]+>", "", str(body)))


def json_(obj: Any, expanded: bool = True, **kw: Any) -> None:  # noqa: ARG001
    REC.block("json", text=json.dumps(_plain_deep(obj), indent=1, default=str)[:20000])


def _plain_deep(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _plain_deep(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain_deep(v) for v in o]
    return _plain(o)


def metric(label: str, value: Any, delta: Any = None, **kw: Any) -> None:
    REC.block("metric", label=str(label), value=str(value), delta=None if delta is None else str(delta))


def write(*objs: Any, **kw: Any) -> None:
    for o in objs:
        mod = type(o).__module__
        if isinstance(o, str):
            REC.block("markdown", text=o)
        elif mod.startswith("pandas"):
            dataframe(o)
        elif mod.startswith("plotly"):
            plotly_chart(o)
        elif mod.startswith("matplotlib"):
            pyplot(o)
        elif isinstance(o, (dict, list)):
            json_(o)
        else:
            REC.block("text", text=str(o))


def dataframe(data: Any = None, *a: Any, **kw: Any) -> Any:
    import pandas as pd

    try:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    except Exception:  # noqa: BLE001
        REC.block("text", text=str(data))
        return data
    p = REC.file("table", "csv")
    df.to_csv(p, index=not isinstance(df.index, pd.RangeIndex))
    REC.block("table", file=p.name, rows=int(len(df)), cols=[str(c) for c in df.columns][:40])
    return data


table = dataframe


def data_editor(data: Any, *a: Any, **kw: Any) -> Any:
    dataframe(data)
    return data


def _chart(kind: str):
    def f(data: Any = None, *a: Any, x: Any = None, y: Any = None, **kw: Any) -> None:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(9, 4.6))
        cols = [y] if isinstance(y, str) else (list(y) if y is not None else [c for c in df.columns if c != x])
        xs = df[x] if x is not None else df.index
        for c in cols:
            if kind == "bar":
                ax.bar(xs, df[c], label=str(c))
            elif kind == "scatter":
                ax.scatter(xs, df[c], s=12, label=str(c))
            else:
                ax.plot(xs, df[c], label=str(c))
        if len(cols) > 1:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        pyplot(fig)

    return f


line_chart, area_chart, bar_chart, scatter_chart = _chart("line"), _chart("line"), _chart("bar"), _chart("scatter")


def pyplot(fig: Any = None, *a: Any, **kw: Any) -> None:
    import matplotlib.pyplot as plt

    fig = fig or plt.gcf()
    p = REC.file("figure", "png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    REC.block("figure", file=p.name)


def plotly_chart(fig: Any, *a: Any, **kw: Any) -> None:
    from aerosynapse.nativelab.plotly2mpl import render

    p = REC.file("figure", "png")
    try:
        render(fig, p)
        REC.block("figure", file=p.name)
    except Exception as e:  # noqa: BLE001
        REC.block("warning", text=f"a chart could not be drawn: {type(e).__name__}: {e}")


def altair_chart(*a: Any, **kw: Any) -> None:
    REC.block("info", text="(an Altair chart: not drawn in Studio)")


vega_lite_chart = altair_chart


def image(img: Any, caption: str | None = None, **kw: Any) -> None:
    p = REC.file("image", "png")
    try:
        if isinstance(img, (str, Path)):
            from PIL import Image

            Image.open(img).save(p)
        else:
            from PIL import Image

            if hasattr(img, "save"):
                img.save(p)
            else:
                import numpy as np

                a = np.asarray(img)
                if a.dtype != np.uint8:
                    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(a).save(p)
        REC.block("figure", file=p.name, caption=caption)
    except Exception as e:  # noqa: BLE001
        REC.block("warning", text=f"an image could not be shown: {e}")


def progress(value: float, text: str | None = None) -> Any:
    v = float(value) / (100 if float(value) > 1 else 1)
    if REC.progress:
        with contextlib.suppress(OSError):
            REC.progress.write_text(json.dumps({"progress": v, "text": text or ""}))
    REC.block("progress", value=v, text=text or "")
    return _Container("progress")


def download_button(label: str, data: Any, file_name: str | None = None, mime: str | None = None, **kw: Any) -> bool:
    d = REC.out / "downloads"
    d.mkdir(exist_ok=True)
    name = re.sub(r"[^\w.\-]+", "_", file_name or "download.bin")
    if hasattr(data, "read"):
        data = data.read()
    (d / name).write_bytes(data if isinstance(data, bytes) else str(data).encode("utf-8"))
    REC.block("download", label=str(label), file="downloads/" + name)
    return False


# ---------------------------------------------------------------- input widgets
def number_input(
    label: str,
    min_value: Any = None,
    max_value: Any = None,
    value: Any = "min",
    step: Any = None,
    format: Any = None,
    key: Any = None,
    help: Any = None,
    **kw: Any,
) -> Any:  # noqa: A002
    if value == "min" or value is None:
        value = min_value if min_value is not None else 0.0
    v = REC.answer("number_input", label, key, value, min=min_value, max=max_value, step=step, help=help)
    v = _num(v, value)
    if min_value is not None:
        v = max(v, min_value)
    if max_value is not None:
        v = min(v, max_value)
    return v


def slider(
    label: str,
    min_value: Any = None,
    max_value: Any = None,
    value: Any = None,
    step: Any = None,
    format: Any = None,
    key: Any = None,
    help: Any = None,
    **kw: Any,
) -> Any:  # noqa: A002
    if value is None:
        value = min_value if min_value is not None else 0
    if isinstance(value, (list, tuple)):
        v = REC.answer("slider_range", label, key, list(value), min=min_value, max=max_value, step=step, help=help)
        try:
            a, b = (float(x) for x in (v.split(",") if isinstance(v, str) else v))
            return (_num(a, value[0]), _num(b, value[1]))
        except Exception:  # noqa: BLE001
            return tuple(value)
    v = REC.answer("slider", label, key, value, min=min_value, max=max_value, step=step, help=help)
    v = _num(v, value)
    if min_value is not None:
        v = max(v, min_value)
    if max_value is not None:
        v = min(v, max_value)
    return v


def _choice(kind: str, label: str, options: Any, index: int | None, format_func: Any, key: Any, help: Any) -> Any:
    opts = list(options)
    shown = [str(format_func(o)) for o in opts]
    if not opts:
        REC.answer(kind, label, key, None, options=[], help=help)
        return None
    d = shown[index if index is not None and 0 <= index < len(opts) else 0]
    v = REC.answer(kind, label, key, d, options=shown, help=help)
    if v in shown:
        return opts[shown.index(v)]
    if v in opts:
        return v
    return opts[shown.index(d)]


def selectbox(
    label: str,
    options: Any,
    index: int | None = 0,
    format_func: Any = str,
    key: Any = None,
    help: Any = None,
    **kw: Any,
) -> Any:
    return _choice("selectbox", label, options, index, format_func, key, help)


def radio(
    label: str,
    options: Any,
    index: int | None = 0,
    format_func: Any = str,
    key: Any = None,
    help: Any = None,
    **kw: Any,
) -> Any:
    return _choice("radio", label, options, index, format_func, key, help)


def segmented_control(
    label: str, options: Any, default: Any = None, format_func: Any = str, key: Any = None, help: Any = None, **kw: Any
) -> Any:
    opts = list(options)
    return _choice("radio", label, opts, opts.index(default) if default in opts else 0, format_func, key, help)


pills = segmented_control


def select_slider(
    label: str,
    options: Any = (),
    value: Any = None,
    format_func: Any = str,
    key: Any = None,
    help: Any = None,
    **kw: Any,
) -> Any:
    opts = list(options)
    return _choice("selectbox", label, opts, opts.index(value) if value in opts else 0, format_func, key, help)


def multiselect(
    label: str, options: Any, default: Any = None, format_func: Any = str, key: Any = None, help: Any = None, **kw: Any
) -> Any:
    opts = list(options)
    shown = [str(format_func(o)) for o in opts]
    d = [str(format_func(o)) for o in (default or [])] if not isinstance(default, str) else [default]
    v = REC.answer("multiselect", label, key, d, options=shown, help=help)
    if isinstance(v, str):
        v = [s.strip() for s in v.split(";") if s.strip()]
    return [opts[shown.index(s)] for s in v if s in shown]


def checkbox(label: str, value: bool = False, key: Any = None, help: Any = None, **kw: Any) -> bool:
    v = REC.answer("checkbox", label, key, bool(value), help=help)
    return v in (True, "true", "1", 1, "yes")


toggle = checkbox


def text_input(label: str, value: str = "", key: Any = None, help: Any = None, **kw: Any) -> str:
    return str(REC.answer("text_input", label, key, value or "", help=help))


def text_area(label: str, value: str = "", key: Any = None, help: Any = None, **kw: Any) -> str:
    return str(REC.answer("text_area", label, key, value or "", help=help))


class _Upload(io.BytesIO):
    def __init__(self, path: str):
        super().__init__(Path(path).read_bytes())
        self.name = Path(path).name
        self.size = len(self.getvalue())

    def getbuffer(self):  # noqa: D401
        return memoryview(self.getvalue())


def file_uploader(
    label: str, type: Any = None, accept_multiple_files: bool = False, key: Any = None, help: Any = None, **kw: Any
) -> Any:  # noqa: A002
    v = REC.answer(
        "file_uploader",
        label,
        key,
        "",
        types=list(type) if isinstance(type, (list, tuple)) else ([type] if type else None),
        help=help,
    )
    if v and Path(str(v)).exists():
        f = _Upload(str(v))
        return [f] if accept_multiple_files else f
    return [] if accept_multiple_files else None


def button(label: str, key: Any = None, help: Any = None, **kw: Any) -> bool:
    k = REC.key("button", label, key)
    REC.widgets.append(
        {
            "key": k,
            "kind": "button",
            "label": label,
            "help": help,
            "sidebar": REC.sidebar > 0,
            **({"section": " / ".join(REC.section)} if REC.section else {}),
        }
    )
    return REC.press == k


form_submit_button = button


def color_picker(label: str, value: str = "#000000", key: Any = None, **kw: Any) -> str:
    return str(REC.answer("text_input", label, key, value))


def date_input(label: str, value: Any = None, key: Any = None, **kw: Any) -> Any:
    return REC.answer("text_input", label, key, str(value) if value else "")


time_input = date_input


# ---------------------------------------------------------------- layout
class _Container:
    def __init__(self, name: str | None = None, sidebar: bool = False):
        self.name, self.is_sidebar = name, sidebar

    def __enter__(self):
        if self.is_sidebar:
            REC.sidebar += 1
        elif self.name:
            REC.section.append(self.name)
        return self

    def __exit__(self, *a: Any) -> bool:
        if self.is_sidebar:
            REC.sidebar -= 1
        elif self.name:
            REC.section.pop()
        return False

    def __getattr__(self, name: str) -> Any:
        f = getattr(MODULE, name)
        if not callable(f):
            return f

        @functools.wraps(f)
        def g(*a: Any, **kw: Any) -> Any:
            with self:
                return f(*a, **kw)

        return g

    # a progress bar or an empty slot can be updated
    def progress(self, value: float, text: str | None = None) -> Any:
        return progress(value, text)

    def empty(self) -> Any:
        return _Container()


def columns(spec: Any, **kw: Any) -> list[_Container]:
    n = spec if isinstance(spec, int) else len(spec)
    return [_Container() for _ in range(n)]


def tabs(names: Any) -> list[_Container]:
    return [_Container(str(n)) for n in names]


def expander(label: str, expanded: bool = False, **kw: Any) -> _Container:
    return _Container(str(label))


def container(**kw: Any) -> _Container:
    return _Container()


empty = container
popover = expander


def form(key: str, **kw: Any) -> _Container:
    return _Container()


@contextlib.contextmanager
def spinner(text: str = "", **kw: Any):
    yield


def status(label: str = "", **kw: Any) -> _Container:
    return _Container(str(label))


def toast(body: str = "", **kw: Any) -> None:
    REC.block("info", text=str(body))


def balloons() -> None:
    pass


snow = balloons


def set_page_config(**kw: Any) -> None:
    pass


def stop() -> None:
    raise StopPage


def rerun(**kw: Any) -> None:
    raise RerunPage


experimental_rerun = rerun


def _cache(func: Any = None, **kw: Any) -> Any:
    def deco(f: Any) -> Any:
        @functools.wraps(f)
        def g(*a: Any, **k: Any) -> Any:
            try:
                key = (f.__module__, f.__qualname__, repr(a), repr(sorted(k.items())))
            except Exception:  # noqa: BLE001
                return f(*a, **k)
            if key not in CACHE:
                CACHE[key] = f(*a, **k)
            return CACHE[key]

        g.clear = lambda: CACHE.clear()  # type: ignore[attr-defined]
        return g

    return deco(func) if callable(func) else deco


cache_data = cache_resource = cache = _cache


class _Query(dict):
    def to_dict(self) -> dict:
        return dict(self)

    def update(self, *a: Any, **kw: Any) -> None:
        dict.update(self, *a, **kw)


class _ColumnConfig:
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


# ======================================================================== the module itself
MODULE = types.ModuleType("streamlit")
for _name, _obj in list(globals().items()):
    if not _name.startswith("_") and callable(_obj) and _name not in ("Recorder", "StopPage", "RerunPage", "MODULE"):
        setattr(MODULE, _name, _obj)
MODULE.json = json_
MODULE.session_state = _State()
MODULE.query_params = _Query()
MODULE.column_config = _ColumnConfig()
MODULE.sidebar = _Container(sidebar=True)
MODULE.__version__ = "aerosynapse-native"
MODULE.__file__ = __file__


def _missing(name: str) -> Any:  # any other streamlit call: recorded, harmless
    if name.startswith("__"):  # module attributes (__file__, __path__ ...) are not streamlit calls
        raise AttributeError(name)

    def f(*a: Any, **kw: Any) -> Any:
        REC.block("info", text=f"(st.{name}: not shown in Studio)")
        return _Container()

    return f


MODULE.__getattr__ = _missing  # type: ignore[attr-defined]


def install() -> None:
    """Make ``import streamlit`` give this module."""
    sys.modules["streamlit"] = MODULE
    comp = types.ModuleType("streamlit.components")
    v1 = types.ModuleType("streamlit.components.v1")
    v1.html = lambda *a, **kw: None  # type: ignore[attr-defined]
    comp.v1 = v1  # type: ignore[attr-defined]
    sys.modules["streamlit.components"] = comp
    sys.modules["streamlit.components.v1"] = v1


def begin(rec: Recorder, session: dict[str, Any]) -> None:
    """Start a page run: the recorder, and the app's session state (one object for widgets and the page's own use)."""
    global REC, SESSION
    REC = rec
    SESSION = session if isinstance(session, _State) else _State(session)
    MODULE.session_state = SESSION


def session() -> dict[str, Any]:
    return SESSION
