"""Phase 7 Step 7.4: smoke tests for every figure function, on the inputs the real runs gave them.

    python -m aerosynapse.thermal.viz.capture <out_dir> <label> <module> [script args ...]

imports experiments.<module> (its __main__ block does not run), wraps every figure function of the project (see
figure_functions: the public plotting functions of src/viz and every function of a script that draws a figure)
and calls the script's main() with the given arguments. The arguments of each call are pickled to
<out_dir>/blobs/<sha>.pkl.gz (content-addressed: functions called on the same inputs share one file) and listed in
<out_dir>/<label>.capture.json (at most MAX_CALLS per function and script); paths are stored relative to the
repository, Experiment objects as stubs and unpicklable objects as Uncaptured placeholders (recorded), so a fixture
replays on any machine. save_fig writes into a
scratch folder, never into docs/figures. No network is trained: the commands are the saved-results re-render
commands of Step 7.3.

replay(entry, blob, out) calls the function on the stored arguments with save_fig redirected to `out` (used by
tests/unit/test_figure_smoke.py).
"""

from __future__ import annotations

import ast
import functools
import gzip
import hashlib
import importlib
import io
import pickle

from aerosynapse.thermal import _compat
import sys
import types
from pathlib import Path, PurePath

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
MAX_CALLS = 3
# src/viz modules that hold infrastructure, not figures
NOT_FIGURES = {"__init__", "style", "catalogue", "gallery", "capture"}
PLOT_CALLS = {"subplots", "figure", "subplot_mosaic", "save_fig"}


def _is_figure(fn: ast.FunctionDef) -> bool:
    """Creates or saves a figure, or is annotated as returning one."""
    ret = ast.unparse(fn.returns) if fn.returns is not None else ""
    return bool(_calls(fn) & PLOT_CALLS) or ret.endswith("Figure")


def _calls(fn: ast.FunctionDef) -> set[str]:
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def figure_functions() -> dict[str, list[str]]:
    """{module: [function, ...]}: public top-level functions of src/viz that create a figure, and functions of the
    experiment scripts that create or save one (main() excluded: scripts that draw inside main() are listed by
    inline_figures)."""
    out: dict[str, list[str]] = {}
    for f in sorted((__import__("aerosynapse.thermal._home", fromlist=["HOME"]).CODE / "viz").glob("*.py")):
        if f.stem in NOT_FIGURES:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        names = [
            n.name for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_") and _is_figure(n)
        ]
        if names:
            out[f"aerosynapse.thermal.viz.{f.stem}"] = names
    for f in sorted((__import__("aerosynapse.thermal._home", fromlist=["HOME"]).CODE / "experiments").glob("p*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "main" and _is_figure(n)]
        if names:
            out[f"aerosynapse.thermal.experiments.{f.stem}"] = names
    return out


def inline_figures() -> list[str]:
    """Scripts whose main() itself draws a figure (figure code not in a separate function)."""
    out = []
    for f in sorted((__import__("aerosynapse.thermal._home", fromlist=["HOME"]).CODE / "experiments").glob("p*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        own = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}  # e.g. a script's own figure()
        if any(
            isinstance(n, ast.FunctionDef) and n.name == "main" and _calls(n) & (PLOT_CALLS - {"save_fig"} - own)
            for n in tree.body
        ):
            out.append(f"aerosynapse.thermal.experiments.{f.stem}")
    return out


# ---------------------------------------------------------------- portable pickling
def _repo_path(rel: str, absolute: str) -> Path:
    return ROOT / rel if rel else Path(absolute)


class ExperimentStub:
    """What a figure function may use of an Experiment: dir (replay() points it at its scratch folder), exp_id,
    config, metrics, log() (rows kept in memory) and fig_path()."""

    def __init__(self, exp_id: str, config: dict | None = None):
        self.exp_id, self.config, self.dir, self.metrics, self.rows = exp_id, config or {}, None, {}, []

    def log(self, **row) -> None:
        self.rows.append(row)

    def fig_path(self, name: str) -> Path:
        return Path(self.dir) / name


def _experiment_stub(exp_id: str, config: dict | None = None) -> ExperimentStub:
    return ExperimentStub(exp_id, config)


def _rebuild(cls, state: dict):
    obj = object.__new__(cls)
    obj.__dict__.update(state)
    return obj


class Uncaptured:
    """Stands in for an argument that cannot be pickled (a sparse LU factorisation, a local lambda); using it in a
    replay raises, so a figure function that needs it fails visibly instead of drawing something else."""

    def __init__(self, what: str):
        self.what = what

    def __getattr__(self, name):
        if name.startswith("__") or name == "what":
            raise AttributeError(name)
        raise RuntimeError(f"argument not captured: {self.what}")

    def __call__(self, *a, **k):
        raise RuntimeError(f"argument not captured: {self.what}")


PLAIN = (dict, list, tuple, set, frozenset, str, bytes, int, float, complex, bool, type(None))


class _Pickler(pickle.Pickler):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.substituted: list[str] = []

    def reducer_override(self, obj):
        if isinstance(obj, PurePath):
            p = Path(obj)
            try:
                return _repo_path, (p.resolve().relative_to(ROOT).as_posix(), "")
            except ValueError:
                return _repo_path, ("", str(p))
        if type(obj).__name__ == "Experiment" and hasattr(obj, "exp_id"):
            return _experiment_stub, (obj.exp_id, getattr(obj, "config", None))
        if isinstance(obj, type) and "<locals>" in obj.__qualname__ and hasattr(obj, "dir"):
            return _experiment_stub, (obj.__qualname__,)  # a script's local stand-in for an Experiment
        if isinstance(obj, PLAIN) or type(obj).__module__.split(".")[0] in ("numpy", "torch"):
            return NotImplemented
        try:
            pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001 - substituted and recorded
            mod = getattr(obj, "__module__", None) or type(obj).__module__
            what = f"{mod}.{getattr(obj, '__qualname__', type(obj).__qualname__)}"
            if not isinstance(obj, types.FunctionType | types.BuiltinFunctionType | type) and hasattr(obj, "__dict__"):
                # keep the object with its picklable state; only the attributes that cannot be pickled (a sparse LU
                # factorisation, a lambda) become placeholders, recorded when the pickler reaches them
                return _rebuild, (type(obj), dict(obj.__dict__))
            self.substituted.append(what)
            return Uncaptured, (what,)
        return NotImplemented


def dumps(obj, substituted: list | None = None) -> bytes:
    buf = io.BytesIO()
    pk = _Pickler(buf, protocol=pickle.HIGHEST_PROTOCOL)
    pk.dump(obj)
    if substituted is not None:
        substituted += pk.substituted
    return buf.getvalue()


# ---------------------------------------------------------------- capture
def _install(out: Path, label: str, script: str) -> dict:
    """Wrap every figure function. Arguments are stored content-addressed in out/blobs/<sha>.pkl.gz (functions
    called on the same inputs share one blob); the log maps each fixture to its blob."""
    log: dict = {"fixtures": {}, "failed": []}
    counts: dict[str, int] = {}
    (out / "blobs").mkdir(parents=True, exist_ok=True)
    for mod_name, names in figure_functions().items():
        mod = importlib.import_module(mod_name)
        for name in names:
            orig = getattr(mod, name)

            @functools.wraps(orig)
            def wrapper(*args, __orig=orig, __key=f"{mod_name}.{name}", **kwargs):
                n = counts.get(__key, 0)
                if n < MAX_CALLS:
                    counts[__key] = n + 1
                    stem = f"{label}.{__key}.{n}"
                    try:
                        subst: list[str] = []
                        # plain-data module globals the function reads (a script's profile set by main(), e.g.)
                        g = __orig.__globals__
                        glob = {
                            k: g[k]
                            for k in __orig.__code__.co_names
                            if k in g and not k.startswith("__") and isinstance(g[k], PLAIN + (PurePath,))
                        }
                        raw = dumps({"args": args, "kwargs": kwargs, "globals": glob}, subst)
                        sha = hashlib.sha256(raw).hexdigest()[:20]
                        f = out / "blobs" / f"{sha}.pkl.gz"
                        if not f.exists():
                            f.write_bytes(gzip.compress(raw, 6, mtime=0))
                        mod_, fn_ = __key.rsplit(".", 1)
                        log["fixtures"][stem] = {
                            "module": mod_,
                            "function": fn_,
                            "script": script,
                            "blob": sha,
                            "uncaptured": sorted(set(subst)),
                        }
                    except Exception as e:  # noqa: BLE001 - recorded and reported, never hidden
                        log["failed"].append([stem, f"{type(e).__name__}: {e}"[:200]])
                return __orig(*args, **kwargs)

            setattr(mod, name, wrapper)
    return log


def _redirect_save_fig(folder: Path, dpi: float | None = None):
    """Point every loaded module's save_fig at `folder` (docs/figures and results folders are never written)."""
    from aerosynapse.thermal.viz import style

    orig = style.save_fig
    written: list[Path] = []

    def save_fig(fig, name, also_to=None, docs=True):
        style.tidy_log_ticks(fig)
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"{name}.png"
        fig.savefig(p, bbox_inches="tight", **({"dpi": dpi} if dpi else {}))
        written.append(p)
        style.plt.close(fig)
        return [p]

    for m in list(sys.modules.values()):
        if getattr(m, "save_fig", None) is orig:
            m.save_fig = save_fig
    return written, lambda: [
        setattr(m, "save_fig", orig) for m in list(sys.modules.values()) if getattr(m, "save_fig", None) is save_fig
    ]


def capture(out: Path, label: str, script: str, argv: list[str]) -> dict:
    import json

    import matplotlib

    matplotlib.use("Agg")
    out.mkdir(parents=True, exist_ok=True)
    log = _install(out, label, script)
    mod = importlib.import_module(f"aerosynapse.thermal.experiments.{script}")
    _redirect_save_fig(out / "_figures" / label, dpi=72)
    sys.argv = [f"experiments/{script}.py", *argv]
    mod.main()
    (out / f"{label}.capture.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return log


# ---------------------------------------------------------------- replay
def load_bytes(blob: bytes) -> dict:
    return _compat.loads(blob)


def load(blob: Path) -> dict:
    return load_bytes(gzip.decompress(Path(blob).read_bytes()))


def replay(entry: dict, blob: Path, out: Path, dpi: float = 40) -> list[Path]:
    """Call entry's function (module / function of a manifest entry) on the arguments stored in `blob`; every figure
    goes to `out`. Returns the PNGs written."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    rec = load(blob)

    def scratch(o):
        if isinstance(o, ExperimentStub) or (isinstance(o, type) and hasattr(o, "dir")):
            o.dir = out
        return o

    args = [scratch(a) for a in rec["args"]]
    kwargs = {k: scratch(v) for k, v in rec["kwargs"].items()}
    mod = importlib.import_module(_compat.modname(entry["module"]))
    fn = getattr(mod, entry["function"])
    fn = getattr(fn, "__wrapped__", fn)
    written, restore = _redirect_save_fig(out, dpi=dpi)
    glob = rec.get("globals", {})
    saved = {k: mod.__dict__[k] for k in glob if k in mod.__dict__}
    mod.__dict__.update(glob)
    try:
        res = fn(*args, **kwargs)
        if isinstance(res, Figure):
            assert res.axes, "figure without axes"
            p = out / f"{entry['function']}.png"
            res.savefig(p, dpi=dpi)
            written.append(p)
            matplotlib.pyplot.close(res)
    finally:
        restore()
        for k in glob:
            if k in saved:
                mod.__dict__[k] = saved[k]
            else:
                mod.__dict__.pop(k, None)
    return written


if __name__ == "__main__":
    # through the importable module, so the pickled reducers name aerosynapse.thermal.viz.capture, not __main__
    from aerosynapse.thermal.viz import capture as _module

    _module.capture(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:])
