"""Phase 7 Step 7.4: every figure function, called alone on the inputs a real run gave it.

Fixtures are captured by `python -m aerosynapse.thermal.experiments.p7_4_smoke --capture` (src/viz/capture.py). A replay that needs a
git-ignored artefact (network checkpoint, cached array, data/) is skipped with the missing path; large fixtures live
in the git-ignored tests/fixtures/figures_large/ and are skipped where absent (a clean clone).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from aerosynapse.thermal.viz.capture import ROOT, figure_functions, inline_figures, replay

FIX = ROOT / "tests" / "fixtures" / "figures"
FIX_LARGE = ROOT / "tests" / "fixtures" / "figures_large"
MANIFEST = json.loads((FIX / "MANIFEST.json").read_text(encoding="utf-8")) if (FIX / "MANIFEST.json").exists() else {}


def _ignored(path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


# replays measured above SLOW_S by aerosynapse.thermal.experiments.p7_4_smoke run only with -m slow (the driver runs all)
SLOW_S = 5.0


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=pytest.mark.slow) if MANIFEST[n].get("replay_s", 0) > SLOW_S else n
        for n in sorted(MANIFEST)
    ],
)
def test_figure_function(name, tmp_path):
    e = MANIFEST[name]
    blob = (FIX if e["committed"] else FIX_LARGE) / "blobs" / f"{e['blob']}.pkl.gz"
    if not blob.exists():
        pytest.skip(f"large fixture not present (git-ignored): {blob.relative_to(ROOT).as_posix()}")
    try:
        written = replay(e, blob, tmp_path)
    except (FileNotFoundError, OSError) as err:
        missing = getattr(err, "filename", None) or ""
        if missing and _ignored(str(missing)):
            pytest.skip(f"needs git-ignored artefact {missing}")
        raise
    assert written, "the function drew no figure"
    for p in written:
        assert p.exists() and p.stat().st_size > 0


def test_every_figure_function_has_a_fixture():
    covered = {f"{e['module']}.{e['function']}" for e in MANIFEST.values()}
    funcs = {f"{m}.{n}" for m, ns in figure_functions().items() for n in ns}
    assert not funcs - covered, f"figure functions without a fixture: {sorted(funcs - covered)}"
    assert not inline_figures(), "figure code inside main(): move it into a function"


def test_fixture_pickling_is_portable(tmp_path):
    from aerosynapse.thermal.utils.experiment import Experiment
    from aerosynapse.thermal.viz import capture

    ex = object.__new__(Experiment)
    ex.exp_id, ex.dir, ex.config = "x_test", ROOT / "results" / "x_test", {"N": [1, 2]}
    rec = capture.load_bytes(capture.dumps({"p": ROOT / "results" / "a.json", "ex": ex, "outside": tmp_path}))
    assert rec["p"] == ROOT / "results" / "a.json"  # stored relative to the repository
    assert isinstance(rec["ex"], capture.ExperimentStub) and rec["ex"].exp_id == "x_test" and rec["ex"].dir is None
    assert rec["ex"].config == {"N": [1, 2]}
    assert rec["outside"] == tmp_path


def test_figure_function_discovery():
    ff = figure_functions()
    assert "amortisation" in ff["aerosynapse.thermal.viz.cost_figs"]  # annotated -> plt.Figure
    assert ff["aerosynapse.thermal.experiments.p1_1_physics"] == ["figures"]  # moved out of main() in Step 7.4
    assert all(not n.startswith("_") for ns in ff.values() for n in ns)


def test_unpicklable_attributes_become_placeholders():
    from aerosynapse.thermal.viz import capture

    obj = capture.ExperimentStub("x")
    obj.solver = lambda: 1  # a lambda cannot be pickled
    rec = capture.load_bytes(capture.dumps({"o": obj}))
    assert isinstance(rec["o"], capture.ExperimentStub)
    with pytest.raises(RuntimeError, match="not captured"):
        rec["o"].solver()
