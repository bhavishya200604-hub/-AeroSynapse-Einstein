"""Phase 7.5 / 9 tools: the claims audit (registry id resolution, number tracing), the figure gallery, the reproduce
plan and the history replay used to re-draw figures without training."""

import json
import subprocess
import sys
from pathlib import Path

from aerosynapse.thermal.experiments import p9_claims_audit as A
from aerosynapse.thermal.utils import replay as RP

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
REG = {"R6.3a": "inside 0.0024 (0.0015-0.012)", "R6.3b": "inside 0.0077", "R1.7a": "68/69 PASS", "R1.7b": "1.98"}


def test_resolve_exact_range_and_prefix():
    assert A.resolve("R6.3a", REG) == ["R6.3a"]
    assert A.resolve("R6.3a-b", REG) == ["R6.3a", "R6.3b"]
    assert A.resolve("R1.7", REG) == ["R1.7a", "R1.7b"]
    assert A.resolve("R9.9z", REG) == []


def test_in_rows_verbatim_rounded_and_missing():
    assert A.in_rows(0.0024, "0.0024", REG["R6.3a"]) == "verbatim"
    assert A.in_rows(0.002, "0.002", "error 0.00201") == "rounded"
    assert A.in_rows(0.5, "0.5", REG["R6.3a"]) == ""


def test_markdown_audit_flags_untraced_numbers(tmp_path, monkeypatch):
    doc = tmp_path / "doc.md"
    doc.write_text(
        "# Title with Phase 7\n\n- error 0.0024 inside the range [R6.3a]\n- a number from nowhere: 0.4242\n"
        "- wrapped over two lines: 0.0077\n  [R6.3b]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(A, "ROOT", tmp_path)
    r = A.audit_markdown("doc.md", REG)
    assert [c["number"] for c in r["untraced"]] == ["0.4242"]
    assert r["traced"] == 2 and not r["pass"]


def test_registry_rows_carry_their_section_setup():
    reg = A.registry()
    assert "R6.2a" in reg and "[0.3, 3]" in reg["R6.2a"]  # Step 6.2 'Setup' names the training range


def test_gallery_lists_every_catalogued_figure():
    from aerosynapse.thermal.viz import gallery

    out = gallery.build()
    cat = json.loads((ROOT / "docs/figures/catalogue.json").read_text(encoding="utf-8"))
    assert out["figures"] == len(cat)
    page = (ROOT / "docs/figures/index.html").read_text(encoding="utf-8")
    assert all(f'alt="{f["name"]}"' in page for f in cat)


def test_reproduce_dry_run_lists_steps():
    r = subprocess.run(
        [sys.executable, "-m", "aerosynapse.thermal.experiments.p9_reproduce", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0 and "claims audit" in r.stdout and "paper1: make_numbers" in r.stdout


def test_history_replay_matches_the_trainer_rows(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("phase,step,total,rel_l2\nadam,0,2.5,\nadam,50,0.5,0.1\nlbfgs,60,0.25,\n", encoding="utf-8")
    h = RP.history(log)
    assert h[0] == {"phase": "adam", "step": 0.0, "total": 2.5}  # empty cell -> key absent, as in memory
    assert h[1]["rel_l2"] == 0.1 and h[2]["phase"] == "lbfgs"
