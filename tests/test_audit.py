"""The claims audit must catch a fabricated number (negative control) and accept a real one."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.unit


def _audit():
    spec = importlib.util.spec_from_file_location("audit_claims", ROOT / "scripts" / "audit_claims.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_audit_catches_fabricated_numbers(tmp_path):
    if not (ROOT / "results" / "phase1_verification" / "metrics.json").exists():
        pytest.skip("phase1 results not generated")
    a = _audit()
    doc = tmp_path / "doc.md"
    doc.write_text("<!-- sources: phase1_verification -->\nThe heat solver reached order 7.7777 on 12345.6 cells.\n")
    bad = a.scan(doc, a.collect_values(a.doc_sources(doc)))
    assert len(bad) == 2
    import json

    v = json.loads((ROOT / "results/phase1_verification/metrics.json").read_text())["heat1d_cn"]["fitted_order"]
    doc.write_text(f"<!-- sources: phase1_verification -->\nObserved order {v:.3f}.\n")
    assert a.scan(doc, a.collect_values(a.doc_sources(doc))) == []
