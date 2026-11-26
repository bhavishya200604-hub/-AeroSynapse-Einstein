"""Figure catalogue (src/viz/catalogue.py, Phase 7 Step 7.2)."""

import re

from aerosynapse.thermal.viz import catalogue as C


def test_regex_captures_tag_and_other_placeholders():
    rx = C._regex("p4_2_fields_{case}_{tag}")
    m = re.match(rx, "p4_2_fields_C_cavity_smoke")
    assert m and m.group("tag")  # matched; the ambiguous split is resolved against results folders in build()
    assert re.match(C._regex("p6_5_spectra_{tag}"), "p6_5_spectra_default").group("tag") == "default"
    assert re.match(C._regex("p4_5b_flow_rates"), "p4_5b_flow_rates")
    assert not re.match(C._regex("p6_5_spectra_{tag}"), "p6_5_spectral3d_default".replace("spectral3d", "x"))


def test_every_figure_is_catalogued_with_a_valid_command():
    entries = C.build()
    names = {p.stem for p in C.FIG.glob("*.png")} | {p.stem for p in C.FIG.glob("*.pdf")}
    assert {e.name for e in entries} == names
    assert all(e.script for e in entries), [e.name for e in entries if not e.script]
    for e in entries:
        src = (C.ROOT / e.script).read_text(encoding="utf-8")
        for opt in re.findall(r"(--[a-z-]+)", e.rerender or ""):
            assert f'"{opt}"' in src or (opt == "--profile" and "cli_profile(" in src), (e.name, opt)


def test_ambiguous_tags_resolve_to_existing_results():
    by = {e.name: e for e in C.build()}
    e = by.get("p4_2_fields_C_cavity_smoke")
    if e is not None:
        assert e.tag == "smoke" and e.results_present
    e = by.get("p4_5b_fields_D_graetz_hard_psi")
    if e is not None:
        assert e.tag == "hard_psi" and e.results_present
