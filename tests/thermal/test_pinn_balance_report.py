"""Step 4.3 report wording: each statement follows from the numbers it is given."""

from aerosynapse.thermal.experiments.p4_report import balance_table
from aerosynapse.thermal.pinn import spec as S

SPEC = S.load_spec(None, "smoke")


def rec(method, err, case="B_slab 10", val=None, gw=10.0, clamped=0, status="ok", wall=10.0, dim=1):
    return {
        "case": case,
        "problem": case.split()[0],
        "dim": dim,
        "method": method,
        "seed": 0,
        "err": err,
        "rel_l2_p": err,
        "val_loss": val if val is not None else err,
        "weight_spread": 3.0,
        "grad_spread_unweighted": 100.0,
        "grad_spread_weighted": gw,
        "n_clamped": clamped,
        "wall_s": wall,
        "status": status,
    }


def table(recs, tmp_path, refs=None):
    return "\n".join(balance_table(recs, refs or {}, SPEC, tmp_path / "t.md", "t"))


def test_effects_relative_to_fixed(tmp_path):
    txt = table(
        [
            rec("fixed", 1e-1, wall=10),
            rec("relobralo", 2e-2, wall=12),
            rec("ntk", 4e-1, wall=30),
            rec("softadapt", 8e-2),
        ],
        tmp_path,
    )
    assert "≥2x lower: relobralo (5.0x)" in txt and "≥2x HIGHER: ntk (4.0x)" in txt
    assert "within 2x (no effect claimed): softadapt" in txt
    assert "| ntk | 4.000e-01 |" in txt and "| 30 | 3.0x |" in txt
    assert "Best: **relobralo** at 2.00e-02" in txt and "none — no method solved" in txt


def test_failures_and_clamps_are_listed(tmp_path):
    txt = table(
        [rec("fixed", 1e-1), rec("gradnorm", float("nan"), status="diverged"), rec("ntk", 1e-1, clamped=7)], tmp_path
    )
    assert "Runs that did not finish normally: gradnorm." in txt
    assert "(number of clamps): ntk (7)" in txt


def test_cross_case_summary_and_uniform_competitiveness(tmp_path):
    recs = [
        rec("fixed", 1e-1, case="A 1"),
        rec("relobralo", 1e-2, case="A 1"),
        rec("ntk", 1.5e-2, case="A 1"),
        rec("fixed", 1e-1, case="B 1"),
        rec("relobralo", 4e-1, case="B 1"),
        rec("ntk", 9e-2, case="B 1"),
    ]
    txt = table(recs, tmp_path)
    assert "| relobralo | 10.00 | 0.25 | 1 | 1 |" in txt
    assert "Within a factor 2 of the best error in EVERY case: ntk." in txt


def test_gradient_balance_statement(tmp_path):
    recs = [rec("fixed", 1e-1, gw=50.0), rec("a", 9e-2, gw=5.0), rec("b", 1e-2, gw=5.0), rec("c", 1e-1, gw=80.0)]
    txt = table(recs, tmp_path)
    assert "below that of fixed weights: 2/3; of these, 1 also have a ≥2x lower error" in txt
    recs2 = [rec("fixed", 1e-1, gw=50.0), rec("a", 1e-2, gw=5.0), rec("b", 1e-2, gw=5.0)]
    assert "went together with a lower error in most" in table(recs2, tmp_path)


def test_validation_agreement_and_reference(tmp_path):
    recs = [rec("fixed", 1e-1, val=1.0, dim=2, case="C_cavity"), rec("x", 1e-2, val=2.0, dim=2, case="C_cavity")]
    refs = {"C_cavity": {"coarse_fine_diff": {"u": 7e-3, "v": 1e-2, "theta": 2e-4, "p": 6e-2}}}
    txt = table(recs, tmp_path, refs)
    assert "0/1 (0%): NOT reliable" in txt and "Reference uncertainty (coarse-fine): u 7.00e-03" in txt
    assert "p error |" in txt and "1 seed(s)" in txt


def test_inactive_methods_and_time_outliers(tmp_path):
    recs = [rec("fixed", 1e-1, wall=300), rec("a", 1e-1, wall=30), rec("b", 1e-1, wall=35), rec("c", 1e-1, wall=40)]
    recs[1]["weight_spread"] = 1.05
    txt = table(recs, tmp_path)
    assert "Final weights within a factor 1.2 of uniform: a —" in txt
    assert "Wall-time outliers (more than 2x from the case median 38 s): fixed (300 s)" in txt
    assert "cost ratios are not given for this case" in txt and "| 30 | —x |" in txt
    ok = [rec("fixed", 1e-1, wall=30), rec("a", 1e-1, wall=33)]
    assert "Wall-time outliers" not in table(ok, tmp_path) and "| 33 | 1.1x |" in table(ok, tmp_path)
