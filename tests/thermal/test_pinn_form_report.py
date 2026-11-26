"""Step 4.2 report wording: every statement follows from the numbers it is given."""

from aerosynapse.thermal.experiments.p4_report import formulation_table
from aerosynapse.thermal.pinn import spec as S

SPEC = S.load_spec(None, "smoke")
REF = {"C_mms": {"kind": "exact manufactured solution"}}


def rec(variant, err, p=None, case="C_mms", kind="primitive", hard=False, div=1e-2, val=None, seed=0, **kw):
    r = {
        "case": case,
        "variant": variant,
        "seed": seed,
        "kind": kind,
        "hard_bc": hard,
        "gauge": "none",
        "normalise": "none",
        "rel_l2_max": err,
        "rel_l2_u": err,
        "rel_l2_v": err,
        "rel_l2_theta": err,
        "rel_l2_p": p if p is not None else err,
        "max_div": div,
        "val_loss": val if val is not None else err,
        "bc_max_normal": 1e-3,
        "bc_max_tangential": 1e-3,
        "bc_max_theta_dirichlet": 1e-3,
        "bc_max_theta_other": 1e-3,
        "n_params": 2000,
        "wall_s": 100.0,
        "status": "ok",
    }
    r.update(kw)
    return r


def table(recs, tmp_path, refs=REF):
    return "\n".join(formulation_table(recs, refs, SPEC, tmp_path / "t.md", "t"))


def test_effect_wording_and_identity(tmp_path):
    txt = table(
        [
            rec("primitive", 1e-1),
            rec("prim_gauge_point", 1e-1),
            rec("prim_hard", 2.5e-2),
            rec("prim_norm", 3e-1),
            rec("psi", 0.8e-1, kind="streamfunction", div=1e-14),
        ],
        tmp_path,
    )
    assert "| prim_hard | 4.0x lower |" in txt
    assert "| prim_norm | 3.0x HIGHER |" in txt
    assert "| psi | within a factor 2 (no effect claimed) |" in txt
    assert "| prim_gauge_point | identical (to round-off) |" in txt
    assert "(to round-off): prim_gauge_point." in txt
    assert "Best (worst-field error): **prim_hard** at 2.50e-02" in txt and "none — no formulation solved" in txt


def test_divergence_and_hard_constraint_wording(tmp_path):
    ok = table(
        [
            rec("primitive", 1e-1),
            rec(
                "psi_hard",
                1e-1,
                kind="streamfunction",
                hard=True,
                div=1e-14,
                bc_max_normal=1e-15,
                bc_max_theta_dirichlet=1e-15,
            ),
        ],
        tmp_path,
    )
    assert "divergence-free to round-off, as constructed" in ok and "exact to round-off" in ok
    bad = table(
        [rec("primitive", 1e-1), rec("psi_hard", 1e-1, kind="streamfunction", hard=True, div=1e-6, bc_max_normal=1e-4)],
        tmp_path,
    )
    assert "NOT at round-off level" in bad and "NOT at round-off: investigate" in bad


def test_reference_uncertainty_wording(tmp_path):
    refs = {
        "C_cavity": {
            "kind": "Newton",
            "n_coarse": 64,
            "n_fine": 128,
            "newton": {"fine": {"newton_residual": 1e-11}},
            "coarse_fine_diff": {"u": 1e-2, "v": 1e-2, "theta": 1e-4, "p": 6e-2, "nu_top": 1e-4},
        }
    }
    below = table([rec("primitive", 5e-3, p=1e-1, case="C_cavity")], tmp_path, refs)
    assert "not resolvable): u (5.00e-03 < 1.00e-02), v (5.00e-03 < 1.00e-02)" in below
    above = table([rec("primitive", 5e-1, case="C_cavity")], tmp_path, refs)
    assert "they are PINN errors, not reference errors" in above


def test_consistency_and_validation_wording(tmp_path):
    recs = [
        rec("primitive", 1e-1, val=1e-1),
        rec("psi", 1e-2, val=1e-2, kind="streamfunction", div=1e-14),
        rec("primitive", 1e-2, case="C_cavity", val=1e-1),
        rec("psi", 1e-1, case="C_cavity", val=1e-2, kind="streamfunction", div=1e-14),
    ]
    refs = {"C_mms": REF["C_mms"], "C_cavity": {"kind": "x"}}
    txt = table(recs, tmp_path, refs)
    assert "The nominally best variant differs between cases: C_mms → psi, C_cavity → primitive" in txt
    assert "No variant is within a factor 2 of the best in every case" in txt
    close = recs + [
        rec("prim_hard", 1.5e-2, val=1e-2),
        rec("prim_hard", 1.8e-2, case="C_cavity", val=1e-2),
    ]
    assert "of the best worst-field error in EVERY case: prim_hard." in table(close, tmp_path, refs)
    assert "1/2 (50%): it is NOT reliable" in txt and "in C_cavity, primitive has a 10.0x lower error than psi" in txt
    assert "1 seed(s)" in txt
