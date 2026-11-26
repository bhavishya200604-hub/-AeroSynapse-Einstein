"""Phase 3 report wording: diagnoses must follow the measurements; user parameter blocks replace defaults."""

import csv

from aerosynapse.thermal.experiments.p3_report import failure_table
from aerosynapse.thermal.pinn import spec as S


def rec(problem, value, rel, linf=0.1, **kw):
    base = {
        "problem": problem,
        "parameter": "Pe",
        "value": value,
        "rel_l2": rel,
        "linf": linf,
        "rel_h1": rel,
        "meets_target": rel < 0.01,
        "wall_s": 30.0,
        "grad_imbalance_max_over_min": 3.0,
        "final_losses": {"total": 0.3, "pde": 0.1, "bc_left": 0.1, "bc_right": 0.1},
        "reference_solvers": {"fd": {"rel_l2": 1e-3, "wall_s": 0.1}, "spectral": {"rel_l2": 1e-9, "wall_s": 0.5}},
        "spectrum": {
            "k_cut": 20.0,
            "error_high_band_fraction": 0.01,
            "reference_high_band_fraction": 0.0,
            "error_to_reference_ratio_high": 1.0,
            "error_to_reference_ratio_low": 0.01,
        },
    }
    base.update(kw)
    return base


def table(recs, tmp_path, runs_dir=None):
    spec = S.load_spec(None, "smoke")
    return "\n".join(failure_table(recs, {}, spec, tmp_path / "t.md", "t", runs_dir))


def test_error_decreasing_with_parameter_is_not_called_a_boundary_layer(tmp_path):
    txt = table([rec("A_pulse", v, e) for v, e in [(1, 0.11), (10, 0.07), (100, 0.02)]], tmp_path)
    assert "DECREASES with the parameter" in txt and "consistent with sharper solution features" not in txt


def test_error_increasing_is_reported_as_consistent_not_proven(tmp_path):
    txt = table([rec("A_steady", v, e) for v, e in [(1, 1e-4), (10, 1e-3), (100, 1e-2)]], tmp_path)
    assert "consistent with sharper solution features" in txt and "not proven" in txt


def test_spectral_bias_wording_follows_error_band_share(tmp_path):
    low = table([rec("A_steady", v, 0.1 * v) for v in (1, 2, 3)], tmp_path)
    assert "NOT supported as the dominant failure" in low
    hi = {
        "k_cut": 20.0,
        "error_high_band_fraction": 0.6,
        "reference_high_band_fraction": 0.3,
        "error_to_reference_ratio_high": 0.9,
        "error_to_reference_ratio_low": 0.1,
    }
    assert "supported: a median 0.60" in table([rec("A_steady", v, 0.1 * v, spectrum=hi) for v in (1, 2, 3)], tmp_path)


def test_wrong_solution_uses_logged_loss_reduction(tmp_path):
    d = tmp_path / "runs" / "A_steady_100"
    d.mkdir(parents=True)
    with open(d / "log.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase", "step", "total"])
        w.writerows([["adam", 0, 1.1], ["lbfgs", 9, 0.33]])
    recs = [rec("A_steady", 100, 6.7), rec("A_steady", 1, 1e-4), rec("A_steady", 10, 1e-4)]
    txt = table(recs, tmp_path, tmp_path / "runs")
    assert (
        "optimiser stalled" in txt
        and "×3.3" in txt
        and "an optimisation failure, not a lack of representability" in txt
    )


def test_underresolved_reference_does_not_claim_solvability(tmp_path):
    d = tmp_path / "runs" / "A_steady_1000"
    d.mkdir(parents=True)
    with open(d / "log.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase", "step", "total"])
        w.writerows([["adam", 0, 1.1], ["lbfgs", 9, 0.33]])
    bad = rec(
        "A_steady",
        1000,
        14.9,
        reference_solvers={"fd": {"rel_l2": 0.7, "wall_s": 0.1}, "spectral": {"rel_l2": 0.81, "wall_s": 0.1}},
    )
    txt = table([bad, rec("A_steady", 1, 1e-4), rec("A_steady", 10, 1e-4)], tmp_path, tmp_path / "runs")
    assert "under-resolved" in txt and "an optimisation failure, not a lack of representability" not in txt


def test_user_parameter_block_replaces_default_values():
    s = S.load_spec(None, "smoke", S.parse_set(["problems.A_steady.parameter={name: Pe, lo: 1, hi: 500, n: 3}"]))
    pc = s["problems"]["A_steady"]["parameter"]
    assert "values" not in pc and len(S.param_values(pc)) == 3 and abs(S.param_values(pc)[-1] - 500) < 1e-9
    assert s["problems"]["A_steady"]["conditions"]["theta_right"] == 1.0  # only 'parameter' replaces; the rest merges
