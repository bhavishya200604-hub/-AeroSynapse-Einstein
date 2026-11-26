"""Gap run G2 (experiments/p6b_g2_conflict.py): config defaults, pre-registered verdict logic, report."""

from aerosynapse.thermal.experiments import p6b_g2_conflict as G


def _row1(variant, seed, rel, cos):
    c = {"cosine": cos, "cancellation": 0.5, "grad_pde": 1.0, "grad_cond": 1.0, "loss_pde": 0.1, "loss_cond": 0.1}
    return {"variant": variant, "seed": seed, "rel_l2": rel, "status": "ok", "init": c, "final": c}


def _row2(case, seed, err, cos):
    c = {"cosine": cos, "cancellation": 0.5, "grad_pde": 1.0, "grad_bc": 1.0}
    return {"case": case, "select": "x", "run": "r", "seed": seed, "error": err, "init": c, "final": c}


def test_default_config_is_the_preregistered_design():
    cfg = G.load_cfg([], [])
    assert cfg["oned"]["variants"] == ["mlp", "fourier", "siren"] and cfg["oned"]["seeds"] == [0, 1, 2, 3, 4]
    assert cfg["oned"]["fail_threshold"] == 0.1 and cfg["twod"]["points"]["seed"] == 21
    a = cfg["analysis"]
    assert (a["alpha"], a["n_boot"], a["fail_cos_max"], a["success_cos_min"]) == (0.05, 10000, -0.5, -0.3)
    assert G.load_cfg([], ["oned.seeds=[0, 1]"])["oned"]["seeds"] == [0, 1]


def test_supported_when_all_three_conditions_hold():
    cfg = G.load_cfg([], [])
    rows1 = [_row1("mlp", s, 6.0 + s, -0.99 + 0.001 * s) for s in range(5)]
    rows1 += [_row1("siren", s, 1e-3, 0.3 + 0.05 * s) for s in range(5)]
    rows2 = [_row2("C_cavity", s, 0.8, -0.9 + 0.02 * s) for s in range(5)]
    rows2 += [_row2("D_graetz", s, 0.1, 0.0 + 0.03 * s) for s in range(5)]
    an = G.analyse(rows1, rows2, cfg)
    assert an["settings"]["1d"]["verdict"] == "SUPPORTED"
    assert an["settings"]["2d"]["verdict"] == "SUPPORTED"
    assert an["settings"]["1d"]["spearman_cos_error"]["rho"] < 0


def test_not_supported_when_success_cosine_too_low():
    cfg = G.load_cfg([], [])
    rows1 = [_row1("mlp", s, 6.0, -0.99 - 0.001 * s) for s in range(5)]
    rows1 += [_row1("fourier", s, 1e-3, -0.33 + 0.001 * s) for s in range(5)]  # mean below -0.3
    e = G.analyse(rows1, [], cfg)["settings"]["1d"]
    assert e["welch"]["p"] < 0.05 and e["verdict"] == "NOT SUPPORTED"


def test_empty_group_gives_no_test(tmp_path):
    cfg = G.load_cfg([], [])
    rows1 = [_row1("siren", s, 1e-3, 0.3) for s in range(3)]
    an = G.analyse(rows1, [], cfg)
    assert an["settings"]["1d"]["welch"] is None and an["settings"]["1d"]["verdict"] == "NOT SUPPORTED"
    met = {"oned": rows1, "twod": [], "analysis": an, "reproduction": None}
    G.report(met, tmp_path / "g2.md", "t")
    assert "no test" in (tmp_path / "g2.md").read_text(encoding="utf-8")
