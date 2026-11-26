"""Step 4.1 report wording and the experiment's selection logic."""

from aerosynapse.thermal.experiments.p4_report import arch_table
from aerosynapse.thermal.pinn import spec as S

SPEC = S.load_spec(None, "smoke")


def rec(label, rel, val, seed=0, problem="A_steady", value=100.0, status="ok"):
    return {
        "problem": problem,
        "parameter": "Pe",
        "value": value,
        "label": label,
        "seed": seed,
        "rel_l2": rel,
        "linf": rel,
        "rel_h1": rel,
        "val_loss": val,
        "n_params": 2000,
        "wall_s": 30.0,
        "status": status,
    }


def table(recs, sweep, tmp_path):
    return "\n".join(arch_table(recs, sweep, SPEC, tmp_path / "t.md", "t"))


def test_best_architecture_and_improvement_ratio(tmp_path):
    txt = table([rec("mlp", 6.0, 0.3), rec("fourier", 1e-3, 1e-5), rec("siren", 0.5, 0.1)], None, tmp_path)
    assert (
        "Best: **fourier** at rel L2 1.00e-03" in txt
        and "6000.0x lower" in txt
        and "Meeting the target: fourier" in txt
    )


def test_no_architecture_solving_is_stated(tmp_path):
    txt = table([rec("mlp", 6.0, 0.3), rec("fourier", 2.0, 0.2)], None, tmp_path)
    assert "none — no architecture solved this case" in txt
    assert "baseline MLP is the best" in table([rec("mlp", 1e-3, 1e-5), rec("fourier", 1e-2, 1e-4)], None, tmp_path)


def test_validation_loss_proxy_wording(tmp_path):
    good = [rec(f"a{i}", 10**-i, 10 ** (-2 * i)) for i in range(1, 6)]
    assert "usable truth-free proxy" in table(good, None, tmp_path)
    bad = [
        rec(f"a{i}", e, v)
        for i, (e, v) in enumerate([(1e-4, 1e-1), (1e-1, 1e-5), (1e-2, 1e-3), (1e-3, 1e-2), (1.0, 1e-4)])
    ]
    assert "NOT a reliable proxy" in table(bad, None, tmp_path)


def test_pooled_correlation_but_within_case_misranking_is_flagged(tmp_path):
    # pooled: case A errors/val losses are large, case B small -> high pooled Spearman;
    # within case B: siren has 15x lower error than mlp but a HIGHER validation loss -> discordant pair reported
    recs = [
        rec("mlp", 6.0, 3e-1, problem="A_steady", value=100.0),
        rec("siren", 1e-3, 3e-5, problem="A_steady", value=100.0),
        rec("mlp", 2e-2, 4e-2, problem="B_slab", value=10.0),
        rec("siren", 1.3e-3, 1e-1, problem="B_slab", value=10.0),
        rec("rwf", 3e-2, 5e-2, problem="B_slab", value=10.0),
    ]
    txt = table(recs, None, tmp_path)
    assert "WITHIN a case it ranks architectures correctly for only" in txt
    # discordant pairs in B_slab: (siren, mlp) 15.4x and (siren, rwf) 23.1x -> the largest is reported
    assert "Largest misranking: in B_slab 10, siren has a 23.1x lower error than rwf" in txt


def test_sigma_transfer_failure_is_reported(tmp_path):
    sw = {
        "case": {"problem": "A_steady", "value": 100.0},
        "seed": 0,
        "selected_sigma": 0.5,
        "runs": [
            {"sigma": s, "val_loss": v, "rel_l2": e, "status": "ok"}
            for s, v, e in [(0.5, 1e-5, 1e-4), (1.0, 1e-3, 1e-2)]
        ],
    }
    recs = [
        rec("mlp", 6.0, 0.3, problem="A_steady", value=100.0),
        rec("fourier", 1e-3, 1e-5, problem="A_steady", value=100.0),
        rec("mlp", 2e-2, 4e-2, problem="B_slab", value=10.0),
        rec("fourier", 0.7, 3e2, problem="B_slab", value=10.0),
    ]
    txt = table(recs, sw, tmp_path)
    assert "Does the tuned σ transfer?" in txt and "B_slab 10 (Fourier 7.00e-01 vs MLP 2.00e-02)" in txt
    ok = [r for r in recs if r["label"] != "fourier" or r["problem"] == "A_steady"] + [
        rec("fourier", 1e-3, 1e-5, problem="B_slab", value=10.0)
    ]
    assert "Does the tuned σ transfer?" not in table(ok, sw, tmp_path)


def test_sigma_selection_agreement_and_range_edge(tmp_path):
    sw = {
        "case": {"problem": "A_steady", "value": 100.0},
        "seed": 0,
        "runs": [
            {"sigma": s, "val_loss": v, "rel_l2": e, "status": "ok"}
            for s, v, e in [(0.5, 1e-2, 1e-1), (1.0, 1e-4, 1e-3), (2.0, 1e-3, 1e-4), (4.0, 1e-1, 1.0)]
        ],
    }
    txt = table([rec("mlp", 1e-3, 1e-4)], sw, tmp_path)
    assert "Selected σ = **1**" in txt and "DISAGREE (truth-based minimum at σ = 2)" in txt and "extend" not in txt
    sw["runs"][0]["val_loss"] = 1e-9
    assert "end of the tested range" in table([rec("mlp", 1e-3, 1e-4)], sw, tmp_path)


def test_seed_caveat_only_with_few_seeds(tmp_path):
    assert "1 seed(s)" in table([rec("mlp", 1e-3, 1e-4)], None, tmp_path)
    many = [rec("mlp", 1e-3 * (1 + s), 1e-4, seed=s) for s in range(3)]
    assert "seed(s)" not in table(many, None, tmp_path) and "±" in table(many, None, tmp_path)
