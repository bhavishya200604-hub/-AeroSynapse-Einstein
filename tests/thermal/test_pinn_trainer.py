"""Step 3.3: trainer — convergence on an easy problem, L-BFGS gain, checkpoint/resume exactness, logging, divergence."""

import numpy as np
import pytest
import torch

from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import spec as S
from aerosynapse.thermal.pinn.metrics import evaluate_model
from aerosynapse.thermal.pinn.networks import MLP
from aerosynapse.thermal.pinn.trainer import Trainer, component_grad_norms, lr_at

torch.set_default_dtype(torch.float64)


def small_spec(**over):
    s = S.load_spec(None, "smoke", S.parse_set([f"{k}={v}" for k, v in over.items()]))
    s["sampling"].update({"n_pde": 128, "n_ic": 32, "n_bc": 16})
    return s


def make(spec, name="A_steady", val=1.0, seed=0):
    prob = P.build_problem(name, spec["problems"][name], val)
    torch.manual_seed(seed)
    net = MLP(prob.lo, prob.hi, spec["network"]["width"], spec["network"]["depth"])
    return prob, net


def test_scientific_notation_is_accepted_in_files_and_set(tmp_path):
    """Regression: PyYAML reads 3e-3 / 1e+30 as strings; the spec reader must coerce them to floats."""
    f = tmp_path / "u.yaml"
    f.write_text("training:\n  adam: {lr: 3e-3, lr_min: 1.0e-5}\n", encoding="utf-8")
    s = S.load_spec(f, "smoke", S.parse_set(["training.lbfgs.lr=1e+0"]))
    assert s["training"]["adam"]["lr"] == 3e-3 and s["training"]["adam"]["lr_min"] == 1e-5
    assert s["training"]["lbfgs"]["lr"] == 1.0 and isinstance(s["training"]["lbfgs"]["lr"], float)


def test_lr_schedules():
    a = {"epochs": 100, "lr": 1e-2, "lr_min": 1e-4, "schedule": "cosine"}
    assert (
        lr_at(a, 0) == pytest.approx(1e-2)
        and lr_at(a, 100) == pytest.approx(1e-4)
        and lr_at(a, 50) == pytest.approx(5.05e-3)
    )
    assert lr_at({**a, "schedule": "constant"}, 77) == 1e-2
    assert lr_at({**a, "schedule": "exp"}, 100) == pytest.approx(1e-4) and lr_at(
        {**a, "schedule": "exp"}, 50
    ) == pytest.approx(1e-3)


def test_adam_then_lbfgs_converges_and_lbfgs_helps(tmp_path):
    spec = small_spec(
        **{
            "training.adam.epochs": 500,
            "training.lbfgs.iterations": 100,
            "training.log_every": 100,
            "training.eval_every": 100,
            "training.lr": 1,
        }
    )
    prob, net = make(spec)
    ev = lambda m: {k: v for k, v in evaluate_model(m, prob, 1.0, 201, 2).items() if not k.startswith("_")}  # noqa: E731
    tr = Trainer(net, prob, spec, tmp_path, eval_fn=ev)
    out = tr.run()
    assert out["status"] == "ok"
    rows = tr.history
    adam_rows = [r for r in rows if r["phase"] == "adam"]
    lb_rows = [r for r in rows if r["phase"] == "lbfgs"]
    assert adam_rows[-1]["total"] < 0.01 * adam_rows[0]["total"]
    assert lb_rows[-1]["total"] < adam_rows[-1]["total"]  # L-BFGS refines the Adam solution
    assert lb_rows[-1]["rel_l2"] < 5e-3  # Pe = 1 steady boundary-layer-free problem is easy
    for key in ("pde", "bc_left", "bc_right", "total", "lr", "wall_s", "phase", "step"):
        assert key in rows[0]
    assert (tmp_path / "log.csv").exists() and (tmp_path / "checkpoint.pt").exists()


def test_resume_reproduces_uninterrupted_run_exactly(tmp_path):
    over = {
        "training.adam.epochs": 60,
        "training.lbfgs.iterations": 0,
        "training.log_every": 10,
        "training.eval_every": 1000,
        "sampling.resample_every": 7,
        "training.checkpoint_every": 1000,
    }
    spec = small_spec(**over)
    prob, net_a = make(spec, "A_pulse", 10.0)
    Trainer(net_a, prob, spec, tmp_path / "a").run()
    prob, net_b = make(spec, "A_pulse", 10.0)
    t1 = Trainer(net_b, prob, spec, tmp_path / "b")
    r = t1.run(stop_after_adam_epoch=25)
    assert r["status"] == "stopped"
    prob, net_c = make(spec, "A_pulse", 10.0, seed=123)  # different init: must be overwritten by the checkpoint
    Trainer(net_c, prob, spec, tmp_path / "b").run(resume=True)
    for pa, pc in zip(net_a.parameters(), net_c.parameters(), strict=True):
        assert torch.equal(pa, pc)


def test_lbfgs_resume_midway(tmp_path):
    spec = small_spec(
        **{
            "training.adam.epochs": 30,
            "training.lbfgs.iterations": 60,
            "training.lbfgs.inner": 10,
            "training.log_every": 10,
            "training.checkpoint_every": 20,
        }
    )
    prob, n1 = make(spec)
    Trainer(n1, prob, spec, tmp_path / "x").run()
    prob, n2 = make(spec)
    r = Trainer(n2, prob, spec, tmp_path / "y").run(stop_after_lbfgs_step=3)
    assert r == {"status": "stopped", "step": 3}
    prob, n3 = make(spec, seed=5)  # different init: overwritten by the checkpoint
    t3 = Trainer(n3, prob, spec, tmp_path / "y")
    assert t3.run(resume=True)["status"] == "ok"
    assert all(torch.equal(a, b) for a, b in zip(n1.parameters(), n3.parameters(), strict=True))
    assert [r["phase"] for r in t3.history].count("lbfgs") == 6  # all 6 outer L-BFGS steps logged, none duplicated


def test_weights_change_the_optimisation_and_grad_norms_reported(tmp_path):
    spec = small_spec(**{"training.adam.epochs": 20, "training.lbfgs.iterations": 0, "training.log_every": 5})
    prob, n1 = make(spec, "B_slab", 10.0)
    t1 = Trainer(n1, prob, spec, tmp_path / "w1", weights={"pde": 1, "ic": 1, "bc": 1})
    prob, n2 = make(spec, "B_slab", 10.0)
    t2 = Trainer(n2, prob, spec, tmp_path / "w2", weights={"pde": 1, "ic": 1, "bc": 100})
    t1.run()
    t2.run()
    assert not all(torch.equal(a, b) for a, b in zip(n1.parameters(), n2.parameters(), strict=True))
    g = component_grad_norms(n1, prob, t1.batch(0))
    assert set(g) == {"pde", "ic", "bc_left", "bc_right"} and all(np.isfinite(v) and v >= 0 for v in g.values())


def test_divergence_is_detected(tmp_path):
    """A non-finite loss must stop training with status 'diverged' (NaN weight injects it deterministically)."""
    spec = small_spec(**{"training.adam.epochs": 30, "training.lbfgs.iterations": 0})
    prob, net = make(spec)
    out = Trainer(net, prob, spec, tmp_path, weights={"pde": float("nan"), "ic": 1.0, "bc": 1.0}).run()
    assert out["status"] == "diverged" and out["n_log_rows"] == 0


def test_parametric_training_runs(tmp_path):
    spec = small_spec(
        **{"training.adam.epochs": 20, "training.lbfgs.iterations": 0, "problems.A_steady.parametric": "true"}
    )
    prob = P.build_problem("A_steady", spec["problems"]["A_steady"], None)
    torch.manual_seed(0)
    net = MLP(prob.lo, prob.hi, 8, 2)
    out = Trainer(net, prob, spec, tmp_path).run()
    assert out["status"] == "ok" and prob.input_names == ["x", "logp"]
    m = evaluate_model(net, prob, 10.0, 51, 2)
    assert np.isfinite(m["rel_l2"])
