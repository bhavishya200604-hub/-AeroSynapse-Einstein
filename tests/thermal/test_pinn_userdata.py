"""User data inputs (CSV / NPZ / HDF5 / Phase 2 files) for every study, data loss terms, and process isolation (D57)."""

import json

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as S
from aerosynapse.thermal.pinn import userdata as U
from aerosynapse.thermal.pinn.networks import MLP
from aerosynapse.thermal.pinn.trainer import Trainer
from aerosynapse.thermal.utils.isolate import run_tasks

DT = torch.float64
SPEC = S.load_spec(None, "smoke")


def slab():
    return P.build_problem("B_slab", SPEC["problems"]["B_slab"], 10.0)


def write_csv(path, rows, header):
    path.write_text(
        ",".join(header) + "\n" + "\n".join(",".join(f"{v:.17g}" for v in r) for r in rows) + "\n", encoding="utf-8"
    )


def test_csv_mapping_scale_and_domain_filter(tmp_path):
    f = tmp_path / "m.csv"
    write_csv(
        f, [(0.1, 0.1, 50.0), (0.5, 0.2, 60.0), (0.9, 0.9, 70.0), (1.5, 0.1, 80.0)], ["x", "t", "Tm"]
    )  # 2 outside
    cfg = {"file": str(f), "columns": {"theta": "Tm"}, "scale": {"theta": 0.01}, "role": "both"}
    p = slab()
    ud = U.load_source("m", cfg, p.input_names, ["theta"], {"Bi": 10.0}, p.lo, p.hi)
    assert ud.n == 2 and ud.meta["n_dropped"] == 2 and ud.meta["format"] == "csv"
    assert np.allclose(ud.fields["theta"], [0.5, 0.6]) and np.allclose(ud.inputs["t"], [0.1, 0.2])
    with pytest.raises(ValueError, match="input 't' not found"):
        U.load_source("m", {**cfg, "columns": {"theta": "Tm", "t": "time"}}, p.input_names, ["theta"], {}, p.lo, p.hi)


def test_npz_and_h5_points_and_exact_2d_evaluation(tmp_path):
    prob = P2.build_problem2d("C_mms", SPEC, SPEC["formulation"])
    fn = prob.truth["exact_fn"]
    xy = np.random.default_rng(0).uniform(size=(40, 2))
    X, Y = torch.as_tensor(xy[:, 0]), torch.as_tensor(xy[:, 1])
    cols = {"x": xy[:, 0], "y": xy[:, 1], **{k: fn[k](X, Y).numpy() for k in ("u", "v", "p", "theta")}}
    np.savez(tmp_path / "d.npz", **cols)
    with h5py.File(tmp_path / "d.h5", "w") as h:
        for k, v in cols.items():
            h[k] = v

    class Exact(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = nn.Parameter(torch.zeros(1, dtype=DT))

        def forward(self, Z):
            return torch.stack([fn[k](Z[:, 0], Z[:, 1]) for k in ("u", "v", "p", "theta")], -1) + 0 * self.d

    for name in ("d.npz", "d.h5"):
        spec = S.deep_merge(SPEC, {"user_data": {"sources": {"s": {"file": str(tmp_path / name), "role": "both"}}}})
        terms, evals = U.attach(spec, {"problem": "C_mms", "data": "s"}, prob, "cpu", DT, 0)
        m = U.evaluate(prob, Exact(), evals[0])
        assert m["n"] == 40 and all(m[f"rel_l2_{k}"] < 1e-13 for k in ("u", "v", "p", "theta"))
        vecs = terms[0].vectors(Exact())
        assert set(vecs) == {f"data_s_{k}" for k in ("u", "v", "p", "theta")}
        assert all(float(v.detach().abs().max()) < 1e-13 for v in vecs.values())


def test_phase2_dataset_selection(tmp_path):
    f = tmp_path / "caseB.h5"
    x, t = np.linspace(0, 1, 5), np.array([0.0, 0.25, 0.5])
    theta = np.stack([np.full((3, 5), float(i)) for i in range(3)])  # point i has theta == i everywhere
    with h5py.File(f, "w") as h:
        uu = np.array([[0.0], [2.0 / 3.0], [1.0]])  # Bi = 0.1, 10, 100 on a log scale [0.1, 100]
        h["params_unit"], h["split"] = uu, np.zeros(3, int)
        h["coords/x"], h["coords/t"], h["fields/theta"] = x, t, theta
        h["params"] = 0.1 * 1000.0**uu
        h.attrs.update(
            {
                "case": "caseB",
                "param_names": json.dumps(["Bi"]),
                "param_lo": json.dumps([0.1]),
                "param_hi": json.dumps([100.0]),
                "param_scale": json.dumps(["log"]),
                "u_lim": 1.0,
                "min_dist": 0.0,
            }
        )
    p = slab()
    ud = U.load_source("q", {"file": str(f)}, p.input_names, ["theta"], {"Bi": 10.0}, p.lo, p.hi)
    assert ud.meta["format"] == "phase2" and ud.meta["phase2_index"] == 1 and ud.n == 15
    assert np.all(ud.fields["theta"] == 1.0)
    last = U.load_source(
        "q", {"file": str(f), "select": {"index": 2, "time": "last"}}, p.input_names, ["theta"], {}, p.lo, p.hi
    )
    assert last.n == 5 and np.all(last.inputs["t"] == 0.5) and np.all(last.fields["theta"] == 2.0)
    with pytest.raises(ValueError, match="no design point"):
        U.load_source("q", {"file": str(f)}, p.input_names, ["theta"], {"Bi": 3.0}, p.lo, p.hi)


def test_data_loss_trains_and_is_balanced(tmp_path):
    p = slab()
    xt = np.random.default_rng(1).uniform(size=(30, 2)) * [1.0, 0.5]
    write_csv(tmp_path / "d.csv", [(a, b, 0.5) for a, b in xt], ["x", "t", "theta"])
    base = S.deep_merge(
        SPEC,
        {
            "user_data": {"sources": {"d": {"file": str(tmp_path / "d.csv"), "role": "train"}}},
            "training": {"adam": {"epochs": 30}, "lbfgs": {"iterations": 0}, "log_every": 1},
            "sampling": {"n_pde": 32, "n_ic": 8, "n_bc": 4},
        },
    )
    for method in ("fixed", "lr_annealing"):
        # fixed: physics weights 0, so the data misfit is the only objective and must decrease
        w = {"pde": 0.0, "ic": 0.0, "bc": 0.0} if method == "fixed" else {}
        s = S.deep_merge(base, {"loss": {"balancing": {"method": method}, "weights": w}})
        terms, evals = U.attach(s, {"problem": "B_slab", "value": 10.0, "data": "d"}, p, "cpu", DT, 0)
        assert len(terms) == 1 and evals == []  # role train: not evaluated
        torch.manual_seed(0)
        net = MLP(p.lo, p.hi, 12, 2).to(DT)
        tr = Trainer(net, p, s, tmp_path / method, data_terms=terms)
        tr.run()
        h = tr.history
        assert "data_d_theta" in h[0]
        if method == "fixed":
            assert h[-1]["data_d_theta"] < 0.1 * h[0]["data_d_theta"]
        if method == "lr_annealing":
            assert "w_data_d_theta" in h[-1] and tr.balancer.effective()["data_d_theta"] != 1.0
        phys, comp = tr.fixed_losses(tr.batch(0), include_data=False)
        assert not any(k.startswith("data_") for k in comp)


def test_no_data_means_no_terms_and_spec_checks(tmp_path):
    assert U.attach(SPEC, {"problem": "B_slab", "value": 10.0}, slab(), "cpu", DT, 0) == ([], [])
    with pytest.raises(ValueError, match="not defined under user_data.sources"):
        S.load_spec(None, "smoke", {"balance_study": {"cases": [{"problem": "B_slab", "value": 10.0, "data": "nope"}]}})
    with pytest.raises(ValueError, match="user_data.sources.a.role"):
        S.load_spec(None, "smoke", {"user_data": {"sources": {"a": {"file": "x.csv", "role": "fit"}}}})
    s = S.load_spec(
        None,
        "smoke",
        {
            "user_data": {"sources": {"a": {"file": "no/such/file.csv"}}},
            "formulation_study": {"cases": [{"problem": "C_cavity", "data": "a"}, "C_mms"]},
        },
    )
    with pytest.raises(ValueError, match="does not exist"):
        S.check_user_files(s, s["formulation_study"]["cases"])
    with pytest.raises(ValueError, match="execution.workers"):
        S.load_spec(None, "smoke", {"execution": {"workers": 0}})


def test_example_config_and_files_are_usable():
    ex = S.load_spec(S.ROOT / "configs" / "pinn" / "examples" / "user_data_custom.yaml", "smoke")
    for study in ("arch_study", "formulation_study", "balance_study", "strategy_study"):
        S.check_user_files(ex, ex[study]["cases"])
    p = slab()
    terms, evals = U.attach(ex, {"problem": "B_slab", "value": 10.0, "data": "slab_measurements"}, p, "cpu", DT, 0)
    assert len(terms) == 1 and len(evals) == 1 and evals[0].n == 200
    cav = P2.build_problem2d("C_cavity", ex, ex["formulation"], reference=_fake_ref())
    _, ev2 = U.attach(ex, {"problem": "C_cavity", "data": "cavity_probes"}, cav, "cpu", DT, 0)
    assert ev2[0].n == 49 and set(ev2[0].fields) == {"u", "v", "theta"}


def _fake_ref():
    x = (np.arange(8) + 0.5) / 8
    z = np.zeros((8, 8))
    return {"x": x, "y": x, "u": z, "v": z, "p": z, "theta": z, "nu_top": 1.0}


def test_isolated_runs_do_not_depend_on_what_ran_before(tmp_path):
    """D57: with isolation, a run's result is the same whether or not another run preceded it."""
    exe = {"isolate": True, "workers": 1, "threads_per_worker": 0}
    mod = "aerosynapse.thermal._isolation_probe"
    alone = run_tasks([(mod, "probe", {"problem": "B_slab", "value": 10.0, "out_dir": str(tmp_path / "a")})], exe)
    after = run_tasks(
        [
            (mod, "probe", {"problem": "A_steady", "value": 100.0, "out_dir": str(tmp_path / "b0")}),
            (mod, "probe", {"problem": "B_slab", "value": 10.0, "out_dir": str(tmp_path / "b1")}),
        ],
        exe,
    )
    assert after[0]["pid_marker"] == "A_steady" and alone[0]["checksum"] == after[1]["checksum"]
    inproc = run_tasks(
        [(mod, "probe", {"problem": "B_slab", "value": 10.0, "out_dir": str(tmp_path / "c")})], {"isolate": False}
    )
    assert np.isfinite(inproc[0]["checksum"])


def test_run_tasks_more_tasks_than_workers_returns_in_order():
    """Fresh process per task with 2 workers and 6 tasks: results in task order, and the pool shuts down (bounded)."""
    exe = {"isolate": True, "workers": 2, "threads_per_worker": 0}
    seen = []
    out = run_tasks([("json", "dumps", {"obj": i}) for i in range(6)], exe, lambda i, r, s: seen.append(i))
    assert out == [str(i) for i in range(6)] and sorted(seen) == list(range(6))


def test_run_tasks_reports_a_failing_task():
    exe = {"isolate": True, "workers": 2, "threads_per_worker": 0}
    with pytest.raises(RuntimeError, match="JSONDecodeError"):
        run_tasks([("json", "dumps", {"obj": 1}), ("json", "loads", {"s": "not json"})], exe)
