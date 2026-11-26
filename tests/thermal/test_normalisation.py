"""Step 2.3: normalisation (train-only statistics, fixed input bounds) and split integrity."""

import numpy as np

from aerosynapse.thermal.data import design as D
from aerosynapse.thermal.data.normalisation import Normaliser, fit_normaliser

DIMS = [D.Dim("Re", 100, 1000), D.Dim("Bi", 0.1, 100)]


def _design():
    return D.design_case("c", DIMS, {"train": 10, "val": 3, "interp": 4, "extrap": 4}, seed=2)


def test_field_stats_use_train_points_only():
    des = _design()
    rng = np.random.default_rng(0)
    f = rng.standard_normal((des.u.shape[0], 5, 6))
    n1 = fit_normaliser(des, {"x": (0, 1)}, {"theta": f})
    f2 = f.copy()
    f2[des.split != 0] += 100.0  # corrupt every non-train point
    n2 = fit_normaliser(des, {"x": (0, 1)}, {"theta": f2})
    assert n1.field_stats == n2.field_stats and n1.n_fit_points == 10
    tr = f[des.idx("train")]
    assert np.isclose(n1.field_stats["theta"]["mean"], tr.mean()) and np.isclose(
        n1.field_stats["theta"]["std"], tr.std()
    )
    # whole-data statistics would differ -> the leak would be visible
    assert abs(f2.mean() - n2.field_stats["theta"]["mean"]) > 1


def test_input_bounds_fixed_by_design_and_roundtrip(tmp_path):
    des = _design()
    nrm = fit_normaliser(des, {"x": (0, 1), "t": (0.05, 0.5)}, {"theta": np.zeros((des.u.shape[0], 2))})
    lo = nrm.normalise_inputs(x=0.0, t=0.05, Re=100.0, Bi=0.1)
    hi = nrm.normalise_inputs(x=1.0, t=0.5, Re=1000.0, Bi=100.0)
    assert np.allclose(lo, -1) and np.allclose(hi, 1)
    p = des.params
    z = nrm.normalise_inputs(x=0.3 + 0 * p[:, 0], t=0.2 + 0 * p[:, 0], Re=p[:, 0], Bi=p[:, 1])
    assert z.shape == (p.shape[0], 4) and np.all(np.abs(z) <= 1 + 1e-12)
    back = nrm.denormalise_inputs(z)
    assert np.allclose(back["Re"], p[:, 0]) and np.allclose(back["Bi"], p[:, 1]) and np.allclose(back["x"], 0.3)
    # train/val/interp inputs stay inside the training sub-box; extrapolation leaves it (normalised log coords)
    zp = nrm.normalise_inputs(x=0 * p[:, 0], t=0.05 + 0 * p[:, 0], Re=p[:, 0], Bi=p[:, 1])[:, 2:]
    lim = 2 * des.u_lim - 1
    assert np.all(zp[des.split < 3] <= lim + 1e-9) and np.all(zp[des.split == 3].max(axis=1) > lim)
    nrm.save(tmp_path / "n.json")
    again = Normaliser.load(tmp_path / "n.json")
    assert again.field_stats == nrm.field_stats and again.input_names == nrm.input_names


def test_field_zscore_roundtrip():
    des = _design()
    f = np.random.default_rng(1).uniform(2, 5, (des.u.shape[0], 8))
    nrm = fit_normaliser(des, {}, {"u": f})
    assert np.allclose(nrm.denormalise_field("u", nrm.normalise_field("u", f)), f)
    assert abs(nrm.normalise_field("u", f[des.idx("train")]).mean()) < 1e-12
