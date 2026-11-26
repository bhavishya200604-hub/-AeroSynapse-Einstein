"""Step 2.4: HDF5 round trip and file-level leakage check."""

import numpy as np

from aerosynapse.thermal.data import design as D
from aerosynapse.thermal.data import generators as G
from aerosynapse.thermal.data import spec as SP
from aerosynapse.thermal.data.dataset_io import read_dataset, sha256, write_dataset
from aerosynapse.thermal.data.normalisation import fit_normaliser

CFG = SP.load_spec(None, "smoke")["cases"]["caseA"]


def test_roundtrip_and_leakage_from_file(tmp_path):
    dims = [D.Dim("Pe", 1.0, 1000.0)]
    des = D.design_case("caseA", dims, {"train": 8, "val": 2, "interp": 3, "extrap": 3}, seed=5)
    out = G.gen_case_a(des.params[:, 0], CFG)
    info = write_dataset(
        tmp_path / "a.h5",
        des,
        {"x": out["x"], "t": out["t"]},
        {"theta": out["theta"], "theta_steady": out["theta_steady"]},
        config=CFG,
    )
    assert info["bytes"] > 0 and info["sha256"] == sha256(tmp_path / "a.h5")
    ds = read_dataset(tmp_path / "a.h5")
    assert np.array_equal(ds.fields["theta"], out["theta"]) and np.array_equal(ds.design.u, des.u)
    assert ds.design.dims == dims and ds.attrs["case"] == "caseA"
    assert D.check_leakage(ds.design)["all_ok"]
    # splits recovered from the file are identical, and statistics from the file equal those from memory
    assert all(np.array_equal(ds.split_idx(s), des.idx(s)) for s in D.SPLITS)
    n_mem = fit_normaliser(des, {"x": (0, 1)}, {"theta": out["theta"]})
    n_file = fit_normaliser(ds.design, {"x": (0, 1)}, {"theta": ds.fields["theta"]})
    assert n_mem.field_stats == n_file.field_stats
