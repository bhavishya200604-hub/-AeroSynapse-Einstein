"""RAD candidates evaluated in chunks (memory bound) give the same residuals as one pass (src/pinn/strategies.py)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from aerosynapse.thermal.experiments.p6_4_sensitivity import ADAPTERS
from aerosynapse.thermal.pinn.strategies import AdaptiveSampler, pde_residual_abs

CKPT = Path("results/p4_7_ablation_smoke/runs/full-exact_psi_s0/checkpoint.pt")


@pytest.mark.skipif(
    not CKPT.exists(), reason="needs the git-ignored Step 4.7 checkpoint (python experiments/p4_7_ablation.py)"
)
def test_chunked_residuals_equal_single_pass():
    ad = ADAPTERS["D_graetz"]({"surrogate": {"run": "results/p4_7_ablation_smoke", "configuration": "full-exact_psi"}})
    _, prob, net = ad.members[0]
    p0 = next(net.parameters())
    cand = prob.sample(1000, 1, 1, "sobol", 5, p0.device, p0.dtype)["pde"]
    one = pde_residual_abs(net, prob, cand)
    for chunk in (300, 1000, 4096):
        part = pde_residual_abs(net, prob, cand, chunk)
        assert part.shape == one.shape
        assert np.allclose(part, one, rtol=1e-10, atol=1e-14)


def test_sampler_chunk_default_and_config():
    cfg = {"method": "rad", "period": 250, "k": None, "c": None, "n_candidates": 32768, "start_fraction": 0.5}
    assert AdaptiveSampler(cfg, 4096, 1000, 0).chunk == 4096
    assert AdaptiveSampler(cfg | {"chunk": 1024}, 4096, 1000, 0).chunk == 1024
    torch.manual_seed(0)
