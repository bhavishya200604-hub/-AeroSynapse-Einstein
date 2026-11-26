"""Tiny training run used by the process-isolation test (importable by name in a fresh process)."""

import torch

from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import spec as S
from aerosynapse.thermal.pinn.networks import MLP
from aerosynapse.thermal.pinn.trainer import Trainer


def probe(problem: str, value: float, out_dir: str) -> dict:
    s = S.deep_merge(
        S.load_spec(None, "smoke"),
        {
            "training": {"adam": {"epochs": 25}, "lbfgs": {"iterations": 4, "inner": 2}, "log_every": 100},
            "sampling": {"n_pde": 64, "n_ic": 16, "n_bc": 8},
        },
    )
    prob = P.build_problem(problem, s["problems"][problem], value)
    torch.manual_seed(0)
    net = MLP(prob.lo, prob.hi, 16, 2).to(torch.float64)
    Trainer(net, prob, s, out_dir).run()
    return {"checksum": float(sum(p.double().sum() for p in net.parameters())), "pid_marker": problem}
