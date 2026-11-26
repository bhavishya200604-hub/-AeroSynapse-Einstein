r"""DeepONet (Lu et al. 2021): :math:`G(a)(y)\approx\sum_{k=1}^p b_k(a(x_1),\dots,a(x_m))\,t_k(y)+b_0`.

Branch: MLP on the input function sampled at ``m`` fixed sensors (plus scalar parameters). Trunk: MLP on
the query coordinate y. Unlike the FNO it can be queried at arbitrary y, but the sensor locations are
fixed, so it does **not** transfer to a different input discretisation without re-sampling at the sensors.
"""

from __future__ import annotations

import torch
from torch import nn


def _mlp(i: int, o: int, w: int, d: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(d):
        layers += [nn.Linear(i, w), nn.Tanh()]
        i = w
    layers.append(nn.Linear(i, o))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(
        self,
        m_sensors: int,
        n_params: int = 0,
        coord_dim: int = 1,
        p: int = 64,
        width: int = 128,
        depth: int = 3,
        periodic: bool = True,
    ) -> None:
        super().__init__()
        self.periodic = periodic
        self.branch = _mlp(m_sensors + n_params, p, width, depth)
        self.trunk = _mlp(2 * coord_dim if periodic else coord_dim, p, width, depth)
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, sensors: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """sensors (B, m[+P]); y (n, d) → (B, n)."""
        if self.periodic:
            y = torch.cat([torch.cos(2 * torch.pi * y), torch.sin(2 * torch.pi * y)], -1)
        return self.branch(sensors) @ torch.tanh(self.trunk(y)).T + self.b0
