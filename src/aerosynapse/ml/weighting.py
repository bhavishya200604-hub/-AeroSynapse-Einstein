r"""Adaptive loss weighting: learning-rate annealing of Wang, Teng & Perdikaris (2021).

:math:`\hat\lambda_k = \max_\theta|\nabla_\theta L_r| / \overline{|\nabla_\theta L_k|}`, smoothed
:math:`\lambda_k \leftarrow (1-\alpha)\lambda_k + \alpha\hat\lambda_k`. Used only in experiments that
first demonstrate a gradient imbalance (see Phase 2 record ``phase2_loss_imbalance``).
"""

from __future__ import annotations

import torch


def grad_stats(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> tuple[float, float]:
    """(max |g|, mean |g|) of d loss / d params."""
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = torch.cat([g.abs().flatten() for g in gs if g is not None])
    return float(flat.max()), float(flat.mean())


class AnnealingWeights:
    def __init__(self, names: list[str], alpha: float = 0.9) -> None:  # alpha = 0.9 as recommended in the paper
        self.w = {n: 1.0 for n in names}
        self.alpha = alpha
        self.history: list[dict[str, float]] = []

    def update(
        self, residual_loss: torch.Tensor, losses: dict[str, torch.Tensor], params: list[torch.nn.Parameter]
    ) -> dict[str, float]:
        gmax_r, _ = grad_stats(residual_loss, params)
        for k, L in losses.items():
            _, gmean = grad_stats(L, params)
            lam_hat = gmax_r / max(gmean, 1e-16)
            self.w[k] = (1 - self.alpha) * self.w[k] + self.alpha * lam_hat
        self.history.append(dict(self.w))
        return dict(self.w)


def gradient_imbalance(
    residual_loss: torch.Tensor, losses: dict[str, torch.Tensor], params: list[torch.nn.Parameter]
) -> dict[str, float]:
    """Ratio mean|∇L_r| / mean|∇L_k| for each constraint term (≫1 indicates imbalance)."""
    _, gr = grad_stats(residual_loss, params)
    return {k: gr / max(grad_stats(L, params)[1], 1e-16) for k, L in losses.items()}
