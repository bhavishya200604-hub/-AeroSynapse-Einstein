r"""Inverse problem: identify the diffusivity α of :math:`T_t=\alpha T_{xx}` from sparse noisy data.

Classical route: discrete Crank–Nicolson forward model, least-squares misfit
:math:`J(\alpha)=\tfrac12\sum_{n,s}(T^n_s(\alpha)-d^n_s)^2`, gradient by the **discrete adjoint**:
with :math:`A T^{n+1}=B T^n`, :math:`A=I-\tfrac{\Delta t}{2}\alpha L`, :math:`B=I+\tfrac{\Delta t}{2}\alpha L`,

.. math:: A^\top\lambda^{n}=r^{n}+B^\top\lambda^{n+1},\qquad
          \frac{dJ}{d\alpha}=\sum_n (\lambda^{n})^\top\tfrac{\Delta t}{2}L\,(T^{n}+T^{n-1})

(:math:`r^n=H^\top(HT^n-d^n)`), verified against finite differences in the tests. Gauss–Newton
gives an approximate standard error :math:`\sigma\,(J_r^\top J_r)^{-1/2}`.
PINN route: α is a trainable parameter, loss = residual + IC/BC + data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.numerics.fd1d import assemble_1d
from aerosynapse.physics.bc import Dirichlet

FloatArr = NDArray[np.float64]


@dataclass
class HeatInverseSetup:
    n: int = 64
    nt: int = 100
    t_final: float = 0.5
    sensors: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    every: int = 10

    def __post_init__(self) -> None:
        op = assemble_1d(0, 1, self.n, 1.0, 0.0, Dirichlet(0.0), Dirichlet(0.0))
        L = op.A.tolil()
        for i in (0, self.n):
            L.rows[i], L.data[i] = [], []
        self.L = L.tocsr()
        self.x = op.x
        self.dt = self.t_final / self.nt
        self.idx = np.array([int(round(s * self.n)) for s in self.sensors])
        self.obs_steps = list(range(self.every, self.nt + 1, self.every))
        self.u0 = np.sin(np.pi * self.x) + 0.5 * np.sin(3 * np.pi * self.x)

    def forward(self, alpha: float) -> list[FloatArr]:
        I = sp.identity(self.n + 1, format="csc")
        A = (I - 0.5 * self.dt * alpha * self.L).tocsc()
        B = (I + 0.5 * self.dt * alpha * self.L).tocsr()
        lu = spla.splu(A)
        T = [self.u0.copy()]
        for _ in range(self.nt):
            T.append(lu.solve(B @ T[-1]))
        return T

    def observe(self, T: list[FloatArr]) -> FloatArr:
        return np.array([T[k][self.idx] for k in self.obs_steps])

    def misfit_and_grad(self, alpha: float, d: FloatArr) -> tuple[float, float]:
        T = self.forward(alpha)
        R = self.observe(T) - d
        J = 0.5 * float(np.sum(R**2))
        I = sp.identity(self.n + 1, format="csc")
        A = (I - 0.5 * self.dt * alpha * self.L).tocsc()
        B = (I + 0.5 * self.dt * alpha * self.L).tocsr()
        luT = spla.splu(A.T.tocsc())
        lam_next = np.zeros(self.n + 1)
        grad = 0.0
        obs_map = {k: i for i, k in enumerate(self.obs_steps)}
        for n in range(self.nt, 0, -1):
            rhs = B.T @ lam_next
            if n in obs_map:
                r = np.zeros(self.n + 1)
                r[self.idx] = R[obs_map[n]]
                rhs = rhs + r
            lam = luT.solve(rhs)
            grad += float(lam @ (0.5 * self.dt * (self.L @ (T[n] + T[n - 1]))))
            lam_next = lam
        return J, grad

    def identify(self, d: FloatArr, alpha0: float = 0.05, noise_sigma: float | None = None) -> dict[str, float]:
        from scipy.optimize import minimize as spmin

        res = spmin(
            lambda a: self.misfit_and_grad(float(np.exp(a[0])), d)[0],
            [np.log(alpha0)],
            jac=lambda a: np.array([self.misfit_and_grad(float(np.exp(a[0])), d)[1] * np.exp(a[0])]),
            method="L-BFGS-B",
        )
        a_hat = float(np.exp(res.x[0]))
        h = 1e-6 * a_hat
        Jr = (self.observe(self.forward(a_hat + h)) - self.observe(self.forward(a_hat - h))).ravel() / (2 * h)
        resid = (self.observe(self.forward(a_hat)) - d).ravel()
        sig = noise_sigma if noise_sigma is not None else float(np.sqrt(np.sum(resid**2) / max(1, resid.size - 1)))
        se = sig / float(np.sqrt(Jr @ Jr))
        return {"alpha_hat": a_hat, "se": se, "n_iter": int(res.nit), "misfit": float(0.5 * resid @ resid)}


def pinn_identify(setup: HeatInverseSetup, d: FloatArr, iterations: int = 3000, seed: int = 0) -> dict[str, float]:
    """PINN route: u_t = α u_xx with α = exp(θ) trainable; hard IC/BC via trial function."""
    import time

    import torch

    from aerosynapse.ml.autodiff import grad, second
    from aerosynapse.ml.nets import MLP, set_seed

    set_seed(seed)
    dt = torch.get_default_dtype()
    net = MLP(2, 1, 32, 3, lo=[0, 0], hi=[1, setup.t_final], seed=seed)
    log_a = torch.nn.Parameter(torch.tensor(np.log(0.05), dtype=dt))
    opt = torch.optim.Adam([*net.parameters(), log_a], lr=1e-3)
    x_obs = setup.x[setup.idx]
    t_obs = np.array(setup.obs_steps) * setup.dt
    Xd = torch.tensor(np.array([[x, t] for t in t_obs for x in x_obs]), dtype=dt)
    Ud = torch.tensor(d.reshape(-1, 1), dtype=dt)
    rng = np.random.default_rng(seed)

    def u(X: torch.Tensor) -> torch.Tensor:
        x, t = X[:, :1], X[:, 1:2]
        u0 = torch.sin(torch.pi * x) + 0.5 * torch.sin(3 * torch.pi * x)
        out: torch.Tensor = u0 + t * x * (1 - x) * net(X)
        return out

    Xc = torch.tensor(rng.random((1000, 2)) * [1, setup.t_final], dtype=dt, requires_grad=True)
    t0 = time.perf_counter()
    for _ in range(iterations):
        U = u(Xc)
        r = grad(U, Xc)[:, 1:2] - torch.exp(log_a) * second(U, Xc, 0)
        loss = (r**2).mean() + 100.0 * ((u(Xd) - Ud) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"alpha_hat": float(torch.exp(log_a)), "wall_s": time.perf_counter() - t0, "final_loss": float(loss)}
