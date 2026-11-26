"""Step 6.5: training diagnostics of 1D PINNs — residual Jacobian / NTK, Hessian, loss landscape, error spectra.

Loss used throughout (truth-free, the unweighted training objective on a FIXED point set):
    L(theta) = mean(r_pde^2) + mean(r_ic^2) + mean(r_bc^2)      (each group a mean square, as in the trainer)
NTK of the residual (Wang, Yu & Perdikaris; LITERATURE.md Wang2021ntk): K = J J^T with J = d r / d theta
(rows = residual points of every group). Under gradient flow on L the residuals evolve as dr/dt = -K D r (D = per-group
1/n_g weights), so directions with small eigenvalues are learnt slowly; block traces tr(K_g)/n_g compare the groups.
Lazy-regime indicator: relative kernel drift ||K_t - K_0||_F / ||K_0||_F (small for wide networks).
Kernel-target alignment A(K, y) = y^T K y / (||K||_F ||y||^2) (LITERATURE.md Cristianini2002).
Hessian of L: Hessian-vector products by double back-propagation; full matrix for small networks (p ~ 2-3k), extreme
eigenvalues by Lanczos otherwise. Loss landscape (LITERATURE.md Li2018landscape): L(theta* + a d1 + b d2) with
filter-normalised random directions or the two principal directions of the training trajectory.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla
import torch
from scipy.fft import dct

from aerosynapse.thermal.pinn.derivs import evaluate


# --------------------------------------------------------------------------- parameters as one vector
def params(model) -> list[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def flat(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params(model)])


def set_flat(model, v: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in params(model):
            n = p.numel()
            p.copy_(v[i : i + n].reshape(p.shape))
            i += n


# --------------------------------------------------------------------------- residuals with graph
def point_set(problem, n_pde: int, n_ic: int, n_t: int) -> dict:
    """Fixed, deterministic points: interior x (and t) grid, IC points, boundary points."""
    x = np.linspace(0.0, 1.0, n_pde + 2)[1:-1]
    pts = {}
    if problem.has_time:
        t = np.linspace(0.0, problem.T, n_t + 1)[1:]
        T_, X_ = np.meshgrid(t, x, indexing="ij")
        pts["pde"] = {"x": X_.ravel(), "t": T_.ravel()}
        xi = np.linspace(0.0, 1.0, n_ic)
        pts["ic"] = {"x": xi, "t": np.zeros_like(xi)}
        tb = np.linspace(0.0, problem.T, n_t + 1)[1:]
        pts["bc_left"] = {"x": np.zeros_like(tb), "t": tb}
        pts["bc_right"] = {"x": np.ones_like(tb), "t": tb}
    else:
        pts["pde"] = {"x": x}
        pts["bc_left"] = {"x": np.zeros(1)}
        pts["bc_right"] = {"x": np.ones(1)}
    return pts


def _cols(pts: dict, like: torch.Tensor) -> dict:
    return {k: torch.as_tensor(v, dtype=like.dtype, device=like.device) for k, v in pts.items()}


def residuals(model, problem, pts: dict) -> dict[str, torch.Tensor]:
    """Residual vectors per group WITH the graph (differentiable w.r.t. the parameters). 2D problems
    (aerosynapse.thermal.pinn.problems2d and subclasses) define their own trained residual groups: ``pts`` is then their batch
    (problem.sample(...)) and the groups are pde_* / bc_* (/ gauge), as in training (data terms not included)."""
    if hasattr(problem, "residual_vectors"):
        return problem.residual_vectors(model, pts)
    like = next(model.parameters())
    names = problem.input_names
    first, second = problem.derivative_spec()
    out = {}
    c = _cols(pts["pde"], like)
    d = evaluate(model, c, names, first=first, second=second, create_graph=True)
    out["pde"] = problem.residual(d, problem.p_of(c))
    if "ic" in pts:
        c = _cols(pts["ic"], like)
        u = model(torch.stack([c[k] for k in names], dim=1)).squeeze(-1)
        out["ic"] = u - problem.ic_target(c)
    for side in ("left", "right"):
        c = _cols(pts[f"bc_{side}"], like)
        d = evaluate(model, c, names, first=("x",), second=(), create_graph=True)
        out[f"bc_{side}"] = problem.bc_residual(side, d, c)
    return out


def loss(model, problem, pts: dict) -> torch.Tensor:
    r = residuals(model, problem, pts)
    return sum(torch.mean(v**2) for v in r.values())


# --------------------------------------------------------------------------- Jacobian and NTK
def jacobian(model, problem, pts: dict) -> tuple[np.ndarray, list[str]]:
    """J = d r / d theta (rows: residual points of all groups, in order); one backward pass per row."""
    r = residuals(model, problem, pts)
    rows, labels = [], []
    ps = params(model)
    for g, v in r.items():
        for i in range(v.numel()):
            gr = torch.autograd.grad(v[i], ps, retain_graph=True, allow_unused=True)
            rows.append(
                torch.cat(
                    [(z if z is not None else torch.zeros_like(p)).reshape(-1) for z, p in zip(gr, ps, strict=True)]
                )
                .detach()
                .cpu()
                .numpy()
            )
            labels.append(g)
    return np.array(rows), labels


def ntk_stats(J: np.ndarray, labels: list[str], k_top: int = 8) -> dict:
    """Eigen-spectrum of K = J J^T, block traces per group (per point), top eigenvectors of the PDE block."""
    K = J @ J.T
    lam = np.clip(np.linalg.eigvalsh(K)[::-1], 0.0, None)
    lab = np.asarray(labels)
    traces = {
        g: float(np.trace(K[np.ix_(lab == g, lab == g)]) / max((lab == g).sum(), 1)) for g in dict.fromkeys(labels)
    }
    m = np.array([str(g).startswith("pde") for g in lab], bool)  # 1D: "pde"; 2D: pde_mom_x, pde_energy, ...
    Kp = K[np.ix_(m, m)]
    w, V = np.linalg.eigh(Kp)
    order = np.argsort(w)[::-1]
    return {
        "eigenvalues": lam.tolist(),
        "pde_eigenvalues": np.clip(w[order], 0, None).tolist(),
        "pde_eigenvectors": V[:, order[:k_top]].T.tolist(),
        "trace_per_point": traces,
        "condition": float(lam[0] / max(lam[lam > 0][-1] if (lam > 0).any() else 1e-300, 1e-300)),
        "effective_rank": float(np.exp(-np.sum((p := lam / lam.sum())[p > 0] * np.log(p[p > 0])))),
        "_K": K,
    }


def drift(K0: np.ndarray, K: np.ndarray) -> float:
    return float(np.linalg.norm(K - K0) / max(np.linalg.norm(K0), 1e-300))


def alignment(K: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, float)
    return float(y @ K @ y / max(np.linalg.norm(K) * (y @ y), 1e-300))


def output_jacobian(model, problem, x: np.ndarray) -> np.ndarray:
    """d u(x_i) / d theta for a steady problem (rows = points): the NTK of the network OUTPUT, K_uu = J J^T."""
    like = next(model.parameters())
    X = torch.as_tensor(np.asarray(x)[:, None], dtype=like.dtype, device=like.device)
    u = model(X).squeeze(-1)
    ps = params(model)
    rows = []
    for i in range(u.numel()):
        gr = torch.autograd.grad(u[i], ps, retain_graph=True)
        rows.append(torch.cat([z.reshape(-1) for z in gr]).detach().cpu().numpy())
    return np.array(rows)


# --------------------------------------------------------------------------- Hessian
def hvp(model, problem, pts: dict, v: torch.Tensor) -> torch.Tensor:
    ps = params(model)
    L = loss(model, problem, pts)
    g = torch.autograd.grad(L, ps, create_graph=True)
    gv = torch.cat([x.reshape(-1) for x in g]) @ v
    h = torch.autograd.grad(gv, ps)
    return torch.cat([x.reshape(-1) for x in h]).detach()


def hessian_full(model, problem, pts: dict) -> np.ndarray:
    n = flat(model).numel()
    like = next(model.parameters())
    H = np.empty((n, n))
    for i in range(n):
        e = torch.zeros(n, dtype=like.dtype, device=like.device)
        e[i] = 1.0
        H[:, i] = hvp(model, problem, pts, e).cpu().numpy()
    return 0.5 * (H + H.T)


def hessian_extremes(model, problem, pts: dict, tol: float = 1e-6, maxiter: int = 1000) -> dict:
    """Largest eigenvalue (Lanczos, which='LA') and the most negative one as lambda_max - mu, mu = largest eigenvalue of
    (lambda_max I - H) (moves the wanted end of the spectrum to the well-separated top). A non-converged Lanczos is
    reported (converged False, with ARPACK's partial value if any) instead of raising."""
    n = flat(model).numel()
    like = next(model.parameters())

    def mv(v):
        return (
            hvp(model, problem, pts, torch.as_tensor(np.asarray(v).ravel(), dtype=like.dtype, device=like.device))
            .cpu()
            .numpy()
        )

    def top(op):
        try:
            return float(
                np.max(
                    spla.eigsh(op, k=1, which="LA", tol=tol, return_eigenvectors=False, maxiter=maxiter, ncv=min(n, 40))
                )
            ), True
        except spla.ArpackNoConvergence as e:
            ev = getattr(e, "eigenvalues", None)
            return (float(np.max(ev)) if ev is not None and len(ev) else float("nan")), False

    H = spla.LinearOperator((n, n), matvec=mv, dtype=float)
    lmax, ok1 = top(H)
    if not np.isfinite(lmax):
        return {"max": None, "min": None, "converged": False}
    shifted = spla.LinearOperator((n, n), matvec=lambda v: lmax * np.asarray(v).ravel() - mv(v), dtype=float)
    mu, ok2 = top(shifted)
    return {"max": lmax, "min": (lmax - mu) if np.isfinite(mu) else None, "converged": bool(ok1 and ok2)}


# --------------------------------------------------------------------------- loss landscape
def filter_normalised(model, rng: np.random.Generator) -> torch.Tensor:
    """Random direction with every row (filter) of every weight matrix scaled to that row's norm; biases 0."""
    parts = []
    for p in params(model):
        d = torch.as_tensor(rng.standard_normal(tuple(p.shape)), dtype=p.dtype, device=p.device)
        if p.dim() >= 2:
            d = d * (p.detach().norm(dim=1, keepdim=True) / d.norm(dim=1, keepdim=True).clamp_min(1e-12))
        else:
            d = torch.zeros_like(p)
        parts.append(d.reshape(-1))
    return torch.cat(parts)


def pca_directions(traj: list[torch.Tensor], final: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Two leading principal directions of the trajectory (theta_t - theta_final); coordinates of every snapshot."""
    M = torch.stack([t - final for t in traj]).cpu().numpy()
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    d1, d2 = Vt[0], Vt[1]
    coords = M @ np.stack([d1, d2], 1)
    like = final
    return torch.as_tensor(d1, dtype=like.dtype), torch.as_tensor(d2, dtype=like.dtype), coords


def landscape(
    model, problem, pts: dict, d1: torch.Tensor, d2: torch.Tensor, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """L(theta* + a d1 + b d2) on the grid a x b (rows = b); theta* restored afterwards."""
    th = flat(model)
    Z = np.empty((len(b), len(a)))
    try:
        for j, bb in enumerate(b):
            for i, aa in enumerate(a):
                set_flat(model, th + float(aa) * d1 + float(bb) * d2)
                Z[j, i] = float(loss(model, problem, pts).detach())
    finally:
        set_flat(model, th)
    return Z


def interpolate(
    model, problem, pts: dict, th0: torch.Tensor, th1: torch.Tensor, n: int = 51
) -> tuple[np.ndarray, np.ndarray]:
    """L((1 - s) theta_0 + s theta_1) for s in [-0.2, 1.2] (Goodfellow et al. 2015)."""
    th = flat(model)
    s = np.linspace(-0.2, 1.2, n)
    out = np.empty(n)
    try:
        for i, si in enumerate(s):
            set_flat(model, (1 - si) * th0 + si * th1)
            out[i] = float(loss(model, problem, pts).detach())
    finally:
        set_flat(model, th)
    return s, out


# --------------------------------------------------------------------------- spectra
def dct_amplitudes(e: np.ndarray) -> np.ndarray:
    """|DCT-II| (orthonormal) of a signal on a uniform grid: amplitude per cosine mode k = 0, 1, ..."""
    return np.abs(dct(np.asarray(e, float), type=2, norm="ortho"))
