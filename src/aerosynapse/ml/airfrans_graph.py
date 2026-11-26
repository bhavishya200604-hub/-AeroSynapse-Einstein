r"""Training on the AirfRANS graph data downloaded from Kaggle (``graph_airfrans_data_batch_1`` ... ``_10``).

**The data.** 1000 RANS simulations of 2D airfoils (AirfRANS, NeurIPS 2022 Datasets and Benchmarks, ODbL-1.0), one
PyTorch Geometric ``Data`` file per simulation: about 180 000 mesh nodes with

* ``x`` (5): the inlet velocity's two components (the same at every node: speed and angle of attack), the distance to
  the airfoil, and the wall normal (x, y; zero away from the wall);
* ``y`` (4): the fields u, v, p and the turbulent viscosity nu_t;
* ``pos`` (2): the node's position (chord from 0 to 1); ``edge_index``: about 10 neighbours per node.

Every column is min-max scaled to [0, 1] by whoever prepared the files, and the scaling constants are not in them, so
everything here is in those scaled units (no lift or drag coefficients in physical units). The ten folders reuse file
names but hold different simulations; a simulation is identified by (folder, file).

**Safety.** The files are pickles. They are read with PyTorch's weights-only loader and stand-in classes for the
PyG names they mention, so no code from the files runs and PyTorch Geometric is not needed.

**What is trained** (split by simulation, 80 / 10 / 10 with a fixed seed; the test airfoils are never seen):

* *MLP*: a point-wise network (inlet, distance, normal, Fourier features of the position) -> (u, v, p, nu_t), the
  AirfRANS paper's simplest baseline;
* *PointNet*: the same, plus a code of the whole airfoil and flow (a max-pool over the simulation's sampled points), so
  each point knows the shape it sits beside.

**Evaluation**: relative L2 error of each field on the held-out simulations at the sampled points and at every node
of the full mesh, the pressure along the airfoil surface, field maps, error against the inlet conditions.

Run ``python -m aerosynapse.ml.airfrans_graph all --out results/airfrans_graph`` (``extract``, ``train``,
``evaluate`` separately; ``--profile quick|full``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
FIELDS = ["u", "v", "p", "nu_t"]
N_VOL, N_FAR = 6000, 0.5  # sampled volume nodes per simulation; half of them weighted towards the airfoil
_SAFE: list[type] = []


# ---------------------------------------------------------------------------------------------- safe reading
def _stand_ins() -> list[type]:
    if _SAFE:
        return _SAFE
    for modname, names in (
        ("torch_geometric.data.data", ("Data", "DataEdgeAttr", "DataTensorAttr")),
        ("torch_geometric.data.storage", ("GlobalStorage",)),
    ):
        parts = modname.split(".")
        for i in range(1, len(parts) + 1):
            sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
        for n in names:
            c = type(
                n,
                (dict,),
                {
                    "__module__": modname,
                    "__setstate__": lambda self, st: self.update(st if isinstance(st, dict) else {"_state": st}),
                },
            )
            setattr(sys.modules[modname], n, c)
            _SAFE.append(c)
    return _SAFE


def read_graph(path: Path) -> dict[str, np.ndarray]:
    """One simulation: x, y, pos as float32 arrays (edges are not needed here)."""
    with torch.serialization.safe_globals(_stand_ins()):
        d = torch.load(path, weights_only=True)
    m = d["_store"]["_mapping"]
    return {k: m[k].numpy().astype(np.float32) for k in ("x", "y", "pos")}


def graph_files(root: Path = ROOT) -> list[tuple[str, Path]]:
    out = []
    for b in sorted(root.glob("graph_airfrans_data_batch_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])):
        for f in sorted(b.glob("graph_airfrans_data_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            out.append((f"{b.name.rsplit('_', 1)[1]}/{f.stem.rsplit('_', 1)[1]}", f))
    return out


# ---------------------------------------------------------------------------------------------- 1 extraction
def extract(out: Path, n_vol: int = N_VOL, seed: int = 0, limit: int | None = None, log=print) -> dict[str, Any]:
    """All wall nodes and n_vol sampled field nodes of every simulation into out/data.npz (float16 fields)."""
    rng = np.random.default_rng(seed)
    files = graph_files()
    if limit:
        files = files[:limit]
    out.mkdir(parents=True, exist_ok=True)
    sims, X, Y, P, S, owner = [], [], [], [], [], []
    seen: dict[tuple, str] = {}
    dups = 0
    t0 = time.time()
    for i, (sid, f) in enumerate(files):
        g = read_graph(f)
        x, y, pos = g["x"], g["y"], g["pos"]
        key = (round(float(x[0, 0]), 6), round(float(x[0, 1]), 6), len(x), round(float(pos[:, 0].sum()), 3))
        if key in seen:  # the same simulation twice
            dups += 1
            continue
        seen[key] = sid
        surf = np.flatnonzero(x[:, 2] <= x[:, 2].min() + 1e-7)
        rest = np.setdiff1d(np.arange(len(x)), surf, assume_unique=True)
        k_near = int(n_vol * N_FAR)
        w = 1.0 / (x[rest, 2] + 0.02)
        near = rng.choice(rest, k_near, replace=False, p=w / w.sum())
        uni = rng.choice(np.setdiff1d(rest, near, assume_unique=True), n_vol - k_near, replace=False)
        idx = np.concatenate([surf, near, uni])
        sims.append(
            {
                "id": sid,
                "n_nodes": int(len(x)),
                "n_surface": int(len(surf)),
                "inlet": [float(x[0, 0]), float(x[0, 1])],
                "file": str(f.relative_to(ROOT)),
            }
        )
        X.append(x[idx])
        Y.append(y[idx])
        P.append(pos[idx])
        S.append(np.r_[np.ones(len(surf), np.int8), np.zeros(n_vol, np.int8)])
        owner.append(np.full(len(idx), len(sims) - 1, np.int32))
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            log(f"  {i + 1}/{len(files)} simulations read ({time.time() - t0:.0f} s), {dups} duplicates skipped")
    np.savez_compressed(
        out / "data.npz",
        x=np.concatenate(X).astype(np.float16),
        y=np.concatenate(Y).astype(np.float16),
        pos=np.concatenate(P).astype(np.float32),
        surface=np.concatenate(S),
        sim=np.concatenate(owner),
    )
    meta = {"simulations": sims, "n_vol": n_vol, "duplicates_skipped": dups, "seconds": time.time() - t0}
    (out / "simulations.json").write_text(json.dumps(meta, indent=1))
    log(
        f"  {len(sims)} simulations, {sum(len(a) for a in X)} points kept ({(out / 'data.npz').stat().st_size / 1e6:.0f} MB)"
    )
    return meta


# ---------------------------------------------------------------------------------------------- 2 models
def fourier(pos: torch.Tensor, n: int = 8) -> torch.Tensor:
    f = 2.0 ** torch.arange(n, dtype=pos.dtype, device=pos.device) * math.pi / 4
    a = pos[..., None] * f  # (..., 2, n)
    return torch.cat([torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], -1)


def _mlp(i: int, w: int, o: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = i
    for _ in range(depth):
        layers += [nn.Linear(d, w), nn.GELU()]
        d = w
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class PointMLP(nn.Module):
    """(inlet, distance, normal, Fourier(pos)) -> (u, v, p, nu_t) at each point on its own."""

    def __init__(self, width: int = 160, depth: int = 5):
        super().__init__()
        self.net = _mlp(5 + 2 + 32, width, 4, depth)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:  # (B, N, *)
        return self.net(torch.cat([x, pos, fourier(pos)], -1))


class PointNetFlow(nn.Module):
    """The point-wise network plus a code of the whole simulation (a max-pool over its points)."""

    def __init__(self, width: int = 160, depth: int = 4, code: int = 96):
        super().__init__()
        self.enc = _mlp(5 + 2 + 32, width, code, 2)
        self.head = _mlp(5 + 2 + 32 + code, width, 4, depth)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, pos, fourier(pos)], -1)
        g = self.enc(h).amax(dim=-2, keepdim=True).expand(*h.shape[:-1], -1)
        return self.head(torch.cat([h, g], -1))


def split(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    p = np.random.default_rng(seed).permutation(n)
    a, b = int(0.8 * n), int(0.9 * n)
    return {"train": np.sort(p[:a]), "val": np.sort(p[a:b]), "test": np.sort(p[b:])}


class Data:
    def __init__(self, out: Path):
        z = np.load(out / "data.npz")
        self.x, self.y, self.pos = z["x"].astype(np.float32), z["y"].astype(np.float32), z["pos"]
        self.surface, self.sim = z["surface"].astype(bool), z["sim"]
        self.n = int(self.sim.max()) + 1
        starts = np.r_[0, np.flatnonzero(np.diff(self.sim)) + 1, len(self.sim)]
        self.rows = [np.arange(starts[i], starts[i + 1]) for i in range(self.n)]
        self.mu, self.sd = self.y.mean(0), self.y.std(0) + 1e-6  # targets standardised for training

    def batch(self, sims: np.ndarray, k: int, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
        idx = np.stack([rng.choice(self.rows[s], k, replace=len(self.rows[s]) < k) for s in sims])
        t = lambda a: torch.from_numpy(a[idx])  # noqa: E731
        return t(self.x), t(self.pos), torch.from_numpy((self.y[idx] - self.mu) / self.sd)


def train_model(
    kind: str, data: Data, sp: dict[str, np.ndarray], epochs: int, out: Path, seed: int = 0, log=print
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PointMLP() if kind == "mlp" else PointNetFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sims_per, pts = min(16, len(sp["train"])), 2048
    steps_per_epoch = max(1, len(sp["train"]) // sims_per) * 2
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps_per_epoch, pct_start=0.15)
    hist: dict[str, list[float]] = {"epoch": [], "train": [], "val": []}
    best, best_state, t0 = float("inf"), None, time.time()
    vrng = np.random.default_rng(123)
    vb = [data.batch(sp["val"][i : i + 16], 4096, vrng) for i in range(0, len(sp["val"]), 16)]
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for _ in range(steps_per_epoch):
            x, pos, y = data.batch(rng.choice(sp["train"], sims_per, replace=False), pts, rng)
            loss = ((model(x, pos) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.detach().item() / steps_per_epoch
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([((model(x, pos) - y) ** 2).mean().item() for x, pos, y in vb]))
        hist["epoch"].append(ep + 1)
        hist["train"].append(tl)
        hist["val"].append(vl)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        log(f"    {kind} epoch {ep + 1:3d}/{epochs}: train {tl:.4f}  val {vl:.4f}  ({time.time() - t0:.0f} s)")
    model.load_state_dict(best_state)
    torch.save(
        {"kind": kind, "state": best_state, "mu": torch.from_numpy(data.mu), "sd": torch.from_numpy(data.sd)},
        out / f"model_{kind}.pt",
    )
    return {
        "model": model,
        "history": hist,
        "best_val": best,
        "seconds": time.time() - t0,
        "params": sum(p.numel() for p in model.parameters()),
    }


# ---------------------------------------------------------------------------------------------- 3 evaluation
def predict(model: nn.Module, data: Data, x: np.ndarray, pos: np.ndarray, chunk: int = 60000) -> np.ndarray:
    """A whole simulation (any number of points); the PointNet code comes from all of its points at once."""
    with torch.no_grad():
        X, Pp = torch.from_numpy(x[None]), torch.from_numpy(pos[None])
        if isinstance(model, PointNetFlow):
            h = torch.cat([X, Pp, fourier(Pp)], -1)
            g = torch.cat([model.enc(h[:, i : i + chunk]).amax(1) for i in range(0, h.shape[1], chunk)]).amax(
                0, keepdim=True
            )
            outs = [
                model.head(torch.cat([h[:, i : i + chunk], g[:, None].expand(1, h[:, i : i + chunk].shape[1], -1)], -1))
                for i in range(0, h.shape[1], chunk)
            ]
        else:
            outs = [model(X[:, i : i + chunk], Pp[:, i : i + chunk]) for i in range(0, X.shape[1], chunk)]
        return torch.cat(outs, 1)[0].numpy() * data.sd + data.mu


def rel_l2(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per field: ||p - t|| / ||t - mean(t)|| (the fields are scaled, so the deviation from the mean is the signal)."""
    return np.linalg.norm(p - t, axis=0) / (np.linalg.norm(t - t.mean(0), axis=0) + 1e-12)


def evaluate(
    out: Path,
    data: Data,
    sp: dict[str, np.ndarray],
    models: dict[str, nn.Module],
    meta: dict[str, Any],
    n_full: int = 20,
    log=print,
) -> dict[str, Any]:
    sims = meta["simulations"]
    res: dict[str, Any] = {}
    for kind, m in models.items():
        e_s = np.array(
            [
                rel_l2(predict(m, data, data.x[data.rows[s]], data.pos[data.rows[s]]), data.y[data.rows[s]])
                for s in sp["test"]
            ]
        )
        e_w = np.array(
            [
                rel_l2(
                    predict(
                        m,
                        data,
                        data.x[data.rows[s]][data.surface[data.rows[s]]],
                        data.pos[data.rows[s]][data.surface[data.rows[s]]],
                    ),
                    data.y[data.rows[s]][data.surface[data.rows[s]]],
                )
                for s in sp["test"]
            ]
        )
        res[kind] = {
            "test_sampled": dict(zip(FIELDS, e_s.mean(0).tolist(), strict=True)),
            # at the wall u, v and nu_t are constant (no slip), so a relative error has nothing to divide by: pressure only
            "test_surface": {f: (float(e_w[:, i].mean()) if f == "p" else None) for i, f in enumerate(FIELDS)},
            "per_sim_sampled": e_s.tolist(),
            "per_sim_surface": e_w.tolist(),
        }
        log(
            f"  {kind}: held-out relative L2 (sampled points) "
            + ", ".join(f"{f} {100 * v:.2f} %" for f, v in zip(FIELDS, e_s.mean(0), strict=True))
        )
    # every node of the full mesh for some held-out simulations
    full = {k: [] for k in models}
    for s in sp["test"][:n_full]:
        g = read_graph(ROOT / sims[s]["file"])
        for kind, m in models.items():
            full[kind].append(rel_l2(predict(m, data, g["x"], g["pos"]), g["y"]))
    for kind in models:
        e = np.array(full[kind])
        res[kind]["test_full_mesh"] = dict(zip(FIELDS, e.mean(0).tolist(), strict=True))
        res[kind]["n_full"] = len(e)
        log(
            f"  {kind}: held-out relative L2 (all {sum(sims[s]['n_nodes'] for s in sp['test'][:n_full]) // max(1, len(e))} nodes of {len(e)} meshes) "
            + ", ".join(f"{f} {100 * v:.2f} %" for f, v in zip(FIELDS, e.mean(0), strict=True))
        )
    return res


def figures(
    out: Path,
    data: Data,
    sp: dict[str, np.ndarray],
    models: dict[str, Any],
    hist: dict[str, Any],
    res: dict[str, Any],
    meta: dict[str, Any],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    names: list[str] = []

    def save(fig, name):
        with contextlib.suppress(RuntimeError):  # a layout engine set elsewhere in the process
            fig.tight_layout()
        fig.savefig(out / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        names.append(name)

    sty = {
        "mlp": ("#2563eb", "-", "o", "MLP (point-wise)"),
        "pointnet": ("#dc2626", "--", "s", "PointNet (with the airfoil's code)"),
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, h in hist.items():
        c, ls, mk, lab = sty[k]
        ax.semilogy(h["epoch"], h["train"], color=c, ls=ls, lw=1.2, alpha=0.6, label=f"{lab}: train")
        ax.semilogy(h["epoch"], h["val"], color=c, ls=ls, marker=mk, ms=4, lw=2, label=f"{lab}: validation")
    ax.set(
        xlabel="epoch",
        ylabel="mean squared error (standardised fields)",
        title="Training on 800 AirfRANS simulations, validation on 100",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "01_training.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    xb = np.arange(4)
    for i, k in enumerate(models):
        c, _ls, _mk, lab = sty[k]
        for j, (part, alpha, hatch) in enumerate(
            (("test_sampled", 1.0, ""), ("test_full_mesh", 0.6, "//"), ("test_surface", 0.35, ".."))
        ):
            v = [
                100 * res[k][part][f] if res[k][part][f] is not None else 0.0 for f in FIELDS
            ]  # (u, v, nu_t at the wall: none)
            ax.bar(
                xb + (i * 3 + j - 2.5) * 0.13,
                v,
                0.12,
                color=c,
                alpha=alpha,
                hatch=hatch,
                edgecolor="k",
                lw=0.4,
                label=f"{lab}: {part.replace('test_', '').replace('_', ' ')}",
            )
    ax.set_xticks(xb, ["u", "v", "p", "nu_t"])
    ax.set(
        ylabel="relative L2 error on held-out airfoils, %",
        title="Held-out error per field (sampled points, every mesh node; at the wall pressure only)",
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    save(fig, "02_test_error_per_field.png")

    # one held-out simulation: true, predicted, error, on the full mesh near the airfoil
    sims = meta["simulations"]
    s = int(sp["test"][0])
    g = read_graph(ROOT / sims[s]["file"])
    best = min(models, key=lambda k: np.mean(list(res[k]["test_full_mesh"].values())))
    pr = predict(models[best], data, g["x"], g["pos"])
    keep = (g["pos"][:, 0] > -0.6) & (g["pos"][:, 0] < 1.8) & (np.abs(g["pos"][:, 1]) < 0.6)
    tri = mtri.Triangulation(g["pos"][keep, 0], g["pos"][keep, 1])
    fig, axs = plt.subplots(4, 3, figsize=(15, 13))
    for j, f in enumerate(FIELDS):
        t, p = g["y"][keep, j], pr[keep, j]
        lo, hi = np.percentile(t, [1, 99])
        for c_, (z, ttl, cm) in enumerate(
            (
                (t, "AirfRANS (RANS)", "viridis"),
                (p, f"{sty[best][3].split(' (')[0]} prediction", "viridis"),
                (p - t, "prediction - RANS", "coolwarm"),
            )
        ):
            a = axs[j, c_]
            if c_ < 2:
                im = a.tricontourf(tri, z, 40, cmap=cm, vmin=lo, vmax=hi)
            else:
                r = np.percentile(np.abs(z), 99)
                im = a.tricontourf(tri, z, 40, cmap=cm, vmin=-r, vmax=r)
            fig.colorbar(im, ax=a, shrink=0.8)
            a.set_aspect("equal")
            a.set_title(f"{f}: {ttl}", fontsize=10)
    fig.suptitle(
        f"Held-out simulation {sims[s]['id']} (inlet {sims[s]['inlet'][0]:.3f}, {sims[s]['inlet'][1]:.3f} scaled), every mesh node near the airfoil",
        fontweight="bold",
    )
    save(fig, "03_fields_test_airfoil.png")

    # the pressure along the wall of four held-out airfoils
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for a, s in zip(axs.flat, sp["test"][:4], strict=False):
        r = data.rows[int(s)][data.surface[data.rows[int(s)]]]
        xs = data.pos[r, 0]
        o = np.argsort(xs)
        a.plot(xs[o], data.y[r, 2][o], ".", color="k", ms=3, label="RANS")
        for k, m in models.items():
            c, _ls, mk, lab = sty[k]
            p = predict(m, data, data.x[r], data.pos[r])[:, 2]
            a.plot(xs[o], p[o], mk, color=c, ms=2.5, alpha=0.7, label=lab)
        a.set(xlabel="x / chord", ylabel="p (scaled)", title=f"wall pressure, held-out {sims[int(s)]['id']}")
        a.invert_yaxis()
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    save(fig, "04_wall_pressure.png")

    # error against the inlet conditions
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    inl = np.array([sims[int(s)]["inlet"] for s in sp["test"]])
    for a, k in zip(axs, models, strict=False):
        e = np.array(res[k]["per_sim_sampled"]).mean(1) * 100
        sc = a.scatter(inl[:, 0], inl[:, 1], c=e, cmap="magma_r", s=60, edgecolors="k")
        fig.colorbar(sc, ax=a, label="mean relative L2 error, %")
        a.set(
            xlabel="inlet u (scaled)",
            ylabel="inlet v (scaled: the angle of attack)",
            title=f"{sty[k][3]}: error of each held-out airfoil",
        )
    save(fig, "05_error_vs_inlet.png")
    return names


# ---------------------------------------------------------------------------------------------- the whole study
PROFILES = {"smoke": (40, 3), "quick": (None, 20), "full": (None, 50)}


def run(
    out: Path, profile: str = "quick", steps: tuple[str, ...] = ("extract", "train", "evaluate"), log=print
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    limit, epochs = PROFILES[profile]
    lines: list[str] = []

    def say(s: str) -> None:
        log(s)
        lines.append(s)

    t0 = time.time()
    if "extract" in steps or not (out / "data.npz").exists():
        say(f"1  Reading the AirfRANS graph files ({limit if limit else len(graph_files())} files) and sampling them")
        extract(out, limit=limit, log=say)
    if steps == ("extract",):
        return {"extracted": True}
    meta = json.loads((out / "simulations.json").read_text())
    data = Data(out)
    sp = split(data.n)
    say(
        f"   {data.n} simulations: {len(sp['train'])} train, {len(sp['val'])} validation, {len(sp['test'])} held-out test; {len(data.x)} points"
    )
    models, hist, info = {}, {}, {}
    prev = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
    if "train" not in steps and (out / "train_log.txt").exists():
        lines[:0] = (out / "train_log.txt").read_text(encoding="utf-8").splitlines() + ["(evaluation run again)"]
    if "train" in steps or not all((out / f"model_{k}.pt").exists() for k in ("mlp", "pointnet")):
        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        for kind in ("mlp", "pointnet"):
            say(f"2  Training {kind} ({epochs} epochs)")
            r = train_model(kind, data, sp, epochs, out, log=say)
            models[kind], hist[kind] = r["model"], r["history"]
            info[kind] = {"best_val_mse": r["best_val"], "train_s": r["seconds"], "parameters": r["params"]}
        (out / "history.json").write_text(json.dumps(hist))
        (out / "train_log.txt").write_text("\n".join(lines), encoding="utf-8")
    else:
        hist = json.loads((out / "history.json").read_text())
        for kind in ("mlp", "pointnet"):
            ck = torch.load(out / f"model_{kind}.pt", weights_only=True)
            m = PointMLP() if kind == "mlp" else PointNetFlow()
            m.load_state_dict(ck["state"])
            models[kind] = m.eval()
    say("3  Evaluating on the held-out airfoils")
    res = evaluate(out, data, sp, models, meta, n_full=10 if profile == "smoke" else 20, log=say)
    names = figures(out, data, sp, models, hist, res, meta)
    summary = {
        "profile": profile,
        "simulations": data.n,
        "split": {k: len(v) for k, v in sp.items()},
        "points": int(len(data.x)),
        "models": info or prev.get("models", {}),
        "results": {k: {kk: v for kk, v in r.items() if not kk.startswith("per_sim")} for k, r in res.items()},
        "figures": names,
        "seconds": time.time() - t0,
    }
    (out / "result.json").write_text(json.dumps(summary, indent=1))
    head = [
        "AIRFRANS GRAPH DATA: FLOW-FIELD SURROGATES",
        "=" * 60,
        __doc__.split("**What is trained**")[0].replace("**", "").replace("``", "").strip(),
        "",
        "RUN LOG",
        "-" * 60,
    ]
    (out / "summary.txt").write_text(
        "\n".join(head + lines + ["", f"Total {summary['seconds'] / 60:.1f} min. Data: AirfRANS (ODbL-1.0)."]),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train flow-field surrogates on the AirfRANS graph data in the project folder."
    )
    ap.add_argument("step", choices=["extract", "train", "evaluate", "all"])
    ap.add_argument("--out", default=str(ROOT / "results" / "airfrans_graph"))
    ap.add_argument("--profile", default="quick", choices=list(PROFILES))
    a = ap.parse_args(argv)
    steps = ("extract", "train", "evaluate") if a.step == "all" else (a.step,)
    run(Path(a.out), a.profile, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
