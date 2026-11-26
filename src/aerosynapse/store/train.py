"""Train a network from shards on disk without loading the dataset into memory.

Rows are float32 vectors ``[inputs..., outputs...]`` in an :class:`~aerosynapse.store.shards.ArrayWriter` store.
Training streams mini-batches through a bounded shuffle buffer (memory = buffer x row size), normalises with
statistics computed in one streaming pass (Welford), checkpoints model, optimiser and epoch to disk after every
epoch (an interrupted run resumes), and holds out whole shards for validation so no validation row is ever
trained on.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from aerosynapse.store.shards import ArrayReader, stream


def streaming_stats(reader: ArrayReader, shards: list[int]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n, mean, m2 = 0, None, None
    for k in shards:
        for row in reader.shard(k):
            x = np.asarray(row, np.float64).ravel()
            n += 1
            if mean is None:
                mean, m2 = x.copy(), np.zeros_like(x)
            else:
                d = x - mean
                mean += d / n
                m2 += d * (x - mean)
    assert mean is not None and m2 is not None
    return mean, np.sqrt(np.maximum(m2 / max(n - 1, 1), 1e-12))


class _Sub(ArrayReader):
    def __init__(self, base: ArrayReader, keep: list[int]) -> None:
        self.dir, self.shape = base.dir, base.shape
        self.man = {**base.man, "shards": [base.man["shards"][k] for k in keep]}


def mlp(n_in: int, n_out: int, width: int = 128, depth: int = 4) -> nn.Module:
    layers: list[nn.Module] = []
    d = n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def train(store: str | Path, n_in: int, out: str | Path, epochs: int = 60, batch: int = 256, buffer: int = 8192, lr: float = 2e-3, width: int = 128, depth: int = 4,
          val_shards: int = 1, seed: int = 0) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rd = ArrayReader(store)
    ns = len(rd.man["shards"])
    if ns < 2:
        raise ValueError("need at least two shards (one is held out for validation)")
    order = list(np.random.default_rng(seed).permutation(ns))
    val, tr = sorted(order[:val_shards]), sorted(order[val_shards:])
    mean, std = streaming_stats(rd, tr)
    n_out = int(np.prod(rd.shape)) - n_in
    torch.manual_seed(seed)
    net = mlp(n_in, n_out, width, depth)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ck = out / "checkpoint.pt"
    start, hist = 0, []
    if ck.exists():
        s = torch.load(ck, weights_only=False)
        net.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])  # noqa: E702
        start, hist = s["epoch"], s["history"]
    mu, sd = torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
    sub_tr, sub_val = _Sub(rd, tr), _Sub(rd, val)
    t0 = time.time()
    for ep in range(start, epochs):
        net.train()
        tot, nb = 0.0, 0
        for b in stream(sub_tr, batch, buffer, seed=seed + ep):
            z = (torch.from_numpy(b.reshape(len(b), -1)) - mu) / sd
            loss = nn.functional.mse_loss(net(z[:, :n_in]), z[:, n_in:])
            opt.zero_grad(); loss.backward(); opt.step()  # noqa: E702
            tot += float(loss.detach()); nb += 1  # noqa: E702
        sched.step()
        v = evaluate(net, sub_val, n_in, mu, sd)
        hist.append({"epoch": ep + 1, "train_mse": tot / max(nb, 1), "val_mse": v["mse"], "val_r2_mean": float(np.mean(v["r2"]))})
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1, "history": hist}, ck)
    v = evaluate(net, sub_val, n_in, mu, sd)
    torch.save({"model": net.state_dict(), "mean": mean, "std": std, "n_in": n_in, "width": width, "depth": depth}, out / "surrogate.pt")
    rep = {"rows": len(rd), "train_shards": len(tr), "val_shards": len(val), "val_rows": len(sub_val), "epochs": epochs, "seconds": time.time() - t0, "history": hist,
           "val_r2": v["r2"].tolist(), "val_rel_rmse": v["rel_rmse"].tolist(), "buffer_rows": buffer, "row_bytes": int(4 * np.prod(rd.shape)),
           "val_shard_ids": [int(k) for k in val], "train_shard_ids": [int(k) for k in tr]}
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def evaluate(net: nn.Module, rd: ArrayReader, n_in: int, mu: torch.Tensor, sd: torch.Tensor) -> dict[str, Any]:
    net.eval()
    ys, ps = [], []
    with torch.no_grad():
        for k in range(len(rd.man["shards"])):
            a = torch.from_numpy(np.array(rd.shard(k)).reshape(len(rd.shard(k)), -1))  # copy out of the read-only memory map
            z = (a - mu) / sd
            ps.append((net(z[:, :n_in]) * sd[n_in:] + mu[n_in:]).numpy())
            ys.append(a[:, n_in:].numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    r2 = 1 - ((y - p) ** 2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-12)
    rel = np.sqrt(((y - p) ** 2).mean(0)) / np.maximum(np.abs(y).mean(0), 1e-12)
    return {"mse": float((((y - p) / sd[n_in:].numpy()) ** 2).mean()), "r2": r2, "rel_rmse": rel}
