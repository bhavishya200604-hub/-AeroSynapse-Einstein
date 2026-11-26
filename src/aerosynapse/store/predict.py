"""Instant performance estimate of a fighter design from the surrogate trained on the data store.

The surrogate covers designs from the freeform family of :mod:`aerosynapse.fighter.design` (a design file written
by the optimiser carries its ``variables``). It reports only the outputs it predicts well on held-out data and
says so; the others need the full model (``aerosynapse fighter`` or :func:`aerosynapse.fighter.performance.evaluate`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "results" / "surrogate" / "report.json"
MODEL = ROOT / "results" / "surrogate" / "surrogate.pt"
GOOD_R2 = 0.7


def predict(design: dict[str, Any]) -> dict[str, Any]:
    import torch

    from aerosynapse.fighter import design as D
    from aerosynapse.store.train import mlp

    if "variables" not in design:
        raise ValueError("this design has no design-space variables (it was drawn by hand): run the full model or the 3D flow solver instead")
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    ck = torch.load(MODEL, weights_only=False)
    v = D.complete("freeform", {k: float(x) for k, x in design["variables"].items() if isinstance(x, (int, float))})
    z = D.encode("freeform", v)
    n_in = int(ck["n_in"])
    net = mlp(n_in, len(ck["mean"]) - n_in, ck["width"], ck["depth"])
    net.load_state_dict(ck["model"])
    net.eval()
    mu, sd = ck["mean"], ck["std"]
    x = (np.asarray(z, float) - mu[:n_in]) / sd[:n_in]
    with torch.no_grad():
        y = net(torch.tensor(x[None], dtype=torch.float32)).numpy()[0] * sd[n_in:] + mu[n_in:]
    out: dict[str, Any] = {}
    for name, val, r2 in zip(rep["outputs"], y, rep["val_r2"], strict=True):
        if name in rep.get("log_outputs", []):
            val = float(np.exp(val))
        out[name] = {"value": float(val), "held_out_r2": float(r2), "trusted": bool(r2 >= GOOD_R2)}
    return out


def report(design_path: str) -> str:
    p = predict(json.loads(Path(design_path).read_text(encoding="utf-8")))
    lines = ["Surrogate estimate (network trained on 1,117 designs streamed from the data store)", ""]
    for k, v in p.items():
        mark = "" if v["trusted"] else "   (not reliable: held-out R2 below 0.7, use the full model)"
        lines.append(f"{k:26s} {v['value']:12.4g}   R2 {v['held_out_r2']:.2f}{mark}")
    return "\n".join(lines)
