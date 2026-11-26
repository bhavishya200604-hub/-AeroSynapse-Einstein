"""Model serialisation (safe: ``weights_only=True``) and determinism helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_model(model: torch.nn.Module, path: str | Path, meta: dict[str, Any] | None = None) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, Path(path))


def load_model(model: torch.nn.Module, path: str | Path) -> dict[str, Any]:
    """Load weights into an already-constructed ``model``; returns the metadata."""
    obj = torch.load(Path(path), weights_only=True)
    model.load_state_dict(obj["state_dict"])
    return dict(obj.get("meta", {}))


def save_checkpoint(
    path: str | Path, model: torch.nn.Module, opt: torch.optim.Optimizer, step: int, rng_state: Any = None
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "torch_rng": torch.get_rng_state(),
            "extra_rng": rng_state,
        },
        Path(path),
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, opt: torch.optim.Optimizer) -> int:
    obj = torch.load(Path(path), weights_only=False)  # own checkpoints only (contains RNG state)
    model.load_state_dict(obj["model"])
    opt.load_state_dict(obj["opt"])
    torch.set_rng_state(obj["torch_rng"])
    return int(obj["step"])
