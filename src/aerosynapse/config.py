"""Pydantic-validated configuration schema and YAML loading.

Profiles: ``quick`` (CI / laptop), ``standard`` and ``full`` (Colab-scale). Every experiment
config carries a seed and a profile; unknown keys are rejected (``extra="forbid"``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from aerosynapse.exceptions import ConfigurationError

Profile = Literal["quick", "standard", "full"]


class StrictModel(BaseModel):
    """Base model: forbids unknown keys so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class SolverConfig(StrictModel):
    name: str = "convdiff_fv"
    nx: int = Field(128, ge=4)
    nt: int | None = Field(None, ge=1)
    scheme: str = "upwind"
    time_integrator: str = "crank_nicolson"
    cfl: float = Field(0.5, gt=0)


class ModelConfig(StrictModel):
    name: str = "pinn"
    width: int = Field(32, ge=1)
    depth: int = Field(3, ge=1)
    activation: str = "tanh"
    fourier_features: int = Field(0, ge=0)
    fourier_scale: float = Field(1.0, gt=0)


class TrainingConfig(StrictModel):
    optimizer: str = "adam"
    lr: float = Field(1e-3, gt=0)
    iterations: int = Field(2000, ge=1)
    n_collocation: int = Field(2000, ge=1)
    sampler: str = "uniform"
    resample_every: int = Field(500, ge=1)
    lbfgs_iterations: int = Field(0, ge=0)


class ExperimentConfig(StrictModel):
    """Top-level experiment configuration."""

    experiment_id: str
    research_question: str = ""
    hypothesis: str = ""
    profile: Profile = "quick"
    seed: int = 0
    problem: dict[str, Any] = Field(default_factory=dict)
    solver: SolverConfig = Field(default_factory=lambda: SolverConfig())
    model: ModelConfig = Field(default_factory=lambda: ModelConfig())
    training: TrainingConfig = Field(default_factory=lambda: TrainingConfig())
    optimization: dict[str, Any] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=lambda: ["rel_l2", "linf"])


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    p = Path(path)
    if not p.is_file():
        raise ConfigurationError(f"config file not found: {p}")
    with p.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return ExperimentConfig.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError
        raise ConfigurationError(f"invalid config {p}: {exc}") from exc


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write a configuration back to YAML (round-trip safe)."""
    Path(path).write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")
