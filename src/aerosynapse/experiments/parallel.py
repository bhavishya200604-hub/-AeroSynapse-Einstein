"""Run independent jobs in worker processes (one torch thread each) for multi-seed studies."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def _init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass


def run_parallel(fn: Callable[[Any], Any], jobs: Iterable[Any], workers: int | None = None) -> list[Any]:
    jobs = list(jobs)
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 1)))
    if workers == 1:
        _init()
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        return list(ex.map(fn, jobs))
