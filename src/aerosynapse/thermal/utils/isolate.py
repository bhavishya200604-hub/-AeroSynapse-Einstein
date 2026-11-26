"""Run training tasks in fresh processes (D57).

A training result can depend on what ran earlier in the same process (round-off in memory-layout-dependent kernels,
amplified by training: R4.3p). With ``execution.isolate: true`` (default) every task runs in its OWN freshly spawned
interpreter, so each published number is reproducible in isolation, whatever ran before it; ``workers > 1`` runs that
many tasks in parallel (for machines with more cores; ``workers: auto`` = physical cores // threads_per_worker).

A task is ``(module, function, kwargs)``: the function is imported by name in the child, so it must be a top-level
function of an importable module (e.g. ``aerosynapse.thermal.experiments.p4_3_balancing.train_case``) and its arguments and return value
must be picklable.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections.abc import Callable
from multiprocessing.connection import wait
from typing import Any

from aerosynapse.thermal.utils.experiment import ROOT

Task = tuple[str, str, dict[str, Any]]


def _init(threads: int, path: list[str]) -> None:
    for p in reversed(path):
        if p not in sys.path:
            sys.path.insert(0, p)
    if threads > 0:
        import torch

        torch.set_num_threads(threads)


def _call(module: str, func: str, kwargs: dict[str, Any]) -> Any:
    return getattr(importlib.import_module(module), func)(**kwargs)


def n_workers(execution: dict) -> int:
    w = execution.get("workers", 1)
    if w == "auto":
        cores = max((os.cpu_count() or 2) // 2, 1)  # logical -> physical (2-way SMT assumed)
        t = int(execution.get("threads_per_worker", 0) or 1)
        return max(cores // t, 1)
    return int(w)


def run_tasks(
    tasks: list[Task], execution: dict, on_done: Callable[[int, Any, float], None] | None = None
) -> list[Any]:
    """Run tasks; results in task order. ``on_done(index, result, seconds)`` is called as each task finishes."""
    results: list[Any] = [None] * len(tasks)
    if not execution.get("isolate", True):
        for i, (m, f, kw) in enumerate(tasks):
            t0 = time.perf_counter()
            results[i] = _call(m, f, kw)
            if on_done:
                on_done(i, results[i], time.perf_counter() - t0)
        return results
    threads = int(execution.get("threads_per_worker", 0) or 0)
    path = [str(ROOT)] + [p for p in sys.path if p]
    ctx = mp.get_context("spawn")
    # One spawned process per task, started and collected by THIS thread only (D91): a fresh interpreter for every
    # task, no worker pool. (A shared pool with max_tasks_per_child=1 hung / crashed re-spawning workers at shutdown
    # on CPython 3.12.8; starting pools from several threads gave a child an invalid handle, WinError 6.)
    pending = list(enumerate(tasks))
    running: dict[Any, tuple[int, Any, float]] = {}
    try:
        while pending or running:
            while pending and len(running) < n_workers(execution):
                i, (m, f, kw) = pending.pop(0)
                r, w = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_child, args=(w, threads, path, m, f, kw), daemon=True)
                proc.start()
                w.close()  # the child holds the write end; EOF on r if it dies without a result
                running[r] = (i, proc, time.perf_counter())
            for r in wait(list(running)):
                i, proc, t0 = running.pop(r)
                try:
                    ok, payload = r.recv()
                except EOFError:
                    ok, payload = False, None
                r.close()
                proc.join()
                if not ok:
                    detail = payload or f"worker exited with code {proc.exitcode} without a result"
                    raise RuntimeError(f"task {i} ({tasks[i][0]}.{tasks[i][1]}) failed:\n{detail}")
                results[i] = payload
                if on_done:
                    on_done(i, payload, time.perf_counter() - t0)
    finally:
        for r, (_, proc, _) in running.items():  # only after a failure: stop the other tasks
            proc.terminate()
            proc.join(5.0)
            r.close()
    return results


def _child(conn: Any, threads: int, path: list[str], module: str, func: str, kwargs: dict[str, Any]) -> None:
    """Entry point of a task process: run the task, send (True, result) or (False, traceback text)."""
    try:
        _init(threads, path)
        out = (True, _call(module, func, kwargs))
    except BaseException:  # noqa: BLE001  (reported to the parent, which raises)
        out = (False, traceback.format_exc())
    conn.send(out)
    conn.close()
