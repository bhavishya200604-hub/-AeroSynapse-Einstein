"""Background jobs for the workbench: progress, cancellation, result/exception capture."""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

_POOL = ThreadPoolExecutor(max_workers=2)


@dataclass
class Job:
    name: str
    progress: float = 0.0
    message: str = "queued"
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future[Any] | None = None
    error: str | None = None

    def report(self, msg: str, frac: float) -> None:
        self.message, self.progress = msg, max(0.0, min(1.0, frac))

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def done(self) -> bool:
        return self.future is not None and self.future.done()

    def result(self) -> Any:
        assert self.future is not None
        return self.future.result()


def submit(name: str, fn: Callable[[Job], Any]) -> Job:
    job = Job(name)

    def wrapped() -> Any:
        try:
            return fn(job)
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
            raise

    job.future = _POOL.submit(wrapped)
    return job
