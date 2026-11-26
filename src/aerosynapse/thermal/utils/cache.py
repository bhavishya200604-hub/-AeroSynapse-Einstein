"""Disk cache for expensive, deterministic results (Step 6.6: solver solutions are stored on the SSD and reused).

cached(cache_dir, group, key, fn) returns fn() and stores it under <cache_dir>/<group>/<sha1(key)>.pkl; the next call
with the same key loads the file instead of recomputing. The key is any JSON-serialisable dict that fully determines
the result (parameters, grid, tolerances, code version tag). Files are plain pickles of dicts of numpy arrays /
numbers — the cache directory is local and git-ignored; delete it to force recomputation.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any


def key_hash(key: dict) -> str:
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def cached(cache_dir: str | Path, group: str, key: dict, fn: Callable[[], Any]) -> tuple[Any, bool]:
    """(result, hit): hit = True when the result came from the cache."""
    d = Path(cache_dir) / group
    f = d / f"{key_hash(key)}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)["result"], True
    res = fn()
    d.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"key": key, "result": res}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(f)  # atomic: a crash never leaves a half-written entry
    return res, False
