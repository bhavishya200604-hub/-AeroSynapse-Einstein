"""Sharded datasets on disk.

* ``ShardWriter``: JSON-lines records, a new shard every ``max_records``; ``close()`` writes the manifest.
* ``ArrayWriter``: fixed-shape float32 samples (fields, volumes) appended to raw binary shards; each shard
  has its count, shape, bytes and SHA-256 in the manifest, so a reader can check what it reads.
* ``records()`` / ``ArrayReader``: read back one shard at a time (``np.memmap`` for arrays), so memory
  stays at one shard plus the shuffle buffer whatever the dataset size.
* ``stream()``: shuffled mini-batches from shards in random order through a bounded buffer.

Writers are append-only and crash-safe at shard granularity: a shard is listed in the manifest only after
it was closed, and an interrupted run continues with the next shard number.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[3]


def store_root() -> Path:
    p = Path(os.environ.get("AEROSYNAPSE_STORE") or ROOT / "data_store")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(d: Path) -> dict[str, Any]:
    p = d / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"shards": []}


def _save_manifest(d: Path, m: dict[str, Any]) -> None:
    tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(m, indent=1), encoding="utf-8")
    os.replace(tmp, d / "manifest.json")


class ShardWriter:
    def __init__(self, root: str | Path, name: str, fields: list[str] | None = None, max_records: int = 2000) -> None:
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.man = _manifest(self.dir)
        self.man.setdefault("kind", "records")
        self.man["fields"] = fields or self.man.get("fields", [])
        self.max = max_records
        self.n = 0
        self.fh: Any = None
        self.path: Path | None = None

    def _open(self) -> None:
        k = len(self.man["shards"])
        self.path = self.dir / f"shard-{k:05d}.jsonl"
        while self.path.exists() and any(s["file"] == self.path.name for s in self.man["shards"]):
            k += 1
            self.path = self.dir / f"shard-{k:05d}.jsonl"
        self.fh = open(self.path, "w", encoding="utf-8")  # noqa: SIM115  (a shard stays open until it is full)
        self.n = 0

    def add(self, rec: dict[str, Any]) -> None:
        if self.fh is None:
            self._open()
        self.fh.write(json.dumps(rec, default=float) + "\n")
        self.n += 1
        if self.n >= self.max:
            self._close_shard()

    def _close_shard(self) -> None:
        if self.fh is None or self.path is None:
            return
        self.fh.close()
        self.man["shards"].append({"file": self.path.name, "records": self.n, "bytes": self.path.stat().st_size, "sha256": _sha(self.path)})
        _save_manifest(self.dir, self.man)
        self.fh = None

    def close(self) -> None:
        if self.fh is not None and self.n:
            self._close_shard()
        elif self.fh is not None:
            self.fh.close()
            if self.path is not None:
                self.path.unlink(missing_ok=True)
            self.fh = None


def records(path: str | Path, verify: bool = False) -> Iterator[dict[str, Any]]:
    d = Path(path)
    for s in _manifest(d)["shards"]:
        f = d / s["file"]
        if verify and _sha(f) != s["sha256"]:
            raise OSError(f"checksum mismatch in {f}")
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


class ArrayWriter:
    """Append float32 samples of a fixed shape; ``max_bytes`` per shard (default 256 MB)."""

    def __init__(self, root: str | Path, name: str, shape: tuple[int, ...], max_bytes: int = 256 << 20, meta: dict[str, Any] | None = None) -> None:
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.man = _manifest(self.dir)
        if self.man["shards"] and tuple(self.man["shape"]) != tuple(shape):
            raise ValueError(f"{self.dir} holds samples of shape {self.man['shape']}, not {shape}")
        self.man.update({"kind": "arrays", "shape": list(shape), "dtype": "float32", **(meta or {})})
        self.per = max(1, max_bytes // (4 * int(np.prod(shape))))
        self.fh: Any = None
        self.n = 0
        self.path: Path | None = None

    def add(self, x: NDArray[np.floating[Any]]) -> None:
        a = np.ascontiguousarray(x, np.float32)
        if a.shape != tuple(self.man["shape"]):
            raise ValueError(f"sample shape {a.shape} differs from {tuple(self.man['shape'])}")
        if self.fh is None:
            self.path = self.dir / f"shard-{len(self.man['shards']):05d}.bin"
            self.fh = open(self.path, "wb")  # noqa: SIM115  (a shard stays open until it is full)
            self.n = 0
        self.fh.write(a.tobytes())
        self.n += 1
        if self.n >= self.per:
            self._close_shard()

    def _close_shard(self) -> None:
        if self.fh is None or self.path is None:
            return
        self.fh.close()
        self.man["shards"].append({"file": self.path.name, "samples": self.n, "bytes": self.path.stat().st_size, "sha256": _sha(self.path)})
        _save_manifest(self.dir, self.man)
        self.fh = None

    def close(self) -> None:
        if self.fh is not None and self.n:
            self._close_shard()


class ArrayReader:
    def __init__(self, path: str | Path) -> None:
        self.dir = Path(path)
        self.man = _manifest(self.dir)
        self.shape = tuple(self.man["shape"])

    def __len__(self) -> int:
        return sum(s["samples"] for s in self.man["shards"])

    def shard(self, k: int) -> np.memmap[Any, np.dtype[np.float32]]:
        s = self.man["shards"][k]
        return np.memmap(self.dir / s["file"], np.float32, "r", shape=(s["samples"], *self.shape))

    def verify(self) -> bool:
        return all(_sha(self.dir / s["file"]) == s["sha256"] for s in self.man["shards"])


def stream(reader: ArrayReader, batch: int, buffer: int = 4096, seed: int = 0, epochs: int = 1) -> Iterator[NDArray[np.float32]]:
    """Shuffled mini-batches: shards in random order, samples mixed through a buffer of ``buffer`` samples."""
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        buf: list[NDArray[np.float32]] = []
        for k in rng.permutation(len(reader.man["shards"])):
            m = reader.shard(int(k))
            for i in rng.permutation(len(m)):
                buf.append(np.array(m[i]))
                if len(buf) >= buffer:
                    j = rng.permutation(len(buf))
                    take = [buf[t] for t in j[:batch]]
                    buf = [buf[t] for t in j[batch:]]
                    yield np.stack(take)
            del m
        rng.shuffle(buf)
        for s in range(0, len(buf), batch):
            yield np.stack(buf[s : s + batch])
