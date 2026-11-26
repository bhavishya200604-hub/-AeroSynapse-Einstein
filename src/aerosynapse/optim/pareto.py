"""Pareto utilities: non-dominated sorting, crowding distance, exact hypervolume (2D) and 3D slicing."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]


def dominates(a: FloatArr, b: FloatArr) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(F: FloatArr) -> list[list[int]]:
    n = len(F)
    S: list[list[int]] = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                cnt[p] += 1
        if cnt[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(F: FloatArr) -> FloatArr:
    n, m = F.shape
    d = np.zeros(n)
    for j in range(m):
        o = np.argsort(F[:, j])
        d[o[0]] = d[o[-1]] = np.inf
        span = F[o[-1], j] - F[o[0], j] or 1.0
        d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / span
    return d


def hypervolume_2d(F: FloatArr, ref: FloatArr) -> float:
    P = F[np.all(ref > F, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return float(hv)


def hypervolume_3d(F: FloatArr, ref: FloatArr) -> float:
    """Exact HV in 3D by slicing along f3 (O(n² log n))."""
    P = F[np.all(ref > F, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 2])]
    hv = 0.0
    for i in range(len(P)):
        z_next = P[i + 1, 2] if i + 1 < len(P) else ref[2]
        hv += hypervolume_2d(P[: i + 1, :2], ref[:2]) * (z_next - P[i, 2])
    return float(hv)
