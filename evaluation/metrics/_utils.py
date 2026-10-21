"""Internal helpers shared across metric modules."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 1000,
    seed: int = 20260417,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(s) / n)
    samples.sort()
    lo = samples[int(alpha / 2 * n_boot)]
    hi = samples[int((1 - alpha / 2) * n_boot) - 1]
    mean = sum(values) / n
    return mean, lo, hi


def group_by(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple, list[dict[str, Any]]]:
    g: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[tuple(r.get(k) for k in keys)].append(r)
    return g


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    def _ranks(a: list[float]) -> list[float]:
        idx = sorted(range(len(a)), key=lambda i: a[i])
        ranks = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[idx[j + 1]] == a[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based average rank
            for k in range(i, j + 1):
                ranks[idx[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


TBD = "[TBD]"
