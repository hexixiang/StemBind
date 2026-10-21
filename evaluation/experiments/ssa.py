"""Stage Skyline Analysis.

Given per-(model, item, condition H0..H4) correctness, compute the isotonic
skyline + per-step delta with bootstrap CI. The six guards are enforced
upstream (prefill schema audit, irrelevant-control, position randomisation,
subset exclusion for single-stage items). This module only does scoring.

Condition keys expected in prediction rows:
    condition ∈ {"H0","H1","H2","H3","H4","H1_irrelevant"}

Raw chance-normalised accuracy:

    g_i,m,h = (y - 1/K) / (1 - 1/K)    (K = 4 choices on F items)
    A_m,h   = mean_i(g_i,m,h)

Isotonic projection (monotone non-decreasing H0 ≤ H1 ≤ H2 ≤ H3 ≤ H4) uses the
pool-adjacent-violators (PAV) algorithm.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable

H_KEYS = ("H0", "H1", "H2", "H3", "H4")
IRRELEVANT_KEY = "H1_irrelevant"
K_OPTIONS = 4


def _chance_norm(acc: float, k: int = K_OPTIONS) -> float:
    if k <= 1:
        return acc
    return (acc - 1.0 / k) / (1.0 - 1.0 / k)


def _isotonic_nondecreasing(values: list[float]) -> list[float]:
    """Pool-Adjacent-Violators for a non-decreasing fit."""
    if not values:
        return []
    v = list(values)
    w = [1.0] * len(v)
    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1]:
            # pool
            total = v[i] * w[i] + v[i + 1] * w[i + 1]
            wt = w[i] + w[i + 1]
            pooled = total / wt
            v[i] = pooled
            v[i + 1 : i + 2] = []
            w[i] = wt
            w[i + 1 : i + 2] = []
            if i > 0:
                i -= 1
        else:
            i += 1
    # expand pooled means back (simple repeat by original count)
    expanded: list[float] = []
    for val, wt in zip(v, w):
        expanded.extend([val] * int(wt))
    return expanded[: len(values)]


def _per_model_acc(predictions: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in predictions:
        if r.get("task_type") != "F":
            continue
        bucket[(r.get("model"), r.get("condition"))].append(1 if r.get("correct") else 0)
    return bucket


def _bootstrap_ci(vals: list[int], *, seed: int, n_boot: int = 1000) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def skyline(
    predictions: Iterable[dict[str, Any]],
    *,
    seed: int = 20260417,
    n_boot: int = 1000,
) -> dict[str, Any]:
    bucket = _per_model_acc(predictions)
    models = sorted({m for (m, _h) in bucket})
    rows = []
    for m in models:
        raw_acc = {h: (sum(vals) / len(vals) if vals else 0.0) for h in H_KEYS
                   for vals in [bucket.get((m, h), [])]}
        cn = [_chance_norm(raw_acc[h]) for h in H_KEYS]
        iso = _isotonic_nondecreasing(cn)
        cis = {}
        for h_i, h in enumerate(H_KEYS):
            vals = bucket.get((m, h), [])
            cis[h] = _bootstrap_ci(vals, seed=seed + h_i * 1009, n_boot=n_boot)
        deltas_pp = [100.0 * (iso[i] - iso[i - 1]) for i in range(1, len(iso))]
        irrelevant_vals = bucket.get((m, IRRELEVANT_KEY), [])
        irrelevant_acc = sum(irrelevant_vals) / len(irrelevant_vals) if irrelevant_vals else None
        row = {
            "model": m,
            "raw_acc": raw_acc,
            "chance_normalised": {h: cn[i] for i, h in enumerate(H_KEYS)},
            "isotonic_skyline": {h: iso[i] for i, h in enumerate(H_KEYS)},
            "skyline_deltas_pp": {f"H{i}->H{i+1}": deltas_pp[i] for i in range(len(deltas_pp))},
            "ci95": {h: {"low": cis[h][0], "high": cis[h][1]} for h in H_KEYS},
            "irrelevant_control": {
                "acc": irrelevant_acc if irrelevant_acc is not None else "[TBD]",
                "delta_vs_H0_pp": (
                    100.0 * (_chance_norm(irrelevant_acc) - cn[0])
                    if irrelevant_acc is not None else "[TBD]"
                ),
            },
            "h1_jump_pp": deltas_pp[0] if deltas_pp else 0.0,
            "h1_gate_pass": (
                deltas_pp[0] >= 10.0
                and (
                    irrelevant_acc is None
                    or (100.0 * (_chance_norm(irrelevant_acc) - cn[0])) <= 3.0
                )
            ) if deltas_pp else False,
        }
        rows.append(row)
    return {"by_model": rows, "H_keys": list(H_KEYS)}
