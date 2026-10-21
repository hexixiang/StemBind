"""SSA (Stage-wise Stimulus Augmentation) scoring.

Input: F predictions run under conditions H0..H4 and H1_irrelevant (see
`harness/input_conditions.py`). For each model this reports

    raw accuracy per level, with bootstrap 95% CI over items,
    chance-normalised accuracy  g = (acc - 1/K) / (1 - 1/K),  K = 4 options,
    an isotonic (non-decreasing) fit over H0 <= ... <= H4 via pool-adjacent-violators,
    the per-step gains of that fit, and the irrelevant-S1 control vs. H0.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from ..metrics._utils import TBD, bootstrap_ci

H_KEYS = ("H0", "H1", "H2", "H3", "H4")
IRRELEVANT_KEY = "H1_irrelevant"
K_OPTIONS = 4


def _chance_norm(acc: float, k: int = K_OPTIONS) -> float:
    return (acc - 1.0 / k) / (1.0 - 1.0 / k)


def _isotonic_nondecreasing(values: list[float]) -> list[float]:
    """Pool-adjacent-violators fit with equal weights."""
    blocks: list[list[float]] = []  # [mean, size]
    for v in values:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (m1, w1), (m2, w2) = blocks[-2], blocks.pop()
            blocks[-1] = [(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2]
    return [m for m, w in blocks for _ in range(int(w))]


def _per_model_acc(predictions: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in predictions:
        if r.get("task_type") != "F":
            continue
        bucket[(r.get("model"), r.get("condition"))].append(1 if r.get("correct") else 0)
    return bucket


def skyline(predictions: Iterable[dict[str, Any]], *, n_boot: int = 1000) -> dict[str, Any]:
    bucket = _per_model_acc(predictions)
    rows = []
    for m in sorted({m for (m, _h) in bucket}):
        raw_acc, ci95 = {}, {}
        for h in H_KEYS:
            mean, lo, hi = bootstrap_ci([float(v) for v in bucket.get((m, h), [])], n_boot=n_boot)
            raw_acc[h], ci95[h] = mean, {"low": lo, "high": hi}
        cn = [_chance_norm(raw_acc[h]) for h in H_KEYS]
        iso = _isotonic_nondecreasing(cn)
        irrelevant = bucket.get((m, IRRELEVANT_KEY), [])
        irrelevant_acc = sum(irrelevant) / len(irrelevant) if irrelevant else None
        rows.append({
            "model": m,
            "raw_acc": raw_acc,
            "ci95": ci95,
            "chance_normalised": dict(zip(H_KEYS, cn)),
            "isotonic": dict(zip(H_KEYS, iso)),
            "isotonic_gain_pp": {
                f"{H_KEYS[i - 1]}->{H_KEYS[i]}": 100.0 * (iso[i] - iso[i - 1]) for i in range(1, len(iso))
            },
            "irrelevant_control": {
                "acc": irrelevant_acc if irrelevant_acc is not None else TBD,
                "delta_vs_H0_pp": (
                    100.0 * (_chance_norm(irrelevant_acc) - cn[0]) if irrelevant_acc is not None else TBD
                ),
            },
        })
    return {"by_model": rows}
