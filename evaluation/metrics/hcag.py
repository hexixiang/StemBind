"""HCAG: Human-Calibrated Ability Gap.

    HCAG_m = sqrt( Σ_a  w_a * ( max(0, μ_H,a - A_m,a) / (σ_H,a + ε) )^2 )

Inputs:
  * `human_norms`: dict {operation: {"mu": float, "sigma": float, "weight": float}}.
    When mu/sigma are unknown we return "[TBD]" strings rather than invent numbers
    (No fabricated results.)
  * `predictions`: rows with `model`, `operation`, `task_type == "F"`, `correct`.

If weights are omitted, equal weights are used (1 / n_ops).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from ._utils import TBD

EPSILON = 1e-6


def _model_acc_per_op(predictions: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in predictions:
        if r.get("task_type") != "F":
            continue
        counts[(r.get("model"), r.get("operation"))].append(1 if r.get("correct") else 0)
    per_model: dict[str, dict[str, float]] = defaultdict(dict)
    for (model, op), vals in counts.items():
        if vals and op:
            per_model[model][op] = sum(vals) / len(vals)
    return per_model


def hcag_by_model(
    predictions: Iterable[dict[str, Any]],
    *,
    human_norms: dict[str, dict[str, float]] | None,
) -> list[dict[str, Any]]:
    acc = _model_acc_per_op(predictions)
    out: list[dict[str, Any]] = []
    if not human_norms:
        for model in sorted(acc):
            out.append({"model": model, "hcag": TBD, "reason": "human_norms missing"})
        return out

    ops = sorted(human_norms)
    default_w = 1.0 / len(ops) if ops else 0.0
    for model in sorted(acc):
        gaps: list[float] = []
        missing_ops: list[str] = []
        for op in ops:
            norm = human_norms[op]
            mu, sigma = norm.get("mu"), norm.get("sigma")
            if mu is None or sigma is None:
                missing_ops.append(op)
                continue
            w = float(norm.get("weight", default_w))
            a = acc[model].get(op)
            if a is None:
                missing_ops.append(op)
                continue
            gap = max(0.0, float(mu) - float(a)) / (float(sigma) + EPSILON)
            gaps.append(w * gap * gap)
        if missing_ops or not gaps:
            out.append({
                "model": model,
                "hcag": TBD,
                "reason": f"missing_norms_or_preds: {missing_ops}",
            })
        else:
            out.append({"model": model, "hcag": math.sqrt(sum(gaps))})
    return out
