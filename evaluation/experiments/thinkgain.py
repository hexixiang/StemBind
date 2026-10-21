"""ThinkGain: paired direct-vs-thinking analysis.

    ThinkGain@X = Acc(thinking, X) - Acc(direct, X),   X in {P, R, F}

Pairs are given by the caller as (direct model name, thinking model name).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

TASK_LEVELS = ("P", "R", "F")


def _acc(preds: list[dict[str, Any]], model: str, task_type: str) -> tuple[float, int]:
    vals = [bool(r.get("correct")) for r in preds if r.get("model") == model and r.get("task_type") == task_type]
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)


def thinkgain(
    predictions: Iterable[dict[str, Any]],
    *,
    pairs: Mapping[str, tuple[str, str]],
    condition: str = "full-image",
) -> dict[str, Any]:
    preds = [r for r in predictions if r.get("condition") == condition]
    rows = []
    for group, (direct, thinking) in pairs.items():
        entry: dict[str, Any] = {"group": group, "direct": direct, "thinking": thinking}
        for t in TASK_LEVELS:
            acc_d, n_d = _acc(preds, direct, t)
            acc_t, n_t = _acc(preds, thinking, t)
            entry[f"acc_direct_{t}"] = acc_d
            entry[f"acc_thinking_{t}"] = acc_t
            entry[f"n_{t}"] = min(n_d, n_t)
            entry[f"thinkgain_{t}_pp"] = 100.0 * (acc_t - acc_d)
        rows.append(entry)
    return {"pairs": rows}
