"""L1 accuracy: exact answer match + Wilson 95% CI, sliced by various axes."""

from __future__ import annotations

from typing import Any, Iterable

from ._utils import group_by, wilson_ci


def _acc_row(key: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if bool(r.get("correct")))
    lo, hi = wilson_ci(k, n)
    return {
        **key,
        "n": n,
        "correct": k,
        "acc": (k / n) if n else 0.0,
        "ci95_low": lo,
        "ci95_high": hi,
        "errors": sum(1 for r in rows if r.get("error")),
    }


def accuracy_by(
    predictions: Iterable[dict[str, Any]],
    *,
    by: tuple[str, ...] = ("model", "task_type"),
) -> list[dict[str, Any]]:
    groups = group_by(predictions, *by)
    rows = []
    for key_vals, sub in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        key = {name: val for name, val in zip(by, key_vals)}
        rows.append(_acc_row(key, sub))
    return rows


def accuracy_summary(predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    preds = list(predictions)
    return {
        "overall": _acc_row({"scope": "overall"}, preds),
        "by_model_task": accuracy_by(preds, by=("model", "task_type")),
        "by_model_operation_task": accuracy_by(preds, by=("model", "operation", "task_type")),
        "by_model_condition_task": accuracy_by(preds, by=("model", "condition", "task_type")),
    }
