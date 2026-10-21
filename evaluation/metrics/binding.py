"""Binding evidence from Section 4.2 of the paper.

  * R-F gap: Acc(R) - Acc(F) per model.
  * Stem-level failure decomposition. Each stem falls into exactly one of
      solved        F correct
      can't bind    F wrong, R correct, at least one P probe correct
      can't reason  F wrong, R wrong, at least one P probe correct
      can't see     F wrong, no P probe correct
  * Strict Binding Gap: BG = 1 - P(F=1 | P=1, R=1), where P=1 means every P
    probe of the stem is correct. Reported per model (bootstrap 95% CI over
    stems) and pooled over all models.

Only stems with an F, an R, and at least one P prediction are used.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from ._utils import bootstrap_ci

CATEGORIES = ("solved", "cant_bind", "cant_reason", "cant_see")


def _stems(predictions: Iterable[dict[str, Any]], condition: str) -> dict[str, list[dict[str, Any]]]:
    """Group predictions into per-model lists of complete stems."""
    slots: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"F": None, "R": None, "P": []})
    for r in predictions:
        if r.get("condition") != condition:
            continue
        slot = slots[(r.get("model"), r.get("item_id"))]
        ok = bool(r.get("correct"))
        if r.get("task_type") == "P":
            slot["P"].append(ok)
        elif r.get("task_type") in ("F", "R"):
            slot[r["task_type"]] = ok
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model, _item), slot in slots.items():
        if slot["F"] is not None and slot["R"] is not None and slot["P"]:
            by_model[model].append(slot)
    return by_model


def _category(stem: dict[str, Any]) -> str:
    if stem["F"]:
        return "solved"
    if not any(stem["P"]):
        return "cant_see"
    return "cant_bind" if stem["R"] else "cant_reason"


def _binding_gap(stems: list[dict[str, Any]], *, n_boot: int) -> dict[str, Any]:
    f_given_pr = [1.0 if s["F"] else 0.0 for s in stems if s["R"] and all(s["P"])]
    if not f_given_pr:
        return {"n_pr": 0, "binding_gap": None, "ci95_low": None, "ci95_high": None}
    mean, lo, hi = bootstrap_ci(f_given_pr, n_boot=n_boot)
    return {"n_pr": len(f_given_pr), "binding_gap": 1.0 - mean, "ci95_low": 1.0 - hi, "ci95_high": 1.0 - lo}


def binding_summary(
    predictions: Iterable[dict[str, Any]],
    *,
    condition: str = "full-image",
    n_boot: int = 1000,
) -> dict[str, Any]:
    preds = list(predictions)
    acc: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in preds:
        if r.get("condition") == condition and r.get("task_type") in ("F", "R"):
            acc[(r.get("model"), r["task_type"])].append(int(bool(r.get("correct"))))

    stems = _stems(preds, condition)
    rows = []
    for model in sorted({m for m, _ in acc} | set(stems)):
        f, r = acc.get((model, "F"), []), acc.get((model, "R"), [])
        counts = {c: 0 for c in CATEGORIES}
        for s in stems.get(model, []):
            counts[_category(s)] += 1
        n = sum(counts.values())
        rows.append({
            "model": model,
            "acc_F": sum(f) / len(f) if f else None,
            "acc_R": sum(r) / len(r) if r else None,
            "rf_gap": (sum(r) / len(r) - sum(f) / len(f)) if f and r else None,
            "n_stems": n,
            "decomposition": {c: counts[c] / n if n else 0.0 for c in CATEGORIES},
            **_binding_gap(stems.get(model, []), n_boot=n_boot),
        })
    pooled = _binding_gap([s for v in stems.values() for s in v], n_boot=n_boot)
    return {"condition": condition, "by_model": rows, "pooled": pooled}
