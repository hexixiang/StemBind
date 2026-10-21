"""StemCoherent: per-stem P ∧ R ∧ F joint 0/1 scoring.

A model scores 1 on an item iff the F answer, the R probe,
and ALL of the item's P probes are simultaneously correct. This penalises
lucky F guesses and rewards models that are coherent end-to-end on a stem.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from ._utils import wilson_ci


def _item_ok(f_ok: bool | None, r_ok: bool | None, p_all_ok: bool) -> int | None:
    """Return 1 / 0 / None (incomplete: missing probe of any type)."""
    if f_ok is None or r_ok is None:
        return None
    return 1 if (f_ok and r_ok and p_all_ok) else 0


def stem_coherent_by_model(
    predictions: Iterable[dict[str, Any]],
    *,
    condition: str | None = "full-image",
) -> list[dict[str, Any]]:
    """Per-model StemCoherent. Filters by condition if supplied (default full-image)."""
    by_model_item: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {
        "F": None, "R": None, "P": [],
    })
    for r in predictions:
        if condition and r.get("condition") not in (condition, None):
            continue
        key = (r.get("model"), r.get("item_id"))
        slot = by_model_item[key]
        t = r.get("task_type")
        if t == "F":
            slot["F"] = bool(r.get("correct"))
        elif t == "R":
            slot["R"] = bool(r.get("correct"))
        elif t == "P":
            slot["P"].append(bool(r.get("correct")))

    by_model: dict[str, list[int]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    for (model, item_id), slot in by_model_item.items():
        if not slot["P"]:
            skipped[model] += 1
            continue
        p_all = all(slot["P"])
        res = _item_ok(slot["F"], slot["R"], p_all)
        if res is None:
            skipped[model] += 1
            continue
        by_model[model].append(res)

    out = []
    for model in sorted(by_model):
        vals = by_model[model]
        n, k = len(vals), sum(vals)
        lo, hi = wilson_ci(k, n)
        out.append({
            "model": model,
            "n_items": n,
            "stem_coherent_correct": k,
            "stem_coherent": (k / n) if n else 0.0,
            "ci95_low": lo,
            "ci95_high": hi,
            "items_skipped_incomplete": skipped[model],
        })
    return out
