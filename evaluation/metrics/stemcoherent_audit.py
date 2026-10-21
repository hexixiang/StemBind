"""StemCoherent information-gain audit.

Four checks:
  1. rank_change        – top-N model rank change vs Acc
  2. spearman_acc_sc    – Spearman correlation between Acc and StemCoherent
  3. chance_baseline    – expected StemCoherent if every probe were independent
                          at the model's F accuracy (Acc ** (#P probes + 2))
  4. lucky_guess_reduction – Acc - StemCoherent in percentage points
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from ._utils import spearman


def _acc_by_model(predictions: Iterable[dict[str, Any]], task_type: str = "F") -> dict[str, float]:
    agg: dict[str, list[int]] = defaultdict(list)
    for r in predictions:
        if r.get("task_type") != task_type:
            continue
        agg[r.get("model", "unknown")].append(1 if r.get("correct") else 0)
    return {m: sum(v) / len(v) if v else 0.0 for m, v in agg.items()}


def _avg_probes_per_item(predictions: Iterable[dict[str, Any]]) -> float:
    by_item: dict[str, int] = defaultdict(int)
    for r in predictions:
        if r.get("task_type") == "P":
            by_item[r.get("item_id", "")] += 1
    if not by_item:
        return 0.0
    return sum(by_item.values()) / len(by_item)


def _rank(d: dict[str, float]) -> dict[str, int]:
    ordered = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
    return {m: i + 1 for i, (m, _v) in enumerate(ordered)}


def audit(
    *,
    predictions: Iterable[dict[str, Any]],
    stem_coherent_rows: list[dict[str, Any]],
    top_n: int = 10,
) -> dict[str, Any]:
    preds = list(predictions)
    acc_f = _acc_by_model(preds, "F")
    sc_map = {r["model"]: float(r.get("stem_coherent", 0.0)) for r in stem_coherent_rows}
    common = sorted(set(acc_f) & set(sc_map))
    if not common:
        return {
            "n_models": 0,
            "spearman_acc_sc": 0.0,
            "rank_change_top_n": 0.0,
            "chance_baseline": {},
            "lucky_guess_reduction_pp": {},
            "top_n": top_n,
            "note": "no overlapping models",
        }

    x = [acc_f[m] for m in common]
    y = [sc_map[m] for m in common]
    rho = spearman(x, y)

    # rank change on top-N
    acc_rank = _rank(acc_f)
    sc_rank = _rank(sc_map)
    top_acc_models = [m for m, _ in sorted(acc_rank.items(), key=lambda kv: kv[1])[:top_n]]
    changes = sum(
        1 for m in top_acc_models if abs(acc_rank[m] - sc_rank.get(m, acc_rank[m])) >= 1
    )
    rank_change_rate = changes / len(top_acc_models) if top_acc_models else 0.0

    # chance baseline: Acc^(k + 1) where k = avg P probes per item (+1 for R)
    avg_probes = _avg_probes_per_item(preds)
    exponent = 1 + 1 + max(0.0, avg_probes)  # F + R + P-all-correct
    chance = {m: (acc_f[m] ** exponent) for m in common}
    lucky = {m: (acc_f[m] - sc_map[m]) * 100.0 for m in common}

    return {
        "n_models": len(common),
        "top_n": top_n,
        "spearman_acc_sc": rho,
        "rank_change_top_n": rank_change_rate,
        "chance_baseline": chance,
        "lucky_guess_reduction_pp": lucky,
        "avg_probes_per_item": avg_probes,
    }
