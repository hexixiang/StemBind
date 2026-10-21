"""Vision Gain (VG) and Image-Blind Shortcut Index (IBSI).

    VG_m  = Acc(full-image) - Acc(text-only)
    IBSI_m = chance_normalized(text-only) / chance_normalized(full-image)

    chance_normalized(acc, k) = (acc - 1/k) / (1 - 1/k)     (k = n options)

Negative chance_normalized values are clamped to 0 before the ratio; if the
full-image chance-normalised acc is ≤ 0 we report "[TBD]" rather than produce
a misleading Inf.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from ._utils import TBD


def _acc_by_model_condition(
    predictions: Iterable[dict[str, Any]],
    *,
    task_type: str = "F",
) -> dict[tuple[str, str], float]:
    agg: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in predictions:
        if r.get("task_type") != task_type:
            continue
        agg[(r.get("model"), r.get("condition"))].append(1 if r.get("correct") else 0)
    return {k: (sum(v) / len(v)) if v else 0.0 for k, v in agg.items()}


def _chance_norm(acc: float, k: int) -> float:
    if k <= 1:
        return acc
    return (acc - 1.0 / k) / (1.0 - 1.0 / k)


def vg_by_model(
    predictions: Iterable[dict[str, Any]],
    *,
    num_options: int = 4,
) -> list[dict[str, Any]]:
    accs = _acc_by_model_condition(predictions)
    models = sorted({m for (m, _c) in accs})
    out = []
    for m in models:
        full = accs.get((m, "full-image"))
        text = accs.get((m, "text-only"))
        row: dict[str, Any] = {
            "model": m,
            "acc_full_image": full if full is not None else TBD,
            "acc_text_only": text if text is not None else TBD,
        }
        if full is None or text is None:
            row["vg"] = TBD
            row["ibsi"] = TBD
        else:
            row["vg"] = full - text
            cn_full = max(0.0, _chance_norm(full, num_options))
            cn_text = max(0.0, _chance_norm(text, num_options))
            row["ibsi"] = (cn_text / cn_full) if cn_full > 0 else TBD
        out.append(row)
    return out
