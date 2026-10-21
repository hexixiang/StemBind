"""ThinkGain paired analysis.

ThinkGain@X_g = Acc(thinking_g, X) - Acc(nonthinking_g, X),   X ∈ {P, R, F}

Pairs (g) are declared by caller (e.g. G1..G5). The module also reports the
Go/No-Go verdict:

    Pass: (ThinkGain@R - ThinkGain@P) >= 5 pp
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

TASK_LEVELS = ("P", "R", "F")


def _acc(preds: Iterable[dict[str, Any]], model: str, task_type: str) -> tuple[float, int]:
    k = n = 0
    for r in preds:
        if r.get("model") != model or r.get("task_type") != task_type:
            continue
        n += 1
        if r.get("correct"):
            k += 1
    return ((k / n) if n else 0.0), n


def thinkgain(
    predictions: Iterable[dict[str, Any]],
    *,
    pairs: Mapping[str, tuple[str, str]],
    go_threshold_pp: float = 5.0,
) -> dict[str, Any]:
    preds = list(predictions)
    rows = []
    for group, (baseline, thinking) in pairs.items():
        entry: dict[str, Any] = {
            "group": group,
            "baseline": baseline,
            "thinking": thinking,
        }
        for t in TASK_LEVELS:
            acc_base, n_base = _acc(preds, baseline, t)
            acc_think, n_think = _acc(preds, thinking, t)
            entry[f"acc_base_{t}"] = acc_base
            entry[f"acc_think_{t}"] = acc_think
            entry[f"n_{t}"] = min(n_base, n_think)
            entry[f"thinkgain_{t}_pp"] = 100.0 * (acc_think - acc_base)
        r_minus_p = entry["thinkgain_R_pp"] - entry["thinkgain_P_pp"]
        entry["r_minus_p_pp"] = r_minus_p
        entry["verdict"] = "GO" if r_minus_p >= go_threshold_pp else "NO-GO"
        rows.append(entry)
    return {
        "pairs": rows,
        "go_threshold_pp": go_threshold_pp,
        "overall_verdict": (
            "GO" if any(r["verdict"] == "GO" for r in rows) else "NO-GO"
        ),
    }
