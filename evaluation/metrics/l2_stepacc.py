"""L2 StepAcc: per-stage accuracy on F items as judged against S1–S4 GT.

Inputs are predictions joined with a judge result per-row. The expected shape
(added by the judge pipeline, e.g. runner + judge.Judge):

    {
      ...runner.Prediction fields...,
      "judge": {
        "stages": {"S1": 0/1, "S2": 0/1, "S3": 0/1, "S4": 0/1},
        "first_failure": "S1" | ... | "",
      }
    }

We compute:
  * Stage pass rate per (model, stage)
  * First-failure distribution per model
  * Step Consistency (SCS): fraction of rows where a later stage is 1 while an
    earlier stage is 0.
  * Lucky-Guess Rate (LGR): F answer correct but any stage 0.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

STAGE_KEYS = ("S1", "S2", "S3", "S4")


def _judge_stages(row: dict[str, Any]) -> dict[str, int] | None:
    j = row.get("judge") or {}
    stages = j.get("stages")
    if not isinstance(stages, dict):
        return None
    return {k: 1 if int(stages.get(k, 0)) == 1 else 0 for k in STAGE_KEYS}


def stepacc_summary(predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in predictions if r.get("task_type") == "F"]
    per_model: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "n": 0, "stages": {k: 0 for k in STAGE_KEYS}, "first_failure": Counter(),
        "inconsistencies": 0, "lgr": 0,
    })
    for r in rows:
        stages = _judge_stages(r)
        if stages is None:
            continue
        model = r.get("model", "unknown")
        agg = per_model[model]
        agg["n"] += 1
        for k in STAGE_KEYS:
            agg["stages"][k] += stages[k]
        # first failure
        ff = ""
        for k in STAGE_KEYS:
            if stages[k] == 0:
                ff = k
                break
        agg["first_failure"][ff or "none"] += 1
        # inconsistency: any later=1 after earlier=0
        seen_zero = False
        for k in STAGE_KEYS:
            if stages[k] == 0:
                seen_zero = True
            elif seen_zero:
                agg["inconsistencies"] += 1
                break
        # LGR: F correct but any stage wrong
        if r.get("correct") and any(stages[k] == 0 for k in STAGE_KEYS):
            agg["lgr"] += 1

    out: list[dict[str, Any]] = []
    for model, agg in sorted(per_model.items()):
        n = agg["n"]
        row = {
            "model": model,
            "n": n,
            **{f"stepacc_{k}": (agg["stages"][k] / n if n else 0.0) for k in STAGE_KEYS},
            "scs_inconsistency_rate": (agg["inconsistencies"] / n) if n else 0.0,
            "lgr": (agg["lgr"] / n) if n else 0.0,
            "first_failure_dist": dict(agg["first_failure"]),
        }
        out.append(row)
    return {"by_model": out}


def first_failure_rows(predictions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten first-failure stage per F row, for L3 downstream."""
    out: list[dict[str, Any]] = []
    for r in predictions:
        if r.get("task_type") != "F":
            continue
        stages = _judge_stages(r)
        if stages is None:
            continue
        ff = ""
        for k in STAGE_KEYS:
            if stages[k] == 0:
                ff = k
                break
        out.append({
            "model": r.get("model"),
            "item_id": r.get("item_id"),
            "operation": r.get("operation"),
            "perception_type": r.get("perception_type"),
            "first_failure": ff,
            "f_correct": bool(r.get("correct")),
        })
    return out
