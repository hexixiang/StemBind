"""L3 AttrTag: map (first-failure stage, perception type) → attribution label.

Rule table:

    S1 + Type-P  → 感知瓶颈 (perception_bottleneck)
    S1 + Type-R  → 意外感知失败 (unexpected_perception_fail)
    S2 + Type-R  → 关系发现瓶颈 (relation_bottleneck)
    S3 + Type-R  → 结构迁移瓶颈 (mapping_bottleneck)
    S4 (any)     → 规则执行瓶颈 (application_bottleneck)
    any + Type-PR → 混合归因 (mixed_attribution)

If `perception_type` is unknown we label it `unknown_perception_type` and
leave the stage suffix for downstream inspection.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

PERCEPTION_P, PERCEPTION_R, PERCEPTION_PR = "Type-P", "Type-R", "Type-PR"


def classify(first_failure: str, perception_type: str | None) -> str:
    if not first_failure:
        return "no_failure"
    if perception_type == PERCEPTION_PR:
        return f"mixed_attribution/{first_failure}"
    if first_failure == "S4":
        return "application_bottleneck"
    if first_failure == "S1" and perception_type == PERCEPTION_P:
        return "perception_bottleneck"
    if first_failure == "S1" and perception_type == PERCEPTION_R:
        return "unexpected_perception_fail"
    if first_failure == "S2" and perception_type == PERCEPTION_R:
        return "relation_bottleneck"
    if first_failure == "S3" and perception_type == PERCEPTION_R:
        return "mapping_bottleneck"
    if perception_type is None:
        return f"unknown_perception_type/{first_failure}"
    return f"other/{first_failure}_{perception_type}"


def attrtag_rows(first_failure_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in first_failure_rows:
        out.append({
            **r,
            "attr_tag": classify(r.get("first_failure", ""), r.get("perception_type")),
        })
    return out


def attrtag_summary(first_failure_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(first_failure_rows)
    tagged = attrtag_rows(rows)
    by_model: dict[str, Counter] = defaultdict(Counter)
    for r in tagged:
        by_model[r.get("model", "unknown")][r["attr_tag"]] += 1
    return {
        "tagged": tagged,
        "by_model": {m: dict(c) for m, c in by_model.items()},
    }
