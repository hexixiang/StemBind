"""Input conditions.

  - full-image     : image + question (default)
  - text-only      : question only, for the VG image-dependence check
  - H0 .. H4       : SSA (Stage-wise Stimulus Augmentation) on F items. Each
                     level adds one verified stage cumulatively: H1 = S1,
                     H2 = S1-S2, H3 = S1-S3, H4 = S1-S4. Option letters are
                     masked so no condition reveals the answer.
  - H1_irrelevant  : SSA control; S1 of a different stem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .data_loader import STEP_KEYS, Task

SSA_CONDITIONS = ("H0", "H1", "H2", "H3", "H4", "H1_irrelevant")
VALID_CONDITIONS = ("full-image", "text-only", *SSA_CONDITIONS)

_STAGE_LABELS = ("S1 Encode", "S2 Infer", "S3 Map", "S4 Apply")
_OPTION_LETTER = re.compile(r"(?<![A-Za-z])(?:(?i:options?|choices?)\s+)?\(?[A-D]\)?(?![A-Za-z])")


@dataclass
class ConditionContext:
    condition: str
    include_image: bool
    prefill: str = ""


def mask_option_letters(text: str) -> str:
    return _OPTION_LETTER.sub("[option]", text)


def ssa_prefill(task: Task, condition: str) -> str:
    if condition == "H1_irrelevant":
        stages = [(_STAGE_LABELS[0], task.irrelevant_s1)]
    else:
        level = int(condition[1])
        stages = [(label, task.gt_steps.get(key, "")) for key, label in zip(STEP_KEYS, _STAGE_LABELS)][:level]
    return "\n".join(f"{label}: {mask_option_letters(text)}" for label, text in stages)


def apply_condition(task: Task, condition: str) -> ConditionContext:
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Unknown input condition {condition!r}; expected one of {VALID_CONDITIONS}")
    if condition == "text-only":
        return ConditionContext(condition, include_image=False)
    if condition == "full-image":
        return ConditionContext(condition, include_image=True)
    return ConditionContext(condition, include_image=True, prefill=ssa_prefill(task, condition))
