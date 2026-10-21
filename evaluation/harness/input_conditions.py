"""Input-condition logic for ablations.

Four conditions:
  - text-only     : question text only, no image, no caption, no parse
  - caption-only  : natural-language caption appended to the question, no image
  - parse-only    : structured symbolic parse appended to the question, no image
  - full-image    : original image + question (the default)

Downstream consumers (prompt_builder, runner) read the returned dict to decide
whether to attach the image and what extra context block to inject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Condition = Literal["text-only", "caption-only", "parse-only", "full-image"]

VALID_CONDITIONS: tuple[Condition, ...] = (
    "text-only",
    "caption-only",
    "parse-only",
    "full-image",
)


@dataclass
class ConditionContext:
    condition: Condition
    include_image: bool
    extra_text: str = ""          # prepended context block (caption / parse / "")
    note: str = ""                # human-readable note for logging


def _ensure_valid(condition: str) -> Condition:
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Unknown input condition {condition!r}; "
            f"expected one of {VALID_CONDITIONS}"
        )
    return condition  # type: ignore[return-value]


def apply_condition(
    *,
    condition: str,
    caption: str | None = None,
    parse: str | None = None,
) -> ConditionContext:
    """Resolve condition -> (include_image, extra_text) for a single task."""
    c = _ensure_valid(condition)
    if c == "full-image":
        return ConditionContext(c, include_image=True, extra_text="", note="image")
    if c == "text-only":
        return ConditionContext(c, include_image=False, extra_text="", note="no-context")
    if c == "caption-only":
        if not caption:
            # Missing caption is a hard fail; caller must provide it for this
            # condition. Upstream loader should filter out items without it.
            raise ValueError("caption-only condition requires a caption but got None")
        return ConditionContext(
            c,
            include_image=False,
            extra_text=f"[以下是该题的文字描述（代替图像）]\n{caption.strip()}\n",
            note="caption",
        )
    # parse-only
    if not parse:
        raise ValueError("parse-only condition requires a parse but got None")
    return ConditionContext(
        c,
        include_image=False,
        extra_text=f"[以下是该题的结构化解析（代替图像）]\n{parse.strip()}\n",
        note="parse",
    )
