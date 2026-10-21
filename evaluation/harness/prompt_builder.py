"""Prompt construction and answer parsing for F / P / R tasks.

Only the final answer is locked with `<ANSWER>...</ANSWER>`; the model reasons
freely, so the L2 judge sees a natural trace. SSA prefill is inserted before
the question as verified context and does not impose a reasoning template.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any

from .data_loader import Task
from .input_conditions import ConditionContext

ANSWER_LOCK_F = (
    "Answer briefly. If you include explanation, keep it very short and avoid "
    "repetitive self-debate. On the final line, you must give your chosen "
    "option letter in the exact form <ANSWER>X</ANSWER> where X is one of "
    "{A, B, C, D}. You must pick one option even if uncertain. Do not use "
    "stage labels or other structured XML tags."
)
ANSWER_LOCK_P = (
    "Give only the final answer on the last line in the exact form "
    "<ANSWER>...</ANSWER>."
)
ANSWER_LOCK_R = (
    "State the rule briefly. On the last line, give the final answer in the "
    "exact form <ANSWER>...</ANSWER>."
)


# --- Image ----------------------------------------------------------------


def image_data_url(path: Path, *, max_side: int = 768, quality: int = 80) -> str:
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        scale = min(1.0, max_side / max(img.size))
        if scale < 1.0:
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# --- Prompt text ----------------------------------------------------------


def build_prompt_text(task: Task, ctx: ConditionContext) -> str:
    extra = ""
    if ctx.prefill:
        extra = (
            "[The following intermediate results have been externally verified as "
            "correct and may be trusted directly, but they do not contain the "
            "final option.]\n"
            f"{ctx.prefill.strip()}\n\n"
        )
    if task.task_type == "F":
        return (
            "You are an evaluation model for visual reasoning multiple-choice tasks.\n"
            f"{extra}"
            f"Question: {task.question}\n"
            "Options: A / B / C / D\n\n"
            f"{ANSWER_LOCK_F}"
        )
    if task.task_type == "R":
        return (
            "You are an evaluation model for rule-induction probes. Answer the rule "
            "question below using only the known visible figures. Do not solve the "
            "full F task.\n"
            f"{extra}"
            f"Rule question: {task.question}\n\n"
            f"{ANSWER_LOCK_R}"
        )
    if task.task_type == "P":
        return f"{extra}Question: {task.question}\n\n{ANSWER_LOCK_P}"
    raise ValueError(f"Unknown task_type: {task.task_type}")


def build_messages(
    task: Task,
    ctx: ConditionContext,
    *,
    max_side: int = 768,
    jpeg_quality: int = 80,
    encode_image: bool = True,
) -> list[dict[str, Any]]:
    """Build an OpenAI-style chat payload.

    With `encode_image=False` the image is referenced by path instead of
    base64-encoded, which keeps dry runs fast.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": build_prompt_text(task, ctx)}]
    if ctx.include_image:
        if encode_image:
            url = image_data_url(Path(task.image_path), max_side=max_side, quality=jpeg_quality)
            content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            content.append({"type": "image_ref", "path": str(task.image_path)})
    return [{"role": "user", "content": content}]


# --- Output parsing -------------------------------------------------------

_ANSWER_TAG = re.compile(r"<\s*ANSWER\s*>\s*(.+?)\s*<\s*/\s*ANSWER\s*>", re.IGNORECASE | re.DOTALL)
_STANDALONE_LETTER = re.compile(r"(?<![A-Z])([A-D])(?![A-Z])")
_FALLBACK_PATTERNS = (
    re.compile(r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:|=)?\s*\(?([A-D])\)?(?![A-Z])"),
    re.compile(r"(?:答案|选项|选择|应选|故选)\s*(?:是|为|:|：)?\s*([A-D])"),
)


def extract_answer(raw: str) -> str:
    """Return the content of the last <ANSWER>...</ANSWER>, or "" if none."""
    matches = _ANSWER_TAG.findall(raw or "")
    return matches[-1].strip() if matches else ""


def extract_answer_letter(raw: str) -> str:
    """Return the chosen option letter A-D, or "" if none can be found."""
    content = extract_answer(raw).upper()
    m = _STANDALONE_LETTER.search(content)
    if m:
        return m.group(1)
    # Fallback for responses without the tag: only accept a letter attached to
    # an answer phrase near the end, never a bare A-D mention in the reasoning.
    tail = (raw or "")[-1000:].upper()
    for pattern in _FALLBACK_PATTERNS:
        found = pattern.findall(tail)
        if found:
            return found[-1]
    return ""


def strip_answer_tag(raw: str) -> str:
    """Return the trace with the <ANSWER>...</ANSWER> tags removed."""
    return _ANSWER_TAG.sub("", raw or "").strip()
