"""Prompt construction for F / P / R tasks.

Prompt rules enforced here:
  * L2 StepAcc never forces `<S1>...</S1>` output. We only lock the final
    answer letter with `<ANSWER>X</ANSWER>`; the model reasons freely.
  * Four input conditions are handled by first running `apply_condition` then
    passing the resulting `ConditionContext` here.
  * SSA (Exp4) adds H0-H4 prefill blocks as input-side context. Prefill is
    inserted before the question but does NOT impose a reasoning template.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any

from .data_loader import Task
from .input_conditions import ConditionContext

ANSWER_LOCK_F_R = (
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
ANSWER_LOCK_R_OPEN = (
    "State the rule briefly. On the last line, give the final answer in the "
    "exact form <ANSWER>...</ANSWER>."
)


# --- Image helpers --------------------------------------------------------


def image_data_url(path: Path, *, max_side: int = 768, quality: int = 80) -> str:
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        scale = min(1.0, max_side / max(img.size))
        if scale < 1.0:
            new_size = (
                max(1, int(img.width * scale)),
                max(1, int(img.height * scale)),
            )
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# --- Prompt text ----------------------------------------------------------


def _f_text(task: Task, extra: str) -> str:
    return (
        "You are an evaluation model for visual reasoning multiple-choice tasks.\n"
        f"{extra}"
        f"Question: {task.question}\n"
        "Options: A / B / C / D\n\n"
        f"{ANSWER_LOCK_F_R}"
    )


def _r_text(task: Task, extra: str) -> str:
    if not task.choices:
        return (
            "You are an evaluation model for rule-induction probes. Answer the rule "
            "question below using only the known visible figures. Do not solve the "
            "full F task.\n"
            f"{extra}"
            f"Rule question: {task.question}\n\n"
            f"{ANSWER_LOCK_R_OPEN}"
        )
    choices_block = "\n".join(
        f"{chr(65+i)}. {c}" for i, c in enumerate(task.choices)
    )
    return (
        "You are an evaluation model for rule-induction probes. Answer the rule "
        "question below using only the known visible figures. Do not solve the "
        "full F task.\n"
        f"{extra}"
        f"Rule question: {task.question}\n"
        f"Options:\n{choices_block}\n\n"
        f"{ANSWER_LOCK_F_R}"
    )


def _p_text(task: Task, extra: str) -> str:
    return f"{extra}Question: {task.question}\n\n{ANSWER_LOCK_P}"


def build_prompt_text(task: Task, ctx: ConditionContext, *, ssa_prefill: str = "") -> str:
    extra = ""
    if ssa_prefill:
        extra += (
            "[The following intermediate results have been externally verified as "
            "correct and may be trusted directly, but they do not contain the "
            "final option.]\n"
            f"{ssa_prefill.strip()}\n\n"
        )
    if ctx.extra_text:
        extra += ctx.extra_text + "\n"
    if task.task_type == "F":
        return _f_text(task, extra)
    if task.task_type == "R":
        return _r_text(task, extra)
    if task.task_type == "P":
        return _p_text(task, extra)
    raise ValueError(f"Unknown task_type: {task.task_type}")


def build_messages(
    task: Task,
    ctx: ConditionContext,
    *,
    ssa_prefill: str = "",
    max_side: int = 768,
    jpeg_quality: int = 80,
    encode_image: bool = True,
) -> list[dict[str, Any]]:
    """Build OpenAI-chat messages payload.

    `encode_image=False` returns a messages list with `{"image_ref": str(path)}`
    placeholder instead of a base64 url – useful for tests and `--dry-run`.
    """
    text = build_prompt_text(task, ctx, ssa_prefill=ssa_prefill)
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if ctx.include_image:
        if encode_image:
            url = image_data_url(Path(task.image_path), max_side=max_side, quality=jpeg_quality)
            content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            content.append({"type": "image_ref", "path": str(task.image_path)})
    return [{"role": "user", "content": content}]


# --- Output parsing -------------------------------------------------------

_ANSWER_TAG = re.compile(r"<\s*ANSWER\s*>\s*(.+?)\s*<\s*/\s*ANSWER\s*>", re.IGNORECASE | re.DOTALL)
_LETTER = re.compile(r"^[A-D]$")


def extract_answer(raw: str) -> str:
    """Return the content of the LAST <ANSWER>…</ANSWER>, or "" if none."""
    if not raw:
        return ""
    matches = _ANSWER_TAG.findall(raw)
    if not matches:
        return ""
    return matches[-1].strip()


def extract_answer_letter(raw: str) -> str:
    """Coerce the answer tag content into a single A/B/C/D letter."""
    content = extract_answer(raw).upper()
    if _LETTER.match(content):
        return content
    m = re.search(r"[A-D]", content)
    if m:
        return m.group(0)

    # Conservative fallback for models that ignore the answer tag. Only accept
    # letters that are explicitly attached to an answer/choice phrase near the
    # end of the response; do not pick arbitrary A-D mentions from reasoning.
    tail = raw[-1000:].upper()
    patterns = [
        r"(?:最终答案|答案|正确答案|选择|选项|应选|所以选|因此选|故选)\s*(?:是|为|:|：)?\s*([A-D])",
        r"([A-D])\s*(?:项|选项)\s*(?:正确|符合|最合适|最可能)?",
    ]
    for pat in patterns:
        match = re.search(pat, tail)
        if match:
            return match.group(1)
    return ""


def strip_answer_tag(raw: str) -> str:
    """Return the trace with the final <ANSWER>…</ANSWER> removed."""
    if not raw:
        return ""
    return _ANSWER_TAG.sub("", raw).strip()
