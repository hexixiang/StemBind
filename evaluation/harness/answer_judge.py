"""LLM-as-judge answer correctness for F / P / R rows.

This judge compares a candidate final answer against the reference answer. It is
separate from the L2 S1-S4 judge, which evaluates reasoning-stage alignment only
for F rows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .data_loader import Task
from .model_router import ModelCaller
from .prompt_builder import extract_answer
from .runner import score_l1


ANSWER_JUDGE_SYSTEM = (
    "You are the StemBind answer-equivalence judge. Your task is not to solve "
    "the problem again and not to reason from the image. You only decide "
    "whether the candidate response's final answer is equivalent to the "
    "reference answer. Use semantic equivalence when appropriate, but do not "
    "over-infer missing content. Whenever the final answer is unclear or "
    "ambiguous, default to correct=0. Output JSON only."
)

ANSWER_JUDGE_SCHEMA = (
    '{"correct":0,"confidence":"high|medium|low","rationale":"brief English reason"}'
)


@dataclass
class AnswerJudgeConfig:
    model: str
    temperature: float = 0.0
    timeout: int = 120
    extra_payload: dict[str, Any] | None = None


@dataclass
class AnswerJudgeResult:
    correct: int
    confidence: str
    rationale: str
    raw: str = ""
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "correct": int(self.correct),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "raw": self.raw,
            "error": self.error,
        }


def build_answer_judge_prompt(task: Task, row: dict[str, Any]) -> list[dict[str, Any]]:
    choices = ""
    if task.choices:
        choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(task.choices))

    reference = task.gold
    if task.task_type == "R" and task.gold_index >= 0:
        reference = f"{chr(65 + task.gold_index)}. {task.gold}"

    candidate_answer = judge_candidate_answer(task, row)

    user = (
        "Determine whether the candidate response's final answer is correct.\n\n"
        "[Task Type]\n"
        "{task_type}\n\n"
        "[Question]\n"
        "{question}\n\n"
        "[Options]\n"
        "{choices}\n\n"
        "[Reference Answer]\n"
        "{reference}\n\n"
        "[Program-Extracted Candidate Answer]\n"
        "{prediction}\n\n"
        "[Full Candidate Response]\n"
        "===BEGIN RESPONSE===\n"
        "{raw_text}\n"
        "===END RESPONSE===\n\n"
        "[Judging Rules]\n"
        "1. Only decide whether the candidate response's final answer is equivalent to the reference answer. Do not solve the problem again and do not infer a new answer from the question.\n"
        "2. If the candidate response contains <ANSWER>...</ANSWER>, the last <ANSWER> block has highest priority. Otherwise use the clearly stated final answer near the end of the response. If the final answer cannot be determined, set correct=0.\n"
        "3. F multiple-choice tasks: the candidate answer is correct if it clearly gives the same option letter. If it gives only text and no letter, it may still be correct if that text clearly matches the content of the correct option and does not also fit another option.\n"
        "4. R rule probes: accept semantically equivalent descriptions of the correct rule. Different wording, bilingual paraphrases, and minor surface variation are acceptable if the rule meaning is clearly the same. Do not accept a broader, weaker, or only partially overlapping rule.\n"
        "5. P perception short-answer tasks: accept clear semantic equivalence. Allow equivalent number forms (for example 2, two, 2条), concise bilingual equivalents, and common literal variants for shape, color, direction, count, or position. Do not accept partial matches, vague supersets, ambiguous hypernyms, or answers that leave the key attribute unspecified.\n"
        "6. Do not give credit for the amount of explanation or formatting quality. Judge only final-answer equivalence.\n"
        "7. If the candidate response contains multiple conflicting final answers, set correct=0.\n"
        "8. If the candidate answer is missing, blank, off-topic, vague, or only weakly related to the reference answer, set correct=0.\n"
        "9. Use semantic equivalence, but do not over-infer. If you must guess what the model meant, set correct=0.\n"
        "10. confidence must be one of high/medium/low. rationale must be one short English sentence, ideally under 20 words.\n\n"
        "[Output Format]\nOutput strict JSON only. No Markdown, no code fences, and no extra text. The format must be: {schema}"
    ).format(
        task_type=task.task_type,
        question=task.question,
        choices=choices or "[none]",
        reference=reference,
        prediction=candidate_answer,
        raw_text=judge_visible_response(row),
        schema=ANSWER_JUDGE_SCHEMA,
    )
    return [
        {"role": "system", "content": ANSWER_JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


_THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tag(text: str) -> str:
    return _THINK_TAG.sub("", text).strip()


def judge_visible_response(row: dict[str, Any]) -> str:
    """Return the text that answer-judge should inspect.

    Newer Qwen-style responses may place visible chain-of-thought in a separate
    `reasoning_content` field. In that case, `raw_text` should already contain
    only the final answer content, so prefer it directly. The legacy
    `<think>...</think>` cleanup remains as a fallback for older cached runs.
    """
    return strip_think_tag(str(row.get("raw_text", "")))


def judge_candidate_answer(task: Task, row: dict[str, Any]) -> str:
    """Return the answer string shown to the answer judge.

    For F, keep the parser-extracted answer because the task is strict MCQ.
    For P/R, if no <ANSWER> tag is present, fall back to the full visible
    response so the judge can score natural-language answers directly.
    """
    prediction = str(row.get("prediction", "") or "").strip()
    visible = judge_visible_response(row)

    if task.task_type == "F":
        return prediction

    tagged = extract_answer(visible)
    if tagged:
        return tagged
    if prediction:
        return prediction
    return visible


_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_answer_judge_output(raw: str) -> AnswerJudgeResult:
    obj: dict[str, Any] = {}
    s = (raw or "").strip()
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001
        for m in _JSON_BLOCK.findall(s):
            try:
                obj = json.loads(m)
                break
            except Exception:  # noqa: BLE001
                continue

    try:
        correct = 1 if int(obj.get("correct", 0)) == 1 else 0
    except (TypeError, ValueError):
        correct = 0
    confidence = str(obj.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    rationale = str(obj.get("rationale", "")).strip()[:400]
    return AnswerJudgeResult(correct=correct, confidence=confidence, rationale=rationale, raw=raw)


class AnswerJudge:
    def __init__(self, caller: ModelCaller, config: AnswerJudgeConfig):
        self.caller = caller
        self.cfg = config

    def judge(self, *, task: Task, row: dict[str, Any]) -> AnswerJudgeResult:
        messages = build_answer_judge_prompt(task, row)
        raw = self.caller(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            timeout=self.cfg.timeout,
            extra=self.cfg.extra_payload or None,
        )
        return parse_answer_judge_output(raw.final_text)


def exact_fallback_result(task: Task, row: dict[str, Any]) -> AnswerJudgeResult:
    correct, _prediction = score_l1(task, row.get("raw_text", ""))
    return AnswerJudgeResult(
        correct=1 if correct else 0,
        confidence="medium",
        rationale="dry-run exact fallback",
        raw="",
    )
