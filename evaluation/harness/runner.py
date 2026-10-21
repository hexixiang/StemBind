"""Run models over tasks with a per-call cache.

Each (model, task, condition) result is written to
`<out_dir>/cache/<model>__<task_id>__<condition>.json`, so an interrupted run
resumes without repeating finished calls. Transient API errors are not cached
and are retried on the next run. All results go to `<out_dir>/predictions.jsonl`.

L1 correctness (exact option letter for F, normalized text match for P / R) is
computed inline; `answer-judge` later replaces it with LLM-judged correctness
for free-form answers.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .data_loader import Task
from .input_conditions import SSA_CONDITIONS, VALID_CONDITIONS, apply_condition
from .model_router import ModelCaller
from .prompt_builder import build_messages, extract_answer, extract_answer_letter, strip_answer_tag


# --- L1 scoring -----------------------------------------------------------


def _clean(text: Any) -> str:
    s = re.sub(r"\s+", "", str(text or "").strip().lower())
    return re.sub(r"[，。、“”‘’；;：:,.!?？！（）()\[\]{}<>《》|/\\_-]", "", s)


def match_p_answer(pred: str, gold: str) -> bool:
    p, g = _clean(pred), _clean(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    pn, gn = re.findall(r"\d+", p), re.findall(r"\d+", g)
    if pn and gn:
        return pn == gn
    return (len(g) <= 8 and g in p) or (len(p) <= 8 and p in g)


def score_l1(task: Task, raw_text: str) -> tuple[bool, str]:
    if task.task_type == "F":
        letter = extract_answer_letter(raw_text)
        return letter == task.gold, letter
    pred = extract_answer(raw_text) or raw_text.strip()
    if task.task_type == "P":
        return match_p_answer(pred, task.gold), pred
    return _clean(pred) == _clean(task.gold), pred


# --- Prediction record ----------------------------------------------------


@dataclass
class Prediction:
    task_id: str
    item_id: str
    task_type: str
    operation: str
    perception_load: str | None
    model: str
    condition: str
    gold: str
    prediction: str
    correct: bool
    raw_text: str
    trace: str
    reasoning_content: str = ""
    error: str | None = None


_TRANSIENT_ERRORS = (
    "HTTP Error 408", "HTTP Error 409", "HTTP Error 425", "HTTP Error 429",
    "HTTP Error 500", "HTTP Error 502", "HTTP Error 503", "HTTP Error 504",
    "rate limit", "Rate limit", "timed out", "Connection reset", "Connection refused",
    "Connection aborted", "Remote end closed connection", "urlopen error", "SSL:",
    "EOF occurred", "IncompleteRead",
)


def _is_transient_error(error: str) -> bool:
    return any(marker in error for marker in _TRANSIENT_ERRORS)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


# --- Runner ---------------------------------------------------------------


@dataclass
class RunConfig:
    models: list[str]
    conditions: list[str]
    out_dir: Path
    concurrency: int = 1
    timeout: int = 120
    max_retries: int = 2
    max_side: int = 768
    jpeg_quality: int = 80
    encode_image: bool = True
    temperature: float = 0.0
    extra_payload: dict[str, Any] = field(default_factory=dict)


class Runner:
    def __init__(self, caller: ModelCaller, config: RunConfig):
        if not config.models:
            raise ValueError("At least one model is required.")
        for c in config.conditions:
            if c not in VALID_CONDITIONS:
                raise ValueError(f"Unknown condition {c!r}; expected one of {VALID_CONDITIONS}")
        self.caller = caller
        self.cfg = config
        self.cache_dir = Path(config.out_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run_one(self, task: Task, model: str, condition: str) -> Prediction:
        # `label=api_model` records results under `label` while calling `api_model`.
        model, _, api_model = model.partition("=")
        api_model = api_model or model
        cache_file = self.cache_dir / f"{safe_name(model)}__{safe_name(task.task_id)}__{safe_name(condition)}.json"
        if cache_file.exists():
            return Prediction(**json.loads(cache_file.read_text(encoding="utf-8")))

        messages = build_messages(
            task,
            apply_condition(task, condition),
            max_side=self.cfg.max_side,
            jpeg_quality=self.cfg.jpeg_quality,
            encode_image=self.cfg.encode_image,
        )
        raw_text, reasoning, error = "", "", None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                result = self.caller(
                    model=api_model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    timeout=self.cfg.timeout,
                    extra=self.cfg.extra_payload or None,
                )
                raw_text, reasoning, error = result.final_text, result.reasoning_content, None
                break
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                if attempt < self.cfg.max_retries:
                    time.sleep(min(20.0, 2.5 * (attempt + 1)))

        correct, prediction = (False, "") if error else score_l1(task, raw_text)
        pred = Prediction(
            task_id=task.task_id,
            item_id=task.item_id,
            task_type=task.task_type,
            operation=task.operation,
            perception_load=task.perception_load,
            model=model,
            condition=condition,
            gold=task.gold,
            prediction=prediction,
            correct=correct,
            raw_text=raw_text,
            trace=strip_answer_tag(raw_text),
            reasoning_content=reasoning,
            error=error,
        )
        if not (error and _is_transient_error(error)):
            cache_file.write_text(json.dumps(asdict(pred), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return pred

    def run(self, tasks: Iterable[Task]) -> list[Prediction]:
        tasks = list(tasks)
        units = [
            (t, m, c)
            for m in self.cfg.models
            for c in self.cfg.conditions
            for t in tasks
            if c not in SSA_CONDITIONS or t.task_type == "F"  # SSA applies to F items only
        ]
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency)) as ex:
            results = list(ex.map(lambda u: self.run_one(*u), units))
        results.sort(key=lambda r: (r.model, r.condition, r.task_id))
        out_file = Path(self.cfg.out_dir) / "predictions.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        return results
