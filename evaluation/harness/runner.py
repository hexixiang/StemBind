"""Run models over tasks with checkpointed per-call cache.

A single `Prediction` row is appended to `predictions.jsonl`; each call is also
written to `cache/<model>__<task_id>__<condition>.json` so reruns resume
without re-paying for completed model/task/condition triples.

L1 (answer letter / P text match) is computed inline because the logic is
cheap and deterministic. L2 / L3 downstream steps consume the cached traces.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .data_loader import Task
from .input_conditions import VALID_CONDITIONS, apply_condition
from .model_router import ModelCaller, StubModelCaller
from .prompt_builder import (
    build_messages,
    extract_answer,
    extract_answer_letter,
    strip_answer_tag,
)


# --- Scoring helpers ------------------------------------------------------


def _clean(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。、“”‘’；;：:,.!?？！（）()\[\]{}<>《》|/\\_-]", "", s)
    return s


def match_p_answer(pred: str, gold: str) -> bool:
    p, g = _clean(pred), _clean(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    pn = re.findall(r"\d+", p)
    gn = re.findall(r"\d+", g)
    if pn and gn:
        return pn == gn
    if len(g) <= 8 and g in p:
        return True
    if len(p) <= 8 and p in g:
        return True
    return False


def score_l1(task: Task, raw_text: str) -> tuple[bool, str]:
    if task.task_type == "P":
        pred = extract_answer(raw_text) or raw_text.strip()
        return match_p_answer(pred, task.gold), pred
    if task.task_type == "F":
        letter = extract_answer_letter(raw_text)
        return letter == task.gold, letter
    if not task.choices:
        pred = extract_answer(raw_text) or raw_text.strip()
        return _clean(pred) == _clean(task.gold), pred
    letter = extract_answer_letter(raw_text)
    # Legacy R probe: compare predicted letter index vs gold index
    idx = "ABCD".index(letter) if letter in "ABCD" else -1
    return idx == task.gold_index, letter


# --- Prediction record ----------------------------------------------------


@dataclass
class Prediction:
    task_id: str
    item_id: str
    task_type: str
    source: str
    operation: str
    model: str
    condition: str
    gold: str
    prediction: str
    correct: bool
    raw_text: str
    trace: str
    reasoning_content: str = ""
    timestamp: str = ""
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# --- Cache paths ----------------------------------------------------------


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def cache_path(cache_dir: Path, *, model: str, task_id: str, condition: str) -> Path:
    return cache_dir / f"{_safe(model)}__{_safe(task_id)}__{_safe(condition)}.json"


# --- Runner ---------------------------------------------------------------


@dataclass
class RunConfig:
    models: list[str]
    conditions: list[str]
    out_dir: Path
    dry_run: bool = True
    max_side: int = 768
    jpeg_quality: int = 80
    concurrency: int = 1
    timeout: int = 120
    max_retries: int = 2
    temperature: float = 0.0
    encode_image: bool = True
    extra_payload: dict[str, Any] = field(default_factory=dict)
    cache_errors: bool = True

    def validate(self) -> None:
        if not self.models:
            raise ValueError("At least one model is required.")
        for c in self.conditions:
            if c not in VALID_CONDITIONS:
                raise ValueError(f"Unknown condition {c!r}")


class Runner:
    def __init__(self, caller: ModelCaller, config: RunConfig):
        self.caller = caller
        self.cfg = config
        self.cfg.validate()
        self.cfg.out_dir = Path(self.cfg.out_dir)
        (self.cfg.out_dir / "cache").mkdir(parents=True, exist_ok=True)

    # ---- single call ----------------------------------------------------

    def run_one(self, task: Task, model: str, condition: str) -> Prediction:
        cp = cache_path(self.cfg.out_dir / "cache", model=model, task_id=task.task_id, condition=condition)
        if cp.exists():
            cached = json.loads(cp.read_text(encoding="utf-8"))
            return Prediction(**cached)

        ctx = apply_condition(condition=condition, caption=task.caption, parse=task.parse)
        messages = build_messages(
            task,
            ctx,
            max_side=self.cfg.max_side,
            jpeg_quality=self.cfg.jpeg_quality,
            encode_image=self.cfg.encode_image,
        )

        last_exc: Exception | None = None
        raw_text = ""
        reasoning_content = ""
        for attempt in range(self.cfg.max_retries + 1):
            try:
                result = self.caller(
                    model=model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    timeout=self.cfg.timeout,
                    extra=self.cfg.extra_payload or None,
                )
                raw_text = result.final_text
                reasoning_content = result.reasoning_content or ""
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.cfg.max_retries:
                    raw_text = ""
                    break
                time.sleep(min(20.0, 2.5 * (attempt + 1)))

        if last_exc is not None and not raw_text:
            pred = Prediction(
                task_id=task.task_id,
                item_id=task.item_id,
                task_type=task.task_type,
                source=task.source,
                operation=task.operation,
                model=model,
                condition=condition,
                gold=task.gold,
                prediction="",
                correct=False,
                raw_text="",
                reasoning_content="",
                trace="",
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=str(last_exc),
            )
        else:
            correct, prediction = score_l1(task, raw_text)
            pred = Prediction(
                task_id=task.task_id,
                item_id=task.item_id,
                task_type=task.task_type,
                source=task.source,
                operation=task.operation,
                model=model,
                condition=condition,
                gold=task.gold,
                prediction=prediction,
                correct=correct,
                raw_text=raw_text,
                reasoning_content=reasoning_content,
                trace=strip_answer_tag(raw_text),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        if self.cfg.cache_errors or not pred.error or not _is_transient_error(pred.error):
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(
                json.dumps(pred.to_json(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return pred

    # ---- batch ----------------------------------------------------------

    def run(self, tasks: Iterable[Task]) -> list[Prediction]:
        tasks = list(tasks)
        units = [
            (t, m, c)
            for m in self.cfg.models
            for c in self.cfg.conditions
            for t in tasks
            if _condition_feasible(t, c)
        ]
        results: list[Prediction] = []
        if self.cfg.concurrency <= 1:
            for t, m, c in units:
                results.append(self.run_one(t, m, c))
        else:
            with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as ex:
                futures = {ex.submit(self.run_one, t, m, c): (t, m, c) for t, m, c in units}
                for fut in as_completed(futures):
                    results.append(fut.result())
        results.sort(key=lambda r: (r.model, r.condition, r.task_id))
        self._write_predictions(results)
        return results

    # ---- IO -------------------------------------------------------------

    def _write_predictions(self, results: list[Prediction]) -> None:
        path = self.cfg.out_dir / "predictions.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")


def _condition_feasible(task: Task, condition: str) -> bool:
    """Skip conditions that require missing context for this task."""
    if condition == "caption-only" and not task.caption:
        return False
    if condition == "parse-only" and not task.parse:
        return False
    return True


def _is_transient_error(error: str) -> bool:
    markers = (
        "HTTP Error 502",
        "HTTP Error 503",
        "HTTP Error 504",
        "HTTP Error 400",
        "HTTP Error 408",
        "HTTP Error 409",
        "HTTP Error 425",
        "HTTP Error 429",
        "Bad Request",
        "Too Many Requests",
        "rate limit",
        "Rate limit",
        "timed out",
        "TimeoutError",
        "Read timed out",
        "Connection reset",
        "Connection refused",
        "Connection aborted",
        "Remote end closed connection",
        "urlopen error",
        "SSL:",
        "UNEXPECTED_EOF",
        "EOF occurred",
        "IncompleteRead",
        "ChunkedEncodingError",
        "ProtocolError",
        "Bad Gateway",
        "Service Unavailable",
        "Gateway Timeout",
    )
    return any(m in error for m in markers)


# --- Convenience ----------------------------------------------------------


def build_default_runner(config: RunConfig) -> Runner:
    from .model_router import build_caller

    caller = build_caller(dry_run=config.dry_run)
    if config.dry_run:
        caller = StubModelCaller()
    return Runner(caller, config)
