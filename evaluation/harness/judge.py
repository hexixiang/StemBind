"""External LLM judge for L2 StepAcc.

The judge receives (question prompt, natural trace, ground-truth S1-S4) and
returns 4 independent 0/1 verdicts.
  * Stages are judged by semantic alignment against ground truth, NOT by looking
    for explicit `<S1>..<S4>` markers in the trace (models reason freely).
  * Skipped or content-wrong stages both receive 0; the first 0 in the
    S1→S4 order is the first-failure stage used by L3 AttrTag.
  * Ambiguous / missing judge output is treated as 0 (conservative).
  * We support n_repeat ≥ 1 (default 3) with temperature 0 and report the
    majority vote + per-stage std.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .model_router import ChatResult, ModelCaller, StubModelCaller

STAGE_KEYS = ("S1", "S2", "S3", "S4")
GT_FIELDS = ("S1_encode", "S2_infer", "S3_map", "S4_apply")

JUDGE_SYSTEM = (
    "You are a strict but fair alignment judge for the StemBind L2 StepAcc "
    "evaluation. Given (question, candidate natural-language trace, ground-truth "
    "4-stage solution), output for each stage a 0/1 flag indicating whether the "
    "trace contains content semantically aligned with that stage's ground truth. "
    "Do NOT look for explicit <S1> tags. If a stage is missing OR present but "
    "substantively wrong, output 0 for that stage. Output strict JSON."
)

JUDGE_OUTPUT_SCHEMA = (
    '{"S1":0,"S2":0,"S3":0,"S4":0,"first_failure":"S1|S2|S3|S4|none",'
    '"rationale":"brief English reason"}'
)


@dataclass
class JudgeResult:
    stages: dict[str, int]                 # {"S1":0/1, ...}
    first_failure: str                     # "S1".."S4" or "" if all pass
    rationale: str = ""
    std: dict[str, float] = field(default_factory=dict)
    raw: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "stages": dict(self.stages),
            "first_failure": self.first_failure,
            "rationale": self.rationale,
            "std": dict(self.std),
            "raw": list(self.raw),
        }


# --- Prompt ---------------------------------------------------------------


def build_judge_prompt(*, question: str, trace: str, gt_steps: dict[str, str]) -> list[dict[str, Any]]:
    gt_block = "\n".join(
        f"- {key}: {gt_steps.get(key, '[TBD]')}" for key in GT_FIELDS
    )
    user = (
        "Question (translated if needed):\n"
        f"{question}\n\n"
        "Candidate trace (may be free-form, in any language, MAY be empty):\n"
        "===BEGIN TRACE===\n"
        f"{trace if trace else '[empty trace]'}\n"
        "===END TRACE===\n\n"
        "Ground-truth 4-stage reference solution:\n"
        f"{gt_block}\n\n"
        "For each stage (S1 Encode / S2 Infer / S3 Map / S4 Apply), output 1 "
        "if the trace correctly performs that step (semantic alignment, not "
        "required to echo wording), else 0. Missing a step => 0. Wrong content "
        "=> 0. `first_failure` is the smallest stage with 0, or 'none' if all 1.\n"
        f"Output strict JSON matching exactly: {JUDGE_OUTPUT_SCHEMA}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


# --- Parsing --------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json_obj(raw: str) -> dict[str, Any]:
    s = raw.strip()
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass
    for m in _JSON_BLOCK.findall(s):
        try:
            return json.loads(m)
        except Exception:  # noqa: BLE001
            continue
    return {}


def parse_judge_output(raw: str) -> dict[str, Any]:
    obj = _extract_json_obj(raw)
    stages = {}
    for k in STAGE_KEYS:
        v = obj.get(k, 0)
        try:
            stages[k] = 1 if int(v) == 1 else 0
        except (TypeError, ValueError):
            stages[k] = 0
    ff = str(obj.get("first_failure", "")).strip().upper()
    if ff not in STAGE_KEYS and ff not in {"NONE", ""}:
        ff = ""
    if ff == "NONE":
        ff = ""
    # Recompute for consistency against stages (conservative):
    for k in STAGE_KEYS:
        if stages[k] == 0:
            ff = k
            break
    else:
        ff = ""
    return {
        "stages": stages,
        "first_failure": ff,
        "rationale": str(obj.get("rationale", "")).strip()[:400],
    }


# --- Judge engine ---------------------------------------------------------


@dataclass
class JudgeConfig:
    model: str = "gpt-4o"
    n_repeat: int = 1
    temperature: float = 0.0
    timeout: int = 120
    extra_payload: dict[str, Any] = field(default_factory=dict)


class Judge:
    """L2 judge engine. Can be driven by any ModelCaller (stub or live)."""

    def __init__(self, caller: ModelCaller, config: JudgeConfig | None = None):
        self.caller = caller
        self.cfg = config or JudgeConfig()

    def judge(
        self,
        *,
        question: str,
        trace: str,
        gt_steps: dict[str, str],
    ) -> JudgeResult:
        messages = build_judge_prompt(question=question, trace=trace, gt_steps=gt_steps)
        per_stage: dict[str, list[int]] = {k: [] for k in STAGE_KEYS}
        raws: list[str] = []
        first_failures: list[str] = []
        rationale = ""
        for _ in range(max(1, self.cfg.n_repeat)):
            raw = self.caller(
                model=self.cfg.model,
                messages=messages,
                temperature=self.cfg.temperature,
                timeout=self.cfg.timeout,
                extra=self.cfg.extra_payload or None,
            )
            raws.append(raw.final_text)
            parsed = parse_judge_output(raw.final_text)
            for k in STAGE_KEYS:
                per_stage[k].append(parsed["stages"][k])
            first_failures.append(parsed["first_failure"])
            rationale = parsed["rationale"] or rationale

        # Majority vote per stage (ties -> 0, conservative).
        stages = {
            k: 1 if sum(per_stage[k]) > len(per_stage[k]) / 2 else 0
            for k in STAGE_KEYS
        }
        # Recompute first_failure from voted stages.
        ff = ""
        for k in STAGE_KEYS:
            if stages[k] == 0:
                ff = k
                break
        std = {
            k: statistics.pstdev(per_stage[k]) if len(per_stage[k]) > 1 else 0.0
            for k in STAGE_KEYS
        }
        return JudgeResult(stages=stages, first_failure=ff, rationale=rationale, std=std, raw=raws)


# --- Stub judge (tests / dry-run) -----------------------------------------


class StubJudge:
    """Deterministic judge: passes stages whose GT keywords appear in the trace."""

    def __init__(self, keywords_per_stage: int = 6):
        self.k = keywords_per_stage

    def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        timeout: int = 120,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        # Extract trace + gt_steps from the user message we built.
        user_text = ""
        for m in messages:
            if m.get("role") == "user":
                c = m.get("content")
                user_text = c if isinstance(c, str) else ""
                break
        trace = _between(user_text, "===BEGIN TRACE===", "===END TRACE===")
        gt = {}
        for key, label in zip(GT_FIELDS, STAGE_KEYS):
            gt[label] = _after(user_text, f"- {key}:")
        stages = {
            s: 1 if _keyword_overlap(trace, gt.get(s, ""), self.k) else 0
            for s in STAGE_KEYS
        }
        ff = ""
        for s in STAGE_KEYS:
            if stages[s] == 0:
                ff = s
                break
        payload = {
            **stages,
            "first_failure": ff or "none",
            "rationale": "stub: keyword overlap",
        }
        return ChatResult(content=json.dumps(payload, ensure_ascii=False))


def _between(text: str, start: str, end: str) -> str:
    a = text.find(start)
    if a < 0:
        return ""
    b = text.find(end, a + len(start))
    if b < 0:
        return ""
    return text[a + len(start):b].strip()


def _after(text: str, marker: str) -> str:
    a = text.find(marker)
    if a < 0:
        return ""
    line_end = text.find("\n", a)
    return text[a + len(marker): line_end if line_end > 0 else len(text)].strip()


def _keyword_overlap(trace: str, gt: str, k: int) -> bool:
    if not trace or not gt:
        return False
    tokens = [t for t in re.split(r"[\s，。、,./();\[\]:：；;]+", gt) if t]
    if not tokens:
        return False
    # pick first k tokens; pass if any appears in trace
    for tok in tokens[:k]:
        if tok and tok in trace:
            return True
    return False


# --- Factory --------------------------------------------------------------


def build_judge(*, dry_run: bool = True, config: JudgeConfig | None = None) -> Judge:
    if dry_run:
        return Judge(StubJudge(), config or JudgeConfig(n_repeat=1))
    from .model_router import build_caller

    caller = build_caller(dry_run=False)
    return Judge(caller, config or JudgeConfig())
