"""External LLM judge for L2 StepAcc.

The judge receives (question prompt, natural trace, ground-truth S1-S4) and
returns 4 independent 0/1 verdicts.
  * Stages are judged by semantic alignment against ground truth, NOT by looking
    for explicit `<S1>..<S4>` markers in the trace (models reason freely).
  * Skipped or content-wrong stages both receive 0; the first 0 in the
    S1→S4 order is the first-failure stage used by L3 AttrTag.
  * Ambiguous / missing judge output is treated as 0 (conservative).
  * n_repeat >= 1 calls (default 1, GPT-4o at temperature 0 as in the paper)
    are combined by per-stage majority vote; per-stage std is reported.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from .model_router import ChatResult, ModelCaller

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


def _first_failure(stages: dict[str, int]) -> str:
    return next((k for k in STAGE_KEYS if stages[k] == 0), "")


def parse_judge_output(raw: str) -> dict[str, Any]:
    obj = _extract_json_obj(raw)
    stages = {}
    for k in STAGE_KEYS:
        try:
            stages[k] = 1 if int(obj.get(k, 0)) == 1 else 0
        except (TypeError, ValueError):
            stages[k] = 0
    # first_failure is recomputed from the stage flags rather than trusted.
    return {
        "stages": stages,
        "first_failure": _first_failure(stages),
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

    def judge(self, *, question: str, trace: str, gt_steps: dict[str, str]) -> JudgeResult:
        messages = build_judge_prompt(question=question, trace=trace, gt_steps=gt_steps)
        per_stage: dict[str, list[int]] = {k: [] for k in STAGE_KEYS}
        raws: list[str] = []
        rationale = ""
        for _ in range(max(1, self.cfg.n_repeat)):
            raw = self.caller(
                model=self.cfg.model,
                messages=messages,
                temperature=self.cfg.temperature,
                timeout=self.cfg.timeout,
                extra=self.cfg.extra_payload or None,
            ).final_text
            raws.append(raw)
            parsed = parse_judge_output(raw)
            for k in STAGE_KEYS:
                per_stage[k].append(parsed["stages"][k])
            rationale = parsed["rationale"] or rationale

        # Majority vote per stage (ties -> 0, conservative).
        stages = {k: 1 if sum(v) > len(v) / 2 else 0 for k, v in per_stage.items()}
        std = {k: statistics.pstdev(v) if len(v) > 1 else 0.0 for k, v in per_stage.items()}
        return JudgeResult(
            stages=stages, first_failure=_first_failure(stages), rationale=rationale, std=std, raw=raws,
        )


# --- Stub judge (dry-run) -------------------------------------------------


class StubJudge:
    """Offline judge: a stage passes if one of its first ground-truth words appears in the trace."""

    def __call__(self, *, messages: list[dict[str, Any]], **_: Any) -> ChatResult:
        user_text = next((m["content"] for m in messages if m.get("role") == "user"), "")
        start, end = user_text.find("===BEGIN TRACE==="), user_text.find("===END TRACE===")
        trace = user_text[start:end] if 0 <= start < end else ""
        stages = {}
        for key, stage in zip(GT_FIELDS, STAGE_KEYS):
            marker = f"- {key}:"
            line = user_text[user_text.find(marker) + len(marker):].split("\n", 1)[0] if marker in user_text else ""
            words = re.findall(r"\w+", line)[:6]
            stages[stage] = int(any(w in trace for w in words))
        return ChatResult(content=json.dumps({**stages, "rationale": "stub: keyword overlap"}))


def build_judge(*, dry_run: bool, config: JudgeConfig | None = None) -> Judge:
    from .model_router import build_caller

    caller = StubJudge() if dry_run else build_caller(dry_run=False)
    return Judge(caller, config or JudgeConfig())
