"""StemBind evaluation harness CLI.

Defaults to `--dry-run` (stub model + stub judge; no network, no API cost).
Pass `--live` to actually hit the OpenAI-compatible gateway.

Subcommands:
    run         Score F/P/R predictions under the four input conditions
    judge       Run the L2 judge over cached predictions
    metrics     Compute L1 / L2 / L3 / StemCoherent / HCAG / VG + audit
    ssa         Stage Skyline Analysis (Exp4) from H0..H4 predictions
    thinkgain   ThinkGain (Exp6) from paired baseline / thinking predictions
    validate    Load the dataset and print a sanity summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running as `python evaluation/cli.py` from the repository root
_HERE = Path(__file__).resolve().parent
_CODE_DIR = _HERE.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from evaluation.harness.answer_judge import (
    AnswerJudge,
    AnswerJudgeConfig,
    exact_fallback_result,
)
from evaluation.harness.data_loader import expand_tasks, load_dataset
from evaluation.harness.input_conditions import VALID_CONDITIONS
from evaluation.harness.judge import Judge, JudgeConfig, StubJudge, build_judge
from evaluation.harness.model_router import StubModelCaller, build_caller
from evaluation.harness.runner import RunConfig, Runner
from evaluation.metrics.hcag import hcag_by_model
from evaluation.metrics.l1_accuracy import accuracy_summary
from evaluation.metrics.l2_stepacc import first_failure_rows, stepacc_summary
from evaluation.metrics.l3_attrtag import attrtag_summary
from evaluation.metrics.stem_coherent import stem_coherent_by_model
from evaluation.metrics.stemcoherent_audit import audit as stem_audit
from evaluation.metrics.vg import vg_by_model
from evaluation.experiments.ssa import skyline as ssa_skyline
from evaluation.experiments.thinkgain import thinkgain as run_thinkgain


# --- IO helpers -----------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _safe_cache_name(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _build_extra_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build provider-specific request-body fields for OpenAI-compatible APIs."""
    payload: dict[str, Any] = {}
    raw = getattr(args, "extra_payload_json", None)
    if raw:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("--extra-payload-json must decode to a JSON object.")
        payload.update(obj)

    max_tokens = getattr(args, "max_tokens", None)
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    top_p = getattr(args, "top_p", None)
    if top_p is not None:
        payload["top_p"] = top_p
    min_p = getattr(args, "min_p", None)
    if min_p is not None:
        payload["min_p"] = min_p
    presence_penalty = getattr(args, "presence_penalty", None)
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    repetition_penalty = getattr(args, "repetition_penalty", None)
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    top_k = getattr(args, "top_k", None)
    if top_k is not None:
        payload["top_k"] = top_k

    enable_thinking = getattr(args, "enable_thinking", None)
    if enable_thinking is not None:
        chat_kwargs = dict(payload.get("chat_template_kwargs") or {})
        chat_kwargs["enable_thinking"] = enable_thinking == "true"
        payload["chat_template_kwargs"] = chat_kwargs
    return payload


def _add_generation_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--max-tokens", type=int, default=None)
    s.add_argument("--top-p", type=float, default=None)
    s.add_argument("--min-p", type=float, default=None)
    s.add_argument("--presence-penalty", type=float, default=None)
    s.add_argument("--repetition-penalty", type=float, default=None)
    s.add_argument("--top-k", type=int, default=None)
    s.add_argument("--enable-thinking", choices=["true", "false"], default=None,
                   help="Pass chat_template_kwargs.enable_thinking for Qwen-style vLLM servers.")
    s.add_argument("--extra-payload-json", default=None,
                   help="Extra top-level JSON fields merged into the chat/completions request body.")


def _default_pairs() -> dict[str, tuple[str, str]]:
    return {
        "G1": ("gpt-4o", "gpt-o3"),
        "G2": ("gemini-2.5-pro-nothinking", "gemini-2.5-pro-thinking"),
        "G3": ("claude-sonnet-4.5", "claude-sonnet-4.5-thinking"),
        "G4": ("qwen2.5-vl-72b-instruct", "qvq-72b"),
        "G5": ("kimi-vl", "kimi-vl-thinking"),
    }


# --- run ------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    records = load_dataset(
        Path(args.dataset_dir),
        annotated_path=Path(args.annotated) if args.annotated else None,
        caption_path=Path(args.captions) if args.captions else None,
        parse_path=Path(args.parses) if args.parses else None,
    )
    if args.limit:
        records = records[: args.limit]
    tasks = expand_tasks(
        records,
        include_f=args.include_f,
        include_p=args.include_p,
        include_r=args.include_r,
        max_p_per_item=args.max_p_per_item,
    )
    if args.task_limit:
        tasks = tasks[: args.task_limit]
    cfg = RunConfig(
        models=args.models,
        conditions=args.conditions,
        out_dir=Path(args.out_dir),
        dry_run=args.dry_run,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        encode_image=not args.dry_run,
        temperature=args.temperature,
        extra_payload=_build_extra_payload(args),
        cache_errors=args.cache_errors,
    )
    caller = StubModelCaller() if args.dry_run else build_caller(dry_run=False)
    runner = Runner(caller, cfg)
    _dump_json(cfg.out_dir / "config.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "models": args.models,
        "conditions": args.conditions,
        "dry_run": args.dry_run,
        "n_tasks": len(tasks),
        "include": {"F": args.include_f, "P": args.include_p, "R": args.include_r},
        "max_p_per_item": args.max_p_per_item,
        "limit": args.limit,
        "task_limit": args.task_limit,
        "temperature": args.temperature,
        "extra_payload": cfg.extra_payload,
    })
    results = runner.run(tasks)
    print(f"Wrote {len(results)} predictions to {cfg.out_dir}/predictions.jsonl "
          f"(dry_run={args.dry_run})")
    return 0


# --- judge ----------------------------------------------------------------


def cmd_judge(args: argparse.Namespace) -> int:
    preds = _load_jsonl(Path(args.predictions))
    ds_records = load_dataset(Path(args.dataset_dir))
    by_item = {r.item_id: r for r in ds_records}

    judge = build_judge(dry_run=args.dry_run, config=JudgeConfig(
        model=args.judge_model,
        n_repeat=args.n_repeat,
        timeout=args.timeout,
        temperature=args.temperature,
        extra_payload=_build_extra_payload(args),
    ))

    def judge_one(r: dict[str, Any]) -> dict[str, Any]:
        if r.get("task_type") != "F":
            return r
        if r.get("error") or not r.get("trace", ""):
            return {
                **r,
                "judge": {
                    "stages": {"S1": 0, "S2": 0, "S3": 0, "S4": 0},
                    "first_failure": "S1",
                    "rationale": "empty trace or failed prediction",
                    "std": {"S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": 0.0},
                    "raw": [],
                },
            }
        rec = by_item.get(r.get("item_id", ""))
        gt_steps = rec.gt_steps if rec else {}
        question = rec.f_question if rec else ""
        result = judge.judge(
            question=question,
            trace=r.get("trace", ""),
            gt_steps=gt_steps,
        )
        return {**r, "judge": result.to_json()}

    if args.concurrency <= 1:
        out = [judge_one(r) for r in preds]
    else:
        out = [None] * len(preds)
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(judge_one, r): i for i, r in enumerate(preds)}
            for fut in as_completed(futures):
                out[futures[fut]] = fut.result()
    judged_path = Path(args.out or str(Path(args.predictions).with_name("predictions_judged.jsonl")))
    _dump_jsonl(judged_path, out)
    print(f"Judged {sum(1 for r in out if 'judge' in r)} F rows → {judged_path}")
    return 0


# --- answer-judge ---------------------------------------------------------


def cmd_answer_judge(args: argparse.Namespace) -> int:
    preds = _load_jsonl(Path(args.predictions))
    records = load_dataset(Path(args.dataset_dir))
    tasks = {
        t.task_id: t
        for t in expand_tasks(records, max_p_per_item=args.max_p_per_item)
    }
    out_path = Path(args.out or str(Path(args.predictions).with_name("predictions_answer_judged.jsonl")))
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_path.with_suffix("").with_name(out_path.stem + "_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    caller = StubModelCaller() if args.dry_run else build_caller(dry_run=False)
    judge = AnswerJudge(caller, AnswerJudgeConfig(
        model=args.judge_model,
        timeout=args.timeout,
        temperature=args.temperature,
        extra_payload=_build_extra_payload(args),
    ))

    def judge_one(row: dict[str, Any]) -> dict[str, Any]:
        task = tasks.get(str(row.get("task_id", "")))
        if not task:
            result = {
                "correct": 0,
                "confidence": "low",
                "rationale": "missing task metadata",
                "raw": "",
                "error": "missing task metadata",
            }
        else:
            cp = cache_dir / (
                f"{_safe_cache_name(args.judge_model)}__"
                f"{_safe_cache_name(str(row.get('model', '')))}__"
                f"{_safe_cache_name(str(row.get('task_id', '')))}__"
                f"{_safe_cache_name(str(row.get('condition', '')))}.json"
            )
            if cp.exists():
                result = json.loads(cp.read_text(encoding="utf-8"))
            else:
                if args.dry_run:
                    result_obj = exact_fallback_result(task, row)
                else:
                    result_obj = None
                    last_exc: Exception | None = None
                    for _attempt in range(args.max_retries + 1):
                        try:
                            result_obj = judge.judge(task=task, row=row)
                            break
                        except Exception as exc:  # noqa: BLE001
                            last_exc = exc
                    if result_obj is None:
                        result_obj = exact_fallback_result(task, row)
                        result_obj.confidence = "low"
                        result_obj.rationale = "answer judge failed; exact fallback used"
                        result_obj.error = str(last_exc) if last_exc else "unknown judge failure"
                result = result_obj.to_json()
                if not result.get("error"):
                    cp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        out_row = {
            **row,
            "exact_correct": row.get("correct"),
            "answer_judge": result,
        }
        if args.replace_correct:
            out_row["correct"] = bool(result.get("correct"))
        return out_row

    if args.concurrency <= 1:
        out_rows = [judge_one(r) for r in preds]
    else:
        out_rows = [None] * len(preds)
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(judge_one, r): i for i, r in enumerate(preds)}
            for fut in as_completed(futures):
                out_rows[futures[fut]] = fut.result()

    _dump_jsonl(out_path, out_rows)
    n_judged = sum(1 for r in out_rows if r.get("answer_judge"))
    n_correct = sum(1 for r in out_rows if r.get("answer_judge", {}).get("correct") == 1)
    print(f"Answer-judged {n_judged} rows → {out_path} (correct={n_correct})")
    return 0


# --- metrics --------------------------------------------------------------


def cmd_metrics(args: argparse.Namespace) -> int:
    preds = _load_jsonl(Path(args.predictions))
    out_dir = Path(args.out_dir)

    l1 = accuracy_summary(preds)
    sc_rows = stem_coherent_by_model(preds)
    sc_audit = stem_audit(predictions=preds, stem_coherent_rows=sc_rows, top_n=args.audit_top_n)
    vg_rows = vg_by_model(preds)
    hcag_rows = hcag_by_model(
        preds,
        human_norms=json.loads(Path(args.human_norms).read_text(encoding="utf-8"))
        if args.human_norms else None,
    )
    l2 = stepacc_summary(preds)
    ff_rows = first_failure_rows(preds)
    l3 = attrtag_summary(ff_rows)

    _dump_json(out_dir / "metrics_l1.json", l1)
    _dump_json(out_dir / "metrics_l2.json", l2)
    _dump_json(out_dir / "metrics_l3.json", l3)
    _dump_json(out_dir / "stem_coherent.json", sc_rows)
    _dump_json(out_dir / "stem_coherent_audit.json", sc_audit)
    _dump_json(out_dir / "vg.json", vg_rows)
    _dump_json(out_dir / "hcag.json", hcag_rows)
    print(f"Wrote metrics bundle to {out_dir}/")
    return 0


# --- ssa ------------------------------------------------------------------


def cmd_ssa(args: argparse.Namespace) -> int:
    preds = _load_jsonl(Path(args.predictions))
    summary = ssa_skyline(preds, n_boot=args.n_boot)
    _dump_json(Path(args.out), summary)
    print(f"SSA skyline → {args.out} ({len(summary['by_model'])} models)")
    return 0


# --- thinkgain ------------------------------------------------------------


def cmd_thinkgain(args: argparse.Namespace) -> int:
    preds = _load_jsonl(Path(args.predictions))
    pairs = _default_pairs()
    if args.pairs:
        pairs = {}
        for spec in args.pairs:
            group, base, think = spec.split(":")
            pairs[group] = (base, think)
    summary = run_thinkgain(preds, pairs=pairs, go_threshold_pp=args.go_threshold_pp)
    _dump_json(Path(args.out), summary)
    print(f"ThinkGain → {args.out} ({len(summary['pairs'])} pairs, overall={summary['overall_verdict']})")
    return 0


# --- validate -------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    records = load_dataset(Path(args.dataset_dir))
    tasks = expand_tasks(records)
    from collections import Counter
    by_source = Counter(r.source for r in records)
    by_op = Counter(r.operation for r in records)
    by_task = Counter(t.task_type for t in tasks)
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "n_items": len(records),
        "n_tasks": len(tasks),
        "by_source": dict(by_source),
        "by_operation": dict(by_op),
        "by_task_type": dict(by_task),
        "items_with_r_probe": sum(1 for r in records if r.r_probe),
        "items_with_gt_steps": sum(1 for r in records if r.gt_steps),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


# --- main -----------------------------------------------------------------


def _add_run(sp):
    s = sp.add_parser("run", help="Run VLM eval and write predictions.jsonl")
    s.add_argument("--dataset-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--conditions", nargs="+", default=["full-image"], choices=list(VALID_CONDITIONS))
    s.add_argument("--annotated", default=None)
    s.add_argument("--captions", default=None)
    s.add_argument("--parses", default=None)
    s.add_argument("--include-f", action="store_true", default=True)
    s.add_argument("--no-f", dest="include_f", action="store_false")
    s.add_argument("--include-p", action="store_true", default=True)
    s.add_argument("--no-p", dest="include_p", action="store_false")
    s.add_argument("--include-r", action="store_true", default=True)
    s.add_argument("--no-r", dest="include_r", action="store_false")
    s.add_argument("--max-p-per-item", type=int, default=99)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--task-limit", type=int, default=None,
                   help="Limit the expanded F/P/R task list after item expansion.")
    s.add_argument("--concurrency", type=int, default=1)
    s.add_argument("--timeout", type=int, default=120)
    s.add_argument("--max-retries", type=int, default=2)
    s.add_argument("--max-side", type=int, default=768)
    s.add_argument("--jpeg-quality", type=int, default=80)
    s.add_argument("--cache-errors", dest="cache_errors", action="store_true", default=True,
                   help="Write failed calls to cache so reruns skip them. (default)")
    s.add_argument("--no-cache-errors", dest="cache_errors", action="store_false",
                   help="Do not cache transient 5xx/timeout failures; useful for high-concurrency retries.")
    _add_generation_args(s)
    s.set_defaults(fn=cmd_run)


def _add_judge(sp):
    s = sp.add_parser("judge", help="Run L2 judge over cached predictions")
    s.add_argument("--predictions", required=True)
    s.add_argument("--dataset-dir", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--judge-model", default="gpt-4o")
    s.add_argument("--n-repeat", type=int, default=1)
    s.add_argument("--concurrency", type=int, default=1)
    s.add_argument("--timeout", type=int, default=120)
    _add_generation_args(s)
    s.set_defaults(fn=cmd_judge)


def _add_answer_judge(sp):
    s = sp.add_parser("answer-judge", help="LLM-as-judge answer correctness for F/P/R rows")
    s.add_argument("--predictions", required=True)
    s.add_argument("--dataset-dir", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--cache-dir", default=None)
    s.add_argument("--judge-model", required=True,
                   help="Required LLM-as-judge model for semantic/non-A-B-C-D answer grading.")
    s.add_argument("--concurrency", type=int, default=1)
    s.add_argument("--timeout", type=int, default=120)
    s.add_argument("--max-retries", type=int, default=2)
    s.add_argument("--max-p-per-item", type=int, default=99)
    s.add_argument("--replace-correct", dest="replace_correct", action="store_true", default=True,
                   help="Replace row.correct with answer_judge.correct in the output. (default)")
    s.add_argument("--no-replace-correct", dest="replace_correct", action="store_false")
    _add_generation_args(s)
    s.set_defaults(fn=cmd_answer_judge)


def _add_metrics(sp):
    s = sp.add_parser("metrics", help="Compute L1/L2/L3/SC/HCAG/VG")
    s.add_argument("--predictions", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--human-norms", default=None,
                   help="JSON file mapping operation -> {mu, sigma, weight}; HCAG becomes [TBD] if missing.")
    s.add_argument("--audit-top-n", type=int, default=10)
    s.set_defaults(fn=cmd_metrics)


def _add_ssa(sp):
    s = sp.add_parser("ssa", help="Stage Skyline Analysis (Exp4)")
    s.add_argument("--predictions", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-boot", type=int, default=1000)
    s.set_defaults(fn=cmd_ssa)


def _add_thinkgain(sp):
    s = sp.add_parser("thinkgain", help="ThinkGain (Exp6) paired analysis")
    s.add_argument("--predictions", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--pairs", nargs="*", default=None,
                   help="Optional group:baseline:thinking triples; uses built-in defaults if omitted.")
    s.add_argument("--go-threshold-pp", type=float, default=5.0)
    s.set_defaults(fn=cmd_thinkgain)


def _add_validate(sp):
    s = sp.add_parser("validate", help="Load dataset and print sanity summary")
    s.add_argument("--dataset-dir", required=True)
    s.set_defaults(fn=cmd_validate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="StemBind evaluation harness")
    mx = parser.add_mutually_exclusive_group()
    mx.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                    help="Use stub model + stub judge, no network. (default)")
    mx.add_argument("--live", dest="dry_run", action="store_false",
                    help="Use real OpenAI-compatible gateway.")
    sp = parser.add_subparsers(dest="cmd", required=True)
    _add_run(sp); _add_judge(sp); _add_answer_judge(sp); _add_metrics(sp); _add_ssa(sp); _add_thinkgain(sp); _add_validate(sp)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
