"""StemBind evaluation CLI.

Commands run in dry-run mode by default (stub model and judge, no network).
Pass `--live` to call the OpenAI-compatible endpoint in OPENAI_API_BASE.

    validate      Load the dataset and print a summary
    run           Generate F / P / R predictions
    answer-judge  LLM-judged answer correctness for free-form P / R answers
    judge         L2 S1-S4 trace judge on F predictions
    metrics       L1 accuracy, Binding Gap, L2 StepAcc, L3 AttrTag, StemCoherent, VG
    ssa           Score SSA runs (conditions H0..H4, H1_irrelevant)
    thinkgain     Paired direct-vs-thinking deltas
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python evaluation/cli.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.experiments.ssa import skyline as ssa_skyline  # noqa: E402
from evaluation.experiments.thinkgain import thinkgain as run_thinkgain  # noqa: E402
from evaluation.harness.answer_judge import AnswerJudge, AnswerJudgeConfig, exact_fallback_result  # noqa: E402
from evaluation.harness.data_loader import LABELS_PATH, expand_tasks, load_dataset  # noqa: E402
from evaluation.harness.input_conditions import VALID_CONDITIONS  # noqa: E402
from evaluation.harness.judge import JudgeConfig, build_judge  # noqa: E402
from evaluation.harness.model_router import build_caller  # noqa: E402
from evaluation.harness.runner import RunConfig, Runner, safe_name  # noqa: E402
from evaluation.metrics.binding import binding_summary  # noqa: E402
from evaluation.metrics.l1_accuracy import accuracy_summary  # noqa: E402
from evaluation.metrics.l2_stepacc import first_failure_rows, stepacc_summary  # noqa: E402
from evaluation.metrics.l3_attrtag import attrtag_summary  # noqa: E402
from evaluation.metrics.stem_coherent import stem_coherent_by_model  # noqa: E402
from evaluation.metrics.stemcoherent_audit import audit as stem_audit  # noqa: E402
from evaluation.metrics.vg import vg_by_model  # noqa: E402


# --- IO helpers -----------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _parallel_map(fn, rows: list[Any], concurrency: int) -> list[Any]:
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        return list(ex.map(fn, rows))


def _extra_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Provider-specific request fields for OpenAI-compatible APIs (e.g. vLLM)."""
    payload: dict[str, Any] = json.loads(args.extra_payload_json) if args.extra_payload_json else {}
    for name in ("max_tokens", "top_p", "top_k"):
        if getattr(args, name) is not None:
            payload[name] = getattr(args, name)
    if args.enable_thinking is not None:
        payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = args.enable_thinking == "true"
    return payload


# --- Commands -------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    records = load_dataset(Path(args.dataset_dir), labels_path=Path(args.labels))
    tasks = expand_tasks(records)
    print(json.dumps({
        "n_stems": len(records),
        "n_tasks": len(tasks),
        "by_task_type": dict(Counter(t.task_type for t in tasks)),
        "by_operation": dict(sorted(Counter(r.operation for r in records).items())),
    }, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    records = load_dataset(Path(args.dataset_dir), labels_path=Path(args.labels))[: args.limit]
    tasks = expand_tasks(
        records,
        include_f="F" in args.tasks,
        include_p="P" in args.tasks,
        include_r="R" in args.tasks,
        max_p_per_item=args.max_p_per_item,
    )
    cfg = RunConfig(
        models=args.models,
        conditions=args.conditions,
        out_dir=Path(args.out_dir),
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        encode_image=not args.dry_run,
        temperature=args.temperature,
        extra_payload=_extra_payload(args),
    )
    _dump_json(cfg.out_dir / "config.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "models": args.models,
        "conditions": args.conditions,
        "tasks": args.tasks,
        "n_tasks": len(tasks),
        "temperature": args.temperature,
        "extra_payload": cfg.extra_payload,
    })
    results = Runner(build_caller(dry_run=args.dry_run), cfg).run(tasks)
    print(f"Wrote {len(results)} predictions to {cfg.out_dir / 'predictions.jsonl'}")
    return 0


def cmd_answer_judge(args: argparse.Namespace) -> int:
    preds = _load_jsonl(Path(args.predictions))
    records = load_dataset(Path(args.dataset_dir), labels_path=Path(args.labels))
    tasks = {t.task_id: t for t in expand_tasks(records)}
    out_path = Path(args.out or Path(args.predictions).with_name("predictions_answer_judged.jsonl"))
    cache_dir = out_path.with_name(out_path.stem + "_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    judge = AnswerJudge(build_caller(dry_run=args.dry_run), AnswerJudgeConfig(
        model=args.judge_model,
        timeout=args.timeout,
        temperature=args.temperature,
        extra_payload=_extra_payload(args),
    ))

    def judge_one(row: dict[str, Any]) -> dict[str, Any]:
        task = tasks.get(str(row.get("task_id", "")))
        if task is None or row.get("error"):
            return {**row, "answer_judge": None}
        cache_file = cache_dir / "__".join(
            safe_name(str(x)) for x in (args.judge_model, row.get("model"), row.get("task_id"), row.get("condition"))
        )
        cache_file = cache_file.with_suffix(".json")
        if cache_file.exists():
            result = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            if args.dry_run:
                res = exact_fallback_result(task, row, "dry-run exact match")
            else:
                res, error = None, ""
                for _ in range(args.max_retries + 1):
                    try:
                        res = judge.judge(task=task, row=row)
                        break
                    except Exception as exc:  # noqa: BLE001
                        error = str(exc)
                if res is None:
                    res = exact_fallback_result(task, row, "judge failed; exact-match fallback")
                    res.error = error
            result = res.to_json()
            if not result.get("error"):
                cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {**row, "exact_correct": row.get("correct"), "correct": bool(result["correct"]), "answer_judge": result}

    out = _parallel_map(judge_one, preds, args.concurrency)
    _dump_jsonl(out_path, out)
    n_correct = sum(1 for r in out if r.get("correct"))
    print(f"Answer-judged {len(out)} rows -> {out_path} (correct={n_correct})")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    preds = _load_jsonl(Path(args.predictions))
    records = {r.item_id: r for r in load_dataset(Path(args.dataset_dir), labels_path=Path(args.labels))}
    judge = build_judge(dry_run=args.dry_run, config=JudgeConfig(
        model=args.judge_model,
        n_repeat=args.n_repeat,
        timeout=args.timeout,
        temperature=args.temperature,
        extra_payload=_extra_payload(args),
    ))

    def judge_one(r: dict[str, Any]) -> dict[str, Any]:
        if r.get("task_type") != "F" or r.get("condition") != "full-image":
            return r
        rec = records.get(r.get("item_id", ""))
        if r.get("error") or not r.get("trace") or rec is None:
            stages = {k: 0 for k in ("S1", "S2", "S3", "S4")}
            return {**r, "judge": {"stages": stages, "first_failure": "S1", "rationale": "empty trace or failed prediction"}}
        result = judge.judge(question=rec.f_question, trace=r["trace"], gt_steps=rec.gt_steps)
        return {**r, "judge": result.to_json()}

    out = _parallel_map(judge_one, preds, args.concurrency)
    out_path = Path(args.out or Path(args.predictions).with_name("predictions_judged.jsonl"))
    _dump_jsonl(out_path, out)
    print(f"Judged {sum(1 for r in out if 'judge' in r)} F rows -> {out_path}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    preds = _load_jsonl(Path(args.predictions))
    out_dir = Path(args.out_dir)
    sc_rows = stem_coherent_by_model(preds)
    _dump_json(out_dir / "metrics_l1.json", accuracy_summary(preds))
    _dump_json(out_dir / "binding.json", binding_summary(preds))
    _dump_json(out_dir / "metrics_l2.json", stepacc_summary(preds))
    _dump_json(out_dir / "metrics_l3.json", attrtag_summary(first_failure_rows(preds)))
    _dump_json(out_dir / "stem_coherent.json", sc_rows)
    _dump_json(out_dir / "stem_coherent_audit.json", stem_audit(predictions=preds, stem_coherent_rows=sc_rows))
    _dump_json(out_dir / "vg.json", vg_by_model(preds))
    print(f"Wrote metrics to {out_dir}/")
    return 0


def cmd_ssa(args: argparse.Namespace) -> int:
    summary = ssa_skyline(_load_jsonl(Path(args.predictions)), n_boot=args.n_boot)
    _dump_json(Path(args.out), summary)
    print(f"SSA -> {args.out} ({len(summary['by_model'])} models)")
    return 0


def cmd_thinkgain(args: argparse.Namespace) -> int:
    pairs = {}
    for spec in args.pairs:
        direct, thinking = spec.split(":")
        pairs[direct] = (direct, thinking)
    preds = [row for path in args.predictions for row in _load_jsonl(Path(path))]
    summary = run_thinkgain(preds, pairs=pairs)
    _dump_json(Path(args.out), summary)
    print(f"ThinkGain -> {args.out} ({len(pairs)} pairs)")
    return 0


# --- Parser ---------------------------------------------------------------


def _add_dataset_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--dataset-dir", required=True, help="Local copy of the Hugging Face release.")
    s.add_argument("--labels", default=str(LABELS_PATH), help="Per-image operation / answer labels.")


def _add_api_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--concurrency", type=int, default=1)
    s.add_argument("--timeout", type=int, default=120)
    s.add_argument("--max-retries", type=int, default=2)
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--max-tokens", type=int, default=None)
    s.add_argument("--top-p", type=float, default=None)
    s.add_argument("--top-k", type=int, default=None)
    s.add_argument("--enable-thinking", choices=["true", "false"], default=None,
                   help="Set chat_template_kwargs.enable_thinking (Qwen-style vLLM servers).")
    s.add_argument("--extra-payload-json", default=None,
                   help="Extra JSON fields merged into the chat/completions request body.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StemBind evaluation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                      help="Stub model and judge, no network (default).")
    mode.add_argument("--live", dest="dry_run", action="store_false",
                      help="Call the OpenAI-compatible endpoint in OPENAI_API_BASE.")
    sp = parser.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("validate", help="Load the dataset and print a summary")
    _add_dataset_args(s)
    s.set_defaults(fn=cmd_validate)

    s = sp.add_parser("run", help="Generate predictions")
    _add_dataset_args(s)
    s.add_argument("--models", nargs="+", required=True, metavar="MODEL",
                   help="Model names; use label=model to record results under a different name.")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--conditions", nargs="+", default=["full-image"], choices=list(VALID_CONDITIONS))
    s.add_argument("--tasks", nargs="+", default=["F", "P", "R"], choices=["F", "P", "R"])
    s.add_argument("--limit", type=int, default=None, help="Only use the first N stems.")
    s.add_argument("--max-p-per-item", type=int, default=99)
    s.add_argument("--max-side", type=int, default=768, help="Resize images to this longest side.")
    s.add_argument("--jpeg-quality", type=int, default=80)
    _add_api_args(s)
    s.set_defaults(fn=cmd_run)

    s = sp.add_parser("answer-judge", help="LLM-judged answer correctness")
    _add_dataset_args(s)
    s.add_argument("--predictions", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--judge-model", default="gpt-4o")
    _add_api_args(s)
    s.set_defaults(fn=cmd_answer_judge)

    s = sp.add_parser("judge", help="L2 S1-S4 trace judge")
    _add_dataset_args(s)
    s.add_argument("--predictions", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--judge-model", default="gpt-4o")
    s.add_argument("--n-repeat", type=int, default=1)
    _add_api_args(s)
    s.set_defaults(fn=cmd_judge)

    s = sp.add_parser("metrics", help="Compute all metrics")
    s.add_argument("--predictions", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(fn=cmd_metrics)

    s = sp.add_parser("ssa", help="Score SSA runs")
    s.add_argument("--predictions", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-boot", type=int, default=1000)
    s.set_defaults(fn=cmd_ssa)

    s = sp.add_parser("thinkgain", help="Paired direct-vs-thinking deltas")
    s.add_argument("--predictions", nargs="+", required=True, help="One or more judged prediction files.")
    s.add_argument("--out", required=True)
    s.add_argument("--pairs", nargs="+", required=True, metavar="DIRECT:THINKING",
                   help="Model name pairs, e.g. qwen3.5-27b:qwen3.5-27b-thinking")
    s.set_defaults(fn=cmd_thinkgain)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
