"""Load a StemBind-format dataset into typed evaluation records.

One F item yields up to:
  - 1 F task (full item)
  - K P probes
  - 1 R probe

Optional human-annotated fields (D_perception_type, μ_H/σ_H) are read opportunistically
from the annotated jsonl and attached to the record when present. Missing values are
left as None and downstream metrics fall back to "[TBD]" / "unknown".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


# --- Typed records --------------------------------------------------------


@dataclass
class PProbe:
    q: str
    a: str
    panel_ref: str = ""
    attr_type: str = ""


@dataclass
class RProbe:
    q: str
    choices: list[str]
    a: str
    answer_index: int = -1


@dataclass
class ItemRecord:
    item_id: str
    source: str
    image_path: Path
    operation: str          # 9-class AVR operation (RI-* / VP-*)
    gold_letter: str        # A/B/C/D
    f_question: str
    p_probes: list[PProbe] = field(default_factory=list)
    r_probe: RProbe | None = None
    gt_steps: dict[str, str] = field(default_factory=dict)  # S1_encode..S4_apply
    perception_type: str | None = None                      # Type-P / Type-R / Type-PR
    caption: str | None = None      # Exp5 caption-only text
    parse: str | None = None        # Exp5 parse-only symbolic description


@dataclass
class Task:
    task_id: str           # "{item_id}::F" / "{item_id}::P{i}" / "{item_id}::R"
    item_id: str
    task_type: str         # "F" | "P" | "R"
    source: str
    operation: str
    image_path: Path
    question: str
    gold: str              # letter for F/R, free-text for P
    gold_index: int = -1   # for R (index into choices)
    choices: list[str] = field(default_factory=list)
    panel_ref: str = ""
    attr_type: str = ""
    perception_type: str | None = None
    gt_steps: dict[str, str] = field(default_factory=dict)
    caption: str | None = None
    parse: str | None = None


# --- JSONL helpers --------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_item(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["item_id"]: row for row in rows if row.get("item_id")}


def _annotation_payload(row: dict[str, Any]) -> dict[str, Any]:
    parsed = row.get("parsed")
    if isinstance(parsed, dict):
        return parsed
    return row


# --- Dataset loading ------------------------------------------------------


def load_dataset(
    dataset_dir: Path,
    *,
    annotated_path: Path | None = None,
    caption_path: Path | None = None,
    parse_path: Path | None = None,
) -> list[ItemRecord]:
    """Assemble unified ItemRecord list from the dataset directory.

    Required files (as produced by scripts/prepare_data.py):
      manifests/items.jsonl, answers.jsonl,
      machine_labels_with_ability.jsonl, annotations.jsonl

    Optional:
      annotated_path   human-verified jsonl with D_perception_type / overrides
      caption_path     jsonl with {item_id, caption}  (Exp5 caption-only input)
      parse_path       jsonl with {item_id, parse}    (Exp5 parse-only input)
    """
    dataset_dir = Path(dataset_dir)
    manifest = load_jsonl(dataset_dir / "manifests" / "items.jsonl")
    answers = index_by_item(load_jsonl(dataset_dir / "answers.jsonl"))
    labels = index_by_item(load_jsonl(dataset_dir / "machine_labels_with_ability.jsonl"))
    probe_path = dataset_dir / "probe_prelabels.jsonl"
    probes = index_by_item(load_jsonl(probe_path)) if probe_path.exists() else {}
    annotations_path = dataset_dir / "annotations.jsonl"
    annotations = (
        index_by_item(load_jsonl(annotations_path))
        if annotations_path.exists()
        else {}
    )

    # Reasoning (S1..S4 ground truth) – field name in machine labels is already
    # "steps" (prelabelled). If a human-verified "E_steps" override exists use that.
    reasoning = {}
    reasoning_path = dataset_dir / "reasoning.jsonl"
    if reasoning_path.exists():
        reasoning = index_by_item(load_jsonl(reasoning_path))

    annotated = index_by_item(load_jsonl(annotated_path)) if annotated_path else {}
    captions = index_by_item(load_jsonl(caption_path)) if caption_path else {}
    parses = index_by_item(load_jsonl(parse_path)) if parse_path else {}

    records: list[ItemRecord] = []
    for row in manifest:
        item_id = row["item_id"]
        answer_row = answers.get(item_id, {})
        gold = (answer_row.get("answer") or "").strip().upper()
        label_row = labels.get(item_id, {})
        op = label_row.get("A_operation_type") or label_row.get("ability") or ""
        probe_row = probes.get(item_id, {}) or {}
        ann_row = annotated.get(item_id, {})
        english_ann = _annotation_payload(annotations.get(item_id, {}))

        steps = (
            english_ann.get("steps_en")
            or ann_row.get("E_steps")
            or ann_row.get("steps")
            or label_row.get("E_steps")
            or label_row.get("steps")
            or {}
        )

        english_p = english_ann.get("p_probes") or []
        legacy_p = probe_row.get("P_probes") or []
        p_source = english_p if english_p else legacy_p
        p_probes = [
            PProbe(
                q=str(p.get("q", "")).strip(),
                a=str(p.get("a", "")).strip(),
                panel_ref=str(p.get("panel_ref", "")).strip(),
                attr_type=str(p.get("attr_type", "")).strip(),
            )
            for p in p_source
        ]
        r_raw = english_ann.get("r_probe") or probe_row.get("R_probe") or {}
        r_probe: RProbe | None = None
        choices = r_raw.get("choices") or []
        a_text = str(r_raw.get("a", "")).strip()
        if str(r_raw.get("q", "")).strip() and a_text:
            idx = -1
            if isinstance(choices, list) and choices:
                idx = (
                    choices.index(a_text)
                    if a_text in choices
                    else int(r_raw.get("answer_index", -1))
                )
            r_probe = RProbe(
                q=str(r_raw.get("q", "")).strip(),
                choices=[str(c) for c in choices] if isinstance(choices, list) else [],
                a=a_text,
                answer_index=idx,
            )

        records.append(
            ItemRecord(
                item_id=item_id,
                source=str(row.get("source", "unknown")),
                image_path=dataset_dir / row.get("image_path", f"items/{item_id}.png"),
                operation=str(op),
                gold_letter=gold,
                f_question=str(
                    english_ann.get("f_stem_en")
                    or "Solve the full visual reasoning problem in the image and choose the correct option from A, B, C, or D."
                ).strip(),
                p_probes=p_probes,
                r_probe=r_probe,
                gt_steps={str(k): str(v) for k, v in (steps or {}).items()},
                perception_type=ann_row.get("D_perception_type")
                or row.get("D_perception_type"),
                caption=(captions.get(item_id) or {}).get("caption"),
                parse=(parses.get(item_id) or {}).get("parse"),
            )
        )
    return records


def expand_tasks(
    records: Iterable[ItemRecord],
    *,
    include_f: bool = True,
    include_p: bool = True,
    include_r: bool = True,
    max_p_per_item: int = 99,
) -> list[Task]:
    """Expand ItemRecords into flat Task list, one entry per F / P / R instance."""
    out: list[Task] = []
    for rec in records:
        common = {
            "item_id": rec.item_id,
            "source": rec.source,
            "operation": rec.operation,
            "image_path": rec.image_path,
            "perception_type": rec.perception_type,
            "gt_steps": rec.gt_steps,
            "caption": rec.caption,
            "parse": rec.parse,
        }
        if include_f and rec.gold_letter in {"A", "B", "C", "D"}:
            out.append(
                Task(
                    task_id=f"{rec.item_id}::F",
                    task_type="F",
                    question=rec.f_question,
                    gold=rec.gold_letter,
                    **common,
                )
            )
        if include_p:
            for i, p in enumerate(rec.p_probes[:max_p_per_item], 1):
                out.append(
                    Task(
                        task_id=f"{rec.item_id}::P{i}",
                        task_type="P",
                        question=p.q,
                        gold=p.a,
                        panel_ref=p.panel_ref,
                        attr_type=p.attr_type,
                        **common,
                    )
                )
        if include_r and rec.r_probe is not None:
            out.append(
                Task(
                    task_id=f"{rec.item_id}::R",
                    task_type="R",
                    question=rec.r_probe.q,
                    gold=rec.r_probe.a,
                    gold_index=rec.r_probe.answer_index,
                    choices=list(rec.r_probe.choices),
                    **common,
                )
            )
    return out


def iter_records(dataset_dir: Path, **kwargs: Any) -> Iterator[ItemRecord]:
    yield from load_dataset(dataset_dir, **kwargs)
