"""Load the StemBind release into typed evaluation records.

The dataset directory is a local copy of the Hugging Face release
(`metadata.jsonl` + `images/`). Operation, perception-load and answer labels
come from `data/stembind_labels.jsonl` in this repository, joined on
`file_name`.

One stem yields one F task, one R probe, and its P probes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "stembind_labels.jsonl"
STEP_KEYS = ("S1_encode", "S2_infer", "S3_map", "S4_apply")


# --- Typed records --------------------------------------------------------


@dataclass
class PProbe:
    q: str
    a: str
    panel_ref: str = ""
    attr_type: str = ""


@dataclass
class ItemRecord:
    item_id: str
    image_path: Path
    operation: str                 # one of the nine RI-* / VP-* operations
    perception_load: str | None    # Type-P / Type-R / Type-PR, if tagged
    gold_letter: str               # A/B/C/D
    f_question: str
    r_question: str
    r_answer: str
    p_probes: list[PProbe] = field(default_factory=list)
    gt_steps: dict[str, str] = field(default_factory=dict)  # S1_encode..S4_apply


@dataclass
class Task:
    task_id: str           # "{item_id}::F" / "{item_id}::P{i}" / "{item_id}::R"
    item_id: str
    task_type: str         # "F" | "P" | "R"
    operation: str
    perception_load: str | None
    image_path: Path
    question: str
    gold: str              # letter for F, free text for P / R
    panel_ref: str = ""
    attr_type: str = ""
    gt_steps: dict[str, str] = field(default_factory=dict)
    irrelevant_s1: str = ""  # S1 of another stem, for the SSA control


# --- Loading --------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(dataset_dir: Path, *, labels_path: Path = LABELS_PATH) -> list[ItemRecord]:
    dataset_dir = Path(dataset_dir)
    labels = {r["file_name"]: r for r in load_jsonl(labels_path)}
    records: list[ItemRecord] = []
    for row in load_jsonl(dataset_dir / "metadata.jsonl"):
        file_name = row["file_name"]
        label = labels.get(file_name, {})
        records.append(ItemRecord(
            item_id=Path(file_name).stem,
            image_path=dataset_dir / file_name,
            operation=label.get("operation") or "",
            perception_load=label.get("perception_load"),
            gold_letter=(label.get("answer") or "").strip().upper(),
            f_question=row["F"].strip(),
            r_question=row["R"]["q"].strip(),
            r_answer=row["R"]["a"].strip(),
            p_probes=[
                PProbe(
                    q=str(p.get("q", "")).strip(),
                    a=str(p.get("a", "")).strip(),
                    panel_ref=str(p.get("panel_ref", "")).strip(),
                    attr_type=str(p.get("attr_type", "")).strip(),
                )
                for p in row["P"]
            ],
            gt_steps={key: row[f"s{i}"].strip() for i, key in enumerate(STEP_KEYS, 1)},
        ))
    return records


def expand_tasks(
    records: Iterable[ItemRecord],
    *,
    include_f: bool = True,
    include_p: bool = True,
    include_r: bool = True,
    max_p_per_item: int = 99,
) -> list[Task]:
    """Expand records into a flat task list, one entry per F / P / R instance."""
    records = list(records)
    n = len(records)
    out: list[Task] = []
    for i, rec in enumerate(records):
        # SSA irrelevant-S1 control: S1 of the stem half the dataset away.
        other = records[(i + n // 2) % n] if n > 1 else rec
        common = {
            "item_id": rec.item_id,
            "operation": rec.operation,
            "perception_load": rec.perception_load,
            "image_path": rec.image_path,
            "gt_steps": rec.gt_steps,
        }
        if include_f and rec.gold_letter in {"A", "B", "C", "D"}:
            out.append(Task(
                task_id=f"{rec.item_id}::F", task_type="F", question=rec.f_question,
                gold=rec.gold_letter, irrelevant_s1=other.gt_steps.get("S1_encode", ""),
                **common,
            ))
        if include_p:
            for j, p in enumerate(rec.p_probes[:max_p_per_item], 1):
                out.append(Task(
                    task_id=f"{rec.item_id}::P{j}", task_type="P", question=p.q, gold=p.a,
                    panel_ref=p.panel_ref, attr_type=p.attr_type, **common,
                ))
        if include_r and rec.r_question:
            out.append(Task(
                task_id=f"{rec.item_id}::R", task_type="R", question=rec.r_question,
                gold=rec.r_answer, **common,
            ))
    return out
