"""Convert the Hugging Face StemBind release into the harness dataset layout.

Usage:
    # download from the Hub (needs `pip install huggingface_hub`)
    python scripts/prepare_data.py --out-dir data/stembind

    # or reuse a local snapshot that contains metadata.jsonl and images/
    python scripts/prepare_data.py --hf-dir /path/to/Stembind --out-dir data/stembind

Output (see evaluation/README.md):
    OUT_DIR/manifests/items.jsonl
    OUT_DIR/answers.jsonl
    OUT_DIR/machine_labels_with_ability.jsonl
    OUT_DIR/annotations.jsonl
    OUT_DIR/images -> <snapshot>/images
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

HF_REPO = "user48271/Stembind"
ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "stembind_labels.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _download(cache_dir: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=["metadata.jsonl", "images/*"],
        cache_dir=cache_dir,
    ))


def _link_images(src: Path, dst: Path, copy: bool) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if copy:
        shutil.copytree(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-dir", default=None, help="Local snapshot with metadata.jsonl and images/.")
    ap.add_argument("--cache-dir", default=None, help="Hugging Face cache directory for the download.")
    ap.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them.")
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir) if args.hf_dir else _download(args.cache_dir)
    out = Path(args.out_dir)
    stems = _load_jsonl(hf_dir / "metadata.jsonl")
    labels = {r["file_name"]: r for r in _load_jsonl(LABELS)}

    items, answers, ops, anns = [], [], [], []
    for row in stems:
        fn = row["file_name"]
        item_id = Path(fn).stem
        lab = labels[fn]
        items.append({"item_id": item_id, "source": "stembind", "image_path": fn})
        answers.append({"item_id": item_id, "answer": lab["answer"] or ""})
        ops.append({"item_id": item_id, "A_operation_type": lab["operation"]})
        anns.append({"item_id": item_id, "parsed": {
            "f_stem_en": row["F"],
            "p_probes": row["P"],
            "r_probe": {"q": row["R"]["q"], "a": row["R"]["a"]},
            "steps_en": {
                "S1_encode": row["s1"],
                "S2_infer": row["s2"],
                "S3_map": row["s3"],
                "S4_apply": row["s4"],
            },
        }})

    _dump_jsonl(out / "manifests" / "items.jsonl", items)
    _dump_jsonl(out / "answers.jsonl", answers)
    _dump_jsonl(out / "machine_labels_with_ability.jsonl", ops)
    _dump_jsonl(out / "annotations.jsonl", anns)
    _link_images(hf_dir / "images", out / "images", args.copy_images)

    n_p = sum(len(r["P"]) for r in stems)
    n_f = sum(1 for a in answers if a["answer"])
    print(f"Wrote {len(stems)} stems to {out} (P={n_p}, R={len(stems)}, F with answer key={n_f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
