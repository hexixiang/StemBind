# StemBind Dataset Schema

The evaluation CLI reads a dataset directory with the layout below.
`scripts/prepare_data.py` builds it from the
[Hugging Face release](https://huggingface.co/datasets/user48271/Stembind).

```text
DATASET_DIR/
  manifests/
    items.jsonl
  answers.jsonl
  machine_labels_with_ability.jsonl
  annotations.jsonl
```

Optional legacy or auxiliary files:

```text
DATASET_DIR/
  probe_prelabels.jsonl
  reasoning.jsonl
  captions.jsonl
  parses.jsonl
```

## Required Files

`manifests/items.jsonl`

- `item_id`: stable stem identifier.
- `source`: source or split label.
- `image_path`: image path relative to `DATASET_DIR`; required for
  `full-image`, unused by `text-only` dry runs.

`answers.jsonl`

- `item_id`: identifier matching the manifest.
- `answer`: final F-task answer letter, normalized to `A`/`B`/`C`/`D`.

`machine_labels_with_ability.jsonl`

- `item_id`: identifier matching the manifest.
- `A_operation_type`: one of `RI-Pos`, `RI-Sty`, `RI-Attr`, `RI-Qty`,
  `RI-Rel`, `VP-Fold`, `VP-View`, `VP-Rot`, `VP-Closure`.

`annotations.jsonl`

- `item_id`: identifier matching the manifest.
- `parsed.f_stem_en`: English full-item prompt.
- `parsed.p_probes`: list of perception probes with `q`, `a`, `panel_ref`,
  and `attr_type`.
- `parsed.r_probe`: rule probe with `q`, optional `choices`, `a`, and optional
  `answer_index`.
- `parsed.steps_en`: S1--S4 process targets using keys `S1_encode`,
  `S2_infer`, `S3_map`, and `S4_apply`.

## Input Conditions

- `text-only`: evaluates from textual task fields only.
- `caption-only`: requires an auxiliary captions JSONL.
- `parse-only`: requires an auxiliary parses JSONL.
- `full-image`: requires images referenced by `manifests/items.jsonl`.

## Answer Key

The Hugging Face release states the F answer letter inside the `s4` target.
`data/stembind_labels.jsonl` lists it per image together with the operation
label, and `prepare_data.py` copies both into `answers.jsonl` and
`machine_labels_with_ability.jsonl`.
