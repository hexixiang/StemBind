#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OUT_DIR="outputs/smoke"
FIXTURE_DIR="${OUT_DIR}/synthetic_fixture"
RUN_DIR="${OUT_DIR}/run"
JUDGED="${OUT_DIR}/predictions_answer_judged.jsonl"
METRICS_DIR="${OUT_DIR}/metrics"

rm -rf "${OUT_DIR}"
mkdir -p "${FIXTURE_DIR}/manifests" "${RUN_DIR}" "${METRICS_DIR}"

cat > "${FIXTURE_DIR}/manifests/items.jsonl" <<'JSONL'
{"item_id":"synthetic_001","source":"synthetic_fixture","image_path":"items/synthetic_001.png"}
{"item_id":"synthetic_002","source":"synthetic_fixture","image_path":"items/synthetic_002.png"}
JSONL

cat > "${FIXTURE_DIR}/answers.jsonl" <<'JSONL'
{"item_id":"synthetic_001","answer":"A"}
{"item_id":"synthetic_002","answer":"B"}
JSONL

cat > "${FIXTURE_DIR}/machine_labels_with_ability.jsonl" <<'JSONL'
{"item_id":"synthetic_001","A_operation_type":"RI-Pos"}
{"item_id":"synthetic_002","A_operation_type":"VP-Rot"}
JSONL

cat > "${FIXTURE_DIR}/annotations.jsonl" <<'JSONL'
{"item_id":"synthetic_001","parsed":{"f_stem_en":"Synthetic fixture: choose the option that follows a left-to-right position cycle.","r_probe":{"q":"Which rule governs the synthetic sequence?","choices":["position cycle","color swap","count increase","mirror flip"],"a":"position cycle","answer_index":0},"p_probes":[{"q":"Where is the marker in the first synthetic panel?","a":"left cell","panel_ref":"panel_1","attr_type":"position"},{"q":"Where is the marker in the second synthetic panel?","a":"center cell","panel_ref":"panel_2","attr_type":"position"}],"steps_en":{"S1_encode":"Identify marker positions in the synthetic panels.","S2_infer":"Infer a left-to-right position cycle.","S3_map":"Map the next position to the candidate slots.","S4_apply":"Select option A."}}}
{"item_id":"synthetic_002","parsed":{"f_stem_en":"Synthetic fixture: choose the option equivalent to a rotated block.","r_probe":{"q":"Which rule governs the synthetic item?","choices":["rotation equivalence","size ordering","line style","quantity addition"],"a":"rotation equivalence","answer_index":0},"p_probes":[{"q":"What is the synthetic block's coarse shape?","a":"L shape","panel_ref":"panel_1","attr_type":"shape"},{"q":"Which transformation is visually suggested?","a":"rotation","panel_ref":"panel_2","attr_type":"orientation"}],"steps_en":{"S1_encode":"Identify the synthetic L-shaped block.","S2_infer":"Infer rotation equivalence.","S3_map":"Compare rotated candidates to the target.","S4_apply":"Select option B."}}}
JSONL

python3 evaluation/cli.py validate \
  --dataset-dir "${FIXTURE_DIR}"

python3 evaluation/cli.py --dry-run run \
  --dataset-dir "${FIXTURE_DIR}" \
  --out-dir "${RUN_DIR}" \
  --models stub-model \
  --conditions text-only \
  --limit 2 \
  --max-p-per-item 2

python3 evaluation/cli.py --dry-run answer-judge \
  --predictions "${RUN_DIR}/predictions.jsonl" \
  --dataset-dir "${FIXTURE_DIR}" \
  --out "${JUDGED}" \
  --judge-model stub-judge \
  --max-p-per-item 2

python3 evaluation/cli.py metrics \
  --predictions "${JUDGED}" \
  --out-dir "${METRICS_DIR}"

echo "Smoke test completed: ${OUT_DIR}"
