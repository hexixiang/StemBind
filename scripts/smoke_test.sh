#!/usr/bin/env bash
# Offline check of the full pipeline on a two-stem synthetic fixture.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=outputs/smoke
DATA="${OUT}/data"
rm -rf "${OUT}" && mkdir -p "${DATA}"

cat > "${DATA}/metadata.jsonl" <<'JSONL'
{"file_name":"images/000001.png","F":"Choose the option that follows the left-to-right position cycle.","R":{"q":"Which rule governs the sequence?","a":"The marker moves one cell to the right in each panel.","difficulty":"medium"},"P":[{"q":"Where is the marker in the first panel?","a":"left cell","panel_ref":"panel_1","attr_type":"position","difficulty":"easy"},{"q":"Where is the marker in the second panel?","a":"center cell","panel_ref":"panel_2","attr_type":"position","difficulty":"easy"}],"s1":"Identify marker positions in the panels.","s2":"Infer a left-to-right position cycle.","s3":"Map the next position to the missing slot.","s4":"Only option A places the marker in the right cell, so A is correct."}
{"file_name":"images/000002.png","F":"Choose the option equivalent to the rotated block.","R":{"q":"Which rule governs the item?","a":"Rotation equivalence.","difficulty":"medium"},"P":[{"q":"What is the block's coarse shape?","a":"L shape","panel_ref":"panel_1","attr_type":"shape","difficulty":"easy"}],"s1":"Identify the L-shaped block.","s2":"Infer rotation equivalence.","s3":"Compare rotated candidates to the target.","s4":"Option B is the rotated block, so B is correct."}
JSONL

cat > "${DATA}/labels.jsonl" <<'JSONL'
{"file_name":"images/000001.png","operation":"RI-Pos","perception_load":"Type-R","answer":"A"}
{"file_name":"images/000002.png","operation":"VP-Rot","perception_load":"Type-P","answer":"B"}
JSONL

CLI="python3 evaluation/cli.py"
DS=(--dataset-dir "${DATA}" --labels "${DATA}/labels.jsonl")

${CLI} validate "${DS[@]}"
${CLI} run "${DS[@]}" --models stub-model --out-dir "${OUT}/run"
${CLI} answer-judge "${DS[@]}" --predictions "${OUT}/run/predictions.jsonl"
${CLI} judge "${DS[@]}" --predictions "${OUT}/run/predictions_answer_judged.jsonl"
${CLI} metrics --predictions "${OUT}/run/predictions_judged.jsonl" --out-dir "${OUT}/metrics"
${CLI} run "${DS[@]}" --models stub-model --out-dir "${OUT}/ssa" --tasks F \
  --conditions H0 H1 H2 H3 H4 H1_irrelevant
${CLI} ssa --predictions "${OUT}/ssa/predictions.jsonl" --out "${OUT}/ssa/ssa.json" --n-boot 100

${CLI} run "${DS[@]}" --models stub-think=stub-model --out-dir "${OUT}/think"
${CLI} thinkgain --pairs stub-model:stub-think --out "${OUT}/thinkgain.json" \
  --predictions "${OUT}/run/predictions.jsonl" "${OUT}/think/predictions.jsonl"

echo "Smoke test passed: ${OUT}"
