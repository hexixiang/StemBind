# Evaluation

All commands go through `evaluation/cli.py` and run in dry-run mode (stub model, no API calls) unless `--live` is given. `--live` reads `OPENAI_API_BASE` and `OPENAI_API_KEY` and works with any OpenAI-compatible `/chat/completions` endpoint.

## Data

`--dataset-dir` points to a local copy of the [Hugging Face release](https://huggingface.co/datasets/user48271/Stembind):

```text
data/stembind/
├── metadata.jsonl   # one stem per line: file_name, F, R, P, s1-s4
└── images/          # 000000.png ... 002297.png
```

| Field | Content |
|---|---|
| `F` | Full-item instruction (four options, A-D, shown in the image) |
| `R` | Rule probe `{q, a}`, free-form |
| `P` | Perception probes `[{q, a, panel_ref, attr_type}]`, 6-7 per stem |
| `s1`-`s4` | S1 Encode, S2 Infer, S3 Map, S4 Apply targets |

The operation type, perception-load tag and F answer letter of each image are in [`data/stembind_labels.jsonl`](../data/stembind_labels.jsonl), joined on `file_name`.

## Pipeline

| Step | Command | Output |
|---|---|---|
| Predictions | `run` | `predictions.jsonl` |
| Answer grading (P / R are free-form) | `answer-judge` | `predictions_answer_judged.jsonl` |
| S1-S4 trace judging (optional) | `judge` | `predictions_judged.jsonl` |
| Metrics | `metrics` | `metrics/*.json` |

`run` caches every call under `<out-dir>/cache/`, so an interrupted run resumes where it stopped. Use `--concurrency` for parallel requests and `--limit N` to try the first N stems.

## Metrics

| File | Metric |
|---|---|
| `metrics_l1.json` | P / R / F accuracy with Wilson 95% CI, by operation and condition |
| `binding.json` | R-F gap, stem-level failure decomposition (solved / can't bind / can't reason / can't see), strict Binding Gap `1 - P(F=1 \| P=1, R=1)` |
| `metrics_l2.json` | L2 StepAcc per stage S1-S4, first-failure distribution, lucky-guess rate |
| `metrics_l3.json` | L3 AttrTag: first-failure stage crossed with the perception-load tag |
| `stem_coherent.json`, `stem_coherent_audit.json` | StemCoherent (all P, R and F correct on a stem) and its rank audit |
| `vg.json` | Vision Gain: F accuracy with the image minus text-only (needs a `--conditions text-only` run) |

## Diagnostics

**SSA (Stage-wise Stimulus Augmentation).** F items are re-run with verified stage content added cumulatively: `H0` (none) to `H4` (S1-S4), plus the `H1_irrelevant` control (S1 of another stem). Option letters are masked in all injected text.

```bash
python evaluation/cli.py --live run --dataset-dir data/stembind --models your-model \
  --tasks F --conditions H0 H1 H2 H3 H4 H1_irrelevant --out-dir outputs/ssa
python evaluation/cli.py ssa --predictions outputs/ssa/predictions.jsonl --out outputs/ssa/ssa.json
```

**ThinkGain.** Thinking minus direct accuracy on P, R and F. When both modes use the same served model, give each run its own label with `label=model`:

```bash
python evaluation/cli.py --live run --dataset-dir data/stembind --out-dir outputs/qwen-direct \
  --models qwen3.5-27b=Qwen/Qwen3.5-27B --enable-thinking false
python evaluation/cli.py --live run --dataset-dir data/stembind --out-dir outputs/qwen-thinking \
  --models qwen3.5-27b-thinking=Qwen/Qwen3.5-27B --enable-thinking true
# after answer-judge on both runs:
python evaluation/cli.py thinkgain --pairs qwen3.5-27b:qwen3.5-27b-thinking --out thinkgain.json \
  --predictions outputs/qwen-{direct,thinking}/predictions_answer_judged.jsonl
```

## Offline check

```bash
bash scripts/smoke_test.sh
```

Runs every command above on a two-stem synthetic fixture without network access.
