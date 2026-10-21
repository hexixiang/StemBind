"""StemBind metrics.

- l1_accuracy         Acc + Wilson CI per (model, task_type, operation, condition)
- binding             R-F gap, stem-level failure decomposition, strict Binding Gap
- l2_stepacc          S1-S4 stage accuracy + first-failure distribution (from judge)
- l3_attrtag          First-failure stage x perception-load attribution labels
- stem_coherent       Per-stem P and R and F joint scoring
- stemcoherent_audit  Rank change / Spearman / lucky-guess audit of StemCoherent
- vg                  Vision Gain (full-image minus text-only) and IBSI
"""
