"""StemBind metrics.

- l1_accuracy       Acc + Wilson CI per (model, task_type, source, operation)
- l2_stepacc        Stage accuracy + first-failure distribution (from judge)
- l3_attrtag        First-failure × perception-type attribution labels
- stem_coherent     P ∧ R ∧ F per-item joint scoring
- hcag              Human-Calibrated Ability Gap (human-SD weighted)
- vg                Vision Gain and IBSI
- stemcoherent_audit  Rank-change / Spearman / chance / lucky-guess audit
"""
