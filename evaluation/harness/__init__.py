"""StemBind evaluation harness.

Modular components:

- data_loader: load unified dataset into evaluation records
- prompt_builder: F / P / R / L2-trace prompts (only locks <ANSWER>X</ANSWER>)
- input_conditions: text-only / caption-only / parse-only / full-image
- model_router: route model names to API callers (real + stub)
- runner: orchestrate calls with checkpointed cache
- judge: LLM judge for L2 S1-S4 alignment on natural traces
"""
