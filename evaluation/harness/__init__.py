"""StemBind evaluation harness.

- data_loader       load the Hugging Face release into F / P / R tasks
- input_conditions  full-image, text-only and SSA H0..H4 conditions
- prompt_builder    prompts and <ANSWER> parsing
- model_router      OpenAI-compatible caller and offline stub
- runner            cached, resumable prediction runs
- answer_judge      LLM-judged answer correctness for free-form P / R answers
- judge             L2 judge aligning free traces to S1-S4 targets
"""
