---
name: evolve-refine
description: >-
  Generates an improved version of an existing attack skill by analyzing its
  strategy_prompt/wrap_function, identifying weak points based on failure analysis,
  and producing a refined version that addresses the specific failure mode.
metadata:
  category: evolve
  stage:
  - evolve
---

# evolve-refine

## Role

Reads the target skill's actual content (strategy_prompt or wrap_function_code), combines it with the failure analysis, and generates an improved version that fixes identified weaknesses.

## Inputs

- **action_args.skill_name** — target skill to refine
- **action_args.dominant_dimensions** — failure dimensions (Refusal Triggered / Semantic Drift / Insufficient Context)
- **action_args.analysis** — root cause from failure-analyzer

## Behavior

1. Reads target skill's `run.py` and extracts the core content
2. Detects skill mode (llm_rewrite / deterministic_template / hybrid)
3. Builds improvement instruction based on failure type and extracted content
4. Calls `generate_and_validate_skill` with mode preservation enforced
5. Output: new skill under `skills/new_skills/` that is immediately invoked
