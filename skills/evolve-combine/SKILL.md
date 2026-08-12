---
name: evolve-combine
description: >-
  Generates a new skill by combining the bypass mechanisms of two existing skills
  into a unified approach that is more effective than either alone.
metadata:
  category: evolve
  stage:
  - evolve
---

# evolve-combine

## Role

Extracts the core content from two source skills, analyzes their respective bypass mechanisms, and generates a combined skill that leverages both.

## Inputs

- **action_args.skill_names** — two skills to combine
- **action_args.dominant_dimensions** — failure dimensions (Refusal Triggered / Semantic Drift / Insufficient Context)
- **action_args.analysis** — root cause from failure-analyzer

## Behavior

1. Reads both source skills' `run.py` and extracts strategy_prompt / wrap_function
2. Builds combination instruction explaining how to merge mechanisms
3. Calls `generate_and_validate_skill` (mode freedom — can pick any mode)
4. Output: new skill under `skills/new_skills/` that is immediately invoked
