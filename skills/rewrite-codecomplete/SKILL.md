---
name: rewrite-codecomplete
description: >-
  Embeds the seed prompt into a Python code completion task where the model is
  asked to enhance print() statements, making it appear as a programming exercise.
metadata:
  category: attack
  stage:
  - search
  mode: deterministic_template
---

# rewrite-codecomplete

## Technique

Wraps the seed prompt into a Python function where the model is asked to "improve" print statements. The seed prompt becomes the function's purpose description, and the model fills in detailed steps as print content. Deterministic — no LLM call needed, 100% intent preservation. Based on ReNeLLM (NAACL 2024) code completion scenario nesting.

## I/O protocol

| Direction | Format |
|-----------|--------|
| stdin     | `SkillContext` JSON |
| stdout    | `SkillExecutionResult` JSON |
