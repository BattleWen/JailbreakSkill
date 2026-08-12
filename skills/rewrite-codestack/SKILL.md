---
name: rewrite-codestack
description: >-
  CodeAttack — wraps the seed prompt into a Python code completion task using deque
  stack operations. Words are pushed in reverse order, then the code asks the model
  to decode the stack and complete the task.
metadata:
  category: attack
  stage:
  - search
  mode: deterministic_template
---

# rewrite-codestack

## Technique

Split the seed prompt into words, push them onto a Python deque stack in reverse order (popping reconstructs the original), and wrap it in a Python function with comments instructing the model to "follow the code." Two variants: (0) steps template requesting specific actionable instructions, (1) paragraphs template requesting detailed 100-word paragraphs. Deterministic — no LLM call needed, 100% intent preservation through code completion context.

## I/O protocol

| Direction | Format |
|-----------|--------|
| stdin     | `SkillContext` JSON |
| stdout    | `SkillExecutionResult` JSON |
