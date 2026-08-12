---
name: rewrite-base64
description: >-
  Base64 encoding attack — encodes the seed prompt in Base64 with various framing
  strategies (full encode+decode instruction, input-only, output-only, raw).
metadata:
  category: attack
  stage:
  - search
  mode: deterministic_template
---

# rewrite-base64

## Technique

Encode the seed prompt using Base64 with four framing variants: (0) full encode with explicit decode-and-respond instruction, (1) encode with neutral "respond normally" framing, (2) plain input requesting Base64-encoded output, (3) raw Base64 with no instruction wrapper. Deterministic — no LLM call needed, 100% intent preservation through verbatim encoding.

## I/O protocol

| Direction | Format |
|-----------|--------|
| stdin     | `SkillContext` JSON |
| stdout    | `SkillExecutionResult` JSON |
