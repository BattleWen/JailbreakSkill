# Skill extraction tools

This package contains command-line entry points that collect external evidence,
derive reusable skill packages, and evaluate a named skill in isolation.

| Module | Purpose |
| --- | --- |
| `crawl_external_text` | Collect and rank verified external evidence |
| `generate_base_skill_from_external` | End-to-end paper discovery, collection, generation, and optional promotion |
| `generate_base_skill_from_external_text` | Generate from an existing TXT, Markdown, JSON, JSONL, or CSV snapshot |
| `evaluate_skill_asr` | Evaluate one skill against a configured target and guard |

Run every module from the repository root with `python -m`. Use `--help` for
the complete option set. Credentials are read from the environment and must not
be placed in command arguments or committed configuration.
