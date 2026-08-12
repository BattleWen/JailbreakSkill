# SkillTeaming

SkillTeaming is a two-stage LLM red-teaming framework built around reusable
prompt-rewrite skills. Stage 1 searches a fixed skill library with shared UCB
memory; Stage 2 analyzes failures and evolves category-specific skills. The
repository also includes tools that extract new skills from external evidence.

> [!WARNING]
> This repository contains adversarial prompts and red-teaming transformations.
> Use it only on systems you own or are authorized to evaluate, with appropriate
> access controls, monitoring, and human review.

## Repository layout

```text
core/                 Runtime, planning, evaluation, memory, and skill loading
skill_extraction/     External-evidence collection, skill generation, and evaluation
skills/               Stable built-in rewrite and evolution skills
configs/              Safe configuration template and workflow definitions
data/                 AdvBench, HarmBench, and JBB Original benchmark inputs
main.py               Two-stage command-line entry point
```

Runtime outputs such as `runs/`, `memory/`, generated skills, reports, caches,
and local configuration are intentionally ignored.

## Installation

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/BattleWen/SkillTeaming.git
cd SkillTeaming
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
cp configs/config.template.yaml configs/config.yaml
```

Fill `.env` with endpoints, model names, and credentials for your own
OpenAI-compatible services. No deployment-specific endpoint or credential is
committed to this repository. The planner endpoint is inherited by model-backed
skills and meta-skills unless the configuration is extended with role-specific
settings.

Verify the local installation without making an API request:

```bash
python main.py --help
python main.py \
  --config configs/config.yaml \
  --output-dir outputs/smoke \
  --start 0 \
  --end 0
```

## Run Stage 1

The following evaluates one HarmBench row. It makes real calls to the target
and guard endpoints configured in `.env`.

```bash
python main.py \
  --config configs/config.yaml \
  --seed-prompt-file data/HarmBench.jsonl \
  --output-dir outputs/harmbench-demo \
  --start 0 \
  --end 1 \
  --stage1-only
```

Use `--target-query-budget N` to bound target-model calls per Stage 1 prompt,
or `--stage1-skill NAME` to evaluate one rewrite skill.

## Extract a skill from external evidence

Generate one base skill from an exact arXiv paper:

```bash
python -m skill_extraction.generate_base_skill_from_external \
  --arxiv-id 2512.23173 \
  --skill-name rewrite-paper-method-2512-23173 \
  --crawl-out outputs/extraction/paper.jsonl \
  --paper-manifest-out outputs/extraction/papers.json \
  --report-out outputs/extraction/generation.json \
  --pipeline-report-out outputs/extraction/pipeline.json \
  --config configs/config.yaml
```

The extraction package also exposes:

```bash
python -m skill_extraction.crawl_external_text --help
python -m skill_extraction.generate_base_skill_from_external_text --help
python -m skill_extraction.evaluate_skill_asr --help
```

Generated packages pass schema and runtime validation before registration.
That validation does not by itself establish paper fidelity or attack
effectiveness; inspect the evidence and reports before using a generated skill.

## Datasets

The repository includes project-formatted copies of AdvBench, HarmBench, and
the 55-row `Source=Original` subset of JBB-Behaviors. They contain harmful or
sensitive requests intended for controlled safety evaluation. Provenance,
transformations, checksums, citations, and third-party license notices are in
[`data/README.md`](data/README.md).

## Configuration and secrets

- Keep credentials only in `.env` or the process environment.
- Never commit `configs/config.yaml`; it is ignored by Git.
- Use `SKILLTEAMING_NO_PROXY` for deployment-specific proxy bypass hosts.
- External discovery credentials are optional and are never written to crawl
  or generation reports.

## Project license

A project-level source-code license has not yet been selected. The licenses in
`data/licenses/` apply only to their respective third-party datasets and do not
grant rights to the SkillTeaming source code.
