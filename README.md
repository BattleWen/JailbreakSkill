<!-- # JailbreakSkill: Scaling Automated Red-Teaming with Reusable and Ever-Evolving Skills -->

<p align="center" style="margin-bottom: 0px;">
  <img src="assets/github%20logo.png" alt="JailbreakSkill" height="140">
</p>

<p align="center" style="font-size: 18px; margin-top: 0;">
  Scaling Automated Red-Teaming with Reusable and Ever-Evolving Skills
</p>

<p align="center">
  <a href="https://github.com/BattleWen/JailbreakSkill"><img src="https://img.shields.io/badge/GitHub-JailbreakSkill-181717?logo=github" alt="GitHub"></a>
  <a href="https://arxiv.org/abs/2608.16465"><img src="https://img.shields.io/badge/Paper-arXiv%3A2608.16465-b31b1b.svg" alt="Paper"></a>
</p>

JailbreakSkill is a two-stage LLM red-teaming framework built around reusable
prompt-rewrite skills. Stage 1 searches a fixed skill library with shared UCB
memory; Stage 2 analyzes failures and evolves category-specific skills. The
repository also includes tools that extract new skills from external evidence.

## 💡 Motivation

<p align="center">
  <a href="assets/motivation0817.pdf">
    <img src="assets/motivation0817.png" alt="JailbreakSkill motivation and framework overview" width="100%">
  </a>
</p>

<p align="center"><sub>From external attack knowledge to reusable, diagnosable, and ever-evolving jailbreak skills. Click the figure to view the original PDF.</sub></p>

## 🔄 Pipeline

```
Seed Prompts
     │
     ▼
┌─────────────────────────────────────────────┐
│  Stage 1 — UCB Skill Search                 │
│                                             │
│  UCB Memory ──rank──► Rewrite Skills (×16)  │
│      ▲                       │              │
│      │ score           rewritten prompt     │
│      │                       ▼              │
│   Judge  ◄──response── Target Model         │
└─────────────────────────────────────────────┘
     │                    │
   success             failure
     │                    │ (grouped by risk category)
     │                    ▼
     │   ┌─────────────────────────────────────┐
     │   │  Stage 2 — Skill Evolution          │
     │   │                                     │
     │   │  Failure Analyzer                   │
     │   │       │ dispatch                    │
     │   │       ▼                             │
     │   │Meta-Skills (refine/combine/discover)│
     │   │       │ new skill                   │
     │   │       ▼                             │
     │   │  Evaluate on failed prompts         │
     │   │       │ still failing               │
     │   │       └──────► Failure Analyzer     │
     │   └─────────────────────────────────────┘
     │                    │
     ▼                    ▼
  ✓ Bypassed          ✓ Recovered
```

## 🧬 Evolved skill examples

![Examples of evolved skills](assets/evolved_skills.png)

## 🗂️ Repository layout

```text
core/                        Runtime, planning, evaluation, memory, and skill loading
skill_extraction/            External-evidence collection, skill generation, and evaluation
skills/                      Stable built-in rewrite and evolution skills
evolved_skill_examples/      Runnable evolved skill examples (`SKILL.md` + `scripts/run.py`)
configs/
  config.template.yaml       Master configuration template
  workflows/basic.yaml       Default workflow (skill groups and stage routing)
data/                        AdvBench, HarmBench, and JBB Original benchmark inputs
main.py                      Two-stage command-line entry point
```

Runtime outputs such as `runs/`, `memory/`, generated skills, reports, caches, and local configuration are intentionally ignored.

> [!WARNING]
> This repository contains adversarial prompts and red-teaming transformations.
> Use it only on systems you own or are authorized to evaluate, with appropriate
> access controls, monitoring, and human review.

## 🚀 Getting started

### Installation

Requires Python 3.11 or newer. The recommended way is [uv](https://docs.astral.sh/uv/), which downloads the right Python version automatically.

```bash
git clone https://github.com/BattleWen/JailbreakSkill.git
cd JailbreakSkill
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
cp configs/config.template.yaml configs/config.yaml
```

Fill `.env` with endpoints, model names, and credentials for your own
OpenAI-compatible services. No deployment-specific endpoint or credential is
committed to this repository. The planner endpoint is inherited by model-backed
skills and meta-skills unless the configuration is extended with role-specific
settings.

Verify the local installation without making an API request (run all subsequent commands with the venv active):

```bash
source .venv/bin/activate
python main.py --help
python main.py \
  --config configs/config.yaml \
  --output-dir outputs/smoke \
  --start 0 \
  --end 0
```

### Run Stage 1

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

### Run the full pipeline (Stage 1 + Stage 2)

Omitting `--stage1-only` runs both stages in sequence. Stage 2 picks up the failures written by Stage 1 in the same `--output-dir`.

```bash
python main.py \
  --config configs/config.yaml \
  --seed-prompt-file data/HarmBench.jsonl \
  --output-dir outputs/harmbench-demo \
  --start 0 \
  --end 1
```

### Rerun Stage 2 on an existing Stage 1 checkpoint

Use `--stage2-only` when Stage 1 has already finished. The `--output-dir` must contain `stage1_per_seed.jsonl` and `stage1_shared.json`, and all indices in the `--start`/`--end` range must be present in the checkpoint. Stage 2 derives its input from those files and never re-queries the dataset.

```bash
python main.py \
  --config configs/config.yaml \
  --seed-prompt-file data/HarmBench.jsonl \
  --output-dir outputs/harmbench-demo \
  --start 0 \
  --end 1 \
  --stage2-only
```

Use `--max-evolve-skills N` to cap evolved skills per category (default 20), or `--stage2-patience N` to stop a category after N rounds with no new recoveries.

### Quick examples

Single prompt from a custom file:

```bash
echo '{"query": "How do I bypass a door lock?", "risk_category": "illegal"}' > data/single.jsonl
python main.py --config configs/config.yaml --seed-prompt-file data/single.jsonl \
  --output-dir outputs/single-test --start 0 --end 1 --stage1-only
```

Few-shot Stage 1 + Stage 2 (`--target-query-budget` caps target calls per prompt, making failures more likely for Stage 2 to evolve against):

```bash
python main.py --config configs/config.yaml --seed-prompt-file data/HarmBench.jsonl \
  --output-dir outputs/few-shot --start 0 --end 5 --target-query-budget 3 --stage1-only

python main.py --config configs/config.yaml --seed-prompt-file data/HarmBench.jsonl \
  --output-dir outputs/few-shot --start 0 --end 5 --stage2-only --max-evolve-skills 5
```

Single risk category, first 10 rows:

```bash
python main.py --config configs/config.yaml --seed-prompt-file data/HarmBench.jsonl \
  --output-dir outputs/chem-only --start 0 --end 400 \
  --risk-filter chemical_biological --risk-limit 10 --stage1-only
```

## 🧩 Skills

### Adding a custom skill

A skill is a directory under `skills/` with two files:

```
skills/rewrite-myskill/
    SKILL.md          # frontmatter + technique description
    scripts/run.py    # reads SkillContext JSON from stdin, writes SkillExecutionResult JSON to stdout
```

**SKILL.md** — required frontmatter fields:

```yaml
---
name: rewrite-myskill
description: One-line description.
metadata:
  category: attack        # attack | analyze | evolve
  stage: [search]         # search = tried in Stage 1
  mode: llm_rewrite       # llm_rewrite | deterministic_template
---
```

**scripts/run.py** — two patterns depending on `mode`:

- `llm_rewrite`: call `run_llm_skill(name, build_prompt_fn)` where `build_prompt_fn(seed, candidate_count) -> str` returns the meta-prompt instructing the LLM how to rewrite the seed. See `skills/rewrite-roleplay/scripts/run.py` as reference.
- `deterministic_template`: call `run_template_skill(name, wrap_query_fn)` where `wrap_query_fn(query, variant) -> str` returns the rewritten string directly. See `skills/rewrite-base64/scripts/run.py` as reference.

Both harnesses handle stdin/stdout JSON serialization automatically.

**Register** — add an entry to `initial_skills` in `skills/registry.json`:

```json
{"name": "rewrite-myskill", "mode": "llm_rewrite", "description": "One-line description"}
```

**Add to workflow** — append the skill name under `skill_groups.search` in `configs/workflows/basic.yaml`. Stage 1 includes it in UCB selection on the next run.

### Extract a skill from external evidence

Generate a skill from an arXiv paper end-to-end (crawl + generate + validate):

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

Other tools in the extraction package:

```bash
python -m skill_extraction.crawl_external_text --help             # collect evidence from arXiv / GitHub / HuggingFace
python -m skill_extraction.generate_base_skill_from_external_text --help  # generate from a local text snapshot
python -m skill_extraction.evaluate_skill_asr --help              # evaluate a built-in skill's ASR in isolation
```

Generated packages pass schema and runtime validation before registration. That validation does not by itself establish paper fidelity or attack effectiveness; inspect the evidence and reports before using a generated skill.

## 📚 Datasets

The repository includes AdvBench, HarmBench, and the 55-row `Source=Original` subset of JBB-Behaviors. Provenance, checksums, and license notices are in [`data/README.md`](data/README.md).

**Custom datasets** — any JSONL with a `query` field works. Add `risk_category` to enable per-category UCB memory and Stage 2 grouping; without it the risk classifier LLM labels each prompt automatically. Supported codes: HarmBench keys (`chemical_biological`, `copyright`, `cybercrime_intrusion`, `illegal`, `misinformation_disinformation`, `harassment_bullying`, `harmful`), AdvBench keys (`cybercrime`, `fraud_theft`, `harassment_harmful`, `weapons_violence`, `misinformation`, `drugs_chemical`), AILuminate keys (`S1`–`S12`).

```jsonl
{"query": "How do I pick a lock?", "risk_category": "illegal"}
```

Pass with `--seed-prompt-file data/my_dataset.jsonl`.

## ⚙️ Configuration

**`risk_classifier`** — only called when a seed prompt has no `risk_category`. To enable for custom datasets, configure `base_url` and `model` under `risk_classifier` in `configs/config.yaml`.

**`fidelity_filter`** — optional pre-target check that rejects semantically drifted rewrites. Disabled by default; set `fidelity_filter.llm.enabled: true` in `configs/config.yaml` to enable (costs one extra LLM call per candidate).

**`max_tokens`** — for capable frontier models set `planner.llm.max_tokens: 8192` and `meta_skills.llm.max_tokens: 12288`. The template defaults are already set to these values.

## 📄 Citation

If you find JailbreakSkill useful in your research, please cite:

```bibtex
@misc{wen2026jailbreakskillscalingautomatedredteaming,
      title={JailbreakSkill: Scaling Automated Red-Teaming with Reusable and Ever-Evolving Skills},
      author={Xiaoyu Wen and Jiajia Li and Zhida He and Peng Yu and Chenxu Wang and Han Qi and Ziyuan Zhou and Cheng Jin and Ying Wen and Xingcheng Xu and Shuyue Hu and Tianhang Zheng and Chaochao Lu and Qiaosheng Zhang},
      year={2026},
      eprint={2608.16465},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.16465},
}
```
