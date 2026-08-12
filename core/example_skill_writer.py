"""Generate base rewrite skills from seed/adversarial JSONL examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.meta_skill_writer import (
    _sanitize_skill_name,
    generate_and_validate_skill,
)
from core.skill_loader import SkillLoader
from core.utils import read_markdown_frontmatter, read_yaml


DEFAULT_MAX_EXAMPLES = 50
DEFAULT_MAX_CHARS_PER_FIELD = 1500


class ExampleSkillWriterError(ValueError):
    """Raised when example input or generation options are invalid."""


@dataclass(frozen=True)
class ExamplePair:
    """One seed prompt and its adversarial rewrite."""

    seed_prompt: str
    adversarial_prompt: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""
        return {
            "seed_prompt": self.seed_prompt,
            "adversarial_prompt": self.adversarial_prompt,
        }


@dataclass(frozen=True)
class LoadedExamplePairs:
    """Loaded and sampled example-pair payload."""

    examples: list[ExamplePair]
    examples_total: int


@dataclass(frozen=True)
class GeneratedBaseSkill:
    """Summary of a generated base skill."""

    generated_skill_name: str
    generated_skill_dir: str
    examples_total: int
    examples_used: int
    mode: str
    workflow_registered: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a stable CLI summary payload."""
        return {
            "generated_skill_name": self.generated_skill_name,
            "generated_skill_dir": self.generated_skill_dir,
            "examples_total": self.examples_total,
            "examples_used": self.examples_used,
            "mode": self.mode,
            "workflow_registered": self.workflow_registered,
        }


def load_example_pairs(
    path: Path,
    *,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    max_chars_per_field: int = DEFAULT_MAX_CHARS_PER_FIELD,
) -> LoadedExamplePairs:
    """Read, validate, truncate, and deterministically sample example pairs."""
    if max_examples <= 0:
        raise ExampleSkillWriterError("max_examples must be positive")
    if max_chars_per_field <= 0:
        raise ExampleSkillWriterError("max_chars_per_field must be positive")

    examples: list[ExamplePair] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExampleSkillWriterError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ExampleSkillWriterError(
                    f"{path}:{line_number}: expected a JSON object"
                )

            seed_prompt = _required_text(record, "seed_prompt", path, line_number)
            adversarial_prompt = _required_text(
                record, "adversarial_prompt", path, line_number
            )
            examples.append(
                ExamplePair(
                    seed_prompt=seed_prompt[:max_chars_per_field],
                    adversarial_prompt=adversarial_prompt[:max_chars_per_field],
                )
            )

    if not examples:
        raise ExampleSkillWriterError(f"{path}: no valid example pairs found")

    return LoadedExamplePairs(
        examples=_uniform_sample(examples, max_examples),
        examples_total=len(examples),
    )


def build_generation_context(
    *,
    examples: list[ExamplePair],
    existing_skills: list[str],
    static: bool = False,
) -> dict[str, Any]:
    """Build the meta-skill generation context for example-driven learning."""
    if not examples:
        raise ExampleSkillWriterError("at least one example pair is required")

    if static:
        instruction = (
            "Learn one reusable base rewrite skill from the provided "
            "seed/adversarial prompt pairs. Infer the common transformation "
            "pattern, then create one skill that applies that pattern to new "
            "seed prompts. Do not copy example-specific names, topics, "
            "numbers, or wording except inside brief illustrative examples when needed."
        )
    else:
        instruction = (
            "Learn a reusable base rewrite skill from the provided seed/adversarial "
            "prompt pairs. Infer the common transformation pattern, then create a "
            "general skill that applies that pattern to new seed prompts. Do not "
            "copy example-specific names, topics, numbers, or wording except inside "
            "brief illustrative examples when needed."
        )

    return {
        "task": "learn_from_examples",
        "instruction": instruction,
        "example_pairs": [example.to_dict() for example in examples],
        "existing_skills": list(existing_skills),
        "seed_prompt": examples[0].seed_prompt,
        "static_skill": bool(static),
    }


def generate_base_skill_from_examples(
    *,
    project_root: Path,
    examples_path: Path,
    backend_config: dict[str, Any],
    skill_name: str = "",
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    max_chars_per_field: int = DEFAULT_MAX_CHARS_PER_FIELD,
    overwrite: bool = False,
    workflow_name: str = "basic",
    static: bool = False,
) -> GeneratedBaseSkill:
    """Generate a base skill from example pairs and register it in a workflow."""
    requested_name = _validate_requested_skill_name(skill_name) if skill_name else ""
    loaded = load_example_pairs(
        examples_path,
        max_examples=max_examples,
        max_chars_per_field=max_chars_per_field,
    )
    existing_skills = _discover_existing_skill_names(project_root)
    if requested_name and requested_name in existing_skills and not overwrite:
        raise ExampleSkillWriterError(
            f"Skill '{requested_name}' already exists. Use overwrite to replace it."
        )
    if (
        requested_name
        and requested_name in existing_skills
        and overwrite
        and not (project_root / "skills" / requested_name).exists()
    ):
        raise ExampleSkillWriterError(
            f"Skill '{requested_name}' exists outside the base skills directory; "
            "refusing to create a duplicate base skill."
        )

    generation_context = build_generation_context(
        examples=loaded.examples,
        existing_skills=existing_skills,
        static=static,
    )
    generated_name, skill_dir = generate_and_validate_skill(
        backend_config=backend_config,
        generation_context=generation_context,
        project_root=project_root,
        source_meta_skill="examples-to-base-skill",
        base_skill_name=requested_name,
        existing_skill_names=set(existing_skills),
        destination="base_skills",
        allow_overwrite=overwrite,
        deduplicate_name=not requested_name,
    )

    workflow_registered = register_skill_in_workflow(
        project_root=project_root,
        skill_name=generated_name,
        workflow_name=workflow_name,
    )

    frontmatter = read_markdown_frontmatter(skill_dir / "SKILL.md")
    metadata = frontmatter.get("metadata", {})
    mode = metadata.get("mode", "") if isinstance(metadata, dict) else ""

    return GeneratedBaseSkill(
        generated_skill_name=generated_name,
        generated_skill_dir=str(skill_dir.relative_to(project_root)),
        examples_total=loaded.examples_total,
        examples_used=len(loaded.examples),
        mode=str(mode),
        workflow_registered=workflow_registered,
    )


def register_skill_in_workflow(
    *,
    project_root: Path,
    skill_name: str,
    workflow_name: str = "basic",
) -> bool:
    """Append a skill to a workflow search group if it is not already present."""
    workflow_path = project_root / "configs" / "workflows" / f"{workflow_name}.yaml"
    workflow = read_yaml(workflow_path)
    skill_groups = dict(workflow.get("skill_groups", {}))
    search_group = list(skill_groups.get("search", []))
    if skill_name in search_group:
        return False

    search_group.append(skill_name)
    skill_groups["search"] = search_group
    workflow["skill_groups"] = skill_groups
    workflow_path.write_text(
        yaml.safe_dump(workflow, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return True


def _required_text(
    record: dict[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    value = record.get(key)
    if value is None:
        raise ExampleSkillWriterError(f"{path}:{line_number}: missing field '{key}'")
    text = str(value).strip()
    if not text:
        raise ExampleSkillWriterError(f"{path}:{line_number}: empty field '{key}'")
    return text


def _uniform_sample(examples: list[ExamplePair], max_examples: int) -> list[ExamplePair]:
    if len(examples) <= max_examples:
        return list(examples)
    if max_examples == 1:
        return [examples[0]]

    last_index = len(examples) - 1
    sampled: list[ExamplePair] = []
    seen: set[int] = set()
    for i in range(max_examples):
        index = round(i * last_index / (max_examples - 1))
        if index in seen:
            continue
        sampled.append(examples[index])
        seen.add(index)
    return sampled


def _discover_existing_skill_names(project_root: Path) -> list[str]:
    specs = SkillLoader(
        project_root=project_root,
        skill_roots=[
            project_root / "skills",
            project_root / "skills" / "new_skills",
        ],
    ).discover()
    return sorted({spec.name for spec in specs})


def _validate_requested_skill_name(skill_name: str) -> str:
    normalized = _sanitize_skill_name(skill_name)
    if normalized != skill_name:
        raise ExampleSkillWriterError(
            f"Invalid skill name '{skill_name}'. Use lowercase letters, digits, and hyphens."
        )
    if not normalized.startswith("rewrite-"):
        raise ExampleSkillWriterError("skill name must start with 'rewrite-'")
    return normalized
