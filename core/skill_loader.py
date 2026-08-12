"""Load skill specifications from machine-readable SKILL.md frontmatter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.schemas import SkillSpec
from core.utils import read_markdown_frontmatter


class SkillLoader:
    """Discover and validate skills under configured roots."""

    FRONTMATTER_REQUIRED_FIELDS = {"name", "description"}
    METADATA_REQUIRED_FIELDS = {"category", "stage"}
    DEFAULT_ENTRY = "scripts/run.py"

    def __init__(self, project_root: Path, skill_roots: list[Path] | None = None) -> None:
        self.project_root = project_root
        self.skill_roots = list(
            dict.fromkeys(
                skill_roots
                or [
                    project_root / "skills",
                ]
            )
        )

    def discover(self) -> list[SkillSpec]:
        """Scan all roots and return validated specs."""
        specs: list[SkillSpec] = []
        for root in self.skill_roots:
            if not root.exists():
                continue
            for skill_doc in sorted(root.glob("*/SKILL.md")):
                spec = self._load_one(skill_doc)
                if spec is not None:
                    specs.append(spec)
        return specs

    def _load_one(self, skill_doc: Path) -> SkillSpec | None:
        """Load and validate one indexed skill spec from frontmatter."""
        frontmatter = read_markdown_frontmatter(skill_doc)
        if not frontmatter:
            return None

        raw = self._spec_from_frontmatter(frontmatter)
        self._validate_frontmatter(skill_doc, raw)

        spec = SkillSpec.from_dict(raw)
        spec.root_dir = str(skill_doc.parent.resolve())
        entry_path = skill_doc.parent / spec.entry
        if not entry_path.exists():
            raise ValueError(f"Missing entry script for {spec.name}: {entry_path}")

        return spec

    def _spec_from_frontmatter(self, frontmatter: dict[str, Any]) -> dict[str, Any]:
        """Materialize a minimal SkillSpec payload from frontmatter metadata."""
        metadata = frontmatter.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        name = str(frontmatter.get("name", "")).strip()
        raw_stage = metadata.get("stage", frontmatter.get("stage", []))
        category = str(metadata.get("category", frontmatter.get("category", "")))
        raw_applicability_terms = metadata.get("applicability_terms", [])
        if isinstance(raw_applicability_terms, str):
            raw_applicability_terms = [raw_applicability_terms]
        elif not isinstance(raw_applicability_terms, list):
            raw_applicability_terms = []
        raw_evaluation_profile = metadata.get("evaluation_profile", {})
        if not isinstance(raw_evaluation_profile, dict):
            raw_evaluation_profile = {}
        raw_classic_components = metadata.get("classic_components", [])
        if isinstance(raw_classic_components, str):
            raw_classic_components = [raw_classic_components]
        elif not isinstance(raw_classic_components, list):
            raw_classic_components = []
        structured_list_fields: dict[str, list[object]] = {}
        for field_name in (
            "classic_matches",
            "classic_component_roles",
            "ablation_plan",
        ):
            raw_values = metadata.get(field_name, [])
            if isinstance(raw_values, str):
                raw_values = [raw_values]
            elif not isinstance(raw_values, list):
                raw_values = []
            structured_list_fields[field_name] = raw_values
        raw = {
            "name": name,
            "description": str(frontmatter.get("description", "")).strip(),
            "category": category,
            "stage": raw_stage,
            "entry": str(metadata.get("entry", frontmatter.get("entry", self.DEFAULT_ENTRY))),
            "family": str(metadata.get("family", frontmatter.get("family", name))).strip(),
            "status": str(metadata.get("status", frontmatter.get("status", "active"))).strip() or "active",
            "mode": str(metadata.get("mode", frontmatter.get("mode", "llm_rewrite"))).strip(),
            "target_domain": str(metadata.get("target_domain", "")).strip(),
            "applicability_terms": list(raw_applicability_terms),
            "attack_surface": str(metadata.get("attack_surface", "")).strip(),
            "red_team_objective": str(
                metadata.get("red_team_objective", "")
            ).strip(),
            "scope_boundary": str(metadata.get("scope_boundary", "")).strip(),
            "evaluation_profile": dict(raw_evaluation_profile),
            "prior_art_relation": str(
                metadata.get("prior_art_relation", "")
            ).strip(),
            "classic_components": list(raw_classic_components),
            "classic_matches": list(structured_list_fields["classic_matches"]),
            "classic_component_roles": list(
                structured_list_fields["classic_component_roles"]
            ),
            "mechanism_type": str(metadata.get("mechanism_type", "")).strip(),
            "novelty_delta": str(metadata.get("novelty_delta", "")).strip(),
            "interaction_hypothesis": str(
                metadata.get("interaction_hypothesis", "")
            ).strip(),
            "ablation_plan": list(structured_list_fields["ablation_plan"]),
        }
        return raw

    def _validate_frontmatter(self, skill_doc: Path, raw_spec: dict[str, object]) -> None:
        """Validate that SKILL.md frontmatter contains a full machine spec."""
        frontmatter = read_markdown_frontmatter(skill_doc)
        if not frontmatter:
            raise ValueError(f"Missing YAML frontmatter in {skill_doc}")

        missing = {
            field
            for field in self.FRONTMATTER_REQUIRED_FIELDS
            if raw_spec.get(field) is None or raw_spec.get(field) == "" or raw_spec.get(field) == []
        }
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing frontmatter fields in {skill_doc}: {missing_text}")

        if str(raw_spec["name"]) != skill_doc.parent.name:
            raise ValueError(f"Frontmatter name must match directory name in {skill_doc}")

        metadata = frontmatter.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Frontmatter metadata must be a mapping in {skill_doc}")

        missing_metadata = {
            field
            for field in self.METADATA_REQUIRED_FIELDS
            if raw_spec.get(field) is None or raw_spec.get(field) == "" or raw_spec.get(field) == []
        }
        if missing_metadata:
            missing_text = ", ".join(sorted(missing_metadata))
            raise ValueError(f"Missing metadata fields in {skill_doc}: {missing_text}")

        profile = metadata.get("evaluation_profile", {})
        if profile:
            if not isinstance(profile, dict):
                raise ValueError(
                    f"evaluation_profile must be a mapping in {skill_doc}"
                )
            for field_name in (
                "profile_id",
                "target_domain",
                "taxonomy",
                "origin",
                "definition",
                "probe_set_id",
            ):
                if field_name in profile and not isinstance(profile[field_name], str):
                    raise ValueError(
                        f"evaluation_profile.{field_name} must be a string in {skill_doc}"
                    )
            for field_name in (
                "include",
                "exclude",
                "dataset_risk_labels",
                "evidence_ids",
            ):
                if field_name in profile and (
                    not isinstance(profile[field_name], list)
                    or not all(isinstance(value, str) for value in profile[field_name])
                ):
                    raise ValueError(
                        f"evaluation_profile.{field_name} must be a string list in {skill_doc}"
                    )
            profile_target = str(profile.get("target_domain", "")).strip()
            spec_target = str(raw_spec.get("target_domain", "")).strip()
            if spec_target and profile_target != spec_target:
                raise ValueError(
                    f"evaluation_profile.target_domain must match target_domain in {skill_doc}"
                )
            if spec_target and not str(profile.get("profile_id", "")).strip():
                raise ValueError(
                    f"evaluation_profile.profile_id is required in {skill_doc}"
                )
            if spec_target and not profile.get("include"):
                raise ValueError(
                    f"evaluation_profile.include must be non-empty in {skill_doc}"
                )

        stage = raw_spec.get("stage")
        if not isinstance(stage, list) or not stage:
            raise ValueError(f"Frontmatter stage must be a non-empty list in {skill_doc}")
