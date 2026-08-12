"""Registry for loaded skill specifications."""

from __future__ import annotations

from collections.abc import Iterable
import re

from core.schemas import SkillSpec


class SkillRegistry:
    """Store and filter skill specs by name, family, category, or stage."""

    def __init__(self, specs: Iterable[SkillSpec] | None = None) -> None:
        self._by_name: dict[str, SkillSpec] = {}
        self._by_family: dict[str, list[SkillSpec]] = {}
        if specs:
            self.register_many(specs)

    def register(self, spec: SkillSpec, *, replace: bool = False) -> None:
        """Register a single skill spec."""
        if spec.name in self._by_name and not replace:
            raise ValueError(f"Duplicate skill name: {spec.name}")
        self._by_name[spec.name] = spec
        family_specs = [
            existing for existing in self._by_family.get(spec.family, []) if existing.name != spec.name
        ]
        family_specs.append(spec)
        self._by_family[spec.family] = sorted(family_specs, key=lambda item: item.name)

    def register_many(self, specs: Iterable[SkillSpec]) -> None:
        """Register many skill specs."""
        for spec in specs:
            self.register(spec)

    def get(self, name: str) -> SkillSpec:
        """Get a skill spec by name."""
        if name not in self._by_name:
            available = ", ".join(self.names())
            raise KeyError(f"Unknown skill '{name}'. Available skills: {available}")
        return self._by_name[name]

    def all(self) -> list[SkillSpec]:
        """Return all registered specs."""
        return list(self._by_name.values())

    def families(self) -> list[str]:
        """Return all registered skill family identifiers."""
        return sorted(self._by_family)

    def get_family(self, family: str) -> list[SkillSpec]:
        """Return all specs that belong to the same family."""
        return list(self._by_family.get(family, []))

    def names(self) -> list[str]:
        """Return all registered skill names."""
        return sorted(self._by_name)

    def filter(
        self,
        *,
        names: list[str] | None = None,
        family: str | None = None,
        category: str | None = None,
        stage: str | None = None,
        status: str | None = None,
    ) -> list[SkillSpec]:
        """Filter skills by family, category, and stage."""
        results = self.all()
        if names is not None:
            allowed = set(names)
            results = [spec for spec in results if spec.name in allowed]
        if family is not None:
            results = [spec for spec in results if spec.family == family]
        if category is not None:
            results = [spec for spec in results if spec.category == category]
        if stage is not None:
            results = [spec for spec in results if stage in spec.stage]
        if status is not None:
            results = [spec for spec in results if spec.status == status]
        return results

    def filter_applicable(
        self,
        *,
        category: str | None = None,
        stage: str | None = None,
        names: list[str] | None = None,
        seed_prompt: str = "",
        risk_types: list[str] | None = None,
    ) -> list[SkillSpec]:
        """Filter active skills and keep narrow external skills inside their domain."""
        results = self.filter(
            names=names,
            category=category,
            stage=stage,
            status="active",
        )
        return [
            spec
            for spec in results
            if self._matches_declared_scope(
                spec,
                seed_prompt=seed_prompt,
                risk_types=risk_types or [],
            )
        ]

    @staticmethod
    def _matches_declared_scope(
        spec: SkillSpec,
        *,
        seed_prompt: str,
        risk_types: list[str],
    ) -> bool:
        if not spec.target_domain:
            return True
        haystack = " ".join([seed_prompt, *risk_types]).casefold()
        profile = spec.evaluation_profile if isinstance(spec.evaluation_profile, dict) else {}
        raw_include = profile.get("include", [])
        raw_exclude = profile.get("exclude", [])
        raw_dataset_labels = profile.get("dataset_risk_labels", [])
        include_labels = [
            str(value).strip()
            for value in (raw_include if isinstance(raw_include, list) else [])
            if str(value).strip()
        ]
        exclude_labels = [
            str(value).strip()
            for value in (raw_exclude if isinstance(raw_exclude, list) else [])
            if str(value).strip()
        ]
        dataset_labels = [
            str(value).strip()
            for value in (
                raw_dataset_labels if isinstance(raw_dataset_labels, list) else []
            )
            if str(value).strip()
        ]
        normalized_haystack = " ".join(haystack.split())
        for label in exclude_labels:
            normalized_label = " ".join(label.casefold().split())
            if not normalized_label or normalized_label.startswith("other "):
                continue
            unicode_specific = any(ord(char) > 127 for char in normalized_label) and len(
                re.sub(r"\s+", "", normalized_label)
            ) >= 4
            if (
                len(normalized_label.split()) > 1 or unicode_specific
            ) and normalized_label in normalized_haystack:
                return False

        normalized_risk_types = {
            " ".join(str(value).casefold().split())
            for value in risk_types
            if str(value).strip()
        }
        structured_labels = {
            " ".join(str(value).casefold().split())
            for value in [
                profile.get("profile_id", ""),
                profile.get("target_domain", ""),
                *include_labels,
                *dataset_labels,
            ]
            if str(value).strip()
        }
        if normalized_risk_types & structured_labels:
            return True
        haystack_tokens = set(re.findall(r"\w+", haystack, flags=re.UNICODE))
        if not haystack_tokens:
            return False
        generic_tokens = {
            "and", "the", "for", "with", "from", "into", "assistance", "instructions",
            "instruction", "generation", "content", "request", "requests", "response",
            "responses", "targeted", "controlled", "specific", "creation", "methods",
        }
        cues = [
            spec.target_domain,
            *include_labels,
            *dataset_labels,
            *spec.applicability_terms,
        ]
        for cue in cues:
            normalized = " ".join(str(cue).casefold().split())
            if not normalized:
                continue
            if normalized in haystack:
                return True
            cue_tokens = {
                token
                for token in re.findall(r"\w+", normalized, flags=re.UNICODE)
                if len(token) >= 4 and token not in generic_tokens
            }
            overlap = cue_tokens & haystack_tokens
            required_overlap = 1 if len(cue_tokens) == 1 else 2
            if len(overlap) >= required_overlap:
                return True
        return False
