"""Workflow loading helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.utils import read_yaml


@dataclass
class Workflow:
    """Minimal workflow config for skill groups and routing policy."""

    name: str
    description: str
    initial_stage: str
    skill_groups: dict[str, list[str]] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "Workflow":
        """Load a workflow config from YAML."""
        raw = read_yaml(path)
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            initial_stage=str(raw["initial_stage"]),
            skill_groups=dict(raw.get("skill_groups", {})),
            policy=dict(raw.get("policy", {})),
        )

    def get_group(self, name: str) -> list[str]:
        """Get a skill group by name."""
        return list(self.skill_groups.get(name, []))

    def get_policy(self, key: str, default: Any = None) -> Any:
        """Get one policy value."""
        return self.policy.get(key, default)
