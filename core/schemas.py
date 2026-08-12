"""Dataclasses and JSON helpers shared by the framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclasses into JSON-serializable structures."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


@dataclass
class SkillSpec:
    """Machine-readable definition for a skill package."""

    name: str
    description: str
    category: str
    stage: list[str]
    entry: str = "scripts/run.py"
    family: str = ""
    status: str = "active"
    root_dir: str = ""
    mode: str = "llm_rewrite"
    target_domain: str = ""
    applicability_terms: list[str] = field(default_factory=list)
    attack_surface: str = ""
    red_team_objective: str = ""
    scope_boundary: str = ""
    evaluation_profile: dict[str, Any] = field(default_factory=dict)
    prior_art_relation: str = ""
    classic_components: list[str] = field(default_factory=list)
    classic_matches: list[str] = field(default_factory=list)
    classic_component_roles: list[str] = field(default_factory=list)
    mechanism_type: str = ""
    novelty_delta: str = ""
    interaction_hypothesis: str = ""
    ablation_plan: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize the minimal runtime/planner fields."""
        if not self.family:
            self.family = self.name
        if not self.status:
            self.status = "active"
        if not self.entry:
            self.entry = "scripts/run.py"
        if self.evaluation_profile:
            if not isinstance(self.evaluation_profile, dict):
                raise ValueError("evaluation_profile must be a mapping")
            for field_name in (
                "include",
                "exclude",
                "dataset_risk_labels",
                "evidence_ids",
            ):
                value = self.evaluation_profile.get(field_name, [])
                if not isinstance(value, list) or not all(
                    isinstance(item, str) for item in value
                ):
                    raise ValueError(
                        f"evaluation_profile.{field_name} must be a string list"
                    )
            if self.target_domain:
                profile_target = " ".join(
                    str(self.evaluation_profile.get("target_domain", ""))
                    .casefold()
                    .split()
                )
                outer_target = " ".join(self.target_domain.casefold().split())
                if profile_target != outer_target:
                    raise ValueError(
                        "evaluation_profile.target_domain must match target_domain"
                    )
                if not str(self.evaluation_profile.get("profile_id", "")).strip():
                    raise ValueError("evaluation_profile.profile_id is required")
                if not self.evaluation_profile.get("include"):
                    raise ValueError("evaluation_profile.include must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert the spec to a plain dictionary."""
        return _jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillSpec":
        """Build a spec from a plain dictionary."""
        raw_profile = data.get("evaluation_profile", {})
        evaluation_profile = dict(raw_profile) if isinstance(raw_profile, dict) else {}
        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            category=str(data["category"]),
            stage=list(data["stage"]),
            entry=str(data.get("entry", "scripts/run.py")),
            family=str(data.get("family", data["name"])),
            status=str(data.get("status", "active")),
            root_dir=str(data.get("root_dir", "")),
            mode=str(data.get("mode", "llm_rewrite")),
            target_domain=str(data.get("target_domain", "")).strip(),
            applicability_terms=[
                str(value).strip()
                for value in data.get("applicability_terms", [])
                if str(value).strip()
            ],
            attack_surface=str(data.get("attack_surface", "")).strip(),
            red_team_objective=str(data.get("red_team_objective", "")).strip(),
            scope_boundary=str(data.get("scope_boundary", "")).strip(),
            evaluation_profile=evaluation_profile,
            prior_art_relation=str(data.get("prior_art_relation", "")).strip(),
            classic_components=[
                str(value).strip()
                for value in data.get("classic_components", [])
                if str(value).strip()
            ],
            classic_matches=[
                str(value).strip()
                for value in data.get("classic_matches", [])
                if str(value).strip()
            ],
            classic_component_roles=[
                str(value).strip()
                for value in data.get("classic_component_roles", [])
                if str(value).strip()
            ],
            mechanism_type=str(data.get("mechanism_type", "")).strip(),
            novelty_delta=str(data.get("novelty_delta", "")).strip(),
            interaction_hypothesis=str(
                data.get("interaction_hypothesis", "")
            ).strip(),
            ablation_plan=[
                str(value).strip()
                for value in data.get("ablation_plan", [])
                if str(value).strip()
            ],
        )


@dataclass
class SkillContext:
    """Context passed into a skill execution."""

    run_id: str
    step_id: int
    seed_prompt: str
    target_profile: dict[str, Any]
    memory_summary: dict[str, Any]
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the context to a plain dictionary."""
        return _jsonable(self)



@dataclass
class SkillExecutionResult:
    """Structured output returned by a skill script."""

    skill_name: str
    candidates: list[dict[str, Any]]
    rationale: str | None
    artifacts: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a plain dictionary."""
        return _jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillExecutionResult":
        """Build a skill result from a plain dictionary."""
        return cls(
            skill_name=str(data["skill_name"]),
            candidates=list(data.get("candidates", [])),
            rationale=data.get("rationale"),
            artifacts=dict(data.get("artifacts", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EvalResult:
    """Toy evaluation result for a candidate batch."""

    success: bool
    notes: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the evaluation to a plain dictionary."""
        return _jsonable(self)



@dataclass
class AgentState:
    """Mutable planner state tracked during the run."""

    run_id: str
    current_step: int  # Budget-consuming steps (invoke_skill, analyze_memory, invoke_meta_skill)
    trace_step_id: int  # All main loop iterations (for compact_trace recording)
    seed_prompt: str
    memory_summary: dict[str, Any]
    last_eval: dict[str, Any]
    active_workflow_stage: str
    available_skills: list[str]
    budget_remaining: dict[str, Any]
    pending_candidates: list[dict[str, Any]] = field(default_factory=list)
    last_responses: list[dict[str, Any]] = field(default_factory=list)
    workflow_name: str = "basic"
    planner_flags: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    current_prompt_bucket: str = "general"
    current_risk_type: str = "unclassified"
    seed_risk_types: list[str] = field(default_factory=list)  # Risk types from seed prompt (set once at start)
    selected_skill_names: list[str] = field(default_factory=list)
    current_run_skill_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    skill_trace_samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    meta_attempts_history: list[dict[str, Any]] = field(default_factory=list)
    warmup_mode: bool = False
    stop_on_success: bool = False  # Stage 1: halt as soon as a skill scores 5
    representative_prompts: list[str] = field(default_factory=list)  # Category-evolve: N representative seeds
    category_evolve_mode: bool = False  # Category-level multi-seed evolution (Stage 2)
    _analyzed_trace_cursor: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the state to a plain dictionary."""
        return _jsonable(self)


@dataclass
class PlanStep:
    """Planner action emitted by the rule-based planner."""

    action_type: str
    target: str | None
    args: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the plan step to a plain dictionary."""
        return _jsonable(self)
