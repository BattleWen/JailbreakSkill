"""Deterministic planner for red-team action selection."""

from __future__ import annotations

from typing import Any

from core.registry import SkillRegistry
from core.schemas import AgentState, PlanStep
from core.workflow import Workflow


SEARCH_STAGE = "search"
ANALYSIS_STAGE = "analyze"
META_STAGE = "evolve"
STOP_STAGE = "stop"

DEEP_EXPLOIT_CANDIDATE_COUNT = 3
NEAR_MISS_THRESHOLD = 3


class DeterministicPlanner:
    """Deterministic planner that uses UCB for search and rule-based routing for other stages."""

    def __init__(self, config: dict[str, Any] | None = None, defaults: dict[str, Any] | None = None) -> None:
        self.defaults = dict(defaults or {})
        self.default_candidate_count = int(self.defaults.get("candidate_count", 1))
        self.enable_deep_exploit: bool = False

    # ==================== Helper Methods (State & Workflow) ====================

    def _deterministic_plan(self, state: AgentState) -> list[PlanStep]:
        """Keep queued execution and evaluation transitions local and deterministic."""
        if state.active_workflow_stage == STOP_STAGE:
            return [PlanStep(STOP_STAGE, None, {}, "")]
        if state.pending_candidates and not state.last_responses:
            return [
                PlanStep(
                    action_type="execute_candidates",
                    target=None,
                    args={"count": len(state.pending_candidates)},
                    reason="",
                )
            ]
        if state.pending_candidates and state.last_responses:
            return [
                PlanStep(
                    action_type="evaluate_candidates",
                    target=None,
                    args={"count": len(state.pending_candidates)},
                    reason="",
                )
            ]
        if (
            state.active_workflow_stage == SEARCH_STAGE
            and "environment_calls" in state.budget_remaining
            and state.budget_remaining["environment_calls"] <= 0
        ):
            state.planner_flags["stage1_stop_reason"] = "target_query_budget_exhausted"
            state.active_workflow_stage = STOP_STAGE
            return [PlanStep(STOP_STAGE, None, {}, "Target-query budget exhausted.")]
        if state.budget_remaining.get("steps", 0) <= 0:
            return [PlanStep(STOP_STAGE, None, {}, "Budget exhausted.")]
        return []

    def _workflow_for_state(self, state: AgentState, workflows: dict[str, Workflow]) -> Workflow:
        """Return the requested workflow, falling back to the configured default if needed."""
        if state.workflow_name in workflows:
            return workflows[state.workflow_name]
        if "basic" in workflows:
            return workflows["basic"]
        return next(iter(workflows.values()))

    def _search_pool(
        self,
        workflow: Workflow,
        registry: SkillRegistry,
        *,
        seed_prompt: str = "",
        risk_types: list[str] | None = None,
    ) -> list[str]:
        """Return active search-stage attack skills for the current workflow."""
        declared = workflow.get_group("search")
        if declared:
            allowed_specs = {
                spec.name: spec
                for spec in registry.filter_applicable(
                    names=declared,
                    category="attack",
                    stage=SEARCH_STAGE,
                    seed_prompt=seed_prompt,
                    risk_types=risk_types,
                )
            }
            return [
                skill_name
                for skill_name in declared
                if skill_name in allowed_specs
            ]
        return [
            spec.name
            for spec in registry.filter_applicable(
                category="attack",
                stage=SEARCH_STAGE,
                seed_prompt=seed_prompt,
                risk_types=risk_types,
            )
        ]

    def _sort_search_pool_by_memory(
        self,
        search_pool: list[str],
        memory,
        current_risk_types: list[str] | str,
    ) -> list[str]:
        """Sort search pool by average UCB score across relevant risk types."""
        if isinstance(current_risk_types, str):
            current_risk_types = [current_risk_types]

        if not current_risk_types:
            all_risk_types = memory.get_all_risk_types() if hasattr(memory, 'get_all_risk_types') else []
            current_risk_types = all_risk_types if all_risk_types else ["unclassified"]

        untried_skills: list[str] = []
        tried_skills: list[tuple[str, float]] = []

        for skill_name in search_pool:
            ucb_scores = []

            for risk_type in current_risk_types:
                stats = memory.get_skill_stats(skill_name, risk_type=risk_type)
                if stats and stats.get("attempts", 0) > 0:
                    ucb_scores.append(stats.get("ucb_score", 0.0))

            if ucb_scores:
                avg_ucb = sum(ucb_scores) / len(ucb_scores)
                tried_skills.append((skill_name, avg_ucb))
            else:
                untried_skills.append(skill_name)

        tried_skills.sort(key=lambda x: x[1], reverse=True)
        tried_skill_names = [name for name, _ in tried_skills]

        return untried_skills + tried_skill_names

    def _analysis_targets(self, workflow: Workflow, registry: SkillRegistry) -> list[str]:
        """Return active analysis skills."""
        declared = workflow.get_group("analyze")
        if declared:
            return [
                spec.name
                for spec in registry.filter(names=declared, category="analyze", status="active")
            ]
        return [spec.name for spec in registry.filter(category="analyze", status="active")]

    def _best_score_from_traces(self, state: AgentState) -> tuple[int, str, str]:
        """Find the best judge_score from all skill trace samples.

        Returns (best_score, best_skill_name, best_candidate_text).
        """
        best_score = 0
        best_skill = ""
        best_text = ""
        for skill_name, traces in state.skill_trace_samples.items():
            for trace in traces:
                score = trace.get("judge_score", 1)
                if score > best_score:
                    best_score = score
                    best_skill = skill_name
                    best_text = trace.get("candidate_text", "")
        return best_score, best_skill, best_text

    # ==================== Main Planning Interface ====================

    def plan(
        self,
        state: AgentState,
        workflows: dict[str, Workflow],
        registry: SkillRegistry,
        memory=None,
    ) -> list[PlanStep]:
        """Deterministic planning for search, analysis, and meta stages."""
        deterministic_plan = self._deterministic_plan(state)
        if deterministic_plan:
            state.planner_flags["planner_backend"] = "deterministic"
            state.planner_flags["planner_mode"] = "deterministic_transition"
            return deterministic_plan

        # Search stage: sequential UCB-ordered scan (with optional deep exploit)
        if state.active_workflow_stage == SEARCH_STAGE:
            workflow = self._workflow_for_state(state, workflows)
            search_pool = self._search_pool(
                workflow,
                registry,
                seed_prompt=state.seed_prompt,
                risk_types=state.seed_risk_types,
            )

            if memory is not None:
                current_risk_types = state.seed_risk_types
                if not current_risk_types:
                    current_risk_types = state.last_eval.get("risk_types", [])
                if not current_risk_types and hasattr(memory, 'get_all_risk_types'):
                    current_risk_types = memory.get_all_risk_types()
                search_pool = self._sort_search_pool_by_memory(
                    search_pool, memory, current_risk_types,
                )
                state.planner_flags["search_pool_sorted_by_memory"] = True
                state.planner_flags["search_pool_risk_types"] = current_risk_types

            # Deep exploit: retry a near-miss skill if flag is set
            if self.enable_deep_exploit:
                deep_target = state.planner_flags.get("deep_exploit_target")
                retries_remaining = state.planner_flags.get("deep_exploit_retries", 0)
                if deep_target and retries_remaining > 0:
                    state.planner_flags["deep_exploit_retries"] = retries_remaining - 1
                    state.planner_flags["planner_backend"] = "deterministic"
                    state.planner_flags["planner_mode"] = "deep_exploit"
                    return [
                        PlanStep(
                            action_type="invoke_skill",
                            target=deep_target,
                            args={
                                "mode": SEARCH_STAGE,
                                "candidate_count": DEEP_EXPLOIT_CANDIDATE_COUNT,
                            },
                            reason=f"Deep exploit: retrying {deep_target} (retries left: {retries_remaining - 1})",
                        )
                    ]
                # Clear exhausted deep exploit state
                if deep_target and retries_remaining <= 0:
                    state.planner_flags.pop("deep_exploit_target", None)
                    state.planner_flags.pop("deep_exploit_retries", None)

            # Default: sequential UCB scan
            untried = [s for s in search_pool if s not in state.selected_skill_names]
            if untried:
                next_skill = untried[0]
                state.planner_flags["planner_backend"] = "deterministic"
                state.planner_flags["planner_mode"] = "sequential_ucb_scan"
                return [
                    PlanStep(
                        action_type="invoke_skill",
                        target=next_skill,
                        args={
                            "mode": SEARCH_STAGE,
                            "candidate_count": self.default_candidate_count,
                        },
                        reason=f"Sequential UCB scan: {next_skill}",
                    )
                ]

            # All skills exhausted
            if state.warmup_mode:
                state.active_workflow_stage = STOP_STAGE
                state.planner_flags["warmup_all_skills_tried"] = True
                return [PlanStep(STOP_STAGE, None, {}, "")]
            state.active_workflow_stage = ANALYSIS_STAGE
            return self.plan(state, workflows, registry, memory)

        # Analysis stage: deterministic — always invoke failure-analyzer
        if state.active_workflow_stage == ANALYSIS_STAGE:
            workflow = self._workflow_for_state(state, workflows)
            analysis_targets = self._analysis_targets(workflow, registry)
            target = analysis_targets[0] if analysis_targets else "failure-analyzer"
            state.planner_flags["planner_backend"] = "deterministic"
            state.planner_flags["planner_mode"] = "deterministic_analysis"
            return [
                PlanStep(
                    action_type="analyze_memory",
                    target=target,
                    args={"mode": ANALYSIS_STAGE},
                    reason="",
                )
            ]

        # Meta stage: deterministic — read meta_dispatch from failure-analyzer
        if state.active_workflow_stage == META_STAGE:
            meta_dispatch = self._read_meta_dispatch(state)
            if not meta_dispatch:
                # Fallback: stop immediately — LLM likely refused due to sensitive content
                return [PlanStep(STOP_STAGE, None, {}, "No meta dispatch from failure-analyzer.")]
            state.planner_flags.pop("meta_no_dispatch_count", None)
            meta_skill_name = str(meta_dispatch.get("meta_skill_name", ""))
            meta_args = dict(meta_dispatch.get("args", {}))
            if meta_dispatch.get("reason"):
                meta_args["reason"] = str(meta_dispatch["reason"])

            meta_args["mode"] = META_STAGE
            state.planner_flags["planner_backend"] = "deterministic"
            state.planner_flags["planner_mode"] = "deterministic_meta"
            state.planner_flags["meta_dispatch"] = meta_dispatch
            return [
                PlanStep(
                    action_type="invoke_meta_skill",
                    target=meta_skill_name,
                    args=meta_args,
                    reason=f"Dispatched by failure-analyzer: {meta_dispatch.get('meta_skill_name', '')}",
                )
            ]

        # Fallback: stop
        return [PlanStep(STOP_STAGE, None, {}, "Unknown stage; stopping.")]

    def _read_meta_dispatch(self, state: AgentState) -> dict[str, Any] | None:
        """Read the meta_dispatch decision from failure-analyzer artifacts."""
        fa_artifacts = state.artifacts.get("failure-analyzer", {})
        if not isinstance(fa_artifacts, dict):
            return None
        decision = fa_artifacts.get("planner_decision", {})
        if isinstance(decision, dict) and decision.get("meta_dispatch"):
            return dict(decision["meta_dispatch"])
        return None

    def route_after_evaluation(
        self,
        state: AgentState,
        workflows: dict[str, Workflow],
        registry: SkillRegistry | None = None,
    ) -> None:
        """Route the workflow stage after evaluation."""
        if (
            state.last_eval.get("success", False)
            and (not state.warmup_mode or state.stop_on_success)
            and not state.category_evolve_mode
        ):
            state.planner_flags["early_stop_reason"] = f"{state.active_workflow_stage}_success"
            state.active_workflow_stage = STOP_STAGE
            state.planner_flags["successful_skill"] = state.last_eval.get("best_skill")
            return

        if state.active_workflow_stage == SEARCH_STAGE:
            workflow = self._workflow_for_state(state, workflows)

            # Deep exploit trigger: if enabled and last skill scored >= 3, retry it
            if self.enable_deep_exploit and not state.planner_flags.get("deep_exploit_target"):
                best_score, best_skill, _ = self._best_score_from_traces(state)
                if best_score >= NEAR_MISS_THRESHOLD and best_skill:
                    state.planner_flags["deep_exploit_target"] = best_skill
                    state.planner_flags["deep_exploit_retries"] = DEEP_EXPLOIT_CANDIDATE_COUNT
                    return

            all_search_skills = set(
                self._search_pool(
                    workflow,
                    registry or SkillRegistry([]),
                    seed_prompt=state.seed_prompt,
                    risk_types=state.seed_risk_types,
                )
            )
            tried_skills = set(state.selected_skill_names)
            exhausted = all_search_skills and all_search_skills.issubset(tried_skills)

            # Also check if deep exploit is still pending
            if not exhausted and self.enable_deep_exploit:
                deep_target = state.planner_flags.get("deep_exploit_target")
                retries = state.planner_flags.get("deep_exploit_retries", 0)
                if deep_target and retries > 0:
                    return

            if exhausted:
                if state.warmup_mode:
                    state.active_workflow_stage = STOP_STAGE
                    state.planner_flags["warmup_all_skills_tried"] = True
                    return
                state.active_workflow_stage = ANALYSIS_STAGE
                return
            return

        if state.active_workflow_stage == META_STAGE:
            state.active_workflow_stage = ANALYSIS_STAGE
            state.planner_flags["stage_router_backend"] = "deterministic"
            state.planner_flags["stage_router_mode"] = "meta_fail_to_analysis"
            state.planner_flags["stage_router_reason"] = "Meta evaluation failed, returning to analysis."
            return

    def advance_after_action(
        self,
        state: AgentState,
        plan_step: PlanStep,
        workflows: dict[str, Workflow],
    ) -> None:
        """Route the workflow stage after one action completes."""
        if state.active_workflow_stage == SEARCH_STAGE:
            return

        if state.active_workflow_stage == ANALYSIS_STAGE:
            state.active_workflow_stage = META_STAGE
            state.planner_flags["stage_router_backend"] = "deterministic"
            state.planner_flags["stage_router_mode"] = "analysis_to_meta"
            state.planner_flags["stage_router_next_stage"] = META_STAGE
            state.planner_flags["stage_router_reason"] = "Analysis always advances to meta."
            return

        if state.active_workflow_stage == META_STAGE:
            return


LLMPlanner = DeterministicPlanner
