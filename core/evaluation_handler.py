"""Evaluation handler for the planner loop.

Evaluates candidate responses against the judge, updates persistent and short-term
memory, and records trace samples for meta-skill feedback.
"""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import Any

from core.evaluator import MockEvaluator
from core.persistent_memory import PersistentMemory
from core.run_report import CompactRunRecorder
from core.schemas import AgentState
from core.short_term_memory import update_short_term_memory
from core.utils import utc_now_iso

logger = logging.getLogger(__name__)


class EvaluationHandler:
    """Handles candidate evaluation and memory updates."""

    def __init__(self, evaluator: MockEvaluator, exp_name: str = "default") -> None:
        self._evaluator = evaluator
        self._exp_name = exp_name

    def evaluate(
        self,
        state: AgentState,
        memory: PersistentMemory,
        recorder: CompactRunRecorder,
    ) -> dict[str, Any]:
        """Evaluate latest responses and persist them into memory."""
        if len(state.pending_candidates) != len(state.last_responses):
            raise RuntimeError(
                f"Candidate/response count mismatch before evaluation: "
                f"{len(state.pending_candidates)} candidates vs "
                f"{len(state.last_responses)} responses. "
                "This indicates a bug in _execute_candidates."
            )

        eval_result = self._evaluator.evaluate(
            state.pending_candidates,
            state.last_responses,
            seed_prompt=state.seed_prompt,
            seed_risk_types=state.seed_risk_types,
        )
        eval_payload = eval_result.to_dict()

        best_index = eval_result.metadata.get("best_candidate_index")
        skill_names = [candidate.get("source_skill") for candidate in state.pending_candidates]
        best_skill = None
        if isinstance(best_index, int) and 0 <= best_index < len(skill_names):
            best_skill = skill_names[best_index]

        eval_payload.update({
            "step_id": state.current_step,
            "skill_names": skill_names,
            "best_skill": best_skill,
            "seed_risk_type": str(
                eval_payload.get("metadata", {}).get("seed_risk_type", "unclassified")
            ),
            "seed_risk_types": list(
                eval_payload.get("metadata", {}).get("seed_risk_types", [])
            ),
            "risk_types": list(eval_payload.get("metadata", {}).get("risk_types", [])),
            "primary_risk_type": str(
                eval_payload.get("metadata", {}).get("primary_risk_type", "unclassified")
            ),
        })
        state.last_eval = eval_payload
        state.current_risk_type = str(eval_payload.get("primary_risk_type", state.current_risk_type))

        score_bundles = {
            int(bundle["candidate_index"]): bundle
            for bundle in eval_payload.get("metadata", {}).get("score_bundles", [])
        }

        should_update_long_term = self._should_update_long_term(state, eval_payload)

        if should_update_long_term:
            for candidate, response in zip(state.pending_candidates, state.last_responses):
                bundle = score_bundles.get(int(response.get("candidate_index", -1)), {})
                if bundle.get("not_scored") or bundle.get("execution_error"):
                    continue
                candidate_success = bool(bundle.get("candidate_success", eval_payload["success"]))
                judge_score = int(bundle.get("judge_score", 1))
                risk_types = list(bundle.get("risk_types", []))
                if not risk_types:
                    risk_types = ["unclassified"]

                skill_name = str(candidate.get("source_skill", "unknown"))
                memory.update(
                    skill_name=skill_name,
                    risk_types=risk_types,
                    judge_score=judge_score,
                    success=candidate_success,
                )
            memory.save()

        self._update_short_term_memory(state, eval_payload, score_bundles)
        self._store_trace_samples(state, score_bundles)
        self._update_registry_scores(score_bundles, state, exp_name=self._exp_name)

        recorder.record_evaluation(
            step_id=state.current_step,
            timestamp=utc_now_iso(),
            result=eval_payload,
            candidates=list(state.pending_candidates),
            responses=list(state.last_responses),
        )

        state.pending_candidates = []
        state.last_responses = []
        if state.planner_flags.pop("stop_after_evaluate", False):
            state.active_workflow_stage = "stop"
        return eval_payload

    @staticmethod
    def _should_update_long_term(state: AgentState, eval_payload: dict[str, Any]) -> bool:
        """Determine whether to update long-term memory based on stage."""
        if state.active_workflow_stage == "search":
            return True
        elif state.active_workflow_stage == "evolve":
            return eval_payload.get("success", False)
        return True

    @staticmethod
    def _update_short_term_memory(
        state: AgentState,
        eval_payload: dict[str, Any],
        score_bundles: dict[int, dict[str, Any]],
    ) -> None:
        """Update short-term memory with results grouped by skill."""
        skill_to_results: dict[str, list[dict[str, Any]]] = {}
        for candidate, response in zip(state.pending_candidates, state.last_responses):
            bundle = score_bundles.get(int(response.get("candidate_index", -1)), {})
            if bundle.get("not_scored") or bundle.get("execution_error"):
                continue
            skill_name = str(candidate.get("source_skill", "unknown"))

            if skill_name not in skill_to_results:
                skill_to_results[skill_name] = []
            skill_to_results[skill_name].append({
                "judge_score": int(bundle.get("judge_score", 1)),
                "success": bool(bundle.get("candidate_success", eval_payload["success"])),
            })

        for skill_name, results in skill_to_results.items():
            update_short_term_memory(state, skill_name, results)

    @staticmethod
    def _store_trace_samples(
        state: AgentState,
        score_bundles: dict[int, dict[str, Any]],
        max_samples_per_skill: int = 5,
    ) -> None:
        """Store candidate trace samples for meta-skill feedback."""
        for candidate, response in zip(state.pending_candidates, state.last_responses):
            bundle = score_bundles.get(int(response.get("candidate_index", -1)), {})
            if bundle.get("not_scored") or bundle.get("execution_error"):
                continue
            skill_name = str(candidate.get("source_skill", "unknown"))
            if skill_name not in state.skill_trace_samples:
                state.skill_trace_samples[skill_name] = []
            samples = state.skill_trace_samples[skill_name]
            entry = {
                "candidate_text": str(candidate.get("text", "")),
                "response_text": str(response.get("response_text", "")),
                "judge_score": int(bundle.get("judge_score", 1)),
                "success": bool(bundle.get("candidate_success", False)),
            }
            samples.append(entry)
            if len(samples) > max_samples_per_skill:
                samples.sort(key=lambda x: x["judge_score"])
                del samples[0]

    @staticmethod
    def _update_registry_scores(
        score_bundles: dict[int, dict[str, Any]],
        state: "AgentState",
        exp_name: str = "default",
    ) -> None:
        """Update registry.json with max_score and success for evolved skills (best-effort)."""
        try:
            registry_path = Path(__file__).resolve().parents[1] / "skills" / f"registry_{exp_name}.json"
            if not registry_path.exists():
                return

            updated_skills: dict[str, int] = {}
            for candidate, response in zip(state.pending_candidates, state.last_responses):
                bundle = score_bundles.get(int(response.get("candidate_index", -1)), {})
                if bundle.get("not_scored") or bundle.get("execution_error"):
                    continue
                skill_name = str(candidate.get("source_skill", ""))
                score = int(bundle.get("judge_score", 1))
                if skill_name:
                    updated_skills[skill_name] = max(updated_skills.get(skill_name, 0), score)

            if not updated_skills:
                return

            data = _json.loads(registry_path.read_text(encoding="utf-8"))
            changed = False
            for entry in data.get("evolved_skills", []):
                name = entry.get("name", "")
                if name in updated_skills:
                    new_score = updated_skills[name]
                    if new_score > entry.get("max_score", 0):
                        entry["max_score"] = new_score
                        if new_score >= 5:
                            entry["success"] = True
                        changed = True

            if changed:
                registry_path.write_text(
                    _json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
        except Exception:
            pass
