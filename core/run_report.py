"""Helpers for building compact per-run JSON reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CompactRunRecorder:
    """Collect compact per-step trace data in memory and emit one JSON report."""

    def __init__(self, *, run_id: str, workflow: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.workflow = workflow
        self.run_dir = str(run_dir)
        self._steps_by_id: dict[int, dict[str, Any]] = {}
        self._ordered_step_ids: list[int] = []
        self._candidates_by_id: dict[str, dict[str, Any]] = {}
        self._ordered_candidate_ids: list[str] = []

    def record_skill_call(
        self,
        *,
        step_id: int,
        timestamp: str,
        skill_name: str,
        plan_reason: str,
        context_summary: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Record one skill execution under a planner step."""
        step = self._ensure_step(step_id)
        candidate_ids: list[str] = []
        for candidate in list(result.get("candidates", [])):
            candidate_id = self._candidate_id(candidate)
            if not candidate_id:
                continue
            candidate_ids.append(candidate_id)
            self._upsert_candidate(candidate)
        step["skill_calls"].append(
            self._drop_empty(
                {
                    "timestamp": timestamp,
                    "skill_name": skill_name,
                    "plan_reason": plan_reason,
                    "context_summary": self._compact_context_summary(context_summary),
                    "candidate_ids": candidate_ids,
                    "candidate_count": len(candidate_ids),
                    "artifacts": self._compact_skill_artifacts(result.get("artifacts", {})),
                    "metadata": self._compact_skill_metadata(result.get("metadata", {})),
                }
            )
        )

    def record_environment_call(
        self,
        *,
        step_id: int,
        timestamp: str,
        candidate: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Record one environment execution under a planner step."""
        step = self._ensure_step(step_id)
        candidate_id = self._candidate_id(candidate)
        if candidate_id:
            self._upsert_candidate(candidate)
            self._attach_response(candidate_id=candidate_id, result=result)
        step.setdefault("environment_calls", []).append(
            self._drop_empty(
                {
                    "timestamp": timestamp,
                    "candidate_id": candidate_id,
                }
            )
        )

    def record_evaluation(
        self,
        *,
        step_id: int,
        timestamp: str,
        result: dict[str, Any],
        candidates: list[dict[str, Any]],
        responses: list[dict[str, Any]],
    ) -> None:
        """Record one evaluation payload under a planner step."""
        step = self._ensure_step(step_id)
        metadata = dict(result.get("metadata", {}))
        bundles = list(metadata.get("score_bundles", []))

        for candidate in candidates:
            self._upsert_candidate(candidate)

        for candidate, response in zip(candidates, responses):
            candidate_id = self._candidate_id(candidate)
            if not candidate_id:
                continue
            self._attach_response(candidate_id=candidate_id, result=response)

        for bundle in bundles:
            candidate_index = bundle.get("candidate_index")
            if not isinstance(candidate_index, int):
                continue
            if candidate_index < 0 or candidate_index >= len(candidates):
                continue
            candidate_id = self._candidate_id(candidates[candidate_index])
            if not candidate_id:
                continue
            self._attach_evaluation(candidate_id=candidate_id, bundle=bundle)

        step["evaluation"] = self._drop_empty(
            {
                "timestamp": timestamp,
                "success": result.get("success"),
            }
        )

    def record_analysis_context(
        self,
        *,
        step_id: int,
        timestamp: str,
        skill_trace_samples: dict[str, list[dict[str, Any]]],
        meta_attempts_history: list[dict[str, Any]],
        representative_prompts: list[str] | None = None,
    ) -> None:
        """Record the input context fed to failure-analyzer.

        Records the full candidate / response text for each trace sample, so the
        audit trace reflects exactly what the analyzer LLM was given. For
        category-evolve, also records the representative failed prompts the
        analyzer must generalize across.
        """
        step = self._ensure_step(step_id)

        trace_samples: dict[str, list[dict[str, Any]]] = {}
        for skill_name, samples in skill_trace_samples.items():
            trace_samples[skill_name] = [
                self._drop_empty({
                    "judge_score": s.get("judge_score", 1),
                    "success": s.get("success", False),
                    "candidate_text": str(s.get("candidate_text", "")),
                    "response_text": str(s.get("response_text", "")),
                })
                for s in samples
            ]

        history_summary = []
        for attempt in meta_attempts_history:
            entry: dict[str, Any] = {
                "meta_skill": attempt.get("meta_skill"),
                "generated_skill": attempt.get("generated_skill"),
                "success": attempt.get("success"),
            }
            if attempt.get("eval_scores"):
                entry["max_score"] = attempt.get("max_score")
                entry["avg_score"] = attempt.get("avg_score")
            args = attempt.get("args", {})
            if args.get("skill_name"):
                entry["target_skill"] = args["skill_name"]
            if args.get("skill_names"):
                entry["target_skills"] = args["skill_names"]
            if args.get("discover_direction"):
                entry["direction"] = args["discover_direction"]
            history_summary.append(entry)

        step["analysis_input"] = self._drop_empty({
            "timestamp": timestamp,
            "representative_prompts": list(representative_prompts or []),
            "trace_samples": trace_samples,
            "meta_attempts_count": len(meta_attempts_history),
            "meta_attempts_history": history_summary,
        })

    def record_meta_context(
        self,
        *,
        step_id: int,
        timestamp: str,
        meta_skill_name: str,
        action_args: dict[str, Any],
    ) -> None:
        """Record the input context fed to a meta-skill."""
        step = self._ensure_step(step_id)
        step["meta_input"] = self._drop_empty({
            "timestamp": timestamp,
            "meta_skill_name": meta_skill_name,
            "action_args": dict(action_args),
        })

    def record_step_summary(
        self,
        *,
        step_id: int,
        timestamp: str,
        action_type: str,
        target: str | None,
        plan_reason: str,
        planner_args: dict[str, Any],
        stage_before: str,
        stage_after: str,
        selected_skill_names: list[str],
        planner_flags: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Record the planner-facing summary for one step."""
        step = self._ensure_step(step_id)
        step["planner"] = self._drop_empty(
            {
                "timestamp": timestamp,
                "action_type": action_type,
                "target": target,
                "plan_reason": plan_reason,
                "planner_args": self._drop_empty(dict(planner_args)),
                "stage_before": stage_before,
                "stage_after": stage_after,
                "selected_skill_names": list(selected_skill_names),
                "planner_flags": self._drop_empty(dict(planner_flags)),
                "result": self._drop_empty(dict(result)),
            }
        )

    def build_steps_trace(self, *, summary: dict[str, Any]) -> dict[str, Any]:
        """Emit the step-centric compact run trace."""
        compact_steps = [
            self._build_compact_step(step_id) for step_id in sorted(self._ordered_step_ids)
        ]
        return self._drop_empty(
            {
                "run_id": summary.get("run_id", self.run_id),
                "workflow": summary.get("workflow", self.workflow),
                "models": summary.get("models"),
                "final_stage": summary.get("final_stage"),
                "steps_completed": summary.get("steps_completed"),
                "budget": summary.get("budget"),
                "stop_reason": summary.get("stop_reason"),
                "finished_at": summary.get("finished_at"),
                "steps": compact_steps,
            }
        )

    def _build_compact_step(self, step_id: int) -> dict[str, Any]:
        """Return one concise, step-oriented trace entry."""
        step = self._steps_by_id[step_id]
        planner = dict(step.get("planner", {}))
        candidate_ids = self._step_candidate_ids(step)
        executed_skills_raw = [
            str(call.get("skill_name", "")).strip()
            for call in step.get("skill_calls", [])
            if str(call.get("skill_name", "")).strip()
        ]
        if not executed_skills_raw:
            executed_skills_raw = list(planner.get("selected_skill_names", []))
        # Deduplicate while preserving order
        seen: set[str] = set()
        executed_skills = [s for s in executed_skills_raw if not (s in seen or seen.add(s))]

        compact_step: dict[str, Any] = {
            "step_id": step_id,
            "action_type": planner.get("action_type"),
            "executed_skills": executed_skills,
        }

        # Include input context for analysis/meta steps
        step_input = self._build_step_input(step, planner)
        if step_input:
            compact_step["input"] = step_input

        compact_step["output"] = self._build_step_output(step=step, candidate_ids=candidate_ids)
        return self._drop_empty(compact_step)

    def _build_step_output(
        self,
        *,
        step: dict[str, Any],
        candidate_ids: list[str],
    ) -> dict[str, Any]:
        """Build the key outputs for one step."""
        payload: dict[str, Any] = {}
        if step.get("skill_calls"):
            payload["skill_results"] = [
                self._compact_skill_result(call)
                for call in step.get("skill_calls", [])
                if self._compact_skill_result(call)
            ]
        if step.get("environment_calls"):
            payload["responses"] = [
                self._compact_response_brief(str(call.get("candidate_id", "")).strip())
                for call in step.get("environment_calls", [])
                if self._compact_response_brief(str(call.get("candidate_id", "")).strip())
            ]
        if step.get("evaluation"):
            payload["evaluation"] = self._compact_evaluation_summary(dict(step.get("evaluation", {})))
            payload["candidate_results"] = [
                self._compact_candidate_result(candidate_id)
                for candidate_id in candidate_ids
                if self._compact_candidate_result(candidate_id)
            ]
        return self._drop_empty(payload)

    def _build_step_input(self, step: dict[str, Any], planner: dict[str, Any]) -> dict[str, Any]:
        """Build the input context for analysis/meta steps."""
        payload: dict[str, Any] = {}

        # Analysis step: use pre-recorded analysis_input
        if step.get("analysis_input"):
            ai = dict(step["analysis_input"])
            ai.pop("timestamp", None)
            payload["analysis"] = self._drop_empty(ai)

        # Meta step: use pre-recorded meta_input
        if step.get("meta_input"):
            mi = dict(step["meta_input"])
            mi.pop("timestamp", None)
            payload["meta"] = self._drop_empty(mi)

        return self._drop_empty(payload)

    def _step_candidate_ids(self, step: dict[str, Any]) -> list[str]:
        """Return ordered candidate ids touched by one step."""
        ordered: list[str] = []
        seen: set[str] = set()
        for call in step.get("skill_calls", []):
            for candidate_id in call.get("candidate_ids", []):
                candidate_id = str(candidate_id).strip()
                if candidate_id and candidate_id not in seen:
                    ordered.append(candidate_id)
                    seen.add(candidate_id)
        for call in step.get("environment_calls", []):
            candidate_id = str(call.get("candidate_id", "")).strip()
            if candidate_id and candidate_id not in seen:
                ordered.append(candidate_id)
                seen.add(candidate_id)
        for candidate_id in dict(step.get("evaluation", {})).get("candidate_ids", []):
            candidate_id = str(candidate_id).strip()
            if candidate_id and candidate_id not in seen:
                ordered.append(candidate_id)
                seen.add(candidate_id)
        return ordered

    def _compact_candidate_brief(self, candidate_id: str) -> dict[str, Any]:
        """Return a short candidate summary for one id."""
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return {}
        text = str(candidate.get("text", "")).strip()
        return self._drop_empty(
            {
                "source_skill": candidate.get("source_skill"),
                "text_preview": self._text_preview(text),
            }
        )

    def _compact_response_brief(self, candidate_id: str) -> dict[str, Any]:
        """Return a short response summary for one candidate id."""
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return {}
        response = dict(candidate.get("response", {}))
        text = str(response.get("text", "")).strip()
        return self._drop_empty(
            {
                "text_preview": self._text_preview(text) if text else "",
                "style": response.get("style"),
                "blocked_by_guardrail": response.get("blocked_by_guardrail"),
                "target_error": response.get("target_error"),
                "attempts": response.get("attempts"),
                "error_type": response.get("error_type"),
                "error": response.get("error"),
            }
        )

    def _compact_candidate_result(self, candidate_id: str) -> dict[str, Any]:
        """Return one candidate-level evaluation result."""
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return {}
        evaluation = dict(candidate.get("evaluation", {}))
        if not evaluation:
            return {}
        return self._drop_empty(
            {
                "judge_score": evaluation.get("judge_score"),
                "success": evaluation.get("success"),
                "risk_type": evaluation.get("seed_risk_type"),
                "not_scored": evaluation.get("not_scored"),
                "execution_error": evaluation.get("execution_error"),
            }
        )

    def _compact_skill_result(self, skill_call: dict[str, Any]) -> dict[str, Any]:
        """Return the key output of one skill call."""
        payload: dict[str, Any] = {}
        skill_name = str(skill_call.get("skill_name", "")).strip()
        if skill_name:
            payload["skill_name"] = skill_name
        payload["generated_candidates"] = [
            self._compact_candidate_brief(candidate_id)
            for candidate_id in skill_call.get("candidate_ids", [])
            if self._compact_candidate_brief(candidate_id)
        ]
        payload["artifacts"] = self._compact_artifact_summary(skill_call.get("artifacts", {}))
        metadata = skill_call.get("metadata") or {}
        if metadata.get("error"):
            payload["error"] = metadata["error"]
        return self._drop_empty(payload)

    def _compact_evaluation_summary(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        """Return only the step-level evaluation outcome."""
        return self._drop_empty(
            {
                "success": evaluation.get("success"),
            }
        )

    def _compact_artifact_summary(self, artifacts: Any) -> dict[str, Any]:
        """Return only the artifact fields that explain what changed."""
        artifact_dict = dict(artifacts) if isinstance(artifacts, dict) else {}
        if not artifact_dict:
            return {}
        payload: dict[str, Any] = {}

        # Analysis output: extract key decision info from planner_decision
        planner_decision = artifact_dict.get("planner_decision")
        if isinstance(planner_decision, dict):
            meta_dispatch = planner_decision.get("meta_dispatch")
            if meta_dispatch:
                payload["meta_dispatch"] = meta_dispatch
            else:
                payload["meta_dispatch"] = None

        # Generated skill artifacts
        if "generated_skill_name" in artifact_dict:
            payload["generated_skill_name"] = artifact_dict["generated_skill_name"]
        if "generated_skill_dir" in artifact_dict:
            payload["generated_skill_dir"] = artifact_dict["generated_skill_dir"]
        if "source_skill" in artifact_dict:
            payload["source_skill"] = artifact_dict["source_skill"]
        if "source_skills" in artifact_dict:
            payload["source_skills"] = artifact_dict["source_skills"]

        return self._drop_empty(payload)

    def _ensure_step(self, step_id: int) -> dict[str, Any]:
        """Return the mutable record for one step."""
        if step_id not in self._steps_by_id:
            self._steps_by_id[step_id] = {
                "step_id": step_id,
                "planner": {},
                "skill_calls": [],
            }
            self._ordered_step_ids.append(step_id)
        return self._steps_by_id[step_id]

    def _candidate_id(self, candidate: dict[str, Any]) -> str:
        """Return the normalized candidate id."""
        return str(candidate.get("candidate_id", "")).strip()

    def _upsert_candidate(self, candidate: dict[str, Any]) -> None:
        """Store the canonical candidate record once and update it in place."""
        candidate_id = self._candidate_id(candidate)
        if not candidate_id:
            return
        if candidate_id not in self._candidates_by_id:
            self._candidates_by_id[candidate_id] = self._drop_empty(
                {
                    "candidate_id": candidate_id,
                    "text": str(candidate.get("text", "")).strip(),
                    "source_skill": candidate.get("source_skill"),
                    "prompt_bucket": candidate.get("prompt_bucket"),
                    "risk_type": candidate.get("risk_type"),
                    "selection_id": candidate.get("selection_id"),
                    "selection_rank": candidate.get("selection_rank"),
                }
            )
            self._ordered_candidate_ids.append(candidate_id)
            return

        existing = self._candidates_by_id[candidate_id]
        for key, value in self._drop_empty(
            {
                "text": str(candidate.get("text", "")).strip(),
                "source_skill": candidate.get("source_skill"),
                "prompt_bucket": candidate.get("prompt_bucket"),
                "risk_type": candidate.get("risk_type"),
                "selection_id": candidate.get("selection_id"),
                "selection_rank": candidate.get("selection_rank"),
            }
        ).items():
            existing.setdefault(key, value)

    def _attach_response(self, *, candidate_id: str, result: dict[str, Any]) -> None:
        """Attach one environment response to the canonical candidate."""
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return
        candidate["response"] = self._drop_empty(
            {
                "text": str(result.get("response_text", "")).strip(),
                "style": result.get("style"),
                "backend": result.get("backend"),
                "model_name": result.get("model_name"),
                "blocked_by_guardrail": result.get("blocked_by_guardrail"),
                "target_error": result.get("target_error"),
                "attempts": result.get("attempts"),
                "error_type": result.get("error_type"),
                "error": result.get("error"),
            }
        )

    def _attach_evaluation(self, *, candidate_id: str, bundle: dict[str, Any]) -> None:
        """Attach one candidate-level evaluation summary."""
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return
        candidate["evaluation"] = self._drop_empty(
            {
                "success": bundle.get("candidate_success"),
                "judge_score": bundle.get("judge_score"),
                "seed_risk_type": bundle.get("seed_risk_type"),
                "candidate_risk_type": bundle.get("primary_risk_type"),
                "not_scored": bundle.get("not_scored"),
                "execution_error": bundle.get("execution_error"),
            }
        )

    def _compact_context_summary(self, context_summary: dict[str, Any]) -> dict[str, Any]:
        """Keep only context counters that are useful during replay."""
        return self._drop_empty(
            {
                "prior_candidate_count": context_summary.get("prior_candidate_count"),
                "memory_total_entries": context_summary.get("memory_total_entries"),
            }
        )

    def _compact_skill_artifacts(self, artifacts: Any) -> dict[str, Any]:
        """Drop obvious duplicates from skill artifacts."""
        artifact_dict = dict(artifacts) if isinstance(artifacts, dict) else {}
        artifact_dict.pop("original_seed", None)
        artifact_dict.pop("candidate_count", None)
        return self._drop_empty(artifact_dict)

    def _compact_skill_metadata(self, metadata: Any) -> dict[str, Any]:
        """Keep only metadata that helps debug the skill runtime itself."""
        metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
        return self._drop_empty(
            {
                "protocol_version": metadata_dict.get("protocol_version"),
                "entry_path": metadata_dict.get("entry_path"),
                "stderr": metadata_dict.get("stderr"),
                "error": metadata_dict.get("error"),
            }
        )

    def _text_preview(self, text: str) -> str:
        """Return normalized text."""
        return " ".join(text.split())

    def _drop_empty(self, payload: Any) -> Any:
        """Recursively remove null, empty-string, empty-list, and empty-dict values."""
        if isinstance(payload, dict):
            compacted: dict[str, Any] = {}
            for key, value in payload.items():
                cleaned = self._drop_empty(value)
                if cleaned in (None, "", [], {}):
                    continue
                compacted[key] = cleaned
            return compacted
        if isinstance(payload, list):
            return [cleaned for item in payload if (cleaned := self._drop_empty(item)) not in (None, "", [], {})]
        return payload
