"""Load warmup trace data and convert to skill_trace_samples format."""

from __future__ import annotations

from typing import Any


def extract_skill_traces(compact_trace: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert compact_trace.json steps into skill_trace_samples format.

    Output format matches EvaluationHandler._store_trace_samples:
    {
        "rewrite-base64": [{"candidate_text": ..., "response_text": ..., "judge_score": ..., "success": ...}],
        ...
    }
    """
    skill_traces: dict[str, list[dict[str, Any]]] = {}

    for step in compact_trace.get("steps", []):
        executed_skills = step.get("executed_skills", [])
        if not executed_skills:
            continue

        skill_name = executed_skills[0]
        output = step.get("output", {})

        skill_results = output.get("skill_results", [])
        candidates: list[str] = []
        for sr in skill_results:
            for gc in sr.get("generated_candidates", []):
                candidates.append(gc.get("text_preview", ""))

        responses = output.get("responses", [])
        response_texts = [r.get("text_preview", "") for r in responses]

        candidate_results = output.get("candidate_results", [])

        if skill_name not in skill_traces:
            skill_traces[skill_name] = []

        for i, cr in enumerate(candidate_results):
            if cr.get("not_scored") or cr.get("execution_error"):
                continue
            entry = {
                "candidate_text": candidates[i] if i < len(candidates) else "",
                "response_text": response_texts[i] if i < len(response_texts) else "",
                "judge_score": int(cr.get("judge_score", 1)),
                "success": bool(cr.get("success", False)),
            }
            skill_traces[skill_name].append(entry)

    return skill_traces
