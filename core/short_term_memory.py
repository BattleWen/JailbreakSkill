"""Short-term memory management for current run skill performance tracking."""

from __future__ import annotations

from typing import Any

from core.schemas import AgentState


def update_short_term_memory(
    state: AgentState,
    skill_name: str,
    candidates_results: list[dict[str, Any]],
) -> None:
    """Update short-term memory with evaluation results from one skill invocation.

    Args:
        state: Current agent state
        skill_name: Skill name
        candidates_results: List of evaluation results for all candidates.
                           Each result should have:
                           - "judge_score": int (1-5)
                           - "success": bool
    """
    if not candidates_results:
        return

    scores = [
        int(r.get("judge_score", 1))
        for r in candidates_results
    ]
    successes = sum(
        1 for r in candidates_results
        if r.get("success", False)
    )
    batch_attempts = len(candidates_results)

    if skill_name not in state.current_run_skill_stats:
        state.current_run_skill_stats[skill_name] = {
            "attempts": 0,
            "successes": 0,
            "asr": 0.0,
            "avg_judge_score": 0.0,
            "best_judge_score": 1,
            "worst_judge_score": 5,
            "_total_judge_score": 0,
        }

    stats = state.current_run_skill_stats[skill_name]

    stats["attempts"] += batch_attempts
    stats["successes"] += successes
    stats["_total_judge_score"] += sum(scores)

    stats["avg_judge_score"] = stats["_total_judge_score"] / stats["attempts"]
    stats["asr"] = stats["successes"] / stats["attempts"]

    if scores:
        stats["best_judge_score"] = max(stats["best_judge_score"], max(scores))
        stats["worst_judge_score"] = min(stats["worst_judge_score"], min(scores))


def get_top_skills_in_current_run(
    state: AgentState,
    top_n: int = 5,
    min_attempts: int = 1,
    sort_by: str = "avg_judge_score",
) -> list[str]:
    """Get top N skills by specified metric in current run.

    Args:
        state: Current agent state
        top_n: Number of skills to return
        min_attempts: Minimum attempts required to be considered
        sort_by: Metric to sort by ("avg_judge_score" or "asr")

    Returns:
        List of skill names sorted by metric (descending)
    """
    candidates = [
        (skill_name, stats[sort_by])
        for skill_name, stats in state.current_run_skill_stats.items()
        if stats["attempts"] >= min_attempts
    ]

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [skill_name for skill_name, _ in candidates[:top_n]]


def get_skill_stats_in_current_run(
    state: AgentState,
    skill_name: str,
) -> dict[str, Any] | None:
    """Get statistics for a specific skill in current run."""
    if skill_name not in state.current_run_skill_stats:
        return None

    stats = state.current_run_skill_stats[skill_name]
    return {
        "attempts": stats["attempts"],
        "successes": stats["successes"],
        "asr": stats["asr"],
        "avg_judge_score": stats["avg_judge_score"],
        "best_judge_score": stats["best_judge_score"],
        "worst_judge_score": stats["worst_judge_score"],
    }


def get_current_run_summary(state: AgentState) -> dict[str, Any]:
    """Get summary statistics for current run."""
    total_skills_tried = len(state.current_run_skill_stats)
    total_attempts = sum(
        stats["attempts"]
        for stats in state.current_run_skill_stats.values()
    )
    total_successes = sum(
        stats["successes"]
        for stats in state.current_run_skill_stats.values()
    )
    overall_asr = total_successes / total_attempts if total_attempts > 0 else 0.0

    top_skills = get_top_skills_in_current_run(state, top_n=5, min_attempts=1)

    return {
        "total_skills_tried": total_skills_tried,
        "total_attempts": total_attempts,
        "total_successes": total_successes,
        "overall_asr": overall_asr,
        "top_skills": top_skills,
    }
