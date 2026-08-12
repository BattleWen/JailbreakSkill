"""Materialize an exact Stage-1 checkpoint from single-skill result matrices.

The ``*_all_skills`` experiments evaluate every search skill independently on
every dataset row.  This module replays the production UCB policy over those
frozen outcomes and writes the files consumed by category-level Stage 2.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from core.persistent_memory import PersistentMemory
from core.planner import DeterministicPlanner
from core.utils import utc_now_iso, write_json_atomic
from core.workflow import Workflow


STAGE1_PER_SEED = "stage1_per_seed.jsonl"
STAGE1_PROGRESS = "stage1_progress.jsonl"
STAGE1_MEMORY = "stage1_shared.json"
PROMPT_MEMORY = "prompt_memory.json"
REPLAY_MANIFEST = "offline_replay_manifest.json"


def _load_latest_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    """Return the last JSONL row for each integer dataset index."""
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                index = int(row["index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid indexed JSONL row at {path}:{line_number}") from exc
            rows[index] = row
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL through a sibling temporary file and atomically replace."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _target_was_attempted(
    *,
    source_dir: Path,
    source_runs_root: Path,
    skill_name: str,
    index: int,
    progress: dict[int, dict[str, Any]],
) -> bool:
    """Resolve whether an unscored single-skill row consumed a target call."""
    progress_row = progress.get(index)
    if progress_row is None:
        raise ValueError(
            f"Missing progress row for unscored outcome: {source_dir.name}/"
            f"{skill_name} index={index}"
        )

    recorded_queries = progress_row.get("target_queries_used")
    if recorded_queries is not None:
        return int(recorded_queries) > 0

    run_id = str(progress_row.get("run_id", "")).strip()
    if not run_id:
        raise ValueError(
            f"Missing run_id for unscored outcome: {source_dir.name}/"
            f"{skill_name} index={index}"
        )
    trace_path = source_runs_root / source_dir.name / skill_name / run_id / "compact_trace.json"
    if not trace_path.exists():
        raise FileNotFoundError(
            f"Cannot determine target-query use for index={index}; trace is missing: "
            f"{trace_path}"
        )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_budget = trace.get("budget")
    if isinstance(trace_budget, dict) and trace_budget.get("used_environment_calls") is not None:
        return int(trace_budget["used_environment_calls"]) > 0

    for step in trace.get("steps", []):
        if step.get("action_type") != "invoke_skill":
            continue
        output = step.get("output", {})
        if output.get("responses") or output.get("candidate_results"):
            return True
    return False


def _load_outcomes(
    *,
    source_dir: Path,
    source_runs_root: Path,
    skill_names: list[str],
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[tuple[str, int], bool]]:
    """Load and validate the all-skills outcome matrix."""
    available = {
        path.name
        for path in source_dir.iterdir()
        if path.is_dir() and (path / STAGE1_PER_SEED).exists()
    }
    if available != set(skill_names):
        missing = sorted(set(skill_names) - available)
        extra = sorted(available - set(skill_names))
        raise ValueError(
            f"Expected exactly {len(skill_names)} workflow skills under {source_dir}; "
            f"missing={missing}, extra={extra}"
        )

    outcomes: dict[str, dict[int, dict[str, Any]]] = {}
    attempted: dict[tuple[str, int], bool] = {}
    expected_indices: set[int] | None = None
    canonical_seed: dict[int, tuple[str, str]] = {}

    for skill_name in skill_names:
        skill_dir = source_dir / skill_name
        rows = _load_latest_jsonl(skill_dir / STAGE1_PER_SEED)
        progress = _load_latest_jsonl(skill_dir / STAGE1_PROGRESS)
        indices = set(rows)
        if expected_indices is None:
            expected_indices = indices
        elif indices != expected_indices:
            raise ValueError(f"Dataset index mismatch in {skill_dir}")

        for index, row in rows.items():
            prompt_key = (
                str(row.get("seed_prompt", "")),
                str(row.get("risk_category", "unclassified")),
            )
            if index in canonical_seed and canonical_seed[index] != prompt_key:
                raise ValueError(
                    f"Seed/risk mismatch across skills in {source_dir.name}, index={index}"
                )
            canonical_seed[index] = prompt_key

            score = int(row.get("best_score") or 0)
            if score < 0 or score > 5:
                raise ValueError(
                    f"Invalid score {score} in {skill_dir / STAGE1_PER_SEED}, index={index}"
                )
            if score > 0:
                attempted[(skill_name, index)] = True
                samples = (row.get("skill_traces") or {}).get(skill_name, [])
                if not samples:
                    raise ValueError(
                        f"Scored row has no trace samples: {source_dir.name}/"
                        f"{skill_name} index={index}"
                    )
            else:
                attempted[(skill_name, index)] = _target_was_attempted(
                    source_dir=source_dir,
                    source_runs_root=source_runs_root,
                    skill_name=skill_name,
                    index=index,
                    progress=progress,
                )
        outcomes[skill_name] = rows

    indices = sorted(expected_indices or set())
    if not indices or indices != list(range(indices[-1] + 1)):
        raise ValueError(
            f"Expected a complete contiguous dataset starting at index 0 in {source_dir}"
        )
    return outcomes, attempted


def materialize_stage1_replay(
    *,
    source_dir: Path,
    source_runs_root: Path,
    output_dir: Path,
    workflow_file: Path,
    target_query_budget: int = 10,
    ucb_exploration_weight: float = 0.45,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Replay one all-skills experiment and write a Stage-2-ready checkpoint."""
    source_dir = source_dir.resolve()
    source_runs_root = source_runs_root.resolve()
    output_dir = output_dir.resolve()
    if target_query_budget <= 0:
        raise ValueError("target_query_budget must be positive")

    workflow = Workflow.from_file(workflow_file.resolve())
    skill_names = workflow.get_group("search")
    if not skill_names:
        raise ValueError(f"Workflow has no search skills: {workflow_file}")

    managed_paths = [
        output_dir / STAGE1_PER_SEED,
        output_dir / STAGE1_PROGRESS,
        output_dir / STAGE1_MEMORY,
        output_dir / PROMPT_MEMORY,
        output_dir / REPLAY_MANIFEST,
    ]
    existing = [path for path in managed_paths if path.exists()]
    stage2_paths = [
        output_dir / "stage2_results.json",
        output_dir / "stage2_progress.jsonl",
        output_dir / "stage2_per_seed.jsonl",
        output_dir / "stage2_evolve.json",
    ]
    if existing and not overwrite:
        if len(existing) == len(managed_paths):
            existing_manifest = json.loads(
                (output_dir / REPLAY_MANIFEST).read_text(encoding="utf-8")
            )
            same_checkpoint = (
                existing_manifest.get("source_dir") == str(source_dir)
                and existing_manifest.get("target_query_budget") == target_query_budget
                and existing_manifest.get("ucb_exploration_weight")
                == ucb_exploration_weight
            )
            if same_checkpoint:
                return existing_manifest
        raise FileExistsError(
            f"A different or partial replay checkpoint exists under {output_dir}; "
            "pass overwrite=True only before Stage 2 has started"
        )
    if overwrite and any(path.exists() for path in stage2_paths):
        raise FileExistsError(
            f"Refusing to overwrite Stage 1 inputs after Stage 2 started under {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes, target_attempted = _load_outcomes(
        source_dir=source_dir,
        source_runs_root=source_runs_root,
        skill_names=skill_names,
    )
    indices = sorted(outcomes[skill_names[0]])

    planner = DeterministicPlanner()
    with tempfile.TemporaryDirectory(prefix="agentic-redteam-offline-replay-") as tmp_dir:
        memory = PersistentMemory(
            Path(tmp_dir),
            ucb_exploration_weight=ucb_exploration_weight,
        )
        # Replaying should persist exactly once, after the complete matrix is built.
        memory.save = lambda: None  # type: ignore[method-assign]

        per_seed_rows: list[dict[str, Any]] = []
        progress_rows: list[dict[str, Any]] = []
        prompt_memory: dict[str, list[dict[str, Any]]] = {}
        selected_counts: Counter[str] = Counter()
        unscored_target_calls = 0

        for index in indices:
            canonical = outcomes[skill_names[0]][index]
            seed_prompt = str(canonical.get("seed_prompt", ""))
            risk_category = str(canonical.get("risk_category", "unclassified"))
            selected_skills: list[str] = []
            skill_scores: dict[str, int] = {}
            skill_traces: dict[str, list[dict[str, Any]]] = {}
            target_queries_used = 0
            success = False

            while (
                target_queries_used < target_query_budget
                and len(selected_skills) < len(skill_names)
            ):
                ordered = planner._sort_search_pool_by_memory(
                    skill_names,
                    memory,
                    [risk_category],
                )
                skill_name = next(
                    name for name in ordered if name not in selected_skills
                )
                selected_skills.append(skill_name)
                selected_counts[skill_name] += 1

                source_row = outcomes[skill_name][index]
                score = int(source_row.get("best_score") or 0)
                did_query_target = target_attempted[(skill_name, index)]
                if did_query_target:
                    target_queries_used += 1
                elif score > 0:
                    raise AssertionError("A scored outcome must consume a target query")

                if score > 0:
                    skill_scores[skill_name] = score
                    skill_traces[skill_name] = list(
                        (source_row.get("skill_traces") or {}).get(skill_name, [])
                    )
                    memory.update(
                        skill_name=skill_name,
                        risk_types=[risk_category],
                        judge_score=score,
                        success=(score == 5),
                    )
                elif did_query_target:
                    unscored_target_calls += 1

                if score == 5:
                    success = True
                    break

            best_score = max(skill_scores.values()) if skill_scores else 0
            avg_score = (
                sum(skill_scores.values()) / len(skill_scores)
                if skill_scores
                else 0.0
            )
            if success:
                stop_reason = "search_success"
            elif target_queries_used >= target_query_budget:
                stop_reason = "target_query_budget_exhausted"
            else:
                stop_reason = "all_search_skills_exhausted"

            replay_row = {
                "index": index,
                "seed_prompt": seed_prompt,
                "risk_category": risk_category,
                "success": success,
                "best_score": best_score,
                "avg_score": avg_score,
                "skill_scores": skill_scores,
                "skill_traces": skill_traces,
                "target_query_budget": target_query_budget,
                "target_queries_used": target_queries_used,
                "stop_reason": stop_reason,
                "offline_selected_skills": selected_skills,
                "offline_source_experiment": source_dir.name,
            }
            per_seed_rows.append(replay_row)
            progress_rows.append({
                "index": index,
                "run_id": f"offline-replay-{source_dir.name}-{index}",
                "success": success,
                "steps": len(skill_scores),
                "target_query_budget": target_query_budget,
                "target_queries_used": target_queries_used,
                "stop_reason": stop_reason,
                "offline_selected_skills": selected_skills,
            })

            if not success:
                prompt_memory.setdefault(risk_category, []).append({
                    "index": index,
                    "seed_prompt": seed_prompt,
                    "avg_score": avg_score,
                    "best_score": best_score,
                    "skill_traces": skill_traces,
                })

        matrix = memory.summary()["matrix"]

    success_count = sum(int(row["success"]) for row in per_seed_rows)
    failure_count = len(per_seed_rows) - success_count
    manifest = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "source_experiment": source_dir.name,
        "source_dir": str(source_dir),
        "source_runs_root": str(source_runs_root),
        "workflow_file": str(workflow_file.resolve()),
        "skill_names": skill_names,
        "target_query_budget": target_query_budget,
        "ucb_exploration_weight": ucb_exploration_weight,
        "total_prompts": len(per_seed_rows),
        "successes": success_count,
        "failures": failure_count,
        "asr": success_count / len(per_seed_rows),
        "unscored_target_calls_selected": unscored_target_calls,
        "selected_skill_counts": dict(selected_counts),
        "output_dir": str(output_dir),
    }

    _write_jsonl_atomic(output_dir / STAGE1_PER_SEED, per_seed_rows)
    _write_jsonl_atomic(output_dir / STAGE1_PROGRESS, progress_rows)
    write_json_atomic(output_dir / STAGE1_MEMORY, matrix)
    write_json_atomic(output_dir / PROMPT_MEMORY, prompt_memory)
    write_json_atomic(output_dir / REPLAY_MANIFEST, manifest)
    return manifest
