"""Persistent memory for cross-run skill performance tracking."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any


class PersistentMemory:
    """Persistent risk_type x skill matrix with automatic disk sync.

    Structure:
        {
            "harmful_content": {
                "rewrite-emoji": {
                    "attempts": 10,
                    "successes": 3,
                    "avg_judge_score": 2.8,
                    "asr": 0.30,
                    "ucb_score": 3.25,
                    "_total_judge_score": 28
                }
            }
        }

    UCB formula uses avg_judge_score (continuous 1-5 signal) instead of binary ASR
    for finer-grained exploration/exploitation.
    """

    def __init__(
        self,
        memory_dir: Path,
        ucb_exploration_weight: float = 0.45,
        filename: str = "risk_matrix.json",
        baseline_file: str | Path | None = None,
    ) -> None:
        """Load from disk or initialize from a frozen baseline.

        Args:
            memory_dir: Directory to store memory files (will be created if not exists)
            ucb_exploration_weight: UCB exploration coefficient (default 0.45)
            filename: Name of the matrix JSON file (default "risk_matrix.json")
            baseline_file: Optional path to a frozen warmup matrix. If the target
                memory file does not exist, the baseline is copied as the starting
                point. The baseline file itself is never written to.
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / filename
        self.baseline_file = Path(baseline_file) if baseline_file else None
        self.ucb_exploration_weight = ucb_exploration_weight

        self._matrix: dict[str, dict[str, dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        """Load matrix from disk, falling back to baseline if target doesn't exist."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    self._matrix = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[Warning] Failed to load memory from {self.memory_file}: {e}")
                self._matrix = {}
        elif self.baseline_file and self.baseline_file.exists():
            try:
                with open(self.baseline_file, "r", encoding="utf-8") as f:
                    self._matrix = json.load(f)
                self.save()
                print(f"[Info] Initialized memory from baseline: {self.baseline_file}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[Warning] Failed to load baseline from {self.baseline_file}: {e}")
                self._matrix = {}
        else:
            self._matrix = {}

    def save(self) -> None:
        """Save matrix to disk."""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self._matrix, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[Error] Failed to save memory to {self.memory_file}: {e}")

    def update(
        self,
        skill_name: str,
        risk_types: list[str],
        judge_score: int,
        success: bool,
    ) -> None:
        """Update matrix with one evaluation result.

        Args:
            skill_name: Skill name
            risk_types: List of risk types this evaluation belongs to
            judge_score: Judge score (1-5) from evaluator
            success: Whether this was a successful attack (judge_score == 5)
        """
        for risk_type in risk_types:
            if not risk_type:
                risk_type = "unclassified"

            if risk_type not in self._matrix:
                self._matrix[risk_type] = {}

            risk_bucket = self._matrix[risk_type]

            if skill_name not in risk_bucket:
                risk_bucket[skill_name] = {
                    "attempts": 0,
                    "successes": 0,
                    "avg_judge_score": 0.0,
                    "asr": 0.0,
                    "ucb_score": 0.0,
                    "_total_judge_score": 0.0,
                }

            cell = risk_bucket[skill_name]

            cell["attempts"] += 1
            cell["successes"] += int(success)
            cell["_total_judge_score"] += judge_score

            cell["avg_judge_score"] = cell["_total_judge_score"] / cell["attempts"]
            cell["asr"] = cell["successes"] / cell["attempts"]

            self._update_all_ucb_in_risk_type(risk_type)

        self.save()

    def _update_all_ucb_in_risk_type(self, risk_type: str) -> None:
        """Recompute UCB scores for all skills in a given risk_type."""
        if risk_type not in self._matrix:
            return

        risk_bucket = self._matrix[risk_type]
        total_attempts = sum(cell.get("attempts", 0) for cell in risk_bucket.values())

        if total_attempts == 0:
            return

        for skill_name, cell in risk_bucket.items():
            skill_attempts = cell.get("attempts", 0)
            if skill_attempts == 0:
                continue

            avg_score = cell.get("avg_judge_score", 1.0)
            if total_attempts > 1:
                exploration_bonus = self.ucb_exploration_weight * math.sqrt(
                    math.log(total_attempts) / skill_attempts
                )
                cell["ucb_score"] = avg_score + exploration_bonus
            else:
                cell["ucb_score"] = avg_score + self.ucb_exploration_weight

    def get_skill_stats(
        self,
        skill_name: str,
        risk_type: str | None = None,
    ) -> dict[str, Any]:
        """Get statistics for a specific skill.

        Args:
            skill_name: Skill name to query
            risk_type: If None, aggregate across all risk_types

        Returns:
            Dict with aggregated stats (attempts, successes, avg_judge_score, asr, ucb_score)
            Returns empty dict if skill not found.
        """
        if risk_type:
            risk_bucket = self._matrix.get(risk_type, {})
            cell = risk_bucket.get(skill_name, {})
            if not cell:
                return {}
            return {
                "attempts": cell.get("attempts", 0),
                "successes": cell.get("successes", 0),
                "avg_judge_score": cell.get("avg_judge_score", 0.0),
                "asr": cell.get("asr", 0.0),
                "ucb_score": cell.get("ucb_score", 0.0),
            }
        else:
            total_attempts = 0
            total_successes = 0
            total_judge_score = 0.0
            max_ucb_score = 0.0

            for risk_bucket in self._matrix.values():
                cell = risk_bucket.get(skill_name, {})
                if cell:
                    total_attempts += cell.get("attempts", 0)
                    total_successes += cell.get("successes", 0)
                    total_judge_score += cell.get("_total_judge_score", 0.0)
                    max_ucb_score = max(max_ucb_score, cell.get("ucb_score", 0.0))

            if total_attempts == 0:
                return {}

            return {
                "attempts": total_attempts,
                "successes": total_successes,
                "avg_judge_score": total_judge_score / total_attempts,
                "asr": total_successes / total_attempts,
                "ucb_score": max_ucb_score,
            }

    def get_all_skills(self) -> list[str]:
        """Get all skill names in memory (across all risk_types)."""
        all_skills = set()
        for risk_bucket in self._matrix.values():
            all_skills.update(risk_bucket.keys())
        return sorted(all_skills)

    def get_all_risk_types(self) -> list[str]:
        """Get all risk types in memory."""
        return list(self._matrix.keys())

    def summary(self) -> dict[str, Any]:
        """Get a summary of memory state for logging/debugging."""
        all_skills = self.get_all_skills()
        risk_types = list(self._matrix.keys())

        total_attempts = sum(
            cell["attempts"]
            for risk_bucket in self._matrix.values()
            for cell in risk_bucket.values()
        )

        return {
            "total_skills": len(all_skills),
            "risk_types": risk_types,
            "total_attempts": total_attempts,
            "matrix": copy.deepcopy(self._matrix),
        }

    def clear(self) -> None:
        """Clear all memory (for testing/debugging)."""
        self._matrix = {}
        self.save()
