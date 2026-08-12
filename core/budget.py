"""Simple budget tracking for planner steps and tool usage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetManager:
    """Track and consume step, skill-call, and environment-call budgets."""

    max_steps: int
    max_skill_calls: int
    max_environment_calls: int
    used_steps: int = 0
    used_skill_calls: int = 0
    used_environment_calls: int = 0

    def remaining(self) -> dict[str, int]:
        """Return the remaining budget in each category."""
        return {
            "steps": max(self.max_steps - self.used_steps, 0),
            "skill_calls": max(self.max_skill_calls - self.used_skill_calls, 0),
            "environment_calls": max(
                self.max_environment_calls - self.used_environment_calls,
                0,
            ),
        }

    def can_continue(self) -> bool:
        """Return whether the run can continue."""
        remaining = self.remaining()
        return all(value > 0 for value in remaining.values())

    def consume_step(self) -> None:
        """Consume one planner step."""
        self.used_steps += 1

    def consume_skill(self) -> None:
        """Consume one skill call."""
        self.used_skill_calls += 1

    def consume_environment(self) -> None:
        """Consume one environment call."""
        self.used_environment_calls += 1
