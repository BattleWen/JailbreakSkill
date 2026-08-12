"""Skill execution harness — eliminates boilerplate in skill scripts.

Each skill only needs to define its core logic function. The harness handles:
- JSON stdin/stdout protocol
- Context parsing (candidate_count, seed_prompt, extra, backend_config)
- Result serialization with standard metadata structure
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

from core.skill_runtime import (
    finalize_verbatim_constraints_candidate,
    parse_candidate_count,
    request_model_candidates_from_context,
)


@dataclass
class SkillInput:
    """Parsed skill execution context."""

    seed: str
    candidate_count: int
    extra: dict[str, Any]
    backend_config: dict[str, Any]
    raw_context: dict[str, Any]


def _read_context() -> SkillInput:
    """Read and parse SkillContext JSON from stdin."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    context = json.load(sys.stdin)
    action_args = dict(context.get("extra", {}).get("action_args", {}))
    candidate_count = parse_candidate_count(action_args)
    seed = str(context.get("seed_prompt", ""))
    extra = dict(context.get("extra", {}))
    backend_config = dict(extra.get("skill_model_backend", {}))

    return SkillInput(
        seed=seed,
        candidate_count=candidate_count,
        extra=extra,
        backend_config=backend_config,
        raw_context=context,
    )


def _write_result(
    skill_name: str,
    candidates: list[dict[str, str]],
    original_seed: str,
    generation_backend: dict[str, str],
    *,
    rationale: str = "",
) -> None:
    """Write SkillExecutionResult JSON to stdout."""
    result = {
        "skill_name": skill_name,
        "candidates": candidates,
        "rationale": rationale,
        "artifacts": {
            "original_seed": original_seed,
            "candidate_count": len(candidates),
            "generation_backend": generation_backend.get("backend", "unknown"),
            "generation_model": generation_backend.get("model", "unknown"),
        },
        "metadata": {
            "protocol_version": "1",
            "generation_backend": generation_backend,
        },
    }
    json.dump(result, sys.stdout, ensure_ascii=False)


def run_llm_skill(
    skill_name: str,
    build_prompt_fn: Callable[[str, int], str],
    *,
    rationale: str = "",
    candidate_transform_fn: Callable[[str, str, int], str] | None = None,
) -> None:
    """Standard entry point for LLM Rewrite skills.

    Args:
        skill_name: Skill identifier (must match SKILL.md name)
        build_prompt_fn: Function(seed, candidate_count) -> user_prompt string
        rationale: Optional rationale for trace output
        candidate_transform_fn: Optional deterministic postprocessor receiving
            (candidate_text, original_seed, candidate_index).
    """
    ctx = _read_context()
    user_prompt = build_prompt_fn(ctx.seed, ctx.candidate_count)
    candidates, backend = request_model_candidates_from_context(
        backend_config=ctx.backend_config,
        user_prompt=user_prompt,
        candidate_count=ctx.candidate_count,
        extra=ctx.extra,
    )
    transformed: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates):
        text = str(candidate.get("text", ""))
        if candidate_transform_fn is not None:
            text = candidate_transform_fn(text, ctx.seed, index)
        text = finalize_verbatim_constraints_candidate(text, ctx.seed, index)
        if text.strip():
            transformed.append({**candidate, "text": text})
    candidates = transformed
    _write_result(skill_name, candidates, ctx.seed, backend, rationale=rationale)


def run_single_llm_skill(
    skill_name: str,
    build_prompt_fn: Callable[[str], str],
    *,
    rationale: str = "",
    candidate_transform_fn: Callable[[str, str, int], str] | None = None,
) -> None:
    """Run one canonical model-authored transformation exactly once."""
    ctx = _read_context()
    candidates, backend = request_model_candidates_from_context(
        backend_config=ctx.backend_config,
        user_prompt=build_prompt_fn(ctx.seed),
        candidate_count=1,
        extra=ctx.extra,
    )
    transformed: list[dict[str, str]] = []
    for candidate in candidates[:1]:
        text = str(candidate.get("text", ""))
        if candidate_transform_fn is not None:
            text = candidate_transform_fn(text, ctx.seed, 0)
        text = finalize_verbatim_constraints_candidate(text, ctx.seed, 0)
        if text.strip():
            transformed.append({**candidate, "text": text})
    _write_result(skill_name, transformed, ctx.seed, backend, rationale=rationale)


def run_template_skill(
    skill_name: str,
    wrap_fn: Callable[[str, int], str],
    *,
    rationale: str = "",
    allowed_risk_labels: list[str] | None = None,
) -> None:
    """Standard entry point for Deterministic Template skills.

    Args:
        skill_name: Skill identifier (must match SKILL.md name)
        wrap_fn: Function(query, variant) -> transformed string
        rationale: Optional rationale for trace output
        allowed_risk_labels: Optional exact risk labels accepted from SkillContext.
    """
    ctx = _read_context()
    declared_labels = {
        " ".join(str(value).casefold().split())
        for value in (allowed_risk_labels or [])
        if str(value).strip()
    }
    raw_runtime_labels = ctx.extra.get("seed_risk_types", [])
    if isinstance(raw_runtime_labels, str):
        raw_runtime_labels = [raw_runtime_labels]
    elif not isinstance(raw_runtime_labels, list):
        raw_runtime_labels = []
    runtime_labels = {
        " ".join(str(value).casefold().split())
        for value in raw_runtime_labels
        if str(value).strip() and str(value).casefold() != "unclassified"
    }
    if declared_labels and runtime_labels and not (declared_labels & runtime_labels):
        _write_result(
            skill_name,
            [],
            ctx.seed,
            {"backend": "deterministic", "model": "template"},
            rationale="scope mismatch: supplied risk labels are outside the skill profile",
        )
        return
    candidates = [
        {"text": wrap_fn(ctx.seed, i)}
        for i in range(ctx.candidate_count)
    ]
    _write_result(
        skill_name,
        candidates,
        ctx.seed,
        {"backend": "deterministic", "model": "template"},
        rationale=rationale,
    )


def run_single_template_skill(
    skill_name: str,
    wrap_fn: Callable[[str], str],
    *,
    rationale: str = "",
    allowed_risk_labels: list[str] | None = None,
) -> None:
    """Run one canonical deterministic implementation exactly once.

    Unlike :func:`run_template_skill`, this entry point deliberately ignores a
    larger requested ``candidate_count``.  It is used by generated external
    skills whose source alternatives are resolved during authorship instead of
    being exposed as runtime variants.
    """
    ctx = _read_context()
    declared_labels = {
        " ".join(str(value).casefold().split())
        for value in (allowed_risk_labels or [])
        if str(value).strip()
    }
    raw_runtime_labels = ctx.extra.get("seed_risk_types", [])
    if isinstance(raw_runtime_labels, str):
        raw_runtime_labels = [raw_runtime_labels]
    elif not isinstance(raw_runtime_labels, list):
        raw_runtime_labels = []
    runtime_labels = {
        " ".join(str(value).casefold().split())
        for value in raw_runtime_labels
        if str(value).strip() and str(value).casefold() != "unclassified"
    }
    if declared_labels and runtime_labels and not (declared_labels & runtime_labels):
        _write_result(
            skill_name,
            [],
            ctx.seed,
            {"backend": "deterministic", "model": "template"},
            rationale="scope mismatch: supplied risk labels are outside the skill profile",
        )
        return
    candidate = str(wrap_fn(ctx.seed))
    candidates = [{"text": candidate}] if candidate.strip() else []
    _write_result(
        skill_name,
        candidates,
        ctx.seed,
        {"backend": "deterministic", "model": "template"},
        rationale=rationale,
    )
