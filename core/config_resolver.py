"""Configuration resolution for the planner loop.

Handles: LLM backend config inheritance, timeout calculation, config normalization,
and model name extraction.
"""

from __future__ import annotations

import os
from typing import Any

from core.embedding_client import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL

LLM_BACKEND = "llm"


class ConfigResolver:
    """Resolves and normalizes configuration for the planner loop."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @staticmethod
    def normalize(config: dict[str, Any]) -> dict[str, Any]:
        """Normalize config sections and default guard and environment to enabled."""
        normalized = dict(config)

        evaluator_config = dict(normalized.get("evaluator", {}))
        guard_config = dict(evaluator_config.get("guard_model", {}))
        guard_enabled = guard_config.get("enabled")
        guard_config["enabled"] = True if guard_enabled is None else bool(guard_enabled)
        evaluator_config["guard_model"] = guard_config
        normalized["evaluator"] = evaluator_config

        environment_config = dict(normalized.get("environment", {}))
        environment_backend = environment_config.get("backend")
        environment_config["backend"] = LLM_BACKEND if environment_backend is None else environment_backend
        normalized["environment"] = environment_config

        return normalized

    def resolve_backend(self, *keys: str) -> dict[str, Any]:
        """Resolve a nested LLM backend config, inheriting planner endpoint if configured."""
        section = self.config
        for k in keys:
            section = dict(section.get(k, {}))
        if not section:
            return {"enabled": False}

        if bool(section.get("inherit_planner_endpoint", False)):
            planner_config = dict(self.config.get("planner", {}).get("llm", {}))
            for key in ("base_url", "model", "api_key"):
                if not section.get(key):
                    section[key] = planner_config.get(key, "")
        return section

    def resolve_skill_backend(self) -> dict[str, Any]:
        """Resolve the model backend config passed to model-backed skills."""
        return self._apply_llm_env_overrides(
            self.resolve_backend("skills", "llm"),
            prefix="SKILL",
        )

    def resolve_meta_backend(self) -> dict[str, Any]:
        """Resolve the model backend config used by model-backed meta-skills."""
        return self._apply_llm_env_overrides(
            self.resolve_backend("meta_skills", "llm"),
            prefix="META_SKILL",
        )

    def resolve_meta_author_backend(self) -> dict[str, Any]:
        """Resolve the external-skill author, falling back to the shared meta backend."""
        raw = dict(self.config.get("meta_skills", {}).get("author_llm", {}))
        base = (
            self.resolve_backend("meta_skills", "author_llm")
            if raw
            else self.resolve_meta_backend()
        )
        return self._apply_llm_env_overrides(base, prefix="META_AUTHOR")

    def resolve_meta_judge_backend(self) -> dict[str, Any]:
        """Resolve semantic judges, falling back to the shared meta backend."""
        raw = dict(self.config.get("meta_skills", {}).get("judge_llm", {}))
        base = (
            self.resolve_backend("meta_skills", "judge_llm")
            if raw
            else self.resolve_meta_backend()
        )
        return self._apply_llm_env_overrides(base, prefix="META_JUDGE")

    @staticmethod
    def _apply_llm_env_overrides(section: dict[str, Any], *, prefix: str) -> dict[str, Any]:
        """Apply role-specific .env overrides to one resolved LLM backend."""
        resolved = dict(section)
        for config_key, env_suffix in (
            ("base_url", "BASE_URL"),
            ("model", "MODEL"),
            ("api_key", "API_KEY"),
        ):
            value = os.getenv(f"{prefix}_{env_suffix}")
            if value:
                resolved[config_key] = value
        max_tokens_value = os.getenv(f"{prefix}_MAX_TOKENS")
        if max_tokens_value:
            try:
                max_tokens = int(max_tokens_value)
            except ValueError as exc:
                raise ValueError(
                    f"{prefix}_MAX_TOKENS must be a positive integer"
                ) from exc
            if max_tokens <= 0:
                raise ValueError(f"{prefix}_MAX_TOKENS must be a positive integer")
            resolved["max_tokens"] = max_tokens
        timeout_value = os.getenv(f"{prefix}_TIMEOUT_SECONDS")
        if timeout_value:
            try:
                timeout_seconds = int(timeout_value)
            except ValueError as exc:
                raise ValueError(
                    f"{prefix}_TIMEOUT_SECONDS must be a positive integer"
                ) from exc
            if timeout_seconds <= 0:
                raise ValueError(
                    f"{prefix}_TIMEOUT_SECONDS must be a positive integer"
                )
            resolved["timeout_seconds"] = timeout_seconds
        disable_thinking_value = os.getenv(f"{prefix}_DISABLE_THINKING")
        if disable_thinking_value:
            normalized = disable_thinking_value.strip().casefold()
            if normalized not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
                raise ValueError(
                    f"{prefix}_DISABLE_THINKING must be a boolean"
                )
            resolved["disable_thinking"] = normalized in {"1", "true", "yes", "on"}
        return resolved

    def resolve_embedding_backend(self) -> dict[str, Any]:
        """Resolve the dedicated embedding backend used by external-text ingestion."""
        section = dict(self.config.get("embeddings", {}))
        if not section:
            section = {
                "enabled": True,
                "base_url": "https://api.openai.com/v1",
                "model": DEFAULT_EMBEDDING_MODEL,
                "dimensions": DEFAULT_EMBEDDING_DIMENSIONS,
                "api_key": "",
            }
        return section

    def resolve_external_search(self) -> dict[str, Any]:
        """Resolve external web-search backends without persisting credentials."""
        section = dict(self.config.get("external_search", {}))
        bocha = dict(section.get("bocha", {}))
        bocha.setdefault("enabled", True)
        bocha.setdefault("base_url", "https://api.bochaai.com/v1")
        bocha.setdefault("freshness", "noLimit")
        bocha.setdefault("summary", False)

        env_base_url = os.getenv("BOCHA_BASE_URL")
        if env_base_url:
            bocha["base_url"] = env_base_url
        env_api_key = os.getenv("BOCHA_API_KEY")
        if env_api_key:
            bocha["api_key"] = env_api_key
        else:
            api_key = str(bocha.get("api_key", "")).strip()
            if api_key.startswith("${") and api_key.endswith("}"):
                api_key = ""
            bocha["api_key"] = api_key

        section["bocha"] = bocha
        return section

    def executor_timeout(self) -> int:
        """Choose a subprocess timeout that safely exceeds nested backend calls."""
        planner_llm = dict(self.config.get("planner", {}).get("llm", {}))
        planner_timeout = int(planner_llm.get("timeout_seconds", 8))
        failure_analysis_retries = max(
            0,
            min(5, int(planner_llm.get("failure_analysis_retries", 2))),
        )
        failure_analysis_timeout = planner_timeout * (failure_analysis_retries + 1)
        # Use the resolved role backends here. Promotion launches generated skills
        # in a subprocess, so reading the raw YAML would ignore role-specific
        # timeout overrides and could leave that subprocess waiting for the global
        # 1800-second meta timeout even when the caller requested a bounded test.
        meta_timeout = int(self.resolve_meta_backend().get("timeout_seconds", 12))
        meta_author_timeout = int(
            self.resolve_meta_author_backend().get("timeout_seconds", meta_timeout)
        )
        meta_judge_timeout = int(
            self.resolve_meta_judge_backend().get("timeout_seconds", meta_timeout)
        )
        skill_timeout = int(self.resolve_skill_backend().get("timeout_seconds", 12))
        evaluator_timeout = int(
            self.config.get("evaluator", {}).get("guard_model", {}).get("timeout_seconds", 8)
        )
        environment_timeout = int(
            dict(self.config.get("environment", {}).get("llm", {})).get("timeout_seconds", 12)
        )
        return max(
            30,
            failure_analysis_timeout,
            meta_timeout,
            meta_author_timeout,
            meta_judge_timeout,
            skill_timeout,
            evaluator_timeout,
            environment_timeout,
        ) + 5

    def model_names(self) -> dict[str, str]:
        """Extract model names from config for trace metadata."""
        cfg = self.config
        return {
            "planner": cfg.get("planner", {}).get("llm", {}).get("model", ""),
            "skills": self.resolve_skill_backend().get("model", ""),
            "meta_skills": self.resolve_meta_backend().get("model", ""),
            "meta_author": self.resolve_meta_author_backend().get("model", ""),
            "meta_judge": self.resolve_meta_judge_backend().get("model", ""),
            "evaluator": cfg.get("evaluator", {}).get("guard_model", {}).get("model", ""),
            "risk_classifier": cfg.get("risk_classifier", {}).get("model", ""),
            "target": cfg.get("environment", {}).get("llm", {}).get("model", ""),
            "embeddings": self.resolve_embedding_backend().get(
                "model",
                DEFAULT_EMBEDDING_MODEL,
            ),
        }
