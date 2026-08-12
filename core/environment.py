"""Environment adapters for mock or LLM target models."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from core.llm_config import LLMClientConfig, call_chat_completion
from core.retry_policy import is_retryable_transport_error
from core.utils import shorten

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


_DEFAULT_POLICY_BLOCK_CODES = frozenset({
    "content_filter",
    "content_policy_violation",
    "cyber_policy",
    "safety_policy_violation",
})
_POLICY_BLOCK_MESSAGE_MARKERS = (
    "content management policy",
    "content policy violation",
    "output blocked by content filtering policy",
    "prompt was filtered",
    "response was filtered",
    "flagged for possible cybersecurity risk",
    "limited access to this content for safety reasons",
)


def _is_retryable_target_error(exc: Exception) -> bool:
    """Return whether a target failure is transient and safe to retry.

    Explicit policy responses, authentication failures, and ordinary 4xx
    errors are not retryable. SDK retries stay disabled so every transport
    attempt remains visible in the trace.
    """
    return is_retryable_transport_error(exc)


def _provider_policy_block_code(
    exc: Exception,
    configured_codes: frozenset[str],
) -> str | None:
    """Identify an explicit provider policy rejection, preferring structured fields."""
    structured_values: list[str] = []
    for attr in ("code", "type"):
        value = getattr(exc, attr, None)
        if value is not None:
            structured_values.append(str(value).lower())

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error_body = body.get("error", body)
        if isinstance(error_body, dict):
            for key in ("code", "type"):
                value = error_body.get(key)
                if value is not None:
                    structured_values.append(str(value).lower())

    for value in structured_values:
        if value in configured_codes:
            return value

    # Some OpenAI-compatible gateways flatten or discard the structured body.
    # Keep a message fallback for explicit policy language only; generic 4xx/5xx,
    # EOF, timeout, and connection failures must never be classified as refusals.
    normalized_message = str(exc).lower()
    for code in configured_codes:
        if code in normalized_message:
            return code
    if any(marker in normalized_message for marker in _POLICY_BLOCK_MESSAGE_MARKERS):
        return "provider_policy_block"
    return None


@dataclass
class MockEnvironment:
    """Toy target model that only returns harmless mock responses."""

    target_profile: dict[str, object]

    def execute(self, candidate_text: str) -> dict[str, object]:
        """Produce a mock response for one candidate text."""
        lowered = candidate_text.lower()

        if "{" in candidate_text and "}" in candidate_text:
            response = "Mock target response: accepted structured input and returned a safe summary."
            style = "structured"
        elif len(candidate_text) > 180:
            response = "Mock target response: prompt is long, returning a short and neutral answer."
            style = "brief"
        elif any(token in lowered for token in ("teacher", "guide", "explain", "friendly")):
            response = (
                "Mock target response: provided a helpful educational reply with a calm tone."
            )
            style = "helpful"
        else:
            response = (
                "Mock target response: safely acknowledged the transformed request theme: "
                + shorten(candidate_text, 72)
            )
            style = "echo"

        return {
            "response_text": response,
            "style": style,
            "candidate_text": candidate_text,
            "backend": "mock",
            "model_name": str(self.target_profile.get("model_name", "mock-target-model")),
        }


class OpenAICompatibleEnvironment:
    """Remote target model behind an OpenAI-compatible chat-completions endpoint.

    Raises RuntimeError if the LLM endpoint is unavailable or misconfigured.
    """

    def __init__(self, target_profile: dict[str, object], config: dict[str, Any] | None = None) -> None:
        self.target_profile = dict(target_profile)
        self.config = dict(config or {})
        self.cfg = LLMClientConfig.from_dict(
            self.config,
            defaults={
                "timeout_seconds": 12,
                "temperature": 0.0,
                "max_tokens": 512,
                "enabled": True,
                "transport": "openai_sdk" if OpenAI is not None else "http",
            },
            env_overrides={
                "base_url": "TARGET_BASE_URL",
                "model": "TARGET_MODEL",
                "api_key": "TARGET_API_KEY",
            },
        )
        self.enabled = self.cfg.enabled
        self.include_system_prompt = bool(self.config.get("include_system_prompt", False))
        self.send_sampling_params = bool(self.config.get("send_sampling_params", False))
        self.disable_thinking = bool(self.config.get("disable_thinking", False))
        self.system_prompt = str(self.config.get("system_prompt", ""))
        self.target_transport_retries = max(
            0, int(self.config.get("target_transport_retries", 3))
        )
        self.target_retry_backoff_seconds = max(
            0.0, float(self.config.get("target_retry_backoff_seconds", 1.0))
        )
        configured_policy_codes = self.config.get("policy_block_error_codes", [])
        if isinstance(configured_policy_codes, str):
            configured_policy_codes = [configured_policy_codes]
        self.policy_block_error_codes = _DEFAULT_POLICY_BLOCK_CODES | frozenset(
            str(code).strip().lower()
            for code in configured_policy_codes
            if str(code).strip()
        )

    def execute(self, candidate_text: str) -> dict[str, object]:
        """Execute one candidate against the remote LLM environment."""
        if not self.enabled:
            raise RuntimeError(
                "Remote environment is disabled. "
                "Set environment.llm.enabled to true in config.yaml"
            )

        if not self.cfg.base_url or not self.cfg.model:
            raise RuntimeError(
                "Remote environment requires base_url and model configuration. "
                "Set TARGET_BASE_URL and TARGET_MODEL environment variables, "
                "or configure environment.llm.base_url and environment.llm.model in config.yaml"
            )

        max_attempts = self.target_transport_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                response_text = self._call_remote_environment(candidate_text)
            except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary
                policy_code = _provider_policy_block_code(
                    exc, self.policy_block_error_codes
                )
                if policy_code is not None:
                    logging.getLogger(__name__).warning(
                        "Target provider rejected a candidate by policy (%s).",
                        policy_code,
                    )
                    return {
                        "response_text": "",
                        "style": "provider_policy_block",
                        "candidate_text": candidate_text,
                        "backend": "llm",
                        "model_name": self.cfg.model,
                        "blocked_by_guardrail": True,
                        "policy_block_code": policy_code,
                        "attempts": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }

                retryable = _is_retryable_target_error(exc)
                if retryable and attempt < max_attempts:
                    delay = self.target_retry_backoff_seconds * (2 ** (attempt - 1))
                    logging.getLogger(__name__).warning(
                        "Target transport call failed with %s (attempt %d/%d); "
                        "retrying in %.1fs.",
                        type(exc).__name__,
                        attempt,
                        max_attempts,
                        delay,
                    )
                    if delay:
                        time.sleep(delay)
                    continue

                if retryable:
                    message = (
                        f"Target transport call failed with {type(exc).__name__} "
                        f"after {attempt} attempts; marking it unscored."
                    )
                else:
                    message = (
                        f"Target environment call failed with {type(exc).__name__}; "
                        "marking it unscored."
                    )
                logging.getLogger(__name__).warning(message)
                return {
                    "response_text": "",
                    "style": "target_execution_error",
                    "candidate_text": candidate_text,
                    "backend": "llm",
                    "model_name": self.cfg.model,
                    "target_error": True,
                    "not_scored": True,
                    "attempts": attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

            return {
                "response_text": response_text,
                "style": self._classify_style(candidate_text, response_text),
                "candidate_text": candidate_text,
                "backend": "llm",
                "model_name": self.cfg.model,
                "attempts": attempt,
            }

        raise AssertionError("unreachable target transport retry loop")

    def _call_remote_environment(self, candidate_text: str) -> str:
        """Send one harmless candidate to the remote target model."""
        extra = self._build_sampling_params() if self.send_sampling_params else {}
        if self.disable_thinking:
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return call_chat_completion(
            self.cfg,
            messages=self._build_messages(candidate_text),
            extra_params=extra,
            context="Environment",
            # Disable hidden SDK retries. The application loop above owns the
            # bounded transport retry policy and records every actual attempt.
            sdk_max_retries=0,
        )

    def _build_messages(self, candidate_text: str) -> list[dict[str, str]]:
        """Build OpenAI-compatible chat messages for the target model call."""
        messages: list[dict[str, str]] = []
        if self.include_system_prompt and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": candidate_text})
        return messages

    def _build_sampling_params(self) -> dict[str, object]:
        """Build optional sampling params for environments that accept them."""
        if not self.send_sampling_params:
            return {}
        params: dict[str, object] = {
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.send_temperature:
            params["temperature"] = self.cfg.temperature
        if self.cfg.top_p != 1.0:
            params["top_p"] = self.cfg.top_p
        return params

    def _classify_style(self, candidate_text: str, response_text: str) -> str:
        """Classify response style so the evaluator can keep using simple heuristics."""
        lowered_response = response_text.lower()
        lowered_candidate = candidate_text.lower()
        if any(marker in lowered_response for marker in ("cannot", "decline", "unsafe", "out of scope", "refuse")):
            return "refusal"
        if "{" in response_text and "}" in response_text:
            return "structured"
        if len(response_text) < 80:
            return "brief"
        if any(token in lowered_candidate for token in ("teacher", "guide", "friendly", "explain")):
            return "helpful"
        return "echo"


def build_environment(
    target_profile: dict[str, object],
    config: dict[str, Any] | None = None,
) -> MockEnvironment | OpenAICompatibleEnvironment:
    """Construct the configured environment backend."""
    config = dict(config or {})
    backend = str(config.get("backend", "mock"))

    if backend == "llm":
        llm_config = dict(config.get("llm", {}))
        return OpenAICompatibleEnvironment(
            target_profile=target_profile,
            config=llm_config,
        )
    return MockEnvironment(target_profile=target_profile)
