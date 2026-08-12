"""Shared LLM client configuration and call dispatch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from core.skill_runtime import (
    extract_content,
    extract_response_diagnostics,
    post_chat_completion,
)


@dataclass(frozen=True, slots=True)
class LLMClientConfig:
    """Parsed LLM endpoint configuration."""

    base_url: str
    model: str
    api_key: str
    timeout_seconds: int = 12
    temperature: float = 0.9
    max_tokens: int = 1000
    top_p: float = 1.0
    transport: str = "http"
    enabled: bool = False
    send_temperature: bool = True

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        defaults: dict[str, Any] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> "LLMClientConfig":
        """Build config from a raw dictionary.

        Args:
            raw: The raw config dict.
            defaults: Field-level defaults that override class defaults.
            env_overrides: Mapping of field_name -> ENV_VAR_NAME for env lookups.
        """
        d = defaults or {}
        env = env_overrides or {}

        def _str_field(field: str, fallback: str = "") -> str:
            env_var = env.get(field)
            if env_var:
                env_value = os.getenv(env_var)
                if env_value:
                    return env_value
            return str(raw.get(field, d.get(field, fallback)))

        return cls(
            base_url=_str_field("base_url").rstrip("/"),
            model=_str_field("model"),
            api_key=_str_field("api_key"),
            timeout_seconds=int(raw.get("timeout_seconds", d.get("timeout_seconds", 12))),
            temperature=float(raw.get("temperature", d.get("temperature", 0.9))),
            max_tokens=int(raw.get("max_tokens", d.get("max_tokens", 1000))),
            top_p=float(raw.get("top_p", d.get("top_p", 1.0))),
            transport=str(raw.get("transport", d.get("transport", "http"))),
            enabled=bool(raw.get("enabled", d.get("enabled", False))),
            send_temperature=bool(raw.get("send_temperature", d.get("send_temperature", True))),
        )

    def validate_endpoint(self) -> None:
        """Raise RuntimeError if base_url or model is empty."""
        if not self.base_url or not self.model:
            raise RuntimeError(
                f"LLM endpoint requires base_url and model; "
                f"got base_url={self.base_url!r}, model={self.model!r}"
            )


def call_chat_completion(
    config: LLMClientConfig,
    *,
    messages: list[dict[str, Any]],
    extra_params: dict[str, Any] | None = None,
    context: str = "API",
    sdk_max_retries: int | None = None,
    diagnostics_out: dict[str, Any] | None = None,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint.

    Dispatches to the OpenAI SDK or raw HTTP based on config.transport.
    Returns the extracted text content from the response.
    """
    params = dict(extra_params) if extra_params else {}

    if config.transport == "openai_sdk":
        return _call_sdk(
            config,
            messages=messages,
            extra_params=params,
            max_retries=sdk_max_retries,
            diagnostics_out=diagnostics_out,
        )

    body: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_tokens": config.max_tokens,
    }
    if config.send_temperature:
        body["temperature"] = config.temperature
    if config.top_p != 1.0:
        body["top_p"] = config.top_p
    body.update(params)

    payload = post_chat_completion(
        base_url=config.base_url,
        body=body,
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        context=context,
    )
    if diagnostics_out is not None:
        diagnostics_out.update(extract_response_diagnostics(payload))
    return extract_content(payload)


def _call_sdk(
    config: LLMClientConfig,
    *,
    messages: list[dict[str, Any]],
    extra_params: dict[str, Any],
    max_retries: int | None,
    diagnostics_out: dict[str, Any] | None = None,
) -> str:
    """Internal: call via OpenAI Python SDK."""
    if OpenAI is None:
        raise RuntimeError("openai package is not installed for openai_sdk transport.")
    client_kwargs: dict[str, Any] = {
        "base_url": config.base_url,
        "api_key": config.api_key,
        "timeout": config.timeout_seconds,
    }
    if max_retries is not None:
        client_kwargs["max_retries"] = max(0, int(max_retries))
    client = OpenAI(**client_kwargs)
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_tokens": config.max_tokens,
    }
    if config.send_temperature:
        kwargs["temperature"] = config.temperature
    if config.top_p != 1.0:
        kwargs["top_p"] = config.top_p
    kwargs.update(extra_params)
    completion = client.chat.completions.create(**kwargs)
    payload = completion.model_dump()
    if diagnostics_out is not None:
        diagnostics_out.update(extract_response_diagnostics(payload))
    return extract_content(payload)
