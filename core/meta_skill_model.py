"""Shared OpenAI-compatible helper for model-backed meta-skills.

Raises RuntimeError if the meta-skill LLM backend is unavailable or fails.
"""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator

from core.llm_config import LLMClientConfig, call_chat_completion
from core.skill_runtime import extract_json_object


class MetaArtifactSchemaError(RuntimeError):
    """Raised when a model response does not satisfy a supplied JSON Schema."""

    def __init__(
        self,
        errors: list[str],
        *,
        response_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.errors = errors
        self.response_diagnostics = dict(response_diagnostics or {})
        super().__init__(
            _response_error_message(errors, self.response_diagnostics)
        )


class MetaArtifactResponseError(RuntimeError):
    """Raised for retryable malformed model output, not transport/config failures."""

    def __init__(
        self,
        errors: list[str],
        *,
        response_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.errors = errors
        self.response_diagnostics = dict(response_diagnostics or {})
        super().__init__(
            _response_error_message(errors, self.response_diagnostics)
        )


def generate_meta_artifact(
    *,
    backend_config: dict[str, Any],
    system_prompt: str,
    user_payload: dict[str, Any],
    allow_unescaped_control_chars: bool = False,
    response_schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Generate one structured meta-skill artifact from a remote model.

    Raises RuntimeError if backend is disabled, misconfigured, or fails.

    Returns:
        tuple: (artifacts, rationale, metadata)
    """
    cfg = LLMClientConfig.from_dict(
        backend_config,
        defaults={"timeout_seconds": 60, "temperature": 0.4, "max_tokens": 8192},
    )

    if not cfg.enabled:
        raise RuntimeError(
            "Meta-skill backend is disabled. "
            "Set meta_skills.llm.enabled to true in config.yaml"
        )

    cfg.validate_endpoint()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    raw_extra_body = backend_config.get("extra_body")
    extra_params = dict(raw_extra_body) if isinstance(raw_extra_body, dict) else {}
    if bool(backend_config.get("disable_thinking", False)):
        chat_template_kwargs = dict(extra_params.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = False
        extra_params["chat_template_kwargs"] = chat_template_kwargs
    response_diagnostics: dict[str, Any] = {}
    content = call_chat_completion(
        cfg,
        messages=messages,
        extra_params=extra_params or None,
        context="Meta-skill backend",
        diagnostics_out=response_diagnostics,
    )

    if not content or not content.strip():
        raise MetaArtifactResponseError(
            ["Meta-skill backend returned empty response"],
            response_diagnostics=response_diagnostics,
        )

    try:
        json_str = extract_json_object(content)
        parsed = json.loads(json_str, strict=not allow_unescaped_control_chars)
    except (RuntimeError, json.JSONDecodeError) as exc:
        parse_detail = type(exc).__name__
        if isinstance(exc, json.JSONDecodeError):
            parse_detail += (
                f" ({exc.msg}) at line {exc.lineno}, column {exc.colno}"
            )
        raise MetaArtifactResponseError(
            [
                "Meta-skill JSON parse failed (possibly truncated at max_tokens). "
                f"Response length={len(content)}, parser={parse_detail}"
            ],
            response_diagnostics=response_diagnostics,
        ) from exc

    if not isinstance(parsed, dict):
        raise MetaArtifactResponseError(
            ["Meta-skill backend must return one JSON object"],
            response_diagnostics=response_diagnostics,
        )

    if response_schema is not None:
        validation_errors = sorted(
            Draft202012Validator(response_schema).iter_errors(parsed),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if validation_errors:
            messages = []
            for error in validation_errors:
                path = ".".join(str(part) for part in error.absolute_path) or "$"
                messages.append(_safe_schema_error(path=path, error=error))
            raise MetaArtifactSchemaError(
                messages,
                response_diagnostics=response_diagnostics,
            )

    artifacts = dict(parsed.get("artifacts", {}))
    rationale = str(parsed.get("rationale", "")).strip()

    # Fallback: some OpenAI-compatible backends flatten the requested
    # ``artifacts`` object and return its fields at the top level.  Recognize
    # several artifact families instead of relying on strategy_prompt, which is
    # legitimately absent from deterministic skills.
    artifact_hint_keys = {
        "skill_name",
        "skill_mode",
        "strategy_prompt",
        "wrap_function_code",
        "technique_doc",
        "mechanism",
    }
    if not artifacts and artifact_hint_keys.intersection(parsed):
        rationale = rationale or str(parsed.pop("rationale", "")).strip()
        parsed.pop("rationale", None)
        artifacts = parsed

    if not artifacts:
        raise MetaArtifactResponseError(
            ["Meta-skill backend returned empty artifacts"],
            response_diagnostics=response_diagnostics,
        )

    return (
        artifacts,
        rationale,
        {
            "backend": "openai_compatible",
            "model": cfg.model,
            "response_diagnostics": response_diagnostics,
        },
    )


def _response_error_message(
    errors: list[str],
    response_diagnostics: dict[str, Any],
) -> str:
    """Format an exception with bounded telemetry but no model-authored text."""
    message = "; ".join(errors)
    if not response_diagnostics:
        return message
    serialized = json.dumps(
        response_diagnostics,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{message}; response_diagnostics={serialized}"


def _safe_schema_error(*, path: str, error: Any) -> str:
    """Describe a schema failure without echoing model-controlled values or keys."""
    validator = str(error.validator or "constraint")
    if validator == "required":
        expected = [str(value) for value in error.validator_value]
        present = set(error.instance) if isinstance(error.instance, dict) else set()
        missing = [value for value in expected if value not in present]
        detail = "missing required field(s): " + ", ".join(missing)
    elif validator == "additionalProperties":
        detail = "contains fields not allowed by the schema"
    elif validator == "type":
        detail = f"must have type {error.validator_value}"
    elif validator in {"enum", "const"}:
        detail = f"must satisfy the {validator} constraint"
    elif validator == "pattern":
        detail = "must match the required pattern"
    elif validator in {
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
    }:
        detail = f"violates {validator}={error.validator_value}"
    else:
        detail = f"violates schema constraint {validator}"
    return f"{path}: {detail}"
