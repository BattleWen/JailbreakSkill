"""Shared retry classification for LLM transport failures."""

from __future__ import annotations

from urllib.error import HTTPError, URLError

try:
    import httpx
except ImportError:  # pragma: no cover - installed with the OpenAI SDK
    httpx = None  # type: ignore[assignment]

try:
    from openai import APIConnectionError
except ImportError:  # pragma: no cover - optional dependency at runtime
    APIConnectionError = None  # type: ignore[assignment,misc]


_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 429})


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Return a cycle-safe cause/context chain, outermost first."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _coerce_http_status(value: object) -> int | None:
    """Normalize an HTTP status-like value without accepting booleans/callables."""
    if value is None or isinstance(value, bool) or callable(value):
        return None
    try:
        status_code = int(value)
    except (TypeError, ValueError):
        return None
    return status_code if 100 <= status_code <= 599 else None


def _http_status_code(exc: BaseException) -> int | None:
    """Extract status codes used by urllib, SDK, requests/httpx-style errors."""
    if isinstance(exc, HTTPError):
        return _coerce_http_status(exc.code)

    for attr in ("status_code", "http_status", "status"):
        status_code = _coerce_http_status(getattr(exc, attr, None))
        if status_code is not None:
            return status_code

    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            status_code = _coerce_http_status(getattr(response, attr, None))
            if status_code is not None:
                return status_code
    return None


def is_retryable_transport_error(exc: Exception) -> bool:
    """Return whether an LLM request failure is plausibly transient.

    An explicit HTTP status takes precedence over broad transport base classes.
    This matters for ``urllib.error.HTTPError``, which also inherits from
    ``URLError``: ordinary 4xx responses must not become network retries.
    """
    chain = _exception_chain(exc)

    # Inspect the entire wrapped exception chain first. A precise HTTP status
    # must win over a generic connection/URL-error superclass or wrapper.
    for current in chain:
        status_code = _http_status_code(current)
        if status_code is not None:
            return (
                status_code in _RETRYABLE_HTTP_STATUS_CODES
                or 500 <= status_code <= 599
            )

    transient_types: tuple[type[BaseException], ...] = (
        ConnectionError,
        TimeoutError,
        EOFError,
        URLError,
    )
    for current in chain:
        if isinstance(current, transient_types):
            return True
        if APIConnectionError is not None and isinstance(current, APIConnectionError):
            return True
        if httpx is not None and isinstance(current, httpx.TransportError):
            return True
    return False
