"""OpenAI-compatible embedding client with persistent vector caching."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import struct
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib import error as urllib_error
from urllib import request as urllib_request


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_MAX_INPUT_TOKENS = 8191
DEFAULT_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBEDDING_CACHE_PATH = "data/external_embedding_cache.sqlite3"
DEFAULT_EMBEDDING_MAX_RETRIES = 6
DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS = 1.0


class EmbeddingClientError(RuntimeError):
    """Raised when embeddings cannot be generated or validated."""


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for the external-text embedding backend."""

    enabled: bool = True
    base_url: str = DEFAULT_EMBEDDING_BASE_URL
    model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    api_key: str = ""
    timeout_seconds: int = 60
    batch_size: int = 32
    max_batch_tokens: int = 100_000
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    cache_path: str = DEFAULT_EMBEDDING_CACHE_PATH
    max_retries: int = DEFAULT_EMBEDDING_MAX_RETRIES
    retry_backoff_seconds: float = DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any] | None,
        *,
        project_root: Path | None = None,
    ) -> "EmbeddingConfig":
        payload = dict(raw or {})
        base_url = str(
            os.getenv("EMBEDDING_BASE_URL")
            or payload.get("base_url")
            or DEFAULT_EMBEDDING_BASE_URL
        ).rstrip("/")
        model = str(
            os.getenv("EMBEDDING_MODEL")
            or payload.get("model")
            or DEFAULT_EMBEDDING_MODEL
        )
        dimensions = int(
            os.getenv("EMBEDDING_DIMENSIONS")
            or payload.get("dimensions", DEFAULT_EMBEDDING_DIMENSIONS)
        )
        api_key = str(
            os.getenv("EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("API_KEY")
            or payload.get("api_key", "")
        )
        if api_key.startswith("${") and api_key.endswith("}"):
            api_key = ""
        cache_path = str(payload.get("cache_path") or DEFAULT_EMBEDDING_CACHE_PATH)
        if project_root is not None and cache_path and not Path(cache_path).is_absolute():
            cache_path = str(project_root / cache_path)
        config = cls(
            enabled=bool(payload.get("enabled", True)),
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            timeout_seconds=int(
                os.getenv("EMBEDDING_TIMEOUT_SECONDS")
                or payload.get("timeout_seconds", 60)
            ),
            batch_size=int(payload.get("batch_size", 32)),
            max_batch_tokens=int(payload.get("max_batch_tokens", 100_000)),
            max_input_tokens=int(payload.get("max_input_tokens", DEFAULT_MAX_INPUT_TOKENS)),
            cache_path=cache_path,
            max_retries=int(
                os.getenv("EMBEDDING_MAX_RETRIES")
                or payload.get("max_retries", DEFAULT_EMBEDDING_MAX_RETRIES)
            ),
            retry_backoff_seconds=float(
                os.getenv("EMBEDDING_RETRY_BACKOFF_SECONDS")
                or payload.get(
                    "retry_backoff_seconds", DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS
                )
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.enabled:
            raise EmbeddingClientError("Embedding backend is disabled")
        if not self.base_url:
            raise EmbeddingClientError("Embedding backend requires base_url")
        if not self.model:
            raise EmbeddingClientError("Embedding backend requires model")
        if "api.openai.com" in self.base_url and not self.api_key:
            raise EmbeddingClientError(
                "OpenAI embedding endpoint requires EMBEDDING_API_KEY, OPENAI_API_KEY, or API_KEY"
            )
        if self.dimensions <= 0:
            raise EmbeddingClientError("Embedding dimensions must be positive")
        if self.timeout_seconds <= 0:
            raise EmbeddingClientError("Embedding timeout must be positive")
        if self.batch_size <= 0:
            raise EmbeddingClientError("Embedding batch_size must be positive")
        if self.max_batch_tokens <= 0 or self.max_input_tokens <= 0:
            raise EmbeddingClientError("Embedding token limits must be positive")
        if self.max_retries <= 0 or self.retry_backoff_seconds < 0:
            raise EmbeddingClientError("Embedding retry settings must be non-negative")


class EmbeddingCache:
    """SQLite cache keyed by endpoint, model, dimensions, and normalized text."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    endpoint_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    text_hash TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (endpoint_hash, model, dimensions, text_hash)
                )
                """
            )

    def get_many(
        self,
        *,
        endpoint_hash: str,
        model: str,
        dimensions: int,
        text_hashes: Iterable[str],
    ) -> dict[str, list[float]]:
        hashes = list(dict.fromkeys(text_hashes))
        if not hashes:
            return {}
        found: dict[str, list[float]] = {}
        with self._connect() as connection:
            for start in range(0, len(hashes), 500):
                batch = hashes[start:start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"""
                    SELECT text_hash, vector FROM embeddings
                    WHERE endpoint_hash = ? AND model = ? AND dimensions = ?
                      AND text_hash IN ({placeholders})
                    """,
                    [endpoint_hash, model, dimensions, *batch],
                ).fetchall()
                for text_hash, blob in rows:
                    vector = list(struct.unpack(f"<{dimensions}f", blob))
                    found[str(text_hash)] = normalize_vector(vector)
        return found

    def put_many(
        self,
        *,
        endpoint_hash: str,
        model: str,
        dimensions: int,
        records: Iterable[tuple[str, int, list[float]]],
    ) -> None:
        rows = []
        for text_hash, token_count, vector in records:
            blob = struct.pack(f"<{dimensions}f", *vector)
            rows.append((endpoint_hash, model, dimensions, text_hash, token_count, blob))
        if not rows:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                    (endpoint_hash, model, dimensions, text_hash, token_count, vector)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


class EmbeddingClient:
    """Generate normalized embeddings through an OpenAI-compatible endpoint."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.config.validate()
        self.endpoint_hash = hashlib.sha256(config.base_url.encode("utf-8")).hexdigest()[:16]
        self.cache = EmbeddingCache(Path(config.cache_path)) if config.cache_path else None
        self._encoding: Any | None = None
        self._encoding_loaded = False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        prepared = [self._prepare_text(text) for text in texts]
        text_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text, _ in prepared]
        cached = (
            self.cache.get_many(
                endpoint_hash=self.endpoint_hash,
                model=self.config.model,
                dimensions=self.config.dimensions,
                text_hashes=text_hashes,
            )
            if self.cache is not None
            else {}
        )

        vectors_by_hash = dict(cached)
        missing: list[tuple[str, str, int]] = []
        seen_missing: set[str] = set()
        for (text, token_count), text_hash in zip(prepared, text_hashes):
            if text_hash not in vectors_by_hash and text_hash not in seen_missing:
                missing.append((text_hash, text, token_count))
                seen_missing.add(text_hash)

        for batch in self._batches(missing):
            batch_vectors = self._request_embeddings([item[1] for item in batch])
            cache_records: list[tuple[str, int, list[float]]] = []
            for (text_hash, _text, token_count), vector in zip(batch, batch_vectors):
                vectors_by_hash[text_hash] = vector
                cache_records.append((text_hash, token_count, vector))
            if self.cache is not None:
                self.cache.put_many(
                    endpoint_hash=self.endpoint_hash,
                    model=self.config.model,
                    dimensions=self.config.dimensions,
                    records=cache_records,
                )

        try:
            return [vectors_by_hash[text_hash] for text_hash in text_hashes]
        except KeyError as exc:  # pragma: no cover - defensive consistency check
            raise EmbeddingClientError("Embedding response did not cover all inputs") from exc

    def _prepare_text(self, text: str) -> tuple[str, int]:
        normalized = normalize_embedding_text(text)
        if not normalized:
            raise EmbeddingClientError("Embedding input cannot be empty")
        tokens = self._encode(normalized)
        if tokens is not None:
            if len(tokens) > self.config.max_input_tokens:
                tokens = tokens[: self.config.max_input_tokens]
                normalized = self._decode(tokens)
            return normalized, len(tokens)

        # Conservative fallback used only until the declared tiktoken dependency is installed.
        max_chars = max(1, self.config.max_input_tokens // 4)
        normalized = normalized[:max_chars]
        return normalized, min(len(normalized) * 4, self.config.max_input_tokens)

    def _load_encoding(self) -> Any | None:
        if self._encoding_loaded:
            return self._encoding
        self._encoding_loaded = True
        try:
            import tiktoken  # type: ignore[import-not-found]

            try:
                self._encoding = tiktoken.encoding_for_model(self.config.model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self._encoding = None
        return self._encoding

    def _encode(self, text: str) -> list[int] | None:
        encoding = self._load_encoding()
        return (
            list(encoding.encode(text, disallowed_special=()))
            if encoding is not None
            else None
        )

    def _decode(self, tokens: list[int]) -> str:
        encoding = self._load_encoding()
        if encoding is None:  # pragma: no cover - paired with _encode
            return ""
        return str(encoding.decode(tokens))

    def _batches(
        self,
        records: list[tuple[str, str, int]],
    ) -> Iterable[list[tuple[str, str, int]]]:
        batch: list[tuple[str, str, int]] = []
        token_total = 0
        for record in records:
            token_count = record[2]
            if batch and (
                len(batch) >= self.config.batch_size
                or token_total + token_count > self.config.max_batch_tokens
            ):
                yield batch
                batch = []
                token_total = 0
            batch.append(record)
            token_total += token_count
        if batch:
            yield batch

    def _request_embeddings(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps(
            {
                "model": self.config.model,
                "input": inputs,
                "dimensions": self.config.dimensions,
                "encoding_format": "float",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib_request.Request(
            f"{self.config.base_url}/embeddings",
            data=body,
            headers=headers,
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with urllib_request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return self._parse_response(payload, expected_count=len(inputs))
            except urllib_error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 and not 500 <= exc.code < 600:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                    raise EmbeddingClientError(
                        f"Embedding API HTTP {exc.code}: {detail}"
                    ) from exc
            except (urllib_error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        raise EmbeddingClientError(
            f"Embedding API failed after {self.config.max_retries} attempts: {last_error}"
        )

    def _parse_response(self, payload: Any, *, expected_count: int) -> list[list[float]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise EmbeddingClientError("Embedding API response missing data array")
        rows = payload["data"]
        if len(rows) != expected_count:
            raise EmbeddingClientError(
                f"Embedding API returned {len(rows)} vectors for {expected_count} inputs"
            )
        indexed: dict[int, list[float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise EmbeddingClientError("Embedding API returned a non-object row")
            index = int(row.get("index", -1))
            raw_vector = row.get("embedding")
            if index < 0 or not isinstance(raw_vector, list):
                raise EmbeddingClientError("Embedding API row is missing index or vector")
            vector = [float(value) for value in raw_vector]
            if len(vector) != self.config.dimensions:
                raise EmbeddingClientError(
                    f"Embedding vector has {len(vector)} dimensions; "
                    f"expected {self.config.dimensions}"
                )
            indexed[index] = normalize_vector(vector)
        if sorted(indexed) != list(range(expected_count)):
            raise EmbeddingClientError("Embedding API response indexes are incomplete or duplicated")
        return [indexed[index] for index in range(expected_count)]


def normalize_embedding_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding non-Latin text."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(normalized.split()).strip()


def normalize_vector(vector: list[float]) -> list[float]:
    if not vector or any(not math.isfinite(value) for value in vector):
        raise EmbeddingClientError("Embedding vector contains invalid values")
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise EmbeddingClientError("Embedding vector has zero or invalid norm")
    return [value / norm for value in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} != {len(b)}")
    return math.fsum(left * right for left, right in zip(a, b))


def similarity_clusters(
    vectors: list[list[float]],
    *,
    threshold: float,
) -> list[list[int]]:
    """Return deterministic union-find clusters for cosine-similar vectors."""
    if not 0 <= threshold <= 1:
        raise ValueError("similarity threshold must be between 0 and 1")
    parents = list(range(len(vectors)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            if cosine_similarity(vectors[left], vectors[right]) >= threshold:
                union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in range(len(vectors)):
        grouped.setdefault(find(index), []).append(index)
    return [grouped[root] for root in sorted(grouped)]
