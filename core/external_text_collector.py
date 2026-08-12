"""Collect and normalize public external text for skill generation."""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import io
import json
import os
import re
import socket
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from core.meta_skill_model import generate_meta_artifact
from core.skill_loader import SkillLoader


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _url_origin(url: str) -> tuple[str, str, int | None]:
    parsed = urllib_parse.urlsplit(url)
    scheme = parsed.scheme.casefold()
    default_port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return scheme, (parsed.hostname or "").casefold(), parsed.port or default_port


class UnsafeExternalUrlError(ValueError):
    """Raised when a crawl target is not a public HTTP(S) destination."""


class PaperBundleValidationError(ValueError):
    """Raised when an exact paper bundle cannot be verified without ambiguity."""


_ARXIV_ID_PATTERN = re.compile(
    r"(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)


def normalize_arxiv_id(value: str) -> str:
    """Return a version-free canonical arXiv identifier or fail closed."""
    raw = urllib_parse.unquote(str(value).strip())
    if not raw:
        raise PaperBundleValidationError("arxiv_id_required")
    raw = re.sub(r"(?i)^arxiv:\s*", "", raw).strip()
    parsed = urllib_parse.urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme.casefold() not in {"http", "https"}:
            raise PaperBundleValidationError("invalid_arxiv_id")
        hostname = (parsed.hostname or "").rstrip(".").casefold()
        if hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            raise PaperBundleValidationError("invalid_arxiv_host")
        path = parsed.path.strip("/")
        for prefix in ("abs/", "html/", "pdf/"):
            if path.casefold().startswith(prefix):
                path = path[len(prefix) :]
                break
        raw = path
    raw = re.sub(r"(?i)\.pdf$", "", raw).strip("/")
    if not _ARXIV_ID_PATTERN.fullmatch(raw):
        raise PaperBundleValidationError("invalid_arxiv_id")
    return re.sub(r"(?i)v\d+$", "", raw)


def normalize_github_repo(value: str) -> str:
    """Normalize an explicit GitHub ``owner/repository`` identifier."""
    raw = urllib_parse.unquote(str(value).strip())
    parsed = urllib_parse.urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme.casefold() not in {"http", "https"} or (
            parsed.hostname or ""
        ).rstrip(".").casefold() not in {"github.com", "www.github.com"}:
            raise PaperBundleValidationError("invalid_github_repo")
        raw = parsed.path.strip("/")
    raw = re.sub(r"(?i)\.git$", "", raw).strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", raw):
        raise PaperBundleValidationError("invalid_github_repo")
    return raw


def normalize_huggingface_dataset_id(value: str) -> str:
    """Normalize an explicit Hugging Face ``owner/dataset`` identifier."""
    raw = urllib_parse.unquote(str(value).strip())
    parsed = urllib_parse.urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme.casefold() not in {"http", "https"} or (
            parsed.hostname or ""
        ).rstrip(".").casefold() not in {"huggingface.co", "www.huggingface.co"}:
            raise PaperBundleValidationError("invalid_huggingface_dataset")
        path = parsed.path.strip("/")
        if not path.casefold().startswith("datasets/"):
            raise PaperBundleValidationError("invalid_huggingface_dataset")
        raw = path[len("datasets/") :]
    raw = raw.strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", raw):
        raise PaperBundleValidationError("invalid_huggingface_dataset")
    return raw


_NON_PUBLIC_HOST_SUFFIXES = (
    ".internal",
    ".intranet",
    ".lan",
    ".local",
    ".localhost",
)
_NON_PUBLIC_HOSTNAMES = frozenset(
    {"internal", "intranet", "lan", "local", "localhost"}
)


def _is_public_ip_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return bool(address.is_global and not address.is_multicast)


def _ensure_public_http_url(url: str, *, resolve_dns: bool) -> str:
    """Validate a public-web URL before fetching it or following a redirect."""
    try:
        parsed = urllib_parse.urlsplit(str(url).strip())
        port = parsed.port
    except ValueError as exc:
        raise UnsafeExternalUrlError("invalid external URL") from exc
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise UnsafeExternalUrlError("external URL must use HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeExternalUrlError("external URL must not contain userinfo")
    hostname = (parsed.hostname or "").rstrip(".").casefold()
    if not hostname:
        raise UnsafeExternalUrlError("external URL is missing a hostname")
    if hostname in _NON_PUBLIC_HOSTNAMES or hostname.endswith(
        _NON_PUBLIC_HOST_SUFFIXES
    ):
        raise UnsafeExternalUrlError("external URL hostname is not public")

    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        if not _is_public_ip_address(str(literal_ip)):
            raise UnsafeExternalUrlError("external URL IP address is not public")
        return parsed.geturl()

    if resolve_dns:
        try:
            addresses = socket.getaddrinfo(
                hostname,
                port or (443 if parsed.scheme.casefold() == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise UnsafeExternalUrlError(
                "external URL hostname could not be resolved"
            ) from exc
        resolved_ips = {
            str(address[4][0]).split("%", 1)[0]
            for address in addresses
            if len(address) >= 5 and address[4]
        }
        if not resolved_ips or any(
            not _is_public_ip_address(address)
            for address in resolved_ips
        ):
            raise UnsafeExternalUrlError(
                "external URL hostname resolved to a non-public address"
            )
    return parsed.geturl()


class _SameOriginAuthorizationRedirectHandler(urllib_request.HTTPRedirectHandler):
    """Never forward source credentials when an HTTP redirect changes origin."""

    _SENSITIVE_HEADERS = frozenset(
        {"authorization", "x-goog-api-key", "x-api-key"}
    )

    def __init__(self, *, resolve_dns: bool = True) -> None:
        super().__init__()
        self.resolve_dns = resolve_dns

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _ensure_public_http_url(newurl, resolve_dns=self.resolve_dns)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and _url_origin(req.full_url) != _url_origin(newurl):
            redirected.headers = {
                key: value
                for key, value in redirected.headers.items()
                if key.casefold() not in self._SENSITIVE_HEADERS
            }
            redirected.unredirected_hdrs = {
                key: value
                for key, value in redirected.unredirected_hdrs.items()
                if key.casefold() not in self._SENSITIVE_HEADERS
            }
        return redirected


class _SameOriginOnlyRedirectHandler(_SameOriginAuthorizationRedirectHandler):
    """Reject cross-origin redirects for requests carrying query credentials."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _ensure_public_http_url(newurl, resolve_dns=self.resolve_dns)
        if _url_origin(req.full_url) != _url_origin(newurl):
            raise UnsafeExternalUrlError(
                "credentialed external request cannot redirect across origins"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


DEFAULT_CRAWL_DIR = Path("data") / "external_crawls"
DEFAULT_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE_SEARCH_ENDPOINT = "https://lite.duckduckgo.com/lite/"
GOOGLE_GROUNDING_MODEL = "gemini-3-flash-preview"
GOOGLE_GROUNDING_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GOOGLE_GROUNDING_MODEL}:generateContent"
)
SERPAPI_SEARCH_ENDPOINT = "https://serpapi.com/search.json"
BOCHA_SEARCH_BASE_URL = "https://api.bochaai.com/v1"
EXTERNAL_DEFAULT_SOURCES = ("arxiv", "github", "huggingface", "google")
EXTERNAL_ALL_SOURCES = EXTERNAL_DEFAULT_SOURCES
DEFAULT_EXTERNAL_QUERY_PROFILE = "ai-safety-text"
ALTERNATIVE_REWRITE_QUERY_PROFILE = "ai-safety-rewrite-alt"
REPOSITORY_REWRITE_QUERY_PROFILE = "ai-safety-rewrite-redteam"
DEFAULT_MAX_SOURCE_AGE_DAYS = 3 * 365
MAX_PAPER_DISCOVERY_LIMIT = 50
MAX_GOOGLE_COMPANION_CANDIDATES_PER_SOURCE = 3
MAX_PAPER_COMPANION_CANDIDATES_PER_SOURCE = 3
MAX_TOTAL_COMPANION_CANDIDATES_PER_SOURCE = 5
MAX_PAPER_CITATION_CANDIDATES_PER_PRIMARY = 5
MAX_PAPER_CITATION_EXPANSION_CANDIDATES = 5
MAX_PAPER_CITATION_DOMAIN_BRIDGES = 1
PAPER_SELECTION_POLICY = "mechanism_suitability_v4"
PAPER_SELECTION_SCHEMA_VERSION = 6
_LEGACY_PAPER_SELECTION_CONTRACTS = frozenset(
    {
        ("evidence_tiered_v1", 3),
        ("example_grounded_v2", 4),
        ("example_grounded_v3", 5),
    }
)


def supports_paper_selection_contract(policy: str, schema_version: int) -> bool:
    """Accept the current manifest contract and explicitly supported snapshots."""
    normalized = str(policy).strip().casefold()
    try:
        version = int(schema_version)
    except (TypeError, ValueError):
        return False
    return (normalized, version) in {
        (PAPER_SELECTION_POLICY.casefold(), PAPER_SELECTION_SCHEMA_VERSION),
        *_LEGACY_PAPER_SELECTION_CONTRACTS,
    }


_GITHUB_API_VERSION = "2026-03-10"
_GITHUB_ANONYMOUS_API_TIMEOUT_SECONDS = 10
_GITHUB_ANONYMOUS_README_TIMEOUT_SECONDS = 8
_GITHUB_README_TIMEOUT_SECONDS = 15
_GITHUB_README_FALLBACK_FILENAMES = (
    "README.md",
    "README.MD",
    "Readme.md",
    "README.rst",
    "README.txt",
)
_GITHUB_ARXIV_DISCOVERY_TIMEOUT_SECONDS = 10
_GITHUB_EVIDENCE_MAX_FILES = 8
_GITHUB_EVIDENCE_MAX_FILE_BYTES = 120_000
_GITHUB_EVIDENCE_MAX_FILE_CHARS = 20_000
_GITHUB_EVIDENCE_MAX_TOTAL_CHARS = 60_000
_GITHUB_EVIDENCE_SUFFIXES = frozenset(
    {".md", ".mdx", ".rst", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".csv", ".py"}
)
_GITHUB_EVIDENCE_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".github",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "vendor",
    }
)
_EVIDENCE_ROLE_MARKERS: dict[str, tuple[str, ...]] = {
    "mechanism": (
        "method",
        "mechanism",
        "technique",
        "attack",
        "algorithm",
        "approach",
    ),
    "implementation": (
        "implementation",
        "config",
        "template",
        "prompt",
        "schema",
        "pipeline",
        "usage",
    ),
    "examples": ("example", "examples", "demo", "sample", "case-study"),
    "evaluation": (
        "evaluation",
        "eval",
        "benchmark",
        "experiment",
        "result",
        "ablation",
        "metric",
    ),
    "domain-evidence": (
        "risk",
        "harm",
        "safety",
        "domain",
        "category",
        "defamation",
        "misinformation",
    ),
    "dataset-schema": ("dataset schema", "features", "column", "split", "config"),
    "limitations": ("limitation", "limitations", "failure mode", "caveat"),
}
_HUGGINGFACE_MAX_CARD_CHARS = 50_000
_HUGGINGFACE_MAX_PREVIEW_CHARS = 12_000
_HUGGINGFACE_MAX_TOTAL_CHARS = 70_000
_HUGGINGFACE_PREVIEW_ROWS = 6
_HUGGINGFACE_VIEWER_TIMEOUT_SECONDS = 5
_HUGGINGFACE_HUB_TIMEOUT_SECONDS = 8
_HUGGINGFACE_RAW_MAX_BYTES = 256 * 1024
_HUGGINGFACE_RAW_MAX_FILES = 3
_HUGGINGFACE_TREE_MAX_ENTRIES = 2_000
_HUGGINGFACE_TREE_MAX_PAGES = 4
_HUGGINGFACE_RAW_SUFFIXES = (".jsonl", ".ndjson", ".json", ".csv", ".tsv")
_ARXIV_PDF_MAX_BYTES = 20 * 1024 * 1024
_ARXIV_PDF_MAX_PAGES = 100
_ARXIV_PDF_MAX_EXTRACTED_CHARS = 200_000
_GITHUB_AGGREGATOR_NAME_PATTERN = re.compile(
    r"(?:^|[-_/\s])(?:awesome|list|digest|roadmap|interviews?|daily)(?:$|[-_/\s])",
    re.IGNORECASE,
)
_GITHUB_AGGREGATOR_DESCRIPTION_PATTERN = re.compile(
    r"\b(?:awesome|curated|maintained|comprehensive)\s+(?:list|collection)\b|"
    r"\b(?:list|collection|index|catalog(?:ue)?)\s+of\s+"
    r"(?:papers?|resources?|links?|repos(?:itories)?|tools?)\b|"
    r"\b(?:reading|paper|resource|link)\s+list\b|"
    r"\bdaily\s+(?:papers?|digest)\b",
    re.IGNORECASE,
)
_GITHUB_SECURITY_ANCHORS = (
    "jailbreak",
    "adversarial prompt",
    "red team",
    "red-team",
    "red teaming",
    "red-teaming",
    "prompt attack",
    "refusal robustness",
    "harmful behavior",
    "content safety",
    "misinformation",
    "defamation",
    "targeted harm",
)
_GITHUB_METHOD_MARKERS = (
    "method",
    "attack",
    "algorithm",
    "pipeline",
    "optimization",
    "implementation",
    "reproduce",
    "reproduction",
    "usage",
    "quick start",
    "evaluation",
    "benchmark",
    "approach",
    "technique",
    "mechanism",
    "improv",
    "using",
)
_GITHUB_PUBLICATION_MARKERS = (
    "arxiv.org",
    "doi.org",
    "bibtex",
    "citation",
    "@article",
    "@inproceedings",
)
_GITHUB_NON_ATTACK_REPO_PATTERN = re.compile(
    r"\b(?:defen[cs]e|defensive|guardrail|sanitiz(?:er|ation)?|detector|"
    r"security auditor|defensive proxy|mitigation|input validation)\b",
    re.IGNORECASE,
)
_GITHUB_UNSUPPORTED_SCOPE_PATTERN = re.compile(
    r"\b(?:multi[- ]turn|multimodal|multi-modal|vision-language|tool[- ]use|"
    r"tool chain|agentic attack|agent attack)\b",
    re.IGNORECASE,
)
_GITHUB_TRANSFORMATION_MARKERS = (
    "generate",
    "rewrite",
    "construct",
    "craft",
    "optimize",
    "append",
    "embed",
    "compose",
    "trigger",
    "payload",
    "template",
    "prompt",
    "insert",
    "select",
    "prefix",
    "prepend",
    "steer",
    "reinforce",
    "sample",
    "replace",
    "order",
    "condition",
)
_CYBER_ONLY_MARKERS = (
    "cve-",
    "malware",
    "credential theft",
    "privilege escalation",
    "remote code execution",
    "network intrusion",
    "endpoint security",
    "cloud misconfiguration",
    "sql injection",
)
_AI_MODEL_ANCHORS = (
    "llm",
    "language model",
    "generative ai",
    "model safety",
    "content safety",
    "prompt",
    "jailbreak",
)
_GITHUB_FAMILY_ANCHORS: dict[str, tuple[str, ...]] = {
    "reasoning-model": ("reasoning model", "large reasoning model", "lrm"),
    "schema-exploitation": (
        "schema jailbreak",
        "schema exploitation",
        "structured prompt",
    ),
    "trusted-document": (
        "paper summary attack",
        "trusted document",
        "authoritative document",
    ),
    "context-coherent": ("context coherent", "context-coherent", "semantic camouflage"),
    "single-turn-composition": (
        "single turn",
        "single-turn",
        "compositional jailbreak",
    ),
    "long-context": ("many shot", "many-shot", "long context", "long-context"),
    "reasoning-distraction": (
        "reasoning distraction",
        "chain of thought",
        "chain-of-thought",
    ),
    "adaptive-template": ("adaptive jailbreak", "adaptive attack", "prompt template"),
    "transfer-trigger": (
        "adversarial trigger",
        "query agnostic",
        "query-agnostic",
        "transferable",
    ),
    "automated-one-shot": (
        "automated jailbreak",
        "automatic jailbreak",
        "prompt optimization",
    ),
    "semantic-camouflage": (
        "semantic prompt",
        "semantic camouflage",
        "context coherent",
        "semantic jailbreak",
    ),
    "trusted-context": (
        "trusted document",
        "document context",
        "paper summary",
        "authoritative context",
    ),
    "reasoning-perturbation": (
        "reasoning prompt",
        "reasoning distraction",
        "chain of thought",
        "reasoning jailbreak",
    ),
    "compositional-rewrite": (
        "compositional prompt",
        "compositional jailbreak",
        "single turn",
        "single-turn",
    ),
    "structured-reframing": (
        "structured prompt",
        "schema jailbreak",
        "schema exploitation",
        "structured jailbreak",
    ),
    "policy-configuration": (
        "policy puppetry",
        "policy file",
        "policy configuration",
        "configuration prompt",
        "universal bypass",
    ),
    # Repository-derived rewrite families.  These are intentionally broader than
    # individual skill names: the downstream LLM must infer the concrete mechanism
    # and any risk-domain binding from README evidence instead of from the query.
    "representation-obfuscation": (
        "prompt obfuscation",
        "character-level",
        "character level",
        "encoding attack",
        "cipher prompt",
        "leetspeak",
    ),
    "multilingual-rewrite": (
        "multilingual jailbreak",
        "cross-lingual jailbreak",
        "translation attack",
        "translation jailbreak",
    ),
    "structured-completion": (
        "structured prompt",
        "schema jailbreak",
        "json jailbreak",
        "xml jailbreak",
        "table completion",
        "form filling",
    ),
    "code-task-framing": (
        "code completion",
        "code generation jailbreak",
        "program synthesis jailbreak",
        "code task framing",
    ),
    "narrative-role-framing": (
        "roleplay jailbreak",
        "role-play jailbreak",
        "hypothetical framing",
        "fictional framing",
        "narrative jailbreak",
        "story jailbreak",
    ),
    "authority-context-framing": (
        "research framing",
        "scholarly framing",
        "historical framing",
        "authority framing",
        "expert persona",
    ),
    "semantic-transformation": (
        "semantic inversion",
        "semantic transformation",
        "reverse prompt",
        "contradiction jailbreak",
        "euphemism jailbreak",
    ),
    "decomposition-continuation": (
        "prompt decomposition",
        "task chain jailbreak",
        "continuation attack",
        "completion framing",
        "multi-hop jailbreak",
    ),
    "context-fragmentation": (
        "prompt splitting",
        "payload splitting",
        "prompt fragmentation",
        "fragmentation attack",
        "interleaved prompt",
    ),
    "risk-domain-rewrite": (
        "defamation",
        "targeted misinformation",
        "harmful behavior",
        "harmful content",
        "content safety",
        "risk category",
    ),
}

# Repository search can match README text even when a repository's short metadata
# omits the method name.  For these profile families, defer semantic identity checks
# to `_assess_github_readme`; metadata still rejects forks, archives, aggregators,
# defensive-only projects, and unsupported runtime scopes.
_GITHUB_README_DISCOVERY_FAMILIES = frozenset(
    {
        "representation-obfuscation",
        "multilingual-rewrite",
        "structured-completion",
        "code-task-framing",
        "narrative-role-framing",
        "authority-context-framing",
        "semantic-transformation",
        "decomposition-continuation",
        "context-fragmentation",
        "compositional-rewrite",
        "risk-domain-rewrite",
    }
)


class GitHubAPIError(RuntimeError):
    """GitHub API failure with rate-limit context safe for diagnostics."""

    def __init__(
        self, message: str, *, status: int = 0, metadata: dict[str, Any] | None = None
    ):
        super().__init__(message)
        self.status = status
        self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class GitHubEvidenceAssessment:
    eligible: bool
    score: float
    evidence_type: str = ""
    selected_text: str = ""
    selected_sections: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """One source-aware search query and its semantic family."""

    family: str
    query: str
    body_relevance_gate: str = ""
    fallback_for_family: bool = False
    companion_for_family: bool = False


@dataclass(frozen=True, slots=True)
class QueryBodyRelevanceAssessment:
    """Body-only admission result for a narrowly scoped discovery query."""

    eligible: bool
    reason: str
    evidence_terms: tuple[str, ...] = ()
    topic_term: str = ""


@dataclass(frozen=True, slots=True)
class TextAttackRuntimeAssessment:
    """Whether a paper can supply a single-turn textual attack mechanism."""

    eligible: bool
    reason: str
    evidence_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GoogleLinkDiscovery:
    """Credential-free Google/fallback link-discovery result."""

    results: tuple[tuple[str, str], ...]
    backend: str = ""
    errors: tuple[str, ...] = ()


_TARGETED_DEFAMATION_BODY_GATE = "targeted-defamation-evidence-v1"
_REWRITE_REQUIRED_QUERY_FAMILIES = (
    "risk-domain-rewrite",
    "compositional-rewrite",
)
_REWRITE_BREADTH_FAMILY_TARGET = 3
_QUERY_BODY_LOCAL_RADIUS = 600
_TEXT_ATTACK_SCOPE_LOCAL_RADIUS = 260
_TEXT_ATTACK_SCOPE_LEAD_CHARS = 5_000
_TEXT_ATTACK_STRONGLY_STATEFUL_LEAD_CHARS = 2_000
_TEXT_ATTACK_UNSUPPORTED_RUNTIME_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:audio|acoustic|speech|spoken)\b",
        r"\b(?:voice|vocal)(?:[-\s]+(?:input|prompt|attack|jailbreak|"
        r"model|modality|content|cloning))\b",
        r"\b(?:text[-\s]*to[-\s]*speech|tts)\b",
        r"\b(?:multimodal|multi-modal|vision[-\s]*language|image|video)\b",
        r"\bvisual(?:[-\s]+(?:language|input|prompt|attack|jailbreak|"
        r"model|modality|content))\b",
        r"\b(?:web|browser|tool[-\s]*using|autonomous|embodied)\s+"
        r"(?:llm\s+)?agents?\b",
        r"\b(?:tool[-\s]*use|function[-\s]*calling|agent(?:ic)?\s+runtime)\b",
        r"\bonline[-\s]+web[-\s]+search\b",
        r"\b(?:search[-\s]+augmented|web[-\s]+search[-\s]+enabled)\s+"
        r"(?:llm\s+)?agents?\b",
        r"\bpoison(?:ing|ed)?\s+(?:(?:long[-\s]+term\s+)?memor(?:y|ies)|"
        r"(?:rag[-\s]+)?knowledge[-\s]+bases?)\b",
        r"\b(?:rag|retrieval[-\s]+augmented(?:\s+generation)?|"
        r"long[-\s]+term\s+memory|knowledge[-\s]+bases?)\b.{0,80}"
        r"\b(?:poison(?:ing|ed)?|backdoors?)\b",
        r"\b(?:multi[-\s]*turn|multi[-\s]*round|dialogue[-\s]*based|"
        r"conversation[-\s]*based)\b",
    )
)
_TEXT_ATTACK_STRONGLY_STATEFUL_RUNTIME_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:attacker(?:\s+llm)?|attack(?:er)?\s+model)\s+"
        r"iteratively\s+quer(?:y|ies)\s+(?:the\s+)?target(?:\s+llm|\s+model)?\b",
        r"\b(?:full\s+)?conversation\s+history\s+.{0,80}"
        r"iteratively\s+(?:refin|updat)\w*\s+(?:the\s+)?attack\b",
        r"\btarget(?:\s+llm|\s+model)?\s+(?:response|feedback)s?\s+.{0,80}"
        r"iteratively\s+(?:refin|updat)\w*\b",
    )
)
_TEXT_ATTACK_CORE_CLAIM_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bwe\s+(?:propose|present|introduce|develop|study|evaluate|benchmark|"
        r"train|build|design)\b",
        r"\bthis\s+(?:paper|work|study)\s+"
        r"(?:proposes|presents|introduces|develops|studies|evaluates|benchmarks)\b",
        r"\bour\s+(?:attack|method|approach|framework|benchmark|dataset|system)\b",
    )
)
_TEXT_ATTACK_DEFENSE_ONLY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:detect(?:ion|or|ing)?|classif(?:ication|ier|ying)|moderation)\b",
        r"\b(?:defen[cs]e|defensive|guardrail|mitigat(?:e|es|ed|ing|ion)|"
        r"filter(?:s|ed|ing)?|safeguard(?:s|ing)?)\b",
    )
)
_TEXT_ATTACK_DEFENSIVE_TITLE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:defending|protecting|safeguarding)\b.{0,120}"
        r"\b(?:against|from)\b.{0,120}\b(?:attacks?|jailbreak(?:s|ing)?)\b",
        r"\b(?:jailbreak|prompt[- ]attack)\s+defen[cs]es?\b",
        r"\bmitigat\w*\b.{0,120}\b(?:attacks?|jailbreak(?:s|ing)?)\b",
    )
)
_TEXT_ATTACK_ANCHOR_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bjailbreak(?:s|ing|ed)?\b",
        r"\b(?:adversarial|attack|malicious|harmful)\s+prompts?\b",
        r"\bprompt[-\s]*(?:based\s+)?(?:attack|injection)s?\b",
        r"\bred[-\s]*team(?:ing)?\b",
        r"\b(?:refusal|safety|alignment)\s+(?:bypass|evasion|attack)s?\b",
    )
)
_TEXT_ATTACK_DIRECT_TEXT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:jailbreak|adversarial|attack|malicious|harmful)\s+prompts?\b",
        r"\bprompts?\s+(?:for|to)\s+jailbreak\b",
        r"\btext(?:ual|[-\s]*only)?\s+(?:jailbreak|prompt|attack)s?\b",
        r"\bsingle[-\s]*turn\s+(?:jailbreak|prompt|attack|rewrite)s?\b",
    )
)
_TEXT_ATTACK_CONSTRUCTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:generat|craft|construct|creat|rewrit|transform|optim|perturb|"
        r"obfuscat|encod|translat|decompos|compos)(?:e|es|ed|ing|ion|ions|ized)?\b",
        r"\b(?:prefix|suffix|template|payload|trigger)s?\b",
        r"\battack\s+(?:method|approach|algorithm|framework|strategy)\b",
    )
)
_TEXT_ATTACK_EVALUATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:benchmark|evaluation|evaluate|dataset|corpus|"
        r"red[-\s]*team)(?:s|ed|ing)?\b",
        r"\b(?:attack|jailbreak)\s+(?:success|robustness|transferability)\b",
    )
)
_QUERY_BODY_TOPIC_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bdefam(?:ation|atory|e|ed|es|ing)\b",
        r"\blibel(?:ous|ed|ing)?\b",
        r"\bslander(?:ous|ed|ing)?\b",
        r"\bcharacter\s+assassination\b",
        r"\bsmear\s+campaigns?\b",
        r"\breputation(?:al)?\s+(?:harm|damage)\b",
        r"\b(?:false|fabricated|unfounded|baseless)\s+"
        r"(?:allegations?|accusations?|claims?|rumou?rs?)\b",
    )
)
_QUERY_BODY_MODEL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bllms?\b",
        r"\blarge\s+language\s+models?\b",
        r"\blanguage\s+models?\b",
        r"\bgenerative\s+ai\b",
        r"\bchatbots?\b",
        r"\bai\s+(?:models?|systems?|assistants?)\b",
        r"\bmodel\s+(?:outputs?|responses?)\b",
        # Evaluation appendices often render the tested model as a transcript
        # slot rather than repeating "LLM" beside every hazard definition.
        # Keep this narrow to explicit response slots/assessment language so a
        # generic mention of an autonomous agent cannot satisfy the gate.
        r"\b(?:assistant|agent|model)\s*:\s*\{?\s*"
        r"(?:response|answer|completion)\b",
        r"\bsafety\s+assessment\s+for\s+(?:the\s+)?"
        r"(?:assistant|agent|model)\b",
    )
)
_QUERY_BODY_ACTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bgenerat(?:e|es|ed|ing|ion)\b",
        r"\bproduc(?:e|es|ed|ing|tion)\b",
        r"\bwrit(?:e|es|ten|ing)\b",
        r"\bcreat(?:e|es|ed|ing|ion)\b",
        r"\bfabricat(?:e|es|ed|ing|ion)\b",
        r"\bspread(?:s|ing)?\b",
        r"\bpublish(?:es|ed|ing)?\b",
        r"\b(?:red[\s-]*team(?:ing)?|benchmarks?|"
        r"evaluat(?:e|es|ed|ing|ion)|assess(?:es|ed|ing|ment)?|"
        r"refusal|safety\s+(?:test(?:s|ing)?|evaluation))\b",
    )
)
_QUERY_BODY_TARGET_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:named|identifiable|specific)\s+(?:person|individual|company|"
        r"organisation|organization|employer|target)\b",
        r"\b(?:person|individual|company|business|institution|organisation|"
        r"organization|employer|public\s+figure|president|member\s+of\s+congress|"
        r"politician|official|candidate)\b",
        r"\b(?:their|his|her|its|someone(?:'s)?)\s+reputation\b",
    )
)


@dataclass(frozen=True, slots=True)
class SourceEvidenceDocument:
    """One bounded, read-only document inside a source evidence package."""

    path: str
    role: str
    text: str
    url: str = ""
    revision: str = ""
    size_bytes: int = 0
    truncated: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)


_FRONTIER_QUERY_SPECS = (
    QuerySpec("reasoning-model", '"reasoning model" jailbreak adversarial prompt'),
    QuerySpec(
        "schema-exploitation", 'schema exploitation "structured prompt" jailbreak'
    ),
    QuerySpec("trusted-document", '"paper summary" authoritative document jailbreak'),
    QuerySpec("context-coherent", '"context coherent" semantic camouflage jailbreak'),
    QuerySpec(
        "single-turn-composition", '"single turn" compositional payload jailbreak'
    ),
    QuerySpec("long-context", '"many shot" "long context" jailbreak'),
    QuerySpec(
        "reasoning-distraction", "reasoning distraction chain-of-thought jailbreak"
    ),
    QuerySpec(
        "adaptive-template", '"model specific" adaptive jailbreak prompt template'
    ),
    QuerySpec(
        "transfer-trigger",
        'transferable "query agnostic" adversarial trigger jailbreak',
    ),
    QuerySpec(
        "automated-one-shot", 'automated "single turn" prompt optimization jailbreak'
    ),
)


def _source_query_specs(source: str) -> tuple[QuerySpec, ...]:
    if source == "github":
        return (
            QuerySpec(
                "reasoning-model", '"reasoning model jailbreak" in:name,description'
            ),
            QuerySpec("schema-exploitation", '"schema jailbreak" in:name,description'),
            QuerySpec(
                "trusted-document",
                '"paper summary attack" jailbreak in:name,description',
            ),
            QuerySpec(
                "context-coherent", '"context coherent jailbreak" in:name,description'
            ),
            QuerySpec(
                "single-turn-composition",
                '"single turn jailbreak" compositional in:name,description',
            ),
            QuerySpec("long-context", '"many shot jailbreak" in:name,description'),
            QuerySpec(
                "reasoning-distraction",
                '"reasoning distraction" jailbreak in:name,description',
            ),
            QuerySpec(
                "adaptive-template", '"adaptive jailbreak" template in:name,description'
            ),
            QuerySpec(
                "transfer-trigger",
                '"adversarial trigger" jailbreak in:name,description',
            ),
            QuerySpec(
                "automated-one-shot", '"automated jailbreak" in:name,description'
            ),
        )
    if source == "arxiv":
        return (
            QuerySpec(
                "reasoning-model",
                'all:jailbreak AND (all:"reasoning model" OR all:LRM)',
            ),
            QuerySpec(
                "schema-exploitation",
                'all:jailbreak AND (all:schema OR all:"structured prompt")',
            ),
            QuerySpec(
                "trusted-document",
                'all:jailbreak AND (all:"paper summary" OR all:"authoritative document")',
            ),
            QuerySpec(
                "context-coherent",
                'all:jailbreak AND (all:"context coherent" OR all:"semantic camouflage")',
            ),
            QuerySpec(
                "single-turn-composition",
                'all:jailbreak AND all:"single turn" AND all:compositional',
            ),
            QuerySpec(
                "long-context",
                'all:jailbreak AND (all:"many shot" OR all:"long context")',
            ),
            QuerySpec(
                "reasoning-distraction",
                'all:jailbreak AND (all:"reasoning distraction" OR all:"chain of thought")',
            ),
            QuerySpec(
                "adaptive-template",
                'all:jailbreak AND all:adaptive AND all:"prompt template"',
            ),
            QuerySpec(
                "transfer-trigger",
                'all:jailbreak AND (all:transferable OR all:"query agnostic")',
            ),
            QuerySpec(
                "automated-one-shot",
                'all:jailbreak AND all:automated AND all:"single turn"',
            ),
        )
    if source == "huggingface":
        return (
            QuerySpec("reasoning-model", "reasoning jailbreak"),
            QuerySpec("schema-exploitation", "schema jailbreak"),
            QuerySpec("trusted-document", "document jailbreak"),
            QuerySpec("context-coherent", "context jailbreak"),
            QuerySpec("single-turn-composition", "single turn jailbreak"),
            QuerySpec("long-context", "many shot jailbreak"),
            QuerySpec("reasoning-distraction", "chain of thought jailbreak"),
            QuerySpec("adaptive-template", "adaptive jailbreak"),
            QuerySpec("transfer-trigger", "adversarial trigger"),
            QuerySpec("automated-one-shot", "automated jailbreak"),
        )
    return _FRONTIER_QUERY_SPECS


_REPOSITORY_REWRITE_GITHUB_SPECS = (
    QuerySpec(
        "representation-obfuscation",
        'jailbreak "prompt obfuscation" in:readme',
    ),
    QuerySpec("multilingual-rewrite", "jailbreak multilingual in:readme"),
    QuerySpec(
        "structured-completion",
        'jailbreak "structured prompt" in:readme',
    ),
    QuerySpec("code-task-framing", 'jailbreak "code completion" in:readme'),
    QuerySpec("narrative-role-framing", "jailbreak roleplay in:readme"),
    QuerySpec(
        "authority-context-framing",
        'jailbreak "expert persona" in:readme',
    ),
    QuerySpec(
        "semantic-transformation",
        'jailbreak "semantic transformation" in:readme',
    ),
    QuerySpec(
        "decomposition-continuation",
        'jailbreak "prompt decomposition" in:readme',
    ),
    QuerySpec(
        "context-fragmentation",
        'jailbreak "prompt splitting" in:readme',
    ),
    QuerySpec("compositional-rewrite", '"compositional jailbreak" in:readme'),
    QuerySpec(
        "risk-domain-rewrite",
        'defamation LLM jailbreak in:readme',
        body_relevance_gate=_TARGETED_DEFAMATION_BODY_GATE,
    ),
    QuerySpec(
        "risk-domain-rewrite",
        '"reputational harm" LLM "adversarial prompt" in:readme',
        body_relevance_gate=_TARGETED_DEFAMATION_BODY_GATE,
    ),
    QuerySpec(
        "risk-domain-rewrite",
        '(JailbreakBench OR HarmBench OR AILuminate OR "jailbreak benchmark") '
        '(defamation OR "false accusations" OR "usage policies" '
        'OR "hazard categories") '
        '"adversarial prompt" in:readme',
        body_relevance_gate=_TARGETED_DEFAMATION_BODY_GATE,
        fallback_for_family=True,
        companion_for_family=True,
    ),
)


def _repository_rewrite_risk_query_specs(source: str) -> tuple[QuerySpec, ...]:
    """Return source-native, mechanism-oriented targeted-risk discovery queries."""
    if source == "arxiv":
        return (
            QuerySpec(
                "risk-domain-rewrite",
                "(all:defamation OR all:defamatory OR all:libel OR all:slander) "
                'AND (all:LLM OR all:"language model") '
                'AND (all:generation OR all:evaluation OR all:jailbreak '
                'OR all:"red teaming" OR all:"adversarial prompt")',
                body_relevance_gate=_TARGETED_DEFAMATION_BODY_GATE,
            ),
            QuerySpec(
                "risk-domain-rewrite",
                '(all:"reputational harm" OR all:"false allegations" '
                'OR all:"smear campaign") '
                'AND (all:LLM OR all:"generative AI") '
                "AND (all:generation OR all:safety OR all:evaluation)",
                body_relevance_gate=_TARGETED_DEFAMATION_BODY_GATE,
            ),
            QuerySpec(
                "risk-domain-rewrite",
                '(all:JailbreakBench OR all:HarmBench OR all:AILuminate '
                'OR all:"jailbreak benchmark") '
                'AND (all:defamation OR all:"false accusations" '
                'OR all:"usage policies" OR all:"hazard categories") '
                'AND (all:"adversarial prompt" OR all:"automated red teaming" '
                'OR all:evaluation)',
                body_relevance_gate=_TARGETED_DEFAMATION_BODY_GATE,
                fallback_for_family=True,
                companion_for_family=True,
            ),
        )
    if source == "google":
        return (
            QuerySpec(
                "risk-domain-rewrite",
                "(defamation OR defamatory OR libel OR slander) "
                '(LLM OR "language model") '
                '(generation OR evaluation OR jailbreak OR "red teaming" '
                'OR "adversarial prompt")',
                body_relevance_gate=_TARGETED_DEFAMATION_BODY_GATE,
            ),
            QuerySpec(
                "risk-domain-rewrite",
                '("reputational harm" OR "false allegations" OR "smear campaign") '
                '(LLM OR "generative AI") (generation OR safety OR evaluation)',
                body_relevance_gate=_TARGETED_DEFAMATION_BODY_GATE,
            ),
            QuerySpec(
                "risk-domain-rewrite",
                '(JailbreakBench OR HarmBench OR AILuminate '
                'OR "jailbreak benchmark") '
                '(defamation OR "false accusations" OR "usage policies" '
                'OR "hazard categories") '
                '("adversarial prompt" OR "automated red teaming" '
                'OR evaluation)',
                body_relevance_gate=_TARGETED_DEFAMATION_BODY_GATE,
                fallback_for_family=True,
                companion_for_family=True,
            ),
        )
    return tuple(
        QuerySpec(
            spec.family,
            re.sub(r"\s+in:readme\b", "", spec.query).replace('"', "").strip(),
            body_relevance_gate=spec.body_relevance_gate,
            fallback_for_family=spec.fallback_for_family,
            companion_for_family=spec.companion_for_family,
        )
        for spec in _REPOSITORY_REWRITE_GITHUB_SPECS
        if spec.family == "risk-domain-rewrite"
    )


def _repository_rewrite_example_query_specs(source: str) -> tuple[QuerySpec, ...]:
    """Return a discovery-only lane for papers with concrete rewrite artifacts."""
    if source == "arxiv":
        query = (
            'all:jailbreak AND (all:"prompt template" OR all:"worked example" '
            'OR all:"example prompt" OR all:pseudocode)'
        )
    elif source == "google":
        query = (
            'LLM jailbreak ("prompt template" OR "worked example" '
            'OR "example prompt" OR pseudocode) paper'
        )
    elif source == "github":
        query = 'jailbreak "prompt template" example in:readme'
    else:
        query = "jailbreak prompt template examples"
    return (QuerySpec("reproducible-artifact", query),)


def _repository_rewrite_query_specs(source: str) -> tuple[QuerySpec, ...]:
    """Render one shared rewrite taxonomy into source-native query syntax."""
    if source == "github":
        return (
            *_REPOSITORY_REWRITE_GITHUB_SPECS,
            *_repository_rewrite_example_query_specs(source),
        )
    rendered: list[QuerySpec] = []
    for spec in _REPOSITORY_REWRITE_GITHUB_SPECS:
        if spec.family == "risk-domain-rewrite":
            continue
        query = re.sub(r"\s+in:readme\b", "", spec.query).strip()
        if source == "arxiv":
            terms = re.findall(r'"[^"]+"|[A-Za-z][A-Za-z0-9-]*', query)
            query = " AND ".join(f"all:{term}" for term in terms)
        elif source == "huggingface":
            query = query.replace('"', "")
        rendered.append(
            QuerySpec(
                spec.family,
                query,
                body_relevance_gate=spec.body_relevance_gate,
                fallback_for_family=spec.fallback_for_family,
                companion_for_family=spec.companion_for_family,
            )
        )
    return (
        *rendered,
        *_repository_rewrite_risk_query_specs(source),
        *_repository_rewrite_example_query_specs(source),
    )


EXTERNAL_QUERY_PROFILES: dict[str, dict[str, tuple[QuerySpec, ...]]] = {
    DEFAULT_EXTERNAL_QUERY_PROFILE: {
        source: _source_query_specs(source) for source in EXTERNAL_ALL_SOURCES
    },
    ALTERNATIVE_REWRITE_QUERY_PROFILE: {
        "github": (
            QuerySpec(
                "policy-configuration",
                '"policy puppetry" LLM in:name,description',
            ),
            QuerySpec(
                "policy-configuration",
                '"policy file" jailbreak LLM in:name,description',
            ),
        ),
        "arxiv": (
            QuerySpec(
                "policy-configuration", 'all:"policy puppetry" AND all:LLM'
            ),
            QuerySpec(
                "policy-configuration",
                'all:"policy file" AND all:jailbreak AND all:LLM',
            ),
        ),
        "google": (
            QuerySpec(
                "policy-configuration", '"policy puppetry" LLM jailbreak'
            ),
            QuerySpec(
                "policy-configuration", '"policy file" LLM alignment bypass'
            ),
        ),
        "huggingface": (
            QuerySpec("policy-configuration", "policy puppetry jailbreak"),
            QuerySpec("policy-configuration", "policy file LLM jailbreak"),
        ),
    },
    REPOSITORY_REWRITE_QUERY_PROFILE: {
        source: _repository_rewrite_query_specs(source)
        for source in EXTERNAL_ALL_SOURCES
    },
}


@dataclass(frozen=True)
class ExternalCollectedItem:
    """One fetched and cleaned external text item."""

    url: str
    text: str
    title: str = ""
    status: int = 0
    content_type: str = ""
    source_query: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ReadableHTMLParser(HTMLParser):
    """Extract title, visible text, and result links from simple HTML."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header"}
    _BLOCK_TAGS = {
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "pre",
        "section",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.result_links: list[tuple[str, str]] = []
        self._skip_depth = 0
        self._in_title = False
        self._capture_link = False
        self._current_href = ""
        self._current_link_text: list[str] = []
        self.published_dates: list[str] = []
        self.updated_dates: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag == "meta":
            key = (
                attrs_dict.get("property")
                or attrs_dict.get("name")
                or attrs_dict.get("itemprop")
                or ""
            ).casefold()
            value = attrs_dict.get("content", "").strip()
            if value and key in {
                "article:published_time",
                "datepublished",
                "date",
                "dc.date",
            }:
                self.published_dates.append(value)
            if value and key in {
                "article:modified_time",
                "datemodified",
                "last-modified",
                "dc.modified",
            }:
                self.updated_dates.append(value)
        elif tag == "time" and attrs_dict.get("datetime"):
            self.published_dates.append(attrs_dict["datetime"].strip())
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK_TAGS and self.text_parts:
            self.text_parts.append("\n\n")
        if tag == "a":
            classes = attrs_dict.get("class", "")
            href = attrs_dict.get("href", "")
            class_names = set(classes.split())
            if class_names.intersection({"result__a", "result-link"}) and href:
                self._capture_link = True
                self._current_href = href
                self._current_link_text = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS and self.text_parts:
            self.text_parts.append("\n\n")
        if tag == "a" and self._capture_link:
            title = " ".join(" ".join(self._current_link_text).split()).strip()
            if self._current_href:
                self.result_links.append((title, self._current_href))
            self._capture_link = False
            self._current_href = ""
            self._current_link_text = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        if self._capture_link:
            self._current_link_text.append(text)
        self.text_parts.append(text)

    @property
    def title(self) -> str:
        return " ".join(" ".join(self.title_parts).split()).strip()

    @property
    def text(self) -> str:
        raw = " ".join(self.text_parts)
        paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", raw)]
        return "\n\n".join(part for part in paragraphs if part).strip()


class _ArxivCitationMetadataParser(HTMLParser):
    """Read the identity fields published on an official arXiv abstract page."""

    _FIELDS = frozenset({"citation_arxiv_id", "citation_title", "citation_date"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "meta":
            return
        attrs_dict = {key.casefold(): value or "" for key, value in attrs}
        name = attrs_dict.get("name", "").strip().casefold()
        content = attrs_dict.get("content", "").strip()
        if name in self._FIELDS and content and name not in self.values:
            self.values[name] = unescape(content)


def _arxiv_abs_citation_metadata(html_text: str) -> dict[str, str]:
    parser = _ArxivCitationMetadataParser()
    parser.feed(html_text)
    return dict(parser.values)


def _normalize_arxiv_citation_date(value: str) -> str:
    """Normalize arXiv's ``YYYY/MM/DD`` citation date to UTC ISO format."""
    raw = str(value).strip()
    for date_format in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(raw, date_format).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        return parsed.isoformat().replace("+00:00", "Z")
    return _normalize_timestamp(raw)


def html_to_text(html_text: str) -> tuple[str, str]:
    """Return (title, visible_text) from HTML."""
    parser = _ReadableHTMLParser()
    parser.feed(html_text)
    return parser.title, unescape(parser.text)


def _arxiv_html_to_text(html_text: str) -> tuple[str, str]:
    """Preserve paper headings so bounded retention can sample later sections."""
    annotated = re.sub(
        r"(?is)<h([1-6])\b[^>]*>",
        lambda match: "\n\n" + "#" * int(match.group(1)) + " ",
        html_text,
    )
    annotated = re.sub(r"(?is)</h[1-6]\s*>", "\n\n", annotated)
    return html_to_text(annotated)


_PAPER_EXAMPLE_SECTION_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:\d+(?:\.\d+)*[.)]?[ \t]+)?"
    r"(?:worked[ -]+examples?|prompt[ -]+examples?|attack[ -]+examples?|"
    r"qualitative[ -]+examples?|case[ -]+stud(?:y|ies)|demonstrations?)[ \t]*$"
)
_PAPER_INPUT_LABEL_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:[-*][ \t]+)?(?:original(?:[ -]+(?:request|prompt|query))?|"
    r"input(?:[ -]+prompt)?|user(?:[ -]+(?:request|prompt|query))?|"
    r"harmful(?:[ -]+(?:request|behavior))?|query)[ \t]*(?:[:：][ \t]*|$)"
)
_PAPER_OUTPUT_LABEL_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:[-*][ \t]+)?(?:adversarial|jailbreak|attack|rewritten|"
    r"transformed|optimized)(?:[ -]+(?:request|prompt|query|output))?"
    r"[ \t]*(?:[:：][ \t]*|$)|"
    r"^[ \t]*(?:[-*][ \t]+)?output(?:[ -]+prompt)?[ \t]*(?:[:：][ \t]*|$)"
)
_PAPER_USER_LABEL_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:user|human)[ \t]*(?:[:：][ \t]*|$)"
)
_PAPER_ASSISTANT_LABEL_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:assistant|model|chatgpt)[ \t]*(?:[:：][ \t]*|$)"
)
_PAPER_PROMPT_TEMPLATE_PATTERN = re.compile(
    r"\b(?:(?:jailbreak|attack|adversarial|rewrite)[ -]+prompt[ -]+template|"
    r"prompt[ -]+template(?:[ -]+for[ -]+(?:an?[ -]+)?)?"
    r"(?:jailbreak|attack|adversarial[ -]+rewrite)?|"
    r"template[ -]+(?:jailbreak|attack|adversarial)[ -]+prompt)\b",
    re.IGNORECASE,
)
_PAPER_CODE_TEMPLATE_ASSIGNMENT_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:prompt[_ -]?)?template[ \t]*=[ \t]*(?:[rubf]{0,2})?"
    r"(?:\"\"\"|'''|\"|')"
)
_PAPER_EXPLICIT_ATTACK_TEMPLATE_LABEL_PATTERN = re.compile(
    r"\b(?:jailbreak|attack|adversarial|rewrite|rewritten|transformed)\b",
    re.IGNORECASE,
)
_PAPER_TEMPLATE_REWRITE_DIRECTIVE_PATTERN = re.compile(
    r"\b(?:rewrite|rewrites|rewriting|transform|transforms|transforming|"
    r"construct|constructs|constructing)\b",
    re.IGNORECASE,
)
_PAPER_OFFENSIVE_EXAMPLE_CONTEXT_PATTERN = re.compile(
    r"\b(?:jailbreak|adversarial(?:[ -]+(?:attack|prompt|suffix|rewrite))?|"
    r"attack[ -]+(?:method|prompt|example|rewrite)|red[ -]+team(?:ing)?|"
    r"(?:bypass|evade|weaken)\w*[ -]+(?:refusal|guardrail|safety|alignment)|"
    r"(?:unsafe|harmful)[ -]+(?:request|prompt)[ -]+rewrite)\b",
    re.IGNORECASE,
)
_PAPER_OFFENSIVE_TEMPLATE_PREFIX_PATTERN = re.compile(
    r"\b(?:persona[ -]+)?jailbreak(?:[ -]+attack)?[ -]+prompt[ -]+rewrite\b|"
    r"\b(?:adversarial|attack)[ -]+prompt[ -]+"
    r"(?:rewrite|template|construction)\b|"
    r"\b(?:use|adopt|simulate)\b[^.!?\n]{0,80}"
    r"\b(?:jailbreak|adversarial|attack)[ -]+persona\b",
    re.IGNORECASE,
)
_PAPER_BENIGN_REWRITE_PATTERN = re.compile(
    r"\b(?:grammar|grammatical|spelling|punctuation|clarity|readability|"
    r"proofread(?:ing)?|copy[ -]*edit(?:ing)?|correct[ -]+(?:typos?|grammar|"
    r"spelling)|polish(?:ing)?[ -]+(?:the[ -]+)?(?:prose|writing)|"
    r"improve[ -]+(?:the[ -]+)?(?:wording|readability))\b",
    re.IGNORECASE,
)
_PAPER_LOCAL_ATTACK_GENERATION_PATTERN = re.compile(
    r"\b(?:jailbreak|adversarial[ -]+(?:attack|prompt|rewrite)|"
    r"attack[ -]+(?:prompt|rewrite)|red[ -]+team(?:ing)?|"
    r"persona[ -]+(?:frame|framed|prompt|wrapper)|role[ -]*(?:play|framed)|"
    r"(?:bypass|evade|weaken)\w*[ -]+(?:refusal|guardrail|safety|alignment))\b",
    re.IGNORECASE,
)
_PAPER_DEFENSIVE_ARTIFACT_PATTERN = re.compile(
    r"\b(?:classifier|classification|detector|detection|evaluator|judge|scorer?|"
    r"moderator|moderation|reviewer|auditor|verifier|defen[cs]e|"
    r"safety[ -]+label|maliciousness|decide[ -]+whether|classify|detect|"
    r"score|review|audit|verify)\b",
    re.IGNORECASE,
)
_PAPER_DEFENSIVE_TEMPLATE_ROLE_PATTERN = re.compile(
    r"\b(?:prompt[ -]+template|template)\b[\s:=-]{0,12}"
    r"(?:(?:for|used[ -]+by)\s+(?:an?\s+)?)?"
    r"(?:classifier|detector|evaluator|scorer|moderator|reviewer|auditor|"
    r"verifier)\b|"
    r"\b(?:classifier|detector|evaluator|scorer|moderator|reviewer|auditor|"
    r"verifier)\b[\s:=-]{0,12}\b(?:prompt[ -]+template|template)\b",
    re.IGNORECASE,
)
_PAPER_DEFENSIVE_TEMPLATE_TASK_PATTERN = re.compile(
    r"\b(?:decide[ -]+whether|determine[ -]+whether|classif(?:y|ies)|"
    r"detect(?:s)?|evaluat(?:e|es)|scor(?:e|es)|moderat(?:e|es)|"
    r"judg(?:e|es)|review(?:s)?|audit(?:s)?|verif(?:y|ies)|assess(?:es)?|"
    r"check(?:s)?|(?:answer|explain|report|state|respond)\s+whether)\b"
    r"[^.!?\n]{0,220}(?:\{\{?|\[|<|\b(?:prompt|request|query|question|"
    r"instruction|input|content)\b)",
    re.IGNORECASE,
)
_PAPER_TEMPLATE_REWRITE_NEGATION_PATTERN = re.compile(
    r"\b(?:(?:do|does|did|must|should|will|can)\s+not"
    r"(?:[\s,]+\w+){0,6}|never(?:[\s,]+\w+){0,4}|without|"
    r"avoid(?:s|ed|ing)?(?:[\s,]+\w+){0,3}|"
    r"refrain(?:s|ed|ing)?\s+from(?:[\s,]+\w+){0,2}|"
    r"(?:decline|refuse)(?:s|d|ing)?\s+to(?:[\s,]+\w+){0,2}|"
    r"abstain(?:s|ed|ing)?\s+from(?:[\s,]+\w+){0,2}|"
    r"(?:am|is|are|be|being|been)\s+prohibited\s+from"
    r"(?:[\s,]+\w+){0,2}|"
    r"under\s+no\s+circumstances(?:[\s,]+\w+){0,3}|"
    r"instead\s+of|rather\s+than)[\s,]*$",
    re.IGNORECASE,
)
_PAPER_EXPLICIT_EXAMPLE_PROMPT_PATTERN = re.compile(
    r"\b(?:(?:worked|complete|concrete|illustrative|example|sample)[ -]+"
    r"(?:jailbreak|attack|adversarial|rewritten|transformed)[ -]+prompt|"
    r"(?:jailbreak|attack|adversarial|rewritten|transformed)[ -]+prompt[ -]+"
    r"(?:example|sample|demonstration))\b",
    re.IGNORECASE,
)
_PAPER_TEMPLATE_PLACEHOLDER_PATTERN = re.compile(
    r"\{\{?(?:QUERY|REQUEST|QUESTION(?:_TEXT)?|PERSONA(?:_TEXT)?|GOAL|TASK|"
    r"PROMPT|INSTRUCTION|BEHAVIOR|USER_INPUT)"
    r"(?:[-_ ][^{}\n]{0,60})?\}\}?|"
    r"\[(?:INSERT|QUERY|REQUEST|QUESTION|PERSONA|GOAL|TASK|PROMPT|"
    r"INSTRUCTION)[^\]\n]{0,80}\]|<(?:query|request|goal|task|prompt|"
    r"instruction)(?:[-_ ][^>\n]{0,60})?>",
    re.IGNORECASE,
)
_PAPER_ALGORITHM_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:algorithm|pseudocode)[ \t]*(?:\d+|[A-Z][A-Za-z0-9_-]*)?\b"
)


def _assess_paper_example_evidence(text: str) -> dict[str, Any]:
    """Find concrete, local paper artifacts useful for authoring a rewrite skill.

    Generic prose such as ``for example`` is intentionally ignored.  A complete
    artifact needs a locally ordered original-to-adversarial pair or a substantive
    prompt template with a real placeholder.  Worked-example headings, target
    transcripts, and executable-looking
    pseudocode are retained as partial evidence so they can improve recall without
    being mistaken for a complete demonstration.
    """
    body = unicodedata.normalize("NFKC", str(text)).replace("\r\n", "\n")
    headings = list(re.finditer(r"(?im)^(#{1,6})[ \t]+([^\n]+)$", body))
    excluded_heading = re.compile(
        r"\b(?:related[ -]+work|prior[ -]+work|background|survey|"
        r"literature[ -]+review|baselines?|comparison[ -]+methods?)\b",
        re.IGNORECASE,
    )
    evaluation_heading = re.compile(
        r"\b(?:evaluation|experiment(?:s|al)?|results?|benchmark(?:ing)?)\b",
        re.IGNORECASE,
    )
    example_heading = re.compile(
        r"\b(?:worked|prompt|attack|qualitative)[ -]+examples?|"
        r"\bcase[ -]+stud(?:y|ies)|\bdemonstrations?\b",
        re.IGNORECASE,
    )

    def _active_heading_titles(position: int) -> list[str]:
        active: list[tuple[int, str]] = []
        for heading in headings:
            if heading.start() > position:
                break
            level = len(heading.group(1))
            while active and active[-1][0] >= level:
                active.pop()
            active.append((level, heading.group(2).strip()))
        return [title for _level, title in active]

    def _belongs_to_excluded_section(match: re.Match[str]) -> bool:
        return any(
            excluded_heading.search(title)
            for title in _active_heading_titles(match.start())
        )

    def _is_source_owned_example(match: re.Match[str]) -> bool:
        titles = _active_heading_titles(match.start())
        if not any(evaluation_heading.search(title) for title in titles):
            return True
        if any(example_heading.search(title) for title in titles):
            return True
        context = body[
            max(0, match.start() - 1200) : min(len(body), match.end() + 1600)
        ]
        return bool(
            re.search(
                r"\b(?:our|the[ -]+proposed)[ -]+(?:method|attack|approach|"
                r"framework|algorithm|prompt|rewrite)|\bwe[ -]+(?:generate|"
                r"construct|rewrite|transform|optimi[sz]e|propose)\b",
                context,
                flags=re.IGNORECASE,
            )
        )

    def _has_local_attack_context(match: re.Match[str], radius: int = 1600) -> bool:
        context = body[
            max(0, match.start() - radius) : min(len(body), match.end() + radius)
        ]
        return bool(
            re.search(
                r"\b(?:jailbreak|attack[ -]+prompt|adversarial[ -]+prompt|"
                r"prompt[ -]+rewrite|rewritten[ -]+prompt|transformed[ -]+prompt)\b",
                context,
                flags=re.IGNORECASE,
            )
        )

    def _has_local_rewrite_context(match: re.Match[str], radius: int = 700) -> bool:
        context = body[
            max(0, match.start() - radius) : min(len(body), match.end() + radius)
        ]
        return bool(
            re.search(
                r"\b(?:jailbreak[ -]+attack[ -]+prompt[ -]+rewrite|"
                r"attack[ -]+prompt[ -]+rewrite|prompt[ -]+rewrite|"
                r"rewritten[ -]+prompt|transformed[ -]+prompt|"
                r"rewrite[ -]+(?:the[ -]+)?(?:input|query|request|prompt))\b",
                context,
                flags=re.IGNORECASE,
            )
        )

    def _is_attacker_side_artifact(
        match: re.Match[str],
        *,
        artifact_text: str,
    ) -> bool:
        return bool(
            _PAPER_OFFENSIVE_EXAMPLE_CONTEXT_PATTERN.search(artifact_text)
            and _is_source_owned_example(match)
        )

    def _is_defensive_template_artifact(text: str) -> bool:
        return bool(
            _PAPER_DEFENSIVE_TEMPLATE_ROLE_PATTERN.search(text)
            or _PAPER_DEFENSIVE_TEMPLATE_TASK_PATTERN.search(text)
            or re.search(
                r"\b(?:return|emit|produce)\b[^.!?\n]{0,120}"
                r"\b(?:safety|policy|maliciousness|attack)[ -]+"
                r"(?:label|score|classification)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _has_affirmative_template_rewrite(text: str) -> bool:
        placeholder = _PAPER_TEMPLATE_PLACEHOLDER_PATTERN.search(text)
        if placeholder is None:
            return False

        def _directive_governs_placeholder(between: str) -> bool:
            if re.search(r"[.;!?]", between):
                return False
            if re.search(
                r"\b(?:act|answer|write|produce|generate|respond|return|"
                r"classify|detect|score|judge|review|audit|verify|assess|"
                r"explain|report|state)\b",
                between,
                flags=re.IGNORECASE,
            ):
                return False
            if re.search(
                r"\b(?:summary|title|header|label|metadata|schema|record|"
                r"field|key|configuration|table|caption)\b",
                between,
                flags=re.IGNORECASE,
            ):
                return False
            words = [word.lower() for word in re.findall(r"[A-Za-z]+", between)]
            if not words:
                return True
            if re.search(
                r"\b(?:jailbreak|adversarial|attack|persona|role[ -]*play)"
                r"[ -]+(?:prompt|wrapper|frame|framing|payload)\b"
                r"[^.!?]{0,40}\b(?:around|from|using|with|for|of)\s*$",
                between,
                flags=re.IGNORECASE,
            ):
                return True
            if re.fullmatch(
                r"[\s,:-]*(?:(?:while\s+)?preserv(?:e|ing)\s+"
                r"(?:all\s+|every\s+)?(?:named\s+)?constraints?[\s,:-]*)",
                between,
                flags=re.IGNORECASE,
            ):
                return True
            direct_object_modifiers = {
                "a",
                "an",
                "the",
                "this",
                "that",
                "following",
                "provided",
                "given",
                "original",
                "user",
                "source",
                "target",
                "entire",
                "full",
                "whole",
                "complete",
                "below",
                "above",
                "directly",
                "fully",
                "completely",
                "faithfully",
                "verbatim",
            }
            target_nouns = {
                "request",
                "query",
                "question",
                "prompt",
                "instruction",
                "task",
                "behavior",
                "input",
                "text",
                "content",
            }
            return all(
                word in direct_object_modifiers or word in target_nouns
                for word in words
            ) and any(word in target_nouns for word in words)

        for directive in _PAPER_TEMPLATE_REWRITE_DIRECTIVE_PATTERN.finditer(text):
            prefix = text[max(0, directive.start() - 80) : directive.start()]
            if re.search(r"\bnot\s+only\s*$", prefix, flags=re.IGNORECASE):
                negated = False
            else:
                negated = bool(
                    _PAPER_TEMPLATE_REWRITE_NEGATION_PATTERN.search(prefix)
                )
            if negated:
                continue
            if directive.end() <= placeholder.start():
                between = text[directive.end() : placeholder.start()]
                if _directive_governs_placeholder(between):
                    return True
            elif directive.start() >= placeholder.end():
                between = text[placeholder.end() : directive.start()]
                if re.fullmatch(
                    r"[\s,:-]*(?:(?:then|please)\s+)?",
                    between,
                    flags=re.IGNORECASE,
                ):
                    return True
        return False

    def _is_attacker_side_pair(
        left: re.Match[str],
        right: re.Match[str],
    ) -> bool:
        context = body[
            max(0, left.start() - 400) : min(
                len(body), _artifact_content_end(right) + 400
            )
        ]
        return bool(
            _PAPER_OFFENSIVE_EXAMPLE_CONTEXT_PATTERN.search(context)
            and not _PAPER_DEFENSIVE_ARTIFACT_PATTERN.search(context)
            and _is_source_owned_example(left)
            and _is_source_owned_example(right)
        )

    next_artifact_boundary = re.compile(
        r"(?im)^[ \t]*(?:#{1,6}[ \t]+|(?:[-*][ \t]+)?(?:original|input|"
        r"user|human|query|output|assistant|model|chatgpt|adversarial|jailbreak|"
        r"attack|rewritten|transformed|optimized)(?:[ -]+(?:request|prompt|"
        r"query|output|behavior))?[ \t]*(?:[:：][ \t]*|$))"
    )

    def _artifact_content_end(match: re.Match[str], max_chars: int = 550) -> int:
        tail = body[match.end() : match.end() + max_chars]
        boundary = next_artifact_boundary.search(tail.lstrip("\n"))
        if boundary is None:
            return min(len(body), match.end() + max_chars)
        leading_newlines = len(tail) - len(tail.lstrip("\n"))
        return match.end() + leading_newlines + boundary.start()

    def _nontrivial_labels(pattern: re.Pattern[str]) -> list[re.Match[str]]:
        matches: list[re.Match[str]] = []
        for match in pattern.finditer(body):
            content = body[match.end() : _artifact_content_end(match)]
            if (
                len(re.findall(r"[A-Za-z0-9]+", content)) >= 5
                and not _belongs_to_excluded_section(match)
                and _is_source_owned_example(match)
            ):
                matches.append(match)
        return matches

    input_labels = _nontrivial_labels(_PAPER_INPUT_LABEL_PATTERN)
    output_labels = [
        match
        for match in _nontrivial_labels(_PAPER_OUTPUT_LABEL_PATTERN)
        if re.search(
            r"\b(?:adversarial|jailbreak|attack|rewritten|transformed|optimized)\b",
            match.group(0),
            flags=re.IGNORECASE,
        )
        or _has_local_rewrite_context(match)
    ]
    transcript_inputs = [
        match
        for match in _nontrivial_labels(_PAPER_USER_LABEL_PATTERN)
        if _has_local_attack_context(match)
    ]
    transcript_outputs = [
        match
        for match in _nontrivial_labels(_PAPER_ASSISTANT_LABEL_PATTERN)
        if _has_local_attack_context(match)
    ]

    def _ordered_local_pairs(
        first: list[re.Match[str]], second: list[re.Match[str]], radius: int = 1200
    ) -> list[tuple[re.Match[str], re.Match[str]]]:
        pairs: list[tuple[re.Match[str], re.Match[str]]] = []
        used_second: set[int] = set()
        for left in first:
            for right in second:
                if (
                    right.start() in used_second
                    or right.start() < left.end()
                    or right.start() - left.end() > radius
                ):
                    continue
                between = body[left.end() : right.start()]
                if re.search(r"(?m)^#{1,6}[ \t]+", between):
                    continue
                pairs.append((left, right))
                used_second.add(right.start())
                break
        return pairs

    io_pairs = [
        (left, right)
        for left, right in _ordered_local_pairs(input_labels, output_labels)
        if _is_attacker_side_pair(left, right)
    ]
    transcript_pairs = _ordered_local_pairs(transcript_inputs, transcript_outputs)
    io_pair_count = len(io_pairs)
    transcript_pair_count = len(transcript_pairs)
    raw_template_matches = [
        match
        for match in _PAPER_PROMPT_TEMPLATE_PATTERN.finditer(body)
        if not _belongs_to_excluded_section(match)
        and _is_source_owned_example(match)
    ]
    raw_template_matches.extend(
        match
        for match in _PAPER_CODE_TEMPLATE_ASSIGNMENT_PATTERN.finditer(body)
        if not _belongs_to_excluded_section(match)
        and _is_source_owned_example(match)
    )
    raw_template_matches.sort(key=lambda match: match.start())
    template_block_matches: list[re.Match[str]] = []
    complete_template_artifacts: list[tuple[re.Match[str], int]] = []
    for match in raw_template_matches:
        surrounding = body[max(0, match.start() - 180) : match.end() + 240]
        if re.search(
            r"\b(?:(?:do|does|did|is|are|was|were|will)[ -]+not[ -]+"
            r"(?:release|provide|publish|disclose|include|make[ -]+available)|"
            r"not[ -]+publicly[ -]+available|unreleased|undisclosed|omitted|"
            r"unavailable|future[ -]+work|could[ -]+define)\b",
            surrounding,
            flags=re.IGNORECASE,
        ):
            continue
        block_end = min(len(body), match.end() + 1400)
        next_heading = re.search(r"(?m)^#{1,6}[ \t]+", body[match.end() : block_end])
        if next_heading is not None:
            block_end = match.end() + next_heading.start()
        block = body[match.end() : block_end]
        placeholder = _PAPER_TEMPLATE_PLACEHOLDER_PATTERN.search(block)
        if placeholder is None:
            continue
        before_placeholder = block[: placeholder.start()]
        previous_boundaries = list(
            re.finditer(r"(?:[.!?](?=\s)|\n\s*\n)", before_placeholder)
        )
        local_start = (
            previous_boundaries[-1].end() if previous_boundaries else 0
        )
        after_placeholder = block[placeholder.end() :]
        next_boundary = re.search(
            r"(?:[.!?](?=\s|$)|\n\s*\n)",
            after_placeholder,
        )
        local_end = (
            placeholder.end() + next_boundary.end()
            if next_boundary is not None
            else min(len(block), placeholder.end() + 500)
        )
        local_instruction = block[local_start:local_end]
        artifact_instruction = block[
            : min(len(block), placeholder.end() + 500)
        ]
        instruction_text = _PAPER_TEMPLATE_PLACEHOLDER_PATTERN.sub(
            " ", artifact_instruction
        )
        local_instruction_text = _PAPER_TEMPLATE_PLACEHOLDER_PATTERN.sub(
            " ", local_instruction
        )
        instruction_words = re.findall(r"[A-Za-z0-9]+", instruction_text)
        template_prefix = block[
            max(0, placeholder.start() - 240) : placeholder.start()
        ]
        has_explicit_attack_template_label = bool(
            _PAPER_EXPLICIT_ATTACK_TEMPLATE_LABEL_PATTERN.search(
                match.group(0) + " " + template_prefix
            )
        )
        artifact_semantics = match.group(0) + " " + local_instruction
        if _PAPER_OFFENSIVE_TEMPLATE_PREFIX_PATTERN.search(template_prefix):
            artifact_semantics += " " + template_prefix
        defensive_semantics = match.group(0) + " " + local_instruction
        has_rewrite_directive = _has_affirmative_template_rewrite(
            local_instruction
        )
        normalized_local_instruction = _PAPER_TEMPLATE_PLACEHOLDER_PATTERN.sub(
            " PLACEHOLDER ", local_instruction
        )
        has_attack_constraint_preflight = bool(
            re.search(
                r"\b(?:verify|audit|review|check)\b[^.!?\n]{0,100}"
                r"\b(?:(?:all|every|the|named)\s+)*constraints?\s+"
                r"(?:in|of|from)\s+PLACEHOLDER\b[^.!?\n]{0,80}"
                r"\b(?:then|and)\s+(?:rewrite|transform|construct)\s+"
                r"(?:it|the\s+(?:request|query|question|prompt|instruction|"
                r"input|text|content))\b",
                normalized_local_instruction,
                flags=re.IGNORECASE,
            )
        )
        is_evaluator_template = bool(
            _is_defensive_template_artifact(defensive_semantics)
            and not (
                has_rewrite_directive
                and has_attack_constraint_preflight
            )
        )
        is_benign_rewrite = bool(
            _PAPER_BENIGN_REWRITE_PATTERN.search(local_instruction)
            and not _PAPER_LOCAL_ATTACK_GENERATION_PATTERN.search(
                local_instruction
            )
        )
        has_unbound_or_negated_rewrite = bool(
            _PAPER_TEMPLATE_REWRITE_DIRECTIVE_PATTERN.search(local_instruction)
            and not has_rewrite_directive
        )
        attacker_side = _is_attacker_side_artifact(
            match,
            artifact_text=artifact_semantics,
        )
        has_directive = bool(
            re.search(
                r"\b(?:act|answer|write|produce|generate|rewrite|transform|respond|"
                r"follow|use|include|ignore|complete|return|format|first|then|"
                r"place|insert|preserve)\b",
                instruction_text,
                flags=re.IGNORECASE,
            )
        )
        if (
            len(instruction_words) < 6
            or not has_directive
            or is_evaluator_template
            or is_benign_rewrite
            or not attacker_side
            or has_unbound_or_negated_rewrite
            or (
                not has_explicit_attack_template_label
                and not has_rewrite_directive
            )
        ):
            continue
        template_block_matches.append(match)
        artifact_end = min(
            block_end,
            match.end() + max(local_end + 200, 700),
        )
        complete_template_artifacts.append((match, artifact_end))
    explicit_example_matches = [
        match
        for match in _PAPER_EXPLICIT_EXAMPLE_PROMPT_PATTERN.finditer(body)
        if not _belongs_to_excluded_section(match)
        and _is_source_owned_example(match)
    ]
    section_matches = [
        match
        for match in _PAPER_EXAMPLE_SECTION_PATTERN.finditer(body)
        if not _belongs_to_excluded_section(match)
        and _is_source_owned_example(match)
        and re.search(
            r"\b(?:jailbreak|attack|adversarial|prompt|rewrite|transform)\w*\b",
            body[match.start() : match.end() + 1800],
            flags=re.IGNORECASE,
        )
    ]
    algorithm_matches: list[re.Match[str]] = []
    for match in _PAPER_ALGORITHM_PATTERN.finditer(body):
        if _belongs_to_excluded_section(match):
            continue
        window = body[match.start() : match.end() + 2600]
        numbered_steps = len(
            re.findall(r"(?m)^[ \t]*(?:\d+[:.)]|[-*])[ \t]+", window)
        )
        if (
            numbered_steps >= 2
            and re.search(r"\binput\b", window, flags=re.IGNORECASE)
            and re.search(
                r"\b(?:output|return|emit|generate|construct)\w*\b",
                window,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"\b(?:jailbreak|adversarial[ -]+(?:prompt|suffix)|attack[ -]+"
                r"prompt|prompt[ -]+(?:rewrite|transform|construction|"
                r"optimi[sz]ation)|rewrit(?:e|ing)[ -]+(?:the[ -]+)?(?:input|"
                r"query|request|prompt)|transformed[ -]+prompt)\b",
                window,
                flags=re.IGNORECASE,
            )
            and _is_source_owned_example(match)
        ):
            algorithm_matches.append(match)

    # A User/Assistant transcript is target-model behavior, not by itself an
    # attacker-side rewrite example.  It remains useful partial evidence only.
    complete = bool(io_pair_count or complete_template_artifacts)
    partial = bool(
        template_block_matches
        or transcript_pair_count
        or explicit_example_matches
        or section_matches
        or algorithm_matches
    )
    status = "complete" if complete else "partial" if partial else "none"
    signals: list[str] = []
    for present, name in (
        (io_pair_count > 0, "labeled_input_output_pair"),
        (transcript_pair_count > 0, "user_assistant_transcript"),
        (bool(complete_template_artifacts), "prompt_template_with_placeholder"),
        (
            bool(template_block_matches) and not complete_template_artifacts,
            "prompt_template_block",
        ),
        (bool(explicit_example_matches), "explicit_example_prompt"),
        (bool(section_matches), "worked_example_section"),
        (bool(algorithm_matches), "algorithm_or_pseudocode"),
    ):
        if present:
            signals.append(name)
    score = min(
        100,
        50 * min(io_pair_count, 1)
        + 10 * min(transcript_pair_count, 1)
        + 45 * min(len(complete_template_artifacts), 1)
        + 10
        * min(
            int(bool(template_block_matches) and not complete_template_artifacts),
            1,
        )
        + 20 * min(len(explicit_example_matches), 1)
        + 15 * min(len(section_matches), 1)
        + 10 * min(len(algorithm_matches), 1),
    )
    complete_spans = [
        (left.start(), _artifact_content_end(right)) for left, right in io_pairs
    ] + [
        (match.start(), artifact_end)
        for match, artifact_end in complete_template_artifacts
    ]
    partial_spans = [
        (left.start(), _artifact_content_end(right))
        for left, right in transcript_pairs
    ] + [
        (match.start(), min(len(body), match.end() + 1200))
        for match in (
            *template_block_matches,
            *explicit_example_matches,
            *section_matches,
            *algorithm_matches,
        )
    ]
    primary_span = min(complete_spans or partial_spans, default=(-1, -1))
    return {
        "status": status,
        "score": score,
        "signals": signals,
        "anchor_start": primary_span[0],
        "anchor_end": primary_span[1],
        "io_pair_count": io_pair_count,
        "transcript_pair_count": transcript_pair_count,
        "prompt_template_count": len(complete_template_artifacts),
        "worked_example_count": len(section_matches),
        "algorithm_count": len(algorithm_matches),
    }


_ARXIV_SECTION_PRIORITY_PATTERNS = (
    (12, re.compile(r"\bablation|component analysis|sensitivity\b", re.I)),
    (
        10,
        re.compile(
            r"\bmethod(?:ology)?|approach|algorithm|attack|prompt construction|implementation\b",
            re.I,
        ),
    ),
    (
        8,
        re.compile(
            r"\bexperiment|evaluation|result|benchmark|dataset|metric\b",
            re.I,
        ),
    ),
    (7, re.compile(r"\bappendix|template|hyperparameter|reproduc", re.I)),
    (5, re.compile(r"\babstract|introduction|threat model\b", re.I)),
)


def _bounded_arxiv_paper_text(text: str, *, max_chars: int = 50_000) -> str:
    """Retain role-balanced paper sections instead of truncating only the prefix."""
    normalized = str(text).strip()
    if len(normalized) <= max_chars:
        return normalized

    def _span_window(source: str, start: int, end: int, budget: int) -> str:
        if budget <= 0:
            return ""
        bounded_start = max(0, min(int(start), len(source)))
        bounded_end = max(bounded_start, min(int(end), len(source)))
        span_length = bounded_end - bounded_start
        if span_length >= budget:
            return source[bounded_start : bounded_start + budget]
        padding = budget - span_length
        window_start = max(0, bounded_start - padding // 3)
        window_end = min(len(source), window_start + budget)
        window_start = max(0, window_end - budget)
        return source[window_start:window_end]

    sections = [
        section.strip()
        for section in re.split(r"(?m)(?=^#{1,6}\s+)", normalized)
        if section.strip()
    ]
    if len(sections) < 2:
        example_evidence = _assess_paper_example_evidence(normalized)
        anchor_start = int(example_evidence.get("anchor_start", -1))
        anchor_end = int(example_evidence.get("anchor_end", anchor_start))
        if anchor_start >= 0:
            markers = (
                "\n\n[Middle worked-example context]\n\n",
                "\n\n[Later paper text retained]\n\n",
            )
            content_budget = max(0, max_chars - sum(map(len, markers)))
            head_budget = content_budget // 4
            example_budget = content_budget // 2
            tail_budget = content_budget - head_budget - example_budget
            rendered = (
                normalized[:head_budget]
                + markers[0]
                + _span_window(
                    normalized,
                    anchor_start,
                    anchor_end,
                    example_budget,
                )
                + markers[1]
                + normalized[-tail_budget:]
            )
            return rendered[:max_chars]
        marker = "\n\n[Later paper text retained]\n\n"
        head_budget = (max_chars - len(marker)) * 4 // 5
        tail_budget = max_chars - len(marker) - head_budget
        return normalized[:head_budget] + marker + normalized[-tail_budget:]

    ranked: list[tuple[int, int, str, dict[str, Any]]] = []
    for index, section in enumerate(sections):
        heading = section.splitlines()[0][:300]
        score = max(
            (
                priority
                for priority, pattern in _ARXIV_SECTION_PRIORITY_PATTERNS
                if pattern.search(heading)
            ),
            default=1,
        )
        example_evidence = _assess_paper_example_evidence(section)
        if example_evidence["status"] == "complete":
            score = max(score, 24)
        elif example_evidence["status"] == "partial":
            score = max(score, 14)
        # Narrow risk-domain evidence may occur only in a late appendix or a
        # single benchmark-label page.  Rank the complete section/page body,
        # not merely its heading, so bounding cannot silently discard it.
        topic_match = next(
            (
                match
                for pattern in _QUERY_BODY_TOPIC_PATTERNS
                if (match := pattern.search(section)) is not None
            ),
            None,
        )
        if topic_match is not None:
            score = max(score, 20)
        if index == 0:
            score = max(score, 6)
        ranked.append((score, index, section, example_evidence))

    selected: dict[int, str] = {}
    remaining = max_chars
    for _score, index, section, example_evidence in sorted(
        ranked,
        key=lambda value: (-value[0], value[1]),
    ):
        if remaining < 800:
            break
        section_budget = min(len(section), 8_000, remaining)
        topic_match = next(
            (
                match
                for pattern in _QUERY_BODY_TOPIC_PATTERNS
                if (match := pattern.search(section)) is not None
            ),
            None,
        )
        example_anchor = int(example_evidence.get("anchor_start", -1))
        example_end = int(example_evidence.get("anchor_end", example_anchor))
        if (
            topic_match is not None
            and example_anchor >= 0
            and len(section) > section_budget
        ):
            heading = section.splitlines()[0][:300]
            markers = (
                "\n[Method section lead]\n",
                "\n[Domain-relevant page context]\n",
                "\n[Worked-example context]\n",
            )
            content_budget = max(
                0,
                section_budget - len(heading) - sum(map(len, markers)),
            )
            lead_budget = content_budget // 5
            domain_budget = content_budget * 2 // 5
            example_budget = content_budget - lead_budget - domain_budget
            bounded = (
                heading
                + markers[0]
                + section[len(section.splitlines()[0]) :][:lead_budget]
                + markers[1]
                + _span_window(
                    section,
                    topic_match.start(),
                    topic_match.end(),
                    domain_budget,
                )
                + markers[2]
                + _span_window(
                    section,
                    example_anchor,
                    example_end,
                    example_budget,
                )
            )[:section_budget]
        elif topic_match is not None and len(section) > section_budget:
            heading = section.splitlines()[0][:300]
            context_budget = max(0, section_budget - len(heading) - 34)
            start = max(0, topic_match.start() - context_budget // 2)
            end = min(len(section), start + context_budget)
            start = max(0, end - context_budget)
            bounded = (
                heading
                + "\n[Domain-relevant page context]\n"
                + section[start:end]
            )[:section_budget]
        elif (
            len(section) > section_budget
            and example_anchor >= 0
            and example_anchor > section_budget // 2
        ):
            heading = section.splitlines()[0][:300]
            marker_budget = len("\n[Method section lead]\n\n[Worked-example context]\n")
            content_budget = max(0, section_budget - len(heading) - marker_budget)
            lead_budget = content_budget // 2
            example_budget = content_budget - lead_budget
            bounded = (
                heading
                + "\n[Method section lead]\n"
                + section[len(section.splitlines()[0]) :][:lead_budget]
                + "\n[Worked-example context]\n"
                + _span_window(
                    section,
                    example_anchor,
                    example_end,
                    example_budget,
                )
            )[:section_budget]
        else:
            bounded = section[:section_budget]
        selected[index] = bounded
        remaining -= len(bounded) + 2
    rendered = "\n\n".join(selected[index] for index in sorted(selected))
    return rendered[:max_chars]


def extract_html_dates(html_text: str) -> tuple[str, str]:
    """Extract normalized publication and modification timestamps from HTML metadata."""
    parser = _ReadableHTMLParser()
    parser.feed(html_text)
    published = list(parser.published_dates)
    updated = list(parser.updated_dates)

    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>([\s\S]*?)</script>',
        html_text,
        flags=re.IGNORECASE,
    ):
        try:
            payload = json.loads(unescape(match.group(1)).strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for key, value in _walk_json_dates(payload):
            if key == "datepublished":
                published.append(value)
            elif key == "datemodified":
                updated.append(value)

    return _first_normalized_timestamp(published), _first_normalized_timestamp(updated)


def extract_visible_date(text: str) -> str:
    """Extract a full English publication date rendered in visible page text."""
    match = re.search(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},\s+\d{4}\b",
        text[:5000],
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    try:
        parsed = datetime.strptime(match.group(0), "%B %d, %Y").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return ""
    return parsed.isoformat().replace("+00:00", "Z")


def _walk_json_dates(value: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).casefold()
            if (
                normalized_key in {"datepublished", "datemodified"}
                and str(child).strip()
            ):
                found.append((normalized_key, str(child).strip()))
            else:
                found.extend(_walk_json_dates(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_json_dates(child))
    return found


class DuckDuckGoChallengeError(RuntimeError):
    """Raised when DuckDuckGo serves a human-verification page."""


class DuckDuckGoUnexpectedResponseError(RuntimeError):
    """Raised when a search response is neither results nor an explicit empty page."""


_DUCKDUCKGO_CHALLENGE_MARKERS = (
    "anomaly.js",
    "challenge-form",
    "anomaly-modal",
    "bots use duckduckgo",
    "complete the following challenge",
)
_DUCKDUCKGO_EMPTY_MARKERS = (
    "no results",
    "did not match any documents",
    "no more results",
)


def parse_duckduckgo_results(html_text: str) -> list[tuple[str, str]]:
    """Extract result links from DuckDuckGo's HTML endpoint."""
    parser = _ReadableHTMLParser()
    parser.feed(html_text)

    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for title, raw_href in parser.result_links:
        href = _normalize_duckduckgo_href(raw_href)
        if not href or href in seen:
            continue
        seen.add(href)
        results.append((title, href))
    return results


def _normalize_duckduckgo_href(href: str) -> str:
    href = unescape(href)
    parsed = urllib_parse.urlparse(href)
    if parsed.scheme in {"http", "https"} and "duckduckgo.com" not in parsed.netloc:
        try:
            return _ensure_public_http_url(href, resolve_dns=False)
        except UnsafeExternalUrlError:
            return ""
    query = urllib_parse.parse_qs(parsed.query)
    uddg = query.get("uddg", [""])[0]
    if uddg:
        try:
            return _ensure_public_http_url(
                urllib_parse.unquote(uddg),
                resolve_dns=False,
            )
        except UnsafeExternalUrlError:
            return ""
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = urllib_parse.urljoin("https://duckduckgo.com", href)
    parsed = urllib_parse.urlparse(href)
    if parsed.scheme in {"http", "https"} and "duckduckgo.com" not in parsed.netloc:
        try:
            return _ensure_public_http_url(href, resolve_dns=False)
        except UnsafeExternalUrlError:
            return ""
    return ""


def fetch_url(
    url: str,
    *,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    max_bytes: int = 2_000_000,
    source_query: str = "",
) -> ExternalCollectedItem:
    """Fetch one URL and normalize its text content."""
    url = _ensure_public_http_url(url, resolve_dns=True)
    req = urllib_request.Request(url, headers={"User-Agent": user_agent})
    opener = urllib_request.build_opener(_SameOriginAuthorizationRedirectHandler())
    try:
        with opener.open(req, timeout=timeout) as response:
            raw = response.read(max_bytes + 1)
            status = int(getattr(response, "status", 200) or 200)
            geturl = getattr(response, "geturl", None)
            final_url = str(geturl() if callable(geturl) else url)
            content_type = response.headers.get("content-type", "")
            charset = response.headers.get_content_charset() or "utf-8"
            http_last_modified = response.headers.get("last-modified", "")
    except urllib_error.HTTPError as exc:
        raw = exc.read(max_bytes + 1)
        status = int(exc.code)
        final_url = str(exc.geturl() or url)
        content_type = exc.headers.get("content-type", "")
        charset = exc.headers.get_content_charset() or "utf-8"
        http_last_modified = exc.headers.get("last-modified", "")
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    truncated = len(raw) > max_bytes
    raw = raw[:max_bytes]
    decoded = raw.decode(charset, errors="replace")
    metadata: dict[str, Any] = {"truncated": truncated, "final_url": final_url}
    if "html" in content_type.lower() or "<html" in decoded[:1000].lower():
        title, text = html_to_text(decoded)
        published_at, updated_at = extract_html_dates(decoded)
        if published_at:
            metadata["source_published_at"] = published_at
        if updated_at:
            metadata["source_updated_at"] = updated_at
    else:
        title = ""
        text = " ".join(decoded.split())
    normalized_http_date = _normalize_timestamp(http_last_modified)
    if normalized_http_date:
        metadata["http_last_modified"] = normalized_http_date

    return ExternalCollectedItem(
        url=url,
        title=title,
        text=text,
        status=status,
        content_type=content_type,
        source_query=source_query,
        metadata=metadata,
    )


def search_duckduckgo(
    query: str,
    *,
    limit: int = 10,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[tuple[str, str]]:
    """Search DuckDuckGo HTML and return (title, url) result pairs."""
    encoded_query = urllib_parse.urlencode({"q": query})
    requests = (
        urllib_request.Request(
            DEFAULT_SEARCH_ENDPOINT,
            data=encoded_query.encode("utf-8"),
            headers={
                "User-Agent": user_agent,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        ),
        urllib_request.Request(
            f"{DEFAULT_SEARCH_ENDPOINT}?{encoded_query}",
            headers={"User-Agent": user_agent},
            method="GET",
        ),
        urllib_request.Request(
            f"{DUCKDUCKGO_LITE_SEARCH_ENDPOINT}?{encoded_query}",
            headers={"User-Agent": user_agent},
            method="GET",
        ),
    )
    explicit_empty_response = False
    challenge_error: Exception | None = None
    protocol_error: Exception | None = None
    transport_error: Exception | None = None
    deadline = time.monotonic() + max(float(timeout), 0.1)
    for req in requests:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            transport_error = TimeoutError(
                "DuckDuckGo search exhausted its total timeout budget"
            )
            break
        try:
            with urllib_request.urlopen(req, timeout=remaining) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                html_text = response.read(1_000_000).decode(charset, errors="replace")
        except Exception as exc:
            transport_error = exc
            continue
        lowered = html_text.casefold()
        if any(marker in lowered for marker in _DUCKDUCKGO_CHALLENGE_MARKERS):
            challenge_error = DuckDuckGoChallengeError(
                "DuckDuckGo returned a human-verification challenge"
            )
            continue
        results = parse_duckduckgo_results(html_text)[:limit]
        if results:
            return results
        if any(marker in lowered for marker in _DUCKDUCKGO_EMPTY_MARKERS):
            explicit_empty_response = True
            continue
        protocol_error = DuckDuckGoUnexpectedResponseError(
            "DuckDuckGo returned an unrecognized response without search results"
        )
    if challenge_error is not None:
        raise challenge_error
    if protocol_error is not None:
        raise protocol_error
    if explicit_empty_response:
        return []
    if transport_error is not None:
        raise transport_error
    return []


def search_serpapi_google(
    query: str,
    *,
    api_key: str,
    limit: int = 10,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[tuple[str, str]]:
    """Discover URLs from SerpAPI Google organic results.

    Only ``organic_results[*].title`` and ``organic_results[*].link`` are
    retained. Search snippets, answer boxes, AI overviews, and SerpAPI metadata
    are deliberately ignored and never become source evidence.
    """
    if not api_key.strip():
        raise ValueError("SerpAPI Google search requires an API key")
    if limit <= 0:
        return []
    try:
        response = _http_get_json(
            SERPAPI_SEARCH_ENDPOINT,
            params={
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "output": "json",
                "num": limit,
            },
            timeout=timeout,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
            same_origin_redirects_only=True,
        )
    except Exception as exc:
        # SerpAPI authenticates with a query parameter. Never propagate a
        # transport exception whose URL may contain the credential.
        raise RuntimeError(
            f"SerpAPI Google search request failed ({type(exc).__name__})"
        ) from None

    if not isinstance(response, dict):
        raise RuntimeError("SerpAPI Google search returned an invalid response")
    search_metadata = response.get("search_metadata", {})
    status = (
        str(search_metadata.get("status", "")).strip().casefold()
        if isinstance(search_metadata, dict)
        else ""
    )
    if response.get("error") or status == "error":
        raise RuntimeError("SerpAPI Google search returned an error")

    organic_results = response.get("organic_results", [])
    if not isinstance(organic_results, list):
        raise RuntimeError("SerpAPI Google search returned invalid organic results")
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for result in organic_results:
        if not isinstance(result, dict):
            continue
        uri = str(result.get("link", "")).strip()
        title = str(result.get("title", "")).strip()
        if api_key in title or api_key in urllib_parse.unquote(uri):
            continue
        try:
            uri = _ensure_public_http_url(uri, resolve_dns=False)
        except UnsafeExternalUrlError:
            continue
        parsed = urllib_parse.urlsplit(uri)
        key = uri.casefold().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        results.append((title or parsed.netloc, uri))
        if len(results) >= limit:
            return results
    if not results:
        raise RuntimeError("SerpAPI Google search returned no organic result URLs")
    return results


def search_bocha_web(
    query: str,
    *,
    api_key: str,
    base_url: str = BOCHA_SEARCH_BASE_URL,
    freshness: str = "noLimit",
    summary: bool = False,
    limit: int = 10,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[tuple[str, str]]:
    """Discover public URLs from Bocha Web Search structured results."""
    if not api_key.strip():
        raise ValueError("Bocha Web Search requires an API key")
    if limit <= 0:
        return []
    normalized_base_url = str(base_url).strip().rstrip("/")
    if not normalized_base_url:
        raise ValueError("Bocha Web Search requires a base URL")
    endpoint = (
        normalized_base_url
        if normalized_base_url.casefold().endswith("/web-search")
        else normalized_base_url + "/web-search"
    )
    try:
        endpoint = _ensure_public_http_url(endpoint, resolve_dns=False)
        response = _http_post_json(
            endpoint,
            payload={
                "query": query,
                "freshness": str(freshness).strip() or "noLimit",
                "summary": bool(summary),
                "count": min(int(limit), 50),
            },
            timeout=timeout,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
    except Exception as exc:
        # Never propagate transport details that could contain credentials.
        raise RuntimeError(
            f"Bocha Web Search request failed ({type(exc).__name__})"
        ) from None

    if not isinstance(response, dict):
        raise RuntimeError("Bocha Web Search returned an invalid response")

    # Bocha's current API wraps the SearchResponse in ``data`` and reports a
    # business-level status code alongside it. Keep accepting the legacy
    # top-level SearchResponse so existing compatible gateways continue to
    # work.
    response_code = response.get("code")
    if response_code is not None:
        try:
            response_ok = int(response_code) == 200
        except (TypeError, ValueError):
            response_ok = False
        if not response_ok:
            # Do not include the provider message because it may echo a key or
            # other request metadata.
            raise RuntimeError("Bocha Web Search returned an unsuccessful response")

    search_response: dict[str, Any] = response
    if "data" in response:
        data = response.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("Bocha Web Search returned invalid response data")
        search_response = data

    web_pages = search_response.get("webPages", {})
    if not isinstance(web_pages, dict):
        raise RuntimeError("Bocha Web Search returned invalid web pages")
    values = web_pages.get("value", [])
    if not isinstance(values, list):
        raise RuntimeError("Bocha Web Search returned invalid result items")

    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for result in values:
        if not isinstance(result, dict):
            continue
        uri = str(result.get("url", "")).strip()
        title = str(result.get("name", "")).strip()
        if api_key in title or api_key in urllib_parse.unquote(uri):
            continue
        try:
            uri = _ensure_public_http_url(uri, resolve_dns=False)
        except UnsafeExternalUrlError:
            continue
        parsed = urllib_parse.urlsplit(uri)
        key = uri.casefold().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        results.append((title or parsed.netloc, uri))
        if len(results) >= limit:
            break
    return results


def search_google_grounding(
    query: str,
    *,
    api_key: str,
    limit: int = 10,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[tuple[str, str]]:
    """Discover URLs through Gemini Grounding with Google Search.

    Only structured grounding URLs and titles are returned. Model-authored text
    and snippets are deliberately ignored and never become source evidence.
    """
    if not api_key.strip():
        raise ValueError("Google Search grounding requires an API key")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Use Google Search to find primary technical pages that "
                            "directly document the following AI-safety red-teaming "
                            f"topic. Ground the answer in sources. Query: {query}"
                        )
                    }
                ],
            }
        ],
        "tools": [{"google_search": {}}],
        "generationConfig": {"maxOutputTokens": 256},
    }
    response = _http_post_json(
        GOOGLE_GROUNDING_ENDPOINT,
        payload=payload,
        timeout=timeout,
        headers={
            "User-Agent": user_agent,
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
        },
    )
    candidates = response.get("candidates", []) if isinstance(response, dict) else []
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        grounding = candidate.get("groundingMetadata", {})
        chunks = (
            grounding.get("groundingChunks", [])
            if isinstance(grounding, dict)
            else []
        )
        for chunk in chunks:
            web = chunk.get("web", {}) if isinstance(chunk, dict) else {}
            uri = str(web.get("uri", "")).strip() if isinstance(web, dict) else ""
            title = str(web.get("title", "")).strip() if isinstance(web, dict) else ""
            parsed = urllib_parse.urlsplit(uri)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            try:
                uri = _ensure_public_http_url(uri, resolve_dns=False)
            except UnsafeExternalUrlError:
                continue
            key = uri.casefold().rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            results.append((title or parsed.netloc, uri))
            if len(results) >= limit:
                return results
    if not results:
        raise RuntimeError("Google Search grounding returned no web source URLs")
    return results


def expand_search_queries(
    *,
    backend_config: dict[str, Any],
    project_root: Path,
    query: str,
    max_queries: int = 8,
) -> list[str]:
    """Use the meta-model to generate open-vocabulary search queries."""
    if max_queries <= 1 or not bool(backend_config.get("enabled", False)):
        return [query]

    existing = _skill_name_description_summary(project_root, limit=20)
    system_prompt = (
        "You generate web-search queries for finding public text about prompt "
        "rewriting patterns. Use open-vocabulary concepts such as framing, format, "
        "role, discourse pattern, structural wrapper, and completion pattern. "
        "Do not rely on a fixed keyword list. Return strict JSON only."
    )
    user_payload = {
        "task": "expand_external_skill_search_queries",
        "base_query": query,
        "existing_skill_summary": existing,
        "required_output_schema": {
            "artifacts": {
                "queries": f"array of 5 to {max_queries} concise web-search query strings",
            },
            "rationale": "string",
        },
    }
    try:
        artifacts, _rationale, _metadata = generate_meta_artifact(
            backend_config=backend_config,
            system_prompt=system_prompt,
            user_payload=user_payload,
        )
    except Exception:
        return [query]

    raw_queries = artifacts.get("queries", [])
    queries = [query]
    if isinstance(raw_queries, list):
        queries.extend(str(item).strip() for item in raw_queries if str(item).strip())
    elif isinstance(raw_queries, str):
        queries.extend(
            part.strip() for part in re.split(r"[\n;]+", raw_queries) if part.strip()
        )
    return _dedupe_preserve_order(queries)[:max_queries]


def collect_external_text(
    *,
    urls: list[str] | None = None,
    query: str = "",
    backend_config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    expand_query: bool = True,
    search_limit: int = 20,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    delay_seconds: float = 0.0,
) -> list[ExternalCollectedItem]:
    """Collect external text from explicit URLs or search results."""
    target_urls: list[tuple[str, str]] = [
        (url, "") for url in (urls or []) if url.strip()
    ]
    if query:
        queries = [query]
        if expand_query and backend_config is not None and project_root is not None:
            queries = expand_search_queries(
                backend_config=backend_config,
                project_root=project_root,
                query=query,
                max_queries=8,
            )
        seen_urls = {url for url, _source_query in target_urls}
        remaining = max(search_limit - len(target_urls), 0)
        for expanded_query in queries:
            if remaining <= 0:
                break
            results = search_duckduckgo(
                expanded_query,
                limit=remaining,
                timeout=timeout,
                user_agent=user_agent,
            )
            for _title, result_url in results:
                if result_url in seen_urls:
                    continue
                seen_urls.add(result_url)
                target_urls.append((result_url, expanded_query))
                remaining -= 1
                if remaining <= 0:
                    break

    items: list[ExternalCollectedItem] = []
    for index, (url, source_query) in enumerate(target_urls):
        try:
            item = fetch_url(
                url,
                timeout=timeout,
                user_agent=user_agent,
                source_query=source_query,
            )
            if item.text.strip():
                items.append(item)
        except Exception as exc:
            items.append(
                ExternalCollectedItem(
                    url=url,
                    text="",
                    status=0,
                    source_query=source_query,
                    metadata={"error": str(exc)},
                )
            )
        if delay_seconds > 0 and index + 1 < len(target_urls):
            time.sleep(delay_seconds)
    return items


def read_urls_file(path: Path) -> list[str]:
    """Read one URL per line from a text file."""
    urls: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def write_crawl_snapshot(items: list[ExternalCollectedItem], out_path: Path) -> Path:
    """Write collected items as JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def default_snapshot_path(project_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / DEFAULT_CRAWL_DIR / f"external_crawl_{timestamp}.jsonl"


def _skill_name_description_summary(
    project_root: Path, limit: int
) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    try:
        specs = SkillLoader(project_root).discover()
    except Exception:
        return summaries
    for spec in specs[:limit]:
        summaries.append({"name": spec.name, "description": spec.description})
    return summaries


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = " ".join(value.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _normalize_timestamp(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _first_normalized_timestamp(values: list[str]) -> str:
    for value in values:
        normalized = _normalize_timestamp(value)
        if normalized:
            return normalized
    return ""


def _resolve_as_of(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    else:
        normalized = _normalize_timestamp(value)
        if not normalized:
            raise ValueError(f"Invalid as_of timestamp: {value}")
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _render_query_with_cutoff(
    source: str,
    query: str,
    cutoff: datetime,
    as_of: datetime,
) -> str:
    cutoff_date = cutoff.strftime("%Y-%m-%d")
    if source == "github" and not re.search(r"\bcreated\s*:", query, re.IGNORECASE):
        return f"{query} fork:false archived:false created:>={cutoff_date}"
    if source == "arxiv" and "submittedDate:" not in query:
        start = cutoff.strftime("%Y%m%d0000")
        end = as_of.strftime("%Y%m%d2359")
        return f"({query}) AND submittedDate:[{start} TO {end}]"
    if source == "google" and not re.search(
        r"\bafter:", query, re.IGNORECASE
    ):
        return f"{query} after:{cutoff_date}"
    return query


def _render_arxiv_native_query(query: str) -> str:
    """Convert one plain discovery query to conservative arXiv field syntax."""
    normalized = " ".join(str(query).split())
    if re.search(r"\b(?:all|ti|abs|au|cat):", normalized):
        return normalized
    terms = [
        term
        for term in re.findall(r'"[^"\r\n]+"|[A-Za-z0-9][A-Za-z0-9_-]*', normalized)
        if term.casefold() not in {"and", "or", "not"}
    ]
    return " AND ".join(f"all:{term}" for term in terms) or normalized


def _annotate_freshness(
    raw_item: "RawCollectedItem",
    *,
    as_of: datetime,
    cutoff: datetime,
) -> "RawCollectedItem":
    metadata = dict(raw_item.metadata or {})
    metadata["freshness_evaluated_at"] = as_of.isoformat().replace("+00:00", "Z")
    metadata["freshness_cutoff"] = cutoff.isoformat().replace("+00:00", "Z")
    if metadata.get("skipped_reason") or metadata.get("error"):
        metadata.update(
            {"freshness_eligible": False, "freshness_reason": "collector_skipped"}
        )
        return replace(raw_item, metadata=metadata)

    effective_at = _normalize_timestamp(
        metadata.get("source_effective_at")
        or metadata.get("source_updated_at")
        or metadata.get("source_published_at")
    )
    if not effective_at:
        metadata.update(
            {"freshness_eligible": False, "freshness_reason": "missing_source_date"}
        )
        return replace(raw_item, metadata=metadata)

    effective_dt = datetime.fromisoformat(effective_at.replace("Z", "+00:00"))
    age_days = max(0.0, (as_of - effective_dt).total_seconds() / 86400)
    eligible = effective_dt >= cutoff and effective_dt <= as_of + timedelta(days=1)
    metadata.update(
        {
            "source_effective_at": effective_at,
            "source_age_days": round(age_days, 3),
            "freshness_eligible": eligible,
            "freshness_reason": "within_window"
            if eligible
            else "outside_freshness_window",
        }
    )
    return replace(raw_item, metadata=metadata)


def _github_api_headers(
    *,
    accept: str = "application/vnd.github+json",
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, str]:
    headers = {
        "Accept": accept,
        "User-Agent": user_agent,
        "X-GitHub-Api-Version": _GITHUB_API_VERSION,
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _huggingface_headers(
    *, user_agent: str = DEFAULT_USER_AGENT
) -> dict[str, str]:
    """Build Hugging Face headers without exposing token values in records."""
    headers = {"User-Agent": user_agent}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_raw_readme_urls(repo_name: str, default_branch: str) -> list[str]:
    branch = default_branch.strip()
    if not branch:
        return []
    return [
        f"https://raw.githubusercontent.com/{repo_name}/{branch}/{filename}"
        for filename in _GITHUB_README_FALLBACK_FILENAMES
    ]


def _classify_github_evidence_path(path: str, size_bytes: int) -> tuple[str, int] | None:
    """Classify a safe text path for role-balanced, read-only evidence retrieval."""
    normalized = path.strip().replace("\\", "/")
    lowered = normalized.casefold()
    parts = tuple(part for part in lowered.split("/") if part)
    filename = parts[-1] if parts else ""
    if not filename or any(part in _GITHUB_EVIDENCE_EXCLUDED_PARTS for part in parts):
        return None
    if filename.startswith("readme") or filename in {
        "license",
        "license.md",
        "changelog.md",
        "citation.cff",
    }:
        return None
    if any(
        marker in lowered
        for marker in (
            ".env",
            "credential",
            "private-key",
            "private_key",
            "secret",
            "token",
            "id_rsa",
        )
    ):
        return None
    suffix = Path(filename).suffix.casefold()
    if suffix not in _GITHUB_EVIDENCE_SUFFIXES:
        return None
    if size_bytes <= 0 or size_bytes > _GITHUB_EVIDENCE_MAX_FILE_BYTES:
        return None

    matches: list[tuple[int, str]] = []
    for role, markers in _EVIDENCE_ROLE_MARKERS.items():
        hits = sum(marker in lowered for marker in markers)
        if hits:
            role_bonus = {
                "mechanism": 5,
                "implementation": 4,
                "evaluation": 3,
                "examples": 2,
                "domain-evidence": 1,
            }.get(role, 0)
            matches.append((hits * 10 + role_bonus, role))
    if not matches:
        return None
    score, role = max(matches)
    if suffix == ".py" and not any(
        marker in lowered
        for marker in ("prompt", "template", "attack", "rewrite", "eval", "example")
    ):
        return None
    score += max(0, 4 - normalized.count("/"))
    return role, score


def _select_github_evidence_candidates(
    tree_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select bounded GitHub blobs in role round-robin order."""
    by_role: dict[str, list[dict[str, Any]]] = {}
    raw_tree = tree_payload.get("tree", []) if isinstance(tree_payload, dict) else []
    for entry in raw_tree:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        path = str(entry.get("path", ""))
        size = int(entry.get("size", 0) or 0)
        classified = _classify_github_evidence_path(path, size)
        if classified is None:
            continue
        role, score = classified
        candidate = {
            "path": path,
            "sha": str(entry.get("sha", "")),
            "size": size,
            "role": role,
            "score": score,
        }
        if candidate["sha"]:
            by_role.setdefault(role, []).append(candidate)

    for values in by_role.values():
        values.sort(key=lambda value: (-int(value["score"]), str(value["path"])))
    role_order = (
        "mechanism",
        "implementation",
        "examples",
        "evaluation",
        "domain-evidence",
    )
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < _GITHUB_EVIDENCE_MAX_FILES:
        added = False
        for role in role_order:
            values = by_role.get(role, [])
            if index < len(values):
                selected.append(values[index])
                added = True
                if len(selected) >= _GITHUB_EVIDENCE_MAX_FILES:
                    break
        if not added:
            break
        index += 1
    return selected


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(value.casefold() in lowered for value in values)


_SOURCE_AUTHORSHIP_PATTERN = re.compile(
    r"\b(?:we|this\s+(?:paper|work|study))\s+"
    r"(?:propos(?:e|ed)|introduc(?:e|ed)|present(?:s|ed)?|develop(?:s|ed)?|"
    r"design(?:s|ed)?|adapt(?:s|ed)?|extend(?:s|ed)?|combin(?:e|es|ed))\b",
    re.IGNORECASE,
)
_SOURCE_OWNED_METHOD_PATTERN = re.compile(
    r"\bour\s+(?:(?:new|novel|proposed|introduced|developed|designed|adapted|"
    r"extended|combined)\s+){0,2}(?:attack|jailbreak|method|technique|mechanism|"
    r"algorithm|approach|procedure|pipeline)\b",
    re.IGNORECASE,
)
_SOURCE_OWNED_DEFENSE_OBJECTIVE_PATTERN = re.compile(
    r"\b(?:for\s+defending|defen[cs]e\s+objective|resulting\s+defen[cs]e|"
    r"universal\s+defen[cs]e|system[- ]level\s+defen[cs]es?|"
    r"jailbreaking\s+defen[cs]e|enforc\w*\s+safe\s+outputs?|"
    r"ensur\w*\s+harmless\s+(?:outputs?|responses?)|maintain\w*\s+refusal)\b",
    re.IGNORECASE,
)
_ATTACK_METHOD_CLAIM_PATTERN = re.compile(
    r"\b(?:jailbreak(?:s|ing)?|prompt[- ]attack|adversarial[- ]prompts?|"
    r"prompt[- ]rewrite|guardrail[- ]bypass|safety[- ]bypass|red[- ]team(?:ing)?)\b",
    re.IGNORECASE,
)
_METHOD_OBJECT_PATTERN = re.compile(
    r"\b(?:attack|jailbreak(?:s|ing)?|method|technique|mechanism|algorithm|approach|"
    r"procedure|pipeline)\b",
    re.IGNORECASE,
)
_METHOD_OPERATION_PATTERN = re.compile(
    r"\b(?:construct|craft|generat|rewrit|transform|optimiz|optimis|compos|"
    r"append|prepend|insert|select|search|perturb|encod|obfuscat)\w*\b",
    re.IGNORECASE,
)
_GENERIC_RESEARCH_OBJECT_PATTERN = re.compile(
    r"^\s*(?:a|an|the|our)?\s*(?:new\s+|novel\s+)?"
    r"(?:[a-z0-9][a-z0-9_-]*\s+){0,4}"
    r"(?:benchmark|dataset|corpus|survey|taxonomy|metric|index|score|evaluation|"
    r"analysis|risk profile|threat model|defen[cs]e objective)\b",
    re.IGNORECASE,
)
_BENCHMARK_PRIMARY_PATTERN = re.compile(
    r"\b(?:benchmark(?:ing)?|dataset|risk profile|empirical analysis|"
    r"comparative evaluation|safety evaluation|evaluation framework)\b",
    re.IGNORECASE,
)
_METHOD_SOURCE_ATTRIBUTION_PATTERN = re.compile(
    r"\b(?:ideas?|behaviou?rs?|prompts?|examples?|artifacts?)\b.{0,100}"
    r"\b(?:sourc|deriv|adapt|draw)\w*\s+from\b",
    re.IGNORECASE | re.DOTALL,
)
_ARXIV_CITATION_PATTERN = re.compile(
    r"(?i)(?:(?:https?://(?:www\.)?arxiv\.org)?/(?:abs|html|pdf)/|"
    r"arxiv\s*:\s*)"
    r"((?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*(?:\.[A-Z]{2})?/\d{7})"
    r"(?:v\d+)?)(?:\.pdf)?"
)


def _source_authored_attack_method_claim(text: str) -> bool:
    """Require a local author-owned attack-method claim, not a generic contribution."""
    body = _query_body_without_package_headers(text)
    if any(
        pattern.search(body[:1_000])
        for pattern in _TEXT_ATTACK_DEFENSIVE_TITLE_PATTERNS
    ):
        return False
    claims = [
        (match, True, True)
        for match in _SOURCE_AUTHORSHIP_PATTERN.finditer(body)
    ] + [
        (
            match,
            False,
            bool(
                re.search(
                    r"\b(?:new|novel|proposed|introduced|developed|designed|"
                    r"adapted|extended|combined)\b",
                    match.group(0),
                    flags=re.IGNORECASE,
                )
            ),
        )
        for match in _SOURCE_OWNED_METHOD_PATTERN.finditer(body)
    ]
    for claim, has_claim_verb, allow_following_sentence in claims:
        sentence_start = max(
            body.rfind(".", 0, claim.start()),
            body.rfind("!", 0, claim.start()),
            body.rfind("?", 0, claim.start()),
            body.rfind("\n", 0, claim.start()),
        ) + 1
        sentence_end_match = re.search(r"[.!?](?:\s|$)", body[claim.end() :])
        sentence_end = (
            claim.end() + sentence_end_match.end()
            if sentence_end_match
            else min(len(body), claim.end() + 500)
        )
        if allow_following_sentence:
            following_end_match = re.search(
                r"[.!?](?:\s|$)", body[sentence_end:]
            )
            if following_end_match:
                sentence_end += following_end_match.end()
        context = " ".join(
            body[sentence_start : min(sentence_end, claim.end() + 700)].split()
        )
        if _SOURCE_OWNED_DEFENSE_OBJECTIVE_PATTERN.search(context):
            continue
        if not _ATTACK_METHOD_CLAIM_PATTERN.search(context):
            continue
        if not (
            _METHOD_OBJECT_PATTERN.search(context)
            and _METHOD_OPERATION_PATTERN.search(context)
        ):
            continue
        if has_claim_verb:
            claim_tail = body[claim.end() : claim.end() + 240]
            if _GENERIC_RESEARCH_OBJECT_PATTERN.search(claim_tail):
                continue
        return True
    return False


def _plain_citation_context(value: str) -> str:
    without_tags = re.sub(r"(?is)<[^>]+>", " ", str(value))
    return " ".join(unescape(without_tags).split())


def _citation_context_window(text: str, start: int, end: int) -> str:
    lower_bound = max(0, start - 500)
    upper_bound = min(len(text), end + 500)
    prefix = text[lower_bound:start]
    suffix = text[end:upper_bound]
    # PDF extraction commonly puts ``arXiv preprint`` on a fresh line after
    # the cited title, while arXiv HTML wraps the same phrase in inline tags.
    # Newlines/tags are therefore presentation details, not reference
    # boundaries.  Sentence punctuation is the stable local delimiter.
    boundary = re.compile(r"(?is)[.!?;](?:\s+|(?=<))")
    prefix_matches = list(boundary.finditer(prefix))
    if prefix_matches:
        selected_boundary_index = -1
        trailing_fragment = _plain_citation_context(
            prefix[prefix_matches[-1].end() :]
        ).casefold()
        if (
            len(prefix_matches) >= 2
            and re.fullmatch(
                r"(?:(?:arxiv\s+)?preprint|arxiv|technical\s+report)\s*",
                trailing_fragment,
            )
        ):
            # Include exactly one preceding sentence: bibliography entries
            # often format ``Paper title. arXiv preprint arXiv:...``.
            selected_boundary_index = -2
        lower_bound += prefix_matches[selected_boundary_index].end()
    suffix_match = boundary.search(suffix)
    if suffix_match:
        upper_bound = end + suffix_match.start()
    return text[lower_bound:upper_bound]


def _ranked_attack_method_arxiv_citations(
    text: str,
    *,
    current_arxiv_id: str = "",
) -> list[dict[str, Any]]:
    """Extract bounded, locally attack-method-relevant arXiv citations."""
    current_id = ""
    if current_arxiv_id:
        try:
            current_id = normalize_arxiv_id(current_arxiv_id)
        except PaperBundleValidationError:
            current_id = ""
    best_by_id: dict[str, dict[str, Any]] = {}
    raw_text = str(text)
    for match in _ARXIV_CITATION_PATTERN.finditer(raw_text):
        try:
            canonical_id = normalize_arxiv_id(match.group(1))
        except PaperBundleValidationError:
            continue
        if canonical_id == current_id:
            continue
        context = _plain_citation_context(
            _citation_context_window(raw_text, match.start(), match.end())
        )
        attack_hit = bool(_ATTACK_METHOD_CLAIM_PATTERN.search(context))
        method_hit = bool(_METHOD_OBJECT_PATTERN.search(context))
        operation_hit = bool(_METHOD_OPERATION_PATTERN.search(context))
        source_attribution_hit = bool(
            _METHOD_SOURCE_ATTRIBUTION_PATTERN.search(context)
        )
        if not attack_hit or not (
            method_hit and (operation_hit or source_attribution_hit)
        ):
            continue
        generic_hit = bool(_BENCHMARK_PRIMARY_PATTERN.search(context))
        score = (
            4
            + 2 * int(method_hit)
            + 3 * int(operation_hit)
            + 2 * int(source_attribution_hit)
            - int(generic_hit)
        )
        record = {
            "arxiv_id": canonical_id,
            "score": score,
            "context": context[:700],
            "explicit_source_attribution": source_attribution_hit,
            "extraction_origin": "body_local",
            "evidence_terms": list(
                dict.fromkeys(
                    value
                    for value in (
                        _query_body_first_pattern_match(
                            context, (_ATTACK_METHOD_CLAIM_PATTERN,)
                        ),
                        _query_body_first_pattern_match(
                            context, (_METHOD_OBJECT_PATTERN,)
                        ),
                        _query_body_first_pattern_match(
                            context, (_METHOD_OPERATION_PATTERN,)
                        ),
                    )
                    if value
                )
            ),
        }
        existing = best_by_id.get(canonical_id)
        if existing is None or int(record["score"]) > int(existing["score"]):
            best_by_id[canonical_id] = record
    return sorted(
        best_by_id.values(),
        key=lambda value: (-int(value["score"]), str(value["arxiv_id"])),
    )[:MAX_PAPER_CITATION_CANDIDATES_PER_PRIMARY]


def _has_quantified_attack_result(text: str) -> bool:
    """Return whether an attack/safety metric is paired with a measured value."""
    metric = (
        r"(?:attack[- ]success(?: rate)?|asr|jailbreak[- ]success(?: rate)?|"
        r"bypass rate|refusal rate|harmful(?: completion| response| output)? rate|"
        r"robustness|robust accuracy)"
    )
    formatted_value = (
        r"(?:\d{1,3}(?:\.\d+)?\s*%|0?\.\d+|"
        r"\d{1,3}(?:\.\d+)?\s*(?:percentage\s+points?|points?))"
    )
    relation = (
        r"(?:=|:|was|is|at|of|to|by|reached|achieved|measured at|"
        r"improved to|improved by|increased to|increased by|"
        r"decreased to|decreased by|dropped to|dropped by)"
    )
    bare_related_value = r"\d{1,3}(?:\.\d+)?\s*%?"
    return bool(
        re.search(
            rf"\b{metric}\b.{{0,48}}\b{relation}\s+{bare_related_value}",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        or re.search(
            rf"\b{metric}\b\s*[=:\u2191\u2193]\s*{bare_related_value}",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"{formatted_value}.{{0,48}}\b{metric}\b",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def _infer_evidence_roles(text: str, *, path: str = "") -> list[str]:
    """Infer broad evidence roles without treating them as mechanism labels."""
    searchable = f"{path}\n{text[:12000]}".casefold()
    roles = [
        role
        for role, markers in _EVIDENCE_ROLE_MARKERS.items()
        if any(marker in searchable for marker in markers)
    ]
    return roles or ["overview"]


def _evidence_quality_metadata(
    text: str,
    documents: list[SourceEvidenceDocument],
    *,
    extraction_method: str,
    authenticated: bool = False,
    source_group: str = "",
    revision: str = "",
) -> dict[str, Any]:
    """Return a source-neutral quality gate and a serializable evidence manifest."""
    rendered_documents = _rendered_evidence_documents(text, documents)
    declared_roles = list(
        dict.fromkeys(
            document.role for document in rendered_documents if document.role
        )
    )
    content_roles = list(
        dict.fromkeys(
            role
            for document in rendered_documents
            for role in _infer_evidence_roles(document.text)
            if role != "overview"
        )
    )
    roles = list(dict.fromkeys([*declared_roles, *content_roles]))
    content_text = "\n\n".join(document.text for document in rendered_documents)
    lowered = content_text.casefold()
    method_count = sum(marker in lowered for marker in _GITHUB_METHOD_MARKERS)
    transformation_count = sum(
        marker in lowered for marker in _GITHUB_TRANSFORMATION_MARKERS
    )
    strong_method_hit = _contains_any(
        content_text,
        (
            "method",
            "algorithm",
            "pipeline",
            "optimization",
            "implementation",
            "reproduce",
            "approach",
            "technique",
            "mechanism",
        ),
    )
    security_hit = _contains_any(content_text, _GITHUB_SECURITY_ANCHORS)
    cyber_only = (
        sum(marker in lowered for marker in _CYBER_ONLY_MARKERS) >= 2
        and not _contains_any(content_text, _AI_MODEL_ANCHORS)
    )
    mechanism_hit = strong_method_hit and method_count >= 2
    actionable_hit = transformation_count >= 2
    evaluation_hit = _contains_any(
        content_text, _EVIDENCE_ROLE_MARKERS["evaluation"]
    )
    domain_hit = _contains_any(
        content_text, _EVIDENCE_ROLE_MARKERS["domain-evidence"]
    )
    component_analysis_hit = bool(
        re.search(
            r"\b(?:ablations?(?: study| analysis| experiment)?|"
            r"component(?:[- ]wise)?[- ](?:analysis|study|evaluation)|"
            r"sensitivity analysis)\b",
            content_text,
            flags=re.IGNORECASE,
        )
    )
    empirical_comparison_hit = bool(
        re.search(
            r"\b(?:empirical(?:ly)?\s+compar(?:e|ed|ison)|"
            r"compar(?:e|ed|ison)\s+(?:to|with|against)|"
            r"baselines?|versus|vs\.?|outperform(?:s|ed|ing)?)\b",
            content_text,
            flags=re.IGNORECASE,
        )
    )
    explicit_procedure_hit = bool(
        re.search(
            r"\b(?:algorithm\s*(?:\d+|one)|pseudocode|step\s*(?:1|one)|"
            r"(?:explicit|iterative|multi[- ]step)\s+procedure)\b",
            content_text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\bprocedure\b.{0,120}\b(?:first|then|next|finally)\b",
            content_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    example_evidence = _assess_paper_example_evidence(content_text)
    quantified_attack_result_hit = _has_quantified_attack_result(content_text)
    source_authored_attack_method_claim = _source_authored_attack_method_claim(
        content_text
    )
    reasons: list[str] = []
    if len(content_text.strip()) < 500:
        reasons.append("insufficient_text")
    if not security_hit:
        reasons.append("missing_red_team_identity")
    if cyber_only:
        reasons.append("unsupported_cyber_only_scope")
    if not mechanism_hit:
        reasons.append("missing_mechanism_evidence")
    if not actionable_hit:
        reasons.append("missing_implementation_evidence")
    eligible = not reasons
    novelty_denied = _source_denies_new_attack_mechanism(content_text)
    packaging_marker_count = _dataset_packaging_marker_count(content_text)
    novelty_claimed = _contains_any(
        content_text,
        (
            "we propose",
            "we introduce",
            "we present a new",
            "we develop a new",
            "novel attack",
            "novel jailbreak",
            "새로운 공격 기법을 제안",
        ),
    )
    dataset_repack_only = bool(
        extraction_method.startswith("huggingface_")
        and packaging_marker_count >= 3
        and not novelty_claimed
    )
    mechanism_extraction_eligible = bool(
        eligible and not novelty_denied and not dataset_repack_only
    )
    if not mechanism_extraction_eligible:
        content_roles = [role for role in content_roles if role != "mechanism"]
        roles = [role for role in roles if role != "mechanism"]
    mechanism_extraction_reason = (
        "source_claims_no_new_attack_mechanism"
        if novelty_denied
        else "dataset_benchmark_or_repack_without_novel_mechanism"
        if dataset_repack_only
        else "quality_gate_failed"
        if not eligible
        else "source_supported_mechanism"
    )
    score = (
        min(len(content_text.strip()) / 2000.0, 2.0)
        + (2.0 if security_hit else 0.0)
        + (2.0 if mechanism_hit else 0.0)
        + (2.0 if actionable_hit else 0.0)
        + (
            1.5
            if example_evidence["status"] == "complete"
            else 0.5
            if example_evidence["status"] == "partial"
            else 0.0
        )
        + min(len(set(roles)), 4) * 0.5
        + min(len(rendered_documents), 4) * 0.25
    )
    return {
        "evidence_quality_eligible": eligible,
        "evidence_quality_score": round(score, 3),
        "evidence_quality_reason": "quality_verified"
        if eligible
        else ",".join(reasons),
        "evidence_roles": roles,
        "evidence_content_roles": content_roles,
        "evidence_document_count": len(rendered_documents),
        "evidence_char_count": len(text),
        "evidence_extraction_method": extraction_method,
        "example_evidence_status": str(example_evidence["status"]),
        "example_evidence_score": int(example_evidence["score"]),
        "example_evidence_signals": list(example_evidence["signals"]),
        "example_evidence_counts": {
            "input_output_pairs": int(example_evidence["io_pair_count"]),
            "user_assistant_transcripts": int(
                example_evidence["transcript_pair_count"]
            ),
            "prompt_templates": int(example_evidence["prompt_template_count"]),
            "worked_examples": int(example_evidence["worked_example_count"]),
            "algorithms": int(example_evidence["algorithm_count"]),
        },
        "source_authenticated": bool(authenticated),
        "source_group": source_group,
        "evidence_role": "package",
        "evidence_path": source_group,
        "source_revision": revision,
        "risk_domain_binding_eligible": bool(
            eligible and evaluation_hit and domain_hit
        ),
        "mechanism_extraction_eligible": mechanism_extraction_eligible,
        "mechanism_extraction_reason": mechanism_extraction_reason,
        "source_authored_attack_method_claim": (
            source_authored_attack_method_claim
        ),
        "advanced_mechanism_eligible": bool(
            mechanism_extraction_eligible
            and source_authored_attack_method_claim
            and evaluation_hit
            and (
                component_analysis_hit
                or (
                    empirical_comparison_hit and quantified_attack_result_hit
                )
                or (explicit_procedure_hit and quantified_attack_result_hit)
            )
        ),
        "advanced_mechanism_reason": (
            "source_authored_attack_method_with_advanced_evaluation"
            if (
                mechanism_extraction_eligible
                and source_authored_attack_method_claim
                and evaluation_hit
                and (
                    component_analysis_hit
                    or (
                        empirical_comparison_hit
                        and quantified_attack_result_hit
                    )
                    or (
                        explicit_procedure_hit
                        and quantified_attack_result_hit
                    )
                )
            )
            else "missing_source_authored_attack_method_claim"
            if not source_authored_attack_method_claim
            else "missing_advanced_evaluation_evidence"
        ),
        "evidence_documents": [
            {
                "path": document.path,
                "role": document.role,
                "url": document.url,
                "revision": document.revision,
                "size_bytes": document.size_bytes,
                "chars": len(document.text),
                "truncated": document.truncated,
                "sha256": hashlib.sha256(
                    document.text.encode("utf-8", errors="replace")
                ).hexdigest(),
                **(
                    {"source_sampling": dict(document.provenance)}
                    if document.provenance
                    else {}
                ),
            }
            for document in rendered_documents
        ],
    }


def _safe_evidence_header_value(value: str, *, max_chars: int = 500) -> str:
    return " ".join(str(value).split())[:max_chars] or "unknown"


def _safe_evidence_body(value: str) -> str:
    """Quote reserved package boundaries found inside untrusted source text."""
    return re.sub(
        r"(?m)^(?=## Evidence document:|Evidence role:)",
        "> ",
        str(value).strip(),
    )


def _rendered_evidence_documents(
    text: str,
    documents: list[SourceEvidenceDocument],
) -> list[SourceEvidenceDocument]:
    """Return only document bodies actually present in the bounded package text."""
    rendered: list[SourceEvidenceDocument] = []
    for document in documents:
        safe_path = _safe_evidence_header_value(document.path)
        safe_role = _safe_evidence_header_value(document.role, max_chars=80)
        marker = f"## Evidence document: {safe_path}\nEvidence role: {safe_role}\n"
        marker_start = text.find(marker)
        if marker_start < 0:
            continue
        body_start = marker_start + len(marker)
        next_start = text.find("\n\n## Evidence document:", body_start)
        body = text[body_start : next_start if next_start >= 0 else len(text)].strip()
        if not body:
            continue
        original_body = _safe_evidence_body(document.text)
        rendered.append(
            replace(
                document,
                path=safe_path,
                role=safe_role,
                text=body,
                truncated=document.truncated or len(body) < len(original_body),
            )
        )
    return rendered


def _render_evidence_package(
    title: str,
    documents: list[SourceEvidenceDocument],
    *,
    max_total_chars: int,
) -> str:
    """Render source documents with stable boundaries for downstream chunking."""
    parts = [f"# Evidence package: {title}"]
    used = len(parts[0])
    for document in documents:
        safe_path = _safe_evidence_header_value(document.path)
        safe_role = _safe_evidence_header_value(document.role, max_chars=80)
        heading = (
            f"\n\n## Evidence document: {safe_path}\n"
            f"Evidence role: {safe_role}\n"
        )
        remaining = max_total_chars - used - len(heading)
        if remaining <= 0:
            break
        body = _safe_evidence_body(document.text)[:remaining]
        if not body:
            continue
        parts.extend([heading, body])
        used += len(heading) + len(body)
    return "".join(parts).strip()


def _query_body_without_package_headers(text: str) -> str:
    """Return normalized source body text without collector-authored headers."""
    normalized = unicodedata.normalize("NFKC", str(text)).replace("\r\n", "\n")
    normalized = re.sub(
        r"(?im)^[ \t]{0,3}#{1,6}[ \t]+Evidence[ \t]+"
        r"(?:package|document)[ \t]*:[^\n]*(?:\n|$)",
        "",
        normalized,
    )
    normalized = re.sub(
        r"(?im)^[ \t]*Evidence[ \t]+role[ \t]*:[^\n]*(?:\n|$)",
        "",
        normalized,
    )
    return normalized.strip()


def _best_verified_bundle_example_evidence(
    bundle: list[ExternalCollectedItem],
) -> tuple[dict[str, Any], str, bool, str]:
    """Choose the strongest concrete rewrite example in a verified paper bundle."""
    ranked: list[
        tuple[tuple[int, int, int], dict[str, Any], str, bool, str]
    ] = []
    status_rank = {"complete": 2, "partial": 1, "none": 0}
    for item in bundle:
        source = str(item.metadata.get("external_source", "")).casefold()
        role = str(item.metadata.get("paper_role", "")).casefold()
        is_primary = bool(
            source == "arxiv"
            and role == "primary"
            and item.metadata.get("paper_relation_verified") is True
        )
        is_verified_companion = bool(
            source in {"github", "huggingface"}
            and role == "companion"
            and item.metadata.get("paper_relation_verified") is True
            and str(item.metadata.get("paper_companion_usage", "")).casefold()
            != "domain_evidence_only"
        )
        if not (is_primary or is_verified_companion):
            continue
        assessment = _assess_paper_example_evidence(
            _query_body_without_package_headers(item.text)
        )
        evidence_source = (
            "primary_bounded_body"
            if is_primary
            else f"verified_companion:{source}"
        )
        truncated = bool(
            item.metadata.get("arxiv_source_bounded")
            or item.metadata.get("arxiv_pdf_extraction_truncated")
            or any(
                bool(document.get("truncated"))
                for document in list(item.metadata.get("evidence_documents", []) or [])
                if isinstance(document, dict)
            )
        )
        bounding_method = str(
            item.metadata.get(
                "arxiv_source_bounding_method",
                "verified_companion_evidence_package"
                if is_verified_companion
                else "role_balanced_example_grounded_v3",
            )
        )
        ranked.append(
            (
                (
                    status_rank.get(str(assessment["status"]), 0),
                    int(assessment["score"]),
                    1 if is_primary else 0,
                ),
                assessment,
                evidence_source,
                truncated,
                bounding_method,
            )
        )
    if not ranked:
        return (
            _assess_paper_example_evidence(""),
            "primary_bounded_body",
            False,
            "role_balanced_example_grounded_v3",
        )
    _rank, assessment, source, truncated, method = max(
        ranked,
        key=lambda value: value[0],
    )
    return assessment, source, truncated, method


def _query_body_pattern_hit(
    text: str, patterns: tuple[re.Pattern[str], ...]
) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _query_body_first_pattern_match(
    text: str, patterns: tuple[re.Pattern[str], ...]
) -> str:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return " ".join(match.group(0).split())
    return ""


def _text_attack_local_pair_evidence(
    text: str,
    first_patterns: tuple[re.Pattern[str], ...],
    second_patterns: tuple[re.Pattern[str], ...],
    *,
    radius: int = _TEXT_ATTACK_SCOPE_LOCAL_RADIUS,
) -> tuple[str, ...]:
    """Return one nearby pair without treating distant related work as scope."""
    for pattern in first_patterns:
        for match in pattern.finditer(text):
            start = max(0, match.start() - radius)
            end = min(len(text), match.end() + radius)
            second = _query_body_first_pattern_match(
                text[start:end], second_patterns
            )
            if second:
                return tuple(
                    dict.fromkeys(
                        (
                            " ".join(match.group(0).split()),
                            second,
                        )
                    )
                )
    return ()


def _assess_text_attack_runtime(
    title: str,
    text: str,
) -> TextAttackRuntimeAssessment:
    """Admit only papers usable for single-turn, text-only attack rewrites.

    The title is authoritative for the paper's primary runtime. Body-level
    exclusions are limited to lead-section scope claims, so a textual jailbreak
    paper is not rejected merely because related work mentions other modalities.
    Detection and defense papers are admitted only when they also describe how
    an offensive prompt is constructed or transformed. A jailbreak benchmark is
    valid textual attack evidence even when it evaluates existing prompts rather
    than introducing a new generator.
    """
    normalized_title = unicodedata.normalize("NFKC", str(title)).casefold()
    body = _query_body_without_package_headers(text).casefold()
    lead = body[:_TEXT_ATTACK_SCOPE_LEAD_CHARS]
    combined = f"{normalized_title}\n{lead}".strip()

    defensive_title_term = _query_body_first_pattern_match(
        normalized_title,
        _TEXT_ATTACK_DEFENSIVE_TITLE_PATTERNS,
    )
    if defensive_title_term:
        return TextAttackRuntimeAssessment(
            False,
            "detection_classification_moderation_or_defense_only",
            (defensive_title_term,),
        )

    unsupported_title_term = _query_body_first_pattern_match(
        normalized_title,
        _TEXT_ATTACK_UNSUPPORTED_RUNTIME_PATTERNS,
    )
    if unsupported_title_term:
        return TextAttackRuntimeAssessment(
            False,
            "unsupported_non_text_or_stateful_runtime_scope",
            (unsupported_title_term,),
        )

    unsupported_scope_evidence = _text_attack_local_pair_evidence(
        lead,
        _TEXT_ATTACK_CORE_CLAIM_PATTERNS,
        _TEXT_ATTACK_UNSUPPORTED_RUNTIME_PATTERNS,
    )
    if unsupported_scope_evidence:
        return TextAttackRuntimeAssessment(
            False,
            "unsupported_non_text_or_stateful_runtime_scope",
            unsupported_scope_evidence,
        )

    strongly_stateful_term = _query_body_first_pattern_match(
        lead[:_TEXT_ATTACK_STRONGLY_STATEFUL_LEAD_CHARS],
        _TEXT_ATTACK_STRONGLY_STATEFUL_RUNTIME_PATTERNS,
    )
    if strongly_stateful_term:
        return TextAttackRuntimeAssessment(
            False,
            "unsupported_non_text_or_stateful_runtime_scope",
            (strongly_stateful_term,),
        )

    construction_evidence = _text_attack_local_pair_evidence(
        combined,
        _TEXT_ATTACK_ANCHOR_PATTERNS,
        _TEXT_ATTACK_CONSTRUCTION_PATTERNS,
    )
    if not construction_evidence:
        construction_evidence = _text_attack_local_pair_evidence(
            combined,
            _TEXT_ATTACK_CONSTRUCTION_PATTERNS,
            _TEXT_ATTACK_ANCHOR_PATTERNS,
        )

    defense_title_term = _query_body_first_pattern_match(
        normalized_title,
        _TEXT_ATTACK_DEFENSE_ONLY_PATTERNS,
    )
    defense_scope_evidence = _text_attack_local_pair_evidence(
        lead,
        _TEXT_ATTACK_CORE_CLAIM_PATTERNS,
        _TEXT_ATTACK_DEFENSE_ONLY_PATTERNS,
    )
    if (
        defense_title_term or defense_scope_evidence
    ) and not construction_evidence:
        evidence_terms = tuple(
            dict.fromkeys(
                term
                for term in (defense_title_term, *defense_scope_evidence)
                if term
            )
        )
        return TextAttackRuntimeAssessment(
            False,
            "detection_classification_moderation_or_defense_only",
            evidence_terms,
        )

    direct_text_term = _query_body_first_pattern_match(
        combined,
        _TEXT_ATTACK_DIRECT_TEXT_PATTERNS,
    )
    evaluation_evidence = _text_attack_local_pair_evidence(
        combined,
        _TEXT_ATTACK_ANCHOR_PATTERNS,
        _TEXT_ATTACK_EVALUATION_PATTERNS,
    )
    evidence_terms = tuple(
        dict.fromkeys(
            term
            for term in (
                *construction_evidence,
                direct_text_term,
                *evaluation_evidence,
            )
            if term
        )
    )
    if not evidence_terms:
        return TextAttackRuntimeAssessment(
            False,
            "missing_text_prompt_attack_generation_or_transformation_evidence",
        )
    return TextAttackRuntimeAssessment(
        True,
        "single_turn_text_attack_runtime_verified",
        evidence_terms,
    )


def _assess_query_body_relevance(
    text: str,
    body_relevance_gate: str,
) -> QueryBodyRelevanceAssessment:
    """Apply a named relevance rule to source body text and nothing else.

    Search queries, result titles, URLs, and metadata are intentionally absent
    from this interface. A narrow topic mention is admitted only when model and
    generation/evaluation language occurs in the same bounded local context.
    Dataset rows may establish the same link with an explicit target entity
    instead of repeating a model noun on every row.
    """
    if not body_relevance_gate:
        return QueryBodyRelevanceAssessment(True, "body_gate_not_required")
    if body_relevance_gate != _TARGETED_DEFAMATION_BODY_GATE:
        return QueryBodyRelevanceAssessment(False, "unsupported_body_relevance_gate")

    body = _query_body_without_package_headers(text).casefold()
    topic_matches = [
        match
        for pattern in _QUERY_BODY_TOPIC_PATTERNS
        for match in pattern.finditer(body)
    ]
    if not topic_matches:
        return QueryBodyRelevanceAssessment(False, "missing_local_narrow_topic")

    for match in topic_matches:
        start = max(0, match.start() - _QUERY_BODY_LOCAL_RADIUS)
        end = min(len(body), match.end() + _QUERY_BODY_LOCAL_RADIUS)
        context = body[start:end]
        action_term = _query_body_first_pattern_match(
            context, _QUERY_BODY_ACTION_PATTERNS
        )
        model_term = _query_body_first_pattern_match(
            context, _QUERY_BODY_MODEL_PATTERNS
        )
        target_term = _query_body_first_pattern_match(
            context, _QUERY_BODY_TARGET_PATTERNS
        )
        if action_term and (model_term or target_term):
            return QueryBodyRelevanceAssessment(
                True,
                "body_local_relevance_verified",
                tuple(
                    dict.fromkeys(
                        term
                        for term in (
                            " ".join(match.group(0).split()),
                            action_term,
                            model_term,
                            target_term,
                        )
                        if term
                    )
                ),
                " ".join(match.group(0).split()),
            )
    return QueryBodyRelevanceAssessment(
        False, "missing_local_model_generation_or_evaluation_context"
    )


def _github_aggregator_metadata(full_name: str, description: str) -> bool:
    return bool(
        _GITHUB_AGGREGATOR_NAME_PATTERN.search(full_name)
        or _GITHUB_AGGREGATOR_DESCRIPTION_PATTERN.search(description)
    )


def _github_metadata_rejection(repo: dict[str, Any], family: str) -> str:
    if bool(repo.get("archived", False)):
        return "archived"
    if bool(repo.get("fork", False)):
        return "fork"
    full_name = str(repo.get("full_name", ""))
    description = str(repo.get("description", "") or "")
    if _GITHUB_NON_ATTACK_REPO_PATTERN.search(description):
        return "defense_or_audit_repo"
    if _GITHUB_UNSUPPORTED_SCOPE_PATTERN.search(description):
        return "unsupported_runtime_scope"
    if _github_aggregator_metadata(full_name, description):
        return "aggregator_metadata"
    if family in _GITHUB_README_DISCOVERY_FAMILIES:
        return ""
    topics = " ".join(
        str(value) for value in repo.get("topics", []) if str(value).strip()
    )
    searchable = f"{full_name} {description} {topics}".casefold()
    anchors = _GITHUB_FAMILY_ANCHORS.get(family) or tuple(
        anchor for values in _GITHUB_FAMILY_ANCHORS.values() for anchor in values
    )
    if not _contains_any(searchable, anchors):
        return "missing_family_anchor_in_metadata"
    if not _contains_any(searchable, _GITHUB_SECURITY_ANCHORS):
        return "missing_security_anchor_in_metadata"
    return ""


def _split_markdown_sections(markdown: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", markdown))
    if not matches:
        return [("document", markdown)]
    sections: list[tuple[str, str]] = []
    prefix = markdown[: matches[0].start()].strip()
    if prefix:
        sections.append(("preamble", prefix))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections.append((match.group(1).strip(), markdown[match.start() : end].strip()))
    return sections


def _sanitize_github_evidence_section(section: str) -> str:
    cleaned = re.sub(r"!\[[^\]]*\]\([^\n)]*\)", "", section)
    cleaned = re.sub(r"<img\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    return "\n".join(
        line.rstrip() for line in cleaned.splitlines() if line.strip()
    ).strip()


def _github_link_aggregator(readme: str) -> bool:
    lines = [line for line in readme.splitlines() if line.strip()]
    if not lines:
        return False
    linked_lines = sum(
        bool(re.search(r"\[[^\]]+\]\(https?://", line)) for line in lines
    )
    return linked_lines >= 30 and linked_lines / len(lines) >= 0.45


def _github_mechanism_bullet_count(section: str) -> int:
    count = 0
    for line in section.splitlines():
        if not re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line):
            continue
        if _contains_any(line, _GITHUB_TRANSFORMATION_MARKERS):
            count += 1
    return count


def _normalized_title_tokens(value: str) -> set[str]:
    stop = {"the", "a", "an", "of", "for", "with", "and", "to", "in", "on", "via"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 2 and token not in stop
    }


_GENERIC_PAPER_TITLE_TOKENS = {
    "aligned",
    "attack",
    "attacks",
    "automated",
    "benchmark",
    "evaluation",
    "evaluating",
    "framework",
    "generated",
    "generating",
    "generation",
    "jailbreak",
    "jailbreaking",
    "jailbreaks",
    "language",
    "large",
    "llm",
    "llms",
    "method",
    "methods",
    "model",
    "models",
    "open",
    "prompt",
    "prompts",
    "robust",
    "robustness",
    "safety",
    "scalable",
    "transferable",
}


def _paper_title_identifier_tokens(value: str) -> set[str]:
    """Return acronym/camel-case tokens likely to identify one named artifact."""
    identifiers: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]*", value):
        letters = "".join(character for character in token if character.isalpha())
        if len(letters) < 3:
            continue
        internal_upper = any(character.isupper() for character in letters[1:])
        acronym = letters.isupper() and len(letters) >= 3
        if internal_upper or acronym or any(character.isdigit() for character in token):
            identifiers.add(token.casefold())
    return identifiers


def _paper_identity_match_basis(
    evidence_text: str,
    paper_title: str,
    paper_url_or_id: str,
) -> str:
    """Return the strong identity signal linking one companion to a paper."""
    try:
        expected_id = normalize_arxiv_id(paper_url_or_id)
    except PaperBundleValidationError:
        expected_id = ""
    if expected_id:
        for candidate in _ARXIV_ID_PATTERN.findall(evidence_text):
            try:
                if normalize_arxiv_id(candidate).casefold() == expected_id.casefold():
                    return "arxiv_id"
            except PaperBundleValidationError:
                continue

    title_tokens = _normalized_title_tokens(paper_title)
    if not title_tokens:
        return ""
    evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_text.casefold()))
    identifier_tokens = _paper_title_identifier_tokens(paper_title)
    if identifier_tokens & evidence_tokens:
        return "title_identifier"
    overlap = len(title_tokens & evidence_tokens)
    distinctive_tokens = title_tokens - _GENERIC_PAPER_TITLE_TOKENS
    distinctive_overlap = len(distinctive_tokens & evidence_tokens)
    if not distinctive_tokens or distinctive_overlap == 0:
        return ""
    # Avoid accepting a companion solely because it contains one generic word.
    minimum_hits = 1 if len(title_tokens) == 1 else 2 if len(title_tokens) <= 3 else 3
    distinctive_minimum_hits = 1 if len(distinctive_tokens) <= 2 else 2
    if (
        overlap >= minimum_hits
        and overlap / len(title_tokens) >= 0.6
        and distinctive_overlap >= distinctive_minimum_hits
        and distinctive_overlap / len(distinctive_tokens) >= 0.5
    ):
        return "title_tokens"
    return ""


def _paper_identity_matches(readme: str, paper_title: str, paper_url: str) -> bool:
    return bool(_paper_identity_match_basis(readme, paper_title, paper_url))


def _assess_github_readme(
    *,
    readme: str,
    family: str,
    repo_name: str,
    description: str,
    paper_title: str = "",
    paper_url: str = "",
) -> GitHubEvidenceAssessment:
    evidence_content = re.sub(
        r"(?m)^## Evidence document:[^\n]*\nEvidence role:[^\n]*\n?",
        "",
        readme,
    )
    if _GITHUB_AGGREGATOR_NAME_PATTERN.search(repo_name) or _github_link_aggregator(
        evidence_content
    ):
        return GitHubEvidenceAssessment(False, 0.0, reason="aggregator_readme")
    if paper_title and not _paper_identity_matches(
        evidence_content, paper_title, paper_url
    ):
        return GitHubEvidenceAssessment(False, 0.0, reason="paper_identity_mismatch")

    anchors = _GITHUB_FAMILY_ANCHORS.get(family) or tuple(
        anchor for values in _GITHUB_FAMILY_ANCHORS.values() for anchor in values
    )
    publication_evidence = bool(paper_title) or _contains_any(
        evidence_content, _GITHUB_PUBLICATION_MARKERS
    )
    document_context = f"{repo_name}\n{description}\n{evidence_content[:20000]}"
    if not _contains_any(document_context, anchors) or not _contains_any(
        document_context,
        _GITHUB_SECURITY_ANCHORS,
    ):
        return GitHubEvidenceAssessment(False, 0.0, reason="missing_document_identity")
    if not publication_evidence:
        return GitHubEvidenceAssessment(
            False, 0.0, reason="missing_publication_evidence"
        )

    total_method_count = sum(
        marker in evidence_content.casefold() for marker in _GITHUB_METHOD_MARKERS
    )
    total_transformation_count = sum(
        marker in evidence_content.casefold()
        for marker in _GITHUB_TRANSFORMATION_MARKERS
    )
    strong_method_hit = _contains_any(
        evidence_content,
        (
            "method",
            "algorithm",
            "pipeline",
            "optimization",
            "implementation",
            "reproduce",
            "approach",
            "technique",
            "mechanism",
        ),
    )
    if not strong_method_hit or total_method_count < 2:
        return GitHubEvidenceAssessment(
            False, 0.0, reason="missing_repository_mechanism_evidence"
        )
    if total_transformation_count < 2:
        return GitHubEvidenceAssessment(
            False, 0.0, reason="missing_repository_operational_evidence"
        )

    ranked_sections: list[tuple[float, str, str]] = []
    for heading, section in _split_markdown_sections(evidence_content):
        evidence_section = _sanitize_github_evidence_section(section)
        if not evidence_section:
            continue
        context = f"{heading}\n{evidence_section}"
        section_roles = set(_infer_evidence_roles(context, path=heading))
        method_count = sum(
            marker in context.casefold() for marker in _GITHUB_METHOD_MARKERS
        )
        transformation_count = sum(
            marker in context.casefold() for marker in _GITHUB_TRANSFORMATION_MARKERS
        )
        mechanism_bullets = _github_mechanism_bullet_count(evidence_section)
        role_evidence = bool(
            section_roles
            & {"mechanism", "implementation", "examples", "evaluation", "domain-evidence"}
        )
        procedural_detail = method_count >= 2 or mechanism_bullets >= 2
        operational_detail = transformation_count >= 2
        if not role_evidence and not procedural_detail and not operational_detail:
            continue
        score = (
            (2.0 if "mechanism" in section_roles else 0.0)
            + (2.0 if operational_detail else 0.0)
            + (1.0 if "evaluation" in section_roles else 0.0)
            + min(method_count, 3)
            + min(transformation_count, 3)
            + min(mechanism_bullets, 2)
        )
        ranked_sections.append((score, heading, evidence_section))

    if not ranked_sections:
        return GitHubEvidenceAssessment(False, 0.0, reason="no_method_evidence_section")
    ranked_sections.sort(key=lambda value: (-value[0], value[1].casefold()))
    selected: list[str] = []
    section_names: list[str] = []
    total_chars = 0
    for _score, heading, section in ranked_sections[:6]:
        remaining = 12000 - total_chars
        if remaining <= 0:
            break
        selected.append(section[:remaining])
        section_names.append(heading)
        total_chars += min(len(section), remaining)
    evidence_type = "paper_method_repo" if paper_title else "method_repo"
    preamble = [f"Repository: {repo_name}", f"Description: {description}"]
    if paper_title:
        preamble.extend([f"Paper: {paper_title}", f"Paper URL: {paper_url}"])
    focused = "\n".join(preamble) + "\n\n" + "\n\n".join(selected)
    return GitHubEvidenceAssessment(
        True,
        min(10.0, ranked_sections[0][0]),
        evidence_type=evidence_type,
        selected_text=focused,
        selected_sections=section_names,
        reason="method_evidence_verified",
    )


def _extract_github_links_with_context(text: str) -> list[tuple[str, str]]:
    decoded = unescape(text).replace("\\/", "/")
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(decoded):
        owner = match.group(1)
        repo = match.group(2).removesuffix(".git").rstrip(".,);'\"")
        repo_name = f"{owner}/{repo}"
        context = re.sub(
            r"<[^>]+>", " ", decoded[max(0, match.start() - 180) : match.end() + 180]
        )
        if repo_name.casefold() in seen:
            continue
        if not _contains_any(
            context, ("code", "implementation", "repository", "available", "source")
        ):
            continue
        seen.add(repo_name.casefold())
        results.append((repo_name, " ".join(context.split())))
    return results


def _extract_paper_companion_links(text: str) -> tuple[list[str], list[str]]:
    """Extract only canonical repository/dataset URLs directly present in a paper."""
    decoded = unescape(str(text)).replace("\\/", "/")
    github: list[str] = []
    huggingface: list[str] = []
    github_seen: set[str] = set()
    huggingface_seen: set[str] = set()
    github_pattern = re.compile(
        r"https?://(?:www\.)?github\.com/"
        r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?"
        r"(?=[\s<>?#'\"),.;:]|$)",
        re.IGNORECASE,
    )
    huggingface_pattern = re.compile(
        r"https?://(?:www\.)?huggingface\.co/datasets/"
        r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)"
        r"(?=[\s<>?#'\"),.;:]|$)",
        re.IGNORECASE,
    )
    for match in github_pattern.finditer(decoded):
        repo = f"{match.group(1)}/{match.group(2)}"
        if repo.casefold() not in github_seen:
            github_seen.add(repo.casefold())
            github.append(repo)
    for match in huggingface_pattern.finditer(decoded):
        dataset = f"{match.group(1)}/{match.group(2)}"
        if dataset.casefold() not in huggingface_seen:
            huggingface_seen.add(dataset.casefold())
            huggingface.append(dataset)
    return github[:5], huggingface[:5]


@dataclass(frozen=True)
class RawCollectedItem:
    """source-neutral raw item collected from an external source."""

    text: str
    source: str
    url: str = ""
    title: str = ""
    collected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseExternalCollector:
    """Small local adapter matching external-source collector semantics."""

    source_name = "unknown"
    default_query = "LLM jailbreak prompt"

    def __init__(
        self,
        *,
        timeout: int = 30,
        user_agent: str = DEFAULT_USER_AGENT,
        delay_seconds: float = 0.0,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self._quality_rejected = 0
        self._quality_rejection_counts: dict[str, int] = {}
        self._collection_errors: list[str] = []

    def collect(self, query: str = "", max_items: int = 20) -> list[RawCollectedItem]:
        raise NotImplementedError

    def collect_query(self, query: str, max_items: int) -> list[RawCollectedItem]:
        """Collect query-driven items without fixed known targets."""
        return self.collect(query=query, max_items=max_items)

    def _query(self, query: str) -> str:
        return query.strip() or self.default_query

    def _delay(self, index: int, total: int) -> None:
        if self.delay_seconds > 0 and index + 1 < total:
            time.sleep(self.delay_seconds)

    def _record_quality_rejection(self, reason: str) -> None:
        self._quality_rejected += 1
        self._quality_rejection_counts[reason] = (
            self._quality_rejection_counts.get(reason, 0) + 1
        )

    def _record_collection_error(self, reason: str) -> None:
        normalized = str(reason).strip()
        if normalized and normalized not in self._collection_errors:
            self._collection_errors.append(normalized)

    def diagnostic_item(self) -> RawCollectedItem | None:
        if not self._quality_rejected and not self._collection_errors:
            return None
        metadata: dict[str, Any] = {
            "external_source": self.source_name,
            "collector": self.source_name,
            "diagnostic": True,
            "quality_rejected": self._quality_rejected,
            "quality_rejection_counts": dict(self._quality_rejection_counts),
        }
        if self._collection_errors:
            metadata["error"] = "; ".join(self._collection_errors)
        return RawCollectedItem(
            text="",
            source=self.source_name,
            title=f"{self.source_name} collection diagnostics",
            metadata=metadata,
        )


class GitHubExternalCollector(BaseExternalCollector):
    """Precision-first GitHub method-repository collector."""

    source_name = "github"
    default_query = "LLM jailbreak prompt"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.authenticated = bool(
            os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        )
        self._started_at = time.monotonic()
        self.diagnostics: dict[str, Any] = {
            "search_candidates": 0,
            "paper_candidates": 0,
            "metadata_rejected": 0,
            "readme_rejected": 0,
            "paper_linked_items": 0,
            "accepted_items": 0,
            "request_errors": 0,
            "github_requests": 0,
            "arxiv_requests": 0,
            "tree_requests": 0,
            "blob_requests": 0,
            "tree_truncated": 0,
            "evidence_files_selected": 0,
            "evidence_chars_selected": 0,
            "evidence_rejected": 0,
            "multi_file_bundles": 0,
            "rejection_counts": {},
            "rejected_samples": [],
        }
        self._repo_cache: dict[str, dict[str, Any]] = {}
        self._readme_cache: dict[str, str] = {}
        self._tree_cache: dict[str, dict[str, Any]] = {}
        self._blob_cache: dict[tuple[str, str], str] = {}
        self._arxiv_failures = 0
        self._arxiv_disabled = False

    def collect(self, query: str = "", max_items: int = 20) -> list[RawCollectedItem]:
        return self.collect_query(self._query(query), max_items, family="custom")

    def collect_repo_for_paper(
        self,
        repo_name: str,
        *,
        arxiv_id: str,
        paper_title: str,
        paper_url: str = "",
        paper_published_at: str = "",
        paper_direct_link: bool = False,
    ) -> RawCollectedItem:
        """Collect one explicit repository and prove that it belongs to a paper."""
        requested_repo = normalize_github_repo(repo_name)
        canonical_id = normalize_arxiv_id(arxiv_id)
        canonical_paper_url = paper_url or f"https://arxiv.org/abs/{canonical_id}"
        try:
            repo = self._get_repo(requested_repo)
        except GitHubAPIError as exc:
            raise PaperBundleValidationError(
                "github_companion_fetch_failed"
            ) from exc
        returned_repo = str(repo.get("full_name", "")).strip()
        if returned_repo.casefold() != requested_repo.casefold():
            raise PaperBundleValidationError("github_companion_identity_mismatch")
        item = self._repo_item(
            repo,
            family="custom",
            discovery=(
                "paper_direct_link"
                if paper_direct_link
                else "explicit_paper_bundle"
            ),
            paper={
                "title": paper_title,
                "url": canonical_paper_url,
                "published_at": paper_published_at,
                "link_context": (
                    "paper direct link"
                    if paper_direct_link
                    else "explicit companion"
                ),
            },
        )
        if item is None:
            raise PaperBundleValidationError("github_companion_not_eligible")
        match_basis = _paper_identity_match_basis(
            item.text,
            paper_title,
            canonical_id,
        )
        if not match_basis:
            raise PaperBundleValidationError("github_companion_paper_mismatch")
        return replace(
            item,
            metadata={
                **item.metadata,
                "companion_identity_verified": True,
                "companion_identity_basis": match_basis,
                "paper_arxiv_id": canonical_id,
                "paper_title": paper_title,
                "paper_url": canonical_paper_url,
                "paper_published_at": paper_published_at,
            },
        )

    def collect_query(
        self,
        query: str,
        max_items: int,
        *,
        family: str = "custom",
        cutoff: datetime | None = None,
        arxiv_discovery: bool = True,
    ) -> list[RawCollectedItem]:
        items: list[RawCollectedItem] = []
        if arxiv_discovery and cutoff is not None and family in _GITHUB_FAMILY_ANCHORS:
            items.extend(
                self._discover_from_arxiv(family, cutoff, max_items=min(2, max_items))
            )
        seen = {item.url.casefold() for item in items if item.url}
        if len(items) < max_items:
            direct = self._search_repos(
                self._query(query), max_items - len(items), family=family
            )
            for item in direct:
                if item.url and item.url.casefold() in seen:
                    continue
                items.append(item)
                if item.url:
                    seen.add(item.url.casefold())
                if len(items) >= max_items:
                    break
        return items[:max_items]

    def _search_repos(
        self, query: str, max_items: int, *, family: str
    ) -> list[RawCollectedItem]:
        try:
            payload = self._request_json(
                "https://api.github.com/search/repositories",
                params={"q": query, "per_page": min(max(10, max_items * 4), 30)},
            )
        except GitHubAPIError as exc:
            return [self._error_item(exc)]

        items: list[RawCollectedItem] = []
        repos = payload.get("items", []) if isinstance(payload, dict) else []
        for index, repo in enumerate(repos):
            if len(items) >= max_items or not isinstance(repo, dict):
                break
            self.diagnostics["search_candidates"] += 1
            rejection = _github_metadata_rejection(repo, family)
            if rejection:
                self._reject(
                    rejection,
                    stage="metadata",
                    repo=str(repo.get("full_name", "")),
                    family=family,
                    discovery="repository_search",
                )
                continue
            item = self._repo_item(repo, family=family, discovery="repository_search")
            if item is not None:
                items.append(item)
            self._delay(index, len(repos))
        return items

    def _discover_from_arxiv(
        self,
        family: str,
        cutoff: datetime,
        *,
        max_items: int,
    ) -> list[RawCollectedItem]:
        if self._arxiv_disabled:
            return []
        query = next(
            (
                spec.query
                for spec in _source_query_specs("arxiv")
                if spec.family == family
            ),
            "",
        )
        if not query:
            return []
        rendered = _render_query_with_cutoff(
            "arxiv", query, cutoff, datetime.now(timezone.utc)
        )
        try:
            self.diagnostics["arxiv_requests"] += 1
            xml_text = _http_get_text(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": rendered,
                    "start": 0,
                    "max_results": 2,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
                timeout=min(self.timeout, _GITHUB_ARXIV_DISCOVERY_TIMEOUT_SECONDS),
                headers={"User-Agent": self.user_agent},
            )
            root = ET.fromstring(xml_text)
        except Exception:
            self._arxiv_failures += 1
            if self._arxiv_failures >= 2:
                self._arxiv_disabled = True
                self.diagnostics["arxiv_discovery_disabled"] = True
            self._reject("arxiv_discovery_error")
            return []

        items: list[RawCollectedItem] = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = _xml_text(entry.find("atom:title", ns))
            summary = _xml_text(entry.find("atom:summary", ns))
            paper_url = _xml_text(entry.find("atom:id", ns))
            published_at = _normalize_timestamp(
                _xml_text(entry.find("atom:published", ns))
            )
            published_dt = (
                datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                if published_at
                else None
            )
            if published_dt is None or published_dt < cutoff:
                continue
            links = _extract_github_links_with_context(summary)
            if not links and paper_url:
                try:
                    self.diagnostics["arxiv_requests"] += 1
                    raw_html = _http_get_text(
                        paper_url.replace("/abs/", "/html/"),
                        timeout=min(
                            self.timeout, _GITHUB_ARXIV_DISCOVERY_TIMEOUT_SECONDS
                        ),
                        headers={"User-Agent": self.user_agent},
                    )
                    links = _extract_github_links_with_context(raw_html)
                except Exception:
                    links = []
            for repo_name, link_context in links:
                self.diagnostics["paper_candidates"] += 1
                try:
                    repo = self._get_repo(repo_name)
                except GitHubAPIError:
                    self._reject(
                        "github_repo_lookup_error",
                        repo=repo_name,
                        family=family,
                        discovery="arxiv_link",
                    )
                    continue
                item = self._repo_item(
                    repo,
                    family=family,
                    discovery="arxiv_link",
                    paper={
                        "title": title,
                        "url": paper_url,
                        "published_at": published_at,
                        "link_context": link_context,
                    },
                )
                if item is not None:
                    items.append(item)
                    self.diagnostics["paper_linked_items"] += 1
                if len(items) >= max_items:
                    return items
        return items

    def _repo_item(
        self,
        repo: dict[str, Any],
        *,
        family: str,
        discovery: str,
        paper: dict[str, str] | None = None,
    ) -> RawCollectedItem | None:
        full_name = str(repo.get("full_name", "")).strip()
        if not full_name:
            return None
        if bool(repo.get("archived", False)) or bool(repo.get("fork", False)):
            self._reject(
                "archived_or_fork",
                stage="metadata",
                repo=full_name,
                family=family,
                discovery=discovery,
            )
            return None
        description = str(repo.get("description", "") or "")
        if _github_aggregator_metadata(full_name, description):
            self._reject(
                "aggregator_metadata",
                stage="metadata",
                repo=full_name,
                family=family,
                discovery=discovery,
            )
            return None
        if _GITHUB_NON_ATTACK_REPO_PATTERN.search(description):
            self._reject(
                "defense_or_audit_repo",
                stage="metadata",
                repo=full_name,
                family=family,
                discovery=discovery,
            )
            return None
        if _GITHUB_UNSUPPORTED_SCOPE_PATTERN.search(description):
            self._reject(
                "unsupported_runtime_scope",
                stage="metadata",
                repo=full_name,
                family=family,
                discovery=discovery,
            )
            return None
        try:
            readme = self._get_readme(
                full_name,
                default_branch=str(repo.get("default_branch", "") or ""),
            )
        except GitHubAPIError:
            self._reject(
                "readme_request_error",
                stage="readme",
                repo=full_name,
                family=family,
                discovery=discovery,
            )
            return None
        if not readme:
            self._reject(
                "missing_readme",
                stage="readme",
                repo=full_name,
                family=family,
                discovery=discovery,
            )
            return None
        paper = dict(paper or {})
        default_branch = str(repo.get("default_branch", "") or "").strip()
        if not default_branch:
            self._reject(
                "missing_default_branch",
                stage="evidence",
                repo=full_name,
                family=family,
                discovery=discovery,
            )
            return None
        try:
            tree_payload = self._get_tree(full_name, default_branch)
            evidence_documents = self._get_repository_evidence_documents(
                full_name,
                tree_payload,
                char_budget=max(
                    0,
                    _GITHUB_EVIDENCE_MAX_TOTAL_CHARS
                    - min(len(readme), _GITHUB_EVIDENCE_MAX_FILE_CHARS),
                ),
            )
        except GitHubAPIError as exc:
            if exc.status in {403, 429}:
                raise
            self._reject(
                "github_evidence_request_error",
                stage="evidence",
                repo=full_name,
                family=family,
                discovery=discovery,
            )
            return None
        if not evidence_documents:
            self._reject(
                "missing_non_readme_evidence",
                stage="evidence",
                repo=full_name,
                family=family,
                discovery=discovery,
            )
            return None

        tree_revision = str(tree_payload.get("sha", "") or default_branch)
        documents = [
            SourceEvidenceDocument(
                path="README.md",
                role="overview",
                text=readme[:_GITHUB_EVIDENCE_MAX_FILE_CHARS],
                url=f"https://github.com/{full_name}/blob/{default_branch}/README.md",
                revision=tree_revision,
                size_bytes=len(readme.encode("utf-8", errors="replace")),
                truncated=len(readme) > _GITHUB_EVIDENCE_MAX_FILE_CHARS,
            ),
            *evidence_documents,
        ]
        evidence_text = _render_evidence_package(
            full_name,
            documents,
            max_total_chars=_GITHUB_EVIDENCE_MAX_TOTAL_CHARS,
        )
        assessment = _assess_github_readme(
            readme=evidence_text,
            family=family,
            repo_name=full_name,
            description=description,
            paper_title=paper.get("title", ""),
            paper_url=paper.get("url", ""),
        )
        if not assessment.eligible:
            self._reject(
                assessment.reason,
                stage="evidence",
                repo=full_name,
                family=family,
                discovery=discovery,
            )
            return None

        created_at = _normalize_timestamp(repo.get("created_at"))
        paper_date = _normalize_timestamp(paper.get("published_at"))
        updated_at = _normalize_timestamp(repo.get("updated_at"))
        pushed_at = _normalize_timestamp(repo.get("pushed_at"))
        source_group = f"github:{full_name}@{tree_revision}"
        quality_metadata = _evidence_quality_metadata(
            evidence_text,
            documents,
            extraction_method="github_repository_evidence_pack",
            authenticated=self.authenticated,
            source_group=source_group,
            revision=tree_revision,
        )
        if not quality_metadata["evidence_quality_eligible"]:
            self._reject(
                str(quality_metadata["evidence_quality_reason"]),
                stage="evidence",
                repo=full_name,
                family=family,
                discovery=discovery,
            )
            return None
        metadata: dict[str, Any] = {
            "repo": full_name,
            "stars": repo.get("stargazers_count", 0),
            "topics": repo.get("topics", []),
            "source_published_at": created_at,
            "source_updated_at": updated_at,
            "source_effective_at": paper_date or pushed_at or updated_at or created_at,
            "repo_pushed_at": pushed_at,
            "archived": bool(repo.get("archived", False)),
            "github_discovery": discovery,
            "github_evidence_eligible": True,
            "github_evidence_score": assessment.score,
            "github_evidence_type": assessment.evidence_type,
            "selected_sections": assessment.selected_sections,
            "github_authenticated": self.authenticated,
            "github_default_branch": default_branch,
            "github_tree_revision": tree_revision,
            "github_tree_truncated": bool(tree_payload.get("truncated", False)),
            **quality_metadata,
        }
        if paper:
            metadata.update(
                {
                    "paper_title": paper.get("title", ""),
                    "paper_url": paper.get("url", ""),
                    "paper_published_at": paper_date,
                    "github_link_context": paper.get("link_context", ""),
                }
            )
        self.diagnostics["accepted_items"] += 1
        self.diagnostics["multi_file_bundles"] += 1
        return RawCollectedItem(
            text=evidence_text,
            source=self.source_name,
            url=str(repo.get("html_url", f"https://github.com/{full_name}")),
            title=full_name,
            metadata=metadata,
        )

    def _get_repo(self, repo_name: str) -> dict[str, Any]:
        key = repo_name.casefold()
        if key not in self._repo_cache:
            payload = self._request_json(f"https://api.github.com/repos/{repo_name}")
            self._repo_cache[key] = payload if isinstance(payload, dict) else {}
        return self._repo_cache[key]

    def _get_readme(self, repo_name: str, *, default_branch: str = "") -> str:
        key = repo_name.casefold()
        if key not in self._readme_cache:
            self._readme_cache[key] = self._request_text(
                f"https://api.github.com/repos/{repo_name}/readme",
                accept="application/vnd.github.raw+json",
                fallback_urls=_github_raw_readme_urls(repo_name, default_branch),
            )
        return self._readme_cache[key]

    def _get_tree(self, repo_name: str, default_branch: str) -> dict[str, Any]:
        key = f"{repo_name.casefold()}@{default_branch}"
        if key not in self._tree_cache:
            self.diagnostics["tree_requests"] += 1
            quoted_branch = urllib_parse.quote(default_branch, safe="")
            payload = self._request_json(
                f"https://api.github.com/repos/{repo_name}/git/trees/{quoted_branch}",
                params={"recursive": 1},
            )
            self._tree_cache[key] = payload if isinstance(payload, dict) else {}
            if bool(self._tree_cache[key].get("truncated", False)):
                self.diagnostics["tree_truncated"] += 1
        return self._tree_cache[key]

    def _get_blob_text(self, repo_name: str, sha: str) -> str:
        key = (repo_name.casefold(), sha)
        if key not in self._blob_cache:
            self.diagnostics["blob_requests"] += 1
            self._blob_cache[key] = self._request_text(
                f"https://api.github.com/repos/{repo_name}/git/blobs/{sha}",
                accept="application/vnd.github.raw+json",
            )
        return self._blob_cache[key]

    def _get_repository_evidence_documents(
        self,
        repo_name: str,
        tree_payload: dict[str, Any],
        *,
        char_budget: int,
    ) -> list[SourceEvidenceDocument]:
        documents: list[SourceEvidenceDocument] = []
        remaining = max(0, char_budget)
        for candidate in _select_github_evidence_candidates(tree_payload):
            if remaining <= 0:
                break
            try:
                raw_text = self._get_blob_text(repo_name, str(candidate["sha"]))
            except GitHubAPIError as exc:
                if exc.status in {403, 429}:
                    raise
                continue
            if not raw_text or "\x00" in raw_text:
                continue
            if raw_text.startswith("version https://git-lfs.github.com/spec/v1"):
                continue
            cleaned = raw_text.strip()
            if len(cleaned) < 80:
                continue
            selected_chars = min(
                len(cleaned),
                _GITHUB_EVIDENCE_MAX_FILE_CHARS,
                remaining,
            )
            if selected_chars < 80:
                continue
            selected_text = cleaned[:selected_chars]
            sha = str(candidate["sha"])
            path = str(candidate["path"])
            documents.append(
                SourceEvidenceDocument(
                    path=path,
                    role=str(candidate["role"]),
                    text=selected_text,
                    url=f"https://github.com/{repo_name}/blob/{sha}/{urllib_parse.quote(path)}",
                    revision=sha,
                    size_bytes=int(candidate["size"]),
                    truncated=len(cleaned) > selected_chars,
                )
            )
            remaining -= selected_chars
        self.diagnostics["evidence_files_selected"] += len(documents)
        selected_total = sum(len(document.text) for document in documents)
        self.diagnostics["evidence_chars_selected"] += selected_total
        return documents

    def _request_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        request_timeout = self.timeout
        if not self.authenticated:
            request_timeout = min(self.timeout, _GITHUB_ANONYMOUS_API_TIMEOUT_SECONDS)
        for attempt in range(2):
            try:
                self.diagnostics["github_requests"] += 1
                return _http_get_json(
                    url,
                    params=params,
                    timeout=request_timeout,
                    headers=_github_api_headers(user_agent=self.user_agent),
                )
            except urllib_error.HTTPError as exc:
                if self._maybe_retry(exc, attempt):
                    continue
                raise self._github_error(exc) from exc
            except Exception as exc:
                self.diagnostics["request_errors"] += 1
                self.diagnostics["last_error"] = {"status": 0, "error": str(exc)}
                raise GitHubAPIError(f"GitHub request failed: {exc}") from exc
        raise GitHubAPIError("GitHub request failed after retry")

    def _request_text(
        self,
        url: str,
        *,
        accept: str,
        fallback_urls: list[str] | None = None,
    ) -> str:
        timeout_cap = (
            _GITHUB_README_TIMEOUT_SECONDS
            if self.authenticated
            else _GITHUB_ANONYMOUS_README_TIMEOUT_SECONDS
        )
        http_error: urllib_error.HTTPError | None = None
        generic_error: Exception | None = None
        for attempt in range(2):
            try:
                self.diagnostics["github_requests"] += 1
                return _http_get_text(
                    url,
                    timeout=min(self.timeout, timeout_cap),
                    headers=_github_api_headers(
                        accept=accept, user_agent=self.user_agent
                    ),
                )
            except urllib_error.HTTPError as exc:
                if self._maybe_retry(exc, attempt):
                    continue
                http_error = exc
                break
            except Exception as exc:
                generic_error = exc
                break
        fallback_text = self._request_readme_fallback(
            fallback_urls,
            timeout_cap=timeout_cap,
        )
        if fallback_text is not None:
            return fallback_text
        if http_error is not None:
            raise self._github_error(http_error) from http_error
        if generic_error is not None:
            self.diagnostics["request_errors"] += 1
            self.diagnostics["last_error"] = {"status": 0, "error": str(generic_error)}
            raise GitHubAPIError(
                f"GitHub request failed: {generic_error}"
            ) from generic_error
        raise GitHubAPIError("GitHub request failed after retry")

    def _request_readme_fallback(
        self,
        urls: list[str] | None,
        *,
        timeout_cap: int,
    ) -> str | None:
        if not urls:
            return None
        for url in urls:
            try:
                self.diagnostics["raw_readme_requests"] = (
                    int(self.diagnostics.get("raw_readme_requests", 0)) + 1
                )
                return _http_get_text(
                    url,
                    timeout=min(self.timeout, timeout_cap),
                    headers={"User-Agent": self.user_agent},
                )
            except urllib_error.HTTPError as exc:
                if int(exc.code) == 404:
                    continue
            except Exception:
                continue
        return None

    def _maybe_retry(self, exc: urllib_error.HTTPError, attempt: int) -> bool:
        if attempt > 0 or int(exc.code) not in {403, 429}:
            return False
        retry_after = str(exc.headers.get("retry-after", "") or "").strip()
        reset = str(exc.headers.get("x-ratelimit-reset", "") or "").strip()
        wait_seconds = int(retry_after) if retry_after.isdigit() else 0
        if not wait_seconds and reset.isdigit():
            wait_seconds = max(0, int(reset) - int(time.time()))
        if 0 < wait_seconds <= 60:
            time.sleep(wait_seconds)
            return True
        return False

    def _github_error(self, exc: urllib_error.HTTPError) -> GitHubAPIError:
        self.diagnostics["request_errors"] += 1
        metadata = {
            "status": int(exc.code),
            "rate_limit_remaining": exc.headers.get("x-ratelimit-remaining", ""),
            "rate_limit_reset": exc.headers.get("x-ratelimit-reset", ""),
            "retry_after": exc.headers.get("retry-after", ""),
            "github_authenticated": self.authenticated,
        }
        self.diagnostics["last_error"] = dict(metadata)
        return GitHubAPIError(
            f"GitHub API returned HTTP {exc.code}",
            status=int(exc.code),
            metadata=metadata,
        )

    def _error_item(self, exc: GitHubAPIError) -> RawCollectedItem:
        item = _skipped_raw_item(self.source_name, str(exc), is_error=True)
        return replace(item, metadata={**item.metadata, **exc.metadata})

    def _reject(
        self,
        reason: str,
        *,
        stage: str = "other",
        repo: str = "",
        family: str = "",
        discovery: str = "",
    ) -> None:
        if stage == "metadata":
            self.diagnostics["metadata_rejected"] += 1
        elif stage == "readme":
            self.diagnostics["readme_rejected"] += 1
        elif stage == "evidence":
            self.diagnostics["evidence_rejected"] += 1
        counts = self.diagnostics["rejection_counts"]
        counts[reason] = counts.get(reason, 0) + 1
        if repo:
            sample = {
                "repo": repo,
                "family": family,
                "stage": stage,
                "reason": reason,
                "discovery": discovery,
            }
            samples = self.diagnostics["rejected_samples"]
            sample_key = (repo.casefold(), family, stage, reason, discovery)
            existing_keys = {
                (
                    str(value.get("repo", "")).casefold(),
                    value.get("family", ""),
                    value.get("stage", ""),
                    value.get("reason", ""),
                    value.get("discovery", ""),
                )
                for value in samples
            }
            if sample_key not in existing_keys and len(samples) < 25:
                samples.append(sample)

    def diagnostic_item(self) -> RawCollectedItem:
        elapsed_seconds = round(time.monotonic() - self._started_at, 3)
        return RawCollectedItem(
            text="",
            source=self.source_name,
            title="github collection diagnostics",
            metadata={
                "external_source": self.source_name,
                "collector": self.source_name,
                "diagnostic": True,
                "github_authenticated": self.authenticated,
                "elapsed_seconds": elapsed_seconds,
                **self.diagnostics,
            },
        )


class ArxivExternalCollector(BaseExternalCollector):
    """Collect titles and abstracts from recent arXiv papers."""

    source_name = "arxiv"
    default_query = "LLM jailbreak attack"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # ``_get_full_text`` intentionally keeps its historical ``str`` return
        # type.  Store bounded extraction provenance separately so callers and
        # tests that mock that method remain compatible.
        self._last_full_text_metadata: dict[str, Any] = {}

    def collect_id(self, arxiv_id: str) -> RawCollectedItem:
        """Fetch exactly one arXiv paper by ``id_list`` and require HTML text."""
        canonical_id = normalize_arxiv_id(arxiv_id)
        try:
            xml_text = _http_get_text(
                "https://export.arxiv.org/api/query",
                params={
                    "id_list": canonical_id,
                    "start": 0,
                    "max_results": 1,
                },
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )
        except urllib_error.HTTPError as exc:
            # The Atom endpoint may rate-limit an otherwise reachable arXiv
            # host. Fall back only to the official abstract page and require
            # its citation_arxiv_id metadata to match exactly before using the
            # HTML paper body. Feed responses that parse but are ambiguous or
            # mismatched still fail closed below and never take this fallback.
            if int(exc.code) not in {403, 408, 429, 500, 502, 503, 504}:
                raise PaperBundleValidationError("arxiv_exact_fetch_failed") from exc
            return self._collect_id_from_abs_page(canonical_id, api_error=exc)
        except (urllib_error.URLError, TimeoutError, ConnectionError) as exc:
            # Transport failures do not weaken identity verification: the
            # independent official abs-page route below must still publish a
            # matching citation_arxiv_id before HTML/PDF evidence is admitted.
            return self._collect_id_from_abs_page(canonical_id, api_error=exc)
        except Exception as exc:
            raise PaperBundleValidationError("arxiv_exact_fetch_failed") from exc
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            # A gateway/challenge page from the Atom host is recoverable only
            # through the independently identity-checked official abs page.
            return self._collect_id_from_abs_page(canonical_id, api_error=exc)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if len(entries) != 1:
            raise PaperBundleValidationError("arxiv_exact_one_required")
        entry_link = _xml_text(entries[0].find("atom:id", ns))
        try:
            returned_id = normalize_arxiv_id(entry_link)
        except PaperBundleValidationError as exc:
            raise PaperBundleValidationError("arxiv_response_id_missing") from exc
        if returned_id.casefold() != canonical_id.casefold():
            raise PaperBundleValidationError("arxiv_response_id_mismatch")
        full_text_attempts = 1
        item = self._entry_item(entries[0], require_full_text=True)
        if item is None:
            # HTML/PDF availability can fail transiently even after the exact
            # Atom identity has been verified. Retry the same verified entry
            # once without relaxing URL, identity, size, or parser checks.
            full_text_attempts = 2
            item = self._entry_item(entries[0], require_full_text=True)
        if item is None:
            raise PaperBundleValidationError("arxiv_full_text_required")
        return replace(
            item,
            metadata={
                **item.metadata,
                "arxiv_id": canonical_id,
                "arxiv_exact_fetch": True,
                "arxiv_identity_verified": True,
                "arxiv_identity_verification": "atom_id_list",
                "arxiv_full_text_attempts": full_text_attempts,
            },
        )

    def _collect_id_from_abs_page(
        self,
        canonical_id: str,
        *,
        api_error: Exception,
    ) -> RawCollectedItem:
        abs_url = f"https://arxiv.org/abs/{canonical_id}"
        try:
            abs_html = _http_get_text(
                abs_url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                same_origin_redirects_only=True,
            )
        except Exception as exc:
            raise PaperBundleValidationError("arxiv_exact_fetch_failed") from exc
        if "<html" not in abs_html[:2000].casefold():
            raise PaperBundleValidationError("arxiv_abs_html_required") from api_error

        citation = _arxiv_abs_citation_metadata(abs_html)
        returned_value = citation.get("citation_arxiv_id", "")
        try:
            returned_id = normalize_arxiv_id(returned_value)
        except PaperBundleValidationError as exc:
            raise PaperBundleValidationError("arxiv_response_id_missing") from exc
        if returned_id.casefold() != canonical_id.casefold():
            raise PaperBundleValidationError("arxiv_response_id_mismatch")

        atom_namespace = "http://www.w3.org/2005/Atom"
        entry = ET.Element(f"{{{atom_namespace}}}entry")
        ET.SubElement(entry, f"{{{atom_namespace}}}id").text = abs_url
        ET.SubElement(entry, f"{{{atom_namespace}}}title").text = citation.get(
            "citation_title", canonical_id
        )
        published_at = _normalize_arxiv_citation_date(
            citation.get("citation_date", "")
        )
        if published_at:
            ET.SubElement(entry, f"{{{atom_namespace}}}published").text = published_at

        full_text_attempts = 1
        item = self._entry_item(entry, require_full_text=True)
        if item is None:
            full_text_attempts = 2
            item = self._entry_item(entry, require_full_text=True)
        if item is None:
            raise PaperBundleValidationError("arxiv_full_text_required")
        return replace(
            item,
            metadata={
                **item.metadata,
                "arxiv_id": canonical_id,
                "arxiv_exact_fetch": True,
                "arxiv_identity_verified": True,
                "arxiv_identity_verification": "official_abs_citation_meta",
                "arxiv_full_text_attempts": full_text_attempts,
                "arxiv_api_fallback": True,
                "arxiv_api_fallback_reason": type(api_error).__name__,
                "arxiv_api_fallback_status": int(
                    getattr(api_error, "code", 0) or 0
                ),
            },
        )

    def collect(self, query: str = "", max_items: int = 20) -> list[RawCollectedItem]:
        return self.collect_query(query, max_items)

    def discover_ids(self, query: str, max_items: int) -> list[RawCollectedItem]:
        """List canonical IDs from the Atom feed without fetching paper bodies."""
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        rendered_query = self._query(query)
        if not re.search(r"\b(?:all|ti|abs|au|cat):", rendered_query):
            rendered_query = f"all:{rendered_query}"
        try:
            xml_text = _http_get_text(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": rendered_query,
                    "start": 0,
                    "max_results": min(max(max_items * 3, 10), 30),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )
            root = ET.fromstring(xml_text)
        except Exception as exc:
            return [
                _skipped_raw_item(
                    self.source_name,
                    f"arXiv ID discovery failed: {type(exc).__name__}",
                    is_error=True,
                )
            ]

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items: list[RawCollectedItem] = []
        for entry in root.findall("atom:entry", ns):
            link = _xml_text(entry.find("atom:id", ns))
            try:
                canonical_id = normalize_arxiv_id(link)
            except PaperBundleValidationError:
                continue
            published_at = _normalize_timestamp(
                _xml_text(entry.find("atom:published", ns))
            )
            updated_at = _normalize_timestamp(
                _xml_text(entry.find("atom:updated", ns))
            )
            items.append(
                RawCollectedItem(
                    text="",
                    source=self.source_name,
                    url=link,
                    title=_xml_text(entry.find("atom:title", ns)),
                    metadata={
                        "arxiv_id": canonical_id,
                        "arxiv_discovery_only": True,
                        "source_published_at": published_at,
                        "source_updated_at": updated_at,
                        "source_effective_at": updated_at or published_at,
                    },
                )
            )
            if len(items) >= max_items:
                break
        return items

    def collect_query(self, query: str, max_items: int) -> list[RawCollectedItem]:
        rendered_query = self._query(query)
        if not re.search(r"\b(?:all|ti|abs|au|cat):", rendered_query):
            rendered_query = f"all:{rendered_query}"
        try:
            xml_text = _http_get_text(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": rendered_query,
                    "start": 0,
                    "max_results": min(max(max_items * 3, 10), 30),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )
            root = ET.fromstring(xml_text)
        except Exception as exc:
            return [
                _skipped_raw_item(
                    self.source_name,
                    f"arXiv collection failed: {exc}",
                    is_error=True,
                )
            ]

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        ready_items: list[RawCollectedItem] = []
        discovery_items: list[RawCollectedItem] = []
        for entry in root.findall("atom:entry", ns):
            collected_item = self._entry_item(entry)
            if collected_item is None:
                continue
            if (
                collected_item.metadata.get("full_text")
                and collected_item.metadata.get("evidence_quality_eligible")
            ):
                ready_items.append(collected_item)
            else:
                discovery_items.append(collected_item)
            if len(ready_items) >= max_items:
                break
        return (ready_items + discovery_items)[:max_items]

    def _entry_item(
        self,
        entry: Any,
        *,
        require_full_text: bool = False,
    ) -> RawCollectedItem | None:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        title = _xml_text(entry.find("atom:title", ns))
        summary = _xml_text(entry.find("atom:summary", ns))
        link = _xml_text(entry.find("atom:id", ns))
        if not link or (not summary and not require_full_text):
            return None
        published_at = _normalize_timestamp(
            _xml_text(entry.find("atom:published", ns))
        )
        updated_at = _normalize_timestamp(_xml_text(entry.find("atom:updated", ns)))
        try:
            canonical_id = normalize_arxiv_id(link)
        except PaperBundleValidationError:
            canonical_id = hashlib.sha256(
                link.encode("utf-8", errors="replace")
            ).hexdigest()[:16]
        canonical_abs_url = f"https://arxiv.org/abs/{canonical_id}"
        canonical_html_url = f"https://arxiv.org/html/{canonical_id}"
        self._last_full_text_metadata = {}
        full_text = self._get_full_text(
            canonical_abs_url,
            expected_arxiv_id=(canonical_id if require_full_text else ""),
        )
        if require_full_text and not full_text:
            return None
        source_text = full_text or f"{title}\n\n{summary}".strip()
        companion_github, companion_huggingface = _extract_paper_companion_links(
            source_text
        )
        revision = updated_at or published_at
        full_text_metadata = dict(self._last_full_text_metadata)
        full_text_format = str(
            full_text_metadata.get("arxiv_full_text_format", "html")
            if full_text
            else ""
        ).casefold()
        full_text_url = (
            str(full_text_metadata.get("arxiv_full_text_url", "")).strip()
            if full_text
            else ""
        )
        if full_text and not full_text_url:
            full_text_url = canonical_html_url
        documents = [
            SourceEvidenceDocument(
                path=(
                    "paper.pdf"
                    if full_text and full_text_format == "pdf"
                    else "paper.html"
                    if full_text
                    else "abstract.txt"
                ),
                role="mechanism" if full_text else "overview",
                text=source_text,
                url=full_text_url if full_text else canonical_abs_url,
                revision=revision,
                size_bytes=len(source_text.encode("utf-8", errors="replace")),
                truncated=bool(full_text and len(full_text) >= 50_000),
            )
        ]
        text = _render_evidence_package(
            title or canonical_id,
            documents,
            max_total_chars=55_000,
        )
        quality_metadata = _evidence_quality_metadata(
            text,
            documents,
            extraction_method=(
                str(
                    full_text_metadata.get(
                        "arxiv_full_text_extraction_method",
                        "arxiv_html_role_balanced_full_text",
                    )
                )
                if full_text
                else "arxiv_abstract_only"
            ),
            source_group=f"arxiv:{canonical_id}@{revision}",
            revision=revision,
        )
        if not full_text:
            quality_metadata.update(
                {
                    "evidence_quality_eligible": False,
                    "evidence_quality_reason": "full_text_required",
                    "risk_domain_binding_eligible": False,
                    "advanced_mechanism_eligible": False,
                }
            )
        return RawCollectedItem(
            text=text,
            source=self.source_name,
            url=canonical_abs_url,
            title=title,
            metadata={
                "arxiv_id": canonical_id,
                "source_published_at": published_at,
                "source_updated_at": updated_at,
                "source_effective_at": updated_at or published_at,
                "full_text": bool(full_text),
                **full_text_metadata,
                "paper_companion_github_repos": companion_github,
                "paper_companion_huggingface_datasets": companion_huggingface,
                **quality_metadata,
            },
        )

    def _get_full_text(
        self,
        abstract_url: str,
        *,
        expected_arxiv_id: str = "",
    ) -> str:
        self._last_full_text_metadata = {}
        if not abstract_url:
            return ""
        try:
            requested_id = normalize_arxiv_id(abstract_url)
        except PaperBundleValidationError:
            return ""
        canonical_id = ""
        if expected_arxiv_id:
            try:
                canonical_id = normalize_arxiv_id(expected_arxiv_id)
            except PaperBundleValidationError:
                return ""
            if canonical_id.casefold() != requested_id.casefold():
                return ""
        # Atom feeds commonly publish ``http://arxiv.org/abs/...`` links.
        # Construct the official HTTPS URL directly so the strict redirect
        # guard does not reject arXiv's HTTP-to-HTTPS protocol upgrade.
        html_url = f"https://arxiv.org/html/{requested_id}"
        try:
            html_text = _http_get_text(
                html_url,
                timeout=min(self.timeout, 15),
                headers={"User-Agent": self.user_agent},
                same_origin_redirects_only=True,
            )
        except Exception:
            html_text = ""
        if "<html" in html_text[:2000].casefold():
            html_identity_valid = True
            if canonical_id:
                identity_candidates: set[str] = set()
                for match in re.finditer(
                    r"(?i)(?:arxiv\s*:\s*|/(?:abs|html)/)"
                    r"((?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})(?:v\d+)?)",
                    html_text,
                ):
                    try:
                        identity_candidates.add(normalize_arxiv_id(match.group(1)))
                    except PaperBundleValidationError:
                        continue
                # An explicitly different identity is a hard failure.  A page
                # without any identity is commonly arXiv's "HTML unavailable"
                # shell, for which the exact official PDF is the safe fallback.
                if identity_candidates and canonical_id not in identity_candidates:
                    return ""
                html_identity_valid = canonical_id in identity_candidates

            if html_identity_valid:
                _title, text = _arxiv_html_to_text(html_text)
                extracted_chars = len(text)
                text = self._prepare_arxiv_full_text(text, link_source=html_text)
                if len(text.strip()) > 500:
                    citation_candidates = _ranked_attack_method_arxiv_citations(
                        html_text,
                        current_arxiv_id=requested_id,
                    )
                    self._last_full_text_metadata = {
                        "arxiv_full_text_format": "html",
                        "arxiv_full_text_url": html_url,
                        "arxiv_full_text_extraction_method": (
                            "arxiv_html_role_balanced_full_text"
                        ),
                        "arxiv_source_extracted_chars": extracted_chars,
                        "arxiv_source_retained_chars": len(text),
                        "arxiv_source_bounded": len(text) < extracted_chars,
                        "arxiv_source_bounding_method": (
                            "role_balanced_example_grounded_v2"
                        ),
                        "paper_attack_method_arxiv_citations": (
                            citation_candidates
                        ),
                    }
                    return text

        return self._get_pdf_full_text(requested_id)

    def _prepare_arxiv_full_text(self, text: str, *, link_source: str) -> str:
        """Bound extracted paper text while retaining directly cited artifacts."""
        github_links, huggingface_links = _extract_paper_companion_links(link_source)
        if github_links or huggingface_links:
            direct_links = [
                *(f"https://github.com/{repo}" for repo in github_links),
                *(
                    f"https://huggingface.co/datasets/{dataset}"
                    for dataset in huggingface_links
                ),
            ]
            link_appendix = (
                "\n\nCompanion links directly cited by this paper:\n"
                + "\n".join(direct_links)
            )
            text_budget = max(0, 50_000 - len(link_appendix))
            return (
                _bounded_arxiv_paper_text(text, max_chars=text_budget)
                + link_appendix
            ).strip()
        return _bounded_arxiv_paper_text(text, max_chars=50_000)

    def _get_pdf_full_text(self, canonical_id: str) -> str:
        """Extract bounded text from the exact official arXiv PDF, or fail closed."""
        try:
            requested_id = normalize_arxiv_id(canonical_id)
        except PaperBundleValidationError:
            return ""
        pdf_url = f"https://arxiv.org/pdf/{requested_id}.pdf"
        try:
            # Ask for one byte beyond the accepted bound so a truncated prefix
            # can never be mistaken for a complete, trusted PDF.
            pdf_bytes = _http_get_bytes(
                pdf_url,
                timeout=min(self.timeout, 20),
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/pdf",
                    "Accept-Encoding": "identity",
                },
                max_bytes=_ARXIV_PDF_MAX_BYTES + 1,
                same_origin_redirects_only=True,
            )
        except Exception:
            return ""
        if (
            not pdf_bytes.startswith(b"%PDF-")
            or len(pdf_bytes) > _ARXIV_PDF_MAX_BYTES
        ):
            return ""

        try:
            # Lazy import keeps non-arXiv collection paths independent of the
            # optional PDF parser and makes parser failures fail closed here.
            from pypdf import PdfReader

            # arXiv PDFs occasionally contain recoverable cross-reference
            # quirks.  Non-strict parsing avoids rejecting those ordinary
            # papers; every parser/extraction exception still fails closed,
            # and identity is anchored by the verified exact Atom ID plus the
            # canonical official PDF URL constructed above.
            reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
            if reader.is_encrypted:
                return ""
            page_count = len(reader.pages)
            if page_count < 1 or page_count > _ARXIV_PDF_MAX_PAGES:
                return ""

            parts: list[str] = []
            extracted_chars = 0
            extraction_truncated = False
            for page_index, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if not isinstance(page_text, str) or not page_text.strip():
                    continue
                remaining = _ARXIV_PDF_MAX_EXTRACTED_CHARS - extracted_chars
                if remaining <= 0:
                    extraction_truncated = True
                    break
                page_heading = f"# PDF page {page_index + 1}\n"
                if len(page_text) > remaining:
                    parts.append(page_heading + page_text[:remaining])
                    extracted_chars += remaining
                    extraction_truncated = True
                    break
                parts.append(page_heading + page_text)
                extracted_chars += len(page_text)
                if (
                    extracted_chars >= _ARXIV_PDF_MAX_EXTRACTED_CHARS
                    and page_index + 1 < page_count
                ):
                    extraction_truncated = True
                    break
        except Exception:
            return ""

        extracted_text = "\n\n".join(parts).strip()
        if len(extracted_text) <= 500:
            return ""
        text = self._prepare_arxiv_full_text(
            extracted_text,
            link_source=extracted_text,
        )
        if len(text.strip()) <= 500:
            return ""
        citation_candidates = _ranked_attack_method_arxiv_citations(
            extracted_text,
            current_arxiv_id=requested_id,
        )
        self._last_full_text_metadata = {
            "arxiv_full_text_format": "pdf",
            "arxiv_full_text_url": pdf_url,
            "arxiv_full_text_extraction_method": (
                "arxiv_pdf_pypdf_role_balanced_full_text"
            ),
            "arxiv_pdf_download_bytes": len(pdf_bytes),
            "arxiv_pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            "arxiv_pdf_page_count": page_count,
            "arxiv_pdf_extracted_chars": extracted_chars,
            "arxiv_pdf_extraction_truncated": extraction_truncated,
            "arxiv_source_extracted_chars": len(extracted_text),
            "arxiv_source_retained_chars": len(text),
            "arxiv_source_bounded": len(text) < len(extracted_text),
            "arxiv_source_bounding_method": "role_balanced_example_grounded_v2",
            "paper_attack_method_arxiv_citations": citation_candidates,
        }
        return text


class GoogleExternalCollector(BaseExternalCollector):
    """Collect full pages discovered through configured web-search backends."""

    source_name = "google"
    default_query = "LLM jailbreak prompt techniques"

    def __init__(
        self,
        *,
        search_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        resolved_search_config = dict(search_config or {})
        bocha_config = dict(resolved_search_config.get("bocha", {}))
        self.bocha_enabled = bool(bocha_config.get("enabled", True))
        self.bocha_api_key = str(
            os.environ.get("BOCHA_API_KEY")
            or bocha_config.get("api_key")
            or ""
        ).strip()
        self.bocha_base_url = str(
            os.environ.get("BOCHA_BASE_URL")
            or bocha_config.get("base_url")
            or BOCHA_SEARCH_BASE_URL
        ).strip()
        self.bocha_freshness = str(
            bocha_config.get("freshness") or "noLimit"
        ).strip()
        self.bocha_summary = bool(bocha_config.get("summary", False))
        # SerpAPI is an optional discovery backend for the existing google
        # source, not a sixth external source. Keep credentials in memory only.
        self.serpapi_api_key = str(
            os.environ.get("SERPAPI_API_KEY")
            or os.environ.get("SERPAPI_KEY")
            or ""
        ).strip()
        # Model-backed Google grounding is intentionally not part of the
        # default discovery chain. External skill generation must not acquire
        # a hidden dependency on a second generation model.
        self.authenticated = bool(
            (self.bocha_enabled and self.bocha_api_key)
            or self.serpapi_api_key
        )

    def collect(self, query: str = "", max_items: int = 20) -> list[RawCollectedItem]:
        return self.collect_query(query, max_items)

    def discover_links(self, query: str, max_results: int) -> GoogleLinkDiscovery:
        """Discover public result links without scraping or admitting their text."""
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        errors: list[str] = []
        results: list[tuple[str, str]] = []
        search_backend = ""
        search_limit = min(max(int(max_results), 1), 30)
        if self.bocha_enabled and self.bocha_api_key:
            try:
                results = search_bocha_web(
                    self._query(query),
                    api_key=self.bocha_api_key,
                    base_url=self.bocha_base_url,
                    freshness=self.bocha_freshness,
                    summary=self.bocha_summary,
                    limit=search_limit,
                    timeout=self.timeout,
                    user_agent=self.user_agent,
                )
                if results:
                    search_backend = "bocha"
            except Exception as exc:
                errors.append(f"Bocha Web Search: {type(exc).__name__}")
        if not results and self.serpapi_api_key:
            try:
                results = search_serpapi_google(
                    self._query(query),
                    api_key=self.serpapi_api_key,
                    limit=search_limit,
                    timeout=self.timeout,
                    user_agent=self.user_agent,
                )
                search_backend = "serpapi_google"
            except Exception as exc:
                errors.append(f"SerpAPI Google search: {type(exc).__name__}")
        try:
            if not results:
                results = search_duckduckgo(
                    self._query(query),
                    limit=search_limit,
                    timeout=self.timeout,
                    user_agent=self.user_agent,
                )
                search_backend = "duckduckgo"
        except Exception as exc:
            errors.append(f"DuckDuckGo: {type(exc).__name__}")
        return GoogleLinkDiscovery(
            results=tuple(results[:search_limit]),
            backend=search_backend,
            errors=tuple(errors),
        )

    def collect_query(self, query: str, max_items: int) -> list[RawCollectedItem]:
        items: list[RawCollectedItem] = []
        discovery = self.discover_links(
            query,
            min(max(max_items * 3, 10), 30),
        )
        results = list(discovery.results)
        if not results and discovery.errors:
            return [
                _skipped_raw_item(
                    self.source_name,
                    "web search failed: " + "; ".join(discovery.errors),
                    is_error=True,
                )
            ]

        for index, (_title, result_url) in enumerate(results):
            if len(items) >= max_items:
                break
            item = self._scrape_url(
                result_url,
                search_backend=discovery.backend or "unknown",
            )
            if item:
                items.append(item)
            self._delay(index, len(results))
        return items[:max_items]

    def _scrape_url(
        self,
        url: str,
        *,
        search_backend: str,
    ) -> RawCollectedItem | None:
        try:
            item = fetch_url(
                url,
                timeout=min(self.timeout, 15),
                user_agent=self.user_agent,
                max_bytes=500_000,
            )
        except Exception as exc:
            self._record_collection_error(
                f"page_fetch_failed:{type(exc).__name__}"
            )
            return None
        if item.status >= 400:
            self._record_collection_error(f"page_fetch_http_{item.status}")
            return None
        if len(item.text.strip()) <= 100:
            self._record_quality_rejection("insufficient_text")
            return None
        metadata = dict(item.metadata)
        resolved_url = str(metadata.get("final_url") or item.url or url)
        if not (
            metadata.get("source_published_at")
            or metadata.get("source_updated_at")
        ):
            visible_date = extract_visible_date(item.text)
            if visible_date:
                metadata["source_published_at"] = visible_date
        effective_date = (
            metadata.get("source_updated_at")
            or metadata.get("source_published_at")
            or metadata.get("http_last_modified")
        )
        if not effective_date:
            self._record_quality_rejection("missing_source_date")
            return None
        revision = str(
            effective_date
        )
        document = SourceEvidenceDocument(
            path=resolved_url,
            role="overview",
            text=item.text[:50_000],
            url=resolved_url,
            revision=revision,
            size_bytes=len(item.text.encode("utf-8", errors="replace")),
            truncated=bool(metadata.get("truncated")) or len(item.text) > 50_000,
        )
        text = _render_evidence_package(
            item.title or url,
            [document],
            max_total_chars=55_000,
        )
        source_group = "web:" + hashlib.sha256(
            resolved_url.casefold().rstrip("/").encode("utf-8")
        ).hexdigest()[:20]
        bocha_authenticated = bool(
            search_backend == "bocha"
            and self.bocha_enabled
            and self.bocha_api_key
        )
        serpapi_authenticated = bool(
            search_backend == "serpapi_google" and self.serpapi_api_key
        )
        discovery_authenticated = bool(
            bocha_authenticated or serpapi_authenticated
        )
        google_grounding_authenticated = False
        quality_metadata = _evidence_quality_metadata(
            text,
            [document],
            extraction_method=f"full_web_page_after_{search_backend}_discovery",
            authenticated=discovery_authenticated,
            source_group=f"{source_group}@{revision}",
            revision=revision,
        )
        if not quality_metadata["evidence_quality_eligible"]:
            self._record_quality_rejection(
                str(quality_metadata["evidence_quality_reason"])
            )
            return None
        return RawCollectedItem(
            text=text,
            source=self.source_name,
            url=resolved_url,
            title=item.title or resolved_url.rstrip("/").split("/")[-1] or resolved_url,
            metadata={
                **metadata,
                "search_backend": search_backend,
                "bocha_authenticated": bocha_authenticated,
                "google_search_authenticated": serpapi_authenticated,
                "serpapi_authenticated": serpapi_authenticated,
                "google_grounding_authenticated": google_grounding_authenticated,
                "source_effective_at": effective_date,
                **quality_metadata,
            },
        )


def _is_huggingface_commit_sha(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{40}", str(value).strip()))


_ATTACK_MECHANISM_DENIAL_PATTERNS = (
    r"\b(?:does|do|did)\s+not\s+(?:propose|introduce|present|develop)\b.{0,100}\b(?:attack|jailbreak|technique|method|mechanism)s?\b",
    r"\bno\s+(?:new|novel)\s+(?:attack|jailbreak|technique|method|mechanism)s?\b",
    r"새로운\s*공격\s*기법.{0,30}(?:제안|소개).{0,20}(?:않|아니)",
)
_DATASET_PACKAGING_MARKERS = (
    "benchmark",
    "dataset",
    "corpus",
    "translated",
    "translation",
    "deduplic",
    "redistribut",
    "repack",
    "벤치마크",
    "데이터셋",
    "번역",
    "중복 제거",
    "재배포",
)


def _source_denies_new_attack_mechanism(text: str) -> bool:
    lowered = text.casefold()
    return any(
        re.search(pattern, lowered, flags=re.DOTALL)
        for pattern in _ATTACK_MECHANISM_DENIAL_PATTERNS
    )


def _dataset_packaging_marker_count(text: str) -> int:
    lowered = text.casefold()
    return sum(marker in lowered for marker in _DATASET_PACKAGING_MARKERS)


def _huggingface_dataset_candidate_score(dataset: dict[str, Any]) -> float:
    """Prefer documented, safety-relevant datasets over fresh anonymous copies."""
    dataset_id = str(dataset.get("id", ""))
    description = str(dataset.get("description", "") or "")
    tags = [str(tag).casefold() for tag in dataset.get("tags", [])]
    searchable = f"{dataset_id}\n{description}\n{' '.join(tags)}".casefold()
    score = 0.0
    if len(description.strip()) >= 120:
        score += 4.0
    if any(tag in {"format:json", "format:jsonl", "format:csv"} for tag in tags):
        score += 4.0
    if any(
        marker in searchable
        for marker in ("ai-safety", "jailbreak", "red-team", "red_team")
    ):
        score += 3.0
    if any(tag.startswith("arxiv:") for tag in tags):
        score += 1.5
    score += min(float(dataset.get("likes", 0) or 0), 50.0) / 25.0
    score += min(float(dataset.get("downloads", 0) or 0), 10_000.0) / 10_000.0
    if re.search(r"(?:^|[/_-])id[-_]\d+|unlearn[-_]seed", dataset_id, re.I):
        score -= 5.0
    if _source_denies_new_attack_mechanism(searchable):
        score -= 10.0
    elif _dataset_packaging_marker_count(searchable) >= 3:
        score -= 3.0
    return score


def _safe_huggingface_data_path(path: str) -> bool:
    normalized = str(path).strip()
    if not normalized or normalized.startswith(("/", "\\")) or "\\" in normalized:
        return False
    parts = normalized.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _select_huggingface_raw_data_files(
    tree: list[Any],
) -> list[dict[str, Any]]:
    selected: list[tuple[tuple[int, int, int, int, str], dict[str, Any]]] = []
    suffix_rank = {
        ".jsonl": 0,
        ".ndjson": 0,
        ".csv": 1,
        ".tsv": 1,
        ".json": 2,
    }
    for entry in tree[:_HUGGINGFACE_TREE_MAX_ENTRIES]:
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        path = str(entry.get("path", ""))
        if not _safe_huggingface_data_path(path):
            continue
        lowered = path.casefold()
        suffix = Path(lowered).suffix
        if suffix not in _HUGGINGFACE_RAW_SUFFIXES:
            continue
        if Path(lowered).name in {
            "dataset_infos.json",
            "metadata.json",
            "config.json",
        }:
            continue
        size = int(entry.get("size", 0) or 0)
        if size <= 32:
            continue
        split_rank = 0 if "train" in lowered else 1 if any(
            value in lowered for value in ("validation", "valid", "dev")
        ) else 2 if "test" in lowered else 3
        data_rank = 0 if any(
            value in lowered
            for value in ("data", "dataset", "sample", "prompt", "jailbreak")
        ) else 1
        size_rank = 0 if size <= 5_000_000 else 1
        rank = (split_rank, data_rank, suffix_rank[suffix], size_rank, lowered)
        selected.append((rank, entry))
    selected.sort(key=lambda value: value[0])
    return [entry for _rank, entry in selected[:_HUGGINGFACE_RAW_MAX_FILES]]


def _bounded_huggingface_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:1000]
    if depth >= 3:
        return str(value)[:1000]
    if isinstance(value, dict):
        return {
            str(key)[:200]: _bounded_huggingface_value(child, depth=depth + 1)
            for key, child in list(value.items())[:32]
        }
    if isinstance(value, list):
        return [
            _bounded_huggingface_value(child, depth=depth + 1)
            for child in value[:16]
        ]
    return str(value)[:1000]


def _parse_huggingface_structured_rows(
    text: str,
    *,
    path: str,
    max_rows: int,
) -> list[dict[str, Any]]:
    if not text.strip() or "\x00" in text:
        return []
    if text.lstrip().startswith("version https://git-lfs.github.com/spec/v1"):
        return []
    suffix = Path(path.casefold()).suffix
    rows: list[Any] = []
    if suffix in {".jsonl", ".ndjson"}:
        lines = text.splitlines()
        for line in lines:
            if len(rows) >= max_rows:
                break
            candidate = line.strip()
            if not candidate or len(candidate) > 100_000:
                continue
            try:
                rows.append(json.loads(candidate))
            except json.JSONDecodeError:
                continue
    elif suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, list):
            rows.extend(payload[:max_rows])
        elif isinstance(payload, dict):
            nested = next(
                (
                    payload[key]
                    for key in ("rows", "data", "examples", "train", "test")
                    if isinstance(payload.get(key), list)
                ),
                None,
            )
            rows.extend(nested[:max_rows] if isinstance(nested, list) else [payload])
    elif suffix in {".csv", ".tsv"}:
        reader = csv.DictReader(
            io.StringIO(text),
            delimiter="\t" if suffix == ".tsv" else ",",
        )
        if not reader.fieldnames or len(reader.fieldnames) > 64:
            return []
        for row in reader:
            rows.append(dict(row))
            if len(rows) >= max_rows:
                break
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not row:
            continue
        bounded = _bounded_huggingface_value(row)
        if isinstance(bounded, dict) and any(
            value is not None and value != "" and value != [] and value != {}
            for value in bounded.values()
        ):
            normalized.append(dict(list(bounded.items())[:64]))
        if len(normalized) >= max_rows:
            break
    return normalized


def _huggingface_observed_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"


def _infer_huggingface_observed_schema(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fields: dict[str, set[str]] = {}
    for row in rows:
        for key, value in row.items():
            name = str(key)[:200]
            fields.setdefault(name, set()).add(_huggingface_observed_type(value))
            if len(fields) >= 64:
                break
    return [
        {"name": name, "observed_types": sorted(types)}
        for name, types in fields.items()
    ]


def _huggingface_native_search_query(query: str) -> str:
    """Reduce web-style queries to terms supported by Hub dataset search."""
    normalized = " ".join(str(query).casefold().split())
    # Preserve narrow risk-discovery terms before collapsing generic red-team
    # language; otherwise a defamation query becomes only ``red teaming``.
    if "defamation" in normalized or "defamatory" in normalized:
        return "defamation"
    if "reputational harm" in normalized or "reputation harm" in normalized:
        return "reputational harm"
    if "jailbreak" in normalized:
        return "jailbreak"
    if "prompt injection" in normalized:
        return "prompt injection"
    if "adversarial" in normalized and (
        "prompt" in normalized or "trigger" in normalized
    ):
        return "adversarial prompts"
    if "red team" in normalized or "red-team" in normalized:
        return "red teaming"
    tokens = re.findall(r"[a-z0-9][a-z0-9_-]+", normalized)
    return " ".join(tokens[:2]) or "jailbreak"


class HuggingFaceExternalCollector(BaseExternalCollector):
    """Collect versioned dataset cards, schemas, and bounded sample evidence."""

    source_name = "huggingface"
    default_query = "jailbreak"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.authenticated = bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        )
        self._viewer_circuit_open = False
        self._viewer_circuit_reason = ""
        self._viewer_last_status = "not_attempted"
        self._search_cache: dict[str, list[RawCollectedItem]] = {}

    def collect(self, query: str = "", max_items: int = 20) -> list[RawCollectedItem]:
        return self.collect_query(query, max_items)

    def collect_query(self, query: str, max_items: int) -> list[RawCollectedItem]:
        native_query = _huggingface_native_search_query(self._query(query))
        cache_key = f"{native_query}\0{max_items}"
        if cache_key not in self._search_cache:
            self._search_cache[cache_key] = self._search_datasets(
                native_query,
                max_items,
            )
        return list(self._search_cache[cache_key])

    def _search_datasets(self, query: str, max_items: int) -> list[RawCollectedItem]:
        items: list[RawCollectedItem] = []
        supporting_items: list[RawCollectedItem] = []
        try:
            payload = _http_get_json(
                "https://huggingface.co/api/datasets",
                params={
                    "search": query,
                    "limit": min(max(max_items * 3, 10), 30),
                    "sort": "lastModified",
                },
                timeout=self.timeout,
                headers=_huggingface_headers(user_agent=self.user_agent),
            )
        except Exception as exc:
            return [
                _skipped_raw_item(
                    self.source_name,
                    f"Hugging Face search failed: {exc}",
                    is_error=True,
                )
            ]

        if not isinstance(payload, list):
            return items
        candidates = [dataset for dataset in payload if isinstance(dataset, dict)]
        candidates.sort(
            key=lambda dataset: (
                -_huggingface_dataset_candidate_score(dataset),
                str(dataset.get("id", "")).casefold(),
            )
        )
        # The most recently modified results often contain thin mirrors or
        # benchmark repacks. Inspect the full bounded search page so one useful
        # documented dataset is not hidden behind those candidates.
        candidate_limit = min(len(candidates), max(max_items * 5, 10))
        for dataset in candidates[:candidate_limit]:
            if len(items) >= max_items or not isinstance(dataset, dict):
                break
            collected_item = self._dataset_item(dataset, query=query)
            if collected_item is None:
                continue
            if collected_item.metadata["mechanism_extraction_eligible"]:
                items.append(collected_item)
            else:
                supporting_items.append(collected_item)
        return (items + supporting_items)[:max_items]

    def collect_dataset_for_paper(
        self,
        dataset_id: str,
        *,
        arxiv_id: str,
        paper_title: str,
        paper_url: str = "",
        paper_published_at: str = "",
        paper_direct_link: bool = False,
    ) -> RawCollectedItem:
        """Collect one explicit dataset and prove that it belongs to a paper."""
        requested_dataset = normalize_huggingface_dataset_id(dataset_id)
        canonical_id = normalize_arxiv_id(arxiv_id)
        quoted_id = urllib_parse.quote(requested_dataset, safe="/")
        try:
            payload = _http_get_json(
                f"https://huggingface.co/api/datasets/{quoted_id}",
                timeout=min(self.timeout, _HUGGINGFACE_HUB_TIMEOUT_SECONDS),
                headers=_huggingface_headers(user_agent=self.user_agent),
            )
        except Exception as exc:
            raise PaperBundleValidationError(
                "huggingface_companion_fetch_failed"
            ) from exc
        if not isinstance(payload, dict):
            raise PaperBundleValidationError("huggingface_companion_invalid_response")
        returned_dataset = str(payload.get("id", "")).strip()
        if returned_dataset.casefold() != requested_dataset.casefold():
            raise PaperBundleValidationError(
                "huggingface_companion_identity_mismatch"
            )
        item = self._dataset_item(
            payload,
            query="",
            detail_hint=payload,
            paper_title=paper_title,
            paper_url_or_id=paper_url or canonical_id,
        )
        if item is None:
            reason = str(
                getattr(self, "_last_dataset_rejection", "")
                or "not_eligible"
            )
            raise PaperBundleValidationError(
                f"huggingface_companion_{reason}"
            )
        match_basis = str(item.metadata.get("companion_identity_basis", ""))
        if not match_basis:
            raise PaperBundleValidationError("huggingface_companion_paper_mismatch")
        return replace(
            item,
            metadata={
                **item.metadata,
                "companion_identity_verified": True,
                "companion_identity_basis": match_basis,
                "paper_arxiv_id": canonical_id,
                "paper_title": paper_title,
                "paper_url": paper_url or f"https://arxiv.org/abs/{canonical_id}",
                "paper_published_at": paper_published_at,
            },
        )

    def _dataset_item(
        self,
        dataset: dict[str, Any],
        *,
        query: str,
        detail_hint: dict[str, Any] | None = None,
        paper_title: str = "",
        paper_url_or_id: str = "",
    ) -> RawCollectedItem | None:
        self._last_dataset_rejection = ""
        dataset_id = str(dataset.get("id", "")).strip()
        if not dataset_id:
            self._last_dataset_rejection = "missing_id"
            return None
        card_document, detail, card_provenance = self._get_dataset_card(
            dataset_id,
            revision_hint=str(dataset.get("sha", "") or ""),
            detail_hint=detail_hint,
        )
        if card_document is None:
            self._last_dataset_rejection = "missing_card_schema_or_samples"
            self._record_quality_rejection(self._last_dataset_rejection)
            return None
        identity_basis = ""
        if paper_url_or_id:
            identity_text = "\n".join(
                [
                    card_document.text,
                    str(dataset.get("description", "") or ""),
                    str(detail.get("description", "") or ""),
                    " ".join(str(tag) for tag in dataset.get("tags", [])),
                    " ".join(str(tag) for tag in detail.get("tags", [])),
                ]
            )
            identity_basis = _paper_identity_match_basis(
                identity_text,
                paper_title,
                paper_url_or_id,
            )
            if not identity_basis:
                self._last_dataset_rejection = "paper_mismatch"
                self._record_quality_rejection(self._last_dataset_rejection)
                return None

        documents: list[SourceEvidenceDocument] = [card_document]
        preview_documents = self._get_dataset_preview_documents(
            dataset_id,
            max_items=_HUGGINGFACE_PREVIEW_ROWS,
        )
        preview_roles = {document.role for document in preview_documents}
        preview_backend = "dataset_viewer"
        preview_revision = ""
        preview_revision_verified = False
        preview_fallback_reason = ""
        if not {"dataset-schema", "examples"}.issubset(preview_roles):
            repository_revision = str(
                detail.get("sha") or dataset.get("sha") or ""
            ).strip()
            hub_documents, hub_provenance = self._get_hub_raw_preview_documents(
                dataset_id,
                revision=repository_revision,
                max_items=_HUGGINGFACE_PREVIEW_ROWS,
            )
            hub_roles = {document.role for document in hub_documents}
            if {"dataset-schema", "examples"}.issubset(hub_roles):
                preview_documents = hub_documents
                preview_roles = hub_roles
                preview_backend = str(hub_provenance["backend"])
                preview_revision = str(hub_provenance["revision"])
                preview_revision_verified = bool(
                    hub_provenance["revision_verified"]
                )
                preview_fallback_reason = self._viewer_last_status
        documents.extend(preview_documents)
        if not {"dataset-schema", "examples"}.issubset(preview_roles):
            self._last_dataset_rejection = "missing_card_schema_or_samples"
            self._record_quality_rejection(self._last_dataset_rejection)
            return None
        evidence_text = _render_evidence_package(
            dataset_id,
            documents,
            max_total_chars=_HUGGINGFACE_MAX_TOTAL_CHARS,
        )
        single_commit = bool(
            preview_backend == "hub_raw"
            and preview_revision_verified
            and card_provenance["revision_verified"]
            and card_provenance["revision"] == preview_revision
            and card_document.revision == preview_revision
        )
        package_revision = preview_revision if single_commit else ""
        source_group = f"huggingface:{dataset_id}"
        if single_commit:
            source_group += f"@{package_revision}"
        extraction_method = (
            "huggingface_card_hub_raw_schema_samples"
            if preview_backend == "hub_raw"
            else "huggingface_card_dataset_viewer_schema_samples"
        )
        quality_metadata = _evidence_quality_metadata(
            evidence_text,
            documents,
            extraction_method=extraction_method,
            authenticated=self.authenticated,
            source_group=source_group,
            revision=package_revision,
        )
        if quality_metadata["evidence_quality_eligible"]:
            card_only_text = _render_evidence_package(
                dataset_id,
                [card_document],
                max_total_chars=_HUGGINGFACE_MAX_CARD_CHARS,
            )
            card_quality = _evidence_quality_metadata(
                card_only_text,
                [card_document],
                extraction_method="huggingface_card_only_assessment",
            )
            if not card_quality["mechanism_extraction_eligible"]:
                quality_metadata["mechanism_extraction_eligible"] = False
                quality_metadata["mechanism_extraction_reason"] = (
                    str(card_quality["mechanism_extraction_reason"])
                    if card_quality["evidence_quality_eligible"]
                    else "dataset_samples_without_documented_mechanism"
                )
                quality_metadata["advanced_mechanism_eligible"] = False
                quality_metadata["evidence_roles"] = [
                    role
                    for role in quality_metadata["evidence_roles"]
                    if role != "mechanism"
                ]
                quality_metadata["evidence_content_roles"] = [
                    role
                    for role in quality_metadata["evidence_content_roles"]
                    if role != "mechanism"
                ]
        if not quality_metadata["evidence_quality_eligible"]:
            self._last_dataset_rejection = str(
                quality_metadata["evidence_quality_reason"]
            )
            self._record_quality_rejection(self._last_dataset_rejection)
            return None
        return RawCollectedItem(
            text=evidence_text,
            source=self.source_name,
            url=f"https://huggingface.co/datasets/{dataset_id}",
            title=dataset_id,
            metadata={
                "dataset": dataset_id,
                "huggingface_search_query": query,
                "downloads": dataset.get("downloads", 0),
                "likes": dataset.get("likes", 0),
                "huggingface_authenticated": self.authenticated,
                "huggingface_card_kind": card_provenance["kind"],
                "huggingface_card_revision": card_provenance["revision"],
                "huggingface_card_revision_verified": card_provenance[
                    "revision_verified"
                ],
                "huggingface_preview_backend": preview_backend,
                "huggingface_preview_fallback_reason": preview_fallback_reason,
                "huggingface_preview_revision": preview_revision,
                "huggingface_preview_revision_verified": preview_revision_verified,
                "huggingface_package_revision_scope": (
                    "single_commit" if single_commit else "mixed_or_unpinned"
                ),
                "source_published_at": _normalize_timestamp(
                    dataset.get("createdAt") or detail.get("createdAt")
                ),
                "source_updated_at": _normalize_timestamp(
                    dataset.get("lastModified") or detail.get("lastModified")
                ),
                "source_effective_at": _normalize_timestamp(
                    dataset.get("lastModified")
                    or detail.get("lastModified")
                    or dataset.get("createdAt")
                    or detail.get("createdAt")
                ),
                **(
                    {
                        "companion_identity_verified": True,
                        "companion_identity_basis": identity_basis,
                    }
                    if identity_basis
                    else {}
                ),
                **quality_metadata,
            },
        )

    def _scrape_dataset(
        self, dataset_id: str, max_items: int
    ) -> list[RawCollectedItem]:
        documents = self._get_dataset_preview_documents(
            dataset_id,
            max_items=max_items,
        )
        return [
            RawCollectedItem(
                text=document.text,
                source=self.source_name,
                url=f"https://huggingface.co/datasets/{dataset_id}",
                title=dataset_id,
                metadata={
                    "dataset": dataset_id,
                    "evidence_role": document.role,
                    "evidence_path": document.path,
                },
            )
            for document in documents
            if document.role == "examples"
        ]

    def _get_dataset_preview_documents(
        self,
        dataset_id: str,
        *,
        max_items: int,
    ) -> list[SourceEvidenceDocument]:
        if self._viewer_circuit_open:
            self._viewer_last_status = (
                f"circuit_open_after:{self._viewer_circuit_reason}"
                if self._viewer_circuit_reason
                else "circuit_open_after_unavailable"
            )
            return []

        viewer_timeout = min(self.timeout, _HUGGINGFACE_VIEWER_TIMEOUT_SECONDS)
        try:
            splits_payload = _http_get_json(
                "https://datasets-server.huggingface.co/splits",
                params={"dataset": dataset_id},
                timeout=viewer_timeout,
                headers=_huggingface_headers(user_agent=self.user_agent),
            )
            splits = (
                splits_payload.get("splits", [])
                if isinstance(splits_payload, dict)
                else []
            )
            choices = [row for row in splits if isinstance(row, dict)]
            if not choices:
                self._viewer_last_status = "no_splits"
                return []
            choices.sort(
                key=lambda row: (
                    str(row.get("split", "")) != "train",
                    str(row.get("config", "")) != "default",
                    str(row.get("config", "")),
                    str(row.get("split", "")),
                )
            )
        except Exception as exc:
            # Dataset Viewer outages should not multiply the full crawl timeout
            # by every candidate. Open a per-collector circuit and use the Hub
            # repository fallback for the remaining candidates.
            self._viewer_circuit_open = True
            self._viewer_last_status = (
                f"splits_unavailable:{type(exc).__name__}"
            )
            self._viewer_circuit_reason = self._viewer_last_status
            return []

        documents: list[SourceEvidenceDocument] = [
            SourceEvidenceDocument(
                path="dataset-server/splits.json",
                role="overview",
                text=json.dumps(
                    {"dataset": dataset_id, "splits": choices},
                    ensure_ascii=False,
                    default=str,
                    indent=2,
                )[:_HUGGINGFACE_MAX_PREVIEW_CHARS],
                url=f"https://datasets-server.huggingface.co/splits?dataset={urllib_parse.quote(dataset_id, safe='')}",
                revision="",
            )
        ]
        remaining_rows = max_items
        first_rows_errors = 0
        for selected in choices[:2]:
            if remaining_rows <= 0:
                break
            config = str(selected.get("config", "default"))
            split = str(selected.get("split", "train"))
            try:
                payload = _http_get_json(
                    "https://datasets-server.huggingface.co/first-rows",
                    params={
                        "dataset": dataset_id,
                        "config": config,
                        "split": split,
                    },
                    timeout=viewer_timeout,
                    headers=_huggingface_headers(user_agent=self.user_agent),
                )
            except Exception:
                first_rows_errors += 1
                continue
            raw_features = (
                payload.get("features", []) if isinstance(payload, dict) else []
            )
            features = [
                feature
                for feature in raw_features
                if isinstance(feature, dict) and str(feature.get("name", "")).strip()
            ]
            if features:
                schema_text = json.dumps(
                    {
                        "dataset": dataset_id,
                        "config": config,
                        "split": split,
                        "features": features,
                    },
                    ensure_ascii=False,
                    default=str,
                    indent=2,
                )
                documents.append(
                    SourceEvidenceDocument(
                        path=f"config/{config}/split/{split}/features.json",
                        role="dataset-schema",
                        text=schema_text[:_HUGGINGFACE_MAX_PREVIEW_CHARS],
                        url=(
                            "https://datasets-server.huggingface.co/first-rows?"
                            + urllib_parse.urlencode(
                                {
                                    "dataset": dataset_id,
                                    "config": config,
                                    "split": split,
                                }
                            )
                        ),
                        revision="",
                        size_bytes=len(schema_text.encode("utf-8", errors="replace")),
                        truncated=len(schema_text) > _HUGGINGFACE_MAX_PREVIEW_CHARS,
                    )
                )
            # A row sample without the corresponding feature declaration is not
            # enough to establish dataset schema. Keep schema and examples
            # coupled to the same Dataset Viewer response.
            if not features:
                continue
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            for row_index, row in enumerate(rows[:remaining_rows]):
                row_data = row.get("row", {}) if isinstance(row, dict) else row
                text = json.dumps(row_data, ensure_ascii=False, default=str, indent=2)
                if len(text) <= 50:
                    continue
                bounded = text[: min(3000, _HUGGINGFACE_MAX_PREVIEW_CHARS)]
                documents.append(
                    SourceEvidenceDocument(
                        path=f"config/{config}/split/{split}/row/{row_index}.json",
                        role="examples",
                        text=bounded,
                        url=f"https://huggingface.co/datasets/{dataset_id}/viewer/{config}/{split}",
                        revision="",
                        size_bytes=len(text.encode("utf-8", errors="replace")),
                        truncated=len(text) > len(bounded),
                    )
                )
                remaining_rows -= 1
                if remaining_rows <= 0:
                    break
        roles = {document.role for document in documents}
        if {"dataset-schema", "examples"}.issubset(roles):
            self._viewer_last_status = "ok"
        elif first_rows_errors and not roles.intersection(
            {"dataset-schema", "examples"}
        ):
            self._viewer_circuit_open = True
            self._viewer_last_status = "first_rows_unavailable"
            self._viewer_circuit_reason = self._viewer_last_status
        else:
            self._viewer_last_status = "incomplete"
        return documents

    def _get_hub_raw_preview_documents(
        self,
        dataset_id: str,
        *,
        revision: str,
        max_items: int,
    ) -> tuple[list[SourceEvidenceDocument], dict[str, Any]]:
        provenance: dict[str, Any] = {
            "backend": "hub_raw",
            "revision": "",
            "revision_verified": False,
            "path": "",
        }
        revision = str(revision).strip()
        if not _is_huggingface_commit_sha(revision):
            return [], provenance

        quoted_id = urllib_parse.quote(dataset_id, safe="/")
        quoted_revision = urllib_parse.quote(revision, safe="")
        tree_url = (
            f"https://huggingface.co/api/datasets/{quoted_id}/tree/"
            f"{quoted_revision}"
        )
        tree_payload: list[Any] = []
        next_url = tree_url
        next_params: dict[str, Any] | None = {
            "recursive": "true",
            "expand": "false",
        }
        for _page in range(_HUGGINGFACE_TREE_MAX_PAGES):
            try:
                page_payload, response_headers = _http_get_json_page(
                    next_url,
                    params=next_params,
                    timeout=min(self.timeout, _HUGGINGFACE_HUB_TIMEOUT_SECONDS),
                    headers=_huggingface_headers(user_agent=self.user_agent),
                )
            except Exception:
                return [], provenance
            if not isinstance(page_payload, list):
                return [], provenance
            tree_payload.extend(page_payload)
            if len(tree_payload) >= _HUGGINGFACE_TREE_MAX_ENTRIES:
                tree_payload = tree_payload[:_HUGGINGFACE_TREE_MAX_ENTRIES]
                break
            candidate_next = _http_next_link(response_headers.get("link", ""))
            if not candidate_next:
                break
            parsed_next = urllib_parse.urlsplit(candidate_next)
            parsed_tree = urllib_parse.urlsplit(tree_url)
            if (
                _url_origin(candidate_next) != _url_origin(tree_url)
                or parsed_next.path != parsed_tree.path
            ):
                return [], provenance
            next_url = _ensure_public_http_url(candidate_next, resolve_dns=False)
            next_params = None
        if not tree_payload:
            return [], provenance

        for entry in _select_huggingface_raw_data_files(tree_payload):
            path = str(entry.get("path", "")).strip()
            if not path:
                continue
            quoted_path = urllib_parse.quote(path, safe="/")
            raw_url = (
                f"https://huggingface.co/datasets/{quoted_id}/resolve/"
                f"{quoted_revision}/{quoted_path}"
            )
            headers = _huggingface_headers(user_agent=self.user_agent)
            headers.update(
                {
                    "Range": f"bytes=0-{_HUGGINGFACE_RAW_MAX_BYTES - 1}",
                    "Accept-Encoding": "identity",
                }
            )
            try:
                raw_bytes = _http_get_bytes(
                    raw_url,
                    timeout=min(self.timeout, _HUGGINGFACE_HUB_TIMEOUT_SECONDS),
                    headers=headers,
                    max_bytes=_HUGGINGFACE_RAW_MAX_BYTES,
                )
            except Exception:
                continue
            raw_text = raw_bytes.decode("utf-8", errors="replace")
            rows = _parse_huggingface_structured_rows(
                raw_text,
                path=path,
                max_rows=max_items,
            )
            fields = _infer_huggingface_observed_schema(rows)
            if not rows or not fields:
                continue

            source_file_size = int(entry.get("size", 0) or 0)
            source_truncated = bool(
                source_file_size > len(raw_bytes) or len(rows) >= max_items
            )
            source_sampling = {
                "source_file_size_bytes": source_file_size,
                "raw_bytes_read": len(raw_bytes),
                "byte_range": (
                    f"bytes=0-{len(raw_bytes) - 1}" if raw_bytes else "bytes=0--1"
                ),
                "row_limit": max_items,
                "sampled_rows": len(rows),
                "source_sampled": source_truncated,
                "source_truncated": source_truncated,
                "raw_chunk_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            }

            schema_text = json.dumps(
                {
                    "schema_kind": "observed_from_bounded_repository_rows",
                    "dataset": dataset_id,
                    "path": path,
                    "revision": revision,
                    "sampled_rows": len(rows),
                    "source_sampling": source_sampling,
                    "fields": fields,
                },
                ensure_ascii=False,
                default=str,
                indent=2,
            )
            examples_text = json.dumps(
                {
                    "dataset": dataset_id,
                    "path": path,
                    "revision": revision,
                    "source_sampling": source_sampling,
                    "rows": rows,
                },
                ensure_ascii=False,
                default=str,
                indent=2,
            )
            documents = [
                SourceEvidenceDocument(
                    path=f"{path}#observed-schema.json",
                    role="dataset-schema",
                    text=schema_text[:_HUGGINGFACE_MAX_PREVIEW_CHARS],
                    url=raw_url,
                    revision=revision,
                    size_bytes=len(schema_text.encode("utf-8", errors="replace")),
                    truncated=len(schema_text) > _HUGGINGFACE_MAX_PREVIEW_CHARS,
                    provenance=source_sampling,
                ),
                SourceEvidenceDocument(
                    path=f"{path}#sample-rows.json",
                    role="examples",
                    text=examples_text[:_HUGGINGFACE_MAX_PREVIEW_CHARS],
                    url=raw_url,
                    revision=revision,
                    size_bytes=len(examples_text.encode("utf-8", errors="replace")),
                    truncated=len(examples_text) > _HUGGINGFACE_MAX_PREVIEW_CHARS,
                    provenance=source_sampling,
                ),
            ]
            return (
                documents,
                {
                    "backend": "hub_raw",
                    "revision": revision,
                    "revision_verified": True,
                    "path": path,
                },
            )
        return [], provenance

    def _get_dataset_card(
        self,
        dataset_id: str,
        *,
        revision_hint: str = "",
        detail_hint: dict[str, Any] | None = None,
    ) -> tuple[
        SourceEvidenceDocument | None,
        dict[str, Any],
        dict[str, Any],
    ]:
        detail: dict[str, Any] = dict(detail_hint or {})
        if not detail:
            try:
                payload = _http_get_json(
                    f"https://huggingface.co/api/datasets/{dataset_id}",
                    timeout=min(self.timeout, 15),
                    headers=_huggingface_headers(user_agent=self.user_agent),
                )
                if isinstance(payload, dict):
                    detail = payload
            except Exception:
                pass

        quoted_id = urllib_parse.quote(dataset_id, safe="/")
        # This is the exact ref used by the resolve request. Dates such as
        # lastModified describe freshness, not an immutable repository revision.
        revision = str(detail.get("sha") or revision_hint or "main")
        quoted_revision = urllib_parse.quote(revision, safe="")
        resolve_url = (
            f"https://huggingface.co/datasets/{quoted_id}/resolve/"
            f"{quoted_revision}/README.md"
        )
        try:
            card = _http_get_text(
                resolve_url,
                timeout=min(self.timeout, 15),
                # The redirect handler retains auth on Hugging Face and strips it
                # before any cross-origin CDN redirect.
                headers=_huggingface_headers(user_agent=self.user_agent),
            ).strip()
            if card:
                bounded = card[:_HUGGINGFACE_MAX_CARD_CHARS]
                return (
                    SourceEvidenceDocument(
                        path="README.md",
                        role="overview",
                        text=bounded,
                        url=resolve_url,
                        revision=revision,
                        size_bytes=len(card.encode("utf-8", errors="replace")),
                        truncated=len(card) > len(bounded),
                    ),
                    detail,
                    {
                        "kind": "repository_readme",
                        "revision": revision,
                        "revision_verified": _is_huggingface_commit_sha(revision),
                    },
                )
        except Exception:
            pass
        api_url = f"https://huggingface.co/api/datasets/{quoted_id}"
        description = str(detail.get("description", "") or "").strip()
        if description:
            return (
                _huggingface_api_description_document(
                    description,
                    url=api_url,
                    path="api/dataset-description.txt",
                ),
                detail,
                {
                    "kind": "api_description",
                    "revision": "",
                    "revision_verified": False,
                },
            )
        card_data = detail.get("cardData", {})
        if isinstance(card_data, dict):
            description = str(card_data.get("description", "")).strip()
            if description:
                return (
                    _huggingface_api_description_document(
                        description,
                        url=api_url,
                        path="api/card-data-description.txt",
                    ),
                    detail,
                    {
                        "kind": "api_card_data",
                        "revision": "",
                        "revision_verified": False,
                    },
                )
        if isinstance(card_data, str):
            description = card_data.strip()
            if description:
                return (
                    _huggingface_api_description_document(
                        description,
                        url=api_url,
                        path="api/card-data.txt",
                    ),
                    detail,
                    {
                        "kind": "api_card_data",
                        "revision": "",
                        "revision_verified": False,
                    },
                )
        return (
            None,
            detail,
            {"kind": "missing", "revision": "", "revision_verified": False},
        )


def _huggingface_api_description_document(
    text: str,
    *,
    url: str,
    path: str,
) -> SourceEvidenceDocument:
    """Represent API metadata honestly instead of labeling it as a README."""
    bounded = text[:_HUGGINGFACE_MAX_CARD_CHARS]
    return SourceEvidenceDocument(
        path=path,
        role="overview",
        text=bounded,
        url=url,
        revision="",
        size_bytes=len(text.encode("utf-8", errors="replace")),
        truncated=len(text) > len(bounded),
    )




def _google_result_companion_id(url: str, source: str) -> str:
    """Extract one canonical companion identifier from a search-result URL."""
    parsed = urllib_parse.urlsplit(str(url).strip())
    hostname = (parsed.hostname or "").rstrip(".").casefold()
    parts = [urllib_parse.unquote(part) for part in parsed.path.split("/") if part]
    try:
        if source == "github":
            if hostname not in {"github.com", "www.github.com"} or len(parts) < 2:
                return ""
            if parts[0].casefold() in {
                "about",
                "apps",
                "collections",
                "enterprise",
                "features",
                "marketplace",
                "orgs",
                "search",
                "settings",
                "sponsors",
                "topics",
            }:
                return ""
            return normalize_github_repo(f"{parts[0]}/{parts[1]}")
        if source == "huggingface":
            if hostname not in {"huggingface.co", "www.huggingface.co"}:
                return ""
            if len(parts) < 3 or parts[0].casefold() != "datasets":
                return ""
            return normalize_huggingface_dataset_id(
                f"{parts[1]}/{parts[2]}"
            )
    except PaperBundleValidationError:
        return ""
    return ""


def _discover_paper_companions_with_google(
    *,
    arxiv_id: str,
    paper_title: str,
    include_github: bool,
    include_huggingface: bool,
    timeout: int,
    user_agent: str,
    delay_seconds: float,
    external_search_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use Google discovery only to find companion URLs, never evidence text."""
    enabled_sources = [
        source
        for source, enabled in (
            ("github", include_github),
            ("huggingface", include_huggingface),
        )
        if enabled
    ]
    diagnostics: dict[str, Any] = {
        "attempted": bool(enabled_sources),
        "queries": [],
        "github": [],
        "huggingface": [],
        "result_records": [],
    }
    if not enabled_sources:
        return diagnostics

    title_phrase = " ".join(str(paper_title).replace('"', " ").split())[:180]
    identity_query = f'"{arxiv_id}"'
    if title_phrase:
        identity_query = f'"{title_phrase}" {identity_query}'
    collector = GoogleExternalCollector(
        timeout=timeout,
        user_agent=user_agent,
        delay_seconds=delay_seconds,
        search_config=external_search_config,
    )
    seen: dict[str, set[str]] = {"github": set(), "huggingface": set()}
    for source_index, source in enumerate(enabled_sources):
        site_clause = (
            "site:github.com"
            if source == "github"
            else "site:huggingface.co/datasets"
        )
        query = f"{identity_query} {site_clause}"
        discovery = collector.discover_links(query, 10)
        diagnostics["queries"].append(
            {
                "source": source,
                "query": query,
                "backend": discovery.backend,
                "result_count": len(discovery.results),
                "errors": list(discovery.errors),
            }
        )
        for result_title, result_url in discovery.results:
            if (
                len(diagnostics[source])
                >= MAX_GOOGLE_COMPANION_CANDIDATES_PER_SOURCE
            ):
                continue
            candidate = _google_result_companion_id(result_url, source)
            if not candidate or candidate.casefold() in seen[source]:
                continue
            seen[source].add(candidate.casefold())
            diagnostics[source].append(candidate)
            canonical_url = (
                f"https://github.com/{candidate}"
                if source == "github"
                else f"https://huggingface.co/datasets/{candidate}"
            )
            diagnostics["result_records"].append(
                {
                    "source": source,
                    "candidate": candidate,
                    "title": " ".join(str(result_title).split())[:240],
                    "url": canonical_url,
                    "backend": discovery.backend,
                }
            )
        collector._delay(source_index, len(enabled_sources))
    return diagnostics


def _validated_precollected_arxiv_primary(
    primary: RawCollectedItem,
    *,
    canonical_id: str,
) -> RawCollectedItem:
    """Validate an internally reused exact-arXiv fetch before trusting it."""
    if not isinstance(primary, RawCollectedItem):
        raise PaperBundleValidationError("precollected_arxiv_primary_invalid")
    if str(primary.source).strip().casefold() != "arxiv":
        raise PaperBundleValidationError("precollected_arxiv_source_mismatch")

    metadata = dict(primary.metadata or {})
    raw_metadata_id = str(metadata.get("arxiv_id", "")).strip()
    try:
        metadata_id = normalize_arxiv_id(raw_metadata_id)
    except PaperBundleValidationError as exc:
        raise PaperBundleValidationError(
            "precollected_arxiv_identity_missing"
        ) from exc
    if metadata_id.casefold() != canonical_id.casefold():
        raise PaperBundleValidationError("precollected_arxiv_identity_mismatch")

    try:
        url_id = normalize_arxiv_id(primary.url)
    except PaperBundleValidationError as exc:
        raise PaperBundleValidationError("precollected_arxiv_url_invalid") from exc
    if url_id.casefold() != canonical_id.casefold():
        raise PaperBundleValidationError("precollected_arxiv_url_mismatch")
    if (
        metadata.get("arxiv_exact_fetch") is not True
        or metadata.get("arxiv_identity_verified") is not True
        or not str(metadata.get("arxiv_identity_verification", "")).strip()
    ):
        raise PaperBundleValidationError(
            "precollected_arxiv_identity_unverified"
        )
    return primary


def collect_paper_evidence_bundle(
    arxiv_id: str,
    *,
    github_repo: str | None = None,
    huggingface_dataset: str | None = None,
    include_github: bool = True,
    include_huggingface: bool = True,
    google_companion_discovery: bool = False,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    delay_seconds: float = 0.0,
    max_source_age_days: int = DEFAULT_MAX_SOURCE_AGE_DAYS,
    as_of: datetime | str | None = None,
    external_search_config: dict[str, Any] | None = None,
    _precollected_primary: RawCollectedItem | None = None,
) -> list[ExternalCollectedItem]:
    """Collect one exact paper plus verified GitHub/Hugging Face companions.

    ``None``, an empty string, or ``"auto"`` discovers at most one companion
    per source. Canonical URLs directly present in the paper are preferred;
    optional Google discovery can supplement them, but its result text is never
    admitted as evidence. Explicit companions must independently mention the
    paper ID or sufficiently match its title. Any explicit mismatch aborts the
    whole bundle.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_source_age_days <= 0:
        raise ValueError("max_source_age_days must be positive")
    canonical_id = normalize_arxiv_id(arxiv_id)
    if _precollected_primary is None:
        arxiv_collector = ArxivExternalCollector(
            timeout=timeout,
            user_agent=user_agent,
        )
        primary = arxiv_collector.collect_id(canonical_id)
    else:
        primary = _validated_precollected_arxiv_primary(
            _precollected_primary,
            canonical_id=canonical_id,
        )
    paper_title = primary.title
    paper_url = primary.url or f"https://arxiv.org/abs/{canonical_id}"
    paper_published_at = str(primary.metadata.get("source_published_at", ""))
    bundle_id = f"arxiv:{canonical_id}"
    as_of_dt = _resolve_as_of(as_of)
    cutoff = as_of_dt - timedelta(days=max_source_age_days)

    github_links = list(
        primary.metadata.get("paper_companion_github_repos", []) or []
    )
    huggingface_links = list(
        primary.metadata.get("paper_companion_huggingface_datasets", []) or []
    )
    if not github_links and not huggingface_links:
        github_links, huggingface_links = _extract_paper_companion_links(primary.text)

    github_auto = not str(github_repo or "").strip() or str(
        github_repo or ""
    ).strip().casefold() == "auto"
    huggingface_auto = not str(huggingface_dataset or "").strip() or str(
        huggingface_dataset or ""
    ).strip().casefold() == "auto"
    google_discovery = {
        "attempted": False,
        "queries": [],
        "github": [],
        "huggingface": [],
        "result_records": [],
    }
    def _google_candidates_for(source: str) -> list[str]:
        discovered = _discover_paper_companions_with_google(
            arxiv_id=canonical_id,
            paper_title=paper_title,
            include_github=source == "github",
            include_huggingface=source == "huggingface",
            timeout=timeout,
            user_agent=user_agent,
            delay_seconds=delay_seconds,
            external_search_config=external_search_config,
        )
        google_discovery["attempted"] = bool(
            google_discovery.get("attempted") or discovered.get("attempted")
        )
        google_discovery["queries"].extend(discovered.get("queries", []))
        google_discovery["result_records"].extend(
            discovered.get("result_records", [])
        )
        existing = {
            str(candidate).casefold()
            for candidate in list(google_discovery.get(source, []) or [])
        }
        for candidate in list(discovered.get(source, []) or []):
            if str(candidate).casefold() not in existing:
                google_discovery[source].append(str(candidate))
                existing.add(str(candidate).casefold())
        return list(discovered.get(source, []) or [])

    raw_members: list[tuple[RawCollectedItem, str, str]] = [
        (primary, "primary", "exact_arxiv_id")
    ]
    auto_rejections: dict[str, list[str]] = {"github": [], "huggingface": []}
    admitted_companion_ids: dict[str, str] = {}

    github_candidates: list[tuple[str, str]] = []
    if include_github:
        if github_auto:
            github_candidates.extend(
                (value, "paper_direct_link") for value in github_links
            )
        else:
            github_candidates.append((str(github_repo), "explicit"))
    github_candidates = github_candidates[
        :MAX_PAPER_COMPANION_CANDIDATES_PER_SOURCE
    ]
    github_collector = (
        GitHubExternalCollector(
            timeout=timeout,
            user_agent=user_agent,
            delay_seconds=delay_seconds,
        )
        if include_github
        else None
    )

    def _try_github_candidates(candidates: list[tuple[str, str]]) -> bool:
        if github_collector is None:
            return False
        for candidate_index, (candidate, discovery_method) in enumerate(candidates):
            try:
                companion = github_collector.collect_repo_for_paper(
                    candidate,
                    arxiv_id=canonical_id,
                    paper_title=paper_title,
                    paper_url=paper_url,
                    paper_published_at=paper_published_at,
                    paper_direct_link=discovery_method == "paper_direct_link",
                )
            except PaperBundleValidationError as exc:
                if discovery_method == "explicit":
                    raise
                auto_rejections["github"].append(str(exc))
                github_collector._delay(candidate_index, len(candidates))
                continue
            companion = replace(
                companion,
                metadata={
                    **companion.metadata,
                    "paper_companion_discovery_method": discovery_method,
                },
            )
            companion = _annotate_freshness(
                companion,
                as_of=as_of_dt,
                cutoff=cutoff,
            )
            if (
                companion.metadata.get("freshness_eligible") is not True
                or companion.metadata.get("evidence_quality_eligible") is not True
            ):
                reason = "github_companion_not_fresh_quality_evidence"
                if discovery_method == "explicit":
                    raise PaperBundleValidationError(reason)
                auto_rejections["github"].append(reason)
                github_collector._delay(candidate_index, len(candidates))
                continue
            admitted_companion_ids["github"] = normalize_github_repo(candidate)
            raw_members.append(
                (
                    companion,
                    "companion",
                    str(
                        companion.metadata.get(
                            "companion_identity_basis", "paper_direct_link"
                        )
                    ),
                )
            )
            return True
        return False

    github_added = _try_github_candidates(github_candidates)
    if (
        not github_added
        and include_github
        and github_auto
        and google_companion_discovery
    ):
        google_candidates = [
            (candidate, "google_search")
            for candidate in _google_candidates_for("github")
            if candidate.casefold()
            not in {item.casefold() for item in github_links}
        ][
            : max(
                0,
                MAX_TOTAL_COMPANION_CANDIDATES_PER_SOURCE
                - len(github_candidates),
            )
        ]
        _try_github_candidates(google_candidates)

    huggingface_candidates: list[tuple[str, str]] = []
    if include_huggingface:
        if huggingface_auto:
            huggingface_candidates.extend(
                (value, "paper_direct_link") for value in huggingface_links
            )
        else:
            huggingface_candidates.append((str(huggingface_dataset), "explicit"))
    huggingface_candidates = huggingface_candidates[
        :MAX_PAPER_COMPANION_CANDIDATES_PER_SOURCE
    ]
    huggingface_collector = (
        HuggingFaceExternalCollector(
            timeout=timeout,
            user_agent=user_agent,
            delay_seconds=delay_seconds,
        )
        if include_huggingface
        else None
    )

    def _try_huggingface_candidates(candidates: list[tuple[str, str]]) -> bool:
        if huggingface_collector is None:
            return False
        for candidate_index, (candidate, discovery_method) in enumerate(candidates):
            try:
                companion = huggingface_collector.collect_dataset_for_paper(
                    candidate,
                    arxiv_id=canonical_id,
                    paper_title=paper_title,
                    paper_url=paper_url,
                    paper_published_at=paper_published_at,
                    paper_direct_link=discovery_method == "paper_direct_link",
                )
            except PaperBundleValidationError as exc:
                if discovery_method == "explicit":
                    raise
                auto_rejections["huggingface"].append(str(exc))
                huggingface_collector._delay(candidate_index, len(candidates))
                continue
            companion = replace(
                companion,
                metadata={
                    **companion.metadata,
                    "paper_companion_discovery_method": discovery_method,
                },
            )
            companion = _annotate_freshness(
                companion,
                as_of=as_of_dt,
                cutoff=cutoff,
            )
            if (
                companion.metadata.get("freshness_eligible") is not True
                or companion.metadata.get("evidence_quality_eligible") is not True
            ):
                reason = "huggingface_companion_not_fresh_quality_evidence"
                if discovery_method == "explicit":
                    raise PaperBundleValidationError(reason)
                auto_rejections["huggingface"].append(reason)
                huggingface_collector._delay(candidate_index, len(candidates))
                continue
            admitted_companion_ids["huggingface"] = (
                normalize_huggingface_dataset_id(candidate)
            )
            raw_members.append(
                (
                    companion,
                    "companion",
                    str(
                        companion.metadata.get(
                            "companion_identity_basis", "paper_direct_link"
                        )
                    ),
                )
            )
            return True
        return False

    huggingface_added = _try_huggingface_candidates(huggingface_candidates)
    if (
        not huggingface_added
        and include_huggingface
        and huggingface_auto
        and google_companion_discovery
    ):
        google_candidates = [
            (candidate, "google_search")
            for candidate in _google_candidates_for("huggingface")
            if candidate.casefold()
            not in {item.casefold() for item in huggingface_links}
        ][
            : max(
                0,
                MAX_TOTAL_COMPANION_CANDIDATES_PER_SOURCE
                - len(huggingface_candidates),
            )
        ]
        _try_huggingface_candidates(google_candidates)

    retained_members: list[ExternalCollectedItem] = []
    for raw_item, paper_role, relation_basis in raw_members:
        metadata = {
            **raw_item.metadata,
            "paper_bundle_id": bundle_id,
            "paper_role": paper_role,
            "paper_relation_verified": True,
            "paper_relation_basis": relation_basis,
            "paper_arxiv_id": canonical_id,
            "paper_title": paper_title,
            "paper_url": paper_url,
        }
        if paper_role == "primary":
            metadata.update(
                {
                    "paper_auto_companion_candidates": {
                        "github": github_links,
                        "huggingface": huggingface_links,
                    },
                    "paper_auto_companion_rejections": auto_rejections,
                    "paper_google_companion_discovery": google_discovery,
                }
            )
        annotated = _annotate_freshness(
            replace(raw_item, metadata=metadata),
            as_of=as_of_dt,
            cutoff=cutoff,
        )
        item = _external_from_raw_item(
            annotated,
            source_query=f"arxiv_id:{canonical_id}",
        )
        if paper_role == "companion" and (
            item.metadata.get("freshness_eligible") is not True
            or item.metadata.get("evidence_quality_eligible") is not True
        ):
            source = str(item.metadata.get("external_source", raw_item.source))
            discovery_method = str(
                item.metadata.get("paper_companion_discovery_method", "")
            )
            reason = f"{source}_companion_not_fresh_quality_evidence"
            if discovery_method == "explicit":
                raise PaperBundleValidationError(reason)
            auto_rejections.setdefault(source, []).append(reason)
            admitted_companion_ids.pop(source, None)
            continue
        retained_members.append(item)

    bundle_companion_sources = sorted(
        {
            str(item.metadata.get("external_source", ""))
            for item in retained_members
            if str(item.metadata.get("paper_role", "")).casefold()
            == "companion"
        }
    )
    google_confirmed_sources = sorted(
        source
        for source, admitted_id in admitted_companion_ids.items()
        if source in bundle_companion_sources
        and admitted_id.casefold()
        in {
            str(candidate).casefold()
            for candidate in list(google_discovery.get(source, []) or [])
        }
    )
    member_count = len(retained_members)
    return [
        replace(
            item,
            metadata={
                **item.metadata,
                "paper_bundle_member_count": member_count,
                "paper_bundle_companion_sources": bundle_companion_sources,
                "paper_google_confirmed_companion_sources": google_confirmed_sources,
                **(
                    {"paper_auto_companion_rejections": auto_rejections}
                    if str(item.metadata.get("paper_role", "")).casefold()
                    == "primary"
                    else {}
                ),
            },
        )
        for item in retained_members
    ]


_EXTERNAL_COLLECTOR_CLASSES: dict[str, type[BaseExternalCollector]] = {
    "github": GitHubExternalCollector,
    "arxiv": ArxivExternalCollector,
    "google": GoogleExternalCollector,
    "huggingface": HuggingFaceExternalCollector,
}


def normalize_external_sources(
    sources: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Normalize repeated/comma-separated external source names."""
    raw_sources = list(sources or EXTERNAL_DEFAULT_SOURCES)
    expanded: list[str] = []
    for raw_source in raw_sources:
        expanded.extend(part.strip().lower() for part in str(raw_source).split(","))

    if any(source == "all" for source in expanded):
        expanded = list(EXTERNAL_ALL_SOURCES)

    normalized: list[str] = []
    seen: set[str] = set()
    valid = set(EXTERNAL_ALL_SOURCES)
    for source in expanded:
        if not source:
            continue
        if source not in valid:
            raise ValueError(
                f"Unsupported external source: {source}. "
                f"Supported sources: {', '.join(EXTERNAL_ALL_SOURCES)}"
            )
        if source not in seen:
            normalized.append(source)
            seen.add(source)
    return normalized or list(EXTERNAL_DEFAULT_SOURCES)


def get_external_query_profile(
    profile_name: str = DEFAULT_EXTERNAL_QUERY_PROFILE,
) -> dict[str, list[QuerySpec]]:
    """Return a copy of one configured external-source query profile."""
    if profile_name not in EXTERNAL_QUERY_PROFILES:
        available = ", ".join(sorted(EXTERNAL_QUERY_PROFILES))
        raise ValueError(
            f"Unknown external-source query profile: {profile_name}. Available: {available}"
        )
    return {
        source: list(query_specs)
        for source, query_specs in EXTERNAL_QUERY_PROFILES[profile_name].items()
    }


def parse_source_query_overrides(
    values: list[str] | tuple[str, ...] | None,
) -> dict[str, list[str]]:
    """Parse repeated SOURCE=QUERY CLI values."""
    parsed: dict[str, list[str]] = {}
    for value in values or []:
        source, separator, query = str(value).partition("=")
        source = source.strip().lower()
        query = query.strip()
        if not separator or source not in EXTERNAL_ALL_SOURCES or not query:
            raise ValueError(
                "--source-query must use SOURCE=QUERY with a supported external source"
            )
        parsed.setdefault(source, []).append(query)
    return parsed


def resolve_external_query_map(
    *,
    sources: list[str] | tuple[str, ...] | None,
    query_profile: str = "",
    query: str = "",
    queries: list[str] | tuple[str, ...] | None = None,
    source_queries: dict[str, list[str]] | None = None,
) -> dict[str, list[QuerySpec]]:
    """Resolve profile, common, and source-native queries in stable order."""
    selected_sources = normalize_external_sources(sources)
    profile = get_external_query_profile(query_profile) if query_profile else {}
    common_queries = [
        str(value).strip() for value in (queries or []) if str(value).strip()
    ]
    if query.strip():
        common_queries.insert(0, query.strip())

    resolved: dict[str, list[QuerySpec]] = {}
    for source in selected_sources:
        specs = list(profile.get(source, []))
        specs.extend(QuerySpec("custom", value) for value in common_queries)
        specs.extend(
            QuerySpec("custom-source", value)
            for value in (source_queries or {}).get(source, [])
            if str(value).strip()
        )
        deduped: list[QuerySpec] = []
        seen: set[str] = set()
        for spec in specs:
            key = " ".join(spec.query.casefold().split())
            if key and key not in seen:
                deduped.append(spec)
                seen.add(key)
        resolved[source] = deduped
    return resolved


def collect_external_source_text(
    *,
    sources: list[str] | tuple[str, ...] | None = None,
    query: str = "",
    queries: list[str] | tuple[str, ...] | None = None,
    queries_by_source: dict[str, list[QuerySpec] | list[str]] | None = None,
    query_profile: str = "",
    source_queries: dict[str, list[str]] | None = None,
    max_items: int = 20,
    per_query_limit: int = 3,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    delay_seconds: float = 0.0,
    max_source_age_days: int = DEFAULT_MAX_SOURCE_AGE_DAYS,
    as_of: datetime | str | None = None,
    github_arxiv_discovery: bool = True,
    external_search_config: dict[str, Any] | None = None,
) -> list[ExternalCollectedItem]:
    """Collect external text using source-neutral source collectors.

    max_items is applied per source. Query results are merged round-robin so one
    broad query cannot consume the whole source budget.
    """
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if per_query_limit <= 0:
        raise ValueError("per_query_limit must be positive")
    if max_source_age_days <= 0:
        raise ValueError("max_source_age_days must be positive")

    as_of_dt = _resolve_as_of(as_of)
    cutoff = as_of_dt - timedelta(days=max_source_age_days)

    items: list[ExternalCollectedItem] = []
    selected_sources = normalize_external_sources(sources)
    query_map = resolve_external_query_map(
        sources=selected_sources,
        query_profile=query_profile,
        query=query,
        queries=queries,
        source_queries=source_queries,
    )
    if queries_by_source:
        for source, raw_specs in queries_by_source.items():
            if source not in selected_sources:
                continue
            query_map[source] = [
                spec
                if isinstance(spec, QuerySpec)
                else QuerySpec("custom-source", str(spec))
                for spec in raw_specs
                if (spec.query if isinstance(spec, QuerySpec) else str(spec)).strip()
            ]

    for source in selected_sources:
        collector_cls = _EXTERNAL_COLLECTOR_CLASSES[source]
        collector_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "user_agent": user_agent,
            "delay_seconds": delay_seconds,
        }
        if source == "google":
            collector_kwargs["search_config"] = external_search_config
        collector = collector_cls(**collector_kwargs)
        source_raw_items: list[RawCollectedItem] = []
        source_query_diagnostics: list[RawCollectedItem] = []
        seen_keys: set[str] = set()

        def _append_raw(raw_item: RawCollectedItem) -> None:
            key = _raw_item_key(raw_item)
            if key in seen_keys or len(source_raw_items) >= max_items:
                return
            seen_keys.add(key)
            source_raw_items.append(raw_item)

        specs = query_map.get(source, [])
        if not specs:
            default_query = str(
                getattr(collector, "default_query", "LLM jailbreak prompt")
            )
            specs = [QuerySpec("default", default_query)]
        ranked_specs = list(enumerate(specs))
        buckets: list[list[RawCollectedItem]] = []
        execution_rank = 0

        def _execute_query(
            query_rank: int,
            spec: QuerySpec,
            *,
            execution_wave: str,
        ) -> tuple[list[RawCollectedItem], list[RawCollectedItem], bool]:
            nonlocal execution_rank
            collect_query = getattr(collector, "collect_query", None)
            executed_query = _render_query_with_cutoff(
                source, spec.query, cutoff, as_of_dt
            )
            current_execution_rank = execution_rank
            execution_rank += 1
            request_limit = per_query_limit
            if spec.body_relevance_gate:
                # Overfetch only for body-gated discovery so irrelevant search
                # hits do not consume the requested result budget. The extra
                # work is capped at 20 candidates (or the caller's base limit).
                request_limit = max(
                    per_query_limit,
                    min(per_query_limit * 3, 20),
                )
            try:
                if isinstance(collector, GitHubExternalCollector) and callable(
                    collect_query
                ):
                    raw_bucket = collect_query(
                        executed_query,
                        request_limit,
                        family=spec.family,
                        cutoff=cutoff,
                        arxiv_discovery=github_arxiv_discovery,
                    )
                elif callable(collect_query):
                    raw_bucket = collect_query(executed_query, request_limit)
                else:
                    raw_bucket = collector.collect(
                        query=executed_query, max_items=request_limit
                    )
            except Exception as exc:
                raw_bucket = [
                    _skipped_raw_item(
                        source,
                        f"{source} query failed ({type(exc).__name__})",
                        is_error=True,
                    )
                ]

            admitted: list[RawCollectedItem] = []
            diagnostics: list[RawCollectedItem] = []
            rejection_counts: dict[str, int] = {}
            terminal_failure = False
            for raw_item in list(raw_bucket)[:request_limit]:
                if raw_item.metadata.get("diagnostic"):
                    terminal_failure = terminal_failure or bool(
                        raw_item.metadata.get("error")
                        or raw_item.metadata.get("skipped_reason")
                    )
                    diagnostics.append(raw_item)
                    continue
                if raw_item.metadata.get("error") or raw_item.metadata.get(
                    "skipped_reason"
                ):
                    terminal_failure = True
                    diagnostic_metadata = dict(raw_item.metadata or {})
                    diagnostic_metadata["diagnostic"] = True
                    diagnostics.append(replace(raw_item, metadata=diagnostic_metadata))
                    continue
                if not raw_item.text.strip():
                    continue
                gated_item, rejection_reason = _apply_query_body_relevance_gate(
                    raw_item,
                    spec,
                )
                if gated_item is None:
                    rejection_counts[rejection_reason] = (
                        rejection_counts.get(rejection_reason, 0) + 1
                    )
                    continue
                admitted.append(gated_item)
                if len(admitted) >= per_query_limit:
                    break

            if rejection_counts:
                diagnostics.append(
                    _query_gate_diagnostic_raw_item(source, rejection_counts)
                )

            def _annotate_query_item(raw_item: RawCollectedItem) -> RawCollectedItem:
                return _annotate_freshness(
                    _with_query_metadata(
                        raw_item,
                        spec.family,
                        spec.query,
                        query_rank,
                        query_profile,
                        executed_query=executed_query,
                        execution_rank=current_execution_rank,
                        execution_wave=execution_wave,
                    ),
                    as_of=as_of_dt,
                    cutoff=cutoff,
                )

            return (
                [_annotate_query_item(raw_item) for raw_item in admitted],
                [_annotate_query_item(raw_item) for raw_item in diagnostics],
                terminal_failure,
            )

        use_family_scheduler = bool(
            query_profile == REPOSITORY_REWRITE_QUERY_PROFILE
            and not (queries_by_source and source in queries_by_source)
        )
        required_reservations: dict[str, RawCollectedItem] = {}
        deferred_scheduler_items: list[RawCollectedItem] = []
        if use_family_scheduler:
            executed_ranks: set[int] = set()
            scheduled_usable_keys: set[str] = set()
            deferred_scheduler_keys: set[str] = set()
            covered_families: set[str] = set()

            def _scheduler_item_is_usable(
                raw_item: RawCollectedItem,
                spec: QuerySpec,
            ) -> bool:
                metadata = raw_item.metadata or {}
                if (
                    not raw_item.text.strip()
                    or metadata.get("diagnostic")
                    or not metadata.get("freshness_eligible")
                    or not metadata.get("evidence_quality_eligible")
                    or not metadata.get("mechanism_extraction_eligible")
                ):
                    return False
                if spec.family == "risk-domain-rewrite":
                    return bool(metadata.get("risk_domain_binding_eligible"))
                return True

            def _schedule_query(
                query_rank: int,
                spec: QuerySpec,
                *,
                execution_wave: str,
            ) -> int:
                executed_ranks.add(query_rank)
                bucket, diagnostics, _terminal_failure = _execute_query(
                    query_rank,
                    spec,
                    execution_wave=execution_wave,
                )
                source_query_diagnostics.extend(diagnostics)
                usable_bucket: list[RawCollectedItem] = []
                for raw_item in bucket:
                    key = _raw_item_key(raw_item)
                    if _scheduler_item_is_usable(raw_item, spec):
                        if key in scheduled_usable_keys:
                            continue
                        scheduled_usable_keys.add(key)
                        usable_bucket.append(raw_item)
                        continue
                    if (
                        key in scheduled_usable_keys
                        or key in deferred_scheduler_keys
                    ):
                        continue
                    deferred_scheduler_keys.add(key)
                    deferred_scheduler_items.append(raw_item)
                if usable_bucket:
                    buckets.append(usable_bucket)
                new_unique = len(usable_bucket)
                if new_unique:
                    covered_families.add(spec.family)
                    if (
                        spec.family in _REWRITE_REQUIRED_QUERY_FAMILIES
                        and (
                            spec.family not in required_reservations
                            or spec.companion_for_family
                        )
                    ):
                        # Prefer a successful companion as the family's reserved
                        # item: unlike a taxonomy-only primary hit, it is meant
                        # to join an operational method with a body-gated domain
                        # example.  The primary remains in its normal bucket and
                        # is retained whenever the source budget has room.
                        required_reservations[spec.family] = usable_bucket[0]
                return new_unique

            for family in _REWRITE_REQUIRED_QUERY_FAMILIES:
                for query_rank, spec in ranked_specs:
                    if (
                        spec.family != family
                        or spec.fallback_for_family
                        or family in covered_families
                    ):
                        continue
                    _schedule_query(
                        query_rank,
                        spec,
                        execution_wave="required-primary",
                    )

            for family in _REWRITE_REQUIRED_QUERY_FAMILIES:
                if family in covered_families:
                    continue
                for query_rank, spec in ranked_specs:
                    if spec.family != family or not spec.fallback_for_family:
                        continue
                    _schedule_query(
                        query_rank,
                        spec,
                        execution_wave="required-fallback",
                    )
                    if family in covered_families:
                        break

            # A narrow discovery query often finds a domain taxonomy or audit
            # page but misses the method repository that contains both the
            # operational rewrite and a domain-labelled evaluation row.  A
            # companion query supplements that evidence even after a primary
            # succeeds; if it already ran as fallback, executed_ranks prevents
            # a duplicate request.
            for family in _REWRITE_REQUIRED_QUERY_FAMILIES:
                for query_rank, spec in ranked_specs:
                    if (
                        spec.family != family
                        or not spec.companion_for_family
                        or query_rank in executed_ranks
                    ):
                        continue
                    _schedule_query(
                        query_rank,
                        spec,
                        execution_wave="required-companion",
                    )

            breadth_goal = min(
                _REWRITE_BREADTH_FAMILY_TARGET,
                max(0, max_items - len(_REWRITE_REQUIRED_QUERY_FAMILIES)),
            )
            breadth_covered: set[str] = set()
            breadth_families = list(
                dict.fromkeys(
                    spec.family
                    for _query_rank, spec in ranked_specs
                    if spec.family not in _REWRITE_REQUIRED_QUERY_FAMILIES
                    and not spec.fallback_for_family
                )
            )
            for family in breadth_families:
                if len(breadth_covered) >= breadth_goal:
                    break
                for query_rank, spec in ranked_specs:
                    if (
                        spec.family != family
                        or spec.fallback_for_family
                        or query_rank in executed_ranks
                    ):
                        continue
                    _schedule_query(
                        query_rank,
                        spec,
                        execution_wave="breadth",
                    )
                    if family in covered_families:
                        breadth_covered.add(family)
                        break

            for query_rank, spec in ranked_specs:
                if len(scheduled_usable_keys) >= max_items:
                    break
                if (
                    query_rank in executed_ranks
                    or spec.fallback_for_family
                    or spec.family in _REWRITE_REQUIRED_QUERY_FAMILIES
                ):
                    continue
                _schedule_query(
                    query_rank,
                    spec,
                    execution_wave="optional",
                )
        else:
            for query_rank, spec in ranked_specs:
                bucket, diagnostics, _terminal_failure = _execute_query(
                    query_rank,
                    spec,
                    execution_wave="round-robin",
                )
                buckets.append(bucket)
                source_query_diagnostics.extend(diagnostics)

        if use_family_scheduler:
            for family in _REWRITE_REQUIRED_QUERY_FAMILIES:
                reserved = required_reservations.get(family)
                if reserved is not None:
                    _append_raw(reserved)

        max_bucket_size = max((len(bucket) for bucket in buckets), default=0)
        for item_rank in range(max_bucket_size):
            for bucket in buckets:
                if item_rank < len(bucket):
                    _append_raw(bucket[item_rank])
                if len(source_raw_items) >= max_items:
                    break
            if len(source_raw_items) >= max_items:
                break
        if use_family_scheduler and len(source_raw_items) < max_items:
            for raw_item in deferred_scheduler_items:
                _append_raw(raw_item)
                if len(source_raw_items) >= max_items:
                    break

        for raw_item in source_raw_items:
            source_query = str(raw_item.metadata.get("source_query", ""))
            item = _external_from_raw_item(raw_item, source_query=source_query)
            if (
                item.text.strip()
                or item.metadata.get("error")
                or item.metadata.get("skipped_reason")
            ):
                items.append(item)
        for raw_diagnostic in source_query_diagnostics:
            source_query = str(raw_diagnostic.metadata.get("source_query", ""))
            items.append(
                _external_from_raw_item(
                    raw_diagnostic,
                    source_query=source_query,
                )
            )
        diagnostic_item = getattr(collector, "diagnostic_item", None)
        if callable(diagnostic_item):
            raw_diagnostic = diagnostic_item()
            if raw_diagnostic is not None:
                items.append(_external_from_raw_item(raw_diagnostic, source_query=""))
    return items


def _arxiv_id_from_result_url(url: str) -> str:
    try:
        return normalize_arxiv_id(url)
    except PaperBundleValidationError:
        return ""


def _paper_discovery_error(exc: Exception) -> str:
    if isinstance(exc, PaperBundleValidationError):
        reason = " ".join(str(exc).split())
        if re.fullmatch(r"[a-z0-9_:-]+", reason, re.IGNORECASE):
            return reason
    return type(exc).__name__


def collect_ranked_paper_evidence_batch(
    *,
    sources: list[str] | tuple[str, ...] | None = None,
    queries: list[str] | tuple[str, ...] | None = None,
    query_profile: str = DEFAULT_EXTERNAL_QUERY_PROFILE,
    source_queries: dict[str, list[str]] | None = None,
    discovery_limit: int = 10,
    per_query_limit: int = 3,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    delay_seconds: float = 0.0,
    max_source_age_days: int = DEFAULT_MAX_SOURCE_AGE_DAYS,
    as_of: datetime | str | None = None,
    google_companion_discovery: bool = True,
    desired_generation_candidates: int = 1,
    enable_candidate_backfill: bool = True,
    candidate_bundles_out: dict[str, list[ExternalCollectedItem]] | None = None,
    external_search_config: dict[str, Any] | None = None,
) -> tuple[list[ExternalCollectedItem], dict[str, Any]]:
    """Discover papers, retain a bounded eligible queue, and select one deterministically.

    Query discovery finishes before exact-paper verification begins. For a
    required body gate, ``discovery_limit`` sizes the initial verification
    batch and, when ``enable_candidate_backfill`` is true, a bounded candidate
    pool supplies replacements until the requested eligible-paper count is met.
    When GitHub or Hugging Face companions were requested,
    selection-ready means an eligible primary plus at least one verified
    companion; an eligible primary without companions remains a deterministic
    exhaustion fallback. When candidate backfill is enabled, strong attack-method
    citations found in an admitted benchmark primary may add a small bounded
    verification lane. Only exact arXiv content and relation-verified companion
    artifacts enter the returned
    writer snapshot. ``desired_generation_candidates`` counts verified eligible
    paper bundles; concrete examples remain a ranking preference rather than a
    condition that forces verification of the entire candidate pool. When
    ``candidate_bundles_out`` is supplied, it also receives every verified
    generation-eligible bundle, including citation-domain bridges.
    """
    if discovery_limit <= 0:
        raise ValueError("discovery_limit must be positive")
    if discovery_limit > MAX_PAPER_DISCOVERY_LIMIT:
        raise ValueError(
            f"discovery_limit must not exceed {MAX_PAPER_DISCOVERY_LIMIT}"
        )
    if per_query_limit <= 0:
        raise ValueError("per_query_limit must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_source_age_days <= 0:
        raise ValueError("max_source_age_days must be positive")
    if desired_generation_candidates <= 0:
        raise ValueError("desired_generation_candidates must be positive")
    if (
        not enable_candidate_backfill
        and desired_generation_candidates > discovery_limit
    ):
        raise ValueError(
            "desired_generation_candidates cannot exceed discovery_limit when "
            "candidate backfill is disabled"
        )
    if candidate_bundles_out is not None:
        candidate_bundles_out.clear()
    as_of_dt = _resolve_as_of(as_of)
    cutoff = as_of_dt - timedelta(days=max_source_age_days)
    selected_sources = normalize_external_sources(sources)
    if "arxiv" not in selected_sources:
        raise ValueError("arxiv-first discovery requires the arxiv external source")

    normalized_source_queries = {
        str(source).strip().casefold(): [
            str(query).strip() for query in raw_queries if str(query).strip()
        ]
        for source, raw_queries in dict(source_queries or {}).items()
    }
    unsupported_source_queries = sorted(
        source
        for source, raw_queries in normalized_source_queries.items()
        if source not in {"arxiv", "google"} and raw_queries
    )
    if unsupported_source_queries:
        raise ValueError(
            "arxiv-first --source-query supports only arxiv and google; "
            "companion sources are discovered from each verified paper: "
            + ", ".join(unsupported_source_queries)
        )

    def _discovery_body_gate(spec: QuerySpec) -> str:
        if spec.body_relevance_gate:
            return spec.body_relevance_gate
        if any(pattern.search(spec.query) for pattern in _QUERY_BODY_TOPIC_PATTERNS):
            return _TARGETED_DEFAMATION_BODY_GATE
        return ""

    explicit_scope_queries = [
        *[str(query).strip() for query in list(queries or []) if str(query).strip()],
        *normalized_source_queries.get("arxiv", []),
        *normalized_source_queries.get("google", []),
    ]
    required_body_relevance_gates = sorted(
        {
            gate_id
            for query in explicit_scope_queries
            if (
                gate_id := _discovery_body_gate(
                    QuerySpec("custom", query)
                )
            )
        }
    )
    # Narrow-domain and multi-paper runs may need spare candidates because search
    # snippets are not trusted as body evidence and individual full-paper checks
    # may fail. The requested discovery limit remains the size of the initial
    # verification batch. When enabled, this bounded pool supplies replacements
    # until the requested eligible queue is full, or becomes the eligible
    # no-companion fallback after bounded exhaustion.
    needs_candidate_backfill = bool(
        enable_candidate_backfill
        and (required_body_relevance_gates or desired_generation_candidates > 1)
    )
    candidate_pool_limit = (
        min(
            MAX_PAPER_DISCOVERY_LIMIT,
            max(
                discovery_limit * 2,
                discovery_limit + per_query_limit,
                desired_generation_candidates + per_query_limit,
            ),
        )
        if needs_candidate_backfill
        else discovery_limit
    )

    def _prioritize_discovery_specs(specs: list[QuerySpec]) -> list[QuerySpec]:
        # A narrow user query defines the desired *application domain*, not a
        # requirement that the same paper also introduce the attack mechanism.
        # Keep a bounded mechanism-discovery lane after the domain-first queries
        # so a domain benchmark and a general method paper can be considered as
        # separate, auditable evidence roles instead of forcing their accidental
        # intersection in one document.
        def _priority(spec: QuerySpec) -> int:
            if spec.family == "custom-source":
                return 0
            if spec.family == "custom":
                return 1
            if (
                required_body_relevance_gates
                and _discovery_body_gate(spec)
                and any(
                    anchor in spec.query.casefold()
                    for anchor in ("jailbreakbench", "harmbench", "ailuminate")
                )
            ):
                # These benchmark lanes combine public adversarial-prompt
                # artifacts with explicit hazard taxonomies.  Running them
                # before sparse legal-domain keyword queries improves recall
                # without bypassing the same full-body relevance gate.
                return 2
            # When an explicit query establishes a narrow domain scope, run
            # profile queries carrying the same body gate before generic
            # technique fallbacks. This keeps a transient/empty custom search
            # from filling the candidate batch with off-domain papers.
            if required_body_relevance_gates and _discovery_body_gate(spec):
                return 3
            if spec.family == "reproducible-artifact":
                return 4
            return 5

        return [
            spec
            for _index, spec in sorted(
                enumerate(specs),
                key=lambda value: (_priority(value[1]), value[0]),
            )
        ]

    arxiv_candidate_ids: list[str] = []
    arxiv_discovered_ids: list[str] = []
    provenance_by_id: dict[str, list[dict[str, Any]]] = {}
    domain_candidate_limit = (
        max(discovery_limit, candidate_pool_limit // 2)
        if required_body_relevance_gates
        else candidate_pool_limit
    )

    def _record_arxiv_id(candidate: str, provenance: dict[str, Any]) -> None:
        if not candidate or candidate in provenance_by_id:
            if candidate:
                provenance_by_id[candidate].append(provenance)
            return
        if (
            provenance.get("discovery_lane") == "domain"
            and sum(
                any(
                    route.get("discovery_lane") == "domain"
                    for route in provenance_by_id.get(arxiv_id, [])
                )
                for arxiv_id in arxiv_candidate_ids
            )
            >= domain_candidate_limit
        ):
            return
        if len(arxiv_candidate_ids) >= candidate_pool_limit:
            return
        arxiv_candidate_ids.append(candidate)
        provenance_by_id[candidate] = [provenance]

    arxiv_discovery_items: list[ExternalCollectedItem] = []
    arxiv_query_map = resolve_external_query_map(
        sources=["arxiv"],
        query_profile=query_profile,
        queries=queries,
        source_queries={
            "arxiv": normalized_source_queries.get("arxiv", []),
        },
    )
    prioritized_arxiv_specs = _prioritize_discovery_specs(
        arxiv_query_map.get("arxiv", [])
    ) or [
        QuerySpec("default", ArxivExternalCollector.default_query)
    ]
    arxiv_query_budget = min(
        len(prioritized_arxiv_specs),
        max(
            4,
            min(
                10,
                2
                * (
                    (candidate_pool_limit + per_query_limit - 1)
                    // per_query_limit
                ),
            ),
        ),
    )
    arxiv_specs = prioritized_arxiv_specs[:arxiv_query_budget]
    reserved_artifact_spec = next(
        (
            spec
            for spec in prioritized_arxiv_specs
            if spec.family == "reproducible-artifact"
        ),
        None,
    )
    if (
        reserved_artifact_spec is not None
        and all(spec.family != "reproducible-artifact" for spec in arxiv_specs)
    ):
        arxiv_specs[-1] = reserved_artifact_spec
    arxiv_artifact_lane_pending = bool(
        any(spec.family == "reproducible-artifact" for spec in arxiv_specs)
    )
    arxiv_collector = ArxivExternalCollector(
        timeout=timeout,
        user_agent=user_agent,
        delay_seconds=delay_seconds,
    )
    arxiv_queries_executed = 0
    arxiv_artifact_queries_executed = 0
    for query_rank, spec in enumerate(arxiv_specs):
        if len(arxiv_candidate_ids) >= candidate_pool_limit:
            break
        reserved_artifact_slots = int(
            arxiv_artifact_lane_pending
            and spec.family != "reproducible-artifact"
        )
        query_candidate_ceiling = candidate_pool_limit - reserved_artifact_slots
        if len(arxiv_candidate_ids) >= query_candidate_ceiling:
            continue
        executed_query = _render_query_with_cutoff(
            "arxiv", _render_arxiv_native_query(spec.query), cutoff, as_of_dt
        )
        arxiv_queries_executed += 1
        if spec.family == "reproducible-artifact":
            arxiv_artifact_lane_pending = False
            arxiv_artifact_queries_executed += 1
        remaining = query_candidate_ceiling - len(arxiv_candidate_ids)
        request_limit = per_query_limit
        if required_body_relevance_gates and _discovery_body_gate(spec):
            request_limit = min(max(per_query_limit * 2, per_query_limit), 20)
        raw_items = arxiv_collector.discover_ids(
            executed_query,
            min(request_limit, remaining),
        )
        for raw_item in raw_items:
            annotated = _annotate_freshness(
                _with_query_metadata(
                    raw_item,
                    spec.family,
                    spec.query,
                    query_rank,
                    query_profile,
                    executed_query=executed_query,
                    execution_rank=query_rank,
                    execution_wave="arxiv_id_discovery",
                ),
                as_of=as_of_dt,
                cutoff=cutoff,
            )
            item = _external_from_raw_item(
                annotated,
                source_query=spec.query,
            )
            arxiv_discovery_items.append(item)
            if item.metadata.get("diagnostic") or item.metadata.get("error"):
                continue
            raw_id = str(item.metadata.get("arxiv_id") or item.url)
            try:
                canonical_id = normalize_arxiv_id(raw_id)
            except PaperBundleValidationError:
                continue
            if (
                canonical_id not in provenance_by_id
                and len(arxiv_candidate_ids) >= query_candidate_ceiling
            ):
                continue
            _record_arxiv_id(
                canonical_id,
                {
                    "channel": "arxiv_api_query",
                    "discovery_lane": (
                        "domain" if _discovery_body_gate(spec) else "mechanism"
                    ),
                    "query_family": spec.family,
                    "source_query": spec.query,
                    "executed_query": executed_query,
                    "body_relevance_gate": _discovery_body_gate(spec),
                    "body_relevance_gate_required": bool(
                        _discovery_body_gate(spec)
                    ),
                    "title": item.title,
                    "url": item.url,
                },
            )
            if (
                canonical_id in arxiv_candidate_ids
                and canonical_id not in arxiv_discovered_ids
            ):
                arxiv_discovered_ids.append(canonical_id)
        arxiv_collector._delay(query_rank, len(arxiv_specs))

    google_candidate_ids: list[str] = []
    google_reserved_slots = (
        min(per_query_limit, discovery_limit - 1)
        if (
            "google" in selected_sources
            and required_body_relevance_gates
            and discovery_limit > 1
        )
        else 0
    )
    google_initial_target = min(
        discovery_limit,
        max(
            discovery_limit - len(arxiv_candidate_ids),
            google_reserved_slots,
        ),
    )
    google_candidate_target = (
        candidate_pool_limit
        if "google" in selected_sources and required_body_relevance_gates
        else google_initial_target
    )
    google_arxiv_discovery: list[dict[str, Any]] = []
    google_artifact_queries_executed = 0
    if "google" in selected_sources and google_candidate_target > 0:
        query_map = resolve_external_query_map(
            sources=["google"],
            query_profile=query_profile,
            queries=queries,
            source_queries={
                "google": normalized_source_queries.get("google", []),
            },
        )
        google_collector = GoogleExternalCollector(
            timeout=timeout,
            user_agent=user_agent,
            delay_seconds=delay_seconds,
            search_config=external_search_config,
        )
        google_specs = _prioritize_discovery_specs(
            query_map.get("google", [])
        )
        google_artifact_lane_pending = bool(
            any(spec.family == "reproducible-artifact" for spec in google_specs)
        )
        google_query_spread = min(len(google_specs), google_candidate_target)
        google_unique_per_query = (
            max(
                2 if required_body_relevance_gates else 1,
                (
                    google_candidate_target
                    + google_query_spread
                    - 1
                )
                // google_query_spread,
            )
            if google_query_spread
            else 0
        )
        for google_query_index, spec in enumerate(google_specs):
            if len(google_candidate_ids) >= google_candidate_target:
                break
            reserved_artifact_slots = int(
                google_artifact_lane_pending
                and spec.family != "reproducible-artifact"
            )
            query_candidate_ceiling = (
                google_candidate_target - reserved_artifact_slots
            )
            if len(google_candidate_ids) >= query_candidate_ceiling:
                continue
            if spec.family == "reproducible-artifact":
                google_artifact_lane_pending = False
                google_artifact_queries_executed += 1
            base_query = re.sub(
                r"\s+site:arxiv\.org(?:/abs)?\b",
                "",
                spec.query,
                flags=re.IGNORECASE,
            ).strip()
            discovery_query = _render_query_with_cutoff(
                "google",
                f"{base_query} site:arxiv.org",
                cutoff,
                as_of_dt,
            )
            google_request_limit = min(
                max(per_query_limit * 3, google_unique_per_query * 2, 5),
                20,
            )
            discovery = google_collector.discover_links(
                discovery_query,
                google_request_limit,
            )
            query_record: dict[str, Any] = {
                "query_family": spec.family,
                "query": discovery_query,
                "backend": discovery.backend,
                "errors": list(discovery.errors),
                "result_count": len(discovery.results),
                "candidate_limit": google_unique_per_query,
                "accepted_arxiv_ids": [],
            }
            query_unique_added = 0
            for result_title, result_url in discovery.results:
                canonical_id = _arxiv_id_from_result_url(result_url)
                if not canonical_id:
                    continue
                provenance = {
                    "channel": "google_arxiv_url",
                    "discovery_lane": (
                        "domain" if _discovery_body_gate(spec) else "mechanism"
                    ),
                    "backend": discovery.backend,
                    "query_family": spec.family,
                    "source_query": discovery_query,
                    "body_relevance_gate": _discovery_body_gate(spec),
                    "body_relevance_gate_required": bool(
                        _discovery_body_gate(spec)
                    ),
                    "title": " ".join(str(result_title).split())[:240],
                    "url": f"https://arxiv.org/abs/{canonical_id}",
                }
                if canonical_id in provenance_by_id:
                    provenance_by_id[canonical_id].append(provenance)
                elif (
                    len(google_candidate_ids) < query_candidate_ceiling
                    and query_unique_added < google_unique_per_query
                ):
                    google_candidate_ids.append(canonical_id)
                    provenance_by_id[canonical_id] = [provenance]
                    query_unique_added += 1
                else:
                    continue
                if canonical_id not in query_record["accepted_arxiv_ids"]:
                    query_record["accepted_arxiv_ids"].append(canonical_id)
            google_arxiv_discovery.append(query_record)
            google_collector._delay(google_query_index, len(google_specs))

    google_take = min(len(google_candidate_ids), google_initial_target)
    arxiv_keep = min(
        len(arxiv_candidate_ids),
        discovery_limit - google_take,
    )
    initial_verification_ids = [
        *arxiv_candidate_ids[:arxiv_keep],
        *google_candidate_ids[:google_take],
    ]
    # A reservation protects discovery-route diversity, but it must not leave
    # the initial verification batch under-filled when arXiv search is empty.
    if len(initial_verification_ids) < discovery_limit:
        for canonical_id in google_candidate_ids[google_take:]:
            if canonical_id in initial_verification_ids:
                continue
            initial_verification_ids.append(canonical_id)
            google_take += 1
            if len(initial_verification_ids) >= discovery_limit:
                break

    backfill_ids: list[str] = []
    remaining_google = google_candidate_ids[google_take:]
    remaining_arxiv = arxiv_candidate_ids[arxiv_keep:]
    if enable_candidate_backfill:
        for offset in range(max(len(remaining_google), len(remaining_arxiv))):
            # Google and the arXiv API rank different false positives. Alternate
            # their leftovers so one discovery backend cannot monopolize backfill.
            for source_pool in (remaining_google, remaining_arxiv):
                if offset >= len(source_pool):
                    continue
                canonical_id = source_pool[offset]
                if (
                    canonical_id in initial_verification_ids
                    or canonical_id in backfill_ids
                ):
                    continue
                backfill_ids.append(canonical_id)
                if (
                    len(initial_verification_ids) + len(backfill_ids)
                    >= candidate_pool_limit
                ):
                    break
            if (
                len(initial_verification_ids) + len(backfill_ids)
                >= candidate_pool_limit
            ):
                break
    verification_queue_ids = [*initial_verification_ids, *backfill_ids]
    verification_queue_limit = (
        min(
            MAX_PAPER_DISCOVERY_LIMIT,
            len(verification_queue_ids) + MAX_PAPER_CITATION_EXPANSION_CANDIDATES,
        )
        if enable_candidate_backfill
        else len(initial_verification_ids)
    )
    citation_expansion_arxiv_ids: list[str] = []
    citation_expansion_records: list[dict[str, Any]] = []
    citation_domain_bridge_records: list[dict[str, Any]] = []
    citation_domain_bridges_by_id: dict[str, list[dict[str, Any]]] = {}

    candidate_records: list[dict[str, Any]] = []
    verified_arxiv_ids: list[str] = []
    verified_bundles: dict[str, list[ExternalCollectedItem]] = {}
    include_github = "github" in selected_sources
    include_huggingface = "huggingface" in selected_sources
    companion_preferred = bool(include_github or include_huggingface)
    use_google_companions = bool(
        google_companion_discovery and "google" in selected_sources
    )
    metadata_required_primary_flags = (
        "freshness_eligible",
        "evidence_quality_eligible",
        "mechanism_extraction_eligible",
    )
    required_primary_flags = (
        *metadata_required_primary_flags,
        "source_authored_attack_method_claim",
        "text_attack_runtime_eligible",
    )

    def _gate_requirements_for(canonical_id: str) -> dict[str, bool]:
        # Global gates describe the requested application domain.  A candidate
        # discovered through the mechanism lane must still be assessed against
        # them, but the gate is required *for that paper* only when one of its
        # own discovery routes asserted the domain.  This distinction enables a
        # truthful mechanism-only fallback without claiming domain coverage.
        gate_requirements: dict[str, bool] = {
            gate_id: False for gate_id in required_body_relevance_gates
        }
        for provenance in provenance_by_id.get(canonical_id, []):
            gate_id = str(provenance.get("body_relevance_gate", "")).strip()
            if not gate_id:
                continue
            gate_requirements[gate_id] = bool(
                gate_requirements.get(gate_id, False)
                or provenance.get("body_relevance_gate_required", False)
            )
        return gate_requirements

    def _selection_ready(record: dict[str, Any]) -> bool:
        return bool(
            record.get("status") == "eligible"
            and (
                not required_body_relevance_gates
                or record.get("domain_evidence_status")
                in {"same_bundle", "citation_graph"}
            )
            and (
                not companion_preferred
                or bool(record.get("companion_sources", []))
            )
        )

    def _precheck_primary(
        primary: ExternalCollectedItem,
        canonical_id: str,
    ) -> tuple[
        list[dict[str, Any]],
        TextAttackRuntimeAssessment,
        list[str],
        bool,
    ]:
        provenance_routes = provenance_by_id.get(canonical_id, [])
        gate_requirements = _gate_requirements_for(canonical_id)
        body_gate_assessments: list[dict[str, Any]] = []
        for gate_id, required in gate_requirements.items():
            assessment = _assess_query_body_relevance(primary.text, gate_id)
            body_gate_assessments.append(
                {
                    "gate_id": gate_id,
                    "required": required,
                    "eligible": assessment.eligible,
                    "reason": assessment.reason,
                    "evidence_terms": list(assessment.evidence_terms),
                    "topic_term": assessment.topic_term,
                }
            )
        required_gate_assessments = [
            assessment
            for assessment in body_gate_assessments
            if assessment["required"]
        ]
        has_ungated_route = any(
            not str(provenance.get("body_relevance_gate", "")).strip()
            for provenance in provenance_routes
        )
        gate_failed = bool(
            required_gate_assessments
            and not any(
                assessment["eligible"]
                for assessment in required_gate_assessments
            )
        ) or bool(
            not required_gate_assessments
            and body_gate_assessments
            and not has_ungated_route
            and not any(
                assessment["eligible"] for assessment in body_gate_assessments
            )
        )
        citation_derived = any(
            provenance.get("channel") == "paper_citation"
            for provenance in provenance_routes
        )
        body_gate_deferred = bool(
            gate_failed
            and citation_derived
            and primary.metadata.get("source_authored_attack_method_claim") is True
        )
        rejection_reasons: list[str] = []
        text_attack_runtime_assessment = _assess_text_attack_runtime(
            primary.title,
            primary.text,
        )
        if not text_attack_runtime_assessment.eligible:
            rejection_reasons.append(
                "primary_text_attack_runtime_eligible_required"
            )
        for flag in metadata_required_primary_flags:
            if primary.metadata.get(flag) is not True:
                rejection_reasons.append(f"primary_{flag}_required")
        if not (
            primary.metadata.get("source_authored_attack_method_claim") is True
            or primary.metadata.get("advanced_mechanism_eligible") is True
        ):
            rejection_reasons.append(
                "primary_source_authored_attack_method_claim_required"
            )
        return (
            body_gate_assessments,
            text_attack_runtime_assessment,
            rejection_reasons,
            body_gate_deferred,
        )

    def _citation_candidates_for(
        primary: ExternalCollectedItem,
        canonical_id: str,
    ) -> list[dict[str, Any]]:
        combined: dict[str, dict[str, Any]] = {}
        raw_candidates = list(
            primary.metadata.get("paper_attack_method_arxiv_citations", []) or []
        )
        raw_candidates.extend(
            _ranked_attack_method_arxiv_citations(
                primary.text,
                current_arxiv_id=canonical_id,
            )
        )
        for raw in raw_candidates:
            if not isinstance(raw, dict):
                continue
            try:
                citation_id = normalize_arxiv_id(str(raw.get("arxiv_id", "")))
            except PaperBundleValidationError:
                continue
            if citation_id == canonical_id:
                continue
            try:
                score = int(raw.get("score", 0) or 0)
            except (TypeError, ValueError):
                continue
            context = _plain_citation_context(str(raw.get("context", "")))[:700]
            if score <= 0 or not context:
                continue
            local_source_attribution = bool(
                _METHOD_SOURCE_ATTRIBUTION_PATTERN.search(context)
            )
            record = {
                "arxiv_id": citation_id,
                "score": score,
                "context": context,
                "explicit_source_attribution": bool(
                    local_source_attribution
                ),
                "extraction_origin": (
                    "body_local" if local_source_attribution else str(
                        raw.get("extraction_origin", "")
                    )
                ),
                "evidence_terms": [
                    str(value)[:120]
                    for value in list(raw.get("evidence_terms", []) or [])[:6]
                    if str(value).strip()
                ],
            }
            existing = combined.get(citation_id)
            if existing is None or score > int(existing.get("score", 0) or 0):
                combined[citation_id] = record
        return sorted(
            combined.values(),
            key=lambda value: (-int(value["score"]), str(value["arxiv_id"])),
        )[:MAX_PAPER_CITATION_CANDIDATES_PER_PRIMARY]

    def _verified_fixed_companion(
        item: ExternalCollectedItem,
        *,
        expected_bundle_id: str,
        expected_arxiv_id: str,
    ) -> bool:
        """Require a relation-verified, revision-pinned companion snapshot."""
        return bool(
            str(item.metadata.get("external_source", "")).casefold()
            in {"github", "huggingface"}
            and str(item.metadata.get("paper_role", "")).casefold()
            == "companion"
            and item.metadata.get("paper_relation_verified") is True
            and item.metadata.get("companion_identity_verified") is True
            and str(item.metadata.get("paper_relation_basis", "")).strip()
            and str(item.metadata.get("paper_bundle_id", ""))
            == expected_bundle_id
            and str(item.metadata.get("paper_arxiv_id", ""))
            == expected_arxiv_id
            and item.metadata.get("freshness_eligible") is True
            and item.metadata.get("evidence_quality_eligible") is True
            and str(item.metadata.get("source_group", "")).strip()
            and str(item.metadata.get("source_revision", "")).strip()
        )

    def _verified_auditable_bundle_member(
        item: ExternalCollectedItem,
        *,
        expected_bundle_id: str,
        expected_arxiv_id: str,
    ) -> bool:
        """Accept only an exact primary or a revision-pinned companion."""
        source = str(item.metadata.get("external_source", "")).casefold()
        role = str(item.metadata.get("paper_role", "")).casefold()
        shared_identity = bool(
            item.metadata.get("paper_relation_verified") is True
            and str(item.metadata.get("paper_relation_basis", "")).strip()
            and str(item.metadata.get("paper_bundle_id", ""))
            == expected_bundle_id
            and str(item.metadata.get("paper_arxiv_id", ""))
            == expected_arxiv_id
            and item.metadata.get("freshness_eligible") is True
            and item.metadata.get("evidence_quality_eligible") is True
        )
        if not shared_identity:
            return False
        if source == "arxiv" and role == "primary":
            try:
                url_id = normalize_arxiv_id(item.url)
            except PaperBundleValidationError:
                return False
            return url_id == expected_arxiv_id
        return _verified_fixed_companion(
            item,
            expected_bundle_id=expected_bundle_id,
            expected_arxiv_id=expected_arxiv_id,
        )

    def _register_citation_domain_bridges(
        *,
        citation_id: str,
        citation: dict[str, Any],
        citation_source: ExternalCollectedItem,
        citing_arxiv_id: str,
        eligible_required_gates: list[dict[str, Any]],
        related_items: list[ExternalCollectedItem] | None,
    ) -> None:
        """Retain audited domain-only evidence for a cited method paper.

        The edge must be a body-local attack-method citation from an exact
        arXiv primary or revision-pinned companion.  Domain evidence may be a
        different member of that same verified bundle.  The copied item keeps
        its origin and is never promoted to implementation evidence.
        """
        if (
            related_items is None
            or citation.get("extraction_origin") != "body_local"
            or int(citation.get("score", 0) or 0) < 7
        ):
            return
        origin_bundle_id = f"arxiv:{citing_arxiv_id}"
        if not _verified_auditable_bundle_member(
            citation_source,
            expected_bundle_id=origin_bundle_id,
            expected_arxiv_id=citing_arxiv_id,
        ):
            return
        required_gate_ids = {
            str(assessment.get("gate_id", ""))
            for assessment in eligible_required_gates
            if str(assessment.get("gate_id", ""))
        }
        if not required_gate_ids:
            return

        bridge_bucket = citation_domain_bridges_by_id.setdefault(citation_id, [])
        ranked_domain_items = sorted(
            related_items,
            key=lambda item: (
                {
                    "github": 0,
                    "huggingface": 1,
                    "arxiv": 2,
                }.get(
                    str(item.metadata.get("external_source", "")).casefold(),
                    3,
                ),
                item.url,
            ),
        )
        for domain_item in ranked_domain_items:
            if not _verified_auditable_bundle_member(
                domain_item,
                expected_bundle_id=origin_bundle_id,
                expected_arxiv_id=citing_arxiv_id,
            ):
                continue
            gate_id = str(domain_item.metadata.get("query_body_gate_id", ""))
            if not (
                gate_id in required_gate_ids
                and domain_item.metadata.get("query_body_gate_required") is True
                and domain_item.metadata.get("query_body_relevance_eligible") is True
                and domain_item.metadata.get("risk_domain_binding_eligible") is True
            ):
                continue
            bridge_key = (
                origin_bundle_id,
                citation_source.url.casefold().rstrip("/"),
                domain_item.url.casefold().rstrip("/"),
                gate_id,
            )
            if any(candidate["bridge_key"] == bridge_key for candidate in bridge_bucket):
                continue
            audit_record = {
                "target_arxiv_id": citation_id,
                "citing_arxiv_id": citing_arxiv_id,
                "origin_bundle_id": origin_bundle_id,
                "citation_evidence_source": str(
                    citation_source.metadata.get("external_source", "")
                ),
                "citation_evidence_role": "companion",
                "citation_evidence_url": citation_source.url,
                "citation_score": int(citation.get("score", 0) or 0),
                "citation_explicit_source_attribution": bool(
                    citation.get("explicit_source_attribution") is True
                ),
                "citation_relation_strength": (
                    "explicit_source_attribution"
                    if citation.get("explicit_source_attribution") is True
                    else "body_local_method_citation"
                ),
                "domain_evidence_source": str(
                    domain_item.metadata.get("external_source", "")
                ),
                "domain_evidence_url": domain_item.url,
                "body_relevance_gate": gate_id,
                "usage": "domain_evidence_only",
                "attached": False,
            }
            bridge_bucket.append(
                {
                    "bridge_key": bridge_key,
                    "item": domain_item,
                    "citation": dict(citation),
                    "citation_source": citation_source,
                    "audit_record": audit_record,
                }
            )
            citation_domain_bridge_records.append(audit_record)
            # One strongest audited domain witness per citation is enough; the
            # attachment stage is intentionally bounded to the same cardinality.
            return

    def _attach_citation_domain_bridges(
        bundle: list[ExternalCollectedItem],
        canonical_id: str,
    ) -> tuple[list[ExternalCollectedItem], list[dict[str, Any]]]:
        """Attach bounded graph-linked evidence while preserving its origin."""
        primary_members = [
            item
            for item in bundle
            if str(item.metadata.get("paper_role", "")).casefold() == "primary"
            and str(item.metadata.get("external_source", "")).casefold() == "arxiv"
            and item.metadata.get("paper_relation_verified") is True
            and str(item.metadata.get("paper_bundle_id", ""))
            == f"arxiv:{canonical_id}"
        ]
        if len(primary_members) != 1:
            return bundle, []
        primary = primary_members[0]
        existing_urls = {item.url.casefold().rstrip("/") for item in bundle}
        candidates = sorted(
            citation_domain_bridges_by_id.get(canonical_id, []),
            key=lambda candidate: (
                0
                if str(
                    candidate["item"].metadata.get("external_source", "")
                ).casefold()
                == "github"
                else 1,
                str(candidate["item"].url),
            ),
        )
        attached: list[dict[str, Any]] = []
        expanded_bundle = list(bundle)
        for candidate in candidates:
            if len(attached) >= MAX_PAPER_CITATION_DOMAIN_BRIDGES:
                break
            domain_item = candidate["item"]
            normalized_url = domain_item.url.casefold().rstrip("/")
            if normalized_url in existing_urls:
                continue
            citation = candidate["citation"]
            citation_source = candidate["citation_source"]
            origin_metadata = domain_item.metadata
            bridge_metadata = {
                **origin_metadata,
                "paper_bundle_id": f"arxiv:{canonical_id}",
                "paper_role": "companion",
                "paper_relation_verified": True,
                "paper_relation_basis": "verified_citation_domain_bridge",
                "paper_arxiv_id": canonical_id,
                "paper_title": str(
                    primary.metadata.get("paper_title") or primary.title
                ),
                "paper_url": str(
                    primary.metadata.get("paper_url") or primary.url
                ),
                "paper_companion_usage": "domain_evidence_only",
                "paper_relation_scope": "domain_evidence_only",
                "paper_companion_discovery_method": (
                    "verified_citation_domain_bridge"
                ),
                "paper_bridge_relation_verified": True,
                "paper_bridge_origin_bundle_id": str(
                    origin_metadata.get("paper_bundle_id", "")
                ),
                "paper_bridge_origin_arxiv_id": str(
                    origin_metadata.get("paper_arxiv_id", "")
                ),
                "paper_bridge_origin_title": str(
                    origin_metadata.get("paper_title", "")
                ),
                "paper_bridge_origin_relation_basis": str(
                    origin_metadata.get("paper_relation_basis", "")
                ),
                "paper_bridge_citation_source": str(
                    citation_source.metadata.get("external_source", "")
                ),
                "paper_bridge_citation_source_url": citation_source.url,
                "paper_bridge_citation_score": int(
                    citation.get("score", 0) or 0
                ),
                "paper_bridge_citation_context": str(
                    citation.get("context", "")
                )[:700],
                "paper_bridge_citation_explicit_source_attribution": bool(
                    citation.get("explicit_source_attribution") is True
                ),
                "paper_bridge_citation_relation_strength": str(
                    candidate["audit_record"].get(
                        "citation_relation_strength", ""
                    )
                ),
                "mechanism_extraction_eligible": False,
                "advanced_mechanism_eligible": False,
                "source_authored_attack_method_claim": False,
            }
            expanded_bundle.append(
                replace(
                    domain_item,
                    source_query=(
                        f"arxiv_id:{canonical_id}:citation-domain-bridge"
                    ),
                    metadata=bridge_metadata,
                )
            )
            existing_urls.add(normalized_url)
            candidate["audit_record"]["attached"] = True
            attached.append(dict(candidate["audit_record"]))

        if not attached:
            return bundle, []
        companion_sources = sorted(
            {
                str(item.metadata.get("external_source", ""))
                for item in expanded_bundle
                if str(item.metadata.get("paper_role", "")).casefold()
                == "companion"
                and item.metadata.get("paper_relation_verified") is True
            }
        )
        domain_evidence_sources = sorted(
            {
                str(item.metadata.get("external_source", ""))
                for item in expanded_bundle
                if str(
                    item.metadata.get("paper_companion_usage", "")
                ).casefold()
                == "domain_evidence_only"
            }
        )
        member_count = len(expanded_bundle)
        expanded_bundle = [
            replace(
                item,
                metadata={
                    **item.metadata,
                    "paper_bundle_member_count": member_count,
                    "paper_bundle_companion_sources": companion_sources,
                    "paper_bundle_domain_evidence_sources": (
                        domain_evidence_sources
                    ),
                },
            )
            for item in expanded_bundle
        ]
        return expanded_bundle, attached

    def _expand_attack_method_citations(
        *,
        primary: ExternalCollectedItem,
        canonical_id: str,
        body_gate_assessments: list[dict[str, Any]],
        insert_after: int,
        related_items: list[ExternalCollectedItem] | None = None,
    ) -> None:
        eligible_required_gates = [
            assessment
            for assessment in body_gate_assessments
            if assessment.get("required") is True
            and assessment.get("eligible") is True
        ]
        if (
            not eligible_required_gates
            or primary.metadata.get("freshness_eligible") is not True
            or primary.metadata.get("evidence_quality_eligible") is not True
            or primary.metadata.get("source_authored_attack_method_claim") is True
            or not _BENCHMARK_PRIMARY_PATTERN.search(
                _query_body_without_package_headers(primary.text)
            )
        ):
            return

        if related_items is None:
            citation_sources: list[ExternalCollectedItem] = [primary]
        else:
            citation_sources = [
                item
                for item in related_items
                if str(item.metadata.get("paper_role", "")).casefold()
                in {"primary", "companion"}
                and item.metadata.get("paper_relation_verified") is True
                and item.metadata.get("freshness_eligible") is True
                and item.metadata.get("evidence_quality_eligible") is True
            ]
        insertion_offset = 0
        for citation_source in citation_sources:
            citation_source_name = str(
                citation_source.metadata.get("external_source", "arxiv")
            )
            citation_source_role = str(
                citation_source.metadata.get("paper_role", "primary")
            )
            for citation in _citation_candidates_for(
                citation_source,
                canonical_id,
            ):
                citation_id = str(citation["arxiv_id"])
                is_new_expansion = citation_id not in citation_expansion_arxiv_ids
                if (
                    is_new_expansion
                    and len(citation_expansion_arxiv_ids)
                    >= MAX_PAPER_CITATION_EXPANSION_CANDIDATES
                ):
                    continue
                provenance_records = [
                    {
                        "channel": "paper_citation",
                        "discovery_lane": "mechanism",
                        "query_family": "paper-citation-expansion",
                        "source_query": f"cited-by:{canonical_id}",
                        "body_relevance_gate": str(assessment["gate_id"]),
                        "body_relevance_gate_required": True,
                        "body_relevance_gate_deferred_to_bundle": True,
                        "citing_arxiv_id": canonical_id,
                        "citation_evidence_source": citation_source_name,
                        "citation_evidence_role": citation_source_role,
                        "citation_evidence_url": citation_source.url,
                        "citation_score": int(citation["score"]),
                        "citation_explicit_source_attribution": bool(
                            citation.get("explicit_source_attribution") is True
                        ),
                        "citation_context": str(citation["context"]),
                        "citation_evidence_terms": list(
                            citation.get("evidence_terms", [])
                        ),
                        "url": f"https://arxiv.org/abs/{citation_id}",
                    }
                    for assessment in eligible_required_gates
                ]
                existing_provenance = provenance_by_id.setdefault(citation_id, [])
                for provenance in provenance_records:
                    if provenance not in existing_provenance:
                        existing_provenance.append(provenance)
                if citation_id in verified_arxiv_ids:
                    continue

                insertion_index = min(
                    insert_after + 1 + insertion_offset,
                    len(verification_queue_ids),
                )
                if citation_id in verification_queue_ids:
                    current_index = verification_queue_ids.index(citation_id)
                    if current_index > insert_after:
                        verification_queue_ids.pop(current_index)
                        verification_queue_ids.insert(insertion_index, citation_id)
                        insertion_offset += 1
                elif len(verification_queue_ids) < verification_queue_limit:
                    verification_queue_ids.insert(insertion_index, citation_id)
                    insertion_offset += 1
                else:
                    continue
                _register_citation_domain_bridges(
                    citation_id=citation_id,
                    citation=citation,
                    citation_source=citation_source,
                    citing_arxiv_id=canonical_id,
                    eligible_required_gates=eligible_required_gates,
                    related_items=related_items,
                )
                if is_new_expansion:
                    citation_expansion_arxiv_ids.append(citation_id)
                citation_expansion_records.append(
                    {
                        "arxiv_id": citation_id,
                        "citing_arxiv_id": canonical_id,
                        "citation_evidence_source": citation_source_name,
                        "citation_evidence_role": citation_source_role,
                        "score": int(citation["score"]),
                        "citation_explicit_source_attribution": bool(
                            citation.get("explicit_source_attribution") is True
                        ),
                        "body_relevance_gates": [
                            str(assessment["gate_id"])
                            for assessment in eligible_required_gates
                        ],
                    }
                )

    def _annotate_bundle_body_relevance(
        bundle: list[ExternalCollectedItem],
        canonical_id: str,
    ) -> tuple[list[ExternalCollectedItem], list[dict[str, Any]], bool]:
        gate_requirements = _gate_requirements_for(canonical_id)
        desired_gate_ids = set(gate_requirements)
        rows: list[dict[str, Any]] = []
        by_item_index: dict[int, list[dict[str, Any]]] = {}
        for item_index, item in enumerate(bundle):
            paper_role = str(item.metadata.get("paper_role", "")).casefold()
            relation_eligible = bool(
                paper_role == "primary"
                or (
                    paper_role == "companion"
                    and item.metadata.get("paper_relation_verified") is True
                )
            )
            if not relation_eligible:
                continue
            for gate_id, required in gate_requirements.items():
                assessment = _assess_query_body_relevance(item.text, gate_id)
                payload = {
                    "gate_id": gate_id,
                    "required": required,
                    "eligible": assessment.eligible,
                    "reason": assessment.reason,
                    "evidence_terms": list(assessment.evidence_terms),
                    "topic_term": assessment.topic_term,
                    "item_index": item_index,
                    "paper_role": paper_role,
                    "external_source": str(
                        item.metadata.get("external_source", "")
                    ),
                }
                rows.append(payload)
                by_item_index.setdefault(item_index, []).append(payload)

        annotated: list[ExternalCollectedItem] = []
        for item_index, item in enumerate(bundle):
            matches = [
                row
                for row in by_item_index.get(item_index, [])
                if row["eligible"] is True
            ]
            if not matches:
                annotated.append(item)
                continue
            primary_match = next(
                (row for row in matches if row["required"] is True),
                matches[0],
            )
            annotated.append(
                replace(
                    item,
                    metadata={
                        **item.metadata,
                        "query_body_gate_required": bool(
                            primary_match["required"]
                        ),
                        "query_body_gate_id": str(primary_match["gate_id"]),
                        "query_body_relevance_eligible": True,
                        "query_body_relevance_reason": str(
                            primary_match["reason"]
                        ),
                        "query_body_relevance_terms": list(
                            primary_match["evidence_terms"]
                        ),
                        "query_body_relevance_topic_term": str(
                            primary_match["topic_term"]
                        ),
                        "query_body_relevance_assessments": [
                            {
                                key: value
                                for key, value in row.items()
                                if key not in {"item_index"}
                            }
                            for row in matches
                        ],
                    },
                )
            )
        desired_satisfied = bool(
            not desired_gate_ids
            or all(
                any(
                    row["eligible"] is True
                    and str(row["gate_id"]) == desired_gate_id
                    for row in rows
                )
                for desired_gate_id in desired_gate_ids
            )
        )
        return annotated, rows, desired_satisfied

    for paper_index, canonical_id in enumerate(verification_queue_ids):
        pending_citation_expansions = any(
            citation_id not in verified_arxiv_ids
            for citation_id in citation_expansion_arxiv_ids
        )
        if (
            all(
                initial_id in verified_arxiv_ids
                for initial_id in initial_verification_ids
            )
            and sum(
                record.get("status") == "eligible"
                for record in candidate_records
            )
            >= desired_generation_candidates
            and any(_selection_ready(record) for record in candidate_records)
            and not pending_citation_expansions
        ):
            break
        if paper_index > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        verified_arxiv_ids.append(canonical_id)
        record: dict[str, Any] = {
            "arxiv_id": canonical_id,
            "discovery_provenance": provenance_by_id.get(canonical_id, []),
            "verification_order": paper_index + 1,
            "verification_phase": (
                "citation_expansion"
                if any(
                    provenance.get("channel") == "paper_citation"
                    for provenance in provenance_by_id.get(canonical_id, [])
                )
                else "initial"
                if canonical_id in initial_verification_ids
                else "backfill"
            ),
            "status": "rejected",
            "rejection_reasons": [],
            "paper_bundle_id": "",
            "title": "",
            "companion_sources": [],
            "direct_companion_sources": [],
            "google_confirmed_companion_sources": [],
            "google_companion_discovery": {},
            "companion_discovery_attempted": False,
            "citation_companion_discovery_attempted": False,
            "citation_companion_sources": [],
            "citation_domain_bridge_sources": [],
            "citation_domain_bridge_count": 0,
            "body_relevance_assessments": [],
            "bundle_body_relevance_assessments": [],
            "body_relevance_gate_deferred_to_bundle": False,
            "generation_eligible": False,
            "mechanism_strength": "missing",
            "domain_evidence_status": (
                "deferred_to_promotion"
                if required_body_relevance_gates
                else "not_requested"
            ),
            "domain_binding_deferred": bool(required_body_relevance_gates),
            "selection_tier": "ineligible",
            "selection_policy": PAPER_SELECTION_POLICY,
            "selection_schema_version": PAPER_SELECTION_SCHEMA_VERSION,
            "selection_degradations": [],
            "selection_ready": False,
            "evidence_quality_score": 0.0,
            "example_evidence_status": "none",
            "example_evidence_score": 0,
            "example_evidence_signals": [],
            "example_evidence_source": "primary_bounded_body",
            "example_source_truncated": False,
            "example_source_bounding_method": "role_balanced_example_grounded_v2",
            "source_effective_at": "",
            "effective_rank": 0.0,
            "score": 0.0,
            "precheck": {
                "status": "pending",
                "rejection_reasons": [],
                "primary_flags": {},
                "text_attack_runtime_assessment": {},
            },
        }
        try:
            precollected_primary = arxiv_collector.collect_id(canonical_id)
            precollected_primary = _validated_precollected_arxiv_primary(
                precollected_primary,
                canonical_id=canonical_id,
            )
            precollected_primary = _annotate_freshness(
                precollected_primary,
                as_of=as_of_dt,
                cutoff=cutoff,
            )
            precheck_primary = _external_from_raw_item(
                precollected_primary,
                source_query=f"arxiv_id:{canonical_id}",
            )
        except Exception as exc:
            reason = _paper_discovery_error(exc)
            record["rejection_reasons"] = [reason]
            record["precheck"] = {
                "status": "fetch_failed",
                "rejection_reasons": [reason],
                "primary_flags": {},
                "text_attack_runtime_assessment": {},
            }
            candidate_records.append(record)
            continue

        (
            body_gate_assessments,
            text_attack_runtime_assessment,
            precheck_rejection_reasons,
            body_gate_deferred,
        ) = _precheck_primary(precheck_primary, canonical_id)
        primary_flags = {
            flag: precheck_primary.metadata.get(flag) is True
            for flag in metadata_required_primary_flags
        }
        primary_flags["source_authored_attack_method_claim"] = bool(
            precheck_primary.metadata.get(
                "source_authored_attack_method_claim"
            )
            is True
            or precheck_primary.metadata.get("advanced_mechanism_eligible")
            is True
        )
        primary_flags["advanced_mechanism_eligible"] = (
            precheck_primary.metadata.get("advanced_mechanism_eligible")
            is True
        )
        primary_flags["risk_domain_binding_eligible"] = (
            precheck_primary.metadata.get("risk_domain_binding_eligible")
            is True
        )
        primary_flags["text_attack_runtime_eligible"] = (
            text_attack_runtime_assessment.eligible
        )
        primary_domain_relevance_satisfied = bool(
            not required_body_relevance_gates
            or all(
                any(
                    assessment["eligible"] is True
                    and str(assessment["gate_id"]) == gate_id
                    for assessment in body_gate_assessments
                )
                for gate_id in required_body_relevance_gates
            )
        )
        mechanism_strength = (
            "advanced"
            if primary_flags["advanced_mechanism_eligible"]
            else "standard"
            if primary_flags["source_authored_attack_method_claim"]
            and primary_flags["mechanism_extraction_eligible"]
            and primary_flags["text_attack_runtime_eligible"]
            else "missing"
        )
        generation_eligible = not precheck_rejection_reasons
        text_attack_runtime_payload = {
            "eligible": text_attack_runtime_assessment.eligible,
            "reason": text_attack_runtime_assessment.reason,
            "evidence_terms": list(text_attack_runtime_assessment.evidence_terms),
        }
        effective_at = str(
            precheck_primary.metadata.get("source_effective_at", "")
        )
        try:
            effective_rank = datetime.fromisoformat(
                effective_at.replace("Z", "+00:00")
            ).timestamp()
        except (TypeError, ValueError):
            effective_rank = 0.0
        quality_score = float(
            precheck_primary.metadata.get("evidence_quality_score", 0.0) or 0.0
        )
        example_evidence = _assess_paper_example_evidence(
            _query_body_without_package_headers(precheck_primary.text)
        )
        record.update(
            {
                "paper_bundle_id": f"arxiv:{canonical_id}",
                "title": precheck_primary.title,
                "rejection_reasons": list(precheck_rejection_reasons),
                "body_relevance_assessments": body_gate_assessments,
                "body_relevance_gate_deferred_to_bundle": body_gate_deferred,
                "evidence_quality_score": quality_score,
                "example_evidence_status": str(example_evidence["status"]),
                "example_evidence_score": int(example_evidence["score"]),
                "example_evidence_signals": list(example_evidence["signals"]),
                "example_evidence_source": "primary_bounded_body",
                "example_source_truncated": bool(
                    precheck_primary.metadata.get("arxiv_source_bounded")
                    or precheck_primary.metadata.get(
                        "arxiv_pdf_extraction_truncated"
                    )
                ),
                "example_source_bounding_method": str(
                    precheck_primary.metadata.get(
                        "arxiv_source_bounding_method",
                        "role_balanced_example_grounded_v2",
                    )
                ),
                "source_effective_at": effective_at,
                "effective_rank": effective_rank,
                "score": round(quality_score, 3),
                "generation_eligible": generation_eligible,
                "mechanism_strength": mechanism_strength,
                "domain_evidence_status": (
                    "same_bundle"
                    if primary_domain_relevance_satisfied
                    and required_body_relevance_gates
                    else "deferred_to_promotion"
                    if required_body_relevance_gates
                    else "not_requested"
                ),
                "domain_binding_deferred": bool(
                    required_body_relevance_gates
                    and not primary_domain_relevance_satisfied
                ),
                "precheck": {
                    "status": (
                        "rejected" if precheck_rejection_reasons else "passed"
                    ),
                    "rejection_reasons": list(precheck_rejection_reasons),
                    "primary_flags": primary_flags,
                    "body_relevance_gate_deferred_to_bundle": (
                        body_gate_deferred
                    ),
                    "text_attack_runtime_assessment": (
                        text_attack_runtime_payload
                    ),
                },
            }
        )
        _expand_attack_method_citations(
            primary=precheck_primary,
            canonical_id=canonical_id,
            body_gate_assessments=body_gate_assessments,
            insert_after=paper_index,
        )
        domain_context_expansion = bool(
            required_body_relevance_gates
            and primary_domain_relevance_satisfied
            and precheck_primary.metadata.get("freshness_eligible") is True
            and precheck_primary.metadata.get("evidence_quality_eligible")
            is True
            and not generation_eligible
            and _BENCHMARK_PRIMARY_PATTERN.search(
                _query_body_without_package_headers(precheck_primary.text)
            )
        )
        if domain_context_expansion:
            record["companion_discovery_attempted"] = True
            record["citation_companion_discovery_attempted"] = True
            try:
                citation_bundle = collect_paper_evidence_bundle(
                    canonical_id,
                    include_github=include_github,
                    include_huggingface=include_huggingface,
                    google_companion_discovery=use_google_companions,
                    timeout=timeout,
                    user_agent=user_agent,
                    delay_seconds=delay_seconds,
                    max_source_age_days=max_source_age_days,
                    as_of=as_of_dt,
                    external_search_config=external_search_config,
                    _precollected_primary=precollected_primary,
                )
                (
                    citation_bundle,
                    citation_bundle_body_rows,
                    citation_bundle_body_satisfied,
                ) = _annotate_bundle_body_relevance(
                    citation_bundle,
                    canonical_id,
                )
                record["bundle_body_relevance_assessments"] = (
                    citation_bundle_body_rows
                )
                citation_companions = [
                    item
                    for item in citation_bundle
                    if str(item.metadata.get("paper_role", "")).casefold()
                    == "companion"
                    and item.metadata.get("paper_relation_verified") is True
                ]
                record["citation_companion_sources"] = sorted(
                    {
                        str(item.metadata.get("external_source", ""))
                        for item in citation_companions
                        if str(item.metadata.get("external_source", ""))
                    }
                )
                if citation_bundle_body_satisfied:
                    _expand_attack_method_citations(
                        primary=precheck_primary,
                        canonical_id=canonical_id,
                        body_gate_assessments=body_gate_assessments,
                        insert_after=paper_index,
                        related_items=citation_bundle,
                    )
            except Exception as exc:
                record["citation_companion_discovery_error"] = (
                    _paper_discovery_error(exc)
                )
        if precheck_rejection_reasons:
            candidate_records.append(record)
            continue

        record["companion_discovery_attempted"] = bool(
            include_github or include_huggingface
        )
        try:
            bundle = collect_paper_evidence_bundle(
                canonical_id,
                include_github=include_github,
                include_huggingface=include_huggingface,
                google_companion_discovery=use_google_companions,
                timeout=timeout,
                user_agent=user_agent,
                delay_seconds=delay_seconds,
                max_source_age_days=max_source_age_days,
                as_of=as_of_dt,
                external_search_config=external_search_config,
                _precollected_primary=precollected_primary,
            )
        except Exception as exc:
            record["rejection_reasons"] = [_paper_discovery_error(exc)]
            candidate_records.append(record)
            continue

        (
            bundle,
            bundle_body_relevance_assessments,
            bundle_body_relevance_satisfied,
        ) = _annotate_bundle_body_relevance(bundle, canonical_id)
        same_bundle_domain_satisfied = bundle_body_relevance_satisfied
        attached_domain_bridges: list[dict[str, Any]] = []
        if required_body_relevance_gates and not bundle_body_relevance_satisfied:
            bundle, attached_domain_bridges = _attach_citation_domain_bridges(
                bundle,
                canonical_id,
            )
            if attached_domain_bridges:
                (
                    bundle,
                    bundle_body_relevance_assessments,
                    bundle_body_relevance_satisfied,
                ) = _annotate_bundle_body_relevance(bundle, canonical_id)
        primary = next(
            (
                item
                for item in bundle
                if str(item.metadata.get("paper_role", "")).casefold()
                == "primary"
            ),
            None,
        )
        if primary is None:
            record["rejection_reasons"] = ["verified_arxiv_primary_required"]
            candidate_records.append(record)
            continue
        (
            example_evidence,
            example_evidence_source,
            example_source_truncated,
            example_source_bounding_method,
        ) = _best_verified_bundle_example_evidence(bundle)
        record.update(
            {
                "example_evidence_status": str(example_evidence["status"]),
                "example_evidence_score": int(example_evidence["score"]),
                "example_evidence_signals": list(example_evidence["signals"]),
                "example_evidence_source": example_evidence_source,
                "example_source_truncated": example_source_truncated,
                "example_source_bounding_method": example_source_bounding_method,
            }
        )
        rejection_reasons: list[str] = []
        if any(
            item.metadata.get("freshness_eligible") is not True
            or item.metadata.get("evidence_quality_eligible") is not True
            for item in bundle
        ):
            rejection_reasons.append("all_bundle_members_must_be_fresh_quality_evidence")

        companion_sources = sorted(
            {
                str(item.metadata.get("external_source", ""))
                for item in bundle
                if str(item.metadata.get("paper_role", "")).casefold()
                == "companion"
                and item.metadata.get("paper_relation_verified") is True
                and str(
                    item.metadata.get("paper_companion_usage", "")
                ).casefold()
                != "domain_evidence_only"
            }
        )
        google_confirmed_sources = sorted(
            {
                str(source)
                for source in list(
                    primary.metadata.get(
                        "paper_google_confirmed_companion_sources", []
                    )
                    or []
                )
                if str(source)
            }
        )
        direct_companion_sources = sorted(
            {
                str(item.metadata.get("external_source", ""))
                for item in bundle
                if str(item.metadata.get("paper_role", "")).casefold()
                == "companion"
                and item.metadata.get("paper_relation_verified") is True
                and str(
                    item.metadata.get("paper_companion_discovery_method", "")
                )
                == "paper_direct_link"
            }
        )
        eligible = not rejection_reasons
        domain_evidence_status = (
            "not_requested"
            if not required_body_relevance_gates
            else "same_bundle"
            if same_bundle_domain_satisfied
            else "citation_graph"
            if attached_domain_bridges and bundle_body_relevance_satisfied
            else "deferred_to_promotion"
        )
        domain_binding_deferred = bool(
            domain_evidence_status == "deferred_to_promotion"
        )
        selection_tier = (
            "same_bundle_bound"
            if domain_evidence_status == "same_bundle"
            else "citation_graph_bound"
            if domain_evidence_status == "citation_graph"
            else "mechanism_only"
        )
        selection_degradations: list[str] = []
        if mechanism_strength == "standard":
            selection_degradations.append(
                "advanced_mechanism_evidence_not_required"
            )
        if domain_binding_deferred:
            selection_degradations.append(
                "domain_binding_deferred_to_promotion"
            )
        if example_evidence["status"] == "partial":
            selection_degradations.append(
                "complete_paper_example_not_found_partial_artifact_only"
            )
        elif example_evidence["status"] == "none":
            selection_degradations.append(
                "concrete_paper_example_not_found"
            )
        tier_score = (
            1000.0
            if domain_evidence_status == "same_bundle"
            else 800.0
            if domain_evidence_status == "citation_graph"
            else 700.0
            if domain_evidence_status == "not_requested"
            else 500.0
        )
        score = (
            (tier_score if eligible else 0.0)
            + (
                200.0
                if example_evidence["status"] == "complete"
                else 75.0
                if example_evidence["status"] == "partial"
                else 0.0
            )
            + float(example_evidence["score"])
            + (50.0 if mechanism_strength == "advanced" else 0.0)
            + 100.0 * len(companion_sources)
            + 10.0 * len(direct_companion_sources)
            + quality_score
        )
        record.update(
            {
                "paper_bundle_id": str(
                    primary.metadata.get("paper_bundle_id", "")
                ),
                "title": primary.title,
                "status": "eligible" if eligible else "rejected",
                "rejection_reasons": rejection_reasons,
                "generation_eligible": eligible,
                "mechanism_strength": mechanism_strength,
                "domain_evidence_status": domain_evidence_status,
                "domain_binding_deferred": domain_binding_deferred,
                "selection_tier": selection_tier,
                "selection_degradations": selection_degradations,
                "companion_sources": companion_sources,
                "direct_companion_sources": direct_companion_sources,
                "google_confirmed_companion_sources": google_confirmed_sources,
                "google_companion_discovery": primary.metadata.get(
                    "paper_google_companion_discovery", {}
                ),
                "body_relevance_assessments": body_gate_assessments,
                "bundle_body_relevance_assessments": (
                    bundle_body_relevance_assessments
                ),
                "citation_domain_bridge_sources": sorted(
                    {
                        str(bridge.get("domain_evidence_source", ""))
                        for bridge in attached_domain_bridges
                        if str(bridge.get("domain_evidence_source", ""))
                    }
                ),
                "citation_domain_bridge_count": len(attached_domain_bridges),
                "evidence_quality_score": quality_score,
                "source_effective_at": effective_at,
                "effective_rank": effective_rank,
                "score": round(score, 3),
            }
        )
        record["selection_ready"] = _selection_ready(record)
        if eligible:
            verified_bundles[canonical_id] = bundle
        candidate_records.append(record)

    ranked_records = sorted(
        candidate_records,
        key=lambda record: (
            0
            if record.get("status") == "eligible"
            else 1,
            {
                "complete": 0,
                "partial": 1,
                "none": 2,
            }.get(str(record.get("example_evidence_status", "")), 3),
            {
                "same_bundle_bound": 0,
                "citation_graph_bound": 1,
                "mechanism_only": 2,
            }.get(str(record.get("selection_tier", "")), 99),
            -float(record.get("example_evidence_score", 0.0) or 0.0),
            0
            if record.get("mechanism_strength") == "advanced"
            else 1,
            0 if _selection_ready(record) else 1,
            -len(record.get("companion_sources", [])),
            -len(record.get("direct_companion_sources", [])),
            -float(record.get("evidence_quality_score", 0.0) or 0.0),
            -float(record.get("effective_rank", 0.0) or 0.0),
            str(record.get("arxiv_id", "")),
        ),
    )
    for rank, record in enumerate(ranked_records, start=1):
        record["component_score"] = float(record.get("score", 0.0) or 0.0)
        # Public ``score`` is monotonic with the actual lexicographic generation
        # order.  ``component_score`` retains the older additive diagnostic.
        record["score"] = float(len(ranked_records) - rank + 1)
        record["score_semantics"] = "generation_rank_monotonic_v1"
        record["generation_rank"] = rank
        record["rank"] = rank
        record.pop("effective_rank", None)

    selected_record = next(
        (record for record in ranked_records if record.get("status") == "eligible"),
        None,
    )
    if selected_record is not None:
        selected_record["selected"] = True
    for record in ranked_records:
        record.setdefault("selected", False)

    def _annotated_candidate_bundle(
        record: dict[str, Any],
    ) -> list[ExternalCollectedItem]:
        canonical_id = str(record.get("arxiv_id", ""))
        return [
            replace(
                item,
                metadata={
                    **item.metadata,
                    "paper_discovery_mode": "arxiv_first_batch",
                    "paper_discovery_candidate_count": len(candidate_records),
                    "paper_discovery_candidate_pool_count": len(
                        verification_queue_ids
                    ),
                    "paper_discovery_rank": int(record["rank"]),
                    "paper_discovery_score": float(record["score"]),
                    "paper_selection_tier": str(record.get("selection_tier", "")),
                    "selection_tier": str(record.get("selection_tier", "")),
                    "paper_selection_policy": PAPER_SELECTION_POLICY,
                    "paper_selection_schema_version": (
                        PAPER_SELECTION_SCHEMA_VERSION
                    ),
                    "selection_policy": PAPER_SELECTION_POLICY,
                    "selection_schema_version": PAPER_SELECTION_SCHEMA_VERSION,
                    "domain_evidence_status": str(
                        record.get("domain_evidence_status", "")
                    ),
                    "domain_binding_deferred": bool(
                        record.get("domain_binding_deferred", False)
                    ),
                    "selection_degradations": list(
                        record.get("selection_degradations", [])
                    ),
                    "mechanism_strength": str(
                        record.get("mechanism_strength", "")
                    ),
                    "example_evidence_status": str(
                        record.get("example_evidence_status", "none")
                    ),
                    "example_evidence_score": int(
                        record.get("example_evidence_score", 0) or 0
                    ),
                    "example_evidence_signals": list(
                        record.get("example_evidence_signals", []) or []
                    ),
                    "example_evidence_source": str(
                        record.get(
                            "example_evidence_source", "primary_bounded_body"
                        )
                    ),
                    "example_source_truncated": bool(
                        record.get("example_source_truncated", False)
                    ),
                    "example_source_bounding_method": str(
                        record.get(
                            "example_source_bounding_method",
                            "role_balanced_example_grounded_v2",
                        )
                    ),
                    "paper_selection_ready": bool(
                        record.get("selection_ready", False)
                    ),
                    "paper_selection_selected": bool(record.get("selected", False)),
                    "paper_selection_fallback_without_companion": bool(
                        companion_preferred and not record.get("selection_ready", False)
                    ),
                },
            )
            for item in verified_bundles.get(canonical_id, [])
        ]

    eligible_records = [
        record for record in ranked_records if record.get("status") == "eligible"
    ]
    annotated_candidate_bundles = {
        str(record["arxiv_id"]): _annotated_candidate_bundle(record)
        for record in eligible_records
    }
    if candidate_bundles_out is not None:
        candidate_bundles_out.update(annotated_candidate_bundles)
    selected_items = (
        list(annotated_candidate_bundles.get(str(selected_record["arxiv_id"]), []))
        if selected_record is not None
        else []
    )

    manifest = {
        "schema_version": PAPER_SELECTION_SCHEMA_VERSION,
        "collection_mode": "arxiv_first_batch",
        "selection_policy": PAPER_SELECTION_POLICY,
        "status": "selected" if selected_record is not None else "no_suitable_paper",
        "query_profile": query_profile,
        "requested_sources": selected_sources,
        "required_body_relevance_gates": required_body_relevance_gates,
        "required_primary_precheck_flags": list(required_primary_flags),
        "paper_discovery_limit": discovery_limit,
        "desired_generation_candidates": desired_generation_candidates,
        "available_generation_candidates": len(eligible_records),
        "paper_candidate_backfill_enabled": bool(enable_candidate_backfill),
        "generation_candidate_target_satisfied": bool(
            len(eligible_records) >= desired_generation_candidates
        ),
        "generation_candidate_shortfall": max(
            0, desired_generation_candidates - len(eligible_records)
        ),
        "preferred_example_evidence_status": "complete",
        "example_evidence_policy": "ranking_and_diagnostic",
        "selection_failure_reasons": (
            ["no_generation_eligible_verified_paper"]
            if selected_record is None
            else []
        ),
        "complete_example_candidate_count": sum(
            record.get("status") == "eligible"
            and record.get("example_evidence_status") == "complete"
            for record in ranked_records
        ),
        "partial_example_candidate_count": sum(
            record.get("status") == "eligible"
            and record.get("example_evidence_status") == "partial"
            for record in ranked_records
        ),
        "observed_partial_example_candidate_count": sum(
            record.get("example_evidence_status") == "partial"
            for record in ranked_records
        ),
        "none_example_candidate_count": sum(
            record.get("status") == "eligible"
            and record.get("example_evidence_status") == "none"
            for record in ranked_records
        ),
        "selection_preferences": [
            "prefer_complete_verified_bundle_rewrite_example",
            "domain_evidence_tier",
            "example_evidence_score",
            "mechanism_strength",
            "verified_companion_evidence",
        ],
        "as_of": as_of_dt.isoformat().replace("+00:00", "Z"),
        "freshness_cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "max_source_age_days": max_source_age_days,
        # Discovery, queueing, and verification are separate stages. Keep each
        # list explicit so a rejected initial hit and its successful replacement
        # are visible rather than looking like an unstable search result.
        "discovered_arxiv_ids": verification_queue_ids,
        "initial_verification_arxiv_ids": initial_verification_ids,
        "backfill_arxiv_ids": backfill_ids,
        "citation_expansion_arxiv_ids": citation_expansion_arxiv_ids,
        "citation_expansion_records": citation_expansion_records,
        "citation_domain_bridge_records": citation_domain_bridge_records,
        "verified_arxiv_ids": verified_arxiv_ids,
        "paper_precheck_summary": {
            "attempted": len(candidate_records),
            "passed": sum(
                record.get("precheck", {}).get("status") == "passed"
                for record in candidate_records
            ),
            "rejected": sum(
                record.get("precheck", {}).get("status") == "rejected"
                for record in candidate_records
            ),
            "fetch_failed": sum(
                record.get("precheck", {}).get("status") == "fetch_failed"
                for record in candidate_records
            ),
            "companion_discovery_attempts": sum(
                record.get("companion_discovery_attempted") is True
                for record in candidate_records
            ),
        },
        "paper_discovery_source_allocation": {
            "paper_candidate_backfill_enabled": bool(enable_candidate_backfill),
            "arxiv_discovered": len(arxiv_candidate_ids),
            "google_unique_discovered": len(google_candidate_ids),
            "google_reserved_slots": google_reserved_slots,
            "arxiv_selected_for_verification": arxiv_keep,
            "google_selected_for_verification": google_take,
            "candidate_pool_limit": candidate_pool_limit,
            "candidate_pool_size": len(verification_queue_ids),
            "domain_candidate_limit": domain_candidate_limit,
            "domain_lane_candidates": sum(
                any(
                    route.get("discovery_lane") == "domain"
                    for route in provenance_by_id.get(arxiv_id, [])
                )
                for arxiv_id in verification_queue_ids
            ),
            "mechanism_lane_candidates": sum(
                any(
                    route.get("discovery_lane") == "mechanism"
                    for route in provenance_by_id.get(arxiv_id, [])
                )
                for arxiv_id in verification_queue_ids
            ),
            "verification_queue_limit": verification_queue_limit,
            "citation_expansion_limit": (
                MAX_PAPER_CITATION_EXPANSION_CANDIDATES
            ),
            "citation_expansion_candidates": len(
                citation_expansion_arxiv_ids
            ),
            "initial_verification_target": discovery_limit,
            "initial_verification_size": len(initial_verification_ids),
            "verification_attempts": len(verified_arxiv_ids),
            "backfill_verification_attempts": sum(
                record.get("verification_phase") == "backfill"
                for record in candidate_records
            ),
            "citation_verification_attempts": sum(
                record.get("verification_phase") == "citation_expansion"
                for record in candidate_records
            ),
            "arxiv_artifact_queries_executed": arxiv_artifact_queries_executed,
            "google_artifact_queries_executed": google_artifact_queries_executed,
        },
        "arxiv_discovery_summary": {
            "status": "ok"
            if arxiv_discovered_ids
            else "error"
            if any(item.metadata.get("error") for item in arxiv_discovery_items)
            else "empty",
            "query_budget": arxiv_query_budget,
            "queries_executed": arxiv_queries_executed,
            "artifact_queries_executed": arxiv_artifact_queries_executed,
            "records": len(arxiv_discovery_items),
            "discovered_id_count": len(arxiv_discovered_ids),
            "errors": list(
                dict.fromkeys(
                    str(item.metadata.get("error", ""))
                    for item in arxiv_discovery_items
                    if str(item.metadata.get("error", ""))
                )
            ),
        },
        "google_arxiv_discovery": google_arxiv_discovery,
        "candidates": ranked_records,
        "selected_arxiv_id": (
            str(selected_record["arxiv_id"]) if selected_record else ""
        ),
        "selected_paper_bundle_id": (
            str(selected_record.get("paper_bundle_id", ""))
            if selected_record
            else ""
        ),
    }
    return selected_items, manifest


def summarize_external_collection(
    items: list[ExternalCollectedItem],
    *,
    sources: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build source-level diagnostics without treating an empty crawl as success."""
    selected_sources = normalize_external_sources(sources)
    by_source: dict[str, dict[str, Any]] = {}
    total_content = 0
    total_usable = 0
    total_mechanism_usable = 0
    total_errors = 0
    total_skipped = 0
    total_diagnostics = 0
    total_evidence_rejected = 0
    role_counts: dict[str, int] = {}
    skill_ready_groups: set[str] = set()

    for source in selected_sources:
        source_items = [
            item for item in items if item.metadata.get("external_source") == source
        ]
        content_items = [
            item
            for item in source_items
            if item.text.strip() and not item.metadata.get("diagnostic")
        ]
        usable_items = [
            item
            for item in content_items
            if item.metadata.get("freshness_eligible")
            and item.metadata.get("evidence_quality_eligible", True)
            and (source != "github" or item.metadata.get("github_evidence_eligible"))
        ]
        mechanism_usable_items = [
            item
            for item in usable_items
            if item.metadata.get("mechanism_extraction_eligible", True) is not False
        ]
        evidence_rejected_items = [
            item
            for item in content_items
            if item.metadata.get("evidence_quality_eligible") is False
        ]
        freshness_rejected_items = [
            item
            for item in content_items
            if not item.metadata.get("freshness_eligible")
        ]
        error_items = [item for item in source_items if item.metadata.get("error")]
        skipped_items = [
            item for item in source_items if item.metadata.get("skipped_reason")
        ]
        diagnostics = [item for item in source_items if item.metadata.get("diagnostic")]
        diagnostic_evidence_rejected = sum(
            int(item.metadata.get("quality_rejected", 0) or 0)
            + int(item.metadata.get("evidence_rejected", 0) or 0)
            for item in diagnostics
        )
        reported_evidence_rejected = (
            len(evidence_rejected_items) + diagnostic_evidence_rejected
        )
        quality_rejection_counts: dict[str, int] = {}
        for diagnostic in diagnostics:
            raw_counts = diagnostic.metadata.get("quality_rejection_counts", {})
            if not isinstance(raw_counts, dict):
                continue
            for reason, count in raw_counts.items():
                quality_rejection_counts[str(reason)] = (
                    quality_rejection_counts.get(str(reason), 0) + int(count or 0)
                )
        error_messages = list(
            dict.fromkeys(
                str(item.metadata.get("error", "")).strip()
                for item in error_items
                if str(item.metadata.get("error", "")).strip()
            )
        )

        if usable_items:
            source_status = "ok" if not error_items else "partial"
        elif error_items:
            source_status = "error"
        elif reported_evidence_rejected:
            source_status = "evidence_rejected"
        elif content_items:
            source_status = "freshness_rejected"
        elif skipped_items:
            source_status = "skipped"
        else:
            source_status = "empty"

        by_source[source] = {
            "status": source_status,
            "records": len(source_items),
            "content_items": len(content_items),
            "usable_items": len(usable_items),
            "mechanism_usable_items": len(mechanism_usable_items),
            "freshness_rejected": len(freshness_rejected_items),
            "evidence_rejected": reported_evidence_rejected,
            "quality_rejection_counts": dict(sorted(quality_rejection_counts.items())),
            "errors": len(error_items),
            "skipped": len(skipped_items),
            "diagnostic_records": len(diagnostics),
            "error_messages": error_messages,
        }
        total_content += len(content_items)
        total_usable += len(usable_items)
        total_mechanism_usable += len(mechanism_usable_items)
        total_errors += len(error_items)
        total_skipped += len(skipped_items)
        total_diagnostics += len(diagnostics)
        total_evidence_rejected += reported_evidence_rejected
        for item in mechanism_usable_items:
            group = str(item.metadata.get("source_group", "")).strip()
            if group:
                skill_ready_groups.add(group)
            raw_roles = item.metadata.get("evidence_roles", [])
            roles = raw_roles if isinstance(raw_roles, list) else [raw_roles]
            for role in roles:
                normalized_role = str(role).strip()
                if normalized_role:
                    role_counts[normalized_role] = role_counts.get(normalized_role, 0) + 1

    non_ok_sources = [
        source for source, summary in by_source.items() if summary["status"] != "ok"
    ]
    if total_usable == 0:
        if total_errors:
            status = "error"
        elif total_evidence_rejected:
            status = "evidence_rejected"
        elif any(
            summary["status"] == "freshness_rejected"
            for summary in by_source.values()
        ):
            status = "freshness_rejected"
        elif total_skipped:
            status = "skipped"
        else:
            status = "empty"
    elif non_ok_sources:
        status = "partial"
    else:
        status = "ok"

    github_diagnostics = [
        item.metadata
        for item in items
        if item.metadata.get("external_source") == "github"
        and item.metadata.get("diagnostic")
    ]
    github_token_configured = bool(
        os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    )
    return {
        "status": status,
        "external_sources": selected_sources,
        "records": len(items),
        "content_items": total_content,
        "usable_items": total_usable,
        "mechanism_usable_items": total_mechanism_usable,
        "errors": total_errors,
        "skipped": total_skipped,
        "diagnostic_records": total_diagnostics,
        "evidence_rejected": total_evidence_rejected,
        "skill_ready_groups": sorted(skill_ready_groups),
        "skill_ready_group_count": len(skill_ready_groups),
        "evidence_role_counts": dict(sorted(role_counts.items())),
        "non_ok_sources": non_ok_sources,
        "by_source": by_source,
        "github": {
            "authenticated": bool(
                github_token_configured
                or any(
                    bool(diagnostic.get("github_authenticated", False))
                    for diagnostic in github_diagnostics
                )
            ),
            "token_configured": github_token_configured,
            "authentication_observed": bool(github_diagnostics),
            "rejected_candidates": sum(
                int(diagnostic.get("metadata_rejected", 0))
                + int(diagnostic.get("readme_rejected", 0))
                + int(diagnostic.get("evidence_rejected", 0))
                for diagnostic in github_diagnostics
            ),
            "paper_linked_items": sum(
                int(diagnostic.get("paper_linked_items", 0))
                for diagnostic in github_diagnostics
            ),
        },
    }


def _with_query_metadata(
    raw_item: RawCollectedItem,
    family: str,
    source_query: str,
    query_rank: int,
    query_profile: str,
    *,
    executed_query: str = "",
    execution_rank: int | None = None,
    execution_wave: str = "",
) -> RawCollectedItem:
    metadata = dict(raw_item.metadata or {})
    metadata.update(
        {
            "query_family": family,
            "source_query": source_query,
            "query_rank": query_rank,
            "query_profile": query_profile,
            "executed_query": executed_query or source_query,
        }
    )
    if execution_rank is not None:
        metadata["query_execution_rank"] = execution_rank
    if execution_wave:
        metadata["query_execution_wave"] = execution_wave
    return replace(raw_item, metadata=metadata)


def _apply_query_body_relevance_gate(
    raw_item: RawCollectedItem,
    spec: QuerySpec,
) -> tuple[RawCollectedItem | None, str]:
    """Filter one result through its body-only gate before deduplication."""
    if not spec.body_relevance_gate:
        return raw_item, ""
    assessment = _assess_query_body_relevance(
        raw_item.text,
        spec.body_relevance_gate,
    )
    if not assessment.eligible:
        return None, assessment.reason
    metadata = dict(raw_item.metadata or {})
    metadata.update(
        {
            "query_body_gate_required": True,
            "query_body_gate_id": spec.body_relevance_gate,
            "query_body_relevance_eligible": True,
            "query_body_relevance_reason": assessment.reason,
            "query_body_relevance_terms": list(assessment.evidence_terms),
            "query_body_relevance_topic_term": assessment.topic_term,
            # A query admission check can only narrow the collector's existing
            # evidence decision. Missing or false evidence never becomes true.
            "risk_domain_binding_eligible": bool(
                metadata.get("risk_domain_binding_eligible", False)
            )
            and assessment.eligible,
        }
    )
    return replace(raw_item, metadata=metadata), ""


def _query_gate_diagnostic_raw_item(
    source: str,
    rejection_counts: dict[str, int],
) -> RawCollectedItem:
    """Build a body-free aggregate diagnostic for rejected query candidates."""
    total = sum(max(0, int(count)) for count in rejection_counts.values())
    return RawCollectedItem(
        text="",
        source=source,
        title=f"{source} query relevance diagnostics",
        metadata={
            "external_source": source,
            "collector": source,
            "diagnostic": True,
            "quality_rejected": total,
            "quality_rejection_counts": {
                str(reason): int(count)
                for reason, count in sorted(rejection_counts.items())
                if int(count) > 0
            },
            "query_body_gate_required": True,
            "query_body_relevance_eligible": False,
        },
    )


def _raw_item_key(raw_item: RawCollectedItem) -> str:
    if raw_item.url:
        parsed = urllib_parse.urlsplit(raw_item.url)
        canonical = urllib_parse.urlunsplit(
            (
                parsed.scheme.casefold(),
                parsed.netloc.casefold(),
                parsed.path.rstrip("/"),
                parsed.query,
                "",
            )
        )
        if canonical:
            return f"url:{canonical}"
    normalized = " ".join(raw_item.text.casefold().split())
    return "text:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _query_matches_text(query: str, text: str) -> bool:
    query_tokens = [
        token for token in re.findall(r"[a-z0-9]+", query.casefold()) if len(token) > 2
    ]
    lowered = text.casefold()
    return bool(query_tokens) and all(token in lowered for token in query_tokens)


def _external_from_raw_item(
    raw_item: RawCollectedItem,
    *,
    source_query: str,
) -> ExternalCollectedItem:
    metadata = dict(raw_item.metadata or {})
    metadata.setdefault("external_source", raw_item.source)
    metadata.setdefault("collector", raw_item.source)
    return ExternalCollectedItem(
        url=raw_item.url,
        text=raw_item.text,
        title=raw_item.title,
        source_query=source_query,
        fetched_at=raw_item.collected_at,
        metadata=metadata,
    )


def _skipped_raw_item(
    source: str, reason: str, *, is_error: bool = False
) -> RawCollectedItem:
    metadata: dict[str, Any] = {
        "external_source": source,
        "collector": source,
        "skipped_reason": reason,
    }
    if is_error:
        metadata["error"] = reason
    return RawCollectedItem(
        text="",
        source=source,
        title=f"{source} skipped",
        metadata=metadata,
    )


def _http_get_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
    headers: dict[str, str] | None = None,
    max_bytes: int = 1_000_000,
    same_origin_redirects_only: bool = False,
) -> str:
    return _http_request_text(
        url,
        params=params,
        timeout=timeout,
        headers=headers,
        max_bytes=max_bytes,
        same_origin_redirects_only=same_origin_redirects_only,
    )


def _http_get_bytes(
    url: str,
    *,
    timeout: int = 30,
    headers: dict[str, str] | None = None,
    max_bytes: int = 1_000_000,
    same_origin_redirects_only: bool = False,
) -> bytes:
    req = urllib_request.Request(url, headers=headers or {})
    redirect_handler = (
        _SameOriginOnlyRedirectHandler()
        if same_origin_redirects_only
        else _SameOriginAuthorizationRedirectHandler()
    )
    opener = urllib_request.build_opener(redirect_handler)
    with opener.open(req, timeout=timeout) as response:
        return response.read(max_bytes + 1)[:max_bytes]


def _http_get_json_page(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
    headers: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str]]:
    if params:
        separator = "&" if urllib_parse.urlparse(url).query else "?"
        url = url + separator + urllib_parse.urlencode(params)
    req = urllib_request.Request(url, headers=headers or {})
    opener = urllib_request.build_opener(_SameOriginAuthorizationRedirectHandler())
    with opener.open(req, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        raw = response.read(1_000_001)[:1_000_000]
        response_headers = {
            str(key).casefold(): str(value)
            for key, value in response.headers.items()
        }
    return json.loads(raw.decode(charset, errors="replace")), response_headers


def _http_next_link(value: str) -> str:
    for part in str(value).split(","):
        match = re.match(r'\s*<([^>]+)>\s*;\s*rel=["\']?next["\']?', part, re.I)
        if match:
            return match.group(1).strip()
    return ""


def _http_get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
    headers: dict[str, str] | None = None,
    same_origin_redirects_only: bool = False,
) -> Any:
    return json.loads(
        _http_get_text(
            url,
            params=params,
            timeout=timeout,
            headers=headers,
            same_origin_redirects_only=same_origin_redirects_only,
        )
    )


def _http_post_json(
    url: str,
    *,
    payload: dict[str, Any],
    timeout: int = 30,
    headers: dict[str, str] | None = None,
) -> Any:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib_request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    opener = urllib_request.build_opener(_SameOriginAuthorizationRedirectHandler())
    with opener.open(req, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        raw = response.read(1_000_000).decode(charset, errors="replace")
    return json.loads(raw)


def _http_request_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    timeout: int = 30,
    headers: dict[str, str] | None = None,
    max_bytes: int = 1_000_000,
    same_origin_redirects_only: bool = False,
) -> str:
    if params:
        separator = "&" if urllib_parse.urlparse(url).query else "?"
        url = url + separator + urllib_parse.urlencode(params)
    body = urllib_parse.urlencode(data).encode("utf-8") if data is not None else None
    req = urllib_request.Request(url, data=body, headers=headers or {})
    redirect_handler = (
        _SameOriginOnlyRedirectHandler()
        if same_origin_redirects_only
        else _SameOriginAuthorizationRedirectHandler()
    )
    opener = urllib_request.build_opener(redirect_handler)
    with opener.open(req, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        raw = response.read(max_bytes + 1)[:max_bytes]
        return raw.decode(charset, errors="replace")


def _xml_text(element: Any) -> str:
    return str(getattr(element, "text", "") or "").strip()
