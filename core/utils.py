"""Shared helpers for file IO, timestamps, and small utility functions."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


_DOTENV_LOADED: set[Path] = set()


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path



def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace a JSON file so an interrupted write keeps the old copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file, loading sibling/parent .env files and expanding env vars."""
    _load_dotenv_for(path)
    return yaml.safe_load(os.path.expandvars(path.read_text(encoding="utf-8"))) or {}


def _load_dotenv_for(path: Path) -> None:
    """Load the nearest .env file without overriding existing environment values."""
    for parent in (path.resolve().parent, *path.resolve().parents):
        env_path = parent / ".env"
        if env_path.exists():
            _load_dotenv(env_path)
            return


def _load_dotenv(path: Path) -> None:
    resolved = path.resolve()
    if resolved in _DOTENV_LOADED:
        return
    _DOTENV_LOADED.add(resolved)

    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def read_markdown_frontmatter(path: Path) -> dict[str, Any]:
    """Read YAML frontmatter from a markdown file if present."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}

    lines = text.splitlines()
    if not lines:
        return {}

    try:
        closing_index = lines[1:].index("---") + 1
    except ValueError:
        return {}

    payload = "\n".join(lines[1:closing_index])
    return yaml.safe_load(payload) or {}



def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str = "run") -> str:
    """Create a unique run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def shorten(text: str, limit: int = 80) -> str:
    """Shorten text for logs."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def read_seed_prompt(path: Path, index: int) -> tuple[str, list[str]]:
    """Read one prompt and its risk types from a JSONL dataset by row index."""
    from core.constants import RISK_CATEGORY_MAP

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                record = json.loads(line)
                query = str(record.get("query", "")).strip()
                risk_code = record.get("risk_category", "")
                risk_name = RISK_CATEGORY_MAP.get(risk_code, "")
                if risk_name:
                    seed_risk_types = [risk_name]
                else:
                    semantic_cat = record.get("SemanticCategory", "")
                    seed_risk_types = [semantic_cat] if semantic_cat else ["unclassified"]
                return query, seed_risk_types
    raise IndexError(f"Index {index} out of range for {path}")


def extract_skill_scores(run_dir: str | Path) -> dict[str, int]:
    """Extract per-skill max judge scores from compact_trace.json.
    Step 0 (direct query without skill) is recorded under 'direct_query'."""
    trace_path = Path(run_dir) / "compact_trace.json"
    if not trace_path.exists():
        return {}
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    skill_scores: dict[str, int] = {}
    for step in trace.get("steps", []):
        executed_skills = step.get("executed_skills", [])
        skill_name = executed_skills[0] if executed_skills else "direct_query"
        candidate_results = step.get("output", {}).get("candidate_results", [])
        for cr in candidate_results:
            if cr.get("not_scored") or cr.get("execution_error"):
                continue
            score = int(cr.get("judge_score", 1))
            skill_scores[skill_name] = max(skill_scores.get(skill_name, 0), score)
    return skill_scores


def load_completed_indices(progress_file: Path) -> set[int]:
    """Load successfully processed indices from a progress JSONL file.

    Error rows are diagnostic checkpoints, not completed work. Skipping them
    lets a resumed run retry transient failures such as expired API tokens.
    """
    if not progress_file.exists():
        return set()
    completed: set[int] = set()
    with progress_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if entry.get("error"):
                    continue
                completed.add(int(entry["index"]))
    return completed


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    """Append one JSON line to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_skill_script(skill_name: str, project_root: Path, exp_name: str = "default") -> str:
    """Read a skill's run.py source from disk."""
    for base_dir in ["skills", f"skills/new_skills_{exp_name}"]:
        script_path = project_root / base_dir / skill_name / "scripts" / "run.py"
        if script_path.exists():
            return script_path.read_text(encoding="utf-8")
    return ""
