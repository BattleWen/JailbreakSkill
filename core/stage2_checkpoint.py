"""Durable per-prompt checkpoints for Stage 2 category evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from core.utils import append_jsonl, utc_now_iso


class Stage2PromptCheckpoint:
    """Append human-readable Stage 2 events and restore completed evaluations."""

    SCHEMA_VERSION = 2
    _PREVIEW_LIMIT = 200

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._completed: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self._evolved_skills_by_category: dict[str, list[str]] = {}
        self._load()

    @staticmethod
    def prompt_key(entry: dict[str, Any]) -> str:
        """Build a stable identity that detects both index and prompt changes."""
        identity = {
            "index": entry.get("index"),
            "seed_prompt_sha256": hashlib.sha256(
                str(entry.get("seed_prompt", "")).encode("utf-8")
            ).hexdigest(),
        }
        encoded = json.dumps(
            identity,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def skill_fingerprint(*, root_dir: str, entry: str, skill_name: str) -> str:
        """Hash the executed script so edited skills do not reuse stale scores."""
        entry_path = Path(root_dir) / entry
        try:
            payload = entry_path.read_bytes()
        except OSError:
            payload = f"{skill_name}\0{entry}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get_completed(
        self,
        *,
        category: str,
        skill_name: str,
        skill_fingerprint: str,
        prompt_key: str,
    ) -> dict[str, Any] | None:
        """Return a defensive copy of a completed result, if one exists."""
        record = self._completed.get(
            (category, skill_name, skill_fingerprint, prompt_key)
        )
        return copy.deepcopy(record) if record is not None else None

    def get_evolved_skills(self, category: str) -> list[str]:
        """Return skills already generated against ``category``, in order."""
        return list(self._evolved_skills_by_category.get(str(category), []))

    @classmethod
    def _preview(cls, value: object, *, limit: int | None = None) -> str:
        """Return a compact, single-line preview suitable for a JSONL event."""
        text = " ".join(str(value or "").split())
        resolved_limit = cls._PREVIEW_LIMIT if limit is None else max(1, int(limit))
        if len(text) <= resolved_limit:
            return text
        return f"{text[:resolved_limit - 1]}…"

    def append_evolved_skill(
        self,
        *,
        category: str,
        skill_name: str,
        run_id: str,
    ) -> None:
        """Persist one consumed category-evolution attempt.

        The record is written immediately after a skill is generated so a
        restart cannot reset ``max_evolve_skills`` for that category. Duplicate
        names are ignored because the in-process loop does not count them twice.
        """
        category = str(category)
        skill_name = str(skill_name)
        if not category or not skill_name:
            raise ValueError(
                "Stage 2 evolved-skill records require category and skill_name"
            )

        evolved_skills = self._evolved_skills_by_category.setdefault(category, [])
        if skill_name in evolved_skills:
            return

        record = {
            "schema_version": self.SCHEMA_VERSION,
            "record_type": "evolved_skill",
            "timestamp": utc_now_iso(),
            "category": category,
            "evolved_skill_number": len(evolved_skills) + 1,
            "skill_name": skill_name,
            "run_id": run_id,
        }
        append_jsonl(self.path, record)
        evolved_skills.append(skill_name)

    def append_prompt_evaluation(
        self,
        *,
        category: str,
        skill_name: str,
        skill_fingerprint: str,
        prompt_key: str,
        dataset_index: object,
        prompt: str,
        status: str,
        score: int,
        candidate_text: str | None,
        response_text: str | None,
        run_id: str,
        generated_run_dir: str,
        judge_reason: str = "",
        error_type: str = "",
        error_message: str = "",
        attempts: int | None = None,
    ) -> None:
        """Append one concise, self-contained prompt-evaluation event.

        Completed records retain only the text and score needed to resume Stage 2.
        Retryable failures retain short diagnostics but are never restored as
        completed work. Detailed evaluator metadata belongs in ``compact_trace``.
        """
        if status not in {"completed", "retryable_error"}:
            raise ValueError(f"Unsupported Stage 2 checkpoint status: {status}")
        category = str(category)
        skill_name = str(skill_name)
        skill_fingerprint = str(skill_fingerprint)
        prompt_key = str(prompt_key)
        if not all((category, skill_name, skill_fingerprint, prompt_key)):
            raise ValueError(
                "Stage 2 prompt records require category, skill, and checkpoint keys"
            )

        resolved_score = int(score)
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "record_type": "prompt_evaluation",
            "timestamp": utc_now_iso(),
            "status": status,
            "category": category,
            "skill_name": skill_name,
            "dataset_index": dataset_index,
            "prompt_preview": self._preview(prompt),
            "score": resolved_score,
            "success": status == "completed" and resolved_score == 5,
        }
        if judge_reason:
            record["judge_reason"] = self._preview(judge_reason)

        if status == "completed":
            # ``None`` distinguishes a skill that produced no candidate from a
            # valid empty string. These are the only large fields retained.
            record["candidate_text"] = (
                None if candidate_text is None else str(candidate_text)
            )
            record["response_text"] = (
                None if response_text is None else str(response_text)
            )
        else:
            record["retryable"] = True
            if candidate_text is not None:
                record["candidate_preview"] = self._preview(candidate_text)
            if response_text is not None:
                record["response_preview"] = self._preview(response_text)
            if error_type:
                record["error_type"] = str(error_type)
            if error_message:
                record["error_message"] = self._preview(error_message)
            if attempts is not None:
                record["attempts"] = max(0, int(attempts))

        # Operational provenance follows the human-readable result fields;
        # opaque checkpoint identities stay last so a raw line is easy to scan.
        record.update({
            "run_id": run_id,
            "generated_run_dir": str(generated_run_dir),
            "skill_fingerprint": skill_fingerprint,
            "prompt_key": prompt_key,
        })
        append_jsonl(self.path, record)
        if status == "completed":
            self._completed[
                (category, skill_name, skill_fingerprint, prompt_key)
            ] = copy.deepcopy(record)

    @staticmethod
    def _normalize_completed_record(record: dict[str, Any]) -> dict[str, Any]:
        """Flatten a v1 or v2 completed record into the readable v2 shape."""
        normalized = {
            "schema_version": int(record.get("schema_version", 1) or 1),
            "record_type": "prompt_evaluation",
            "timestamp": str(record.get("timestamp", "")),
            "status": "completed",
            "category": str(record["category"]),
            "skill_name": str(record["skill_name"]),
            "dataset_index": record.get("dataset_index"),
            "prompt_preview": str(record.get("prompt_preview", "")),
            "run_id": str(record.get("run_id", "")),
            "generated_run_dir": str(record.get("generated_run_dir", "")),
            "skill_fingerprint": str(record["skill_fingerprint"]),
            "prompt_key": str(record["prompt_key"]),
        }

        legacy_result = record.get("result")
        if isinstance(legacy_result, dict):
            candidate = legacy_result.get("candidate")
            outcome = legacy_result.get("outcome")
            candidate_text = (
                candidate.get("text") if isinstance(candidate, dict) else None
            )
            response_text = (
                outcome.get("response_text") if isinstance(outcome, dict) else None
            )
            score = int(legacy_result.get("score", 1))
            if not normalized["prompt_preview"] and isinstance(candidate, dict):
                normalized["prompt_preview"] = str(
                    candidate.get("acceptance_prompt", "")
                )

            eval_payload = legacy_result.get("eval_payload")
            if isinstance(eval_payload, dict):
                metadata = eval_payload.get("metadata")
                bundles = (
                    metadata.get("score_bundles", [])
                    if isinstance(metadata, dict)
                    else []
                )
                if bundles and isinstance(bundles[0], dict):
                    reason = bundles[0].get("reason")
                    if reason:
                        normalized["judge_reason"] = str(reason)
        else:
            candidate_text = record.get("candidate_text")
            response_text = record.get("response_text")
            score = int(record.get("score", 1))
            if record.get("judge_reason"):
                normalized["judge_reason"] = str(record["judge_reason"])

        normalized.update({
            "score": score,
            "success": bool(record.get("success", score == 5)),
            "candidate_text": (
                None if candidate_text is None else str(candidate_text)
            ),
            "response_text": (
                None if response_text is None else str(response_text)
            ),
        })
        return normalized

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            checkpoint_file = self.path.open("r", encoding="utf-8")
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "Could not load Stage 2 checkpoint %s: %s", self.path, exc
            )
            return

        with checkpoint_file:
            for line_number, line in enumerate(checkpoint_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError("checkpoint record is not an object")
                    record_type = record.get("record_type")
                    if record_type == "evolved_skill":
                        category = str(record["category"])
                        skill_name = str(record["skill_name"])
                        if not category or not skill_name:
                            raise ValueError("empty category or skill_name")
                        evolved_skills = self._evolved_skills_by_category.setdefault(
                            category, []
                        )
                        if skill_name not in evolved_skills:
                            evolved_skills.append(skill_name)
                        continue

                    # Ignore obsolete fixed prior-skill selections. Historical
                    # skills are rediscovered and reused on every category.
                    if record_type == "prior_skill_selection":
                        continue
                    if record_type != "prompt_evaluation":
                        raise ValueError(f"unsupported record type: {record_type}")
                    if record.get("status") != "completed":
                        continue

                    normalized = self._normalize_completed_record(record)
                    key = (
                        normalized["category"],
                        normalized["skill_name"],
                        normalized["skill_fingerprint"],
                        normalized["prompt_key"],
                    )
                    self._completed[key] = normalized
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    logging.getLogger(__name__).warning(
                        "Ignoring malformed Stage 2 checkpoint line %d in %s: %s",
                        line_number,
                        self.path,
                        exc,
                    )
