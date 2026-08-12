"""CLI for generating a base rewrite skill from crawled external text."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.proxy_bypass  # noqa: F401,E402

from core.config_resolver import ConfigResolver
from core.external_skill_generator import (
    DEFAULT_MAX_CHARS_PER_ITEM,
    DEFAULT_MAX_ITEMS,
    DEFAULT_MAX_SOURCE_AGE_DAYS,
    DEFAULT_MECHANISM_DEDUP_THRESHOLD,
    DEFAULT_SKILL_SIM_THRESHOLD,
    DEFAULT_TEXT_DEDUP_THRESHOLD,
    generate_base_skill_from_external,
)
from core.embedding_client import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL
from core.utils import read_yaml


DEFAULT_RIGOROUS_CANDIDATE_SKILLS = 3
DEFAULT_RIGOROUS_QUALITY_PROBE_COUNT = 3


def generate_base_skill_from_external_rigorous(**kwargs):
    """Import the compatibility writer only for explicit rigorous runs."""
    from core.external_text_skill_writer import (
        generate_base_skill_from_external_text as legacy_writer,
    )

    return legacy_writer(**kwargs)


def _parse_pipeline_mode(value: str) -> str:
    """Normalize the retired mode spelling without exposing it in CLI help."""
    normalized = str(value).strip().casefold()
    if normalized == "fast":
        return "standard"
    if normalized in {"standard", "rigorous"}:
        return normalized
    raise argparse.ArgumentTypeError("choose either standard or rigorous")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a base rewrite skill from external text JSONL/JSON/CSV/TXT."
    )
    parser.add_argument(
        "--pipeline-mode",
        type=_parse_pipeline_mode,
        choices=("standard", "rigorous"),
        default="standard",
        help=(
            "standard (default) uses one author call and local validation; rigorous "
            "uses the legacy multi-judge pipeline."
        ),
    )
    parser.add_argument("--source", required=True, help="External text source file.")
    parser.add_argument(
        "--source-type",
        default="auto",
        choices=["auto", "txt", "md", "json", "jsonl", "csv"],
        help="Input parsing mode.",
    )
    parser.add_argument("--text-field", default="text", help="Text field for JSON/JSONL/CSV records.")
    parser.add_argument("--skill-name", default="", help="Optional exact skill name, e.g. rewrite-external-v1.")
    parser.add_argument(
        "--target-domain",
        default="",
        help=(
            "Optional open-vocabulary harmful-capability or policy-risk domain. Standard "
            "mode may leave a general transformation unbound; rigorous mode requires "
            "source-backed domain evidence."
        ),
    )
    parser.add_argument(
        "--max-items", "--max-chunks", dest="max_items", type=int, default=DEFAULT_MAX_ITEMS,
        help="Maximum source-aware chunks retained after embedding deduplication.",
    )
    parser.add_argument(
        "--max-chars-per-item",
        type=int,
        default=DEFAULT_MAX_CHARS_PER_ITEM,
        help="Max retained characters per external text item.",
    )
    parser.add_argument(
        "--dedup-threshold", "--embedding-dedup-threshold",
        dest="dedup_threshold",
        type=float,
        default=None,
        help="Configured embedding-model cosine threshold for external chunks.",
    )
    parser.add_argument(
        "--mechanism-dedup-threshold", type=float, default=None
    )
    parser.add_argument(
        "--skill-sim-threshold",
        type=float,
        default=None,
        help="Embedding threshold for generated skill vs existing skills.",
    )
    duplicate_group = parser.add_mutually_exclusive_group()
    duplicate_group.add_argument(
        "--ignore-existing-skill-duplicates",
        dest="ignore_existing_skill_duplicates",
        action="store_true",
        help=(
            "Do not reject a candidate because a similar repository skill exists "
            "(default). Standard mode always skips this semantic check."
        ),
    )
    duplicate_group.add_argument(
        "--enforce-existing-skill-duplicates",
        dest="ignore_existing_skill_duplicates",
        action="store_false",
        help="Opt in to repository-level semantic duplicate rejection in rigorous mode.",
    )
    parser.set_defaults(ignore_existing_skill_duplicates=True)
    parser.add_argument("--candidate-skills", type=int, default=None)
    parser.add_argument(
        "--quality-probes",
        type=int,
        default=None,
        help="Defaults to 0 in standard mode and 3 in rigorous mode.",
    )
    parser.add_argument("--report-out", default="")
    parser.add_argument(
        "--max-source-age-days",
        type=int,
        default=DEFAULT_MAX_SOURCE_AGE_DAYS,
        help=(
            "Reject source evidence older than this rolling window "
            f"(default: {DEFAULT_MAX_SOURCE_AGE_DAYS} days / 3 years)."
        ),
    )
    parser.add_argument("--as-of", default="", help="UTC ISO date for reproducible freshness checks.")
    parser.add_argument("--embedding-model", default="", help=f"Defaults to {DEFAULT_EMBEDDING_MODEL}.")
    parser.add_argument("--embedding-dimensions", type=int, default=0, help=f"Defaults to {DEFAULT_EMBEDDING_DIMENSIONS}.")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-cache", default="")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help=(
            "Path to config.yaml with meta_skills.llm, optional author_llm/judge_llm, "
            "and skills.llm settings."
        ),
    )
    parser.add_argument("--workflow", default="basic", help="Workflow YAML basename to update.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Replace requested skill if it exists.")
    args = parser.parse_args()
    if args.candidate_skills is None:
        args.candidate_skills = (
            1
            if args.pipeline_mode == "standard"
            else DEFAULT_RIGOROUS_CANDIDATE_SKILLS
        )
    if args.quality_probes is None:
        args.quality_probes = (
            0
            if args.pipeline_mode == "standard"
            else DEFAULT_RIGOROUS_QUALITY_PROBE_COUNT
        )
    if args.candidate_skills <= 0:
        parser.error("--candidate-skills must be positive")
    if args.quality_probes < 0 or (
        args.pipeline_mode == "rigorous" and args.quality_probes == 0
    ):
        parser.error(
            "--quality-probes must be non-negative and positive in rigorous mode"
        )
    if args.pipeline_mode == "standard":
        if args.candidate_skills != 1:
            parser.error(
                "standard mode authors exactly one skill; use --pipeline-mode rigorous "
                "for multiple candidates"
            )
        if args.quality_probes != 0:
            parser.error(
                "standard mode does not run runtime probes; use --pipeline-mode rigorous"
            )
        if not args.ignore_existing_skill_duplicates:
            parser.error(
                "standard mode skips semantic duplicate checks; use --pipeline-mode rigorous"
            )
    return args


def main() -> int:
    args = parse_args()
    config = ConfigResolver.normalize(read_yaml(Path(args.config)))
    resolver = ConfigResolver(config)
    backend_config = resolver.resolve_meta_backend()
    author_backend_config = resolver.resolve_meta_author_backend()
    judge_backend_config = (
        resolver.resolve_meta_judge_backend()
        if args.pipeline_mode == "rigorous"
        else {}
    )
    skill_backend_config = (
        resolver.resolve_skill_backend()
        if args.pipeline_mode == "rigorous"
        else {}
    )
    embedding_config = (
        resolver.resolve_embedding_backend()
        if args.pipeline_mode == "rigorous"
        else {}
    )
    if args.embedding_model:
        embedding_config["model"] = str(args.embedding_model)
    if args.embedding_dimensions:
        embedding_config["dimensions"] = int(args.embedding_dimensions)
    if args.embedding_base_url:
        embedding_config["base_url"] = str(args.embedding_base_url)
    if args.embedding_cache:
        embedding_config["cache_path"] = str(args.embedding_cache)

    writer = (
        generate_base_skill_from_external
        if args.pipeline_mode == "standard"
        else generate_base_skill_from_external_rigorous
    )
    summary = writer(
        project_root=PROJECT_ROOT,
        source_path=Path(args.source),
        backend_config=backend_config,
        author_backend_config=author_backend_config,
        judge_backend_config=judge_backend_config,
        skill_backend_config=skill_backend_config,
        embedding_config=embedding_config,
        skill_name=str(args.skill_name),
        target_domain=str(args.target_domain),
        source_type=str(args.source_type),
        text_field=str(args.text_field),
        max_items=int(args.max_items),
        max_chars_per_item=int(args.max_chars_per_item),
        text_dedup_threshold=float(
            args.dedup_threshold
            if args.dedup_threshold is not None
            else embedding_config.get("text_dedup_threshold", DEFAULT_TEXT_DEDUP_THRESHOLD)
        ),
        mechanism_dedup_threshold=float(
            args.mechanism_dedup_threshold
            if args.mechanism_dedup_threshold is not None
            else embedding_config.get("mechanism_dedup_threshold", DEFAULT_MECHANISM_DEDUP_THRESHOLD)
        ),
        skill_sim_threshold=float(
            args.skill_sim_threshold
            if args.skill_sim_threshold is not None
            else embedding_config.get("skill_duplicate_threshold", DEFAULT_SKILL_SIM_THRESHOLD)
        ),
        ignore_existing_skill_duplicates=bool(
            args.ignore_existing_skill_duplicates
        ),
        candidate_skills=int(args.candidate_skills),
        quality_probe_count=int(args.quality_probes),
        report_out=Path(args.report_out) if args.report_out else None,
        overwrite=bool(args.overwrite),
        workflow_name=str(args.workflow),
        max_source_age_days=int(args.max_source_age_days),
        as_of=str(args.as_of) or None,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0 if summary.status == "generated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
