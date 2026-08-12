"""CLI for crawling public external text into a reproducible JSONL snapshot."""

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
from core.external_text_collector import (
    EXTERNAL_DEFAULT_SOURCES,
    DEFAULT_EXTERNAL_QUERY_PROFILE,
    DEFAULT_MAX_SOURCE_AGE_DAYS,
    DEFAULT_USER_AGENT,
    MAX_PAPER_DISCOVERY_LIMIT,
    collect_ranked_paper_evidence_batch,
    collect_paper_evidence_bundle,
    default_snapshot_path,
    normalize_external_sources,
    parse_source_query_overrides,
    summarize_external_collection,
    write_crawl_snapshot,
)
from core.utils import read_yaml, write_json


_GENERIC_DISCOVERY_OPTIONS = (
    "--external-source",
    "--query-profile",
    "--query",
    "--source-query",
    "--paper-discovery-limit",
    "--paper-manifest-out",
)


def _option_was_supplied(argv: list[str], option: str) -> bool:
    return any(token == option or token.startswith(option + "=") for token in argv)


def _validate_source_mode(args: argparse.Namespace, argv: list[str]) -> None:
    paper_mode = bool(str(args.arxiv_id).strip())
    companion_requested = bool(
        str(args.github_repo).strip() or str(args.huggingface_dataset).strip()
    )
    if not args.paper_companions and not paper_mode:
        raise ValueError("--no-paper-companions requires --arxiv-id")
    if not args.paper_companions and companion_requested:
        raise ValueError(
            "--no-paper-companions cannot be combined with --github-repo or "
            "--huggingface-dataset"
        )
    if companion_requested and not paper_mode:
        raise ValueError("--github-repo and --huggingface-dataset require --arxiv-id")
    conflicting = [
        option
        for option in _GENERIC_DISCOVERY_OPTIONS
        if _option_was_supplied(argv, option)
    ]
    if paper_mode and conflicting:
        raise ValueError(
            "--arxiv-id exact-paper mode cannot be combined with generic "
            "discovery options: " + ", ".join(conflicting)
        )
    candidate_backfill_option_supplied = bool(
        _option_was_supplied(argv, "--paper-candidate-backfill")
        or _option_was_supplied(argv, "--no-paper-candidate-backfill")
    )
    if paper_mode and candidate_backfill_option_supplied:
        raise ValueError(
            "--paper-candidate-backfill/--no-paper-candidate-backfill applies "
            "only to automatic paper discovery"
        )
    if args.paper_discovery_limit <= 0:
        raise ValueError("--paper-discovery-limit must be positive")
    if args.paper_discovery_limit > MAX_PAPER_DISCOVERY_LIMIT:
        raise ValueError(
            f"--paper-discovery-limit must not exceed {MAX_PAPER_DISCOVERY_LIMIT}"
        )
    if not paper_mode:
        selected_sources = normalize_external_sources(args.external_source)
        if "arxiv" not in selected_sources:
            raise ValueError("automatic paper discovery requires --external-source arxiv")
        overrides = parse_source_query_overrides(args.source_query)
        unsupported = sorted(
            source
            for source, queries in overrides.items()
            if source not in {"arxiv", "google"} and queries
        )
        if unsupported:
            raise ValueError(
                "automatic paper discovery accepts --source-query only for "
                "arxiv and google: " + ", ".join(unsupported)
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=(
            "Discover and rank arXiv papers, then collect one verified paper "
            "bundle from four external sources into JSONL."
        ),
        epilog=(
            "Optional source credentials are read only from the environment: "
            "GITHUB_TOKEN or GH_TOKEN for GitHub, and HF_TOKEN or "
            "HUGGINGFACE_TOKEN for Hugging Face, BOCHA_API_KEY for Bocha, and "
            "SERPAPI_API_KEY or SERPAPI_KEY for SerpAPI Google results. The "
            "google source tries Bocha first when configured, then SerpAPI, "
            "then best-effort public DuckDuckGo "
            "discovery; that fallback may be "
            "challenged or time out. Google links are used only for discovery "
            "and cross-confirmation, never as paper mechanism evidence. "
            "Hugging Face uses Dataset Viewer first and a pinned Hub raw-data "
            "fallback when needed. arXiv pages use public access. "
            "Token values are never written to the crawl snapshot."
        ),
    )
    parser.add_argument(
        "--arxiv-id",
        default="",
        help=(
            "Exact arXiv identifier for single-paper evidence mode. The exact "
            "paper is primary evidence; only relation-verified GitHub/Hugging "
            "Face companions are admitted."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="YAML config containing optional external_search.bocha settings.",
    )
    parser.add_argument(
        "--github-repo",
        default="",
        help=(
            "Optional related GitHub repository (owner/repo or URL). Empty/auto "
            "discovers a canonical repository linked directly by the paper."
        ),
    )
    parser.add_argument(
        "--huggingface-dataset",
        default="",
        help=(
            "Optional related Hugging Face dataset (owner/name or URL). Empty/auto "
            "discovers a canonical dataset linked directly by the paper."
        ),
    )
    parser.add_argument(
        "--no-paper-companions",
        dest="paper_companions",
        action="store_false",
        default=True,
        help=(
            "In exact-paper mode, collect only the arXiv paper and skip all "
            "GitHub/Hugging Face companions, including links found in the paper."
        ),
    )
    parser.add_argument(
        "--external-source",
        action="append",
        default=[],
        help=(
            "external source to collect from. May be repeated or comma-separated. "
            "Supported: "
            + ", ".join(EXTERNAL_DEFAULT_SOURCES)
            + ", all. Defaults to "
            + ",".join(EXTERNAL_DEFAULT_SOURCES)
            + "."
        ),
    )
    parser.add_argument(
        "--query-profile",
        default=DEFAULT_EXTERNAL_QUERY_PROFILE,
        help="Source-aware external-source query profile.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Additional common query; repeatable.",
    )
    parser.add_argument(
        "--source-query",
        action="append",
        default=[],
        help="Additional native query as SOURCE=QUERY; repeatable.",
    )
    parser.add_argument("--out", default="", help="Output JSONL path.")
    parser.add_argument(
        "--paper-discovery-limit",
        type=int,
        default=10,
        help=(
            "Maximum number of canonical arXiv IDs to discover and verify "
            f"(at most {MAX_PAPER_DISCOVERY_LIMIT})."
        ),
    )
    candidate_backfill_group = parser.add_mutually_exclusive_group()
    candidate_backfill_group.add_argument(
        "--paper-candidate-backfill",
        dest="paper_candidate_backfill",
        action="store_true",
        default=True,
        help=(
            "Allow automatic discovery to verify replacement papers from a "
            "bounded candidate pool when the initial verification batch is "
            "insufficient (default)."
        ),
    )
    candidate_backfill_group.add_argument(
        "--no-paper-candidate-backfill",
        dest="paper_candidate_backfill",
        action="store_false",
        help=(
            "Verify only the initial paper batch; do not use later arXiv/Google "
            "candidate IDs as replacements."
        ),
    )
    parser.add_argument(
        "--paper-manifest-out",
        default="",
        help=(
            "JSON manifest for all discovered paper IDs, verification results, "
            "scores, and the selected bundle. Defaults next to --out."
        ),
    )
    parser.add_argument(
        "--per-query-limit", type=int, default=3, help="Maximum items per query."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help=(
            "network timeout ceiling in seconds; individual collector operations "
            "may cap or share this budget"
        ),
    )
    parser.add_argument(
        "--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent."
    )
    parser.add_argument(
        "--delay-seconds", type=float, default=0.0, help="Delay between source fetches."
    )
    parser.add_argument(
        "--max-source-age-days",
        type=int,
        default=DEFAULT_MAX_SOURCE_AGE_DAYS,
        help=(
            "Reject source evidence older than this rolling window "
            f"(default: {DEFAULT_MAX_SOURCE_AGE_DAYS} days / 3 years)."
        ),
    )
    parser.add_argument("--as-of", default="")
    parser.add_argument(
        "--no-google-companion-discovery",
        dest="google_companion_discovery",
        action="store_false",
        default=True,
        help="Disable Bocha/SerpAPI/DuckDuckGo discovery of paper companions.",
    )
    args = parser.parse_args(raw_argv)
    try:
        _validate_source_mode(args, raw_argv)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    args = parse_args()
    config = ConfigResolver.normalize(read_yaml(Path(args.config)))
    external_search_config = ConfigResolver(config).resolve_external_search()
    out_path = Path(args.out) if args.out else default_snapshot_path(PROJECT_ROOT)

    if args.arxiv_id:
        items = collect_paper_evidence_bundle(
            str(args.arxiv_id),
            github_repo=str(args.github_repo) or None,
            huggingface_dataset=str(args.huggingface_dataset) or None,
            include_github=bool(args.paper_companions),
            include_huggingface=bool(args.paper_companions),
            google_companion_discovery=bool(args.google_companion_discovery),
            timeout=int(args.timeout),
            user_agent=str(args.user_agent),
            delay_seconds=float(args.delay_seconds),
            max_source_age_days=int(args.max_source_age_days),
            as_of=str(args.as_of) or None,
            external_search_config=external_search_config,
        )
        external_sources = list(
            dict.fromkeys(
                str(item.metadata.get("external_source", ""))
                for item in items
                if str(item.metadata.get("external_source", ""))
            )
        )
        collection_mode = "exact_arxiv_paper_bundle"
        query_profile = ""
        paper_manifest: dict[str, object] | None = None
    else:
        requested_sources = normalize_external_sources(args.external_source)
        items, paper_manifest = collect_ranked_paper_evidence_batch(
            sources=requested_sources,
            queries=list(args.query),
            query_profile=str(args.query_profile),
            source_queries=parse_source_query_overrides(args.source_query),
            discovery_limit=int(args.paper_discovery_limit),
            per_query_limit=int(args.per_query_limit),
            timeout=int(args.timeout),
            user_agent=str(args.user_agent),
            delay_seconds=float(args.delay_seconds),
            max_source_age_days=int(args.max_source_age_days),
            as_of=str(args.as_of) or None,
            google_companion_discovery=bool(args.google_companion_discovery),
            enable_candidate_backfill=bool(args.paper_candidate_backfill),
            external_search_config=external_search_config,
        )
        external_sources = list(
            dict.fromkeys(
                str(item.metadata.get("external_source", ""))
                for item in items
                if str(item.metadata.get("external_source", ""))
            )
        ) or ["arxiv"]
        manifest_path = (
            Path(args.paper_manifest_out)
            if args.paper_manifest_out
            else out_path.with_name(out_path.stem + "_papers.json")
        )
        write_json(manifest_path, paper_manifest)
        collection_mode = "arxiv_first_batch"
        query_profile = str(args.query_profile)
    write_crawl_snapshot(items, out_path)

    summary = summarize_external_collection(items, sources=external_sources)
    summary.update(
        {
            "out_path": str(out_path),
            "collection_mode": collection_mode,
            "query_profile": query_profile,
            "paper_bundle_id": (
                str(items[0].metadata.get("paper_bundle_id", ""))
                if items
                else ""
            ),
            "paper_manifest_path": (
                str(manifest_path) if paper_manifest is not None else ""
            ),
            "discovered_arxiv_ids": (
                list(paper_manifest.get("discovered_arxiv_ids", []))
                if paper_manifest is not None
                else [str(args.arxiv_id)]
            ),
        }
    )
    if paper_manifest is not None:
        summary.update(
            {
                "paper_candidate_backfill_enabled": bool(
                    paper_manifest.get(
                        "paper_candidate_backfill_enabled",
                        args.paper_candidate_backfill,
                    )
                ),
                "generation_candidate_target_satisfied": bool(
                    paper_manifest.get(
                        "generation_candidate_target_satisfied", False
                    )
                ),
                "generation_candidate_shortfall": int(
                    paper_manifest.get("generation_candidate_shortfall", 0) or 0
                ),
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if int(summary["usable_items"]) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
