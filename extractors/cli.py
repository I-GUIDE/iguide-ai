"""CLI entry point for GitHub-repo ingestion (code + notebook).

    python -m extractors.cli <github_url> --element-id <uuid> [--targets opensearch,mcp,skill]
    python -m extractors.cli <github_url> --dry-run          # inspect, emit nothing

Until now this printed a manifest and emitted NOTHING, while accepting --targets and
--reingest, which implied the opposite. It now emits, and requires --element-id so derived
doc_ids anchor on a real platform element instead of on repo_id (which would seed docs into
the agent KB that no element can claim). Use --dry-run for the old inspect-only behavior.

Dataset/publication ingestion is delivered through the webhook, not this CLI.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from .base import VALID_TARGETS
from .ingest import ingest_from_github


def _parse_targets(raw: str) -> List[str]:
    targets = [t.strip() for t in (raw or "").split(",") if t.strip()]
    bad = [t for t in targets if t not in VALID_TARGETS]
    if bad:
        raise SystemExit(f"unknown --targets value(s): {bad}; valid: {list(VALID_TARGETS)}")
    return targets or list(VALID_TARGETS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extractors.cli",
        description="Ingest a GitHub repo: extract notebook + code assets to OpenSearch / MCP / SKILL.",
    )
    p.add_argument("url", help="GitHub repository URL")
    p.add_argument("--element-id", default="",
                   help="platform element id that derived doc_ids anchor on (required unless --dry-run)")
    p.add_argument("--dry-run", action="store_true",
                   help="extract and print the manifest without emitting to any target")
    p.add_argument("--ref", default="", help="git ref/branch/tag/sha to check out (default: repo default)")
    p.add_argument("--targets", default=",".join(VALID_TARGETS),
                   help=f"comma-separated emit targets (default: all of {list(VALID_TARGETS)})")
    p.add_argument("--reingest", action="store_true", help="re-extract and upsert even if already ingested")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    targets = _parse_targets(args.targets)
    try:
        manifest = ingest_from_github(args.url, ref=args.ref, targets=targets,
                                      element_id=args.element_id, dry_run=args.dry_run,
                                      reingest=args.reingest)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    print(manifest.to_json())
    if args.dry_run:
        print("# dry run: nothing was emitted", file=sys.stderr)
    else:
        print(f"# emitted to: {', '.join(targets)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
