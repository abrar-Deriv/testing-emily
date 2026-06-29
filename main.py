"""
Financial Content Intelligence Pipeline
----------------------------------------
Usage:
    python main.py run                          # full pipeline, all sources
    python main.py run --sources fed,imf        # specific sources only
    python main.py run --dry-run                # extract only, no LLM calls
    python main.py sources                      # list available sources
"""
from __future__ import annotations

import argparse
import logging
import sys


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> None:
    from src.orchestrator import run_pipeline

    source_filter = [s.strip() for s in args.sources.split(",")] if args.sources else None
    run_pipeline(source_filter=source_filter, dry_run=args.dry_run)


def cmd_sources(_: argparse.Namespace) -> None:
    from src.orchestrator import _ALL_EXTRACTORS

    print("Available sources:")
    for name in _ALL_EXTRACTORS:
        print(f"  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Financial Content Intelligence Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full pipeline")
    run_p.add_argument(
        "--sources",
        default=None,
        help="Comma-separated list of source keys (e.g. fed,imf,yahoo_eurusd)",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract content only; skip all LLM stages",
    )

    sub.add_parser("sources", help="List available source extractors")

    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "sources":
        cmd_sources(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
