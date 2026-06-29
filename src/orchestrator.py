from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.briefing.generator import generate_briefings
from src.extractors.base import BaseExtractor
from src.extractors.fed import FedExtractor
from src.extractors.imf import IMFExtractor
from src.extractors.reuters import ReutersExtractor
from src.extractors.yahoo import YahooMarketsExtractor, YahooQuoteExtractor
from src.llm.client import LLMClient
from src.llm.cost_tracker import CostTracker
from src.models.schemas import ContentItem, PipelineResult
from src.pipeline.entity_recognition import recognize_entities
from src.pipeline.entity_resolution import resolve_entities
from src.pipeline.qa import run_qa
from src.pipeline.sentiment import score_sentiments

logger = logging.getLogger(__name__)

_ALL_EXTRACTORS: dict[str, type[BaseExtractor]] = {
    "yahoo_eurusd": YahooQuoteExtractor,
    "yahoo_markets": YahooMarketsExtractor,
    "fed": FedExtractor,
    "imf": IMFExtractor,
    "reuters": ReutersExtractor,
}

OUTPUT_DIR = Path("output")


def _run_extractor(name: str, cls: type[BaseExtractor]) -> tuple[str, list[ContentItem]]:
    logger.info("Starting extractor: %s", name)
    try:
        extractor = cls()
        items = extractor.extract()
        logger.info("Extractor %s: %d items", name, len(items))
        return name, items
    except Exception:
        logger.exception("Extractor %s failed", name)
        return name, []


def _dedup_items(items: list[ContentItem]) -> list[ContentItem]:
    """Remove duplicate ContentItems by content_hash (cross-source dedup for cost savings)."""
    seen_hashes: set[str] = set()
    unique: list[ContentItem] = []
    for item in items:
        if item.content_hash not in seen_hashes:
            seen_hashes.add(item.content_hash)
            unique.append(item)
        else:
            logger.debug("Dedup: skipping duplicate content from %s", item.source_name)
    return unique


def run_pipeline(
    source_filter: list[str] | None = None,
    dry_run: bool = False,
) -> PipelineResult:
    """
    Main pipeline orchestration:
    1. Extract from sources (parallel)
    2. Dedup
    3. Entity recognition (LLM)
    4. Entity resolution (deterministic + LLM)
    5. Per-entity sentiment (batched LLM)
    6. QA
    7. Audience briefings
    8. Write output files
    """
    run_id = str(uuid.uuid4())
    started_at = datetime.utcnow()
    tracker = CostTracker()
    llm = LLMClient(tracker)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Content extraction (parallel)
    extractors = _ALL_EXTRACTORS
    if source_filter:
        extractors = {k: v for k, v in _ALL_EXTRACTORS.items() if k in source_filter}

    all_items: list[ContentItem] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_run_extractor, name, cls): name
            for name, cls in extractors.items()
        }
        for future in as_completed(futures):
            _, items = future.result()
            all_items.extend(items)

    logger.info("Total extracted: %d items across all sources", len(all_items))

    if not all_items:
        logger.warning("No content extracted from any source. Check network access.")

    # Cap items per source to keep LLM processing fast (5 items × 5 sources = 25 max)
    _MAX_ITEMS_PER_SOURCE = 5
    capped: list[ContentItem] = []
    source_counts: dict[str, int] = {}
    for item in all_items:
        count = source_counts.get(item.source_name, 0)
        if count < _MAX_ITEMS_PER_SOURCE:
            capped.append(item)
            source_counts[item.source_name] = count + 1
    if len(capped) < len(all_items):
        logger.info("Capped to %d items (was %d) to stay within time budget", len(capped), len(all_items))
    all_items = capped

    # Step 2: Dedup (cost optimization: don't process duplicate bodies twice)
    unique_items = _dedup_items(all_items)
    logger.info("After dedup: %d unique items (removed %d)", len(unique_items), len(all_items) - len(unique_items))

    if dry_run:
        logger.info("Dry run: skipping LLM stages.")
        return PipelineResult(
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.utcnow(),
            content_items=unique_items,
            entities=[],
            sentiments=[],
            qa_flags=[],
            cost_records=[],
            total_cost_usd=0.0,
        )

    # Step 3: Entity recognition
    logger.info("Running entity recognition on %d items...", len(unique_items))
    mentions = recognize_entities(unique_items, llm)

    # Step 4: Entity resolution
    logger.info("Running entity resolution on %d mentions...", len(mentions))
    registry, resolved_mentions = resolve_entities(mentions, llm)
    entity_names = {e.canonical_id: e.canonical_name for e in registry.all_entities}

    # Step 5: Per-entity sentiment (batched)
    logger.info("Running sentiment scoring...")
    sentiments = score_sentiments(unique_items, resolved_mentions, entity_names, llm)

    # Step 6: QA
    logger.info("Running QA checks...")
    qa_flags = run_qa(unique_items, resolved_mentions, sentiments, registry.all_entities)

    # Step 7: Audience briefings
    logger.info("Generating audience briefings...")
    briefings = generate_briefings(registry.all_entities, sentiments, qa_flags, llm)

    finished_at = datetime.utcnow()
    result = PipelineResult(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        content_items=unique_items,
        entities=registry.all_entities,
        sentiments=sentiments,
        qa_flags=qa_flags,
        cost_records=tracker.records,
        total_cost_usd=tracker.total_cost,
        briefings=briefings,
    )

    # Step 8: Write outputs
    _write_outputs(result, tracker)

    return result


def _write_outputs(result: PipelineResult, tracker: CostTracker) -> None:
    # Full pipeline result JSON
    result_path = OUTPUT_DIR / "pipeline_result.json"
    with open(result_path, "w") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2, default=str)
    logger.info("Pipeline result written to %s", result_path)

    # Per-audience briefing markdown files
    for audience, text in result.briefings.items():
        brief_path = OUTPUT_DIR / f"briefing_{audience}.md"
        brief_path.write_text(text, encoding="utf-8")
        logger.info("Briefing written: %s", brief_path)

    # Cost report JSON
    cost_report = {
        "run_id": result.run_id,
        "total_cost_usd": tracker.total_cost,
        "by_stage": tracker.summary_by_stage(),
        "by_source": tracker.summary_by_source(),
        "records": [r.model_dump() for r in tracker.records],
    }
    cost_path = OUTPUT_DIR / "cost_report.json"
    with open(cost_path, "w") as f:
        json.dump(cost_report, f, indent=2)
    logger.info("Cost report written to %s", cost_path)

    # Print summary to stdout
    _print_summary(result, tracker)


def _print_summary(result: PipelineResult, tracker: CostTracker) -> None:
    duration = (result.finished_at - result.started_at).total_seconds()
    print("\n" + "=" * 60)
    print("FINANCIAL INTELLIGENCE PIPELINE — RUN SUMMARY")
    print("=" * 60)
    print(f"Run ID       : {result.run_id}")
    print(f"Duration     : {duration:.1f}s")
    print(f"Sources      : {len({it.source_name for it in result.content_items})}")
    print(f"Content items: {len(result.content_items)}")
    print(f"Entities     : {len(result.entities)}")
    print(f"Sentiments   : {len(result.sentiments)}")
    print(f"QA flags     : {len(result.qa_flags)}")
    print(f"Est. cost    : ${tracker.total_cost:.6f}")

    print("\nCost by stage:")
    for stage, cost in tracker.summary_by_stage().items():
        print(f"  {stage:<15} ${cost:.6f}")

    print("\nCost by source:")
    for src, cost in tracker.summary_by_source().items():
        print(f"  {src:<30} ${cost:.6f}")

    if result.qa_flags:
        print(f"\nQA FLAGS ({len(result.qa_flags)}):")
        for flag in result.qa_flags[:10]:
            print(f"  [{flag.flag_type}] {flag.details[:100]}")
        if len(result.qa_flags) > 10:
            print(f"  ... and {len(result.qa_flags) - 10} more (see pipeline_result.json)")

    print("\nOutput files:")
    for f in sorted(Path("output").glob("*")):
        print(f"  {f}")
    print("=" * 60)
