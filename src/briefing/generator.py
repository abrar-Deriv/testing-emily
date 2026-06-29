from __future__ import annotations

import logging
from collections import defaultdict

from src.config import settings
from src.llm.client import LLMClient
from src.models.schemas import CanonicalEntity, EntitySentiment, QAFlag

logger = logging.getLogger(__name__)

_AUDIENCES = {
    "trader": (
        "You are writing for an active FX/equity trader who needs fast, actionable intelligence.\n"
        "Format: bullet points, entity sentiment table (entity | signal | confidence | key level), "
        "conflict flags clearly highlighted. Be concise and direct. Prioritize EUR/USD, rates, and central bank moves."
    ),
    "analyst": (
        "You are writing for a macro research analyst who needs comprehensive narrative intelligence.\n"
        "Format: flowing paragraphs with cross-source thematic analysis, confidence caveats, source attribution. "
        "Highlight divergences, emerging trends, and data inconsistencies."
    ),
    "executive": (
        "You are writing for a senior executive with 2 minutes to read.\n"
        "Format: 3-5 sentences maximum. Top-line market situation, biggest risk, biggest opportunity. "
        "No jargon, no tables. Plain business language."
    ),
}


def _build_sentiment_summary(sentiments: list[EntitySentiment]) -> str:
    """Aggregate sentiments by entity, averaging scores across sources."""
    by_entity: dict[str, list[EntitySentiment]] = defaultdict(list)
    for s in sentiments:
        by_entity[s.canonical_name].append(s)

    lines: list[str] = []
    for name, sents in sorted(by_entity.items()):
        avg_score = sum(s.score for s in sents) / len(sents)
        avg_conf = sum(s.confidence for s in sents) / len(sents)
        label_counts: dict[str, int] = defaultdict(int)
        for s in sents:
            label_counts[s.label] += 1
        dominant_label = max(label_counts, key=lambda k: label_counts[k])
        sources = list({s.source_name for s in sents})
        evidence = sents[0].evidence_span if sents else ""
        lines.append(
            f"- {name}: {dominant_label} (avg score {avg_score:+.2f}, confidence {avg_conf:.2f}) "
            f"| sources: {', '.join(sources)} | evidence: \"{evidence[:100]}...\""
        )
    return "\n".join(lines) if lines else "No entities with sentiment found."


def _build_qa_summary(flags: list[QAFlag]) -> str:
    if not flags:
        return "No QA flags raised."
    lines: list[str] = []
    for f in flags:
        lines.append(f"[{f.flag_type.upper()}] {f.details}")
    return "\n".join(lines)


def generate_briefings(
    entities: list[CanonicalEntity],
    sentiments: list[EntitySentiment],
    flags: list[QAFlag],
    llm: LLMClient,
) -> dict[str, str]:
    """Generate audience-specific briefings. Returns a dict of audience → markdown text."""
    sentiment_summary = _build_sentiment_summary(sentiments)
    qa_summary = _build_qa_summary(flags)
    entity_count = len(entities)

    briefings: dict[str, str] = {}

    context = (
        f"## Entity Sentiment Summary ({entity_count} entities identified)\n"
        f"{sentiment_summary}\n\n"
        f"## QA Flags\n"
        f"{qa_summary}"
    )

    for audience, persona_prompt in _AUDIENCES.items():
        messages = [
            {
                "role": "system",
                "content": (
                    f"{persona_prompt}\n\n"
                    "Generate a financial intelligence briefing based on the data below. "
                    "Today's data was sourced from Yahoo Finance, Federal Reserve, IMF, and Reuters."
                ),
            },
            {
                "role": "user",
                "content": context,
            },
        ]

        try:
            text = llm.chat(
                model=settings.model_briefing,
                messages=messages,
                stage="briefing",
                temperature=0.3,
            )
            briefings[audience] = text
            logger.info("Generated %s briefing (%d chars)", audience, len(text))
        except Exception:
            logger.exception("Failed to generate %s briefing", audience)
            briefings[audience] = f"[Briefing generation failed for audience: {audience}]"

    return briefings
