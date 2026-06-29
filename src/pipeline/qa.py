from __future__ import annotations

import logging
from collections import defaultdict

from src.models.schemas import (
    CanonicalEntity,
    ContentItem,
    EntityMention,
    EntitySentiment,
    QAFlag,
)

logger = logging.getLogger(__name__)

# Sentiment conflict: both sentiments must have confidence above this threshold
_CONFLICT_CONFIDENCE_THRESHOLD = 0.6

# Numeric inconsistency: flag when the same metric differs by more than this fraction
_NUMERIC_INCONSISTENCY_THRESHOLD = 0.005  # 0.5%


def _labels_conflict(a: str, b: str) -> bool:
    """Two labels conflict if one is bullish and the other is bearish."""
    return {a, b} == {"bullish", "bearish"}


def detect_sentiment_conflicts(
    sentiments: list[EntitySentiment],
) -> list[QAFlag]:
    """
    Flag entities with contradictory sentiment signals across different sources.
    Only flags conflicts where both signals are high-confidence.
    """
    flags: list[QAFlag] = []

    # Group sentiments by canonical_id
    by_entity: dict[str, list[EntitySentiment]] = defaultdict(list)
    for s in sentiments:
        by_entity[s.canonical_id].append(s)

    for cid, entity_sentiments in by_entity.items():
        if len(entity_sentiments) < 2:
            continue

        # Compare all pairs from different sources
        for i, s1 in enumerate(entity_sentiments):
            for s2 in entity_sentiments[i + 1 :]:
                if s1.source_name == s2.source_name:
                    continue
                if (
                    s1.confidence >= _CONFLICT_CONFIDENCE_THRESHOLD
                    and s2.confidence >= _CONFLICT_CONFIDENCE_THRESHOLD
                    and _labels_conflict(s1.label, s2.label)
                ):
                    flags.append(
                        QAFlag(
                            flag_type="sentiment_conflict",
                            entity_id=cid,
                            details=(
                                f"'{s1.canonical_name}' is {s1.label} ({s1.confidence:.2f}) "
                                f"per {s1.source_name}, but {s2.label} ({s2.confidence:.2f}) "
                                f"per {s2.source_name}."
                            ),
                            sources=[s1.source_name, s2.source_name],
                        )
                    )

    logger.info("QA: %d sentiment conflicts detected", len(flags))
    return flags


def detect_unresolved_entities(
    mentions: list[EntityMention],
    registry_entities: list[CanonicalEntity],
) -> list[QAFlag]:
    """
    Flag entities that could not be confidently resolved:
    - canonical_id is None
    - resolution_confidence < 0.5
    """
    flags: list[QAFlag] = []
    seen: set[str] = set()

    for mention in mentions:
        if mention.canonical_id is None:
            key = mention.surface_form
            if key not in seen:
                seen.add(key)
                flags.append(
                    QAFlag(
                        flag_type="unresolved_entity",
                        entity_id=None,
                        details=f"Entity '{mention.surface_form}' (type: {mention.entity_type}) could not be resolved.",
                        sources=[mention.source_content_id],
                    )
                )
        elif mention.resolution_confidence < 0.5:
            key = mention.canonical_id
            if key not in seen:
                seen.add(key)
                flags.append(
                    QAFlag(
                        flag_type="unresolved_entity",
                        entity_id=mention.canonical_id,
                        details=(
                            f"Entity '{mention.surface_form}' resolved to '{mention.canonical_id}' "
                            f"with low confidence ({mention.resolution_confidence:.2f})."
                        ),
                        sources=[mention.source_content_id],
                    )
                )

    logger.info("QA: %d unresolved entity flags", len(flags))
    return flags


def detect_numeric_inconsistencies(
    items: list[ContentItem],
) -> list[QAFlag]:
    """
    Flag numeric data that appears inconsistent across sources for the same metric.
    Uses simple heuristics: look for numbers adjacent to the same keyword across items.
    """
    import re

    flags: list[QAFlag] = []

    # Metric keywords mapped to how they appear in financial text
    metric_patterns: dict[str, str] = {
        "EUR/USD rate": r"(?:EUR/USD|EURUSD|euro.*dollar)\D{0,20}(\d+\.\d{2,6})",
        "USD/JPY rate": r"(?:USD/JPY|USDJPY|dollar.*yen)\D{0,20}(\d{2,4}\.\d{1,4})",
        "interest rate": r"(?:interest rate|fed funds rate|federal funds rate)\D{0,20}(\d+\.?\d*)\s*%",
        "CPI": r"(?:CPI|consumer price index)\D{0,20}(\d+\.?\d*)\s*%",
        "GDP": r"(?:GDP|gross domestic product)\D{0,20}(\d+\.?\d*)\s*%",
    }

    for metric_name, pattern in metric_patterns.items():
        readings: list[tuple[float, str]] = []  # (value, source_name)

        for item in items:
            text = f"{item.title} {item.body}"
            for m in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    val = float(m.group(1))
                    readings.append((val, item.source_name))
                except (ValueError, IndexError):
                    continue

        if len(readings) < 2:
            continue

        # Check if any pair of readings differs by more than threshold
        for i, (v1, src1) in enumerate(readings):
            for v2, src2 in readings[i + 1 :]:
                if src1 == src2:
                    continue
                if v1 == 0:
                    continue
                pct_diff = abs(v1 - v2) / abs(v1)
                if pct_diff > _NUMERIC_INCONSISTENCY_THRESHOLD:
                    flags.append(
                        QAFlag(
                            flag_type="numeric_inconsistency",
                            entity_id=None,
                            details=(
                                f"'{metric_name}' shows inconsistent values across sources: "
                                f"{v1} ({src1}) vs {v2} ({src2}), "
                                f"difference: {pct_diff:.1%}."
                            ),
                            sources=[src1, src2],
                        )
                    )

    # Deduplicate flags by (metric, sources pair)
    seen_details: set[str] = set()
    unique_flags: list[QAFlag] = []
    for f in flags:
        key = f.details
        if key not in seen_details:
            seen_details.add(key)
            unique_flags.append(f)

    logger.info("QA: %d numeric inconsistency flags", len(unique_flags))
    return unique_flags


def run_qa(
    items: list[ContentItem],
    mentions: list[EntityMention],
    sentiments: list[EntitySentiment],
    registry_entities: list[CanonicalEntity],
) -> list[QAFlag]:
    flags: list[QAFlag] = []
    flags.extend(detect_sentiment_conflicts(sentiments))
    flags.extend(detect_unresolved_entities(mentions, registry_entities))
    flags.extend(detect_numeric_inconsistencies(items))
    logger.info("QA complete: %d total flags", len(flags))
    return flags
