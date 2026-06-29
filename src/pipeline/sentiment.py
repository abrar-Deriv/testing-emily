from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import settings
from src.llm.client import LLMClient
from src.models.schemas import ContentItem, EntityMention, EntitySentiment

logger = logging.getLogger(__name__)

_MAX_WORKERS = 6

_SYSTEM_PROMPT = """\
You are a financial sentiment analysis system. Your task is to analyze the sentiment of specific
financial entities within a given text.

CRITICAL: Sentiment must be ENTITY-SPECIFIC — not the overall document sentiment.
A single article can be bullish on EUR while simultaneously bearish on USD.

For each entity provided, return a sentiment assessment:
- label: "bullish" (positive outlook), "bearish" (negative outlook), or "neutral"
- score: -1.0 (most bearish) to +1.0 (most bullish), 0.0 = neutral
- confidence: 0.0 (uncertain) to 1.0 (highly certain)
- evidence_span: the exact quote or phrase from the text that most strongly drove the sentiment

Return ONLY valid JSON:
{
  "sentiments": [
    {
      "entity_surface_form": "EUR/USD",
      "label": "bullish",
      "score": 0.7,
      "confidence": 0.85,
      "evidence_span": "euro surged to a two-month high against the dollar"
    }
  ]
}

Rules:
- If the entity is mentioned but sentiment is unclear, use "neutral" with low confidence.
- evidence_span must be verbatim text from the article (max 150 chars).
- Do NOT include an entity in the output if it is not mentioned in the text.
"""


def _truncate_context(text: str, surface_form: str, window: int = 800) -> str:
    """
    Return a window of text centred around the first occurrence of the entity surface form.
    This reduces token usage while preserving the most relevant context.
    """
    idx = text.lower().find(surface_form.lower())
    if idx < 0:
        return text[:window]
    start = max(0, idx - window // 2)
    end = min(len(text), idx + window // 2)
    return text[start:end]


def score_sentiments(
    items: list[ContentItem],
    mentions: list[EntityMention],
    entity_names: dict[str, str],  # canonical_id → canonical_name
    llm: LLMClient,
) -> list[EntitySentiment]:
    """
    Produce per-entity sentiment for every (content_item, entity) pair.

    Batching optimization: group all entities from the same article into a single
    LLM call, cutting API calls by ~3-5x compared to one call per entity.
    """
    # Build: content_id → {canonical_id → [EntityMention]}
    item_map: dict[str, ContentItem] = {it.id: it for it in items}
    entity_groups: dict[str, dict[str, list[EntityMention]]] = defaultdict(lambda: defaultdict(list))

    for mention in mentions:
        if not mention.canonical_id:
            continue
        entity_groups[mention.source_content_id][mention.canonical_id].append(mention)

    def _score_one(content_id: str, entities: dict) -> list[EntitySentiment]:
        item = item_map.get(content_id)
        if not item:
            return []

        full_text = f"{item.title} {item.body}"
        canonical_ids = list(entities.keys())

        entity_lines: list[str] = []
        for cid in canonical_ids:
            name = entity_names.get(cid, cid)
            surface = entities[cid][0].surface_form
            entity_lines.append(f"- {name} (surface form: \"{surface}\")")

        entity_list_str = "\n".join(entity_lines)
        article_excerpt = full_text[: settings.max_chunk_tokens * 3]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Analyze the sentiment for each of the following entities as they appear "
                    f"in the article below.\n\n"
                    f"Entities:\n{entity_list_str}\n\n"
                    f"Article (source: {item.source_name}):\n{article_excerpt}"
                ),
            },
        ]

        try:
            result = llm.chat_json(
                model=settings.model_sentiment,
                messages=messages,
                stage="sentiment",
                source_name=item.source_name,
            )
        except Exception:
            logger.exception("Sentiment scoring failed for content_id=%s", content_id)
            return []

        sentiments: list[EntitySentiment] = []
        for sent_data in result.get("sentiments", []):
            if not isinstance(sent_data, dict):
                continue

            surface = sent_data.get("entity_surface_form", "")
            label = sent_data.get("label", "neutral")
            score = float(sent_data.get("score", 0.0))
            confidence = float(sent_data.get("confidence", 0.5))
            evidence = sent_data.get("evidence_span", "")

            if label not in ("bullish", "bearish", "neutral"):
                label = "neutral"
            score = max(-1.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))

            matched_cid: str | None = None
            for cid in canonical_ids:
                name = entity_names.get(cid, cid)
                if (
                    surface.lower() in name.lower()
                    or name.lower() in surface.lower()
                    or any(m.surface_form.lower() == surface.lower() for m in entities[cid])
                ):
                    matched_cid = cid
                    break

            if not matched_cid:
                for cid in canonical_ids:
                    name = entity_names.get(cid, cid).lower()
                    words = set(name.split())
                    if any(w in surface.lower() for w in words if len(w) > 2):
                        matched_cid = cid
                        break

            if not matched_cid:
                logger.debug("Could not match sentiment surface '%s' to any canonical entity", surface)
                continue

            sentiments.append(
                EntitySentiment(
                    canonical_id=matched_cid,
                    canonical_name=entity_names.get(matched_cid, matched_cid),
                    label=label,
                    score=score,
                    confidence=confidence,
                    evidence_span=evidence[:300],
                    source_content_id=content_id,
                    source_name=item.source_name,
                )
            )
        return sentiments

    # Run all per-article sentiment calls in parallel
    all_sentiments: list[EntitySentiment] = []
    logger.info("Sentiment scoring: %d articles to process (parallel workers: %d)", len(entity_groups), _MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_score_one, cid, ents): cid
            for cid, ents in entity_groups.items()
        }
        for future in as_completed(futures):
            try:
                all_sentiments.extend(future.result())
            except Exception:
                logger.exception("Sentiment future raised an exception")

    logger.info("Sentiment scoring: %d sentiments produced", len(all_sentiments))
    return all_sentiments
