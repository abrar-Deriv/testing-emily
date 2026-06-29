from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import settings
from src.llm.client import LLMClient
from src.models.schemas import ContentItem, EntityMention

logger = logging.getLogger(__name__)

# How many content items to pack into a single LLM call.
# Higher = fewer API calls (faster + cheaper); lower = more accurate span offsets.
_BATCH_SIZE = 5

# Max parallel LLM calls at once.
_MAX_WORKERS = 6

_SYSTEM_PROMPT = """\
You are a financial NLP system specialized in Named Entity Recognition (NER) for financial texts.

Your task is to identify ALL financial entities in the provided texts and return them as structured JSON.

Entity taxonomy:
- currency_pair: e.g. EUR/USD, GBP/JPY, EURUSD, cable, fiber
- index: e.g. S&P 500, SPX, Dow Jones, FTSE 100, DAX, Nikkei 225
- central_bank: e.g. Federal Reserve, ECB, Bank of England, BOJ, PBOC, "the Fed"
- indicator: e.g. CPI, NFP, GDP, PCE, PMI, interest rate, unemployment rate, inflation
- company: e.g. Goldman Sachs, JPMorgan, Deutsche Bank
- person: e.g. Jerome Powell, Christine Lagarde, Janet Yellen

You will receive multiple numbered articles. For each article, return its entities under the matching article_id key.

Return ONLY valid JSON:
{
  "articles": {
    "0": {
      "entities": [
        {"surface_form": "the Fed", "entity_type": "central_bank"}
      ]
    },
    "1": {
      "entities": [
        {"surface_form": "EUR/USD", "entity_type": "currency_pair"}
      ]
    }
  }
}

Rules:
- Keys in "articles" must match the article_id numbers provided.
- Do NOT hallucinate entities not present in the text.
- Common abbreviations: "the Fed" = Federal Reserve, "ECB" = European Central Bank.
- If an article has no financial entities, return an empty entities list for it.
"""


def _build_text(item: ContentItem) -> str:
    return f"{item.title} {item.body}"


def _process_batch(
    batch: list[ContentItem],
    llm: LLMClient,
) -> list[EntityMention]:
    """Send a batch of items in one LLM call and return all EntityMentions."""
    # Build numbered article blocks
    article_blocks: list[str] = []
    for idx, item in enumerate(batch):
        text = _build_text(item)[: settings.max_chunk_tokens * 3]
        article_blocks.append(f"[Article {idx}] Source: {item.source_name}\n{text}")

    user_content = "\n\n---\n\n".join(article_blocks)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract financial entities from each article:\n\n{user_content}"},
    ]

    try:
        result = llm.chat_json(
            model=settings.model_extraction,
            messages=messages,
            stage="extraction",
            source_name=batch[0].source_name if batch else "",
        )
    except Exception:
        logger.exception("Entity recognition batch failed (%d items)", len(batch))
        return []

    mentions: list[EntityMention] = []
    articles_data = result.get("articles", {})

    for idx, item in enumerate(batch):
        text = _build_text(item)
        article_result = articles_data.get(str(idx), {})
        raw_entities = article_result.get("entities", [])

        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue
            surface = ent.get("surface_form", "").strip()
            etype = ent.get("entity_type", "unknown").strip()
            if not surface:
                continue

            # Hallucination guard: surface must actually appear in the text
            if surface.lower() not in text.lower():
                logger.debug("Hallucinated entity '%s' — skipping", surface)
                continue

            idx_in_text = text.lower().find(surface.lower())
            span_start = idx_in_text if idx_in_text >= 0 else 0
            span_end = span_start + len(surface)

            mentions.append(
                EntityMention(
                    surface_form=surface,
                    entity_type=etype,
                    span_start=span_start,
                    span_end=span_end,
                    source_content_id=item.id,
                )
            )

    return mentions


def recognize_entities(
    items: list[ContentItem],
    llm: LLMClient,
) -> list[EntityMention]:
    """
    Run entity recognition over a list of ContentItems.

    Speed optimizations:
    - Items are grouped into batches of _BATCH_SIZE (5 items → 1 LLM call instead of 5).
    - Batches are processed in parallel via ThreadPoolExecutor (_MAX_WORKERS concurrent calls).
    This reduces wall-clock time by ~10-20x vs sequential single-item calls.
    """
    # Filter out very short items
    valid_items = [it for it in items if len(_build_text(it).strip()) >= 20]

    # Split into batches
    batches = [valid_items[i : i + _BATCH_SIZE] for i in range(0, len(valid_items), _BATCH_SIZE)]
    logger.info(
        "Entity recognition: %d items → %d batches (parallel workers: %d)",
        len(valid_items), len(batches), _MAX_WORKERS,
    )

    all_mentions: list[EntityMention] = []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(_process_batch, batch, llm): batch
            for batch in batches
        }
        for future in as_completed(future_to_batch):
            try:
                mentions = future.result()
                all_mentions.extend(mentions)
            except Exception:
                logger.exception("Batch future raised an exception")

    logger.info("Entity recognition: %d mentions found across %d items", len(all_mentions), len(valid_items))
    return all_mentions
