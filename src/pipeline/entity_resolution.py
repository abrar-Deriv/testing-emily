from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict

from src.config import settings
from src.llm.client import LLMClient
from src.models.schemas import CanonicalEntity, EntityMention

logger = logging.getLogger(__name__)

# Deterministic alias expansion rules applied before hitting the LLM.
# Key = normalized surface form, value = (canonical_name, entity_type)
_KNOWN_ALIASES: dict[str, tuple[str, str]] = {
    # Central banks
    "the fed": ("Federal Reserve", "central_bank"),
    "fed": ("Federal Reserve", "central_bank"),
    "federal reserve": ("Federal Reserve", "central_bank"),
    "us central bank": ("Federal Reserve", "central_bank"),
    "u.s. central bank": ("Federal Reserve", "central_bank"),
    "fomc": ("Federal Reserve", "central_bank"),
    "ecb": ("European Central Bank", "central_bank"),
    "european central bank": ("European Central Bank", "central_bank"),
    "boe": ("Bank of England", "central_bank"),
    "bank of england": ("Bank of England", "central_bank"),
    "boj": ("Bank of Japan", "central_bank"),
    "bank of japan": ("Bank of Japan", "central_bank"),
    "pboc": ("People's Bank of China", "central_bank"),
    "peoples bank of china": ("People's Bank of China", "central_bank"),
    "snb": ("Swiss National Bank", "central_bank"),
    "swiss national bank": ("Swiss National Bank", "central_bank"),
    "rba": ("Reserve Bank of Australia", "central_bank"),
    # Currency pairs
    "eurusd": ("EUR/USD", "currency_pair"),
    "eur/usd": ("EUR/USD", "currency_pair"),
    "euro dollar": ("EUR/USD", "currency_pair"),
    "fiber": ("EUR/USD", "currency_pair"),
    "gbpusd": ("GBP/USD", "currency_pair"),
    "gbp/usd": ("GBP/USD", "currency_pair"),
    "cable": ("GBP/USD", "currency_pair"),
    "usdjpy": ("USD/JPY", "currency_pair"),
    "usd/jpy": ("USD/JPY", "currency_pair"),
    "dollar yen": ("USD/JPY", "currency_pair"),
    "usdcny": ("USD/CNY", "currency_pair"),
    "usd/cny": ("USD/CNY", "currency_pair"),
    "dollar yuan": ("USD/CNY", "currency_pair"),
    # Indices
    "s&p 500": ("S&P 500", "index"),
    "s&p500": ("S&P 500", "index"),
    "spx": ("S&P 500", "index"),
    "sp500": ("S&P 500", "index"),
    "dow jones": ("Dow Jones", "index"),
    "djia": ("Dow Jones", "index"),
    "dow": ("Dow Jones", "index"),
    "nasdaq": ("NASDAQ", "index"),
    "nasdaq composite": ("NASDAQ", "index"),
    "ftse 100": ("FTSE 100", "index"),
    "ftse100": ("FTSE 100", "index"),
    "dax": ("DAX", "index"),
    "nikkei": ("Nikkei 225", "index"),
    "nikkei 225": ("Nikkei 225", "index"),
    # Economic indicators
    "cpi": ("CPI", "indicator"),
    "consumer price index": ("CPI", "indicator"),
    "nfp": ("NFP", "indicator"),
    "non-farm payrolls": ("NFP", "indicator"),
    "non farm payrolls": ("NFP", "indicator"),
    "gdp": ("GDP", "indicator"),
    "gross domestic product": ("GDP", "indicator"),
    "pce": ("PCE", "indicator"),
    "personal consumption expenditures": ("PCE", "indicator"),
    "pmi": ("PMI", "indicator"),
    "purchasing managers index": ("PMI", "indicator"),
    "unemployment rate": ("Unemployment Rate", "indicator"),
    "interest rate": ("Interest Rate", "indicator"),
    "federal funds rate": ("Federal Funds Rate", "indicator"),
    "fed funds rate": ("Federal Funds Rate", "indicator"),
    "inflation": ("Inflation", "indicator"),
}


def _normalize(text: str) -> str:
    """Lowercase, strip leading 'the', collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"^the\s+", "", text)
    text = re.sub(r"[''`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _make_canonical_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug


class EntityRegistry:
    """In-memory store of resolved canonical entities, keyed by canonical_id."""

    def __init__(self) -> None:
        self._entities: dict[str, CanonicalEntity] = {}
        # Maps normalized alias → canonical_id for fast lookup
        self._alias_map: dict[str, str] = {}

    def register(self, canonical_name: str, entity_type: str, aliases: list[str]) -> CanonicalEntity:
        cid = _make_canonical_id(canonical_name)
        if cid not in self._entities:
            self._entities[cid] = CanonicalEntity(
                canonical_id=cid,
                canonical_name=canonical_name,
                entity_type=entity_type,
                aliases=[],
            )
        entity = self._entities[cid]
        for alias in aliases:
            norm = _normalize(alias)
            if norm not in entity.aliases:
                entity.aliases.append(norm)
            self._alias_map[norm] = cid
        return entity

    def lookup(self, surface_form: str) -> CanonicalEntity | None:
        norm = _normalize(surface_form)
        cid = self._alias_map.get(norm)
        return self._entities.get(cid) if cid else None

    @property
    def all_entities(self) -> list[CanonicalEntity]:
        return list(self._entities.values())


_RESOLUTION_SYSTEM_PROMPT = """\
You are a financial entity resolution system. Given a list of entity surface forms that could not
be resolved by deterministic rules, cluster them into canonical entities and assign canonical names.

Return ONLY valid JSON:
{
  "resolutions": [
    {
      "surface_forms": ["Powell", "Jerome Powell", "Fed Chair Powell"],
      "canonical_name": "Jerome Powell",
      "entity_type": "person",
      "confidence": 0.95
    }
  ]
}

Rules:
- Group surface forms that clearly refer to the same real-world entity.
- canonical_name should be the most formal/complete name.
- entity_type must be one of: currency_pair, index, central_bank, indicator, company, person.
- confidence is 0.0–1.0; use < 0.5 for very uncertain groupings.
- Do NOT merge entities that might be different (err on the side of splitting).
"""


def resolve_entities(
    mentions: list[EntityMention],
    llm: LLMClient,
) -> tuple[EntityRegistry, list[EntityMention]]:
    """
    Two-pass resolution:
    1. Deterministic: apply _KNOWN_ALIASES lookup.
    2. LLM: resolve remaining unknown clusters.

    Returns an EntityRegistry and the updated mentions list with canonical_id filled in.
    """
    registry = EntityRegistry()

    # Pre-populate registry with all known aliases
    seen_canonicals: set[str] = set()
    for norm_alias, (canon_name, etype) in _KNOWN_ALIASES.items():
        cid = _make_canonical_id(canon_name)
        if cid not in seen_canonicals:
            registry.register(canon_name, etype, list({norm_alias, canon_name.lower()}))
            seen_canonicals.add(cid)
        else:
            entity = registry._entities[cid]
            if norm_alias not in entity.aliases:
                entity.aliases.append(norm_alias)
                registry._alias_map[norm_alias] = cid

    # Pass 1: deterministic resolution
    unresolved: list[EntityMention] = []
    for mention in mentions:
        entity = registry.lookup(mention.surface_form)
        if entity:
            mention.canonical_id = entity.canonical_id
            mention.resolution_confidence = 1.0
        else:
            unresolved.append(mention)

    logger.info(
        "Entity resolution pass 1: %d resolved, %d unresolved",
        len(mentions) - len(unresolved),
        len(unresolved),
    )

    if not unresolved:
        return registry, mentions

    # Pass 2: LLM resolution for unknowns
    # Group unresolved by (normalized surface_form, entity_type) to avoid sending duplicates
    surface_groups: dict[str, list[EntityMention]] = defaultdict(list)
    for m in unresolved:
        surface_groups[_normalize(m.surface_form)].append(m)

    unique_surfaces = list(surface_groups.keys())

    # Send in chunks of 30 to keep prompts manageable
    chunk_size = 30
    for i in range(0, len(unique_surfaces), chunk_size):
        chunk = unique_surfaces[i : i + chunk_size]
        entity_list_str = "\n".join(
            f"- \"{sf}\" (type hint: {surface_groups[sf][0].entity_type})" for sf in chunk
        )
        messages = [
            {"role": "system", "content": _RESOLUTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Cluster and canonicalize these financial entity surface forms:\n\n"
                    f"{entity_list_str}"
                ),
            },
        ]
        try:
            result = llm.chat_json(
                model=settings.model_extraction,
                messages=messages,
                stage="resolution",
            )
        except Exception:
            logger.exception("LLM entity resolution failed for chunk %d", i)
            continue

        for res in result.get("resolutions", []):
            if not isinstance(res, dict):
                continue
            canon_name = res.get("canonical_name", "").strip()
            etype = res.get("entity_type", "unknown")
            confidence = float(res.get("confidence", 0.5))
            surface_forms: list[str] = res.get("surface_forms", [])

            if not canon_name or not surface_forms:
                continue

            entity = registry.register(canon_name, etype, surface_forms)

            for sf in surface_forms:
                norm = _normalize(sf)
                for mention in surface_groups.get(norm, []):
                    mention.canonical_id = entity.canonical_id
                    mention.resolution_confidence = confidence

    # Any still-unresolved: auto-register as their own canonical entity
    for m in unresolved:
        if not m.canonical_id:
            entity = registry.register(
                m.surface_form, m.entity_type, [m.surface_form]
            )
            m.canonical_id = entity.canonical_id
            m.resolution_confidence = 0.3  # low confidence — flag for QA

    logger.info(
        "Entity resolution complete. Registry has %d canonical entities.",
        len(registry.all_entities),
    )
    return registry, mentions
