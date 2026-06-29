from __future__ import annotations

import hashlib
import logging
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.models.schemas import ContentItem, ExtractedNumber

logger = logging.getLogger(__name__)

# Matches numbers with optional unit suffixes (%, bps, bn, trn, etc.)
_NUMBER_RE = re.compile(
    r"(?P<value>-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    r"(?:\s*(?P<unit>%|bps|bp|bn|trn|mn|billion|trillion|million|USD|EUR|GBP|JPY|pct|pp))?",
    re.IGNORECASE,
)


def extract_numbers(text: str) -> list[ExtractedNumber]:
    """Pull numeric data points out of a text string."""
    results: list[ExtractedNumber] = []
    seen: set[str] = set()
    for m in _NUMBER_RE.finditer(text):
        raw = m.group("value").replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        unit = (m.group("unit") or "").strip()
        # Grab up to 60 chars of surrounding context
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 30)
        ctx = text[start:end].strip()
        key = f"{val}|{unit}"
        if key not in seen:
            seen.add(key)
            results.append(ExtractedNumber(value=val, unit=unit, context_span=ctx))
    return results


def content_hash(text: str) -> str:
    """SHA-256 fingerprint used to deduplicate overlapping content across sources."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalized.encode()).hexdigest()


def make_content_item(
    *,
    source_url: str,
    source_name: str,
    title: str,
    body: str,
    published_at: datetime | None = None,
) -> ContentItem:
    combined = f"{title} {body}"
    return ContentItem(
        id=str(uuid.uuid4()),
        source_url=source_url,
        source_name=source_name,
        title=title,
        body=body,
        published_at=published_at,
        numbers=extract_numbers(combined),
        content_hash=content_hash(combined),
    )


def build_http_client() -> httpx.Client:
    return httpx.Client(
        timeout=settings.http_timeout,
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )


class BaseExtractor(ABC):
    source_name: str = ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _get(self, url: str, client: httpx.Client) -> httpx.Response:
        logger.debug("GET %s", url)
        resp = client.get(url)
        resp.raise_for_status()
        return resp

    @abstractmethod
    def extract(self) -> list[ContentItem]:
        """Fetch and return normalized ContentItem list."""
