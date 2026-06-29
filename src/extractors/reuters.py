from __future__ import annotations

import logging
import re
from datetime import datetime

from bs4 import BeautifulSoup

from src.extractors.base import BaseExtractor, build_http_client, make_content_item
from src.models.schemas import ContentItem

logger = logging.getLogger(__name__)

_REUTERS_URL = "https://www.reuters.com/markets/currencies/"


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


class ReutersExtractor(BaseExtractor):
    """
    Extracts currency market headlines from Reuters.
    Reuters serves JS-rendered content, so plain HTTP often returns minimal HTML.
    We attempt a plain HTTP scrape first; Playwright fallback is handled by the
    orchestrator via PlaywrightReutersExtractor if this returns empty results.
    """

    source_name = "Reuters"

    def extract(self) -> list[ContentItem]:
        # Try plain HTTP; on any failure (including 401/403) fall through to Playwright
        items: list[ContentItem] = []
        try:
            with build_http_client() as client:
                resp = self._get(_REUTERS_URL, client)
            soup = BeautifulSoup(resp.text, "lxml")
            items = self._parse_html(soup)
        except Exception:
            logger.info("Reuters plain HTTP failed — trying Playwright fallback")

        if not items:
            logger.info("Reuters: no items from plain HTTP — trying Playwright fallback")
            items = self._playwright_extract()

        logger.info("Reuters: extracted %d items", len(items))
        return items

    def _parse_html(self, soup: BeautifulSoup) -> list[ContentItem]:
        items: list[ContentItem] = []
        seen: set[str] = set()

        for el in soup.find_all(["h3", "h2", "a"], limit=200):
            title_text = el.get_text(strip=True)
            if len(title_text) < 20 or title_text in seen:
                continue
            # Filter for finance-related headlines
            if not re.search(r"(dollar|euro|yen|pound|yuan|currency|forex|fed|rate|bank|inflation)", title_text, re.I):
                continue
            seen.add(title_text)

            sibling = el.find_next_sibling("p")
            body = sibling.get_text(strip=True) if sibling else ""

            time_el = el.find_next("time")
            pub = _parse_iso(time_el.get("datetime") if time_el else None)

            items.append(
                make_content_item(
                    source_url=_REUTERS_URL,
                    source_name=self.source_name,
                    title=title_text,
                    body=body,
                    published_at=pub,
                )
            )
            if len(items) >= 20:
                break
        return items

    def _playwright_extract(self) -> list[ContentItem]:
        """Playwright-based fallback for JS-rendered Reuters pages."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.warning("Playwright not installed; skipping Reuters fallback")
            return []

        items: list[ContentItem] = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    user_agent=(
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )
                )
                page.goto(_REUTERS_URL, timeout=30000, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)
                html = page.content()
                browser.close()

            soup = BeautifulSoup(html, "lxml")
            items = self._parse_html(soup)

            # If still empty, grab all visible text as a single item
            if not items:
                page_text = soup.get_text(" ", strip=True)[:5000]
                if page_text:
                    items.append(
                        make_content_item(
                            source_url=_REUTERS_URL,
                            source_name=self.source_name,
                            title="Reuters Currencies Market Overview",
                            body=page_text,
                        )
                    )
        except Exception:
            logger.exception("Playwright Reuters extraction failed")

        return items
