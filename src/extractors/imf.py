from __future__ import annotations

import logging
import re
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

from src.extractors.base import BaseExtractor, build_http_client, make_content_item
from src.models.schemas import ContentItem

logger = logging.getLogger(__name__)

_BASE = "https://www.imf.org"
_LIST_URL = f"{_BASE}/en/News"


def _parse_date(raw: str) -> datetime | None:
    raw = raw.strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


class IMFExtractor(BaseExtractor):
    source_name = "IMF"

    def extract(self) -> list[ContentItem]:
        # Try plain HTTP first; fall back to Playwright if blocked (e.g. 403)
        html: str | None = self._fetch_html_plain()
        if html is None:
            logger.info("IMF plain HTTP blocked — trying Playwright fallback")
            html = self._fetch_html_playwright()
        if html is None:
            logger.error("IMF: all extraction methods failed; returning empty")
            return []

        return self._parse_listing(html)

    def _fetch_html_plain(self) -> str | None:
        try:
            with build_http_client() as client:
                resp = client.get(
                    _LIST_URL,
                    headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Referer": "https://www.google.com/",
                    },
                )
                if resp.status_code == 200:
                    return resp.text
                logger.debug("IMF plain HTTP status: %d", resp.status_code)
                return None
        except Exception:
            logger.debug("IMF plain HTTP request failed")
            return None

    def _fetch_html_playwright(self) -> str | None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.warning("Playwright not installed; cannot fetch IMF via browser")
            return None

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
                page.goto(_LIST_URL, timeout=30000, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)
                html = page.content()
                browser.close()
            return html
        except Exception:
            logger.exception("Playwright IMF extraction failed")
            return None

    def _parse_listing(self, html: str) -> list[ContentItem]:
        items: list[ContentItem] = []
        soup = BeautifulSoup(html, "lxml")
        article_links: list[tuple[str, str, str | None]] = []

        for anchor in soup.find_all("a", href=True):
            href: str = anchor["href"]
            href_lower = href.lower()
            if "/en/news/articles/" not in href_lower and "/en/publications/" not in href_lower:
                continue
            full_url = href if href.startswith("http") else f"{_BASE}{href}"
            title_text = anchor.get_text(strip=True)
            if len(title_text) < 15:
                continue

            parent = anchor.parent
            date_str: str | None = None
            if parent:
                nearby = parent.get_text(" ", strip=True)
                dm = re.search(
                    r"(January|February|March|April|May|June|July|August|"
                    r"September|October|November|December)\s+\d{1,2},\s+\d{4}",
                    nearby,
                )
                if dm:
                    date_str = dm.group(0)
            article_links.append((full_url, title_text, date_str))

        seen: set[str] = set()
        with build_http_client() as client:
            for full_url, title_text, date_str in article_links:
                if full_url in seen or len(items) >= 20:
                    break
                seen.add(full_url)
                published_at = _parse_date(date_str) if date_str else None
                body = self._fetch_body(full_url, client)
                items.append(
                    make_content_item(
                        source_url=full_url,
                        source_name=self.source_name,
                        title=title_text,
                        body=body,
                        published_at=published_at,
                    )
                )

        logger.info("IMF: extracted %d items", len(items))
        return items

    def _fetch_body(self, url: str, client: httpx.Client) -> str:
        try:
            resp = client.get(url, follow_redirects=True)
            if resp.status_code != 200:
                return ""
            soup = BeautifulSoup(resp.text, "lxml")
            for sel in ["div.imf-article-body", "div#content", "article", "main"]:
                el = soup.select_one(sel)
                if el:
                    return el.get_text(" ", strip=True)[:8000]
            return soup.get_text(" ", strip=True)[:4000]
        except Exception:
            logger.debug("Could not fetch IMF article body: %s", url)
            return ""
