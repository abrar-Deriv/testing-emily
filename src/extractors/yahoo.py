from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from src.extractors.base import BaseExtractor, build_http_client, make_content_item
from src.models.schemas import ContentItem

logger = logging.getLogger(__name__)

_EURUSD_URL = "https://finance.yahoo.com/quote/EURUSD%3DX/"
_MARKETS_URL = "https://finance.yahoo.com/markets/"


class YahooQuoteExtractor(BaseExtractor):
    """Extracts the EUR/USD quote page — price, stats, and any embedded news."""

    source_name = "Yahoo Finance (EUR/USD)"

    def extract(self) -> list[ContentItem]:
        items: list[ContentItem] = []
        with build_http_client() as client:
            try:
                resp = self._get(_EURUSD_URL, client)
            except Exception:
                logger.exception("Failed to fetch Yahoo EUR/USD quote page")
                return items

            soup = BeautifulSoup(resp.text, "lxml")

            # Try to pull structured data from Next.js __NEXT_DATA__ JSON blob
            next_data = self._extract_next_data(resp.text)
            if next_data:
                item = self._parse_next_data(next_data)
                if item:
                    items.append(item)
                    return items

            # Fallback: scrape visible price/stat elements
            body_parts: list[str] = []
            price_el = soup.select_one("fin-streamer[data-symbol='EURUSD=X']")
            if price_el:
                body_parts.append(f"EUR/USD price: {price_el.get_text(strip=True)}")

            for row in soup.select("tr"):
                tds = row.find_all("td")
                if len(tds) == 2:
                    body_parts.append(f"{tds[0].get_text(strip=True)}: {tds[1].get_text(strip=True)}")

            title = "EUR/USD Exchange Rate"
            body = " | ".join(body_parts) if body_parts else soup.get_text(" ", strip=True)[:3000]
            if body:
                items.append(
                    make_content_item(
                        source_url=_EURUSD_URL,
                        source_name=self.source_name,
                        title=title,
                        body=body[:6000],
                    )
                )

        logger.info("Yahoo EUR/USD: extracted %d items", len(items))
        return items

    def _extract_next_data(self, html: str) -> dict | None:
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', html, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None

    def _parse_next_data(self, data: dict) -> ContentItem | None:
        try:
            # Navigate the deeply nested structure — path varies by Yahoo version
            stores = (
                data.get("props", {})
                .get("pageProps", {})
                .get("financialData", {})
            )
            if not stores:
                return None
            price = stores.get("regularMarketPrice", {}).get("raw", "N/A")
            change = stores.get("regularMarketChangePercent", {}).get("fmt", "N/A")
            body = f"EUR/USD rate: {price}, change: {change}%"
            return make_content_item(
                source_url=_EURUSD_URL,
                source_name=self.source_name,
                title="EUR/USD Exchange Rate",
                body=body,
            )
        except (KeyError, AttributeError, TypeError):
            return None


class YahooMarketsExtractor(BaseExtractor):
    """Extracts market headlines from Yahoo Finance Markets page."""

    source_name = "Yahoo Finance Markets"

    def extract(self) -> list[ContentItem]:
        items: list[ContentItem] = []
        with build_http_client() as client:
            try:
                resp = self._get(_MARKETS_URL, client)
            except Exception:
                logger.exception("Failed to fetch Yahoo Markets page")
                return items

            soup = BeautifulSoup(resp.text, "lxml")

            # Yahoo markets renders headline lists in <h3> tags inside article/section elements
            seen_titles: set[str] = set()
            for el in soup.find_all(["h3", "h2"]):
                title_text = el.get_text(strip=True)
                if len(title_text) < 15 or title_text in seen_titles:
                    continue
                seen_titles.add(title_text)

                # Try to grab a summary/description from a sibling <p>
                sibling = el.find_next_sibling("p")
                body = sibling.get_text(strip=True) if sibling else ""

                # Try to find a timestamp
                pub = None
                time_el = el.find_next("time")
                if time_el and time_el.get("datetime"):
                    try:
                        pub = datetime.fromisoformat(time_el["datetime"].replace("Z", "+00:00"))
                    except ValueError:
                        pass

                items.append(
                    make_content_item(
                        source_url=_MARKETS_URL,
                        source_name=self.source_name,
                        title=title_text,
                        body=body,
                        published_at=pub,
                    )
                )
                if len(items) >= 30:
                    break

            # Fallback: if no headlines found, create one item with page text
            if not items:
                page_text = soup.get_text(" ", strip=True)[:5000]
                items.append(
                    make_content_item(
                        source_url=_MARKETS_URL,
                        source_name=self.source_name,
                        title="Yahoo Finance Markets Overview",
                        body=page_text,
                    )
                )

        logger.info("Yahoo Markets: extracted %d items", len(items))
        return items
