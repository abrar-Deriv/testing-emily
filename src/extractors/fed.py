from __future__ import annotations

import logging
import re
from datetime import datetime

from bs4 import BeautifulSoup

from src.extractors.base import BaseExtractor, build_http_client, make_content_item
from src.models.schemas import ContentItem

logger = logging.getLogger(__name__)

_BASE = "https://www.federalreserve.gov"
_LIST_URL = f"{_BASE}/newsevents/pressreleases.htm"


def _parse_date(raw: str) -> datetime | None:
    raw = raw.strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


class FedExtractor(BaseExtractor):
    source_name = "Federal Reserve"

    def extract(self) -> list[ContentItem]:
        items: list[ContentItem] = []
        with build_http_client() as client:
            try:
                resp = self._get(_LIST_URL, client)
            except Exception:
                logger.exception("Failed to fetch Fed press release list")
                return items

            soup = BeautifulSoup(resp.text, "lxml")

            # The press release page lists releases in a table with rows containing
            # a date column and a title/link column.
            rows = soup.select("div.row.eventlist") or soup.select("div.col-xs-12")

            # Fallback: grab all links under the newsevents content area
            content_div = soup.find("div", id="article") or soup.find("main") or soup
            links = content_div.find_all("a", href=True)  # type: ignore[union-attr]

            seen_hrefs: set[str] = set()
            for link in links:
                href: str = link["href"]
                if not href.endswith(".htm") and not href.endswith(".html"):
                    continue
                if "pressrelease" not in href and "monetary" not in href and "fomc" not in href:
                    continue
                full_url = href if href.startswith("http") else f"{_BASE}{href}"
                if full_url in seen_hrefs:
                    continue
                seen_hrefs.add(full_url)

                title_text = link.get_text(strip=True)
                if len(title_text) < 10:
                    continue

                # Try to find a nearby date string
                parent = link.parent
                date_str = ""
                if parent:
                    sibling_text = parent.get_text(" ", strip=True)
                    date_match = re.search(
                        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                        r"\s+\d{1,2},\s+\d{4}",
                        sibling_text,
                    )
                    if date_match:
                        date_str = date_match.group(0)

                published_at = _parse_date(date_str) if date_str else None

                # Fetch article body (up to 20 articles)
                if len(items) >= 20:
                    break
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

        logger.info("Fed: extracted %d items", len(items))
        return items

    def _fetch_body(self, url: str, client) -> str:
        try:
            resp = self._get(url, client)
            soup = BeautifulSoup(resp.text, "lxml")
            article = soup.find("div", id="article") or soup.find("div", class_="col-xs-12 col-sm-8")
            if article:
                return article.get_text(" ", strip=True)[:8000]
            return soup.get_text(" ", strip=True)[:4000]
        except Exception:
            logger.debug("Could not fetch Fed article body: %s", url)
            return ""
