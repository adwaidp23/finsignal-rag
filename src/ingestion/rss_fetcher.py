"""
src/ingestion/rss_fetcher.py
----------------------------
Fetch financial news headlines from Google News RSS and Yahoo Finance RSS
feeds for a list of tickers.

No API key required. Output is written directly to the SQLite cache.

Usage
-----
    from src.ingestion.rss_fetcher import RSSFetcher
    from src.cache.db import Database

    db = Database()
    fetcher = RSSFetcher(db)
    fetcher.fetch_ticker("AAPL")
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import feedparser

from src.cache.db import Database, Headline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feed URL templates
# ---------------------------------------------------------------------------

GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search"
    "?q={query}+stock+when:7d&hl=en-US&gl=US&ceid=US:en"
)

YAHOO_FINANCE_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

# User-agent to avoid being blocked by RSS endpoints
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SentimentBot/1.0)"}


class RSSFetcher:
    """
    Fetches headlines from multiple RSS sources for a given ticker and
    writes new (non-duplicate) entries to the SQLite cache.
    """

    def __init__(self, db: Database, sleep_between_requests: float = 1.0):
        self.db = db
        self.sleep = sleep_between_requests

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ticker(self, ticker: str, company_name: Optional[str] = None) -> int:
        """
        Fetch headlines for a single ticker from all configured RSS sources.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol, e.g. "AAPL"
        company_name : str, optional
            Company name used to broaden Google News search, e.g. "Apple Inc"

        Returns
        -------
        int
            Number of *new* headlines inserted (duplicates silently skipped).
        """
        inserted = 0
        query = company_name or ticker

        sources = [
            ("Yahoo Finance", self._fetch_yahoo(ticker)),
            ("Google News", self._fetch_google_news(query, ticker)),
        ]

        for source_name, headlines in sources:
            for h in headlines:
                row_id = self.db.insert_headline(h)
                if row_id is not None:
                    inserted += 1
                    logger.debug("Inserted [%s] %s", source_name, h.title[:80])
            time.sleep(self.sleep)

        logger.info("Fetched %s: %d new headlines inserted", ticker, inserted)
        return inserted

    def fetch_tickers(
        self, tickers: list[str], company_names: Optional[dict[str, str]] = None
    ) -> dict[str, int]:
        """Fetch headlines for multiple tickers. Returns {ticker: new_count}."""
        company_names = company_names or {}
        results = {}
        for ticker in tickers:
            results[ticker] = self.fetch_ticker(
                ticker, company_names.get(ticker)
            )
        return results

    # ------------------------------------------------------------------
    # Internal feed fetchers
    # ------------------------------------------------------------------

    def _fetch_yahoo(self, ticker: str) -> list[Headline]:
        url = YAHOO_FINANCE_RSS.format(ticker=ticker)
        return self._parse_feed(url, ticker, source="Yahoo Finance")

    def _fetch_google_news(self, query: str, ticker: str) -> list[Headline]:
        url = GOOGLE_NEWS_RSS.format(query=query.replace(" ", "+"))
        return self._parse_feed(url, ticker, source="Google News")

    def _parse_feed(self, url: str, ticker: str, source: str) -> list[Headline]:
        """Parse an RSS feed URL and return a list of Headline objects."""
        headlines: list[Headline] = []
        try:
            feed = feedparser.parse(url, request_headers=HEADERS)
            if feed.bozo:
                logger.warning("Malformed feed from %s: %s", source, url)

            for entry in feed.entries:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                if not title or not link:
                    continue

                # Normalize published timestamp to ISO-8601
                published = self._parse_published(entry)

                headlines.append(
                    Headline(
                        ticker=ticker,
                        title=title,
                        url=link,
                        source=source,
                        published=published,
                    )
                )
        except Exception as exc:
            logger.error("Failed to fetch %s feed for %s: %s", source, ticker, exc)

        return headlines

    @staticmethod
    def _parse_published(entry) -> str:
        """Extract and normalize the published timestamp from a feed entry."""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                return dt.isoformat()
            except Exception:
                pass
        # Fallback: use current UTC time
        return datetime.now(timezone.utc).isoformat()
