"""
src/ingestion/edgar_fetcher.py
------------------------------
Download and parse SEC 10-K / 10-Q filings for a given ticker using the
SEC EDGAR full-text search API and sec-edgar-downloader.

Key design decisions:
  - Raw HTML is treated as scratch space (ephemeral) — the parsed plain text
    is written immediately to SQLite, so the raw file is disposable.
  - Skips filings already in the DB (idempotent re-runs).
  - Respects SEC fair-use rate limit (~10 req/sec) via built-in sleeps.

Usage
-----
    from src.ingestion.edgar_fetcher import EdgarFetcher
    from src.cache.db import Database

    db = Database()
    fetcher = EdgarFetcher(db)
    fetcher.fetch_ticker("AAPL", filing_types=["10-K"], num_filings=3)
"""

import logging
import re
import time
from pathlib import Path

from src.cache.db import Database
from src.ingestion.chunker import chunk_filing

logger = logging.getLogger(__name__)

# SEC requires a proper User-Agent with contact email
SEC_USER_AGENT = "SentimentProject research@example.com"

# Raw filings download directory — intentionally ephemeral / gitignored
RAW_FILINGS_DIR = Path(__file__).resolve().parents[2] / "data" / "sec_filings"

# Seconds between EDGAR HTTP requests (respect ~10 req/sec guideline)
SEC_SLEEP = 0.15


NON_US_SUFFIXES = (".SS", ".PA", ".AS", ".BR", ".NS", ".L", ".HK", ".TW", ".KS", ".BO", ".TO", ".DE")


def is_non_us_ticker(ticker: str) -> bool:
    return any(ticker.upper().endswith(suffix) for suffix in NON_US_SUFFIXES)


class EdgarFetcher:
    """
    Downloads filings via sec-edgar-downloader, parses HTML → plain text,
    chunks the text, and writes everything to SQLite.
    """

    def __init__(self, db: Database, raw_dir: Path = RAW_FILINGS_DIR):
        self.db = db
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ticker(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        num_filings: int = 5,
        sections_filter: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Download, parse, chunk, and cache filings for a ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker, e.g. "AAPL".
        filing_types : list[str]
            Filing types to fetch. Defaults to ["10-K", "10-Q"].
        num_filings : int
            Maximum number of filings per type to download.
        sections_filter : list[str], optional
            If set, only chunk these ITEM sections (e.g. ["ITEM 1A", "ITEM 7"]).

        Returns
        -------
        dict
            {filing_type: number_of_new_chunks_inserted}
        """
        filing_types = filing_types or ["10-K", "10-Q"]

        if is_non_us_ticker(ticker):
            logger.info("Skipping SEC EDGAR download for non-US ticker %s (SEC filings are US-only)", ticker)
            return {ftype: 0 for ftype in filing_types}

        results: dict[str, int] = {}

        for filing_type in filing_types:
            new_chunks = self._process_filing_type(
                ticker, filing_type, num_filings, sections_filter
            )
            results[filing_type] = new_chunks
            time.sleep(SEC_SLEEP)

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_filing_type(
        self,
        ticker: str,
        filing_type: str,
        num_filings: int,
        sections_filter: list[str] | None,
    ) -> int:
        """Download, parse, and insert filings. Returns new chunk count."""
        try:
            from sec_edgar_downloader import Downloader
        except ImportError:
            logger.error(
                "sec-edgar-downloader not installed. Run: pip install sec-edgar-downloader"
            )
            return 0

        dl = Downloader("SentimentProject", "research@example.com", str(self.raw_dir))

        try:
            dl.get(filing_type, ticker, limit=num_filings, download_details=True)
        except Exception as exc:
            logger.error("EDGAR download failed for %s %s: %s", ticker, filing_type, exc)
            return 0

        # Walk the downloaded files
        ticker_dir = self.raw_dir / "sec-edgar-filings" / ticker / filing_type
        if not ticker_dir.exists():
            logger.warning("No files found at %s", ticker_dir)
            return 0

        total_new = 0

        for filing_dir in sorted(ticker_dir.iterdir()):
            if not filing_dir.is_dir():
                continue

            fiscal_period, filed_date = self._extract_period_and_date(filing_dir)

            if self.db.has_filing(ticker, filing_type, fiscal_period):
                logger.info(
                    "Skipping %s %s %s — already in DB",
                    ticker,
                    filing_type,
                    fiscal_period,
                )
                continue

            # Find the primary HTML document
            html_file = self._find_primary_html(filing_dir)
            if not html_file:
                logger.warning("No HTML file found in %s", filing_dir)
                continue

            text = self._html_to_text(html_file.read_text(encoding="utf-8", errors="ignore"))
            if len(text) < 500:
                logger.warning("Filing text suspiciously short (%d chars): %s", len(text), html_file)
                continue

            chunks = chunk_filing(
                text=text,
                ticker=ticker,
                filing_type=filing_type,
                fiscal_period=fiscal_period,
                filed_date=filed_date,
                sections_filter=sections_filter,
            )

            for chunk in chunks:
                self.db.insert_filing_chunk(chunk)

            total_new += len(chunks)
            logger.info(
                "Processed %s %s %s → %d chunks cached",
                ticker,
                filing_type,
                fiscal_period,
                len(chunks),
            )

        return total_new

    @staticmethod
    def _extract_period_and_date(filing_dir: Path) -> tuple[str, str]:
        """
        Extract fiscal period and filed date from the filing directory name.
        sec-edgar-downloader names dirs like: 0000320193-25-000085 (accession number)
        We fall back to the directory mtime for date.
        """
        # Try to read primary-document metadata file if present
        meta_file = filing_dir / "filing-details.json"
        if meta_file.exists():
            import json

            try:
                meta = json.loads(meta_file.read_text())
                filed_date = meta.get("filedAt", "")[:10]  # YYYY-MM-DD
                period = meta.get("periodOfReport", filed_date)[:7]  # YYYY-MM
                # Map to quarter
                fiscal_period = _to_quarter(period)
                return fiscal_period, filed_date
            except Exception:
                pass

        # Fallback: use directory name as identifier
        dir_name = filing_dir.name
        import datetime
        filed_date = datetime.date.today().isoformat()
        return dir_name, filed_date

    @staticmethod
    def _find_primary_html(filing_dir: Path) -> Path | None:
        """Return the largest HTML file in the filing directory (usually the main doc)."""
        html_files = list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html"))
        if not html_files:
            return None
        return max(html_files, key=lambda f: f.stat().st_size)

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Strip HTML tags and normalize whitespace for plain-text parsing."""
        # Remove script/style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Remove all remaining tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Decode common HTML entities
        text = (
            text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&#160;", " ")
        )
        # Collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _to_quarter(year_month: str) -> str:
    """Convert 'YYYY-MM' to 'YYYY-QN'. E.g. '2025-11' → '2025-Q4'."""
    try:
        year, month = year_month.split("-")
        q = (int(month) - 1) // 3 + 1
        return f"{year}-Q{q}"
    except Exception:
        return year_month
