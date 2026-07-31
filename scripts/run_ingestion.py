"""
scripts/run_ingestion.py
-------------------------
Week 1 ingestion pipeline: fetch RSS → download EDGAR filings → chunk → score sentiment.

Run this locally (or via GitHub Actions on a schedule).
The Streamlit app never calls this — it only reads the DB.

Usage
-----
    # Score a default watchlist
    python scripts/run_ingestion.py

    # Score specific tickers
    python scripts/run_ingestion.py --tickers AAPL MSFT NVDA

    # RSS only (no EDGAR download)
    python scripts/run_ingestion.py --rss-only

    # EDGAR only
    python scripts/run_ingestion.py --edgar-only --tickers AAPL --filings 10-K --num 2
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache.db import Database
from src.ingestion.edgar_fetcher import EdgarFetcher
from src.ingestion.rss_fetcher import RSSFetcher
from src.sentiment.finbert_scorer import FinBERTScorer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingestion")

# ---------------------------------------------------------------------------
# Default watchlist across all supported markets
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST: dict[str, str] = {
    # NASDAQ
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Google",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    # NYSE
    "JPM": "JPMorgan Chase",
    "BRK-B": "Berkshire Hathaway",
    "WMT": "Walmart",
    "V": "Visa",
    "DIS": "Walt Disney",
    # Shanghai
    "600519.SS": "Kweichow Moutai",
    "601398.SS": "ICBC",
    "601857.SS": "PetroChina",
    # Euronext
    "MC.PA": "LVMH",
    "RMS.PA": "Hermes International",
    "AIR.PA": "Airbus",
    "ASML.AS": "ASML Holding",
    # NSE
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()

    watchlist = _build_watchlist(args)
    logger.info("Running ingestion for %d tickers: %s", len(watchlist), list(watchlist.keys()))

    db = Database()
    rss_fetcher = RSSFetcher(db)
    edgar_fetcher = EdgarFetcher(db)
    scorer = FinBERTScorer()

    for ticker, company in watchlist.items():
        logger.info("=" * 60)
        logger.info("Processing: %s (%s)", ticker, company)
        logger.info("=" * 60)

        # --- Step 1: RSS headlines ---
        if not args.edgar_only:
            logger.info("[1/3] Fetching RSS headlines for %s…", ticker)
            new_headlines = rss_fetcher.fetch_ticker(ticker, company)
            logger.info("  → %d new headlines", new_headlines)

        # --- Step 2: SEC EDGAR filings ---
        if not args.rss_only:
            logger.info("[2/3] Downloading EDGAR filings for %s…", ticker)
            filing_types = args.filings if args.filings else ["10-K", "10-Q"]
            edgar_results = edgar_fetcher.fetch_ticker(
                ticker,
                filing_types=filing_types,
                num_filings=args.num,
                sections_filter=["ITEM 1A", "ITEM 7", "ITEM 7A", "ITEM 8"],
            )
            for ftype, n_chunks in edgar_results.items():
                logger.info("  → %s: %d new chunks", ftype, n_chunks)

        # --- Step 3: FinBERT scoring ---
        if not args.skip_sentiment:
            logger.info("[3/3] Scoring unscored headlines via FinBERT for %s…", ticker)
            scored = scorer.score_unscored(db, ticker=ticker)
            logger.info("  → %d headlines scored", scored)

        # Be gentle — small pause between tickers
        time.sleep(0.5)

    logger.info("=" * 60)
    logger.info("Ingestion complete.")
    db.close()


def _build_watchlist(args) -> dict[str, str]:
    if args.tickers:
        return {t.upper(): DEFAULT_WATCHLIST.get(t.upper(), t.upper()) for t in args.tickers}
    if args.market:
        try:
            from app.streamlit_app import MARKETS
            m_watchlist = MARKETS.get(args.market, {})
            if m_watchlist:
                return m_watchlist
        except Exception:
            pass
    return DEFAULT_WATCHLIST


def _parse_args():
    parser = argparse.ArgumentParser(description="Financial Sentiment Ingestion Pipeline")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Ticker symbols to process (default: full watchlist)",
    )
    parser.add_argument(
        "--market",
        choices=["NASDAQ", "NYSE", "Shanghai", "Euronext", "NSE"],
        help="Target market to ingest all tickers for",
    )
    parser.add_argument(
        "--rss-only",
        action="store_true",
        help="Only fetch RSS headlines, skip EDGAR downloads",
    )
    parser.add_argument(
        "--edgar-only",
        action="store_true",
        help="Only download EDGAR filings, skip RSS",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip FinBERT scoring (useful when just testing ingestion)",
    )
    parser.add_argument(
        "--filings",
        nargs="+",
        choices=["10-K", "10-Q"],
        help="Filing types to download (default: 10-K and 10-Q)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of filings per type to download (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
