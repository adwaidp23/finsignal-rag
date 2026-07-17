"""
scripts/build_index.py
-----------------------
One-time local script to build the FAISS vector index from SQLite filing chunks.

Run this manually after running the ingestion pipeline.
The resulting index files are committed to git so deployed apps never rebuild.

Usage
-----
    # Build index for all tickers in the DB
    python scripts/build_index.py

    # Build for specific tickers only
    python scripts/build_index.py --tickers AAPL MSFT NVDA

    # Force rebuild even if index already exists
    python scripts/build_index.py --force

    # Build only 10-K filings
    python scripts/build_index.py --filing-type 10-K

After building, commit the index:
    git add data/faiss_index/
    git commit -m "rebuild FAISS index - <date> - <tickers>"

NOTE: If data/faiss_index/ grows beyond ~50 MB, switch to Git LFS:
    git lfs track "data/faiss_index/*.faiss"
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.faiss_builder import FAISSBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_index")


def main():
    args = _parse_args()

    logger.info("=" * 55)
    logger.info("FAISS Index Builder")
    logger.info("=" * 55)

    builder = FAISSBuilder()

    tickers = args.tickers if args.tickers else None
    logger.info(
        "Building index — tickers=%s, filing_type=%s, force=%s",
        tickers or "ALL",
        args.filing_type or "ALL",
        args.force,
    )

    builder.build(
        tickers=tickers,
        filing_type=args.filing_type,
        force_rebuild=args.force,
    )

    # Print stats if index exists
    try:
        stats = builder.index_stats()
        logger.info("=" * 55)
        logger.info("Index stats:")
        logger.info("  Total vectors : %d", stats["total_vectors"])
        logger.info("  Embedding model: %s", stats["embedding_model"])
        logger.info("  Index path    : %s", stats["index_path"])
        logger.info("=" * 55)
        logger.info("Done. Commit the index with:")
        logger.info("  git add data/faiss_index/")
        logger.info("  git commit -m 'rebuild index'")
    except Exception as e:
        logger.warning("Could not read index stats: %s", e)


def _parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index from SQLite chunks")
    parser.add_argument(
        "--tickers", nargs="+", help="Tickers to index (default: all in DB)"
    )
    parser.add_argument(
        "--filing-type",
        choices=["10-K", "10-Q"],
        help="Filter to a specific filing type",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if index already exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
