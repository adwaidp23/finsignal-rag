"""
scripts/smoke_test.py
---------------------
Verifies that Week 1 components are importable and the DB schema
creates correctly. Run this after `pip install -r requirements.txt`.

Usage
-----
    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print("=" * 55)
print("Week 1 Smoke Test")
print("=" * 55)

errors = []

# --- 1. Core imports ---
print("\n[1] Testing core imports…")
try:
    from src.utils.secrets import get_secret
    from src.utils.config import GROQ_MODEL_FAST, FINBERT_MODEL, DB_PATH
    print(f"  ✓ secrets + config (DB_PATH={DB_PATH})")
except Exception as e:
    errors.append(f"Core imports: {e}")
    print(f"  ✗ {e}")

# --- 2. SQLite cache ---
print("\n[2] Testing SQLite cache (schema creation)…")
try:
    import tempfile
    from pathlib import Path as P
    from src.cache.db import Database, Headline, SentimentScore, FilingChunk

    with tempfile.TemporaryDirectory() as tmp:
        db = Database(P(tmp) / "test.db")

        # Insert a test headline
        h = Headline(
            ticker="TEST",
            title="Test headline: earnings beat expectations",
            url="https://example.com/test",
            source="TestFeed",
            published="2026-07-17T00:00:00+00:00",
        )
        hid = db.insert_headline(h)
        assert hid is not None, "insert_headline returned None"

        # Duplicate should be silently skipped
        dup_id = db.insert_headline(h)
        assert dup_id is None, "Duplicate headline was not skipped"

        # Insert sentiment
        db.insert_sentiment(SentimentScore(hid, "positive", 0.92, "ProsusAI/finbert"))

        # Read it back
        rows = db.get_unscored_headlines("TEST", "ProsusAI/finbert")
        assert len(rows) == 0, f"Expected 0 unscored, got {len(rows)}"

        summary = db.get_sentiment_summary("TEST")
        assert "positive" in summary
        assert summary["positive"]["count"] == 1

        db.close()
    print("  ✓ Database schema, insert, dedup, sentiment read/write")
except Exception as e:
    errors.append(f"SQLite cache: {e}")
    print(f"  ✗ {e}")

# --- 3. Chunker ---
print("\n[3] Testing chunker…")
try:
    from src.ingestion.chunker import chunk_filing, _split_by_item_header

    sample_text = """
Cover page preamble text here.

ITEM 1. Business

Apple Inc. designs, manufactures and markets smartphones.

ITEM 1A. Risk Factors

The company faces significant competition from Android manufacturers.
Supply chain disruptions may impact production. Regulatory changes in
multiple jurisdictions represent ongoing risk. These are material risks.

ITEM 7. MD&A

Revenue increased 12% year over year driven by iPhone sales.
Services segment grew 15% to $24.2 billion.
""" * 3  # multiply to exceed chunk_size

    sections = _split_by_item_header(sample_text)
    assert "ITEM 1" in sections, f"Missing ITEM 1, got keys: {list(sections.keys())}"
    assert "ITEM 1A" in sections, f"Missing ITEM 1A"
    assert "PREAMBLE" in sections, "Preamble should be captured"

    chunks = chunk_filing(
        text=sample_text,
        ticker="AAPL",
        filing_type="10-K",
        fiscal_period="2025-Q4",
        filed_date="2025-11-01",
    )
    assert len(chunks) > 0, "No chunks produced"
    assert all(c.ticker == "AAPL" for c in chunks)
    assert all(c.section for c in chunks)
    print(f"  ✓ chunk_filing produced {len(chunks)} chunks from {len(sections)} sections")
except Exception as e:
    errors.append(f"Chunker: {e}")
    print(f"  ✗ {e}")

# --- 4. RSS Fetcher (import only — don't hit network) ---
print("\n[4] Testing RSS fetcher import…")
try:
    from src.ingestion.rss_fetcher import RSSFetcher
    import feedparser
    print(f"  ✓ RSSFetcher + feedparser v{feedparser.__version__}")
except Exception as e:
    errors.append(f"RSS fetcher: {e}")
    print(f"  ✗ {e}")

# --- 5. EDGAR fetcher (import only) ---
print("\n[5] Testing EDGAR fetcher import…")
try:
    from src.ingestion.edgar_fetcher import EdgarFetcher
    print("  ✓ EdgarFetcher")
except Exception as e:
    errors.append(f"EDGAR fetcher: {e}")
    print(f"  ✗ {e}")

# --- 6. FinBERT scorer (import only — don't load model weights) ---
print("\n[6] Testing FinBERT scorer import…")
try:
    from src.sentiment.finbert_scorer import FinBERTScorer
    scorer = FinBERTScorer()
    assert scorer._pipeline is None, "Pipeline should be lazy (not loaded yet)"
    print("  ✓ FinBERTScorer (lazy load confirmed — model not yet downloaded)")
except Exception as e:
    errors.append(f"FinBERT scorer: {e}")
    print(f"  ✗ {e}")

# --- Summary ---
print("\n" + "=" * 55)
if errors:
    print(f"FAILED — {len(errors)} error(s):")
    for err in errors:
        print(f"  ✗ {err}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED ✓")
    print("Week 1 scaffold is ready.")
    print("\nNext step: python scripts/run_ingestion.py --rss-only --tickers AAPL MSFT")
print("=" * 55)
