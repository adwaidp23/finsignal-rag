"""
scripts/test_dashboard_backend.py
---------------------------------
Test the exact calls made by the Streamlit dashboard tabs to locate
which backend function is raising an exception.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache.db import Database
from src.agents.synthesis_agent import SynthesisAgent
from src.backtest.backtester import Backtester

print("="*60)
print("Testing Streamlit Dashboard Backend Code")
print("="*60)

db = Database()

# Ticker to test
ticker = "AAPL"
print(f"Testing ticker: {ticker}\n")

# --- 1. Test Filings (Filing Context retrieval) ---
print("[1] Testing SEC Filing retrieval...")
try:
    chunks = db.get_filing_chunks(ticker)
    print(f"  ✓ Success: Found {len(chunks)} chunks for {ticker}")
    if chunks:
        # Check grouping logic
        filings_seen = set()
        options = []
        for row in chunks:
            key = f"{row['filing_type']} — {row['fiscal_period']} ({row['filed_date'][:10]})"
            if key not in filings_seen:
                filings_seen.add(key)
                options.append(key)
        print(f"  ✓ Grouping logic succeeded. Filings: {options[:5]}")
except Exception as e:
    print(f"  ✗ Error in Filing Context:")
    import traceback
    traceback.print_exc()

print()

# --- 2. Test Synthesis (Analysis Tab) ---
print("[2] Testing Synthesis Agent...")
try:
    agent = SynthesisAgent()
    # Pull sentiment summary first
    from datetime import date, timedelta
    since = (date.today() - timedelta(days=7)).isoformat()
    sentiment_summary = db.get_sentiment_summary(ticker, since=since)
    print(f"  ✓ Sentiment summary for {ticker}: {sentiment_summary}")
    
    # Run synthesis from DB
    report = agent.synthesize_from_db(ticker, days_back=7)
    print(f"  ✓ Synthesis report returned keys: {list(report.keys())}")
    if "error" in report:
        print(f"  ⚠️ Agent returned error field: {report['error']}")
except Exception as e:
    print(f"  ✗ Error in Synthesis:")
    import traceback
    traceback.print_exc()

print()

# --- 3. Test Backtesting ---
print("[3] Testing Backtest Engine...")
try:
    backtester = Backtester(db)
    df, correlation = backtester.run_backtest(ticker, days_back=45, forward_days=5)
    print(f"  ✓ Backtest run succeeded. Dataframe shape: {df.shape}, correlation: {correlation:.4f}")
except Exception as e:
    print(f"  ✗ Error in Backtester:")
    import traceback
    traceback.print_exc()

db.close()
print("="*60)
