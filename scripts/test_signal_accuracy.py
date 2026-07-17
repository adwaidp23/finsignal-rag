"""
scripts/test_signal_accuracy.py
-------------------------------
Runs the backtest engine across all available tickers in the database
to print the Pearson correlation signal accuracy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache.db import Database
from src.backtest.backtester import Backtester

print("="*60)
print("Evaluating Trading Signal Accuracy (Sentiment vs Returns)")
print("="*60)

db = Database()
backtester = Backtester(db)

# Get unique tickers from headlines
tickers = [row['ticker'] for row in db.conn.execute("SELECT DISTINCT ticker FROM headlines").fetchall()]

print(f"Found {len(tickers)} tickers in database: {tickers}\n")

overall_corr = 0.0
valid_tickers = 0

for ticker in tickers:
    try:
        # Run 60-day lookback, 5-day forward return window
        df, correlation = backtester.run_backtest(ticker, days_back=60, forward_days=5)
        if not df.empty:
            print(f"[{ticker}] Backtest successful. Datapoints: {len(df)} | Pearson Correlation: {correlation:+.4f}")
            overall_corr += correlation
            valid_tickers += 1
        else:
            print(f"[{ticker}] Insufficient data for backtest correlation.")
    except Exception as e:
        print(f"[{ticker}] Backtest failed: {e}")

if valid_tickers > 0:
    print("-" * 60)
    print(f"Average Correlation (Accuracy Signal): {overall_corr / valid_tickers:+.4f}")
    if abs(overall_corr / valid_tickers) > 0.1:
        print("Verdict: ADEQUATE SIGNAL (Strong enough for directional leaning)")
    else:
        print("Verdict: WEAK SIGNAL (Further tuning/data history required)")

print("="*60)
