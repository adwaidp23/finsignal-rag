"""
src/backtest/backtester.py
--------------------------
Calculates historical correlation between news sentiment signals
and stock returns using free yfinance data.

Key Design:
  - Fetches stock price data via yfinance (safely rate-limited).
  - Groups sentiment scores from SQLite by date.
  - Aligns sentiment signals with N-day forward returns.
  - Computes correlation metrics (Pearson/Spearman) to evaluate signal utility.
  - Outputs data suitable for Plotly visualization in Streamlit.
"""

import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
import yfinance as yf
from typing import Optional, Tuple
from src.cache.db import Database
from src.utils.config import DB_PATH, YFINANCE_SLEEP_SECONDS

logger = logging.getLogger(__name__)

class Backtester:
    """
    Quantifies correlation between sentiment scores and subsequent asset returns.
    """
    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database(DB_PATH)

    def get_ticker_history(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical close prices from yfinance safely.
        """
        logger.info(f"Fetching yfinance data for {ticker} from {start_date} to {end_date}")
        time.sleep(YFINANCE_SLEEP_SECONDS)
        
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date)
            if df.empty:
                logger.warning(f"No price history returned for {ticker}")
                return pd.DataFrame(columns=["Close"])
            return df[["Close"]]
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {ticker}: {e}")
            return pd.DataFrame(columns=["Close"])

    def calculate_sentiment_signals(self, ticker: str, start_date: str) -> pd.DataFrame:
        """
        Pulls sentiment scores from SQLite and aggregates them daily.
        Assigns weights: positive = +1, negative = -1, neutral = 0.
        """
        query = """
            SELECT date(h.published) as date, s.label, s.score
            FROM headlines h
            JOIN sentiment s ON s.headline_id = h.id
            WHERE h.ticker = ? AND h.published >= ?
        """
        rows = self.db.conn.execute(query, (ticker, start_date)).fetchall()
        if not rows:
            return pd.DataFrame(columns=["date", "sentiment_score"])

        df = pd.DataFrame([dict(r) for r in rows])
        df['weight'] = df['label'].map({'positive': 1.0, 'negative': -1.0, 'neutral': 0.0})
        df['weighted_score'] = df['weight'] * df['score']
        
        # Aggregate daily mean weighted score
        daily_sent = df.groupby('date')['weighted_score'].mean().reset_index()
        daily_sent = daily_sent.rename(columns={'weighted_score': 'sentiment_score'})
        daily_sent['date'] = pd.to_datetime(daily_sent['date'])
        return daily_sent

    def run_backtest(self, ticker: str, days_back: int = 30, forward_days: int = 5) -> Tuple[pd.DataFrame, float]:
        """
        Correlates daily sentiment scores against N-day forward stock returns.
        """
        start_dt = datetime.now() - timedelta(days=days_back)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        # 1. Get daily aggregated sentiment
        sentiment_df = self.calculate_sentiment_signals(ticker, start_date)
        if sentiment_df.empty:
            logger.warning(f"No sentiment data found to run backtest for {ticker}")
            return pd.DataFrame(), 0.0

        # 2. Get price history
        # Fetch slightly more history to calculate forward returns at the boundaries
        price_start = (start_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        price_end = (datetime.now() + timedelta(days=forward_days + 2)).strftime("%Y-%m-%d")
        price_df = self.get_ticker_history(ticker, price_start, price_end)
        
        if price_df.empty:
            return pd.DataFrame(), 0.0

        # Reset index to make Date a column
        price_df = price_df.reset_index()
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.tz_localize(None)

        # 3. Calculate forward returns: (Price_future - Price_current) / Price_current
        price_df['forward_close'] = price_df['Close'].shift(-forward_days)
        price_df['forward_return'] = (price_df['forward_close'] - price_df['Close']) / price_df['Close']

        # 4. Merge sentiment and returns on date
        merged = pd.merge(sentiment_df, price_df, left_on='date', right_on='Date', how='inner')
        if merged.empty:
            logger.warning(f"No overlapping dates between sentiment and price data for {ticker}")
            return pd.DataFrame(), 0.0

        # 5. Compute Pearson correlation coefficient
        correlation = float(merged['sentiment_score'].corr(merged['forward_return']))
        if pd.isna(correlation):
            correlation = 0.0

        return merged[['date', 'sentiment_score', 'Close', 'forward_return']], correlation
