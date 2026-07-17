"""
src/sentiment/finbert_scorer.py
--------------------------------
Pre-compute FinBERT sentiment scores for unscored headlines and write
results to SQLite.

Design: runs at ingestion time (NOT inside the Streamlit app), so the
deployed app never loads the model — it only reads pre-computed scores.
This keeps free-tier CPU hosts responsive.

Model: ProsusAI/finbert (downloaded once by HuggingFace, cached locally)

Usage
-----
    from src.sentiment.finbert_scorer import FinBERTScorer
    from src.cache.db import Database

    db = Database()
    scorer = FinBERTScorer()
    scored = scorer.score_unscored(db, ticker="AAPL")
    print(f"Scored {scored} headlines")
"""

import logging
from typing import Optional

from src.cache.db import Database, SentimentScore

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 512
BATCH_SIZE = 16  # safe for CPU; reduce if OOM


class FinBERTScorer:
    """
    Lazy-loads FinBERT on first use. The model is downloaded once by
    HuggingFace and cached in ~/.cache/huggingface/.
    """

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self._pipeline = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_unscored(self, db: Database, ticker: Optional[str] = None) -> int:
        """
        Score all headlines in the DB that don't yet have a sentiment entry.

        Parameters
        ----------
        db : Database
        ticker : str, optional
            If provided, only score headlines for this ticker.
            If None, scores across all tickers (useful for batch night runs).

        Returns
        -------
        int
            Number of headlines scored.
        """
        tickers = [ticker] if ticker else self._all_tickers(db)
        total = 0

        for t in tickers:
            rows = db.get_unscored_headlines(t, self.model_name)
            if not rows:
                logger.info("%s: no unscored headlines", t)
                continue

            logger.info("%s: scoring %d headlines via FinBERT…", t, len(rows))
            ids = [r["id"] for r in rows]
            texts = [r["title"] for r in rows]

            results = self._run_batch(texts)

            for headline_id, result in zip(ids, results):
                db.insert_sentiment(
                    SentimentScore(
                        headline_id=headline_id,
                        label=result["label"].lower(),   # "positive"|"negative"|"neutral"
                        score=round(result["score"], 4),
                        model_version=self.model_name,
                    )
                )

            total += len(ids)
            logger.info("%s: wrote %d sentiment scores", t, len(ids))

        return total

    def score_texts(self, texts: list[str]) -> list[dict]:
        """
        Score arbitrary texts (not tied to DB). Returns list of
        {"label": str, "score": float} dicts.

        Useful for quick testing.
        """
        return self._run_batch(texts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_pipeline(self):
        """Lazy-load the HuggingFace pipeline on first call."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline

            logger.info("Loading FinBERT model (%s)…", self.model_name)
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=MAX_LENGTH,
                device=-1,  # CPU — change to 0 for GPU if available
            )
            logger.info("FinBERT loaded successfully")
        except Exception as exc:
            logger.error("Failed to load FinBERT: %s", exc)
            raise

    def _run_batch(self, texts: list[str]) -> list[dict]:
        """Run inference in batches. Returns list of {label, score}."""
        self._load_pipeline()
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                batch_results = self._pipeline(batch)
                results.extend(batch_results)
            except Exception as exc:
                logger.error("FinBERT batch %d failed: %s", i, exc)
                # Fill failed batch with neutral scores rather than crashing
                results.extend(
                    [{"label": "neutral", "score": 0.0}] * len(batch)
                )

        return results

    @staticmethod
    def _all_tickers(db: Database) -> list[str]:
        rows = db.conn.execute(
            "SELECT DISTINCT ticker FROM headlines ORDER BY ticker"
        ).fetchall()
        return [r["ticker"] for r in rows]
