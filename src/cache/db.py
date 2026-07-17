"""
src/cache/db.py
---------------
SQLite cache layer for sentiment scores, parsed filing chunks,
and Groq LLM responses.

Schema
------
headlines     – raw headlines + RSS metadata per ticker/date
sentiment     – FinBERT scores keyed by (headline_id, model_version)
filing_chunks – parsed + chunked SEC filing text with metadata
llm_cache     – Groq synthesis responses keyed by (ticker, date, prompt_hash)

Usage
-----
    from src.cache.db import Database

    db = Database()                          # opens / creates data/sentiment.db
    db.insert_headline(...)
    db.get_sentiment(ticker="AAPL", date="2026-07-17")
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

# Default DB path — relative to project root
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sentiment.db"


@dataclass
class Headline:
    ticker: str
    title: str
    url: str
    source: str
    published: str  # ISO-8601 string
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SentimentScore:
    headline_id: int
    label: str          # "positive" | "negative" | "neutral"
    score: float        # confidence [0, 1]
    model_version: str  # e.g. "ProsusAI/finbert"


@dataclass
class FilingChunk:
    ticker: str
    filing_type: str    # "10-K" | "10-Q"
    fiscal_period: str  # e.g. "2025-Q4"
    filed_date: str     # ISO-8601
    section: str        # e.g. "ITEM 1A"
    chunk_index: int
    text: str


@dataclass
class LLMCacheEntry:
    ticker: str
    analysis_date: str
    prompt_hash: str
    response: str
    model: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class Database:
    """Thin wrapper around sqlite3 with typed insert/query helpers."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        self.conn.executescript("""
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS headlines (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                title       TEXT    NOT NULL,
                url         TEXT    UNIQUE,
                source      TEXT,
                published   TEXT,
                fetched_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sentiment (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                headline_id     INTEGER NOT NULL REFERENCES headlines(id),
                label           TEXT    NOT NULL,
                score           REAL    NOT NULL,
                model_version   TEXT    NOT NULL,
                UNIQUE(headline_id, model_version)
            );

            CREATE TABLE IF NOT EXISTS filing_chunks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                filing_type     TEXT    NOT NULL,
                fiscal_period   TEXT    NOT NULL,
                filed_date      TEXT    NOT NULL,
                section         TEXT    NOT NULL,
                chunk_index     INTEGER NOT NULL,
                text            TEXT    NOT NULL,
                UNIQUE(ticker, filing_type, fiscal_period, section, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS llm_cache (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                analysis_date   TEXT    NOT NULL,
                prompt_hash     TEXT    NOT NULL,
                response        TEXT    NOT NULL,
                model           TEXT    NOT NULL,
                created_at      TEXT    NOT NULL,
                UNIQUE(ticker, analysis_date, prompt_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_headlines_ticker_pub
                ON headlines(ticker, published);
            CREATE INDEX IF NOT EXISTS idx_chunks_ticker
                ON filing_chunks(ticker, filing_type, fiscal_period);
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Headlines
    # ------------------------------------------------------------------

    def insert_headline(self, h: Headline) -> Optional[int]:
        """Insert a headline; return its rowid or None if duplicate (by URL)."""
        try:
            cur = self.conn.execute(
                """
                INSERT INTO headlines (ticker, title, url, source, published, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (h.ticker, h.title, h.url, h.source, h.published, h.fetched_at),
            )
            self.conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            # Duplicate URL — already ingested
            return None

    def get_unscored_headlines(
        self, ticker: str, model_version: str
    ) -> list[sqlite3.Row]:
        """Return headlines for a ticker that don't yet have a sentiment score."""
        return self.conn.execute(
            """
            SELECT h.id, h.title FROM headlines h
            LEFT JOIN sentiment s
                ON s.headline_id = h.id AND s.model_version = ?
            WHERE h.ticker = ? AND s.id IS NULL
            ORDER BY h.published DESC
            """,
            (model_version, ticker),
        ).fetchall()

    def get_headlines_with_sentiment(
        self, ticker: str, since: Optional[str] = None
    ) -> list[sqlite3.Row]:
        """Return headlines + latest sentiment scores for a ticker."""
        query = """
            SELECT h.title, h.url, h.published, h.source,
                   s.label, s.score
            FROM headlines h
            JOIN sentiment s ON s.headline_id = h.id
            WHERE h.ticker = ?
        """
        params: list[Any] = [ticker]
        if since:
            query += " AND h.published >= ?"
            params.append(since)
        query += " ORDER BY h.published DESC"
        return self.conn.execute(query, params).fetchall()

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    def insert_sentiment(self, s: SentimentScore) -> None:
        """Insert or replace a sentiment score (upsert on unique constraint)."""
        self.conn.execute(
            """
            INSERT INTO sentiment (headline_id, label, score, model_version)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(headline_id, model_version) DO UPDATE SET
                label = excluded.label,
                score = excluded.score
            """,
            (s.headline_id, s.label, s.score, s.model_version),
        )
        self.conn.commit()

    def get_sentiment_summary(self, ticker: str, since: Optional[str] = None) -> dict:
        """Return aggregated sentiment counts and mean score for a ticker."""
        query = """
            SELECT s.label, COUNT(*) as cnt, AVG(s.score) as avg_score
            FROM sentiment s
            JOIN headlines h ON h.id = s.headline_id
            WHERE h.ticker = ?
        """
        params: list[Any] = [ticker]
        if since:
            query += " AND h.published >= ?"
            params.append(since)
        query += " GROUP BY s.label"
        rows = self.conn.execute(query, params).fetchall()
        return {r["label"]: {"count": r["cnt"], "avg_score": r["avg_score"]} for r in rows}

    # ------------------------------------------------------------------
    # Filing chunks
    # ------------------------------------------------------------------

    def insert_filing_chunk(self, fc: FilingChunk) -> None:
        """Insert a filing chunk (ignore duplicates)."""
        self.conn.execute(
            """
            INSERT OR IGNORE INTO filing_chunks
                (ticker, filing_type, fiscal_period, filed_date, section, chunk_index, text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fc.ticker,
                fc.filing_type,
                fc.fiscal_period,
                fc.filed_date,
                fc.section,
                fc.chunk_index,
                fc.text,
            ),
        )
        self.conn.commit()

    def get_filing_chunks(
        self,
        ticker: str,
        filing_type: Optional[str] = None,
        fiscal_period: Optional[str] = None,
    ) -> list[sqlite3.Row]:
        """Retrieve filing chunks, optionally filtered by type and period."""
        query = "SELECT * FROM filing_chunks WHERE ticker = ?"
        params: list[Any] = [ticker]
        if filing_type:
            query += " AND filing_type = ?"
            params.append(filing_type)
        if fiscal_period:
            query += " AND fiscal_period = ?"
            params.append(fiscal_period)
        query += " ORDER BY filed_date DESC, section, chunk_index"
        return self.conn.execute(query, params).fetchall()

    def has_filing(self, ticker: str, filing_type: str, fiscal_period: str) -> bool:
        """Check if we've already ingested a filing to avoid duplicate work."""
        row = self.conn.execute(
            """
            SELECT 1 FROM filing_chunks
            WHERE ticker = ? AND filing_type = ? AND fiscal_period = ?
            LIMIT 1
            """,
            (ticker, filing_type, fiscal_period),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # LLM cache
    # ------------------------------------------------------------------

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def get_llm_response(
        self, ticker: str, analysis_date: str, prompt: str
    ) -> Optional[str]:
        """Return a cached LLM response if it exists for (ticker, date, prompt)."""
        h = self.hash_prompt(prompt)
        row = self.conn.execute(
            """
            SELECT response FROM llm_cache
            WHERE ticker = ? AND analysis_date = ? AND prompt_hash = ?
            """,
            (ticker, analysis_date, h),
        ).fetchone()
        return row["response"] if row else None

    def set_llm_response(self, entry: LLMCacheEntry) -> None:
        """Cache an LLM response (upsert)."""
        self.conn.execute(
            """
            INSERT INTO llm_cache
                (ticker, analysis_date, prompt_hash, response, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, analysis_date, prompt_hash) DO UPDATE SET
                response = excluded.response,
                model    = excluded.model
            """,
            (
                entry.ticker,
                entry.analysis_date,
                entry.prompt_hash,
                entry.response,
                entry.model,
                entry.created_at,
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
