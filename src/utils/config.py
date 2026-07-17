"""
src/utils/config.py
-------------------
Central configuration — all tuneable values in one place.
Override via environment variables or by editing this file.
"""

import os
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[2]

# ── Models ──────────────────────────────────────────────────────────────────
# Keep model names in config — not hardcoded in modules —
# so you can swap quickly if Groq deprecates a model.
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")   # router
GROQ_MODEL_HEAVY = os.getenv("GROQ_MODEL_HEAVY", "llama-3.3-70b-versatile")  # synthesis

# ── Paths ────────────────────────────────────────────────────────────────────
DB_PATH = ROOT / "data" / "sentiment.db"
FAISS_INDEX_PATH = ROOT / "data" / "faiss_index"
RAW_FILINGS_PATH = ROOT / "data" / "sec_filings"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))  # top-k chunks to retrieve

# ── Rate limiting ─────────────────────────────────────────────────────────────
RSS_SLEEP_SECONDS = float(os.getenv("RSS_SLEEP_SECONDS", "1.0"))
SEC_SLEEP_SECONDS = float(os.getenv("SEC_SLEEP_SECONDS", "0.15"))
YFINANCE_SLEEP_SECONDS = float(os.getenv("YFINANCE_SLEEP_SECONDS", "0.5"))

# ── FinBERT ──────────────────────────────────────────────────────────────────
FINBERT_BATCH_SIZE = int(os.getenv("FINBERT_BATCH_SIZE", "16"))
FINBERT_MAX_LENGTH = int(os.getenv("FINBERT_MAX_LENGTH", "512"))

# ── Groq ─────────────────────────────────────────────────────────────────────
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
