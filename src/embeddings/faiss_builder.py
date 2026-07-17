"""
src/embeddings/faiss_builder.py
--------------------------------
Builds a FAISS vector index from filing chunks stored in SQLite,
using BGE-small-en-v1.5 embeddings (local, no API calls).

Design decisions:
  - Loads chunks from SQLite (already parsed + chunked in Week 1).
  - Attaches full metadata to each LangChain Document so the retriever
    can filter by ticker, filing_type, fiscal_period before similarity search.
  - Saves the index to data/faiss_index/ for committing to git.
  - On app startup, load_index() is called instead of rebuild — fast and
    deployment-safe on ephemeral cloud hosts.

Usage
-----
    from src.embeddings.faiss_builder import FAISSBuilder

    builder = FAISSBuilder()
    builder.build(tickers=["AAPL", "MSFT"])   # writes data/faiss_index/
    vectorstore = builder.load_index()
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.cache.db import Database
from src.utils.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    DB_PATH,
    RETRIEVER_K,
)

logger = logging.getLogger(__name__)


class FAISSBuilder:
    """
    Builds and loads a FAISS index from the SQLite filing_chunks table.
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        index_path: Path = FAISS_INDEX_PATH,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.db = db or Database(DB_PATH)
        self.index_path = index_path
        self.embedding_model = embedding_model
        self._embeddings = None   # lazy load

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        tickers: Optional[list[str]] = None,
        filing_type: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> None:
        """
        Build the FAISS index from SQLite chunks and save to disk.

        Parameters
        ----------
        tickers : list[str], optional
            Only index chunks for these tickers. None = all tickers in DB.
        filing_type : str, optional
            Filter to "10-K" or "10-Q". None = all types.
        force_rebuild : bool
            If False (default) and index already exists, skip rebuild.
        """
        if not force_rebuild and self._index_exists():
            logger.info("FAISS index already exists at %s — skipping rebuild. "
                        "Pass force_rebuild=True to override.", self.index_path)
            return

        documents = self._load_documents(tickers, filing_type)
        if not documents:
            logger.error("No documents found in DB — run ingestion pipeline first.")
            return

        logger.info("Building FAISS index from %d chunks…", len(documents))
        embeddings = self._get_embeddings()

        vectorstore = FAISS.from_documents(documents, embeddings)
        self.index_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.index_path))

        logger.info(
            "FAISS index saved to %s (%d documents indexed)",
            self.index_path,
            len(documents),
        )

    def load_index(self) -> FAISS:
        """
        Load the FAISS index from disk.

        This is what the Streamlit app calls at startup — never rebuild in prod.

        The allow_dangerous_deserialization=True flag is required by LangChain
        because the docstore uses pickle. This is safe here since we built the
        index ourselves from trusted local data (not from an untrusted source).
        """
        if not self._index_exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run: python scripts/build_index.py"
            )

        embeddings = self._get_embeddings()
        vectorstore = FAISS.load_local(
            str(self.index_path),
            embeddings,
            allow_dangerous_deserialization=True,  # safe: we built this index locally
        )
        logger.info("FAISS index loaded from %s", self.index_path)
        return vectorstore

    def get_retriever(self, ticker: Optional[str] = None, k: int = RETRIEVER_K):
        """
        Return a LangChain retriever, optionally pre-filtered by ticker.

        Uses metadata filtering so cross-ticker contamination is avoided
        once the index contains multiple companies.
        """
        vectorstore = self.load_index()

        search_kwargs: dict = {"k": k}
        if ticker:
            search_kwargs["filter"] = {"ticker": ticker}

        return vectorstore.as_retriever(search_kwargs=search_kwargs)

    def index_stats(self) -> dict:
        """Return basic stats about the loaded index."""
        vectorstore = self.load_index()
        return {
            "total_vectors": vectorstore.index.ntotal,
            "index_path": str(self.index_path),
            "embedding_model": self.embedding_model,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_documents(
        self,
        tickers: Optional[list[str]],
        filing_type: Optional[str],
    ) -> list[Document]:
        """Convert SQLite filing_chunks rows → LangChain Documents with metadata."""
        if tickers is None:
            tickers = self._all_tickers()

        documents: list[Document] = []
        for ticker in tickers:
            rows = self.db.get_filing_chunks(ticker, filing_type=filing_type)
            for row in rows:
                documents.append(
                    Document(
                        page_content=row["text"],
                        metadata={
                            "ticker": row["ticker"],
                            "filing_type": row["filing_type"],
                            "fiscal_period": row["fiscal_period"],
                            "filed_date": row["filed_date"],
                            "section": row["section"],
                            "chunk_index": row["chunk_index"],
                        },
                    )
                )
            logger.info("Loaded %d chunks for %s", len(rows), ticker)

        return documents

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load the BGE embedding model."""
        if self._embeddings is None:
            logger.info("Loading embedding model: %s…", self.embedding_model)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": True,  # required for BGE models
                    "batch_size": 32,
                },
            )
            logger.info("Embedding model loaded.")
        return self._embeddings

    def _index_exists(self) -> bool:
        return (self.index_path / "index.faiss").exists()

    def _all_tickers(self) -> list[str]:
        rows = self.db.conn.execute(
            "SELECT DISTINCT ticker FROM filing_chunks ORDER BY ticker"
        ).fetchall()
        return [r["ticker"] for r in rows]
