"""
src/ingestion/chunker.py
------------------------
SEC filing text → structured chunks with metadata.

Strategy (from the plan §5):
  1. Split by ITEM section headers (regex on "ITEM X." / "ITEM XA." patterns)
     — respects the actual structure of 10-K/10-Q filings.
  2. Within each section, use RecursiveCharacterTextSplitter
     (chunk_size=800, chunk_overlap=100) for dense financial prose.
  3. Attach metadata to every chunk: ticker, filing_type, fiscal_period,
     filed_date, section, chunk_index.

Usage
-----
    from src.ingestion.chunker import chunk_filing

    chunks = chunk_filing(
        text="...full filing text...",
        ticker="AAPL",
        filing_type="10-K",
        fiscal_period="2025-Q4",
        filed_date="2025-11-01",
    )
    # chunks is a list[FilingChunk]
"""

import logging
import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.cache.db import FilingChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Matches "ITEM 1.", "ITEM 1A.", "ITEM 10." etc. (case-insensitive)
ITEM_HEADER_PATTERN = re.compile(
    r"(ITEM\s+\d+[A-Z]?\.)", re.IGNORECASE
)

# Sections we care about most for financial analysis
HIGH_VALUE_SECTIONS = {
    "ITEM 1": "Business",
    "ITEM 1A": "Risk Factors",
    "ITEM 7": "MD&A",
    "ITEM 7A": "Quantitative Disclosures",
    "ITEM 8": "Financial Statements",
    "ITEM 9A": "Controls and Procedures",
}

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_filing(
    text: str,
    ticker: str,
    filing_type: str,
    fiscal_period: str,
    filed_date: str,
    sections_filter: Optional[list[str]] = None,
) -> list[FilingChunk]:
    """
    Split a full filing text into structured chunks.

    Parameters
    ----------
    text : str
        Full plain-text content of the filing.
    ticker : str
        Stock ticker, e.g. "AAPL".
    filing_type : str
        "10-K" or "10-Q".
    fiscal_period : str
        e.g. "2025-Q4".
    filed_date : str
        ISO-8601 date string, e.g. "2025-11-01".
    sections_filter : list[str], optional
        If provided, only include these section keys (e.g. ["ITEM 1A", "ITEM 7"]).
        Defaults to all sections found.

    Returns
    -------
    list[FilingChunk]
        Ordered list of FilingChunk dataclass instances ready for DB insertion.
    """
    sections = _split_by_item_header(text)

    if not sections:
        # No headers found — treat entire text as a single "PREAMBLE" section
        logger.warning(
            "%s %s %s: no ITEM headers found, chunking as preamble",
            ticker,
            filing_type,
            fiscal_period,
        )
        sections = {"PREAMBLE": text}

    chunks: list[FilingChunk] = []

    for section_key, section_text in sections.items():
        normalized_key = section_key.strip().upper()

        # Apply section filter if provided
        if sections_filter and normalized_key not in [
            s.upper() for s in sections_filter
        ]:
            continue

        section_label = HIGH_VALUE_SECTIONS.get(normalized_key, normalized_key)
        section_chunks = _splitter.split_text(section_text.strip())

        for idx, chunk_text in enumerate(section_chunks):
            if not chunk_text.strip():
                continue
            chunks.append(
                FilingChunk(
                    ticker=ticker,
                    filing_type=filing_type,
                    fiscal_period=fiscal_period,
                    filed_date=filed_date,
                    section=f"{normalized_key} — {section_label}",
                    chunk_index=idx,
                    text=chunk_text,
                )
            )

    logger.info(
        "%s %s %s: %d sections → %d chunks",
        ticker,
        filing_type,
        fiscal_period,
        len(sections),
        len(chunks),
    )
    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_by_item_header(text: str) -> dict[str, str]:
    """
    Split filing text by ITEM headers.

    Returns a dict mapping header label (e.g. "ITEM 1A") → section body text.

    Edge cases handled:
    - Preamble text before the first ITEM header is stored under key "PREAMBLE"
      (not silently dropped — preamble often has cover page / CIK info).
    - Consecutive headers with no body text between them are kept as empty strings.
    """
    parts = ITEM_HEADER_PATTERN.split(text)
    # parts layout after split: [preamble, 'ITEM 1.', text1, 'ITEM 1A.', text1a, ...]

    result: dict[str, str] = {}

    # Preamble (before first ITEM header)
    preamble = parts[0].strip()
    if preamble:
        result["PREAMBLE"] = preamble

    # Pair up (header, body) — step by 2 starting at index 1
    i = 1
    while i < len(parts) - 1:
        raw_header = parts[i]                        # e.g. "ITEM 1A."
        body = parts[i + 1] if i + 1 < len(parts) else ""

        # Normalize: strip trailing dot and whitespace
        key = re.sub(r"\.$", "", raw_header.strip()).upper()  # "ITEM 1A"
        result[key] = body
        i += 2

    return result
