"""
tests/eval_retrieval.py
------------------------
RAG Retrieval Evaluation — Week 2 verification step.

Tests whether the FAISS retriever surfaces relevant filing chunks
for a set of hand-crafted queries per ticker.

Metric: Hit-rate = fraction of queries where at least one correct chunk
appears in the top-k results.

Even an informal 8/10 hit-rate is a credible, quantified claim for a
portfolio README.

Usage
-----
    # Run all eval queries
    python tests/eval_retrieval.py

    # Run for a specific ticker only
    python tests/eval_retrieval.py --ticker AAPL

    # Show retrieved chunk text (verbose)
    python tests/eval_retrieval.py --verbose

Output
------
    Prints a per-query pass/fail table and overall hit-rate.
    Results are also saved to data/eval_results.json for README inclusion.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.faiss_builder import FAISSBuilder
from src.utils.config import RETRIEVER_K

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_retrieval")

EVAL_RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_results.json"

# ---------------------------------------------------------------------------
# Eval query bank
# Hand-crafted queries — each has a list of keyword substrings that a
# "correct" chunk should contain (case-insensitive OR match).
# ---------------------------------------------------------------------------

EVAL_QUERIES: list[dict] = [
    # --- AAPL ---
    {
        "ticker": "AAPL",
        "query": "What are Apple's main risk factors related to supply chain?",
        "expected_keywords": ["supply chain", "supplier", "manufacturing", "component"],
    },
    {
        "ticker": "AAPL",
        "query": "How did Apple's Services segment perform in terms of revenue?",
        "expected_keywords": ["services", "revenue", "subscription", "app store"],
    },
    {
        "ticker": "AAPL",
        "query": "What does Apple say about competition and market share?",
        "expected_keywords": ["competition", "competitive", "market", "android"],
    },
    # --- MSFT ---
    {
        "ticker": "MSFT",
        "query": "What are Microsoft's cloud revenue growth drivers?",
        "expected_keywords": ["azure", "cloud", "revenue", "growth"],
    },
    {
        "ticker": "MSFT",
        "query": "What cybersecurity risks does Microsoft disclose?",
        "expected_keywords": ["cybersecurity", "security", "breach", "attack", "threat"],
    },
    {
        "ticker": "MSFT",
        "query": "How does Microsoft describe its AI strategy?",
        "expected_keywords": ["artificial intelligence", "ai", "copilot", "openai"],
    },
    # --- NVDA ---
    {
        "ticker": "NVDA",
        "query": "What are NVIDIA's data center revenue trends?",
        "expected_keywords": ["data center", "revenue", "gpu", "compute"],
    },
    {
        "ticker": "NVDA",
        "query": "What export control risks does NVIDIA face?",
        "expected_keywords": ["export", "control", "china", "regulation", "license"],
    },
    # --- GOOGL ---
    {
        "ticker": "GOOGL",
        "query": "How does Alphabet describe advertising revenue risks?",
        "expected_keywords": ["advertising", "revenue", "search", "competition"],
    },
    {
        "ticker": "GOOGL",
        "query": "What regulatory and antitrust concerns does Alphabet disclose?",
        "expected_keywords": ["antitrust", "regulatory", "competition", "doj", "eu"],
    },
]


@dataclass
class EvalResult:
    ticker: str
    query: str
    expected_keywords: list[str]
    hit: bool
    top_chunks: list[str] = field(default_factory=list)
    matched_keyword: Optional[str] = None


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


def run_eval(
    ticker_filter: Optional[str] = None,
    k: int = RETRIEVER_K,
    verbose: bool = False,
) -> dict:
    """
    Run all eval queries and return a summary dict.
    """
    builder = FAISSBuilder()

    try:
        vectorstore = builder.load_index()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run the ingestion pipeline and build_index.py first.")
        sys.exit(1)

    queries = EVAL_QUERIES
    if ticker_filter:
        queries = [q for q in queries if q["ticker"] == ticker_filter.upper()]
        if not queries:
            print(f"No eval queries defined for ticker: {ticker_filter}")
            sys.exit(1)

    results: list[EvalResult] = []

    print(f"\n{'=' * 65}")
    print(f"RAG Retrieval Evaluation   (k={k})")
    print(f"{'=' * 65}")
    print(f"{'#':<3} {'Ticker':<6} {'Hit':<5} {'Query':<50}")
    print(f"{'-' * 65}")

    for i, q in enumerate(queries, 1):
        ticker = q["ticker"]
        query = q["query"]
        keywords = q["expected_keywords"]

        # Retrieve top-k chunks, filtered by ticker
        search_kwargs = {"k": k, "filter": {"ticker": ticker}}
        try:
            docs = vectorstore.similarity_search(query, **search_kwargs)
        except Exception:
            # Fallback: no metadata filter if FAISS version doesn't support it
            docs = vectorstore.similarity_search(query, k=k)
            docs = [d for d in docs if d.metadata.get("ticker") == ticker]

        # Check if any returned chunk contains a relevant keyword
        hit = False
        matched_kw = None
        top_texts = [d.page_content for d in docs]

        for chunk_text in top_texts:
            chunk_lower = chunk_text.lower()
            for kw in keywords:
                if kw.lower() in chunk_lower:
                    hit = True
                    matched_kw = kw
                    break
            if hit:
                break

        result = EvalResult(
            ticker=ticker,
            query=query,
            expected_keywords=keywords,
            hit=hit,
            top_chunks=top_texts,
            matched_keyword=matched_kw,
        )
        results.append(result)

        status = "PASS" if hit else "FAIL"
        short_query = query[:48] + ".." if len(query) > 50 else query
        print(f"{i:<3} {ticker:<6} {status:<5} {short_query}")

        if verbose:
            for j, chunk in enumerate(top_texts, 1):
                print(f"      Chunk {j}: {chunk[:120].strip()}…")
            if matched_kw:
                print(f"      Matched keyword: '{matched_kw}'")
            print()

    # Summary
    total = len(results)
    hits = sum(1 for r in results if r.hit)
    hit_rate = hits / total if total > 0 else 0.0

    print(f"\n{'=' * 65}")
    print(f"Hit-rate: {hits}/{total} ({hit_rate:.0%})")

    if hit_rate >= 0.8:
        print("EXCELLENT — ready for portfolio README claim (≥80% hit-rate)")
    elif hit_rate >= 0.6:
        print("ACCEPTABLE — consider improving chunking or expanding index")
    else:
        print("NEEDS WORK — review chunking strategy and re-run ingestion")

    print(f"{'=' * 65}\n")

    # Save results
    summary = {
        "evaluated_at": datetime.utcnow().isoformat(),
        "k": k,
        "total_queries": total,
        "hits": hits,
        "hit_rate": round(hit_rate, 4),
        "results": [asdict(r) for r in results],
    }
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {EVAL_RESULTS_PATH}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation")
    parser.add_argument("--ticker", help="Run eval for a single ticker only")
    parser.add_argument(
        "--k", type=int, default=RETRIEVER_K, help=f"Top-k chunks to retrieve (default: {RETRIEVER_K})"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show retrieved chunk text"
    )
    args = parser.parse_args()
    run_eval(ticker_filter=args.ticker, k=args.k, verbose=args.verbose)


if __name__ == "__main__":
    main()
