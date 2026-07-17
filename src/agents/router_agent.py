"""
src/agents/router_agent.py
---------------------------
Router Agent — decides whether a query can be answered from the local
FAISS index or needs a DuckDuckGo web search fallback.

Routing logic:
  1. Try local FAISS retrieval (fast, free, no API).
  2. If retrieved chunks have low relevance score OR if the query is about
     very recent news (< 7 days), route to DuckDuckGo web search.
  3. The routing decision itself is made by Llama 3.1-8B (fast, cheap)
     via Groq — one call per ticker analysis.

LLM used: llama-3.1-8b-instant (router — high-volume, low-cost)
LLM used: NOT llama-3.3-70b — that's reserved for synthesis only.

Usage
-----
    from src.agents.router_agent import RouterAgent
    from src.embeddings.faiss_builder import FAISSBuilder

    builder = FAISSBuilder()
    agent = RouterAgent(builder)
    result = agent.route(ticker="AAPL", query="Supply chain risks in latest 10-K")
    # result: {"source": "local"|"web", "context": [...], "query": "..."}
"""

import logging
import time
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.embeddings.faiss_builder import FAISSBuilder
from src.utils.secrets import get_secret
from src.utils.config import GROQ_MODEL_FAST, GROQ_TEMPERATURE, RETRIEVER_K

logger = logging.getLogger(__name__)

# Minimum FAISS relevance score to trust local results (cosine distance; lower = more similar)
LOCAL_RELEVANCE_THRESHOLD = 0.45

# DuckDuckGo search result count
DDG_MAX_RESULTS = 5

ROUTER_SYSTEM_PROMPT = """You are a financial research routing assistant.
Given a ticker symbol and a research query, decide whether the query
should be answered from:
  A) LOCAL — the company's SEC filings (10-K/10-Q) already in our database
  B) WEB   — a live web search for recent news or data not in our filings

Rules:
- Choose LOCAL if the query is about risk factors, MD&A, financial statements,
  business description, or anything typically found in annual/quarterly filings.
- Choose WEB if the query is about events in the last 7 days, earnings calls,
  breaking news, or analyst price targets not found in filings.
- Respond with ONLY one word: LOCAL or WEB. No explanation."""


class RouterAgent:
    """
    Routes a research query to either the local FAISS index or DuckDuckGo.
    Uses Llama 3.1-8B (fast/cheap) for the routing decision itself.
    """

    def __init__(
        self,
        faiss_builder: Optional[FAISSBuilder] = None,
        groq_api_key: Optional[str] = None,
    ):
        self.faiss_builder = faiss_builder or FAISSBuilder()
        self._groq_api_key = groq_api_key or get_secret("GROQ_API_KEY")
        self._llm = None       # lazy init
        self._vectorstore = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        ticker: str,
        query: str,
        force_source: Optional[str] = None,
    ) -> dict:
        """
        Route a query and return retrieved context.

        Parameters
        ----------
        ticker : str
            Stock ticker, e.g. "AAPL".
        query : str
            Research question to answer.
        force_source : str, optional
            "local" or "web" to bypass routing logic (useful for testing).

        Returns
        -------
        dict with keys:
            source   : "local" | "web"
            context  : list of {"text": str, "metadata": dict}
            query    : the original query
            ticker   : the ticker
        """
        source = force_source or self._decide_source(ticker, query)
        logger.info("[%s] Router → %s (query: %s…)", ticker, source.upper(), query[:50])

        if source == "local":
            context = self._fetch_local(ticker, query)
            # Fallback to web if local returns nothing useful
            if not context:
                logger.info("[%s] Local returned no results — falling back to web", ticker)
                source = "web"
                context = self._fetch_web(ticker, query)
        else:
            context = self._fetch_web(ticker, query)

        return {
            "source": source,
            "context": context,
            "query": query,
            "ticker": ticker,
        }

    # ------------------------------------------------------------------
    # Internal: routing decision
    # ------------------------------------------------------------------

    def _decide_source(self, ticker: str, query: str) -> str:
        """Ask Llama 3.1-8B to decide LOCAL vs WEB."""
        try:
            llm = self._get_llm()
            messages = [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Ticker: {ticker}\nQuery: {query}"
                ),
            ]
            response = llm.invoke(messages)
            decision = response.content.strip().upper()
            if "WEB" in decision:
                return "web"
            return "local"
        except Exception as exc:
            logger.warning("Router LLM call failed (%s) — defaulting to local", exc)
            return "local"

    # ------------------------------------------------------------------
    # Internal: local retrieval
    # ------------------------------------------------------------------

    def _fetch_local(self, ticker: str, query: str) -> list[dict]:
        """Retrieve top-k chunks from FAISS, filtered to the given ticker."""
        try:
            vs = self._get_vectorstore()
            # similarity_search_with_score returns (Document, score) tuples
            # score is L2 distance; lower = more similar
            results = vs.similarity_search_with_score(
                query,
                k=RETRIEVER_K,
                filter={"ticker": ticker},
            )
            context = []
            for doc, score in results:
                if score <= LOCAL_RELEVANCE_THRESHOLD:
                    context.append({
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "score": round(float(score), 4),
                    })
            logger.info(
                "[%s] Local retrieval: %d/%d chunks above threshold",
                ticker,
                len(context),
                len(results),
            )
            return context
        except Exception as exc:
            logger.error("[%s] Local retrieval failed: %s", ticker, exc)
            return []

    # ------------------------------------------------------------------
    # Internal: web search fallback
    # ------------------------------------------------------------------

    def _fetch_web(self, ticker: str, query: str) -> list[dict]:
        """Search DuckDuckGo for recent news about the ticker."""
        try:
            from duckduckgo_search import DDGS

            search_query = f"{ticker} {query}"
            context = []

            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=DDG_MAX_RESULTS))

            for r in results:
                context.append({
                    "text": f"{r.get('title', '')}\n{r.get('body', '')}",
                    "metadata": {
                        "source": "web",
                        "url": r.get("href", ""),
                        "ticker": ticker,
                    },
                    "score": None,
                })

            # Be gentle — basic rate limit respect
            time.sleep(1.5)
            logger.info("[%s] Web search returned %d results", ticker, len(context))
            return context

        except Exception as exc:
            logger.error("[%s] DuckDuckGo search failed: %s", ticker, exc)
            return []

    # ------------------------------------------------------------------
    # Lazy initialisers
    # ------------------------------------------------------------------

    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            if not self._groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY not set. Add it to .env or platform secrets."
                )
            self._llm = ChatGroq(
                model=GROQ_MODEL_FAST,
                groq_api_key=self._groq_api_key,
                temperature=GROQ_TEMPERATURE,
            )
        return self._llm

    def _get_vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = self.faiss_builder.load_index()
        return self._vectorstore
