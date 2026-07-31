"""
src/agents/synthesis_agent.py
------------------------------
Synthesis Agent — combines sentiment scores + retrieved filing context
to generate a structured "sentiment vs. fundamentals divergence" report.

LLM used: llama-3.3-70b-versatile (heavy model — reserved for final synthesis only)
Cache: every Groq response is stored in SQLite keyed by (ticker, date, prompt_hash)
       so re-running the same ticker on the same day costs zero additional API calls.

Output format: structured dict suitable for direct Streamlit rendering.

IMPORTANT framing: outputs are labelled as "sentiment vs. fundamentals divergence"
— NOT buy/sell recommendations. This is intentional for responsible design and
to avoid financial advice framing in a portfolio project.

Usage
-----
    from src.agents.synthesis_agent import SynthesisAgent

    agent = SynthesisAgent()
    report = agent.synthesize(
        ticker="AAPL",
        sentiment_summary={"positive": {"count": 12, "avg_score": 0.87}, ...},
        context=[{"text": "...", "metadata": {...}}, ...],
        context_source="local",
    )
"""

import json
import logging
from datetime import date
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.cache.db import Database, LLMCacheEntry
from src.utils.secrets import get_secret
from src.utils.config import GROQ_MODEL_HEAVY, GROQ_TEMPERATURE, DB_PATH

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are a financial research analyst assistant.
Your role is to identify divergences between news sentiment and fundamental
disclosures in SEC filings — NOT to make buy or sell recommendations.

Given:
  - A ticker symbol
  - Recent news sentiment summary (positive/negative/neutral counts and scores)
  - Relevant excerpts from SEC filings (10-K/10-Q)

Produce a structured JSON report with these exact keys:
{
  "ticker": "...",
  "analysis_date": "YYYY-MM-DD",
  "sentiment_signal": "bullish" | "bearish" | "mixed" | "neutral",
  "sentiment_summary": "1-2 sentence plain-English summary of the news sentiment",
  "fundamentals_summary": "1-2 sentence summary of what the filings say",
  "divergence_assessment": "1-2 sentences describing any gap between sentiment and fundamentals",
  "divergence_level": "high" | "medium" | "low" | "none",
  "key_risks": ["risk 1", "risk 2", "risk 3"],
  "data_sources": ["SEC 10-K", "Google News RSS"] (list what data was used),
  "disclaimer": "This analysis is for informational purposes only and does not constitute financial advice."
}

Respond with ONLY valid JSON. No markdown, no preamble."""


class SynthesisAgent:
    """
    Generates a sentiment vs. fundamentals divergence report using
    Llama 3.3-70B, with SQLite caching to save Groq quota.
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        groq_api_key: Optional[str] = None,
    ):
        self.db = db or Database(DB_PATH)
        self._groq_api_key = groq_api_key or get_secret("GROQ_API_KEY")
        self._llm = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        ticker: str,
        sentiment_summary: dict,
        context: list[dict],
        context_source: str = "local",
        analysis_date: Optional[str] = None,
    ) -> dict:
        """
        Generate a divergence report for a ticker.

        Parameters
        ----------
        ticker : str
        sentiment_summary : dict
            Output of db.get_sentiment_summary() — {label: {count, avg_score}}
        context : list[dict]
            Retrieved chunks from RouterAgent — [{"text": ..., "metadata": ...}]
        context_source : str
            "local" or "web" — included in data_sources field
        analysis_date : str, optional
            ISO date string. Defaults to today.

        Returns
        -------
        dict
            Structured report (see SYNTHESIS_SYSTEM_PROMPT for schema).
            Returns an error dict if synthesis fails.
        """
        today = analysis_date or date.today().isoformat()
        prompt = self._build_prompt(ticker, sentiment_summary, context, today)

        # --- Check SQLite cache first ---
        cached = self.db.get_llm_response(ticker, today, prompt)
        if cached:
            logger.info("[%s] Returning cached synthesis for %s", ticker, today)
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                logger.warning("[%s] Cached response was not valid JSON — re-generating", ticker)

        # --- Call Groq ---
        logger.info("[%s] Calling Groq (%s) for synthesis…", ticker, GROQ_MODEL_HEAVY)
        raw_response = self._call_llm(prompt)

        # --- Parse and cache ---
        report = self._parse_response(raw_response, ticker, today)
        self.db.set_llm_response(
            LLMCacheEntry(
                ticker=ticker,
                analysis_date=today,
                prompt_hash=Database.hash_prompt(prompt),
                response=json.dumps(report),
                model=GROQ_MODEL_HEAVY,
            )
        )

        return report

    def synthesize_from_db(self, ticker: str, days_back: int = 7) -> dict:
        """
        Convenience method: pulls sentiment + context automatically from
        the DB and runs synthesis. Useful for the Streamlit app.
        """
        from datetime import timedelta

        since = (date.today() - timedelta(days=days_back)).isoformat()
        sentiment_summary = self.db.get_sentiment_summary(ticker, since=since)

        if not sentiment_summary:
            return {
                "error": f"No sentiment data found for {ticker} in the last {days_back} days.",
                "ticker": ticker,
                "analysis_date": date.today().isoformat(),
            }

        # Use filing chunks as context (simplified — router handles this in full pipeline)
        chunks = self.db.get_filing_chunks(ticker)[:5]
        context = [
            {
                "text": row["text"],
                "metadata": {
                    "ticker": row["ticker"],
                    "section": row["section"],
                    "filing_type": row["filing_type"],
                },
            }
            for row in chunks
        ]

        return self.synthesize(ticker, sentiment_summary, context)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        ticker: str,
        sentiment_summary: dict,
        context: list[dict],
        analysis_date: str,
    ) -> str:
        """Assemble the user prompt from sentiment data and retrieved context."""
        sentiment_text = self._format_sentiment(sentiment_summary)
        context_text = self._format_context(context)

        return f"""Ticker: {ticker}
Analysis Date: {analysis_date}

=== NEWS SENTIMENT (last 7 days) ===
{sentiment_text}

=== FILING EXCERPTS ===
{context_text}

Generate the JSON divergence report now."""

    @staticmethod
    def _format_sentiment(summary: dict) -> str:
        if not summary:
            return "No sentiment data available."
        lines = []
        total = sum(v["count"] for v in summary.values())
        for label, data in summary.items():
            pct = data["count"] / total * 100 if total > 0 else 0
            lines.append(
                f"  {label.capitalize()}: {data['count']} articles "
                f"({pct:.0f}%), avg confidence {data['avg_score']:.2f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_context(context: list[dict]) -> str:
        if not context:
            return "No filing context available."
        parts = []
        for i, c in enumerate(context[:5], 1):  # Cap at 5 chunks to stay under token limit
            meta = c.get("metadata", {})
            section = meta.get("section", "Unknown section")
            filing = meta.get("filing_type", "Filing")
            source = meta.get("source", "local")
            header = f"[{i}] {filing} — {section}" if source != "web" else f"[{i}] Web: {meta.get('url', '')}"
            parts.append(f"{header}\n{c['text'][:600]}")
        return "\n\n---\n\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """Make the Groq API call. Returns raw text response."""
        llm = self._get_llm()
        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as exc:
            logger.error("Groq synthesis call failed: %s", exc)
            raise

    def _parse_response(self, raw: str, ticker: str, analysis_date: str) -> dict:
        """Parse JSON from LLM response, with graceful fallback."""
        # Strip markdown code fences if model adds them
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean

        try:
            report = json.loads(clean)
            # Ensure required fields are present
            report.setdefault("ticker", ticker)
            report.setdefault("analysis_date", analysis_date)
            report.setdefault(
                "disclaimer",
                "This analysis is for informational purposes only and does not constitute financial advice.",
            )
            return report
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON response: %s\nRaw: %s", exc, raw[:200])
            return {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "error": "Failed to parse synthesis response.",
                "raw_response": raw[:500],
                "disclaimer": "This analysis is for informational purposes only and does not constitute financial advice.",
            }

    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            if not self._groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY not set. Add it to .env or platform secrets."
                )
            self._llm = ChatGroq(
                model=GROQ_MODEL_HEAVY,
                groq_api_key=self._groq_api_key,
                temperature=GROQ_TEMPERATURE,
            )
        return self._llm
