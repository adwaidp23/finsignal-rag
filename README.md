# FinSignal-RAG — Financial Sentiment + RAG Trading Signal Dashboard

> **A fully free-tier, $0-cost financial research assistant** that cross-references
> news sentiment against SEC 10-K fundamentals using FinBERT + FAISS + Groq Llama 3.3-70B.
> Built as a portfolio-grade project demonstrating modern agentic AI architecture.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Cost: $0](https://img.shields.io/badge/cost-%240-brightgreen.svg)](#cost-breakdown)
[![RAG Eval: 80% Hit Rate](https://img.shields.io/badge/RAG_Eval-80%25_hit_rate-orange.svg)](#rag-evaluation-results)

---

## What It Does

FinSignal-RAG answers the question: **"Does recent news sentiment match what the company disclosed in their SEC filings?"**

When the answer is *no*, that divergence is a potential research signal worth investigating.

```
RSS Headlines ──→ FinBERT ──→ Sentiment Score
                                    │
SEC 10-K/10-Q ──→ FAISS RAG ──→ Filing Context
                                    │
                             ┌──────▼──────┐
                             │ Llama 3.3-70B│
                             │ (via Groq)   │
                             └──────┬──────┘
                                    │
                         Divergence Report (JSON)
                                    │
                         Streamlit Dashboard
```

---

## Architecture

### Components

| Layer | Technology | Free? |
|---|---|---|
| **News ingestion** | Google News RSS + Yahoo Finance RSS via `feedparser` | ✅ Free |
| **Filings ingestion** | SEC EDGAR Full-Text API via `sec-edgar-downloader` | ✅ Free |
| **Sentiment scoring** | FinBERT (`ProsusAI/finbert`, local HuggingFace) | ✅ Free |
| **Embeddings** | BGE-small (`BAAI/bge-small-en-v1.5`, local) | ✅ Free |
| **Vector store** | FAISS (index committed to repo, never rebuilt on deploy) | ✅ Free |
| **LLM — router** | Llama 3.1-8B-Instant via Groq API | ✅ Free tier |
| **LLM — synthesis** | Llama 3.3-70B-Versatile via Groq API | ✅ Free tier |
| **LLM cache** | SQLite (keyed by ticker + date + prompt hash) | ✅ Free |
| **Price data** | `yfinance` with rate limiting | ✅ Free |
| **Dashboard** | Streamlit | ✅ Free |
| **Hosting** | Hugging Face Spaces or Streamlit Community Cloud | ✅ Free tier |

### Two-Tier Agent Design

```
User query
    │
    ▼
┌─────────────────────────────────────────────┐
│             Router Agent (Llama 3.1-8B)     │
│  Decides: LOCAL FAISS vs. DuckDuckGo WEB    │
└──────────────────┬──────────────────────────┘
         LOCAL ────┤──── WEB
         │         │     │
         ▼         │     ▼
    FAISS search   │  DuckDuckGo
    (semantic)     │  (recent news)
         │         │     │
         └─────────┼─────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │        Synthesis Agent (Llama 3.3-70B)   │
    │  Sentiment + Fundamentals → JSON report  │
    └──────────────────────────────────────────┘
```

- **Router uses Llama 3.1-8B** (fast, high-volume): ~1 Groq call per ticker analysis
- **Synthesis uses Llama 3.3-70B** (heavy, cached): ~1 Groq call per ticker per day
- Every synthesis response is cached in SQLite — same ticker same day = 0 extra API calls

### Persistence Strategy

Both Hugging Face Spaces and Streamlit Community Cloud use **ephemeral disks**.

This project solves that by:
1. **FAISS index**: built once locally, committed to the repo as `data/faiss_index/`. App loads it at startup — never rebuilds.
2. **SQLite DB**: pre-populated locally via ingestion scripts, committed to repo. App reads only.
3. **SEC filings**: parsed text stored in SQLite on first ingest; raw HTML is disposable.

---

## RAG Evaluation Results

Evaluated against 10 hand-written test queries across AAPL, MSFT, NVDA, and GOOGL.

| Metric | Value |
|---|---|
| **Top-3 Hit Rate** | **8 / 10 (80%)** |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Chunking | Section-first (ITEM headers) → RecursiveCharacterTextSplitter 800/100 |
| Index | FAISS flat L2, ~40MB |

**Hit detail:**
- ✅ AAPL — Supply chain risks
- ✅ AAPL — Services revenue
- ✅ AAPL — Competition and market share
- ✅ MSFT — Cloud revenue growth
- ✅ MSFT — Cybersecurity risks
- ✅ MSFT — AI strategy
- ❌ NVDA — Data center revenue (filing not indexed)
- ❌ NVDA — Export controls (filing not indexed)
- ✅ GOOGL — Advertising revenue risks
- ✅ GOOGL — Antitrust concerns

*NVDA misses: NVIDIA's most recent 10-K was not yet in the EDGAR index at eval time. Indexing it would likely bring hit rate to 10/10.*

Run the eval yourself:
```bash
python tests/eval_retrieval.py
```

---

## Groq Rate Limit Budget

| Task | Calls/ticker | Notes |
|---|---|---|
| Router decision | 1 | Fast model (8B) — high allowance |
| Synthesis report | 1 | Heavy model (70B), SQLite cached |
| Web fallback synthesis | ~0.2 | Only triggered ~20% of the time |

**~2.2 Groq calls per ticker.** The free tier allows 14,400 req/day and 500K tokens/day.
A 20–50 ticker watchlist refreshed 3× daily uses < 5% of the free quota.

---

## Setup

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/) (no credit card required)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/finsignal-rag
cd finsignal-rag
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard loads pre-computed data from SQLite and the committed FAISS index — no ingestion needed to see it work.

---

## Data Ingestion (local pipeline)

Ingestion is **separate from the app** — run it locally or on a schedule via GitHub Actions.

```bash
# Fetch RSS news + SEC filings for the default watchlist (10 tickers)
python scripts/run_ingestion.py

# RSS only — faster, no EDGAR download
python scripts/run_ingestion.py --rss-only --tickers AAPL MSFT NVDA

# EDGAR 10-K only
python scripts/run_ingestion.py --edgar-only --tickers AAPL --filings 10-K --num 2

# After adding new filings, rebuild the FAISS index
python scripts/build_index.py --force

# Commit the updated index
git add data/faiss_index/
git commit -m "rebuild index - YYYY-MM-DD"
```

---

## Backtesting

The backtesting tab correlates the daily FinBERT sentiment signal against N-day forward stock returns using `yfinance`.

```
Pearson r = sentiment_score_daily vs. Close_price_{t+N}
```

Interpret r as:
- `r ≥ +0.10` → Positive correlation (bullish sentiment → price gains)
- `r ≤ -0.10` → Contrarian / mean-reversion signal
- `-0.10 < r < +0.10` → Weak linear signal

> ⚠️ This is for research and educational purposes only. Not financial advice.

Run signal accuracy across all tickers in the DB:
```bash
python scripts/test_signal_accuracy.py
```

---

## Project Structure

```
finsignal-rag/
├── app/
│   └── streamlit_app.py         # Streamlit dashboard ("Ink Terminal" design)
├── data/
│   ├── faiss_index/             # Pre-built FAISS index (committed to repo)
│   ├── sentiment.db             # SQLite cache (headlines, sentiment, filings, LLM cache)
│   └── eval_results.json        # RAG evaluation output
├── scripts/
│   ├── run_ingestion.py         # RSS + EDGAR ingestion pipeline
│   ├── build_index.py           # One-time FAISS index builder
│   ├── smoke_test.py            # Import + schema validation
│   ├── test_dashboard_backend.py
│   └── test_signal_accuracy.py
├── src/
│   ├── agents/
│   │   ├── router_agent.py      # Llama 3.1-8B: LOCAL vs WEB routing
│   │   └── synthesis_agent.py   # Llama 3.3-70B: divergence report generation
│   ├── backtest/
│   │   └── backtester.py        # Sentiment vs. forward returns (yfinance)
│   ├── cache/
│   │   └── db.py                # SQLite wrapper (headlines, sentiment, filings, LLM cache)
│   ├── embeddings/
│   │   └── faiss_builder.py     # BGE-small embedding + FAISS index management
│   ├── ingestion/
│   │   ├── chunker.py           # Section-first + RecursiveCharacterTextSplitter
│   │   ├── edgar_fetcher.py     # SEC EDGAR API downloader + parser
│   │   └── rss_fetcher.py       # Google News + Yahoo Finance RSS
│   ├── sentiment/
│   │   └── finbert_scorer.py    # FinBERT pre-computation pipeline
│   └── utils/
│       ├── config.py            # All tuneable constants (model names, paths, etc.)
│       └── secrets.py           # Cross-platform secret resolution (st.secrets / os.environ)
├── tests/
│   └── eval_retrieval.py        # RAG hit-rate evaluator
├── .env.example
├── requirements.txt
└── README.md
```

---

## Deployment

### Streamlit Community Cloud

1. Push repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → point to `app/streamlit_app.py`.
3. In **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
4. Deploy. The pre-built FAISS index and SQLite DB load from the committed files.

### Hugging Face Spaces

1. Create a new Space with **Streamlit** as the SDK.
2. Push this repo.
3. In **Settings → Repository secrets**, add `GROQ_API_KEY`.
4. Point the Space's entry point to `app/streamlit_app.py`.

---

## Cost Breakdown

| Component | Cost |
|---|---|
| FinBERT inference (local CPU) | $0 |
| BGE-small embeddings (local CPU) | $0 |
| FAISS index operations | $0 |
| Groq API (free tier, ~2.2 calls/ticker) | $0 |
| SEC EDGAR API | $0 |
| Google News RSS / Yahoo Finance RSS | $0 |
| yfinance price data | $0 |
| Streamlit Community Cloud hosting | $0 |
| **Total** | **$0** |

---

## Risk Notes

- **Model availability**: Groq model names are stored in `src/utils/config.py`, not hardcoded — swap models in one place if Groq deprecates them.
- **Rate limits**: Groq limits are per-organization. DuckDuckGo has no official limit — the agent includes retry/backoff.
- **yfinance**: Can IP-block on aggressive polling — the backtester adds `time.sleep()` between calls.
- **Responsible AI**: All outputs are framed as "sentiment vs. fundamentals divergence" research — never as buy/sell recommendations.

---

## License

MIT. See [LICENSE](LICENSE).

---

*Built to demonstrate: agentic RAG architecture · free-tier engineering discipline · financial NLP · responsible AI framing*
