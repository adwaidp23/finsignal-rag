# Financial News Sentiment + RAG Trading Signal Assistant
## Upgraded Free-Tier Project Plan (v2 — revised after technical review)

**Total cost: $0.** Every component below runs on a free tier or is fully open-source/local. No credit card required anywhere in this stack.

> **v2 changelog:** resolved the Groq/OpenAI SDK ambiguity, added a persistence strategy for ephemeral cloud disks, defined a filing-chunking strategy, added a FinBERT CPU-latency plan, expanded secrets management, added a RAG evaluation metric, split the overloaded final week, and added the SQLite cache layer to the architecture diagram.

---

## 1. System Architecture (with persistence layer shown)

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Data Ingestion   │ --> │  Sentiment Layer   │ --> │  SQLite Cache     │
│ RSS + SEC EDGAR   │     │  FinBERT (local,   │     │ (sentiment scores,│
│                    │     │  pre-computed)     │     │  parsed filings)  │
└──────────────────┘     └───────────────────┘     └──────────────────┘
                                                              │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │   FAISS Index (committed   │
                                                │   to repo, rebuilt only    │
                                                │   on content change)       │
                                                └───────────────────────────┘
                                                              │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │      Router Agent          │
                                                │  local FAISS → DuckDuckGo  │
                                                │   (langchain-groq client)  │
                                                └───────────────────────────┘
                                                              │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │    Synthesis Agent         │
                                                │ sentiment + fundamentals   │
                                                │   → SQLite cache (again)   │
                                                └───────────────────────────┘
                                                              │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │   Streamlit Dashboard      │
                                                │ (HF Spaces / Streamlit     │
                                                │  Cloud, secrets via        │
                                                │  platform settings)        │
                                                └───────────────────────────┘
```

---

## 2. Free-Tier Component Matrix

| Layer | Tool | Free Limit | Notes |
|---|---|---|---|
| LLM reasoning | Groq (Llama 3.1 8B) via **`langchain-groq`** | 14,400 req/day, 500K tokens/day, 30 RPM | No card needed. Use for router + synthesis steps only. |
| LLM reasoning (heavier) | Groq (Llama 3.3 70B) | ~1,000 req/day | Reserve for final report synthesis only. |
| Sentiment scoring | FinBERT (HuggingFace, local) | Unlimited | Pre-computed at ingestion time, not on-demand — see §5. |
| Embeddings | BGE-small (sentence-transformers, local) | Unlimited | No API calls for embedding generation. |
| Vector store | FAISS | Unlimited | Index **committed to the repo**, not rebuilt on every deploy — see §4. |
| News data | Google News RSS / Yahoo Finance RSS | Unlimited | No key required. |
| Filings data | SEC EDGAR full-text API | Unlimited (fair-use ~10 req/sec) | Parsed text cached to SQLite immediately — raw HTML not relied upon. |
| Web search fallback | DuckDuckGo Search | Unlimited (soft limits) | Add retry/backoff. |
| Price data (backtesting) | `yfinance` | Unlimited (unofficial) | Add `time.sleep()` between calls and reuse a `requests.Session()` to reduce IP-block risk. |
| Hosting | Hugging Face Spaces or Streamlit Community Cloud | Free tier | Secrets set via platform dashboard, not `.env` — see §6. |
| Version control | GitHub | Free | Public repo for portfolio. |

---

## 3. Dependency Fix: Groq vs. OpenAI

The original plan listed `openai` while targeting Groq, which is confusing without explanation. Resolved by using **`langchain-groq`** directly, which exposes `ChatGroq` as a first-class LangChain chat model — no OpenAI-compatible shim needed.

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0.2
)
```

If you specifically want the raw SDK instead of LangChain's wrapper, use `groq` (the official client) — but pick one and document it in the README so it's unambiguous.

---

## 4. Persistence Strategy (new)

Both Hugging Face Spaces and Streamlit Community Cloud use **ephemeral disks** — anything written at runtime disappears on redeploy or after inactivity sleep. This affects two things in the original plan:

- **FAISS index**: build it once locally, then **commit the serialized index files** (`index.faiss`, `index.pkl`) to the repo. On startup, the app loads the committed index instead of rebuilding it. Only rebuild when you intentionally add new filings — via a manual local script, not on every deploy.
- **SEC filings**: `sec-edgar-downloader` saves raw HTML to disk, which is fine locally but won't survive cloud redeploys. Fix: parse each filing to clean text **once**, store the parsed text in SQLite (which you're already committing/persisting via a mounted volume or periodic export), and treat the raw HTML as disposable scratch space, not a data store.

```python
# One-time local build script (run manually, not on app startup)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("data/faiss_index")  # commit this folder to git
```

```python
# App startup — just load, never rebuild
vectorstore = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
```

---

## 5. Chunking Strategy for SEC Filings (new — was previously undefined)

Chunking is the single biggest lever on retrieval quality, so it's worth being explicit:

- **Split by section headers first** (Item 1A "Risk Factors," Item 7 "MD&A," etc.) using the filing's actual structure rather than blind character counts — SEC filings have consistent `ITEM X.` headers you can regex-split on before chunking within each section.
- **Within each section**, use `RecursiveCharacterTextSplitter` with `chunk_size=800, chunk_overlap=100` as a reasonable default for dense financial prose.
- **Attach metadata to every chunk**: ticker, filing type (10-K/10-Q), fiscal period, section name, and filing date. This lets the retriever filter by ticker/date before doing a similarity search, which matters once your index has more than one company in it.

```python
import re

def split_by_item(filing_text: str) -> dict:
    sections = re.split(r'(ITEM\s+\d+[A-Z]?\.)', filing_text)
    return {sections[i]: sections[i+1] for i in range(1, len(sections)-1, 2)}
```

---

## 6. FinBERT CPU Latency Plan (new)

Free-tier hosts (HF Spaces, Streamlit Cloud) are CPU-only with 2–16 GB RAM. Running FinBERT live, per-request, inside the deployed app will make the UI feel slow (10–30 seconds for a 20-headline batch).

**Fix: pre-compute, don't infer live.**
- Sentiment scoring happens in the **ingestion pipeline** (a scheduled local or GitHub Actions script), not inside the Streamlit app.
- Scores are written to SQLite alongside the headline/date/ticker.
- The deployed app only **reads** pre-computed scores — it never loads the FinBERT model at all, which also shrinks the app's memory footprint on the free host.
- If you want a "live" demo feel for judges/reviewers, add a small pre-cached demo dataset for 5–10 tickers with full pre-computed history, and clearly label live-fetch mode as "may take up to 30s" if you keep it as an option.

---

## 7. Secrets Management (new — was previously just "python-dotenv")

- **Local development**: `.env` file (gitignored) + `python-dotenv`, as originally planned.
- **Hugging Face Spaces**: secrets go in the Space's *Settings → Repository secrets*, injected as environment variables at runtime — `.env` is not used and should not be committed.
- **Streamlit Community Cloud**: secrets go in the app's *Settings → Secrets*, written in TOML format, accessed via `st.secrets["GROQ_API_KEY"]` instead of `os.environ`.
- Add a small abstraction so the same code works in both places:

```python
import os
import streamlit as st

def get_secret(key: str) -> str:
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, "")
```

---

## 8. RAG Retrieval Evaluation (new)

To claim the retrieval layer actually works, add a lightweight eval before relying on it:

- Hand-write 8–10 test queries per ticker (e.g., "What did the company say about supply chain risk in the most recent 10-K?").
- Manually verify the top-3 retrieved chunks actually contain the answer.
- Track a simple **hit-rate** (queries where a correct chunk appeared in top-3) as a metric in your README — even an informal 8/10 is a credible, quantified claim for a portfolio project.

---

## 9. Rate Limit Budget Plan (Groq) — unchanged, confirmed accurate

| Task | Frequency | Est. Groq calls/day |
|---|---|---|
| Router agent decision (local vs. web) | Per ticker analysis | 1 per ticker |
| Synthesis report generation | Per ticker analysis | 1 per ticker |
| Web search fallback synthesis (only if triggered) | ~20% of tickers | 0.2 per ticker |

~2.2 calls/ticker → ~6,000 tickers/day theoretical ceiling on the free tier. A 20–50 ticker watchlist, refreshed a few times daily, uses a small fraction of this.

---

## 10. Build Timeline (revised — Week 4 split, was overloaded)

| Week | Focus | Notes |
|---|---|---|
| 1 | Ingestion: RSS + SEC EDGAR downloader; define chunking strategy; pre-compute FinBERT sentiment into SQLite | No API keys needed this week |
| 2 | Build FAISS index locally with BGE embeddings; run the retrieval eval (§8); commit index to repo | Test retrieval quality before moving on |
| 3 | Router + synthesis agents via `langchain-groq`; SQLite caching for LLM responses | Sign up for Groq key; test rate-limit behavior early |
| 4 | Streamlit dashboard (reads pre-computed data only) + deploy to HF Spaces/Streamlit Cloud with proper secrets | Confirm FAISS index and SQLite cache load correctly post-deploy |
| 5 | Backtesting module (`yfinance` + correlation chart) + polish + README with eval numbers | Split out from Week 4 as its own week |

---

## 11. requirements.txt (revised)

```
langchain
langchain-community
langchain-groq
sentence-transformers
faiss-cpu
transformers
torch
duckduckgo-search
feedparser
sec-edgar-downloader
yfinance
streamlit
python-dotenv
plotly
```

(`openai` removed — not used; `groq`/`langchain-groq` is the actual dependency.)

---

## 12. Risk Notes (Free-Tier Specific)

- **Model availability drift**: keep the model name in a config variable, not hardcoded.
- **Rate limits are per-organization**, not per key.
- **DuckDuckGo search** has no official documented rate limit — add retry/backoff.
- **yfinance** can IP-block on aggressive polling — add delays and reuse a session.
- **Framing**: keep all outputs framed as "sentiment vs. fundamentals divergence," never buy/sell recommendations.

---

## 13. Stretch Goals (Still Free)

- **LangGraph state machine** — nodes: `fetch_news → score_sentiment → retrieve_context → decide_web_fallback → synthesize`, each as an explicit node with a typed state object passed between them. Worth a proper state/edge diagram in the README since it's the highest-leverage addition for portfolio impact.
- **Source citation UI** in Streamlit using `st.expander` to show which filing paragraph or headline drove each conclusion.
- **Backtest correlation chart** — sentiment divergence score vs. 5-day forward price return, via `yfinance` + `plotly`.

---

## 14. Portfolio Signal Self-Assessment (post-revision)

| Dimension | Before | After v2 fixes |
|---|---|---|
| Technical depth | 8/10 | 8/10 (unchanged — already strong) |
| Cost discipline | 10/10 | 10/10 |
| Completeness | 6/10 | 9/10 (persistence, chunking, secrets, latency all now addressed) |
| Differentiation | 8/10 | 8/10 |
| Responsible AI | 9/10 | 9/10 |
