"""
app/streamlit_app.py
---------------------
Streamlit dashboard for the Financial Sentiment + RAG Trading Signal Assistant.

Design principles:
  - App ONLY reads pre-computed data from SQLite — never runs FinBERT live.
  - FAISS index is loaded once at startup from the committed index files.
  - Groq synthesis is called on-demand with SQLite caching (zero repeated cost).
  - Secrets resolved via get_secret() — works on HF Spaces, Streamlit Cloud, local.

Run locally:
    streamlit run app/streamlit_app.py

Deploy:
    HF Spaces:         push to repo, set GROQ_API_KEY in Space secrets
    Streamlit Cloud:   connect repo, set GROQ_API_KEY in app secrets (TOML)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

from src.cache.db import Database
from src.utils.config import DB_PATH, FAISS_INDEX_PATH
from src.utils.secrets import get_secret

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SentimentRAG — Financial Signal Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f7fe;
        color: #1b2559;
    }
    
    .stApp { background-color: #f4f7fe; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12);
    }

    .divergence-high   { border-left: 4px solid #ef4444; }
    .divergence-medium { border-left: 4px solid #f59e0b; }
    .divergence-low    { border-left: 4px solid #10b981; }
    .divergence-none   { border-left: 4px solid #4318ff; }

    .signal-bullish { color: #10b981; font-weight: 700; }
    .signal-bearish { color: #ef4444; font-weight: 700; }
    .signal-mixed   { color: #f59e0b; font-weight: 700; }
    .signal-neutral { color: #a3aed0; font-weight: 700; }

    .disclaimer {
        font-size: 0.75rem;
        color: #a3aed0;
        border-top: 1px solid #e2e8f0;
        padding-top: 0.5rem;
        margin-top: 1rem;
    }
    
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Default watchlist
# ---------------------------------------------------------------------------

MARKETS = {
    "NASDAQ": {
        "AAPL": "Apple Inc.", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta Platforms", "NVDA": "NVIDIA", "TSLA": "Tesla", "AVGO": "Broadcom", "PEP": "PepsiCo", "COST": "Costco", "CSCO": "Cisco", "TMUS": "T-Mobile", "ADBE": "Adobe", "TXN": "Texas Instruments", "NFLX": "Netflix", "CMCSA": "Comcast", "AMD": "Advanced Micro Devices", "INTU": "Intuit", "QCOM": "Qualcomm", "AMGN": "Amgen", "HON": "Honeywell", "INTC": "Intel", "ISRG": "Intuitive Surgical", "GILD": "Gilead Sciences", "SBUX": "Starbucks", "BKNG": "Booking Holdings", "VRTX": "Vertex", "MDLZ": "Mondelez", "REGN": "Regeneron", "ADI": "Analog Devices", "ADP": "Automatic Data Processing", "PANW": "Palo Alto Networks", "MU": "Micron Technology", "SNPS": "Synopsys", "KLAC": "KLA Corp", "MELI": "MercadoLibre", "CDNS": "Cadence Design", "CSX": "CSX Corp", "PYPL": "PayPal", "MAR": "Marriott", "MNST": "Monster Beverage", "ASML": "ASML Holding", "ORLY": "O'Reilly Automotive", "CTAS": "Cintas", "LRCX": "Lam Research", "NXPI": "NXP Semiconductors", "FTNT": "Fortinet", "KDP": "Keurig Dr Pepper", "PAYX": "Paychex", "PCAR": "PACCAR"
    },
    "NYSE": {
        "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase", "V": "Visa", "JNJ": "Johnson & Johnson", "WMT": "Walmart", "PG": "Procter & Gamble", "MA": "Mastercard", "HD": "Home Depot", "CVX": "Chevron", "MRK": "Merck", "ABBV": "AbbVie", "KO": "Coca-Cola", "BAC": "Bank of America", "PFE": "Pfizer", "TMO": "Thermo Fisher", "DIS": "Walt Disney", "MCD": "McDonald's", "CRM": "Salesforce", "ACN": "Accenture", "DHR": "Danaher", "ABT": "Abbott Labs", "LIN": "Linde", "WFC": "Wells Fargo", "NKE": "NIKE", "PM": "Philip Morris", "NEE": "NextEra Energy", "RTX": "RTX Corp", "MS": "Morgan Stanley", "UPS": "United Parcel Service", "BA": "Boeing", "UNP": "Union Pacific", "IBM": "IBM", "C": "Citigroup", "BLK": "BlackRock", "GS": "Goldman Sachs", "LMT": "Lockheed Martin", "DE": "Deere & Co", "SYK": "Stryker", "GE": "General Electric", "AMT": "American Tower", "T": "AT&T", "MMM": "3M", "CVS": "CVS Health", "MO": "Altria", "SPGI": "S&P Global", "PLD": "Prologis", "CAT": "Caterpillar", "CB": "Chubb", "CI": "Cigna", "TJX": "TJX Companies"
    },
    "Shanghai": {
        "600519.SS": "Kweichow Moutai", "601398.SS": "ICBC", "601288.SS": "Agricultural Bank of China", "601939.SS": "CCB", "601857.SS": "PetroChina", "601988.SS": "Bank of China", "600036.SS": "China Merchants Bank", "601088.SS": "China Shenhua Energy", "601628.SS": "China Life Insurance", "600900.SS": "China Yangtze Power", "601318.SS": "Ping An Insurance", "600028.SS": "Sinopec", "601166.SS": "Industrial Bank", "601328.SS": "Bank of Communications", "601816.SS": "CGN Power", "600030.SS": "CITIC Securities", "603259.SS": "WuXi AppTec", "600276.SS": "Hengrui Medicine", "600438.SS": "Tongwei", "600000.SS": "SPDB", "600887.SS": "Yili", "601888.SS": "China Tourism", "600031.SS": "Sany Heavy", "600104.SS": "SAIC Motor", "601138.SS": "Foxconn Industrial", "600690.SS": "Haier Smart Home", "601899.SS": "Zijin Mining", "600048.SS": "Poly Developments", "601012.SS": "LONGi", "603993.SS": "China Molybdenum", "600809.SS": "Shanxi Xinghuacun Fen Wine", "601211.SS": "Guotai Junan", "600018.SS": "SIPG", "601800.SS": "China Communications Construction", "601668.SS": "China State Construction", "601390.SS": "China Railway", "601111.SS": "Air China", "601006.SS": "Daqin Railway", "600016.SS": "China Minsheng Bank", "601229.SS": "Bank of Shanghai", "601998.SS": "China CITIC Bank", "600019.SS": "Baoshan Iron & Steel", "601989.SS": "China Shipbuilding", "600111.SS": "China Northern Rare Earth", "601225.SS": "Shaanxi Coal", "600009.SS": "Shanghai International Airport", "601901.SS": "Founder Securities", "601688.SS": "Huatai Securities", "601878.SS": "Zheshang Securities", "600999.SS": "China Merchants Securities"
    },
    "Euronext": {
        "MC.PA": "LVMH", "RMS.PA": "Hermes International", "OR.PA": "L'Oreal", "SU.PA": "Schneider Electric", "TTE.PA": "TotalEnergies", "AIR.PA": "Airbus", "SAN.PA": "Sanofi", "SAF.PA": "Safran", "AI.PA": "Air Liquide", "EL.PA": "EssilorLuxottica", "BN.PA": "Danone", "CS.PA": "AXA", "BNP.PA": "BNP Paribas", "VINC.PA": "Vinci", "CAP.PA": "Capgemini", "ENGI.PA": "Engie", "GLE.PA": "Societe Generale", "SGO.PA": "Saint-Gobain", "ORA.PA": "Orange", "ACA.PA": "Credit Agricole", "STLAP.PA": "Stellantis", "PUB.PA": "Publicis Groupe", "MICP.PA": "Michelin", "LR.PA": "Legrand", "VE.PA": "Veolia", "EN.PA": "Bouygues", "HO.PA": "Thales", "RI.PA": "Pernod Ricard", "ASML.AS": "ASML Holding", "AD.AS": "Ahold Delhaize", "HEIA.AS": "Heineken", "INGA.AS": "ING Groep", "PRX.AS": "Prosus", "AKZA.AS": "Akzo Nobel", "URW.AS": "Unibail-Rodamco", "MT.AS": "ArcelorMittal", "WKL.AS": "Wolters Kluwer", "KPN.AS": "KPN", "NN.AS": "NN Group", "RAND.AS": "Randstad", "PHIA.AS": "Philips", "ABI.BR": "Anheuser-Busch InBev", "BPOST.BR": "Bpost", "UCB.BR": "UCB", "SOLB.BR": "Solvay", "KBC.BR": "KBC Group", "PROX.BR": "Proximus", "AGS.BR": "Ageas", "ARGX.BR": "Argenx", "GBLB.BR": "Groupe Bruxelles Lambert"
    },
    "NSE": {
        "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services", "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank", "BHARTIARTL.NS": "Bharti Airtel", "SBIN.NS": "State Bank of India", "INFY.NS": "Infosys", "ITC.NS": "ITC Limited", "HINDUNILVR.NS": "Hindustan Unilever", "LT.NS": "Larsen & Toubro", "BAJFINANCE.NS": "Bajaj Finance", "AXISBANK.NS": "Axis Bank", "HCLTECH.NS": "HCL Technologies", "KOTAKBANK.NS": "Kotak Mahindra Bank", "MARUTI.NS": "Maruti Suzuki", "SUNPHARMA.NS": "Sun Pharmaceuticals", "TATAMOTORS.NS": "Tata Motors", "TATASTEEL.NS": "Tata Steel", "NTPC.NS": "NTPC Limited", "ULTRACEMCO.NS": "UltraTech Cement", "M&M.NS": "Mahindra & Mahindra", "POWERGRID.NS": "Power Grid Corp", "ASIANPAINT.NS": "Asian Paints", "TITAN.NS": "Titan Company", "BAJAJFINSV.NS": "Bajaj Finserv", "ONGC.NS": "ONGC", "NESTLEIND.NS": "Nestle India", "WIPRO.NS": "Wipro", "ADANIENT.NS": "Adani Enterprises", "ADANIPORTS.NS": "Adani Ports", "COALINDIA.NS": "Coal India", "HINDALCO.NS": "Hindalco Industries", "GRASIM.NS": "Grasim Industries", "JSWSTEEL.NS": "JSW Steel", "TECHM.NS": "Tech Mahindra", "DRREDDY.NS": "Dr. Reddy's Labs", "INDUSINDBK.NS": "IndusInd Bank", "CIPLA.NS": "Cipla", "APOLLOHOSP.NS": "Apollo Hospitals", "EICHERMOT.NS": "Eicher Motors", "DIVISLAB.NS": "Divi's Laboratories", "HDFCLIFE.NS": "HDFC Life", "SBILIFE.NS": "SBI Life Insurance", "LTIM.NS": "LTIMindtree", "HEROMOTOCO.NS": "Hero MotoCorp", "BAJAJ-AUTO.NS": "Bajaj Auto", "BRITANNIA.NS": "Britannia Industries", "TATACONSUM.NS": "Tata Consumer Products", "BPCL.NS": "Bharat Petroleum", "TRENT.NS": "Trent Limited"
    }
}

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading database…")
def load_db():
    return Database(DB_PATH)


@st.cache_resource(show_spinner="Loading FAISS index…")
def load_faiss():
    """Load the committed FAISS index. Returns None if not built yet."""
    try:
        from src.embeddings.faiss_builder import FAISSBuilder
        builder = FAISSBuilder()
        return builder.load_index()
    except FileNotFoundError:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_sentiment_summary(ticker: str, days_back: int = 7):
    db = load_db()
    since = (date.today() - timedelta(days=days_back)).isoformat()
    return db.get_sentiment_summary(ticker, since=since)


@st.cache_data(ttl=300, show_spinner=False)
def get_headlines(ticker: str, days_back: int = 7):
    db = load_db()
    since = (date.today() - timedelta(days=days_back)).isoformat()
    rows = db.get_headlines_with_sentiment(ticker, since=since)
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 SentimentRAG")
        st.markdown("*Financial Signal Assistant*")
        st.divider()

        market = st.selectbox("Select Market", options=list(MARKETS.keys()), index=0)
        current_watchlist = MARKETS[market]
        
        ticker = st.selectbox(
            "Select Ticker",
            options=list(current_watchlist.keys()),
            format_func=lambda t: f"{t} — {current_watchlist[t]}",
            index=0,
        )

        days_back = st.slider("Lookback window (days)", 3, 30, 7)

        st.divider()
        st.markdown("**Data sources**")
        st.markdown("- 📰 Google News / Yahoo RSS")
        st.markdown("- 📄 SEC EDGAR 10-K/10-Q")
        st.markdown("- 🤖 FinBERT (local)")
        st.markdown("- ⚡ Groq Llama 3.3-70B")
        st.divider()

        groq_key = get_secret("GROQ_API_KEY")
        if groq_key:
            st.success("Groq API key ✓")
        else:
            st.warning("No GROQ_API_KEY — synthesis disabled")

        faiss_loaded = load_faiss() is not None
        if faiss_loaded:
            st.success("FAISS index ✓")
        else:
            st.warning("FAISS index not built yet")

        st.caption("v1.0 · Free tier · $0 cost")

    return ticker, days_back, current_watchlist


# ---------------------------------------------------------------------------
# Main dashboard sections
# ---------------------------------------------------------------------------


def render_sentiment_overview(ticker: str, days_back: int):
    st.markdown(f"### 📰 News Sentiment — {ticker} (last {days_back} days)")

    summary = get_sentiment_summary(ticker, days_back)

    if not summary:
        st.info(
            f"No sentiment data for **{ticker}** yet. "
            "Run `python scripts/run_ingestion.py --rss-only --tickers "
            f"{ticker}` to fetch headlines."
        )
        return

    # Metric cards
    total = sum(v["count"] for v in summary.values())
    col1, col2, col3, col4 = st.columns(4)

    pos = summary.get("positive", {})
    neg = summary.get("negative", {})
    neu = summary.get("neutral", {})

    with col1:
        st.metric("Total Headlines", total)
    with col2:
        pct = pos.get("count", 0) / total * 100 if total else 0
        st.metric("🟢 Positive", f"{pos.get('count', 0)} ({pct:.0f}%)",
                  delta=f"avg {pos.get('avg_score', 0):.2f}")
    with col3:
        pct = neg.get("count", 0) / total * 100 if total else 0
        st.metric("🔴 Negative", f"{neg.get('count', 0)} ({pct:.0f}%)",
                  delta=f"avg {neg.get('avg_score', 0):.2f}", delta_color="inverse")
    with col4:
        pct = neu.get("count", 0) / total * 100 if total else 0
        st.metric("⚪ Neutral", f"{neu.get('count', 0)} ({pct:.0f}%)")

    # Donut chart
    labels = list(summary.keys())
    values = [v["count"] for v in summary.values()]
    colors = {"positive": "#10b981", "negative": "#ef4444", "neutral": "#a3aed0"}
    chart_colors = [colors.get(l, "#888") for l in labels]

    fig = go.Figure(go.Pie(
        labels=[l.capitalize() for l in labels],
        values=values,
        hole=0.55,
        marker_colors=chart_colors,
        textinfo="label+percent",
        textfont_size=13,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=250,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_headline_table(ticker: str, days_back: int):
    st.markdown("#### Recent Headlines")

    headlines = get_headlines(ticker, days_back)
    if not headlines:
        st.caption("No headlines found for this ticker/period.")
        return

    df = pd.DataFrame(headlines)[["published", "source", "label", "score", "title"]]
    df["published"] = pd.to_datetime(df["published"]).dt.strftime("%Y-%m-%d %H:%M")
    df["score"] = df["score"].map(lambda x: f"{x:.2f}" if x else "—")
    df = df.rename(columns={
        "published": "Published", "source": "Source",
        "label": "Sentiment", "score": "Confidence", "title": "Headline"
    })

    def highlight_sentiment(val):
        colors = {"positive": "color: #10b981", "negative": "color: #ef4444", "neutral": "color: #a3aed0"}
        return colors.get(val, "")

    st.dataframe(
        df.style.map(highlight_sentiment, subset=["Sentiment"]),
        use_container_width=True,
        height=300,
        hide_index=True,
    )


def render_synthesis(ticker: str, days_back: int):
    st.markdown("### 🤖 Divergence Analysis")
    st.caption("Powered by Groq Llama 3.3-70B · Responses cached in SQLite")

    groq_key = get_secret("GROQ_API_KEY")
    if not groq_key:
        st.warning("Set **GROQ_API_KEY** to enable synthesis.")
        return

    if st.button(f"Generate Divergence Report for {ticker}", type="primary"):
        with st.spinner("Routing query → retrieving context → synthesizing…"):
            try:
                from src.agents.synthesis_agent import SynthesisAgent
                agent = SynthesisAgent()
                report = agent.synthesize_from_db(ticker, days_back=days_back)
                _render_report(report)
            except Exception as e:
                st.error(f"Synthesis failed: {e}")


def _render_report(report: dict):
    if "error" in report:
        st.error(report["error"])
        return

    divergence = report.get("divergence_level", "none")
    signal = report.get("sentiment_signal", "neutral")
    div_class = f"divergence-{divergence}"
    sig_class = f"signal-{signal}"

    st.markdown(f"""
    <div class="metric-card {div_class}">
        <h4>Divergence Level: <span style="text-transform:uppercase">{divergence}</span>
        &nbsp;|&nbsp; Signal: <span class="{sig_class}">{signal.upper()}</span></h4>
        <p><strong>Sentiment:</strong> {report.get('sentiment_summary', '—')}</p>
        <p><strong>Fundamentals:</strong> {report.get('fundamentals_summary', '—')}</p>
        <p><strong>Divergence:</strong> {report.get('divergence_assessment', '—')}</p>
    </div>
    """, unsafe_allow_html=True)

    risks = report.get("key_risks", [])
    if risks:
        st.markdown("**Key Risks Identified**")
        for r in risks:
            st.markdown(f"- {r}")

    sources = report.get("data_sources", [])
    if sources:
        st.caption(f"Data sources: {', '.join(sources)}")

    st.markdown(
        f'<p class="disclaimer">{report.get("disclaimer", "")}</p>',
        unsafe_allow_html=True,
    )


def render_filing_context(ticker: str):
    st.markdown("### 📄 SEC Filing Context")

    db = load_db()
    chunks = db.get_filing_chunks(ticker)

    if not chunks:
        st.info(
            f"No filing data for **{ticker}** yet. "
            f"Run `python scripts/run_ingestion.py --edgar-only --tickers {ticker}`"
        )
        return

    # Group by filing_type + fiscal_period
    filings_seen = set()
    options = []
    for row in chunks:
        key = f"{row['filing_type']} — {row['fiscal_period']} ({row['filed_date'][:10]})"
        if key not in filings_seen:
            filings_seen.add(key)
            options.append(key)

    selected = st.selectbox("Filing", options[:10])
    if not selected:
        return

    # Show chunks for selected filing
    parts = selected.split(" — ")
    filing_type = parts[0]
    fiscal_period = parts[1].split(" ")[0]

    filtered = [
        row for row in chunks
        if row["filing_type"] == filing_type and row["fiscal_period"] == fiscal_period
    ]

    sections = {}
    for row in filtered:
        sections.setdefault(row["section"], []).append(row["text"])

    for section, texts in list(sections.items())[:5]:
        with st.expander(f"📑 {section}"):
            for text in texts[:3]:
                st.markdown(f"> {text[:500]}…" if len(text) > 500 else f"> {text}")


def render_backtest(ticker: str):
    st.markdown("### 📈 Sentiment vs. Price Backtesting")
    st.caption("Quantify correlation between news sentiment and N-day forward stock returns using yfinance.")

    col1, col2 = st.columns(2)
    with col1:
        lookback = st.slider("Backtest history (days)", 15, 90, 45, key="bt_lookback")
    with col2:
        forward_days = st.slider("Forward return window (days)", 1, 10, 5, key="bt_forward")

    if st.button("Run Correlation Backtest", type="secondary"):
        with st.spinner("Fetching yfinance price data & processing signals…"):
            try:
                from src.backtest.backtester import Backtester
                backtester = Backtester(load_db())
                df, correlation = backtester.run_backtest(ticker, days_back=lookback, forward_days=forward_days)

                if df.empty:
                    st.warning("No overlapping daily sentiment & stock price data found. Make sure headlines are ingested and scored first.")
                    return

                # Display correlation metric
                st.markdown(f"#### Pearson Correlation Coefficient: `{correlation:.4f}`")
                
                # Dynamic interpretation
                if abs(correlation) < 0.1:
                    st.info("💡 **Interpretation:** Weak/no linear correlation. Sentiment and forward returns move independently.")
                elif correlation >= 0.1:
                    st.success("💡 **Interpretation:** Positive correlation. High/bullish news sentiment historically preceded positive price moves.")
                else:
                    st.warning("💡 **Interpretation:** Negative correlation. High sentiment historically preceded negative price moves (possible contrarian/mean-reversion indicator).")

                # Double axis line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df['sentiment_score'],
                    name="Daily Sentiment Signal", line=dict(color="#4318ff", width=3),
                    yaxis="y1"
                ))
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df['Close'],
                    name="Close Price ($)", line=dict(color="#05cd99", width=2, dash='dot'),
                    yaxis="y2"
                ))

                fig.update_layout(
                    title=f"{ticker} Sentiment vs Close Price over last {lookback} days",
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Sentiment Signal (-1 to +1)", title_font=dict(color="#4318ff"), tickfont=dict(color="#4318ff")),
                    yaxis2=dict(title="Stock Close Price ($)", title_font=dict(color="#05cd99"), tickfont=dict(color="#05cd99"), anchor="x", overlaying="y", side="right"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)"),
                    margin=dict(t=50, b=50, l=50, r=50),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Return scatter plot
                fig_scatter = px.scatter(
                    df, x="sentiment_score", y="forward_return",
                    labels={"sentiment_score": "Sentiment Score", "forward_return": f"{forward_days}-Day Forward Return"},
                    title="Returns vs Sentiment Score Scatter Plot",
                    trendline="ols",
                    color_discrete_sequence=["#4318ff"]
                )
                fig_scatter.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=50, b=50, l=50, r=50),
                    height=350
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            except Exception as e:
                st.error(f"Backtest execution failed: {e}")


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main():
    ticker, days_back, current_watchlist = render_sidebar()

    st.markdown(f"## 📊 {ticker} Dashboard")
    st.caption(f"{current_watchlist.get(ticker, ticker)} · Updated {date.today().isoformat()}")
    st.markdown("<br>", unsafe_allow_html=True)

    # Top Row: KPIs
    render_sentiment_overview(ticker, days_back)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Middle Row: Main Chart + Right Side Widgets
    col_main, col_side = st.columns([2.5, 1])
    with col_main:
        render_backtest(ticker)
        st.markdown("<br>", unsafe_allow_html=True)
        render_headline_table(ticker, days_back)
        
    with col_side:
        render_synthesis(ticker, days_back)
        st.markdown("<br>", unsafe_allow_html=True)
        render_filing_context(ticker)


if __name__ == "__main__":
    main()
