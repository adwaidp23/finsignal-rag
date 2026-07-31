"""
app/streamlit_app.py
---------------------
FinSignal-RAG — Financial Sentiment + RAG Trading Signal Dashboard.
Design: "Ink Terminal" — dark navy, JetBrains Mono numerals, WCAG AA contrast.
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
    page_title="FinSignal RAG — Financial Research Terminal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design System — "Ink Terminal"
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* TOKENS */
:root {
  --ink-deep:       #0A0F1E;
  --ink-surface:    #111827;
  --ink-card:       #161D2E;
  --ink-border:     #1E2D40;
  --accent:         #00C2A8;
  --accent-dim:     rgba(0,194,168,0.12);
  --accent-blue:    #3B82F6;
  --text-primary:   #F1F5F9;
  --text-secondary: #94A3B8;
  --text-muted:     #475569;
  --pos:   #10B981;  --pos-dim: rgba(16,185,129,0.12);
  --neg:   #F43F5E;  --neg-dim: rgba(244,63,94,0.12);
  --neu:   #6B7280;
  --warn:  #F59E0B;
}

/* GLOBAL */
html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--ink-deep) !important;
  color: var(--text-primary) !important;
}

/* Dot-grid signature texture */
.stApp::before {
  content: '';
  position: fixed; top:0; left:0; right:0; bottom:0;
  background-image: radial-gradient(circle, #1E2D40 1px, transparent 1px);
  background-size: 28px 28px;
  opacity: 0.35;
  pointer-events: none;
  z-index: 0;
}

/* STREAMLIT CHROME */
header, footer, #MainMenu { visibility: hidden !important; height: 0 !important; }
.block-container {
  padding: 1.8rem 2.2rem 3rem 2.2rem !important;
  max-width: 100% !important;
  position: relative; z-index: 1;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
  background: var(--ink-surface) !important;
  border-right: 1px solid var(--ink-border) !important;
  position: relative; z-index: 10;
}
[data-testid="stSidebar"] .block-container { padding: 1.8rem 1.2rem !important; }

.sidebar-logo {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 1.6rem; padding-bottom: 1.2rem;
  border-bottom: 1px solid var(--ink-border);
}
.sidebar-logo-icon {
  font-size: 1.4rem; width: 40px; height: 40px;
  background: var(--accent-dim); border: 1px solid var(--accent);
  border-radius: 10px; display: flex; align-items: center; justify-content: center;
}
.sidebar-logo-name { font-size: 1rem; font-weight: 800; color: var(--text-primary) !important; letter-spacing: -0.01em; line-height: 1; }
.sidebar-logo-tag  { font-size: 0.68rem; color: var(--accent) !important; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; }

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
  font-size: 0.68rem !important; font-weight: 700 !important;
  letter-spacing: 0.1em !important; text-transform: uppercase !important;
  color: var(--text-secondary) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: var(--ink-card) !important; border: 1px solid var(--ink-border) !important;
  border-radius: 8px !important; color: var(--text-primary) !important; font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(0,194,168,0.2) !important;
}

.status-row {
  display: flex; align-items: center; gap: 8px;
  padding: 0.45rem 0; font-size: 0.78rem; color: var(--text-secondary); font-weight: 500;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.status-dot-live  { background: var(--pos); box-shadow: 0 0 6px var(--pos); }
.status-dot-idle  { background: var(--neu); }
.status-dot-error { background: var(--neg); box-shadow: 0 0 6px var(--neg); }
.status-label { color: var(--text-primary) !important; }
.sidebar-divider { height: 1px; background: var(--ink-border); margin: 1.2rem 0; }

/* MASTHEAD */
.masthead {
  display: flex; align-items: flex-start; justify-content: space-between;
  margin-bottom: 2rem; padding-bottom: 1.4rem; border-bottom: 1px solid var(--ink-border);
}
.masthead-ticker {
  font-family: 'JetBrains Mono', monospace;
  font-size: 2rem; font-weight: 700; color: var(--text-primary);
  line-height: 1; letter-spacing: -0.02em;
}
.masthead-company { font-size: 0.82rem; color: var(--text-secondary); margin-top: 0.3rem; font-weight: 500; }
.masthead-right { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }
.masthead-date {
  font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--text-secondary);
  background: var(--ink-card); border: 1px solid var(--ink-border); border-radius: 8px;
  padding: 0.35rem 0.9rem; font-weight: 500;
}
.masthead-live-badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.67rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--pos); background: var(--pos-dim); border: 1px solid rgba(16,185,129,0.25);
  border-radius: 20px; padding: 0.2rem 0.7rem;
}
.masthead-live-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--pos); animation: blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
@media (prefers-reduced-motion: reduce) { .masthead-live-dot { animation: none; } }

/* CARDS */
.ink-card {
  background: var(--ink-card); border: 1px solid var(--ink-border);
  border-radius: 16px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
  position: relative; overflow: hidden;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.ink-card::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,194,168,0.3), transparent);
}
.ink-card:hover { border-color: rgba(0,194,168,0.25); box-shadow: 0 0 30px rgba(0,194,168,0.06); }
@media (prefers-reduced-motion: reduce) { .ink-card { transition: none; } }

.card-title { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.25rem; }
.card-heading { font-size: 1.05rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.1rem; }
.card-sub { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.9rem; line-height: 1.5; }

/* KPI METRICS */
[data-testid="stMetric"] {
  background: var(--ink-card) !important; border: 1px solid var(--ink-border) !important;
  border-radius: 14px !important; padding: 1.1rem 1.3rem !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stMetric"]:hover { border-color: rgba(0,194,168,0.3) !important; box-shadow: 0 0 20px rgba(0,194,168,0.05) !important; }
[data-testid="stMetricLabel"] p { font-size: 0.67rem !important; font-weight: 700 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: var(--text-secondary) !important; }
[data-testid="stMetricValue"]   { font-family: 'JetBrains Mono', monospace !important; font-size: 1.75rem !important; font-weight: 700 !important; color: var(--text-primary) !important; }
.delta-hint { font-size: 0.63rem; color: var(--text-muted); font-family: 'Inter', sans-serif; margin-top: 0.15rem; }

/* BUTTONS */
.stButton > button {
  border-radius: 8px !important; font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important; font-size: 0.82rem !important; padding: 0.55rem 1.4rem !important;
  border: 1px solid var(--accent) !important; background: transparent !important;
  color: var(--accent) !important; letter-spacing: 0.02em;
  transition: all 0.18s ease !important;
}
.stButton > button:hover { background: var(--accent-dim) !important; box-shadow: 0 0 16px rgba(0,194,168,0.2) !important; transform: translateY(-1px) !important; }
.stButton > button:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 3px !important; }
.stButton > button[kind="primary"] { background: var(--accent) !important; color: var(--ink-deep) !important; font-weight: 700 !important; }
.stButton > button[kind="primary"]:hover { background: #00d4b8 !important; box-shadow: 0 4px 20px rgba(0,194,168,0.35) !important; }

/* SKELETON SHIMMER */
@keyframes shimmer {
  0%   { background-position: -600px 0; }
  100% { background-position:  600px 0; }
}
.skeleton {
  border-radius: 8px;
  background: linear-gradient(90deg, #161D2E 25%, #1E2D40 50%, #161D2E 75%);
  background-size: 600px 100%;
  animation: shimmer 1.4s ease-in-out infinite;
}
@media (prefers-reduced-motion: reduce) { .skeleton { animation: none; background: #1E2D40; } }
.skeleton-text-sm  { height: 12px; width: 60%; margin-bottom: 8px; }
.skeleton-text-md  { height: 14px; width: 80%; margin-bottom: 8px; }
.skeleton-text-lg  { height: 36px; width: 45%; margin-bottom: 12px; }
.skeleton-rect     { height: 140px; width: 100%; margin-bottom: 12px; }
.skeleton-donut    { height: 200px; width: 200px; border-radius: 50%; margin: 0 auto 12px; }
.skeleton-bar      { height: 220px; width: 100%; }
.skeleton-row      { display: flex; gap: 12px; margin-bottom: 16px; }
.skeleton-col      { flex: 1; }

/* STEP STATUS (st.status override) */
[data-testid="stStatusWidget"] {
  background: var(--ink-card) !important;
  border: 1px solid var(--ink-border) !important;
  border-radius: 12px !important;
  color: var(--text-secondary) !important;
}

/* SPINNER override */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* DIVERGENCE */
.metric-card { background: var(--ink-card); border: 1px solid var(--ink-border); border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
.divergence-high   { border-left: 4px solid var(--neg); }
.divergence-medium { border-left: 4px solid var(--warn); }
.divergence-low    { border-left: 4px solid var(--pos); }
.divergence-none   { border-left: 4px solid var(--accent); }
.signal-bullish { color: var(--pos) !important; font-weight: 700; }
.signal-bearish { color: var(--neg) !important; font-weight: 700; }
.signal-mixed   { color: var(--warn) !important; font-weight: 700; }
.signal-neutral { color: var(--text-secondary) !important; font-weight: 600; }

/* DIVERGENCE ARROWS */
.div-arrows { display: flex; gap: 1rem; margin: 1rem 0; }
.div-arrow-card { flex: 1; background: var(--ink-surface); border: 1px solid var(--ink-border); border-radius: 10px; padding: 0.9rem 1rem; text-align: center; }
.div-arrow-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.3rem; }
.div-arrow-icon  { font-size: 1.6rem; line-height: 1; }
.div-arrow-text  { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 600; margin-top: 0.25rem; }
.divergence-flag { display: flex; align-items: center; gap: 8px; background: rgba(244,63,94,0.1); border: 1px solid rgba(244,63,94,0.25); border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.78rem; color: var(--neg); font-weight: 600; margin-top: 0.5rem; }
.align-flag { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.2); color: var(--pos); }

/* EMPTY STATES */
.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 3rem 2rem; text-align: center; }
.empty-icon  { font-size: 2.2rem; margin-bottom: 0.8rem; opacity: 0.5; }
.empty-title { font-size: 0.9rem; font-weight: 700; color: var(--text-secondary); margin-bottom: 0.4rem; }
.empty-body  { font-size: 0.78rem; color: var(--text-muted); max-width: 280px; line-height: 1.5; }
.empty-cmd   { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; background: var(--ink-surface); border: 1px solid var(--ink-border); border-radius: 6px; padding: 0.3rem 0.7rem; color: var(--accent); margin-top: 0.8rem; display: inline-block; }

/* TABLES */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; border: 1px solid var(--ink-border) !important; }

/* EXPANDERS */
[data-testid="stExpander"] details { background: var(--ink-surface) !important; border: 1px solid var(--ink-border) !important; border-radius: 10px !important; margin-bottom: 0.5rem !important; }
[data-testid="stExpander"] summary { color: var(--text-primary) !important; font-weight: 600 !important; font-size: 0.85rem !important; padding: 0.75rem 1rem !important; }
[data-testid="stExpander"] summary:hover { color: var(--accent) !important; }
[data-testid="stExpander"] summary:focus-visible { outline: 2px solid var(--accent) !important; }

/* ALERTS */
[data-testid="stAlert"] { border-radius: 10px !important; border: none !important; background: var(--ink-surface) !important; }

.disclaimer { font-size: 0.68rem; color: var(--text-muted); border-top: 1px solid var(--ink-border); padding-top: 0.7rem; margin-top: 1rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Markets data
# ---------------------------------------------------------------------------

MARKETS = {
    "NASDAQ": {
        "AAPL": "Apple Inc.", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
        "META": "Meta Platforms", "NVDA": "NVIDIA", "TSLA": "Tesla", "AVGO": "Broadcom",
        "PEP": "PepsiCo", "COST": "Costco", "CSCO": "Cisco", "TMUS": "T-Mobile",
        "ADBE": "Adobe", "TXN": "Texas Instruments", "NFLX": "Netflix", "CMCSA": "Comcast",
        "AMD": "Advanced Micro Devices", "INTU": "Intuit", "QCOM": "Qualcomm", "AMGN": "Amgen",
        "HON": "Honeywell", "INTC": "Intel", "ISRG": "Intuitive Surgical", "GILD": "Gilead Sciences",
        "SBUX": "Starbucks", "BKNG": "Booking Holdings", "VRTX": "Vertex", "MDLZ": "Mondelez",
        "REGN": "Regeneron", "ADI": "Analog Devices", "ADP": "Automatic Data Processing",
        "PANW": "Palo Alto Networks", "MU": "Micron Technology", "SNPS": "Synopsys",
        "KLAC": "KLA Corp", "MELI": "MercadoLibre", "CDNS": "Cadence Design", "CSX": "CSX Corp",
        "PYPL": "PayPal", "MAR": "Marriott", "MNST": "Monster Beverage", "ASML": "ASML Holding",
        "ORLY": "O'Reilly Automotive", "CTAS": "Cintas", "LRCX": "Lam Research",
        "NXPI": "NXP Semiconductors", "FTNT": "Fortinet", "KDP": "Keurig Dr Pepper",
        "PAYX": "Paychex", "PCAR": "PACCAR",
    },
    "NYSE": {
        "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase", "V": "Visa",
        "JNJ": "Johnson & Johnson", "WMT": "Walmart", "PG": "Procter & Gamble",
        "MA": "Mastercard", "HD": "Home Depot", "CVX": "Chevron", "MRK": "Merck",
        "ABBV": "AbbVie", "KO": "Coca-Cola", "BAC": "Bank of America", "PFE": "Pfizer",
        "TMO": "Thermo Fisher", "DIS": "Walt Disney", "MCD": "McDonald's", "CRM": "Salesforce",
        "ACN": "Accenture", "DHR": "Danaher", "ABT": "Abbott Labs", "LIN": "Linde",
        "WFC": "Wells Fargo", "NKE": "NIKE", "PM": "Philip Morris", "NEE": "NextEra Energy",
        "RTX": "RTX Corp", "MS": "Morgan Stanley", "UPS": "United Parcel Service", "BA": "Boeing",
        "UNP": "Union Pacific", "IBM": "IBM", "C": "Citigroup", "BLK": "BlackRock",
        "GS": "Goldman Sachs", "LMT": "Lockheed Martin", "DE": "Deere & Co", "SYK": "Stryker",
        "GE": "General Electric", "AMT": "American Tower", "T": "AT&T", "MMM": "3M",
        "CVS": "CVS Health", "MO": "Altria", "SPGI": "S&P Global", "PLD": "Prologis",
        "CAT": "Caterpillar", "CB": "Chubb", "CI": "Cigna", "TJX": "TJX Companies",
    },
    "Shanghai": {
        "600519.SS": "Kweichow Moutai", "601398.SS": "ICBC", "601288.SS": "Agricultural Bank of China",
        "601939.SS": "CCB", "601857.SS": "PetroChina", "601988.SS": "Bank of China",
        "600036.SS": "China Merchants Bank", "601088.SS": "China Shenhua Energy",
        "601628.SS": "China Life Insurance", "600900.SS": "China Yangtze Power",
        "601318.SS": "Ping An Insurance", "600028.SS": "Sinopec", "601166.SS": "Industrial Bank",
        "601328.SS": "Bank of Communications", "601816.SS": "CGN Power",
        "600030.SS": "CITIC Securities", "603259.SS": "WuXi AppTec",
        "600276.SS": "Hengrui Medicine", "600438.SS": "Tongwei", "600000.SS": "SPDB",
        "600887.SS": "Yili", "601888.SS": "China Tourism", "600031.SS": "Sany Heavy",
        "600104.SS": "SAIC Motor", "601138.SS": "Foxconn Industrial",
        "600690.SS": "Haier Smart Home", "601899.SS": "Zijin Mining",
        "600048.SS": "Poly Developments", "601012.SS": "LONGi", "603993.SS": "China Molybdenum",
        "600809.SS": "Shanxi Fen Wine", "601211.SS": "Guotai Junan", "600018.SS": "SIPG",
        "601800.SS": "China Comm Construction", "601668.SS": "China State Construction",
        "601390.SS": "China Railway", "601111.SS": "Air China", "601006.SS": "Daqin Railway",
        "600016.SS": "China Minsheng Bank", "601229.SS": "Bank of Shanghai",
        "601998.SS": "China CITIC Bank", "600019.SS": "Baoshan Iron & Steel",
        "601989.SS": "China Shipbuilding", "600111.SS": "China Northern Rare Earth",
        "601225.SS": "Shaanxi Coal", "600009.SS": "Shanghai Intl Airport",
        "601901.SS": "Founder Securities", "601688.SS": "Huatai Securities",
        "601878.SS": "Zheshang Securities", "600999.SS": "China Merchants Securities",
    },
    "Euronext": {
        "MC.PA": "LVMH", "RMS.PA": "Hermes International", "OR.PA": "L'Oreal",
        "SU.PA": "Schneider Electric", "TTE.PA": "TotalEnergies", "AIR.PA": "Airbus",
        "SAN.PA": "Sanofi", "SAF.PA": "Safran", "AI.PA": "Air Liquide",
        "EL.PA": "EssilorLuxottica", "BN.PA": "Danone", "CS.PA": "AXA",
        "BNP.PA": "BNP Paribas", "VINC.PA": "Vinci", "CAP.PA": "Capgemini",
        "ENGI.PA": "Engie", "GLE.PA": "Societe Generale", "SGO.PA": "Saint-Gobain",
        "ORA.PA": "Orange", "ACA.PA": "Credit Agricole", "STLAP.PA": "Stellantis",
        "PUB.PA": "Publicis Groupe", "MICP.PA": "Michelin", "LR.PA": "Legrand",
        "VE.PA": "Veolia", "EN.PA": "Bouygues", "HO.PA": "Thales", "RI.PA": "Pernod Ricard",
        "ASML.AS": "ASML Holding", "AD.AS": "Ahold Delhaize", "HEIA.AS": "Heineken",
        "INGA.AS": "ING Groep", "PRX.AS": "Prosus", "AKZA.AS": "Akzo Nobel",
        "URW.AS": "Unibail-Rodamco", "MT.AS": "ArcelorMittal", "WKL.AS": "Wolters Kluwer",
        "KPN.AS": "KPN", "NN.AS": "NN Group", "RAND.AS": "Randstad", "PHIA.AS": "Philips",
        "ABI.BR": "Anheuser-Busch InBev", "BPOST.BR": "Bpost", "UCB.BR": "UCB",
        "SOLB.BR": "Solvay", "KBC.BR": "KBC Group", "PROX.BR": "Proximus",
        "AGS.BR": "Ageas", "ARGX.BR": "Argenx", "GBLB.BR": "Groupe Bruxelles Lambert",
    },
    "NSE": {
        "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services",
        "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank",
        "BHARTIARTL.NS": "Bharti Airtel", "SBIN.NS": "State Bank of India",
        "INFY.NS": "Infosys", "ITC.NS": "ITC Limited",
        "HINDUNILVR.NS": "Hindustan Unilever", "LT.NS": "Larsen & Toubro",
        "BAJFINANCE.NS": "Bajaj Finance", "AXISBANK.NS": "Axis Bank",
        "HCLTECH.NS": "HCL Technologies", "KOTAKBANK.NS": "Kotak Mahindra Bank",
        "MARUTI.NS": "Maruti Suzuki", "SUNPHARMA.NS": "Sun Pharmaceuticals",
        "TATAMOTORS.NS": "Tata Motors", "TATASTEEL.NS": "Tata Steel",
        "NTPC.NS": "NTPC Limited", "ULTRACEMCO.NS": "UltraTech Cement",
        "M&M.NS": "Mahindra & Mahindra", "POWERGRID.NS": "Power Grid Corp",
        "ASIANPAINT.NS": "Asian Paints", "TITAN.NS": "Titan Company",
        "BAJAJFINSV.NS": "Bajaj Finserv", "ONGC.NS": "ONGC",
        "NESTLEIND.NS": "Nestle India", "WIPRO.NS": "Wipro",
        "ADANIENT.NS": "Adani Enterprises", "ADANIPORTS.NS": "Adani Ports",
        "COALINDIA.NS": "Coal India", "HINDALCO.NS": "Hindalco Industries",
        "GRASIM.NS": "Grasim Industries", "JSWSTEEL.NS": "JSW Steel",
        "TECHM.NS": "Tech Mahindra", "DRREDDY.NS": "Dr. Reddy's Labs",
        "INDUSINDBK.NS": "IndusInd Bank", "CIPLA.NS": "Cipla",
        "APOLLOHOSP.NS": "Apollo Hospitals", "EICHERMOT.NS": "Eicher Motors",
        "DIVISLAB.NS": "Divi's Laboratories", "HDFCLIFE.NS": "HDFC Life",
        "SBILIFE.NS": "SBI Life Insurance", "LTIM.NS": "LTIMindtree",
        "HEROMOTOCO.NS": "Hero MotoCorp", "BAJAJ-AUTO.NS": "Bajaj Auto",
        "BRITANNIA.NS": "Britannia Industries", "TATACONSUM.NS": "Tata Consumer Products",
        "BPCL.NS": "Bharat Petroleum", "TRENT.NS": "Trent Limited",
    },
}

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="⏳ Connecting to database…")
def load_db():
    return Database(DB_PATH)


@st.cache_resource(show_spinner="⏳ Loading FAISS vector index…")
def load_faiss():
    try:
        from src.embeddings.faiss_builder import FAISSBuilder
        builder = FAISSBuilder()
        return builder.load_index()
    except FileNotFoundError:
        return None


def _skeleton_kpi():
    """Renders 4 KPI skeleton cards while sentiment data loads."""
    cols = st.columns(4)
    for col in cols:
        with col:
            st.markdown("""
            <div class="ink-card" style="padding:1.1rem 1.3rem;">
              <div class="skeleton skeleton-text-sm" style="width:55%;"></div>
              <div class="skeleton skeleton-text-lg"></div>
              <div class="skeleton skeleton-text-sm" style="width:40%;"></div>
            </div>
            """, unsafe_allow_html=True)


def _skeleton_chart():
    """Renders a shimmer placeholder for chart areas."""
    st.markdown("""
    <div style="padding:0.5rem 0;">
      <div class="skeleton skeleton-text-sm" style="width:30%;margin-bottom:6px;"></div>
      <div class="skeleton skeleton-text-md" style="width:55%;margin-bottom:16px;"></div>
      <div class="skeleton skeleton-bar"></div>
    </div>
    """, unsafe_allow_html=True)


def _skeleton_table():
    """Renders shimmer rows for table areas."""
    for _ in range(5):
        st.markdown("""
        <div class="skeleton-row">
          <div class="skeleton skeleton-col" style="height:14px;max-width:110px;"></div>
          <div class="skeleton skeleton-col" style="height:14px;max-width:70px;"></div>
          <div class="skeleton skeleton-col" style="height:14px;max-width:60px;"></div>
          <div class="skeleton skeleton-col" style="height:14px;"></div>
        </div>
        """, unsafe_allow_html=True)


@st.cache_data(ttl=15, show_spinner=False)
def get_sentiment_summary(ticker: str, days_back: int = 7):
    db = load_db()
    since = (date.today() - timedelta(days=days_back)).isoformat()
    summary = db.get_sentiment_summary(ticker, since=since)
    if not summary:
        summary = db.get_sentiment_summary(ticker, since=None)
    return summary


@st.cache_data(ttl=15, show_spinner=False)
def get_headlines(ticker: str, days_back: int = 7):
    db = load_db()
    since = (date.today() - timedelta(days=days_back)).isoformat()
    rows = db.get_headlines_with_sentiment(ticker, since=since)
    if not rows:
        rows = db.get_headlines_with_sentiment(ticker, since=None)
    return [dict(r) for r in rows]


@st.cache_data(ttl=15, show_spinner=False)
def get_filing_chunks_cached(ticker: str):
    db = load_db()
    return db.get_filing_chunks(ticker)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
          <div class="sidebar-logo-icon">📡</div>
          <div>
            <div class="sidebar-logo-name">FinSignal RAG</div>
            <div class="sidebar-logo-tag">Research Terminal</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        market = st.selectbox("Select Market", options=list(MARKETS.keys()), index=0)
        current_watchlist = MARKETS[market]

        ticker = st.selectbox(
            "Select Ticker",
            options=list(current_watchlist.keys()),
            format_func=lambda t: f"{t}  —  {current_watchlist[t]}",
            index=0,
        )

        days_back = st.slider("Lookback Window (days)", 3, 30, 7)

        if st.button("🔄 Refresh Data", use_container_width=True, help="Clear cache and reload latest data"):
            st.cache_data.clear()
            st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Data Sources</div>', unsafe_allow_html=True)

        groq_key = get_secret("GROQ_API_KEY")
        faiss_ok = load_faiss() is not None

        indicators = [
            ("Google News / Yahoo RSS", "live"),
            ("SEC EDGAR 10-K / 10-Q", "live"),
            ("FinBERT (local)", "live"),
            ("Groq Llama 3.3-70B", "live" if groq_key else "error"),
            ("FAISS Vector Index", "live" if faiss_ok else "idle"),
        ]
        for label, status in indicators:
            st.markdown(f"""
            <div class="status-row">
              <div class="status-dot status-dot-{status}"></div>
              <span class="status-label">{label}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.65rem;color:#475569;">v2.0 · Free Tier · $0 cost · {date.today().isoformat()}</div>',
            unsafe_allow_html=True
        )

    return ticker, days_back, current_watchlist


# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------

def render_kpi_row(ticker: str, days_back: int):
    summary = get_sentiment_summary(ticker, days_back)

    if not summary:
        st.markdown(f"""
        <div class="empty-state">
          <div class="empty-icon">📭</div>
          <div class="empty-title">No Sentiment Data</div>
          <div class="empty-body">No headlines found for <strong>{ticker}</strong> in the last {days_back} days.
          Run ingestion to populate this section.</div>
          <code class="empty-cmd">python scripts/run_ingestion.py --rss-only --tickers {ticker}</code>
        </div>
        """, unsafe_allow_html=True)
        return

    total = sum(v["count"] for v in summary.values())
    pos = summary.get("positive", {})
    neg = summary.get("negative", {})
    neu = summary.get("neutral", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Headlines", total)
        st.markdown('<div class="delta-hint">Across all sentiment classes</div>', unsafe_allow_html=True)
    with c2:
        pct = pos.get("count", 0) / total * 100 if total else 0
        st.metric("Positive", f"{pos.get('count', 0)}",
                  delta=f"↑ {pct:.0f}%  ·  avg conf {pos.get('avg_score', 0):.2f}")
        st.markdown('<div class="delta-hint">Avg FinBERT confidence score</div>', unsafe_allow_html=True)
    with c3:
        pct = neg.get("count", 0) / total * 100 if total else 0
        st.metric("Negative", f"{neg.get('count', 0)}",
                  delta=f"↓ {pct:.0f}%  ·  avg conf {neg.get('avg_score', 0):.2f}", delta_color="inverse")
        st.markdown('<div class="delta-hint">Avg FinBERT confidence score</div>', unsafe_allow_html=True)
    with c4:
        pct = neu.get("count", 0) / total * 100 if total else 0
        st.metric("Neutral", f"{neu.get('count', 0)}", delta=f"{pct:.0f}% of total")
        st.markdown('<div class="delta-hint">No directional signal</div>', unsafe_allow_html=True)

    # Donut with center count label
    labels = list(summary.keys())
    values = [v["count"] for v in summary.values()]
    colors_map = {"positive": "#10B981", "negative": "#F43F5E", "neutral": "#6B7280"}
    chart_colors = [colors_map.get(l, "#6B7280") for l in labels]

    fig = go.Figure(go.Pie(
        labels=[l.capitalize() for l in labels],
        values=values,
        hole=0.62,
        marker=dict(colors=chart_colors, line=dict(color="#161D2E", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#F1F5F9"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b style='font-size:22px'>{total}</b><br>articles",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#F1F5F9", family="JetBrains Mono"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color="#94A3B8", size=12), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.1),
        margin=dict(t=10, b=10, l=10, r=10), height=240,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Headline Table
# ---------------------------------------------------------------------------

def render_headline_table(ticker: str, days_back: int):
    headlines = get_headlines(ticker, days_back)
    if not headlines:
        st.markdown(f"""
        <div class="empty-state">
          <div class="empty-icon">📰</div>
          <div class="empty-title">No Headlines Found</div>
          <div class="empty-body">No news articles for <strong>{ticker}</strong> in this window.
          Headline sentiment will appear here automatically after ingestion.</div>
          <code class="empty-cmd">python scripts/run_ingestion.py --rss-only --tickers {ticker}</code>
        </div>
        """, unsafe_allow_html=True)
        return

    df = pd.DataFrame(headlines)[["published", "source", "label", "score", "title"]]
    df["published"] = pd.to_datetime(df["published"]).dt.strftime("%Y-%m-%d %H:%M")
    df["score"] = df["score"].map(lambda x: f"{x:.2f}" if x else "—")
    df = df.rename(columns={
        "published": "Published", "source": "Source",
        "label": "Sentiment", "score": "Confidence", "title": "Headline"
    })

    def highlight_sentiment(val):
        m = {
            "positive": "color: #10B981; font-weight: 600",
            "negative": "color: #F43F5E; font-weight: 600",
            "neutral":  "color: #6B7280",
        }
        return m.get(val, "")

    st.dataframe(
        df.style.map(highlight_sentiment, subset=["Sentiment"]),
        use_container_width=True, height=280, hide_index=True,
    )


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def render_backtest(ticker: str):
    st.markdown('<div class="card-title">Backtesting Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">Sentiment vs. Forward Price Return</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Pearson correlation between FinBERT daily signal and N-day forward stock return via yfinance</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        lookback = st.slider("History (days)", 15, 90, 45, key="bt_lookback")
    with c2:
        forward_days = st.slider("Forward return window (days)", 1, 10, 5, key="bt_forward")

    if st.button("▶  Run Backtest", type="primary", key="run_backtest"):
        with st.spinner("Fetching price data & computing correlation…"):
            try:
                from src.backtest.backtester import Backtester
                backtester = Backtester(load_db())
                df, correlation = backtester.run_backtest(ticker, days_back=lookback, forward_days=forward_days)

                if df.empty:
                    st.markdown(f"""
                    <div class="empty-state">
                      <div class="empty-icon">📈</div>
                      <div class="empty-title">Insufficient Data</div>
                      <div class="empty-body">No overlapping daily sentiment and price data for <strong>{ticker}</strong> yet.
                      Backtest results appear here once at least 2 weeks of headlines accumulate.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    return

                corr_color = "#10B981" if correlation >= 0.1 else "#F43F5E" if correlation <= -0.1 else "#6B7280"
                interp = (
                    "Positive correlation — bullish sentiment historically preceded price gains."
                    if correlation >= 0.1 else
                    "Negative correlation — contrarian / mean-reversion indicator."
                    if correlation <= -0.1 else
                    "Weak linear signal — sentiment and returns are independent for now."
                )
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1rem;padding:1rem;
                            background:rgba(0,0,0,0.2);border-radius:10px;border:1px solid #1E2D40;">
                  <div style="font-family:'JetBrains Mono',monospace;font-size:2.2rem;font-weight:700;color:{corr_color};">{correlation:+.4f}</div>
                  <div style="font-size:0.78rem;color:#94A3B8;max-width:300px;line-height:1.5;">
                    <strong style="color:#F1F5F9;">Pearson r</strong><br>{interp}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df['sentiment_score'], name="Sentiment Signal",
                    line=dict(color="#00C2A8", width=2.5), yaxis="y1",
                    hovertemplate="<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df['Close'], name="Close Price ($)",
                    line=dict(color="#F59E0B", width=2, dash="dot"), yaxis="y2",
                    hovertemplate="<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>",
                ))
                fig.update_layout(
                    xaxis=dict(gridcolor="#1E2D40", tickfont=dict(color="#94A3B8")),
                    yaxis=dict(title="Sentiment Score", title_font=dict(color="#00C2A8"), tickfont=dict(color="#00C2A8"), gridcolor="#1E2D40"),
                    yaxis2=dict(title="Close Price ($)", title_font=dict(color="#F59E0B"), tickfont=dict(color="#F59E0B"), anchor="x", overlaying="y", side="right"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(22,29,46,0.95)", font=dict(color="#94A3B8")),
                    margin=dict(t=10, b=40, l=50, r=60), height=340, hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.scatter(df, x="sentiment_score", y="forward_return", trendline="ols",
                                  labels={"sentiment_score": "Sentiment Score", "forward_return": f"{forward_days}d Return"},
                                  color_discrete_sequence=["#00C2A8"])
                fig2.update_traces(marker=dict(size=7, opacity=0.7))
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94A3B8"),
                    xaxis=dict(gridcolor="#1E2D40", tickfont=dict(color="#94A3B8")),
                    yaxis=dict(gridcolor="#1E2D40", tickfont=dict(color="#94A3B8")),
                    margin=dict(t=10, b=40, l=50, r=20), height=280,
                )
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Backtest failed: {e}")
    else:
        st.markdown(f"""
        <div class="empty-state">
          <div class="empty-icon">📊</div>
          <div class="empty-title">Ready to Backtest</div>
          <div class="empty-body">Configure the lookback window above, then click <strong>Run Backtest</strong>.
          Results show sentiment signal correlated against {ticker}'s forward price return.</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Divergence / Synthesis
# ---------------------------------------------------------------------------

def render_synthesis(ticker: str, days_back: int):
    st.markdown('<div class="card-title">AI Synthesis</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">Divergence Signal</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">News sentiment vs. SEC 10-K fundamentals</div>', unsafe_allow_html=True)

    groq_key = get_secret("GROQ_API_KEY")
    if not groq_key:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🔑</div>
          <div class="empty-title">API Key Required</div>
          <div class="empty-body">Set <strong>GROQ_API_KEY</strong> in your .env file to enable the Llama 3.3-70B synthesis agent.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    if st.button(f"⚡  Analyze {ticker}", type="primary", key="run_synthesis"):
        with st.spinner("Routing → retrieving context → synthesizing…"):
            try:
                from src.agents.synthesis_agent import SynthesisAgent
                agent = SynthesisAgent()
                report = agent.synthesize_from_db(ticker, days_back=days_back)
                _render_report(report)
            except Exception as e:
                st.error(f"Synthesis failed: {e}")
    else:
        st.markdown(f"""
        <div class="empty-state">
          <div class="empty-icon">🤖</div>
          <div class="empty-title">Analysis Ready</div>
          <div class="empty-body">Click <strong>Analyze {ticker}</strong> to run the two-tier Llama 3.3-70B agent.
          It cross-references news sentiment against the latest 10-K and flags divergences.</div>
        </div>
        """, unsafe_allow_html=True)


def _render_report(report: dict):
    if "error" in report:
        st.error(report["error"])
        return

    divergence = report.get("divergence_level", "none")
    signal = report.get("sentiment_signal", "neutral")
    div_class = f"divergence-{divergence}"

    sent_arrow = "↑" if signal == "bullish" else "↓" if signal == "bearish" else "→"
    sent_color = "#10B981" if signal == "bullish" else "#F43F5E" if signal == "bearish" else "#6B7280"
    fund_arrow = "↑" if divergence in ("low", "none") else "↓"
    fund_color = "#10B981" if divergence in ("low", "none") else "#F43F5E"
    flagged = divergence in ("high", "medium")

    st.markdown(f"""
    <div class="div-arrows">
      <div class="div-arrow-card">
        <div class="div-arrow-label">News Sentiment</div>
        <div class="div-arrow-icon" style="color:{sent_color};">{sent_arrow}</div>
        <div class="div-arrow-text" style="color:{sent_color};">{signal.upper()}</div>
      </div>
      <div class="div-arrow-card">
        <div class="div-arrow-label">Fundamentals</div>
        <div class="div-arrow-icon" style="color:{fund_color};">{fund_arrow}</div>
        <div class="div-arrow-text" style="color:{fund_color};">10-K FILING</div>
      </div>
    </div>
    <div class="{'divergence-flag' if flagged else 'divergence-flag align-flag'}">
      {"⚠️  Divergence Flagged — " + divergence.upper() if flagged else "✅  Signals Aligned — No Divergence"}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card {div_class}" style="margin-top:1rem;">
      <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
                  color:#475569;margin-bottom:0.7rem;">Assessment</div>
      <p style="font-size:0.8rem;color:#94A3B8;margin:0.35rem 0;line-height:1.5;">
        <strong style="color:#F1F5F9;">Sentiment:</strong> {report.get('sentiment_summary','—')}</p>
      <p style="font-size:0.8rem;color:#94A3B8;margin:0.35rem 0;line-height:1.5;">
        <strong style="color:#F1F5F9;">Fundamentals:</strong> {report.get('fundamentals_summary','—')}</p>
      <p style="font-size:0.8rem;color:#94A3B8;margin:0.35rem 0;line-height:1.5;">
        <strong style="color:#F1F5F9;">Divergence:</strong> {report.get('divergence_assessment','—')}</p>
    </div>
    """, unsafe_allow_html=True)

    risks = report.get("key_risks", [])
    if risks:
        st.markdown('<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin:0.8rem 0 0.4rem 0;">Key Risks</div>', unsafe_allow_html=True)
        for r in risks[:4]:
            st.markdown(f'<div style="font-size:0.77rem;color:#94A3B8;padding:0.25rem 0 0.25rem 0.7rem;border-left:2px solid #1E2D40;margin-bottom:0.3rem;line-height:1.5;">{r}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer">{report.get("disclaimer","")}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SEC Filings Context
# ---------------------------------------------------------------------------

def render_filing_context(ticker: str):
    st.markdown('<div class="card-title">Source Context</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">SEC Filing RAG Chunks</div>', unsafe_allow_html=True)

    db = load_db()
    chunks = db.get_filing_chunks(ticker)

    if not chunks:
        st.markdown(f"""
        <div class="empty-state">
          <div class="empty-icon">📄</div>
          <div class="empty-title">No Filing Data</div>
          <div class="empty-body">No 10-K chunks indexed for <strong>{ticker}</strong>.
          US-listed companies only. Run EDGAR ingestion to populate.</div>
          <code class="empty-cmd">python scripts/run_ingestion.py --tickers {ticker} --filings 10-K</code>
        </div>
        """, unsafe_allow_html=True)
        return

    filings_seen = set()
    options = []
    for row in chunks:
        key = f"{row['filing_type']} — {row['fiscal_period']} ({row['filed_date'][:10]})"
        if key not in filings_seen:
            filings_seen.add(key)
            options.append(key)

    selected = st.selectbox("Filing", options[:10], key="filing_select")
    if not selected:
        return

    parts = selected.split(" — ")
    filing_type = parts[0]
    fiscal_period = parts[1].split(" ")[0]

    filtered = [row for row in chunks
                if row["filing_type"] == filing_type and row["fiscal_period"] == fiscal_period]

    sections = {}
    for row in filtered:
        sections.setdefault(row["section"], []).append(row["text"])

    for section, texts in list(sections.items())[:5]:
        with st.expander(f"📑 {section}"):
            for text in texts[:3]:
                display = text[:500] + "…" if len(text) > 500 else text
                st.markdown(
                    f'<div style="font-size:0.77rem;color:#94A3B8;line-height:1.6;'
                    f'border-left:2px solid #00C2A8;padding-left:0.8rem;">{display}</div>',
                    unsafe_allow_html=True
                )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main():
    ticker, days_back, current_watchlist = render_sidebar()
    company_name = current_watchlist.get(ticker, ticker)

    # Masthead
    st.markdown(f"""
    <div class="masthead">
      <div>
        <div class="masthead-ticker">{ticker}</div>
        <div class="masthead-company">{company_name} &nbsp;·&nbsp; Sentiment vs Fundamentals Divergence</div>
      </div>
      <div class="masthead-right">
        <div class="masthead-date">📅 &nbsp;{date.today().strftime("%b %d, %Y")}</div>
        <div class="masthead-live-badge">
          <div class="masthead-live-dot"></div> LIVE
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI sentiment overview card
    st.markdown('<div class="ink-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Sentiment Overview</div>', unsafe_allow_html=True)
    render_kpi_row(ticker, days_back)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main grid: 2.4 : 1 split
    col_main, col_side = st.columns([2.4, 1], gap="large")

    with col_main:
        st.markdown('<div class="ink-card">', unsafe_allow_html=True)
        render_backtest(ticker)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="ink-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Latest News</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Headline Sentiment Feed</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-sub">FinBERT-scored articles · last {days_back} days</div>', unsafe_allow_html=True)
        render_headline_table(ticker, days_back)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="ink-card">', unsafe_allow_html=True)
        render_synthesis(ticker, days_back)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="ink-card">', unsafe_allow_html=True)
        render_filing_context(ticker)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
