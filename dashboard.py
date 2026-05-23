from __future__ import annotations

import math
import re
from html import escape
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent
BOOKS_PATH = BASE_DIR / "Books.csv"
RATINGS_PATH = BASE_DIR / "Ratings.csv"
USERS_PATH = BASE_DIR / "Users.csv"

RANDOM_STATE = 42
TOP_K = 10
RELEVANT_THRESHOLD = 7
MIN_USER_EXPLICIT_RATINGS = 10
MIN_BOOK_EXPLICIT_RATINGS = 10
CANDIDATE_POOL = 400
CO_LIKE_NEIGHBORS = 100
USER_CF_NEIGHBORS = 80

BEST_WEIGHTS = {
    "popularity": 0.0,
    "content": 0.2,
    "colike": 0.2,
    "author": 0.2,
    "usercf": 0.4,
}

METRICS = pd.DataFrame(
    [
        {
            "model": "Tuned Hybrid",
            "precision@10": 0.0184,
            "recall@10": 0.184,
            "map@10": 0.085278,
            "ndcg@10": 0.108227,
            "novelty": 10.437587,
            "diversity": 0.805809,
            "catalog_coverage": 0.388593,
            "evaluated_users": 250,
        },
        {
            "model": "User-Based CF",
            "precision@10": 0.0152,
            "recall@10": 0.152,
            "map@10": 0.068665,
            "ndcg@10": 0.087804,
            "novelty": 9.509319,
            "diversity": 0.909200,
            "catalog_coverage": 0.208863,
            "evaluated_users": 250,
        },
        {
            "model": "Content-Based",
            "precision@10": 0.0148,
            "recall@10": 0.148,
            "map@10": 0.055997,
            "ndcg@10": 0.077285,
            "novelty": 11.223513,
            "diversity": 0.708603,
            "catalog_coverage": 0.458350,
            "evaluated_users": 250,
        },
        {
            "model": "Co-Like ItemKNN",
            "precision@10": 0.0136,
            "recall@10": 0.136,
            "map@10": 0.064925,
            "ndcg@10": 0.081265,
            "novelty": 11.445849,
            "diversity": 0.916302,
            "catalog_coverage": 0.395979,
            "evaluated_users": 250,
        },
        {
            "model": "Author Affinity",
            "precision@10": 0.0116,
            "recall@10": 0.116,
            "map@10": 0.039427,
            "ndcg@10": 0.057086,
            "novelty": 11.048442,
            "diversity": 0.763637,
            "catalog_coverage": 0.371769,
            "evaluated_users": 250,
        },
        {
            "model": "Popularity",
            "precision@10": 0.0008,
            "recall@10": 0.008,
            "map@10": 0.002667,
            "ndcg@10": 0.003949,
            "novelty": 9.159728,
            "diversity": 0.719135,
            "catalog_coverage": 0.006976,
            "evaluated_users": 250,
        },
    ]
)

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#C084FC", "#818CF8", "#38BDF8", "#34D399", "#FBBF24", "#FB7185"]


st.set_page_config(
    page_title="BookMind · Sistem Rekomendasi Buku",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

    :root {
        --bg: #000000;
        --surface: #050505;
        --surface-2: #080808;
        --surface-3: #0B0B0B;
        --border: rgba(150,130,255,0.14);
        --border-glow: rgba(192,132,252,0.38);
        --text: #EEE8FF;
        --muted: #8B7FA8;
        --soft: #C4B5E0;
        --purple: #C084FC;
        --indigo: #818CF8;
        --cyan: #38BDF8;
        --green: #34D399;
        --amber: #FBBF24;
        --rose: #FB7185;
        --gold: #F59E0B;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .main,
    main {
        font-family: 'DM Sans', sans-serif;
        background: #000000 !important;
        color: var(--text);
    }

    .stApp::before {
        content: none;
    }
    [data-testid="stHeader"],
    [data-testid="stDecoration"] {
        background: #000000 !important;
    }

    .block-container {
        position: relative;
        z-index: 1;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: var(--bg) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg,
            rgba(130,80,255,.10) 0%,
            rgba(56,189,248,.05) 40%,
            transparent 80%) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        background: rgba(16,14,26,.92);
        border: 1px solid var(--border);
        border-radius: 20px;
        margin: 72px 14px 0 14px;
        padding: 24px 16px 20px;
        box-shadow: 0 30px 70px rgba(0,0,0,.55);
        backdrop-filter: blur(18px);
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }

    .sidebar-logo {
        text-align: center;
        margin-bottom: 22px;
        padding-bottom: 18px;
        border-bottom: 1px solid var(--border);
    }
    .sidebar-logo .logo-icon {
        font-size: 2.4rem;
        display: block;
        margin-bottom: 6px;
    }
    .sidebar-logo .logo-name {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 900;
        letter-spacing: -.01em;
        background: linear-gradient(135deg, var(--purple), var(--cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sidebar-logo .logo-tag {
        font-size: 0.72rem;
        color: var(--muted) !important;
        font-weight: 400;
        letter-spacing: .04em;
        margin-top: 2px;
    }
    .nav-section-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
        color: var(--muted) !important;
        margin: 16px 0 8px;
        padding-left: 2px;
    }

    /* ── TYPOGRAPHY ── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text) !important;
    }
    p, label, span, div { color: var(--text); }

    /* ── HERO BANNER ── */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg,
            rgba(100,60,200,.30) 0%,
            rgba(16,14,26,.95) 45%,
            rgba(20,40,80,.28) 100%);
        border-radius: 28px;
        padding: 40px 44px 36px;
        border: 1px solid var(--border-glow);
        box-shadow:
            0 0 0 1px rgba(130,80,255,.08),
            0 30px 100px rgba(0,0,0,.60),
            inset 0 1px 0 rgba(255,255,255,.06);
        margin-bottom: 30px;
    }
    .hero::before {
        content: "";
        position: absolute;
        width: 500px; height: 500px;
        right: -200px; top: -250px;
        background: radial-gradient(circle, rgba(192,132,252,.22), transparent 58%);
        filter: blur(4px);
        pointer-events: none;
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 300px; height: 300px;
        left: -80px; bottom: -120px;
        background: radial-gradient(circle, rgba(56,189,248,.12), transparent 62%);
        pointer-events: none;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: .10em;
        text-transform: uppercase;
        background: rgba(192,132,252,.12);
        border: 1px solid rgba(192,132,252,.32);
        border-radius: 999px;
        padding: 6px 14px;
        margin-bottom: 18px;
        color: var(--purple) !important;
        box-shadow: 0 0 30px rgba(192,132,252,.14);
    }
    .hero-eyebrow .dot {
        width: 6px; height: 6px;
        background: var(--purple);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--purple);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .hero-title {
        font-family: 'Syne', sans-serif !important;
        font-size: clamp(2.1rem, 4.5vw, 3.6rem);
        font-weight: 900;
        line-height: 1.02;
        margin: 0 0 14px;
        color: #fff !important;
        letter-spacing: -.02em;
        max-width: 880px;
    }
    .hero-title .accent {
        background: linear-gradient(135deg, var(--purple), var(--cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-desc {
        color: rgba(200,190,230,.82) !important;
        font-size: 1.05rem;
        font-weight: 300;
        max-width: 700px;
        line-height: 1.6;
        margin: 0;
    }
    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 22px;
    }
    .hero-pill {
        font-size: 0.73rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,.20);
        background: rgba(255,255,255,.04);
        color: var(--soft) !important;
    }

    /* ── METRIC CARDS ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, var(--surface), var(--bg));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 20px 22px;
        box-shadow:
            0 20px 50px rgba(0,0,0,.45),
            inset 0 1px 0 rgba(255,255,255,.04);
        transition: border-color .2s, box-shadow .2s;
        position: relative;
        overflow: hidden;
    }
    div[data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--purple), var(--cyan));
        opacity: .7;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--border-glow);
        box-shadow: 0 20px 60px rgba(0,0,0,.55), 0 0 28px rgba(192,132,252,.10);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label * {
        color: var(--muted) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: .04em !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        font-family: 'Syne', sans-serif !important;
        font-weight: 900 !important;
        font-size: 2rem !important;
        color: var(--text) !important;
    }

    /* ── SECTION HEADER ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 32px 0 18px;
    }
    .section-header .icon {
        width: 36px; height: 36px;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.05rem;
        background: linear-gradient(135deg, rgba(192,132,252,.20), rgba(56,189,248,.12));
        border: 1px solid rgba(192,132,252,.25);
        flex-shrink: 0;
    }
    .section-header h3 {
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -.01em;
    }
    .section-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border-glow), transparent);
    }

    /* ── BOOK CARDS ── */
    .book-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        column-gap: 16px;
        row-gap: 34px;
        align-items: stretch;
        margin: 4px 0 28px;
    }
    .book-grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .book-grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }

    .book-card {
        background: linear-gradient(160deg, var(--surface), var(--bg));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 14px 14px 16px;
        min-height: 430px;
        height: 100%;
        box-shadow: 0 16px 40px rgba(0,0,0,.44);
        transition: transform .22s cubic-bezier(.34,1.56,.64,1), border-color .2s, box-shadow .2s;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }
    .book-card-compact {
        min-height: 462px;
    }
    .book-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 100px;
        background: linear-gradient(180deg, rgba(192,132,252,.06), transparent);
        pointer-events: none;
    }
    .book-card:hover {
        transform: translateY(-5px);
        border-color: var(--border-glow);
        box-shadow:
            0 28px 70px rgba(0,0,0,.60),
            0 0 40px rgba(192,132,252,.14);
    }
    .book-cover-wrap {
        width: 100%;
        height: 210px;
        border-radius: 14px;
        overflow: hidden;
        background: linear-gradient(145deg, var(--surface-3), var(--surface));
        border: 1px solid var(--border);
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 14px;
        position: relative;
        flex: 0 0 auto;
    }
    .book-card-compact .book-cover-wrap {
        height: 286px;
    }
    .book-cover {
        width: 100%; height: 100%;
        object-fit: cover; display: block;
    }
    .book-rank-badge {
        position: absolute;
        top: 8px; left: 8px;
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 900;
        background: rgba(8,6,15,.85);
        border: 1px solid rgba(192,132,252,.4);
        color: var(--purple) !important;
        border-radius: 8px;
        padding: 3px 8px;
        backdrop-filter: blur(10px);
        letter-spacing: .04em;
        z-index: 2;
    }
    .book-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--text) !important;
        line-height: 1.25;
        margin-bottom: 7px;
        height: 3.35rem;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
    }
    .book-author {
        font-size: 0.76rem;
        color: var(--muted) !important;
        line-height: 1.3;
        margin-bottom: 4px;
        font-style: italic;
        height: 1.05rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .book-publisher {
        font-size: 0.72rem;
        color: var(--muted) !important;
        opacity: .7;
        margin-bottom: 12px;
        height: 1rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .book-score-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
        font-size: 0.70rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(192,132,252,.10);
        border: 1px solid rgba(192,132,252,.28);
        color: var(--purple) !important;
        width: fit-content;
        margin-top: auto;
    }
    .book-score-badge .score-dot {
        width: 5px; height: 5px;
        background: var(--purple);
        border-radius: 50%;
        display: inline-block;
        flex: 0 0 auto;
    }
    .no-cover-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: var(--muted) !important;
        font-size: 0.76rem;
        font-weight: 600;
        gap: 8px;
        height: 100%;
        opacity: .6;
    }
    @media(max-width: 1180px) {
        .book-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    }
    @media(max-width: 900px) {
        .book-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media(max-width: 640px) {
        .book-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px 12px; }
        .book-card { min-height: 398px; }
        .book-cover-wrap { height: 190px; }
    }

    /* ── DATAFRAME ── */
    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid var(--border);
        background: var(--surface);
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border);
        background: transparent !important;
        padding-bottom: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(16,14,26,.85);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 8px 18px;
        color: var(--muted) !important;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all .2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(192,132,252,.20), rgba(56,189,248,.12)) !important;
        border-color: rgba(192,132,252,.50) !important;
        color: var(--purple) !important;
        box-shadow: 0 0 24px rgba(192,132,252,.16);
    }

    /* ── SELECTS & INPUTS ── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: var(--surface) !important;
        border-color: var(--border) !important;
        border-radius: 14px !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input { color: var(--text) !important; }

    /* ── BUTTONS ── */
    button[kind="secondary"], button[kind="primary"] {
        border-radius: 12px !important;
        border: 1px solid rgba(192,132,252,.38) !important;
        background: linear-gradient(135deg, rgba(192,132,252,.18), rgba(56,189,248,.10)) !important;
        color: var(--text) !important;
        font-weight: 600 !important;
    }

    /* ── RADIO / SIDEBAR NAV ── */
    .stRadio label { font-size: 0.9rem !important; }
    div[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
    }

    /* ── CALLOUT ── */
    .info-callout {
        background: rgba(56,189,248,.06);
        border: 1px solid rgba(56,189,248,.22);
        border-left: 3px solid var(--cyan);
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        font-size: 0.9rem;
        color: rgba(200,230,255,.88) !important;
        margin: 14px 0 20px;
        line-height: 1.6;
    }

    /* ── MODEL COMPARISON TABLE ── */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }
    .metric-mini {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-mini .mm-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--muted) !important;
        margin-bottom: 6px;
    }
    .metric-mini .mm-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 900;
        color: var(--purple) !important;
    }
    .metric-big {
        text-align: left;
        min-height: 88px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: linear-gradient(145deg, var(--surface), var(--bg));
        border-radius: 20px;
        box-shadow: 0 20px 50px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.04);
    }
    .metric-big .mm-value {
        color: var(--text) !important;
        font-size: clamp(1.35rem, 2vw, 1.85rem);
        white-space: nowrap;
        letter-spacing: -.03em;
    }

    /* ── WEIGHT VIZ ── */
    .weight-bar-wrap { margin: 6px 0 20px; }
    .weight-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .weight-label {
        width: 120px;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--soft) !important;
        text-transform: capitalize;
    }
    .weight-track {
        flex: 1;
        height: 8px;
        background: rgba(255,255,255,.06);
        border-radius: 99px;
        overflow: hidden;
    }
    .weight-fill {
        height: 100%;
        border-radius: 99px;
    }
    .weight-pct {
        width: 44px;
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--text) !important;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def normalize_text(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    value = str(value).lower().strip()
    value = re.sub(r"&amp;", "&", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or "unknown"


def minmax_dict(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if hi == lo:
        return {key: 1.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


@st.cache_data(show_spinner=False)
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    books_raw = pd.read_csv(BOOKS_PATH, low_memory=False)
    ratings_raw = pd.read_csv(RATINGS_PATH)
    users_raw = pd.read_csv(USERS_PATH)
    return books_raw, ratings_raw, users_raw


@st.cache_data(show_spinner=False)
def prepare_data() -> dict[str, object]:
    books_raw, ratings_raw, users_raw = load_raw_data()
    books = books_raw.copy()
    manual_fixes = {
        "078946697X": {"Book-Title": "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)", "Book-Author": "Michael Teitelbaum", "Year-Of-Publication": "2000", "Publisher": "DK Publishing Inc"},
        "0789466953": {"Book-Title": "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)", "Book-Author": "James Buckley", "Year-Of-Publication": "2000", "Publisher": "DK Publishing Inc"},
        "2070426769": {"Book-Title": "Peuple du ciel - Suivi de Les bergers", "Book-Author": "Jean-Marie Gustave Le Clezio", "Year-Of-Publication": "2003", "Publisher": "Gallimard"},
    }
    for isbn, fixes in manual_fixes.items():
        mask = books["ISBN"].eq(isbn)
        for col, value in fixes.items():
            books.loc[mask, col] = value
    books["Book-Author"] = books["Book-Author"].fillna("Unknown")
    books["Publisher"] = books["Publisher"].fillna("Unknown").str.replace("&amp;", "&", regex=False)
    books["title_key"] = books["Book-Title"].map(normalize_text)
    books["author_key"] = books["Book-Author"].map(normalize_text)
    books["publisher_key"] = books["Publisher"].map(normalize_text)
    books["book_key"] = books["title_key"] + " | " + books["author_key"]
    books["book_id"] = pd.factorize(books["book_key"])[0].astype("int32")
    books["year_numeric"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
    catalog = (
        books.groupby("book_id").agg(
            Book_Title=("Book-Title", "first"), Book_Author=("Book-Author", "first"),
            Publisher=("Publisher", "first"), title_key=("title_key", "first"),
            author_key=("author_key", "first"), publisher_key=("publisher_key", "first"),
            n_isbn=("ISBN", "nunique"), year_min=("year_numeric", "min"),
            year_max=("year_numeric", "max"), image_url=("Image-URL-M", "first"),
        ).reset_index()
    )
    isbn_to_book_id = books[["ISBN", "book_id"]].drop_duplicates()
    explicit_events = (
        ratings_raw[ratings_raw["Book-Rating"].between(1, 10)]
        .merge(isbn_to_book_id, on="ISBN", how="inner")
        .groupby(["User-ID", "book_id"], as_index=False)
        .agg(rating=("Book-Rating", "max"))
    )
    def filter_interactions(events: pd.DataFrame) -> pd.DataFrame:
        filtered = events.copy()
        while True:
            before = len(filtered)
            user_counts = filtered["User-ID"].value_counts()
            book_counts = filtered["book_id"].value_counts()
            filtered = filtered[
                filtered["User-ID"].isin(user_counts[user_counts >= MIN_USER_EXPLICIT_RATINGS].index)
                & filtered["book_id"].isin(book_counts[book_counts >= MIN_BOOK_EXPLICIT_RATINGS].index)
            ].copy()
            if len(filtered) == before:
                return filtered
    warm_events = filter_interactions(explicit_events)
    def train_val_test_split(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_idx = set(events.index)
        val_rows, test_rows = [], []
        for _, group in events.groupby("User-ID"):
            relevant = group[group["rating"] >= RELEVANT_THRESHOLD]
            if len(group) < 5 or len(relevant) < 2:
                continue
            picked = relevant.sample(n=2, random_state=RANDOM_STATE)
            val_rows.append(picked.iloc[[0]])
            test_rows.append(picked.iloc[[1]])
            train_idx -= set(picked.index)
        val = pd.concat(val_rows, ignore_index=False)
        test = pd.concat(test_rows, ignore_index=False)
        train = events.loc[sorted(train_idx)].copy()
        eligible_users = set(val["User-ID"]) & set(test["User-ID"])
        return (train[train["User-ID"].isin(eligible_users)].copy(),
                val[val["User-ID"].isin(eligible_users)].copy(),
                test[test["User-ID"].isin(eligible_users)].copy())
    train_events, val_events, test_events = train_val_test_split(warm_events)
    return {
        "books_raw": books_raw, "ratings_raw": ratings_raw, "users_raw": users_raw,
        "books": books, "catalog": catalog, "explicit_events": explicit_events,
        "warm_events": warm_events, "train_events": train_events,
        "val_events": val_events, "test_events": test_events,
    }


@st.cache_resource(show_spinner=False)
def build_recommender() -> dict[str, object]:
    data = prepare_data()
    catalog = data["catalog"]
    train_events = data["train_events"]
    train_seen = train_events.groupby("User-ID")["book_id"].apply(set).to_dict()
    popularity_stats = (
        train_events.groupby("book_id")
        .agg(rating_count=("rating", "size"), avg_rating=("rating", "mean"))
        .reset_index()
    )
    global_mean = train_events["rating"].mean()
    m = popularity_stats["rating_count"].quantile(0.75)
    popularity_stats["popularity_score"] = (
        (popularity_stats["rating_count"] / (popularity_stats["rating_count"] + m)) * popularity_stats["avg_rating"]
        + (m / (popularity_stats["rating_count"] + m)) * global_mean
    )
    popularity_rank = popularity_stats.sort_values(["popularity_score", "rating_count"], ascending=False)["book_id"].tolist()
    popularity_score = dict(zip(popularity_stats["book_id"], popularity_stats["popularity_score"]))
    model_book_ids = sorted(train_events["book_id"].unique())
    catalog_model = catalog[catalog["book_id"].isin(model_book_ids)].reset_index(drop=True).copy()
    catalog_model["content_text"] = (
        catalog_model["title_key"].fillna("") + " author_"
        + catalog_model["author_key"].fillna("").str.replace(" ", "_")
        + " publisher_" + catalog_model["publisher_key"].fillna("").str.replace(" ", "_")
    )
    tfidf = TfidfVectorizer(min_df=2, max_features=20000, ngram_range=(1, 2), stop_words="english")
    content_matrix = tfidf.fit_transform(catalog_model["content_text"])
    book_id_to_content_idx = pd.Series(catalog_model.index.values, index=catalog_model["book_id"]).to_dict()
    liked_train = train_events[train_events["rating"] >= RELEVANT_THRESHOLD].copy()
    user_codes, user_uniques = pd.factorize(liked_train["User-ID"])
    item_codes, item_uniques = pd.factorize(liked_train["book_id"])
    liked_item_user = csr_matrix(
        (np.ones(len(liked_train)), (item_codes, user_codes)),
        shape=(len(item_uniques), len(user_uniques)),
    )
    book_id_to_liked_code = {book_id: code for code, book_id in enumerate(item_uniques)}
    liked_code_to_book_id = {code: book_id for code, book_id in enumerate(item_uniques)}
    co_like_model = NearestNeighbors(metric="cosine", algorithm="brute")
    co_like_model.fit(liked_item_user)
    co_like_neighbors: dict[int, list[tuple[int, float]]] = {}
    for book_id, code_idx in book_id_to_liked_code.items():
        n_neighbors = min(CO_LIKE_NEIGHBORS + 1, liked_item_user.shape[0])
        distances, indices = co_like_model.kneighbors(liked_item_user[code_idx], n_neighbors=n_neighbors)
        neighbors = []
        for distance, neighbor_idx in zip(distances.ravel(), indices.ravel()):
            neighbor_book_id = liked_code_to_book_id[neighbor_idx]
            if neighbor_book_id == book_id:
                continue
            similarity = 1 - distance
            if similarity > 0:
                neighbors.append((neighbor_book_id, similarity))
        co_like_neighbors[book_id] = neighbors
    liked_history = liked_train.groupby("User-ID")[["book_id", "rating"]].apply(
        lambda df: list(df.itertuples(index=False, name=None))
    ).to_dict()
    liked_user_item = liked_item_user.T.tocsr()
    user_cf_model = NearestNeighbors(metric="cosine", algorithm="brute")
    user_cf_model.fit(liked_user_item)
    user_id_to_liked_code = {user_id: code for code, user_id in enumerate(user_uniques)}
    liked_code_to_user_id = {code: user_id for code, user_id in enumerate(user_uniques)}
    book_to_author = dict(zip(catalog["book_id"], catalog["author_key"]))
    author_to_books = catalog_model.groupby("author_key")["book_id"].apply(list).to_dict()
    return {
        **data, "train_seen": train_seen, "popularity_stats": popularity_stats,
        "popularity_rank": popularity_rank, "popularity_score": popularity_score,
        "catalog_model": catalog_model, "content_matrix": content_matrix,
        "book_id_to_content_idx": book_id_to_content_idx,
        "co_like_neighbors": co_like_neighbors, "liked_history": liked_history,
        "liked_user_item": liked_user_item, "user_cf_model": user_cf_model,
        "user_id_to_liked_code": user_id_to_liked_code,
        "liked_code_to_user_id": liked_code_to_user_id,
        "book_to_author": book_to_author, "author_to_books": author_to_books,
    }


def get_model() -> dict[str, object]:
    with st.spinner("⚙️ Memuat data dan membangun model rekomendasi…"):
        return build_recommender()


# ─── scoring ────────────────────────────────────────────────────────────────

def score_popularity(model, user_id, pool=CANDIDATE_POOL):
    seen = model["train_seen"].get(user_id, set())
    candidates = [b for b in model["popularity_rank"] if b not in seen][:pool]
    return {b: model["popularity_score"].get(b, 0.0) for b in candidates}


def score_content(model, user_id, pool=CANDIDATE_POOL):
    seen = model["train_seen"].get(user_id, set())
    history = model["liked_history"].get(user_id, [])
    if not history:
        return {}
    seed_ids = [b for b, _ in history]
    seed_indices = [model["book_id_to_content_idx"][b] for b in seed_ids if b in model["book_id_to_content_idx"]]
    if not seed_indices:
        return {}
    seed_matrix = model["content_matrix"][seed_indices]
    avg_seed = np.asarray(seed_matrix.mean(axis=0))
    sims = cosine_similarity(avg_seed, model["content_matrix"]).ravel()
    top_idx = np.argpartition(sims, -min(pool, len(sims)))[-min(pool, len(sims)):]
    catalog_model = model["catalog_model"]
    scores = {}
    for i in top_idx:
        bid = int(catalog_model.iloc[i]["book_id"])
        if bid not in seen and np.isfinite(sims[i]):
            scores[bid] = float(sims[i])
    return scores


def score_colike(model, user_id, pool=CANDIDATE_POOL):
    seen = model["train_seen"].get(user_id, set())
    history = model["liked_history"].get(user_id, [])
    if not history:
        return {}
    aggregated: dict[int, float] = defaultdict(float)
    for seed_bid, rating in history:
        weight = rating / 10.0
        for neighbor_bid, sim in model["co_like_neighbors"].get(seed_bid, []):
            if neighbor_bid not in seen:
                aggregated[neighbor_bid] += weight * sim
    return dict(list(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))[:pool])


def score_author(model, user_id, pool=CANDIDATE_POOL):
    seen = model["train_seen"].get(user_id, set())
    history = model["liked_history"].get(user_id, [])
    if not history:
        return {}
    author_weights: dict[str, float] = defaultdict(float)
    for bid, rating in history:
        author = model["book_to_author"].get(bid)
        if author and author != "unknown":
            author_weights[author] += rating / 10.0
    scores: dict[int, float] = {}
    for author, weight in author_weights.items():
        for bid in model["author_to_books"].get(author, []):
            if bid not in seen:
                pop = model["popularity_score"].get(bid, 0.0)
                scores[bid] = scores.get(bid, 0.0) + weight * (1.0 + 0.05 * pop)
    return dict(list(sorted(scores.items(), key=lambda x: x[1], reverse=True))[:pool])


def score_usercf(model, user_id, pool=CANDIDATE_POOL, n_neighbors=USER_CF_NEIGHBORS):
    seen = model["train_seen"].get(user_id, set())
    code = model["user_id_to_liked_code"].get(user_id)
    if code is None:
        return {}
    n_nbrs = min(n_neighbors + 1, model["liked_user_item"].shape[0])
    distances, indices = model["user_cf_model"].kneighbors(model["liked_user_item"][code], n_neighbors=n_nbrs)
    scores: dict[int, float] = defaultdict(float)
    for dist, nbr_code in zip(distances.ravel(), indices.ravel()):
        if model["liked_code_to_user_id"].get(nbr_code) == user_id:
            continue
        sim = 1 - dist
        nbr_uid = model["liked_code_to_user_id"][nbr_code]
        for bid, rating in model["liked_history"].get(nbr_uid, []):
            if bid not in seen:
                scores[bid] += sim * (rating / 10.0)
    return dict(list(sorted(scores.items(), key=lambda x: x[1], reverse=True))[:pool])


def recommend_for_user(user_id: int) -> pd.DataFrame:
    model = get_model()
    scores_by_source = {
        "popularity": minmax_dict(score_popularity(model, user_id)),
        "content": minmax_dict(score_content(model, user_id)),
        "colike": minmax_dict(score_colike(model, user_id)),
        "author": minmax_dict(score_author(model, user_id)),
        "usercf": minmax_dict(score_usercf(model, user_id)),
    }
    combined: dict[int, float] = defaultdict(float)
    source_hits: dict[int, list[str]] = defaultdict(list)
    for source, w in BEST_WEIGHTS.items():
        for bid, score in scores_by_source[source].items():
            weighted = w * score
            combined[bid] += weighted
            if weighted > 0:
                source_hits[bid].append(source)
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    recs = pd.DataFrame(ranked, columns=["book_id", "hybrid_score"])
    recs["sources"] = recs["book_id"].map(lambda b: ", ".join(source_hits.get(b, [])))
    return recs.merge(model["catalog"], on="book_id", how="left")


def recommend_for_book(seed_book_id: int) -> pd.DataFrame:
    model = get_model()
    popularity_score = model["popularity_score"]
    scores_by_source: dict[str, dict[int, float]] = {}
    if seed_book_id in model["book_id_to_content_idx"]:
        seed_idx = model["book_id_to_content_idx"][seed_book_id]
        sims = cosine_similarity(model["content_matrix"][seed_idx], model["content_matrix"]).ravel()
        top_idx = np.argpartition(sims, -min(CANDIDATE_POOL, len(sims)))[-min(CANDIDATE_POOL, len(sims)):]
        catalog_model = model["catalog_model"]
        content_scores = {
            int(catalog_model.iloc[i]["book_id"]): float(sims[i])
            for i in top_idx if np.isfinite(sims[i])
        }
        scores_by_source["content"] = minmax_dict(content_scores)
    colike_scores = {int(b): float(s) for b, s in model["co_like_neighbors"].get(seed_book_id, [])[:CANDIDATE_POOL]}
    scores_by_source["colike"] = minmax_dict(colike_scores)
    author_scores: dict[int, float] = {}
    seed_author = model["book_to_author"].get(seed_book_id)
    if seed_author and seed_author != "unknown":
        for bid in model["author_to_books"].get(seed_author, []):
            if bid != seed_book_id:
                author_scores[int(bid)] = 1.0 + 0.05 * popularity_score.get(bid, 0.0)
    scores_by_source["author"] = minmax_dict(author_scores)
    popular_candidates = [b for b in model["popularity_rank"] if b != seed_book_id][:CANDIDATE_POOL]
    scores_by_source["popularity"] = minmax_dict({int(b): float(popularity_score.get(b, 0.0)) for b in popular_candidates})
    item_weights = {"content": 0.45, "colike": 0.35, "author": 0.15, "popularity": 0.05}
    combined: dict[int, float] = defaultdict(float)
    source_hits: dict[int, list[str]] = defaultdict(list)
    for src, src_scores in scores_by_source.items():
        for bid, score in src_scores.items():
            if bid == seed_book_id:
                continue
            weighted = item_weights[src] * score
            combined[bid] += weighted
            if weighted > 0:
                source_hits[bid].append(src)
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    recs = pd.DataFrame(ranked, columns=["book_id", "similarity_score"])
    recs["sources"] = recs["book_id"].map(lambda b: ", ".join(source_hits.get(b, [])))
    return recs.merge(model["catalog"], on="book_id", how="left")


def attach_catalog(model, frame):
    return frame.merge(model["catalog"], on="book_id", how="left")


def format_pct(v: float) -> str:
    return f"{v:.1%}"


def metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-mini metric-big">
        <div class="mm-label">{escape(label)}</div>
        <div class="mm-value">{escape(value)}</div>
    </div>
    """


def valid_image_url(value: object) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return value if value.lower().startswith(("http://", "https://")) else ""


# ─── plot styling ────────────────────────────────────────────────────────────

def style_plot(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#C4B5E0"),
        margin=dict(l=12, r=12, t=52, b=12),
        title_font=dict(family="Syne, sans-serif", color="#EEE8FF", size=15),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#C4B5E0")),
    )
    fig.update_xaxes(gridcolor="rgba(150,130,255,.10)", zerolinecolor="rgba(150,130,255,.14)", color="#8B7FA8")
    fig.update_yaxes(gridcolor="rgba(150,130,255,.10)", zerolinecolor="rgba(150,130,255,.14)", color="#8B7FA8")
    return fig


# ─── UI components ───────────────────────────────────────────────────────────

def section_header(icon: str, title: str):
    st.markdown(
        f"""<div class="section-header">
            <div class="icon">{icon}</div>
            <h3>{title}</h3>
            <div class="section-line"></div>
        </div>""",
        unsafe_allow_html=True,
    )


def book_card_html(row: pd.Series, rank: int, score_label: str | None = None, compact: bool = False) -> str:
    title = escape(str(row.get("Book_Title", "Unknown title")))
    author = escape(str(row.get("Book_Author", "Unknown author")))
    publisher = escape(str(row.get("Publisher", "")))
    image_url = valid_image_url(row.get("image_url", ""))
    card_class = "book-card book-card-compact" if compact else "book-card"

    if image_url:
        cover = f'<img class="book-cover" src="{escape(image_url)}" alt="{title} cover" onerror="this.remove();">'
    else:
        cover = '<div class="no-cover-placeholder"><span style="font-size:2rem">📖</span><span>No Cover</span></div>'

    score_html = ""
    if score_label:
        clean_label = escape(str(score_label))
        score_html = f'<span class="book-score-badge"><span class="score-dot"></span><span>{clean_label}</span></span>'

    pub_html = f'<div class="book-publisher">{publisher}</div>' if publisher and not compact else '<div class="book-publisher"></div>'

    # Jangan pakai multiline HTML di sini. Streamlit markdown kadang membaca card kedua
    # dan seterusnya sebagai teks biasa kalau ada newline/indentasi di dalam raw HTML block.
    return (
        f'<div class="{card_class}">'
        f'<div class="book-cover-wrap">{cover}<div class="book-rank-badge">#{rank}</div></div>'
        f'<div class="book-title">{title}</div>'
        f'<div class="book-author">{author}</div>'
        f'{pub_html}'
        f'{score_html}'
        f'</div>'
    )


def render_book_grid(frame: pd.DataFrame, score_col: str | None = None, max_items: int = 10, columns: int = 5) -> None:
    items = frame.head(max_items).reset_index(drop=True)
    cards: list[str] = []

    for idx, row in items.iterrows():
        score_label = None
        if score_col and score_col in items.columns:
            v = row[score_col]
            score_label = f"{float(v):.3f}" if isinstance(v, (int, float, np.floating)) else str(v)
        cards.append(book_card_html(row, rank=idx + 1, score_label=score_label))

    st.markdown(
        f'<div class="book-grid cols-{columns}">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_hero(subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
            <span class="hero-eyebrow">
                <span class="dot"></span>
                Book-Crossing · Hybrid Recommender System
            </span>
            <div class="hero-title">Dashboard Sistem<br><span class="accent">Rekomendasi Buku</span></div>
            <p class="hero-desc">{subtitle}</p>
            <div class="hero-pills">
                <span class="hero-pill">📊 5 Model Ensemble</span>
                <span class="hero-pill">📚 Book-Crossing Dataset</span>
                <span class="hero-pill">⚡ Top-10 Recommendations</span>
                <span class="hero-pill">🎯 Tuned Hybrid Weights</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── pages ───────────────────────────────────────────────────────────────────

def page_overview(model: dict) -> None:
    render_hero("Eksplorasi dataset, pantau kinerja model, dan simulasikan rekomendasi Top-10 dengan cover buku.")

    # ── KPI row ──────────────────────────────────────────────────────────────
    section_header("🗂️", "Statistik Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Buku (Metadata)", f"{len(model['books_raw']):,}"), unsafe_allow_html=True)
    c2.markdown(metric_card("Total Rating", f"{len(model['ratings_raw']):,}"), unsafe_allow_html=True)
    c3.markdown(metric_card("Pengguna", f"{len(model['users_raw']):,}"), unsafe_allow_html=True)
    c4.markdown(metric_card("Warm Interactions", f"{len(model['warm_events']):,}"), unsafe_allow_html=True)

    # ── Best model KPIs ───────────────────────────────────────────────────────
    section_header("🏆", "Performa Model Terbaik — Tuned Hybrid")
    best = METRICS.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric_card("Recall@10", f"{best['recall@10']:.3f}"), unsafe_allow_html=True)
    m2.markdown(metric_card("NDCG@10", f"{best['ndcg@10']:.3f}"), unsafe_allow_html=True)
    m3.markdown(metric_card("MAP@10", f"{best['map@10']:.3f}"), unsafe_allow_html=True)
    m4.markdown(metric_card("Catalog Coverage", format_pct(float(best["catalog_coverage"]))), unsafe_allow_html=True)

    st.markdown(
        """<div class="info-callout">
        Model final menggabungkan lima sinyal: <strong>Bayesian Popularity</strong>,
        <strong>Content-Based TF-IDF</strong>, <strong>Co-Like Item KNN</strong>,
        <strong>Author Affinity</strong>, dan <strong>User-Based CF</strong>.
        Bobot dioptimalkan pada validation set; hasil akhir dilaporkan pada test set (250 user).
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Popular books ─────────────────────────────────────────────────────────
    section_header("🔥", "Buku Paling Populer")
    popular_books = (
        model["popularity_stats"]
        .sort_values(["popularity_score", "rating_count"], ascending=False)
        .head(10)
        .merge(model["catalog"], on="book_id", how="left")
    )
    render_book_grid(popular_books, score_col="popularity_score", max_items=10, columns=5)

    # ── Hybrid weights bar ────────────────────────────────────────────────────
    section_header("⚖️", "Bobot Sinyal Hybrid")
    grad_map = {
        "usercf": "linear-gradient(90deg,#C084FC,#818CF8)",
        "content": "linear-gradient(90deg,#818CF8,#38BDF8)",
        "colike": "linear-gradient(90deg,#38BDF8,#34D399)",
        "author": "linear-gradient(90deg,#34D399,#FBBF24)",
        "popularity": "linear-gradient(90deg,#FBBF24,#FB7185)",
    }
    label_map = {
        "usercf": "User-Based CF",
        "content": "Content-Based",
        "colike": "Co-Like KNN",
        "author": "Author Affinity",
        "popularity": "Popularity",
    }
    bars_html = '<div class="weight-bar-wrap">'
    for k, w in sorted(BEST_WEIGHTS.items(), key=lambda x: -x[1]):
        pct = int(w * 100)
        bars_html += f"""
        <div class="weight-row">
            <div class="weight-label">{label_map.get(k, k)}</div>
            <div class="weight-track">
                <div class="weight-fill" style="width:{pct*2.5}%;background:{grad_map.get(k,'#fff')};"></div>
            </div>
            <div class="weight-pct">{pct}%</div>
        </div>"""
    bars_html += "</div>"
    st.markdown(bars_html, unsafe_allow_html=True)


def page_data(model: dict) -> None:
    render_hero("Eksplorasi distribusi rating, penerbit teratas, dan pola aktivitas pengguna.")

    books = model["books"]
    ratings_raw = model["ratings_raw"]
    catalog = model["catalog"]
    explicit_events = model["explicit_events"]
    warm_events = model["warm_events"]

    section_header("⭐", "Distribusi Rating")
    rating_counts = ratings_raw["Book-Rating"].value_counts().sort_index().reset_index()
    rating_counts.columns = ["rating", "count"]
    fig_rating = px.bar(
        rating_counts, x="rating", y="count", text_auto=True,
        title="Distribusi Semua Rating (termasuk implisit=0)",
        color="count", color_continuous_scale=["#1E1A35", "#C084FC"],
    )
    fig_rating.update_coloraxes(showscale=False)
    st.plotly_chart(style_plot(fig_rating), use_container_width=True)

    section_header("📅", "Tren Publikasi & Penerbit")
    c1, c2 = st.columns(2)
    year_dist = books["year_numeric"].dropna()
    year_dist = year_dist[(year_dist >= 1900) & (year_dist <= 2005)]
    fig_year = px.histogram(year_dist, nbins=55, title="Tahun Publikasi", labels={"value": "Tahun", "count": "Jumlah"})
    c1.plotly_chart(style_plot(fig_year), use_container_width=True)

    top_publishers = catalog["Publisher"].value_counts().head(10).reset_index()
    top_publishers.columns = ["publisher", "book_count"]
    fig_pub = px.bar(
        top_publishers, x="book_count", y="publisher", orientation="h",
        title="Top 10 Penerbit", color="book_count",
        color_continuous_scale=["#17142A", "#818CF8"],
    )
    fig_pub.update_coloraxes(showscale=False)
    c2.plotly_chart(style_plot(fig_pub), use_container_width=True)

    section_header("👤", "Aktivitas Pengguna & Buku")
    c3, c4 = st.columns(2)
    activity = explicit_events.groupby("User-ID").size().reset_index(name="ratings")
    c3.plotly_chart(
        style_plot(px.histogram(activity, x=activity["ratings"].clip(upper=100), nbins=50,
            title="Aktivitas User (clip 100)", labels={"x": "Jumlah rating eksplisit"},
            color_discrete_sequence=["#C084FC"])),
        use_container_width=True,
    )
    book_activity = explicit_events.groupby("book_id").size().reset_index(name="ratings")
    c4.plotly_chart(
        style_plot(px.histogram(book_activity, x=book_activity["ratings"].clip(upper=100), nbins=50,
            title="Popularitas Buku (clip 100)", labels={"x": "Jumlah rating eksplisit"},
            color_discrete_sequence=["#38BDF8"])),
        use_container_width=True,
    )

    section_header("📋", "Ringkasan Dataset Modeling")
    summary = pd.DataFrame({
        "Tahap": ["Raw ratings", "Explicit events", "Warm events", "Canonical books"],
        "Jumlah": [len(ratings_raw), len(explicit_events), len(warm_events), catalog["book_id"].nunique()],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("🔍 Lihat Sampel Katalog Buku"):
        st.dataframe(
            catalog[["book_id", "Book_Title", "Book_Author", "Publisher", "n_isbn", "image_url"]].head(50),
            use_container_width=True, hide_index=True,
        )


def page_evaluation() -> None:
    render_hero("Perbandingan enam model pada metrik ranking, novelty, diversity, dan catalog coverage.")

    section_header("📊", "Tabel Metrik Semua Model")
    styled_metrics = METRICS.copy()
    styled_metrics.columns = [c.replace("@", " @") for c in styled_metrics.columns]
    st.dataframe(styled_metrics, use_container_width=True, hide_index=True)

    # Radar chart
    section_header("🎯", "Profil Multi-Metrik")
    radar_metrics = ["recall@10", "precision@10", "ndcg@10", "map@10", "diversity", "catalog_coverage"]
    radar_df = METRICS.copy()
    for col in radar_metrics:
        col_min, col_max = radar_df[col].min(), radar_df[col].max()
        if col_max > col_min:
            radar_df[col + "_norm"] = (radar_df[col] - col_min) / (col_max - col_min)
        else:
            radar_df[col + "_norm"] = 1.0
    norm_cols = [c + "_norm" for c in radar_metrics]
    fig_radar = go.Figure()
    colors = ["#C084FC", "#818CF8", "#38BDF8", "#34D399", "#FBBF24", "#FB7185"]
    for i, row in radar_df.iterrows():
        values = [row[c] for c in norm_cols]
        values += [values[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=values, theta=radar_metrics + [radar_metrics[0]],
            fill="toself", name=row["model"],
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=colors[i % len(colors)].replace("FC", "30").replace("F8", "20").replace("D3", "20"),
            opacity=.85,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(16,14,26,.6)",
            radialaxis=dict(visible=True, range=[0, 1], color="#8B7FA8", gridcolor="rgba(150,130,255,.12)"),
            angularaxis=dict(color="#C4B5E0", gridcolor="rgba(150,130,255,.12)"),
        ),
        showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#C4B5E0")),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="#C4B5E0"),
        title=dict(text="Perbandingan Multi-Metrik (Normalized)", font=dict(family="Syne", color="#EEE8FF")),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    section_header("📈", "Perbandingan Metrik")
    metric = st.selectbox(
        "Pilih metrik",
        ["recall@10", "precision@10", "map@10", "ndcg@10", "catalog_coverage", "novelty", "diversity"],
    )
    sorted_metrics = METRICS.sort_values(metric, ascending=False)
    fig_bar = px.bar(
        sorted_metrics, x="model", y=metric, color="model", text_auto=".3f",
        title=f"Perbandingan {metric}",
        color_discrete_sequence=["#C084FC", "#818CF8", "#38BDF8", "#34D399", "#FBBF24", "#FB7185"],
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(style_plot(fig_bar), use_container_width=True)

    st.markdown(
        """<div class="info-callout">
        <strong>Tuned Hybrid</strong> dipilih sebagai model final karena memberikan kombinasi terbaik
        pada metrik ranking utama (Recall, MAP, NDCG@10) sekaligus menjaga diversity dan catalog coverage
        yang sehat. Evaluasi dilakukan pada 250 pengguna test set.
        </div>""",
        unsafe_allow_html=True,
    )


def page_recommender(model: dict) -> None:
    render_hero("Simulasikan rekomendasi Top-10 berdasarkan profil pengguna atau kemiripan buku.")

    tab_user, tab_book = st.tabs(["👤  Berdasarkan User", "📖  Berdasarkan Buku"])

    with tab_user:
        train_events = model["train_events"]
        user_counts = train_events.groupby("User-ID").size().sort_values(ascending=False)
        user_options = user_counts.index.astype(int).tolist()

        c1, c2 = st.columns([1, 2])
        selected_user = c1.selectbox("Pilih User-ID", user_options, index=0)
        c1.markdown(f'<div style="font-size:.8rem;color:var(--muted);margin-top:6px;">📊 {user_counts.loc[selected_user]} rating dalam data training</div>', unsafe_allow_html=True)

        user_history = train_events[train_events["User-ID"].eq(selected_user)].sort_values("rating", ascending=False)
        history_view = attach_catalog(model, user_history).head(15)
        c2.markdown('<div style="font-size:.82rem;color:var(--muted);margin-bottom:8px;font-weight:600;">Riwayat Rating Tertinggi</div>', unsafe_allow_html=True)
        c2.dataframe(history_view[["Book_Title", "Book_Author", "Publisher", "rating"]], use_container_width=True, hide_index=True)

        section_header("✨", f"Rekomendasi Top-10 untuk User #{selected_user}")
        recs = recommend_for_user(int(selected_user))
        render_book_grid(recs, score_col="hybrid_score", max_items=10, columns=5)

        with st.expander("📋 Detail Skor & Sumber"):
            display_cols = ["Book_Title", "Book_Author", "Publisher", "hybrid_score", "sources", "n_isbn"]
            st.dataframe(
                recs[display_cols].assign(hybrid_score=lambda df: df["hybrid_score"].round(4)),
                use_container_width=True, hide_index=True,
            )
            fig_score = px.bar(
                recs.sort_values("hybrid_score"), x="hybrid_score", y="Book_Title",
                orientation="h", title="Skor Hybrid per Buku",
                labels={"hybrid_score": "Hybrid Score", "Book_Title": ""},
                color="hybrid_score", color_continuous_scale=["#1E1A35", "#C084FC"],
            )
            fig_score.update_coloraxes(showscale=False)
            st.plotly_chart(style_plot(fig_score), use_container_width=True)

    with tab_book:
        catalog_model = model["catalog_model"].copy()
        catalog_model["display_name"] = catalog_model["Book_Title"].astype(str) + " — " + catalog_model["Book_Author"].astype(str)
        catalog_model = catalog_model.sort_values("display_name").reset_index(drop=True)

        selected_display = st.selectbox("Pilih buku acuan", catalog_model["display_name"].tolist(), index=min(25, len(catalog_model) - 1))
        seed = catalog_model[catalog_model["display_name"].eq(selected_display)].iloc[0]

        c1, c2 = st.columns([1, 3])
        c1.markdown(book_card_html(seed, rank=1, score_label="Acuan", compact=True), unsafe_allow_html=True)
        c2.markdown('<div style="font-size:.82rem;color:var(--muted);margin-bottom:8px;font-weight:600;">Detail Buku yang Dipilih</div>', unsafe_allow_html=True)
        c2.dataframe(
            pd.DataFrame([seed])[["Book_Title", "Book_Author", "Publisher", "n_isbn"]],
            use_container_width=True, hide_index=True,
        )

        section_header("🔗", "Buku Serupa Top-10")
        book_recs = recommend_for_book(int(seed["book_id"]))
        render_book_grid(book_recs, score_col="similarity_score", max_items=10, columns=5)

        with st.expander("📋 Detail Skor Kemiripan"):
            st.dataframe(
                book_recs[["Book_Title", "Book_Author", "Publisher", "similarity_score", "sources", "n_isbn"]]
                .assign(similarity_score=lambda df: df["similarity_score"].round(4)),
                use_container_width=True, hide_index=True,
            )
            fig_book_score = px.bar(
                book_recs.sort_values("similarity_score"), x="similarity_score", y="Book_Title",
                orientation="h", title="Skor Kemiripan Buku",
                labels={"similarity_score": "Similarity", "Book_Title": ""},
                color="similarity_score", color_continuous_scale=["#1E1A35", "#38BDF8"],
            )
            fig_book_score.update_coloraxes(showscale=False)
            st.plotly_chart(style_plot(fig_book_score), use_container_width=True)


def page_about() -> None:
    render_hero("Penjelasan arsitektur model, pipeline data, dan metodologi evaluasi.")

    section_header("🔄", "Pipeline Data")
    st.markdown(
        """<div class="info-callout">
        Rating eksplisit <strong>1–10</strong> digunakan sebagai sinyal utama. Rating <strong>0</strong>
        merepresentasikan interaksi implisit dan tidak diperlakukan sebagai sinyal negatif.
        Metadata buku disatukan dari level ISBN ke level buku kanonik menggunakan kombinasi
        judul dan penulis yang sudah dinormalisasi.
        </div>""",
        unsafe_allow_html=True,
    )

    section_header("🧠", "Arsitektur Model Hybrid")
    signals = [
        ("🏆", "Bayesian Popularity", "Bobot 0% — sebagai baseline; menghindari dominasi buku populer dalam ensemble final.", "#C084FC"),
        ("📝", "Content-Based TF-IDF", "Bobot 20% — kemiripan berdasarkan representasi TF-IDF judul, penulis, dan penerbit.", "#818CF8"),
        ("🔗", "Co-Like Item KNN", "Bobot 20% — item yang sering di-like bersama oleh pengguna yang sama.", "#38BDF8"),
        ("✍️", "Author Affinity", "Bobot 20% — preferensi penulis berdasarkan riwayat rating pengguna.", "#34D399"),
        ("👥", "User-Based CF", "Bobot 40% — rekomendasi dari pengguna berselera serupa (cosine similarity).", "#FBBF24"),
    ]
    for icon, name, desc, color in signals:
        st.markdown(
            f"""<div style="display:flex;gap:14px;align-items:flex-start;padding:14px;
                background:var(--surface);border:1px solid var(--border);border-radius:14px;margin-bottom:10px;">
                <div style="font-size:1.5rem;flex-shrink:0">{icon}</div>
                <div>
                    <div style="font-family:Syne,sans-serif;font-weight:700;color:{color};margin-bottom:4px;">{name}</div>
                    <div style="font-size:0.86rem;color:var(--soft);line-height:1.5;">{desc}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    section_header("📐", "Evaluasi")
    st.markdown(
        """<div class="info-callout">
        Split data: <strong>Train → Validation → Test</strong>. Dari setiap pengguna yang memenuhi syarat,
        2 buku dengan rating ≥ 7 diambil — satu untuk validasi, satu untuk test.
        Bobot hybrid dipilih pada validation set dengan grid search; hasil akhir dilaporkan pada
        <strong>test set (250 pengguna)</strong>.
        Metrik: <strong>Precision@10</strong>, <strong>Recall@10</strong>, <strong>MAP@10</strong>,
        <strong>NDCG@10</strong>, Novelty, Diversity, Catalog Coverage.
        </div>""",
        unsafe_allow_html=True,
    )


# ─── sidebar & main ──────────────────────────────────────────────────────────

def main() -> None:
    # Sidebar
    st.sidebar.markdown(
        """<div class="sidebar-logo">
            <span class="logo-icon">📚</span>
            <div class="logo-name">BookMind</div>
            <div class="logo-tag">Hybrid Recommender System</div>
        </div>
        <div class="nav-section-label">Navigasi</div>""",
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "",
        ["🏠  Overview", "📊  Data", "📈  Evaluasi", "🤖  Rekomendasi", "📖  Metodologi"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        """<hr style="border-color:rgba(150,130,255,.14);margin:20px 0 14px">
        <div style="font-size:.72rem;color:#8B7FA8;line-height:1.5;">
            Dataset: <strong style="color:#C4B5E0">Book-Crossing</strong><br>
            Model: <strong style="color:#C4B5E0">Tuned Hybrid</strong><br>
            Evaluasi: <strong style="color:#C4B5E0">250 users · Top-10</strong>
        </div>""",
        unsafe_allow_html=True,
    )

    if "Evaluasi" in page:
        page_evaluation()
        return
    if "Metodologi" in page:
        page_about()
        return

    model = get_model()

    if "Overview" in page:
        page_overview(model)
    elif "Data" in page:
        page_data(model)
    elif "Rekomendasi" in page:
        page_recommender(model)


if __name__ == "__main__":
    main()