"""
Neuroplasticity & Articulation Coach — train your communication skills daily.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from trainer import compute_streak, load_progress, render_trainer, render_trainer_sidebar

load_dotenv()


def _get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""


OPENAI_API_KEY = _get_api_key()


@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


st.set_page_config(
    page_title="Neuroplasticity & Articulation Coach",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Styling ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;500;600;700&display=swap');

:root {
    --indigo:       #4F46E5;
    --indigo-deep:  #3730A3;
    --violet:       #7C3AED;
    --mint:         #34D399;
    --bg:           #F7F7FC;
    --ink:          #1E1B2E;
    --slate:        #64748B;
    --white:        #FFFFFF;
}

.stApp {
    background-color: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.main .block-container {
    max-width: 900px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #241E4E 0%, #16123A 100%) !important;
    border-right: 1px solid rgba(124,58,237,0.25);
}
section[data-testid="stSidebar"] * {
    color: #C7C4E0 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(124,58,237,0.3) !important;
}
section[data-testid="stSidebar"] .stMetric label {
    color: #8B87B8 !important;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
    color: #A78BFA !important;
    font-weight: 700;
    font-size: 1.1rem;
}
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
    color: #A78BFA !important;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.8rem;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(124,58,237,0.15) !important;
    border-color: #A78BFA !important;
}

/* ===== Hero ===== */
.hero-heading {
    font-family: 'Sora', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--ink);
    text-align: center;
    margin-bottom: 0.35rem;
    line-height: 1.25;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.97rem;
    color: var(--slate);
    text-align: center;
    margin-bottom: 1.6rem;
    line-height: 1.65;
    max-width: 560px;
    margin-left: auto;
    margin-right: auto;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
    justify-content: center;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    border-radius: 10px 10px 0 0;
    color: var(--slate);
}
.stTabs [aria-selected="true"] {
    color: var(--indigo) !important;
}

/* ===== Buttons ===== */
.main .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--indigo), var(--violet)) !important;
    border: none !important;
    color: #FFFFFF !important;
    border-radius: 10px;
    font-weight: 600;
    box-shadow: 0 3px 12px rgba(79,70,229,0.25);
}
.main .stButton > button[kind="primary"]:hover {
    box-shadow: 0 5px 18px rgba(79,70,229,0.4);
    transform: translateY(-1px);
}
.main .stButton > button {
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    transition: all 0.2s ease;
}

/* ===== Metrics in main area ===== */
.main [data-testid="stMetric"] {
    background: var(--white);
    border: 1px solid #E5E3F0;
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    box-shadow: 0 1px 4px rgba(30,27,46,0.05);
}
.main [data-testid="stMetric"] label {
    color: var(--slate) !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.main [data-testid="stMetricValue"] {
    color: var(--indigo) !important;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
}

/* ===== Info boxes ===== */
.stAlert {
    border-radius: 12px;
}

/* ===== Hide default Streamlit chrome ===== */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---- Sidebar ----
with st.sidebar:
    st.markdown("# 🧠 NeuroSpeak")
    st.caption("Neuroplasticity & Articulation Coach")
    st.divider()

    render_trainer_sidebar()

    st.divider()
    st.markdown("**The method**")
    st.markdown(
        "- 🔥 Warm up the articulators\n"
        "- 🎤 Retrieve ideas under pressure\n"
        "- ✂️ Prune verbal clutter\n"
        "- 🏗️ Automate structure\n"
        "- 📚 Grow your lexicon"
    )
    st.caption(
        "Five minutes daily beats an hour weekly — neuroplasticity rewards "
        "frequency, sleep consolidates each rep."
    )
    st.divider()

    history = load_progress()
    streak = compute_streak(history)
    if streak >= 3:
        st.success(f"🔥 {streak}-day streak — the wiring is happening!")

    st.caption("Progress is stored locally in `.progress/`.")


# ---- Main ----
if not OPENAI_API_KEY:
    st.markdown('<p class="hero-heading">Neuroplasticity &amp; Articulation Coach</p>', unsafe_allow_html=True)
    st.error(
        "Set **OPENAI_API_KEY** to get started — in a `.env` file (local) "
        "or Streamlit Secrets (cloud). Copy `.env.example` to `.env` and add your key."
    )
    st.stop()

render_trainer(get_openai_client())
