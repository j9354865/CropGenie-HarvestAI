"""
CropGenie HarvestAI — Streamlit Frontend
=========================================

A farmer-friendly chat interface with voice input/output and
weather-based farming recommendations.

Run with:
    streamlit run frontend/app.py

The FastAPI backend must be running first on http://localhost:8000.
"""

import hashlib
import json
import os
from datetime import datetime
from io import BytesIO

import pandas as pd
from PIL import Image
import requests
import streamlit as st

# ── Configuration ────────────────────────────────────────────────────

# Read backend URL from Streamlit Secrets (for Streamlit Cloud deployment)
# or fall back to localhost for local development.
# In Streamlit Cloud Secrets, add:  BACKEND_URL = "https://your-backend.onrender.com"
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8001")

LANGUAGES = {
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "English (India)": "en-IN",
    "اردو (Urdu)": "ur-PK",
    "پښتو (Pashto)": "ps-AF",
}


# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="CropGenie — AI Agriculture Assistant",
    page_icon="🌾",
    layout="centered",
)

# Custom CSS for a polished, farmer-friendly look
st.markdown("""
<style>
    /* ==============================================================
       CROPGENIE GLOBAL THEME — Agriculture colour palette
       ==============================================================
       A cohesive green / earthy theme applied across the whole app.
       All individual page cards (irrigation, calendar, etc.) keep
       their own colours — this block sets the *surrounding* look.
    */

    /* --- Colour variables (easy to tweak in one place) --- */
    :root {
        --cg-bg:           #F1F5EE;   /* soft sage-cream page background   */
        --cg-sidebar-bg:   #1B3A1A;   /* deep forest-green sidebar         */
        --cg-sidebar-text: #E8F0E4;   /* light text on dark sidebar        */
        --cg-accent:       #2E7D32;   /* farming-green (buttons, headers)  */
        --cg-accent-hover: #256628;   /* darker green on hover             */
        --cg-gold:         #D4A012;   /* warm wheat-gold for highlights    */
        --cg-text:         #2C2C2C;   /* dark charcoal for body text       */
        --cg-muted:        #5A6B5A;   /* sage-grey for captions / labels   */
        --cg-card:         #FFFFFF;   /* white card surfaces                */
        --cg-border:       #D5DDD2;   /* subtle sage border                */
    }

    /* ── 1. Main page background ──────────────────────────────── */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main {
        background-color: var(--cg-bg);
    }

    /* ── 2. Sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            var(--cg-sidebar-bg) 0%,
            #234D22 100%
        );
        color: var(--cg-sidebar-text);
    }
    /* Sidebar text elements — override Streamlit's default dark text */
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown div,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label p,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: var(--cg-sidebar-text) !important;
    }
    /* Sidebar captions — lighter muted tone */
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #A8C4A0 !important;
    }
    /* Sidebar dividers */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }
    /* Sidebar selectbox / toggle — make the interactive controls readable */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
    section[data-testid="stSidebar"] [data-testid="stToggle"] label {
        color: var(--cg-sidebar-text) !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.12) !important;
        border-color: rgba(255,255,255,0.2) !important;
        color: var(--cg-sidebar-text) !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: var(--cg-sidebar-text) !important;
    }
    /* Sidebar "Clear Conversation" button */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: rgba(255,255,255,0.1);
        color: var(--cg-sidebar-text);
        border: 1px solid rgba(255,255,255,0.25);
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(255,255,255,0.2);
        border-color: rgba(255,255,255,0.4);
        color: #ffffff;
    }

    /* ── 3. Headers & titles ─────────────────────────────────── */
    h1, h2, h3,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3 {
        color: var(--cg-accent) !important;
    }
    /* h4 and below keep charcoal — used for card sub-headings */
    h4, .stMarkdown h4 {
        color: var(--cg-text) !important;
    }

    /* ── 4. Buttons (primary action buttons across all pages) ── */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: var(--cg-accent);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.25);
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: var(--cg-accent-hover);
        box-shadow: 0 4px 14px rgba(46, 125, 50, 0.35);
        transform: translateY(-1px);
    }
    /* Secondary / default buttons — subtle outline style */
    .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
        border-radius: 8px;
        border: 1px solid var(--cg-border);
        color: var(--cg-accent);
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
        border-color: var(--cg-accent);
        background-color: rgba(46, 125, 50, 0.04);
    }

    /* ── 5. Input fields ─────────────────────────────────────── */
    /* Text inputs, number inputs, text areas */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    [data-testid="stChatInput"] textarea {
        border-radius: 8px !important;
        border: 1.5px solid var(--cg-border) !important;
        background-color: var(--cg-card) !important;
        color: var(--cg-text) !important;
        transition: border-color 0.2s ease;
    }
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus,
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--cg-accent) !important;
        box-shadow: 0 0 0 2px rgba(46, 125, 50, 0.12) !important;
    }
    /* Selectbox dropdowns */
    [data-baseweb="select"] > div {
        border-radius: 8px !important;
        border: 1.5px solid var(--cg-border) !important;
        background-color: var(--cg-card) !important;
    }
    [data-baseweb="select"]:focus-within > div {
        border-color: var(--cg-accent) !important;
        box-shadow: 0 0 0 2px rgba(46, 125, 50, 0.12) !important;
    }
    /* Input labels */
    .stTextInput label,
    .stNumberInput label,
    .stTextArea label,
    .stSelectbox label {
        color: var(--cg-text) !important;
        font-weight: 500 !important;
    }

    /* ── 6. Body text & captions ─────────────────────────────── */
    .main .stMarkdown p,
    .main .stMarkdown li {
        color: var(--cg-text);
    }
    .main [data-testid="stCaptionContainer"],
    .main .stCaption {
        color: var(--cg-muted) !important;
    }

    /* ── 7. Streamlit info / warning / error boxes — harmonise ── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    /* ── 8. Dividers ─────────────────────────────────────────── */
    .main hr {
        border-color: var(--cg-border);
    }

    /* ── 9. Expanders ────────────────────────────────────────── */
    .streamlit-expanderHeader {
        color: var(--cg-accent) !important;
        font-weight: 600 !important;
    }

    /* ── 10. Dataframe / table styling ───────────────────────── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Larger font for readability */
    .stChatMessage { font-size: 1.05rem; }

    /* Make the mic button area more prominent */
    [data-testid="stAudioInput"] button {
        min-height: 48px;
        min-width: 48px;
    }

    /* Sidebar header styling */
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* Weather metric cards */
    .weather-metric {
        text-align: center;
        padding: 0.6rem 0.4rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    .weather-metric .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2e7d32;
    }
    .weather-metric .label {
        font-size: 0.82rem;
        color: #666;
        margin-top: 2px;
    }

    /* Forecast card */
    .forecast-card {
        background: #f0f7f0;
        border-radius: 10px;
        padding: 0.7rem;
        text-align: center;
        height: 100%;
    }
    .forecast-card .day {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .forecast-card .temp {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1b5e20;
    }
    .forecast-card .desc {
        font-size: 0.82rem;
        color: #555;
    }
    .forecast-card .rain {
        font-size: 0.82rem;
        color: #1565c0;
    }

    /* Profit dashboard cards */
    .profit-card {
        text-align: center;
        padding: 0.8rem;
        border-radius: 12px;
        min-height: 90px;
    }
    .profit-card .value {
        font-size: 1.5rem;
        font-weight: 800;
    }
    .profit-card .label {
        font-size: 0.8rem;
        color: #555;
        margin-top: 2px;
    }
    .profit-good { background: #e8f5e9; }
    .profit-good .value { color: #2e7d32; }
    .profit-risk { background: #fff3e0; }
    .profit-risk .value { color: #e65100; }
    .profit-neutral { background: #e3f2fd; }
    .profit-neutral .value { color: #1565c0; }
    .profit-cost { background: #fce4ec; }
    .profit-cost .value { color: #c62828; }

    /* Disease detection styles */
    .disease-image-preview {
        text-align: center;
        padding: 1rem;
    }
    .disease-image-preview img {
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        max-width: 100%;
        max-height: 380px;
        object-fit: contain;
    }
    .health-badge {
        display: inline-block;
        padding: 0.45rem 1.4rem;
        border-radius: 24px;
        font-weight: 700;
        font-size: 1rem;
        text-align: center;
    }
    .health-healthy   { background: #e8f5e9; color: #1b5e20; border: 2px solid #81c784; }
    .health-diseased  { background: #ffebee; color: #b71c1c; border: 2px solid #ef5350; }
    .health-nutrient_deficiency { background: #fff3e0; color: #e65100; border: 2px solid #ff9800; }
    .health-unknown   { background: #eceff1; color: #37474f; border: 2px solid #90a4ae; }
    .disease-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #2e7d32;
    }
    .disease-card-risk {
        background: #fff8e1;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #ff9800;
    }
    .disease-card .card-title {
        font-weight: 700;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
        color: #555;
    }
    .disease-card .card-body {
        font-size: 0.95rem;
        color: #333;
    }

    /* Fertilizer recommendation styles */
    .fert-card {
        text-align: center;
        padding: 0.8rem;
        border-radius: 12px;
        min-height: 90px;
    }
    .fert-card .value {
        font-size: 1.5rem;
        font-weight: 800;
    }
    .fert-card .label {
        font-size: 0.8rem;
        color: #555;
        margin-top: 2px;
    }
    .fert-rec   { background: #e8f5e9; }
    .fert-rec .value   { color: #2e7d32; }
    .fert-info  { background: #e3f2fd; }
    .fert-info .value  { color: #1565c0; }
    .fert-warn  { background: #fff3e0; }
    .fert-warn .value  { color: #e65100; }

    /* Location / Regional Guide styles */
    .zone-badge {
        display: inline-block;
        padding: 0.45rem 1.4rem;
        border-radius: 24px;
        font-weight: 700;
        font-size: 1.05rem;
        text-align: center;
        background: #e8f5e9;
        color: #1b5e20;
        border: 2px solid #81c784;
    }
    .region-card {
        text-align: center;
        padding: 0.8rem;
        border-radius: 12px;
        min-height: 90px;
    }
    .region-card .value {
        font-size: 1.5rem;
        font-weight: 800;
    }
    .region-card .label {
        font-size: 0.8rem;
        color: #555;
        margin-top: 2px;
    }
    .region-zone    { background: #e8f5e9; }
    .region-zone .value    { color: #2e7d32; }
    .region-weather { background: #e3f2fd; }
    .region-weather .value { color: #1565c0; }
    .region-info    { background: #f3e5f5; }
    .region-info .value    { color: #7b1fa2; }
    .region-warn    { background: #fff3e0; }
    .region-warn .value    { color: #e65100; }
    .zone-detail-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #2e7d32;
    }
    .zone-detail-card .card-title {
        font-weight: 700;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
        color: #555;
    }
    .zone-detail-card .card-body {
        font-size: 0.95rem;
        color: #333;
    }

    /* Sidebar navigation — tuned for the dark forest-green sidebar */
    div[data-testid="stSidebarNav"] { display: none; }

    [data-testid="stSidebarContent"] .stRadio > div[role="radiogroup"] > label {
        padding: 0.55rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 2px;
        border-left: 3px solid transparent;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebarContent"] .stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebarContent"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
        background: rgba(255, 255, 255, 0.14);
        border-left-color: var(--cg-gold);
    }
    [data-testid="stSidebarContent"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] p {
        font-size: 0.92rem;
        font-weight: 500;
        color: var(--cg-sidebar-text) !important;
    }
    [data-testid="stSidebarContent"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) p {
        font-weight: 700;
        color: #ffffff !important;
    }
    /* Radio circle indicators — lightened for dark bg */
    [data-testid="stSidebarContent"] .stRadio [data-baseweb="radio"] > div:first-child {
        border-color: rgba(255,255,255,0.5) !important;
    }
    [data-testid="stSidebarContent"] .stRadio [data-baseweb="radio"]:has(input:checked) > div:first-child {
        border-color: var(--cg-gold) !important;
        background-color: var(--cg-gold) !important;
    }

    /* Market trend styles */
    .market-card {
        text-align: center;
        padding: 0.8rem;
        border-radius: 12px;
        min-height: 90px;
    }
    .market-card .value {
        font-size: 1.5rem;
        font-weight: 800;
    }
    .market-card .label {
        font-size: 0.8rem;
        color: #555;
        margin-top: 2px;
    }
    .market-up   { background: #e8f5e9; }
    .market-up .value   { color: #2e7d32; }
    .market-down { background: #ffebee; }
    .market-down .value { color: #c62828; }
    .market-flat { background: #e3f2fd; }
    .market-flat .value { color: #1565c0; }
    .market-info { background: #f3e5f5; }
    .market-info .value { color: #7b1fa2; }
    .trend-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .trend-rising  { background: #e8f5e9; color: #2e7d32; border: 2px solid #81c784; }
    .trend-falling { background: #ffebee; color: #c62828; border: 2px solid #ef5350; }
    .trend-stable  { background: #e3f2fd; color: #1565c0; border: 2px solid #64b5f6; }

    /* Irrigation advisory styles */
    .irrigation-card {
        text-align: center;
        padding: 1rem;
        border-radius: 14px;
        min-height: 100px;
    }
    .irrigation-card .value {
        font-size: 1.6rem;
        font-weight: 800;
    }
    .irrigation-card .label {
        font-size: 0.82rem;
        color: #555;
        margin-top: 3px;
    }
    .irr-go   { background: #e8f5e9; }
    .irr-go .value   { color: #2e7d32; }
    .irr-wait { background: #fff3e0; }
    .irr-wait .value { color: #e65100; }
    .irr-info { background: #e3f2fd; }
    .irr-info .value { color: #1565c0; }
    .irr-weather { background: #f3e5f5; }
    .irr-weather .value { color: #7b1fa2; font-size: 1.2rem; }
    .decision-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 24px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .decision-irrigate { background: #e8f5e9; color: #2e7d32; border: 2px solid #81c784; }
    .decision-wait     { background: #fff3e0; color: #e65100; border: 2px solid #ffb74d; }

    /* Farming calendar styles */
    .cal-stage {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin-bottom: 8px;
        border-left: 4px solid;
    }
    .cal-sowing   { background: #e8f5e9; border-color: #4caf50; }
    .cal-water    { background: #e3f2fd; border-color: #1e88e5; }
    .cal-fert     { background: #fff8e1; border-color: #ffa000; }
    .cal-spray    { background: #fce4ec; border-color: #e53935; }
    .cal-harvest  { background: #f3e5f5; border-color: #8e24aa; }
    .cal-stage h4 { margin: 0 0 4px 0; font-size: 0.95rem; }
    .cal-stage p, .cal-stage li { margin: 2px 0; font-size: 0.88rem; }

    /* Government support styles */
    .gov-card {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .gov-answer   { background: #e8f5e9; border-color: #2e7d32; }
    .gov-disclaimer {
        background: #fff3e0;
        border-color: #ef6c00;
        font-size: 0.82rem;
        color: #6d4c00;
    }
    .gov-card h4 { margin: 0 0 6px 0; font-size: 0.95rem; }
    .gov-card p  { margin: 2px 0; font-size: 0.9rem; line-height: 1.5; }
    .gov-province-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        background: #e3f2fd;
        color: #1565c0;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Crop recommendation styles */
    .crop-card {
        text-align: center;
        padding: 0.8rem;
        border-radius: 12px;
        min-height: 90px;
    }
    .crop-card .value {
        font-size: 1.5rem;
        font-weight: 800;
    }
    .crop-card .label {
        font-size: 0.8rem;
        color: #555;
        margin-top: 2px;
    }
    .crop-rec { background: #e8f5e9; }
    .crop-rec .value { color: #2e7d32; }
    .crop-info { background: #f5f5f5; }
    .crop-info .value { color: #333; }
    .crop-alt { background: #fff3e0; }
    .crop-alt .value { color: #e65100; }

    /* Grafting advisor styles */
    .graft-card {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .graft-advice { background: #e8f5e9; border-color: #2e7d32; }
    .graft-meta   { background: #f5f5f5; border-color: #9e9e9e; }
    .graft-card h4 { margin: 0 0 6px 0; font-size: 0.95rem; }
    .graft-card p  { margin: 2px 0; font-size: 0.9rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ─────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "audio_data" not in st.session_state:
    st.session_state.audio_data = []

if "last_audio_fingerprint" not in st.session_state:
    st.session_state.last_audio_fingerprint = None

if "auto_play" not in st.session_state:
    st.session_state.auto_play = True

if "processing" not in st.session_state:
    st.session_state.processing = False

if "weather_result" not in st.session_state:
    st.session_state.weather_result = None

if "weather_error" not in st.session_state:
    st.session_state.weather_error = None

if "profit_result" not in st.session_state:
    st.session_state.profit_result = None

if "profit_error" not in st.session_state:
    st.session_state.profit_error = None

if "disease_result" not in st.session_state:
    st.session_state.disease_result = None

if "disease_error" not in st.session_state:
    st.session_state.disease_error = None

if "fert_result" not in st.session_state:
    st.session_state.fert_result = None

if "fert_error" not in st.session_state:
    st.session_state.fert_error = None

if "region_result" not in st.session_state:
    st.session_state.region_result = None

if "region_error" not in st.session_state:
    st.session_state.region_error = None

if "market_result" not in st.session_state:
    st.session_state.market_result = None

if "market_error" not in st.session_state:
    st.session_state.market_error = None

if "irrigation_result" not in st.session_state:
    st.session_state.irrigation_result = None

if "irrigation_error" not in st.session_state:
    st.session_state.irrigation_error = None

if "calendar_result" not in st.session_state:
    st.session_state.calendar_result = None

if "calendar_error" not in st.session_state:
    st.session_state.calendar_error = None

if "gov_support_result" not in st.session_state:
    st.session_state.gov_support_result = None

if "gov_support_error" not in st.session_state:
    st.session_state.gov_support_error = None

if "crop_rec_result" not in st.session_state:
    st.session_state.crop_rec_result = None

if "crop_rec_error" not in st.session_state:
    st.session_state.crop_rec_error = None

if "grafting_result" not in st.session_state:
    st.session_state.grafting_result = None

if "grafting_error" not in st.session_state:
    st.session_state.grafting_error = None

if "best_market_result" not in st.session_state:
    st.session_state.best_market_result = None

if "best_market_error" not in st.session_state:
    st.session_state.best_market_error = None

if "harvest_loss_result" not in st.session_state:
    st.session_state.harvest_loss_result = None

if "harvest_loss_error" not in st.session_state:
    st.session_state.harvest_loss_error = None

# ── Offline Mode session state ────────────────────────────────────────
# Tracks whether the backend is reachable and stores cached data with
# timestamps so we can show "last known" data when the connection drops.
if "offline_mode" not in st.session_state:
    st.session_state.offline_mode = False

# Cached timestamps — one per data page.  Set when a successful API call
# returns, so we can display "last updated at …" when offline.
if "weather_cached_at" not in st.session_state:
    st.session_state.weather_cached_at = None
if "market_cached_at" not in st.session_state:
    st.session_state.market_cached_at = None
if "calendar_cached_at" not in st.session_state:
    st.session_state.calendar_cached_at = None
if "irrigation_cached_at" not in st.session_state:
    st.session_state.irrigation_cached_at = None
if "profit_cached_at" not in st.session_state:
    st.session_state.profit_cached_at = None
if "best_market_cached_at" not in st.session_state:
    st.session_state.best_market_cached_at = None


# ── Helper: Load the offline knowledge base ──────────────────────────
# This JSON file lives next to this script so it can always be read from
# disk, even when there is zero internet connectivity.
@st.cache_data  # only loaded once per session — very fast
def load_offline_knowledge_base() -> dict:
    """Read the bundled farming knowledge base JSON from disk."""
    kb_path = os.path.join(os.path.dirname(__file__), "offline_knowledge_base.json")
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ── Helper: Check backend connectivity ────────────────────────────────
def check_backend_connectivity() -> bool:
    """
    Try a quick request to the backend health endpoint.
    Returns True if the backend is reachable, False otherwise.
    Also updates st.session_state.offline_mode.
    """
    try:
        resp = requests.get(f"{BACKEND_URL}/api/health", timeout=3)
        if resp.status_code == 200:
            st.session_state.offline_mode = False
            return True
    except (requests.ConnectionError, requests.Timeout, Exception):
        pass
    st.session_state.offline_mode = True
    return False


# Run the connectivity check on every page load.
# This is lightweight (3-second timeout) and keeps the offline status fresh.
_backend_online = check_backend_connectivity()


# ── Helper Functions ─────────────────────────────────────────────────

def call_speech_to_text(audio_bytes: bytes, language: str) -> str | None:
    """Send audio to backend STT endpoint; return transcribed text or None."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/speech-to-text",
            files={"audio": ("recording.webm", audio_bytes, "audio/octet-stream")},
            data={"language": language},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")
    except requests.ConnectionError:
        st.error("Cannot reach the backend server. Is it running on port 8001?")
        return None
    except requests.Timeout:
        st.error("Speech recognition is taking too long. Please try again.")
        return None
    except Exception as exc:
        st.error(f"Speech recognition error: {exc}")
        return None


# Maximum messages to send with each chat request (prevents token bloat
# and hitting rate limits faster).  Mirrors the backend MAX_HISTORY_MESSAGES.
_MAX_CHAT_HISTORY = 10


def call_chat_api(message: str, history: list[dict], language: str) -> str | None:
    """Send message + history to backend chat endpoint; return AI reply or None."""
    # Cap history to the most recent _MAX_CHAT_HISTORY messages.
    # This prevents token bloat that causes rate-limit errors.
    trimmed_history = history[-_MAX_CHAT_HISTORY:] if len(history) > _MAX_CHAT_HISTORY else history

    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "message": message,
                "history": trimmed_history,
                "language": language,
            },
            timeout=120,
        )

        # Handle 429 (duplicate request from dedup guard)
        if resp.status_code == 429:
            return None  # silently ignore duplicate

        # Handle 503 (rate limit / AI unavailable) specifically
        if resp.status_code == 503:
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                pass

            # Check if the server gave us an explicit wait time
            import re as _re
            wait_match = _re.search(r"wait about (\d+)\s*s", detail, _re.IGNORECASE)
            if wait_match:
                seconds = int(wait_match.group(1))
                st.warning(
                    f"AI is rate-limited. Please wait about **{seconds} seconds** "
                    f"and try again."
                )
            elif "rate limit" in detail.lower() or "busy" in detail.lower():
                st.warning(
                    "AI is busy right now. Please wait a few seconds and try again."
                )
            else:
                st.warning(
                    "AI is temporarily unavailable. Please try again in a moment."
                )
            return None

        resp.raise_for_status()
        return resp.json().get("response", "")

    except requests.ConnectionError:
        st.error("Cannot reach the backend server. Is it running on port 8001?")
        return None
    except requests.Timeout:
        st.error("The AI is taking too long to respond. Please try again.")
        return None
    except Exception as exc:
        st.error(f"Chat error: {exc}")
        return None


def call_text_to_speech(text: str, language: str) -> bytes | None:
    """
    Send text to backend TTS endpoint; return MP3 audio bytes or None.

    Retries once on failure to handle transient network issues.
    """
    for attempt in range(2):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/api/text-to-speech",
                json={"text": text, "language": language},
                timeout=60,
            )
            resp.raise_for_status()
            audio = resp.content
            if audio and len(audio) > 100:
                return audio
        except requests.Timeout:
            if attempt == 0:
                continue  # retry once
            st.warning("Voice is taking too long. You can still read the text above.")
        except requests.ConnectionError:
            st.warning("Cannot reach server for voice. You can still read the text above.")
            return None
        except Exception:
            if attempt == 0:
                continue  # retry once
            st.warning("Voice playback unavailable. You can still read the text above.")
    return None


def call_weather_api(city: str) -> dict | None:
    """
    Call /api/weather with a city name.
    Returns the full WeatherResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/weather",
            json={"city": city},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.weather_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.weather_error = (
            "Weather lookup is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.weather_error = detail or f"Weather error: {exc}"
        return None
    except Exception as exc:
        st.session_state.weather_error = f"Weather error: {exc}"
        return None


def call_predict_api(payload: dict) -> dict | None:
    """
    Call /api/predict-profit with crop, acres, city, and soil parameters.
    Returns the full ProfitResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/predict-profit",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.profit_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.profit_error = (
            "Prediction is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.profit_error = detail or f"Prediction error: {exc}"
        return None
    except Exception as exc:
        st.session_state.profit_error = f"Prediction error: {exc}"
        return None


def _compress_image(image_bytes: bytes, max_mb: float = 3.0) -> tuple[bytes, str]:
    """
    Resize/compress an image if it exceeds *max_mb*.
    Returns (compressed_bytes, mime_type).  If the image is already small
    enough it is returned unchanged with its original MIME type inferred
    from the file extension handled by Streamlit.
    """
    threshold = int(max_mb * 1024 * 1024)
    if len(image_bytes) <= threshold:
        return image_bytes, "image/jpeg"

    img = Image.open(BytesIO(image_bytes))
    # Convert RGBA to RGB for JPEG encoding
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    # Resize if either dimension exceeds 1200px
    max_dim = 1200
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue(), "image/jpeg"


def call_disease_api(
    image_bytes: bytes,
    mime_type: str,
    language: str,
) -> dict | None:
    """
    Call /api/detect-disease with an image file and language code.
    Returns the DiseaseDetectionResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/detect-disease",
            files={"image": ("plant.jpg", image_bytes, mime_type)},
            data={"language": language},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.disease_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.disease_error = (
            "Analysis is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.disease_error = detail or f"Analysis error: {exc}"
        return None
    except Exception as exc:
        st.session_state.disease_error = f"Analysis error: {exc}"
        return None


def call_fertilizer_api(payload: dict) -> dict | None:
    """
    Call /api/recommend-fertilizer with crop, soil, NPK, and context.
    Returns the FertilizerResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/recommend-fertilizer",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.fert_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.fert_error = (
            "Recommendation is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.fert_error = detail or f"Fertilizer error: {exc}"
        return None
    except Exception as exc:
        st.session_state.fert_error = f"Fertilizer error: {exc}"
        return None


def call_location_api(payload: dict) -> dict | None:
    """
    Call /api/location-insights with a city and language code.
    Returns the LocationInsightsResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/location-insights",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.region_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.region_error = (
            "Regional analysis is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.region_error = detail or f"Regional analysis error: {exc}"
        return None
    except Exception as exc:
        st.session_state.region_error = f"Regional analysis error: {exc}"
        return None


def call_market_api(payload: dict) -> dict | None:
    """
    Call /api/market-trend with a commodity and language code.
    Returns the MarketTrendResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/market-trend",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.market_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.market_error = (
            "Market analysis is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.market_error = detail or f"Market error: {exc}"
        return None
    except Exception as exc:
        st.session_state.market_error = f"Market error: {exc}"
        return None


def call_best_market_api(payload: dict) -> dict | None:
    """
    Call /api/best-market/recommend with crop + location + language.
    Returns the BestMarketResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/best-market/recommend",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.best_market_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.best_market_error = (
            "Market search is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.best_market_error = detail or f"Best market error: {exc}"
        return None
    except Exception as exc:
        st.session_state.best_market_error = f"Best market error: {exc}"
        return None


def call_irrigation_api(payload: dict) -> dict | None:
    """
    Call /api/irrigation/recommend with crop + location + optional fields.
    Returns the IrrigationResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/irrigation/recommend",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.irrigation_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.irrigation_error = (
            "Irrigation analysis is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.irrigation_error = detail or f"Irrigation error: {exc}"
        return None
    except Exception as exc:
        st.session_state.irrigation_error = f"Irrigation error: {exc}"
        return None


def call_harvest_loss_api(payload: dict) -> dict | None:
    """
    Call /api/harvest-loss/predict with crop, location, growth stage, etc.
    Returns the HarvestLossResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/harvest-loss/predict",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.harvest_loss_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.harvest_loss_error = (
            "Harvest loss analysis is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.harvest_loss_error = (
            detail or f"Harvest loss error: {exc}"
        )
        return None
    except Exception as exc:
        st.session_state.harvest_loss_error = f"Harvest loss error: {exc}"
        return None


def call_calendar_api(payload: dict) -> dict | None:
    """
    Call /api/calendar/generate with crop + location + optional sowing date.
    Returns the CalendarResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/calendar/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.calendar_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.calendar_error = (
            "Calendar generation is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.calendar_error = detail or f"Calendar error: {exc}"
        return None
    except Exception as exc:
        st.session_state.calendar_error = f"Calendar error: {exc}"
        return None


def call_gov_support_api(payload: dict) -> dict | None:
    """
    Call /api/gov-support/ask with a farmer's question + optional province.
    Returns the GovSupportResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/gov-support/ask",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.gov_support_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.gov_support_error = (
            "The request is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.gov_support_error = detail or f"Gov support error: {exc}"
        return None
    except Exception as exc:
        st.session_state.gov_support_error = f"Gov support error: {exc}"
        return None


def call_crop_recommend_api(payload: dict) -> dict | None:
    """
    Call /api/crop/recommend with NPK + climate parameters.
    Returns the CropRecommendationResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/crop/recommend",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.crop_rec_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.crop_rec_error = (
            "Recommendation is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.crop_rec_error = detail or f"Crop recommendation error: {exc}"
        return None
    except Exception as exc:
        st.session_state.crop_rec_error = f"Crop recommendation error: {exc}"
        return None


def call_grafting_api(payload: dict) -> dict | None:
    """
    Call /api/grafting/advise with base_crop, goal, and language.
    Returns the GraftingResponse dict or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/grafting/advise",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.session_state.grafting_error = (
            "Cannot reach the backend server. Is it running on port 8001?"
        )
        return None
    except requests.Timeout:
        st.session_state.grafting_error = (
            "The request is taking too long. Please try again."
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.session_state.grafting_error = detail or f"Grafting error: {exc}"
        return None
    except Exception as exc:
        st.session_state.grafting_error = f"Grafting error: {exc}"
        return None


def submit_message(text: str, language: str) -> None:
    """
    Add the user's message to history, call the AI, store the response,
    and generate TTS audio if auto-play is enabled.

    If the AI call fails, the user message is removed from history so
    unanswered messages don't accumulate and waste token quota.
    """
    if not text.strip():
        return

    # ── Frontend dedup guard: prevent rapid-fire double submissions ──
    import time as _time
    now = _time.monotonic()
    last_submit = st.session_state.get("_last_submit_time", 0)
    if (now - last_submit) < 1.5:  # 1.5-second minimum between submissions
        return
    st.session_state._last_submit_time = now

    st.session_state.messages.append({"role": "user", "content": text})
    history = st.session_state.messages[:-1]

    with st.spinner("🌱 Thinking..."):
        reply = call_chat_api(text, history, language)

    if not reply:
        # Remove the user message so unanswered messages don't pollute
        # the history and waste tokens on the next request.
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            st.session_state.messages.pop()
        st.rerun()
        return

    st.session_state.messages.append({"role": "assistant", "content": reply})

    if st.session_state.auto_play:
        with st.spinner("🔊 Generating voice..."):
            audio = call_text_to_speech(reply, language)
        st.session_state.audio_data.append(audio)
    else:
        st.session_state.audio_data.append(None)

    st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🌾 CropGenie")
    st.caption("AI Agriculture Assistant")
    st.divider()

    # Navigation
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "chat"

    _nav_options = [
        "chat", "weather", "profit", "disease", "fert", "crop_rec",
        "grafting", "region", "market", "best_market", "irrigation",
        "calendar", "gov_support", "harvest_loss", "offline",
    ]
    _nav_labels = {
        "chat": "💬  Chat Assistant",
        "weather": "🌤️  Weather",
        "profit": "📊  Profit Prediction",
        "disease": "🔍  Disease Detection",
        "fert": "🧪  Fertilizer",
        "crop_rec": "🌾  Crop Recommendation",
        "grafting": "🌱  Grafting Advisor",
        "region": "📍  Regional Guide",
        "market": "📈  Market Trends",
        "best_market": "🏪  Best Market",
        "irrigation": "💧  Irrigation",
        "calendar": "🗓️  Farming Calendar",
        "gov_support": "🏛️  Gov Support",
        "harvest_loss": "⏳  Harvest Loss",
        "offline": "📴  Offline Mode",
    }

    page = st.radio(
        "Navigate",
        options=_nav_options,
        format_func=lambda x: _nav_labels[x],
        key="nav_radio",
        index=_nav_options.index(
            st.session_state.get("nav_page", "chat")
            if st.session_state.get("nav_page", "chat") in _nav_options
            else "chat"
        ),
        label_visibility="collapsed",
    )
    st.session_state.nav_page = page

    st.divider()

    # Language selector
    language_display = st.selectbox(
        "🗣️ Language",
        options=list(LANGUAGES.keys()),
        index=0,
        help="Choose the language for voice input and output.",
    )
    language_code = LANGUAGES[language_display]

    # Voice settings
    st.session_state.auto_play = st.toggle(
        "🔊 Auto-play voice responses",
        value=st.session_state.auto_play,
        help="Automatically speak the AI's answers aloud.",
    )

    st.divider()

    # Clear conversation
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.audio_data = []
        st.session_state.last_audio_fingerprint = None
        st.session_state.weather_result = None
        st.session_state.weather_error = None
        st.session_state.profit_result = None
        st.session_state.profit_error = None
        st.session_state.disease_result = None
        st.session_state.disease_error = None
        st.session_state.fert_result = None
        st.session_state.fert_error = None
        st.session_state.region_result = None
        st.session_state.region_error = None
        st.session_state.market_result = None
        st.session_state.market_error = None
        st.session_state.irrigation_result = None
        st.session_state.irrigation_error = None
        st.session_state.calendar_result = None
        st.session_state.calendar_error = None
        st.session_state.gov_support_result = None
        st.session_state.gov_support_error = None
        st.session_state.crop_rec_result = None
        st.session_state.crop_rec_error = None
        st.session_state.grafting_result = None
        st.session_state.grafting_error = None
        st.session_state.best_market_result = None
        st.session_state.best_market_error = None
        # Reset offline caches
        st.session_state.weather_cached_at = None
        st.session_state.market_cached_at = None
        st.session_state.calendar_cached_at = None
        st.session_state.irrigation_cached_at = None
        st.session_state.profit_cached_at = None
        st.session_state.best_market_cached_at = None
        st.rerun()

    st.divider()
    st.caption("Chat and photo check by AI")
    st.caption("Voice: speak and listen")
    st.caption("Weather: live forecast")
    st.caption("Crop advice: smart models")


# ── Page Content ──────────────────────────────────────────────────────

st.title("🌾 CropGenie HarvestAI")

# ── Offline Banner ───────────────────────────────────────────────────
# Show a prominent warning banner on EVERY page when the backend is
# unreachable, so the farmer always knows they are seeing limited data.
if st.session_state.offline_mode:
    st.warning(
        "📴 **You are offline (no internet).**\n\n"
        "AI chat, weather, and market prices cannot load right now. "
        "Some things you already looked at may still show below. "
        "Go to **📴 Offline Mode** for crop tips that work without internet."
    )


# ─────────────────────────────────────────────────────────────────────
# PAGE 1 — Chat Assistant
# ─────────────────────────────────────────────────────────────────────
if page == "chat":
    st.caption("Ask about crops, diseases, fertilizer — type or speak your question!")

    # Render conversation history
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "👨‍🌾" if msg["role"] == "user" else "🌱"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

        if msg["role"] == "assistant" and idx < len(st.session_state.audio_data):
            audio_bytes = st.session_state.audio_data[idx]
            if audio_bytes is not None:
                st.audio(audio_bytes, format="audio/mpeg")

    # ── Voice Input ─────────────────────────────────────────────────
    st.divider()

    col_mic, col_spacer = st.columns([1, 5])
    with col_mic:
        st.markdown("**🎤 Voice:**")

    audio_value = st.audio_input(
        "Record your question",
        key="mic_recorder",
        label_visibility="collapsed",
    )

    if audio_value is not None and not st.session_state.processing:
        audio_bytes = audio_value.getvalue()
        fingerprint = hashlib.md5(audio_bytes).hexdigest()

        if fingerprint != st.session_state.last_audio_fingerprint:
            st.session_state.last_audio_fingerprint = fingerprint
            st.session_state.processing = True
            try:
                with st.spinner("🎤 Listening..."):
                    transcribed = call_speech_to_text(audio_bytes, language_code)

                if transcribed and not transcribed.startswith("Sorry") and not transcribed.startswith("The speech") and not transcribed.startswith("Speech recognition"):
                    submit_message(transcribed, language_code)
                elif transcribed:
                    st.warning(transcribed)
            except Exception as exc:
                st.error(f"Voice error: {exc}. Please type your question instead.")
            finally:
                st.session_state.processing = False

    # ── Text Input ──────────────────────────────────────────────────
    user_input = st.chat_input(
        "Type your farming question here...",
        key="chat_input_box",
    )

    if user_input:
        submit_message(user_input, language_code)

    # ── Empty State ─────────────────────────────────────────────────
    if not st.session_state.messages:
        st.info(
            "👋 **Welcome to CropGenie!**\n\n"
            "I am here to help you with farming. You can:\n\n"
            "  • **Type** your question in the box below\n"
            "  • **Speak** your question using the microphone\n\n"
            "Try asking:\n"
            '  * "When should I plant wheat?"\n'
            '  * "My tomato leaves have spots. What do I do?"\n'
            '  * "Which fertilizer should I use for rice?"\n'
        )


# ─────────────────────────────────────────────────────────────────────
# PAGE 2 — Weather Recommendations
# ─────────────────────────────────────────────────────────────────────
elif page == "weather":
    st.caption("Get farming tips based on today's weather in your area.")

    # ── Location input ──────────────────────────────────────────────
    col_city, col_btn = st.columns([3, 1])
    with col_city:
        city_input = st.text_input(
            "Enter your city or town",
            placeholder="e.g. Lahore, Peshawar, Mumbai, Istanbul",
            key="weather_city_input",
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)  # spacer to align with input
        get_weather = st.button("🌤️ Get Recommendation", use_container_width=True)

    # ── Fetch weather data on button click ──────────────────────────
    # CACHING LOGIC: On success, store result + timestamp.
    #                On failure, keep the old cached result (if any) and
    #                show it with a "may be outdated" warning.
    if get_weather:
        if not city_input.strip():
            st.warning("Please enter a city or town name.")
        else:
            st.session_state.weather_error = None
            with st.spinner(f"🌍 Fetching weather for {city_input}..."):
                result = call_weather_api(city_input.strip())
                if result:
                    # ✅ Success — store fresh data + timestamp
                    st.session_state.weather_result = result
                    st.session_state.weather_cached_at = datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    st.session_state.weather_error = None
                else:
                    # ❌ Failed — weather_result still holds old cached data
                    #    (or None if nothing was ever fetched)
                    pass

    # ── Display errors ──────────────────────────────────────────────
    if st.session_state.weather_error:
        st.error(st.session_state.weather_error)

    # ── Show cached-data banner if showing stale data ───────────────
    if (
        st.session_state.offline_mode
        and st.session_state.weather_result
        and st.session_state.weather_cached_at
    ):
        st.info(
            f"⚠️ **Showing old weather data** "
            f"(from {st.session_state.weather_cached_at}) \u2014 "
            f"could not get new data (no internet). This may be out of date."
        )

    # ── If offline and nothing was ever cached, guide the user ──────
    if (
        st.session_state.offline_mode
        and not st.session_state.weather_result
    ):
        st.info(
            "📴 **No weather data right now.**\n\n"
            "Weather needs internet to work. "
            "Try the **📴 Offline Mode** page for farming tips "
            "that work without internet."
        )

    # ── Display weather results ─────────────────────────────────────
    data = st.session_state.weather_result
    if data:
        current = data["current"]
        forecast = data.get("forecast", [])
        advice = data.get("advice", "")

        st.divider()

        # ── Current Weather Card ────────────────────────────────────
        st.subheader(f"📍 {current['city']} — Current Weather")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="weather-metric">'
                f'<div class="value">{current["temperature"]}°C</div>'
                f'<div class="label">Temperature</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="weather-metric">'
                f'<div class="value">{current["humidity"]}%</div>'
                f'<div class="label">Humidity</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="weather-metric">'
                f'<div class="value">{current["wind_speed"]}</div>'
                f'<div class="label">Wind (m/s)</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="weather-metric">'
                f'<div class="value">{current["feels_like"]}°C</div>'
                f'<div class="label">Feels Like</div></div>',
                unsafe_allow_html=True,
            )

        st.caption(f"☁️ {current['description']}")

        # ── 3-Day Forecast ──────────────────────────────────────────
        if forecast:
            st.subheader("📅 3-Day Forecast")
            fcols = st.columns(len(forecast))
            for i, day in enumerate(forecast):
                with fcols[i]:
                    rain_pct = int(day["rain_chance"] * 100)
                    st.markdown(
                        f'<div class="forecast-card">'
                        f'<div class="day">{day["date"]}</div>'
                        f'<div class="temp">{day["temp_min"]}–{day["temp_max"]}°C</div>'
                        f'<div class="desc">{day["description"]}</div>'
                        f'<div class="rain">🌧️ {rain_pct}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── AI Farming Advice ───────────────────────────────────────
        if advice:
            st.divider()
            st.subheader("🌱 AI Farming Advice")
            st.markdown(advice)


# ─────────────────────────────────────────────────────────────────────
# PAGE 3 — Profit Prediction
# ─────────────────────────────────────────────────────────────────────
elif page == "profit":
    st.caption("See how much crop and money you can make this season.")

    CROPS = ["Wheat", "Rice", "Cotton", "Maize", "Sugarcane"]

    # ── Input form ──────────────────────────────────────────────────
    form_col1, form_col2, form_col3 = st.columns(3)
    with form_col1:
        crop_select = st.selectbox(
            "🌾 Crop", options=CROPS, key="profit_crop"
        )
    with form_col2:
        acres_input = st.number_input(
            "📐 Farm Size (acres)",
            min_value=0.5, max_value=500.0, value=5.0, step=0.5,
            key="profit_acres",
        )
    with form_col3:
        city_input_profit = st.text_input(
            "📍 City",
            placeholder="e.g. Lahore, Peshawar",
            key="profit_city",
        )

    # Advanced soil/fertilizer inputs (collapsible)
    with st.expander("⚙️ Advanced — Soil & Fertilizer Parameters"):
        adv1, adv2, adv3, adv4 = st.columns(4)
        with adv1:
            fert_input = st.number_input(
                "Fertilizer (kg/acre)", 40.0, 90.0, 65.0, 1.0,
                key="profit_fert"
            )
        with adv2:
            n_input = st.number_input(
                "Nitrogen (N)", 50.0, 85.0, 70.0, 1.0,
                key="profit_n"
            )
        with adv3:
            p_input = st.number_input(
                "Phosphorus (P)", 15.0, 28.0, 21.0, 0.5,
                key="profit_p"
            )
        with adv4:
            k_input = st.number_input(
                "Potassium (K)", 12.0, 24.0, 18.0, 0.5,
                key="profit_k"
            )

    st.markdown("")  # spacer

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_btn = st.button(
            "📊 Predict Profit", use_container_width=True, type="primary"
        )

    # ── Run prediction on button click ─────────────────────────────
    if predict_btn:
        if not city_input_profit.strip():
            st.warning("Please enter a city or town name.")
        else:
            st.session_state.profit_result = None
            st.session_state.profit_error = None
            payload = {
                "crop": crop_select,
                "acres": acres_input,
                "city": city_input_profit.strip(),
                "fertilizer": fert_input,
                "nitrogen": n_input,
                "phosphorus": p_input,
                "potassium": k_input,
            }
            with st.spinner(f"🔬 Analysing {crop_select} for {city_input_profit}..."):
                result = call_predict_api(payload)
                if result:
                    st.session_state.profit_result = result
                    st.session_state.profit_error = None

    # ── Display errors ──────────────────────────────────────────────
    if st.session_state.profit_error:
        st.error(st.session_state.profit_error)

    # ── Display profit dashboard ────────────────────────────────────
    pdata = st.session_state.profit_result
    if pdata:
        st.divider()

        # Header
        st.subheader(f"🌾 {pdata['crop']} — {pdata['acres']} acres in {pdata['weather_city']}")

        # ── Metric cards ────────────────────────────────────────────
        profit_class = "profit-risk" if pdata["has_risk"] else "profit-good"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="profit-card profit-neutral">'
                f'<div class="value">{pdata["predicted_yield"]}</div>'
                f'<div class="label">Yield / acre (maunds)</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="profit-card profit-neutral">'
                f'<div class="value">{pdata["total_yield"]:.0f}</div>'
                f'<div class="label">Total Yield (maunds)</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="profit-card {profit_class}">'
                f'<div class="value">₨{pdata["profit"]:,.0f}</div>'
                f'<div class="label">Est. Profit (PKR)</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="profit-card profit-good">'
                f'<div class="value">{pdata["profit_margin"]}%</div>'
                f'<div class="label">Profit Margin</div></div>',
                unsafe_allow_html=True,
            )

        # ── Revenue vs Cost bar chart ──────────────────────────────
        st.markdown("**💰 Revenue vs Cost**")
        chart_df = pd.DataFrame({
            "Category": ["Revenue", "Cost", "Profit"],
            "Amount (PKR)": [pdata["revenue"], pdata["cost"], pdata["profit"]],
        })
        st.bar_chart(chart_df.set_index("Category"), color="#2e7d32")

        # ── Weather context ────────────────────────────────────────
        st.caption(
            f"🌡️ Weather: {pdata['temperature']}°C, "
            f"Humidity: {pdata['humidity']}% "
            f"(live data from {pdata['weather_city']})"
        )

        # ── Risk warning ───────────────────────────────────────────
        if pdata["has_risk"] and pdata["risk_warning"]:
            st.warning(
                f"**⚠️ Weather Risk Detected**\n\n{pdata['risk_warning']}"
            )
        else:
            st.success("✅ No weather risks detected for this crop and location.")


# ─────────────────────────────────────────────────────────────────────
# PAGE 4 — Disease Detection
# ─────────────────────────────────────────────────────────────────────
elif page == "disease":
    st.caption("Upload a photo of a sick plant \u2014 we will tell you what's wrong and how to fix it.")

    # ── Image upload ──────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload or take a photo of the affected plant",
        type=["jpg", "jpeg", "png", "webp"],
        key="disease_image_upload",
        help="JPEG, PNG or WebP — max 5 MB. Use your device camera for best results.",
    )

    # ── Image preview + Analyze button ────────────────────────────
    if uploaded_file is not None:
        # Show preview centered
        col_left, col_img, col_right = st.columns([1, 3, 1])
        with col_img:
            st.image(uploaded_file, use_container_width=True)

        st.markdown("")  # spacer
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            analyze_btn = st.button(
                "🔍 Analyse Plant",
                use_container_width=True,
                type="primary",
                key="disease_analyze_btn",
            )

        # ── Run analysis on button click ──────────────────────────
        if analyze_btn:
            st.session_state.disease_result = None
            st.session_state.disease_error = None

            raw_bytes = uploaded_file.getvalue()
            image_bytes, mime_type = _compress_image(raw_bytes, max_mb=3.0)

            with st.spinner("🔬 Analysing your plant... This may take a few seconds."):
                result = call_disease_api(image_bytes, mime_type, language_code)
                if result:
                    st.session_state.disease_result = result
                    st.session_state.disease_error = None

    # ── Display errors ────────────────────────────────────────────
    if st.session_state.disease_error:
        st.error(st.session_state.disease_error)

    # ── Display results ───────────────────────────────────────────
    ddata = st.session_state.disease_result
    if ddata:
        st.divider()

        # Health status badge — centered
        status = ddata.get("health_status", "unknown")
        status_display = {
            "healthy": "✅ Healthy",
            "diseased": "🦠 Diseased",
            "nutrient_deficiency": "🍂 Nutrient Deficiency",
            "unknown": "❓ Unable to Determine",
        }.get(status, "❓ Unknown")

        badge_html = (
            f'<div style="text-align:center; margin-bottom:0.5rem;">'
            f'<span class="health-badge health-{status}">{status_display}</span>'
            f'</div>'
        )
        st.markdown(badge_html, unsafe_allow_html=True)

        # Plant name — prominent
        plant_name = ddata.get("plant_name", "Unknown")
        st.markdown(
            f'<div style="text-align:center; font-size:1.4rem; font-weight:700; '
            f'margin-bottom:1rem;">🌱 {plant_name}</div>',
            unsafe_allow_html=True,
        )

        # Summary
        summary = ddata.get("summary", "")
        if summary:
            if status == "healthy":
                st.success(f"**{summary}**")
            elif status == "unknown":
                st.info(f"**{summary}**")
            else:
                st.warning(f"**{summary}**")

        st.markdown("")  # spacer

        # ── Disease name + Cause cards ────────────────────────────
        disease_name = ddata.get("disease_name")
        cause = ddata.get("cause")

        if disease_name or cause:
            col_d, col_c = st.columns(2)
            with col_d:
                if disease_name:
                    st.markdown(
                        f'<div class="disease-card-risk">'
                        f'<div class="card-title">🦠 Disease / Issue</div>'
                        f'<div class="card-body">{disease_name}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with col_c:
                if cause:
                    st.markdown(
                        f'<div class="disease-card-risk">'
                        f'<div class="card-title">🔎 Likely Cause</div>'
                        f'<div class="card-body">{cause}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Treatment Steps ───────────────────────────────────────
        treatment = ddata.get("treatment", [])
        if treatment:
            st.markdown(
                f'<div class="disease-card">'
                f'<div class="card-title">💊 Recommended Treatment</div>',
                unsafe_allow_html=True,
            )
            for i, step in enumerate(treatment, 1):
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{i}. {step}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Prevention Tips ───────────────────────────────────────
        prevention = ddata.get("prevention", [])
        if prevention:
            st.markdown(
                f'<div class="disease-card">'
                f'<div class="card-title">🛡️ Prevention Tips</div>',
                unsafe_allow_html=True,
            )
            for tip in prevention:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {tip}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Empty state message ─────────────────────────────────────
        if not disease_name and not treatment and not prevention and status == "healthy":
            st.success(
                "🎉 Your plant looks healthy! Continue regular care "
                "and monitoring to keep it in great shape."
            )

    # ── Tab empty state ───────────────────────────────────────────
    if uploaded_file is None and not st.session_state.disease_result:
        st.info(
            "📸 **How to check plant disease:**\n\n"
            "1. **Upload a photo** of the sick plant, leaf, or fruit\n"
            "2. Click **Check Plant** to get your answer\n"
            "3. See the result — what disease it is, what caused it, "
            "and how to fix it\n\n"
            "**Tips for best results:**\n"
            "  \u2022 Take photos in good light (daylight is best)\n"
            "  \u2022 Get close to the sick part of the plant\n"
            "  \u2022 Show both healthy and sick parts if possible"
        )


# ─────────────────────────────────────────────────────────────────────
# PAGE 5 — Fertilizer Recommendation
# ─────────────────────────────────────────────────────────────────────
elif page == "fert":
    st.caption("Get fertilizer advice for your crop and soil type.")

    # Dataset-driven options
    CROPS_FERT = [
        "Barley", "Cotton", "Ground Nuts", "Maize", "Millets",
        "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat",
    ]
    SOIL_TYPES = ["Black", "Clayey", "Loamy", "Red", "Sandy"]

    # ── Input form ──────────────────────────────────────────────────
    form_col1, form_col2, form_col3 = st.columns(3)
    with form_col1:
        crop_fert = st.selectbox(
            "🌾 Crop", options=CROPS_FERT, key="fert_crop"
        )
    with form_col2:
        soil_type = st.selectbox(
            "🌱 Soil Type", options=SOIL_TYPES, key="fert_soil"
        )
    with form_col3:
        acres_fert = st.number_input(
            "📐 Farm Size (acres)",
            min_value=0.5, max_value=500.0, value=5.0, step=0.5,
            key="fert_acres",
        )

    # NPK inputs
    st.markdown("**🧪 Soil Nutrient Levels (N-P-K)**")
    npk_col1, npk_col2, npk_col3 = st.columns(3)
    with npk_col1:
        n_fert = st.number_input(
            "Nitrogen (N)", 0.0, 100.0, 20.0, 1.0,
            key="fert_n"
        )
    with npk_col2:
        p_fert = st.number_input(
            "Phosphorus (P)", 0.0, 100.0, 20.0, 1.0,
            key="fert_p"
        )
    with npk_col3:
        k_fert = st.number_input(
            "Potassium (K)", 0.0, 50.0, 5.0, 0.5,
            key="fert_k"
        )

    # Optional weather / environmental inputs
    with st.expander("⚙️ Weather Info (optional \u2014 we get it from your city)"):
        env1, env2, env3, env4 = st.columns(4)
        with env1:
            city_fert = st.text_input(
                "City (for auto T/H)",
                placeholder="e.g. Lahore",
                key="fert_city",
            )
        with env2:
            temp_fert = st.number_input(
                "Temperature °C (manual)", 15.0, 50.0, 30.0, 0.5,
                key="fert_temp",
            )
        with env3:
            hum_fert = st.number_input(
                "Humidity % (manual)", 20.0, 100.0, 55.0, 1.0,
                key="fert_hum",
            )
        with env4:
            moist_fert = st.number_input(
                "Soil Moisture %", 10.0, 80.0, 40.0, 1.0,
                key="fert_moist",
            )

    st.markdown("")  # spacer

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        fert_btn = st.button(
            "🧪 Recommend Fertilizer", use_container_width=True, type="primary"
        )

    # ── Run prediction on button click ─────────────────────────────
    if fert_btn:
        st.session_state.fert_result = None
        st.session_state.fert_error = None

        # If city is provided, let backend fetch weather; otherwise send manual T/H
        payload = {
            "crop": crop_fert,
            "soil_type": soil_type,
            "nitrogen": n_fert,
            "phosphorus": p_fert,
            "potassium": k_fert,
            "acres": acres_fert,
            "moisture": moist_fert,
            "language": language_code,
        }
        if city_fert.strip():
            payload["city"] = city_fert.strip()
        else:
            payload["temperature"] = temp_fert
            payload["humidity"] = hum_fert

        with st.spinner(f"🔬 Analysing {crop_fert} on {soil_type} soil..."):
            result = call_fertilizer_api(payload)
            if result:
                st.session_state.fert_result = result
                st.session_state.fert_error = None

    # ── Display errors ──────────────────────────────────────────────
    if st.session_state.fert_error:
        st.error(st.session_state.fert_error)

    # ── Display fertilizer dashboard ────────────────────────────────
    fdata = st.session_state.fert_result
    if fdata:
        st.divider()

        # Header
        st.subheader(f"🧪 Fertilizer for {fdata['crop']} — {fdata['soil_type']} soil, {fdata['acres']} acres")

        # ── Recommended Fertilizer Card ─────────────────────────────
        st.markdown(
            f'<div class="fert-card fert-rec">'
            f'<div class="value">{fdata["fertilizer_name"]}</div>'
            f'<div class="label">✅ Recommended Fertilizer</div></div>',
            unsafe_allow_html=True,
        )

        # ── Input conditions cards ──────────────────────────────────
        st.markdown("**📋 Input Conditions**")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="fert-card fert-info">'
                f'<div class="value">N:{fdata["nitrogen"]}</div>'
                f'<div class="label">Nitrogen</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="fert-card fert-info">'
                f'<div class="value">P:{fdata["phosphorus"]}</div>'
                f'<div class="label">Phosphorus</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="fert-card fert-info">'
                f'<div class="value">K:{fdata["potassium"]}</div>'
                f'<div class="label">Potassium</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="fert-card fert-warn">'
                f'<div class="value">{fdata["temperature"]}°C</div>'
                f'<div class="label">Temperature</div></div>',
                unsafe_allow_html=True,
            )

        # ── AI Explanation ──────────────────────────────────────────
        st.divider()
        st.subheader("🤖 AI Explanation")
        st.markdown(fdata["explanation"])

        # ── Environment context caption ─────────────────────────────
        st.caption(
            f"🌡️ Environment: {fdata['temperature']}°C, "
            f"Humidity: {fdata['humidity']}%, "
            f"Soil Moisture: {fdata['moisture']}%"
        )

    # ── Tab empty state ───────────────────────────────────────────
    if not st.session_state.fert_result:
        st.info(
            "🧪 **How to get fertilizer advice:**\n\n"
            "1. **Pick your crop** and **soil type** from the lists\n"
            "2. **Enter N-P-K numbers** (from your soil test \u2014 "
            "Nitrogen, Phosphorus, Potassium)\n"
            "3. **Enter your city** (optional) to get weather info\n"
            "4. Click **Recommend Fertilizer** to get your answer\n\n"
            "**Tip:** Get a soil test before planting to know your N-P-K numbers."
        )


# ─────────────────────────────────────────────────────────────────────
# PAGE 6 — Regional Guide (Location-Based Recommendations)
# ─────────────────────────────────────────────────────────────────────
elif page == "region":
    st.caption("Find out which farming zone your area is in and get tips for your region.")

    # ── Input form ──────────────────────────────────────────────────
    col_city, col_btn = st.columns([3, 1])
    with col_city:
        city_input = st.text_input(
            "📍 Enter your city or district",
            placeholder="e.g. Lahore, Peshawar, Quetta, Skardu, Tharparkar",
            key="region_city",
        )
    with col_btn:
        st.markdown("")  # spacer for alignment
        region_btn = st.button(
            "📍 Get Insights", use_container_width=True, type="primary",
            key="region_btn",
        )

    # ── Run analysis on button click ─────────────────────────────
    if region_btn:
        if not city_input.strip():
            st.warning("Please enter a city or district name.")
        else:
            st.session_state.region_result = None
            st.session_state.region_error = None
            payload = {
                "city": city_input.strip(),
                "language": language_code,
            }
            with st.spinner(f"🔍 Analysing region for {city_input}..."):
                result = call_location_api(payload)
                if result:
                    st.session_state.region_result = result
                    st.session_state.region_error = None

    # ── Display errors ──────────────────────────────────────────────
    if st.session_state.region_error:
        st.error(st.session_state.region_error)

    # ── Display regional guide dashboard ────────────────────────────
    rdata = st.session_state.region_result
    if rdata:
        st.divider()

        # ── Zone badge — centered ─────────────────────────────────
        zone_icon = rdata.get("zone_icon", "📍")
        zone_name = rdata["zone_name"]
        district_name = rdata["district"]
        zone_key = rdata["zone_key"]

        badge_html = (
            f'<div style="text-align:center; margin-bottom:0.5rem;">'
            f'<span class="zone-badge">{zone_icon} {zone_key}: {zone_name}</span>'
            f'</div>'
        )
        st.markdown(badge_html, unsafe_allow_html=True)

        # District name — prominent
        st.markdown(
            f'<div style="text-align:center; font-size:1.4rem; font-weight:700; '
            f'margin-bottom:1rem;">📍 {district_name}</div>',
            unsafe_allow_html=True,
        )

        # ── Quick stat cards ─────────────────────────────────────
        st.markdown("**📊 At a Glance**")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            crops_short = rdata.get("crops", [])
            crop_display = ", ".join(crops_short[:3]) if crops_short else "—"
            st.markdown(
                f'<div class="region-card region-zone">'
                f'<div class="value" style="font-size:1rem;">{crop_display}</div>'
                f'<div class="label">Typical Crops</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            soil_text = rdata.get("soil", "—")
            # Truncate for card display
            soil_short = (soil_text[:45] + "...") if len(soil_text) > 48 else soil_text
            st.markdown(
                f'<div class="region-card region-info">'
                f'<div class="value" style="font-size:1rem;">{soil_short}</div>'
                f'<div class="label">Soil Type</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            temp_val = rdata.get("temperature")
            temp_display = f"{temp_val:.0f}°C" if temp_val is not None else "—"
            st.markdown(
                f'<div class="region-card region-weather">'
                f'<div class="value">{temp_display}</div>'
                f'<div class="label">Temperature</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            hum_val = rdata.get("humidity")
            hum_display = f"{hum_val}%" if hum_val is not None else "—"
            st.markdown(
                f'<div class="region-card region-weather">'
                f'<div class="value">{hum_display}</div>'
                f'<div class="label">Humidity</div></div>',
                unsafe_allow_html=True,
            )

        # ── Zone Detail Cards ────────────────────────────────────
        st.markdown("")  # spacer

        # Climate
        climate = rdata.get("climate", "")
        if climate:
            st.markdown(
                f'<div class="zone-detail-card">'
                f'<div class="card-title">🌡️ Climate</div>'
                f'<div class="card-body">{climate}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Irrigation
        irrigation = rdata.get("irrigation", "")
        if irrigation:
            st.markdown(
                f'<div class="zone-detail-card">'
                f'<div class="card-title">💧 Irrigation</div>'
                f'<div class="card-body">{irrigation}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Challenges
        challenges = rdata.get("challenges", [])
        if challenges:
            st.markdown(
                f'<div class="zone-detail-card" style="border-left-color: #ff9800;">'
                f'<div class="card-title">⚠️ Key Challenges</div>',
                unsafe_allow_html=True,
            )
            for ch in challenges:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {ch}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── AI Regional Insights ─────────────────────────────────
        insights = rdata.get("insights", "")
        if insights:
            st.divider()
            st.subheader("🤖 AI Regional Insights")
            st.markdown(insights)

        # ── Weather context caption ─────────────────────────────
        weather_desc = rdata.get("weather_description")
        if weather_desc and temp_val is not None:
            st.caption(
                f"🌤️ Current weather in {district_name}: "
                f"{temp_val:.0f}°C, {hum_val or 0}% humidity, {weather_desc}"
            )

    # ── Tab empty state ───────────────────────────────────────────
    if not st.session_state.region_result:
        st.info(
            "📍 **How to use the Regional Guide:**\n\n"
            "1. **Enter your city or area name** "
            "(like Lahore, Peshawar, Quetta)\n"
            "2. Click **Get Insights** to see your farming zone\n"
            "3. Read the results \u2014 what grows best in your area, "
            "what problems to watch for, and farming tips\n\n"
            "**Covers 157 districts** across all provinces of Pakistan, "
            "in 10 official farming zones."
        )

elif page == "market":
    st.markdown("### 📈 Market Prices — When to Sell")
    st.markdown(
        "See how crop prices go up and down over the year. "
        "Find out the best time to sell your crop. "
        "Prices in **Rs. per maund (40 kg)**."
    )

    # Commodity options (key → display name)
    _market_crops = [
        ("paddy", "Paddy / Rice (Dhaan)"),
        ("mango", "Mango (Aam)"),
        ("gram", "Gram / Chana"),
        ("masoor", "Masoor Lentil (Daal Masoor)"),
        ("mash", "Mash (Daal Mash)"),
        ("moong", "Moong (Daal Moong)"),
        ("guava", "Guava (Amrood)"),
        ("carrot", "Carrot (Gajar)"),
        ("garlic", "Garlic (Lehsan)"),
        ("green chilli", "Green Chilli (Hari Mirch)"),
        ("groundnut", "Groundnut (Moongphali)"),
        ("lemon", "Lemon (Limu)"),
        ("grapes", "Grapes (Angoor)"),
        ("kinnow", "Kinnow"),
        ("orange", "Orange (Malta)"),
        ("banana", "Banana (Kela)"),
        ("mustard", "Mustard (Sarson)"),
        ("jaggery", "Jaggery / Gur"),
        ("ginger", "Ginger (Adrak)"),
        ("millet", "Millet (Bajra)"),
        ("apple", "Apple (Saib)"),
        ("canola", "Canola"),
    ]
    _crop_labels = {k: v for k, v in _market_crops}

    col_m1, col_m2, col_m3 = st.columns([2, 2, 1])
    with col_m1:
        selected_crop_label = st.selectbox(
            "Select Crop / Commodity",
            options=[v for _, v in _market_crops],
            index=0,
            key="market_crop_select",
        )
        # Reverse lookup: display name → key
        selected_crop_key = next(
            k for k, v in _market_crops if v == selected_crop_label
        )

    with col_m2:
        market_city = st.text_input(
            "Your City / District (optional)",
            key="market_city_input",
            placeholder="e.g. Multan, Faisalabad, Lahore",
        )

    with col_m3:
        st.write("")  # spacer
        analyze_btn = st.button(
            "📈  Analyse Trend", key="market_analyze_btn", type="primary"
        )

    if analyze_btn:
        st.session_state.market_error = None
        st.session_state.market_result = None
        with st.spinner(f"Analysing price trends for **{selected_crop_label}**…"):
            result = call_market_api({
                "commodity": selected_crop_key,
                "city": market_city.strip() if market_city else "",
                "language": language_code,
            })
            if result:
                st.session_state.market_result = result

    # ── Display results ────────────────────────────────────────────
    if st.session_state.market_error:
        st.error(st.session_state.market_error)
        st.session_state.market_error = None

    if st.session_state.market_result:
        mr = st.session_state.market_result

        # Trend badge
        trend = mr.get("trend", "stable")
        trend_icon = {"rising": "📈", "falling": "📉", "stable": "➡️"}.get(trend, "❓")
        trend_cls = {"rising": "trend-rising", "falling": "trend-falling", "stable": "trend-stable"}.get(trend, "trend-stable")
        st.markdown(
            f'<div style="text-align:center; margin: 0.5rem 0;">'
            f'<span class="trend-badge {trend_cls}">'
            f"{trend_icon} {trend.upper()}</span></div>",
            unsafe_allow_html=True,
        )

        # ── Zone / location badge ────────────────────────────────
        _zone_name = mr.get("zone_name")
        _zone_icon = mr.get("zone_icon", "")
        _matched_city = mr.get("matched_city")
        if _zone_name and _matched_city:
            st.markdown(
                f'<div style="text-align:center; margin: 0.3rem 0 0.8rem 0;">'
                f'<span style="display:inline-block; padding: 0.25rem 0.9rem; '
                f'border-radius: 20px; background: #f3e5f5; color: #7b1fa2; '
                f'font-size: 0.85rem; font-weight: 600; border: 1px solid #ce93d8;">'
                f"{_zone_icon} {_matched_city} — {_zone_name}</span></div>",
                unsafe_allow_html=True,
            )
        elif _matched_city:
            st.markdown(
                f'<div style="text-align:center; margin: 0.3rem 0 0.8rem 0;">'
                f'<span style="display:inline-block; padding: 0.25rem 0.9rem; '
                f'border-radius: 20px; background: #f3e5f5; color: #7b1fa2; '
                f'font-size: 0.85rem; font-weight: 600; border: 1px solid #ce93d8;">'
                f"📍 {_matched_city}</span></div>",
                unsafe_allow_html=True,
            )

        # ── Price line chart ─────────────────────────────────────
        chart_data = mr.get("chart_data", [])
        _price_unit = mr.get("price_unit", "Rs. / maund (40 kg)")
        _unit_short = (
            "Rs. / maund" if "maund" in _price_unit
            else "Rs. / 100 pcs" if "100" in _price_unit
            else "Rs. / dozen" if "dozen" in _price_unit
            else "Rs."
        )
        if chart_data:
            import pandas as pd

            chart_df = pd.DataFrame(chart_data)
            chart_df["price"] = chart_df["price"].astype(float)
            chart_df.index = chart_df["date"]
            chart_df = chart_df.rename(columns={"price": f"Avg Price ({_unit_short})"})

            st.markdown("#### 📊 Monthly Price Trend")
            st.line_chart(chart_df[[f"Avg Price ({_unit_short})"]], use_container_width=True)

        # ── Summary cards ────────────────────────────────────────
        latest_price = mr.get("latest_price", 0)
        pct_change = mr.get("pct_change_3m", 0)
        best_months = mr.get("best_months", [])
        date_range_end = mr.get("date_range_end", "")

        # Card color class based on trend
        price_card_cls = {
            "rising": "market-up", "falling": "market-down", "stable": "market-flat"
        }.get(trend, "market-flat")

        pct_icon = "🔺" if pct_change > 0 else "🔻" if pct_change < 0 else "➡️"
        pct_card_cls = "market-up" if pct_change > 0 else "market-down" if pct_change < 0 else "market-flat"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="market-card {price_card_cls}">'
                f'<div class="value">Rs. {latest_price:,.0f}</div>'
                f'<div class="label">Latest Avg ({_unit_short})</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="market-card {pct_card_cls}">'
                f'<div class="value">{pct_icon} {pct_change:+.1f}%</div>'
                f'<div class="label">3-Month Change</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="market-card market-up">'
                f'<div class="value">{", ".join(best_months[:3]) or "—"}</div>'
                f'<div class="label">Best Months to Sell</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="market-card market-info">'
                f'<div class="value">{date_range_end[:7] or "—"}</div>'
                f'<div class="label">Data Until</div></div>',
                unsafe_allow_html=True,
            )

        # ── Seasonal pattern table ───────────────────────────────
        seasonal = mr.get("seasonal_stats", [])
        if seasonal:
            st.markdown("#### 📅 Monthly Seasonal Pattern")
            season_df = pd.DataFrame(seasonal)
            season_df = season_df.rename(columns={
                "month": "Month",
                "avg_price": f"Avg ({_unit_short})",
                "min_price": f"Min ({_unit_short})",
                "max_price": f"Max ({_unit_short})",
            })
            st.dataframe(
                season_df[["Month", f"Avg ({_unit_short})", f"Min ({_unit_short})", f"Max ({_unit_short})"]],
                use_container_width=True,
                hide_index=True,
            )

        # ── AI Advice ────────────────────────────────────────────
        advice = mr.get("advice", "")
        if advice:
            st.markdown("#### 🤖 AI Selling Advice")
            st.markdown(advice)

        # ── Disclaimer ───────────────────────────────────────────
        disclaimer = mr.get("disclaimer", "")
        if disclaimer:
            st.caption(f"*{disclaimer}*")

elif page == "best_market":
    st.markdown("### 🏪 Best Market — Where to Sell")
    st.markdown(
        "Find out **which mandi pays the most** for your crop. "
        "Prices in **Rs. per maund (40 kg)** and "
        "**Rs. per bori (100 kg bag)**. "
        "We also check if going to a far mandi is worth the travel cost."
    )

    col_b1, col_b2, col_b3 = st.columns([2, 2, 1])
    with col_b1:
        bm_crop = st.text_input(
            "Crop Name",
            key="best_market_crop_input",
            placeholder="e.g. paddy, basmati, mango, garlic, gram",
        )
    with col_b2:
        bm_location = st.text_input(
            "Your City / District (optional)",
            key="best_market_location_input",
            placeholder="e.g. Okara, Multan, Lahore",
        )
    with col_b3:
        st.write("")  # spacer
        bm_btn = st.button(
            "🏪  Find Best Market", key="best_market_btn", type="primary"
        )

    if bm_btn:
        if not bm_crop.strip():
            st.session_state.best_market_error = "Please enter a crop name first."
        else:
            st.session_state.best_market_error = None
            st.session_state.best_market_result = None
            with st.spinner(f"Searching mandi prices for **{bm_crop.strip()}**…"):
                result = call_best_market_api({
                    "crop": bm_crop.strip(),
                    "location": bm_location.strip() if bm_location else "",
                    "language": language_code,
                })
                if result:
                    st.session_state.best_market_result = result

    # ── Display results ────────────────────────────────────────
    if st.session_state.best_market_error:
        st.error(st.session_state.best_market_error)
        st.session_state.best_market_error = None

    if st.session_state.best_market_result:
        bmr = st.session_state.best_market_result

        unit_label = bmr.get("unit_label", "Rs. / maund (40 kg)")
        bag_weight = bmr.get("bag_weight_kg")

        # ── Header: matched crop ──────────────────────
        st.markdown(f"#### 🌾 {bmr['crop']}")
        st.caption(
            f"Prices: {unit_label}"
            + (
                f" • per bori assumes {bag_weight} kg bag"
                if bag_weight else ""
            )
            + f" • Data until {bmr.get('last_date', '')}"
        )

        # ── Market ranking table ─────────────────────
        markets = bmr.get("markets", [])
        if markets:
            table_rows = []
            for i, m in enumerate(markets, 1):
                row = {
                    "Rank": i,
                    "Mandi": m["market"],
                    "Avg Price": f"Rs. {m['avg_price']:,.0f}",
                }
                if m.get("price_per_bori"):
                    row["Per Bori"] = f"Rs. {m['price_per_bori']:,.0f}"
                else:
                    row["Per Bori"] = "—"
                row["Range"] = (
                    f"Rs. {m['min_price']:,.0f} – {m['max_price']:,.0f}"
                )
                table_rows.append(row)
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # ── Summary cards ─────────────────────────
        cb1, cb2, cb3 = st.columns(3)
        with cb1:
            st.markdown(
                f'<div class="market-card market-up">'
                f'<div class="value">Rs. {bmr["national_avg"]:,.0f}</div>'
                f'<div class="label">National Average</div></div>',
                unsafe_allow_html=True,
            )
        with cb2:
            if bmr.get("home_market"):
                st.markdown(
                    f'<div class="market-card market-flat">'
                    f'<div class="value">Rs. {bmr["home_price"]:,.0f}</div>'
                    f'<div class="label">Your Market: {bmr["home_market"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="market-card market-info">'
                    f'<div class="value">📍</div>'
                    f'<div class="label">Add your city to compare</div></div>',
                    unsafe_allow_html=True,
                )
        with cb3:
            if bmr.get("best_gain_vs_home") is not None:
                gain_cls = (
                    "market-up" if bmr["best_gain_vs_home"] > 0 else "market-down"
                )
                st.markdown(
                    f'<div class="market-card {gain_cls}">'
                    f'<div class="value">Rs. {bmr["best_gain_vs_home"]:+,.0f}</div>'
                    f'<div class="label">Best Gain vs Your Market ({bmr["best_gain_pct"]:+.1f}%)</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                top_name = markets[0]["market"] if markets else "—"
                st.markdown(
                    f'<div class="market-card market-info">'
                    f'<div class="value">🏆</div>'
                    f'<div class="label">{top_name}</div></div>',
                    unsafe_allow_html=True,
                )

        # ── AI advice (with transport estimate) ─────
        advice = bmr.get("advice", "")
        if advice:
            st.markdown("#### 🤖 AI Selling Recommendation")
            st.markdown(advice)

        # ── Disclaimer ──────────────────────────
        disclaimer = bmr.get("disclaimer", "")
        if disclaimer:
            st.caption(f"*{disclaimer}*")

    else:
        st.info(
            "🏪 **How to find the best mandi:**\n\n"
            "1. **Type your crop** (like paddy, mango, garlic)\n"
            "2. **Enter your city** so we can compare your local mandi\n"
            "3. Click **Find Best Market** — see which mandi "
            "pays the most and if it's worth the travel\n\n"
            "Based on Pakistani mandi price data (2007–2022, 138 markets)."
        )

elif page == "irrigation":
    st.markdown("### 💧 Watering Guide")
    st.markdown(
        "Find out **when** to water your crop and **how much** water to give. "
        "We check today's weather to give you the best advice."
    )

    # ── Input fields ─────────────────────────────────────────────
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        irr_crop = st.text_input(
            "Crop Name",
            key="irr_crop_input",
            placeholder="e.g. Wheat, Cotton, Rice, Maize",
        )
        irr_location = st.text_input(
            "Location (City / District)",
            key="irr_location_input",
            placeholder="e.g. Multan, Faisalabad, Lahore",
        )

    with col_i2:
        irr_soil = st.selectbox(
            "Soil Type (optional)",
            options=["Not specified", "Loamy", "Clayey", "Sandy", "Silty", "Sandy Loam"],
            index=0,
            key="irr_soil_select",
        )
        irr_days = st.number_input(
            "Days since last irrigation (optional)",
            min_value=0, max_value=60, value=None,
            key="irr_days_input",
            help="Leave blank if you're not sure",
        )

    # Irrigate button
    irrigate_btn = st.button(
        "💧  Get Irrigation Advice", key="irr_analyze_btn", type="primary"
    )

    if irrigate_btn:
        # Validate required fields
        if not irr_crop.strip():
            st.warning("Please enter a crop name.")
        elif not irr_location.strip():
            st.warning("Please enter a location.")
        else:
            st.session_state.irrigation_error = None
            st.session_state.irrigation_result = None

            payload = {
                "crop_name": irr_crop.strip(),
                "location": irr_location.strip(),
                "soil_type": irr_soil if irr_soil != "Not specified" else None,
                "last_irrigation_days_ago": irr_days,
            }

            with st.spinner(
                f"Analysing irrigation needs for **{irr_crop}** in **{irr_location}**…"
            ):
                result = call_irrigation_api(payload)
                if result:
                    st.session_state.irrigation_result = result

    # ── Display results ──────────────────────────────────────────
    if st.session_state.irrigation_error:
        st.error(st.session_state.irrigation_error)
        st.session_state.irrigation_error = None

    if st.session_state.irrigation_result:
        ir = st.session_state.irrigation_result

        decision = ir.get("decision", "Irrigate now")
        reason = ir.get("reason", "")
        water_amount = ir.get("water_amount", "N/A")
        crop = ir.get("crop", "")
        location = ir.get("location", "")
        temp = ir.get("temperature", 0)
        humidity = ir.get("humidity", 0)
        rain_text = ir.get("rain_forecast_text", "")
        raw_advice = ir.get("raw_advice", "")

        # ── Decision badge ───────────────────────────────────────
        is_wait = "wait" in decision.lower()
        badge_cls = "decision-wait" if is_wait else "decision-irrigate"
        badge_icon = "⏳" if is_wait else "💧"

        st.markdown(
            f'<div style="text-align:center; margin: 0.5rem 0;">'
            f'<span class="decision-badge {badge_cls}">'
            f"{badge_icon} {decision}</span></div>",
            unsafe_allow_html=True,
        )

        # Reason text
        if reason:
            st.markdown(
                f'<div style="text-align:center; color: #555; '
                f'font-size: 0.95rem; margin-bottom: 0.8rem;">'
                f"{reason}</div>",
                unsafe_allow_html=True,
            )

        # ── Summary cards ────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)

        # Card 1: Water amount
        water_cls = "irr-go" if not is_wait else "irr-wait"
        water_icon = "💧" if not is_wait else "🚫"
        with c1:
            st.markdown(
                f'<div class="irrigation-card {water_cls}">'
                f'<div class="value">{water_icon} {water_amount.title()}</div>'
                f'<div class="label">Water Amount</div></div>',
                unsafe_allow_html=True,
            )

        # Card 2: Temperature
        with c2:
            st.markdown(
                f'<div class="irrigation-card irr-info">'
                f'<div class="value">🌡️ {temp:.1f}°C</div>'
                f'<div class="label">Current Temperature</div></div>',
                unsafe_allow_html=True,
            )

        # Card 3: Humidity
        with c3:
            st.markdown(
                f'<div class="irrigation-card irr-info">'
                f'<div class="value">💨 {humidity}%</div>'
                f'<div class="label">Humidity</div></div>',
                unsafe_allow_html=True,
            )

        # Card 4: Rain forecast
        # Show a short version of the rain forecast
        rain_short = rain_text[:50] + "…" if len(rain_text) > 50 else rain_text
        with c4:
            st.markdown(
                f'<div class="irrigation-card irr-weather">'
                f'<div class="value">🌧️</div>'
                f'<div class="label" style="font-size: 0.72rem;">{rain_short}</div></div>',
                unsafe_allow_html=True,
            )

        # ── Full AI advice ───────────────────────────────────────
        if raw_advice:
            st.markdown("#### 🤖 AI Irrigation Advice")
            st.markdown(raw_advice)

        # ── Details summary ──────────────────────────────────────
        with st.expander("📋 View Input Details"):
            st.write(f"**Crop:** {crop}")
            st.write(f"**Location:** {location}")
            st.write(f"**Temperature:** {temp:.1f}°C")
            st.write(f"**Humidity:** {humidity}%")
            st.write(f"**Rain Forecast:** {rain_text}")
            st.write(f"**Decision:** {decision}")
            st.write(f"**Water Guidance:** {water_amount}")

elif page == "calendar":
    st.markdown("### 🗓️ Your Farming Calendar")
    st.markdown(
        "Get a full schedule for your crop — when to plant, "
        "when to water, when to put fertilizer, and when to harvest. "
        "Made for your area."
    )

    # ── Input fields ─────────────────────────────────────────────
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        cal_crop = st.text_input(
            "Crop Name",
            key="cal_crop_input",
            placeholder="e.g. Wheat, Cotton, Rice, Maize",
        )
        cal_location = st.text_input(
            "Region / District / Province",
            key="cal_location_input",
            placeholder="e.g. Punjab, Multan, Sindh",
        )
    with col_c2:
        cal_sowing = st.text_input(
            "Planned Sowing Date (optional)",
            key="cal_sowing_input",
            placeholder="e.g. 15 Nov 2026",
            help="Leave blank and AI will suggest the ideal sowing window.",
        )

    # Generate button
    cal_btn = st.button(
        "🗓️  Generate Calendar", key="cal_generate_btn", type="primary"
    )

    if cal_btn:
        if not cal_crop.strip():
            st.warning("Please enter a crop name.")
        elif not cal_location.strip():
            st.warning("Please enter a location or region.")
        else:
            st.session_state.calendar_error = None
            st.session_state.calendar_result = None

            payload = {
                "crop_name": cal_crop.strip(),
                "location": cal_location.strip(),
                "sowing_date": cal_sowing.strip() if cal_sowing.strip() else None,
            }

            with st.spinner(
                f"Generating farming calendar for **{cal_crop}** "
                f"in **{cal_location}**…"
            ):
                result = call_calendar_api(payload)
                if result:
                    st.session_state.calendar_result = result

    # ── Display results ──────────────────────────────────────────
    if st.session_state.calendar_error:
        st.error(st.session_state.calendar_error)
        st.session_state.calendar_error = None

    if st.session_state.calendar_result:
        cr = st.session_state.calendar_result
        cal = cr.get("calendar", {})
        crop = cr.get("crop", "")
        location = cr.get("location", "")
        sowing_provided = cr.get("sowing_date_provided", False)

        # Header
        header_icon = "📅" if sowing_provided else "🌱"
        st.markdown(
            f"#### {header_icon} {crop} — {location}"
            + (" (custom sowing date)" if sowing_provided else " (AI-suggested sowing window)")
        )

        # ── 1. Sowing ───────────────────────────────────────────
        sowing = cal.get("sowing", "")
        if sowing and sowing != "See full advice below":
            st.markdown(
                f'<div class="cal-stage cal-sowing">'
                f"<h4>🌱 Sowing</h4><p>{sowing}</p></div>",
                unsafe_allow_html=True,
            )

        # ── 2. Watering Schedule ────────────────────────────────
        watering = cal.get("watering_schedule", [])
        if watering:
            items = "".join(f"<li>{w}</li>" for w in watering)
            st.markdown(
                f'<div class="cal-stage cal-water">'
                f"<h4>💧 Watering Schedule</h4>"
                f"<ul>{items}</ul></div>",
                unsafe_allow_html=True,
            )

        # ── 3. Fertilizer Schedule ──────────────────────────────
        fertilizer = cal.get("fertilizer_schedule", [])
        if fertilizer:
            items = "".join(f"<li>{f}</li>" for f in fertilizer)
            st.markdown(
                f'<div class="cal-stage cal-fert">'
                f"<h4>🧪 Fertilizer Schedule</h4>"
                f"<ul>{items}</ul></div>",
                unsafe_allow_html=True,
            )

        # ── 4. Spray Reminders ─────────────────────────────────
        sprays = cal.get("spray_reminders", [])
        if sprays:
            items = "".join(f"<li>{s}</li>" for s in sprays)
            st.markdown(
                f'<div class="cal-stage cal-spray">'
                f"<h4>🛡️ Pest / Disease Spray Reminders</h4>"
                f"<ul>{items}</ul></div>",
                unsafe_allow_html=True,
            )

        # ── 5. Harvesting ──────────────────────────────────────
        harvesting = cal.get("harvesting", "")
        if harvesting and harvesting != "See full advice below":
            st.markdown(
                f'<div class="cal-stage cal-harvest">'
                f"<h4>🌾 Harvesting</h4><p>{harvesting}</p></div>",
                unsafe_allow_html=True,
            )

        # ── Full AI advice (expandable) ────────────────────────
        raw = cr.get("raw_advice", "")
        if raw:
            with st.expander("📋 View Full AI Calendar"):
                st.markdown(raw)



elif page == "gov_support":
    st.markdown("### 🏛️ Government Help for Farmers")
    st.markdown(
        "Ask about government help for farmers — like money support, "
        "cheap loans, crop insurance, and free seeds or fertilizer."
    )

    # ── Input fields ─────────────────────────────────────────────
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        gov_query = st.text_area(
            "Your Question",
            key="gov_query_input",
            height=100,
            placeholder=(
                "e.g. What subsidies are available for wheat farmers?\n"
                "e.g. How can I apply for a Kisan Card?\n"
                "e.g. Kya ZTBL se qarz mil sakta hai?"
            ),
        )
    with col_g2:
        gov_province = st.selectbox(
            "Province (optional)",
            options=[
                "",
                "Punjab",
                "Sindh",
                "KPK",
                "Balochistan",
                "Gilgit-Baltistan",
                "AJK",
            ],
            key="gov_province_input",
            help="Select your province for region-specific information.",
        )

    # Ask button
    gov_btn = st.button(
        "🏛️  Ask About Government Support",
        key="gov_ask_btn",
        type="primary",
    )

    if gov_btn:
        if not gov_query.strip():
            st.warning("Please type your question about government support.")
        else:
            st.session_state.gov_support_error = None
            st.session_state.gov_support_result = None

            payload = {
                "query": gov_query.strip(),
                "province": gov_province if gov_province else None,
            }

            with st.spinner("Searching government support information…"):
                result = call_gov_support_api(payload)
                if result:
                    st.session_state.gov_support_result = result

    # ── Display results ──────────────────────────────────────────
    if st.session_state.gov_support_error:
        st.error(st.session_state.gov_support_error)
        st.session_state.gov_support_error = None

    if st.session_state.gov_support_result:
        gr = st.session_state.gov_support_result
        answer = gr.get("answer", "")
        province = gr.get("province", "")
        disclaimer = gr.get("disclaimer", "")
        original_query = gr.get("query", "")

        # Province badge
        if province and province != "General (all provinces)":
            st.markdown(
                f'<span class="gov-province-badge">📍 {province}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="gov-province-badge">🇵🇰 Pakistan-wide</span>',
                unsafe_allow_html=True,
            )

        # Question recap
        st.markdown(f"**Your question:** *{original_query}*")

        # Answer card
        if answer:
            st.markdown(
                f'<div class="gov-card gov-answer">'
                f"<h4>💡 Answer</h4><p>{answer}</p></div>",
                unsafe_allow_html=True,
            )

        # Disclaimer card
        if disclaimer:
            st.markdown(
                f'<div class="gov-card gov-disclaimer">'
                f"<h4>⚠️ Disclaimer</h4><p>{disclaimer}</p></div>",
                unsafe_allow_html=True,
            )

        # Helpful resources expander
        with st.expander("📞 Helpful Government Resources"):
            st.markdown(
                """
**Key Contacts & Resources:**
- **Zarai Taraqiati Bank (ZTBL):** Agricultural loans and financing
  - Helpline: 111-228-228
  - Website: ztbl.com.pk

- **Punjab Agriculture Department:**
  - Helpline: 1245
  - Website: agriculture.punjab.gov.pk

- **Sindh Agriculture Department:**
  - Website: agriculture.sindh.gov.pk

- **KPK Agriculture Department:**
  - Website: agriculture.kp.gov.pk

- **Federal Ministry of National Food Security:**
  - Website: mnfsr.gov.pk

- **Crop Insurance (private + government partnerships):**
  - Ask your nearest Agriculture Extension Office about current schemes

- **Kisan Card Programs:**
  - Check eligibility and application at your local tehsil office
                """
            )


# ─────────────────────────────────────────────────────────────────────
# PAGE 11 — Smart Crop Recommendation
# ─────────────────────────────────────────────────────────────────────
elif page == "crop_rec":
    st.markdown("### 🌾 Which Crop Should I Grow?")
    st.markdown(
        "Tell us about your soil and weather. We will suggest "
        "the best crop for your farm and explain why."
    )

    # ── Input form ──────────────────────────────────────────────────
    st.markdown("**🧪 Soil Nutrient Levels (N-P-K)**")
    npk_col1, npk_col2, npk_col3 = st.columns(3)
    with npk_col1:
        n_crop = st.number_input(
            "Nitrogen (N)", 0.0, 200.0, 50.0, 1.0,
            key="crop_rec_n",
        )
    with npk_col2:
        p_crop = st.number_input(
            "Phosphorus (P)", 0.0, 200.0, 50.0, 1.0,
            key="crop_rec_p",
        )
    with npk_col3:
        k_crop = st.number_input(
            "Potassium (K)", 0.0, 250.0, 50.0, 1.0,
            key="crop_rec_k",
        )

    st.markdown("**🌡️ Climate & Soil Conditions**")
    clim_col1, clim_col2, clim_col3, clim_col4 = st.columns(4)
    with clim_col1:
        temp_crop = st.number_input(
            "Temperature °C", 0.0, 55.0, 25.0, 0.5,
            key="crop_rec_temp",
        )
    with clim_col2:
        hum_crop = st.number_input(
            "Humidity %", 0.0, 100.0, 60.0, 1.0,
            key="crop_rec_hum",
        )
    with clim_col3:
        ph_crop = st.number_input(
            "Soil pH", 0.0, 14.0, 6.5, 0.1,
            key="crop_rec_ph",
        )
    with clim_col4:
        rain_crop = st.number_input(
            "Rainfall (mm)", 0.0, 500.0, 100.0, 5.0,
            key="crop_rec_rain",
        )

    st.markdown("")  # spacer

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        crop_btn = st.button(
            "🌾 Recommend Crop", use_container_width=True, type="primary"
        )

    # ── Run prediction on button click ─────────────────────────────
    if crop_btn:
        st.session_state.crop_rec_result = None
        st.session_state.crop_rec_error = None

        payload = {
            "nitrogen": n_crop,
            "phosphorus": p_crop,
            "potassium": k_crop,
            "temperature": temp_crop,
            "humidity": hum_crop,
            "ph": ph_crop,
            "rainfall": rain_crop,
        }

        with st.spinner("🔬 Analysing soil and climate conditions..."):
            result = call_crop_recommend_api(payload)
            if result:
                st.session_state.crop_rec_result = result
                st.session_state.crop_rec_error = None

    # ── Display errors ──────────────────────────────────────────────
    if st.session_state.crop_rec_error:
        st.error(st.session_state.crop_rec_error)

    # ── Display crop recommendation dashboard ───────────────────────
    cdata = st.session_state.crop_rec_result
    if cdata:
        st.divider()

        # Header
        st.subheader(f"🌾 Recommended Crop: {cdata['recommended_crop'].title()}")

        # ── Top Recommended Crop Card ─────────────────────────────
        conf_pct = f"{cdata['confidence']:.0%}"
        st.markdown(
            f'<div class="crop-card crop-rec">'
            f'<div class="value">{cdata["recommended_crop"].title()}</div>'
            f'<div class="label">✅ Top Recommendation — {conf_pct} confidence</div></div>',
            unsafe_allow_html=True,
        )

        # ── Top 3 Alternatives ─────────────────────────────────────
        st.markdown("**📊 Top 3 Crop Alternatives**")
        alt_cols = st.columns(3)
        for i, alt in enumerate(cdata.get("top_3_alternatives", [])):
            with alt_cols[i]:
                alt_conf = f"{alt['confidence']:.0%}"
                css_class = "crop-rec" if i == 0 else "crop-alt"
                st.markdown(
                    f'<div class="crop-card {css_class}">'
                    f'<div class="value">{alt["crop"].title()}</div>'
                    f'<div class="label">{alt_conf}</div></div>',
                    unsafe_allow_html=True,
                )

        # ── Input conditions cards ──────────────────────────────────
        st.markdown("**📋 Input Conditions**")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="crop-card crop-info">'
                f'<div class="value">N:{n_crop}</div>'
                f'<div class="label">Nitrogen</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="crop-card crop-info">'
                f'<div class="value">P:{p_crop}</div>'
                f'<div class="label">Phosphorus</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="crop-card crop-info">'
                f'<div class="value">K:{k_crop}</div>'
                f'<div class="label">Potassium</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="crop-card crop-info">'
                f'<div class="value">pH:{ph_crop}</div>'
                f'<div class="label">Soil pH</div></div>',
                unsafe_allow_html=True,
            )

        m5, m6, m7 = st.columns(3)
        with m5:
            st.markdown(
                f'<div class="crop-card crop-info">'
                f'<div class="value">{temp_crop}°C</div>'
                f'<div class="label">Temperature</div></div>',
                unsafe_allow_html=True,
            )
        with m6:
            st.markdown(
                f'<div class="crop-card crop-info">'
                f'<div class="value">{hum_crop}%</div>'
                f'<div class="label">Humidity</div></div>',
                unsafe_allow_html=True,
            )
        with m7:
            st.markdown(
                f'<div class="crop-card crop-info">'
                f'<div class="value">{rain_crop}mm</div>'
                f'<div class="label">Rainfall</div></div>',
                unsafe_allow_html=True,
            )

        # ── AI Explanation ──────────────────────────────────────────
        st.divider()
        st.subheader("🤖 AI Explanation")
        st.markdown(cdata["explanation"])

    # ── Tab empty state ───────────────────────────────────────────
    if not st.session_state.crop_rec_result:
        st.info(
            "🌾 **How to get crop suggestions:**\n\n"
            "1. **Enter N-P-K numbers** \u2014 from your soil test "
            "(Nitrogen, Phosphorus, Potassium)\n"
            "2. **Enter your weather info** \u2014 temperature, how much "
            "rain, soil pH, and humidity\n"
            "3. Click **Recommend Crop** to see which crop is best "
            "for your farm\n\n"
            "**Tip:** Get a soil test before planting to know your numbers."
        )


# ─────────────────────────────────────────────────────────────────────
# PAGE 12 — Grafting Advisor (Paiwand Guide)
# ─────────────────────────────────────────────────────────────────────
elif page == "grafting":
    st.markdown("### 🌱 Grafting Guide (Paiwand)")
    st.markdown(
        "Find out which plants can be joined together to grow better. "
        "Learn the best way and best time to do it."
    )

    # ── Input form ──────────────────────────────────────────────────
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        base_crop_graft = st.text_input(
            "🌳 Your Crop / Tree",
            placeholder="e.g. Mango, Citrus, Apple, Guava",
            key="graft_base_crop",
        )
    with form_col2:
        graft_goals = [
            "General improvement",
            "Better yield",
            "Disease resistance",
            "Faster fruit-bearing",
            "Better taste / quality",
            "Drought tolerance",
            "Less water usage",
        ]
        goal_graft = st.selectbox(
            "🎯 Goal (optional)",
            options=graft_goals,
            key="graft_goal",
        )

    # Language selector for grafting advice
    graft_lang_options = {
        "English": "en",
        "اردو (Urdu)": "ur",
        "پښتو (Pashto)": "ps",
    }
    graft_lang_display = st.selectbox(
        "🗣️ Advice Language",
        options=list(graft_lang_options.keys()),
        index=0,
        key="graft_lang",
    )
    graft_lang_code = graft_lang_options[graft_lang_display]

    st.markdown("")  # spacer

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        graft_btn = st.button(
            "🌱 Get Grafting Advice", use_container_width=True, type="primary"
        )

    # ── Run on button click ────────────────────────────────────────
    if graft_btn:
        st.session_state.grafting_result = None
        st.session_state.grafting_error = None

        if not base_crop_graft.strip():
            st.session_state.grafting_error = (
                "Please enter a crop or tree name."
            )
        else:
            payload = {
                "base_crop": base_crop_graft.strip(),
                "language": graft_lang_code,
            }
            # Only send goal if it's not the default placeholder
            if goal_graft != "General improvement":
                payload["goal"] = goal_graft

            with st.spinner(
                f"🔬 Researching grafting options for "
                f"{base_crop_graft.strip()}..."
            ):
                result = call_grafting_api(payload)
                if result:
                    st.session_state.grafting_result = result
                    st.session_state.grafting_error = None

    # ── Display errors ──────────────────────────────────────────────
    if st.session_state.grafting_error:
        st.error(st.session_state.grafting_error)

    # ── Display grafting advice dashboard ─────────────────────────
    gdata = st.session_state.grafting_result
    if gdata:
        st.divider()

        # Header
        st.subheader(
            f"🌱 Grafting Advice for {gdata['base_crop']}"
        )

        # Meta info card
        lang_display_map = {"en": "English", "ur": "اردو", "ps": "پښتو"}
        lang_label = lang_display_map.get(gdata["language"], gdata["language"])
        st.markdown(
            f'<div class="graft-card graft-meta">'
            f"<h4>📋 Details</h4>"
            f"<p><b>Crop:</b> {gdata['base_crop']} &nbsp;|&nbsp; "
            f"<b>Goal:</b> {gdata['goal']} &nbsp;|&nbsp; "
            f"<b>Language:</b> {lang_label}</p></div>",
            unsafe_allow_html=True,
        )

        # Advice card
        st.markdown(
            f'<div class="graft-card graft-advice">'
            f"<h4>💡 Grafting Advice</h4>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(gdata["advice"])

    # ── Tab empty state ───────────────────────────────────────────
    if not st.session_state.grafting_result:
        st.info(
            "🌱 **How to use the Grafting Guide:**\n\n"
            "1. **Enter your crop or tree name** "
            "(like Mango, Apple, Guava)\n"
            "2. **Tell us what you want** (optional) \u2014 like better fruit "
            "or stronger plants\n"
            "3. Click **Get Grafting Advice** for easy steps\n\n"
            "**Tip:** Grafting (paiwand) works best with plants from the same family. "
            "For example, different mango types can be joined together, "
            "but mango and citrus cannot."
        )


# ─────────────────────────────────────────────────────────────────────
# PAGE 14 — Harvest Loss Predictor (yield loss at harvest delay intervals)
# ─────────────────────────────────────────────────────────────────────
elif page == "harvest_loss":
    st.markdown("### ⏳ Harvest Loss — When to Cut?")
    st.markdown(
        "Find out **how much crop and money** you will lose if you "
        "wait too long to harvest. We check the weather and market "
        "prices to help you decide."
    )

    # ── Input fields ─────────────────────────────────────────────
    col_hl1, col_hl2 = st.columns(2)
    with col_hl1:
        hl_crop = st.text_input(
            "Crop Name",
            key="hl_crop_input",
            placeholder="e.g. Wheat, Rice, Cotton, Maize",
        )
        hl_location = st.text_input(
            "Location (City / District)",
            key="hl_location_input",
            placeholder="e.g. Multan, Peshawar, Faisalabad",
        )

    with col_hl2:
        hl_stage = st.selectbox(
            "Growth Stage",
            options=[
                "early grain filling",
                "nearing maturity",
                "fully mature",
                "overripe",
            ],
            index=1,
            key="hl_stage_select",
            help="Current growth stage of your crop.",
        )
        hl_planned = st.text_input(
            "Planned Harvest Date (optional)",
            key="hl_planned_input",
            placeholder="e.g. 15 April 2027",
            help="Leave blank to use today as Day 0.",
        )

    # Predict button
    hl_btn = st.button(
        "⏳  Predict Harvest Loss", key="hl_predict_btn", type="primary"
    )

    if hl_btn:
        # Validate required fields
        if not hl_crop.strip():
            st.warning("Please enter a crop name.")
        elif not hl_location.strip():
            st.warning("Please enter a location.")
        else:
            st.session_state.harvest_loss_error = None
            st.session_state.harvest_loss_result = None

            # Map the frontend language to the short code the route expects
            hl_lang_map = {"English": "en", "اردو": "ur", "پښتو": "ps"}
            hl_lang = hl_lang_map.get(language_display, "en")

            payload = {
                "crop_name": hl_crop.strip(),
                "location": hl_location.strip(),
                "growth_stage": hl_stage,
                "planned_harvest_date": hl_planned.strip() if hl_planned.strip() else None,
                "language": hl_lang,
            }

            with st.spinner(
                f"Analysing harvest timing for **{hl_crop}** in "
                f"**{hl_location}**\u2026"
            ):
                result = call_harvest_loss_api(payload)
                if result:
                    st.session_state.harvest_loss_result = result

    # ── Display results ──────────────────────────────────────────
    if st.session_state.harvest_loss_error:
        st.error(st.session_state.harvest_loss_error)
        st.session_state.harvest_loss_error = None

    if st.session_state.harvest_loss_result:
        hl = st.session_state.harvest_loss_result

        crop_name = hl.get("crop", "")
        loc = hl.get("location", "")
        badge = hl.get("decision_badge", "HARVEST NOW")
        advice = hl.get("advice_text", "")
        scenarios = hl.get("scenarios", [])
        weather_used = hl.get("weather_used", False)
        market_used = hl.get("market_used", False)

        # ── Decision badge (large, colour-coded) ────────────────
        # Map the three labels to colours
        badge_styles = {
            "HARVEST NOW": (
                "#2e7d32", "#ffffff", "🟢"  # green bg, white text
            ),
            "WAIT 2 DAYS": (
                "#f9a825", "#212121", "🟡"  # amber bg, dark text
            ),
            "HARVEST BEFORE RAIN": (
                "#c62828", "#ffffff", "🔴"  # red bg, white text
            ),
        }
        bg, fg, icon = badge_styles.get(
            badge, ("#2e7d32", "#ffffff", "🟢")
        )

        st.markdown(
            f'<div style="text-align:center; margin: 1rem 0;">'
            f'<div style="display:inline-block; background:{bg}; '
            f'color:{fg}; padding: 16px 36px; border-radius: 12px; '
            f'font-size: 1.6rem; font-weight: 700; '
            f'box-shadow: 0 4px 12px rgba(0,0,0,0.15);">'
            f'{icon} {badge}</div></div>',
            unsafe_allow_html=True,
        )

        # Subtitle
        st.markdown(
            f'<div style="text-align:center; color:#666; '
            f'font-size:0.9rem; margin-bottom:1rem;">'
            f'Crop: <strong>{crop_name}</strong> • '
            f'Location: <strong>{loc}</strong></div>',
            unsafe_allow_html=True,
        )

        # ── Scenario cards (Day 0 / Day 5 / Day 10) ────────────
        if scenarios:
            st.markdown("#### 📊 Harvest Delay Scenarios")

            cols = st.columns(len(scenarios))
            for idx, sc in enumerate(scenarios):
                day = sc.get("delay_days", 0)
                yld = sc.get("expected_yield_pct", 0)
                loss = sc.get("loss_pct", 0)

                # Colour: green = low loss, yellow = moderate, red = high
                if loss <= 5:
                    card_bg = "#e8f5e9"  # light green
                    card_border = "#4caf50"
                elif loss <= 15:
                    card_bg = "#fff8e1"  # light amber
                    card_border = "#ffc107"
                else:
                    card_bg = "#ffebee"  # light red
                    card_border = "#f44336"

                day_label = f"Day {day}" if day > 0 else "Today"
                with cols[idx]:
                    st.markdown(
                        f'<div style="background:{card_bg}; '
                        f'border-left: 5px solid {card_border}; '
                        f'padding: 16px; border-radius: 8px; '
                        f'text-align:center; margin-bottom:0.5rem;">'
                        f'<div style="font-size:1.1rem; '
                        f'font-weight:700; margin-bottom:8px;">'
                        f'📅 {day_label}</div>'
                        f'<div style="font-size:2rem; '
                        f'font-weight:800; color:{card_border};">'
                        f'{yld:.0f}%</div>'
                        f'<div style="font-size:0.8rem; '
                        f'color:#555;">Expected Yield</div>'
                        f'<div style="margin-top:8px; '
                        f'font-size:0.85rem; '
                        f'color:#333;">'
                        f'Loss: <strong>{loss:.0f}%</strong></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Data source indicators ─────────────────────────────
        src_parts = []
        if weather_used:
            src_parts.append("✅ Live weather")
        else:
            src_parts.append("⚠️ Weather estimate (no live data)")
        if market_used:
            src_parts.append("✅ Market prices")
        else:
            src_parts.append("⚠️ Market estimate (no live data)")
        st.caption(" | ".join(src_parts))

        # ── Full AI advice text ──────────────────────────────────
        if advice:
            st.markdown("#### 🤖 AI Harvest Advice")
            st.markdown(advice)

        # ── Voice audio (respects auto-play toggle) ─────────────
        if advice:
            hl_lang_map_rev = {"en": "en-US", "ur": "ur-PK", "ps": "ps-AF"}
            hl_tts_lang = hl_lang_map_rev.get(
                hl.get("language", "en"), "en-US"
            )

            if st.session_state.auto_play:
                with st.spinner("🔊 Generating voice..."):
                    audio = call_text_to_speech(advice, hl_tts_lang)
                if audio:
                    st.audio(audio, format="audio/mpeg", autoplay=True)
            else:
                # Manual play button when auto-play is off
                if st.button("🔊 Listen to Advice", key="hl_tts_btn"):
                    with st.spinner("🔊 Generating voice..."):
                        audio = call_text_to_speech(advice, hl_tts_lang)
                    if audio:
                        st.audio(audio, format="audio/mpeg", autoplay=True)

        # ── Input details (collapsible) ─────────────────────────
        with st.expander("📋 View Input Details"):
            st.write(f"**Crop:** {crop_name}")
            st.write(f"**Location:** {loc}")
            st.write(f"**Decision:** {badge}")
            st.write(f"**Weather data used:** {'Yes' if weather_used else 'No'}")
            st.write(f"**Market data used:** {'Yes' if market_used else 'No'}")

    else:
        st.info(
            "⏳ **How to check harvest loss:**\n\n"
            "1. **Type your crop** (like Wheat, Rice, Cotton)\n"
            "2. **Enter your city** so we can check the weather\n"
            "3. **Pick the growth stage** — is your crop almost "
            "ripe, fully ripe, or too ripe?\n"
            "4. Click **Predict Harvest Loss** to see what happens "
            "if you wait to harvest\n\n"
            "We use real weather and market prices to help you decide."
        )


# ─────────────────────────────────────────────────────────────────────
# PAGE 15 — Offline Mode (general crop knowledge — no internet needed)
# ─────────────────────────────────────────────────────────────────────
elif page == "offline":
    st.markdown("### 📴 Offline Mode — Farming Tips Without Internet")
    st.markdown(
        "Read basic farming tips for common Pakistani crops. "
        "This information is saved in the app. You do **not** need "
        "internet to read it."
    )

    # ── Load the bundled knowledge base ────────────────────────────
    kb = load_offline_knowledge_base()

    if not kb:
        st.error(
            "Offline knowledge base file not found. "
            "Please ensure `offline_knowledge_base.json` is in the "
            "`frontend/` folder."
        )
    else:
        # ── Connection status indicator ────────────────────────────
        if st.session_state.offline_mode:
            st.warning(
                "📴 **You are offline.** Showing saved crop "
                "tips only \u2014 no live weather, market prices, "
                "or AI answers."
            )
        else:
            st.success(
                "🟢 **You are online.** Offline Mode is still available "
                "for quick reference. All live features are also working."
            )

        # ── Crop selector ──────────────────────────────────────────
        # Build a nice display label: "Gandum (گندم) — Wheat"
        _crop_keys = list(kb.keys())
        _crop_display = {
            k: f"{kb[k]['local_name']} — {k.capitalize()}"
            for k in _crop_keys
        }

        selected_crop = st.selectbox(
            "Select a crop to learn about",
            options=_crop_keys,
            format_func=lambda k: _crop_display[k],
            key="offline_crop_select",
        )

        if selected_crop:
            info = kb[selected_crop]

            st.divider()

            # ── Crop header ────────────────────────────────────────
            st.subheader(
                f"🌿 {info['local_name']} — {selected_crop.capitalize()}"
            )

            # ── Season & water info cards ──────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    '<div class="weather-metric">'
                    f'<div class="value">🌱</div>'
                    f'<div class="label">Sowing Season</div>'
                    f'<div style="font-size:0.82rem; margin-top:4px;">'
                    f'{info["sowing_season"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    '<div class="weather-metric">'
                    f'<div class="value">🌾</div>'
                    f'<div class="label">Harvesting Season</div>'
                    f'<div style="font-size:0.82rem; margin-top:4px;">'
                    f'{info["harvesting_season"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    '<div class="weather-metric">'
                    f'<div class="value">💧</div>'
                    f'<div class="label">Water Needs</div>'
                    f'<div style="font-size:0.82rem; margin-top:4px;">'
                    f'{info["water_needs"]}</div></div>',
                    unsafe_allow_html=True,
                )

            # ── Common diseases / pests ────────────────────────────
            st.markdown("#### 🐛 Common Diseases & Pests")
            st.caption(
                "General prevention tips — for accurate diagnosis, use the "
                "🔍 Disease Detection page with a photo (requires internet)."
            )
            for issue in info.get("common_issues", []):
                st.markdown(f"  • {issue}")

            # ── Fertilizer guidance ────────────────────────────────
            st.markdown("#### 🧪 General Fertilizer Guidance")
            st.caption(
                "This is general advice — for soil-specific recommendations, "
                "use the 🧪 Fertilizer page (requires internet)."
            )
            st.info(info.get("fertilizer_tip", "No fertilizer data available."))

            # ── General tips ───────────────────────────────────────
            st.markdown("#### 💡 Farming Tips")
            for tip in info.get("general_tips", []):
                st.markdown(f"  • {tip}")

            # ── Disclaimer ─────────────────────────────────────────
            st.divider()
            st.caption(
                "⚠️ **Disclaimer:** This is general reference information "
                "for Pakistani crops. Actual practices may vary based on your "
                "specific soil, climate zone, and local conditions. "
                "For smart farming advice, connect to the internet "
                "and use the other CropGenie features."
            )

