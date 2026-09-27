"""
app.py  —  Iris Data Science Studio
Complete Data Science Lifecycle Platform:
1. Business Understanding & Data Collection
2. Data Cleaning & Preprocessing Studio (Automated Recommendations & 1-Click Execution)
3. Automated Exploratory Data Analysis (EDA) Studio & Dashboard Exporter
4. Automated Feature Engineering Studio (1-Click Execution & Plain-English Explanations)
5. Communication & Executive Reporting
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go

from engine.parser import load_csv
from engine.type_inference import infer_all_types, get_columns_by_type
from engine.quality import full_quality_report
from engine.eda import all_column_stats, correlation_matrix
from engine.viz import (
    histogram_kde, box_plot, bar_chart, missing_heatmap,
    correlation_heatmap, scatter_with_regression, grouped_box,
    outlier_strip
)
from engine.exporter import build_pdf_report, build_cleaned_csv
from engine.html_exporter import build_html_dashboard
from engine.business import DOMAINS, OBJECTIVES, get_domain_templates, get_objective_configs, suggest_target_column
from engine.cleaning import (
    impute_missing, drop_sparse_columns, remove_duplicates,
    treat_outliers_iqr, treat_outliers_zscore, clean_structural_errors,
    convert_column_type, encode_categorical, scale_feature,
    drop_constant_features, drop_high_correlation_features,
    get_encoding_suggestions, auto_encode_all_categorical,
    get_scaling_suggestions, auto_scale_all_numeric,
    detect_cleaning_recommendations, compute_data_health_score, get_before_after_stats,
    detect_structural_errors
)

from engine.eda import all_column_stats, correlation_matrix, normality_test, detect_class_imbalance, generate_eda_key_findings
from engine.eda_automation import generate_eda_chart_recommendations, build_eda_standalone_dashboard
from engine.feature_engineering import (

    get_feature_engineering_suggestions, apply_transformation,
    auto_engineer_all_features, generate_fe_plan, create_interaction_features, quick_feature_importance
)
from engine.gemini_client import (
    explain_file_overview, explain_data_quality, explain_column,
    explain_correlations, chat_with_iris, generate_executive_summary, generate_stage_insights
)
from engine.ml_recommender import recommend_ml_models
from ui.styles import inject_custom_css
from ui.components import (
    render_kpi_card, render_iris_says, render_stage_header,
    render_health_scorecard, render_recommendation_card,
    render_data_preview_table
)



# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Data Science Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Enterprise Design System ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap');

/* ── DESIGN TOKENS ── */
:root {
    --bg-base:      #08080f;
    --bg-surface:   #0e0e1a;
    --bg-elevated:  #141426;
    --bg-hover:     #1a1a30;
    --border-dim:   rgba(255,255,255,0.06);
    --border-soft:  rgba(124,58,237,0.18);
    --border-glow:  rgba(124,58,237,0.45);
    --violet-600:   #7C3AED;
    --violet-500:   #8B5CF6;
    --violet-400:   #A78BFA;
    --violet-glow:  rgba(124,58,237,0.14);
    --cyan-500:     #06B6D4;
    --cyan-400:     #22D3EE;
    --cyan-glow:    rgba(6,182,212,0.10);
    --green-500:    #10B981;
    --amber-500:    #F59E0B;
    --red-500:      #EF4444;
    --pink-500:     #EC4899;
    --text-primary: #F1F5F9;
    --text-secondary:#94A3B8;
    --text-muted:   #475569;
    --radius-sm:    10px;
    --radius-md:    14px;
    --radius-lg:    20px;
    --radius-xl:    24px;
    --shadow-sm:    0 2px 8px rgba(0,0,0,0.3);
    --shadow-md:    0 8px 24px rgba(0,0,0,0.4);
    --shadow-lg:    0 16px 48px rgba(0,0,0,0.5);
    --shadow-glow:  0 0 40px rgba(124,58,237,0.12);
}

/* ── BASE ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg-base) !important;
    color: var(--text-primary);
    font-size: 15px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

p, div, span, li { font-family: 'Inter', sans-serif; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-dim) !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
}

/* Sidebar brand logo area */
.iris-brand {
    padding: 28px 20px 20px;
    border-bottom: 1px solid var(--border-dim);
    margin-bottom: 8px;
    text-align: center;
}
.iris-brand-icon {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 6px;
    filter: drop-shadow(0 0 20px rgba(124,58,237,0.7));
}
.iris-brand-name {
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #A78BFA 0%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.iris-brand-tagline {
    font-size: 0.7rem;
    color: var(--text-muted);
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Sidebar step stepper */
.sidebar-section-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 12px 16px 6px;
}
.sidebar-step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 14px;
    margin: 2px 8px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.18s ease;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-muted);
    border: 1px solid transparent;
    text-decoration: none;
}
.sidebar-step-done {
    color: var(--text-secondary);
}
.sidebar-step-done .step-dot {
    background: var(--green-500);
    box-shadow: 0 0 10px rgba(16,185,129,0.4);
}
.sidebar-step-active {
    background: rgba(124,58,237,0.12);
    border-color: rgba(124,58,237,0.3);
    color: var(--violet-400);
    font-weight: 600;
}
.sidebar-step-active .step-dot {
    background: var(--violet-500);
    box-shadow: 0 0 12px rgba(124,58,237,0.6);
    animation: pulse-dot 2s ease-in-out infinite;
}
.sidebar-step-locked {
    opacity: 0.4;
}
.sidebar-step-locked .step-dot {
    background: var(--text-muted);
}
.step-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.step-number {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.68rem;
    font-weight: 700;
    flex-shrink: 0;
    border: 1.5px solid currentColor;
    opacity: 0.7;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.3); }
}

/* ── KPI METRIC CARDS ── */
div[data-testid="stMetric"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-soft) !important;
    border-top: 2px solid var(--violet-600) !important;
    border-radius: var(--radius-md) !important;
    padding: 18px 20px !important;
    box-shadow: var(--shadow-md), var(--shadow-glow) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg), 0 0 50px rgba(124,58,237,0.18) !important;
}
div[data-testid="stMetricLabel"] > div {
    color: var(--text-muted) !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stMetricValue"] > div {
    color: var(--text-primary) !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.1 !important;
}

/* ── GLASS CARDS ── */
.glass-card {
    background: rgba(20, 20, 38, 0.85);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md), inset 0 1px 0 rgba(255,255,255,0.04);
    padding: 24px 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    border-color: rgba(124,58,237,0.35);
    box-shadow: var(--shadow-lg), 0 0 40px rgba(124,58,237,0.1);
}

/* Accent-top cards (for section headers) */
.card-violet {
    border-top: 2px solid var(--violet-500);
}
.card-cyan {
    border-top: 2px solid var(--cyan-500);
}
.card-green {
    border-top: 2px solid var(--green-500);
}
.card-amber {
    border-top: 2px solid var(--amber-500);
}
.card-red {
    border-top: 2px solid var(--red-500);
}
.card-pink {
    border-top: 2px solid var(--pink-500);
}

/* ── RECOMMENDATION CARDS (Priority system) ── */
.rec-card {
    display: flex;
    gap: 16px;
    align-items: flex-start;
    background: var(--bg-elevated);
    border: 1px solid var(--border-dim);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.rec-card:hover {
    border-color: var(--border-soft);
    transform: translateX(3px);
}
.rec-card-critical { border-left: 3px solid var(--red-500); }
.rec-card-high     { border-left: 3px solid var(--amber-500); }
.rec-card-medium   { border-left: 3px solid var(--cyan-500); }
.rec-card-low      { border-left: 3px solid var(--text-muted); }
.rec-priority-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.rec-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.rec-reason {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── STAGE HEADER SYSTEM ── */
.stage-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,58,237,0.12);
    color: var(--violet-400);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 14px;
}
.stage-header {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 8px;
}
.stage-sub {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.7;
    max-width: 700px;
    margin-bottom: 28px;
}
.section-divider {
    border: none;
    border-top: 1px solid var(--border-dim);
    margin: 24px 0;
}

/* ── IRIS AI INSIGHT PANEL ── */
.iris-insight {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    background: linear-gradient(135deg, rgba(124,58,237,0.07) 0%, rgba(6,182,212,0.04) 100%);
    border: 1px solid rgba(124,58,237,0.22);
    border-left: 3px solid var(--violet-500);
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 18px 22px;
    margin: 20px 0;
    position: relative;
    overflow: hidden;
}
.iris-insight::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--violet-500), transparent);
}
.iris-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--violet-600), var(--cyan-500));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    box-shadow: 0 0 16px rgba(124,58,237,0.4);
}
.iris-insight-body {
    flex: 1;
}
.iris-insight-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--violet-400);
    margin-bottom: 6px;
}
.iris-insight-text {
    font-size: 0.88rem;
    color: #CBD5E1;
    line-height: 1.75;
}

/* Why card (EDA explanations) */
.why-card {
    background: rgba(14,14,26,0.7);
    border: 1px solid var(--border-dim);
    border-left: 3px solid var(--cyan-500);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 14px 18px;
    margin-top: 14px;
    color: var(--text-secondary);
    font-size: 0.84rem;
    line-height: 1.65;
}

/* ── HEALTH SCORECARD ── */
.health-score-ring {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.04em;
}
.health-bar-wrap {
    background: var(--bg-base);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin-top: 6px;
}
.health-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

/* ── STEP / FEATURE CARDS ── */
.step-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-lg);
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-md);
    transition: all 0.2s ease;
}

.feature-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-dim);
    border-radius: var(--radius-md);
    padding: 20px;
    transition: all 0.2s ease;
    height: 100%;
}
.feature-card:hover {
    border-color: var(--border-soft);
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
}

/* Domain / objective selector cards */
.domain-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-dim);
    border-radius: var(--radius-md);
    padding: 18px 14px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;
}
.domain-card:hover {
    border-color: var(--border-soft);
    background: var(--bg-hover);
    transform: translateY(-2px);
}
.domain-card-selected {
    border-color: var(--violet-500) !important;
    background: rgba(124,58,237,0.12) !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.15);
}
.domain-icon { font-size: 1.8rem; margin-bottom: 8px; display: block; }
.domain-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── DATA TABLE ENHANCEMENTS ── */
.type-badge {
    display: inline-block;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 2px 2px;
}

/* ── BUTTONS ── */
div.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.18s ease !important;
    border: 1px solid transparent !important;
}

/* Primary CTA — gradient */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid*="primary"] {
    background: linear-gradient(135deg, var(--violet-600) 0%, #6D28D9 100%) !important;
    color: #fff !important;
    border: 1px solid rgba(124,58,237,0.4) !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #6D28D9 0%, #5B21B6 100%) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.45) !important;
    transform: translateY(-1px) !important;
}

/* Download buttons */
div[data-testid="stDownloadButton"] > button {
    background: rgba(6,182,212,0.08) !important;
    border: 1px solid rgba(6,182,212,0.35) !important;
    color: var(--cyan-400) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(6,182,212,0.18) !important;
    border-color: var(--cyan-500) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(6,182,212,0.2) !important;
}

/* ── TABS ── */
div[data-testid="stTabs"] > div > div > button {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
    border-bottom: 2px solid transparent !important;
}
div[data-testid="stTabs"] > div > div > button[aria-selected="true"] {
    color: var(--violet-400) !important;
    border-bottom: 2px solid var(--violet-500) !important;
    background: rgba(124,58,237,0.07) !important;
}
div[data-testid="stTabs"] > div > div > button:hover {
    color: var(--text-secondary) !important;
    background: rgba(255,255,255,0.03) !important;
}

/* ── INPUTS & SELECTS ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] > div > div > input,
div[data-testid="stTextArea"] > div > div > textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stTextInput"] > div > div > input:focus,
div[data-testid="stTextArea"] > div > div > textarea:focus {
    border-color: rgba(124,58,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important;
}

/* ── SUCCESS / WARNING / INFO ALERTS ── */
div[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-size: 0.86rem !important;
    border: 1px solid !important;
}

/* ── EXPANDERS ── */
div[data-testid="stExpander"] {
    border: 1px solid var(--border-dim) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-elevated) !important;
}
div[data-testid="stExpander"]:hover {
    border-color: var(--border-soft) !important;
}

/* ── DATAFRAMES ── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-dim) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

/* ── CLEANING LOG TIMELINE ── */
.timeline-item {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-dim);
}
.timeline-item:last-child { border-bottom: none; }
.timeline-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green-500);
    margin-top: 6px;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(16,185,129,0.4);
}
.timeline-text {
    font-size: 0.83rem;
    color: var(--text-secondary);
    line-height: 1.5;
}
.timeline-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-top: 2px;
}

/* ── MISC ── */
.divider {
    border: none;
    border-top: 1px solid var(--border-dim);
    margin: 20px 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb {
    background: rgba(124,58,237,0.3);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.5); }

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
STEPS = [
    "business_collection",
    "data_cleaning",
    "eda",
    "feature_engineering",
    "communication"
]

def init_state():
    defaults = {
        "step": "business_collection",
        "df": None,
        "meta": None,
        "col_types": None,
        "cols_by_type": None,
        "quality": None,
        "col_stats": None,
        "corr": None,
        "business_goal": "Understand key drivers and trends in data",
        "cleaning_log": [],
        "transformation_log": [],
        "fe_explanations": [],
        "iris_says": {},
        "chat_history": [],
        "dataset_context": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
S = st.session_state


# ── Sidebar ────────────────────────────────────────────────────────────────────
STEP_META = [
    ("business_collection", "🎯", "1. Business & Data Collection"),
    ("data_cleaning",        "🧹", "2. Data Cleaning & Preprocessing"),
    ("eda",                 "📊", "3. Exploratory Data Analysis (EDA)"),
    ("feature_engineering", "⚙️", "4. Feature Engineering"),
    ("communication",       "📢", "5. Communication & Export"),
]

with st.sidebar:
    # ── Brand Header ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="iris-brand">
        <span class="iris-brand-icon">🌸</span>
        <div class="iris-brand-name">Iris Studio</div>
        <div class="iris-brand-tagline">Data Science Lifecycle</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Step Stepper ─────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section-label'>Pipeline</div>", unsafe_allow_html=True)

    STEP_LABELS = [
        ("business_collection", "Business & Data"),
        ("data_cleaning",       "Data Cleaning"),
        ("eda",                 "Exploration (EDA)"),
        ("feature_engineering", "Feature Engineering"),
        ("communication",       "Export & Report"),
    ]
    step_idx = [s[0] for s in STEP_LABELS].index(S["step"]) if S["step"] in [s[0] for s in STEP_LABELS] else 0

    for i, (sid, label) in enumerate(STEP_LABELS):
        num = i + 1
        if i < step_idx:
            # Completed — clickable
            st.markdown(f"""
            <div class="sidebar-step-item sidebar-step-done">
                <div class="step-dot" style="background:#10B981; box-shadow:0 0 8px rgba(16,185,129,0.5);"></div>
                <div style="flex:1; font-size:0.82rem; font-weight:500; color:#94A3B8;">
                    <span style="font-size:0.65rem; font-weight:700; color:#475569; letter-spacing:0.08em;">STEP {num}</span><br/>{label}
                </div>
                <span style="font-size:0.75rem; color:#10B981;">✓</span>
            </div>
            """, unsafe_allow_html=True)
            # Invisible button for navigation
            if st.button(f"Go to {label}", key=f"side_nav_{sid}", use_container_width=True,
                         help=f"Return to {label}"):
                S["step"] = sid
                st.rerun()
        elif i == step_idx:
            # Active
            st.markdown(f"""
            <div class="sidebar-step-item sidebar-step-active">
                <div class="step-dot"></div>
                <div style="flex:1;">
                    <span style="font-size:0.65rem; font-weight:700; color:#A78BFA; letter-spacing:0.08em;">STEP {num} — ACTIVE</span><br/>{label}
                </div>
                <span style="font-size:0.65rem; color:#A78BFA;">▶</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Locked
            st.markdown(f"""
            <div class="sidebar-step-item sidebar-step-locked">
                <div class="step-dot"></div>
                <div style="flex:1;">
                    <span style="font-size:0.65rem; font-weight:700; letter-spacing:0.08em;">STEP {num}</span><br/>{label}
                </div>
                <span style="font-size:0.7rem; opacity:0.4;">🔒</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Download Links ────────────────────────────────────────────────────────
    if S["df"] is not None and S["step"] != "business_collection":
        st.markdown("<div style='border-top:1px solid rgba(255,255,255,0.05); margin:8px 8px 0;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-section-label'>Downloads</div>", unsafe_allow_html=True)

        csv_bytes = build_cleaned_csv(S["df"])
        st.download_button("Cleaned CSV", csv_bytes, "iris_cleaned.csv", "text/csv",
                           use_container_width=True, key="side_dl_csv")

        try:
            html_str = build_html_dashboard(
                S["meta"], S["col_types"], S["quality"] or {},
                S["col_stats"] or {}, S["corr"] or {},
                {}, S["iris_says"], S["df"]
            )
            st.download_button("Full HTML Dashboard", html_str, "iris_dashboard.html", "text/html",
                               use_container_width=True, key="side_dl_html")
        except Exception:
            pass


# ── Helpers ─────────────────────────────────────────────────────────────────────
def iris_says_block(key: str, compute_fn=None, *args):
    if key not in S["iris_says"] and compute_fn:
        with st.spinner("Iris is analyzing..."):
            S["iris_says"][key] = compute_fn(*args)
    text = S["iris_says"].get(key, "")
    if text:
        st.markdown(f"""
        <div class="iris-insight">
            <div class="iris-avatar">🌸</div>
            <div class="iris-insight-body">
                <div class="iris-insight-label">Iris AI Analysis</div>
                <div class="iris-insight-text">{text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def next_step(current: str):
    idx = STEPS.index(current)
    if idx + 1 < len(STEPS):
        S["step"] = STEPS[idx + 1]
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Business Understanding & Data Collection
# ══════════════════════════════════════════════════════════════════════════════
if S["step"] == "business_collection":
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 16px; position:relative;'>
        <div style='font-size:3.5rem; line-height:1; margin-bottom:16px;
                    filter: drop-shadow(0 0 40px rgba(124,58,237,0.7));'>🌸</div>
        <h1 style='
            font-family: Inter, sans-serif;
            font-size:2.8rem; font-weight:900; letter-spacing:-0.04em; margin:0 0 12px;
            background: linear-gradient(135deg, #F1F5F9 0%, #A78BFA 45%, #22D3EE 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; line-height:1.1;'>
            Iris Data Science Studio
        </h1>
        <p style='color:#94A3B8; font-size:1rem; max-width:600px; margin:0 auto 8px; line-height:1.75;'>
            Select your domain objective, ingest your dataset, and let Iris automate data quality checks, narrative EDA, and feature engineering.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("🎯 Stage 1A: Domain & Business Objective Setup", expanded=True):
        st.markdown("<div style='color:#94A3B8; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px;'>Select Industry Domain</div>", unsafe_allow_html=True)
        domains = get_domain_templates()
        d_keys = list(domains.keys())
        d_cols = st.columns(len(d_keys))
        
        selected_dom = S.get("domain", "general")
        for idx, key in enumerate(d_keys):
            d_info = domains[key]
            is_sel = (selected_dom == key)
            b_type = "primary" if is_sel else "secondary"
            clean_name = d_info['name'].split('&')[0].strip()
            b_label = f"✓ {d_info['icon']} {clean_name}" if is_sel else f"{d_info['icon']} {clean_name}"
            with d_cols[idx]:
                if st.button(b_label, type=b_type, use_container_width=True, key=f"dom_btn_{key}"):
                    S["domain"] = key
                    S["business_goal"] = d_info["default_goal"]
                    st.rerun()
                    
        st.markdown("<div style='margin-top:16px; color:#94A3B8; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px;'>Select Objective Template</div>", unsafe_allow_html=True)
        selected_obj = S.get("objective", "explore")
        objs = [
            ("predict", "📈 Predict Outcomes"),
            ("segment", "👥 Segment Data"),
            ("detect", "🔍 Detect Anomalies"),
            ("explore", "📊 Explore & Report")
        ]
        o_cols = st.columns(4)
        for idx, (o_key, o_label) in enumerate(objs):
            is_sel = (selected_obj == o_key)
            b_type = "primary" if is_sel else "secondary"
            lbl = f"✓ {o_label}" if is_sel else o_label
            with o_cols[idx]:
                if st.button(lbl, type=b_type, use_container_width=True, key=f"obj_{o_key}"):
                    S["objective"] = o_key
                    st.rerun()
                    
        # Executive summary bar for active selection
        cur_dom_name = domains.get(S.get('domain', 'general'), {}).get('name', 'General Analytics')
        cur_obj_label = dict(objs).get(S.get('objective', 'explore'), 'Explore & Report')
        
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:12px; background:rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.25); padding:10px 16px; border-radius:10px; margin: 16px 0 14px;'>
            <div style='font-size:0.75rem; font-weight:700; color:#A78BFA; text-transform:uppercase; letter-spacing:0.08em;'>Active Setup:</div>
            <div style='font-size:0.85rem; font-weight:600; color:#F1F5F9;'>
                Domain: <span style='color:#38BDF8;'>{cur_dom_name}</span> &nbsp;|&nbsp; Objective: <span style='color:#34D399;'>{cur_obj_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
                
        goal_input = st.text_area(
            "Business Objective Narrative:",
            value=S.get("business_goal", "Identify key analytical patterns, customer segments, or predictor variables."),
            height=70,
            placeholder="e.g. Reduce customer churn rate by identifying high-risk signals"
        )
        S["business_goal"] = goal_input

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📂 Stage 1B: Data Collection & Profiling")
    
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"], label_visibility="hidden")
        st.markdown("<div style='text-align:center; color:#8B5CF6; font-size:0.85rem; margin:16px 0 8px; font-weight:700;'>— OR LOAD DEMO DATASET —</div>", unsafe_allow_html=True)
        
        d1, d2, d3 = st.columns(3)
        sample_bytes = None
        
        with d1:
            if st.button("🚢 Titanic Survival", use_container_width=True, key="demo_titanic"):
                path = os.path.join(os.path.dirname(__file__), "samples", "titanic.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()
            if st.button("🏥 Heart Disease", use_container_width=True, key="demo_heart"):
                path = os.path.join(os.path.dirname(__file__), "samples", "heart_disease.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()
                        
        with d2:
            if st.button("🏠 Housing Prices", use_container_width=True, key="demo_housing"):
                path = os.path.join(os.path.dirname(__file__), "samples", "housing.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()
            if st.button("📱 Customer Churn", use_container_width=True, key="demo_churn"):
                path = os.path.join(os.path.dirname(__file__), "samples", "customer_churn.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()
                        
        with d3:
            if st.button("🌸 Iris Flowers", use_container_width=True, key="demo_iris"):
                path = os.path.join(os.path.dirname(__file__), "samples", "iris.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()
            if st.button("🛒 E-Commerce Sales", use_container_width=True, key="demo_ecom"):
                path = os.path.join(os.path.dirname(__file__), "samples", "ecommerce_sales.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()

        file_bytes = uploaded.read() if uploaded else sample_bytes
        if file_bytes:
            with st.spinner("Parsing dataset... 🌸"):
                df, meta = load_csv(file_bytes)
                col_types = infer_all_types(df)
                cols_by_type = get_columns_by_type(df, col_types)
            S["df"] = df
            S["meta"] = meta
            S["col_types"] = col_types
            S["cols_by_type"] = cols_by_type
            S["target_col"] = suggest_target_column(df, S.get("domain", "general"))
            for k in ["quality", "col_stats", "corr", "iris_says"]:
                S[k] = {} if k == "iris_says" else None

    if S["df"] is not None:
        df = S["df"]
        meta = S["meta"]
        col_types = S["col_types"]
        health = compute_data_health_score(df)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Dataset Health & Profiling Snapshot")
        
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            render_kpi_card("Total Rows", f"{len(df):,}", f"{meta['encoding']} encoding", "#7C3AED")
        with p2:
            render_kpi_card("Total Columns", str(len(df.columns)), f"{health['missing_cols']} missing cols", "#06B6D4")
        with p3:
            render_kpi_card("Health Score", f"{health['score']} / 100", "Data Quality Index", "#10B981" if health['score'] >= 80 else "#F59E0B")
        with p4:
            s_target = S.get("target_col", "None")
            render_kpi_card("Target Feature", str(s_target), "Suggested Target", "#EC4899")

        st.markdown("<br>", unsafe_allow_html=True)
        render_data_preview_table(df, key_prefix="stage1")


        type_summary = {}
        for t in set(col_types.values()):
            type_summary[t] = [c for c, ct in col_types.items() if ct == t]

        st.markdown("**Inferred Feature Types:**", unsafe_allow_html=True)
        chips = ""
        type_colors = {
            "numeric": "#8B5CF6", "categorical": "#06B6D4", "datetime": "#F59E0B",
            "text": "#EC4899", "boolean": "#3B82F6", "id": "#94A3B8"
        }
        for t, cols in type_summary.items():
            color = type_colors.get(t, "#888")
            chips += f'<span class="type-badge" style="background:{color}22; color:{color}; border:1px solid {color}55;">{t} ({len(cols)})</span> '
        st.markdown(chips, unsafe_allow_html=True)

        render_iris_says(f"Dataset successfully loaded! Health index is **{health['score']}/100**. Target candidate column suggested: **'{S.get('target_col')}'**.", "Iris Data Profiler")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next: Data Cleaning & Preprocessing Studio →", type="primary", use_container_width=True):
            next_step("business_collection")



# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Data Cleaning and Preprocessing Studio
# ══════════════════════════════════════════════════════════════════════════════
elif S["step"] == "data_cleaning":
    render_stage_header(2, "Data Cleaning & Preprocessing Studio", "Recommendations-first data preparation: automated diagnosis, health scorecard, and 1-click fixes.")

    df = S["df"]
    if S["quality"] is None:
        with st.spinner("Auditing data quality..."):
            S["quality"] = full_quality_report(df, S["col_types"])

    q = S["quality"]
    health = compute_data_health_score(df)

    render_health_scorecard(health["score"], health["missing_cols"], health["dup_rows"], health["outlier_cols"])

    recs = detect_cleaning_recommendations(df)
    if recs:
        with st.expander(f"💡 Iris Data Cleaning Recommendations ({len(recs)} Actionable Suggestions)", expanded=True):
            r_col1, r_col2 = st.columns([3, 1])
            with r_col1:
                for rec in recs:
                    render_recommendation_card(rec["priority"], rec["title"], rec["reason"], rec["col_name"])
            with r_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("⚡ Apply All Fixes (1-Click)", type="primary", use_container_width=True, key="btn_apply_all_recs"):
                    new_df = df.copy()
                    applied_cnt = 0
                    for rec in recs:
                        if rec["action_type"] == "drop_col" and rec["col_name"] in new_df.columns:
                            new_df = new_df.drop(columns=[rec["col_name"]])
                            applied_cnt += 1
                        elif rec["action_type"] == "impute" and rec["col_name"] in new_df.columns:
                            s_num = pd.to_numeric(new_df[rec["col_name"]], errors="coerce")
                            if s_num.notnull().sum() > len(new_df) * 0.5:
                                new_df[rec["col_name"]] = s_num.fillna(s_num.median())
                            else:
                                new_df[rec["col_name"]] = new_df[rec["col_name"]].fillna(new_df[rec["col_name"]].mode().iloc[0] if len(new_df[rec["col_name"]].mode())>0 else "Unknown")
                            applied_cnt += 1
                        elif rec["action_type"] == "drop_dups":
                            new_df = new_df.drop_duplicates().reset_index(drop=True)
                            applied_cnt += 1
                        elif rec["action_type"] == "cap_outliers" and rec["col_name"] in new_df.columns:
                            s_num = pd.to_numeric(new_df[rec["col_name"]], errors="coerce").astype(float)
                            q1, q3 = s_num.quantile(0.25), s_num.quantile(0.75)
                            iqr = q3 - q1
                            if iqr > 0:
                                new_df[rec["col_name"]] = s_num.clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr)
                            applied_cnt += 1

                    
                    S["df"] = new_df
                    S["quality"] = full_quality_report(S["df"], S["col_types"])
                    msg = f"Applied {applied_cnt} recommended data cleaning actions in 1 click!"
                    S["cleaning_log"].append(msg)
                    st.toast(msg, icon="⚡")
                    st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    t_miss, t_dup, t_out, t_struct, t_enc, t_scale, t_red = st.tabs([
        "Missing Values",
        "Duplicates",
        "Treat Outliers",
        "Structural Errors",
        "Categorical Encoding",
        "Feature Scaling",
        "Data Reduction"
    ])


    with t_miss:
        st.markdown("### 🩹 Handling Missing Values")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Fill gaps using statistical imputation (mean, median, mode) or drop sparse rows and columns if data loss is acceptable.</div>", unsafe_allow_html=True)

        miss_df = q["missing"]
        if len(miss_df) > 0:
            st.dataframe(miss_df, use_container_width=True)
            
            st.markdown("**Batch Quick Imputation:**")
            g1, g2, g3, g4, g5 = st.columns(5)
            with g1:
                if st.button("📊 Numeric → Median", use_container_width=True, key="btn_imp_med"):
                    for c, ct in S["col_types"].items():
                        if ct == "numeric" and S["df"][c].isnull().sum() > 0:
                            S["df"], msg = impute_missing(S["df"], c, "median")
                            S["cleaning_log"].append(msg)
                    S["quality"] = full_quality_report(S["df"], S["col_types"])
                    st.toast("Imputed numeric features with median!", icon="📊")
                    st.rerun()
            with g2:
                if st.button("📈 Numeric → Mean", use_container_width=True, key="btn_imp_mean"):
                    for c, ct in S["col_types"].items():
                        if ct == "numeric" and S["df"][c].isnull().sum() > 0:
                            S["df"], msg = impute_missing(S["df"], c, "mean")
                            S["cleaning_log"].append(msg)
                    S["quality"] = full_quality_report(S["df"], S["col_types"])
                    st.toast("Imputed numeric features with mean!", icon="📈")
                    st.rerun()
            with g3:
                if st.button("🏷️ Categorical → Mode", use_container_width=True, key="btn_imp_mode"):
                    for c, ct in S["col_types"].items():
                        if ct in ("categorical", "boolean") and S["df"][c].isnull().sum() > 0:
                            S["df"], msg = impute_missing(S["df"], c, "mode")
                            S["cleaning_log"].append(msg)
                    S["quality"] = full_quality_report(S["df"], S["col_types"])
                    st.toast("Imputed categorical features with mode!", icon="🏷️")
                    st.rerun()
            with g4:
                if st.button("🗑️ Drop Sparse Columns (>40%)", use_container_width=True, key="btn_drop_sparse"):
                    S["df"], msg = drop_sparse_columns(S["df"], 40.0)
                    S["col_types"] = infer_all_types(S["df"])
                    S["cleaning_log"].append(msg)
                    S["quality"] = full_quality_report(S["df"], S["col_types"])
                    st.toast(msg, icon="🗑️")
                    st.rerun()
            with g5:
                if st.button("❌ Drop Rows with Nulls", use_container_width=True, key="btn_drop_null_rows"):
                    n_before = len(S["df"])
                    S["df"] = S["df"].dropna().reset_index(drop=True)
                    msg = f"Dropped {n_before - len(S['df'])} rows with missing values."
                    S["cleaning_log"].append(msg)
                    S["quality"] = full_quality_report(S["df"], S["col_types"])
                    st.toast(msg, icon="❌")
                    st.rerun()

            st.markdown("<br>**Per-Column Targeted Imputation:**", unsafe_allow_html=True)
            col_target = st.selectbox("Select column to impute:", miss_df["column"].tolist(), key="select_imp_col")
            imp_method = st.selectbox("Imputation Method:", ["median", "mean", "mode", "constant", "ffill", "drop_rows", "drop_column"], key="select_imp_method")
            const_val = st.text_input("Constant value (if method='constant'):", value="0", key="input_const_val") if imp_method == "constant" else None

            if st.button("✨ Apply Imputation", type="primary", key="btn_apply_single_imp"):
                S["df"], msg = impute_missing(S["df"], col_target, imp_method, fill_value=const_val)
                S["col_types"] = infer_all_types(S["df"])
                S["cleaning_log"].append(msg)
                S["quality"] = full_quality_report(S["df"], S["col_types"])
                st.toast(msg, icon="✨")
                st.rerun()
        else:
            st.success("🎉 Complete Dataset: Zero missing values detected.")

        if df.isnull().any().any():
            st.markdown("<br>**Missing Heatmap:**", unsafe_allow_html=True)
            st.plotly_chart(missing_heatmap(df), use_container_width=True)

    with t_dup:
        st.markdown("### 👥 Removing Duplicate Entries")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Eliminate redundant or repeated entries to prevent skewing model analysis.</div>", unsafe_allow_html=True)
        
        n_dups = int(df.duplicated().sum())
        if n_dups > 0:
            st.warning(f"⚠️ Detected **{n_dups} duplicate rows** ({round(n_dups/len(df)*100, 2)}% of dataset).")
            with st.expander("Preview Duplicate Rows"):
                st.dataframe(df[df.duplicated(keep=False)].head(20), use_container_width=True)
            if st.button("🗑️ Eliminate All Duplicate Rows", type="primary", key="btn_remove_dups"):
                S["df"], msg = remove_duplicates(S["df"])
                S["cleaning_log"].append(msg)
                S["quality"] = full_quality_report(S["df"], S["col_types"])
                st.toast(msg, icon="👥")
                st.rerun()
        else:
            st.success("✅ Clean Dataset: Zero duplicate rows found.")

    with t_out:
        st.markdown("### 🎯 Treating Extreme Outliers")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Detect and manage extreme anomalies using Interquartile Range (IQR) or Z-score methods.</div>", unsafe_allow_html=True)

        num_cols = [c for c, t in S["col_types"].items() if t == "numeric"]
        out_df = q["outliers"]
        if len(out_df) > 0:
            st.dataframe(out_df, use_container_width=True)
            if st.button("🎯 Cap All IQR Outliers to Boundaries (1-Click)", type="primary", key="btn_cap_all_outliers"):
                new_df = S["df"].copy()
                for _, r in out_df.iterrows():
                    c = r["column"]
                    if c in new_df.columns:
                        s_n = pd.to_numeric(new_df[c], errors="coerce").astype(float)
                        q1, q3 = s_n.quantile(0.25), s_n.quantile(0.75)
                        iqr = q3 - q1
                        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        new_df[c] = s_n.clip(lower=lo, upper=hi)

                S["df"] = new_df
                S["quality"] = full_quality_report(S["df"], S["col_types"])
                msg = "Capped extreme outliers across all numeric columns to IQR boundaries."
                S["cleaning_log"].append(msg)
                st.toast(msg, icon="🎯")
                st.rerun()
        else:
            st.info("No significant outliers detected.")

        if num_cols:
            st.markdown("<br>**Per-Column Outlier Treatment:**", unsafe_allow_html=True)
            col_out = st.selectbox("Select numeric column to treat:", num_cols, key="select_out_col")
            out_method = st.radio("Outlier Detection Method:", ["IQR (Interquartile Range)", "Z-Score (|Z| > threshold)"], horizontal=True, key="radio_out_method")
            out_action = st.radio("Action:", ["cap (Clip to boundary values)", "drop (Remove outlier rows)"], horizontal=True, key="radio_out_action")

            if st.button("🎯 Apply Outlier Treatment to Column", key="btn_apply_outlier"):
                act = "cap" if "cap" in out_action else "drop"
                if "IQR" in out_method:
                    S["df"], msg = treat_outliers_iqr(S["df"], col_out, action=act)
                else:
                    S["df"], msg = treat_outliers_zscore(S["df"], col_out, action=act)
                S["cleaning_log"].append(msg)
                S["quality"] = full_quality_report(S["df"], S["col_types"])
                st.toast(msg, icon="🎯")
                st.rerun()

    with t_struct:
        st.markdown("### 🔤 Correcting Structural Errors & Types")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Iris automatically scans text features for whitespace padding, mixed capitalization, and improper data types.</div>", unsafe_allow_html=True)

        struct_issues = detect_structural_errors(S["df"])
        if struct_issues:
            st.markdown(f"#### 🔍 Detected Structural Issues ({len(struct_issues)} Inconsistencies Found)")
            for issue in struct_issues:
                render_recommendation_card("HIGH", issue["title"], issue["reason"], issue["column"])
            
            if st.button("⚡ Fix All Structural Errors (1-Click)", type="primary", key="btn_fix_all_struct"):
                new_df = S["df"].copy()
                fixed_cnt = 0
                for issue in struct_issues:
                    col = issue["column"]
                    act = issue["recommended_action"]
                    if act in ("trim_whitespace", "titlecase"):
                        new_df, _ = clean_structural_errors(new_df, col, act)
                        fixed_cnt += 1
                    elif act == "cast_numeric":
                        new_df, _ = convert_column_type(new_df, col, "numeric")
                        fixed_cnt += 1
                S["df"] = new_df
                S["col_types"] = infer_all_types(new_df)
                S["quality"] = full_quality_report(S["df"], S["col_types"])
                msg = f"Fixed {fixed_cnt} structural errors automatically!"
                S["cleaning_log"].append(msg)
                st.toast(msg, icon="🔤")
                st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.success("✅ Clean Text Features: Zero structural errors or whitespace padding detected.")
            st.markdown("<br>", unsafe_allow_html=True)

        st1, st2 = st.columns(2)
        with st1:
            st.markdown("**Targeted String Cleanup:**")
            str_col = st.selectbox("Select Text/Categorical Column:", [c for c, t in S["col_types"].items() if t in ("text", "categorical", "id")], key="select_str_col")
            str_act = st.selectbox("Action:", [
                ("trim_whitespace", "Trim Extra Leading/Trailing Whitespaces"),
                ("lowercase", "Convert to lowercase"),
                ("uppercase", "Convert to UPPERCASE"),
                ("titlecase", "Convert to Title Case"),
                ("remove_special", "Remove Special Characters & Punctuation"),
            ], format_func=lambda x: x[1], key="select_str_act")

            if st.button("🔤 Clean String Errors", type="primary", key="btn_clean_string"):
                S["df"], msg = clean_structural_errors(S["df"], str_col, str_act[0])
                S["cleaning_log"].append(msg)
                st.toast(msg, icon="🔤")
                st.rerun()

        with st2:
            st.markdown("**Data Type Correction:**")
            type_col = st.selectbox("Select Column to Re-cast Type:", S["df"].columns.tolist(), key="select_type_col")
            target_type = st.selectbox("Target Data Type:", ["numeric", "categorical", "datetime", "boolean"], key="select_target_type")

            if st.button("🔄 Convert Data Type", type="primary", key="btn_convert_type"):
                S["df"], msg = convert_column_type(S["df"], type_col, target_type)
                S["col_types"] = infer_all_types(S["df"])
                S["cleaning_log"].append(msg)
                st.toast(msg, icon="🔄")
                st.rerun()


    with t_enc:
        st.markdown("### 🔢 Categorical Encoding — Automated Recommendations")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Iris analyzes your categorical columns and recommends the optimal encoding method for each feature.</div>", unsafe_allow_html=True)

        cat_suggs = get_encoding_suggestions(S["df"], S["col_types"])
        if cat_suggs:
            st.markdown("""
            <div style='background:linear-gradient(135deg, rgba(139,92,246,0.15), rgba(6,182,212,0.1));
                        border:1px solid #8B5CF6; border-radius:14px; padding:18px; margin-bottom:20px;'>
                <div style='font-size:1.1rem; font-weight:700; color:#8B5CF6;'>⚡ One-Click Automated Encoding</div>
                <div style='color:#9CA3AF; font-size:0.85rem; margin-top:4px;'>Apply all recommended encoding methods across all categorical features instantly.</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("⚡ Auto-Encode All Categorical Features (Recommended)", type="primary", key="btn_auto_encode_all"):
                S["df"], msg = auto_encode_all_categorical(S["df"], S["col_types"])
                S["col_types"] = infer_all_types(S["df"])
                S["cleaning_log"].append(msg)
                st.toast(msg, icon="⚡")
                st.rerun()

            st.markdown("<br>**Column-by-Column Automated Recommendations:**", unsafe_allow_html=True)
            for sug in cat_suggs:
                c_name = sug["column"]
                n_u = sug["n_unique"]
                m_name = sug["method_name"]
                reason = sug["reason"]
                m_code = sug["method"]

                c_info, c_action = st.columns([3, 1])
                with c_info:
                    st.markdown(f"**{c_name}** ({n_u} unique categories) → 💡 **{m_name}**")
                    st.caption(reason)
                with c_action:
                    if st.button(f"Apply to {c_name}", key=f"btn_apply_enc_{c_name}", use_container_width=True):
                        S["df"], msg = encode_categorical(S["df"], c_name, m_code)
                        S["col_types"] = infer_all_types(S["df"])
                        S["cleaning_log"].append(msg)
                        st.toast(msg, icon="🔢")
                        st.rerun()
                st.markdown("<hr style='border-color:rgba(139,92,246,0.1); margin:8px 0;'>", unsafe_allow_html=True)
        else:
            st.info("No categorical features remaining for encoding.")

    with t_scale:
        st.markdown("### 📏 Feature Scaling — Automated Recommendations")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Iris analyzes your numeric distributions and recommends Standardization (Z-score) vs. Min-Max scaling per column.</div>", unsafe_allow_html=True)

        scale_suggs = get_scaling_suggestions(S["df"], S["col_types"])
        if scale_suggs:
            st.markdown("""
            <div style='background:linear-gradient(135deg, rgba(6,182,212,0.15), rgba(139,92,246,0.1));
                        border:1px solid #06B6D4; border-radius:14px; padding:18px; margin-bottom:20px;'>
                <div style='font-size:1.1rem; font-weight:700; color:#06B6D4;'>⚡ One-Click Automated Scaling</div>
                <div style='color:#9CA3AF; font-size:0.85rem; margin-top:4px;'>Standardize high-variance numeric columns and scale bounded features into uniform ranges.</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("⚡ Auto-Scale All Numeric Features (Recommended)", type="primary", key="btn_auto_scale_all"):
                S["df"], msg = auto_scale_all_numeric(S["df"], S["col_types"])
                S["col_types"] = infer_all_types(S["df"])
                S["cleaning_log"].append(msg)
                st.toast(msg, icon="⚡")
                st.rerun()

            st.markdown("<br>**Column-by-Column Automated Recommendations:**", unsafe_allow_html=True)
            for sug in scale_suggs:
                c_name = sug["column"]
                m_name = sug["method_name"]
                reason = sug["reason"]
                m_code = sug["method"]
                min_v, max_v = sug["min"], sug["max"]

                c_info, c_action = st.columns([3, 1])
                with c_info:
                    st.markdown(f"**{c_name}** [Range: {min_v:.1f} to {max_v:.1f}] → 💡 **{m_name}**")
                    st.caption(reason)
                with c_action:
                    if m_code != "none":
                        if st.button(f"Apply to {c_name}", key=f"btn_apply_scale_{c_name}", use_container_width=True):
                            S["df"], msg = scale_feature(S["df"], c_name, m_code)
                            S["col_types"] = infer_all_types(S["df"])
                            S["cleaning_log"].append(msg)
                            st.toast(msg, icon="📏")
                            st.rerun()
                    else:
                        st.caption("✅ Scaled")
                st.markdown("<hr style='border-color:rgba(6,182,212,0.1); margin:8px 0;'>", unsafe_allow_html=True)
        else:
            st.info("No numeric features remaining for scaling.")

    with t_red:
        st.markdown("### ✂️ Simple Data Reduction")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>One-click actions to drop useless constant columns and remove redundant collinear features.</div>", unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("""
            <div class="feature-card" style="padding:20px; text-align:center;">
                <div style="font-size:2rem; margin-bottom:6px;">🚫</div>
                <div style="font-weight:700; color:#8B5CF6; margin-bottom:4px;">Constant Columns</div>
                <div style="color:#9CA3AF; font-size:0.82rem; margin-bottom:14px;">Drop features with 0 variance (only 1 unique value).</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚫 Drop Constant Columns (1-Click)", use_container_width=True, key="btn_drop_const"):
                S["df"], msg = drop_constant_features(S["df"])
                S["col_types"] = infer_all_types(S["df"])
                S["cleaning_log"].append(msg)
                st.toast(msg, icon="🚫")
                st.rerun()

        with r2:
            st.markdown("""
            <div class="feature-card" style="padding:20px; text-align:center;">
                <div style="font-size:2rem; margin-bottom:6px;">⚡</div>
                <div style="font-weight:700; color:#06B6D4; margin-bottom:4px;">Redundant Collinear Features</div>
                <div style="color:#9CA3AF; font-size:0.82rem; margin-bottom:14px;">Drop feature pairs with correlation |r| > 0.90.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("⚡ Drop Redundant Collinear Features (1-Click)", use_container_width=True, key="btn_drop_high_corr"):
                S["df"], msg = drop_high_correlation_features(S["df"], 0.90)
                S["col_types"] = infer_all_types(S["df"])
                S["cleaning_log"].append(msg)
                st.toast(msg, icon="⚡")
                st.rerun()

        st.markdown("<br>**Manual Quick Drop:**", unsafe_allow_html=True)
        cols_to_drop = st.multiselect("Select columns to remove:", S["df"].columns.tolist(), key="select_manual_drop")
        if st.button("🗑️ Drop Selected Columns", type="primary", key="btn_manual_drop"):
            if cols_to_drop:
                S["df"] = S["df"].drop(columns=cols_to_drop)
                S["col_types"] = infer_all_types(S["df"])
                msg = f"Manually dropped {len(cols_to_drop)} columns: {', '.join(cols_to_drop)}"
                S["cleaning_log"].append(msg)
                st.toast(msg, icon="🗑️")
                st.rerun()

    if S["cleaning_log"]:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#475569; margin-bottom:8px;'>Cleaning Actions Applied</div>", unsafe_allow_html=True)
        timeline_html = "<div style='padding: 4px 0;'>"
        for log_msg in S["cleaning_log"]:
            timeline_html += f"""
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-text">{log_msg}</div>
            </div>"""
        timeline_html += "</div>"
        st.markdown(timeline_html, unsafe_allow_html=True)

    iris_says_block("quality", explain_data_quality, q)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Next: Exploratory Data Analysis (EDA) →", type="primary", use_container_width=True):
        next_step("data_cleaning")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Automated Exploratory Data Analysis (EDA) Studio
# ══════════════════════════════════════════════════════════════════════════════
elif S["step"] == "eda":
    render_stage_header(3, "Exploratory Data Analysis (EDA) Narrative Studio", "Iris converts complex feature distributions into plain-English data stories, interactive charts, and executive insights.")

    df = S["df"]
    col_types = S["col_types"]

    if S["col_stats"] is None:
        with st.spinner("Computing statistics..."):
            S["col_stats"] = all_column_stats(df, col_types)
    if S["corr"] is None:
        with st.spinner("Computing correlations..."):
            S["corr"] = correlation_matrix(df, col_types)

    findings = generate_eda_key_findings(df, col_types)
    if findings:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #06B6D4;">
            <div style="font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 10px;">📊 Iris EDA Narrative Key Findings</div>
            <ul style="color: #CBD5E1; margin: 0; padding-left: 20px; line-height: 1.7;">
        """ + "".join([f"<li>{f}</li>" for f in findings]) + """
            </ul>
        </div>
        """, unsafe_allow_html=True)

    eda_recs = generate_eda_chart_recommendations(df, col_types, S["col_stats"], S["corr"])


    try:
        eda_html_str = build_eda_standalone_dashboard(df, col_types, S["col_stats"], S["corr"], eda_recs)
        st.download_button(
            "🌐 Export Standalone EDA Charts Dashboard (HTML)",
            data=eda_html_str,
            file_name="iris_eda_charts_dashboard.html",
            mime="text/html",
            type="primary",
            use_container_width=True,
            key="btn_download_eda_html"
        )
    except Exception as e:
        st.error(f"Error generating EDA dashboard HTML: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    tab_auto_charts, tab_col_explorer, tab_corr_matrix = st.tabs([
        "📊 1. Automated Recommended Charts",
        "🔎 2. Feature Deep-Dive Explorer",
        "🔗 3. Correlation & Bivariate Matrix"
    ])

    with tab_auto_charts:
        st.markdown(f"### 💡 {len(eda_recs)} Automated Chart Recommendations for your Dataset")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:20px;'>Below are high-value visualizations chosen by Iris based on feature distributions, statistical properties, and correlation strength.</div>", unsafe_allow_html=True)

        for rec in eda_recs:
            st.markdown(f"""
            <div style="background:#151528; border:1px solid rgba(139,92,246,0.25); border-radius:16px; padding:20px; margin-bottom:24px;">
                <div style="display:flex; gap:8px; margin-bottom:8px;">
                    <span class="type-badge" style="background:rgba(139,92,246,0.15); color:#8B5CF6; border:1px solid rgba(139,92,246,0.35);">{rec['category']}</span>
                    <span class="type-badge" style="background:rgba(6,182,212,0.15); color:#06B6D4; border:1px solid rgba(6,182,212,0.35);">{rec['chart_type']}</span>
                </div>
                <div style="font-size:1.25rem; font-weight:800; color:#F3F4F6; margin-bottom:6px;">{rec['title']}</div>
                <div style="color:#9CA3AF; font-size:0.83rem;">Target Features: <strong>{', '.join(rec['columns'])}</strong></div>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(rec["fig"], use_container_width=True)

            st.markdown(f"""
            <div class="why-card">
                <strong>❓ Why are we making this chart?</strong><br>
                {rec['why']}
            </div>
            <br><hr style="border-color:rgba(139,92,246,0.15); margin:24px 0;">
            """, unsafe_allow_html=True)

    with tab_col_explorer:
        all_cols = df.columns.tolist()
        selected_col = st.selectbox("Select feature for deep-dive exploration:", all_cols, key="select_eda_col")
        ctype = col_types.get(selected_col, "unknown")
        stats = S["col_stats"].get(selected_col, {})

        st.markdown(f"### {selected_col} (`{ctype}`)")

        chip_keys = ["count", "missing", "unique", "mean", "median", "std", "min", "max", "skewness"]
        chips_html = "<div style='display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px;'>"
        for k in chip_keys:
            if k in stats and stats[k] is not None:
                chips_html += f'<div style="background:#151528; border:1px solid rgba(139,92,246,0.25); border-radius:10px; padding:6px 14px; font-size:0.8rem;"><span style="color:#9CA3AF;">{k.title()}:</span> <strong style="color:#06B6D4;">{stats[k]}</strong></div>'
        chips_html += "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)

        if ctype == "numeric":
            col_s = df[selected_col].iloc[:, 0] if isinstance(df[selected_col], pd.DataFrame) else df[selected_col]
            s = pd.to_numeric(col_s, errors="coerce").dropna()
            plot_df = pd.DataFrame({selected_col: s})

            t1, t2, t3, t4 = st.tabs(["📊 Distribution Histogram", "🎻 Violin Plot", "📦 Box Plot", "🎯 Outlier Strip"])
            with t1:
                fig = px.histogram(plot_df, x=selected_col, nbins=40, color_discrete_sequence=["#8B5CF6"], marginal="rug")
                fig.update_layout(plot_bgcolor="#0F0F20", paper_bgcolor="#151528", font=dict(color="#F3F4F6"))
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                fig = px.violin(plot_df, y=selected_col, box=True, points="outliers", color_discrete_sequence=["#06B6D4"])
                fig.update_layout(plot_bgcolor="#0F0F20", paper_bgcolor="#151528", font=dict(color="#F3F4F6"))
                st.plotly_chart(fig, use_container_width=True)
            with t3:
                fig = px.box(plot_df, y=selected_col, points="all", color_discrete_sequence=["#EC4899"])
                fig.update_layout(plot_bgcolor="#0F0F20", paper_bgcolor="#151528", font=dict(color="#F3F4F6"))
                st.plotly_chart(fig, use_container_width=True)
            with t4:
                q1v, q3v = s.quantile(0.25), s.quantile(0.75)
                iqrv = q3v - q1v
                lo, hi = q1v - 1.5*iqrv, q3v + 1.5*iqrv
                color = s.apply(lambda v: "Outlier" if v < lo or v > hi else "Normal")
                tmp = pd.DataFrame({"value": s, "type": color})
                fig = px.strip(tmp, y="value", color="type", color_discrete_map={"Normal":"#8B5CF6", "Outlier":"#EC4899"})
                fig.update_layout(plot_bgcolor="#0F0F20", paper_bgcolor="#151528", font=dict(color="#F3F4F6"))
                st.plotly_chart(fig, use_container_width=True)

        elif ctype in ("categorical", "boolean"):
            col_s = df[selected_col].iloc[:, 0] if isinstance(df[selected_col], pd.DataFrame) else df[selected_col]
            vc = col_s.value_counts().head(20).reset_index()
            vc.columns = ["Category", "Count"]
            t1, t2 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart"])
            with t1:
                fig = px.bar(vc, x="Count", y="Category", orientation="h", color="Count", color_continuous_scale=["#151528", "#06B6D4"])
                fig.update_layout(plot_bgcolor="#0F0F20", paper_bgcolor="#151528", font=dict(color="#F3F4F6"), yaxis=dict(title=selected_col))
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                fig = px.pie(vc, names="Category", values="Count", hole=0.35, color_discrete_sequence=["#8B5CF6","#06B6D4","#EC4899","#10B981"])
                fig.update_layout(paper_bgcolor="#151528", font=dict(color="#F3F4F6"))
                st.plotly_chart(fig, use_container_width=True)

        iris_says_block(f"col_{selected_col}", explain_column, selected_col, stats)

    with tab_corr_matrix:
        corr_res = S["corr"]
        if corr_res.get("pearson") is not None:
            st.plotly_chart(correlation_heatmap(corr_res["pearson"]), use_container_width=True)
            if corr_res.get("top_pairs"):
                st.markdown("**Top Correlated Pairs:**")
                st.dataframe(pd.DataFrame(corr_res["top_pairs"]), use_container_width=True)
        else:
            st.info("Need at least 2 numeric features for correlation matrix.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Next: Feature Engineering Studio →", type="primary", use_container_width=True):
        next_step("eda")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Automated Feature Engineering Studio
# ══════════════════════════════════════════════════════════════════════════════
elif S["step"] == "feature_engineering":
    render_stage_header(4, "Feature Engineering & Transformation Lab", "Transparent transformation lab: preview planned transformations, engineer interaction features, and check feature importance.")

    df = S["df"]
    col_types = S["col_types"]

    fe_plan = generate_fe_plan(df, col_types)

    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #EC4899; margin-bottom: 20px;">
        <div style="font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 6px;">⚙️ Iris Feature Engineering Strategy Plan</div>
        <div style="color: #CBD5E1; font-size: 0.9rem;">
            <b>{fe_plan['total_planned_actions']}</b> planned actions will create approximately
            <b>+{fe_plan['estimated_new_columns']}</b> new columns from the existing features.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Per-column action preview table
    if fe_plan["items"]:
        fe_preview_rows = "".join([
            f"""<tr>
                <td style='padding:9px 14px; color:#E5E7EB; font-weight:600;'>{it['col']}</td>
                <td style='padding:9px 14px;'>
                    <span style='background:#8B5CF622; color:#A78BFA; border:1px solid #8B5CF655;
                                 border-radius:12px; padding:3px 10px; font-size:0.8rem; font-weight:700;'>
                        {it['type']}
                    </span>
                </td>
                <td style='padding:9px 14px; color:#06B6D4; font-weight:700; text-align:center;'>+{it['est_cols']}</td>
            </tr>"""
            for it in fe_plan["items"]
        ])
        st.markdown(f"""
        <div style='margin-bottom:24px; border:1px solid rgba(139,92,246,0.2); border-radius:16px; overflow:hidden;'>
            <table style='width:100%; border-collapse:collapse; font-size:0.88rem;'>
                <thead>
                    <tr style='background:rgba(139,92,246,0.15);'>
                        <th style='padding:10px 14px; color:#8B5CF6; text-align:left; font-weight:700;'>Column</th>
                        <th style='padding:10px 14px; color:#8B5CF6; text-align:left; font-weight:700;'>Planned Transformation</th>
                        <th style='padding:10px 14px; color:#8B5CF6; text-align:center; font-weight:700;'>New Cols</th>
                    </tr>
                </thead>
                <tbody>{fe_preview_rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    if st.button("⚡ Apply All Planned Feature Transformations (1-Click)", type="primary", key="btn_auto_fe_all", use_container_width=True):
        new_df, explanations, summary_msg = auto_engineer_all_features(df, col_types)
        S["df"] = new_df
        S["col_types"] = infer_all_types(new_df)
        S["fe_explanations"] = explanations
        S["transformation_log"].append(summary_msg)
        st.toast(summary_msg, icon="⚡")
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("➕ Interaction Feature Builder", expanded=False):
        num_cols = [c for c, t in col_types.items() if t == "numeric"]
        if len(num_cols) >= 2:
            i_col1, i_col2, i_op, i_btn = st.columns([2, 2, 2, 2])
            with i_col1:
                c1 = st.selectbox("Feature 1:", num_cols, key="int_c1")
            with i_col2:
                c2 = st.selectbox("Feature 2:", num_cols, key="int_c2")
            with i_op:
                op = st.selectbox("Operation:", ["product (Multiply)", "ratio (Divide)"], key="int_op")
            with i_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Create Interaction", key="btn_create_int"):
                    op_code = "product" if "product" in op else "ratio"
                    S["df"], msg = create_interaction_features(S["df"], c1, c2, op_code)
                    S["col_types"] = infer_all_types(S["df"])
                    st.toast(msg, icon="➕")
                    st.rerun()
        else:
            st.info("Need at least 2 numeric features for interaction builder.")

    if S.get("target_col") and S.get("target_col") in df.columns:
        fi_dict = quick_feature_importance(df, S["target_col"])
        if fi_dict:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"### 🎯 Feature Importance Preview (Target: '{S['target_col']}')")
            fi_df = pd.DataFrame(list(fi_dict.items()), columns=["Feature", "Importance Score"])
            fig = px.bar(fi_df, x="Importance Score", y="Feature", orientation="h", color="Importance Score", color_continuous_scale="Purples")
            fig.update_layout(plot_bgcolor="#0A0A14", paper_bgcolor="#0A0A14", font=dict(color="#F1F5F9"))
            st.plotly_chart(fig, use_container_width=True)


    # Explanations Section
    if S["fe_explanations"]:
        st.markdown(f"### 💡 Iris Feature Engineering Log ({len(S['fe_explanations'])} features created)")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:20px;'>Below is the plain-English explanation for every feature engineered by Iris:</div>", unsafe_allow_html=True)

        for exp in S["fe_explanations"]:
            c_orig = exp["column"]
            f_new = exp["new_feature"]
            badge = exp["badge"]
            b_color = exp.get("badge_color", "#8B5CF6")
            trans = exp["transformation"]
            why_text = exp["why"]

            st.markdown(f"""
            <div style="background:#151528; border:1px solid rgba(139,92,246,0.25); border-radius:16px; padding:20px; margin-bottom:20px;">
                <div style="display:flex; justify-space-between; align-items:center; margin-bottom:8px;">
                    <div>
                        <span class="type-badge" style="background:{b_color}22; color:{b_color}; border:1px solid {b_color}55;">{badge}</span>
                        <span style="font-size:1.1rem; font-weight:800; color:#F3F4F6; margin-left:8px;">{c_orig} &nbsp;→&nbsp; <span style="color:#06B6D4;">{f_new}</span></span>
                    </div>
                </div>
                <div style="color:#9CA3AF; font-size:0.83rem; margin-bottom:12px;">Method: <strong>{trans}</strong></div>
                <div class="why-card" style="margin-top:0;">
                    <strong>❓ Why did Iris create this feature?</strong><br>
                    {why_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Optional Advanced Custom Feature Studio
    with st.expander("🛠️ Advanced Custom Feature Transformation Studio (Optional)", expanded=False):
        fe_col1, fe_col2 = st.columns([1, 1])
        with fe_col1:
            selected_feature = st.selectbox("Select Column to Transform:", df.columns.tolist(), key="fe_col_select")
            trans_options = [
                ("log", "⚡ Log Transform (log1p) — Fix Skewness"),
                ("standard_scale", "📏 Standard Scaling (Z-Score: mean=0, std=1)"),
                ("minmax_scale", "📐 Min-Max Scaling ([0, 1] range)"),
                ("bin_quantiles", "📦 4-Quantile Binning (Low/Med/High)"),
                ("binary_encode", "🔢 Binary Encoding (0/1)"),
                ("one_hot", "🔥 One-Hot Encoding (Dummy columns)"),
                ("freq_encode", "🗂️ Frequency Encoding"),
                ("datetime_decompose", "📅 Datetime Decomposition"),
                ("drop", "🗑️ Drop Column"),
            ]
            trans_type = st.selectbox("Select Transformation:", [t[0] for t in trans_options], format_func=lambda x: [t[1] for t in trans_options if t[0]==x][0])

            if st.button("✨ Apply Custom Transformation", key="btn_apply_custom_fe", use_container_width=True):
                new_df, status_msg = apply_transformation(df, selected_feature, trans_type)
                S["df"] = new_df
                S["col_types"] = infer_all_types(new_df)
                S["transformation_log"].append(status_msg)
                st.toast(status_msg, icon="✨")
                st.rerun()

        with fe_col2:
            st.markdown("**Distribution Visualizer**")
            s_raw = df[selected_feature]
            if pd.api.types.is_numeric_dtype(s_raw):
                fig_prev = px.histogram(df, x=selected_feature, title=f"Feature: '{selected_feature}'", color_discrete_sequence=["#8B5CF6"])
                fig_prev.update_layout(plot_bgcolor="#0F0F20", paper_bgcolor="#151528", font=dict(color="#F3F4F6"), height=240)
                st.plotly_chart(fig_prev, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Next: Communication & Executive Export →", type="primary", use_container_width=True):
        next_step("feature_engineering")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Communication & Executive Reporting
# ══════════════════════════════════════════════════════════════════════════════
elif S["step"] == "communication":
    render_stage_header(5, "Communication & Executive Debrief", "Executive summary view: plain-English AI narrative, project pipeline metrics, interactive HTML report, and cleaned dataset exports.")

    df = S["df"]
    health = compute_data_health_score(df)
    
    # Project Summary Dashboard KPI Grid
    st.markdown("### 📊 Project Pipeline Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card("Total Rows", f"{len(df):,}", "Final Dataset Size", "#7C3AED")
    with c2:
        render_kpi_card("Total Columns", str(len(df.columns)), f"{len(S.get('col_types', {}))} Inferenced", "#06B6D4")
    with c3:
        render_kpi_card("Health Index", f"{health['score']} / 100", "Post-Cleaning", "#10B981")
    with c4:
        render_kpi_card("Actions Executed", str(len(S.get("cleaning_log", [])) + len(S.get("transformation_log", []))), "Cleaning & FE", "#F59E0B")
    with c5:
        render_kpi_card("Target Feature", str(S.get("target_col", "None")), "Primary Metric", "#EC4899")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Dataset Preview (Cleaned & Engineered)")
    render_data_preview_table(df, key_prefix="stage5")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # AI Executive Summary Panel
    pipeline_stats = {
        "domain": S.get("domain", "General"),
        "business_goal": S.get("business_goal", ""),
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "health_score": health["score"],
        "target_col": S.get("target_col", "None"),
        "cleaning_count": len(S.get("cleaning_log", [])),
        "fe_count": len(S.get("transformation_log", []))
    }
    
    exec_summary_text = generate_executive_summary(pipeline_stats)
    with st.expander("📄 AI Executive Narrative Briefing", expanded=True):
        st.markdown(exec_summary_text)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ML Model Recommendation Engine ──────────────────────────────────────────
    st.markdown("### 🤖 ML Model Recommendation Engine")
    st.markdown(
        "<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:20px;'>"
        "Based on your dataset's statistical properties and target column selection, "
        "Iris recommends the following Machine Learning models for future analysis. "
        "No training is performed — this is a strategic advisory for your next steps."
        "</div>",
        unsafe_allow_html=True
    )

    ml_recs = recommend_ml_models(
        df, S["col_types"], S["col_stats"] or {},
        S["quality"] or {}, target_col=S.get("target_col")
    )

    # Task type banner
    task_color_map = {
        "Supervised Classification": ("#10B981", "#10B98120"),
        "Supervised Regression":     ("#06B6D4", "#06B6D420"),
        "Unsupervised Learning":      ("#F59E0B", "#F59E0B20"),
    }
    task_label = ml_recs["task_type"]
    t_color, t_bg = next(
        ((c, b) for k, (c, b) in task_color_map.items() if k in task_label),
        ("#8B5CF6", "#8B5CF620")
    )
    target_badge = ""
    if ml_recs["target_info"]:
        ti = ml_recs["target_info"]
        target_badge = f"&nbsp;&nbsp;|&nbsp;&nbsp;Target: <b style='color:{t_color};'>{ti.get('name', '')}</b>"

    st.markdown(f"""
    <div style="background:{t_bg}; border:1px solid {t_color}44; border-left:4px solid {t_color};
                border-radius:14px; padding:16px 22px; margin-bottom:24px; display:flex; align-items:center; gap:12px;">
        <span style="font-size:1.6rem;">{'📊' if 'Regression' in task_label else ('🔬' if 'Classification' in task_label else '🔍')}</span>
        <div>
            <div style="font-size:0.75rem; color:{t_color}; font-weight:800; letter-spacing:0.08em; text-transform:uppercase;">Detected Task Type</div>
            <div style="font-size:1.05rem; font-weight:700; color:#F3F4F6; margin-top:2px;">{task_label}{target_badge}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Data Notices
    if ml_recs["data_notices"]:
        for notice in ml_recs["data_notices"]:
            bg = "rgba(245,158,11,0.1)" if "Warning" in notice or "Imbalance" in notice else "rgba(16,185,129,0.08)"
            border = "#F59E0B" if "Warning" in notice or "Imbalance" in notice else "#10B981"
            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {border}33; border-left:3px solid {border};
                        border-radius:10px; padding:11px 18px; margin-bottom:10px; font-size:0.88rem; color:#E5E7EB;">
                {notice}
            </div>""", unsafe_allow_html=True)

    # Model cards grid
    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
    model_badge_colors = {
        "Primary Recommended": ("#10B981", "#10B98118"),
        "Highly Recommended":  ("#06B6D4", "#06B6D418"),
        "Benchmark Baseline":  ("#8B5CF6", "#8B5CF618"),
        "Interpretable Baseline": ("#8B5CF6", "#8B5CF618"),
        "Density Discovery":   ("#F59E0B", "#F59E0B18"),
        "Specialized Choice":  ("#EC4899", "#EC489918"),
        "Feature Space Reduction": ("#3B82F6", "#3B82F618"),
        "Outlier":             ("#F97316", "#F9731618"),
    }

    for model in ml_recs["recommendations"]:
        suit_label = model["suitability"]
        m_color, m_bg = next(
            ((c, b) for k, (c, b) in model_badge_colors.items() if k in suit_label),
            ("#8B5CF6", "#8B5CF618")
        )
        interp_color = "#10B981" if model["interpretability"].startswith("High") else (
            "#F59E0B" if model["interpretability"].startswith("Medium") else "#9CA3AF"
        )
        st.markdown(f"""
        <div style="background:#151528; border:1px solid rgba(139,92,246,0.2); border-radius:18px;
                    padding:22px 26px; margin-bottom:18px; transition:transform 0.2s;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px; margin-bottom:14px;">
                <div>
                    <span style="background:{m_bg}; color:{m_color}; border:1px solid {m_color}55;
                                 border-radius:20px; padding:4px 12px; font-size:0.75rem; font-weight:800;
                                 letter-spacing:0.04em;">{suit_label}</span>
                    <h3 style="color:#F3F4F6; font-size:1.1rem; font-weight:800; margin:8px 0 2px;">{'⚡' if 'Primary' in suit_label else ('✅' if 'Baseline' in suit_label else '🔷')} {model['name']}</h3>
                    <span style="color:#9CA3AF; font-size:0.8rem; font-weight:600;">{model['category']}</span>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.72rem; color:#6B7280; text-transform:uppercase; letter-spacing:0.07em;">Interpretability</div>
                    <div style="color:{interp_color}; font-weight:700; font-size:0.9rem;">{model['interpretability']}</div>
                    <div style="font-size:0.72rem; color:#6B7280; margin-top:4px;">Complexity: {model['complexity']}</div>
                </div>
            </div>
            <div style="background:rgba(139,92,246,0.06); border-left:3px solid {m_color}88;
                        border-radius:8px; padding:12px 16px; margin-bottom:10px; color:#CBD5E1; font-size:0.87rem; line-height:1.6;">
                <strong style='color:#E5E7EB;'>Why this model?</strong><br>{model['why']}
            </div>
            <div style="color:#9CA3AF; font-size:0.82rem;">
                Best for: <em style='color:#D1D5DB;'>{model['best_for']}</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Evaluation Metrics
    if ml_recs["eval_metrics"]:
        metrics_html = "".join([
            f"<span style='background:rgba(139,92,246,0.12); color:#A78BFA; border:1px solid rgba(139,92,246,0.3);"
            f" border-radius:20px; padding:5px 14px; font-size:0.82rem; font-weight:600; margin:4px; display:inline-block;'>{m}</span>"
            for m in ml_recs["eval_metrics"]
        ])
        st.markdown(f"""
        <div style="margin-top:8px; margin-bottom:32px;">
            <div style="font-size:0.8rem; font-weight:700; color:#9CA3AF; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:10px;">Recommended Evaluation Metrics</div>
            {metrics_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📦 Export Reports & Datasets")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("""
        <div class="feature-card" style="text-align:center; padding:24px 16px;">
            <div style="font-size:1.1rem; font-weight:700; color:#8B5CF6; margin-bottom:6px;">Interactive HTML Dashboard</div>
            <div style="color:#9CA3AF; font-size:0.82rem; margin-bottom:16px;">
                Single-file responsive web app with embedded Plotly charts, data quality audit, correlations & feature engineering guide.
            </div>
        </div>
        """, unsafe_allow_html=True)
        try:
            html_str = build_html_dashboard(
                S["meta"], S["col_types"], S["quality"] or {},
                S["col_stats"] or {}, S["corr"] or {},
                {}, S["iris_says"], S["df"]
            )
            st.download_button("Download Full HTML Dashboard", html_str, "iris_dashboard.html", "text/html",
                               use_container_width=True, type="primary", key="comm_dl_html")
        except Exception as e:
            st.error(f"Error generating HTML: {e}")

    with d2:
        st.markdown("""
        <div class="feature-card" style="text-align:center; padding:24px 16px;">
            <div style="font-size:1.1rem; font-weight:700; color:#06B6D4; margin-bottom:6px;">Dark Executive PDF Report</div>
            <div style="color:#9CA3AF; font-size:0.82rem; margin-bottom:16px;">
                High-contrast printable PDF report with executive cover page, KPI metrics breakdown & AI narrative explanations.
            </div>
        </div>
        """, unsafe_allow_html=True)
        try:
            pdf_bytes = build_pdf_report(
                S["meta"], S["col_types"], S["quality"] or {},
                S["col_stats"] or {}, S["corr"] or {},
                {}, S["iris_says"]
            )
            st.download_button("Download PDF Report", pdf_bytes, "iris_report.pdf", "application/pdf",
                               use_container_width=True, type="primary", key="comm_dl_pdf")
        except Exception as e:
            st.error(f"Error generating PDF: {e}")

    with d3:
        st.markdown("""
        <div class="feature-card" style="text-align:center; padding:24px 16px;">
            <div style="font-size:1.1rem; font-weight:700; color:#EC4899; margin-bottom:6px;">Cleaned & Engineered CSV</div>
            <div style="color:#9CA3AF; font-size:0.82rem; margin-bottom:16px;">
                Preprocessed dataset with imputed missing values, capped outliers, and newly engineered feature columns.
            </div>
        </div>
        """, unsafe_allow_html=True)
        try:
            csv_bytes = build_cleaned_csv(S["df"])
            st.download_button("Download Cleaned CSV", csv_bytes, "iris_cleaned.csv", "text/csv",
                               use_container_width=True, type="primary", key="comm_dl_csv")

        except Exception as e:
            st.error(f"Error generating CSV: {e}")

    st.markdown("---")

    # Ask Iris Chat
    st.markdown("### 💬 Ask Iris AI Anything")
    st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Ask questions about your data analysis, business findings, or feature engineering strategy.</div>", unsafe_allow_html=True)

    for msg in S["chat_history"]:
        role_icon = "🌸" if msg["role"] == "iris" else "👤"
        align = "left" if msg["role"] == "iris" else "right"
        bg = "rgba(139,92,246,0.12)" if msg["role"] == "iris" else "rgba(6,182,212,0.12)"
        border = "#8B5CF6" if msg["role"] == "iris" else "#06B6D4"
        st.markdown(f"""
        <div style="text-align:{align}; margin:8px 0;">
            <div style="display:inline-block; background:{bg}; border:1px solid {border};
                        border-radius:12px; padding:12px 16px; max-width:75%;
                        text-align:left; color:#E2E8F0; font-size:0.9rem; line-height:1.6;">
                {role_icon} {msg['text']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_q = st.text_input("Your question:", placeholder="Which features have the strongest relationship? How does log transform help?", label_visibility="collapsed")
        with btn_col:
            send = st.form_submit_button("Ask 🌸", use_container_width=True)

    if send and user_q:
        S["chat_history"].append({"role": "user", "text": user_q})
        with st.spinner("Iris is thinking..."):
            answer = chat_with_iris(user_q, S["dataset_context"])
        S["chat_history"].append({"role": "iris", "text": answer})
        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🔄 Start New Data Science Project", use_container_width=False):
        for key in list(S.keys()):
            del st.session_state[key]
        st.rerun()
