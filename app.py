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
import hashlib
import json
import os
import plotly.express as px
import plotly.graph_objects as go

from engine.parser import get_excel_sheet_names, load_csv, load_excel, load_google_sheet
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
    render_data_preview_table, render_styled_dataframe, render_section_header, ICONS,
    scroll_to_top
)



import base64

def get_logo_base64():
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.jpg")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

LOGO_B64 = get_logo_base64()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Data Science Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Enterprise Design System ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap');

/* ── DESIGN TOKENS ── */
:root {
    --bg-base:      #FAFAFC;      /* Executive Ivory / Soft Cream White */
    --bg-surface:   #FFFFFF;      /* Pure Crisp White Cards */
    --bg-elevated:  #F8FAFC;      /* Soft Light Slate/Cream for secondary items */
    --bg-hover:     #F1F5F9;      /* Light Slate Hover */
    --border-dim:   #E2E8F0;      /* Soft subtle border */
    --border-soft:  rgba(124,58,237,0.22);
    --border-glow:  rgba(124,58,237,0.40);
    --violet-600:   #7C3AED;
    --violet-500:   #8B5CF6;
    --violet-400:   #6D28D9;      /* High contrast violet for light background */
    --violet-glow:  rgba(124,58,237,0.12);
    --cyan-500:     #0891B2;
    --cyan-400:     #0284C7;
    --cyan-glow:    rgba(6,182,212,0.10);
    --green-500:    #059669;
    --amber-500:    #D97706;
    --red-500:      #DC2626;
    --pink-500:     #DB2777;
    --text-primary: #0F172A;      /* Deep Slate Charcoal for main text */
    --text-secondary:#334155;      /* Dark Slate for body text */
    --text-muted:   #64748B;      /* Slate Grey for muted subtext */
    --radius-sm:    10px;
    --radius-md:    14px;
    --radius-lg:    20px;
    --radius-xl:    24px;
    --shadow-sm:    0 2px 8px rgba(0,0,0,0.03);
    --shadow-md:    0 6px 20px rgba(0,0,0,0.05);
    --shadow-lg:    0 12px 36px rgba(0,0,0,0.07);
    --shadow-glow:  0 0 30px rgba(124,58,237,0.08);
}

/* ── BASE ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg-base) !important;
    background-color: #FFFFFF !important;
    background-image: linear-gradient(rgba(124,58,237,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(124,58,237,0.035) 1px, transparent 1px) !important;
    background-size: 36px 36px !important;
    color: var(--text-primary) !important;
    font-size: 15px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* Keep legacy inline copy readable on the white canvas without changing the dark sidebar. */
section[data-testid="stMain"] *[style*="color:#9CA3AF"],
section[data-testid="stMain"] *[style*="color: #9CA3AF"],
section[data-testid="stMain"] *[style*="rgb(156, 163, 175)"] {
    color: #64748B !important;
}
section[data-testid="stMain"] *[style*="color:#8B5CF6"],
section[data-testid="stMain"] *[style*="color:#A78BFA"],
section[data-testid="stMain"] *[style*="rgb(139, 92, 246)"],
section[data-testid="stMain"] *[style*="rgb(167, 139, 250)"] {
    color: #6D28D9 !important;
}
section[data-testid="stMain"] *[style*="color:#06B6D4"],
section[data-testid="stMain"] *[style*="color:#67E8F9"],
section[data-testid="stMain"] *[style*="color:#22D3EE"],
section[data-testid="stMain"] *[style*="rgb(6, 182, 212)"],
section[data-testid="stMain"] *[style*="rgb(103, 232, 249)"],
section[data-testid="stMain"] *[style*="rgb(34, 211, 238)"] {
    color: #0E7490 !important;
}
section[data-testid="stMain"] *[style*="color:#CBD5E1"],
section[data-testid="stMain"] *[style*="color:#D1D5DB"],
section[data-testid="stMain"] *[style*="color:#E2E8F0"],
section[data-testid="stMain"] *[style*="color:#E5E7EB"],
section[data-testid="stMain"] *[style*="color:#F3F4F6"],
section[data-testid="stMain"] *[style*="color:#F8FAFC"],
section[data-testid="stMain"] *[style*="rgb(203, 213, 225)"],
section[data-testid="stMain"] *[style*="rgb(209, 213, 219)"],
section[data-testid="stMain"] *[style*="rgb(226, 232, 240)"],
section[data-testid="stMain"] *[style*="rgb(229, 231, 235)"],
section[data-testid="stMain"] *[style*="rgb(243, 244, 246)"],
section[data-testid="stMain"] *[style*="rgb(248, 250, 252)"] {
    color: #334155 !important;
}

p, div, span, li { font-family: 'Inter', sans-serif; }

/* ── SIDEBAR (Dark Navy Theme) ── */
section[data-testid="stSidebar"] {
    background: #0B0F19 !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
}

/* Sidebar brand logo area */
.iris-brand {
    padding: 24px 16px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 8px;
    text-align: center;
}
.iris-brand-icon {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 6px;
}
.iris-brand-name {
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #A78BFA !important;
}
.iris-brand-tagline {
    font-size: 0.7rem;
    color: #94A3B8 !important;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Sidebar step stepper */
.sidebar-section-label {
    font-size: 0.65rem;
    color: #94A3B8 !important;
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
    color: #CBD5E1 !important;
    border: 1px solid rgba(255,255,255,0.04);
    background: rgba(255,255,255,0.03);
    text-decoration: none;
}
.sidebar-step-done {
    color: #94A3B8 !important;
}
.sidebar-step-done .step-dot {
    background: #10B981;
    box-shadow: 0 0 10px rgba(16,185,129,0.4);
}
.sidebar-step-active {
    background: rgba(124,58,237,0.2) !important;
    border-color: rgba(139,92,246,0.5) !important;
    color: #A78BFA !important;
    font-weight: 600;
}
.sidebar-step-active .step-dot {
    background: #8B5CF6;
    box-shadow: 0 0 12px rgba(124,58,237,0.6);
    animation: pulse-dot 2s ease-in-out infinite;
}
.sidebar-step-locked {
    opacity: 0.4;
}
.sidebar-step-locked .step-dot {
    background: #64748B;
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
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-top: 3px solid var(--violet-600) !important;
    border-radius: var(--radius-md) !important;
    padding: 18px 20px !important;
    box-shadow: var(--shadow-md) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg) !important;
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

/* ── GLASS & ELEVATION CARDS ── */
.glass-card {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md) !important;
    padding: 24px 28px;
    margin-bottom: 20px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    border-color: rgba(124,58,237,0.35) !important;
    box-shadow: var(--shadow-lg) !important;
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
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.rec-card:hover {
    border-color: var(--border-soft) !important;
    transform: translateX(3px);
}
.rec-card-critical { border-left: 3px solid var(--red-500) !important; }
.rec-card-high     { border-left: 3px solid var(--amber-500) !important; }
.rec-card-medium   { border-left: 3px solid var(--cyan-500) !important; }
.rec-card-low      { border-left: 3px solid var(--text-muted) !important; }
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
    color: var(--text-primary) !important;
    margin-bottom: 4px;
}
.rec-reason {
    font-size: 0.8rem;
    color: var(--text-secondary) !important;
    line-height: 1.5;
}

/* ── STAGE HEADER SYSTEM ── */
.stage-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,58,237,0.1);
    color: #6D28D9 !important;
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
    color: var(--text-primary) !important;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 8px;
}
.stage-sub {
    color: var(--text-secondary) !important;
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
    background: rgba(124,58,237,0.05) !important;
    border: 1px solid rgba(124,58,237,0.2) !important;
    border-left: 4px solid var(--violet-600) !important;
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
    background: rgba(124,58,237,0.3);
}
.iris-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: rgba(124,58,237,0.15) !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.iris-insight-body {
    flex: 1;
}
.iris-insight-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6D28D9 !important;
    margin-bottom: 6px;
}
.iris-insight-text {
    font-size: 0.88rem;
    color: #1E293B !important;
    line-height: 1.75;
}

/* Why card (EDA explanations) */
.why-card {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-left: 3px solid #0891B2 !important;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 14px 18px;
    margin-top: 14px;
    color: #334155 !important;
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
    background: #E2E8F0;
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
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: var(--radius-lg);
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-md);
    transition: all 0.2s ease;
}

.feature-card {
    background: #FFFFFF !important;
    border: 1px solid #DDD6FE !important;
    border-radius: 10px;
    padding: 17px 18px;
    transition: all 0.2s ease;
    height: 100%;
    min-height: 128px;
    box-shadow: 0 3px 12px rgba(76,29,149,0.05);
}
.feature-card:hover {
    border-color: rgba(124,58,237,0.6) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(76,29,149,0.12);
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
    background: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
}
div.stButton > button:hover {
    background: #F8FAFC !important;
    color: #6D28D9 !important;
    border-color: #8B5CF6 !important;
}

/* Primary CTA — flat color */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid*="primary"] {
    background: #7C3AED !important;
    color: #FFFFFF !important;
    border: 1px solid #6D28D9 !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.2) !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #6D28D9 !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.3) !important;
    transform: translateY(-1px) !important;
}

/* Download buttons */
div[data-testid="stDownloadButton"] > button {
    background: rgba(8,145,178,0.08) !important;
    border: 1px solid rgba(8,145,178,0.35) !important;
    color: #0891B2 !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(8,145,178,0.16) !important;
    border-color: #0891B2 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(8,145,178,0.2) !important;
}

/* ── TABS ── */
div[data-testid="stTabs"] > div > div > button {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
    border-bottom: 2px solid transparent !important;
}
div[data-testid="stTabs"] > div > div > button[aria-selected="true"] {
    color: #7C3AED !important;
    border-bottom: 2px solid #7C3AED !important;
    background: rgba(124,58,237,0.06) !important;
    font-weight: 700 !important;
}
div[data-testid="stTabs"] > div > div > button:hover {
    color: #0F172A !important;
    background: #F1F5F9 !important;
}

/* ── INPUTS & SELECTS & TEXTAREAS ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] > div > div > input,
div[data-testid="stTextArea"] > div > div > textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea, input, select {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: var(--radius-sm) !important;
    color: #0F172A !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stTextInput"] > div > div > input:focus,
div[data-testid="stTextArea"] > div > div > textarea:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}

/* Input & Radio & Selectbox Labels */
label, .stRadio label, .stSelectbox label, .stTextInput label, .stTextArea label, .stMultiSelect label {
    color: #1E293B !important;
    font-weight: 600 !important;
}

/* ── SUCCESS / WARNING / INFO ALERTS ── */
div[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-size: 0.86rem !important;
    border: 1px solid !important;
}

/* ── EXPANDERS ── */
div[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: var(--radius-md) !important;
    background: #FFFFFF !important;
}
div[data-testid="stExpander"] details summary {
    color: #0F172A !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    background: #F5F1FF !important;
    border-left: 3px solid #7C3AED !important;
    border-radius: var(--radius-md) !important;
}
div[data-testid="stExpander"] details summary * {
    color: #0F172A !important;
}
div[data-testid="stExpander"] details summary:hover {
    background: #F1F5F9 !important;
    color: #6D28D9 !important;
}

/* ── DATAFRAMES ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #DDD6FE !important;
    border-radius: 10px !important;
    overflow: hidden;
    box-shadow: 0 3px 12px rgba(76,29,149,0.06) !important;
    background: #FFFFFF !important;
}
div[data-testid="stDataFrame"] [role="columnheader"] {
    font-weight: 800 !important;
    color: #4C1D95 !important;
    background: #F3E8FF !important;
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
        "dataset_fingerprint": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
S = st.session_state

if S["df"] is not None and not S["df"].columns.is_unique:
    duplicate_columns = S["df"].columns[S["df"].columns.duplicated()].unique().tolist()
    S["df"] = S["df"].loc[:, ~S["df"].columns.duplicated()].copy()
    S["col_types"] = infer_all_types(S["df"])
    S["cols_by_type"] = get_columns_by_type(S["df"], S["col_types"])
    S["quality"] = None
    S["col_stats"] = None
    S["corr"] = None
    S["cleaning_log"].append(
        f"Removed duplicate generated columns from the active dataset: {', '.join(map(str, duplicate_columns))}."
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
STEP_META = [
    ("business_collection", "", "1. Business & Data Collection"),
    ("data_cleaning",        "", "2. Data Cleaning & Preprocessing"),
    ("eda",                 "", "3. Exploratory Data Analysis (EDA)"),
    ("feature_engineering", "", "4. Feature Engineering"),
    ("communication",       "", "5. Communication & Export"),
]

with st.sidebar:
    # ── Brand Header with Uploaded Logo ────────────────────────────────────
    logo_img_html = f'<img src="data:image/jpeg;base64,{LOGO_B64}" style="width:95px; height:95px; object-fit:cover; border-radius:50%; border:2px solid rgba(139,92,246,0.5); box-shadow:0 0 20px rgba(124,58,237,0.3); margin-bottom:8px;" />' if LOGO_B64 else '<div style="font-size:2rem; font-weight:900; color:#8B5CF6;">IRIS</div>'
    st.markdown(f'<div class="iris-brand" style="text-align:center; padding:18px 12px 14px; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:12px;">{logo_img_html}<div class="iris-brand-name" style="font-size:1.1rem; font-weight:800; color:#A78BFA; letter-spacing:-0.02em;">Iris Studio</div><div class="iris-brand-tagline" style="font-size:0.73rem; color:#94A3B8; font-weight:500;">Data Science Lifecycle</div></div>', unsafe_allow_html=True)

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
            chk_icon = ICONS.get("check", "✓")
            st.markdown(f'<div class="sidebar-step-item sidebar-step-done"><div class="step-dot" style="background:#10B981; box-shadow:0 0 8px rgba(16,185,129,0.5);"></div><div style="flex:1; font-size:0.82rem; font-weight:500; color:#94A3B8;"><span style="font-size:0.65rem; font-weight:700; color:#475569; letter-spacing:0.08em;">STEP {num}</span><br/>{label}</div><span style="display:flex; align-items:center; color:#10B981;">{chk_icon}</span></div>', unsafe_allow_html=True)
            # Invisible button for navigation
            if st.button(f"Go to {label}", key=f"side_nav_{sid}", use_container_width=True, help=f"Return to {label}"):
                S["step"] = sid
                S["_scroll_to_stage_top"] = True
                st.rerun()
        elif i == step_idx:
            # Active
            arr_icon = ICONS.get("arrow_r", "▶")
            st.markdown(f'<div class="sidebar-step-item sidebar-step-active"><div class="step-dot"></div><div style="flex:1;"><span style="font-size:0.65rem; font-weight:700; color:#A78BFA; letter-spacing:0.08em;">STEP {num} — ACTIVE</span><br/>{label}</div><span style="display:flex; align-items:center; color:#A78BFA;">{arr_icon}</span></div>', unsafe_allow_html=True)
        else:
            # Locked
            lck_icon = ICONS.get("lock", "🔒")
            st.markdown(f'<div class="sidebar-step-item sidebar-step-locked"><div class="step-dot"></div><div style="flex:1;"><span style="font-size:0.65rem; font-weight:700; letter-spacing:0.08em;">STEP {num}</span><br/>{label}</div><span style="display:flex; align-items:center; opacity:0.4;">{lck_icon}</span></div>', unsafe_allow_html=True)

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
        iris_icon = ICONS.get("iris", "")
        st.markdown(f'<div class="iris-insight"><div class="iris-avatar" style="display:flex; align-items:center; justify-content:center;">{iris_icon}</div><div class="iris-insight-body"><div class="iris-insight-label">Iris AI Analysis</div><div class="iris-insight-text">{text}</div></div></div>', unsafe_allow_html=True)


def next_step(current: str):
    idx = STEPS.index(current)
    if idx + 1 < len(STEPS):
        S["step"] = STEPS[idx + 1]
        S["_scroll_to_stage_top"] = True
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Business Understanding & Data Collection
# ══════════════════════════════════════════════════════════════════════════════
if S["step"] == "business_collection":
    if S.pop("_scroll_to_stage_top", False):
        scroll_to_top()
    hero_logo_html = f'<img src="data:image/jpeg;base64,{LOGO_B64}" style="width:105px; height:105px; object-fit:cover; border-radius:50%; border:3px solid rgba(139,92,246,0.5); box-shadow:0 0 25px rgba(124,58,237,0.25); margin-bottom:14px;" />' if LOGO_B64 else ""
    st.markdown(f"""
    <div style='text-align:center; padding: 24px 0 16px; position:relative;'>
        <div style='margin-bottom:12px;'>{hero_logo_html}</div>
        <h1 style='
            font-family: Inter, sans-serif;
            font-size:2.8rem; font-weight:900; letter-spacing:-0.04em; margin:0 0 12px;
            color:#0F172A; line-height:1.1;'>
            Iris Data Science Studio
        </h1>
        <p style='color:#475569; font-size:1rem; max-width:620px; margin:0 auto 8px; line-height:1.75; font-weight:500;'>
            Select your domain objective, ingest your dataset, and let Iris automate data quality checks, narrative EDA, and feature engineering.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("Stage 1A: Domain & Business Objective Setup", expanded=True):
        st.markdown("<div style='color:#64748B; font-size:0.8rem; font-weight:800; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:12px;'>Select Industry Domain</div>", unsafe_allow_html=True)
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
                    
        st.markdown("<div style='margin-top:20px; color:#64748B; font-size:0.8rem; font-weight:800; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:12px;'>Select Objective Template</div>", unsafe_allow_html=True)
        selected_obj = S.get("objective", "explore")
        objs = [
            ("predict", "Predict Outcomes"),
            ("segment", "Segment Data"),
            ("detect", "Detect Anomalies"),
            ("explore", "Explore & Report")
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
        <div class='stage-objective-summary' style='display:flex; align-items:center; gap:12px; padding:12px 18px; margin:20px 0 16px; box-shadow:0 2px 8px rgba(124,58,237,0.06);'>
            <div style='font-size:0.75rem; font-weight:900; color:#6D28D9; text-transform:uppercase; letter-spacing:0.08em;'>Active Setup:</div>
            <div style='font-size:0.9rem; font-weight:700; color:#0F172A;'>
                Domain: <span style='color:#7C3AED;'>{cur_dom_name}</span> &nbsp;|&nbsp; Objective: <span style='color:#059669;'>{cur_obj_label}</span>
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
    render_section_header("Stage 1B: Data Collection & Profiling", "upload")
    
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        uploaded = st.file_uploader(
            "Upload CSV or Excel workbook",
            type=["csv", "txt", "xlsx", "xls"],
            label_visibility="visible",
        )
        sheet_url = st.text_input(
            "Google Sheets link",
            placeholder="https://docs.google.com/spreadsheets/d/...",
            help="The sheet must be shared as Anyone with the link can view.",
            key="google_sheet_url",
        )
        load_sheet_clicked = st.button(
            "Load Google Sheet",
            use_container_width=True,
            key="btn_load_google_sheet",
        )
        st.markdown("<div style='text-align:center; color:#7C3AED; font-size:0.85rem; margin:16px 0 8px; font-weight:800; letter-spacing:0.05em;'>— OR LOAD DEMO DATASET —</div>", unsafe_allow_html=True)
        
        d1, d2, d3 = st.columns(3)
        sample_bytes = None
        
        with d1:
            if st.button("Titanic Survival", use_container_width=True, key="demo_titanic"):
                path = os.path.join(os.path.dirname(__file__), "samples", "titanic.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()
            if st.button("Heart Disease", use_container_width=True, key="demo_heart"):
                path = os.path.join(os.path.dirname(__file__), "samples", "heart_disease.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()
                        
        with d2:
            if st.button("Housing Prices", use_container_width=True, key="demo_housing"):
                path = os.path.join(os.path.dirname(__file__), "samples", "housing.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()
            if st.button("Customer Churn", use_container_width=True, key="demo_churn"):
                path = os.path.join(os.path.dirname(__file__), "samples", "customer_churn.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()
                        
        with d3:
            if st.button("Iris Flowers", use_container_width=True, key="demo_iris"):
                path = os.path.join(os.path.dirname(__file__), "samples", "iris.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()
            if st.button("E-Commerce Sales", use_container_width=True, key="demo_ecom"):
                path = os.path.join(os.path.dirname(__file__), "samples", "ecommerce_sales.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        sample_bytes = f.read()

        file_bytes = uploaded.getvalue() if uploaded else sample_bytes
        if load_sheet_clicked:
            if not sheet_url.strip():
                st.warning("Paste a Google Sheets link first.")
            else:
                try:
                    with st.spinner("Loading Google Sheet..."):
                        df, meta = load_google_sheet(sheet_url)
                    dataset_fingerprint = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()
                    if dataset_fingerprint != S.get("dataset_fingerprint"):
                        S["cleaning_log"] = []
                        S["dataset_fingerprint"] = dataset_fingerprint
                    S["df"] = df
                    S["meta"] = meta
                    S["col_types"] = infer_all_types(df)
                    S["cols_by_type"] = get_columns_by_type(df, S["col_types"])
                    S["target_col"] = suggest_target_column(df, S.get("domain", "general"))
                    for key in ["quality", "col_stats", "corr", "iris_says"]:
                        S[key] = {} if key == "iris_says" else None
                    st.success("Google Sheet loaded successfully.")
                except Exception as error:
                    st.error(f"Could not load that Google Sheet. Check the URL and sharing access. Details: {error}")

        if file_bytes and not load_sheet_clicked:
            dataset_fingerprint = hashlib.sha256(file_bytes).hexdigest()
            if dataset_fingerprint != S.get("dataset_fingerprint"):
                S["cleaning_log"] = []
                S["dataset_fingerprint"] = dataset_fingerprint
            file_name = uploaded.name.lower() if uploaded else ""
            try:
                with st.spinner("Parsing dataset... 🌸"):
                    if file_name.endswith((".xlsx", ".xls")):
                        sheet_names = get_excel_sheet_names(file_bytes)
                        selected_sheet = st.selectbox(
                            "Worksheet",
                            sheet_names,
                            key=f"excel_sheet_{uploaded.name}",
                        )
                        df, meta = load_excel(file_bytes, selected_sheet)
                    else:
                        df, meta = load_csv(file_bytes)
                    col_types = infer_all_types(df)
                    cols_by_type = get_columns_by_type(df, col_types)
                S["df"] = df
                S["meta"] = meta
                S["col_types"] = col_types
                S["cols_by_type"] = cols_by_type
                S["target_col"] = suggest_target_column(df, S.get("domain", "general"))
                for key in ["quality", "col_stats", "corr", "iris_says"]:
                    S[key] = {} if key == "iris_says" else None
            except Exception as error:
                st.error(f"Could not read this file. Check that it is a supported CSV or Excel workbook. Details: {error}")

    if S["df"] is not None:
        if st.button("Next: Data Cleaning & Preprocessing Studio →", type="primary", key="btn_business_next_top", use_container_width=True):
            next_step("business_collection")

        df = S["df"]
        meta = S["meta"]
        col_types = S["col_types"]
        health = compute_data_health_score(df)

        st.markdown("<br>", unsafe_allow_html=True)
        render_section_header("Dataset Health & Profiling Snapshot", "health", badge=f"Health: {health['score']}/100", badge_color="#10B981" if health['score'] >= 80 else "#F59E0B")
        
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            render_kpi_card("Total Rows", f"{len(df):,}", f"{meta['encoding']} encoding", "#7C3AED")
        with p2:
            render_kpi_card("Total Columns", str(len(df.columns)), f"{health['missing_cols']} missing cols", "#0891B2")
        with p3:
            render_kpi_card("Health Score", f"{health['score']} / 100", "Data Quality Index", "#10B981" if health['score'] >= 80 else "#F59E0B")
        with p4:
            s_target = S.get("target_col", "None")
            render_kpi_card("Target Feature", str(s_target), "Suggested Target", "#7C3AED")

        st.markdown("<br>", unsafe_allow_html=True)
        render_data_preview_table(df, key_prefix="stage1")

        type_summary = {}
        for t in set(col_types.values()):
            type_summary[t] = [c for c, ct in col_types.items() if ct == t]

        st.markdown("**Inferred Feature Types:**", unsafe_allow_html=True)
        chips = ""
        type_colors = {
            "numeric": "#7C3AED", "categorical": "#0891B2", "datetime": "#D97706",
            "text": "#DB2777", "boolean": "#2563EB", "id": "#64748B"
        }
        for t, cols in type_summary.items():
            color = type_colors.get(t, "#7C3AED")
            chips += f'<span class="type-badge" style="background:{color}18; color:{color}; border:1px solid {color}40; font-weight:700;">{t} ({len(cols)})</span> '
        st.markdown(chips, unsafe_allow_html=True)

        render_iris_says(f"Dataset successfully loaded! Health index is {health['score']}/100. Target candidate column suggested: '{S.get('target_col')}'.", "Iris Data Profiler")

        st.markdown("<br>", unsafe_allow_html=True)
        render_section_header("Interactive Feature Directory (Click Any Card to Explore Column)", "explore")
        st.markdown("<div style='color:#475569; font-size:0.88rem; margin-bottom:18px; font-weight:500;'>Click 'Explore Column' on any feature card below to navigate directly to its detailed distribution, box plot, and statistical breakdown.</div>", unsafe_allow_html=True)
        
        f_cols = st.columns(3)
        for idx, col_name in enumerate(df.columns):
            ctype = col_types.get(col_name, "unknown")
            null_cnt = int(df[col_name].isnull().sum())
            null_pct = round((null_cnt / len(df)) * 100, 1) if len(df) > 0 else 0
            uniq_cnt = int(df[col_name].nunique())
            color = type_colors.get(ctype, "#7C3AED")
            
            with f_cols[idx % 3]:
                st.markdown(f'''
                <div class="feature-card" style="border-top:3px solid {color} !important; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-weight:800; font-size:1.0rem; color:#0F172A;">{col_name}</span>
                        <span class="type-badge" style="background:{color}1A; color:{color}; border:1px solid {color}40; font-size:0.7rem; font-weight:700;">{ctype}</span>
                    </div>
                    <div style="font-size:0.82rem; color:#475569; margin-bottom:6px;">
                        Nulls: <strong style="color:{"#DC2626" if null_cnt > 0 else "#059669"};">{null_cnt} ({null_pct}%)</strong> &nbsp;|&nbsp; Distinct: <strong style="color:#7C3AED;">{uniq_cnt:,}</strong>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                if st.button(f"Explore '{col_name}' →", key=f"btn_explore_card_{col_name}", type="primary", use_container_width=True):
                    S["selected_eda_col"] = col_name
                    S["step"] = "eda"
                    S["_scroll_to_stage_top"] = True
                    st.toast(f"Navigating to EDA Deep-Dive for column '{col_name}'")
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next: Data Cleaning & Preprocessing Studio →", type="primary", use_container_width=True):
            next_step("business_collection")



# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Data Cleaning and Preprocessing Studio
# ══════════════════════════════════════════════════════════════════════════════
elif S["step"] == "data_cleaning":
    render_stage_header(2, "Data Cleaning & Preprocessing Studio", "Recommendations-first data preparation: automated diagnosis, health scorecard, and 1-click fixes.")

    if st.button("Next: Exploratory Data Analysis (EDA) →", type="primary", key="btn_clean_next_top", use_container_width=True):
        next_step("data_cleaning")

    df = S["df"]
    if S["quality"] is None:
        with st.spinner("Auditing data quality..."):
            S["quality"] = full_quality_report(df, S["col_types"])

    q = S["quality"]
    health = compute_data_health_score(df)

    render_health_scorecard(health["score"], health["missing_cols"], health["dup_rows"], health["outlier_cols"])

    st.markdown("""
    <div style="background:#F5F1FF; border:1px solid #DDD6FE; border-left:4px solid #7C3AED; border-radius:8px; padding:14px 18px; margin:16px 0 20px;">
        <div style="font-weight:800; color:#5B21B6; font-size:0.95rem; margin-bottom:4px;">Stage 2 Roadmap: Work Through 7 Steps Before Proceeding</div>
        <div style="color:#334155; font-size:0.85rem; line-height:1.65;">
            1. Review automated diagnosis below & execute <strong>Apply All Fixes (1-Click)</strong> if recommendations exist.<br/>
            2. Inspect each of the <strong>7 cleaning tabs below</strong> (Missing Values, Duplicates, Outliers, Structural, Encoding, Scaling, Reduction).<br/>
            3. Once steps show <span style="color:#10B981; font-weight:700;">Clean / Ready</span>, click the <strong>Proceed to Stage 3</strong> button at the bottom of the page.
        </div>
    </div>
    """, unsafe_allow_html=True)

    recs = detect_cleaning_recommendations(df)
    if recs:
        with st.expander(f"Iris Data Cleaning Recommendations ({len(recs)} Actionable Suggestions)", expanded=True):
            r_col1, r_col2 = st.columns([3, 1])
            with r_col1:
                for rec in recs:
                    render_recommendation_card(rec["priority"], rec["title"], rec["reason"], rec["col_name"])
            with r_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Apply All Fixes (1-Click)", type="primary", use_container_width=True, key="btn_apply_all_recs"):
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
                    st.toast(msg)
                    st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── 7-Step Cleaning Workflow Status Bar ───────────────────────────────────
    _miss_count = len(q["missing"]) if q and "missing" in q and hasattr(q["missing"], "__len__") else 0
    _dup_count = int(df.duplicated().sum())
    _out_count = len(q["outliers"]) if q and "outliers" in q and hasattr(q["outliers"], "__len__") else 0
    _struct_issues = detect_structural_errors(S["df"])
    _struct_count = len(_struct_issues)
    _cat_cols = [c for c, t in S["col_types"].items() if t in ("categorical", "boolean")]
    _num_cols_raw = [c for c, t in S["col_types"].items() if t == "numeric"]

    def _step_cell(num, label, count, ok_msg):
        if count == 0:
            bg = "rgba(5,150,105,0.08)"; border = "#059669"
            status_html = f'<div style="font-size:0.75rem; font-weight:800; color:#059669;">{ok_msg}</div>'
        else:
            bg = "rgba(217,119,6,0.08)"; border = "#D97706"
            status_html = f'<div style="font-size:0.75rem; font-weight:800; color:#D97706;">{count} issues</div>'
        return f"""<div style="background:{bg}; border:1px solid {border}40; border-top:3px solid {border};
                            border-radius:10px; padding:12px 14px; text-align:center; min-width:0; box-shadow:0 2px 6px rgba(0,0,0,0.02);">
                    <div style="font-size:0.65rem; color:#64748B; font-weight:800; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:5px;">Step {num}</div>
                    <div style="font-size:0.82rem; font-weight:800; color:#0F172A; margin-bottom:5px;">{label}</div>
                    {status_html}
                </div>"""

    _log_messages = [message.lower() for message in S["cleaning_log"]]
    _encoding_applied = any("encoding" in message or "encoded" in message for message in _log_messages)
    _scaling_applied = any("scaling" in message or "scaled" in message for message in _log_messages)
    _reduction_reviewed = any(
        marker in message
        for message in _log_messages
        for marker in ("data reduction:", "no constant columns", "correlation reduction", "correlation threshold")
    )

    def _workflow_status(label, status, color):
        return f'<div style="font-size:0.75rem; font-weight:800; color:{color};">{status}</div>'

    _encoding_status = "N/A" if not _cat_cols else "Applied" if _encoding_applied else "Not started"
    _scaling_status = "N/A" if not _num_cols_raw else "Applied" if _scaling_applied else "Not started"
    _reduction_status = "Reviewed" if _reduction_reviewed else "Not reviewed"
    _encoding_color = "#059669" if _encoding_applied or not _cat_cols else "#6D28D9"
    _scaling_color = "#059669" if _scaling_applied or not _num_cols_raw else "#0E7490"
    _reduction_color = "#059669" if _reduction_reviewed else "#64748B"
    st.markdown(f"""
    <div style="margin-bottom:20px;">
        <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:0.08em; color:#1E293B; margin-bottom:12px;">Cleaning Workflow — Work through all 7 steps before proceeding</div>
        <div style="display:grid; grid-template-columns:repeat(7,1fr); gap:10px;">
            {_step_cell(1, 'Missing Values', _miss_count, 'Clean')}
            {_step_cell(2, 'Duplicates', _dup_count, 'Clean')}
            {_step_cell(3, 'Outliers', _out_count, 'Clean')}
            {_step_cell(4, 'Structural', _struct_count, 'Clean')}
            <div style="background:rgba(124,58,237,0.06); border:1px solid rgba(124,58,237,0.2); border-top:3px solid #7C3AED; border-radius:10px; padding:12px 14px; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.02);">
                <div style="font-size:0.65rem; color:#64748B; font-weight:800; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:5px;">Step 5</div>
                <div style="font-size:0.82rem; font-weight:800; color:#0F172A; margin-bottom:5px;">Encoding</div>
                {_workflow_status('Encoding', _encoding_status, _encoding_color)}
            </div>
            <div style="background:rgba(8,145,178,0.06); border:1px solid rgba(8,145,178,0.2); border-top:3px solid #0891B2; border-radius:10px; padding:12px 14px; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.02);">
                <div style="font-size:0.65rem; color:#64748B; font-weight:800; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:5px;">Step 6</div>
                <div style="font-size:0.82rem; font-weight:800; color:#0F172A; margin-bottom:5px;">Scaling</div>
                {_workflow_status('Scaling', _scaling_status, _scaling_color)}
            </div>
            <div style="background:rgba(100,116,139,0.06); border:1px solid rgba(100,116,139,0.2); border-top:3px solid #64748B; border-radius:10px; padding:12px 14px; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.02);">
                <div style="font-size:0.65rem; color:#64748B; font-weight:800; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:5px;">Step 7</div>
                <div style="font-size:0.82rem; font-weight:800; color:#0F172A; margin-bottom:5px;">Reduction</div>
                {_workflow_status('Reduction', _reduction_status, _reduction_color)}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t_miss, t_dup, t_out, t_struct, t_enc, t_scale, t_red = st.tabs([
        "1. Missing Values",
        "2. Duplicates",
        "3. Treat Outliers",
        "4. Structural Errors",
        "5. Categorical Encoding",
        "6. Feature Scaling",
        "7. Data Reduction"
    ])


    with t_miss:
        render_section_header("Handling Missing Values", "missing", badge="Step 1 of 7", badge_color="#F59E0B" if _miss_count > 0 else "#10B981")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Fill gaps using statistical imputation (mean, median, mode) or drop sparse rows and columns if data loss is acceptable.</div>", unsafe_allow_html=True)

        miss_df = q["missing"]
        if len(miss_df) > 0:
            render_styled_dataframe(miss_df, use_container_width=True)
            
            st.markdown("**Batch Quick Imputation:**")
            g1, g2, g3, g4, g5 = st.columns(5)
            with g1:
                if st.button("Numeric → Median", use_container_width=True, key="btn_imp_med"):
                    for c, ct in S["col_types"].items():
                        if ct == "numeric" and S["df"][c].isnull().sum() > 0:
                            S["df"], msg = impute_missing(S["df"], c, "median")
                            S["cleaning_log"].append(msg)
                    S["quality"] = full_quality_report(S["df"], S["col_types"])
                    st.toast("Imputed numeric features with median!")
                    st.rerun()
            with g2:
                if st.button("Numeric → Mean", use_container_width=True, key="btn_imp_mean"):
                    for c, ct in S["col_types"].items():
                        if ct == "numeric" and S["df"][c].isnull().sum() > 0:
                            S["df"], msg = impute_missing(S["df"], c, "mean")
                            S["cleaning_log"].append(msg)
                    S["quality"] = full_quality_report(S["df"], S["col_types"])
                    st.toast("Imputed numeric features with mean!")
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
        render_section_header("Removing Duplicate Entries", "dupe", badge="Step 2 of 7", badge_color="#F59E0B" if _dup_count > 0 else "#10B981")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Eliminate redundant or repeated entries to prevent skewing model analysis.</div>", unsafe_allow_html=True)
        
        n_dups = int(df.duplicated().sum())
        if n_dups > 0:
            st.warning(f"Detected **{n_dups} duplicate rows** ({round(n_dups/len(df)*100, 2)}% of dataset).")
            with st.expander("Preview Duplicate Rows"):
                render_styled_dataframe(df[df.duplicated(keep=False)].head(20), use_container_width=True)
            if st.button("Eliminate All Duplicate Rows", type="primary", key="btn_remove_dups"):
                S["df"], msg = remove_duplicates(S["df"])
                S["cleaning_log"].append(msg)
                S["quality"] = full_quality_report(S["df"], S["col_types"])
                st.toast(msg)
                st.rerun()
        else:
            st.success("Clean Dataset: Zero duplicate rows found.")

    with t_out:
        render_section_header("Treating Extreme Outliers", "outlier", badge="Step 3 of 7", badge_color="#F59E0B" if _out_count > 0 else "#10B981")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Detect and manage extreme anomalies using Interquartile Range (IQR) or Z-score methods.</div>", unsafe_allow_html=True)

        num_cols = [c for c, t in S["col_types"].items() if t == "numeric"]
        out_df = q["outliers"]
        if len(out_df) > 0:
            render_styled_dataframe(out_df, use_container_width=True)
            if st.button("Cap All IQR Outliers to Boundaries (1-Click)", type="primary", key="btn_cap_all_outliers"):
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
                st.toast(msg)
                st.rerun()
        else:
            st.info("No significant outliers detected.")

        if num_cols:
            st.markdown("<br>**Per-Column Outlier Treatment:**", unsafe_allow_html=True)
            col_out = st.selectbox("Select numeric column to treat:", num_cols, key="select_out_col")
            out_method = st.radio("Outlier Detection Method:", ["IQR (Interquartile Range)", "Z-Score (|Z| > threshold)"], horizontal=True, key="radio_out_method")
            out_action = st.radio("Action:", ["cap (Clip to boundary values)", "drop (Remove outlier rows)"], horizontal=True, key="radio_out_action")

            if st.button("Apply Outlier Treatment to Column", key="btn_apply_outlier"):
                act = "cap" if "cap" in out_action else "drop"
                if "IQR" in out_method:
                    S["df"], msg = treat_outliers_iqr(S["df"], col_out, action=act)
                else:
                    S["df"], msg = treat_outliers_zscore(S["df"], col_out, action=act)
                S["cleaning_log"].append(msg)
                S["quality"] = full_quality_report(S["df"], S["col_types"])
                st.toast(msg)
                st.rerun()

    with t_struct:
        render_section_header("Correcting Structural Errors & Types", "struct", badge="Step 4 of 7", badge_color="#F59E0B" if _struct_count > 0 else "#10B981")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Iris automatically scans text features for whitespace padding, mixed capitalization, and improper data types.</div>", unsafe_allow_html=True)

        struct_issues = detect_structural_errors(S["df"])
        if struct_issues:
            st.markdown(f"#### Detected Structural Issues ({len(struct_issues)} Inconsistencies Found)")
            for issue in struct_issues:
                render_recommendation_card("HIGH", issue["title"], issue["reason"], issue["column"])
            
            if st.button("Fix All Structural Errors (1-Click)", type="primary", key="btn_fix_all_struct"):
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
                st.toast(msg)
                st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.success("Clean Text Features: Zero structural errors or whitespace padding detected.")
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

            if st.button("Clean String Errors", type="primary", key="btn_clean_string"):
                S["df"], msg = clean_structural_errors(S["df"], str_col, str_act[0])
                S["cleaning_log"].append(msg)
                st.toast(msg)
                st.rerun()

        with st2:
            st.markdown("**Data Type Correction:**")
            type_col = st.selectbox("Select Column to Re-cast Type:", S["df"].columns.tolist(), key="select_type_col")
            target_type = st.selectbox("Target Data Type:", ["numeric", "categorical", "datetime", "boolean"], key="select_target_type")

            if st.button("Convert Data Type", type="primary", key="btn_convert_type"):
                S["df"], msg = convert_column_type(S["df"], type_col, target_type)
                S["col_types"] = infer_all_types(S["df"])
                S["cleaning_log"].append(msg)
                st.toast(msg)
                st.rerun()


    with t_enc:
        render_section_header("Categorical Encoding — Automated Recommendations", "encode", badge="Step 5 of 7", badge_color="#F59E0B" if len(_cat_cols) > 0 else "#10B981")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Iris analyzes your categorical columns and recommends the optimal encoding method for each feature.</div>", unsafe_allow_html=True)

        cat_suggs = get_encoding_suggestions(S["df"], S["col_types"])
        if cat_suggs:
            st.markdown("""
            <div style='background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.3); border-radius:14px; padding:18px; margin-bottom:20px;'>
                <div style='font-size:1.1rem; font-weight:700; color:#8B5CF6;'>One-Click Automated Encoding</div>
                <div style='color:#9CA3AF; font-size:0.85rem; margin-top:4px;'>Apply all recommended encoding methods across all categorical features instantly.</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Auto-Encode All Categorical Features (Recommended)", type="primary", key="btn_auto_encode_all"):
                S["df"], msg = auto_encode_all_categorical(S["df"], S["col_types"])
                S["col_types"] = infer_all_types(S["df"])
                S["cleaning_log"].append(msg)
                st.toast(msg)
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
                        st.toast(msg)
                        st.rerun()
                st.markdown("<hr style='border-color:rgba(139,92,246,0.1); margin:8px 0;'>", unsafe_allow_html=True)
        else:
            st.info("No categorical features remaining for encoding.")

    with t_scale:
        render_section_header("Feature Scaling — Automated Recommendations", "scale", badge="Step 6 of 7", badge_color="#F59E0B" if len(_num_cols_raw) > 0 else "#10B981")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>Iris analyzes your numeric distributions and recommends Standardization (Z-score) vs. Min-Max scaling per column.</div>", unsafe_allow_html=True)

        scale_suggs = get_scaling_suggestions(S["df"], S["col_types"])
        if scale_suggs:
            st.markdown("""
            <div style='background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.3); border-radius:14px; padding:18px; margin-bottom:20px;'>
                <div style='font-size:1.1rem; font-weight:700; color:#06B6D4;'>One-Click Automated Scaling</div>
                <div style='color:#9CA3AF; font-size:0.85rem; margin-top:4px;'>Standardize high-variance numeric columns and scale bounded features into uniform ranges.</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Auto-Scale All Numeric Features (Recommended)", type="primary", key="btn_auto_scale_all"):
                S["df"], msg = auto_scale_all_numeric(S["df"], S["col_types"])
                S["col_types"] = infer_all_types(S["df"])
                S["cleaning_log"].append(msg)
                st.toast(msg)
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
                            st.toast(msg)
                            st.rerun()
                    else:
                        st.caption("✅ Scaled")
                st.markdown("<hr style='border-color:rgba(6,182,212,0.1); margin:8px 0;'>", unsafe_allow_html=True)
        else:
            st.info("No numeric features remaining for scaling.")

    with t_red:
        render_section_header("Simple Data Reduction", "reduce", badge="Step 7 of 7", badge_color="#F59E0B" if (_struct_count+_miss_count+_dup_count+_out_count) > 0 else "#10B981")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:16px;'>One-click actions to drop useless constant columns and remove redundant collinear features.</div>", unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("""
            <div class="feature-card" style="padding:20px; text-align:center;">
                <div style="font-weight:700; color:#8B5CF6; margin-bottom:4px;">Constant Columns</div>
                <div style="color:#9CA3AF; font-size:0.82rem; margin-bottom:14px;">Drop features with 0 variance (only 1 unique value).</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Drop Constant Columns (1-Click)", use_container_width=True, key="btn_drop_const"):
                S["df"], msg = drop_constant_features(S["df"])
                S["col_types"] = infer_all_types(S["df"])
                S["cleaning_log"].append(msg)
                st.toast(msg)
                st.rerun()

        with r2:
            st.markdown("""
            <div class="feature-card" style="padding:20px; text-align:center;">
                <div style="font-weight:700; color:#06B6D4; margin-bottom:4px;">Redundant Collinear Features</div>
                <div style="color:#9CA3AF; font-size:0.82rem; margin-bottom:14px;">Drop feature pairs with correlation |r| > 0.90.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Drop Redundant Collinear Features (1-Click)", use_container_width=True, key="btn_drop_high_corr"):
                S["df"], msg = drop_high_correlation_features(S["df"], 0.90)
                S["col_types"] = infer_all_types(S["df"])
                S["cleaning_log"].append(msg)
                st.toast(msg)
                st.rerun()

        st.markdown("<br>**Manual Quick Drop:**", unsafe_allow_html=True)
        cols_to_drop = st.multiselect("Select columns to remove:", S["df"].columns.tolist(), key="select_manual_drop")
        if st.button("Drop Selected Columns", type="primary", key="btn_manual_drop"):
            if cols_to_drop:
                S["df"] = S["df"].drop(columns=cols_to_drop)
                S["col_types"] = infer_all_types(S["df"])
                msg = f"Manually dropped {len(cols_to_drop)} columns: {', '.join(cols_to_drop)}"
                S["cleaning_log"].append(msg)
                st.toast(msg)
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

    if st.button("Next: Feature Engineering Studio →", type="primary", key="btn_eda_next_top", use_container_width=True):
        next_step("eda")

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
        <div class="glass-card" style="border-left: 4px solid #0891B2;">
            <div style="font-weight: 800; font-size: 1.1rem; color: #0F172A; margin-bottom: 10px;">Iris EDA Narrative Key Findings</div>
            <ul style="color: #334155; margin: 0; padding-left: 20px; line-height: 1.8; font-size:0.92rem;">
        """ + "".join([f"<li>{f}</li>" for f in findings]) + """
            </ul>
        </div>
        """, unsafe_allow_html=True)

    eda_recs = generate_eda_chart_recommendations(df, col_types, S["col_stats"], S["corr"])


    try:
        eda_html_str = build_eda_standalone_dashboard(df, col_types, S["col_stats"], S["corr"], eda_recs)
        st.download_button(
            "Export Standalone EDA Charts Dashboard (HTML)",
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
        "1. Automated Recommended Charts",
        "2. Feature Deep-Dive Explorer",
        "3. Correlation & Bivariate Matrix"
    ])

    with tab_auto_charts:
        render_section_header(f"{len(eda_recs)} Automated Chart Recommendations for your Dataset", "bulb", badge="Auto-Generated", badge_color="#8B5CF6")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:20px;'>Below are high-value visualizations chosen by Iris based on feature distributions, statistical properties, and correlation strength.</div>", unsafe_allow_html=True)
        
        # Group charts by category
        from collections import defaultdict
        grouped_recs = defaultdict(list)
        for rec in eda_recs:
            grouped_recs[rec['category']].append(rec)
            
        for category, recs in grouped_recs.items():
            st.markdown(f"#### {category}")
            st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:8px 0 24px;'>", unsafe_allow_html=True)
            
            # Display charts in a 2-column grid to look more like a dashboard
            for i in range(0, len(recs), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(recs):
                        rec = recs[i+j]
                        with cols[j]:
                            st.markdown(f"""
                            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-top:3px solid #7C3AED; border-radius:14px; padding:18px; margin-bottom:12px; box-shadow:0 4px 14px rgba(0,0,0,0.03);">
                                <div style="display:flex; gap:6px; margin-bottom:8px;">
                                    <span class="type-badge" style="background:#F3E8FF; color:#6D28D9; border:1px solid #DDD6FE; font-size:0.7rem; padding:3px 10px; font-weight:700;">{rec['chart_type']}</span>
                                </div>
                                <div style="font-size:1.05rem; font-weight:800; color:#0F172A; margin-bottom:4px; line-height:1.2;">{rec['title']}</div>
                                <div style="color:#475569; font-size:0.78rem;">Target Features: <strong style="color:#0F172A;">{', '.join(rec['columns'])}</strong></div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.plotly_chart(rec["fig"], use_container_width=True)
                            
                            st.markdown(f"""
                            <div class="why-card" style="font-size:0.84rem; padding:14px 18px; margin-bottom:32px; background:#F8FAFC; border:1px solid #E2E8F0; border-left:3px solid #0891B2; border-radius:0 10px 10px 0; color:#334155;">
                                <strong style="color:#0F172A;">Why are we making this chart?</strong><br>
                                {rec['why']}
                            </div>
                            """, unsafe_allow_html=True)

    with tab_col_explorer:
        all_cols = df.columns.tolist()
        sel_default = S.get("selected_eda_col", all_cols[0] if all_cols else "")
        sel_idx = all_cols.index(sel_default) if sel_default in all_cols else 0
        selected_col = st.selectbox("Select feature for deep-dive exploration:", all_cols, index=sel_idx, key="select_eda_col")
        ctype = col_types.get(selected_col, "unknown")
        stats = S["col_stats"].get(selected_col, {})

        st.markdown(f"<div class='h-section' style='margin-bottom:12px;'>{selected_col} <span style='color:#94A3B8; font-weight:400; font-size:0.9rem;'>({ctype})</span></div>", unsafe_allow_html=True)

        chip_keys = ["count", "missing", "unique", "mean", "median", "std", "min", "max", "skewness"]
        chips_html = "<div style='display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px;'>"
        for k in chip_keys:
            if k in stats and stats[k] is not None:
                chips_html += f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:6px 14px; font-size:0.82rem; box-shadow:0 1px 4px rgba(0,0,0,0.02);"><span style="color:#64748B; font-weight:600;">{k.title()}:</span> <strong style="color:#7C3AED;">{stats[k]}</strong></div>'
        chips_html += "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)

        if ctype == "numeric":
            col_s = df[selected_col].iloc[:, 0] if isinstance(df[selected_col], pd.DataFrame) else df[selected_col]
            s = pd.to_numeric(col_s, errors="coerce").dropna()
            plot_df = pd.DataFrame({selected_col: s})

            t1, t2, t3, t4 = st.tabs(["Distribution Histogram", "Violin Plot", "Box Plot", "Outlier Strip"])
            with t1:
                fig = px.histogram(plot_df, x=selected_col, nbins=40, color_discrete_sequence=["#7C3AED"], marginal="rug")
                fig.update_layout(plot_bgcolor="#FAFAFC", paper_bgcolor="#FFFFFF", font=dict(color="#0F172A"))
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                fig = px.violin(plot_df, y=selected_col, box=True, points="outliers", color_discrete_sequence=["#0891B2"])
                fig.update_layout(plot_bgcolor="#FAFAFC", paper_bgcolor="#FFFFFF", font=dict(color="#0F172A"))
                st.plotly_chart(fig, use_container_width=True)
            with t3:
                fig = px.box(plot_df, y=selected_col, points="all", color_discrete_sequence=["#DB2777"])
                fig.update_layout(plot_bgcolor="#FAFAFC", paper_bgcolor="#FFFFFF", font=dict(color="#0F172A"))
                st.plotly_chart(fig, use_container_width=True)
            with t4:
                q1v, q3v = s.quantile(0.25), s.quantile(0.75)
                iqrv = q3v - q1v
                lo, hi = q1v - 1.5*iqrv, q3v + 1.5*iqrv
                color = s.apply(lambda v: "Outlier" if v < lo or v > hi else "Normal")
                tmp = pd.DataFrame({"value": s, "type": color})
                fig = px.strip(tmp, y="value", color="type", color_discrete_map={"Normal":"#7C3AED", "Outlier":"#DB2777"})
                fig.update_layout(plot_bgcolor="#FAFAFC", paper_bgcolor="#FFFFFF", font=dict(color="#0F172A"))
                st.plotly_chart(fig, use_container_width=True)

        elif ctype in ("categorical", "boolean"):
            col_s = df[selected_col].iloc[:, 0] if isinstance(df[selected_col], pd.DataFrame) else df[selected_col]
            vc = col_s.value_counts().head(20).reset_index()
            vc.columns = ["Category", "Count"]
            t1, t2 = st.tabs(["Bar Chart", "Pie Chart"])
            with t1:
                fig = px.bar(vc, x="Count", y="Category", orientation="h", color="Count", color_continuous_scale=["#E2E8F0", "#0891B2"])
                fig.update_layout(plot_bgcolor="#FAFAFC", paper_bgcolor="#FFFFFF", font=dict(color="#0F172A"), yaxis=dict(title=selected_col))
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                fig = px.pie(vc, names="Category", values="Count", hole=0.35, color_discrete_sequence=["#7C3AED","#0891B2","#DB2777","#059669"])
                fig.update_layout(paper_bgcolor="#FFFFFF", font=dict(color="#0F172A"))
                st.plotly_chart(fig, use_container_width=True)

        iris_says_block(f"col_{selected_col}", explain_column, selected_col, stats)

    with tab_corr_matrix:
        corr_res = S["corr"]
        if corr_res.get("pearson") is not None:
            st.plotly_chart(correlation_heatmap(corr_res["pearson"]), use_container_width=True)
            if corr_res.get("top_pairs"):
                st.markdown("**Top Correlated Pairs:**")
                render_styled_dataframe(pd.DataFrame(corr_res["top_pairs"]), use_container_width=True)
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

    if st.button("Next: Communication & Executive Export →", type="primary", key="btn_fe_next_top", use_container_width=True):
        next_step("feature_engineering")

    df = S["df"]
    col_types = S["col_types"]

    fe_plan = generate_fe_plan(df, col_types)

    st.markdown(f"""
    <div class="glass-card" style="border-left:3px solid #EC4899; margin-bottom:20px; padding:18px 24px;">
        <div style="font-weight:800; font-size:1.1rem; color:#4C1D95; margin-bottom:6px;">Iris Feature Engineering Strategy Plan</div>
        <div style="color:#334155; font-size:0.9rem;">
            <b>{fe_plan['total_planned_actions']}</b> planned actions will create approximately
            <b>+{fe_plan['estimated_new_columns']}</b> new columns from the existing features.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Per-column action preview table
    if fe_plan["items"]:
        fe_preview_rows = "".join([
            f"""<tr>
                <td style='padding:9px 14px; color:#1E293B; font-weight:700; border-bottom:1px solid #EDE9FE;'>{it['col']}</td>
                <td style='padding:9px 14px;'>
                    <span style='display:inline-block; background:#F3E8FF; color:#5B21B6; border:1px solid #DDD6FE;
                                 border-radius:6px; padding:3px 10px; font-size:0.8rem; font-weight:800;'>
                        {it['type']}
                    </span>
                </td>
                <td style='padding:9px 14px; color:#0E7490; font-weight:800; text-align:center; border-bottom:1px solid #EDE9FE;'>+{it['est_cols']}</td>
            </tr>"""
            for it in fe_plan["items"]
        ])
        st.markdown(f"""
        <div style='margin-bottom:24px; border:1px solid #DDD6FE; border-radius:8px; overflow:hidden; box-shadow:0 4px 14px rgba(76,29,149,0.06);'>
            <table style='width:100%; border-collapse:collapse; font-size:0.88rem;'>
                <thead>
                    <tr style='background:#6D28D9;'>
                        <th style='padding:10px 14px; color:#FFFFFF; text-align:left; font-weight:800;'>Column</th>
                        <th style='padding:10px 14px; color:#FFFFFF; text-align:left; font-weight:800;'>Planned Transformation</th>
                        <th style='padding:10px 14px; color:#FFFFFF; text-align:center; font-weight:800;'>New Cols</th>
                    </tr>
                </thead>
                <tbody>{fe_preview_rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Apply All Planned Feature Transformations (1-Click)", type="primary", key="btn_auto_fe_all", use_container_width=True):
        new_df, explanations, summary_msg = auto_engineer_all_features(df, col_types)
        S["df"] = new_df
        S["col_types"] = infer_all_types(new_df)
        S["fe_explanations"] = explanations
        S["transformation_log"].append(summary_msg)
        st.toast(summary_msg)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Interaction Feature Builder", expanded=False):
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
                    st.toast(msg)
                    st.rerun()
        else:
            st.info("Need at least 2 numeric features for interaction builder.")

    if S.get("target_col") and S.get("target_col") in df.columns:
        fi_dict = quick_feature_importance(df, S["target_col"])
        if fi_dict:
            st.markdown("<br>", unsafe_allow_html=True)
            render_section_header(f"Feature Importance Preview (Target: '{S['target_col']}')", "importance", badge="Insight", badge_color="#EC4899")
            fi_df = pd.DataFrame(list(fi_dict.items()), columns=["Feature", "Importance Score"])
            fig = px.bar(fi_df, x="Importance Score", y="Feature", orientation="h", color="Importance Score", color_continuous_scale="Purples")
            fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font=dict(color="#1E293B"), colorway=["#6D28D9", "#0891B2", "#DB2777", "#059669"])
            st.plotly_chart(fig, use_container_width=True)


    # Explanations Section
    if S["fe_explanations"]:
        render_section_header(f"Iris Feature Engineering Log ({len(S['fe_explanations'])} features created)", "log", badge="Auto-Generated", badge_color="#8B5CF6")
        st.markdown("<div style='color:#9CA3AF; font-size:0.88rem; margin-bottom:20px;'>Below is the plain-English explanation for every feature engineered by Iris:</div>", unsafe_allow_html=True)

        for exp in S["fe_explanations"]:
            c_orig = exp["column"]
            f_new = exp["new_feature"]
            badge = exp["badge"]
            b_color = exp.get("badge_color", "#8B5CF6")
            trans = exp["transformation"]
            why_text = exp["why"]

            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #DDD6FE; border-left:4px solid #7C3AED; border-radius:8px; padding:20px; margin-bottom:20px; box-shadow:0 4px 14px rgba(76,29,149,0.06);">
                <div style="display:flex; justify-space-between; align-items:center; margin-bottom:8px;">
                    <div>
                        <span class="type-badge" style="background:{b_color}22; color:{b_color}; border:1px solid {b_color}55;">{badge}</span>
                        <span style="font-size:1.1rem; font-weight:800; color:#1E293B; margin-left:8px;">{c_orig} &nbsp;→&nbsp; <span style="color:#7C3AED;">{f_new}</span></span>
                    </div>
                </div>
                <div style="color:#64748B; font-size:0.83rem; margin-bottom:12px;">Method: <strong style="color:#334155;">{trans}</strong></div>
                <div class="why-card" style="margin-top:0;">
                    <strong>Why did Iris create this feature?</strong><br>
                    {why_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Optional Advanced Custom Feature Studio
    with st.expander("Advanced Custom Feature Transformation Studio (Optional)", expanded=False):
        fe_col1, fe_col2 = st.columns([1, 1])
        with fe_col1:
            selected_feature = st.selectbox("Select Column to Transform:", df.columns.tolist(), key="fe_col_select")
            trans_options = [
                ("log", "Log Transform (log1p) — Fix Skewness"),
                ("standard_scale", "Standard Scaling (Z-Score: mean=0, std=1)"),
                ("minmax_scale", "Min-Max Scaling ([0, 1] range)"),
                ("bin_quantiles", "4-Quantile Binning (Low/Med/High)"),
                ("binary_encode", "Binary Encoding (0/1)"),
                ("one_hot", "One-Hot Encoding (Dummy columns)"),
                ("freq_encode", "Frequency Encoding"),
                ("datetime_decompose", "Datetime Decomposition"),
                ("drop", "Drop Column"),
            ]
            trans_type = st.selectbox("Select Transformation:", [t[0] for t in trans_options], format_func=lambda x: [t[1] for t in trans_options if t[0]==x][0])

            if st.button("Apply Custom Transformation", key="btn_apply_custom_fe", use_container_width=True):
                new_df, status_msg = apply_transformation(df, selected_feature, trans_type)
                S["df"] = new_df
                S["col_types"] = infer_all_types(new_df)
                S["transformation_log"].append(status_msg)
                st.toast(status_msg)
                st.rerun()

        with fe_col2:
            st.markdown("**Distribution Visualizer**")
            s_raw = df[selected_feature]
            if isinstance(s_raw, pd.DataFrame):
                s_raw = s_raw.iloc[:, 0]
            if pd.api.types.is_numeric_dtype(s_raw):
                plot_df = pd.DataFrame({selected_feature: s_raw})
                fig_prev = px.histogram(plot_df, x=selected_feature, title=f"Feature: '{selected_feature}'", color_discrete_sequence=["#8B5CF6"])
                fig_prev.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font=dict(color="#1E293B"), height=300, margin=dict(l=32, r=20, t=48, b=36))
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
    render_section_header("Project Pipeline Summary", "summary")
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
    render_section_header("Dataset Preview (Cleaned & Engineered)", "table")
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
    with st.expander("AI Executive Narrative Briefing", expanded=True):
        st.markdown(exec_summary_text)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ML Model Recommendation Engine ──────────────────────────────────────────
    render_section_header("ML Model Recommendation Engine", "ml", badge="Advisory", badge_color="#3B82F6")
    st.markdown(
        "<div style='color:#475569; font-size:0.88rem; margin-bottom:20px;'>"
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
        <div>
            <div style="font-size:0.75rem; color:{t_color}; font-weight:800; letter-spacing:0.08em; text-transform:uppercase;">Detected Task Type</div>
            <div style="font-size:1.05rem; font-weight:800; color:#1E293B; margin-top:2px;">{task_label}{target_badge}</div>
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
                        border-radius:10px; padding:11px 18px; margin-bottom:10px; font-size:0.88rem; color:#334155; line-height:1.55;">
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
        <div style="background:#FFFFFF; border:1px solid #DDD6FE; border-left:4px solid {m_color}; border-radius:8px;
                padding:22px 26px; margin-bottom:14px; box-shadow:0 4px 14px rgba(76,29,149,0.06);">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px; margin-bottom:14px;">
                <div>
                    <span style="background:{m_bg}; color:{m_color}; border:1px solid {m_color}55;
                                 border-radius:20px; padding:4px 12px; font-size:0.75rem; font-weight:800;
                                 letter-spacing:0.04em;">{suit_label}</span>
                    <h3 style="color:#1E293B; font-size:1.1rem; font-weight:800; margin:8px 0 2px;">{'⚡' if 'Primary' in suit_label else ('✅' if 'Baseline' in suit_label else '🔷')} {model['name']}</h3>
                    <span style="color:#64748B; font-size:0.8rem; font-weight:600;">{model['category']}</span>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.72rem; color:#64748B; text-transform:uppercase; letter-spacing:0.07em;">Interpretability</div>
                    <div style="color:{interp_color}; font-weight:700; font-size:0.9rem;">{model['interpretability']}</div>
                    <div style="font-size:0.72rem; color:#64748B; margin-top:4px;">Complexity: {model['complexity']}</div>
                </div>
            </div>
                <div style="background:#F7F3FF; border:1px solid #E9DFFF; border-left:3px solid {m_color};
                    border-radius:6px; padding:12px 16px; margin-bottom:10px; color:#334155; font-size:0.87rem; line-height:1.6;">
                <strong style='color:#4C1D95;'>Why this model?</strong><br>{model['why']}
            </div>
            <div style="color:#64748B; font-size:0.82rem;">
                Best for: <em style='color:#334155;'>{model['best_for']}</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Evaluation Metrics
    if ml_recs["eval_metrics"]:
        metrics_html = "".join([
            f"<span style='background:#F3E8FF; color:#5B21B6; border:1px solid #DDD6FE;"
            f" border-radius:6px; padding:5px 12px; font-size:0.82rem; font-weight:700; margin:4px; display:inline-block;'>{m}</span>"
            for m in ml_recs["eval_metrics"]
        ])
        st.markdown(f"""
        <div style="margin-top:8px; margin-bottom:32px;">
            <div style="font-size:0.8rem; font-weight:700; color:#475569; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:10px;">Recommended Evaluation Metrics</div>
            {metrics_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Export Reports & Datasets", "export")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("""
        <div class="feature-card" style="text-align:center; padding:24px 16px;">
            <div style="font-size:1.1rem; font-weight:800; color:#5B21B6; margin-bottom:6px;">Interactive HTML Dashboard</div>
            <div style="color:#475569; font-size:0.82rem; margin-bottom:16px;">
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
            <div style="font-size:1.1rem; font-weight:800; color:#0E7490; margin-bottom:6px;">Dark Executive PDF Report</div>
            <div style="color:#475569; font-size:0.82rem; margin-bottom:16px;">
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
            <div style="font-size:1.1rem; font-weight:800; color:#BE185D; margin-bottom:6px;">Cleaned & Engineered CSV</div>
            <div style="color:#475569; font-size:0.82rem; margin-bottom:16px;">
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
    render_section_header("Ask Iris AI Anything", "chat")
    st.markdown("<div style='color:#475569; font-size:0.88rem; margin-bottom:16px;'>Ask questions about your data analysis, business findings, or feature engineering strategy.</div>", unsafe_allow_html=True)

    for msg in S["chat_history"]:
        role_label = "<strong style='color:#5B21B6;'>Iris</strong>" if msg["role"] == "iris" else "<strong style='color:#0E7490;'>You</strong>"
        align = "left" if msg["role"] == "iris" else "right"
        bg = "rgba(139,92,246,0.12)" if msg["role"] == "iris" else "rgba(6,182,212,0.12)"
        border = "#8B5CF6" if msg["role"] == "iris" else "#06B6D4"
        st.markdown(f"""
        <div style="text-align:{align}; margin:8px 0;">
            <div style="display:inline-block; background:{bg}; border:1px solid {border};
                        border-radius:12px; padding:12px 16px; max-width:75%;
                        text-align:left; color:#1E293B; font-size:0.9rem; line-height:1.6;">
                <div style="font-size:0.75rem; text-transform:uppercase; margin-bottom:4px; opacity:0.8;">{role_label}</div>
                {msg['text']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_q = st.text_input("Your question:", placeholder="Which features have the strongest relationship? How does log transform help?", label_visibility="collapsed")
        with btn_col:
            send = st.form_submit_button("Ask Iris", use_container_width=True)

    if send and user_q:
        S["chat_history"].append({"role": "user", "text": user_q})
        with st.spinner("Iris is thinking..."):
            answer = chat_with_iris(user_q, S["dataset_context"])
        S["chat_history"].append({"role": "iris", "text": answer})
        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Start New Data Science Project", use_container_width=False):
        for key in list(S.keys()):
            del st.session_state[key]
        st.rerun()
