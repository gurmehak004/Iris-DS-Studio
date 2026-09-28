"""
ui/styles.py
CSS design system tokens and custom CSS injector for Iris Data Science Studio.
"""

import streamlit as st

CSS_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Base Theme Ivory Light Palette */
:root {
    --bg-dark: #FAFAFC;
    --card-bg: #FFFFFF;
    --card-border: #E2E8F0;
    --accent-violet: #7C3AED;
    --accent-violet-light: #8B5CF6;
    --accent-violet-dark: #6D28D9;
    --text-main: #0F172A;
    --text-muted: #64748B;
}

/* Crisp White & Violet Grid Pattern Background */
.stApp {
    background-color: #FAFAFC !important;
    background-image: 
        radial-gradient(rgba(124, 58, 237, 0.12) 1.2px, transparent 1.2px),
        linear-gradient(to right, rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(124, 58, 237, 0.03) 1px, transparent 1px) !important;
    background-size: 24px 24px, 48px 48px, 48px 48px !important;
    color: var(--text-main) !important;
}

/* ── STRICT VIOLET BUTTON & CARD TEXT COLOR RULE ── */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid*="primary"],
.btn-violet {
    background-color: #7C3AED !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: 1px solid #6D28D9 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.25) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid*="primary"]:hover,
.btn-violet:hover {
    background-color: #6D28D9 !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button[kind="primary"] *,
div.stButton > button[data-testid*="primary"] * {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Glassmorphic / Elevated Cards */
.glass-card {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(124, 58, 237, 0.35) !important;
}

/* KPI Card */
.kpi-card {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-top: 3px solid var(--accent-violet) !important;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.03) !important;
}

.kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-main) !important;
    margin-top: 4px;
}

.kpi-sub {
    font-size: 0.8rem;
    color: #059669 !important;
    margin-top: 2px;
}

/* ── FONT HIERARCHY SYSTEM ── */
.h-page {
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    color: #0F172A !important;
    line-height: 1.15;
}

.h-section {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: #0F172A !important;
    line-height: 1.3;
}

.h-subsection {
    font-size: 1.0rem;
    font-weight: 700;
    color: #1E293B !important;
    line-height: 1.35;
}

.h-card {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0F172A !important;
    line-height: 1.3;
}

.h-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B !important;
}

.h-body {
    font-size: 0.875rem;
    font-weight: 400;
    color: #334155 !important;
    line-height: 1.65;
}

.h-caption {
    font-size: 0.85rem;
    font-weight: 400;
    color: #475569 !important;
    line-height: 1.6;
}

/* Stage Header & Badges */
.stage-badge {
    display: inline-block;
    background: rgba(124, 58, 237, 0.1);
    border: 1px solid rgba(124, 58, 237, 0.3);
    color: #6D28D9 !important;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Iris Says AI Panel */
.iris-insight {
    background: rgba(124, 58, 237, 0.06) !important;
    border: 1px solid rgba(124, 58, 237, 0.25) !important;
    border-left: 4px solid var(--accent-violet) !important;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 20px;
    display: flex;
    gap: 16px;
    align-items: flex-start;
}

.iris-avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: rgba(124, 58, 237, 0.15) !important;
    border: 1px solid rgba(124, 58, 237, 0.35) !important;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.iris-title {
    font-weight: 800;
    font-size: 0.85rem;
    color: #6D28D9 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}

.iris-text {
    font-size: 0.92rem;
    color: #0F172A !important;
    line-height: 1.6;
}

/* Recommendation Pills */
.rec-card-critical {
    background: rgba(239, 68, 68, 0.06) !important;
    border: 1px solid rgba(239, 68, 68, 0.25) !important;
    border-left: 4px solid #EF4444 !important;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.rec-card-warning {
    background: rgba(245, 158, 11, 0.06) !important;
    border: 1px solid rgba(245, 158, 11, 0.25) !important;
    border-left: 4px solid #F59E0B !important;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.rec-card-info {
    background: rgba(6, 182, 212, 0.06) !important;
    border: 1px solid rgba(6, 182, 212, 0.25) !important;
    border-left: 4px solid #06B6D4 !important;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* Styled Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #F1F5F9 !important;
    padding: 6px 10px;
    border-radius: 12px;
    border: 1px solid #E2E8F0 !important;
}

.stTabs [data-baseweb="tab"] {
    height: 40px;
    border-radius: 8px;
    color: #475569 !important;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0 16px;
}

.stTabs [aria-selected="true"] {
    background: #7C3AED !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 10px rgba(124,58,237,0.25) !important;
}

/* ── BOLD DATAFRAME HEADERS WITH COLOR ── */
div[data-testid="stDataFrame"] [role="columnheader"],
div[data-testid="stDataFrame"] th {
    font-weight: 800 !important;
    color: #4C1D95 !important;
    background-color: #F3E8FF !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid #DDD6FE !important;
}
</style>
"""

def inject_custom_css():
    """Injects custom CSS design tokens into Streamlit."""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
