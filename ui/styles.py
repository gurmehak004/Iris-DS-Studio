"""
ui/styles.py
CSS design system tokens and custom CSS injector for Iris Data Science Studio.
"""

import streamlit as st

CSS_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Base Theme Dark Palette */
:root {
    --bg-dark: #0A0A14;
    --card-bg: rgba(22, 22, 40, 0.75);
    --card-border: rgba(124, 58, 237, 0.2);
    --accent-violet: #7C3AED;
    --accent-violet-light: #8B5CF6;
    --accent-cyan: #06B6D4;
    --text-main: #F1F5F9;
    --text-muted: #94A3B8;
}

.stApp {
    background-color: var(--bg-dark);
}

/* Glassmorphic Cards */
.glass-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(124, 58, 237, 0.4);
}

/* KPI Card */
.kpi-card {
    background: rgba(22, 22, 40, 0.85);
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-top: 3px solid var(--accent-violet);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-main);
    margin-top: 4px;
}

.kpi-sub {
    font-size: 0.8rem;
    color: #10B981;
    margin-top: 2px;
}

/* Stage Header & Badges */
.stage-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(6, 182, 212, 0.2));
    border: 1px solid rgba(124, 58, 237, 0.4);
    color: #C4B5FD;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.stage-header {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #F8FAFC;
    margin-bottom: 4px;
}

.stage-sub {
    font-size: 0.95rem;
    color: #94A3B8;
    margin-bottom: 24px;
}

/* Iris Says AI Panel */
.iris-insight {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(6, 182, 212, 0.05));
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-left: 4px solid var(--accent-violet);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    gap: 16px;
    align-items: flex-start;
}

.iris-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7C3AED, #06B6D4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 0 12px rgba(124, 58, 237, 0.4);
    flex-shrink: 0;
}

.iris-title {
    font-weight: 700;
    font-size: 0.85rem;
    color: #C4B5FD;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.iris-text {
    font-size: 0.92rem;
    color: #E2E8F0;
    line-height: 1.5;
}

/* Recommendation Pills */
.rec-card-critical {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-left: 4px solid #EF4444;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.rec-card-warning {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-left: 4px solid #F59E0B;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.rec-card-info {
    background: rgba(6, 182, 212, 0.08);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-left: 4px solid #06B6D4;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* Timeline Cleaning Log */
.timeline-item {
    border-left: 2px solid rgba(124, 58, 237, 0.4);
    padding-left: 16px;
    margin-left: 8px;
    margin-bottom: 14px;
    position: relative;
}

.timeline-item::before {
    content: '';
    position: absolute;
    left: -6px;
    top: 4px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #8B5CF6;
}

/* Styled Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 15, 30, 0.5);
    padding: 6px 10px;
    border-radius: 12px;
    border: 1px solid rgba(124, 58, 237, 0.15);
}

.stTabs [data-baseweb="tab"] {
    height: 40px;
    border-radius: 8px;
    color: #94A3B8;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0 16px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.3), rgba(6, 182, 212, 0.2)) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(124, 58, 237, 0.4) !important;
}
</style>
"""

def inject_custom_css():
    """Injects custom CSS design tokens into Streamlit."""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
