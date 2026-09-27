"""
ui/components.py
Reusable UI component functions for Iris Data Science Studio.
"""

import streamlit as st
from typing import Optional, List, Dict, Any


def render_kpi_card(label: str, value: str, subtext: Optional[str] = None, accent_color: str = "#7C3AED"):
    """Renders an enterprise glassmorphic KPI card."""
    sub_html = f'<div class="kpi-sub">{subtext}</div>' if subtext else ''
    html = f"""
    <div class="kpi-card" style="border-top-color: {accent_color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_iris_says(message: str, title: str = "Iris Insights"):
    """Renders the Iris Says AI Assistant panel."""
    html = f"""
    <div class="iris-insight">
        <div class="iris-avatar">🌸</div>
        <div>
            <div class="iris-title">{title}</div>
            <div class="iris-text">{message}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_stage_header(stage_num: int, title: str, subtitle: str):
    """Renders a stage badge and header."""
    html = f"""
    <div class="stage-badge">Stage {stage_num} of 5</div>
    <div class="stage-header">{title}</div>
    <div class="stage-sub">{subtitle}</div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_health_scorecard(score: int, missing_cols: int, dup_rows: int, outlier_cols: int):
    """Renders the Stage 2 Data Health Scorecard."""
    color = "#10B981" if score >= 80 else "#F59E0B" if score >= 60 else "#EF4444"
    status_missing = "✅ Clean" if missing_cols == 0 else f"⚠️ {missing_cols} columns"
    status_dup = "✅ 0 duplicates" if dup_rows == 0 else f"⚠️ {dup_rows} rows"
    status_outliers = "✅ Clean" if outlier_cols == 0 else f"🔴 {outlier_cols} columns"
    
    html = f"""
    <div class="glass-card" style="border-left: 4px solid {color}; padding: 18px 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC;">🏥 Data Health Scorecard</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {color};">{score} <span style="font-size: 0.9rem; color: #94A3B8;">/ 100</span></div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; text-align: center;">
            <div style="background: rgba(15, 15, 30, 0.6); padding: 10px; border-radius: 8px;">
                <div style="font-size: 0.75rem; color: #94A3B8; uppercase;">Missing Values</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #E2E8F0; margin-top: 4px;">{status_missing}</div>
            </div>
            <div style="background: rgba(15, 15, 30, 0.6); padding: 10px; border-radius: 8px;">
                <div style="font-size: 0.75rem; color: #94A3B8; uppercase;">Duplicates</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #E2E8F0; margin-top: 4px;">{status_dup}</div>
            </div>
            <div style="background: rgba(15, 15, 30, 0.6); padding: 10px; border-radius: 8px;">
                <div style="font-size: 0.75rem; color: #94A3B8; uppercase;">Outliers Detected</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #E2E8F0; margin-top: 4px;">{status_outliers}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_recommendation_card(priority: str, title: str, reason: str, col_name: Optional[str] = None):
    """Renders a prioritized recommendation pill card."""
    card_class = "rec-card-critical" if priority == "CRITICAL" else "rec-card-warning" if priority == "HIGH" else "rec-card-info"
    icon = "🔴" if priority == "CRITICAL" else "🟡" if priority == "HIGH" else "💡"
    
    col_str = f" <span style='background: rgba(124, 58, 237, 0.2); color: #A78BFA; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-family: monospace;'>{col_name}</span>" if col_name else ""
    
    html = f"""
    <div class="{card_class}">
        <div style="font-weight: 700; font-size: 0.95rem; color: #F8FAFC; margin-bottom: 4px;">
            {icon} {title}{col_str}
        </div>
        <div style="font-size: 0.85rem; color: #CBD5E1;">
            {reason}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_data_preview_table(df, key_prefix: str = "preview"):
    """
    Renders an interactive DataFrame preview table with Head, Tail, and Random Sample toggles.
    """
    if df is None:
        return

    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        view_mode = st.radio(
            "Data View Mode:",
            ["Head (First Rows)", "Tail (Last Rows)", "Random Sample"],
            horizontal=True,
            key=f"{key_prefix}_view_mode"
        )
    with p_col2:
        num_rows = st.selectbox("Rows:", [5, 10, 20, 50], index=1, key=f"{key_prefix}_row_count")

    if "Head" in view_mode:
        display_df = df.head(num_rows)
    elif "Tail" in view_mode:
        display_df = df.tail(num_rows)
    else:
        display_df = df.sample(min(num_rows, len(df)), random_state=42)

    st.dataframe(display_df, use_container_width=True)

