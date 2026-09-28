"""
ui/components.py
Reusable UI component functions for Iris Data Science Studio.
Icon-first design: all emojis replaced with inline SVG icons.
"""

import streamlit as st
from typing import Optional, List, Dict, Any


# ── SVG Icon Library ──────────────────────────────────────────────────────────
ICONS = {
    "iris":     '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="1.8"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/></svg>',
    "check":    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "lock":     '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
    "arrow_r":  '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>',
    "chart":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "gear":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
    "bolt":     '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "bulb":     '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="9" y1="18" x2="15" y2="18"/><line x1="10" y1="22" x2="14" y2="22"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/></svg>',
    "trash":    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg>',
    "refresh":  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>',
    "warning":  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "info":     '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#06B6D4" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    "data":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "export":   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "robot":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"/><path d="M12 2v3"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/><circle cx="9" cy="16" r="1" fill="currentColor"/><circle cx="15" cy="16" r="1" fill="currentColor"/></svg>',
    "target":   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "broom":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 21l9-9"/><path d="M12.22 6.22L15 3l6 6-3.22 2.78"/><path d="M18 11.5V22H6V11.5"/><path d="M6 11.5l6-6 6 6"/></svg>',
    "share":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>',
    "predict":  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "segment":  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "detect":   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "explore":  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
    "upload":   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>',
    "health":   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M20.42 4.58a5.4 5.4 0 0 0-7.65 0l-.77.78-.77-.78a5.4 5.4 0 0 0-7.65 7.65l.77.78L12 21l7.65-7.99.77-.78a5.4 5.4 0 0 0 0-7.65z"/></svg>',
    "missing":  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    "outlier":  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polygon points="12 2 2 19 22 19"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "struct":   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="4 7 4 4 20 4 20 7"/><line x1="9" y1="20" x2="15" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/></svg>',
    "encode":   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
    "scale":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
    "reduce":   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/></svg>',
    "dupe":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
    "corr":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 3l18 18"/><path d="M3 21C3 12 12 3 21 3"/></svg>',
    "fe":       '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>',
    "chat":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
}


def _icon(name: str, color: str = "currentColor", size: int = 14) -> str:
    """Returns an SVG icon string with optional color/size override."""
    svg = ICONS.get(name, ICONS["info"])
    svg = svg.replace('stroke="currentColor"', f'stroke="{color}"')
    for orig_size in [10, 12, 13, 14, 20, 22]:
        svg = svg.replace(f'width="{orig_size}" height="{orig_size}"', f'width="{size}" height="{size}"')
    return svg


# ── Components ────────────────────────────────────────────────────────────────

def scroll_to_top():
    """Scrolls Streamlit's main content viewport to the top after a stage change."""
    st.components.v1.html("""
    <script>
        try {
            const parentWindow = window.parent;
            const scrollMain = () => {
                const doc = parentWindow.document;
                const main = doc.querySelector('[data-testid="stMain"]');
                const appView = doc.querySelector('[data-testid="stAppViewContainer"]');
                if (main) {
                    main.scrollTop = 0;
                }
                if (appView) appView.scrollTop = 0;
                doc.documentElement.scrollTop = 0;
                parentWindow.scrollTo(0, 0);
            };
            let attempts = 0;
            const ensureTop = () => {
                scrollMain();
                attempts += 1;
                if (attempts < 8) parentWindow.setTimeout(ensureTop, 100);
            };
            parentWindow.requestAnimationFrame(() => parentWindow.requestAnimationFrame(ensureTop));
        } catch(e) { console.log(e); }
    </script>
    """, height=0, width=0)


def render_styled_dataframe(df, **kwargs):
    """Renders a dataframe with a high-contrast violet header and subtle row bands."""
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#6D28D9"),
            ("color", "#FFFFFF"),
            ("font-weight", "800"),
            ("border-bottom", "2px solid #4C1D95"),
            ("text-align", "left"),
        ]},
        {"selector": "td", "props": [
            ("color", "#1E293B"),
            ("border-bottom", "1px solid #EDE9FE"),
        ]},
        {"selector": "tbody tr:nth-child(even) td", "props": [
            ("background-color", "#F8F6FF"),
        ]},
    ])
    st.dataframe(styled_df, **kwargs)


def render_kpi_card(label: str, value: str, subtext: Optional[str] = None, accent_color: str = "#7C3AED"):
    """Renders an enterprise glassmorphic KPI card."""
    sub_html = f'<div class="kpi-sub" style="color:{accent_color} !important; font-weight:600; font-size:0.78rem; margin-top:3px;">{subtext}</div>' if subtext else ''
    html = f"""
    <div class="kpi-card" style="border-top:3px solid {accent_color} !important; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:16px 20px; box-shadow:0 4px 14px rgba(0,0,0,0.03);">
        <div class="kpi-label" style="font-size:0.72rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; color:#64748B !important;">{label}</div>
        <div class="kpi-value" style="font-size:1.8rem; font-weight:800; color:#0F172A !important; margin-top:4px;">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_iris_says(message: str, title: str = "Iris Insights"):
    """Renders the Iris Says AI Assistant panel with high readability."""
    html = f"""
    <div class="iris-insight">
        <div class="iris-avatar">{ICONS['iris']}</div>
        <div>
            <div class="iris-title" style="color:#6D28D9 !important; font-weight:800; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">{title}</div>
            <div class="iris-text" style="color:#0F172A !important; font-size:0.92rem; line-height:1.6;">{message}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_stage_header(stage_num: int, title: str, subtitle: str):
    """Renders a stage badge and header, scrolling only after a stage transition."""
    if st.session_state.pop("_scroll_to_stage_top", False):
        scroll_to_top()
    html = f"""
    <div class="stage-badge" style="background:#7C3AED1A; color:#6D28D9; border:1px solid #7C3AED40; font-weight:700;">Stage {stage_num} of 5</div>
    <div class="h-page" style="margin-bottom:8px; color:#0F172A !important; font-size:2.2rem; font-weight:900;">{title}</div>
    <div class="h-caption" style="color:#475569 !important; max-width:720px; line-height:1.7; margin-bottom:28px; font-size:0.95rem;">{subtitle}</div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title: str, icon_name: str = "chart", badge: Optional[str] = None, badge_color: str = "#7C3AED"):
    """Renders a full-width structured section header with icon, title and optional right badge."""
    icon_svg = ICONS.get(icon_name, ICONS["chart"])
    badge_html = f'<span style="background:{badge_color}1A; color:{badge_color}; border:1px solid {badge_color}50; border-radius:20px; padding:4px 14px; font-size:0.72rem; font-weight:800; letter-spacing:0.06em; text-transform:uppercase;">{badge}</span>' if badge else ""
    html = f'<div style="display:flex; align-items:center; justify-content:space-between; margin:28px 0 16px; padding-bottom:12px; border-bottom:2px solid #F1F5F9;"><div style="display:flex; align-items:center; gap:10px;"><span style="color:#7C3AED; display:flex; align-items:center;">{icon_svg}</span><span class="h-section" style="color:#0F172A !important; font-weight:800; font-size:1.25rem;">{title}</span></div>{badge_html}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_health_scorecard(score: int, missing_cols: int, dup_rows: int, outlier_cols: int):
    """Renders the Stage 2 Data Health Scorecard with vibrant, clean tinted cards (zero grey panels)."""
    color = "#059669" if score >= 80 else "#D97706" if score >= 60 else "#DC2626"
    bg_tint = "rgba(5,150,105,0.08)" if score >= 80 else "rgba(217,119,6,0.08)" if score >= 60 else "rgba(220,38,38,0.08)"

    def _cell(label, value, zero_label, issue_label, ok_color="#059669", warn_color="#D97706"):
        if value == 0:
            dot = f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{ok_color};margin-right:8px;box-shadow:0 0 8px {ok_color}60;"></span>'
            val = f'{dot}<span style="color:{ok_color};font-weight:800;font-size:0.95rem;">{zero_label}</span>'
            c_bg = "rgba(5,150,105,0.08)"
            c_border = "rgba(5,150,105,0.25)"
        else:
            dot = f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{warn_color};margin-right:8px;box-shadow:0 0 8px {warn_color}60;"></span>'
            val = f'{dot}<span style="color:{warn_color};font-weight:800;font-size:0.95rem;">{value} {issue_label}</span>'
            c_bg = "rgba(217,119,6,0.08)"
            c_border = "rgba(217,119,6,0.25)"
        return f'<div style="background:{c_bg};border:1px solid {c_border};border-top:3px solid {ok_color if value == 0 else warn_color};padding:16px 18px;border-radius:8px;min-height:94px;"><div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.07em;font-weight:800;margin-bottom:8px;">{label}</div><div style="display:flex;align-items:center;">{val}</div></div>'

    html = f"""
    <div style="background:linear-gradient(115deg,#FFFFFF 55%,#F7F3FF); border:1px solid #DDD6FE; border-left:5px solid {color}; border-radius:10px; padding:22px 24px; box-shadow:0 6px 20px rgba(76,29,149,0.07); margin-bottom:24px;">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom:18px;">
            <div style="display:flex; align-items:center; gap:14px;">
                <div style="width:42px; height:42px; border-radius:50%; background:{color}18; display:flex; align-items:center; justify-content:center; border:1.5px solid {color}40;">
                    {_icon('health', color, 20)}
                </div>
                <div>
                    <div class="h-subsection" style="color:#0F172A !important; font-weight:800; font-size:1.2rem;">Data Quality & Health Scorecard</div>
                    <div style="font-size:0.82rem; color:#475569 !important; font-weight:500;">Automated data audit across missing values, duplicates, and extreme outliers</div>
                </div>
            </div>
            <div style="background:{bg_tint}; border:1px solid {color}50; padding:7px 18px; border-radius:8px; text-align:center; min-width:112px;">
                <span style="font-size:2rem; font-weight:900; color:{color}; letter-spacing:-0.03em;">{score}</span>
                <span style="font-size:0.9rem; color:#475569; font-weight:700;"> / 100</span>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px;">
            {_cell("Missing Values", missing_cols, "Clean (0 missing)", "columns with nulls")}
            {_cell("Duplicate Rows", dup_rows, "Clean (0 duplicates)", "duplicate rows")}
            {_cell("Outlier Columns", outlier_cols, "Clean (0 extreme outliers)", "columns with outliers", warn_color="#DC2626")}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_recommendation_card(priority: str, title: str, reason: str, col_name: Optional[str] = None):
    """Renders a prioritized recommendation card with dot-based priority indicators (no emojis)."""
    priority_map = {
        "CRITICAL": ("#DC2626", "rec-card-critical"),
        "HIGH":     ("#D97706", "rec-card-warning"),
        "MEDIUM":   ("#0891B2", "rec-card-info"),
        "LOW":      ("#64748B", "rec-card-info"),
    }
    p_color, card_class = priority_map.get(priority, ("#64748B", "rec-card-info"))

    col_html = f"""<span style="background:rgba(124,58,237,0.1); color:#6D28D9; padding:2px 10px;
                               border-radius:6px; font-size:0.78rem; font-family:monospace; font-weight:700;
                               border:1px solid rgba(124,58,237,0.3); margin-left:8px;">{col_name}</span>""" if col_name else ""

    html = f"""
    <div class="{card_class}" style="margin-bottom:12px; background:#FFFFFF; border-radius:12px; padding:16px 20px; box-shadow:0 2px 10px rgba(0,0,0,0.03);">
        <div style="display:flex; align-items:flex-start; gap:12px;">
            <span style="display:inline-block; width:10px; height:10px; border-radius:50%;
                         background:{p_color}; margin-top:5px; flex-shrink:0; box-shadow:0 0 8px {p_color}60;"></span>
            <div style="flex:1;">
                <div class="h-card" style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:0.95rem;">{title}{col_html}</div>
                <div class="h-caption" style="color:#334155; line-height:1.6; font-size:0.85rem;">{reason}</div>
            </div>
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

