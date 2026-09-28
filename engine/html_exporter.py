"""
engine/html_exporter.py
Generates a stunning, vibrant, high-contrast single-file HTML Data Science Dashboard
with embedded interactive Plotly charts, CRISP-DM lifecycle structure, feature engineering guide,
and AI-driven insights.
"""
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, Any, Optional, List, Tuple
import datetime
from engine.viz import sanitize_fig_for_html

# ── Color Palette ──────────────────────────────────────────────────────────────
BG        = "#090912"
CARD_BG   = "#151528"
PURPLE    = "#8B5CF6"
CYAN      = "#06B6D4"
PINK      = "#EC4899"
GREEN     = "#10B981"
AMBER     = "#F59E0B"
BLUE      = "#3B82F6"

VIBRANT_PALETTE = [PURPLE, CYAN, PINK, GREEN, AMBER, BLUE, "#F97316", "#A855F7"]

LAYOUT_DEFAULTS = dict(
    plot_bgcolor="#0F0F20",
    paper_bgcolor=CARD_BG,
    font=dict(color="#F3F4F6", family="Inter, -apple-system, sans-serif", size=12),
    margin=dict(l=50, r=30, t=60, b=50),
    height=440,
)


def _styled_fig(fig, title: str = ""):
    """Apply high-contrast dark theme to any Plotly figure."""
    fig.update_layout(**LAYOUT_DEFAULTS)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16, color=PURPLE, family="Inter, sans-serif")))
    fig.update_xaxes(gridcolor="#2A2A48", zerolinecolor="#2A2A48", tickfont=dict(color="#D1D5DB", size=11))
    fig.update_yaxes(gridcolor="#2A2A48", zerolinecolor="#2A2A48", tickfont=dict(color="#D1D5DB", size=11))
    return fig


def _fig_to_html(fig) -> str:
    """Convert Plotly figure to HTML string wrapped in a explicit-height card."""
    fig = sanitize_fig_for_html(fig)
    chart_html = pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})
    return f'<div class="chart-card"><div class="chart-wrapper">{chart_html}</div></div>'


def _stat_grid(items: List[Tuple[str, Any, str]]) -> str:
    """Render KPI metric grid cards."""
    cards = ""
    for label, value, sub in items:
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        cards += f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>"""
    return f'<div class="kpi-grid">{cards}</div>'


def _render_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or len(df) == 0:
        return '<p style="color:#9CA3AF;">No data available.</p>'
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, row in df.head(max_rows).iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows += f"<tr>{cells}</tr>"
    return f"""
    <div class="table-wrap">
        <table>
            <thead><tr>{headers}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>"""


def build_html_dashboard(
    meta: Dict,
    col_types: Dict[str, str],
    quality: Dict,
    col_stats: Dict,
    corr: Dict,
    ml_results: Dict,
    iris_says: Dict,
    df: pd.DataFrame,
) -> str:
    now_str = datetime.datetime.now().strftime("%B %d, %Y — %H:%M")
    cleaned_rows = meta.get("cleaned_rows", len(df))
    cleaned_cols = meta.get("cleaned_cols", len(df.columns))
    orig_filename = meta.get("original_filename", "dataset.csv")

    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isnull().sum().sum())
    completeness = round((1 - total_missing / max(total_cells, 1)) * 100, 1)

    type_counts = {}
    for t in col_types.values():
        type_counts[t] = type_counts.get(t, 0) + 1

    # Build Type Chips
    type_chips = ""
    chip_colors = {
        "numeric": "#8B5CF6", "categorical": "#06B6D4", "datetime": "#F59E0B",
        "text": "#EC4899", "boolean": "#3B82F6", "id": "#94A3B8"
    }
    for t, c in type_counts.items():
        color = chip_colors.get(t, "#888")
        type_chips += f'<span class="type-badge" style="background:{color}22; color:{color}; border:1px solid {color}55;">{t} ({c})</span> '

    # ── 1. BUSINESS UNDERSTANDING & DATA COLLECTION ──
    sec_overview = _stat_grid([
        ("Dataset", orig_filename, "Source File"),
        ("Rows", f"{cleaned_rows:,}", "Total Records"),
        ("Columns", f"{cleaned_cols}", "Features Detected"),
        ("Completeness", f"{completeness}%", f"{total_missing:,} missing cells"),
        ("Memory Usage", f"{meta.get('memory_mb', '?')} MB", "RAM Footprint"),
        ("Numeric Features", type_counts.get("numeric", 0), "Continuous/Discrete"),
        ("Categorical Features", type_counts.get("categorical", 0), "Nominal/Ordinal"),
    ])
    sec_overview += f'<div style="margin-top:16px;">{type_chips}</div>'
    if iris_says.get("overview"):
        sec_overview += f'<div class="iris-insight">🌸 <strong>Iris Says:</strong> {iris_says["overview"]}</div>'

    # ── 2. DATA CLEANING & PROCESSING ──
    dup_info = quality.get("duplicates", {})
    dup_rows = dup_info.get("duplicate_rows", 0)
    
    sec_cleaning = ""
    if total_missing == 0:
        sec_cleaning += '<div class="success-banner">✅ Data Quality Clean: No missing values found in dataset!</div>'
    else:
        miss_df = quality.get("missing", pd.DataFrame())
        sec_cleaning += '<h3 class="sub-header">Missing Values Summary</h3>'
        sec_cleaning += _render_table(miss_df)

    outlier_df = quality.get("outliers", pd.DataFrame())
    if len(outlier_df) > 0:
        sec_cleaning += '<h3 class="sub-header">Outliers Summary (IQR Method)</h3>'
        sec_cleaning += _render_table(outlier_df)

    skew_df = quality.get("skewness", pd.DataFrame())
    if len(skew_df) > 0:
        sec_cleaning += '<h3 class="sub-header">Feature Skewness & Kurtosis</h3>'
        sec_cleaning += _render_table(skew_df)

    if iris_says.get("quality"):
        sec_cleaning += f'<div class="iris-insight">🌸 <strong>Iris Says:</strong> {iris_says["quality"]}</div>'

    # ── 3. EXPLORATORY DATA ANALYSIS (EDA) ──
    sec_eda = ""
    # Filter out engineered suffixes to keep the EDA section clean and focused on original data
    ENGINEERED_SUFFIXES = ("_scaled", "_minmax", "_encoded", "_binary", "_log", "_log1p", "_zscore")
    numeric_cols = [c for c, t in col_types.items() if t == "numeric" and not any(c.endswith(sfx) for sfx in ENGINEERED_SUFFIXES)]
    cat_cols = [c for c, t in col_types.items() if t in ("categorical", "boolean") and not any(c.endswith(sfx) for sfx in ENGINEERED_SUFFIXES)]


    # Numeric Chart Layout Enhancement
    if numeric_cols:
        sec_eda += '<h3 class="sub-header">Numeric Feature Distributions & Statistics</h3>'
        clean_num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        
        # Present each important feature as its own high-quality wide chart rather than a cramped grid
        for i, col in enumerate(numeric_cols[:6]): # Limit to top 6 to prevent huge files
            try:
                s = clean_num_df[col].dropna()
                if len(s) > 0:
                    fig_num = go.Figure(data=[go.Histogram(
                        x=s, 
                        marker_color=VIBRANT_PALETTE[i % len(VIBRANT_PALETTE)],
                        nbinsx=40,
                        opacity=0.85
                    )])
                    
                    mean_val = s.mean()
                    median_val = s.median()
                    fig_num.add_vline(
                      x=mean_val, line_dash="dash", line_color="#E2E8F0",
                      annotation_text="Mean", annotation_position="top left",
                    )
                    fig_num.add_vline(
                      x=median_val, line_dash="dot", line_color="#9CA3AF",
                      annotation_text="Median", annotation_position="top right",
                    )
                    
                    _styled_fig(fig_num, f"Distribution: {col}")
                    fig_num.update_layout(height=320, bargap=0.1)
                    sec_eda += _fig_to_html(fig_num)
            except Exception:
                pass


        # Boxplots for outliers
        try:
            fig_box = px.box(clean_num_df.dropna(how="all"), color_discrete_sequence=VIBRANT_PALETTE)
            _styled_fig(fig_box, "Numeric Features Box Plots & Outliers")
            sec_eda += _fig_to_html(fig_box)
        except Exception:
            pass

    # Categorical Bar Charts
    if cat_cols:
        sec_eda += '<h3 class="sub-header">Categorical Value Distributions</h3>'
        for col in cat_cols[:4]:
            vc = df[col].astype(str).value_counts().head(12).reset_index()
            vc.columns = [col, "Count"]
            fig_cat = px.bar(
                vc, x="Count", y=col, orientation="h",
                color="Count", color_continuous_scale=[CARD_BG, CYAN]
            )
            fig_cat.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
            _styled_fig(fig_cat, f"Top Values — {col}")
            sec_eda += _fig_to_html(fig_cat)

    # ── 4. CORRELATIONS & RELATIONSHIPS ──
    sec_corr = ""
    if len(numeric_cols) >= 2:
        try:
            clean_corr_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            corr_mat = clean_corr_df.corr().round(3)
            fig_cm = px.imshow(
                corr_mat, text_auto=True, color_continuous_scale="Viridis",
                zmin=-1, zmax=1
            )
            _styled_fig(fig_cm, "Pearson Correlation Heatmap")
            sec_corr += _fig_to_html(fig_cm)
            
            # Top pair scatter
            pairs = []
            cols_l = list(corr_mat.columns)
            for i in range(len(cols_l)):
                for j in range(i+1, len(cols_l)):
                    r_val = corr_mat.iloc[i, j]
                    if not np.isnan(r_val):
                        pairs.append((abs(r_val), cols_l[i], cols_l[j], r_val))
            pairs.sort(reverse=True)
            if pairs:
                _, c1_name, c2_name, r_v = pairs[0]
                sc_df = pd.DataFrame({
                    c1_name: pd.to_numeric(df[c1_name], errors="coerce"),
                    c2_name: pd.to_numeric(df[c2_name], errors="coerce")
                }).dropna()
                fig_sc = px.scatter(
                    sc_df, x=c1_name, y=c2_name,
                    color_discrete_sequence=[PURPLE]
                )
                _styled_fig(fig_sc, f"Strongest Pair: {c1_name} vs {c2_name} (Pearson r = {r_v:.3f})")
                sec_corr += _fig_to_html(fig_sc)
        except Exception:
            sec_corr += '<p style="color:#9CA3AF;">Correlation analysis unavailable.</p>'
    else:
        sec_corr = '<p style="color:#9CA3AF;">Need at least 2 numeric features for correlation analysis.</p>'


    if iris_says.get("correlations"):
        sec_corr += f'<div class="iris-insight">🌸 <strong>Iris Says:</strong> {iris_says["correlations"]}</div>'

    # ── 5. FEATURE ENGINEERING GUIDE ──
    fe_rows = []
    for col, ct in col_types.items():
        s = (col_stats or {}).get(col, {})
        actions = []
        if ct == "numeric":
            sk = s.get("skewness", 0) or 0
            if abs(sk) > 1.0:
                actions.append(f"⚡ Log Transform (skew={sk:.2f})")
            actions += ["StandardScaler (Z-Score)", "MinMaxScaler [0,1]", "Quantile Binning"]
        elif ct in ("categorical", "boolean"):
            n_u = s.get("unique", 0)
            actions.append("Binary Encode (0/1)" if n_u == 2 else (f"One-Hot Encode ({n_u} dummy cols)" if n_u <= 10 else f"Frequency Encode ({n_u} cats)"))
        elif ct == "datetime":
            actions += ["Extract Year, Month, Day", "Day of Week & IsWeekend"]
        elif ct == "text":
            actions += ["Extract Character Length", "Extract Word Count"]
        elif ct == "id":
            actions.append("Drop Identifier Feature")
            
        fe_rows.append({
            "Column Name": col,
            "Detected Type": ct,
            "Recommended Transformations": " • ".join(actions) or "—"
        })
    fe_df = pd.DataFrame(fe_rows)
    sec_fe = '<p style="color:#9CA3AF; margin-bottom:16px;">Tailored feature engineering strategy to transform raw variables into highly predictive ML features.</p>'
    sec_fe += _render_table(fe_df, max_rows=60)

    # ── ASSEMBLE FULL HTML DOCUMENT ──
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Iris Data Science Dashboard — {orig_filename}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: {BG};
    --card: {CARD_BG};
    --purple: {PURPLE};
    --cyan: {CYAN};
    --pink: {PINK};
    --text: #F3F4F6;
    --muted: #9CA3AF;
    --border: rgba(139, 92, 246, 0.25);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 14px;
    line-height: 1.6;
  }}
  
  /* ── Header Bar ── */
  .header {{
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.18) 0%, rgba(6, 182, 212, 0.12) 100%);
    border-bottom: 1px solid var(--border);
    padding: 32px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
  }}
  .header-left h1 {{
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--purple), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px;
  }}
  .header-left p {{ color: var(--muted); font-size: 0.9rem; }}
  .iris-badge {{
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 30px;
    padding: 10px 24px;
    font-size: 1.2rem;
    color: var(--purple);
    font-weight: 700;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.2);
  }}

  /* ── Sticky Navigation ── */
  nav {{
    position: sticky;
    top: 0;
    z-index: 1000;
    background: rgba(9, 9, 18, 0.92);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 0 48px;
    display: flex;
    gap: 4px;
    overflow-x: auto;
  }}
  nav a {{
    display: inline-block;
    padding: 14px 20px;
    color: var(--muted);
    text-decoration: none;
    font-size: 0.88rem;
    font-weight: 600;
    border-bottom: 3px solid transparent;
    transition: all 0.2s ease;
    white-space: nowrap;
  }}
  nav a:hover {{
    color: var(--purple);
    border-bottom-color: var(--purple);
  }}

  /* ── Main Layout ── */
  main {{
    max-width: 1320px;
    margin: 0 auto;
    padding: 40px 32px 100px;
  }}
  
  .section {{
    margin-bottom: 56px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px 36px;
    scroll-margin-top: 70px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
  }}
  .section-title {{
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--purple);
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid rgba(139, 92, 246, 0.15);
    padding-bottom: 16px;
  }}
  
  .sub-header {{
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--cyan);
    margin: 28px 0 14px;
  }}

  /* ── KPI Cards ── */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }}
  .kpi-card {{
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 14px;
    padding: 16px 18px;
    transition: transform 0.2s ease, border-color 0.2s ease;
  }}
  .kpi-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(139, 92, 246, 0.5);
  }}
  .kpi-label {{
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    margin-bottom: 6px;
  }}
  .kpi-value {{
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--cyan);
  }}
  .kpi-sub {{
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 4px;
  }}

  /* ── Chart Wrappers (CRITICAL FIX FOR VISIBILITY) ── */
  .chart-card {{
    background: #0F0F20;
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 16px;
    padding: 16px;
    margin: 20px 0 28px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
  }}
  .chart-wrapper {{
    min-height: 440px;
    width: 100%;
  }}

  /* ── Type Badges ── */
  .type-badge {{
    display: inline-block;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px;
  }}

  /* ── Insights & Banners ── */
  .iris-insight {{
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.12), rgba(6, 182, 212, 0.08));
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-left: 4px solid var(--purple);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 20px;
    color: #E5E7EB;
    font-size: 0.92rem;
    line-height: 1.7;
  }}
  .success-banner {{
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 12px;
    padding: 14px 18px;
    color: #10B981;
    font-weight: 600;
    margin-bottom: 20px;
  }}

  /* ── Data Tables ── */
  .table-wrap {{
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid rgba(139, 92, 246, 0.2);
    margin: 12px 0 20px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  thead tr {{
    background: rgba(139, 92, 246, 0.15);
  }}
  th {{
    padding: 12px 16px;
    text-align: left;
    color: var(--purple);
    font-weight: 700;
    white-space: nowrap;
    border-bottom: 1px solid rgba(139, 92, 246, 0.25);
  }}
  td {{
    padding: 10px 16px;
    color: #E5E7EB;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  }}
  tr:hover td {{
    background: rgba(139, 92, 246, 0.06);
  }}

  /* ── Footer ── */
  footer {{
    text-align: center;
    padding: 40px;
    color: var(--muted);
    font-size: 0.82rem;
    border-top: 1px solid var(--border);
    margin-top: 60px;
  }}

  @media (max-width: 768px) {{
    .header {{ padding: 24px; flex-direction: column; text-align: center; }}
    nav {{ padding: 0 16px; }}
    main {{ padding: 24px 16px 60px; }}
    .section {{ padding: 20px 18px; }}
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>

<header class="header">
  <div class="header-left">
    <h1>🌸 Iris Data Science Dashboard</h1>
    <p>Dataset: <strong style="color:var(--text);">{orig_filename}</strong>
       &nbsp;·&nbsp; Generated: {now_str}
       &nbsp;·&nbsp; {cleaned_rows:,} rows × {cleaned_cols} columns</p>
  </div>
  <div class="iris-badge">🌸 Iris Studio</div>
</header>

<nav>
  <a href="#s-business">📋 Business & Collection</a>
  <a href="#s-cleaning">🧹 Data Cleaning</a>
  <a href="#s-eda">📊 Exploratory Analysis</a>
  <a href="#s-corr">🔗 Correlations</a>
  <a href="#s-fe">⚙️ Feature Engineering</a>
</nav>

<main>
  <section class="section" id="s-business">
    <div class="section-title"><span>📋</span> Business Understanding & Data Collection</div>
    {sec_overview}
  </section>

  <section class="section" id="s-cleaning">
    <div class="section-title"><span>🧹</span> Data Cleaning and Processing</div>
    {sec_cleaning}
  </section>

  <section class="section" id="s-eda">
    <div class="section-title"><span>📊</span> Exploratory Data Analysis (EDA)</div>
    {sec_eda}
  </section>

  <section class="section" id="s-corr">
    <div class="section-title"><span>🔗</span> Correlations & Relationships</div>
    {sec_corr}
  </section>

  <section class="section" id="s-fe">
    <div class="section-title"><span>⚙️</span> Feature Engineering Strategy</div>
    {sec_fe}
  </section>
</main>

<footer>
  Generated by <strong style="color:var(--purple);">Iris Data Science Studio</strong>
  &nbsp;·&nbsp; {now_str}
</footer>

</body>
</html>"""
    return html_doc
