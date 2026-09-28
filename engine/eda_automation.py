"""
engine/eda_automation.py
Automated Exploratory Data Analysis (EDA) Engine:
1. Generates smart chart recommendations per column & feature combination.
2. Explains the analytical rationale ("Why are we making this chart?").
3. Exports a standalone, high-contrast, interactive Plotly EDA Charts Dashboard HTML file.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, List, Any, Tuple
import datetime
from engine.viz import sanitize_fig_for_html

# ── Color Palette ──────────────────────────────────────────────────────────────
BG        = "#FAFAFC"
CARD_BG   = "#FFFFFF"
PURPLE    = "#7C3AED"
CYAN      = "#0891B2"
PINK      = "#DB2777"
GREEN     = "#059669"
AMBER     = "#D97706"
BLUE      = "#2563EB"

PALETTE   = [PURPLE, CYAN, PINK, GREEN, AMBER, BLUE, "#EA580C", "#9333EA"]

LAYOUT_DEFAULTS = dict(
    plot_bgcolor="#FAFAFC",
    paper_bgcolor=CARD_BG,
    font=dict(color="#0F172A", family="Inter, sans-serif", size=12),
    margin=dict(l=45, r=25, t=55, b=45),
    height=420,
)


def _styled_fig(fig, title: str = ""):
    fig.update_layout(**LAYOUT_DEFAULTS)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15, color=PURPLE, family="Inter, sans-serif")))
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0", tickfont=dict(color="#475569", size=11))
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0", tickfont=dict(color="#475569", size=11))
    if any(getattr(t, "type", None) == "parcoords" for t in fig.data):
        fig.update_layout(margin=dict(l=60, r=60, t=75, b=40))
    return fig


def generate_eda_chart_recommendations(df: pd.DataFrame, col_types: Dict[str, str], col_stats: Dict[str, Any], corr_res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates structured chart recommendations with analytical rationales."""
    recommendations = []
    
    numeric_cols = [c for c, t in col_types.items() if t == "numeric" and c in df.columns]
    cat_cols     = [c for c, t in col_types.items() if t in ("categorical", "boolean") and c in df.columns]
    datetime_cols= [c for c, t in col_types.items() if t == "datetime" and c in df.columns]
    text_cols    = [c for c, t in col_types.items() if t == "text" and c in df.columns]

    # 1. UNIVARIATE NUMERIC DISTRIBUTIONS
    for col in numeric_cols:
        col_s = df[col].iloc[:, 0] if isinstance(df[col], pd.DataFrame) else df[col]
        s = pd.to_numeric(col_s, errors="coerce").dropna()
        if len(s) < 4:
            continue
        stats = col_stats.get(col, {})
        skew = stats.get("skewness", 0.0) or 0.0
        mean_v = stats.get("mean", s.mean())
        med_v = stats.get("median", s.median())

        plot_df = pd.DataFrame({col: s})
        fig = px.histogram(plot_df, x=col, nbins=35, marginal="rug", color_discrete_sequence=[PURPLE])
        _styled_fig(fig, f"Distribution Histogram — {col}")

        why_text = f"Analyzes the central tendency (mean={mean_v:.2f}, median={med_v:.2f}) and distribution spread. "
        if abs(skew) > 1.0:
            why_text += f"Highlight: Significant skewness ({skew:.2f}) indicates an asymmetrical distribution requiring log transformation."
        else:
            why_text += f"Highlight: Relatively symmetric distribution (skewness = {skew:.2f})."

        recommendations.append({
            "id": f"num_dist_{col}",
            "title": f"Numeric Distribution — {col}",
            "chart_type": "Histogram & Rug Plot",
            "columns": [col],
            "category": "Univariate Numeric",
            "why": why_text,
            "fig": fig
        })

    # 2. NUMERIC BOXPLOTS & OUTLIER STRIPS
    if len(numeric_cols) >= 2:
        try:
            num_sub = df[numeric_cols].loc[:, ~df[numeric_cols].columns.duplicated()]
            num_sub = num_sub.apply(pd.to_numeric, errors="coerce")
            fig_box = px.box(num_sub, color_discrete_sequence=PALETTE)
            _styled_fig(fig_box, "Numeric Features Boxplot & Outlier Comparison")
            recommendations.append({
                "id": "all_num_boxplot",
                "title": "Numeric Features Variance & Outliers",
                "chart_type": "Comparative Box Plots",
                "columns": numeric_cols,
                "category": "Univariate Numeric",
                "why": "Compares scale differences, interquartile ranges (IQR), and extreme anomalies across all numerical variables.",
                "fig": fig_box
            })
        except Exception:
            pass

    # 3. CATEGORICAL BAR CHARTS
    for col in cat_cols:
        col_s = df[col].iloc[:, 0] if isinstance(df[col], pd.DataFrame) else df[col]
        vc = col_s.value_counts().head(12).reset_index()
        vc.columns = ["Category", "Count"]
        if len(vc) == 0:
            continue

        fig = px.bar(vc, x="Count", y="Category", orientation="h", color="Count", color_continuous_scale=[CARD_BG, CYAN])
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending", "title": col})
        _styled_fig(fig, f"Frequency Distribution — {col}")

        nu = col_s.nunique()
        n_u = int(nu.iloc[0]) if isinstance(nu, (pd.Series, np.ndarray)) else int(nu)
        why_text = f"Visualizes frequency breakdown across top categories. Total unique categories = {n_u}. "
        why_text += "Helps identify dominant categories and potential class imbalance."

        recommendations.append({
            "id": f"cat_bar_{col}",
            "title": f"Value Frequency — {col}",
            "chart_type": "Horizontal Bar Chart",
            "columns": [col],
            "category": "Univariate Categorical",
            "why": why_text,
            "fig": fig
        })

    # 4. CORRELATION MATRIX HEATMAP
    if len(numeric_cols) >= 2 and corr_res.get("pearson") is not None:
        try:
            corr_mat = corr_res["pearson"].fillna(0)
            fig_cm = px.imshow(corr_mat, text_auto=".2f", color_continuous_scale="Viridis", zmin=-1, zmax=1)
            _styled_fig(fig_cm, "Pearson Correlation Heatmap")
            recommendations.append({
                "id": "corr_matrix_heatmap",
                "title": "Pairwise Correlation Matrix",
                "chart_type": "Heatmap",
                "columns": numeric_cols,
                "category": "Multivariate Relationships",
                "why": "Quantifies linear relationships between all numerical features. Red/green highlights indicate strong positive or negative collinearity.",
                "fig": fig_cm
            })
        except Exception:
            pass

    # 5. BIVARIATE SCATTER PLOTS WITH TRENDLINES
    top_pairs = corr_res.get("top_pairs", [])
    for pair in top_pairs[:3]:
        c1, c2, r_val = pair["col1"], pair["col2"], pair["pearson_r"]
        try:
            s1 = df[c1].iloc[:, 0] if isinstance(df[c1], pd.DataFrame) else df[c1]
            s2 = df[c2].iloc[:, 0] if isinstance(df[c2], pd.DataFrame) else df[c2]
            sc_df = pd.DataFrame({c1: pd.to_numeric(s1, errors="coerce"), c2: pd.to_numeric(s2, errors="coerce")}).dropna()

            try:
                fig_sc = px.scatter(sc_df, x=c1, y=c2, trendline="ols", color_discrete_sequence=[PURPLE], trendline_color_override=CYAN)
            except Exception:
                fig_sc = px.scatter(sc_df, x=c1, y=c2, color_discrete_sequence=[PURPLE])

            _styled_fig(fig_sc, f"Bivariate Relationship: {c1} vs {c2} (r = {r_val:.3f})")
            why_text = f"Plots feature interaction between '{c1}' and '{c2}'. Pearson correlation coefficient r = {r_val:.3f}. "
            why_text += f"Reveals direction and strength of dependency."

            recommendations.append({
                "id": f"sc_{c1}_{c2}",
                "title": f"Bivariate Trend — {c1} vs {c2}",
                "chart_type": "Scatter Plot",
                "columns": [c1, c2],
                "category": "Bivariate Relationships",
                "why": why_text,
                "fig": fig_sc
            })
        except Exception:
            pass

    # 6. GROUPED VIOLIN PLOTS (NUMERIC BY CATEGORICAL)
    if numeric_cols and cat_cols:
        for num_c in numeric_cols[:2]:
            for cat_c in cat_cols[:2]:
                try:
                    s_num = df[num_c].iloc[:, 0] if isinstance(df[num_c], pd.DataFrame) else df[num_c]
                    s_cat = df[cat_c].iloc[:, 0] if isinstance(df[cat_c], pd.DataFrame) else df[cat_c]
                    tmp_full = pd.DataFrame({num_c: pd.to_numeric(s_num, errors="coerce"), cat_c: s_cat.astype(str)})

                    top_cats = tmp_full[cat_c].value_counts().head(6).index.tolist()
                    tmp = tmp_full[tmp_full[cat_c].isin(top_cats)].dropna()
                    if len(tmp) > 0:
                        fig_g = px.violin(tmp, x=cat_c, y=num_c, box=True, color=cat_c, color_discrete_sequence=PALETTE)
                        _styled_fig(fig_g, f"{num_c} Distribution across {cat_c}")
                        why_text = f"Groups numerical variable '{num_c}' across categories of '{cat_c}'. "
                        why_text += f"Highlights median shifts, dispersion, and multimodality across groups."

                        recommendations.append({
                            "id": f"group_violin_{num_c}_{cat_c}",
                            "title": f"Grouped Breakdown — {num_c} by {cat_c}",
                            "chart_type": "Grouped Violin Plot",
                            "columns": [num_c, cat_c],
                            "category": "Bivariate Relationships",
                            "why": why_text,
                            "fig": fig_g
                        })
                except Exception:
                    pass

    # 7. CATEGORICAL DONUT CHARTS
    for col in cat_cols[:2]:
        try:
            col_s = df[col].iloc[:, 0] if isinstance(df[col], pd.DataFrame) else df[col]
            vc = col_s.value_counts().head(8).reset_index()
            vc.columns = ["Category", "Count"]
            if len(vc) > 1:
                fig_donut = px.pie(vc, names="Category", values="Count", hole=0.45, color_discrete_sequence=PALETTE)
                _styled_fig(fig_donut, f"Category Share Donut Chart — {col}")
                recommendations.append({
                    "id": f"cat_donut_{col}",
                    "title": f"Proportional Share — {col}",
                    "chart_type": "Donut Ring Chart",
                    "columns": [col],
                    "category": "Univariate Proportions",
                    "why": f"Displays percentage composition and proportional weight across categories in '{col}'.",
                    "fig": fig_donut
                })
        except Exception:
            pass

    # 8. STACKED / PROPORTIONAL BAR CHARTS (CAT vs CAT)
    if len(cat_cols) >= 2:
        c1, c2 = cat_cols[0], cat_cols[1]
        try:
            s1 = df[c1].iloc[:, 0] if isinstance(df[c1], pd.DataFrame) else df[c1]
            s2 = df[c2].iloc[:, 0] if isinstance(df[c2], pd.DataFrame) else df[c2]
            ct_df = pd.DataFrame({c1: s1.astype(str), c2: s2.astype(str)}).dropna()
            
            # Top categories filter
            top1 = ct_df[c1].value_counts().head(6).index
            top2 = ct_df[c2].value_counts().head(6).index
            ct_sub = ct_df[ct_df[c1].isin(top1) & ct_df[c2].isin(top2)]

            if len(ct_sub) > 0:
                fig_stk = px.histogram(ct_sub, x=c1, color=c2, barmode="group", color_discrete_sequence=PALETTE)
                _styled_fig(fig_stk, f"Cross-Categorical Breakdown: {c1} vs {c2}")
                recommendations.append({
                    "id": f"cat_cross_{c1}_{c2}",
                    "title": f"Cross-Categorical Interaction — {c1} vs {c2}",
                    "chart_type": "Grouped Segmented Bar",
                    "columns": [c1, c2],
                    "category": "Multivariate Categorical",
                    "why": f"Evaluates interaction and sub-group proportions between categorical dimensions '{c1}' and '{c2}'.",
                    "fig": fig_stk
                })
        except Exception:
            pass

    # 9. COLOR-CODED MULTIVARIATE SCATTER PLOT (2 NUMERIC + 1 CAT)
    if len(numeric_cols) >= 2 and cat_cols:
        c1, c2 = numeric_cols[0], numeric_cols[1]
        c_cat = cat_cols[0]
        try:
            s1 = pd.to_numeric(df[c1], errors="coerce")
            s2 = pd.to_numeric(df[c2], errors="coerce")
            sc_mult = pd.DataFrame({c1: s1, c2: s2, c_cat: df[c_cat].astype(str)}).dropna()
            top_cats = sc_mult[c_cat].value_counts().head(5).index
            sc_sub = sc_mult[sc_mult[c_cat].isin(top_cats)]
            
            if len(sc_sub) > 0:
                fig_msc = px.scatter(sc_sub, x=c1, y=c2, color=c_cat, color_discrete_sequence=PALETTE, opacity=0.85)
                _styled_fig(fig_msc, f"Multivariate Scatter: {c1} vs {c2} by {c_cat}")
                recommendations.append({
                    "id": f"multi_sc_{c1}_{c2}_{c_cat}",
                    "title": f"Segmented Scatter — {c1} vs {c2} by {c_cat}",
                    "chart_type": "Color-Segmented Scatter Plot",
                    "columns": [c1, c2, c_cat],
                    "category": "Multivariate Relationships",
                    "why": f"Maps 2D numeric space ({c1} vs {c2}) colored by categorical class '{c_cat}' to uncover cluster boundaries.",
                    "fig": fig_msc
                })
        except Exception:
            pass

    # 10. DENSITY HEATMAP (2D NUMERIC DENSITY)
    if len(numeric_cols) >= 2:
        c1, c2 = numeric_cols[0], numeric_cols[1]
        try:
            s1 = pd.to_numeric(df[c1], errors="coerce")
            s2 = pd.to_numeric(df[c2], errors="coerce")
            dh_df = pd.DataFrame({c1: s1, c2: s2}).dropna()
            if len(dh_df) > 10:
                fig_dh = px.density_heatmap(dh_df, x=c1, y=c2, color_continuous_scale="Viridis", nbinsx=25, nbinsy=25)
                _styled_fig(fig_dh, f"2D Bivariate Density — {c1} vs {c2}")
                recommendations.append({
                    "id": f"density_2d_{c1}_{c2}",
                    "title": f"2D Density Heatmap — {c1} vs {c2}",
                    "chart_type": "2D Density Heatmap",
                    "columns": [c1, c2],
                    "category": "Bivariate Density",
                    "why": f"Visualizes data point concentration and multi-modal clusters between '{c1}' and '{c2}'.",
                    "fig": fig_dh
                })
        except Exception:
            pass

    # 11. PARALLEL COORDINATES PLOT (MULTI-NUMERIC SPACE)
    if len(numeric_cols) >= 3:
        try:
            p_cols = numeric_cols[:5]
            p_df = df[p_cols].apply(pd.to_numeric, errors="coerce").dropna()
            if len(p_df) > 5:
                def _fmt_par_lbl(c_name: str) -> str:
                    if len(c_name) <= 20:
                        return c_name
                    words = c_name.split(' ')
                    lines, cur = [], ''
                    for w in words:
                        if cur and len(cur) + len(w) + 1 > 18:
                            lines.append(cur)
                            cur = w
                        else:
                            cur = (cur + ' ' + w).strip()
                    if cur:
                        lines.append(cur)
                    if len(lines) > 2:
                        return lines[0] + '<br>' + lines[1][:14] + '...'
                    return '<br>'.join(lines)

                labels = {c: _fmt_par_lbl(c) for c in p_cols}
                fig_par = px.parallel_coordinates(p_df, color=p_cols[0], color_continuous_scale="Viridis", labels=labels)
                _styled_fig(fig_par, "")
                fig_par.update_layout(margin=dict(t=75, b=40, l=60, r=60))
                recommendations.append({
                    "id": "parallel_coords",
                    "title": "Multi-Dimensional Parallel Coordinates",
                    "chart_type": "Parallel Coordinates Plot",
                    "columns": p_cols,
                    "category": "High-Dimensional EDA",
                    "why": "Traces continuous multi-variable trajectories across numeric features simultaneously to discover global patterns.",
                    "fig": fig_par
                })
        except Exception:
            pass

    # 12. DATETIME / TIME-SERIES LINE PLOT
    if datetime_cols and numeric_cols:
        d_col = datetime_cols[0]
        n_col = numeric_cols[0]
        try:
            ts_df = pd.DataFrame({
                d_col: pd.to_datetime(df[d_col], errors="coerce"),
                n_col: pd.to_numeric(df[n_col], errors="coerce")
            }).dropna().sort_values(d_col)
            if len(ts_df) > 3:
                fig_ts = px.line(ts_df, x=d_col, y=n_col, color_discrete_sequence=[CYAN])
                _styled_fig(fig_ts, f"Time Series Trend — {n_col} over {d_col}")
                recommendations.append({
                    "id": f"ts_line_{n_col}",
                    "title": f"Temporal Trend — {n_col} over Time",
                    "chart_type": "Time Series Line Chart",
                    "columns": [d_col, n_col],
                    "category": "Temporal Analytics",
                    "why": f"Analyzes trend direction, seasonal patterns, and temporal variations in '{n_col}' over date column '{d_col}'.",
                    "fig": fig_ts
                })
        except Exception:
            pass

    return recommendations


def build_eda_standalone_dashboard(df: pd.DataFrame, col_types: Dict[str, str], col_stats: Dict[str, Any], corr_res: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
    """Builds a standalone interactive EDA Charts Dashboard HTML file."""
    now_str = datetime.datetime.now().strftime("%B %d, %Y — %H:%M")
    total_rows = len(df)
    total_cols = len(df.columns)

    # Convert Plotly figures into HTML chart cards
    charts_html = ""
    for rec in recommendations:
        fig = sanitize_fig_for_html(rec["fig"])
        chart_div = pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})
        cols_badges = " ".join([f'<span class="col-pill">{c}</span>' for c in rec["columns"]])
        
        charts_html += f"""
        <div class="eda-card" id="{rec['id']}">
            <div class="card-header">
                <div>
                    <span class="category-badge">{rec['category']}</span>
                    <span class="chart-type-badge">{rec['chart_type']}</span>
                    <h2 class="card-title">{rec['title']}</h2>
                    <div style="margin-top:6px;">Features: {cols_badges}</div>
                </div>
            </div>
            
            <div class="chart-container">
                {chart_div}
            </div>

            <div class="why-box">
                <div class="why-title">❓ Why are we making this chart?</div>
                <div class="why-text">{rec['why']}</div>
            </div>
        </div>
        """

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Iris EDA Charts Dashboard</title>
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

  .header {{
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.18), rgba(6, 182, 212, 0.12));
    border-bottom: 1px solid var(--border);
    padding: 32px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
  }}
  .header h1 {{
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--purple), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
  }}
  .header p {{ color: var(--muted); font-size: 0.92rem; }}
  
  .iris-badge {{
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 30px;
    padding: 10px 24px;
    font-size: 1.2rem;
    color: var(--purple);
    font-weight: 700;
  }}

  main {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 40px 28px 100px;
  }}

  .eda-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 40px;
    box-shadow: 0 12px 36px rgba(0,0,0,0.4);
  }}
  .card-header {{
    border-bottom: 1px solid rgba(139, 92, 246, 0.15);
    padding-bottom: 16px;
    margin-bottom: 20px;
  }}
  .card-title {{
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--text);
    margin-top: 8px;
  }}

  .category-badge {{
    background: rgba(139, 92, 246, 0.15);
    color: var(--purple);
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
  }}
  .chart-type-badge {{
    background: rgba(6, 182, 212, 0.15);
    color: var(--cyan);
    border: 1px solid rgba(6, 182, 212, 0.35);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-left: 6px;
  }}
  .col-pill {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #D1D5DB;
  }}

  .chart-container {{
    min-height: 420px;
    width: 100%;
    background: #0F0F20;
    border-radius: 16px;
    padding: 16px;
    margin: 20px 0;
  }}

  .why-box {{
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(6, 182, 212, 0.06));
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-left: 4px solid var(--purple);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 20px;
  }}
  .why-title {{
    font-weight: 700;
    color: var(--purple);
    font-size: 0.92rem;
    margin-bottom: 4px;
  }}
  .why-text {{
    color: #E5E7EB;
    font-size: 0.9rem;
    line-height: 1.6;
  }}

  footer {{
    text-align: center;
    padding: 40px;
    color: var(--muted);
    font-size: 0.82rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<header class="header">
  <div>
    <h1>📊 Interactive EDA Charts Dashboard</h1>
    <p>Automated EDA Analysis & Visual Insights &nbsp;·&nbsp; {total_rows:,} rows × {total_cols} columns &nbsp;·&nbsp; {now_str}</p>
  </div>
  <div class="iris-badge">🌸 Iris Studio</div>
</header>

<main>
  {charts_html}
</main>

<footer>
  Generated by <strong style="color:var(--purple);">Iris Data Science Studio</strong> &nbsp;·&nbsp; {now_str}
</footer>

</body>
</html>"""
    return html_doc
