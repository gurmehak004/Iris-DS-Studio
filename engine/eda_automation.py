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
from typing import Dict, List, Any, Tuple, Optional
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
        # Shorten title so it never stretches across or collides with top modebar buttons
        short_title = title if len(title) <= 42 else title[:39] + "..."
        fig.update_layout(
            title=dict(
                text=short_title,
                font=dict(size=13, color=PURPLE, family="Inter, sans-serif"),
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top"
            ),
            margin=dict(l=45, r=25, t=65, b=45)
        )
    else:
        fig.update_layout(margin=dict(l=45, r=25, t=35, b=45))
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0", tickfont=dict(color="#475569", size=11))
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0", tickfont=dict(color="#475569", size=11))
    if any(getattr(t, "type", None) == "parcoords" for t in fig.data):
        fig.update_layout(margin=dict(l=60, r=60, t=75, b=40))
    return fig


def generate_eda_chart_recommendations(
    df: pd.DataFrame,
    col_types: Dict[str, str],
    col_stats: Dict[str, Any],
    corr_res: Dict[str, Any],
    target_col: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Generates a focused, curated set of high-value EDA charts with statistical rationales."""
    recommendations = []
    
    # ── Base Column Filtering ───────────────────────────────────────────────────
    # Exclude redundant derived columns (_minmax, _scaled, _encoded) when original base feature exists
    raw_numeric  = [c for c, t in col_types.items() if t == "numeric" and c in df.columns]
    raw_cat      = [c for c, t in col_types.items() if t in ("categorical", "boolean") and c in df.columns]
    datetime_cols= [c for c, t in col_types.items() if t == "datetime" and c in df.columns]

    derived_suffixes = ("_minmax", "_scaled", "_standardized", "_zscore", "_normalized", "_encoded", "_ohe", "_label")
    numeric_cols = []
    for c in raw_numeric:
        is_derived = False
        for sfx in derived_suffixes:
            if c.endswith(sfx) and c[:-len(sfx)] in raw_numeric:
                is_derived = True
                break
        if not is_derived:
            numeric_cols.append(c)

    cat_cols = []
    for c in raw_cat:
        is_derived = False
        for sfx in derived_suffixes:
            if c.endswith(sfx) and c[:-len(sfx)] in raw_cat:
                is_derived = True
                break
        if not is_derived:
            cat_cols.append(c)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. TARGET VARIABLE ANALYSIS (High-Priority Real Data Science EDA)
    # ─────────────────────────────────────────────────────────────────────────
    has_target = target_col is not None and target_col in df.columns
    if has_target:
        tgt_s = df[target_col].dropna()
        is_discrete_target = col_types.get(target_col) in ("categorical", "boolean") or tgt_s.nunique() <= 10

        if is_discrete_target:
            # 1a. Target Class Countplot with Counts and Percentages
            vc = tgt_s.astype(str).value_counts().reset_index()
            vc.columns = ["Target Class", "Count"]
            total_rows = vc["Count"].sum()
            vc["Percentage"] = (vc["Count"] / total_rows * 100).round(1).astype(str) + "%"
            vc["Label"] = vc["Count"].astype(str) + " (" + vc["Percentage"] + ")"

            fig_tgt = px.bar(
                vc,
                x="Target Class",
                y="Count",
                text="Label",
                color="Target Class",
                color_discrete_sequence=[PURPLE, CYAN, AMBER, PINK, GREEN]
            )
            fig_tgt.update_traces(textposition="outside")
            _styled_fig(fig_tgt, f"Target Class Balance — {target_col}")

            maj_ratio = (vc["Count"].max() / total_rows) * 100 if total_rows > 0 else 0
            why_tgt = f"Visualizes class frequencies of prediction target '{target_col}'. Dominant class comprises {maj_ratio:.1f}% of observations. "
            if maj_ratio > 65:
                why_tgt += "Moderate-to-high class imbalance detected; stratified cross-validation and evaluation metrics like PR-AUC are vital."
            else:
                why_tgt += "Class distribution is balanced across the problem space."

            recommendations.append({
                "id": "target_countplot",
                "title": f"Target Class Distribution — {target_col}",
                "chart_type": "Target Countplot with % Share",
                "columns": [target_col],
                "category": "Target Variable Deep-Dive",
                "why": why_tgt,
                "fig": fig_tgt
            })

            # 1b. Categorical Feature vs Target Countplots (Hue Grouped Bar)
            pred_cat_cols = [c for c in cat_cols if c != target_col][:2]
            for cat_c in pred_cat_cols:
                try:
                    cat_s = df[cat_c].dropna().astype(str)
                    top_vals = cat_s.value_counts().head(6).index.tolist()
                    sub_grp = df[df[cat_c].isin(top_vals)][[cat_c, target_col]].dropna()
                    if len(sub_grp) > 0:
                        fig_grp_cat = px.histogram(
                            sub_grp,
                            x=cat_c,
                            color=target_col,
                            barmode="group",
                            color_discrete_sequence=[PURPLE, CYAN, PINK, AMBER],
                        )
                        _styled_fig(fig_grp_cat, f"{cat_c} vs {target_col} (Grouped Countplot)")
                        recommendations.append({
                            "id": f"countplot_{cat_c}_by_target",
                            "title": f"{cat_c} vs {target_col} — Grouped Countplot",
                            "chart_type": "Grouped Feature Countplot (hue=Target)",
                            "columns": [cat_c, target_col],
                            "category": "Target Variable Deep-Dive",
                            "why": f"Compares how classes of '{target_col}' distribute across categories of '{cat_c}'. Asymmetry in bar proportions highlights strong predictive signal.",
                            "fig": fig_grp_cat
                        })
                except Exception:
                    pass

            # 1c. Numeric Features vs Target (Grouped Boxplots / Distribution)
            pred_num_cols = [c for c in numeric_cols if c != target_col][:2]
            for num_c in pred_num_cols:
                try:
                    sub_num = df[[num_c, target_col]].dropna()
                    sub_num[num_c] = pd.to_numeric(sub_num[num_c], errors="coerce")
                    sub_num = sub_num.dropna()
                    if len(sub_num) > 0:
                        fig_num_tgt = px.box(
                            sub_num,
                            x=target_col,
                            y=num_c,
                            color=target_col,
                            points="outliers",
                            color_discrete_sequence=[PURPLE, CYAN, PINK, AMBER]
                        )
                        fig_num_tgt.update_layout(showlegend=False)
                        _styled_fig(fig_num_tgt, f"{num_c} Distribution across {target_col}")
                        recommendations.append({
                            "id": f"box_{num_c}_by_target",
                            "title": f"{num_c} Separation across {target_col}",
                            "chart_type": "Grouped Boxplot by Target",
                            "columns": [num_c, target_col],
                            "category": "Target Variable Deep-Dive",
                            "why": f"Evaluates how continuous predictor '{num_c}' differentiates across '{target_col}' classes. Distinct medians and non-overlapping IQRs reveal top classification drivers.",
                            "fig": fig_num_tgt
                        })
                except Exception:
                    pass

        else:
            # Continuous Target Distribution
            t_s = pd.to_numeric(tgt_s, errors="coerce").dropna()
            fig_ctgt = px.histogram(pd.DataFrame({target_col: t_s}), x=target_col, nbins=28, marginal="box", color_discrete_sequence=[PURPLE])
            _styled_fig(fig_ctgt, f"Continuous Target Distribution — {target_col}")
            recommendations.append({
                "id": "target_continuous_dist",
                "title": f"Target Distribution — {target_col}",
                "chart_type": "Continuous Target Histogram & Box",
                "columns": [target_col],
                "category": "Target Variable Deep-Dive",
                "why": f"Examines central tendency, spread, and normality of continuous target '{target_col}'.",
                "fig": fig_ctgt
            })

    # ─────────────────────────────────────────────────────────────────────────
    # 2. CORE UNIVARIATE DISTRIBUTIONS (Non-Target Numeric Columns)
    # ─────────────────────────────────────────────────────────────────────────
    # Select top 2 non-target numeric columns by skewness to prevent chart overload
    candidate_num = [c for c in numeric_cols if c != target_col]
    sorted_numeric = sorted(
        candidate_num,
        key=lambda c: abs(float(col_stats.get(c, {}).get("skewness", 0.0) or 0.0)),
        reverse=True
    )[:2]

    for col in sorted_numeric:
        col_s = df[col].iloc[:, 0] if isinstance(df[col], pd.DataFrame) else df[col]
        s = pd.to_numeric(col_s, errors="coerce").dropna()
        if len(s) < 4:
            continue
        stats = col_stats.get(col, {})
        skew = stats.get("skewness", 0.0) or 0.0
        mean_v = stats.get("mean", s.mean())
        med_v = stats.get("median", s.median())

        plot_df = pd.DataFrame({col: s})
        fig = px.histogram(plot_df, x=col, nbins=28, marginal="box", color_discrete_sequence=[PURPLE])
        try:
            fig.add_vline(x=float(mean_v), line_dash="dash", line_color=PINK, line_width=2)
            fig.add_vline(x=float(med_v), line_dash="dot", line_color=CYAN, line_width=2)
            fig.add_annotation(
                text=f"<span style='color:{PINK}; font-weight:bold;'>Mean:</span> {float(mean_v):.2f} &nbsp;|&nbsp; <span style='color:{CYAN}; font-weight:bold;'>Median:</span> {float(med_v):.2f}",
                xref="paper", yref="paper",
                x=0.98, y=0.92,
                xanchor="right", yanchor="top",
                showarrow=False,
                font=dict(size=11, color="#334155", family="Inter, sans-serif"),
                bgcolor="rgba(241, 245, 249, 0.95)",
                bordercolor="#CBD5E1",
                borderwidth=1,
                borderpad=4
            )
        except Exception:
            pass
        _styled_fig(fig, f"Distribution Histogram — {col}")

        why_text = f"Analyzes central tendency (mean={mean_v:.2f}, median={med_v:.2f}) and spread. "
        if abs(skew) > 1.0:
            why_text += f"Highlight: Skewness ({skew:.2f}) indicates an asymmetrical tail requiring transformation."
        else:
            why_text += f"Highlight: Symmetric distribution (skewness = {skew:.2f})."

        recommendations.append({
            "id": f"num_dist_{col}",
            "title": f"Distribution — {col}",
            "chart_type": "Distribution Histogram with Box Marginal",
            "columns": [col],
            "category": "Univariate Distributions",
            "why": why_text,
            "fig": fig
        })

    # ─────────────────────────────────────────────────────────────────────────
    # 3. CORRELATION MATRIX & TOP MULTIVARIATE RELATIONSHIPS
    # ─────────────────────────────────────────────────────────────────────────
    # Pearson Correlation Matrix (Cleaned of duplicate scaled columns)
    if len(numeric_cols) >= 2 and corr_res.get("pearson") is not None:
        try:
            corr_mat = corr_res["pearson"].fillna(0)
            clean_corr_cols = [c for c in corr_mat.columns if c in numeric_cols]
            if len(clean_corr_cols) >= 2:
                corr_sub = corr_mat.loc[clean_corr_cols, clean_corr_cols]
                fig_cm = px.imshow(corr_sub, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                _styled_fig(fig_cm, "Pearson Correlation Heatmap")
                recommendations.append({
                    "id": "corr_matrix_heatmap",
                    "title": "Pairwise Correlation Matrix",
                    "chart_type": "Correlation Heatmap",
                    "columns": clean_corr_cols,
                    "category": "Multivariate Relationships",
                    "why": "Quantifies linear collinearity between clean numerical features without derived duplicates. Highlights multi-collinearity risks.",
                    "fig": fig_cm
                })
        except Exception:
            pass

    # Top Bivariate Scatter Plot with Trendline (Exclude identity / collinear twin columns)
    top_pairs = corr_res.get("top_pairs", [])
    valid_pairs = []
    for p in top_pairs:
        c1, c2, r = p.get("col1"), p.get("col2"), p.get("pearson_r", 0.0)
        # Avoid duplicate collinear scaled columns (|r| >= 0.999)
        if c1 in numeric_cols and c2 in numeric_cols and abs(r) < 0.999:
            valid_pairs.append((c1, c2, r))
        if len(valid_pairs) >= 2:
            break

    # If no valid pair from top_pairs, take the first two numeric columns
    if not valid_pairs and len(numeric_cols) >= 2:
        valid_pairs.append((numeric_cols[0], numeric_cols[1], 0.0))

    for c1, c2, r_val in valid_pairs[:1]:
        try:
            s1 = df[c1].iloc[:, 0] if isinstance(df[c1], pd.DataFrame) else df[c1]
            s2 = df[c2].iloc[:, 0] if isinstance(df[c2], pd.DataFrame) else df[c2]
            sc_df = pd.DataFrame({c1: pd.to_numeric(s1, errors="coerce"), c2: pd.to_numeric(s2, errors="coerce")}).dropna()
            sc_plot_df = sc_df.sample(min(len(sc_df), 800), random_state=42) if len(sc_df) > 800 else sc_df

            try:
                fig_sc = px.scatter(sc_plot_df, x=c1, y=c2, trendline="ols", color_discrete_sequence=[PURPLE], trendline_color_override=CYAN)
            except Exception:
                fig_sc = px.scatter(sc_plot_df, x=c1, y=c2, color_discrete_sequence=[PURPLE])

            title_r = f" (r = {r_val:.2f})" if r_val != 0.0 else ""
            _styled_fig(fig_sc, f"Bivariate Relationship: {c1} vs {c2}{title_r}")
            why_text = f"Explores continuous relationship between '{c1}' and '{c2}'."
            if r_val != 0.0:
                why_text += f" Pearson correlation r = {r_val:.2f} indicates {'positive' if r_val > 0 else 'negative'} co-movement."

            recommendations.append({
                "id": f"sc_{c1}_{c2}",
                "title": f"Bivariate Scatter — {c1} vs {c2}",
                "chart_type": "Scatter Plot with Trendline",
                "columns": [c1, c2],
                "category": "Multivariate Relationships",
                "why": why_text,
                "fig": fig_sc
            })
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # 4. CATEGORICAL FREQUENCY BAR CHART (Fallback or Additional Insight)
    # ─────────────────────────────────────────────────────────────────────────
    non_target_cats = [c for c in cat_cols if c != target_col]
    if non_target_cats and len(recommendations) < 7:
        cat_disp = non_target_cats[0]
        try:
            col_s = df[cat_disp].iloc[:, 0] if isinstance(df[cat_disp], pd.DataFrame) else df[cat_disp]
            vc = col_s.value_counts().head(10).reset_index()
            vc.columns = ["Category", "Count"]
            if len(vc) > 0:
                fig_cat = px.bar(vc, x="Count", y="Category", orientation="h", color="Count", color_continuous_scale=[CARD_BG, CYAN])
                fig_cat.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending", "title": cat_disp})
                _styled_fig(fig_cat, f"Frequency Distribution — {cat_disp}")
                recommendations.append({
                    "id": f"cat_bar_{cat_disp}",
                    "title": f"Value Frequency — {cat_disp}",
                    "chart_type": "Horizontal Bar Chart",
                    "columns": [cat_disp],
                    "category": "Univariate Distributions",
                    "why": f"Visualizes frequency breakdown of categorical variable '{cat_disp}'.",
                    "fig": fig_cat
                })
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # 5. TEMPORAL LINE CHART (If Datetime column is present)
    # ─────────────────────────────────────────────────────────────────────────
    if datetime_cols and numeric_cols:
        d_col = datetime_cols[0]
        n_col = numeric_cols[0]
        try:
            t_df = df[[d_col, n_col]].dropna().copy()
            t_df[d_col] = pd.to_datetime(t_df[d_col], errors="coerce")
            t_df = t_df.dropna().sort_values(d_col)
            if len(t_df) > 5:
                fig_ts = px.line(t_df, x=d_col, y=n_col, color_discrete_sequence=[CYAN])
                _styled_fig(fig_ts, f"Temporal Trend: {n_col} over {d_col}")
                recommendations.append({
                    "id": f"time_series_{n_col}",
                    "title": f"Temporal Trend — {n_col} over {d_col}",
                    "chart_type": "Time Series Line Chart",
                    "columns": [d_col, n_col],
                    "category": "Temporal Analytics",
                    "why": f"Analyzes trend direction and temporal variations in '{n_col}' over time.",
                    "fig": fig_ts
                })
        except Exception:
            pass

    return recommendations


def build_custom_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    trendline: bool = False,
) -> go.Figure:
    """Builds an interactive customized chart based on user parameter choices."""
    sub_df = df.copy()
    cols_needed = [x_col]
    if y_col and y_col != x_col and y_col in sub_df.columns:
        cols_needed.append(y_col)
    if color_col and color_col not in cols_needed and color_col in sub_df.columns:
        cols_needed.append(color_col)

    sub_df = sub_df[cols_needed].dropna()

    title_text = f"{chart_type}: {x_col}"
    if y_col and y_col != x_col:
        title_text += f" vs {y_col}"
    if color_col:
        title_text += f" (Color: {color_col})"

    if chart_type == "Scatter Plot":
        use_trend = "ols" if (trendline and y_col and y_col != x_col and pd.api.types.is_numeric_dtype(sub_df[x_col]) and pd.api.types.is_numeric_dtype(sub_df[y_col])) else None
        try:
            fig = px.scatter(sub_df, x=x_col, y=y_col if y_col else x_col, color=color_col, trendline=use_trend, color_discrete_sequence=PALETTE)
        except Exception:
            fig = px.scatter(sub_df, x=x_col, y=y_col if y_col else x_col, color=color_col, color_discrete_sequence=PALETTE)

    elif chart_type == "Distribution Histogram":
        fig = px.histogram(sub_df, x=x_col, color=color_col, marginal="box", nbins=30, color_discrete_sequence=PALETTE)

    elif chart_type == "Box Plot":
        fig = px.box(sub_df, x=color_col if color_col else None, y=y_col if y_col else x_col, color=color_col, points="outliers", color_discrete_sequence=PALETTE)

    elif chart_type == "Violin Plot":
        fig = px.violin(sub_df, x=color_col if color_col else None, y=y_col if y_col else x_col, color=color_col, box=True, points="outliers", color_discrete_sequence=PALETTE)

    elif chart_type == "Line Trend Chart":
        sort_df = sub_df.sort_values(x_col)
        fig = px.line(sort_df, x=x_col, y=y_col if y_col else x_col, color=color_col, color_discrete_sequence=PALETTE)

    elif chart_type in ("Countplot (Category Frequency)", "Bar Chart"):
        if y_col and y_col != x_col:
            grouped = sub_df.groupby([x_col] + ([color_col] if color_col else []))[y_col].mean().reset_index()
            fig = px.bar(grouped, x=x_col, y=y_col, color=color_col, barmode="group", color_discrete_sequence=PALETTE)
        elif color_col:
            fig = px.histogram(sub_df, x=x_col, color=color_col, barmode="group", color_discrete_sequence=PALETTE)
        else:
            vc = sub_df[x_col].value_counts().head(20).reset_index()
            vc.columns = [x_col, "Count"]
            fig = px.bar(vc, x=x_col, y="Count", text="Count", color="Count", color_continuous_scale="Purples")

    elif chart_type == "2D Density Heatmap":
        fig = px.density_heatmap(sub_df, x=x_col, y=y_col if y_col else x_col, nbinsx=30, nbinsy=30, color_continuous_scale="Viridis")

    else:
        fig = px.scatter(sub_df, x=x_col, y=y_col if y_col else x_col, color=color_col, color_discrete_sequence=PALETTE)

    return _styled_fig(fig, title_text)


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
                <div class="why-title">Why are we making this chart?</div>
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
