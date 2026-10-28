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


def generate_eda_chart_recommendations(df: pd.DataFrame, col_types: Dict[str, str], col_stats: Dict[str, Any], corr_res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates structured chart recommendations with analytical rationales."""
    recommendations = []
    
    numeric_cols = [c for c, t in col_types.items() if t == "numeric" and c in df.columns]
    cat_cols     = [c for c, t in col_types.items() if t in ("categorical", "boolean") and c in df.columns]
    datetime_cols= [c for c, t in col_types.items() if t == "datetime" and c in df.columns]
    text_cols    = [c for c, t in col_types.items() if t == "text" and c in df.columns]

    # 1. UNIVARIATE NUMERIC DISTRIBUTIONS & INDIVIDUAL BOXPLOTS
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
        
        # 1a. Histogram with Central Tendency Reference Lines & Rug (no duplicate/colliding text)
        fig = px.histogram(plot_df, x=col, nbins=35, marginal="rug", color_discrete_sequence=[PURPLE])
        try:
            fig.add_vline(x=float(mean_v), line_dash="dash", line_color=PINK, line_width=2)
            fig.add_vline(x=float(med_v), line_dash="dot", line_color=CYAN, line_width=2)
            fig.add_annotation(
                text=f"<span style='color:{PINK}; font-weight:bold;'>■ Mean:</span> {float(mean_v):.2f} &nbsp;|&nbsp; <span style='color:{CYAN}; font-weight:bold;'>● Median:</span> {float(med_v):.2f}",
                xref="paper", yref="paper",
                x=0.98, y=1.02,
                xanchor="right", yanchor="bottom",
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

        why_text = f"Analyzes central tendency (mean={mean_v:.2f}, median={med_v:.2f}) and distribution spread. "
        if abs(skew) > 1.0:
            why_text += f"Highlight: Significant skewness ({skew:.2f}) indicates an asymmetrical tail requiring transformation."
        else:
            why_text += f"Highlight: Relatively balanced symmetric distribution (skewness = {skew:.2f})."

        recommendations.append({
            "id": f"num_dist_{col}",
            "title": f"Numeric Distribution — {col}",
            "chart_type": "Histogram & Rug Plot",
            "columns": [col],
            "category": "Univariate Numeric",
            "why": why_text,
            "fig": fig
        })

        # 1b. Individual Outlier Boxplot for each numeric column
        fig_ind_box = px.box(plot_df, y=col, points="outliers", color_discrete_sequence=[CYAN])
        _styled_fig(fig_ind_box, f"Outlier & Spread Boxplot — {col}")
        recommendations.append({
            "id": f"num_box_{col}",
            "title": f"Spread & Quartiles — {col}",
            "chart_type": "Box Plot with Outliers",
            "columns": [col],
            "category": "Univariate Numeric",
            "why": f"Identifies IQR spread, median ({med_v:.2f}), and anomalies lying beyond 1.5x IQR boundaries.",
            "fig": fig_ind_box
        })

    # 2. NUMERIC BOXPLOTS & OUTLIER STRIPS (COMPARATIVE)
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

    # 5. BIVARIATE SCATTER PLOTS WITH TRENDLINES (NUMERIC vs NUMERIC)
    # Ensure scatter plots are ALWAYS created when 2+ numeric columns exist!
    pairs_to_plot = []
    top_pairs = corr_res.get("top_pairs", [])
    for p in top_pairs[:6]:
        pairs_to_plot.append((p["col1"], p["col2"], p.get("pearson_r", 0.0)))

    # If top_pairs has fewer than 2 items, add sequential pairs from numeric_cols
    if len(pairs_to_plot) < 2 and len(numeric_cols) >= 2:
        for i in range(len(numeric_cols)):
            for j in range(i + 1, min(len(numeric_cols), i + 3)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                if not any((c1 == p[0] and c2 == p[1]) or (c1 == p[1] and c2 == p[0]) for p in pairs_to_plot):
                    pairs_to_plot.append((c1, c2, 0.0))
                if len(pairs_to_plot) >= 4:
                    break
            if len(pairs_to_plot) >= 4:
                break

    for c1, c2, r_val in pairs_to_plot:
        try:
            s1 = df[c1].iloc[:, 0] if isinstance(df[c1], pd.DataFrame) else df[c1]
            s2 = df[c2].iloc[:, 0] if isinstance(df[c2], pd.DataFrame) else df[c2]
            sc_df = pd.DataFrame({c1: pd.to_numeric(s1, errors="coerce"), c2: pd.to_numeric(s2, errors="coerce")}).dropna()

            try:
                fig_sc = px.scatter(sc_df, x=c1, y=c2, trendline="ols", color_discrete_sequence=[PURPLE], trendline_color_override=CYAN)
            except Exception:
                fig_sc = px.scatter(sc_df, x=c1, y=c2, color_discrete_sequence=[PURPLE])

            title_r = f" (r = {r_val:.3f})" if r_val != 0.0 else ""
            _styled_fig(fig_sc, f"Bivariate Scatter: {c1} vs {c2}{title_r}")
            why_text = f"Analyzes continuous joint distribution and linearity between '{c1}' and '{c2}'."
            if r_val != 0.0:
                why_text += f" Pearson correlation coefficient r = {r_val:.3f} indicates {'positive' if r_val > 0 else 'negative'} co-movement."

            recommendations.append({
                "id": f"sc_{c1}_{c2}",
                "title": f"Bivariate Scatter — {c1} vs {c2}",
                "chart_type": "Scatter Plot with Trendline",
                "columns": [c1, c2],
                "category": "Bivariate Relationships",
                "why": why_text,
                "fig": fig_sc
            })
        except Exception:
            pass

    # 5b. SCATTER MATRIX (SPLOM) FOR NUMERICAL DATASETS
    if len(numeric_cols) >= 3:
        try:
            splom_cols = numeric_cols[:4]
            splom_df = df[splom_cols].apply(pd.to_numeric, errors="coerce").dropna()
            if len(splom_df) > 5:
                fig_splom = px.scatter_matrix(splom_df, dimensions=splom_cols, color_discrete_sequence=[PURPLE])
                _styled_fig(fig_splom, "Pairwise Scatter Matrix (SPLOM)")
                recommendations.append({
                    "id": "scatter_matrix_splom",
                    "title": "Pairwise Scatter Matrix (Pairplot)",
                    "chart_type": "Scatter Matrix (SPLOM)",
                    "columns": splom_cols,
                    "category": "Multivariate Relationships",
                    "why": f"Simultaneously examines pairwise scatter plots and diagonal distributions across {len(splom_cols)} numerical features in one unified grid.",
                    "fig": fig_splom
                })
        except Exception:
            pass

    # 6. GROUPED BOXPLOTS (NUMERIC BY CATEGORICAL)
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
                        category_order = (
                            tmp.groupby(cat_c)[num_c]
                            .median()
                            .sort_values()
                            .index.tolist()
                        )
                        fig_g = px.box(
                            tmp,
                            x=cat_c,
                            y=num_c,
                            color=cat_c,
                            category_orders={cat_c: category_order},
                            points="outliers",
                            color_discrete_sequence=PALETTE,
                        )
                        fig_g.update_layout(showlegend=False)
                        fig_g.update_yaxes(title=num_c)
                        _styled_fig(fig_g, f"{num_c} Distribution across {cat_c}")
                        why_text = f"Groups numerical variable '{num_c}' across categories of '{cat_c}'. "
                        why_text += "Boxes show the middle 50% of values, center lines show medians, and whiskers summarize the remaining spread."

                        recommendations.append({
                            "id": f"group_box_{num_c}_{cat_c}",
                            "title": f"Grouped Breakdown — {num_c} by {cat_c}",
                            "chart_type": "Grouped Box Plot",
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
        fig = px.histogram(sub_df, x=x_col, color=color_col, marginal="rug", nbins=40, color_discrete_sequence=PALETTE)

    elif chart_type == "Box Plot":
        fig = px.box(sub_df, x=color_col if color_col else None, y=y_col if y_col else x_col, color=color_col, points="outliers", color_discrete_sequence=PALETTE)

    elif chart_type == "Violin Plot":
        fig = px.violin(sub_df, x=color_col if color_col else None, y=y_col if y_col else x_col, color=color_col, box=True, points="outliers", color_discrete_sequence=PALETTE)

    elif chart_type == "Line Trend Chart":
        sort_df = sub_df.sort_values(x_col)
        fig = px.line(sort_df, x=x_col, y=y_col if y_col else x_col, color=color_col, color_discrete_sequence=PALETTE)

    elif chart_type == "Bar Chart":
        if y_col and y_col != x_col:
            grouped = sub_df.groupby([x_col] + ([color_col] if color_col else []))[y_col].mean().reset_index()
            fig = px.bar(grouped, x=x_col, y=y_col, color=color_col, color_discrete_sequence=PALETTE)
        else:
            vc = sub_df[x_col].value_counts().head(20).reset_index()
            vc.columns = [x_col, "Count"]
            fig = px.bar(vc, x=x_col, y="Count", color="Count", color_continuous_scale="Purples")

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
