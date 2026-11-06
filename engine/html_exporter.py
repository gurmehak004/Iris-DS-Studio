"""
engine/html_exporter.py
Generates a comprehensive, stunning, high-contrast single-file HTML Data Science Dashboard
with embedded interactive Plotly charts, full lifecycle audit trail, Data Health scorecard,
cleaning & feature engineering logs, machine learning baselines & diagnostics, and Python code export.
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

# ── Design Tokens & Dark Executive Palette ────────────────────────────────────
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
    margin=dict(l=50, r=30, t=55, b=45),
    height=420,
)


def _styled_fig(fig, title: str = ""):
    """Apply high-contrast dark theme to any Plotly figure."""
    if fig is None:
        return None
    fig.update_layout(**LAYOUT_DEFAULTS)
    if title:
        short_title = title if len(title) <= 50 else title[:47] + "..."
        fig.update_layout(title=dict(text=short_title, font=dict(size=14, color=PURPLE, family="Inter, sans-serif")))
    fig.update_xaxes(gridcolor="#2A2A48", zerolinecolor="#2A2A48", tickfont=dict(color="#D1D5DB", size=11))
    fig.update_yaxes(gridcolor="#2A2A48", zerolinecolor="#2A2A48", tickfont=dict(color="#D1D5DB", size=11))
    return fig


def _fig_to_html(fig) -> str:
    """Convert Plotly figure to HTML string wrapped in an explicit-height card."""
    if fig is None:
        return ""
    try:
        fig = sanitize_fig_for_html(fig)
        chart_html = pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})
        return f'<div class="chart-card"><div class="chart-wrapper">{chart_html}</div></div>'
    except Exception:
        return ""


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
        return '<p style="color:#9CA3AF; padding:8px 0;">No issues detected.</p>'
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
    ml_results: Optional[Dict] = None,
    iris_says: Optional[Dict] = None,
    df: Optional[pd.DataFrame] = None,
    cleaning_log: Optional[List[str]] = None,
    fe_log: Optional[List[str]] = None,
    health_score: Optional[Dict] = None,
    target_col: Optional[str] = None,
    train_test_split_enabled: bool = False,
    pipeline_script: Optional[str] = None,
) -> str:
    """
    Constructs an updated, executive-grade, standalone HTML dashboard
    documenting all 5 pipeline stages including machine learning diagnostics.
    """
    if iris_says is None:
        iris_says = {}
    if df is None:
        df = pd.DataFrame()
    if cleaning_log is None:
        cleaning_log = []
    if fe_log is None:
        fe_log = []

    now_str = datetime.datetime.now().strftime("%B %d, %Y — %H:%M")
    cleaned_rows = meta.get("cleaned_rows", len(df))
    cleaned_cols = meta.get("cleaned_cols", len(df.columns))
    orig_filename = meta.get("original_filename") or meta.get("dataset_name") or "dataset.csv"

    total_cells = df.shape[0] * df.shape[1] if len(df) > 0 else 1
    total_missing = int(df.isnull().sum().sum()) if len(df) > 0 else 0
    completeness = round((1 - total_missing / max(total_cells, 1)) * 100, 1)

    type_counts = {}
    for t in col_types.values():
        type_counts[t] = type_counts.get(t, 0) + 1

    # Inferred Column Type Chips
    type_chips = ""
    chip_colors = {
        "numeric": "#8B5CF6", "categorical": "#06B6D4", "datetime": "#F59E0B",
        "text": "#EC4899", "boolean": "#3B82F6", "id": "#94A3B8", "empty": "#64748B"
    }
    for t, c in type_counts.items():
        color = chip_colors.get(t, "#888")
        type_chips += f'<span class="type-badge" style="background:{color}22; color:{color}; border:1px solid {color}55;">{t} ({c})</span> '

    # ── 1. BUSINESS UNDERSTANDING & DATA COLLECTION ───────────────────────────
    target_display = target_col or "None Selected"
    sec_overview = _stat_grid([
        ("Dataset Source", orig_filename, "Active Ingest"),
        ("Total Rows", f"{cleaned_rows:,}", "Observations"),
        ("Features", f"{cleaned_cols}", "Attributes"),
        ("Target Variable", target_display, "Supervised Focus"),
        ("Data Completeness", f"{completeness}%", f"{total_missing:,} missing cells"),
        ("Memory Footprint", f"{meta.get('memory_mb', round(df.memory_usage().sum() / 1024**2, 2) if len(df) > 0 else 0)} MB", "In-Memory RAM"),
        ("Numeric Columns", type_counts.get("numeric", 0), "Continuous/Discrete"),
        ("Categorical Columns", type_counts.get("categorical", 0), "Nominal/Ordinal"),
    ])
    sec_overview += f'<div style="margin:16px 0 24px;">{type_chips}</div>'

    # Sample Data Preview Table (First 10 rows)
    if len(df) > 0:
        sec_overview += '<h3 class="sub-header">Cleaned Dataset Sample Preview (Top 10 Records)</h3>'
        preview_df = df.head(10).copy()
        sec_overview += _render_table(preview_df)

    if iris_says.get("overview"):
        sec_overview += f'<div class="iris-insight">🌸 <strong>Iris AI Briefing:</strong> {iris_says["overview"]}</div>'

    # ── 2. DATA HEALTH SCORECARD & AUDIT ──────────────────────────────────────
    sec_health = ""
    score_val = health_score.get("score", 100) if health_score else 100
    score_color = "#10B981" if score_val >= 80 else ("#F59E0B" if score_val >= 60 else "#EF4444")
    
    sec_health += f"""
    <div style="background:rgba(139,92,246,0.06); border:1px solid rgba(139,92,246,0.25); border-radius:16px; padding:22px 26px; margin-bottom:24px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
            <div>
                <div style="font-size:0.75rem; font-weight:800; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.08em;">Overall Data Quality Health Index</div>
                <div style="font-size:2.8rem; font-weight:900; color:{score_color}; line-height:1.1; margin-top:4px;">{score_val} <span style="font-size:1.1rem; color:#6B7280; font-weight:600;">/ 100</span></div>
            </div>
            <div style="display:flex; gap:16px; flex-wrap:wrap;">
                <div class="kpi-mini"><span>Missing Columns</span><strong>{quality.get('missing_cols', len(quality.get('missing', [])))}</strong></div>
                <div class="kpi-mini"><span>Duplicate Rows</span><strong>{quality.get('duplicates', {}).get('duplicate_rows', int(df.duplicated().sum()) if len(df) > 0 else 0)}</strong></div>
                <div class="kpi-mini"><span>Outlier Features</span><strong>{quality.get('outlier_cols', len(quality.get('outliers', [])))}</strong></div>
                <div class="kpi-mini"><span>Leakage Guard</span><strong style="color:#10B981;">{'Active (80/20)' if train_test_split_enabled else 'Full Mode'}</strong></div>
            </div>
        </div>
    </div>"""

    # Missing Values Table
    miss_df = quality.get("missing", pd.DataFrame())
    if len(miss_df) > 0:
        sec_health += '<h3 class="sub-header">Missing Values by Feature</h3>'
        sec_health += _render_table(miss_df)
    else:
        sec_health += '<div class="success-banner">✅ Zero Missing Values: Dataset is 100% complete across all attributes!</div>'

    # Outliers Table
    outlier_df = quality.get("outliers", pd.DataFrame())
    if len(outlier_df) > 0:
        sec_health += '<h3 class="sub-header">Statistical Outliers Breakdown (IQR & Z-Score)</h3>'
        sec_health += _render_table(outlier_df)

    # Skewness Table
    skew_df = quality.get("skewness", pd.DataFrame())
    if len(skew_df) > 0:
        sec_health += '<h3 class="sub-header">Distribution Skewness & Kurtosis</h3>'
        sec_health += _render_table(skew_df)

    if iris_says.get("quality"):
        sec_health += f'<div class="iris-insight">🌸 <strong>Iris AI Audit Note:</strong> {iris_says["quality"]}</div>'

    # ── 3. DATA CLEANING & PREPROCESSING LOG ──────────────────────────────────
    sec_cleaning = ""
    if cleaning_log:
        sec_cleaning += '<p style="color:#9CA3AF; margin-bottom:14px;">Audit trail of all data cleaning actions applied in this session:</p>'
        sec_cleaning += '<div class="log-grid">'
        for idx, entry in enumerate(cleaning_log):
            sec_cleaning += f"""
            <div class="log-card">
                <span class="log-badge">Action #{idx+1}</span>
                <div class="log-text">{entry}</div>
            </div>"""
        sec_cleaning += '</div>'
    else:
        sec_cleaning += '<div class="success-banner">Dataset verified clean — no manual imputation or dropping required.</div>'

    # ── 4. EXPLORATORY DATA ANALYSIS (EDA) ────────────────────────────────────
    sec_eda = ""
    ENGINEERED_SUFFIXES = ("_scaled", "_minmax", "_encoded", "_binary", "_log", "_log1p", "_zscore")
    numeric_cols = [c for c, t in col_types.items() if t == "numeric" and c in df.columns and not any(c.endswith(sfx) for sfx in ENGINEERED_SUFFIXES)]
    cat_cols     = [c for c, t in col_types.items() if t in ("categorical", "boolean") and c in df.columns and not any(c.endswith(sfx) for sfx in ENGINEERED_SUFFIXES)]

    if numeric_cols:
        sec_eda += '<h3 class="sub-header">Continuous Feature Distributions & Central Tendency</h3>'
        clean_num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        
        # Plot up to 6 distribution histograms with Mean & Median markers
        for i, col in enumerate(numeric_cols[:6]):
            try:
                s = clean_num_df[col].dropna()
                if len(s) >= 4:
                    mean_val = float(s.mean())
                    median_val = float(s.median())
                    fig_hist = px.histogram(
                        pd.DataFrame({col: s}), x=col, nbins=35, marginal="rug",
                        color_discrete_sequence=[VIBRANT_PALETTE[i % len(VIBRANT_PALETTE)]]
                    )
                    fig_hist.add_vline(x=mean_val, line_dash="dash", line_color=PINK, line_width=2)
                    fig_hist.add_vline(x=median_val, line_dash="dot", line_color=CYAN, line_width=2)
                    fig_hist.add_annotation(
                        text=f"Mean: {mean_val:.2f} | Median: {median_val:.2f}",
                        xref="paper", yref="paper", x=0.98, y=1.02, xanchor="right", yanchor="bottom",
                        showarrow=False, font=dict(size=11, color="#CBD5E1")
                    )
                    _styled_fig(fig_hist, f"Distribution Histogram — {col}")
                    fig_hist.update_layout(height=340)
                    sec_eda += _fig_to_html(fig_hist)
            except Exception:
                pass

        # Comparative Box Plot
        if len(numeric_cols) >= 2:
            try:
                fig_box = px.box(clean_num_df.dropna(how="all"), color_discrete_sequence=VIBRANT_PALETTE)
                _styled_fig(fig_box, "Numeric Features Variance & Outliers Comparison")
                fig_box.update_layout(height=380)
                sec_eda += _fig_to_html(fig_box)
            except Exception:
                pass

    # Categorical Frequency Bar Charts & Donut Ring
    if cat_cols:
        sec_eda += '<h3 class="sub-header">Categorical Value Proportions</h3>'
        for col in cat_cols[:3]:
            try:
                vc = df[col].astype(str).value_counts().head(10).reset_index()
                vc.columns = ["Category", "Count"]
                fig_cat = px.bar(vc, x="Count", y="Category", orientation="h", color="Count", color_continuous_scale="Purples")
                fig_cat.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
                _styled_fig(fig_cat, f"Category Counts — {col}")
                fig_cat.update_layout(height=320)
                sec_eda += _fig_to_html(fig_cat)
            except Exception:
                pass

    # ── 5. CORRELATIONS & RELATIONSHIPS ───────────────────────────────────────
    sec_corr = ""
    if len(numeric_cols) >= 2:
        try:
            clean_corr_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            corr_mat = clean_corr_df.corr().round(3)
            fig_cm = px.imshow(corr_mat, text_auto=".2f", color_continuous_scale="Viridis", zmin=-1, zmax=1)
            _styled_fig(fig_cm, "Pearson Pairwise Correlation Heatmap")
            fig_cm.update_layout(height=440)
            sec_corr += _fig_to_html(fig_cm)

            # Top Correlated Feature Pair with Trendline
            pairs = []
            cols_l = list(corr_mat.columns)
            for i in range(len(cols_l)):
                for j in range(i + 1, len(cols_l)):
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
                try:
                    fig_sc = px.scatter(sc_df, x=c1_name, y=c2_name, trendline="ols", color_discrete_sequence=[PURPLE], trendline_color_override=CYAN)
                except Exception:
                    fig_sc = px.scatter(sc_df, x=c1_name, y=c2_name, color_discrete_sequence=[PURPLE])
                _styled_fig(fig_sc, f"Strongest Feature Pair: {c1_name} vs {c2_name} (r = {r_v:.3f})")
                fig_sc.update_layout(height=380)
                sec_corr += _fig_to_html(fig_sc)
        except Exception:
            sec_corr += '<p style="color:#9CA3AF;">Correlation analysis unavailable.</p>'
    else:
        sec_corr = '<p style="color:#9CA3AF;">Need at least 2 numerical features for correlation analysis.</p>'

    if iris_says.get("correlations"):
        sec_corr += f'<div class="iris-insight">🌸 <strong>Iris AI Correlation Finding:</strong> {iris_says["correlations"]}</div>'

    # ── 6. FEATURE ENGINEERING LAB ────────────────────────────────────────────
    sec_fe = ""
    if fe_log:
        sec_fe += '<h3 class="sub-header">Applied Transformations in this Session</h3>'
        sec_fe += '<div class="log-grid" style="margin-bottom:24px;">'
        for idx, entry in enumerate(fe_log):
            sec_fe += f"""
            <div class="log-card">
                <span class="log-badge" style="background:#06B6D425; color:#06B6D4; border-color:#06B6D444;">FE #{idx+1}</span>
                <div class="log-text">{entry}</div>
            </div>"""
        sec_fe += '</div>'

    # Per-column recommended transformations strategy
    fe_rows = []
    for col, ct in col_types.items():
        s = (col_stats or {}).get(col, {})
        actions = []
        if ct == "numeric":
            sk = s.get("skewness", 0) or 0
            if abs(sk) > 1.0:
                actions.append(f"⚡ Log Transform (skew={sk:.2f})")
            actions += ["StandardScaler (Z-Score)", "MinMaxScaler [0,1]", "4-Quantile Binning"]
        elif ct in ("categorical", "boolean"):
            n_u = s.get("unique", 0)
            actions.append("Binary 0/1 Flag" if n_u == 2 else (f"One-Hot ({n_u} cols)" if n_u <= 10 else f"Frequency Encode ({n_u} cats)"))
        elif ct == "datetime":
            actions += ["Extract Year/Month/Day", "Day of Week & IsWeekend"]
        elif ct == "text":
            actions += ["Char Length & Word Count"]
        elif ct == "id":
            actions.append("Drop Identifier Feature")
            
        fe_rows.append({
            "Feature Name": col,
            "Type": ct,
            "Optimal Recommended Engineering Strategy": " • ".join(actions) or "—"
        })
    sec_fe += '<h3 class="sub-header">Strategic Feature Transformation Matrix</h3>'
    sec_fe += _render_table(pd.DataFrame(fe_rows), max_rows=60)

    # ── 7. MACHINE LEARNING BASELINES & DIAGNOSTICS ───────────────────────────
    sec_ml = ""
    if ml_results and "leaderboard" in ml_results:
        best_model = ml_results.get("best_model", "Model")
        best_score = ml_results.get("best_score", 0.0)
        task_label = ml_results.get("task", "Supervised Learning").replace("_", " ").title()
        n_train = ml_results.get("n_train_samples", len(df))
        n_test = ml_results.get("n_test_samples", 0)

        sec_ml += f"""
        <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.3); border-left:4px solid #10B981; border-radius:14px; padding:20px 24px; margin-bottom:24px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
                <div>
                    <span style="font-size:0.75rem; font-weight:800; color:#10B981; text-transform:uppercase; letter-spacing:0.06em;">Top Performing Baseline Model</span>
                    <div style="font-size:1.6rem; font-weight:900; color:#F3F4F6; margin-top:2px;">⭐ {best_model}</div>
                </div>
                <div style="display:flex; gap:20px;">
                    <div style="text-align:right;">
                        <div style="font-size:0.72rem; color:#9CA3AF; font-weight:700; text-transform:uppercase;">Primary Metric</div>
                        <div style="font-size:1.6rem; font-weight:900; color:#10B981;">{best_score:.4f}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.72rem; color:#9CA3AF; font-weight:700; text-transform:uppercase;">Validation Samples</div>
                        <div style="font-size:1.15rem; font-weight:800; color:#E5E7EB; margin-top:4px;">{n_train:,} tr / {n_test:,} ts</div>
                    </div>
                </div>
            </div>
        </div>"""

        # Leaderboard Table
        lb_list = ml_results.get("leaderboard", [])
        if lb_list:
            lb_rows = []
            for idx, item in enumerate(lb_list):
                lb_rows.append({
                    "Rank": f"#{idx+1}",
                    "Model": item.get("model", "") + (" ⭐ (Winner)" if idx == 0 and item.get("status") == "Success" else ""),
                    item.get("primary_label", "Primary Score"): item.get("primary_metric"),
                    item.get("secondary_label", "Secondary"): item.get("secondary_metric"),
                    "Training Latency": item.get("train_time", "0s"),
                    "Execution Status": item.get("status", "Success")
                })
            sec_ml += '<h3 class="sub-header">🏆 Model Performance Leaderboard</h3>'
            sec_ml += _render_table(pd.DataFrame(lb_rows))

        # Diagnostic Visualizations
        if "confusion_matrix" in ml_results and ml_results["confusion_matrix"]:
            try:
                cm_arr = np.array(ml_results["confusion_matrix"])
                class_labels = ml_results.get("class_labels") or [f"Class {i}" for i in range(len(cm_arr))]
                fig_cm = px.imshow(
                    cm_arr, x=class_labels, y=class_labels, text_auto=True,
                    color_continuous_scale="Purples",
                    labels=dict(x="Predicted Class", y="Actual True Class", color="Observations")
                )
                _styled_fig(fig_cm, f"Validation Confusion Matrix — {best_model}")
                fig_cm.update_layout(height=380)
                sec_ml += '<h3 class="sub-header">Confusion Matrix Diagnostics</h3>'
                sec_ml += _fig_to_html(fig_cm)
            except Exception:
                pass
        elif "regression_diagnostics" in ml_results:
            try:
                diag = ml_results["regression_diagnostics"]
                y_tr = diag["y_true"]
                y_pr = diag["y_pred"]
                sc_reg_df = pd.DataFrame({"Actual Target": y_tr, "Predicted Target": y_pr})
                fig_reg = px.scatter(sc_reg_df, x="Actual Target", y="Predicted Target", color_discrete_sequence=[PURPLE])
                mn = min(min(y_tr), min(y_pr))
                mx = max(max(y_tr), max(y_pr))
                fig_reg.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx, line=dict(color=CYAN, dash="dash", width=2))
                _styled_fig(fig_reg, f"Actual vs. Predicted Target Values — {best_model}")
                fig_reg.update_layout(height=380)
                sec_ml += '<h3 class="sub-header">Actual vs. Predicted Diagnostic Fit</h3>'
                sec_ml += _fig_to_html(fig_reg)
            except Exception:
                pass

        # Feature Importances Plot
        fi_dict = ml_results.get("feature_importances", {})
        if fi_dict:
            try:
                fi_df = pd.DataFrame({
                    "Feature": list(fi_dict.keys()),
                    "Importance": list(fi_dict.values())
                }).sort_values("Importance", ascending=True).tail(10)
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#DDD6FE", PURPLE])
                _styled_fig(fig_fi, f"Top Feature Importances — {best_model}")
                fig_fi.update_layout(height=360, coloraxis_showscale=False)
                sec_ml += '<h3 class="sub-header">Key Feature Importances</h3>'
                sec_ml += _fig_to_html(fig_fi)
            except Exception:
                pass

    else:
        sec_ml += f"""
        <div style="background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.25); border-radius:12px; padding:16px 20px; color:#93C5FD; font-size:0.9rem;">
            ℹ️ Baseline model training has not been run for this session. Recommended algorithms for target <strong>`{target_display}`</strong>:
            <strong>XGBoost</strong>, <strong>Random Forest</strong>, and <strong>Logistic/Ridge Regression</strong>.
        </div>"""

    # ── 8. REPRODUCIBLE PYTHON SCRIPT PREVIEW ──────────────────────────────────
    sec_code = ""
    if pipeline_script:
        sec_code += f"""
        <p style="color:#9CA3AF; margin-bottom:14px;">Standalone, production-grade Python script reproducing all cleaning, feature engineering, and model training:</p>
        <div class="code-box"><pre><code>{pipeline_script.replace("<", "&lt;").replace(">", "&gt;")}</code></pre></div>"""
    else:
        sec_code += '<p style="color:#9CA3AF;">Pipeline script export available in Stage 5 of Iris Data Studio.</p>'

    # ── ASSEMBLE FULL HTML DOCUMENT ───────────────────────────────────────────
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Iris Data Science Executive Dashboard — {orig_filename}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: {BG};
    --card: {CARD_BG};
    --purple: {PURPLE};
    --cyan: {CYAN};
    --pink: {PINK};
    --green: {GREEN};
    --text: #F3F4F6;
    --muted: #9CA3AF;
    --border: rgba(139, 92, 246, 0.22);
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
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.20) 0%, rgba(6, 182, 212, 0.15) 100%);
    border-bottom: 1px solid var(--border);
    padding: 30px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
  }}
  .header-left h1 {{
    font-size: 2.1rem;
    font-weight: 900;
    background: linear-gradient(135deg, #A78BFA, var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px;
  }}
  .header-left p {{ color: var(--muted); font-size: 0.92rem; }}
  .iris-badge {{
    background: rgba(139, 92, 246, 0.18);
    border: 1px solid rgba(139, 92, 246, 0.45);
    border-radius: 30px;
    padding: 10px 24px;
    font-size: 1.1rem;
    color: var(--purple);
    font-weight: 800;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.25);
  }}

  /* ── Sticky Navigation ── */
  nav {{
    position: sticky;
    top: 0;
    z-index: 1000;
    background: rgba(9, 9, 18, 0.94);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 0 48px;
    display: flex;
    gap: 4px;
    overflow-x: auto;
  }}
  nav a {{
    display: inline-block;
    padding: 14px 18px;
    color: var(--muted);
    text-decoration: none;
    font-size: 0.86rem;
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
    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
  }}
  .section-title {{
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--purple);
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid rgba(139, 92, 246, 0.18);
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

  .kpi-mini {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 8px 16px;
    text-align: center;
  }}
  .kpi-mini span {{ display: block; font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase; font-weight: 700; }}
  .kpi-mini strong {{ font-size: 1.15rem; color: #F3F4F6; }}

  /* ── Chart Wrappers ── */
  .chart-card {{
    background: #0F0F20;
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 16px;
    padding: 16px;
    margin: 20px 0 28px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
  }}
  .chart-wrapper {{
    min-height: 400px;
    width: 100%;
  }}

  /* ── Badges ── */
  .type-badge {{
    display: inline-block;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px;
  }}

  /* ── Logs & Insights ── */
  .log-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }}
  .log-card {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 14px;
  }}
  .log-badge {{
    display: inline-block;
    background: rgba(139,92,246,0.2);
    color: #A78BFA;
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 800;
    padding: 2px 8px;
    margin-bottom: 6px;
    text-transform: uppercase;
  }}
  .log-text {{ font-size: 0.86rem; color: #E5E7EB; }}

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

  /* ── Tables ── */
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

  /* ── Code Box ── */
  .code-box {{
    background: #0B0B18;
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 12px;
    padding: 20px;
    overflow-x: auto;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.85rem;
    color: #A78BFA;
    line-height: 1.5;
    max-height: 480px;
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
       &nbsp;·&nbsp; {cleaned_rows:,} rows × {cleaned_cols} features</p>
  </div>
  <div class="iris-badge">🌸 Iris Studio</div>
</header>

<nav>
  <a href="#s-business">📋 Business & Data</a>
  <a href="#s-health">🛡️ Quality & Health</a>
  <a href="#s-cleaning">🧹 Cleaning Log</a>
  <a href="#s-eda">📊 Exploratory Analysis</a>
  <a href="#s-corr">🔗 Correlations</a>
  <a href="#s-fe">⚙️ Feature Engineering</a>
  <a href="#s-ml">🤖 Machine Learning Baselines</a>
  <a href="#s-code">💻 Python Pipeline Code</a>
</nav>

<main>
  <section class="section" id="s-business">
    <div class="section-title"><span>📋</span> Business Understanding & Data Ingestion</div>
    {sec_overview}
  </section>

  <section class="section" id="s-health">
    <div class="section-title"><span>🛡️</span> Data Quality & Health Scorecard</div>
    {sec_health}
  </section>

  <section class="section" id="s-cleaning">
    <div class="section-title"><span>🧹</span> Data Cleaning Execution Log</div>
    {sec_cleaning}
  </section>

  <section class="section" id="s-eda">
    <div class="section-title"><span>📊</span> Exploratory Data Analysis (EDA)</div>
    {sec_eda}
  </section>

  <section class="section" id="s-corr">
    <div class="section-title"><span>🔗</span> Correlations & Bivariate Relationships</div>
    {sec_corr}
  </section>

  <section class="section" id="s-fe">
    <div class="section-title"><span>⚙️</span> Feature Engineering Strategy & Log</div>
    {sec_fe}
  </section>

  <section class="section" id="s-ml">
    <div class="section-title"><span>🤖</span> Machine Learning Baselines & Diagnostics</div>
    {sec_ml}
  </section>

  <section class="section" id="s-code">
    <div class="section-title"><span>💻</span> Reproducible Python Pipeline Script</div>
    {sec_code}
  </section>
</main>

<footer>
  Generated by <strong style="color:var(--purple);">Iris Data Science Studio</strong>
  &nbsp;·&nbsp; {now_str}
</footer>

</body>
</html>"""
    return html_doc
