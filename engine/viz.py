"""
engine/viz.py
Auto chart selector + Plotly figure builders.
"""
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, Any

IRIS_PURPLE = "#A78BFA"
IRIS_TEAL   = "#2DD4BF"
IRIS_PINK   = "#F472B6"
BG          = "#0F0F1A"
PAPER_BG    = "#1A1A2E"

PALETTE = [IRIS_PURPLE, IRIS_TEAL, IRIS_PINK, "#60A5FA", "#FBBF24", "#34D399"]

LAYOUT_BASE = dict(
    plot_bgcolor=BG,
    paper_bgcolor=PAPER_BG,
    font=dict(color="#E2E8F0", family="sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def sanitize_fig_for_html(fig):
    """
    Sanitizes a Plotly Figure object by converting any binary base64 encoded data
    ('bdata') into standard Python lists so that Plotly.js in exported standalone
    HTML files can render all traces (such as distribution histograms) correctly.
    """
    if fig is None:
        return None

    def _clean(obj):
        if isinstance(obj, dict):
            if "bdata" in obj and "dtype" in obj:
                dtype_map = {
                    "i1": np.int8, "i2": np.int16, "i4": np.int32, "i8": np.int64,
                    "u1": np.uint8, "u2": np.uint16, "u4": np.uint32, "u8": np.uint64,
                    "f4": np.float32, "f8": np.float64
                }
                dt = dtype_map.get(obj["dtype"], np.float64)
                buf = base64.b64decode(obj["bdata"])
                arr = np.frombuffer(buf, dtype=dt)
                return arr.tolist()
            return {k: _clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    try:
        fig_dict = _clean(fig.to_dict())
        return go.Figure(fig_dict)
    except Exception:
        return fig


def _apply_layout(fig, title=""):
    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, font=dict(size=16, color=IRIS_PURPLE)))
    fig.update_xaxes(gridcolor="#2D2D4E", zerolinecolor="#2D2D4E")
    fig.update_yaxes(gridcolor="#2D2D4E", zerolinecolor="#2D2D4E")
    return fig


# ── Numeric charts ─────────────────────────────────────────────────────────────

def histogram_kde(series: pd.Series, col: str):
    fig = px.histogram(series.dropna(), x=series.name or col,
                       marginal="violin", nbins=40,
                       color_discrete_sequence=[IRIS_PURPLE],
                       labels={series.name or col: col})
    return _apply_layout(fig, f"Distribution — {col}")


def box_plot(series: pd.Series, col: str):
    fig = px.box(series.dropna(), y=series.name or col,
                 color_discrete_sequence=[IRIS_TEAL],
                 points="outliers")
    return _apply_layout(fig, f"Box Plot — {col}")


def scatter_with_regression(df: pd.DataFrame, col1: str, col2: str):
    tmp = df[[col1, col2]].dropna()
    fig = px.scatter(tmp, x=col1, y=col2, trendline="ols",
                     color_discrete_sequence=[IRIS_PURPLE],
                     trendline_color_override=IRIS_TEAL)
    return _apply_layout(fig, f"{col1} vs {col2}")


# ── Categorical charts ──────────────────────────────────────────────────────────

def bar_chart(series: pd.Series, col: str, max_cats: int = 20):
    vc = series.value_counts().head(max_cats).reset_index()
    vc.columns = [col, "count"]
    fig = px.bar(vc, x="count", y=col, orientation="h",
                 color_discrete_sequence=[IRIS_PURPLE])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return _apply_layout(fig, f"Value Counts — {col}")


# ── Missing value heatmap ───────────────────────────────────────────────────────

def missing_heatmap(df: pd.DataFrame):
    sample = df.isnull().astype(int)
    if len(sample) > 500:
        sample = sample.sample(500, random_state=42)
    fig = px.imshow(sample.T,
                    color_continuous_scale=[[0, PAPER_BG], [1, IRIS_PINK]],
                    aspect="auto",
                    labels=dict(x="Row Index", y="Column", color="Missing"))
    fig.update_coloraxes(showscale=False)
    return _apply_layout(fig, "Missing Values Heatmap")


# ── Correlation heatmap ─────────────────────────────────────────────────────────

def correlation_heatmap(corr_matrix: pd.DataFrame):
    fig = px.imshow(corr_matrix,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    text_auto=".2f",
                    aspect="auto")
    return _apply_layout(fig, "Pearson Correlation Matrix")


# ── Time series ─────────────────────────────────────────────────────────────────

def time_series(df: pd.DataFrame, date_col: str, val_col: str):
    tmp = df[[date_col, val_col]].dropna().copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna().sort_values(date_col)
    fig = px.line(tmp, x=date_col, y=val_col,
                  color_discrete_sequence=[IRIS_TEAL])
    return _apply_layout(fig, f"{val_col} over {date_col}")


# ── Grouped box (numeric x categorical) ────────────────────────────────────────

def grouped_box(df: pd.DataFrame, num_col: str, cat_col: str):
    tmp = df[[num_col, cat_col]].dropna()
    top_cats = tmp[cat_col].value_counts().head(8).index
    tmp = tmp[tmp[cat_col].isin(top_cats)]
    fig = px.box(tmp, x=cat_col, y=num_col,
                 color=cat_col,
                 color_discrete_sequence=PALETTE)
    return _apply_layout(fig, f"{num_col} by {cat_col}")


# ── Outlier strip plot ──────────────────────────────────────────────────────────

def outlier_strip(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    color = s.apply(lambda v: "Outlier" if v < low or v > high else "Normal")
    tmp = pd.DataFrame({"value": s, "type": color})
    fig = px.strip(tmp, y="value", color="type",
                   color_discrete_map={"Normal": IRIS_PURPLE, "Outlier": IRIS_PINK})
    return _apply_layout(fig, f"Outliers — {col}")


# ── Feature importance ──────────────────────────────────────────────────────────

def feature_importance_chart(importances: Dict[str, float]):
    df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"])
    df = df.sort_values("Importance", ascending=True).tail(20)
    fig = px.bar(df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale=["#2D2D4E", IRIS_PURPLE])
    return _apply_layout(fig, "Feature Importance")


# ── Confusion matrix ────────────────────────────────────────────────────────────

def confusion_matrix_chart(cm: Any, labels):
    fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                    color_continuous_scale=[[0, PAPER_BG], [1, IRIS_PURPLE]],
                    labels=dict(x="Predicted", y="Actual"))
    return _apply_layout(fig, "Confusion Matrix")


# ── Elbow curve (clustering) ────────────────────────────────────────────────────

def elbow_curve(k_values, inertias):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_values, y=inertias, mode="lines+markers",
                             line=dict(color=IRIS_PURPLE, width=2),
                             marker=dict(color=IRIS_TEAL, size=8)))
    fig.update_xaxes(title="Number of Clusters (k)")
    fig.update_yaxes(title="Inertia")
    return _apply_layout(fig, "K-Means Elbow Curve")
