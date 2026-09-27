"""
engine/eda.py
Per-column stats and correlation analysis.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional


def column_stats(series: pd.Series, col_type: str) -> Dict[str, Any]:
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    s = series.dropna()

    def _scalar_int(val):
        if isinstance(val, (pd.Series, np.ndarray)):
            return int(val.iloc[0]) if len(val) > 0 else 0
        return int(val)

    def _scalar_float(val):
        if isinstance(val, (pd.Series, np.ndarray)):
            return float(val.iloc[0]) if len(val) > 0 else 0.0
        return float(val)

    base = {
        "count": _scalar_int(len(s)),
        "missing": _scalar_int(series.isnull().sum()),
        "unique": _scalar_int(s.nunique()),
        "dtype": str(series.dtype),
        "col_type": col_type,
    }

    if col_type == "numeric":
        base.update({
            "mean": round(_scalar_float(s.mean()), 4),
            "median": round(_scalar_float(s.median()), 4),
            "std": round(_scalar_float(s.std()), 4),
            "min": round(_scalar_float(s.min()), 4),
            "max": round(_scalar_float(s.max()), 4),
            "q1": round(_scalar_float(s.quantile(0.25)), 4),
            "q3": round(_scalar_float(s.quantile(0.75)), 4),
            "iqr": round(_scalar_float(s.quantile(0.75) - s.quantile(0.25)), 4),
            "skewness": round(_scalar_float(s.skew()), 4),
            "kurtosis": round(_scalar_float(s.kurtosis()), 4),
        })
        # Try mode safely
        try:
            base["mode"] = round(_scalar_float(s.mode().iloc[0]), 4)
        except Exception:
            base["mode"] = None

    elif col_type == "categorical":
        vc = s.value_counts()
        base.update({
            "top_value": str(vc.index[0]) if len(vc) > 0 else None,
            "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
            "top_freq_pct": round(float(vc.iloc[0] / len(s) * 100), 2) if len(s) > 0 else 0,
            "value_counts": vc.head(20).to_dict(),
        })

    elif col_type == "datetime":
        parsed = pd.to_datetime(s, errors="coerce")
        base.update({
            "min_date": str(parsed.min()),
            "max_date": str(parsed.max()),
            "range_days": int((parsed.max() - parsed.min()).days) if parsed.notna().any() else 0,
        })

    elif col_type == "text":
        lengths = s.astype(str).str.len()
        base.update({
            "avg_length": round(float(lengths.mean()), 1),
            "max_length": int(lengths.max()),
            "min_length": int(lengths.min()),
        })

    return base


def all_column_stats(df: pd.DataFrame, col_types: Dict[str, str]) -> Dict[str, Dict]:
    res = {}
    for col in df.columns:
        c_type = col_types.get(col, "numeric")
        res[col] = column_stats(df[col], c_type)
    return res


def correlation_matrix(df: pd.DataFrame, col_types: Dict[str, str]) -> Dict[str, Any]:
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    if len(numeric_cols) < 2:
        return {"pearson": None, "spearman": None, "top_pairs": []}

    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    pearson = num_df.corr(method="pearson").round(4)
    spearman = num_df.corr(method="spearman").round(4)

    # Top correlated pairs (excluding diagonal)
    pairs = []
    cols = pearson.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = pearson.iloc[i, j]
            if not np.isnan(r):
                pairs.append({"col1": cols[i], "col2": cols[j], "pearson_r": round(r, 4)})

    pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

    return {
        "pearson": pearson,
        "spearman": spearman,
        "numeric_cols": numeric_cols,
        "top_pairs": pairs[:10],
    }


def normality_test(series: pd.Series) -> Dict[str, Any]:
    """Performs Shapiro-Wilk or D'Agostino normality test on numeric series."""
    s_clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(s_clean) < 8:
        return {"is_normal": False, "p_value": 0.0, "interpretation": "Insufficient sample size"}
    
    # Subsample if large for Shapiro-Wilk limit (5000)
    sample = s_clean.sample(min(len(s_clean), 1000), random_state=42)
    try:
        stat, p_val = stats.shapiro(sample)
        is_normal = p_val > 0.05
        interp = "Normal Distribution (p > 0.05)" if is_normal else "Non-Normal / Skewed (p <= 0.05)"
        return {"stat": round(float(stat), 4), "p_value": round(float(p_val), 4), "is_normal": is_normal, "interpretation": interp}
    except Exception:
        return {"is_normal": False, "p_value": 0.0, "interpretation": "Test inconclusive"}


def detect_class_imbalance(series: pd.Series) -> Dict[str, Any]:
    """Detects class imbalance ratio in categorical column."""
    vc = series.dropna().value_counts()
    if len(vc) < 2:
        return {"is_imbalanced": False, "ratio": 1.0, "majority_pct": 100.0}
    
    top_cnt = vc.iloc[0]
    sec_cnt = vc.iloc[1] if len(vc) > 1 else 1
    total = len(series.dropna())
    ratio = round(top_cnt / max(1, sec_cnt), 2)
    maj_pct = round((top_cnt / total) * 100, 1)
    
    return {
        "is_imbalanced": ratio > 3.0 or maj_pct > 75.0,
        "ratio": ratio,
        "majority_class": str(vc.index[0]),
        "majority_pct": maj_pct
    }


def generate_eda_key_findings(df: pd.DataFrame, col_types: Dict[str, str]) -> List[str]:
    """Generates structured top insights from dataset statistics."""
    findings = []
    if df is None or df.empty:
        return findings

    # 1. Correlation findings
    corr_res = correlation_matrix(df, col_types)
    if corr_res and corr_res.get("top_pairs"):
        top_p = corr_res["top_pairs"][0]
        if abs(top_p["pearson_r"]) > 0.5:
            rel = "positive" if top_p["pearson_r"] > 0 else "negative"
            findings.append(f"Strong {rel} correlation found between **{top_p['col1']}** and **{top_p['col2']}** (r = {top_p['pearson_r']:.2f}).")

    # 2. Skewness / Normality findings
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 0 and abs(s.skew()) > 1.5:
            direction = "right" if s.skew() > 0 else "left"
            findings.append(f"Column **{col}** is heavily {direction}-skewed (skewness = {s.skew():.2f}). Consider log-transform.")
            break

    # 3. Class imbalance findings
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    for col in cat_cols:
        imb = detect_class_imbalance(df[col])
        if imb["is_imbalanced"]:
            findings.append(f"Class imbalance detected in **{col}**: Majority class '{imb['majority_class']}' makes up {imb['majority_pct']}% of records.")
            break

    # 4. Missing data warning
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if len(null_cols) > 0:
        worst_col = null_cols.idxmax()
        worst_pct = (null_cols.max() / len(df)) * 100
        findings.append(f"Missing data alert: **{worst_col}** has the highest missing rate ({worst_pct:.1f}% missing cells).")

    # 5. Dataset shape summary
    findings.append(f"Dataset contains **{len(df):,} rows** across **{len(df.columns)} features** ({len(num_cols)} numeric, {len(cat_cols)} categorical).")

    return findings

