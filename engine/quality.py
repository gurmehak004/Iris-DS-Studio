"""
engine/quality.py
Data quality report: missing values, duplicates, outliers, skewness.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    return pd.DataFrame({
        "column": df.columns,
        "missing_count": missing.values,
        "missing_pct": pct.values,
    }).query("missing_count > 0").sort_values("missing_pct", ascending=False).reset_index(drop=True)


def duplicate_report(df: pd.DataFrame) -> Dict[str, Any]:
    n_dups = df.duplicated().sum()
    return {
        "duplicate_rows": int(n_dups),
        "duplicate_pct": round(n_dups / len(df) * 100, 2),
    }


def outlier_report(df: pd.DataFrame, col_types: Dict[str, str]) -> pd.DataFrame:
    rows = []
    numeric_cols = [c for c, t in col_types.items() if t == "numeric" and c in df.columns]
    for col in numeric_cols:
        col_s = df[col].iloc[:, 0] if isinstance(df[col], pd.DataFrame) else df[col]
        s = col_s.dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        z = np.abs(stats.zscore(s))
        z_outliers = int((z > 3).sum())
        rows.append({
            "column": col,
            "iqr_outliers": iqr_outliers,
            "zscore_outliers": z_outliers,
            "iqr_outlier_pct": round(iqr_outliers / len(s) * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("iqr_outliers", ascending=False).reset_index(drop=True)


def skewness_report(df: pd.DataFrame, col_types: Dict[str, str]) -> pd.DataFrame:
    numeric_cols = [c for c, t in col_types.items() if t == "numeric" and c in df.columns]
    rows = []
    for col in numeric_cols:
        col_s = df[col].iloc[:, 0] if isinstance(df[col], pd.DataFrame) else df[col]
        s = col_s.dropna()
        if len(s) < 4:
            continue
        skew_val = s.skew()
        skew_f = float(skew_val.iloc[0]) if isinstance(skew_val, (pd.Series, np.ndarray)) else float(skew_val)
        kurt_val = s.kurtosis()
        kurt_f = float(kurt_val.iloc[0]) if isinstance(kurt_val, (pd.Series, np.ndarray)) else float(kurt_val)
        rows.append({"column": col, "skewness": round(skew_f, 4), "kurtosis": round(kurt_f, 4)})
    return pd.DataFrame(rows)


def full_quality_report(df: pd.DataFrame, col_types: Dict[str, str]) -> Dict[str, Any]:
    return {
        "missing": missing_report(df),
        "duplicates": duplicate_report(df),
        "outliers": outlier_report(df, col_types),
        "skewness": skewness_report(df, col_types),
        "shape": df.shape,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }
