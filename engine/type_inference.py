"""
engine/type_inference.py
Classifies every column into: numeric, categorical, datetime, text, boolean, id
"""
import pandas as pd
import numpy as np
from typing import Dict


CATEGORY_TYPES = {"numeric", "categorical", "datetime", "text", "boolean", "id"}


def infer_column_type(series: pd.Series) -> str:
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    s = series.dropna()
    if len(s) == 0:
        return "empty"

    # Check column name semantics
    col_name = str(series.name).lower().strip() if hasattr(series, "name") and series.name is not None else ""
    non_id_terms = {"price", "cost", "salary", "wage", "revenue", "sales", "fare", "fee", "val", "value", "target", "score", "rate", "age", "weight", "height", "area", "size"}
    is_name_non_id = any(term in col_name for term in non_id_terms)
    is_name_id = any(col_name.endswith(sfx) for sfx in ("_id", "id", "key", "code", "index", "pk", "uuid")) or col_name in ("id", "index", "number")

    # Boolean check
    unique_vals = set(str(v).strip().lower() for v in s.unique())
    bool_sets = [{"true", "false"}, {"yes", "no"}, {"1", "0"}, {"y", "n"}]
    if any(unique_vals <= b for b in bool_sets):
        return "boolean"

    # Try datetime
    if s.dtype == "object":
        try:
            parsed = pd.to_datetime(s.head(100), infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.7:
                return "datetime"
        except Exception:
            pass

    # Numeric
    if pd.api.types.is_numeric_dtype(s):
        # ID heuristic: strictly integer-like and matches ID semantics or high sample count without non-id terms
        if pd.api.types.is_integer_dtype(s) and (s.nunique() / len(s) > 0.95):
            if is_name_id or (len(s) >= 100 and not is_name_non_id):
                return "id"
        return "numeric"

    # Try coercing object to numeric
    if s.dtype == "object":
        coerced = pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
        if coerced.notna().mean() > 0.8:
            return "numeric"

    # Categorical vs free text: cardinality ratio
    cardinality_ratio = s.nunique() / len(s)
    avg_len = s.astype(str).str.len().mean()

    if cardinality_ratio < 0.05 or s.nunique() <= 20:
        return "categorical"

    if avg_len > 40:
        return "text"

    if cardinality_ratio > 0.9:
        return "id"

    return "categorical"


def infer_all_types(df: pd.DataFrame) -> Dict[str, str]:
    return {col: infer_column_type(df[col]) for col in df.columns}


def get_columns_by_type(df: pd.DataFrame, col_types: Dict[str, str]) -> Dict[str, list]:
    result = {}
    for t in CATEGORY_TYPES | {"empty"}:
        result[t] = [c for c, ct in col_types.items() if ct == t]
    return result
