"""
engine/cleaning.py
Comprehensive Data Cleaning & Preprocessing Engine:
1. Handling Missing Values (Statistical Imputation: Mean, Median, Mode, Constant, Ffill & Sparse Drops)
2. Removing Duplicates (Eliminate redundant rows)
3. Treating Outliers (IQR Method & Z-Score Method — Cap or Drop)
4. Correcting Structural Errors (Trim Whitespace, Capitalisation Fixes, Special Char Cleanup, Type Conversions)
5. Categorical Encoding (Automated suggestions & Batch One-Click Encoding)
6. Feature Scaling (Automated suggestions & Batch One-Click Scaling)
7. Simple Data Reduction (One-Click Constant & Redundant Feature Drop)
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional


# ── 1. MISSING VALUES ─────────────────────────────────────────────────────────
def impute_missing(df: pd.DataFrame, column: str, method: str, fill_value: Any = None) -> Tuple[pd.DataFrame, str]:
    new_df = df.copy()
    if column not in new_df.columns:
        return new_df, f"Column '{column}' not found."
    
    s = new_df[column]
    null_cnt = int(s.isnull().sum())
    if null_cnt == 0:
        return new_df, f"No missing values in '{column}'."

    if method == "mean":
        s_num = pd.to_numeric(s, errors="coerce")
        val = float(s_num.mean())
        new_df[column] = s_num.fillna(val)
        return new_df, f"Imputed {null_cnt} missing values in '{column}' with mean ({val:.2f})."
        
    elif method == "median":
        s_num = pd.to_numeric(s, errors="coerce")
        val = float(s_num.median())
        new_df[column] = s_num.fillna(val)
        return new_df, f"Imputed {null_cnt} missing values in '{column}' with median ({val:.2f})."
        
    elif method == "mode":
        mode_series = s.mode()
        val = mode_series.iloc[0] if len(mode_series) > 0 else "Unknown"
        new_df[column] = s.fillna(val)
        return new_df, f"Imputed {null_cnt} missing values in '{column}' with mode ('{val}')."
        
    elif method == "constant":
        val = fill_value if fill_value is not None else 0
        new_df[column] = s.fillna(val)
        return new_df, f"Imputed {null_cnt} missing values in '{column}' with constant ('{val}')."
        
    elif method == "ffill":
        new_df[column] = s.ffill().bfill()
        return new_df, f"Applied forward/backward fill to '{column}'."
        
    elif method == "drop_rows":
        new_df = new_df.dropna(subset=[column]).reset_index(drop=True)
        return new_df, f"Dropped {null_cnt} rows with missing values in '{column}'."
        
    elif method == "drop_column":
        new_df = new_df.drop(columns=[column])
        return new_df, f"Dropped sparse column '{column}'."

    return new_df, "No changes made."


def drop_sparse_columns(df: pd.DataFrame, threshold_pct: float = 40.0) -> Tuple[pd.DataFrame, str]:
    new_df = df.copy()
    null_pcts = (new_df.isnull().sum() / len(new_df) * 100)
    sparse_cols = null_pcts[null_pcts > threshold_pct].index.tolist()
    if sparse_cols:
        new_df = new_df.drop(columns=sparse_cols)
        return new_df, f"Dropped {len(sparse_cols)} sparse columns (> {threshold_pct}% nulls): {', '.join(sparse_cols)}"
    return new_df, f"No columns exceeded {threshold_pct}% missing threshold."


# ── 2. DUPLICATES ─────────────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        new_df = df.drop_duplicates().reset_index(drop=True)
        return new_df, f"Removed {n_dups} duplicate rows."
    return df.copy(), "No duplicate rows found."


# ── 3. OUTLIERS (IQR & Z-SCORE) ────────────────────────────────────────────────
def treat_outliers_iqr(df: pd.DataFrame, column: str, action: str = "cap", factor: float = 1.5) -> Tuple[pd.DataFrame, str]:
    new_df = df.copy()
    if column not in new_df.columns:
        return new_df, f"Column '{column}' not found."
    
    s_num = pd.to_numeric(new_df[column], errors="coerce").astype(float)
    s_clean = s_num.dropna()
    if len(s_clean) < 4:
        return new_df, f"Insufficient numeric data in '{column}'."

    q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    outliers_mask = (s_num < lower_bound) | (s_num > upper_bound)
    outlier_cnt = int(outliers_mask.sum())

    if outlier_cnt == 0:
        return new_df, f"No IQR outliers detected in '{column}'."

    if action == "cap":
        new_df[column] = s_num.clip(lower=lower_bound, upper=upper_bound)
        return new_df, f"Capped {outlier_cnt} IQR outliers in '{column}' to [{lower_bound:.2f}, {upper_bound:.2f}]."
    elif action == "drop":
        new_df = new_df[~outliers_mask].reset_index(drop=True)
        return new_df, f"Dropped {outlier_cnt} rows with IQR outliers in '{column}'."

    return new_df, "No changes made."


def treat_outliers_zscore(df: pd.DataFrame, column: str, action: str = "cap", threshold: float = 3.0) -> Tuple[pd.DataFrame, str]:
    new_df = df.copy()
    if column not in new_df.columns:
        return new_df, f"Column '{column}' not found."

    s_num = pd.to_numeric(new_df[column], errors="coerce").astype(float)
    s_clean = s_num.dropna()
    if len(s_clean) < 4:
        return new_df, f"Insufficient numeric data in '{column}'."

    mean, std = s_clean.mean(), s_clean.std()
    if std == 0 or pd.isna(std):
        return new_df, f"Zero variance in '{column}'."

    z_scores = np.abs((s_num - mean) / std)
    outliers_mask = z_scores > threshold
    outlier_cnt = int(outliers_mask.sum())

    if outlier_cnt == 0:
        return new_df, f"No Z-Score outliers (|Z| > {threshold}) detected in '{column}'."

    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std

    if action == "cap":
        new_df[column] = s_num.clip(lower=lower_bound, upper=upper_bound)
        return new_df, f"Capped {outlier_cnt} Z-Score outliers in '{column}' to [{lower_bound:.2f}, {upper_bound:.2f}]."

        return new_df, f"Capped {outlier_cnt} Z-Score outliers in '{column}' to [{lower_bound:.2f}, {upper_bound:.2f}]."
    elif action == "drop":
        new_df = new_df[~outliers_mask].reset_index(drop=True)
        return new_df, f"Dropped {outlier_cnt} rows with Z-score outliers in '{column}'."

    return new_df, "No changes made."


# ── 4. STRUCTURAL ERRORS & TYPE CORRECTION ────────────────────────────────────
def clean_structural_errors(df: pd.DataFrame, column: str, action: str) -> Tuple[pd.DataFrame, str]:
    new_df = df.copy()
    if column not in new_df.columns:
        return new_df, f"Column '{column}' not found."
    
    s = new_df[column]
    
    if action == "trim_whitespace":
        if pd.api.types.is_string_dtype(s) or s.dtype == object:
            new_df[column] = s.astype(str).str.strip()
            return new_df, f"Trimmed leading/trailing whitespaces in '{column}'."
        return new_df, f"'{column}' is not text."

    elif action == "lowercase":
        new_df[column] = s.astype(str).str.lower()
        return new_df, f"Converted '{column}' to lowercase."

    elif action == "uppercase":
        new_df[column] = s.astype(str).str.upper()
        return new_df, f"Converted '{column}' to UPPERCASE."

    elif action == "titlecase":
        new_df[column] = s.astype(str).str.title()
        return new_df, f"Converted '{column}' to Title Case."

    elif action == "remove_special":
        new_df[column] = s.astype(str).str.replace(r"[^\w\s]", "", regex=True)
        return new_df, f"Removed special punctuation/symbols from '{column}'."

    return new_df, "No action applied."


def detect_structural_errors(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Scans dataset text/object columns for structural errors:
    - Whitespace padding (leading/trailing spaces)
    - Mixed capitalization (e.g. 'male' vs 'Male')
    - Unparsed numeric strings
    """
    issues = []
    if df is None or df.empty:
        return issues

    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
            
        if s.dtype == object or str(s.dtype) == "string":
            s_str = s.astype(str)
            
            # 1. Whitespace padding check
            has_spaces = (s_str.str.strip() != s_str).any()
            if has_spaces:
                issues.append({
                    "column": col,
                    "issue_type": "whitespace",
                    "title": f"Extra Leading/Trailing Whitespaces in '{col}'",
                    "reason": f"Column '{col}' has text entries with leading/trailing spaces (e.g., ' Male '). This causes duplicate categories in analysis.",
                    "recommended_action": "trim_whitespace",
                    "action_label": "Trim Whitespaces"
                })

            # 2. Inconsistent capitalization check
            lowered = s_str.str.lower()
            if s_str.nunique() > lowered.nunique():
                issues.append({
                    "column": col,
                    "issue_type": "casing",
                    "title": f"Inconsistent Capitalization in '{col}'",
                    "reason": f"Column '{col}' has mixed casing (e.g. 'male' vs 'Male'). Standardize to Title Case or Lowercase for clean categories.",
                    "recommended_action": "titlecase",
                    "action_label": "Convert to Title Case"
                })

            # 3. Numeric stored as string check
            s_num = pd.to_numeric(s, errors="coerce")
            if s_num.notnull().sum() > (len(s) * 0.8):
                issues.append({
                    "column": col,
                    "issue_type": "type_numeric",
                    "title": f"Numeric Data Stored as Text in '{col}'",
                    "reason": f"Column '{col}' is stored as text but contains >80% numeric values. Re-cast to Numeric type for statistics.",
                    "recommended_action": "cast_numeric",
                    "action_label": "Convert to Numeric"
                })

    return issues



def convert_column_type(df: pd.DataFrame, column: str, target_type: str) -> Tuple[pd.DataFrame, str]:
    new_df = df.copy()
    if column not in new_df.columns:
        return new_df, f"Column '{column}' not found."

    try:
        if target_type == "numeric":
            new_df[column] = pd.to_numeric(new_df[column], errors="coerce")
            return new_df, f"Converted '{column}' to numeric (float/int)."
        elif target_type == "datetime":
            new_df[column] = pd.to_datetime(new_df[column], errors="coerce")
            return new_df, f"Converted '{column}' to datetime."
        elif target_type == "categorical":
            new_df[column] = new_df[column].astype(str)
            return new_df, f"Converted '{column}' to categorical text."
        elif target_type == "boolean":
            new_df[column] = new_df[column].astype(bool)
            return new_df, f"Converted '{column}' to boolean."
    except Exception as e:
        return new_df, f"Error converting '{column}' to {target_type}: {e}"

    return new_df, "No conversion applied."


# ── 5. AUTOMATED CATEGORICAL ENCODING & SUGGESTIONS ─────────────────────────────
def get_encoding_suggestions(df: pd.DataFrame, col_types: Dict[str, str]) -> List[Dict[str, Any]]:
    """Generates intelligent categorical encoding recommendations for every categorical column."""
    suggestions = []
    cat_cols = [c for c, t in col_types.items() if t in ("categorical", "boolean") and c in df.columns]
    
    for col in cat_cols:
        s = df[col].dropna()
        n_unique = int(s.nunique())
        
        if n_unique == 2:
            rec_method = "binary"
            rec_name = "Binary Encoding (0/1)"
            reason = "Exactly 2 unique categories -> ideal for 0/1 binary integer flag."
        elif n_unique <= 10:
            rec_method = "one_hot"
            rec_name = f"One-Hot Encoding ({n_unique} dummy columns)"
            reason = f"Low cardinality ({n_unique} categories) -> ideal for One-Hot dummy columns."
        else:
            rec_method = "label"
            rec_name = f"Label/Ordinal Encoding"
            reason = f"High cardinality ({n_unique} unique categories) -> Label Encoding prevents sparse matrix expansion."
            
        suggestions.append({
            "column": col,
            "n_unique": n_unique,
            "method": rec_method,
            "method_name": rec_name,
            "reason": reason
        })
    return suggestions


def encode_categorical(df: pd.DataFrame, column: str, method: str) -> Tuple[pd.DataFrame, str]:
    new_df = df.copy()
    if column not in new_df.columns:
        return new_df, f"Column '{column}' not found."
        
    s = new_df[column]
    
    if method == "one_hot":
        dummies = pd.get_dummies(s, prefix=column, drop_first=False, dtype=int)
        new_df = pd.concat([new_df, dummies], axis=1)
        return new_df, f"Applied One-Hot Encoding to '{column}' ({len(dummies.columns)} dummy columns created)."
        
    elif method == "label":
        categories = s.dropna().unique()
        mapping = {val: idx for idx, val in enumerate(categories)}
        new_col = f"{column}_encoded"
        new_df[new_col] = s.map(mapping).fillna(-1).astype(int)
        return new_df, f"Applied Label Encoding to '{column}' -> Created '{new_col}'."

    elif method == "binary":
        uniques = s.dropna().unique()
        if len(uniques) == 2:
            mapping = {uniques[0]: 0, uniques[1]: 1}
            new_col = f"{column}_binary"
            new_df[new_col] = s.map(mapping).fillna(0).astype(int)
            return new_df, f"Applied Binary Encoding to '{column}' -> Created '{new_col}'."
        else:
            mapping = {val: idx for idx, val in enumerate(uniques)}
            new_col = f"{column}_binary"
            new_df[new_col] = s.map(mapping).fillna(0).astype(int)
            return new_df, f"Applied Integer Encoding to '{column}' -> Created '{new_col}'."

    return new_df, "No encoding applied."


def auto_encode_all_categorical(df: pd.DataFrame, col_types: Dict[str, str]) -> Tuple[pd.DataFrame, str]:
    """Applies recommended encoding to ALL categorical columns automatically in 1 click."""
    new_df = df.copy()
    suggestions = get_encoding_suggestions(new_df, col_types)
    if not suggestions:
        return new_df, "No categorical columns found for encoding."
        
    encoded_count = 0
    actions = []
    for sug in suggestions:
        col = sug["column"]
        method = sug["method"]
        new_df, msg = encode_categorical(new_df, col, method)
        encoded_count += 1
        actions.append(f"{col} ({method})")
        
    return new_df, f"Auto-encoded {encoded_count} categorical columns: {', '.join(actions)}"


# ── 6. AUTOMATED FEATURE SCALING & SUGGESTIONS ──────────────────────────────────
def get_scaling_suggestions(df: pd.DataFrame, col_types: Dict[str, str]) -> List[Dict[str, Any]]:
    """Generates intelligent scaling recommendations for every numeric column."""
    suggestions = []
    num_cols = [c for c, t in col_types.items() if t == "numeric" and c in df.columns]

    # Skip already-engineered columns to prevent double-transformation (e.g. _scaled_scaled)
    ENGINEERED_SUFFIXES = ("_scaled", "_minmax", "_encoded", "_binary", "_log", "_log1p", "_zscore")
    num_cols = [c for c in num_cols if not any(c.endswith(sfx) for sfx in ENGINEERED_SUFFIXES)]

    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 2:
            continue
            
        skew = float(s.skew()) if len(s) > 3 else 0.0
        min_v, max_v = float(s.min()), float(s.max())
        
        if min_v >= 0.0 and max_v <= 1.0:
            rec_method = "none"
            rec_name = "Already Scaled [0, 1]"
            reason = "Values are already bounded between 0 and 1."
        elif abs(skew) > 1.5 or (min_v < 0 and max_v > 100):
            rec_method = "standard"
            rec_name = "Standardization (Z-Score)"
            reason = f"High variance/skew ({skew:.2f}) -> Z-score scaling (Mean=0, Std=1) centers distribution."
        else:
            rec_method = "minmax"
            rec_name = "Min-Max Scaling [0, 1]"
            reason = f"Bounded continuous range [{min_v:.1f}, {max_v:.1f}] -> Min-Max scaling rescales cleanly to [0, 1]."
            
        suggestions.append({
            "column": col,
            "skew": skew,
            "min": min_v,
            "max": max_v,
            "method": rec_method,
            "method_name": rec_name,
            "reason": reason
        })
    return suggestions


def scale_feature(df: pd.DataFrame, column: str, method: str) -> Tuple[pd.DataFrame, str]:
    new_df = df.copy()
    if column not in new_df.columns:
        return new_df, f"Column '{column}' not found."

    s_num = pd.to_numeric(new_df[column], errors="coerce")
    
    if method == "standard":
        mean, std = s_num.mean(), s_num.std()
        if std == 0 or pd.isna(std):
            std = 1.0
        new_col = f"{column}_scaled"
        new_df[new_col] = (s_num - mean) / std
        return new_df, f"Applied Standardization (Z-Score) to '{column}' -> Created '{new_col}' (mean=0, std=1)."
        
    elif method == "minmax":
        min_v, max_v = s_num.min(), s_num.max()
        rng = max_v - min_v
        if rng == 0 or pd.isna(rng):
            rng = 1.0
        new_col = f"{column}_minmax"
        new_df[new_col] = (s_num - min_v) / rng
        return new_df, f"Applied Min-Max Scaling to '{column}' -> Created '{new_col}' ([0, 1] range)."

    return new_df, "No scaling applied."


def auto_scale_all_numeric(df: pd.DataFrame, col_types: Dict[str, str]) -> Tuple[pd.DataFrame, str]:
    """Applies recommended scaling to ALL numeric columns automatically in 1 click."""
    new_df = df.copy()
    suggestions = get_scaling_suggestions(new_df, col_types)
    if not suggestions:
        return new_df, "No numeric columns found for scaling."
        
    scaled_count = 0
    actions = []
    for sug in suggestions:
        col = sug["column"]
        method = sug["method"]
        if method == "none":
            continue
        new_df, msg = scale_feature(new_df, col, method)
        scaled_count += 1
        actions.append(f"{col} ({method})")
        
    return new_df, f"Auto-scaled {scaled_count} numeric columns: {', '.join(actions)}"


# ── 7. SIMPLE DATA REDUCTION ──────────────────────────────────────────────────
def drop_constant_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    new_df = df.copy()
    constant_cols = [c for c in new_df.columns if new_df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        new_df = new_df.drop(columns=constant_cols)
        return new_df, f"Data Reduction: Dropped {len(constant_cols)} zero-variance constant columns: {', '.join(constant_cols)}"
    return new_df, "No constant columns found."


def drop_high_correlation_features(df: pd.DataFrame, threshold: float = 0.90) -> Tuple[pd.DataFrame, str]:
    new_df = df.copy()
    num_cols = new_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return new_df, "Need at least 2 numeric columns for correlation reduction."

    corr_matrix = new_df[num_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    if to_drop:
        new_df = new_df.drop(columns=to_drop)
        return new_df, f"Data Reduction: Dropped {len(to_drop)} redundant collinear features (|r| > {threshold}): {', '.join(to_drop)}"
    return new_df, f"No feature pairs exceeded correlation threshold of {threshold}."


# ── 8. RECOMMENDATIONS & HEALTH SCORE ENGINE ─────────────────────────────────
def detect_cleaning_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Analyzes dataset and returns prioritized cleaning recommendations.
    """
    recs = []
    if df is None or df.empty:
        return recs

    n_rows = len(df)

    # 1. Check ID/High Cardinality columns
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        if n_unique == n_rows and df[col].dtype == "object":
            recs.append({
                "priority": "CRITICAL",
                "title": f"Drop Unique Identifier Column",
                "col_name": col,
                "reason": f"Column '{col}' has 100% unique text values. This is an ID column with no predictive signal.",
                "action_type": "drop_col",
                "suggested_fix": f"Drop column '{col}'"
            })

    # 2. Check Missing Values
    null_counts = df.isnull().sum()
    for col, null_cnt in null_counts.items():
        if null_cnt > 0:
            pct = (null_cnt / n_rows) * 100
            if pct > 40.0:
                recs.append({
                    "priority": "CRITICAL",
                    "title": "Drop Highly Sparse Column",
                    "col_name": col,
                    "reason": f"'{col}' has {pct:.1f}% missing values. Imputation may introduce heavy bias.",
                    "action_type": "drop_col",
                    "suggested_fix": f"Drop column '{col}'"
                })
            else:
                s_num = pd.to_numeric(df[col], errors="coerce")
                is_num = s_num.notnull().sum() > (n_rows * 0.5)
                method = "median" if is_num else "mode"
                recs.append({
                    "priority": "HIGH",
                    "title": f"Impute Missing Values ({pct:.1f}% missing)",
                    "col_name": col,
                    "reason": f"'{col}' has {null_cnt} missing cells. Suggested method: {method.title()} Imputation.",
                    "action_type": "impute",
                    "suggested_fix": f"Impute '{col}' using {method}"
                })

    # 3. Check Duplicate Rows
    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        recs.append({
            "priority": "HIGH",
            "title": f"Remove Duplicate Rows",
            "col_name": None,
            "reason": f"Found {n_dups} duplicate rows ({n_dups/n_rows*100:.1f}% of dataset). Duplicate rows skew statistics.",
            "action_type": "drop_dups",
            "suggested_fix": "Remove duplicate rows"
        })

    # 4. Check Outliers in Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        s_clean = df[col].dropna()
        if len(s_clean) >= 4:
            q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = s_clean[(s_clean < (q1 - 1.5 * iqr)) | (s_clean > (q3 + 1.5 * iqr))]
                if len(outliers) > 0:
                    recs.append({
                        "priority": "MEDIUM",
                        "title": f"Cap Outliers in Numeric Column",
                        "col_name": col,
                        "reason": f"Found {len(outliers)} statistical outliers in '{col}' using IQR fence bounds.",
                        "action_type": "cap_outliers",
                        "suggested_fix": f"Cap IQR outliers in '{col}'"
                    })

    # Sort priority: CRITICAL > HIGH > MEDIUM > INFO
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "INFO": 3}
    recs.sort(key=lambda x: priority_order.get(x["priority"], 99))
    return recs


def compute_data_health_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes an overall Data Health Score (0-100) and breakdown stats.
    """
    if df is None or df.empty:
        return {"score": 100, "missing_cols": 0, "dup_rows": 0, "outlier_cols": 0}

    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    
    # 1. Missing Score (max 40 pts penalty)
    null_cells = int(df.isnull().sum().sum())
    missing_ratio = null_cells / total_cells if total_cells > 0 else 0
    missing_penalty = min(40, missing_ratio * 100 * 2)

    # 2. Duplicate Score (max 30 pts penalty)
    dup_rows = int(df.duplicated().sum())
    dup_ratio = dup_rows / n_rows if n_rows > 0 else 0
    dup_penalty = min(30, dup_ratio * 100 * 3)

    # 3. Outlier Score (max 30 pts penalty)
    outlier_cols = 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        s_clean = df[col].dropna()
        if len(s_clean) >= 4:
            q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0 and len(s_clean[(s_clean < (q1 - 1.5 * iqr)) | (s_clean > (q3 + 1.5 * iqr))]) > 0:
                outlier_cols += 1

    outlier_ratio = outlier_cols / len(num_cols) if len(num_cols) > 0 else 0
    outlier_penalty = min(30, outlier_ratio * 30)

    score = int(max(0, round(100 - missing_penalty - dup_penalty - outlier_penalty)))
    missing_cols = int((df.isnull().sum() > 0).sum())

    return {
        "score": score,
        "missing_cols": missing_cols,
        "dup_rows": dup_rows,
        "outlier_cols": outlier_cols,
        "missing_ratio_pct": round(missing_ratio * 100, 1)
    }


def get_before_after_stats(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns comparative snapshot statistics between raw dataset and cleaned dataset.
    """
    before_health = compute_data_health_score(df_before)
    after_health = compute_data_health_score(df_after)

    return {
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "cols_before": df_before.shape[1],
        "cols_after": df_after.shape[1],
        "score_before": before_health["score"],
        "score_after": after_health["score"],
        "score_diff": after_health["score"] - before_health["score"],
        "missing_cells_before": int(df_before.isnull().sum().sum()),
        "missing_cells_after": int(df_after.isnull().sum().sum())
    }

