"""
engine/feature_engineering.py
Data Science Feature Engineering engine.
Supports automated feature recommendations, 1-click batch feature engineering,
and plain-English explanations for non-technical users.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


def get_feature_engineering_suggestions(df: pd.DataFrame, col_types: Dict[str, str]) -> List[Dict[str, Any]]:
    """Generate smart per-column feature engineering suggestions."""
    suggestions = []
    
    for col, ctype in col_types.items():
        if col not in df.columns:
            continue
            
        col_data = df[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]

        s = col_data.dropna()
        if len(s) == 0:
            continue
            
        col_suggs = {
            "column": col,
            "type": ctype,
            "actions": [],
            "reason": ""
        }
        
        if ctype == "numeric":
            try:
                skew_val = s.skew()
                skew = float(skew_val.iloc[0]) if isinstance(skew_val, (pd.Series, np.ndarray)) else float(skew_val)
            except Exception:
                skew = 0.0

            if abs(skew) > 1.0:
                col_suggs["actions"].append({
                    "name": "Log Transform (log1p)",
                    "type": "log",
                    "badge": "Skew Fix",
                    "desc": f"Fix high skewness ({skew:.2f}) to normalize distribution"
                })
            
            col_suggs["actions"].append({
                "name": "Standard Scaling (Z-Score)",
                "type": "standard_scale",
                "badge": "Scale",
                "desc": "Mean = 0, Std = 1. Ideal for distance-based & linear algorithms"
            })
            
            col_suggs["actions"].append({
                "name": "Min-Max Scaling [0,1]",
                "type": "minmax_scale",
                "badge": "Scale",
                "desc": "Rescale values into strict [0, 1] range"
            })
            
            col_suggs["actions"].append({
                "name": "Bin into 4 Quantiles (Quartiles)",
                "type": "bin_quantiles",
                "badge": "Binning",
                "desc": "Convert continuous variable into Low/Med-Low/Med-High/High categories"
            })
            
        elif ctype in ("categorical", "boolean"):
            nu = s.nunique()
            n_unique = int(nu.iloc[0]) if isinstance(nu, (pd.Series, np.ndarray)) else int(nu)
            if n_unique == 2:
                col_suggs["actions"].append({
                    "name": "Binary Encode (0/1)",
                    "type": "binary_encode",
                    "badge": "Encoding",
                    "desc": "Convert binary categories to 0 and 1 integer flags"
                })
            elif n_unique <= 10:
                col_suggs["actions"].append({
                    "name": f"One-Hot Encode ({n_unique} dummy cols)",
                    "type": "one_hot",
                    "badge": "Encoding",
                    "desc": f"Create {n_unique} binary indicator columns"
                })
            else:
                col_suggs["actions"].append({
                    "name": f"Frequency/Target Encode ({n_unique} categories)",
                    "type": "freq_encode",
                    "badge": "High Cardinality",
                    "desc": "Replace category strings with their occurrence frequency"
                })
                
        elif ctype == "datetime":
            col_suggs["actions"].append({
                "name": "Decompose Datetime (Year, Month, Day, DayOfWeek, IsWeekend)",
                "type": "datetime_decompose",
                "badge": "Time Features",
                "desc": "Extract cyclical and calendar indicators from date"
            })
            
        elif ctype == "text":
            col_suggs["actions"].append({
                "name": "Extract Character Length & Word Count",
                "type": "text_stats",
                "badge": "NLP Feature",
                "desc": "Create text length and word count numerical features"
            })
            
        elif ctype == "id":
            col_suggs["actions"].append({
                "name": "Drop Identifier Column",
                "type": "drop",
                "badge": "Cleanup",
                "desc": "ID columns add noise and cause overfitting"
            })
            
        suggestions.append(col_suggs)
        
    return suggestions


def apply_transformation(df: pd.DataFrame, column: str, transform_type: str, **kwargs) -> Tuple[pd.DataFrame, str]:
    """Applies a specified transformation to the dataframe column."""
    new_df = df.copy()
    if column not in new_df.columns:
        return new_df, f"Column '{column}' not found in dataframe."
        
    s_raw = new_df[column]
    s = s_raw.iloc[:, 0] if isinstance(s_raw, pd.DataFrame) else s_raw
    
    if transform_type == "log":
        s_num = pd.to_numeric(s, errors="coerce")
        min_v = s_num.min()
        shift = abs(min_v) + 1.0 if min_v < 0 else 0.0
        new_col = f"{column}_log"
        new_df[new_col] = np.log1p(s_num + shift)
        return new_df, f"Created log-transformed feature '{new_col}'"
        
    elif transform_type == "standard_scale":
        s_num = pd.to_numeric(s, errors="coerce")
        mean, std = s_num.mean(), s_num.std()
        if std == 0 or pd.isna(std):
            std = 1.0
        new_col = f"{column}_scaled"
        new_df[new_col] = (s_num - mean) / std
        return new_df, f"Created Z-score scaled feature '{new_col}'"
        
    elif transform_type == "minmax_scale":
        s_num = pd.to_numeric(s, errors="coerce")
        min_v, max_v = s_num.min(), s_num.max()
        rng = max_v - min_v
        if rng == 0 or pd.isna(rng):
            rng = 1.0
        new_col = f"{column}_minmax"
        new_df[new_col] = (s_num - min_v) / rng
        return new_df, f"Created MinMax scaled feature '{new_col}'"
        
    elif transform_type == "bin_quantiles":
        s_num = pd.to_numeric(s, errors="coerce")
        labels = ["Q1_Low", "Q2_MedLow", "Q3_MedHigh", "Q4_High"]
        new_col = f"{column}_quantile_bin"
        new_df[new_col] = pd.qcut(s_num, q=4, labels=labels, duplicates="drop")
        return new_df, f"Created 4-quantile binned feature '{new_col}'"
        
    elif transform_type == "binary_encode":
        uniques = s.dropna().unique()
        mapping = {val: i for i, val in enumerate(uniques)}
        new_col = f"{column}_encoded"
        new_df[new_col] = s.map(mapping).fillna(0).astype(int)
        return new_df, f"Created binary encoded feature '{new_col}'"
        
    elif transform_type == "one_hot":
        dummies = pd.get_dummies(s, prefix=column, drop_first=False, dtype=int)
        new_df = pd.concat([new_df, dummies], axis=1)
        return new_df, f"Created {len(dummies.columns)} One-Hot dummy features from '{column}'"
        
    elif transform_type == "freq_encode":
        freqs = s.value_counts(normalize=True).to_dict()
        new_col = f"{column}_freq"
        new_df[new_col] = s.map(freqs).fillna(0.0)
        return new_df, f"Created frequency-encoded feature '{new_col}'"
        
    elif transform_type == "datetime_decompose":
        dt = pd.to_datetime(s, errors="coerce")
        prefix = column
        new_df[f"{prefix}_year"] = dt.dt.year
        new_df[f"{prefix}_month"] = dt.dt.month
        new_df[f"{prefix}_day"] = dt.dt.day
        new_df[f"{prefix}_dayofweek"] = dt.dt.dayofweek
        new_df[f"{prefix}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        return new_df, f"Decomposed '{column}' into 5 datetime features (year, month, day, dayofweek, is_weekend)"
        
    elif transform_type == "text_stats":
        s_str = s.astype(str)
        new_df[f"{column}_char_len"] = s_str.str.len()
        new_df[f"{column}_word_cnt"] = s_str.str.split().str.len()
        return new_df, f"Created text statistics features for '{column}'"
        
    elif transform_type == "drop":
        new_df = new_df.drop(columns=[column])
        return new_df, f"Dropped feature '{column}'"

    return new_df, "No transformation applied."


def auto_engineer_all_features(df: pd.DataFrame, col_types: Dict[str, str]) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    """
    1-Click Automated Feature Engineering Engine.
    Applies optimal transformations across the dataset and returns:
    (new_df, list_of_explanation_cards, summary_message)
    """
    new_df = df.copy()
    explanations = []
    actions_count = 0

    for col, ctype in col_types.items():
        if col not in new_df.columns:
            continue
            
        col_data = new_df[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]

        s = col_data.dropna()
        if len(s) == 0:
            continue

        if ctype == "numeric":
            try:
                skew_val = s.skew()
                skew = float(skew_val.iloc[0]) if isinstance(skew_val, (pd.Series, np.ndarray)) else float(skew_val)
            except Exception:
                skew = 0.0

            # Log transform for skewed features
            if abs(skew) > 1.0:
                new_df, _ = apply_transformation(new_df, col, "log")
                actions_count += 1
                explanations.append({
                    "column": col,
                    "new_feature": f"{col}_log",
                    "badge": "Skew Normalization",
                    "badge_color": "#8B5CF6",
                    "transformation": "Log Transformation (log1p)",
                    "why": f"'{col}' was heavily skewed (skew = {skew:.2f}). Taking the log compresses extreme outlier values so machine learning models can find linear relationships easily."
                })

            # Standardization Z-score
            new_df, _ = apply_transformation(new_df, col, "standard_scale")
            actions_count += 1
            explanations.append({
                "column": col,
                "new_feature": f"{col}_scaled",
                "badge": "Z-Score Scaling",
                "badge_color": "#06B6D4",
                "transformation": "Standard Scaling (Mean = 0, Std = 1)",
                "why": f"Rescales '{col}' to Mean = 0 and Std = 1. This prevents larger magnitude numbers from dominating models like KNN, SVM, or Neural Networks."
            })

        elif ctype in ("categorical", "boolean"):
            nu = s.nunique()
            n_unique = int(nu.iloc[0]) if isinstance(nu, (pd.Series, np.ndarray)) else int(nu)

            if n_unique == 2:
                new_df, _ = apply_transformation(new_df, col, "binary_encode")
                actions_count += 1
                explanations.append({
                    "column": col,
                    "new_feature": f"{col}_encoded",
                    "badge": "Binary Encoding",
                    "badge_color": "#EC4899",
                    "transformation": "Binary 0 / 1 Flag",
                    "why": f"Converts 2-category text feature '{col}' into a numeric integer (0 and 1) so mathematical models can compute probabilities."
                })
            elif n_unique <= 10:
                new_df, _ = apply_transformation(new_df, col, "one_hot")
                actions_count += 1
                explanations.append({
                    "column": col,
                    "new_feature": f"{col}_*",
                    "badge": "One-Hot Encoding",
                    "badge_color": "#10B981",
                    "transformation": f"One-Hot Dummy Indicators ({n_unique} columns)",
                    "why": f"Creates {n_unique} separate 0/1 binary columns for '{col}' categories, preventing artificial order assumptions (e.g. Red > Blue)."
                })

        elif ctype == "datetime":
            new_df, _ = apply_transformation(new_df, col, "datetime_decompose")
            actions_count += 1
            explanations.append({
                "column": col,
                "new_feature": f"{col}_year / month / day / dayofweek",
                "badge": "Time Decomposition",
                "badge_color": "#F59E0B",
                "transformation": "Calendar Signal Extraction",
                "why": f"Extracts seasonal and calendar features (Year, Month, Day of Week, IsWeekend) from raw timestamp '{col}'."
            })

    summary_msg = f"Auto-Engineered {actions_count} new high-performance features across your dataset!"
    return new_df, explanations, summary_msg


# ── ADVANCED FE AUTOMATION HELPERS ───────────────────────────────────────────
def generate_fe_plan(df: pd.DataFrame, col_types: Dict[str, str]) -> Dict[str, Any]:
    """Generates a summary strategy card of planned transformations before execution."""
    plan_items = []
    estimated_new_cols = 0

    for col, ctype in col_types.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue

        if ctype == "numeric":
            skew_val = s.skew()
            if abs(skew_val) > 1.0:
                plan_items.append({"col": col, "type": "Log Transform (Skew Fix)", "est_cols": 1})
                estimated_new_cols += 1
            plan_items.append({"col": col, "type": "Z-Score Scaling", "est_cols": 1})
            estimated_new_cols += 1
        elif ctype in ("categorical", "boolean"):
            nu = s.nunique()
            if nu == 2:
                plan_items.append({"col": col, "type": "Binary 0/1 Encoding", "est_cols": 1})
                estimated_new_cols += 1
            elif nu <= 10:
                plan_items.append({"col": col, "type": f"One-Hot Encoding ({nu} dummy cols)", "est_cols": nu})
                estimated_new_cols += nu
        elif ctype == "datetime":
            plan_items.append({"col": col, "type": "Date Decomposition (Yr, Mo, Day, DoW)", "est_cols": 4})
            estimated_new_cols += 4

    return {
        "total_planned_actions": len(plan_items),
        "estimated_new_columns": estimated_new_cols,
        "items": plan_items
    }


def create_interaction_features(df: pd.DataFrame, col1: str, col2: str, operation: str = "product") -> Tuple[pd.DataFrame, str]:
    """Creates numerical interaction feature between two columns (product or ratio)."""
    new_df = df.copy()
    if col1 not in new_df.columns or col2 not in new_df.columns:
        return new_df, "Columns not found."
        
    s1 = pd.to_numeric(new_df[col1], errors="coerce")
    s2 = pd.to_numeric(new_df[col2], errors="coerce")
    
    if operation == "product":
        new_col = f"{col1}_x_{col2}"
        new_df[new_col] = s1 * s2
        return new_df, f"Created interaction feature '{new_col}' (Product)."
    elif operation == "ratio":
        new_col = f"{col1}_div_{col2}"
        new_df[new_col] = s1 / (s2.replace(0, np.nan))
        return new_df, f"Created interaction feature '{new_col}' (Ratio)."
        
    return new_df, "No interaction feature created."


def quick_feature_importance(df: pd.DataFrame, target_col: str) -> Dict[str, float]:
    """Computes lightweight Random Forest feature importance preview against a target column."""
    if df is None or target_col not in df.columns or len(df) < 10:
        return {}

    num_df = df.select_dtypes(include=[np.number]).dropna()
    if target_col not in num_df.columns or num_df.shape[1] < 2:
        return {}

    X = num_df.drop(columns=[target_col])
    y = num_df[target_col]

    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        is_class = y.nunique() <= 10
        model = RandomForestClassifier(n_estimators=15, random_state=42) if is_class else RandomForestRegressor(n_estimators=15, random_state=42)
        model.fit(X, y)
        importances = dict(zip(X.columns, model.feature_importances_))
        sorted_imp = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
        return {k: round(float(v), 4) for k, v in list(sorted_imp.items())[:10]}
    except Exception:
        return {}

