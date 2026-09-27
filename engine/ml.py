"""
engine/ml.py
ML recommender, auto preprocessing, model training & evaluation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                              r2_score, confusion_matrix, classification_report)
from sklearn.cluster import KMeans
import xgboost as xgb


# ── Recommendation ─────────────────────────────────────────────────────────────

def auto_detect_target(df: pd.DataFrame, col_types: Dict[str, str]) -> str:
    """Auto-detect the best primary target column for the whole dataset."""
    cols = list(df.columns)
    if not cols:
        return ""

    # 1. Common target column names (case-insensitive)
    common_targets = ["survived", "target", "label", "class", "price", "outcome", "churn", "is_cancelled", "salary", "bought", "y"]
    col_lower_map = {c.lower(): c for c in cols}
    for ct in common_targets:
        if ct in col_lower_map:
            col_name = col_lower_map[ct]
            # Check it has at least 2 unique non-null values
            if df[col_name].dropna().nunique() >= 2:
                return col_name

    # 2. Look for binary or low-cardinality categorical/boolean columns (excluding IDs)
    candidate_cols = []
    for c in cols:
        ctype = col_types.get(c, "unknown")
        if ctype in ("boolean", "categorical"):
            n_unq = df[c].dropna().nunique()
            if 2 <= n_unq <= 20:
                candidate_cols.append((n_unq, c))
    
    if candidate_cols:
        # Prefer smallest number of unique values (binary classification)
        candidate_cols.sort(key=lambda x: x[0])
        return candidate_cols[0][1]

    # 3. Look for numeric columns that are not IDs or index-like
    for c in cols:
        ctype = col_types.get(c, "unknown")
        if ctype == "numeric":
            n_unq = df[c].dropna().nunique()
            if 2 <= n_unq < len(df) * 0.95:
                return c

    # 4. Fallback to first non-ID column
    for c in cols:
        if col_types.get(c) != "id":
            return c

    return cols[0]


def recommend_task(df: pd.DataFrame, target_col: str, col_types: Dict[str, str]) -> Dict[str, Any]:
    s = df[target_col].dropna()
    n_unique = s.nunique()
    col_type = col_types.get(target_col, "unknown")

    if n_unique <= 1:
        return {
            "task": "invalid",
            "reason": f"'{target_col}' has only {n_unique} unique value(s). A target variable must have at least 2 distinct values to train a machine learning model.",
            "n_unique_target": n_unique
        }

    if col_type == "id":
        return {
            "task": "invalid",
            "reason": f"'{target_col}' is an Identifier column (unique IDs). Identifier columns cannot be used as target variables.",
            "n_unique_target": n_unique
        }

    if col_type == "numeric":
        if n_unique == 2:
            task = "binary_classification"
            reason = (f"'{target_col}' is a numeric column with exactly 2 unique values ({list(np.sort(s.unique())[:2])}), "
                      "making this a Binary Classification task.")
        elif n_unique <= 10 and pd.api.types.is_integer_dtype(s):
            task = "multiclass_classification"
            reason = (f"'{target_col}' is an integer column with {n_unique} discrete categories, "
                      "suggesting a Multi-class Classification problem.")
        else:
            task = "regression"
            reason = (f"'{target_col}' is a continuous numeric column with {n_unique} unique values, "
                      "suggesting a continuous prediction task (Regression).")
    elif col_type in ("categorical", "boolean"):
        if n_unique == 2:
            task = "binary_classification"
            reason = (f"'{target_col}' has exactly 2 unique values ({list(s.unique()[:2])}), "
                      "which is a classic Binary Classification problem.")
        elif n_unique <= 50:
            task = "multiclass_classification"
            reason = (f"'{target_col}' has {n_unique} categories, "
                      "making this a Multi-class Classification problem.")
        else:
            task = "clustering"
            reason = f"'{target_col}' has too many unique categories ({n_unique}). Unsupervised Clustering is recommended instead."
    else:
        task = "clustering"
        reason = "No valid target column identified. Clustering (unsupervised) is recommended."

    return {"task": task, "reason": reason, "n_unique_target": n_unique}


# ── Preprocessing ──────────────────────────────────────────────────────────────

def build_preprocessor(df: pd.DataFrame, feature_cols: list, col_types: Dict[str, str]):
    numeric_cols = [c for c in feature_cols if col_types.get(c) == "numeric"]
    categorical_cols = [c for c in feature_cols
                        if col_types.get(c) in ("categorical", "boolean") and c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop"), numeric_cols + categorical_cols


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, target_col: str, task: str,
                col_types: Dict[str, str]) -> Dict[str, Any]:

    feature_cols = [c for c in df.columns
                    if c != target_col and col_types.get(c) in ("numeric", "categorical", "boolean")]

    if len(feature_cols) == 0:
        return {"error": "No usable feature columns found (features must be numeric, categorical, or boolean)."}

    preprocessor, used_cols = build_preprocessor(df, feature_cols, col_types)

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    mask = y.notna()
    X, y = X[mask], y[mask]

    if len(y) < 10:
        return {"error": "Too few valid rows available for model training (less than 10 rows)."}

    # Encode target for classification
    le = None
    if task in ("binary_classification", "multiclass_classification"):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    if task == "regression":
        models = {
            "Linear Regression": Pipeline([("pre", preprocessor), ("model", LinearRegression())]),
            "Random Forest": Pipeline([("pre", preprocessor),
                                        ("model", RandomForestRegressor(n_estimators=100, random_state=42))]),
            "XGBoost": Pipeline([("pre", preprocessor),
                                  ("model", xgb.XGBRegressor(n_estimators=100, random_state=42,
                                                               verbosity=0))]),
        }
    else:
        models = {
            "Logistic Regression": Pipeline([("pre", preprocessor),
                                              ("model", LogisticRegression(max_iter=500, random_state=42))]),
            "Random Forest": Pipeline([("pre", preprocessor),
                                        ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
            "XGBoost": Pipeline([("pre", preprocessor),
                                  ("model", xgb.XGBClassifier(n_estimators=100, random_state=42,
                                                                verbosity=0,
                                                                eval_metric="logloss"))]),
        }

    results = {}
    best_score = -np.inf
    best_name = None
    best_pipeline = None

    for name, pipe in models.items():
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            if task == "regression":
                score = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                results[name] = {"r2": round(float(score), 4), "rmse": round(float(np.sqrt(mse)), 4)}
                metric = score
            else:
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                results[name] = {"accuracy": round(float(acc), 4), "f1_weighted": round(float(f1), 4)}
                metric = acc
            if metric > best_score:
                best_score = metric
                best_name = name
                best_pipeline = pipe
        except Exception as e:
            results[name] = {"error": str(e)}

    output = {
        "task": task,
        "best_model": best_name,
        "results": results,
        "feature_cols": used_cols,
        "original_feature_cols": feature_cols,
        "label_encoder": le,
        "best_pipeline": best_pipeline,
    }

    # Feature importances from best model
    if best_pipeline is not None:
        try:
            model_step = best_pipeline.named_steps["model"]
            pre_step = best_pipeline.named_steps["pre"]
            feat_names = pre_step.get_feature_names_out()
            clean_feat_names = [f.split("__")[-1] for f in feat_names]

            if hasattr(model_step, "feature_importances_"):
                fi = model_step.feature_importances_
                if len(fi) == len(clean_feat_names):
                    output["feature_importances"] = dict(zip(clean_feat_names, [round(float(x), 4) for x in fi]))
            elif hasattr(model_step, "coef_"):
                coef = np.abs(model_step.coef_).flatten()
                if len(coef) == len(clean_feat_names):
                    output["feature_importances"] = dict(zip(clean_feat_names, [round(float(x), 4) for x in coef]))
        except Exception:
            pass

        # Confusion matrix for classification
        if task in ("binary_classification", "multiclass_classification"):
            try:
                y_pred_best = best_pipeline.predict(X_test)
                cm = confusion_matrix(y_test, y_pred_best)
                output["confusion_matrix"] = cm
                output["class_labels"] = [str(l) for l in le.classes_] if le else None
                output["classification_report"] = classification_report(
                    y_test, y_pred_best, output_dict=True)
            except Exception:
                pass

    return output


# ── Clustering ─────────────────────────────────────────────────────────────────

def run_clustering(df: pd.DataFrame, col_types: Dict[str, str],
                   k_range=(2, 9)) -> Dict[str, Any]:
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns for clustering."}

    X = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(X) < 10:
        return {"error": "Too few numeric rows available for clustering."}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    inertias = []
    k_vals = list(range(k_range[0], k_range[1] + 1))
    for k in k_vals:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(Xs)
        inertias.append(float(km.inertia_))

    best_k = k_vals[len(k_vals) // 2]
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    labels = km_final.fit_predict(Xs)

    return {
        "k_values": k_vals,
        "inertias": inertias,
        "best_k": best_k,
        "labels": labels,
        "numeric_cols_used": numeric_cols,
    }

