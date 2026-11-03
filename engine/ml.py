"""
engine/ml.py
ML task recommender, automated preprocessing, model training & evaluation.
Supports binary classification, multiclass classification, and regression.
"""
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, List

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
import xgboost as xgb


# ── Utility: Numpy Compatibility ──────────────────────────────────────────────

def to_numpy_compatible_df(df_in: pd.DataFrame, feature_cols: List[str], col_types: Dict[str, str]) -> pd.DataFrame:
    """
    Converts pandas nullable arrays (e.g. StringArray, Int64, Float64) to
    standard NumPy arrays where missing values are represented as float np.nan.
    Prevents Scikit-Learn 'TypeError: boolean value of NA is ambiguous'.
    """
    clean_data = {}
    for col in feature_cols:
        s = df_in[col]
        ctype = col_types.get(col, "unknown")
        is_num = ctype == "numeric" or pd.api.types.is_numeric_dtype(s)
        
        if is_num:
            clean_data[col] = s.to_numpy(dtype=float, na_value=np.nan)
        else:
            clean_data[col] = s.to_numpy(dtype=object, na_value=np.nan)
            
    return pd.DataFrame(clean_data, index=df_in.index)


# ── Target Detection & Task Recommendation ────────────────────────────────────

def auto_detect_target(df: pd.DataFrame, col_types: Dict[str, str]) -> str:
    """Auto-detect the best primary target column for the whole dataset."""
    cols = list(df.columns)
    if not cols:
        return ""

    # 1. Common target column names (case-insensitive)
    common_targets = [
        "survived", "target", "label", "class", "price", "outcome",
        "churn", "is_cancelled", "salary", "bought", "y", "status", "species"
    ]
    col_lower_map = {c.lower(): c for c in cols}
    for ct in common_targets:
        if ct in col_lower_map:
            col_name = col_lower_map[ct]
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
    """Detects whether target implies binary classification, multiclass, or regression."""
    if not target_col or target_col not in df.columns:
        return {"task": "clustering", "reason": "No target column specified. Unsupervised analysis recommended.", "n_unique_target": 0}

    s = df[target_col].dropna()
    n_unique = s.nunique()
    col_type = col_types.get(target_col, "unknown")

    if n_unique <= 1:
        return {
            "task": "invalid",
            "reason": f"'{target_col}' has only {n_unique} unique value(s). A target variable must have at least 2 distinct values to train a model.",
            "n_unique_target": n_unique
        }

    if col_type == "id":
        return {
            "task": "invalid",
            "reason": f"'{target_col}' is an Identifier column (unique IDs). Identifier columns cannot be used as target variables.",
            "n_unique_target": n_unique
        }

    is_num = col_type == "numeric" or pd.api.types.is_numeric_dtype(df[target_col])

    if col_type in ("categorical", "boolean") or (is_num and n_unique <= 10 and pd.api.types.is_integer_dtype(s)):
        if n_unique == 2:
            task = "binary_classification"
            reason = f"'{target_col}' has exactly 2 unique values, making this a Binary Classification task."
        else:
            task = "multiclass_classification"
            reason = f"'{target_col}' has {n_unique} discrete categories, making this a Multi-Class Classification task."
    elif is_num:
        task = "regression"
        reason = f"'{target_col}' is a continuous numeric column with {n_unique} distinct values (Regression task)."
    else:
        if n_unique <= 50:
            task = "multiclass_classification"
            reason = f"'{target_col}' has {n_unique} categories, suggesting Multi-Class Classification."
        else:
            task = "clustering"
            reason = f"'{target_col}' has too many unique categories ({n_unique}). Unsupervised Clustering recommended."

    return {"task": task, "reason": reason, "n_unique_target": n_unique}


# ── Preprocessing Pipeline ────────────────────────────────────────────────────

def build_preprocessor(df: pd.DataFrame, feature_cols: list, col_types: Dict[str, str]):
    numeric_cols = [
        c for c in feature_cols
        if col_types.get(c) == "numeric" or pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [
        c for c in feature_cols
        if c not in numeric_cols and (
            col_types.get(c) in ("categorical", "boolean", "text")
            or pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_categorical_dtype(df[c])
            or pd.api.types.is_bool_dtype(df[c])
        )
    ]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        return None, []

    return ColumnTransformer(transformers, remainder="drop"), numeric_cols + categorical_cols


# ── Training & Leaderboard Engine ─────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    col_types: Dict[str, str],
    custom_splits: Optional[Tuple[pd.DataFrame, pd.DataFrame, Any, Any]] = None,
) -> Dict[str, Any]:
    """
    Trains a portfolio of baseline models (Linear/Logistic, Random Forest, XGBoost).
    Returns an evaluation leaderboard, best model pipeline, feature importances,
    and diagnostic arrays (confusion matrix for classification, actuals/predictions for regression).
    """
    if df is None or len(df) == 0:
        return {"error": "Dataset is empty."}

    if not target_col or target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in dataset."}

    # Identify candidate features (exclude target, ID columns, and empty columns)
    feature_cols = [
        c for c in df.columns
        if c != target_col
        and col_types.get(c) not in ("id", "empty")
        and not c.endswith(("_id", "Id", "ID"))
    ]

    if not feature_cols:
        return {"error": "No usable feature columns found for training."}

    # Prepare numpy-compatible features
    X_clean = to_numpy_compatible_df(df, feature_cols, col_types)
    preprocessor, used_cols = build_preprocessor(X_clean, feature_cols, col_types)

    if preprocessor is None or len(used_cols) == 0:
        return {"error": "Could not construct preprocessing pipeline for features."}

    # Prepare target variable
    s_target = df[target_col]
    le: Optional[LabelEncoder] = None
    y_clean: np.ndarray

    if task in ("binary_classification", "multiclass_classification"):
        # Drop rows where target is NaN
        mask = s_target.notna().to_numpy(dtype=bool)
        if mask.sum() < 10:
            return {"error": "Too few non-null target observations to train (minimum 10 rows required)."}
        
        X_clean = X_clean.iloc[mask].reset_index(drop=True)
        le = LabelEncoder()
        y_clean = le.fit_transform(s_target[mask].astype(str))
        n_classes = len(le.classes_)
        if n_classes < 2:
            return {"error": f"Target column '{target_col}' contains only 1 class. Classification requires at least 2 distinct classes."}
    else:
        # Regression
        y_num = pd.to_numeric(s_target, errors="coerce")
        mask = y_num.notna().to_numpy(dtype=bool)
        if mask.sum() < 10:
            return {"error": "Too few valid numeric target observations (minimum 10 rows required)."}
        X_clean = X_clean.iloc[mask].reset_index(drop=True)
        y_clean = y_num[mask].to_numpy(dtype=float)

    # Train / Test Splitting
    if custom_splits is not None:
        X_train, X_test, y_train, y_test = custom_splits
    else:
        # Stratify classification if every class has at least 2 instances
        stratify_arr = None
        if task in ("binary_classification", "multiclass_classification"):
            _, counts = np.unique(y_clean, return_counts=True)
            if (counts >= 2).all():
                stratify_arr = y_clean

        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.20, random_state=42, stratify=stratify_arr
        )

    # Define model portfolios
    models: Dict[str, Pipeline] = {}
    if task == "regression":
        models = {
            "Ridge Regression": Pipeline([
                ("pre", preprocessor),
                ("model", Ridge(alpha=1.0, random_state=42))
            ]),
            "Random Forest Regressor": Pipeline([
                ("pre", preprocessor),
                ("model", RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1))
            ]),
            "XGBoost Regressor": Pipeline([
                ("pre", preprocessor),
                ("model", xgb.XGBRegressor(
                    n_estimators=80, max_depth=5, learning_rate=0.08,
                    random_state=42, verbosity=0, n_jobs=-1
                ))
            ]),
        }
    else:
        # Classification
        xgb_eval_metric = "logloss" if (le is not None and len(le.classes_) == 2) else "mlogloss"
        models = {
            "Logistic Regression": Pipeline([
                ("pre", preprocessor),
                ("model", LogisticRegression(max_iter=500, random_state=42))
            ]),
            "Random Forest Classifier": Pipeline([
                ("pre", preprocessor),
                ("model", RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1))
            ]),
            "XGBoost Classifier": Pipeline([
                ("pre", preprocessor),
                ("model", xgb.XGBClassifier(
                    n_estimators=80, max_depth=5, learning_rate=0.08,
                    random_state=42, verbosity=0, eval_metric=xgb_eval_metric, n_jobs=-1
                ))
            ]),
        }

    leaderboard = []
    results = {}
    best_score = -np.inf
    best_name = None
    best_pipeline = None
    best_y_pred = None

    for name, pipe in models.items():
        t0 = time.time()
        try:
            pipe.fit(X_train, y_train)
            train_time = round(time.time() - t0, 3)
            y_pred = pipe.predict(X_test)

            if task == "regression":
                r2 = float(r2_score(y_test, y_pred))
                mse = float(mean_squared_error(y_test, y_pred))
                rmse = float(np.sqrt(mse))
                mae = float(mean_absolute_error(y_test, y_pred))

                metrics = {
                    "r2": round(r2, 4),
                    "rmse": round(rmse, 4),
                    "mae": round(mae, 4),
                    "train_time_sec": train_time
                }
                results[name] = metrics
                leaderboard.append({
                    "model": name,
                    "primary_metric": round(r2, 4),
                    "primary_label": "R² Score",
                    "secondary_metric": round(rmse, 4),
                    "secondary_label": "RMSE",
                    "mae": round(mae, 4),
                    "train_time": f"{train_time}s",
                    "status": "Success"
                })
                current_score = r2
            else:
                acc = float(accuracy_score(y_test, y_pred))
                f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

                metrics = {
                    "accuracy": round(acc, 4),
                    "f1_weighted": round(f1_w, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "train_time_sec": train_time
                }
                results[name] = metrics
                leaderboard.append({
                    "model": name,
                    "primary_metric": round(acc, 4),
                    "primary_label": "Accuracy",
                    "secondary_metric": round(f1_w, 4),
                    "secondary_label": "F1 (Weighted)",
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "train_time": f"{train_time}s",
                    "status": "Success"
                })
                current_score = acc

            if current_score > best_score:
                best_score = current_score
                best_name = name
                best_pipeline = pipe
                best_y_pred = y_pred

        except Exception as e:
            results[name] = {"error": str(e)}
            leaderboard.append({
                "model": name,
                "primary_metric": -999.0,
                "primary_label": "Score",
                "secondary_metric": -999.0,
                "secondary_label": "Error",
                "train_time": "0s",
                "status": f"Failed: {str(e)[:45]}"
            })

    # Sort leaderboard in descending order of primary metric
    leaderboard.sort(key=lambda x: x["primary_metric"], reverse=True)

    output: Dict[str, Any] = {
        "task": task,
        "target_col": target_col,
        "best_model": best_name,
        "best_score": best_score,
        "leaderboard": leaderboard,
        "results": results,
        "feature_cols": used_cols,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "label_encoder": le,
        "best_pipeline": best_pipeline,
    }

    # Extract Feature Importances from the best model
    if best_pipeline is not None:
        try:
            model_step = best_pipeline.named_steps["model"]
            pre_step = best_pipeline.named_steps["pre"]
            feat_names = pre_step.get_feature_names_out()
            clean_feat_names = [f.split("__")[-1] for f in feat_names]

            if hasattr(model_step, "feature_importances_"):
                fi = model_step.feature_importances_
                if len(fi) == len(clean_feat_names):
                    sorted_fi = sorted(zip(clean_feat_names, [round(float(x), 4) for x in fi]), key=lambda x: x[1], reverse=True)
                    output["feature_importances"] = dict(sorted_fi[:15])
            elif hasattr(model_step, "coef_"):
                if model_step.coef_.ndim > 1:
                    coef = np.abs(model_step.coef_).mean(axis=0)
                else:
                    coef = np.abs(model_step.coef_).flatten()
                if len(coef) == len(clean_feat_names):
                    sorted_fi = sorted(zip(clean_feat_names, [round(float(x), 4) for x in coef]), key=lambda x: x[1], reverse=True)
                    output["feature_importances"] = dict(sorted_fi[:15])
        except Exception:
            output["feature_importances"] = {}

        # Diagnostic plots & reports
        if task in ("binary_classification", "multiclass_classification") and best_y_pred is not None:
            try:
                cm = confusion_matrix(y_test, best_y_pred)
                output["confusion_matrix"] = cm.tolist()
                output["class_labels"] = [str(lbl) for lbl in le.classes_] if le else None
                output["classification_report"] = classification_report(
                    y_test, best_y_pred, target_names=[str(c) for c in le.classes_] if le else None,
                    output_dict=True, zero_division=0
                )
            except Exception:
                pass
        elif task == "regression" and best_y_pred is not None:
            output["regression_diagnostics"] = {
                "y_true": [round(float(v), 3) for v in y_test[:200]],
                "y_pred": [round(float(v), 3) for v in best_y_pred[:200]],
                "residuals": [round(float(yt - yp), 3) for yt, yp in zip(y_test[:200], best_y_pred[:200])]
            }

    return output


# ── Unsupervised Clustering ───────────────────────────────────────────────────

def run_clustering(df: pd.DataFrame, col_types: Dict[str, str], k_range=(2, 8)) -> Dict[str, Any]:
    numeric_cols = [c for c, t in col_types.items() if t == "numeric" and c in df.columns]
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
        "labels": labels.tolist(),
        "numeric_cols_used": numeric_cols,
    }
