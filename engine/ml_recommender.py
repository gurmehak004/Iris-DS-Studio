"""
engine/ml_recommender.py
Intelligent ML Model Recommendation Engine:
Analyzes dataset characteristics (target column type, skewness, collinearity,
dataset scale, class balance) and recommends a diverse portfolio of Machine Learning
models with plain-English rationales, evaluation metrics, and data considerations.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


def recommend_ml_models(
    df: pd.DataFrame,
    col_types: Dict[str, str],
    col_stats: Dict[str, Any],
    quality: Dict[str, Any],
    target_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyzes dataset properties and recommends a diverse portfolio of ML models
    (Supervised Classification, Supervised Regression, or Unsupervised Learning)
    without executing actual training.
    """
    num_cols = [c for c, t in col_types.items() if t == "numeric" and c in df.columns]
    cat_cols = [c for c, t in col_types.items() if t in ("categorical", "boolean") and c in df.columns]
    total_rows = len(df)
    total_cols = len(df.columns)

    # Detect Task Type
    task_type = "Unsupervised Learning (Clustering & Anomaly Detection)"
    is_classification = False
    is_regression = False
    target_info = {}

    if target_col and target_col in df.columns:
        s_target = df[target_col].dropna()
        n_unique = s_target.nunique()
        t_type = col_types.get(target_col, "numeric")

        # Check if it's effectively numeric (includes 'id' typed columns that are numeric)
        is_numeric_target = t_type in ("numeric", "id") or pd.api.types.is_numeric_dtype(df[target_col])

        if t_type in ("categorical", "boolean") or (is_numeric_target and n_unique <= 15):
            is_classification = True
            is_binary = (n_unique == 2)
            task_type = f"Supervised Classification ({'Binary' if is_binary else 'Multiclass'})"
            target_info = {
                "name": target_col,
                "type": "Classification",
                "unique_classes": n_unique,
                "class_counts": s_target.value_counts().head(10).to_dict()
            }
        elif is_numeric_target:
            is_regression = True
            task_type = "Supervised Regression (Continuous Output)"
            target_info = {
                "name": target_col,
                "type": "Regression",
                "unique_values": n_unique,
                "mean": float(s_target.mean()) if len(s_target) > 0 else 0.0,
                "std": float(s_target.std()) if len(s_target) > 0 else 0.0
            }

    models = []
    eval_metrics = []
    data_notices = []

    # ── 1. SUPERVISED CLASSIFICATION ──
    if is_classification:
        eval_metrics = ["Accuracy", "F1-Score (Macro/Weighted)", "ROC-AUC Score", "Precision & Recall", "Confusion Matrix"]

        # Check Class Imbalance
        val_counts = df[target_col].value_counts(normalize=True)
        min_class_ratio = val_counts.min() if len(val_counts) > 0 else 1.0
        if min_class_ratio < 0.20:
            data_notices.append(f"⚠️ Class Imbalance Warning: Minority class represents only {min_class_ratio*100:.1f}% of data. Recommend SMOTE resampling, class-weight balancing, or PR-AUC metric.")

        # Recommendations
        models.append({
            "name": "XGBoost / LightGBM Classifier",
            "category": "Gradient Boosted Trees (SOTA)",
            "suitability": "⭐ Primary Recommended (High Accuracy)",
            "interpretability": "Medium (Feature Importance & SHAP)",
            "complexity": "Medium-High",
            "why": "Handles tabular data natively, captures non-linear feature interactions, handles missing values, and excels on skewed numerical distributions.",
            "best_for": "Production tabular prediction requiring top accuracy."
        })

        models.append({
            "name": "Random Forest Classifier",
            "category": "Bagged Ensemble Trees",
            "suitability": "⭐ Highly Recommended (Robust Baseline)",
            "interpretability": "Medium (Feature Importance)",
            "complexity": "Medium",
            "why": "Non-parametric ensemble immune to feature scaling or monotonic transformations. Extremely resistant to overfitting.",
            "best_for": "Fast benchmarking with reliable performance across mixed feature types."
        })

        models.append({
            "name": "Logistic Regression (L1/L2 Regularized)",
            "category": "Linear Baseline",
            "suitability": "✅ Benchmark Baseline",
            "interpretability": "High (Direct Feature Odds-Ratios)",
            "complexity": "Low",
            "why": "Fast, highly transparent, and provides probabilistic confidence scores. Ideal for executive explanations and regulatory compliance.",
            "best_for": "Interpretable baseline and verifying linear feature signals."
        })

        if total_rows < 15000:
            models.append({
                "name": "Support Vector Machine (RBF / Linear SVM)",
                "category": "Maximum-Margin Classifier",
                "suitability": "✅ Specialized Choice",
                "interpretability": "Low",
                "complexity": "Medium-High",
                "why": "Effective in high-dimensional spaces. Maximize boundary separation using Z-score scaled continuous features.",
                "best_for": "Clean datasets with scaled numeric attributes."
            })

    # ── 2. SUPERVISED REGRESSION ──
    elif is_regression:
        eval_metrics = ["RMSE (Root Mean Squared Error)", "MAE (Mean Absolute Error)", "R² Score (Coefficient of Determination)", "MAPE"]

        models.append({
            "name": "XGBoost / CatBoost Regressor",
            "category": "Gradient Boosted Trees",
            "suitability": "⭐ Primary Recommended (High Precision)",
            "interpretability": "Medium (Feature Importance)",
            "complexity": "Medium-High",
            "why": "Captures complex non-linear relationships between continuous predictors and target without requiring strict normal distribution assumptions.",
            "best_for": "Predicting numerical targets with high feature interaction."
        })

        models.append({
            "name": "Random Forest Regressor",
            "category": "Bagged Ensemble Regressor",
            "suitability": "⭐ Highly Recommended",
            "interpretability": "Medium",
            "complexity": "Medium",
            "why": "Averages multi-tree predictions to reduce variance. Handles extreme outliers and unscaled features gracefully.",
            "best_for": "Non-linear numeric regression across tabular datasets."
        })

        models.append({
            "name": "Ridge & Lasso Linear Regression (ElasticNet)",
            "category": "Regularized Linear Regression",
            "suitability": "✅ Interpretable Baseline",
            "interpretability": "High (Direct Coefficients)",
            "complexity": "Low",
            "why": "L1 (Lasso) performs automatic feature selection, while L2 (Ridge) prevents overfitting from multicollinear predictors.",
            "best_for": "Understanding feature impact coefficients and linear relationships."
        })

    # ── 3. UNSUPERVISED LEARNING (NO TARGET / EXPLORATORY) ──
    else:
        eval_metrics = ["Silhouette Coefficient", "Davies-Bouldin Index", "Calinski-Harabasz Index", "Explained Variance Ratio"]
        data_notices.append("ℹ️ Exploratory Unsupervised Mode: No target column designated. Models recommended focus on cohort discovery, pattern extraction, and anomaly detection.")

        models.append({
            "name": "K-Means / K-Medoids Clustering",
            "category": "Centroid-Based Partitioning",
            "suitability": "⭐ Primary Recommended (Customer Segmentation)",
            "interpretability": "High (Cluster Centroids & Profiles)",
            "complexity": "Low-Medium",
            "why": "Groups records into K distinct behavioral clusters based on numerical proximity. Excellent for persona creation and customer tiering.",
            "best_for": "Segmenting observations into actionable user personas."
        })

        models.append({
            "name": "DBSCAN (Density-Based Clustering)",
            "category": "Density-Based Spatial Clustering",
            "suitability": "⭐ Density Discovery & Outliers",
            "interpretability": "Medium",
            "complexity": "Medium",
            "why": "Identifies arbitrary-shaped clusters based on spatial density and automatically flags isolated noise records as anomalies.",
            "best_for": "Non-spherical clusters and automatic outlier isolation."
        })

        models.append({
            "name": "Principal Component Analysis (PCA)",
            "category": "Dimensionality Reduction",
            "suitability": "✅ Feature Space Reduction",
            "interpretability": "Medium (Eigenvector Loadings)",
            "complexity": "Low",
            "why": "Transforms high-dimensional correlated numerical columns into a smaller set of uncorrelated principal components while preserving 90%+ variance.",
            "best_for": "Visualizing high-dimensional datasets in 2D/3D."
        })

        models.append({
            "name": "Isolation Forest",
            "category": "Unsupervised Anomaly Detection",
            "suitability": "✅ Outlier & Fraud Detection",
            "interpretability": "Medium (Anomaly Scores)",
            "complexity": "Medium",
            "why": "Isolates anomalous data points by randomly partitioning features. Outliers require fewer splits to isolate.",
            "best_for": "Identifying suspicious, rare, or abnormal records."
        })

    # General Data Notices
    if len(num_cols) >= 2:
        data_notices.append("✅ Scaling Notice: Distance-based algorithms (KNN, K-Means, SVM, Ridge) should use Standardized (Z-score) features created in Stage 4.")
    
    return {
        "task_type": task_type,
        "target_info": target_info,
        "recommendations": models,
        "eval_metrics": eval_metrics,
        "data_notices": data_notices
    }
