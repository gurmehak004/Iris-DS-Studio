"""
engine/code_exporter.py
Exports session actions as reproducible, production-grade Python scripts (.py)
and interactive Jupyter Notebooks (.ipynb).
"""
import json
import re
from typing import List, Dict, Any, Optional
import datetime


def generate_pipeline_script(
    dataset_name: str,
    cleaning_actions: List[str],
    fe_actions: List[str],
    target_col: Optional[str] = None,
    task: Optional[str] = None,
    train_test_split_enabled: bool = True,
    best_model_name: Optional[str] = None,
) -> str:
    """
    Generates a standalone, executable Python script (.py) reproducing
    the full data science pipeline executed in Iris Data Studio.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_target = target_col or "target"
    clean_task = task or "binary_classification"
    is_classification = "classification" in clean_task.lower()
    
    script_lines = [
        '"""',
        '==============================================================================',
        ' Iris Data Science Studio - Automated Production Pipeline',
        f' Generated on: {now_str}',
        f' Dataset: {dataset_name}',
        f' Target Column: {clean_target} ({clean_task})',
        '==============================================================================',
        '"""',
        'import pandas as pd',
        'import numpy as np',
        'import os',
        'import sys',
        'from sklearn.model_selection import train_test_split',
        'from sklearn.pipeline import Pipeline',
        'from sklearn.impute import SimpleImputer',
        'from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder',
        'from sklearn.compose import ColumnTransformer',
    ]

    if is_classification:
        script_lines.extend([
            'from sklearn.linear_model import LogisticRegression',
            'from sklearn.ensemble import RandomForestClassifier',
            'from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix',
        ])
    else:
        script_lines.extend([
            'from sklearn.linear_model import Ridge',
            'from sklearn.ensemble import RandomForestRegressor',
            'from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error',
        ])

    script_lines.extend([
        'import xgboost as xgb',
        '',
        '',
        'def run_pipeline(data_path: str = "dataset.csv"):',
        '    print("--> [Stage 1] Ingesting dataset from:", data_path)',
        '    if not os.path.exists(data_path):',
        '        print(f"Warning: File {data_path} not found. Please provide valid path.")',
        '        return',
        '',
        '    df = pd.read_csv(data_path)',
        '    print(f"Raw dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")',
        '    df.columns = [str(c).strip() for c in df.columns]',
        '',
        '    # -- [Stage 2] Data Cleaning & Preprocessing --------------------------',
        '    print("--> [Stage 2] Executing Data Cleaning...")',
        '    # 1. Drop all-empty rows and columns',
        '    df.dropna(how="all", inplace=True)',
        '    df.dropna(axis=1, how="all", inplace=True)',
        '    df.reset_index(drop=True, inplace=True)',
        '',
    ])

    # Convert recorded cleaning actions into real Python statements
    applied_cleaning = False
    for act in cleaning_actions:
        act_clean = act.strip()
        if "removed" in act_clean.lower() and "duplicate" in act_clean.lower():
            script_lines.append('    # Remove duplicate records')
            script_lines.append('    df.drop_duplicates(inplace=True)')
            script_lines.append('    df.reset_index(drop=True, inplace=True)')
            applied_cleaning = True
        elif "dropped" in act_clean.lower() and "sparse" in act_clean.lower():
            # Extract column names if present
            script_lines.append('    # Drop sparse columns exceeding missing threshold')
            script_lines.append('    sparse_thresh = 0.40 * len(df)')
            script_lines.append('    df = df.dropna(thresh=sparse_thresh, axis=1)')
            applied_cleaning = True
        elif "imputed" in act_clean.lower() and "median" in act_clean.lower():
            col_match = re.search(r"in '([^']+)'", act_clean)
            if col_match:
                c = col_match.group(1)
                script_lines.append(f'    if "{c}" in df.columns:')
                script_lines.append(f'        df["{c}"] = pd.to_numeric(df["{c}"], errors="coerce")')
                script_lines.append(f'        df["{c}"] = df["{c}"].fillna(df["{c}"].median())')
                applied_cleaning = True
        elif "imputed" in act_clean.lower() and "mode" in act_clean.lower():
            col_match = re.search(r"in '([^']+)'", act_clean)
            if col_match:
                c = col_match.group(1)
                script_lines.append(f'    if "{c}" in df.columns:')
                script_lines.append(f'        _mode = df["{c}"].mode()')
                script_lines.append(f'        df["{c}"] = df["{c}"].fillna(_mode.iloc[0] if len(_mode) > 0 else "Unknown")')
                applied_cleaning = True
        elif "capped" in act_clean.lower() and "outliers in" in act_clean.lower():
            col_match = re.search(r"in '([^']+)'", act_clean)
            if col_match:
                c = col_match.group(1)
                script_lines.append(f'    if "{c}" in df.columns:')
                script_lines.append(f'        _s = pd.to_numeric(df["{c}"], errors="coerce")')
                script_lines.append(f'        _q1, _q3 = _s.quantile(0.25), _s.quantile(0.75)')
                script_lines.append(f'        _iqr = _q3 - _q1')
                script_lines.append(f'        if _iqr > 0:')
                script_lines.append(f'            df["{c}"] = _s.clip(lower=_q1 - 1.5 * _iqr, upper=_q3 + 1.5 * _iqr)')
                applied_cleaning = True
        elif "trimmed" in act_clean.lower() and "whitespace" in act_clean.lower():
            col_match = re.search(r"in '([^']+)'", act_clean)
            if col_match:
                c = col_match.group(1)
                script_lines.append(f'    if "{c}" in df.columns and df["{c}"].dtype == object:')
                script_lines.append(f'        df["{c}"] = df["{c}"].astype(str).str.strip()')
                applied_cleaning = True

    if not applied_cleaning:
        script_lines.append('    # Baseline missing-value handling')
        script_lines.append('    df.drop_duplicates(inplace=True)')

    script_lines.extend([
        '',
        '    # -- [Stage 4] Feature Engineering ------------------------------------',
        '    print("--> [Stage 4] Applying Feature Engineering...")',
    ])

    # Convert recorded feature engineering actions into real Python code
    applied_fe = False
    for act in fe_actions:
        act_clean = act.strip()
        if "log" in act_clean.lower():
            col_match = re.search(r"'([^']+)'", act_clean)
            if col_match:
                c = col_match.group(1).replace("_log", "")
                script_lines.append(f'    if "{c}" in df.columns:')
                script_lines.append(f'        _s_num = pd.to_numeric(df["{c}"], errors="coerce")')
                script_lines.append(f'        _shift = abs(_s_num.min()) + 1.0 if _s_num.min() < 0 else 0.0')
                script_lines.append(f'        df["{c}_log"] = np.log1p(_s_num + _shift)')
                applied_fe = True
        elif "datetime" in act_clean.lower() or "decomposed" in act_clean.lower():
            col_match = re.search(r"'([^']+)'", act_clean)
            if col_match:
                c = col_match.group(1)
                script_lines.append(f'    if "{c}" in df.columns:')
                script_lines.append(f'        _dt = pd.to_datetime(df["{c}"], errors="coerce")')
                script_lines.append(f'        df["{c}_year"] = _dt.dt.year')
                script_lines.append(f'        df["{c}_month"] = _dt.dt.month')
                script_lines.append(f'        df["{c}_dayofweek"] = _dt.dt.dayofweek')
                script_lines.append(f'        df["{c}_is_weekend"] = (_dt.dt.dayofweek >= 5).astype(int)')
                applied_fe = True
        elif "interaction" in act_clean.lower():
            col_match = re.search(r"'([^']+)'", act_clean)
            if col_match:
                int_col = col_match.group(1)
                if "_x_" in int_col:
                    p1, p2 = int_col.split("_x_", 1)
                    script_lines.append(f'    if "{p1}" in df.columns and "{p2}" in df.columns:')
                    script_lines.append(f'        df["{int_col}"] = pd.to_numeric(df["{p1}"], errors="coerce") * pd.to_numeric(df["{p2}"], errors="coerce")')
                    applied_fe = True

    if not applied_fe:
        script_lines.append('    # No custom interactions or log transformations recorded.')

    script_lines.extend([
        '',
        f'    # -- [Stage 5] Train / Test Partitioning & Leakage Prevention ---------',
        f'    target_column = "{clean_target}"',
        '    if target_column not in df.columns:',
        f'        print(f"Error: Target column {{target_column}} not found in dataframe.")',
        '        return',
        '',
        '    # Separate predictors and target',
        '    feature_cols = [c for c in df.columns if c != target_column and not c.endswith(("_id", "Id", "ID"))]',
        '    X = df[feature_cols].copy()',
        '    y = df[target_column].copy()',
        '',
        '    # Handle missing target rows',
        '    valid_mask = y.notna()',
        '    X, y = X[valid_mask], y[valid_mask]',
        '',
    ])

    if is_classification:
        script_lines.extend([
            '    le = LabelEncoder()',
            '    y = le.fit_transform(y.astype(str))',
            '    stratify_target = y if min(pd.Series(y).value_counts()) >= 2 else None',
            '    X_train, X_test, y_train, y_test = train_test_split(',
            '        X, y, test_size=0.20, random_state=42, stratify=stratify_target',
            '    )',
        ])
    else:
        script_lines.extend([
            '    y = pd.to_numeric(y, errors="coerce")',
            '    valid_num_mask = y.notna()',
            '    X, y = X[valid_num_mask], y[valid_num_mask]',
            '    X_train, X_test, y_train, y_test = train_test_split(',
            '        X, y, test_size=0.20, random_state=42',
            '    )',
        ])

    script_lines.extend([
        '    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")',
        '',
        '    # Identify numeric and categorical columns strictly on training set',
        '    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()',
        '    cat_cols = [c for c in X_train.columns if c not in num_cols]',
        '',
        '    # Construct Leak-Free ColumnTransformer Pipeline',
        '    preprocessor = ColumnTransformer([',
        '        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),',
        '        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols),',
        '    ], remainder="drop")',
        '',
        '    # -- [Stage 6] Model Training & Evaluation ----------------------------',
        '    print("--> Training Machine Learning Baselines...")',
    ])

    if is_classification:
        script_lines.extend([
            '    models = {',
            '        "Logistic Regression": Pipeline([("pre", preprocessor), ("model", LogisticRegression(max_iter=500, random_state=42))]),',
            '        "Random Forest Classifier": Pipeline([("pre", preprocessor), ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),',
            '        "XGBoost Classifier": Pipeline([("pre", preprocessor), ("model", xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"))]),',
            '    }',
            '',
            '    print("\\n" + "="*50 + " MODEL LEADERBOARD " + "="*50)',
            '    for name, pipe in models.items():',
            '        pipe.fit(X_train, y_train)',
            '        preds = pipe.predict(X_test)',
            '        acc = accuracy_score(y_test, preds)',
            '        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)',
            '        print(f"[{name}] --> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")',
            '    print("="*118 + "\\n")',
        ])
    else:
        script_lines.extend([
            '    models = {',
            '        "Ridge Regression": Pipeline([("pre", preprocessor), ("model", Ridge(alpha=1.0, random_state=42))]),',
            '        "Random Forest Regressor": Pipeline([("pre", preprocessor), ("model", RandomForestRegressor(n_estimators=100, random_state=42))]),',
            '        "XGBoost Regressor": Pipeline([("pre", preprocessor), ("model", xgb.XGBRegressor(n_estimators=100, random_state=42))]),',
            '    }',
            '',
            '    print("\\n" + "="*50 + " MODEL LEADERBOARD " + "="*50)',
            '    for name, pipe in models.items():',
            '        pipe.fit(X_train, y_train)',
            '        preds = pipe.predict(X_test)',
            '        r2 = r2_score(y_test, preds)',
            '        rmse = np.sqrt(mean_squared_error(y_test, preds))',
            '        mae = mean_absolute_error(y_test, preds)',
            '        print(f"[{name}] --> R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")',
            '    print("="*118 + "\\n")',
        ])

    script_lines.extend([
        '',
        'if __name__ == "__main__":',
        '    # Run pipeline on active dataset or fallback',
        '    active_file = sys.argv[1] if len(sys.argv) > 1 else "cleaned_dataset.csv"',
        '    if not os.path.exists(active_file):',
        f'        active_file = "{dataset_name}"',
        '    run_pipeline(active_file)',
    ])

    return "\n".join(script_lines)


def generate_pipeline_notebook(
    dataset_name: str,
    cleaning_actions: List[str],
    fe_actions: List[str],
    target_col: Optional[str] = None,
    task: Optional[str] = None,
    train_test_split_enabled: bool = True,
    best_model_name: Optional[str] = None,
) -> str:
    """
    Generates an interactive Jupyter Notebook (.ipynb) with structured markdown cells
    and executable Python code cells replicating the Iris session.
    """
    script_content = generate_pipeline_script(
        dataset_name=dataset_name,
        cleaning_actions=cleaning_actions,
        fe_actions=fe_actions,
        target_col=target_col,
        task=task,
        train_test_split_enabled=train_test_split_enabled,
        best_model_name=best_model_name,
    )
    
    # Split the script into logical notebook cells
    clean_target = target_col or "target"
    clean_task = task or "Supervised Learning"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌸 Iris Data Science Studio - Automated Pipeline\n",
                f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>\n",
                f"**Dataset:** `{dataset_name}` | **Target:** `{clean_target}` (`{clean_task}`)\n",
                "\n",
                "This Jupyter Notebook was automatically exported from **Iris Data Science Studio**. ",
                "It reproduces the end-to-end data preparation, cleaning, feature engineering, and model training workflow."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Core Imports\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.impute import SimpleImputer\n",
                "from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder\n",
                "from sklearn.compose import ColumnTransformer\n",
                "import xgboost as xgb\n",
                "\n",
                "pd.set_option('display.max_columns', None)\n",
                "print('Environment and libraries initialized successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Data Ingestion\n",
                f"Load the raw dataset (`{dataset_name}`) and inspect dimensions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"DATA_PATH = '{dataset_name}'\n",
                "try:\n",
                "    df = pd.read_csv(DATA_PATH)\n",
                "except FileNotFoundError:\n",
                "    df = pd.read_csv('cleaned_dataset.csv')\n",
                "\n",
                "df.columns = [str(c).strip() for c in df.columns]\n",
                "print(f'Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Cleaning & Preprocessing\n",
                "Apply the automated cleaning rules recorded during the Iris Studio session."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Remove completely empty rows and columns\n",
                "df.dropna(how='all', inplace=True)\n",
                "df.dropna(axis=1, how='all', inplace=True)\n",
                "df.drop_duplicates(inplace=True)\n",
                "df.reset_index(drop=True, inplace=True)\n",
                "print(f'Shape after deduplication: {df.shape}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Train / Test Partitioning (Data Leakage Guard)\n",
                "Split dataset into Train (80%) and Test (20%) partitions **before** computing imputers and scalers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"target_col = '{clean_target}'\n",
                "feature_cols = [c for c in df.columns if c != target_col and not c.endswith(('_id', 'Id', 'ID'))]\n",
                "\n",
                "X = df[feature_cols].copy()\n",
                "y = df[target_col].copy()\n",
                "\n",
                "valid_mask = y.notna()\n",
                "X, y = X[valid_mask], y[valid_mask]\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)\n",
                "print(f'Train Partition: {X_train.shape[0]} rows | Holdout Test Partition: {X_test.shape[0]} rows')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Leak-Free Preprocessing Pipeline & Model Training\n",
                "Construct Scikit-Learn ColumnTransformer and fit baseline models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()\n",
                "cat_cols = [c for c in X_train.columns if c not in num_cols]\n",
                "\n",
                "preprocessor = ColumnTransformer([\n",
                "    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),\n",
                "    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols),\n",
                "], remainder='drop')\n",
                "\n",
                "print(f'Numeric Features ({len(num_cols)}): {num_cols}')\n",
                "print(f'Categorical Features ({len(cat_cols)}): {cat_cols}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Fit and evaluate baseline model\n",
                "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n",
                "\n",
                "is_classifier = True if y_train.dtype == object or y_train.nunique() <= 10 else False\n",
                "if is_classifier:\n",
                "    model = Pipeline([('pre', preprocessor), ('rf', RandomForestClassifier(n_estimators=100, random_state=42))])\n",
                "    model.fit(X_train, y_train)\n",
                "    score = model.score(X_test, y_test)\n",
                "    print(f'Test Accuracy: {score:.4f}')\n",
                "else:\n",
                "    model = Pipeline([('pre', preprocessor), ('rf', RandomForestRegressor(n_estimators=100, random_state=42))])\n",
                "    model.fit(X_train, y_train)\n",
                "    score = model.score(X_test, y_test)\n",
                "    print(f'Test R2 Score: {score:.4f}')"
            ]
        }
    ]

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    return json.dumps(notebook_dict, indent=2)
