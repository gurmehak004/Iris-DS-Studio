"""
engine/gemini_client.py
Gemini API wrapper with prompt templates for each pipeline step.
"""
import os
import json
import logging
from google import genai
from google.genai import types
from typing import Dict, Any, Optional
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


_GEMINI_MODEL = "gemini-2.0-flash"
_GEMINI_TIMEOUT_MS = 6_000


def _create_client(api_key: str) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=_GEMINI_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key

    try:
        return str(st.secrets.get("GEMINI_API_KEY", "") or "").strip()
    except StreamlitSecretNotFoundError:
        return ""


def is_gemini_api_key_configured() -> bool:
    return bool(_get_api_key())


def review_model_recommendations(
    recommendations: Dict[str, Any],
    dataset_summary: Dict[str, Any],
) -> tuple[bool, str]:
    api_key = _get_api_key()
    if not api_key:
        return False, "Iris AI review was not run: no API key is configured for this app."

    prompt = f"""
You are Iris AI, an expert enterprise data science advisor. Never refer to yourself as Gemini or mention Google.
Review the candidate models for this dataset and give a concise, practical recommendation.

Dataset summary:
{json.dumps(dataset_summary, default=str)}

Task type and candidate models:
{json.dumps(recommendations, default=str)}

Recommend the best 1-2 candidates from the supplied list for this dataset, explain
the trade-offs, and mention any relevant data-quality or evaluation cautions. Do not claim
that any model has been trained, tested, or scored. Do not invent evaluation results.
If the target or dataset information is insufficient, say so clearly.
"""
    try:
        client = _create_client(api_key)
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=400),
        )
        if response.text and response.text.strip():
            return True, response.text.strip()
        return False, "Iris AI returned an empty response; statistical recommendations remain available."
    except Exception as exc:
        logging.warning("Iris AI model review failed (%s)", type(exc).__name__)
        return False, (
            "Iris AI review is temporarily unavailable; statistical recommendations "
            "remain available. Check API key configuration and network connectivity."
        )


def _fallback_response(fallback_text: str, reason: str = "") -> str:
    message = fallback_text or "Iris could not generate an extended response."
    if reason:
        return f"{message}\n\n*{reason}*"
    return message


def _answer_chat_locally(question: str, dataset_context: str) -> Optional[str]:
    try:
        context = json.loads(dataset_context)
    except (TypeError, json.JSONDecodeError):
        context = {}

    q = question.casefold().strip()
    target = context.get("target_feature")
    target_type = context.get("target_feature_type", "unknown")
    stats = context.get("target_feature_statistics") or {}
    columns = context.get("columns") or {}
    row_count = context.get("row_count", "unknown")

    # Helper counts
    num_cols = sum(1 for t in columns.values() if t == "numeric") if isinstance(columns, dict) else 0
    cat_cols = sum(1 for t in columns.values() if t in ("categorical", "boolean", "text")) if isinstance(columns, dict) else 0
    total_cols = len(columns) if isinstance(columns, dict) else 0

    # 1. Dataset Profile / Overview / "Tell about data" / Shape / Dimensions
    data_overview_terms = (
        "about data", "about the data", "tell about", "tell me about",
        "overview", "summary", "describe data", "profile", "what is this data",
        "dataset summary", "data details", "understand data", "explain data",
        "data info", "dataset info", "shape", "dimension", "row count", "how many rows",
        "column", "feature"
    )
    if any(term in q for term in data_overview_terms) or q in ("data", "dataset", "info", "overview"):
        lines = []
        lines.append("### 📊 Dataset Profile & Structural Overview\n")
        lines.append(f"- **Scale:** **{row_count:,} rows** × **{total_cols} columns / features**.")
        lines.append(f"- **Feature Breakdown:** **{num_cols}** numeric features, **{cat_cols}** categorical / text features.")

        # High dimensionality warning (p ≈ n or p >= n * 0.4)
        if isinstance(row_count, (int, float)) and row_count > 0 and total_cols > 0:
            if total_cols >= row_count * 0.4 or (total_cols >= 40 and row_count <= 250):
                lines.append(
                    f"\n⚠️ **High-Dimensionality Warning ($p \\approx n$):** You have {total_cols} features for only {row_count} observations. "
                    "When the feature count approaches the sample size, standard algorithms face the *curse of dimensionality* and severe overfitting risk. "
                    "**Recommended Strategy:** In Stage 4 (Feature Engineering), prioritize feature selection—drop low-variance features and redundant correlated columns. "
                    "In Stage 5, favor regularized models (L1 Lasso, Random Forest with constrained `max_depth` and cross-validation) rather than complex unconstrained models."
                )

        if target:
            lines.append(f"\n🎯 **Target Variable:** **`{target}`** (Inferred Type: *{target_type.title()}*)")
            if target_type in ("categorical", "boolean"):
                unique_c = stats.get('unique', 'multiple')
                top_v = stats.get('top_value', 'N/A')
                lines.append(
                    f"- **Task Type:** **Multiclass / Categorical Classification** ({unique_c} unique categories, top class: *'{top_v}'*).\n"
                    "- **Objective:** Train classifiers to accurately map respondent survey attributes to their preferred career field."
                )
            elif target_type == "numeric":
                lines.append(
                    f"- **Task Type:** **Supervised Regression**.\n"
                    f"- **Objective:** Estimate continuous numerical values (Mean: {stats.get('mean', 'N/A')}, Median: {stats.get('median', 'N/A')})."
                )
        else:
            lines.append("\n🎯 **Target Variable:** Not selected yet. Choose a target column in Stage 1 or Stage 5 to unlock predictive modeling.")

        if columns:
            preview_cols = [f"`{col}` ({k})" for col, k in list(columns.items())[:10]]
            suffix = f" *(+ {total_cols - 10} more)*" if total_cols > 10 else ""
            lines.append(f"\n**Sample Features:** {', '.join(preview_cols)}{suffix}")

        lines.append(
            "\n💡 **Next Step in Iris:** Review data quality in **Stage 2 (Cleaning)** to verify missing values and duplicate rows, "
            "then proceed to **Stage 3 (Distributions)** to examine distributions and correlations."
        )
        return "\n".join(lines)

    # 2. Target column questions
    if any(term in q for term in ("target", "predict", "outcome", "label", "dependent variable")):
        if not target:
            return (
                "**Target Feature Overview:**\n\n"
                "A target feature is the outcome column you want a predictive model to learn and estimate. "
                "No target has been selected in this project yet. Please select your outcome in Stage 1 or Stage 5, "
                "ensuring it is not an ID column or a value unavailable at inference time."
            )
        ans = (
            f"**Current Target Feature:** **`{target}`** (Inferred type: *{target_type}*).\n\n"
            f"This is the supervised learning outcome Iris is focusing on. "
        )
        if target_type == "numeric" and stats:
            ans += f"Key metrics: Mean = {stats.get('mean', 'N/A')}, Median = {stats.get('median', 'N/A')}, Range = [{stats.get('min', 'N/A')} to {stats.get('max', 'N/A')}]."
        elif target_type in ("categorical", "boolean") and stats:
            ans += f"It has {stats.get('unique', 'N/A')} unique categories, with most frequent class being *'{stats.get('top_value', 'N/A')}'*."
        return ans

    # 3. Missing values / nulls
    if any(term in q for term in ("missing", "null", "nan", "impute", "empty value")):
        return (
            "**Handling Missing Values in Iris:**\n\n"
            "- **Drop High-Missing Features:** Columns with >40% missing data are usually best dropped as they add noise.\n"
            "- **Numeric Imputation:** Use Median for skewed distributions (robust to outliers) or Mean for symmetric Gaussian distributions.\n"
            "- **Categorical Imputation:** Use Mode (most frequent category) or create a dedicated `'Missing'` category indicator to preserve information."
        )

    # 4. Outliers & IQR / Z-score
    if any(term in q for term in ("outlier", "iqr", "zscore", "z-score", "anomaly", "cap", "trim")):
        return (
            "**Outlier Detection & Capping:**\n\n"
            "- **IQR Method (Robust):** Identifies values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`. Excellent for non-normal or skewed data.\n"
            "- **Z-Score Method:** Identifies observations further than 3 standard deviations from the mean (`|Z| > 3`). Best for bell-shaped distributions.\n"
            "- **Recommendation:** In Stage 2 (Cleaning), use **Cap / Winsorize** instead of dropping rows to preserve dataset size without distorting gradient updates."
        )

    # 5. Skewness and Log Transform
    if any(term in q for term in ("skew", "skewed", "skewness", "log transform", "log1p", "symmetric", "bell curve")):
        return (
            "**Distribution Skewness & Transformations:**\n\n"
            "- Skewness measures asymmetry. A skewness > 1.0 indicates heavy right-skew (long right tail with extreme large numbers, like Income or Prices).\n"
            "- **Log Transformation (`log1p`):** Compresses high values and spreads low values, pulling the distribution closer to a Gaussian bell curve.\n"
            "- **Why it matters:** Linear Regression, Logistic Regression, Neural Networks, and distance algorithms (KNN) converge much faster and perform significantly better with un-skewed inputs."
        )

    # 6. Feature Engineering & Encodings (One-Hot vs Binary vs Frequency)
    if any(term in q for term in ("encode", "encoding", "one hot", "one-hot", "dummy", "binary encode", "frequency encode", "label encode", "scaling", "scale", "standardize")):
        return (
            "**Feature Engineering & Encoding Guide:**\n\n"
            "- **One-Hot Encoding:** Creates binary 0/1 columns for each category. Best for low-cardinality nominal features (<= 10 categories, e.g. Color, Gender).\n"
            "- **Binary / Label Encoding:** Assigns numeric ranks. Ideal for binary flags or ordinal categories (e.g. Low/Medium/High).\n"
            "- **Frequency Encoding:** Replaces category strings with their occurrence percentage. Ideal for high-cardinality categorical features (e.g. ZipCode, City).\n"
            "- **Standardization (Z-Score):** Scales numeric columns to `mean=0, std=1`. Crucial for Ridge, Lasso, SVM, and KNN."
        )

    # 7. Model selection and recommendations
    if any(term in q for term in ("model", "algorithm", "random forest", "xgboost", "logistic", "linear", "recommend", "train", "classification", "regression")):
        if target_type == "numeric":
            return (
                "**Model Recommendations for Continuous Target:**\n\n"
                f"Since your target `{target or 'outcome'}` is **numeric**, this is a **Regression** problem.\n"
                "- **Random Forest Regressor:** Excellent nonlinear baseline, resistant to outliers, handles feature interactions without scaling.\n"
                "- **Gradient Boosting (XGBoost / LightGBM):** Top-tier accuracy on tabular datasets, highly tuned for predictive power.\n"
                "- **Ridge / Lasso Regression:** Great for interpretable linear baselines and feature importance."
            )
        elif target_type in ("categorical", "boolean"):
            return (
                "**Model Recommendations for Categorical Target:**\n\n"
                f"Since your target `{target or 'outcome'}` is **categorical**, this is a **Classification** problem.\n"
                "- **Random Forest Classifier:** Strong ensemble baseline, robust against overfitting, provides instant feature importances.\n"
                "- **XGBoost / LightGBM:** State-of-the-art accuracy for tabular classification tasks.\n"
                "- **Logistic Regression:** Clean probabilistic baseline when scaled with standard scaling."
            )
        return (
            "**Supervised Model Strategy:**\n\n"
            "If your target is numeric continuous, use Gradient Boosting or Random Forest Regressors. "
            "If your target represents distinct categories or classes, use Random Forest or XGBoost Classifiers. "
            "Check Stage 5 for personalized algorithmic candidate cards."
        )

    # 8. Correlations & Bivariate relationships
    if any(term in q for term in ("correlation", "relationship", "correlated", "pearson", "spearman", "bivariate")):
        return (
            "**Interpreting Correlations in Iris:**\n\n"
            "- **Pearson r (-1 to +1):** Measures linear co-movement. Absolute values above 0.7 indicate strong correlation.\n"
            "- **Multicollinearity:** If two independent predictors have r > 0.85, consider dropping one or combining them with PCA/ratios to avoid model variance inflation.\n"
            "- **Target Correlation:** Features with moderate to strong correlation with your target column are your strongest candidate predictors."
        )

    # 9. General guidance / next steps
    return (
        f"**Iris Data Science Assistant:**\n\n"
        f"I'm analyzing your dataset ({row_count:,} rows, {total_cols} columns" +
        (f", target: **`{target}`**" if target else "") + ").\n\n"
        "Here are helpful questions you can ask me:\n"
        "- *'Tell me about the data'* or *'Overview of dataset'*\n"
        "- *'Which features should I log transform?'*\n"
        "- *'How should I handle outliers in my dataset?'*\n"
        "- *'Should I use One-Hot or Frequency encoding?'*\n"
        "- *'What models are recommended for my target column?'*\n"
        "- *'Explain the difference between Pearson and Spearman correlation.'*"
    )


def _safe_generate(prompt: str, fallback_text: str = "") -> str:
    api_key = _get_api_key()
    if not api_key:
        return fallback_text or "Iris built-in analytical response."

    try:
        client = _create_client(api_key)
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=512),
        )
        if response.text and response.text.strip():
            return response.text.strip()
        raise ValueError("Empty response received")
    except Exception as exc:
        logging.warning("Iris AI request failed (%s)", type(exc).__name__)
        return fallback_text or "Iris AI response unavailable. Showing built-in analytical response."



# ── Step 1: File overview ──────────────────────────────────────────────────────

def explain_file_overview(meta: Dict, col_types: Dict) -> str:
    type_summary = ", ".join(f"{v}: {list(col_types.values()).count(v)}"
                             for v in set(col_types.values()))
    prompt = f"""
You are Iris, a friendly data science assistant. Explain this dataset overview in 3-4 plain English sentences.
Be specific, encouraging, and avoid jargon.

Dataset info:
- Rows: {meta['cleaned_rows']}, Columns: {meta['cleaned_cols']}
- Encoding detected: {meta['encoding']}
- Column types found: {type_summary}
- Columns: {list(col_types.keys())}

Mention what kind of data this might be, what the columns suggest, and what analysis could be interesting.
"""
    fallback = (f"Your dataset contains {meta['cleaned_rows']:,} rows and {meta['cleaned_cols']} columns with "
                f"{type_summary} columns detected. It is clean and ready for in-depth data quality check and exploratory visualization!")
    return _safe_generate(prompt, fallback)


# ── Step 2: Data quality ───────────────────────────────────────────────────────

def explain_data_quality(quality: Dict) -> str:
    missing_df = quality["missing"]
    dups = quality["duplicates"]
    outlier_df = quality["outliers"]

    worst_missing = missing_df.head(3).to_dict("records") if len(missing_df) > 0 else []
    worst_outliers = outlier_df.head(3).to_dict("records") if len(outlier_df) > 0 else []

    prompt = f"""
You are Iris, a friendly data science assistant. Explain these data quality findings in 4-5 plain English sentences.
Focus on what matters most and give actionable advice.

Quality findings:
- Duplicate rows: {dups['duplicate_rows']} ({dups['duplicate_pct']}%)
- Top columns with missing values: {json.dumps(worst_missing)}
- Top columns with outliers (IQR method): {json.dumps(worst_outliers)}

Tell the user what they should be concerned about and what they might do about it.
"""
    dup_msg = f"{dups['duplicate_rows']} duplicate row(s) detected." if dups['duplicate_rows'] else "No duplicate rows found."
    miss_msg = f"Columns with missing values: {len(missing_df)}." if len(missing_df) else "No missing values found across all columns."
    out_msg = f"{len(outlier_df)} numeric column(s) have potential outliers." if len(outlier_df) else "No severe outliers detected."
    fallback = f"Data quality check complete: {dup_msg} {miss_msg} {out_msg} Review the quality cards above before moving to column exploration."
    return _safe_generate(prompt, fallback)


# ── Step 3: Column EDA ─────────────────────────────────────────────────────────

def explain_column(col: str, stats: Dict) -> str:
    prompt = f"""
You are Iris, a friendly data science assistant. Explain this column's statistics in 2-3 plain English sentences.
Column: '{col}' (type: {stats.get('col_type', 'unknown')})
Stats: {json.dumps({k: v for k, v in stats.items() if k not in ('value_counts',)}, default=str)}

Highlight anything unusual (high skewness, many missing values, suspicious patterns).
"""
    ctype = stats.get("col_type", "unknown")
    if ctype == "numeric":
        fallback = f"Column '{col}' has mean {stats.get('mean')}, median {stats.get('median')}, and std dev {stats.get('std')}. Range spans from {stats.get('min')} to {stats.get('max')}."
    elif ctype in ("categorical", "boolean"):
        fallback = f"Column '{col}' contains {stats.get('unique')} distinct categories. Top value is '{stats.get('top_value')}' making up {stats.get('top_freq_pct')}% of entries."
    else:
        fallback = f"Column '{col}' is classified as {ctype} with {stats.get('unique')} unique entries and {stats.get('missing')} missing values."
    return _safe_generate(prompt, fallback)


# ── Step 4: Correlations ───────────────────────────────────────────────────────

def explain_correlations(top_pairs: list) -> str:
    prompt = f"""
You are Iris, a friendly data science assistant. Explain the top correlations in this dataset in 4-5 sentences.
Use plain English. Mention whether correlations are strong/weak/negative and what they might mean.

Top correlated pairs (Pearson r): {json.dumps(top_pairs[:5])}
"""
    if top_pairs:
        p1 = top_pairs[0]
        fallback = f"The strongest linear correlation is between '{p1['col1']}' and '{p1['col2']}' with Pearson r = {p1['pearson_r']}. Higher values in one column strongly align with changes in the other."
    else:
        fallback = "Correlation matrix computed. No strong linear correlations detected among the numeric columns."
    return _safe_generate(prompt, fallback)


# ── Step 5: ML recommendation ─────────────────────────────────────────────────

def explain_ml_recommendation(rec: Dict) -> str:
    prompt = f"""
You are Iris, a friendly data science assistant. Explain this ML recommendation in 3-4 plain English sentences.
Make it sound exciting and approachable for a non-technical user.

Recommendation: {rec['task']}
Reason: {rec['reason']}
"""
    fallback = f"Iris recommends **{rec['task']}** because {rec['reason']}. Training baseline models will evaluate performance metrics and feature importance automatically."
    return _safe_generate(prompt, fallback)


# ── Step 6: Model results ─────────────────────────────────────────────────────

def explain_model_results(task: str, best_model: str, results: Dict) -> str:
    prompt = f"""
You are Iris, a friendly data science assistant. Explain these machine learning results in 4-5 plain English sentences.
Avoid technical jargon. Tell the user what the numbers mean in real terms and whether the model is good.

Task: {task}
Best model: {best_model}
All model results: {json.dumps(results)}
"""
    fallback = f"Model training complete! The top performing algorithm was **{best_model}**. Inspect feature importance and evaluation metrics below to understand key drivers."
    return _safe_generate(prompt, fallback)


# ── General chat ───────────────────────────────────────────────────────────────

def chat_with_iris(question: str, dataset_context: str) -> str:
    local_answer = _answer_chat_locally(question, dataset_context)

    if not _get_api_key():
        if local_answer:
            return local_answer
        return (
            "I'm Iris, your data science assistant. Ask me about your columns, target feature, "
            "missing values, skewness, feature encoding, or model recommendations."
        )

    prompt = f"""
You are Iris AI, an elite enterprise data science advisor embedded in Iris Studio.
Never refer to yourself as Gemini, Google, or any external AI service. Your name is Iris AI.
Answer the user's question clearly, practically, and concisely (3-5 sentences).
Use the dataset context provided to give data-specific insights wherever applicable.
Do not embed images, markdown image syntax, or raw HTML.

Dataset context:
{dataset_context}

User question: {question}
"""
    fallback = local_answer or (
        "I'm Iris, your data science assistant. Ask me about your columns, target feature, "
        "missing values, skewness, feature encoding, or model recommendations."
    )
    return _safe_generate(prompt, fallback)


def generate_executive_summary(pipeline_stats: Dict[str, Any]) -> str:
    """Generates a concise plain-English executive summary for Stage 5."""
    prompt = f"""
You are Iris, an enterprise Data Science Lead. Write a professional executive summary for leadership based on the following pipeline execution statistics:

Pipeline Stats:
{json.dumps(pipeline_stats, indent=2)}

Write no more than 150 words. Mention only facts present in the statistics, clearly state
that no model was trained, and give one practical next step. Use short headings.
"""
    fallback = f"""### 📄 Executive Summary Briefing

**Dataset:** {pipeline_stats.get('domain', 'General')} Data Analysis  
**Scope:** {pipeline_stats.get('total_rows', 0):,} records analyzed across {pipeline_stats.get('total_cols', 0)} features.

#### Key Discoveries:
- **Data Quality:** Health score updated from raw baseline to cleaned state.
- **Top Drivers:** Strong correlations identified across core numerical features.
- **Engineered Features:** {pipeline_stats.get('fe_count', 0)} new numerical, dummy, and calendar indicators generated.

#### Strategic Recommendations:
1. **Focus Action on Key Metrics:** Use top predictive features for decision-making.
2. **Automate Quality Pipelines:** Enforce baseline checks prior to model deployment.
3. **Monitor Anomaly Vectors:** Track high-skew and outlier features over time.
"""
    return _safe_generate(prompt, fallback)


def generate_stage_insights(stage_name: str, context_dict: Dict[str, Any]) -> str:
    """Generates 3 bullet insights for a specific pipeline stage."""
    prompt = f"""
You are Iris AI. Generate 3 short, punchy bullet point insights summarizing stage '{stage_name}' based on this data:
{json.dumps(context_dict, indent=2)}
"""
    fallback = f"""• Analyzed {stage_name} metrics with automated validation rules.
• Verified feature health and column distribution integrity.
• Pipeline ready to transition to the next analytical stage."""
    return _safe_generate(prompt, fallback)
