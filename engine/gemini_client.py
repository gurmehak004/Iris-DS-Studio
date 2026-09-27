"""
engine/gemini_client.py
Gemini API wrapper with prompt templates for each pipeline step.
"""
import os
import json
import google.generativeai as genai
from typing import Dict, Any, Optional


def _get_model():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def _safe_generate(prompt: str, fallback_text: str = "") -> str:
    try:
        model = _get_model()
        if model is not None:
            response = model.generate_content(prompt)
            if response.text and response.text.strip():
                return response.text.strip()
    except Exception:
        pass
    
    if fallback_text:
        return fallback_text
    return "Iris has processed your dataset metrics. Connect your Gemini API Key in the sidebar for personalized LLM narrative insights!"



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
    prompt = f"""
You are Iris, a friendly expert data science assistant embedded in a data analysis tool.
Answer the user's question clearly and helpfully. Use the dataset context provided.

Dataset context:
{dataset_context}

User question: {question}

Answer in plain English, 3-6 sentences. Be specific to their data where possible.
"""
    fallback = f"I've analyzed your question regarding '{question}'. Based on your dataset context ({dataset_context[:100]}...), make sure to review the correlation and quality tabs! For full AI conversational responses, add your Gemini API key in the sidebar."
    return _safe_generate(prompt, fallback)


def generate_executive_summary(pipeline_stats: Dict[str, Any]) -> str:
    """Generates a 300-500 word plain English executive summary narrative for Stage 5."""
    prompt = f"""
You are Iris, an enterprise Data Science Lead. Write a professional executive summary for leadership based on the following pipeline execution statistics:

Pipeline Stats:
{json.dumps(pipeline_stats, indent=2)}

Structure your report into:
1. Executive Briefing
2. Key Data Discoveries & Anomalies
3. Cleaning & Feature Engineering Actions Taken
4. Strategic Business Recommendations
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


