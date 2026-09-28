"""
engine/business.py
Domain templates, business objective configurations, target suggestions, and business context metadata.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


DOMAINS = {
    "finance": {
        "name": "Finance & Risk",
        "icon": "",
        "description": "Credit scoring, fraud detection, revenue forecasting, default prediction",
        "default_goal": "Predict financial risk and identify high-value customer transactions to minimize default rate.",
        "target_keywords": ["churn", "default", "fraud", "amount", "price", "profit", "sales", "revenue", "target", "risk"]
    },
    "healthcare": {
        "name": "Healthcare & Life Sciences",
        "icon": "",
        "description": "Patient outcome prediction, diagnosis classification, clinical trial analysis",
        "default_goal": "Analyze patient metrics to classify risk factors and predict clinical outcomes.",
        "target_keywords": ["target", "outcome", "diagnosis", "disease", "survived", "readmitted", "mortality", "severity"]
    },
    "ecommerce": {
        "name": "E-Commerce & Retail",
        "icon": "",
        "description": "Customer lifetime value, sales forecasting, churn prediction, cross-sell analysis",
        "default_goal": "Understand sales drivers and predict customer churn to increase repeat purchase value.",
        "target_keywords": ["sales", "profit", "churn", "segment", "quantity", "discount", "rating", "purchased"]
    },
    "manufacturing": {
        "name": "Manufacturing & Logistics",
        "icon": "",
        "description": "Predictive maintenance, supply chain optimization, yield quality control",
        "default_goal": "Detect equipment failure patterns and optimize delivery timelines to reduce downtime.",
        "target_keywords": ["failure", "downtime", "yield", "defect", "quality", "cost", "delay", "status"]
    },
    "tech": {
        "name": "SaaS & Tech Product",
        "icon": "",
        "description": "User retention, feature usage analytics, subscription churn, conversion rate",
        "default_goal": "Analyze user activity signals to predict churn and increase product engagement.",
        "target_keywords": ["churn", "converted", "retained", "active", "mrr", "plan", "tenure", "usage"]
    },
    "general": {
        "name": "General Analytics",
        "icon": "",
        "description": "General exploratory analysis, regression, classification, or data cleaning",
        "default_goal": "Perform exploratory analysis, clean dataset, and identify key drivers of primary business metrics.",
        "target_keywords": ["target", "label", "class", "result", "y", "output", "price", "score"]
    }
}


OBJECTIVES = {
    "predict": {
        "name": "Predict Outcomes (ML Modeling)",
        "badge": "Machine Learning",
        "description": "Train predictive regression or classification models on your cleaned data.",
        "recommended_stages": ["Stage 1", "Stage 2", "Stage 4"]
    },
    "segment": {
        "name": "Find Customer Segments (Clustering)",
        "badge": "Unsupervised",
        "description": "Group similar entities together using clustering and feature dimensionality reduction.",
        "recommended_stages": ["Stage 1", "Stage 2", "Stage 3"]
    },
    "detect": {
        "name": "Detect Anomalies & Outliers",
        "badge": "Data Quality",
        "description": "Highlight rare events, suspicious rows, or statistical anomalies in metrics.",
        "recommended_stages": ["Stage 2", "Stage 3"]
    },
    "explore": {
        "name": "Explore & Generate Executive Report",
        "badge": "Analytics",
        "description": "Uncover correlations, distribution shapes, and key findings for business leadership.",
        "recommended_stages": ["Stage 1", "Stage 3", "Stage 5"]
    }
}


def get_domain_templates() -> Dict[str, Any]:
    """Returns available industry domain templates."""
    return DOMAINS


def get_objective_configs() -> Dict[str, Any]:
    """Returns objective template configurations."""
    return OBJECTIVES


def suggest_target_column(df: pd.DataFrame, domain: str = "general") -> Optional[str]:
    """
    Suggests probable target column based on column names and domain keywords.
    """
    if df is None or df.empty:
        return None
    
    keywords = DOMAINS.get(domain, DOMAINS["general"])["target_keywords"]
    
    # 1. Exact match checking
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower in ["target", "label", "y", "churn", "survived", "outcome"]:
            return col
            
    # 2. Substring match against domain keywords
    for kw in keywords:
        for col in df.columns:
            if kw in str(col).lower():
                return col
                
    # 3. Fallback to last column if numeric or binary
    last_col = df.columns[-1]
    return last_col
