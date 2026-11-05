"""Explainability and Transaction Investigation module for AML Forensic Analytics.

Stage 5: Integrates Explainable AI (SHAP TreeExplainer for supervised XGBoost models)
and transparent Rule-Based Forensic Attribution when models are unavailable or unlabeled.
Provides investigator dossiers and actionable compliance recommendations.
"""

from src.explainability.engine import (
    compute_explainability,
    get_transaction_explanation,
    get_investigation_dossier,
    generate_rule_based_importance,
)

__all__ = [
    "compute_explainability",
    "get_transaction_explanation",
    "get_investigation_dossier",
    "generate_rule_based_importance",
]

