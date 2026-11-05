"""Unit and integration tests for Stage 5: Explainable AI & Investigation Engine."""

import pytest
import pandas as pd
import numpy as np

from src.utils.data_loader import load_csv
from src.forensic import run_forensic_suite
from src.ml import run_ml_pipeline
from src.scoring import calculate_composite_risk
from src.explainability import (
    compute_explainability,
    get_transaction_explanation,
    get_investigation_dossier,
    generate_rule_based_importance,
)
from src.config import SAMPLE_DATA_PATH


@pytest.fixture(scope="module")
def prepared_pipeline_data():
    """Load sample data and execute Stages 1–4 pipeline."""
    df, _ = load_csv(SAMPLE_DATA_PATH)
    f_res = run_forensic_suite(df)
    ml_res = run_ml_pipeline(df, f_res.get("enriched_df"))
    comp_res = calculate_composite_risk(df, f_res, ml_res)
    return df, f_res, ml_res, comp_res


def test_compute_explainability_labeled(prepared_pipeline_data):
    """Test that explainability engine works on labeled dataset."""
    df, f_res, ml_res, comp_res = prepared_pipeline_data

    exp_res = compute_explainability(
        ml_results=ml_res,
        features_df=ml_res["features_df"],
        feature_names=ml_res["feature_names"],
        composite_results=comp_res,
    )

    assert exp_res["is_valid"] is True
    assert exp_res["method"] in ["shap", "rule_based"]
    assert exp_res["global_importance"] is not None
    assert not exp_res["global_importance"].empty
    assert "Feature" in exp_res["global_importance"].columns
    assert "Importance" in exp_res["global_importance"].columns


def test_compute_explainability_unlabeled_fallback(prepared_pipeline_data):
    """Verify that when labels are absent, SHAP is not faked and rule-based fallback activates."""
    df, _, _, _ = prepared_pipeline_data
    df_unlabeled = df.drop(columns=["is_laundering"])

    f_res = run_forensic_suite(df_unlabeled)
    ml_res = run_ml_pipeline(df_unlabeled, f_res.get("enriched_df"))
    comp_res = calculate_composite_risk(df_unlabeled, f_res, ml_res)

    exp_res = compute_explainability(
        ml_results=ml_res,
        features_df=ml_res["features_df"],
        feature_names=ml_res["feature_names"],
        composite_results=comp_res,
    )

    assert exp_res["is_valid"] is True
    assert exp_res["is_model_based"] is False
    assert exp_res["method"] == "rule_based"
    assert exp_res["shap_values"] is None
    assert "Supervised model not trained" in exp_res["status_message"] or "Rule-Based" in exp_res["status_message"]


def test_transaction_explanation_breakdown(prepared_pipeline_data):
    """Verify local feature contributions and directionality for an individual transaction."""
    df, f_res, ml_res, comp_res = prepared_pipeline_data
    exp_res = compute_explainability(
        ml_results=ml_res,
        features_df=ml_res["features_df"],
        feature_names=ml_res["feature_names"],
        composite_results=comp_res,
    )

    target_tx = comp_res["scored_df"]["transaction_id"].iloc[0]
    tx_exp = get_transaction_explanation(
        tx_id=target_tx,
        explainability_res=exp_res,
        scored_df=comp_res["scored_df"],
        features_df=ml_res["features_df"],
    )

    assert tx_exp["is_valid"] is True
    assert tx_exp["transaction_id"] == target_tx
    assert "contributions_df" in tx_exp
    contrib_df = tx_exp["contributions_df"]
    assert not contrib_df.empty
    assert "Contribution" in contrib_df.columns
    assert "Direction" in contrib_df.columns
    assert "Impact" in contrib_df.columns


def test_investigation_dossier_compliance_language(prepared_pipeline_data):
    """Verify investigation dossier contents and strict compliance non-criminal labeling."""
    df, f_res, ml_res, comp_res = prepared_pipeline_data
    exp_res = compute_explainability(
        ml_results=ml_res,
        features_df=ml_res["features_df"],
        feature_names=ml_res["feature_names"],
        composite_results=comp_res,
    )

    target_tx = "TX_1218"
    dossier = get_investigation_dossier(
        tx_id=target_tx,
        scored_df=comp_res["scored_df"],
        features_df=ml_res["features_df"],
        explainability_res=exp_res,
        forensic_results=f_res,
        ml_results=ml_res,
    )

    assert dossier["is_valid"] is True
    assert dossier["transaction_id"] == target_tx
    assert dossier["composite_risk_score"] >= 0.0
    assert dossier["risk_level"] in ["Low", "Medium", "High", "Critical"]
    assert isinstance(dossier["suspicious_indicators"], list)
    assert len(dossier["suspicious_indicators"]) > 0

    rec = dossier["recommended_action"]
    assert "title" in rec
    assert "priority" in rec
    assert "summary" in rec
    assert "steps" in rec
    assert len(rec["steps"]) >= 2

    # Verification: Must use terms such as "suspicious" or "potentially high-risk"
    # and not definitive criminal declarations
    summary_lower = rec["summary"].lower()
    assert "suspicious" in summary_lower or "potentially high-risk" in summary_lower or "anomaly" in summary_lower

    # Verification: Demo Mode scale distinction
    assert dossier.get("demo_mode") is True
    assert dossier.get("sample_size") == 218
    assert dossier.get("platform_scale") == 5_000_000
    assert "Demo Mode – 5M illustrative transaction scale; actual analysis uses 218 sample transactions." in dossier.get("scale_disclaimer")

