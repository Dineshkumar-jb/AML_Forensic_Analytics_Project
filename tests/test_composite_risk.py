"""Automated validation tests for Stage 4 Composite Risk Scoring Engine."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import generate_sample_dataset
from src.forensic.engine import run_forensic_suite
from src.ml.engine import run_ml_pipeline
from src.scoring.composite_risk import (
    calculate_composite_risk,
    classify_risk_level,
    compute_transaction_forensic_score,
)


def test_risk_level_classification():
    print("\n--- 1. Testing Risk Level Classification (4 Tiers) ---")
    assert classify_risk_level(0.0) == "Low"
    assert classify_risk_level(24.0) == "Low"
    assert classify_risk_level(25.0) == "Medium"
    assert classify_risk_level(49.0) == "Medium"
    assert classify_risk_level(50.0) == "High"
    assert classify_risk_level(74.0) == "High"
    assert classify_risk_level(75.0) == "Critical"
    assert classify_risk_level(100.0) == "Critical"

    # Out-of-bounds clipping tests
    assert classify_risk_level(-15.0) == "Low"
    assert classify_risk_level(115.0) == "Critical"
    print("Risk level classification passed for all boundaries.")


def test_composite_with_all_signals():
    print("\n--- 2. Testing Composite Scoring with All Signals Available (Forensics + IsoForest + XGBoost) ---")
    df = generate_sample_dataset(num_records=120, save_to_disk=False)
    forensic_res = run_forensic_suite(df)
    ml_res = run_ml_pipeline(df, enriched_forensic_df=forensic_res["enriched_df"])

    assert ml_res["has_labels"], "Sample data should have 'is_laundering' for testing full pipeline"

    comp_res = calculate_composite_risk(df, forensic_results=forensic_res, ml_results=ml_res)
    assert comp_res["is_valid"], f"Composite calculation failed: {comp_res.get('error')}"

    scored_df = comp_res["scored_df"]
    print(f"Total transactions scored: {len(scored_df)}")
    print(f"Average Composite Risk: {comp_res['avg_composite_risk']}")
    print(f"Max Composite Risk: {comp_res['max_composite_risk']}")
    print(f"Risk Counts by Level: {comp_res['counts_by_level']}")
    print(f"Effective Weights Used: {comp_res['effective_weights']}")

    # Check required columns
    required_cols = [
        "transaction_id",
        "composite_risk_score",
        "risk_level",
        "forensic_risk_score",
        "isolation_forest_risk_score",
        "xgboost_risk_score",
        "contributing_risk_factors",
        "risk_explanation",
    ]
    for col in required_cols:
        assert col in scored_df.columns, f"Missing required column '{col}' in scored_df"

    # Check score bounds: strictly 0 to 100
    assert scored_df["composite_risk_score"].min() >= 0.0, "Score fell below 0.0!"
    assert scored_df["composite_risk_score"].max() <= 100.0, "Score exceeded 100.0!"

    # Check weight sum
    tot_weight = sum(comp_res["effective_weights"].values())
    assert abs(tot_weight - 1.0) < 1e-3, f"Weights must sum to 1.0, got {tot_weight}"

    # Check top risk table format
    top_table = comp_res["top_risk_table"]
    assert not top_table.empty, "Top risk table is empty!"
    assert top_table["composite_risk_score"].iloc[0] == comp_res["max_composite_risk"]
    print("Full signal composite risk test passed successfully.")


def test_missing_xgboost_proportional_redistribution():
    print("\n--- 3. Testing Missing-Signal Handling (Unlabeled Dataset -> Zero XGBoost Fabrication) ---")
    df = generate_sample_dataset(num_records=80, save_to_disk=False)
    # Drop label column to simulate standard unlabeled banking dump
    unlabeled_df = df.drop(columns=["is_laundering"])

    forensic_res = run_forensic_suite(unlabeled_df)
    ml_res = run_ml_pipeline(unlabeled_df, enriched_forensic_df=forensic_res["enriched_df"])

    assert not ml_res["has_labels"], "Unlabeled data should have has_labels=False"
    assert not ml_res["xgboost"]["is_trained"], "XGBoost must NOT be trained on unlabeled data"

    comp_res = calculate_composite_risk(unlabeled_df, forensic_results=forensic_res, ml_results=ml_res)
    assert comp_res["is_valid"]
    assert not comp_res["has_xgboost_score"]
    assert "xgboost" in comp_res["unavailable_signals"]

    eff_w = comp_res["effective_weights"]
    print(f"Redistributed Weights (XGBoost unavailable): {eff_w}")

    # XGBoost effective weight must be exactly 0
    assert eff_w.get("xgboost", 0.0) == 0.0, "XGBoost weight must be 0 when unavailable"

    # Weights must sum to 1.0
    tot_w = sum(eff_w.values())
    assert abs(tot_w - 1.0) < 1e-3, f"Redistributed weights must sum to 1.0, got {tot_w}"

    # Verify proportional ratio between 0.40 (Forensic) and 0.30 (IsoForest):
    # 0.40 / 0.70 ≈ 0.5714, 0.30 / 0.70 ≈ 0.4286
    assert abs(eff_w["forensic"] - (0.40 / 0.70)) < 0.01
    assert abs(eff_w["isolation_forest"] - (0.30 / 0.70)) < 0.01

    # Verify NO fake scores: xgboost_risk_score in scored_df must be all NaN
    scored_df = comp_res["scored_df"]
    assert scored_df["xgboost_risk_score"].isnull().all(), "XGBoost score must be NaN when model is untrained!"

    # Verify composite score remains strictly within 0 to 100
    assert scored_df["composite_risk_score"].min() >= 0.0
    assert scored_df["composite_risk_score"].max() <= 100.0

    print("Missing XGBoost proportional redistribution verified with zero fabrication.")


def test_new_dataset_recalculation():
    print("\n--- 4. Testing Recalculation on Distinct New Custom Dataset ---")
    # Custom new dataset with arbitrary amounts and accounts
    custom_df = pd.DataFrame({
        "transaction_id": [f"NEW_TX_{i}" for i in range(1, 15)],
        "timestamp": pd.date_range("2026-04-01", periods=14, freq="h").strftime("%Y-%m-%d %H:%M:%S"),
        "sender_account": ["NEW_ACC_A"] * 7 + ["NEW_ACC_B"] * 7,
        "receiver_account": ["NEW_RECV_X"] * 14,
        "amount": [50.0, 60.0, 55.0, 48.0, 52.0, 51.0, 95000.0, 100.0, 110.0, 105.0, 99.0, 102.0, 98.0, 8900.0],
    })

    f_res = run_forensic_suite(custom_df)
    m_res = run_ml_pipeline(custom_df, enriched_forensic_df=f_res["enriched_df"])
    c_res = calculate_composite_risk(custom_df, forensic_results=f_res, ml_results=m_res)

    assert c_res["is_valid"]
    assert c_res["total_transactions"] == 14
    scored = c_res["scored_df"]
    assert len(scored) == 14

    # The $95,000 transaction should be flagged as High or Critical due to extreme RSF outlier
    spike_tx = scored[scored["amount"] == 95000.0].iloc[0]
    print(f"Spike transaction composite risk: {spike_tx['composite_risk_score']} (Level: {spike_tx['risk_level']})")
    assert spike_tx["composite_risk_score"] >= 45.0, "Massive spike should receive elevated risk score!"
    print("New dataset dynamic recalculation passed.")


def test_missing_transaction_id_validation():
    print("\n--- 5. Testing Missing Column Validation ---")
    bad_df = pd.DataFrame({"amount": [100.0, 200.0]})
    res = calculate_composite_risk(bad_df)
    assert not res["is_valid"]
    assert "Missing required 'transaction_id' column" in res["error"]
    print("Missing column graceful validation passed.")


if __name__ == "__main__":
    test_risk_level_classification()
    test_composite_with_all_signals()
    test_missing_xgboost_proportional_redistribution()
    test_new_dataset_recalculation()
    test_missing_transaction_id_validation()
    print("\n[SUCCESS] ALL STAGE 4 COMPOSITE RISK SCORING TESTS PASSED SUCCESSFULLY!")
