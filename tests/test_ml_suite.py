"""Automated validation tests for Stage 3 Machine Learning detection layer."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import generate_sample_dataset
from src.forensic.engine import run_forensic_suite
from src.ml.feature_engineering import extract_ml_features
from src.ml.isolation_forest import train_isolation_forest
from src.ml.xgboost_model import detect_ground_truth_label, train_xgboost
from src.ml.engine import run_ml_pipeline


def test_ml_pipeline():
    print("--- 1. Loading Sample Dataset & Running Forensic Stage 2 ---")
    df = generate_sample_dataset(save_to_disk=False)
    forensic_res = run_forensic_suite(df)
    enriched_forensic_df = forensic_res["enriched_df"]
    print(f"Dataset shape: {df.shape}")

    print("\n--- 2. Testing Feature Engineering ---")
    features_df, feature_names = extract_ml_features(df, enriched_forensic_df)
    assert not features_df.empty, "Feature matrix is empty!"
    assert len(feature_names) >= 10, f"Expected at least 10 features, got {len(feature_names)}"
    print(f"Engineered {len(feature_names)} features: {feature_names[:6]}... (and {len(feature_names)-6} more)")
    assert not features_df.isnull().any().any(), "Feature matrix contains unexpected NaNs!"

    print("\n--- 3. Testing Isolation Forest (Unsupervised) ---")
    iso_res = train_isolation_forest(features_df, contamination=0.08)
    assert iso_res["is_valid"], "Isolation Forest execution failed!"
    print(f"Total evaluated: {iso_res['total_records']}")
    print(f"Anomalies detected: {iso_res['anomaly_count']} ({iso_res['anomaly_percentage']}%)")
    print(f"Mean anomaly score (0-100): {iso_res['mean_anomaly_score']}")
    print(f"Max anomaly score (0-100): {iso_res['max_anomaly_score']}")
    assert iso_res["anomaly_count"] > 0, "Isolation forest should flag outliers at contamination=0.08"
    assert len(iso_res["anomaly_scores_0_100"]) == len(df)

    print("\n--- 4. Testing XGBoost with Ground-Truth Labels ('is_laundering') ---")
    has_labels, label_col, msg = detect_ground_truth_label(df)
    print(f"Label detection: {has_labels} (col: '{label_col}') - {msg}")
    assert has_labels, "Sample dataset should contain 'is_laundering' for testing!"

    xgb_res = train_xgboost(features_df, df[label_col], feature_names)
    assert xgb_res["is_trained"], "XGBoost training failed!"
    print(f"Trained on {xgb_res['train_samples']} samples, tested on {xgb_res['test_samples']} holdout samples.")
    print(f"Authentic Holdout Test Metrics:")
    print(f"  - Accuracy: {xgb_res['accuracy']:.4f}")
    print(f"  - Precision: {xgb_res['precision']:.4f}")
    print(f"  - Recall: {xgb_res['recall']:.4f}")
    print(f"  - F1-Score: {xgb_res['f1_score']:.4f}")
    print(f"  - ROC-AUC: {xgb_res['roc_auc']}")
    print(f"  - Confusion Matrix: {xgb_res['confusion_matrix']}")
    print("\nTop 5 Important Features for XGBoost:")
    print(xgb_res["feature_importances"].head(5).to_string())

    print("\n--- 5. Testing XGBoost Graceful Handling when Labels are Absent ---")
    unlabeled_df = df.drop(columns=["is_laundering"])
    has_unlabeled, _, unlabel_msg = detect_ground_truth_label(unlabeled_df)
    assert not has_unlabeled, "Should report no labels when column is dropped!"
    print(f"Unlabeled dataset handling message: {unlabel_msg}")

    print("\n--- 6. Testing Full ML Pipeline Orchestrator ---")
    pipeline_res = run_ml_pipeline(df, enriched_forensic_df)
    assert pipeline_res["is_valid"]
    print("\nModel Comparison Table:")
    print(pipeline_res["comparison_table"].to_string())

    enriched_df = pipeline_res["enriched_df"]
    assert "iso_forest_score" in enriched_df.columns
    assert "iso_forest_anomaly" in enriched_df.columns
    assert "xgb_risk_score" in enriched_df.columns
    print(f"\nEnriched DataFrame has columns: {list(enriched_df.columns)}")

    print("\n[SUCCESS] ALL STAGE 3 ML DETECTION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_ml_pipeline()
