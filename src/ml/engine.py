"""Machine Learning Detection Engine Orchestrator.

Coordinates:
1. Feature extraction combining transaction attributes & Stage 2 forensic signals
2. Isolation Forest unsupervised anomaly detection
3. Supervised XGBoost classification (when genuine ground-truth labels exist)
4. Model comparison table and DataFrame enrichment
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.ml.feature_engineering import extract_ml_features
from src.ml.isolation_forest import train_isolation_forest
from src.ml.xgboost_model import detect_ground_truth_label, train_xgboost


def run_ml_pipeline(
    df: pd.DataFrame,
    enriched_forensic_df: Optional[pd.DataFrame] = None,
    contamination: float = 0.08,
) -> Dict[str, Any]:
    """Execute complete Machine Learning detection layer.

    Args:
        df: Base transaction DataFrame
        enriched_forensic_df: DataFrame containing Stage 2 forensic indicators
        contamination: Expected proportion of outliers for Isolation Forest

    Returns:
        Dictionary containing extracted features, Isolation Forest results,
        XGBoost results (if labeled), model comparison, and enriched DataFrame.
    """
    if df is None or df.empty:
        return {"is_valid": False, "error": "DataFrame is empty."}

    # 1. Feature Engineering
    features_df, feature_names = extract_ml_features(df, enriched_forensic_df)

    # 2. Unsupervised Anomaly Detection: Isolation Forest
    iso_res = train_isolation_forest(
        features_df=features_df,
        contamination=contamination,
    )

    # 3. Supervised Classification: Check for Genuine Ground Truth
    has_labels, label_col, label_msg = detect_ground_truth_label(df)
    xgb_res = {"is_trained": False, "reason": label_msg}

    if has_labels and label_col is not None:
        target_series = df[label_col]
        xgb_res = train_xgboost(
            features_df=features_df,
            target_series=target_series,
            feature_names=feature_names,
        )
        xgb_res["label_column"] = label_col
        xgb_res["label_message"] = label_msg

    # 4. Model Comparison Table
    comparison_rows = []

    # Isolation Forest Row
    comparison_rows.append({
        "Model": "Isolation Forest",
        "Learning Paradigm": "Unsupervised Anomaly Detection",
        "Ground Truth Dependency": "None (Operates on raw feature geometry)",
        "Flagged Suspicious": f"{iso_res['anomaly_count']} ({iso_res['anomaly_percentage']}%)",
        "Key Evaluation Metric": f"Contamination: {contamination:.2f} | Avg Anomaly Score: {iso_res['mean_anomaly_score']:.1f}",
        "Status": "Active (Unsupervised)",
    })

    # XGBoost Row
    if xgb_res.get("is_trained"):
        conf_mat_str = f"[[{xgb_res['confusion_matrix'][0][0]}, {xgb_res['confusion_matrix'][0][1]}], [{xgb_res['confusion_matrix'][1][0]}, {xgb_res['confusion_matrix'][1][1]}]]"
        xgb_flagged_count = int(np.sum(xgb_res["all_preds"] == 1))
        xgb_flagged_pct = round(xgb_flagged_count / len(df) * 100, 2)
        xgb_metric = (
            f"F1: {xgb_res['f1_score']:.3f} | Recall: {xgb_res['recall']:.3f} | "
            f"Prec: {xgb_res['precision']:.3f} | AUC: {xgb_res['roc_auc']}"
        )
        comparison_rows.append({
            "Model": "XGBoost Classifier",
            "Learning Paradigm": "Supervised Gradient Boosting",
            "Ground Truth Dependency": f"Trained on '{label_col}' (SAR / Audit Labels)",
            "Flagged Suspicious": f"{xgb_flagged_count} ({xgb_flagged_pct}%)",
            "Key Evaluation Metric": xgb_metric,
            "Status": "Active (Supervised)",
        })
    else:
        comparison_rows.append({
            "Model": "XGBoost Classifier",
            "Learning Paradigm": "Supervised Gradient Boosting",
            "Ground Truth Dependency": "Requires Verified Labels (e.g., 'is_laundering')",
            "Flagged Suspicious": "Awaiting Labeled Dataset",
            "Key Evaluation Metric": "N/A (No synthetic labels fabricated)",
            "Status": "Awaiting Labels",
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # 5. DataFrame Enrichment with ML Scores
    enriched_df = (enriched_forensic_df if enriched_forensic_df is not None else df).copy()
    enriched_df["iso_forest_score"] = iso_res["anomaly_scores_0_100"]
    enriched_df["iso_forest_anomaly"] = iso_res["is_anomaly"]

    if xgb_res.get("is_trained"):
        enriched_df["xgb_risk_score"] = xgb_res["all_probs_0_100"]
        enriched_df["xgb_prediction"] = xgb_res["all_preds"]

    return {
        "is_valid": True,
        "features_df": features_df,
        "feature_names": feature_names,
        "isolation_forest": iso_res,
        "xgboost": xgb_res,
        "has_labels": has_labels,
        "comparison_table": comparison_df,
        "enriched_df": enriched_df,
    }
