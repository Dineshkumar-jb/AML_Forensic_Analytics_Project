"""Supervised XGBoost classification module for AML transaction monitoring.

Operates strictly when genuine ground-truth AML labels (e.g., 'is_laundering' or SAR outcome)
exist in the dataset. Does NOT fabricate labels when labels are absent.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb


def detect_ground_truth_label(
    df: pd.DataFrame,
    preferred_cols: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str], str]:
    """Detect if a valid binary AML ground-truth target column is present.

    Returns:
        (has_valid_target, target_col_name, status_message)
    """
    candidates = preferred_cols or ["is_laundering", "label", "is_fraud", "target", "sar_flag"]

    for col in candidates:
        if col in df.columns:
            s = df[col].dropna()
            unique_vals = set(s.unique())
            # Check for valid binary representation (e.g. {0, 1} or {True, False})
            if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                pos_count = int((s == 1).sum() if 1 in unique_vals else (s == True).sum())
                neg_count = int((s == 0).sum() if 0 in unique_vals else (s == False).sum())
                if pos_count >= 2 and neg_count >= 5:
                    return True, col, f"Detected valid target column '{col}' ({pos_count} positive, {neg_count} negative instances)."
                else:
                    return False, col, f"Column '{col}' found, but insufficient class balance ({pos_count} positives) for stratified training."

    return False, None, "No ground-truth AML label column found in the current dataset (Supervised training requires confirmed SAR/audit labels)."


def train_xgboost(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    feature_names: List[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train supervised XGBoost classifier and compute authentic validation metrics.

    Args:
        features_df: Numerical feature DataFrame
        target_series: Binary ground truth target (0 = legitimate, 1 = laundering)
        feature_names: List of feature names
        test_size: Test split ratio
        random_state: Random seed

    Returns:
        Dictionary containing trained model, genuine evaluation metrics, feature importances,
        and predictions.
    """
    if features_df is None or features_df.empty or target_series is None:
        return {"is_trained": False, "reason": "Missing feature matrix or target labels."}

    y = target_series.astype(int).values
    X = features_df.values

    # Stratified Train/Test Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    except ValueError as e:
        # Fallback to non-stratified if class representation is too small
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # Calculate class imbalance weighting: scale_pos_weight = neg_count / pos_count
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos = (neg_count / max(1, pos_count))

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate strictly on the holdout test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Authentic metrics computation
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else None
    except Exception:
        roc_auc = None

    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    # Feature Importance Table
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Full dataset inference for scoring (0 - 100 risk score)
    all_probs = model.predict_proba(X)[:, 1]
    all_preds = model.predict(X)

    return {
        "is_trained": True,
        "model": model,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "confusion_matrix": conf_mat,
        "feature_importances": importance_df,
        "all_preds": all_preds,
        "all_probs_0_100": np.round(all_probs * 100.0, 1),
    }
