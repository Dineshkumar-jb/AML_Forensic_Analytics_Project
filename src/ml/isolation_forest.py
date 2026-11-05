"""Isolation Forest unsupervised anomaly detection module for AML monitoring.

Detects outliers by recursively isolating observation points using random feature splits.
Anomalous points require fewer splits to isolate (shorter path lengths in isolation trees).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def train_isolation_forest(
    features_df: pd.DataFrame,
    contamination: float = 0.08,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit Isolation Forest and generate unsupervised transaction anomaly scores.

    Args:
        features_df: Numerical feature DataFrame
        contamination: Expected proportion of outliers in the data (default 0.08)
        n_estimators: Number of isolation trees in ensemble
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing model, predictions, normalized 0-100 anomaly scores,
        and summary anomaly statistics.
    """
    if features_df is None or features_df.empty:
        return {"is_valid": False, "error": "Features DataFrame is empty."}

    X = features_df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize Isolation Forest
    iso_model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    # Fit and predict: -1 for outlier, 1 for inlier
    preds = iso_model.fit_predict(X_scaled)
    # Map: 1 if anomaly (-1), 0 if normal (+1)
    is_anomaly = (preds == -1).astype(int)

    # Raw decision function: lower / negative values represent anomalies
    raw_decision = iso_model.decision_function(X_scaled)

    # Normalize to intuitive 0-100 scale:
    # Most anomalous (lowest decision score) -> 100
    # Most normal (highest decision score) -> 0
    min_dec = float(np.min(raw_decision))
    max_dec = float(np.max(raw_decision))

    if max_dec > min_dec:
        norm_scores = ((max_dec - raw_decision) / (max_dec - min_dec)) * 100.0
    else:
        norm_scores = np.zeros(len(raw_decision))

    norm_scores = np.clip(norm_scores, 0.0, 100.0)

    total_records = len(features_df)
    anomaly_count = int(np.sum(is_anomaly))
    anomaly_pct = (anomaly_count / total_records * 100.0) if total_records > 0 else 0.0

    return {
        "is_valid": True,
        "model": iso_model,
        "scaler": scaler,
        "contamination": contamination,
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": round(anomaly_pct, 2),
        "is_anomaly": is_anomaly,
        "raw_decision_scores": raw_decision,
        "anomaly_scores_0_100": np.round(norm_scores, 1),
        "mean_anomaly_score": round(float(np.mean(norm_scores)), 1),
        "max_anomaly_score": round(float(np.max(norm_scores)), 1),
    }
