"""Feature engineering module for AML Machine Learning models.

Combines raw transaction attributes with Stage 2 forensic signals:
- Transaction amounts and log-transforms
- Account historical behavioral statistics
- Relative Size Factor (RSF) features
- Duplicate indicators
- Velocity burst indicators
- Digit / Benford indicators
- Typology flags (Structuring, Round amounts)
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.forensic.benford import get_first_digit
from src.config import STRUCTURING_THRESHOLD


def extract_ml_features(
    df: pd.DataFrame,
    enriched_forensic_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract and engineer numerical features suitable for ML anomaly detection and classification.

    Args:
        df: Base transaction DataFrame
        enriched_forensic_df: Optional DataFrame already containing Stage 2 forensic columns

    Returns:
        (feature_matrix, feature_names)
    """
    base_df = enriched_forensic_df.copy() if enriched_forensic_df is not None else df.copy()
    features_df = pd.DataFrame(index=base_df.index)

    # 1. Transaction Amount Features
    amounts = pd.to_numeric(base_df["amount"], errors="coerce").fillna(0.0)
    features_df["amount"] = amounts
    features_df["log_amount"] = np.log1p(np.maximum(0.0, amounts))

    # 2. Digit-Level Feature (Benford context)
    features_df["first_digit"] = amounts.apply(get_first_digit)
    # Structuring flags often cluster on specific leading digits like 9 (e.g. $9,500)
    features_df["is_digit_nine"] = (features_df["first_digit"] == 9).astype(int)

    # 3. Round Number Indicator (Launderers frequently use round multiples)
    features_df["is_round_1000"] = ((amounts % 1000 == 0) & (amounts > 0)).astype(int)
    features_df["is_round_500"] = ((amounts % 500 == 0) & (amounts > 0)).astype(int)

    # 4. Structuring Heuristic (Amounts near $10,000 regulatory reporting threshold)
    features_df["is_near_threshold"] = (
        (amounts >= STRUCTURING_THRESHOLD * 0.88) & (amounts < STRUCTURING_THRESHOLD)
    ).astype(int)

    # 5. Account-Level Behavioral Baseline Statistics
    if "sender_account" in base_df.columns:
        sender_counts = base_df["sender_account"].map(base_df["sender_account"].value_counts()).fillna(1)
        features_df["sender_tx_frequency"] = sender_counts

        sender_means = base_df.groupby("sender_account")["amount"].transform("mean")
        sender_maxs = base_df.groupby("sender_account")["amount"].transform("max")
        sender_stds = base_df.groupby("sender_account")["amount"].transform("std").fillna(0.0)

        features_df["sender_avg_amount"] = sender_means
        features_df["sender_max_amount"] = sender_maxs
        features_df["amount_to_sender_avg"] = np.where(sender_means > 0, amounts / sender_means, 1.0)
        features_df["amount_zscore_sender"] = np.where(sender_stds > 0, (amounts - sender_means) / sender_stds, 0.0)
    else:
        features_df["sender_tx_frequency"] = 1.0
        features_df["sender_avg_amount"] = amounts
        features_df["sender_max_amount"] = amounts
        features_df["amount_to_sender_avg"] = 1.0
        features_df["amount_zscore_sender"] = 0.0

    if "receiver_account" in base_df.columns:
        recv_counts = base_df["receiver_account"].map(base_df["receiver_account"].value_counts()).fillna(1)
        features_df["receiver_inflow_count"] = recv_counts
    else:
        features_df["receiver_inflow_count"] = 1.0

    # 6. Integrated Forensic Signals from Stage 2 (if present)
    if "rsf_risk_score" in base_df.columns:
        features_df["rsf_risk_score"] = pd.to_numeric(base_df["rsf_risk_score"], errors="coerce").fillna(0.0)
    else:
        features_df["rsf_risk_score"] = 0.0

    if "duplicate_risk_score" in base_df.columns:
        features_df["duplicate_risk_score"] = pd.to_numeric(base_df["duplicate_risk_score"], errors="coerce").fillna(0.0)
    else:
        features_df["duplicate_risk_score"] = 0.0

    if "velocity_risk_score" in base_df.columns:
        features_df["velocity_risk_score"] = pd.to_numeric(base_df["velocity_risk_score"], errors="coerce").fillna(0.0)
    else:
        features_df["velocity_risk_score"] = 0.0

    # Fill any remaining NaNs or Infs
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feature_names = list(features_df.columns)
    return features_df, feature_names
