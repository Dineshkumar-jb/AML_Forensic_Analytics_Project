"""Duplicate transaction forensic detection module for AML monitoring.

Identifies repeated or identical transactions across accounts, amounts,
and temporal intervals (common in split testing, layering, and structuring).
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def detect_duplicates(
    df: pd.DataFrame,
    amount_col: str = "amount",
    sender_col: str = "sender_account",
    receiver_col: str = "receiver_account",
    timestamp_col: str = "timestamp",
    window_hours: float = 48.0,
) -> Dict[str, Any]:
    """Detect duplicate transactions across sender, receiver, and amount.

    Args:
        df: Transaction DataFrame
        amount_col: Column for transaction amount
        sender_col: Column for sender account ID
        receiver_col: Column for receiver account ID
        timestamp_col: Column for timestamp
        window_hours: Maximum hours between occurrences to flag temporal proximity

    Returns:
        Dictionary containing duplicate clusters, summary counts, affected accounts,
        and normalized 0-100 duplicate risk score.
    """
    required = [amount_col, sender_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"is_valid": False, "error": f"Missing columns: {missing}"}

    work_df = df.copy()
    work_df["orig_index"] = work_df.index

    # Check pair duplicates: same sender, same receiver, same amount
    group_cols = [sender_col, amount_col]
    if receiver_col in work_df.columns:
        group_cols.append(receiver_col)

    # Find duplicates
    dup_mask = work_df.duplicated(subset=group_cols, keep=False)
    dup_df = work_df[dup_mask].copy()

    if dup_df.empty:
        return {
            "is_valid": True,
            "duplicate_count": 0,
            "duplicate_volume": 0.0,
            "affected_accounts": [],
            "duplicate_clusters": pd.DataFrame(),
            "risk_score": 0.0,
            "tx_duplicate_scores": pd.Series(0.0, index=df.index),
        }

    # Cluster duplicates together
    cluster_records = []
    cluster_id = 0

    for keys, group in dup_df.groupby(group_cols):
        cluster_id += 1
        count = len(group)
        amt = float(keys[1] if isinstance(keys, tuple) else group[amount_col].iloc[0])
        total_vol = amt * count

        # Time delta check if timestamp available
        time_spread_hours = None
        rapid_flag = False
        if timestamp_col in group.columns and pd.api.types.is_datetime64_any_dtype(group[timestamp_col]):
            ts_sorted = group[timestamp_col].sort_values()
            diff_hours = (ts_sorted.max() - ts_sorted.min()).total_seconds() / 3600.0
            time_spread_hours = round(diff_hours, 2)
            if diff_hours <= window_hours:
                rapid_flag = True

        cluster_records.append({
            "cluster_id": f"DUP_{cluster_id:03d}",
            "sender_account": keys[0] if isinstance(keys, tuple) else group[sender_col].iloc[0],
            "receiver_account": keys[2] if (isinstance(keys, tuple) and len(keys) > 2) else "N/A",
            "identical_amount": amt,
            "repeat_count": count,
            "total_cluster_volume": round(total_vol, 2),
            "time_spread_hours": time_spread_hours if time_spread_hours is not None else "N/A",
            "rapid_repetition": rapid_flag,
        })

    cluster_df = pd.DataFrame(cluster_records).sort_values(by="repeat_count", ascending=False).reset_index(drop=True)

    total_dup_txs = len(dup_df)
    total_dup_volume = float(dup_df[amount_col].sum())
    affected_senders = list(dup_df[sender_col].unique())

    # Risk Score calculation (0 - 100):
    # Considers: (1) Percentage of transactions that are duplicates,
    # (2) Maximum repeat count in a single cluster,
    # (3) Rapid repetition flag.
    dup_ratio = total_dup_txs / max(1, len(df))
    max_repeat = int(cluster_df["repeat_count"].max()) if not cluster_df.empty else 0
    has_rapid = bool(cluster_df["rapid_repetition"].any()) if not cluster_df.empty else False

    # Score components
    ratio_score = min(50.0, (dup_ratio / 0.15) * 50.0)
    repeat_score = min(30.0, (max_repeat / 6.0) * 30.0)
    rapid_score = 20.0 if has_rapid else 0.0

    overall_risk_score = round(min(100.0, ratio_score + repeat_score + rapid_score), 1)

    # Assign transaction-level risk scores
    tx_scores = pd.Series(0.0, index=df.index)
    tx_scores.loc[dup_df["orig_index"]] = 75.0
    if has_rapid:
        rapid_indices = []
        for keys, group in dup_df.groupby(group_cols):
            if timestamp_col in group.columns and pd.api.types.is_datetime64_any_dtype(group[timestamp_col]):
                diff_hours = (group[timestamp_col].max() - group[timestamp_col].min()).total_seconds() / 3600.0
                if diff_hours <= window_hours:
                    rapid_indices.extend(group["orig_index"].tolist())
        tx_scores.loc[rapid_indices] = 100.0

    return {
        "is_valid": True,
        "duplicate_count": total_dup_txs,
        "duplicate_volume": round(total_dup_volume, 2),
        "affected_accounts": affected_senders,
        "duplicate_clusters": cluster_df,
        "risk_score": overall_risk_score,
        "tx_duplicate_scores": tx_scores,
    }
