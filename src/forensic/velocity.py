"""Transaction velocity and burst analysis forensic module for AML monitoring.

Analyzes high-frequency activity, rolling time-window transaction bursts,
and sudden spikes in account velocity.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def analyze_velocity(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    account_col: str = "sender_account",
    amount_col: str = "amount",
    window_hours: float = 2.0,
    burst_threshold: int = 4,
) -> Dict[str, Any]:
    """Calculate transaction velocity, burst events, and account frequency metrics.

    Args:
        df: Transaction DataFrame
        timestamp_col: Column containing timestamp
        account_col: Originating account column
        amount_col: Transaction amount column
        window_hours: Rolling time window in hours to detect bursts
        burst_threshold: Minimum transactions within the window to flag as a burst

    Returns:
        Dictionary containing account-level velocity stats, burst events, and 0-100 risk score.
    """
    required = [timestamp_col, account_col, amount_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"is_valid": False, "error": f"Missing required columns for velocity: {missing}"}

    work_df = df.copy()
    work_df["parsed_time"] = pd.to_datetime(work_df[timestamp_col], errors="coerce")
    work_df = work_df.dropna(subset=["parsed_time"]).sort_values("parsed_time").reset_index(drop=True)

    if work_df.empty:
        return {"is_valid": False, "error": "No valid timestamps found for velocity analysis."}

    window_timedelta = pd.Timedelta(hours=window_hours)

    burst_events: List[Dict[str, Any]] = []
    account_burst_counts: Dict[str, int] = {}
    tx_velocity_scores = pd.Series(0.0, index=df.index)

    # Analyze rolling velocity per account
    for acc, group in work_df.groupby(account_col):
        n_tx = len(group)
        if n_tx < 2:
            continue

        group_times = group["parsed_time"].values
        group_amts = group[amount_col].values
        orig_indices = group.index.values

        max_in_window = 1
        max_vol_window = 0.0

        for i in range(n_tx):
            t_start = group_times[i]
            t_end = t_start + np.timedelta64(int(window_hours * 3600), "s")

            # Mask of items in this window
            window_mask = (group_times >= t_start) & (group_times <= t_end)
            count_in_w = int(np.sum(window_mask))
            vol_in_w = float(np.sum(group_amts[window_mask]))

            if count_in_w > max_in_window:
                max_in_window = count_in_w
                max_vol_window = vol_in_w

            if count_in_w >= burst_threshold:
                # Record burst
                burst_indices = orig_indices[window_mask]
                tx_velocity_scores.iloc[burst_indices] = np.maximum(
                    tx_velocity_scores.iloc[burst_indices],
                    min(100.0, (count_in_w / burst_threshold) * 60.0),
                )

        if max_in_window >= burst_threshold:
            account_burst_counts[acc] = max_in_window
            burst_events.append({
                "account": acc,
                "total_transactions": n_tx,
                "max_burst_count": max_in_window,
                "window_hours": window_hours,
                "burst_volume": round(max_vol_window, 2),
                "is_burst": True,
            })

    burst_df = pd.DataFrame(burst_events)
    if not burst_df.empty:
        burst_df = burst_df.sort_values(by="max_burst_count", ascending=False).reset_index(drop=True)

    burst_count = len(burst_df)
    max_burst = int(burst_df["max_burst_count"].max()) if not burst_df.empty else 0

    # Velocity Risk Score (0-100)
    # Scaled by maximum burst intensity and number of accounts bursting
    if burst_count > 0:
        burst_intensity = min(60.0, ((max_burst - burst_threshold + 1) / 5.0) * 60.0)
        account_spread = min(40.0, (burst_count / 5.0) * 40.0)
        velocity_risk_score = round(min(100.0, burst_intensity + account_spread), 1)
    else:
        velocity_risk_score = 0.0

    return {
        "is_valid": True,
        "burst_account_count": burst_count,
        "max_burst_count": max_burst,
        "burst_events_table": burst_df,
        "risk_score": velocity_risk_score,
        "tx_velocity_scores": tx_velocity_scores.round(1),
        "window_hours": window_hours,
        "burst_threshold": burst_threshold,
    }
