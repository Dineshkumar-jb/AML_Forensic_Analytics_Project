"""Relative Size Factor (RSF) forensic module for AML transaction analysis.

Identifies transactions that are disproportionately large relative to an account's
historical activity or second-largest baseline:
RSF = Largest Transaction / Second Largest Transaction
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def calculate_rsf(
    df: pd.DataFrame,
    account_col: str = "sender_account",
    amount_col: str = "amount",
    threshold: float = 5.0,
) -> Dict[str, Any]:
    """Calculate Relative Size Factor (RSF) per account and per transaction.

    Args:
        df: DataFrame containing transactions
        account_col: Account identifier column
        amount_col: Numeric amount column
        threshold: RSF ratio above which an account is flagged as anomalous (default 5.0)

    Returns:
        Dictionary containing account-level metrics, flagged anomalies, and a 0-100 risk score.
    """
    if account_col not in df.columns or amount_col not in df.columns:
        return {
            "is_valid": False,
            "error": f"Required columns '{account_col}' or '{amount_col}' missing.",
        }

    valid_df = df[[account_col, amount_col]].dropna().copy()
    valid_df[amount_col] = pd.to_numeric(valid_df[amount_col], errors="coerce").fillna(0.0)

    account_stats = []

    for acc, group in valid_df.groupby(account_col):
        amounts = group[amount_col].sort_values(ascending=False).values
        n_tx = len(amounts)
        max_amt = float(amounts[0])

        if n_tx >= 2:
            second_max = float(amounts[1])
            # Guard against division by near-zero
            if second_max > 0:
                rsf = max_amt / second_max
            else:
                rsf = 1.0 if max_amt == 0 else 10.0
        else:
            # Single transaction: compare to median of whole population or baseline
            second_max = max_amt
            rsf = 1.0

        median_amt = float(np.median(amounts))
        mean_amt = float(np.mean(amounts))

        account_stats.append({
            "account": acc,
            "tx_count": n_tx,
            "max_amount": round(max_amt, 2),
            "second_max_amount": round(second_max, 2),
            "mean_amount": round(mean_amt, 2),
            "median_amount": round(median_amt, 2),
            "rsf_ratio": round(rsf, 2),
            "is_anomalous": rsf >= threshold and n_tx >= 2,
        })

    rsf_df = pd.DataFrame(account_stats)
    if rsf_df.empty:
        return {"is_valid": False, "error": "No valid accounts found."}

    rsf_df = rsf_df.sort_values(by="rsf_ratio", ascending=False).reset_index(drop=True)
    anomalous_accounts = rsf_df[rsf_df["is_anomalous"]]
    anomaly_count = len(anomalous_accounts)

    # Calculate overall dataset RSF Risk Score (0-100)
    # Based on: (1) Max RSF ratio, (2) Proportion of accounts exceeding threshold
    max_rsf = float(rsf_df["rsf_ratio"].max()) if not rsf_df.empty else 1.0
    anomaly_ratio = anomaly_count / max(1, len(rsf_df))

    # Log-scaled severity of the maximum spike + prevalence of anomalies
    severity_component = min(70.0, (np.log1p(max(0.0, max_rsf - 1.0)) / np.log1p(100.0)) * 70.0)
    prevalence_component = min(30.0, (anomaly_ratio / 0.10) * 30.0)
    rsf_risk_score = round(min(100.0, severity_component + prevalence_component), 1)

    # Transaction-level mapping
    account_rsf_map = rsf_df.set_index("account")["rsf_ratio"].to_dict()
    tx_rsf_scores = df[account_col].map(account_rsf_map).fillna(1.0)

    # Normalize transaction scores: 1.0 -> 0, 10.0+ -> 100
    tx_normalized = (np.clip(tx_rsf_scores - 1.0, 0, 15.0) / 15.0) * 100.0

    return {
        "is_valid": True,
        "account_table": rsf_df,
        "anomalous_accounts": anomalous_accounts,
        "anomaly_count": anomaly_count,
        "max_rsf": round(max_rsf, 2),
        "mean_rsf": round(float(rsf_df["rsf_ratio"].mean()), 2),
        "risk_score": rsf_risk_score,
        "tx_rsf_scores": tx_normalized.round(1),
        "threshold": threshold,
    }
