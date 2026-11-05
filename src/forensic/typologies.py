"""Forensic heuristic typology indicators for AML transaction analysis.

Implements detection for:
1. Structuring / Smurfing (near-threshold amounts)
2. Fan-in (many senders -> one receiver)
3. Fan-out (one sender -> many receivers)
4. Layering / Rapid Pass-through (flow-through transit accounts)
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.config import STRUCTURING_THRESHOLD


def detect_structuring(
    df: pd.DataFrame,
    amount_col: str = "amount",
    sender_col: str = "sender_account",
    threshold: float = STRUCTURING_THRESHOLD,
    lower_pct: float = 0.88,
) -> Dict[str, Any]:
    """Detect structuring (transactions placed just below reporting threshold, e.g. $8,800 to $9,999.99)."""
    if amount_col not in df.columns:
        return {"is_valid": False, "error": f"Column '{amount_col}' missing."}

    lower_bound = threshold * lower_pct
    upper_bound = threshold

    structured_mask = (df[amount_col] >= lower_bound) & (df[amount_col] < upper_bound)
    structured_df = df[structured_mask].copy()

    total_count = len(structured_df)
    total_volume = float(structured_df[amount_col].sum()) if total_count > 0 else 0.0
    unique_senders = list(structured_df[sender_col].unique()) if sender_col in df.columns else []

    # Risk score: scaled by proportion of dataset and repeat behavior
    structuring_ratio = total_count / max(1, len(df))
    # 0 to 100 scale: ratio of 5% or more reaches high risk
    risk_score = min(100.0, (structuring_ratio / 0.05) * 100.0) if total_count > 0 else 0.0

    return {
        "is_valid": True,
        "indicator_name": "Structuring / Smurfing",
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "detected_count": total_count,
        "total_volume": round(total_volume, 2),
        "unique_senders": unique_senders,
        "flagged_transactions": structured_df,
        "risk_score": round(risk_score, 1),
    }


def detect_fan_in(
    df: pd.DataFrame,
    sender_col: str = "sender_account",
    receiver_col: str = "receiver_account",
    amount_col: str = "amount",
    min_senders: int = 3,
) -> Dict[str, Any]:
    """Detect fan-in patterns (multiple distinct senders paying into a single beneficiary)."""
    if sender_col not in df.columns or receiver_col not in df.columns:
        return {"is_valid": False, "error": f"Columns '{sender_col}' or '{receiver_col}' missing."}

    fan_in_records = []
    for receiver, group in df.groupby(receiver_col):
        senders = group[sender_col].unique()
        if len(senders) >= min_senders:
            fan_in_records.append({
                "receiver_account": receiver,
                "distinct_senders_count": len(senders),
                "total_received_txs": len(group),
                "total_consolidated_volume": round(float(group[amount_col].sum()), 2),
                "senders_list": list(senders),
            })

    fan_in_df = pd.DataFrame(fan_in_records)
    if not fan_in_df.empty:
        fan_in_df = fan_in_df.sort_values(by="distinct_senders_count", ascending=False).reset_index(drop=True)

    hub_count = len(fan_in_df)
    max_senders = int(fan_in_df["distinct_senders_count"].max()) if not fan_in_df.empty else 0

    # Risk score: scaled by highest sender count and number of hubs
    if hub_count > 0:
        intensity = min(60.0, ((max_senders - min_senders + 1) / 8.0) * 60.0)
        spread = min(40.0, (hub_count / 3.0) * 40.0)
        risk_score = round(min(100.0, intensity + spread), 1)
    else:
        risk_score = 0.0

    return {
        "is_valid": True,
        "indicator_name": "Fan-in (Mule Consolidation)",
        "hub_count": hub_count,
        "max_senders_to_single_hub": max_senders,
        "hubs_table": fan_in_df,
        "risk_score": risk_score,
        "min_senders_threshold": min_senders,
    }


def detect_fan_out(
    df: pd.DataFrame,
    sender_col: str = "sender_account",
    receiver_col: str = "receiver_account",
    amount_col: str = "amount",
    min_receivers: int = 3,
) -> Dict[str, Any]:
    """Detect fan-out patterns (a single source dispersing funds to multiple distinct beneficiaries)."""
    if sender_col not in df.columns or receiver_col not in df.columns:
        return {"is_valid": False, "error": f"Columns '{sender_col}' or '{receiver_col}' missing."}

    fan_out_records = []
    for sender, group in df.groupby(sender_col):
        receivers = group[receiver_col].unique()
        if len(receivers) >= min_receivers:
            fan_out_records.append({
                "sender_account": sender,
                "distinct_receivers_count": len(receivers),
                "total_dispersed_txs": len(group),
                "total_dispersed_volume": round(float(group[amount_col].sum()), 2),
                "receivers_list": list(receivers),
            })

    fan_out_df = pd.DataFrame(fan_out_records)
    if not fan_out_df.empty:
        fan_out_df = fan_out_df.sort_values(by="distinct_receivers_count", ascending=False).reset_index(drop=True)

    source_count = len(fan_out_df)
    max_receivers = int(fan_out_df["distinct_receivers_count"].max()) if not fan_out_df.empty else 0

    if source_count > 0:
        intensity = min(60.0, ((max_receivers - min_receivers + 1) / 8.0) * 60.0)
        spread = min(40.0, (source_count / 3.0) * 40.0)
        risk_score = round(min(100.0, intensity + spread), 1)
    else:
        risk_score = 0.0

    return {
        "is_valid": True,
        "indicator_name": "Fan-out (Fund Dispersal)",
        "source_count": source_count,
        "max_receivers_from_single_source": max_receivers,
        "sources_table": fan_out_df,
        "risk_score": risk_score,
        "min_receivers_threshold": min_receivers,
    }


def detect_layering_passthrough(
    df: pd.DataFrame,
    sender_col: str = "sender_account",
    receiver_col: str = "receiver_account",
    amount_col: str = "amount",
    timestamp_col: str = "timestamp",
    tolerance_ratio: float = 0.25,
) -> Dict[str, Any]:
    """Detect rapid pass-through transit accounts (accounts that receive funds and shortly evacuate comparable amounts)."""
    required = [sender_col, receiver_col, amount_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"is_valid": False, "error": f"Missing columns: {missing}"}

    all_senders = set(df[sender_col].dropna())
    all_receivers = set(df[receiver_col].dropna())
    # Candidate transit accounts that both send and receive
    transit_candidates = all_senders.intersection(all_receivers)

    layering_records = []
    for acc in transit_candidates:
        inflow = float(df[df[receiver_col] == acc][amount_col].sum())
        outflow = float(df[df[sender_col] == acc][amount_col].sum())

        if inflow > 0 and outflow > 0:
            diff = abs(inflow - outflow)
            ratio = diff / max(inflow, outflow)
            # If net flow through is balanced within tolerance
            if ratio <= tolerance_ratio:
                layering_records.append({
                    "transit_account": acc,
                    "total_inflow": round(inflow, 2),
                    "total_outflow": round(outflow, 2),
                    "net_balance_retained": round(diff, 2),
                    "flow_retention_ratio": round(ratio * 100, 2),
                })

    layering_df = pd.DataFrame(layering_records)
    if not layering_df.empty:
        layering_df = layering_df.sort_values(by="total_inflow", ascending=False).reset_index(drop=True)

    layering_count = len(layering_df)
    risk_score = min(100.0, (layering_count / 2.0) * 100.0) if layering_count > 0 else 0.0

    return {
        "is_valid": True,
        "indicator_name": "Layering (Rapid Pass-through)",
        "detected_transit_accounts": layering_count,
        "transit_table": layering_df,
        "risk_score": round(risk_score, 1),
    }


def analyze_all_typologies(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute all forensic typology indicators."""
    structuring_res = detect_structuring(df)
    fan_in_res = detect_fan_in(df)
    fan_out_res = detect_fan_out(df)
    layering_res = detect_layering_passthrough(df)

    scores = [
        structuring_res.get("risk_score", 0.0),
        fan_in_res.get("risk_score", 0.0),
        fan_out_res.get("risk_score", 0.0),
        layering_res.get("risk_score", 0.0),
    ]
    # Typology indicator score is the weighted or max-weighted combination
    overall_typology_score = round(max(scores) * 0.6 + (sum(scores) / len(scores)) * 0.4, 1)

    return {
        "structuring": structuring_res,
        "fan_in": fan_in_res,
        "fan_out": fan_out_res,
        "layering": layering_res,
        "overall_risk_score": min(100.0, overall_typology_score),
    }
