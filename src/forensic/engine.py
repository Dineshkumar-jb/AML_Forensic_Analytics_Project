"""Forensic Analytics Engine Orchestrator.

Coordinates all Stage 2 forensic analytics modules:
1. Benford's Law (First & Second Digit)
2. Relative Size Factor (RSF)
3. Duplicate Transaction Detection
4. Transaction Velocity & Bursts
5. AML Typology Heuristic Indicators (Structuring, Fan-in/out, Layering)
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from src.forensic.benford import analyze_benford
from src.forensic.rsf import calculate_rsf
from src.forensic.duplicates import detect_duplicates
from src.forensic.velocity import analyze_velocity
from src.forensic.typologies import analyze_all_typologies


def run_forensic_suite(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute complete forensic suite and compile raw findings & 0-100 scores.

    Args:
        df: Normalized transaction DataFrame

    Returns:
        Structured dictionary with test details, enriched DataFrame, and scorecard table.
    """
    if df is None or df.empty:
        return {"is_valid": False, "error": "Input DataFrame is empty."}

    # 1. Benford's Law Analysis
    benford_res = analyze_benford(df)

    # 2. Relative Size Factor (RSF)
    rsf_res = calculate_rsf(df)

    # 3. Duplicate Transaction Detection
    dup_res = detect_duplicates(df)

    # 4. Transaction Velocity Analysis
    velocity_res = analyze_velocity(df)

    # 5. Typology Heuristics
    typology_res = analyze_all_typologies(df)

    # Compile Executive Forensic Scorecard
    scorecard_rows = []

    # Benford entry
    b_score = benford_res.get("overall_risk_score", 0.0) if benford_res.get("is_valid") else 0.0
    b_raw = f"MAD: {benford_res['first_digit'].get('mad', 0.0):.4f} ({benford_res['first_digit'].get('conformity', 'N/A')})" if benford_res.get("is_valid") else "N/A"
    scorecard_rows.append({
        "Forensic Test": "Benford's Law Analysis",
        "Key Metric / Raw Result": b_raw,
        "Detection Target": "Abnormal digit distribution & non-random amounts",
        "Risk Score (0-100)": b_score,
        "Severity": "High" if b_score >= 70 else ("Medium" if b_score >= 40 else "Low"),
    })

    # RSF entry
    rsf_score = rsf_res.get("risk_score", 0.0) if rsf_res.get("is_valid") else 0.0
    rsf_raw = f"Max RSF: {rsf_res.get('max_rsf', 1.0):.1f}x ({rsf_res.get('anomaly_count', 0)} accounts flagged)" if rsf_res.get("is_valid") else "N/A"
    scorecard_rows.append({
        "Forensic Test": "Relative Size Factor (RSF)",
        "Key Metric / Raw Result": rsf_raw,
        "Detection Target": "Transactions exceeding historical baseline ratio",
        "Risk Score (0-100)": rsf_score,
        "Severity": "High" if rsf_score >= 70 else ("Medium" if rsf_score >= 40 else "Low"),
    })

    # Duplicates entry
    dup_score = dup_res.get("risk_score", 0.0) if dup_res.get("is_valid") else 0.0
    dup_raw = f"{dup_res.get('duplicate_count', 0)} transactions (${dup_res.get('duplicate_volume', 0):,.2f})" if dup_res.get("is_valid") else "N/A"
    scorecard_rows.append({
        "Forensic Test": "Duplicate Detection",
        "Key Metric / Raw Result": dup_raw,
        "Detection Target": "Repeated identical amounts across parties",
        "Risk Score (0-100)": dup_score,
        "Severity": "High" if dup_score >= 70 else ("Medium" if dup_score >= 40 else "Low"),
    })

    # Velocity entry
    vel_score = velocity_res.get("risk_score", 0.0) if velocity_res.get("is_valid") else 0.0
    vel_raw = f"{velocity_res.get('burst_account_count', 0)} accounts burst (Peak: {velocity_res.get('max_burst_count', 0)} txs/window)" if velocity_res.get("is_valid") else "N/A"
    scorecard_rows.append({
        "Forensic Test": "Transaction Velocity",
        "Key Metric / Raw Result": vel_raw,
        "Detection Target": "High-frequency rolling window spikes & bursts",
        "Risk Score (0-100)": vel_score,
        "Severity": "High" if vel_score >= 70 else ("Medium" if vel_score >= 40 else "Low"),
    })

    # Typologies entry
    typ_score = typology_res.get("overall_risk_score", 0.0)
    struct_c = typology_res["structuring"].get("detected_count", 0)
    fan_in_c = typology_res["fan_in"].get("hub_count", 0)
    fan_out_c = typology_res["fan_out"].get("source_count", 0)
    layer_c = typology_res["layering"].get("detected_transit_accounts", 0)
    typ_raw = f"Structuring: {struct_c} | Fan-in: {fan_in_c} | Fan-out: {fan_out_c} | Layering: {layer_c}"
    scorecard_rows.append({
        "Forensic Test": "Typology Indicators",
        "Key Metric / Raw Result": typ_raw,
        "Detection Target": "Structuring, Mules, Fan-in/out & Pass-throughs",
        "Risk Score (0-100)": typ_score,
        "Severity": "High" if typ_score >= 70 else ("Medium" if typ_score >= 40 else "Low"),
    })

    scorecard_df = pd.DataFrame(scorecard_rows)

    # Average Forensic Pillar Score (strictly within Forensic tests)
    all_scores = [b_score, rsf_score, dup_score, vel_score, typ_score]
    aggregate_forensic_score = round(float(np.mean(all_scores)), 1)

    # Enriched DataFrame with transaction-level indicators
    enriched_df = df.copy()

    # Add RSF transaction score if available
    if rsf_res.get("is_valid") and "tx_rsf_scores" in rsf_res:
        enriched_df["rsf_risk_score"] = rsf_res["tx_rsf_scores"]

    # Add duplicate score if available
    if dup_res.get("is_valid") and "tx_duplicate_scores" in dup_res:
        enriched_df["duplicate_risk_score"] = dup_res["tx_duplicate_scores"]

    # Add velocity score if available
    if velocity_res.get("is_valid") and "tx_velocity_scores" in velocity_res:
        enriched_df["velocity_risk_score"] = velocity_res["tx_velocity_scores"]

    # Add structuring flag
    if typology_res["structuring"].get("is_valid"):
        lower_b = typology_res["structuring"]["lower_bound"]
        upper_b = typology_res["structuring"]["upper_bound"]
        enriched_df["is_structuring"] = (enriched_df["amount"] >= lower_b) & (enriched_df["amount"] < upper_b)

    return {
        "is_valid": True,
        "benford": benford_res,
        "rsf": rsf_res,
        "duplicates": dup_res,
        "velocity": velocity_res,
        "typologies": typology_res,
        "scorecard": scorecard_df,
        "aggregate_forensic_score": aggregate_forensic_score,
        "enriched_df": enriched_df,
    }
