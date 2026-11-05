"""Composite Risk Scoring Engine for AML Transaction Monitoring.

Stage 4: Synthesizes deterministic forensic indicators with unsupervised and supervised
machine learning signals into a transparent, normalized 0-100 composite risk score.

Risk Levels:
- Low:      0 - 24
- Medium:  25 - 49
- High:    50 - 74
- Critical: 75 - 100

Features automatic proportional weight redistribution when signals (e.g., XGBoost)
are unavailable, ensuring zero fabrication of missing model outputs.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import RISK_TIERS_STAGE4, DEFAULT_COMPOSITE_WEIGHTS


def classify_risk_level(score: float) -> str:
    """Classify risk score into 4-tier standardized AML risk level.

    0 - 24   -> Low
    25 - 49  -> Medium
    50 - 74  -> High
    75 - 100 -> Critical
    """
    s = round(float(np.clip(score, 0.0, 100.0)), 1)
    if s >= 75.0:
        return "Critical"
    elif s >= 50.0:
        return "High"
    elif s >= 25.0:
        return "Medium"
    else:
        return "Low"


def compute_transaction_forensic_score(
    row: pd.Series,
) -> Tuple[float, List[str]]:
    """Compute normalized forensic risk score (0-100) and identify active forensic drivers.

    Combines:
    - RSF risk score
    - Duplicate risk score
    - Velocity risk score
    - Structuring indicator
    """
    factors: List[str] = []

    rsf_s = float(row.get("rsf_risk_score", 0.0)) if pd.notnull(row.get("rsf_risk_score")) else 0.0
    dup_s = float(row.get("duplicate_risk_score", 0.0)) if pd.notnull(row.get("duplicate_risk_score")) else 0.0
    vel_s = float(row.get("velocity_risk_score", 0.0)) if pd.notnull(row.get("velocity_risk_score")) else 0.0
    is_struct = bool(row.get("is_structuring", False)) if pd.notnull(row.get("is_structuring")) else False

    struct_s = 85.0 if is_struct else 0.0

    if rsf_s >= 40.0:
        factors.append(f"RSF Spike ({rsf_s:.0f})")
    if dup_s >= 40.0:
        factors.append(f"Duplicate Repeat ({dup_s:.0f})")
    if vel_s >= 40.0:
        factors.append(f"Velocity Burst ({vel_s:.0f})")
    if is_struct:
        factors.append("Structuring (Near-CTR Threshold)")

    # Max-heavy combination: strong red flags elevate the score while accounting for multiple hits
    component_scores = [rsf_s, dup_s, vel_s, struct_s]
    max_c = max(component_scores)
    mean_c = sum(component_scores) / len(component_scores)

    # 65% weight on the highest forensic red flag, 35% on aggregate breadth
    raw_forensic = (0.65 * max_c) + (0.35 * mean_c)
    forensic_score = float(np.clip(raw_forensic, 0.0, 100.0))

    return round(forensic_score, 1), factors


def generate_risk_explanation(
    composite_score: float,
    risk_level: str,
    forensic_score: float,
    iso_score: float,
    xgb_score: Optional[float],
    factors: List[str],
) -> str:
    """Generate human-readable audit justification for the assigned composite score."""
    factors_str = ", ".join(factors) if factors else "Routine activity baseline"

    if risk_level == "Critical":
        reason = (
            f"Critical Alert ({composite_score:.1f}/100): Transaction exhibits severe multi-signal anomalies. "
            f"Key drivers: {factors_str}. Immediate escalation and SAR filing recommended."
        )
    elif risk_level == "High":
        reason = (
            f"High Risk ({composite_score:.1f}/100): Elevated risk profile driven by {factors_str}. "
            f"Requires secondary compliance officer review."
        )
    elif risk_level == "Medium":
        reason = (
            f"Medium Risk ({composite_score:.1f}/100): Moderate anomalies observed ({factors_str}). "
            f"Place account on heightened monitoring."
        )
    else:
        reason = (
            f"Low Risk ({composite_score:.1f}/100): Activity conforms to normal baseline patterns. "
            f"No significant forensic or ML anomaly markers detected."
        )

    return reason


def calculate_composite_risk(
    df: pd.DataFrame,
    forensic_results: Optional[Dict[str, Any]] = None,
    ml_results: Optional[Dict[str, Any]] = None,
    custom_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Calculate dynamic Composite Risk Score for all transactions.

    Handles missing components through proportional weight redistribution:
    - If XGBoost is unavailable, its weight (default 30%) is proportionally redistributed
      between Forensics (57.1%) and Isolation Forest (42.9%).
    - Zero fake scores or invented probabilities are generated.

    Args:
        df: Transaction DataFrame (dynamically evaluated)
        forensic_results: Output dictionary from Stage 2 run_forensic_suite
        ml_results: Output dictionary from Stage 3 run_ml_pipeline
        custom_weights: Optional user override for base weights

    Returns:
        Structured dictionary containing scored DataFrame, component contributions,
        risk distribution summary, and top-risk transactions table.
    """
    if df is None or df.empty:
        return {"is_valid": False, "error": "Input DataFrame is empty."}

    # Ensure transaction_id exists
    if "transaction_id" not in df.columns:
        return {
            "is_valid": False,
            "error": "Missing required 'transaction_id' column for risk scoring.",
        }

    working_df = df.copy()

    # Merge enriched forensic columns if available
    if forensic_results and forensic_results.get("is_valid") and "enriched_df" in forensic_results:
        f_enriched = forensic_results["enriched_df"]
        for col in ["rsf_risk_score", "duplicate_risk_score", "velocity_risk_score", "is_structuring"]:
            if col in f_enriched.columns and col not in working_df.columns:
                working_df[col] = f_enriched[col]

    # Merge enriched ML columns if available
    if ml_results and ml_results.get("is_valid") and "enriched_df" in ml_results:
        m_enriched = ml_results["enriched_df"]
        for col in ["iso_forest_score", "iso_forest_anomaly", "xgb_risk_score", "xgb_prediction"]:
            if col in m_enriched.columns and col not in working_df.columns:
                working_df[col] = m_enriched[col]

    # 1. Determine Signal Availability
    available_signals: List[str] = []
    unavailable_signals: List[str] = []

    # Forensic availability
    has_forensics = forensic_results is not None and forensic_results.get("is_valid", False)
    if has_forensics:
        available_signals.append("forensic")
    else:
        unavailable_signals.append("forensic")

    # Isolation Forest availability
    has_iso = (
        ml_results is not None
        and ml_results.get("is_valid", False)
        and ml_results.get("isolation_forest", {}).get("is_valid", False)
    )
    if has_iso:
        available_signals.append("isolation_forest")
    else:
        unavailable_signals.append("isolation_forest")

    # XGBoost availability: strictly available ONLY when valid trained model and genuine labels exist
    has_xgb = (
        ml_results is not None
        and ml_results.get("is_valid", False)
        and ml_results.get("has_labels", False)
        and ml_results.get("xgboost", {}).get("is_trained", False)
    )
    if has_xgb:
        available_signals.append("xgboost")
    else:
        unavailable_signals.append("xgboost")

    # 2. Base Weights and Proportional Redistribution
    base_weights = custom_weights or DEFAULT_COMPOSITE_WEIGHTS.copy()

    # Calculate sum of available base weights
    available_base_sum = sum(base_weights.get(sig, 0.0) for sig in available_signals)

    if available_base_sum <= 0:
        return {
            "is_valid": False,
            "error": "No valid analytical signals available to calculate composite risk.",
        }

    # Proportional weight redistribution formula:
    # effective_weight_k = base_weight_k / sum(available_base_weights)
    effective_weights: Dict[str, float] = {}
    for sig in ["forensic", "isolation_forest", "xgboost"]:
        if sig in available_signals:
            effective_weights[sig] = round(base_weights[sig] / available_base_sum, 4)
        else:
            effective_weights[sig] = 0.0

    # Ensure weights sum to exactly 1.0 (guarding floating point rounding)
    active_keys = [k for k in effective_weights if k in available_signals]
    if active_keys:
        tot_eff = sum(effective_weights[k] for k in active_keys)
        diff = 1.0 - tot_eff
        effective_weights[active_keys[0]] = round(effective_weights[active_keys[0]] + diff, 4)

    # 3. Compute Per-Transaction Scores
    n_rows = len(working_df)
    forensic_scores = np.zeros(n_rows)
    all_factors: List[List[str]] = []

    # Compute forensic scores and drivers
    for idx in range(n_rows):
        row = working_df.iloc[idx]
        if has_forensics:
            f_score, factors = compute_transaction_forensic_score(row)
        else:
            f_score, factors = 0.0, []
        forensic_scores[idx] = f_score
        all_factors.append(factors)

    # Isolation Forest scores
    if has_iso and "iso_forest_score" in working_df.columns:
        iso_scores = pd.to_numeric(working_df["iso_forest_score"], errors="coerce").fillna(0.0).values
    else:
        iso_scores = np.zeros(n_rows)

    # XGBoost scores
    if has_xgb and "xgb_risk_score" in working_df.columns:
        xgb_scores = pd.to_numeric(working_df["xgb_risk_score"], errors="coerce").fillna(0.0).values
    else:
        xgb_scores = None

    # Calculate Composite Score for each row
    composite_scores = np.zeros(n_rows)
    risk_levels: List[str] = []
    explanations: List[str] = []
    main_factors_str: List[str] = []

    for idx in range(n_rows):
        f_val = float(forensic_scores[idx])
        iso_val = float(iso_scores[idx])
        xgb_val = float(xgb_scores[idx]) if xgb_scores is not None else None

        # Weighted combination
        c_score = 0.0
        if has_forensics:
            c_score += effective_weights["forensic"] * f_val
        if has_iso:
            c_score += effective_weights["isolation_forest"] * iso_val
        if has_xgb and xgb_val is not None:
            c_score += effective_weights["xgboost"] * xgb_val

        # Clip strictly to 0.0 - 100.0
        c_score = float(np.clip(round(c_score, 1), 0.0, 100.0))
        composite_scores[idx] = c_score

        level = classify_risk_level(c_score)
        risk_levels.append(level)

        # Factor aggregation
        row_factors = list(all_factors[idx])
        if has_iso and iso_val >= 60.0:
            row_factors.append(f"IsoForest Outlier ({iso_val:.0f})")
        if has_xgb and xgb_val is not None and xgb_val >= 60.0:
            row_factors.append(f"XGBoost Typology ({xgb_val:.0f})")

        factors_display = ", ".join(row_factors) if row_factors else "None (Routine)"
        main_factors_str.append(factors_display)

        # Natural language explanation
        exp = generate_risk_explanation(
            composite_score=c_score,
            risk_level=level,
            forensic_score=f_val,
            iso_score=iso_val,
            xgb_score=xgb_val,
            factors=row_factors,
        )
        explanations.append(exp)

    # 4. Construct Output Scored DataFrame
    scored_df = working_df.copy()
    scored_df["composite_risk_score"] = composite_scores
    scored_df["risk_level"] = risk_levels
    scored_df["forensic_risk_score"] = np.round(forensic_scores, 1)
    scored_df["isolation_forest_risk_score"] = np.round(iso_scores, 1)

    if has_xgb and xgb_scores is not None:
        scored_df["xgboost_risk_score"] = np.round(xgb_scores, 1)
    else:
        # Explicit None / NaN to avoid fake data
        scored_df["xgboost_risk_score"] = np.nan

    scored_df["contributing_risk_factors"] = main_factors_str
    scored_df["risk_explanation"] = explanations

    # 5. Summary Statistics
    counts_by_level = {
        "Low": int(np.sum([r == "Low" for r in risk_levels])),
        "Medium": int(np.sum([r == "Medium" for r in risk_levels])),
        "High": int(np.sum([r == "High" for r in risk_levels])),
        "Critical": int(np.sum([r == "Critical" for r in risk_levels])),
    }

    pcts_by_level = {
        k: round(v / n_rows * 100.0, 1) if n_rows > 0 else 0.0
        for k, v in counts_by_level.items()
    }

    # Top Risk Table
    top_cols = [
        "transaction_id",
        "amount",
        "sender_account",
        "receiver_account",
        "composite_risk_score",
        "risk_level",
        "contributing_risk_factors",
        "forensic_risk_score",
        "isolation_forest_risk_score",
    ]
    if has_xgb:
        top_cols.append("xgboost_risk_score")

    available_top_cols = [c for c in top_cols if c in scored_df.columns]
    top_risk_table = (
        scored_df[available_top_cols]
        .sort_values(by="composite_risk_score", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "is_valid": True,
        "scored_df": scored_df,
        "top_risk_table": top_risk_table,
        "total_transactions": n_rows,
        "avg_composite_risk": round(float(np.mean(composite_scores)), 1) if n_rows > 0 else 0.0,
        "max_composite_risk": round(float(np.max(composite_scores)), 1) if n_rows > 0 else 0.0,
        "min_composite_risk": round(float(np.min(composite_scores)), 1) if n_rows > 0 else 0.0,
        "counts_by_level": counts_by_level,
        "percentages_by_level": pcts_by_level,
        "available_signals": available_signals,
        "unavailable_signals": unavailable_signals,
        "base_weights": base_weights,
        "effective_weights": effective_weights,
        "has_xgboost_score": has_xgb,
        "redistribution_applied": (not has_xgb or len(unavailable_signals) > 0),
    }
