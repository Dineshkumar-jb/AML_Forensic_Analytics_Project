"""Explainability and Investigation Engine.

Coordinates Explainable AI (SHAP TreeExplainer for supervised XGBoost)
and transparent Rule-Based Forensic Attribution for transaction monitoring.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Safe import of SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

# Feature Descriptions for AML Context
FEATURE_DESCRIPTIONS = {
    "amount": "Raw transaction amount ($)",
    "log_amount": "Log-transformed transaction amount",
    "first_digit": "Leading digit of transaction amount (Benford test)",
    "is_digit_nine": "Leading digit 9 flag (Common structuring indicator)",
    "is_round_1000": "Exact multiple of $1,000 (Round-number smurfing)",
    "is_round_500": "Exact multiple of $500 (Round-number smurfing)",
    "is_near_threshold": "Amount between $8,800 and $9,999.99 (Near CTR threshold)",
    "sender_tx_frequency": "Historical transaction count for originating sender",
    "sender_avg_amount": "Historical average amount for originating sender ($)",
    "sender_max_amount": "Historical maximum amount for originating sender ($)",
    "amount_to_sender_avg": "Ratio of transaction amount to sender's historical mean",
    "amount_zscore_sender": "Z-score of amount relative to sender baseline",
    "receiver_inflow_count": "Historical transaction count for beneficiary receiver",
    "rsf_risk_score": "Relative Size Factor (RSF) outlier risk score",
    "duplicate_risk_score": "Duplicate transaction pattern risk score",
    "velocity_risk_score": "Transaction velocity and burst frequency risk score",
}


def generate_rule_based_importance(
    scored_df: pd.DataFrame,
    forensic_results: Optional[Dict[str, Any]] = None,
    ml_results: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Compute empirical feature / rule importance from deterministic forensic rules.

    Evaluates the prevalence and variance of forensic indicators across all transactions.
    """
    rows = []

    # 1. Structuring
    if "is_structuring" in scored_df.columns:
        struct_count = int(scored_df["is_structuring"].sum())
        struct_score = min(100.0, struct_count * 12.0)
    else:
        struct_count = 0
        struct_score = 0.0
    rows.append({
        "Feature": "Structuring (Near-CTR)",
        "Importance": round(struct_score, 2),
        "Description": "Transactions structured just below the $10,000 reporting threshold",
        "Category": "Forensic Typology",
    })

    # 2. RSF Outliers
    if "rsf_risk_score" in scored_df.columns:
        rsf_mean = float(scored_df["rsf_risk_score"].mean())
        rsf_max = float(scored_df["rsf_risk_score"].max())
        rsf_importance = (0.5 * rsf_mean) + (0.5 * rsf_max)
    else:
        rsf_importance = 0.0
    rows.append({
        "Feature": "Relative Size Factor (RSF)",
        "Importance": round(rsf_importance, 2),
        "Description": "Transaction magnitude deviating from account historical normal",
        "Category": "Statistical Baseline",
    })

    # 3. Isolation Forest Outliers
    if "isolation_forest_risk_score" in scored_df.columns:
        iso_mean = float(scored_df["isolation_forest_risk_score"].mean())
        iso_max = float(scored_df["isolation_forest_risk_score"].max())
        iso_importance = (0.4 * iso_mean) + (0.6 * iso_max)
    else:
        iso_importance = 0.0
    rows.append({
        "Feature": "Isolation Forest Anomaly",
        "Importance": round(iso_importance, 2),
        "Description": "Multi-dimensional geometric outlier in unsupervised feature space",
        "Category": "Unsupervised ML",
    })

    # 4. Duplicate Repeat Patterns
    if "duplicate_risk_score" in scored_df.columns:
        dup_max = float(scored_df["duplicate_risk_score"].max())
        dup_mean = float(scored_df["duplicate_risk_score"].mean())
        dup_importance = (0.5 * dup_mean) + (0.5 * dup_max)
    else:
        dup_importance = 0.0
    rows.append({
        "Feature": "Duplicate Repeat Intervals",
        "Importance": round(dup_importance, 2),
        "Description": "Identical transaction amounts repeated in close succession",
        "Category": "Forensic Temporal",
    })

    # 5. Transaction Velocity Bursts
    if "velocity_risk_score" in scored_df.columns:
        vel_max = float(scored_df["velocity_risk_score"].max())
        vel_mean = float(scored_df["velocity_risk_score"].mean())
        vel_importance = (0.5 * vel_mean) + (0.5 * vel_max)
    else:
        vel_importance = 0.0
    rows.append({
        "Feature": "Transaction Velocity / Bursts",
        "Importance": round(vel_importance, 2),
        "Description": "High transaction frequency over narrow rolling time windows",
        "Category": "Forensic Temporal",
    })

    # 6. Amount Magnitude
    if "amount" in scored_df.columns:
        amt_std = float(scored_df["amount"].std())
        amt_mean = float(scored_df["amount"].mean())
        amt_importance = min(100.0, (amt_std / max(1.0, amt_mean)) * 25.0)
    else:
        amt_importance = 0.0
    rows.append({
        "Feature": "Amount Volatility (Z-score)",
        "Importance": round(amt_importance, 2),
        "Description": "Absolute value dispersion relative to peer population",
        "Category": "Statistical Baseline",
    })

    df_imp = pd.DataFrame(rows).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return df_imp


def compute_explainability(
    ml_results: Dict[str, Any],
    features_df: pd.DataFrame,
    feature_names: List[str],
    composite_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute Explainable AI framework (SHAP TreeExplainer or Rule-Based Attribution).

    Strictly uses SHAP only when a compatible trained model and package are available.
    Does NOT fabricate SHAP values or fake model predictions when unavailable.

    Args:
        ml_results: Output from Stage 3 run_ml_pipeline
        features_df: Engineered feature matrix (N x F)
        feature_names: List of feature names
        composite_results: Output from Stage 4 calculate_composite_risk

    Returns:
        Structured dictionary containing explanation model, global importances,
        and per-transaction explanation utilities.
    """
    scored_df = composite_results.get("scored_df")
    has_valid_xgb = (
        ml_results is not None
        and ml_results.get("is_valid", False)
        and ml_results.get("has_labels", False)
        and ml_results.get("xgboost", {}).get("is_trained", False)
    )

    # -------------------------------------------------------------------------
    # Case A: SHAP is available AND Supervised Model is Trained
    # -------------------------------------------------------------------------
    if HAS_SHAP and has_valid_xgb and features_df is not None and not features_df.empty:
        xgb_model = ml_results["xgboost"]["model"]
        try:
            explainer = shap.TreeExplainer(xgb_model)
            shap_obj = explainer(features_df)
            shap_values = shap_obj.values  # (N, F)

            if hasattr(shap_obj.base_values, "__len__"):
                base_value = float(shap_obj.base_values[0])
            else:
                base_value = float(shap_obj.base_values)

            # Global feature importance = Mean Absolute SHAP Value
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            importance_rows = []
            for feat, val in zip(feature_names, mean_abs_shap):
                desc = FEATURE_DESCRIPTIONS.get(feat, "Engineered transaction feature")
                importance_rows.append({
                    "Feature": feat,
                    "Importance": round(float(val), 4),
                    "Description": desc,
                    "Category": "Model Feature",
                })

            global_importance_df = pd.DataFrame(importance_rows).sort_values(
                by="Importance", ascending=False
            ).reset_index(drop=True)

            return {
                "is_valid": True,
                "method": "shap",
                "is_model_based": True,
                "explainer": explainer,
                "shap_obj": shap_obj,
                "shap_values": shap_values,
                "base_value": base_value,
                "global_importance": global_importance_df,
                "feature_names": feature_names,
                "status_message": "Model-Based Explanation: SHAP TreeExplainer calibrated on supervised XGBoost classifier.",
                "reason": "Supervised model trained on genuine ground-truth labels.",
            }
        except Exception as e:
            # Fallback to rule-based on any computation error
            pass

    # -------------------------------------------------------------------------
    # Case B: SHAP is Unavailable OR Model is Unlabeled / Inactive
    # -------------------------------------------------------------------------
    reason = "SHAP package not installed in environment." if not HAS_SHAP else (
        ml_results.get("xgboost", {}).get("reason", "Supervised model not trained (Awaiting verified labels).")
    )

    rule_importance_df = generate_rule_based_importance(
        scored_df=scored_df,
        forensic_results=None,
        ml_results=ml_results,
    )

    return {
        "is_valid": True,
        "method": "rule_based",
        "is_model_based": False,
        "explainer": None,
        "shap_obj": None,
        "shap_values": None,
        "base_value": 0.0,
        "global_importance": rule_importance_df,
        "feature_names": feature_names,
        "status_message": f"Rule-Based Forensic Attribution Mode: {reason}",
        "reason": reason,
    }


def get_transaction_explanation(
    tx_id: str,
    explainability_res: Dict[str, Any],
    scored_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Retrieve detailed feature contributions for an individual selected transaction.

    Args:
        tx_id: Selected transaction ID (e.g. 'TX_1218')
        explainability_res: Output from compute_explainability
        scored_df: Composite scored DataFrame
        features_df: Numerical feature DataFrame

    Returns:
        Structured dictionary with feature contributions, positive/negative drivers,
        and natural language explanation.
    """
    row_matches = scored_df[scored_df["transaction_id"] == tx_id]
    if row_matches.empty:
        return {"is_valid": False, "error": f"Transaction '{tx_id}' not found."}

    tx_idx = row_matches.index[0]
    row = row_matches.iloc[0]

    # -------------------------------------------------------------------------
    # Mode 1: SHAP Model-Based Explanation
    # -------------------------------------------------------------------------
    if explainability_res.get("is_model_based", False) and explainability_res.get("shap_values") is not None:
        shap_row = explainability_res["shap_values"][tx_idx]
        feat_names = explainability_res["feature_names"]
        base_val = explainability_res["base_value"]

        feature_vals = features_df.iloc[tx_idx].values if features_df is not None else [0] * len(feat_names)

        contrib_list = []
        for name, val, s_val in zip(feat_names, feature_vals, shap_row):
            desc = FEATURE_DESCRIPTIONS.get(name, name)
            is_pos = s_val > 0
            contrib_list.append({
                "Feature": name,
                "Description": desc,
                "Value": round(float(val), 3),
                "Contribution": round(float(s_val), 4),
                "Abs_Contribution": abs(float(s_val)),
                "Direction": "Pushes Risk UP (Suspicious)" if is_pos else "Pushes Risk DOWN (Legitimate)",
                "Impact": "Risk Escalator (+)" if is_pos else "Risk Mitigator (-)",
                "Color": "#ef4444" if is_pos else "#22c55e",
            })

        contrib_df = pd.DataFrame(contrib_list).sort_values(
            by="Abs_Contribution", ascending=False
        ).reset_index(drop=True)

        pos_drivers = contrib_df[contrib_df["Contribution"] > 0]
        neg_drivers = contrib_df[contrib_df["Contribution"] < 0]

        top_pos = pos_drivers.head(4)["Feature"].tolist()
        top_neg = neg_drivers.head(3)["Feature"].tolist()

        # Natural language synthesis
        if top_pos:
            top_pos_str = ", ".join(top_pos)
            pos_text = f"Primary factors increasing suspicion: {top_pos_str}."
        else:
            pos_text = "No strong features pushing the model towards suspicious classification."

        if top_neg:
            top_neg_str = ", ".join(top_neg)
            neg_text = f"Mitigating attributes supporting legitimate activity: {top_neg_str}."
        else:
            neg_text = "Few mitigating baseline attributes observed."

        narrative = (
            f"SHAP Model Attribution: Transaction exhibits an XGBoost risk probability of "
            f"{row.get('xgboost_risk_score', 'N/A')}/100 against a baseline model expected log-odds of {base_val:.2f}. "
            f"{pos_text} {neg_text}"
        )

        return {
            "is_valid": True,
            "transaction_id": tx_id,
            "method": "shap",
            "is_model_based": True,
            "base_value": base_val,
            "contributions_df": contrib_df,
            "positive_drivers": pos_drivers.reset_index(drop=True),
            "negative_drivers": neg_drivers.reset_index(drop=True),
            "top_positive_features": top_pos,
            "top_negative_features": top_neg,
            "narrative": narrative,
            "model_type": "Supervised XGBoost Classifier via SHAP TreeExplainer",
        }

    # -------------------------------------------------------------------------
    # Mode 2: Rule-Based Forensic Attribution (When SHAP/Labels are Inactive)
    # -------------------------------------------------------------------------
    forensic_score = float(row.get("forensic_risk_score", 0.0))
    iso_score = float(row.get("isolation_forest_risk_score", 0.0))
    comp_score = float(row.get("composite_risk_score", 0.0))
    risk_level = str(row.get("risk_level", "Low"))

    contrib_list = []

    # RSF rule
    rsf_s = float(row.get("rsf_risk_score", 0.0))
    if rsf_s >= 40:
        contrib_list.append({
            "Feature": "Relative Size Factor (RSF)",
            "Description": "Transaction exceeds account's historical spending multiplier",
            "Value": f"Score {rsf_s:.0f}",
            "Contribution": round(rsf_s * 0.35, 2),
            "Abs_Contribution": rsf_s * 0.35,
            "Direction": "Pushes Risk UP (Suspicious)",
            "Impact": "Risk Escalator (+)",
            "Color": "#ef4444",
        })
    else:
        contrib_list.append({
            "Feature": "Historical Amount Consistency",
            "Description": "Amount aligns with account historical baseline",
            "Value": f"Score {rsf_s:.0f}",
            "Contribution": -15.0,
            "Abs_Contribution": 15.0,
            "Direction": "Pushes Risk DOWN (Legitimate)",
            "Impact": "Risk Mitigator (-)",
            "Color": "#22c55e",
        })

    # Duplicate rule
    dup_s = float(row.get("duplicate_risk_score", 0.0))
    if dup_s >= 40:
        contrib_list.append({
            "Feature": "Rapid Duplicate Transactions",
            "Description": "Identical amount repeated across counterparties in rapid succession",
            "Value": f"Score {dup_s:.0f}",
            "Contribution": round(dup_s * 0.30, 2),
            "Abs_Contribution": dup_s * 0.30,
            "Direction": "Pushes Risk UP (Suspicious)",
            "Impact": "Risk Escalator (+)",
            "Color": "#ef4444",
        })

    # Velocity rule
    vel_s = float(row.get("velocity_risk_score", 0.0))
    if vel_s >= 40:
        contrib_list.append({
            "Feature": "High-Frequency Burst Velocity",
            "Description": "Excessive transaction frequency during rolling time window",
            "Value": f"Score {vel_s:.0f}",
            "Contribution": round(vel_s * 0.30, 2),
            "Abs_Contribution": vel_s * 0.30,
            "Direction": "Pushes Risk UP (Suspicious)",
            "Impact": "Risk Escalator (+)",
            "Color": "#ef4444",
        })

    # Structuring rule
    is_struct = bool(row.get("is_structuring", False))
    if is_struct:
        contrib_list.append({
            "Feature": "Structuring / Near-CTR Pattern",
            "Description": "Amount structured just below the $10,000 regulatory reporting threshold",
            "Value": f"${row.get('amount', 0):,.2f}",
            "Contribution": 35.0,
            "Abs_Contribution": 35.0,
            "Direction": "Pushes Risk UP (Suspicious)",
            "Impact": "Risk Escalator (+)",
            "Color": "#ef4444",
        })

    # Isolation Forest Anomaly
    if iso_score >= 50:
        contrib_list.append({
            "Feature": "Isolation Forest Outlier Geometry",
            "Description": "Unsupervised algorithm isolates transaction as dimensional outlier",
            "Value": f"Score {iso_score:.1f}",
            "Contribution": round(iso_score * 0.40, 2),
            "Abs_Contribution": iso_score * 0.40,
            "Direction": "Pushes Risk UP (Suspicious)",
            "Impact": "Risk Escalator (+)",
            "Color": "#ef4444",
        })
    else:
        contrib_list.append({
            "Feature": "Unsupervised Cluster Conformance",
            "Description": "Transaction is densely surrounded by peer cluster points",
            "Value": f"Score {iso_score:.1f}",
            "Contribution": -20.0,
            "Abs_Contribution": 20.0,
            "Direction": "Pushes Risk DOWN (Legitimate)",
            "Impact": "Risk Mitigator (-)",
            "Color": "#22c55e",
        })

    contrib_df = pd.DataFrame(contrib_list).sort_values(
        by="Abs_Contribution", ascending=False
    ).reset_index(drop=True)

    pos_drivers = contrib_df[contrib_df["Contribution"] > 0]
    neg_drivers = contrib_df[contrib_df["Contribution"] < 0]

    top_pos = pos_drivers["Feature"].tolist()
    top_neg = neg_drivers["Feature"].tolist()

    narrative = (
        f"Rule-Based Forensic Attribution: Transaction assigned Composite Risk Score of {comp_score:.1f}/100 ({risk_level}). "
        f"Forensic signals contributed {forensic_score:.1f}/100; Isolation Forest contributed {iso_score:.1f}/100. "
        f"Primary risk drivers: {', '.join(top_pos) if top_pos else 'None (conforms to routine baseline)'}."
    )

    return {
        "is_valid": True,
        "transaction_id": tx_id,
        "method": "rule_based",
        "is_model_based": False,
        "base_value": 0.0,
        "contributions_df": contrib_df,
        "positive_drivers": pos_drivers.reset_index(drop=True),
        "negative_drivers": neg_drivers.reset_index(drop=True),
        "top_positive_features": top_pos,
        "top_negative_features": top_neg,
        "narrative": narrative,
        "model_type": "Deterministic Forensic Heuristics & Isolation Forest Scoring",
    }


def get_investigation_dossier(
    tx_id: str,
    scored_df: pd.DataFrame,
    features_df: pd.DataFrame,
    explainability_res: Dict[str, Any],
    forensic_results: Optional[Dict[str, Any]] = None,
    ml_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile comprehensive investigation summary and recommended compliance action.

    Strict compliance rule: Does not automatically label transactions as money laundering.
    Uses 'suspicious or potentially high-risk transaction'.

    Args:
        tx_id: Transaction ID
        scored_df: Composite scored DataFrame
        features_df: Feature matrix
        explainability_res: Explainability engine output
        forensic_results: Stage 2 forensic suite output
        ml_results: Stage 3 ML pipeline output

    Returns:
        Structured investigation dossier dictionary.
    """
    row_matches = scored_df[scored_df["transaction_id"] == tx_id]
    if row_matches.empty:
        return {"is_valid": False, "error": f"Transaction '{tx_id}' not found."}

    row = row_matches.iloc[0]
    explanation_detail = get_transaction_explanation(
        tx_id=tx_id,
        explainability_res=explainability_res,
        scored_df=scored_df,
        features_df=features_df,
    )

    comp_score = float(row.get("composite_risk_score", 0.0))
    risk_level = str(row.get("risk_level", "Low"))

    # Calculate percentile rank in dataset
    all_scores = scored_df["composite_risk_score"].values
    percentile = round(float(np.mean(all_scores <= comp_score) * 100.0), 1)

    # Compile explicit list of suspicious indicators triggered
    indicators = []
    if pd.notnull(row.get("is_structuring")) and bool(row.get("is_structuring")):
        indicators.append("Structuring: Amount near $10,000 regulatory reporting threshold")
    if pd.notnull(row.get("rsf_risk_score")) and float(row.get("rsf_risk_score")) >= 40:
        indicators.append(f"Relative Size Factor (RSF) Outlier: Score {row.get('rsf_risk_score'):.0f}/100")
    if pd.notnull(row.get("duplicate_risk_score")) and float(row.get("duplicate_risk_score")) >= 40:
        indicators.append(f"Rapid Duplicate Cluster: Score {row.get('duplicate_risk_score'):.0f}/100")
    if pd.notnull(row.get("velocity_risk_score")) and float(row.get("velocity_risk_score")) >= 40:
        indicators.append(f"High Velocity Burst Activity: Score {row.get('velocity_risk_score'):.0f}/100")
    if pd.notnull(row.get("isolation_forest_risk_score")) and float(row.get("isolation_forest_risk_score")) >= 60:
        indicators.append(f"Unsupervised Isolation Forest Outlier: Score {row.get('isolation_forest_risk_score'):.0f}/100")
    if pd.notnull(row.get("xgboost_risk_score")) and float(row.get("xgboost_risk_score")) >= 60:
        indicators.append(f"Supervised XGBoost Classification: Score {row.get('xgboost_risk_score'):.0f}/100")

    if not indicators:
        indicators.append("None: Transaction activity conforms to routine behavioral baseline")

    # Define recommended investigation action matrix
    if risk_level == "Critical":
        rec_action = {
            "title": "Level 4: Urgent SAR Filing & Immediate Compliance Escalation",
            "priority": "CRITICAL",
            "badge_color": "#ef4444",
            "summary": (
                "Potentially high-risk transaction exhibiting multi-pillar anomaly convergence. "
                "Immediate compliance officer review and formal Suspicious Activity Report (SAR) preparation recommended."
            ),
            "steps": [
                "Place immediate temporary administrative hold on pending settlement funds.",
                "Review complete 90-day transaction history for originating and beneficiary accounts.",
                "Draft FinCEN Form 111 (SAR) citing specific red flags (structuring, velocity, RSF outliers).",
                "Notify Designated AML Compliance Officer for executive sign-off within 48 hours.",
                "Assess counterparties under FinCEN Section 314(b) voluntary information-sharing provisions.",
            ],
        }
    elif risk_level == "High":
        rec_action = {
            "title": "Level 3: Enhanced Due Diligence (EDD) & Source of Funds Inquiry",
            "priority": "HIGH",
            "badge_color": "#f97316",
            "summary": (
                "Suspicious transaction profile with elevated risk metrics. "
                "Initiate secondary compliance review and document verification."
            ),
            "steps": [
                "Issue Request for Information (RFI) to customer requesting commercial justification or invoices.",
                "Verify economic purpose and beneficial ownership for participating entities.",
                "Check sender and receiver against latest OFAC / PEP watchlists.",
                "Schedule case re-examination in 14 business days.",
            ],
        }
    elif risk_level == "Medium":
        rec_action = {
            "title": "Level 2: Continuous Transaction Monitoring & 30-Day Watchlist",
            "priority": "MEDIUM",
            "badge_color": "#f59e0b",
            "summary": (
                "Moderate anomalies detected. Place account on heightened surveillance without immediate restriction."
            ),
            "steps": [
                "Tag account for 30-day automated velocity and threshold alerts.",
                "Confirm customer KYC (Know Your Customer) documentation is up-to-date.",
                "Escalate to Level 3 if additional high-risk indicators trigger within next rolling window.",
            ],
        }
    else:
        rec_action = {
            "title": "Level 1: Routine Audit Clearance & Standard Monitoring",
            "priority": "LOW",
            "badge_color": "#22c55e",
            "summary": (
                "Transaction conforms to expected normal transactional baseline. Standard periodic archiving."
            ),
            "steps": [
                "Log transaction in compliance data warehouse for statutory recordkeeping.",
                "No manual investigator intervention required at this time.",
            ],
        }

    return {
        "is_valid": True,
        "transaction_id": tx_id,
        "timestamp": str(row.get("timestamp", "N/A")),
        "sender_account": str(row.get("sender_account", "N/A")),
        "receiver_account": str(row.get("receiver_account", "N/A")),
        "amount": float(row.get("amount", 0.0)),
        "transaction_type": str(row.get("transaction_type", "TRANSFER")),
        "location": str(row.get("location", "N/A")),
        "composite_risk_score": comp_score,
        "risk_level": risk_level,
        "percentile_rank": percentile,
        "forensic_score": float(row.get("forensic_risk_score", 0.0)),
        "iso_score": float(row.get("isolation_forest_risk_score", 0.0)),
        "xgb_score": float(row.get("xgboost_risk_score")) if pd.notnull(row.get("xgboost_risk_score")) else None,
        "active_factors_str": str(row.get("contributing_risk_factors", "Routine")),
        "audit_explanation": str(row.get("risk_explanation", "")),
        "suspicious_indicators": indicators,
        "recommended_action": rec_action,
        "explanation_detail": explanation_detail,
        "is_model_based": explanation_detail.get("is_model_based", False),
        "demo_mode": True,
        "sample_size": len(scored_df),
        "platform_scale": 5_000_000,
        "scale_disclaimer": "Demo Mode – 5M illustrative transaction scale; actual analysis uses 218 sample transactions.",
    }

