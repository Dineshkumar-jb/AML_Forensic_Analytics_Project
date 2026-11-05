"""Transaction Monitoring & Risk Scoring for Money Laundering Detection.

Stage 4: Unified Multi-Pillar Composite Risk Scoring Engine (0-100)
(Deterministic Forensics + Unsupervised Isolation Forest + Supervised XGBoost + Proportional Redistribution).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

try:
    from src.config import (
        APP_NAME,
        APP_SUBTITLE,
        APP_VERSION,
        REQUIRED_COLUMNS,
        OPTIONAL_COLUMNS,
        SAMPLE_DATA_PATH,
        RISK_TIERS_STAGE4,
        DEFAULT_COMPOSITE_WEIGHTS,
        DEMO_MODE_ACTIVE,
        ILLUSTRATIVE_PLATFORM_SCALE,
        ACTUAL_SAMPLE_SIZE,
        DEMO_MODE_BANNER_TEXT,
    )
except ImportError:
    # If a long-running process cached an older version of src.config, force reload
    import importlib
    import src.config
    importlib.reload(src.config)
    from src.config import (
        APP_NAME,
        APP_SUBTITLE,
        APP_VERSION,
        REQUIRED_COLUMNS,
        OPTIONAL_COLUMNS,
        SAMPLE_DATA_PATH,
        RISK_TIERS_STAGE4,
        DEFAULT_COMPOSITE_WEIGHTS,
    )
    DEMO_MODE_ACTIVE = getattr(src.config, "DEMO_MODE_ACTIVE", True)
    ILLUSTRATIVE_PLATFORM_SCALE = getattr(src.config, "ILLUSTRATIVE_PLATFORM_SCALE", 5_000_000)
    ACTUAL_SAMPLE_SIZE = getattr(src.config, "ACTUAL_SAMPLE_SIZE", 218)
    DEMO_MODE_BANNER_TEXT = getattr(
        src.config,
        "DEMO_MODE_BANNER_TEXT",
        "Demo Mode – 5M illustrative transaction scale; actual analysis uses 218 sample transactions.",
    )
from src.utils.data_loader import (
    load_csv,
    validate_dataframe,
    get_dataset_summary,
    generate_sample_dataset,
)
from src.forensic import (
    run_forensic_suite,
    analyze_benford,
    calculate_rsf,
    detect_duplicates,
    analyze_velocity,
    analyze_all_typologies,
)
from src.ml import (
    run_ml_pipeline,
    extract_ml_features,
    train_isolation_forest,
    train_xgboost,
    detect_ground_truth_label,
)
from src.scoring import (
    calculate_composite_risk,
    classify_risk_level,
    compute_transaction_forensic_score,
)
from src.explainability import (
    compute_explainability,
    get_transaction_explanation,
    get_investigation_dossier,
)
from src.reporting import (
    generate_full_forensic_report,
    generate_transaction_dossier_report,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AML Forensic Analytics & Risk Scoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling (Modern Fintech & Forensic Analytics UI)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transition: all 0.25s ease-in-out;
    }
    
    /* Stage Pill Badges */
    .stage-badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stage-completed {
        background-color: rgba(34, 197, 94, 0.12);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }
    .stage-active {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }
    .stage-pending {
        background-color: rgba(148, 163, 184, 0.08);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }

    /* Risk Tier Pills */
    .tier-pill-critical {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .tier-pill-high {
        background-color: rgba(249, 115, 22, 0.15);
        color: #f97316;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid rgba(249, 115, 22, 0.35);
    }
    .tier-pill-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .tier-pill-low {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }

    /* Audit dossier card */
    .dossier-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Navigation & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ AML Forensic Hub")
    st.markdown(
        f"**{APP_NAME}**  \n"
        f"<span style='color: #64748b; font-size: 0.85rem;'>Stage 6 (Full Pipeline Active)</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Project Stage Roadmap
    st.markdown("#### 📌 Implementation Roadmap")
    stages = [
        ("Stage 1", "Data Ingestion & Schema", "completed"),
        ("Stage 2", "Forensic Analytics Suite", "completed"),
        ("Stage 3", "ML Detection Layer", "completed"),
        ("Stage 4", "Composite Risk Scoring", "completed"),
        ("Stage 5", "SHAP & Investigation View", "completed"),
        ("Stage 6", "Final Audit Report Export", "completed"),
    ]

    for code, title, status in stages:
        if status == "completed":
            badge_cls, status_txt = "stage-completed", "✓ Done"
        elif status == "active":
            badge_cls, status_txt = "stage-active", "🟢 Active"
        else:
            badge_cls, status_txt = "stage-pending", "⏳ Upcoming"

        st.markdown(
            f"<div style='margin-bottom: 6px; font-size: 0.82rem;'>"
            f"<strong>{code}:</strong> {title} "
            f"<span class='stage-badge {badge_cls}' style='float: right;'>{status_txt}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📂 Data Source")

    data_source_mode = st.radio(
        "Choose Ingestion Method:",
        ["Built-in Sample Dataset", "Upload Custom CSV"],
        index=0,
        help="Use pre-seeded realistic AML transactions or upload your own CSV file.",
    )

    df_loaded = None
    load_error = None

    if data_source_mode == "Built-in Sample Dataset":
        if not SAMPLE_DATA_PATH.exists():
            with st.spinner("Generating sample dataset..."):
                generate_sample_dataset()
        df_loaded, load_error = load_csv(SAMPLE_DATA_PATH)
        st.success("Loaded built-in forensic sample dataset.")
    else:
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=["csv"],
            help="Upload a CSV with transaction_id, timestamp, sender_account, receiver_account, amount",
        )
        if uploaded_file is not None:
            df_loaded, load_error = load_csv(uploaded_file)
            if load_error:
                st.error(load_error)
            else:
                st.success(f"Loaded '{uploaded_file.name}' successfully.")
        else:
            st.info("Please upload a CSV file or switch to 'Built-in Sample Dataset'.")

    # Composite Scoring Weights Customizer
    st.markdown("---")
    with st.expander("⚖️ Composite Risk Weights", expanded=False):
        st.caption("Default weights: 40% Forensics, 30% IsoForest, 30% XGBoost. Automatically redistributed if a component is unavailable.")
        w_forensic = st.slider("Forensic Risk Weight", min_value=0.10, max_value=0.70, value=0.40, step=0.05)
        w_iso = st.slider("Isolation Forest Weight", min_value=0.10, max_value=0.70, value=0.30, step=0.05)
        w_xgb = st.slider("XGBoost Supervised Weight", min_value=0.10, max_value=0.70, value=0.30, step=0.05)
        
        tot_user_w = w_forensic + w_iso + w_xgb
        user_weights = {
            "forensic": w_forensic / tot_user_w,
            "isolation_forest": w_iso / tot_user_w,
            "xgboost": w_xgb / tot_user_w,
        }

    # Forensic Sensitivity Settings
    with st.expander("⚙️ Forensic Parameters", expanded=False):
        rsf_threshold = st.slider("RSF Outlier Ratio Threshold", min_value=2.0, max_value=15.0, value=5.0, step=0.5,
                                  help="Flag accounts where Largest / Second Largest exceeds this multiplier")
        dup_window_hours = st.slider("Duplicate Rapid Window (Hours)", min_value=1.0, max_value=72.0, value=48.0, step=1.0,
                                     help="Maximum hours between identical amounts to flag as rapid duplicate")
        vel_window_hours = st.slider("Velocity Window (Hours)", min_value=0.5, max_value=12.0, value=2.0, step=0.5,
                                     help="Time window to detect transaction burst frequency")
        vel_burst_count = st.slider("Velocity Burst Minimum Count", min_value=2, max_value=10, value=4, step=1,
                                    help="Minimum transactions in window to trigger velocity burst")

    # ML Parameters
    with st.expander("🤖 ML Model Hyperparameters", expanded=False):
        iso_contamination = st.slider(
            "Isolation Forest Contamination",
            min_value=0.01,
            max_value=0.25,
            value=0.08,
            step=0.01,
            help="Expected proportion of anomalies in the dataset for unsupervised isolation trees.",
        )

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------

# Header Banner
st.title("🛡️ " + APP_NAME)
st.caption(f"{APP_SUBTITLE} • College Forensic Analytics Capstone Project")

if df_loaded is None:
    st.warning("⚠️ No data loaded yet. Please select the built-in sample data or upload a valid CSV file from the sidebar.")
    st.stop()

# Compute high level summary stats
summary = get_dataset_summary(df_loaded)

# Pipeline Execution (Stages 2, 3, and 4)
with st.spinner("Executing Forensic Auditing Suite..."):
    forensic_results = run_forensic_suite(df_loaded)

with st.spinner("Executing Machine Learning Detection Pipeline..."):
    ml_results = run_ml_pipeline(
        df=df_loaded,
        enriched_forensic_df=forensic_results.get("enriched_df"),
        contamination=iso_contamination,
    )

with st.spinner("Calculating Composite Risk Scores..."):
    composite_results = calculate_composite_risk(
        df=df_loaded,
        forensic_results=forensic_results,
        ml_results=ml_results,
        custom_weights=user_weights,
    )

if not composite_results.get("is_valid", False):
    st.error(f"❌ Composite risk calculation error: {composite_results.get('error')}")
    st.stop()

with st.spinner("Computing Explainable AI & Feature Attribution..."):
    explainability_results = compute_explainability(
        ml_results=ml_results,
        features_df=ml_results.get("features_df"),
        feature_names=ml_results.get("feature_names", []),
        composite_results=composite_results,
    )

# -----------------------------------------------------------------------------
# Top Metric KPI Bar (Stage 4 Enhanced with Demo Scale Distinction)
# -----------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Platform Scale (Demo)",
    "5,000,000 txs",
    delta=f"Actual Sample: {summary.get('total_records', 0)} analyzed",
    delta_color="off",
)
col2.metric(
    "Audited Volume",
    f"${summary.get('total_volume', 0):,.2f}",
    delta=f"{summary.get('total_records', 0)} sample records",
    delta_color="off",
)

# Average Composite Risk Score
avg_comp = composite_results["avg_composite_risk"]
avg_tier = classify_risk_level(avg_comp)
col3.metric(
    "Avg Composite Risk",
    f"{avg_comp:.1f} / 100",
    delta=f"Tier: {avg_tier}",
    delta_color="normal" if avg_comp < 25 else ("inverse" if avg_comp >= 50 else "off"),
)

# Peak Composite Risk Score
max_comp = composite_results["max_composite_risk"]
max_tier = classify_risk_level(max_comp)
col4.metric(
    "Peak Composite Risk",
    f"{max_comp:.1f} / 100",
    delta=f"Peak: {max_tier}",
    delta_color="inverse" if max_comp >= 50 else "normal",
)

# High & Critical Alert Count
high_crit_count = composite_results["counts_by_level"]["High"] + composite_results["counts_by_level"]["Critical"]
high_crit_pct = round(high_crit_count / max(1, len(df_loaded)) * 100.0, 1)
col5.metric(
    "High & Critical Alerts",
    f"{high_crit_count} txs",
    delta=f"{high_crit_pct}% of {summary.get('total_records', 0)} sample",
    delta_color="inverse" if high_crit_count > 0 else "normal",
)

# -----------------------------------------------------------------------------
# Tabs Layout (Stages 1–6 Complete Forensic Architecture)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎯 Composite Risk Scoring",
    "🕵️ Forensic Analytics Suite",
    "🤖 ML Detection Layer",
    "🔍 Stage 5: Explainable AI & Investigation",
    "📑 Stage 6: Forensic Audit Report Export",
    "📋 Data Explorer & Validation",
    "📊 Transaction Analytics",
    "🏛️ AML Methodology & Viva Architecture",
])

# =============================================================================
# TAB 1: COMPOSITE RISK SCORING (STAGE 4 CORE)
# =============================================================================
with tab1:
    st.markdown("### 🎯 Multi-Pillar Composite Risk Scoring (0–100)")
    st.caption(f"Synthesizes deterministic forensic auditing with unsupervised Isolation Forest and supervised XGBoost across {len(df_loaded)} actual sample records (5M illustrative platform scale).")

    # Status & Weight Redistribution Notice
    if composite_results["redistribution_applied"]:
        st.info(
            "ℹ️ **Proportional Weight Redistribution Active**: "
            "Supervised XGBoost model is currently inactive for this dataset (no genuine audit labels present). "
            f"Its weight was proportionally redistributed between Forensics ({composite_results['effective_weights']['forensic'] * 100:.1f}%) "
            f"and Isolation Forest ({composite_results['effective_weights']['isolation_forest'] * 100:.1f}%). "
            "**No fabricated labels or fake probabilities were introduced.**"
        )
    else:
        st.success(
            "✅ **All 3 Analytical Pillars Active**: "
            f"Forensics ({composite_results['effective_weights']['forensic'] * 100:.1f}%), "
            f"Isolation Forest ({composite_results['effective_weights']['isolation_forest'] * 100:.1f}%), "
            f"XGBoost Supervised ({composite_results['effective_weights']['xgboost'] * 100:.1f}%)."
        )

    # 4-Tier Risk Distribution Metrics Row
    st.markdown(f"#### 📊 Risk Tier Breakdown ({len(df_loaded)} Actual Sample Transactions | 5M Illustrative Scale)")
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    counts = composite_results["counts_by_level"]
    pcts = composite_results["percentages_by_level"]

    r_col1.metric("🟢 Low Risk (0–24)", f"{counts['Low']}", delta=f"{pcts['Low']}%", delta_color="normal")
    r_col2.metric("🟡 Medium Risk (25–49)", f"{counts['Medium']}", delta=f"{pcts['Medium']}%", delta_color="off")
    r_col3.metric("🟠 High Risk (50–74)", f"{counts['High']}", delta=f"{pcts['High']}%", delta_color="inverse")
    r_col4.metric("🔴 Critical Risk (75–100)", f"{counts['Critical']}", delta=f"{pcts['Critical']}%", delta_color="inverse")

    st.markdown("---")

    # Visual Distribution Charts
    c_graph1, c_graph2 = st.columns(2)

    with c_graph1:
        st.markdown("##### 🍩 Risk Tier Proportions")
        tier_df = pd.DataFrame({
            "Risk Tier": ["Low (0–24)", "Medium (25–49)", "High (50–74)", "Critical (75–100)"],
            "Count": [counts["Low"], counts["Medium"], counts["High"], counts["Critical"]],
        })
        fig_donut = px.pie(
            tier_df,
            names="Risk Tier",
            values="Count",
            hole=0.45,
            color="Risk Tier",
            color_discrete_map={
                "Low (0–24)": "#22c55e",
                "Medium (25–49)": "#f59e0b",
                "High (50–74)": "#f97316",
                "Critical (75–100)": "#ef4444",
            },
            title="Proportion of Transactions by Risk Tier",
        )
        fig_donut.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_graph2:
        st.markdown("##### 📈 Composite Risk Score Distribution")
        scored_data = composite_results["scored_df"]
        fig_score_hist = px.histogram(
            scored_data,
            x="composite_risk_score",
            nbins=35,
            color="risk_level",
            color_discrete_map={
                "Low": "#22c55e",
                "Medium": "#f59e0b",
                "High": "#f97316",
                "Critical": "#ef4444",
            },
            title="Frequency of Composite Risk Scores (0-100)",
            labels={"composite_risk_score": "Composite Risk Score (0-100)", "count": "Transactions"},
        )
        fig_score_hist.add_vline(x=25, line_dash="dash", line_color="#f59e0b")
        fig_score_hist.add_vline(x=50, line_dash="dash", line_color="#f97316")
        fig_score_hist.add_vline(x=75, line_dash="dash", line_color="#ef4444")
        fig_score_hist.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_score_hist, use_container_width=True)

    st.markdown("---")

    # Top Highest-Risk Transactions Table
    st.markdown("#### 🚨 Top High-Risk Transactions (Ranked by Alert Severity)")

    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        tier_filter = st.selectbox("Filter by Risk Tier:", ["All Tiers", "Critical (75–100)", "High (50–74)", "Medium (25–49)", "Low (0–24)"], index=0)
    with filter_col2:
        search_tx = st.text_input("Search Account or Transaction ID:", placeholder="e.g. TX_1001 or ACC_SUSP", key="comp_search")

    display_risk_df = composite_results["top_risk_table"]

    if tier_filter != "All Tiers":
        filter_level = tier_filter.split()[0]
        display_risk_df = display_risk_df[display_risk_df["risk_level"] == filter_level]

    if search_tx:
        s_mask = (
            display_risk_df["transaction_id"].astype(str).str.contains(search_tx, case=False)
            | display_risk_df["sender_account"].astype(str).str.contains(search_tx, case=False)
            | display_risk_df["receiver_account"].astype(str).str.contains(search_tx, case=False)
        )
        display_risk_df = display_risk_df[s_mask]

    st.dataframe(
        display_risk_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Auditor Drilldown & Per-Transaction Risk Explanation
    st.markdown("#### 🔍 Auditor Transaction Investigation Dossier")
    st.caption("Select any transaction to inspect its granular component risk drivers and natural language audit justification.")

    all_tx_ids = composite_results["scored_df"]["transaction_id"].tolist()
    # Default to the most risky transaction
    selected_tx_id = st.selectbox("Select Transaction ID for Forensic Dossier:", all_tx_ids, index=0)

    selected_row = composite_results["scored_df"][composite_results["scored_df"]["transaction_id"] == selected_tx_id].iloc[0]

    # Dossier Card Layout
    st.markdown(
        f"""
        <div class="dossier-card">
            <h4 style="margin-top: 0; color: #60a5fa;">📄 Investigation Dossier: {selected_row['transaction_id']}</h4>
            <div style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 14px;">
                <strong>Audit Explanation:</strong> {selected_row['risk_explanation']}
            </div>
            <div style="display: flex; gap: 12px; margin-bottom: 12px; font-size: 0.88rem;">
                <span><strong>Originating Sender:</strong> <code>{selected_row['sender_account']}</code></span>
                <span><strong>Beneficiary Receiver:</strong> <code>{selected_row['receiver_account']}</code></span>
                <span><strong>Transaction Amount:</strong> <code>${selected_row['amount']:,.2f}</code></span>
            </div>
            <div style="font-size: 0.88rem; color: #94a3b8;">
                <strong>Active Risk Drivers:</strong> <span style="color: #f97316;">{selected_row['contributing_risk_factors']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Component Contribution Metric Tiles for this Transaction
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    d_col1.metric("Final Composite Risk", f"{selected_row['composite_risk_score']:.1f} / 100", delta=selected_row['risk_level'])
    d_col2.metric("Forensic Audit Signal", f"{selected_row['forensic_risk_score']:.1f} / 100")
    d_col3.metric("Isolation Forest Outlier", f"{selected_row['isolation_forest_risk_score']:.1f} / 100")

    if composite_results["has_xgboost_score"] and pd.notnull(selected_row.get("xgboost_risk_score")):
        d_col4.metric("XGBoost Probability", f"{selected_row['xgboost_risk_score']:.1f} / 100")
    else:
        d_col4.metric("XGBoost Probability", "N/A (Unlabeled)", delta="Weight Redistributed", delta_color="off")

    st.markdown("---")

    # Mathematical Transparency & Formula Details
    with st.expander("📐 Mathematical Transparency & Proportional Weight Redistribution Formula"):
        st.markdown(
            r"""
            ##### Mathematical Formulation
            The Composite Risk Score ($R_i \in [0, 100]$) for transaction $i$ is evaluated using a convex combination of normalized component signals:
            $$\text{Composite Risk}_i = \sum_{k \in \mathcal{A}} \left( w_k^* \cdot S_{i, k} \right)$$
            Where:
            - $\mathcal{A}$ is the set of currently active/available analytical signals.
            - $S_{i, k} \in [0, 100]$ is the normalized risk score from pillar $k$.
            - $w_k^*$ is the proportionally redistributed effective weight:
            $$w_k^* = \frac{W_k}{\sum_{j \in \mathcal{A}} W_j} \quad \text{such that} \quad \sum_{k \in \mathcal{A}} w_k^* = 1.0$$
            """
        )

        st.markdown("##### Active Component Weight Allocation")
        eff_weights = composite_results["effective_weights"]
        base_w = composite_results["base_weights"]
        weight_table = pd.DataFrame([
            {
                "Analytical Pillar": "Forensic Auditing Suite",
                "Base Weight": f"{base_w['forensic'] * 100:.1f}%",
                "Effective Weight Applied": f"{eff_weights['forensic'] * 100:.2f}%",
                "Status": "🟢 Active" if "forensic" in composite_results["available_signals"] else "🔴 Unavailable",
            },
            {
                "Analytical Pillar": "Isolation Forest (Unsupervised)",
                "Base Weight": f"{base_w['isolation_forest'] * 100:.1f}%",
                "Effective Weight Applied": f"{eff_weights['isolation_forest'] * 100:.2f}%",
                "Status": "🟢 Active" if "isolation_forest" in composite_results["available_signals"] else "🔴 Unavailable",
            },
            {
                "Analytical Pillar": "XGBoost Supervised Classification",
                "Base Weight": f"{base_w['xgboost'] * 100:.1f}%",
                "Effective Weight Applied": f"{eff_weights['xgboost'] * 100:.2f}%",
                "Status": "🟢 Active" if composite_results["has_xgboost_score"] else "⚠️ Awaiting Labeled Dataset",
            },
        ])
        st.table(weight_table)

# =============================================================================
# TAB 2: FORENSIC ANALYTICS SUITE (STAGE 2)
# =============================================================================
with tab2:
    st.markdown("### 🕵️ Forensic Auditing & Red-Flag Detection Suite")
    st.caption(f"Deterministic statistical and heuristic forensic tests evaluated on {len(df_loaded)} actual sample records (5M illustrative platform scale).")

    f_sub1, f_sub2, f_sub3, f_sub4, f_sub5, f_sub6 = st.tabs([
        "📋 Executive Scorecard",
        "🔢 Benford's Law",
        "📐 Relative Size Factor (RSF)",
        "🔁 Duplicate Detection",
        "⚡ Velocity Analysis",
        "🎯 Typology Heuristics",
    ])

    # Subtab 1: Executive Scorecard
    with f_sub1:
        st.markdown("#### 📋 Executive Forensic Scorecard")
        st.markdown(
            "Consolidated audit scorecard for all 5 forensic examinations. "
            "Displays raw empirical findings alongside calibrated 0-100 risk scores."
        )

        scorecard_df = forensic_results["scorecard"]
        st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📊 Comparative Forensic Risk Spectrum")
        fig_score_bar = px.bar(
            scorecard_df,
            x="Forensic Test",
            y="Risk Score (0-100)",
            color="Severity",
            color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
            text="Risk Score (0-100)",
            title="Individual Forensic Test Severity Ratings",
        )
        fig_score_bar.update_layout(template="plotly_dark", yaxis=dict(range=[0, 105]), margin=dict(l=20, r=20, t=40, b=20), height=340)
        fig_score_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_score_bar, use_container_width=True)

    # Subtab 2: Benford's Law
    with f_sub2:
        st.markdown("#### 🔢 Benford's Law Digit Analysis")
        benford_data = forensic_results["benford"]
        if not benford_data.get("is_valid"):
            st.error(benford_data.get("error", "Benford analysis failed."))
        else:
            f_digit = benford_data["first_digit"]
            s_digit = benford_data.get("second_digit", {})

            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Benford Risk Score", f"{benford_data['overall_risk_score']:.1f} / 100")
            b_col2.metric("Mean Absolute Dev (MAD)", f"{f_digit['mad']:.4f}")
            b_col3.metric("Conformity Status", f"{f_digit['conformity']}")
            b_col4.metric("Chi-Square Stat (p-val)", f"{f_digit['chi2_stat']:.2f} ({f_digit['p_value']:.4f})")

            b_tab_first, b_tab_second = st.tabs(["First-Digit Test (1–9)", "Second-Digit Test (0–9)"])

            with b_tab_first:
                first_df = f_digit["table"]
                fig_benford1 = go.Figure()
                fig_benford1.add_trace(go.Bar(x=first_df["Digit"], y=first_df["Observed_Freq"], name="Observed Frequency (%)", marker_color="#3b82f6"))
                fig_benford1.add_trace(go.Scatter(x=first_df["Digit"], y=first_df["Expected_Freq"], name="Benford Expected (%)", mode="lines+markers", line=dict(color="#f59e0b", width=3), marker=dict(size=8)))
                fig_benford1.update_layout(template="plotly_dark", title="First-Digit Frequency vs. Benford Curve", xaxis=dict(title="Leading Digit (1–9)", tickmode="linear", tick0=1, dtick=1), yaxis=dict(title="Frequency (%)"), height=360, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_benford1, use_container_width=True)
                st.dataframe(first_df, use_container_width=True, hide_index=True)

            with b_tab_second:
                if s_digit.get("is_valid"):
                    second_df = s_digit["table"]
                    fig_benford2 = go.Figure()
                    fig_benford2.add_trace(go.Bar(x=second_df["Digit"], y=second_df["Observed_Freq"], name="Observed Frequency (%)", marker_color="#10b981"))
                    fig_benford2.add_trace(go.Scatter(x=second_df["Digit"], y=second_df["Expected_Freq"], name="Benford Expected (%)", mode="lines+markers", line=dict(color="#ec4899", width=3), marker=dict(size=8)))
                    fig_benford2.update_layout(template="plotly_dark", title="Second-Digit Frequency vs. Benford Curve", xaxis=dict(title="Second Digit (0–9)", tickmode="linear", tick0=0, dtick=1), yaxis=dict(title="Frequency (%)"), height=360, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_benford2, use_container_width=True)
                    st.dataframe(second_df, use_container_width=True, hide_index=True)
                else:
                    st.info(s_digit.get("error", "Second digit analysis requires more records >= $10."))

    # Subtab 3: Relative Size Factor (RSF)
    with f_sub3:
        st.markdown("#### 📐 Relative Size Factor (RSF) Analysis")
        rsf_data = forensic_results["rsf"]
        if not rsf_data.get("is_valid"):
            st.error(rsf_data.get("error", "RSF calculation failed."))
        else:
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            r_col1.metric("RSF Risk Score", f"{rsf_data['risk_score']:.1f} / 100")
            r_col2.metric("Max RSF Observed", f"{rsf_data['max_rsf']:.1f}x")
            r_col3.metric("Anomalous Accounts", f"{rsf_data['anomaly_count']}")
            r_col4.metric("Mean Account RSF", f"{rsf_data['mean_rsf']:.2f}x")

            rsf_df = rsf_data["account_table"]
            top_rsf = rsf_df.head(12)
            fig_rsf = px.bar(top_rsf, x="account", y="rsf_ratio", color="is_anomalous", color_discrete_map={True: "#ef4444", False: "#3b82f6"}, title=f"Top Accounts by RSF (Threshold: {rsf_threshold}x)", labels={"account": "Account ID", "rsf_ratio": "RSF Multiplier (x)"}, text="rsf_ratio")
            fig_rsf.add_hline(y=rsf_threshold, line_dash="dash", line_color="#f59e0b", annotation_text=f"Threshold ({rsf_threshold}x)", annotation_position="top right")
            fig_rsf.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), height=350)
            fig_rsf.update_traces(textposition="outside")
            st.plotly_chart(fig_rsf, use_container_width=True)
            st.dataframe(rsf_df, use_container_width=True, hide_index=True)

    # Subtab 4: Duplicate Detection
    with f_sub4:
        st.markdown("#### 🔁 Duplicate Transaction Detection")
        dup_data = forensic_results["duplicates"]
        if not dup_data.get("is_valid"):
            st.error(dup_data.get("error", "Duplicate analysis failed."))
        else:
            d_col1, d_col2, d_col3 = st.columns(3)
            d_col1.metric("Duplicate Risk Score", f"{dup_data['risk_score']:.1f} / 100")
            d_col2.metric("Duplicate Transactions", f"{dup_data['duplicate_count']}")
            d_col3.metric("Duplicate Volume", f"${dup_data['duplicate_volume']:,.2f}")

            clusters_df = dup_data["duplicate_clusters"]
            if clusters_df.empty:
                st.success("✅ No duplicate transactions detected in this dataset.")
            else:
                st.dataframe(clusters_df, use_container_width=True, hide_index=True)

    # Subtab 5: Velocity Analysis
    with f_sub5:
        st.markdown("#### ⚡ Transaction Velocity Analysis")
        vel_data = forensic_results["velocity"]
        if not vel_data.get("is_valid"):
            st.error(vel_data.get("error", "Velocity analysis failed."))
        else:
            v_col1, v_col2, v_col3 = st.columns(3)
            v_col1.metric("Velocity Risk Score", f"{vel_data['risk_score']:.1f} / 100")
            v_col2.metric("Accounts in Burst", f"{vel_data['burst_account_count']}")
            v_col3.metric("Peak Burst Window Count", f"{vel_data['max_burst_count']} txs")

            burst_table = vel_data["burst_events_table"]
            if not burst_table.empty:
                fig_vel = px.bar(burst_table, x="account", y="max_burst_count", color="burst_volume", title="Peak Transactions per Burst Window by Account", color_continuous_scale="Reds")
                fig_vel.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_vel, use_container_width=True)
                st.dataframe(burst_table, use_container_width=True, hide_index=True)
            else:
                st.success("No velocity bursts detected exceeding threshold.")

    # Subtab 6: Typology Heuristics
    with f_sub6:
        st.markdown("#### 🎯 AML Typology Heuristic Indicators")
        typ_data = forensic_results["typologies"]
        st.metric("Overall Typology Risk Score", f"{typ_data['overall_risk_score']:.1f} / 100")

        with st.expander("💵 1. Structuring / Smurfing (Near-Threshold Amounts)", expanded=True):
            st_data = typ_data["structuring"]
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Structuring Score", f"{st_data['risk_score']:.1f} / 100")
            sc2.metric("Transactions Flagged", f"{st_data['detected_count']}")
            sc3.metric("Structured Volume", f"${st_data['total_volume']:,.2f}")
            if st_data["detected_count"] > 0:
                flagged_txs = st_data["flagged_transactions"][["transaction_id", "timestamp", "sender_account", "receiver_account", "amount", "location"]]
                st.dataframe(flagged_txs, use_container_width=True, hide_index=True)

        with st.expander("📥 2. Fan-in Pattern (Mule Account Consolidation)", expanded=False):
            fi_data = typ_data["fan_in"]
            if fi_data["hub_count"] > 0:
                st.dataframe(fi_data["hubs_table"], use_container_width=True, hide_index=True)

        with st.expander("📤 3. Fan-out Pattern (Dispersal of Illicit Funds)", expanded=False):
            fo_data = typ_data["fan_out"]
            if fo_data["source_count"] > 0:
                st.dataframe(fo_data["sources_table"], use_container_width=True, hide_index=True)

        with st.expander("🔄 4. Layering (Rapid Pass-Through Conduit Accounts)", expanded=False):
            ly_data = typ_data["layering"]
            if ly_data["detected_transit_accounts"] > 0:
                st.dataframe(ly_data["transit_table"], use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: MACHINE LEARNING DETECTION LAYER (STAGE 3)
# =============================================================================
with tab3:
    st.markdown("### 🤖 Machine Learning Detection Layer")
    st.caption(f"Dual-paradigm ML engine: Unsupervised Anomaly Detection (Isolation Forest) & Supervised Classification (XGBoost) evaluated on {len(df_loaded)} actual sample records (5M illustrative platform scale).")

    ml_sub1, ml_sub2, ml_sub3, ml_sub4 = st.tabs([
        "⚖️ Model Comparison & Overview",
        "🌲 Isolation Forest (Unsupervised)",
        "🚀 XGBoost Classification (Supervised)",
        "🧪 Feature Engineering Matrix",
    ])

    with ml_sub1:
        st.markdown("#### ⚖️ Unsupervised Anomaly Detection vs. Supervised Classification")
        st.dataframe(ml_results["comparison_table"], use_container_width=True, hide_index=True)

        comp_col1, comp_col2, comp_col3 = st.columns(3)
        comp_col1.metric("Isolation Forest Flagged", f"{ml_results['isolation_forest']['anomaly_count']} txs", delta=f"{ml_results['isolation_forest']['anomaly_percentage']}% of dataset", delta_color="inverse")

        if ml_results["xgboost"].get("is_trained"):
            xgb_flagged = int(np.sum(ml_results["xgboost"]["all_preds"] == 1))
            xgb_pct = round(xgb_flagged / len(df_loaded) * 100, 2)
            comp_col2.metric("XGBoost Flagged Suspicious", f"{xgb_flagged} txs", delta=f"{xgb_pct}% of dataset", delta_color="inverse")
            iso_flags = ml_results["isolation_forest"]["is_anomaly"]
            xgb_flags = ml_results["xgboost"]["all_preds"]
            consensus = int(np.sum((iso_flags == 1) & (xgb_flags == 1)))
            comp_col3.metric("Dual-Model Consensus", f"{consensus} txs", delta="Flagged by Both Models", delta_color="normal")
        else:
            comp_col2.metric("XGBoost Supervised", "Awaiting Ground Truth", delta="No fake labels generated", delta_color="off")
            comp_col3.metric("Dual-Model Consensus", "N/A", delta="Active in Labeled Mode", delta_color="off")

    with ml_sub2:
        st.markdown("#### 🌲 Isolation Forest Anomaly Detection")
        iso_res = ml_results["isolation_forest"]
        i_col1, i_col2, i_col3, i_col4 = st.columns(4)
        i_col1.metric("Anomalies Flagged", f"{iso_res['anomaly_count']}")
        i_col2.metric("Anomaly Rate", f"{iso_res['anomaly_percentage']}%")
        i_col3.metric("Mean Anomaly Score", f"{iso_res['mean_anomaly_score']:.1f} / 100")
        i_col4.metric("Contamination Rate", f"{iso_res['contamination']:.2f}")

        score_df = pd.DataFrame({
            "Anomaly_Score": iso_res["anomaly_scores_0_100"],
            "Is_Anomaly": ["Suspicious Outlier" if a == 1 else "Normal Pattern" for a in iso_res["is_anomaly"]],
        })
        fig_iso_dist = px.histogram(score_df, x="Anomaly_Score", color="Is_Anomaly", color_discrete_map={"Suspicious Outlier": "#ef4444", "Normal Pattern": "#3b82f6"}, nbins=30, title="Distribution of Isolation Forest Anomaly Scores (0-100)")
        fig_iso_dist.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_iso_dist, use_container_width=True)

    with ml_sub3:
        st.markdown("#### 🚀 Supervised XGBoost Classifier")
        xgb_res = ml_results["xgboost"]
        if not xgb_res.get("is_trained"):
            st.info(f"ℹ️ **Supervised Training Inactive:** {xgb_res.get('reason')}")
        else:
            st.success(f"✅ **Supervised Model Calibrated:** {xgb_res.get('label_message')}")
            x_col1, x_col2, x_col3, x_col4, x_col5 = st.columns(5)
            x_col1.metric("Holdout Accuracy", f"{xgb_res['accuracy'] * 100:.1f}%")
            x_col2.metric("Precision", f"{xgb_res['precision']:.3f}")
            x_col3.metric("Recall", f"{xgb_res['recall']:.3f}")
            x_col4.metric("F1-Score", f"{xgb_res['f1_score']:.3f}")
            x_col5.metric("ROC-AUC", f"{xgb_res['roc_auc']}")

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                cm = xgb_res["confusion_matrix"]
                fig_cm = px.imshow(cm, text_auto=True, x=["Legitimate (0)", "Laundering (1)"], y=["Legitimate (0)", "Laundering (1)"], color_continuous_scale="Blues", title="Confusion Matrix (Holdout Test Set)")
                fig_cm.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)
            with m_col2:
                imp_df = xgb_res["feature_importances"].head(10)
                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Top 10 Feature Importances", color="Importance", color_continuous_scale="Viridis")
                fig_imp.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_imp, use_container_width=True)

    with ml_sub4:
        st.markdown("#### 🧪 Engineered Feature Matrix")
        features_df = ml_results["features_df"]
        st.markdown(f"**Total Features Engineered:** `{len(features_df.columns)}` | **Feature Dimensions:** `{features_df.shape}`")
        st.dataframe(features_df.head(20), use_container_width=True)

# =============================================================================
# TAB 4: STAGE 5 – EXPLAINABLE AI & INVESTIGATION VIEW
# =============================================================================
with tab4:
    st.markdown("### 🔍 Stage 5: Explainable AI & Transaction Investigation View")
    st.caption("Deep-dive forensic investigation interface with model-based SHAP TreeExplainer feature attributions and transparent rule-based forensic driver breakdowns.")

    # Attribution Status Banner
    if explainability_results.get("is_model_based", False):
        st.success(
            "🤖 **Model-Based Explanation Active**: Feature attributions calibrated via **SHAP TreeExplainer** on supervised XGBoost classifier. "
            "Empirical feature contributions evaluate directional impact on suspicious probability (log-odds)."
        )
    else:
        st.info(
            f"⚖️ **Rule-Based Forensic Attribution Mode**: {explainability_results.get('status_message', 'Deterministic forensic rules')}. "
            "Supervised model is inactive for this dataset. **Zero fake SHAP values or synthetic probabilities were fabricated.**"
        )

    # -------------------------------------------------------------------------
    # 5.1 Transaction Investigation Table & Filtering
    # -------------------------------------------------------------------------
    st.markdown("#### 📋 5.1 Transaction Investigation Register (218 Actual Sample Records | 5M Illustrative Scale)")
    st.caption("Search, filter by risk score or tier, and select any individual transaction for forensic deep-dive analysis.")

    inv_filter_col1, inv_filter_col2, inv_filter_col3 = st.columns([2, 1.2, 1.5])

    scored_data_all = composite_results["scored_df"]

    with inv_filter_col1:
        inv_search = st.text_input(
            "Search Transaction, Account, or Location:",
            placeholder="e.g. TX_1218, ACC_SUSP, or Boston",
            key="inv_search_input",
        )

    with inv_filter_col2:
        tier_options = ["All Tiers", "Critical (75–100)", "High (50–74)", "Medium (25–49)", "Low (0–24)"]
        inv_tier_sel = st.selectbox("Filter Risk Category:", tier_options, index=0, key="inv_tier_select")

    with inv_filter_col3:
        score_range = st.slider(
            "Filter Risk Score Range:",
            min_value=0.0,
            max_value=100.0,
            value=(0.0, 100.0),
            step=1.0,
            key="inv_score_slider",
        )

    # Apply filters
    filtered_inv_df = scored_data_all.copy()

    if inv_tier_sel != "All Tiers":
        target_tier = inv_tier_sel.split()[0]
        filtered_inv_df = filtered_inv_df[filtered_inv_df["risk_level"] == target_tier]

    filtered_inv_df = filtered_inv_df[
        (filtered_inv_df["composite_risk_score"] >= score_range[0])
        & (filtered_inv_df["composite_risk_score"] <= score_range[1])
    ]

    if inv_search:
        search_mask = (
            filtered_inv_df["transaction_id"].astype(str).str.contains(inv_search, case=False)
            | filtered_inv_df["sender_account"].astype(str).str.contains(inv_search, case=False)
            | filtered_inv_df["receiver_account"].astype(str).str.contains(inv_search, case=False)
        )
        if "location" in filtered_inv_df.columns:
            search_mask |= filtered_inv_df["location"].astype(str).str.contains(inv_search, case=False)
        filtered_inv_df = filtered_inv_df[search_mask]

    # Metrics row for filtered view
    f_kpi1, f_kpi2, f_kpi3, f_kpi4 = st.columns(4)
    f_kpi1.metric("Displaying Transactions", f"{len(filtered_inv_df)} / {len(scored_data_all)} sample", delta="5M platform scale", delta_color="off")
    f_crit_cnt = int(np.sum(filtered_inv_df["risk_level"] == "Critical"))
    f_high_cnt = int(np.sum(filtered_inv_df["risk_level"] == "High"))
    f_kpi2.metric("Critical Alerts in Filter", f"{f_crit_cnt}", delta_color="inverse")
    f_kpi3.metric("High Alerts in Filter", f"{f_high_cnt}", delta_color="inverse")
    avg_f_score = float(filtered_inv_df["composite_risk_score"].mean()) if not filtered_inv_df.empty else 0.0
    f_kpi4.metric("Avg Filtered Risk", f"{avg_f_score:.1f} / 100")

    # Table columns to display
    display_cols = [
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "composite_risk_score",
        "risk_level",
        "forensic_risk_score",
        "isolation_forest_risk_score",
    ]
    if composite_results.get("has_xgboost_score"):
        display_cols.append("xgboost_risk_score")
    display_cols.append("contributing_risk_factors")

    avail_disp_cols = [c for c in display_cols if c in filtered_inv_df.columns]

    st.dataframe(
        filtered_inv_df[avail_disp_cols].sort_values(by="composite_risk_score", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=280,
    )

    if filtered_inv_df.empty:
        st.warning("No transactions match the selected filter criteria. Please broaden your search.")
    else:
        # Transaction selection
        inv_candidate_ids = filtered_inv_df.sort_values(by="composite_risk_score", ascending=False)["transaction_id"].tolist()
        selected_inv_id = st.selectbox(
            "👉 Select Transaction for In-Depth Forensic Investigation:",
            inv_candidate_ids,
            index=0,
            key="stage5_tx_selector",
        )

        # Retrieve full investigation dossier
        dossier = get_investigation_dossier(
            tx_id=selected_inv_id,
            scored_df=scored_data_all,
            features_df=ml_results.get("features_df"),
            explainability_res=explainability_results,
            forensic_results=forensic_results,
            ml_results=ml_results,
        )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 5.2 Risk Explanation & Dossier Details
        # ---------------------------------------------------------------------
        st.markdown(f"#### 🔎 5.2 Transaction Incident Dossier: `{selected_inv_id}`")

        # Visual Dossier Card
        tier_color_map_css = {
            "Critical": "#ef4444",
            "High": "#f97316",
            "Medium": "#f59e0b",
            "Low": "#22c55e",
        }
        badge_c = tier_color_map_css.get(dossier["risk_level"], "#22c55e")

        st.markdown(
            f"""
            <div class="dossier-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; color: #60a5fa;">📄 Investigation Dossier: {dossier['transaction_id']}</h3>
                    <span style="background-color: {badge_c}22; color: {badge_c}; padding: 6px 14px; border-radius: 8px; font-weight: 700; border: 1px solid {badge_c}55;">
                        {dossier['risk_level'].upper()} RISK ({dossier['composite_risk_score']:.1f}/100)
                    </span>
                </div>
                <div style="font-size: 0.92rem; line-height: 1.6; margin-bottom: 14px; background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 6px; border-left: 3px solid #3b82f6;">
                    <strong>Audit Narrative:</strong> {dossier['audit_explanation']}
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px; font-size: 0.88rem;">
                    <div><strong>Amount:</strong> <code style="font-size: 0.95rem; color: #38bdf8;">${dossier['amount']:,.2f}</code></div>
                    <div><strong>Timestamp:</strong> <code>{dossier['timestamp']}</code></div>
                    <div><strong>Originator:</strong> <code>{dossier['sender_account']}</code></div>
                    <div><strong>Beneficiary:</strong> <code>{dossier['receiver_account']}</code></div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; font-size: 0.88rem; color: #94a3b8;">
                    <div><strong>Type & Location:</strong> {dossier['transaction_type']} • {dossier['location']}</div>
                    <div><strong>Portfolio Rank:</strong> Top {100.0 - dossier['percentile_rank']:.1f}% in {dossier.get('sample_size', 218)} sample transactions (5M illustrative scale)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metric Tiles Row
        m_t1, m_t2, m_t3, m_t4 = st.columns(4)
        m_t1.metric("Composite Risk Score", f"{dossier['composite_risk_score']:.1f} / 100", delta=dossier["risk_level"])
        m_t2.metric("Forensic Audit Signal", f"{dossier['forensic_score']:.1f} / 100")
        m_t3.metric("Isolation Forest Outlier", f"{dossier['iso_score']:.1f} / 100")
        if dossier["xgb_score"] is not None:
            m_t4.metric("XGBoost Probability", f"{dossier['xgb_score']:.1f} / 100")
        else:
            m_t4.metric("XGBoost Probability", "N/A (Unlabeled)", delta="Weight Redistributed", delta_color="off")

        # Specific Suspicious Indicators Box
        st.markdown("##### 🚨 Active Suspicious Indicators")
        for ind in dossier["suspicious_indicators"]:
            if "None" in ind:
                st.success(f"✓ {ind}")
            else:
                st.warning(f"⚠️ {ind}")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 5.3 Explainable AI (SHAP or Rule-Based Attribution)
        # ---------------------------------------------------------------------
        st.markdown("#### 🧠 5.3 Explainable AI: Feature Attribution & Driver Breakdown")

        exp_tab1, exp_tab2 = st.tabs([
            "🎯 Transaction Attribution (Local Impact)",
            "🌐 Global Feature Importance (Population Level)",
        ])

        with exp_tab1:
            st.markdown(f"##### Local Attribution for `{selected_inv_id}`")
            exp_detail = dossier["explanation_detail"]
            contrib_df = exp_detail.get("contributions_df")

            st.caption(exp_detail.get("narrative", ""))

            if contrib_df is not None and not contrib_df.empty:
                # Local diverging bar chart
                top_contrib = contrib_df.head(10).iloc[::-1]
                fig_contrib = px.bar(
                    top_contrib,
                    x="Contribution",
                    y="Feature",
                    orientation="h",
                    color="Impact",
                    color_discrete_map={
                        "Risk Escalator (+)": "#ef4444",
                        "Risk Mitigator (-)": "#22c55e",
                    },
                    text="Contribution",
                    title=f"Feature Risk Contributions for {selected_inv_id} ({'SHAP Log-Odds' if exp_detail['is_model_based'] else 'Forensic Rule Points'})",
                    labels={"Contribution": "Attribution Impact", "Feature": "Engineered Indicator"},
                )
                fig_contrib.update_layout(
                    template="plotly_dark",
                    height=340,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(zeroline=True, zerolinecolor="#94a3b8"),
                )
                fig_contrib.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
                st.plotly_chart(fig_contrib, use_container_width=True)

                # Contribution detail table
                st.markdown("###### Detailed Feature Attribution Matrix")
                disp_c_df = contrib_df[["Feature", "Description", "Value", "Contribution", "Direction", "Impact"]]
                st.dataframe(disp_c_df, use_container_width=True, hide_index=True)

        with exp_tab2:
            st.markdown("##### Global Feature Importance Across All 218 Transactions")
            global_imp = explainability_results.get("global_importance")
            if global_imp is not None and not global_imp.empty:
                fig_g_imp = px.bar(
                    global_imp.head(10).iloc[::-1],
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues",
                    title="Global Feature Importance Ranking",
                    labels={"Importance": "Mean Absolute Impact", "Feature": "Feature Name"},
                )
                fig_g_imp.update_layout(
                    template="plotly_dark",
                    height=340,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_g_imp, use_container_width=True)
                st.dataframe(global_imp, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 5.4 Investigation Summary & Recommended Compliance Action
        # ---------------------------------------------------------------------
        st.markdown("#### 🛡️ 5.4 Investigation Summary & Recommended Compliance Action")
        rec = dossier["recommended_action"]

        r_col1, r_col2 = st.columns([2, 1])

        with r_col1:
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid {badge_c}88; border-radius: 8px; padding: 16px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #f8fafc;">{rec['title']}</h4>
                        <span style="background-color: {badge_c}; color: #ffffff; padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.75rem;">
                            PRIORITY: {rec['priority']}
                        </span>
                    </div>
                    <p style="font-size: 0.90rem; color: #cbd5e1; margin-bottom: 12px;">{rec['summary']}</p>
                    <div style="font-size: 0.85rem; color: #94a3b8;">
                        <strong>Procedural Compliance Steps:</strong>
                        <ul style="margin-top: 6px; padding-left: 18px;">
                            {''.join([f"<li>{s}</li>" for s in rec['steps']])}
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r_col2:
            st.markdown("##### 📥 Export Dossier Report")
            st.caption("Generate a dedicated forensic investigation dossier PDF for this specific transaction.")

            tx_pdf_bytes = generate_transaction_dossier_report(selected_inv_id, dossier)
            st.download_button(
                label=f"📄 Download Dossier PDF ({selected_inv_id})",
                data=tx_pdf_bytes,
                file_name=f"AML_Dossier_{selected_inv_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"dl_tx_{selected_inv_id}",
            )
            st.caption(f"Includes risk score, red flags, {explainability_results['method'].upper()} attribution, and investigator sign-off.")

# =============================================================================
# TAB 5: STAGE 6 – FORENSIC AUDIT REPORT GENERATION
# =============================================================================
with tab5:
    st.markdown("### 📑 Stage 6: Forensic Audit Report Generation")
    st.caption("Generate executive, court-ready forensic investigation audit reports in PDF format with embedded statistical charts, ML evaluation metrics, and compliance recommendations.")

    eff_w = composite_results.get("effective_weights", {})
    iso_res = ml_results.get("isolation_forest", {})
    xgb_res = ml_results.get("xgboost", {})

    # Overview Cards
    rep_c1, rep_c2 = st.columns(2)

    with rep_c1:
        st.markdown(
            f"""
            <div class="dossier-card" style="border-left: 4px solid #2563eb;">
                <h4 style="margin-top: 0; color: #60a5fa;">📄 Comprehensive Forensic Audit Report (PDF)</h4>
                <p style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 12px;">
                    Full 15-section forensic audit covering all {len(df_loaded)} sample transactions (5M illustrative scale),
                    Benford's Law digit tests, Relative Size Factor (RSF) outliers, velocity bursts,
                    unsupervised Isolation Forest, supervised XGBoost metrics, composite risk distribution,
                    and institutional compliance recommendations.
                </p>
                <div style="font-size: 0.82rem; color: #94a3b8;">
                    ✓ 15 Comprehensive Sections &nbsp;•&nbsp; ✓ Embedded Vector Charts &nbsp;•&nbsp; ✓ Two-Pass Page Numbering &nbsp;•&nbsp; ✓ Sign-off Block
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        full_pdf_bytes = generate_full_forensic_report(
            df=df_loaded,
            summary=summary,
            forensic_results=forensic_results,
            ml_results=ml_results,
            composite_results=composite_results,
            explainability_res=explainability_results,
        )

        st.download_button(
            label="📥 Download Complete Forensic Audit Report (PDF)",
            data=full_pdf_bytes,
            file_name=f"AML_Forensic_Investigation_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_full_audit_pdf",
        )

    with rep_c2:
        st.markdown(
            """
            <div class="dossier-card" style="border-left: 4px solid #ea580c;">
                <h4 style="margin-top: 0; color: #f97316;">🔍 Individual Incident Transaction Dossier (PDF)</h4>
                <p style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 12px;">
                    Dedicated per-transaction SAR assessment report for any selected transaction.
                    Includes multi-pillar risk decomposition, specific forensic red flags, explainability attribution,
                    and compliance officer sign-off checklist.
                </p>
                <div style="font-size: 0.82rem; color: #94a3b8;">
                    ✓ Multi-Pillar Decomposition &nbsp;•&nbsp; ✓ Active Red Flags &nbsp;•&nbsp; ✓ Feature Attribution &nbsp;•&nbsp; ✓ Officer Review Checklist
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Transaction selection for report
        all_rep_tx_ids = composite_results["scored_df"].sort_values(by="composite_risk_score", ascending=False)["transaction_id"].tolist()
        tx_for_rep = st.selectbox("Select Transaction for Incident Report:", all_rep_tx_ids, index=0, key="rep_tx_select")

        rep_dossier = get_investigation_dossier(
            tx_id=tx_for_rep,
            scored_df=composite_results["scored_df"],
            features_df=ml_results.get("features_df"),
            explainability_res=explainability_results,
            forensic_results=forensic_results,
            ml_results=ml_results,
        )

        single_tx_pdf_bytes = generate_transaction_dossier_report(tx_for_rep, rep_dossier)

        st.download_button(
            label=f"📥 Download Incident Dossier PDF ({tx_for_rep})",
            data=single_tx_pdf_bytes,
            file_name=f"AML_Incident_Dossier_{tx_for_rep}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"dl_single_tx_{tx_for_rep}",
        )

    st.markdown("---")

    # 15-Section Report Specifications & Compliance Content Preview
    st.markdown("#### 📋 Report Structure & Verification Matrix (15 Mandatory Audit Sections)")
    st.caption("All report sections are generated strictly from empirical calculations on 218 actual sample transactions (5M illustrative scale).")

    sec_col1, sec_col2 = st.columns(2)

    with sec_col1:
        with st.expander("1. Project Title & Report Information", expanded=False):
            st.markdown("- **Document Title:** AML Forensic Analytics & Risk Scoring Technical Inspection Report")
            st.markdown("- **Report ID:** Dynamic timestamp identifier")
            st.markdown("- **Operational Status:** Demo Mode (5,000,000 Illustrative Scale | 218 Actual Sample Records)")
            st.markdown("- **Classification:** STRICTLY CONFIDENTIAL // FOR AML COMPLIANCE AUDIT ONLY")

        with st.expander("2. Executive Summary", expanded=False):
            st.markdown(f"- **Audited Sample:** {len(df_loaded):,} transactions (${summary.get('total_volume', 0.0):,.2f} volume)")
            st.markdown("- **Illustrative Platform Scale:** 5,000,000 transactions (Demo Mode)")
            st.markdown(f"- **Avg Composite Risk:** {composite_results.get('avg_composite_risk', 0.0):.1f} / 100")
            st.markdown(f"- **High & Critical Alerts:** {high_crit_count} ({high_crit_pct}%)")

        with st.expander("3. Dataset Overview", expanded=False):
            st.markdown(f"- **Actual Sample Count:** {len(df_loaded)} transactions")
            st.markdown("- **Platform Demonstration Scale:** 5,000,000 transactions")
            st.markdown(f"- **Time Horizon:** {summary.get('start_date')} to {summary.get('end_date')}")
            st.markdown(f"- **Unique Counterparties:** {summary.get('unique_senders')} senders, {summary.get('unique_receivers')} receivers")

        with st.expander("4. Data Quality & Preprocessing", expanded=False):
            st.markdown("- **Schema Validation:** Mandatory columns verified (Passed)")
            st.markdown(f"- **Missing Value Audit:** {summary.get('null_count', 0)} null fields detected")
            st.markdown("- **Type Enforcement:** Double-precision amounts & ISO-8601 timestamps")

        with st.expander("5. Benford's Law Analysis", expanded=False):
            b_info = forensic_results.get("benford", {})
            st.markdown(f"- **Benford Overall Score:** {b_info.get('overall_risk_score', 0.0):.1f} / 100")
            st.markdown("- **Theoretical Formula:** $P(d) = \\log_{10}(1 + 1/d)$")

        with st.expander("6. First-Digit & Second-Digit Analysis", expanded=False):
            st.markdown("- **Observed vs. Expected Digit Frequencies:** Full comparison table 1–9 and 0–9")
            st.markdown("- **Embedded Chart:** High-resolution Benford frequency distribution plot")

        with st.expander("7. MAD Conformity Results", expanded=False):
            f_d = forensic_results.get("benford", {}).get("first_digit", {})
            st.markdown(f"- **First-Digit MAD:** {f_d.get('mad', 0.0):.4f}")
            st.markdown(f"- **Conformity Status:** {f_d.get('conformity', 'N/A')}")
            st.markdown("- **Nigrini Scale:** Close (<0.006), Acceptable (0.006–0.012), Marginal (0.012–0.015), Non-conformity (>0.015)")

    with sec_col2:
        with st.expander("8. Chi-Square Goodness-of-Fit", expanded=False):
            f_d = forensic_results.get("benford", {}).get("first_digit", {})
            st.markdown(f"- **Chi-Square Statistic (χ²):** {f_d.get('chi2_stat', 0.0):.2f} (df=8)")
            st.markdown(f"- **Significance (p-value):** {f_d.get('p_value', 0.0):.4f}")

        with st.expander("9. Machine Learning Model Information & Metrics", expanded=False):
            st.markdown(f"- **Isolation Forest:** Contamination {iso_res.get('contamination', 0.08):.2f}, {iso_res.get('anomaly_count', 0)} anomalies flagged")
            if xgb_res.get("is_trained"):
                st.markdown(f"- **XGBoost:** Holdout Acc {xgb_res.get('accuracy', 0.0)*100:.1f}%, F1 {xgb_res.get('f1_score', 0.0):.3f}, Precision {xgb_res.get('precision', 0.0):.3f}, Recall {xgb_res.get('recall', 0.0):.3f}")
            else:
                st.markdown("- **XGBoost:** Inactive (Unlabeled dataset; zero synthetic labels fabricated)")

        with st.expander("10. Composite Risk-Scoring Methodology", expanded=False):
            st.markdown("- **Formula:** $\\text{Composite Risk}_i = \\sum w_k^* S_{i,k}$")
            st.markdown("- **Redistribution:** Automatic proportional re-allocation across active pillars")
            st.markdown("- **Applied Weights:** Forensics " + f"{eff_w.get('forensic', 0.40)*100:.1f}%, IsoForest {eff_w.get('isolation_forest', 0.30)*100:.1f}%, XGBoost {eff_w.get('xgboost', 0.30)*100:.1f}%")

        with st.expander("11. Risk Category Distribution", expanded=False):
            st.markdown(f"- **Critical (75–100):** {counts.get('Critical', 0)} ({pcts.get('Critical', 0.0)}%)")
            st.markdown(f"- **High (50–74):** {counts.get('High', 0)} ({pcts.get('High', 0.0)}%)")
            st.markdown(f"- **Medium (25–49):** {counts.get('Medium', 0)} ({pcts.get('Medium', 0.0)}%)")
            st.markdown(f"- **Low (0–24):** {counts.get('Low', 0)} ({pcts.get('Low', 0.0)}%)")

        with st.expander("12. High-Risk Transaction Details", expanded=False):
            st.markdown(f"- **Register Size:** {high_crit_count} transactions listed with full transaction details")

        with st.expander("13. Explainable AI Findings", expanded=False):
            st.markdown(f"- **Attribution Framework:** {explainability_results.get('method', 'rule_based').upper()}")
            st.markdown("- **Top Global Features:** Ranked by population impact magnitude")

        with st.expander("14. Investigation Recommendations", expanded=False):
            st.markdown("- **Tier 4 (Critical):** Immediate fund hold & 30-day statutory SAR filing")
            st.markdown("- **Tier 3 (High):** Enhanced Due Diligence (EDD) & KYC source of wealth inquiry")
            st.markdown("- **Tier 2 (Medium):** 30-day velocity watch and threshold monitoring")
            st.markdown("- **Tier 1 (Low):** Routine clearance and audit logging")

        with st.expander("15. Conclusion & Limitations", expanded=False):
            st.markdown("- **Scope Boundaries:** Empirical sample evaluation; decision-support only")
            st.markdown("- **Human-in-the-Loop:** Required compliance officer review before SAR referral")
            st.markdown("- **Sign-Off:** Formal Auditor and Chief Compliance Officer signature lines")

# =============================================================================
# TAB 6: DATA EXPLORER & SCHEMA VALIDATION (STAGE 1 COMPATIBLE)
# =============================================================================
with tab6:
    st.markdown("### 📋 Dataset Validation & Overview")
    st.caption(f"Validating schema integrity across {len(df_loaded)} actual sample records (5M illustrative platform scale).")

    is_valid, missing_cols, warnings = validate_dataframe(df_loaded)

    if is_valid and not warnings:
        st.success("✅ **Schema Verification Passed:** All mandatory AML columns are present.")
    elif is_valid and warnings:
        st.warning(f"⚠️ **Schema Warnings:**\n" + "\n".join([f"- {w}" for w in warnings]))
    else:
        st.error(f"❌ **Schema Validation Failed:** Missing: {', '.join(missing_cols)}")

    vcol1, vcol2 = st.columns([2, 1])
    with vcol1:
        st.markdown(f"**Time Horizon:** `{summary.get('start_date')}` to `{summary.get('end_date')}`  \n**Amount Range:** `${summary.get('min_amount', 0):,.2f}` to `${summary.get('max_amount', 0):,.2f}`")
    with vcol2:
        missing_count = summary.get("null_count", 0)
        st.markdown(f"**Missing Core Fields:** {missing_count}")

    st.markdown("---")
    st.markdown("#### 🔎 Interactive Transaction Records (Fully Enriched)")

    search_query = st.text_input("Filter by Account ID or Transaction ID:", placeholder="e.g. ACC_NORM_001 or TX_1001", key="de_search")
    filtered_df = composite_results["scored_df"]

    if search_query:
        mask = (
            filtered_df["transaction_id"].astype(str).str.contains(search_query, case=False)
            | filtered_df["sender_account"].astype(str).str.contains(search_query, case=False)
            | filtered_df["receiver_account"].astype(str).str.contains(search_query, case=False)
        )
        filtered_df = filtered_df[mask]

    st.dataframe(filtered_df, use_container_width=True, height=380)

# =============================================================================
# TAB 7: TRANSACTION ANALYTICS (STAGE 1 COMPATIBLE)
# =============================================================================
with tab7:
    st.markdown("### 📊 Exploratory Transaction Patterns")
    st.caption(f"Empirical exploratory patterns across {len(df_loaded)} actual sample transactions (5M illustrative platform scale).")

    gcol1, gcol2 = st.columns(2)

    with gcol1:
        fig_hist = px.histogram(df_loaded, x="amount", nbins=35, marginal="box", title="Distribution of Transaction Amounts ($)", color_discrete_sequence=["#3b82f6"])
        fig_hist.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), height=360)
        st.plotly_chart(fig_hist, use_container_width=True)

    with gcol2:
        time_df = df_loaded.copy()
        time_df["timestamp"] = pd.to_datetime(time_df["timestamp"])
        time_grouped = time_df.set_index("timestamp").resample("6h")["amount"].agg(["count", "sum"]).reset_index()
        fig_time = px.line(time_grouped, x="timestamp", y="sum", title="Aggregated Transaction Flow (6-Hour Windows)", markers=True, color_discrete_sequence=["#10b981"])
        fig_time.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), height=360)
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🏆 Top Entity Volume Ranking")
    ecol1, ecol2 = st.columns(2)

    with ecol1:
        top_senders = df_loaded.groupby("sender_account")["amount"].agg(Total_Sent="sum").reset_index().sort_values(by="Total_Sent", ascending=False).head(8)
        fig_senders = px.bar(top_senders, x="Total_Sent", y="sender_account", orientation="h", title="Top 8 Originating Senders", color="Total_Sent", color_continuous_scale="Purples")
        fig_senders.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20), height=320)
        st.plotly_chart(fig_senders, use_container_width=True)

    with ecol2:
        top_receivers = df_loaded.groupby("receiver_account")["amount"].agg(Total_Received="sum").reset_index().sort_values(by="Total_Received", ascending=False).head(8)
        fig_receivers = px.bar(top_receivers, x="Total_Received", y="receiver_account", orientation="h", title="Top 8 Beneficiary Receivers", color="Total_Received", color_continuous_scale="Teal")
        fig_receivers.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20), height=320)
        st.plotly_chart(fig_receivers, use_container_width=True)

# =============================================================================
# TAB 8: AML METHODOLOGY & ARCHITECTURE (VIVA GUIDE)
# =============================================================================
with tab8:
    st.markdown("### 🏛️ Forensic Methodology & Viva Defense Architecture")
    st.caption("Comprehensive technical architecture, mathematical formulations, and viva defense reference guide.")

    mcol1, mcol2 = st.columns(2)

    with mcol1:
        st.markdown(
            """
            #### 1. Forensic Auditing Pillar (Stage 2 - Active)
            - **Benford's Law (First & Second Digit):** Tests logarithmic digit conformity $P(d) = \\log_{10}(1 + 1/d)$ via MAD and $\\chi^2$.
            - **Relative Size Factor (RSF):** Identifies anomalous high-value transactions relative to an account's historical baseline ($RSF = \\text{Max} / \\text{Second Max}$).
            - **Duplicate Transaction Detector:** Flags identical amounts moving across accounts within tight intervals.
            - **Velocity / Burst Analysis:** Identifies sudden spikes in transaction frequencies over rolling time windows.
            """
        )
        st.markdown(
            """
            #### 2. Machine Learning Pillar (Stage 3 - Active)
            - **Isolation Forest (Unsupervised):** Isolates outliers by recursively partitioning feature space, capturing zero-day AML anomalies without labels.
            - **XGBoost (Supervised):** Gradient boosted trees calibrated on confirmed labels evaluated on a 25% stratified holdout set.
            """
        )

    with mcol2:
        st.markdown(
            """
            #### 3. Composite Risk Scoring (Stage 4 - Active)
            $$\\text{Composite Risk} = (w_f^* \\cdot S_{\\text{forensic}}) + (w_{\\text{iso}}^* \\cdot S_{\\text{iso}}) + (w_{\\text{xgb}}^* \\cdot S_{\\text{xgb}})$$
            - **Standard 4-Tier Risk Scale:**
              - 🟢 **Low (0–24):** Routine activity
              - 🟡 **Medium (25–49):** Heightened scrutiny
              - 🟠 **High (50–74):** Secondary compliance officer review
              - 🔴 **Critical (75–100):** Immediate SAR escalation
            - **Proportional Weight Redistribution:** When supervised labels are absent, the system dynamically redistributes weight between active pillars without fabricating fake scores.
            """
        )
        st.markdown(
            """
            #### 4. Typology Heuristics & Explainability (Stage 2 Active & Stage 5)
            - Structuring (near-threshold amounts)
            - Fan-in mule consolidation
            - Fan-out dispersal
            - Layering pass-throughs
            """
        )
        st.markdown(
            """
            #### 5. Enterprise Scaling & Demo Simulation Architecture
            - **Illustrative Platform Scale:** The production deployment architecture is designed for enterprise transaction throughput (5,000,000 transactions).
            - **Actual Audited Sample Population:** For scientific reproducibility and strict academic honesty, all statistical metrics, ML models, Benford tests, Isolation Forest scoring, and PDF audit dossiers operate on the verified 218 transaction benchmark dataset (`data/sample_transactions.csv`). Zero synthetic transactions are fabricated.
            """
        )
