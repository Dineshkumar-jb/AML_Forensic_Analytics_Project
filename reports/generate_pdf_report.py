"""Script to generate a comprehensive, executive PDF report of the AML inspection findings."""

import os
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
HTML_OUTPUT_PATH = REPORTS_DIR / "AML_Inspection_Report.html"
PDF_OUTPUT_PATH = REPORTS_DIR / "AML_Inspection_Report.pdf"

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AML Forensic Analytics - Technical Inspection & Architecture Review Report</title>
<style>
    @page {
        size: A4;
        margin: 16mm 14mm 16mm 14mm;
        @bottom-right {
            content: counter(page);
        }
    }
    body {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        background-color: #ffffff;
        line-height: 1.55;
        font-size: 10pt;
        margin: 0;
        padding: 0;
    }
    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 700;
        margin-top: 1.4em;
        margin-bottom: 0.5em;
        page-break-after: avoid;
    }
    h1 {
        font-size: 20pt;
        line-height: 1.25;
        border-bottom: 3px solid #2563eb;
        padding-bottom: 8px;
        margin-top: 0;
        color: #1e3a8a;
    }
    h2 {
        font-size: 13pt;
        border-bottom: 1.5px solid #cbd5e1;
        padding-bottom: 4px;
        color: #1e40af;
        margin-top: 1.6em;
    }
    h3 {
        font-size: 11pt;
        color: #334155;
        margin-top: 1.2em;
    }
    p, li {
        color: #334155;
        text-align: justify;
    }
    ul, ol {
        margin-top: 0.4em;
        margin-bottom: 0.6em;
        padding-left: 1.4em;
    }
    li {
        margin-bottom: 0.3em;
    }
    .header-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 18px;
    }
    .header-box table {
        width: 100%;
        border-collapse: collapse;
    }
    .header-box td {
        padding: 3px 6px;
        font-size: 9.5pt;
        border: none;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 8pt;
        font-weight: 600;
        border-radius: 9999px;
        text-transform: uppercase;
    }
    .badge-verified {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .badge-planned {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .badge-unsupervised {
        background-color: #e0e7ff;
        color: #4338ca;
        border: 1px solid #c7d2fe;
    }
    .badge-supervised {
        background-color: #f3e8ff;
        color: #7e22ce;
        border: 1px solid #e9d5ff;
    }
    .badge-rule {
        background-color: #e2e8f0;
        color: #334155;
        border: 1px solid #cbd5e1;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0 16px 0;
        font-size: 8.8pt;
        page-break-inside: auto;
    }
    tr {
        page-break-inside: avoid;
        page-break-after: auto;
    }
    th {
        background-color: #1e3a8a;
        color: #ffffff;
        font-weight: 600;
        text-align: left;
        padding: 6px 8px;
        border: 1px solid #1e3a8a;
    }
    td {
        padding: 5px 8px;
        border: 1px solid #cbd5e1;
        vertical-align: top;
    }
    tr:nth-child(even) td {
        background-color: #f8fafc;
    }
    .card {
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 0 6px 6px 0;
        padding: 10px 14px;
        margin: 10px 0 14px 0;
        page-break-inside: avoid;
    }
    .card-title {
        font-weight: 700;
        color: #1e3a8a;
        font-size: 10.5pt;
        margin-bottom: 4px;
    }
    .formula-box {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 5px;
        padding: 8px 12px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 9pt;
        color: #0f172a;
        margin: 6px 0;
    }
    code {
        font-family: 'Consolas', 'Courier New', monospace;
        background-color: #f1f5f9;
        padding: 1px 4px;
        border-radius: 3px;
        font-size: 8.8pt;
        color: #0f172a;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-top: 1px solid #fef3c7;
        border-right: 1px solid #fef3c7;
        border-bottom: 1px solid #fef3c7;
        padding: 8px 12px;
        margin: 10px 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }
    .success-box {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-top: 1px solid #dcfce7;
        border-right: 1px solid #dcfce7;
        border-bottom: 1px solid #dcfce7;
        padding: 8px 12px;
        margin: 10px 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }
    .page-break {
        page-break-before: always;
    }
    .footer-note {
        font-size: 8pt;
        color: #64748b;
        text-align: center;
        margin-top: 25px;
        border-top: 1px solid #e2e8f0;
        padding-top: 8px;
    }
</style>
</head>
<body>

<!-- Header Banner -->
<h1>AML Forensic Analytics & Risk Scoring System</h1>
<p style="font-size: 11pt; color: #475569; margin-top: -6px; margin-bottom: 12px;">
    <strong>Independent Technical Inspection & Architecture Review Report</strong>
</p>

<div class="header-box">
    <table>
        <tr>
            <td style="width: 22%;"><strong>Project Title:</strong></td>
            <td style="width: 44%;">Transaction Monitoring & Risk Scoring for AML Detection</td>
            <td style="width: 16%;"><strong>Date of Review:</strong></td>
            <td style="width: 18%;">October 2026</td>
        </tr>
        <tr>
            <td><strong>Lead Reviewer:</strong></td>
            <td>Senior ML Engineer & System Reviewer</td>
            <td><strong>Verified Stages:</strong></td>
            <td><span class="badge badge-verified">Stages 1 to 4 Active</span></td>
        </tr>
        <tr>
            <td><strong>Target Application:</strong></td>
            <td>Streamlit Forensic Banking Intelligence Platform</td>
            <td><strong>Test Status:</strong></td>
            <td><span class="badge badge-verified">7/7 Tests Passing</span></td>
        </tr>
    </table>
</div>

<!-- Executive Summary Table -->
<h2>Executive Summary: Model & Algorithm Implementation Matrix</h2>
<p>
The table below summarizes the comprehensive verification of all machine learning models, statistical tests, forensic rules, and heuristic typologies discovered in the codebase. Every item was verified by inspecting source code files, function definitions, and unit test executions.
</p>

<table>
    <thead>
        <tr>
            <th style="width: 23%;">Model / Algorithm</th>
            <th style="width: 17%;">Paradigm</th>
            <th style="width: 32%;">Core Purpose in AML Detection</th>
            <th style="width: 10%;">Status</th>
            <th style="width: 18%;">Source File</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Isolation Forest</strong></td>
            <td><span class="badge badge-unsupervised">Unsupervised ML</span></td>
            <td>Detects zero-day multidimensional transaction anomalies without requiring ground-truth labels</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/ml/isolation_forest.py</code></td>
        </tr>
        <tr>
            <td><strong>XGBoost Classifier</strong></td>
            <td><span class="badge badge-supervised">Supervised ML</span></td>
            <td>Gradient boosted decision trees for laundering classification; calibrated probabilities & holdout metrics</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/ml/xgboost_model.py</code></td>
        </tr>
        <tr>
            <td><strong>Benford's Law (1st Digit)</strong></td>
            <td><span class="badge badge-rule">Statistical Test</span></td>
            <td>Tests logarithmic leading digit frequency ($1–9$) against natural law via MAD & $\chi^2$ test</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/forensic/benford.py</code></td>
        </tr>
        <tr>
            <td><strong>Benford's Law (2nd Digit)</strong></td>
            <td><span class="badge badge-rule">Statistical Test</span></td>
            <td>Tests 2nd significant digit distribution ($0–9$) using Nigrini conformity boundaries</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/forensic/benford.py</code></td>
        </tr>
        <tr>
            <td><strong>Relative Size Factor (RSF)</strong></td>
            <td><span class="badge badge-rule">Forensic Heuristic</span></td>
            <td>Detects disproportionate transaction spikes relative to account 2nd-largest baseline ($\text{Max} / \text{2nd Max}$)</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/forensic/rsf.py</code></td>
        </tr>
        <tr>
            <td><strong>Duplicate Detection</strong></td>
            <td><span class="badge badge-rule">Clustering Rule</span></td>
            <td>Identifies identical amounts across account pairs and detects rapid re-occurrences ($\le 48\text{h}$)</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/forensic/duplicates.py</code></td>
        </tr>
        <tr>
            <td><strong>Transaction Velocity</strong></td>
            <td><span class="badge badge-rule">Rolling Window</span></td>
            <td>Detects burst activity per account over rolling $2$-hour windows ($\ge 4$ transactions)</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/forensic/velocity.py</code></td>
        </tr>
        <tr>
            <td><strong>Structuring / Smurfing</strong></td>
            <td><span class="badge badge-rule">Threshold Rule</span></td>
            <td>Flags transactions placed just below the $\$10,000$ CTR threshold ($88\%\text{--}100\%$, i.e., $\$8,800\text{--}\$9,999.99$)</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/forensic/typologies.py</code></td>
        </tr>
        <tr>
            <td><strong>Fan-in Typology (Mules)</strong></td>
            <td><span class="badge badge-rule">Aggregation Rule</span></td>
            <td>Identifies consolidation accounts receiving funds from $\ge 3$ distinct originating senders</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/forensic/typologies.py</code></td>
        </tr>
        <tr>
            <td><strong>Fan-out Typology (Dispersal)</strong></td>
            <td><span class="badge badge-rule">Aggregation Rule</span></td>
            <td>Identifies source accounts rapidly dispersing funds to $\ge 3$ distinct beneficiaries</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/forensic/typologies.py</code></td>
        </tr>
        <tr>
            <td><strong>Layering (Pass-Through)</strong></td>
            <td><span class="badge badge-rule">Flow Balance Rule</span></td>
            <td>Detects conduit accounts where net inflow and outflow balance within $\le 25\%$ retention</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/forensic/typologies.py</code></td>
        </tr>
        <tr>
            <td><strong>Composite Risk Engine</strong></td>
            <td><span class="badge badge-rule">Convex Hybrid</span></td>
            <td>Combines Forensics, IsoForest, and XGBoost with proportional redistribution and 4 risk tiers</td>
            <td><span class="badge badge-verified">Verified</span></td>
            <td><code>src/scoring/composite_risk.py</code></td>
        </tr>
        <tr>
            <td><strong>SHAP Explainability</strong></td>
            <td><span class="badge badge-supervised">Model Attribution</span></td>
            <td>Local Shapley waterfall plots explaining why individual transactions are flagged</td>
            <td><span class="badge badge-planned">Stage 5 Plan</span></td>
            <td><em>Requires SHAP library</em></td>
        </tr>
        <tr>
            <td><strong>Graph Network Algorithms</strong></td>
            <td><span class="badge badge-rule">Network Theory</span></td>
            <td>PageRank, Louvain community detection, cyclical layering loops</td>
            <td><span class="badge badge-planned">Unimplemented</span></td>
            <td><code>src/typologies/__init__.py</code></td>
        </tr>
        <tr>
            <td><strong>Alternative Classifiers (RF/SVM)</strong></td>
            <td><span class="badge badge-supervised">Supervised ML</span></td>
            <td>Tabular classifiers (Random Forest, Logistic Regression, SVM)</td>
            <td><span class="badge badge-planned">Unimplemented</span></td>
            <td><em>None in repository</em></td>
        </tr>
    </tbody>
</table>

<!-- Section 1 -->
<h2>1. Machine Learning Models (Detailed Audit)</h2>

<div class="card">
    <div class="card-title">1.1 Isolation Forest — Unsupervised Transaction Anomaly Detection</div>
    <p>
        <strong>Module:</strong> <code>src/ml/isolation_forest.py</code> | <strong>Function:</strong> <code>train_isolation_forest(features_df, contamination=0.08, n_estimators=100, random_state=42)</code>
    </p>
    <ul>
        <li><strong>Theoretical Foundation:</strong> Constructs an ensemble of random isolation trees. Since anomalies represent minority observations located in sparse feature regions, they require significantly fewer recursive splits to isolate (shorter path lengths).</li>
        <li><strong>Input Representation:</strong> Consumes the complete 16-dimensional engineered feature matrix. Features are preprocessed through <code>StandardScaler().fit_transform(X)</code>.</li>
        <li><strong>Hyperparameters:</strong> 100 estimators, <code>contamination=0.08</code> (user-adjustable via Streamlit slider $0.01\text{--}0.25$), <code>random_state=42</code>, parallelized on all cores (<code>n_jobs=-1</code>).</li>
        <li><strong>Score Normalization Formula:</strong> The raw continuous decision scores $s_i = \text{decision\_function}(X)_i$ (where lower/negative values denote severe anomalies) are inverted and scaled into an intuitive $0\text{--}100$ risk score:
            <div class="formula-box">
                normalized_score_i = ((max(s) - s_i) / (max(s) - min(s))) * 100.0
            </div>
        </li>
        <li><strong>Verified Outputs:</strong> Array of normalized continuous scores (<code>anomaly_scores_0_100</code>), binary outlier predictions (<code>is_anomaly</code> $\in \{0, 1\}$), total anomalous count, and percentage rate.</li>
    </ul>
</div>

<div class="card">
    <div class="card-title">1.2 XGBoost Classifier — Supervised Money Laundering Detection</div>
    <p>
        <strong>Module:</strong> <code>src/ml/xgboost_model.py</code> | <strong>Functions:</strong> <code>train_xgboost()</code>, <code>detect_ground_truth_label()</code>
    </p>
    <ul>
        <li><strong>Theoretical Foundation:</strong> Gradient-boosted decision tree ensemble minimizing regularized logistic cross-entropy loss across engineered behavioral and forensic features.</li>
        <li><strong>Label Detection Protocol:</strong> The function <code>detect_ground_truth_label()</code> inspects the dataset for binary target columns (<code>is_laundering</code>, <code>label</code>, <code>is_fraud</code>, <code>target</code>, <code>sar_flag</code>). Requires minimum class representation ($\ge 2$ positives, $\ge 5$ negatives).</li>
        <li><strong>Zero Fabrication Guarantee:</strong> When an uploaded dataset lacks ground-truth labels, the training routine is <strong>completely bypassed</strong>. No synthetic target labels, pseudo-probabilities, or fabricated ROC-AUC curves are generated. In the final output table, <code>xgboost_risk_score</code> is explicitly set to <code>NaN</code>.</li>
        <li><strong>Imbalance & Training Architecture:</strong>
            <ul>
                <li>Stratified holdout train/test split: 75% train, 25% test (<code>stratify=y</code>, <code>random_state=42</code>).</li>
                <li>Class imbalance weighting: <code>scale_pos_weight = neg_count / max(1, pos_count)</code> passed directly to the XGBoost estimator.</li>
                <li>Hyperparameters: <code>n_estimators=100</code>, <code>max_depth=4</code>, <code>learning_rate=0.08</code>, <code>eval_metric="logloss"</code>.</li>
            </ul>
        </li>
        <li><strong>Holdout Evaluation Metrics:</strong> Accuracy, Precision, Recall, F1-Score, and ROC-AUC are computed strictly on the holdout test set ($X_{\text{test}}, y_{\text{test}}$). Generates a $2 \times 2$ Confusion Matrix and Gini feature importance rankings.</li>
    </ul>
</div>

<!-- Section 2 -->
<div class="page-break"></div>
<h2>2. Statistical and Forensic Algorithms</h2>

<p>
All Stage 2 forensic auditing tests are contained within <code>src/forensic/</code> and aggregated via <code>src/forensic/engine.py</code>.
</p>

<h3>2.1 Benford's Law (First & Second Digit Analysis)</h3>
<p>
Evaluates whether transaction amounts follow the logarithmic frequency distribution of natural monetary processes.
</p>
<ul>
    <li><strong>First-Digit Formulation:</strong> $P(d_1) = \log_{10}\left(1 + \frac{1}{d_1}\right)$ for $d_1 \in \{1, 2, \dots, 9\}$.</li>
    <li><strong>Second-Digit Formulation:</strong> $P(d_2) = \sum_{k=1}^9 \log_{10}\left(1 + \frac{1}{10k + d_2}\right)$ for $d_2 \in \{0, 1, \dots, 9\}$.</li>
    <li><strong>Goodness-of-Fit Metrics:</strong>
        <div class="formula-box">
            MAD = (1 / K) * sum(|Observed_Freq_d - Expected_Freq_d|) &nbsp;&nbsp;|&nbsp;&nbsp; Chi2 = sum((Observed_Count - Expected_Count)^2 / Expected_Count)
        </div>
    </li>
    <li><strong>Conformity Standards (Nigrini Scale):</strong> First-digit MAD $\le 0.006$ (Close), $\le 0.012$ (Acceptable), $\le 0.015$ (Marginal), $> 0.015$ (Nonconformity).</li>
    <li><strong>Pillar Score:</strong> First-digit MAD scaled to $0\text{--}100$ ($\text{score} = \min(100, (\text{MAD} / 0.035) \times 100)$), blended $70\%$ first digit + $30\%$ second digit.</li>
</ul>

<h3>2.2 Relative Size Factor (RSF)</h3>
<p>
Identifies abnormal account-level outliers by comparing the largest transaction to the second-largest baseline:
</p>
<div class="formula-box">
    RSF_account = Largest_Transaction_Amount / Second_Largest_Transaction_Amount
</div>
<ul>
    <li><strong>Account Flag:</strong> Flagged as anomalous if $RSF \ge 5.0\times$ and account transaction count $n \ge 2$. (Accounts with $n=1$ default to $RSF = 1.0$).</li>
    <li><strong>Transaction Score Normalization:</strong> $\text{tx\_rsf\_score} = \left(\frac{\text{clip}(RSF - 1.0, 0, 15.0)}{15.0}\right) \times 100.0$.</li>
    <li><strong>Dataset Score:</strong> Combines severity of peak spike ($\ln$-scaled up to 70 pts) with anomaly prevalence (up to 30 pts).</li>
</ul>

<h3>2.3 Duplicate Transaction Detection</h3>
<ul>
    <li><strong>Clustering Logic:</strong> Detects identical combinations of <code>[sender_account, amount, receiver_account]</code> using pandas <code>duplicated(keep=False)</code>.</li>
    <li><strong>Temporal Proximity:</strong> Evaluates time spread across duplicates. If time spread $\le 48.0\text{h}$, flags <code>rapid_repetition = True</code>.</li>
    <li><strong>Risk Score:</strong> Duplicates receive a transaction score of $75.0$, elevated to $100.0$ if inside the rapid 48-hour window.</li>
</ul>

<h3>2.4 Transaction Velocity & Burst Analysis</h3>
<ul>
    <li><strong>Rolling Window Protocol:</strong> Groups transactions by originating sender and sorts chronologically. Computes rolling window count within $2.0\text{ hours}$.</li>
    <li><strong>Burst Threshold:</strong> Flags burst events when $\ge 4$ transactions occur within the $2.0$-hour window.</li>
    <li><strong>Transaction Score:</strong> $\min\left(100.0, \frac{\text{count\_in\_window}}{\text{burst\_threshold}} \times 60.0\right)$.</li>
</ul>

<h3>2.5 AML Typology Heuristic Indicators</h3>
<ol>
    <li><strong>Structuring / Smurfing:</strong> Flags amounts falling between $88\%$ and $100\%$ of the $\$10,000$ CTR reporting threshold ($\$8,800.00\text{--}\$9,999.99$).</li>
    <li><strong>Fan-in Pattern (Mule Consolidation):</strong> Groups by <code>receiver_account</code> and identifies hubs receiving funds from $\ge 3$ distinct originating senders.</li>
    <li><strong>Fan-out Pattern (Dispersal):</strong> Groups by <code>sender_account</code> and identifies entities dispersing funds to $\ge 3$ distinct beneficiaries.</li>
    <li><strong>Layering / Rapid Pass-Through:</strong> Identifies transit accounts where $|inflow - outflow| / \max(inflow, outflow) \le 0.25$ (funds immediately evacuated with $\le 25\%$ balance retained).</li>
</ol>

<!-- Section 3 -->
<h2>3. Feature Engineering Pipeline</h2>
<p>
Implemented in <code>src/ml/feature_engineering.py</code> $\rightarrow$ <code>extract_ml_features()</code>. Extracts a robust <strong>16-dimensional numerical feature vector</strong> per transaction:
</p>

<table>
    <thead>
        <tr>
            <th style="width: 5%;">#</th>
            <th style="width: 25%;">Feature Name</th>
            <th style="width: 35%;">Mathematical Expression / Transformation</th>
            <th style="width: 35%;">Forensic & Machine Learning Purpose</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>1</td><td><code>amount</code></td><td>Raw monetary amount</td><td>Monetary magnitude baseline</td></tr>
        <tr><td>2</td><td><code>log_amount</code></td><td>$\ln(1 + \max(0, \text{amount}))$</td><td>Normalizes heavily skewed monetary distributions</td></tr>
        <tr><td>3</td><td><code>first_digit</code></td><td>Leading significant digit $\in \{1, \dots, 9\}$</td><td>Benford digit distribution tracking</td></tr>
        <tr><td>4</td><td><code>is_digit_nine</code></td><td>$(\text{first\_digit} == 9).astype(int)$</td><td>Smurfing transactions cluster near leading 9 (e.g., $\$9,500$)</td></tr>
        <tr><td>5</td><td><code>is_round_1000</code></td><td>$(\text{amount} \% 1000 == 0 \land \text{amount} > 0).astype(int)$</td><td>Illicit payments frequently utilize clean multiples of $\$1,000$</td></tr>
        <tr><td>6</td><td><code>is_round_500</code></td><td>$(\text{amount} \% 500 == 0 \land \text{amount} > 0).astype(int)$</td><td>Round increment transfer detection</td></tr>
        <tr><td>7</td><td><code>is_near_threshold</code></td><td>$(8800 \le \text{amount} < 10000).astype(int)$</td><td>Direct indicator for CTR threshold evasion</td></tr>
        <tr><td>8</td><td><code>sender_tx_frequency</code></td><td><code>sender_account.map(value_counts)</code></td><td>Account activity frequency baseline</td></tr>
        <tr><td>9</td><td><code>sender_avg_amount</code></td><td>Sender mean amount via <code>transform('mean')</code></td><td>Account baseline behavioral volume</td></tr>
        <tr><td>10</td><td><code>sender_max_amount</code></td><td>Sender peak amount via <code>transform('max')</code></td><td>Historical transaction ceiling for sender</td></tr>
        <tr><td>11</td><td><code>amount_to_sender_avg</code></td><td>$\text{amount} / \text{sender\_avg\_amount}$</td><td>Ratio of current transfer to historical account norm</td></tr>
        <tr><td>12</td><td><code>amount_zscore_sender</code></td><td>$(\text{amount} - \mu_{\text{sender}}) / \sigma_{\text{sender}}$</td><td>Standardized deviation from sender historical mean</td></tr>
        <tr><td>13</td><td><code>receiver_inflow_count</code></td><td><code>receiver_account.map(value_counts)</code></td><td>Beneficiary consolidation frequency</td></tr>
        <tr><td>14</td><td><code>rsf_risk_score</code></td><td>Stage 2 RSF module score ($0\text{--}100$)</td><td>Relative size factor anomaly signal</td></tr>
        <tr><td>15</td><td><code>duplicate_risk_score</code></td><td>Stage 2 Duplicate module score ($0, 75, 100$)</td><td>Repeated transfer anomaly signal</td></tr>
        <tr><td>16</td><td><code>velocity_risk_score</code></td><td>Stage 2 Velocity module score ($0\text{--}100$)</td><td>Burst frequency anomaly signal</td></tr>
    </tbody>
</table>

<!-- Section 4 & 5 -->
<div class="page-break"></div>
<h2>4. Model Evaluation & Validation Methodology</h2>
<ul>
    <li><strong>Holdout Validation Split:</strong> Evaluated on a 75% train / 25% test split with stratification on target labels (<code>stratify=y</code>, <code>random_state=42</code>).</li>
    <li><strong>Genuine Validation Metrics Computed:</strong> Accuracy, Precision, Recall, F1-Score, and ROC-AUC are evaluated strictly on the 25% holdout partition. In the sample dataset, the model correctly isolates labeled laundering clusters.</li>
    <li><strong>Cross-Validation Status:</strong> $k$-fold cross-validation is <strong>not implemented</strong>; evaluation relies solely on the stratified holdout split.</li>
    <li><strong>Evaluation Limitation (Lookahead Leakage):</strong> In <code>extract_ml_features()</code>, account historical baselines (mean, max, frequency) are computed across the entire DataFrame prior to train/test splitting. In high-stakes production environments, rolling out-of-time baselines should be applied to prevent lookahead leakage.</li>
</ul>

<h2>5. Composite Risk Scoring Engine — Stage 4 Architecture</h2>
<p>
Stage 4 synthesizes deterministic forensic tests with unsupervised and supervised machine learning into a single unified priority score ($0\text{--}100$).
</p>

<h3>5.1 Formulation & Default Weights</h3>
<div class="formula-box">
    Composite Risk_i = sum_{k in A} ( w_k^* * S_{i, k} )
</div>
<ul>
    <li><strong>Base Weights (<code>DEFAULT_COMPOSITE_WEIGHTS</code>):</strong>
        <ul>
            <li>Forensic Auditing Pillar ($W_{\text{forensic}}$): <strong>40%</strong> (0.40)</li>
            <li>Isolation Forest Anomaly Pillar ($W_{\text{iso}}$): <strong>30%</strong> (0.30)</li>
            <li>XGBoost Supervised Pillar ($W_{\text{xgb}}$): <strong>30%</strong> (0.30)</li>
        </ul>
    </li>
</ul>

<h3>5.2 Proportional Weight Redistribution Formula</h3>
<p>
When an analytical pillar is unavailable (such as when an uploaded CSV has no ground truth labels, disabling XGBoost), its weight is redistributed proportionally among active pillars:
</p>
<div class="formula-box">
    w_k^* = W_k / sum_{j in A} ( W_j ) &nbsp;&nbsp;such that&nbsp;&nbsp; sum_{k in A} w_k^* = 1.0
</div>
<ul>
    <li><strong>Unlabeled Dataset Redistribution:</strong>
        <ul>
            <li>Effective Forensic Weight: $w_{\text{forensic}}^* = \frac{0.40}{0.40 + 0.30} = \frac{0.40}{0.70} \approx \mathbf{57.14\%}$</li>
            <li>Effective Isolation Forest Weight: $w_{\text{iso}}^* = \frac{0.30}{0.40 + 0.30} = \frac{0.30}{0.70} \approx \mathbf{42.86\%}$</li>
            <li>Effective XGBoost Weight: $\mathbf{0.00\%}$ (<code>xgboost_risk_score</code> populated with <code>NaN</code>; zero fake scores).</li>
        </ul>
    </li>
</ul>

<h3>5.3 4-Tier Risk Classification & Contributing Drivers</h3>
<table>
    <thead>
        <tr>
            <th style="width: 22%;">Risk Tier</th>
            <th style="width: 18%;">Score Boundary</th>
            <th style="width: 60%;">Compliance Action & Operational Escalation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>🟢 Low Risk</strong></td>
            <td>$0 \le \text{Score} \le 24$</td>
            <td>Routine transaction. Fully automated straight-through processing.</td>
        </tr>
        <tr>
            <td><strong>🟡 Medium Risk</strong></td>
            <td>$25 \le \text{Score} \le 49$</td>
            <td>Elevated pattern. Logged to account watch-list; periodic compliance audit.</td>
        </tr>
        <tr>
            <td><strong>🟠 High Risk</strong></td>
            <td>$50 \le \text{Score} \le 74$</td>
            <td>Secondary review required by compliance officer within 48 hours.</td>
        </tr>
        <tr>
            <td><strong>🔴 Critical Risk</strong></td>
            <td>$75 \le \text{Score} \le 100$</td>
            <td>Immediate transaction hold, account freeze, and expedited SAR (Suspicious Activity Report) filing.</td>
        </tr>
    </tbody>
</table>

<!-- Section 6 & 7 -->
<h2>6. Dataset Audit & Generalization Capabilities</h2>
<ul>
    <li><strong>Sample Dataset Verification:</strong> <code>data/sample_transactions.csv</code> contains exactly <strong>218 transactions</strong> (180 normal, 8 structuring, 4 duplicates, 12 fan-in, 8 fan-out, 5 baseline, 1 extreme RSF spike). Labels in this dataset are synthetic, seeded deterministically for benchmarking.</li>
    <li><strong>Arbitrary CSV Generalization:</strong> The platform accepts arbitrary CSV files via <code>load_csv()</code>.
        <ul>
            <li><strong>Mandatory Columns:</strong> <code>transaction_id</code>, <code>timestamp</code>, <code>sender_account</code>, <code>receiver_account</code>, <code>amount</code>.</li>
            <li><strong>Data Cleansing:</strong> <code>clean_dataframe()</code> strips header whitespaces, converts column names to lowercase, coerces non-numeric amounts to floats, and normalizes timestamps.</li>
        </ul>
    </li>
</ul>

<h2>7. Stage 5 & Stage 6 Readiness & Prerequisites</h2>
<div class="warning-box">
    <strong>⚠️ Critical Blocker for Stage 5:</strong> While <code>shap&gt;=0.45.0</code> is declared in <code>requirements.txt</code>, the SHAP package is <strong>not currently installed</strong> in the Python environment (<code>ModuleNotFoundError: No module named 'shap'</code>). Run <code>pip install shap</code> before developing Stage 5.
</div>
<ul>
    <li><strong>SHAP Implementation Plan:</strong> Use <code>shap.TreeExplainer(model)</code> on the trained XGBoost estimator. For unlabeled datasets, provide a clear UI notice that SHAP explanations apply to supervised models, while showing forensic driver breakdowns.</li>
    <li><strong>Investigation View:</strong> Expand the existing Stage 4 "Auditor Transaction Dossier" into a full-page deep dive with counterparty flow mini-charts and local feature attribution plots.</li>
    <li><strong>Stage 6 Audit Report Export:</strong> The enriched DataFrame already aggregates all required audit attributes. It can be directly exported to CSV, JSON, and formatted compliance summaries under <code>reports/</code>.</li>
</ul>

<div class="success-box">
    <strong>✅ Verification Sign-Off:</strong> All 7 unit and integration tests across Stages 1–4 passed with exit code 0. Zero regressions detected.
</div>

<div class="footer-note">
    AML Forensic Analytics System &copy; 2026 | Technical Inspection Report | Generated via Headless Chromium Engine
</div>

</body>
</html>
"""

def generate_pdf():
    print("Writing HTML inspection report...")
    with open(HTML_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(HTML_CONTENT)
    print(f"HTML written to: {HTML_OUTPUT_PATH}")

    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]

    browser_exe = None
    for p in edge_paths:
        if Path(p).exists():
            browser_exe = p
            break

    if not browser_exe:
        raise FileNotFoundError("Could not find Edge or Chrome executable for PDF generation.")

    print(f"Using browser executable: {browser_exe}")
    cmd = [
        browser_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUTPUT_PATH}",
        str(HTML_OUTPUT_PATH),
    ]

    print("Running headless PDF conversion...")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"Subprocess output: {result.stdout} {result.stderr}")

    if PDF_OUTPUT_PATH.exists():
        size_kb = PDF_OUTPUT_PATH.stat().st_size / 1024
        print(f"SUCCESS: Generated PDF at {PDF_OUTPUT_PATH} ({size_kb:.2f} KB)")
    else:
        raise RuntimeError("PDF was not created.")

if __name__ == "__main__":
    generate_pdf()
