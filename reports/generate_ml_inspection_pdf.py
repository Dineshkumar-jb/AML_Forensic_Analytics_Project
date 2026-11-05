"""Script to generate the AML ML Model Inspection Report PDF.

Determines whether Isolation Forest and XGBoost use externally pretrained models,
train models from scratch, or load previously saved trained models.
"""

from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
HTML_OUTPUT_PATH = REPORTS_DIR / "AML_ML_Model_Inspection_Report.html"
PDF_OUTPUT_PATH = REPORTS_DIR / "AML_ML_Model_Inspection_Report.pdf"

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AML Forensic Analytics - ML Model Training & Lifecycle Inspection Report</title>
<style>
    @page {
        size: A4;
        margin: 15mm 14mm 15mm 14mm;
        @bottom-right {
            content: "Page " counter(page);
        }
    }
    body {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        background-color: #ffffff;
        line-height: 1.55;
        font-size: 9.5pt;
        margin: 0;
        padding: 0;
    }
    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 700;
        margin-top: 1.3em;
        margin-bottom: 0.4em;
        page-break-after: avoid;
    }
    h1 {
        font-size: 19pt;
        line-height: 1.25;
        border-bottom: 3px solid #1e3a8a;
        padding-bottom: 8px;
        margin-top: 0;
        color: #1e3a8a;
    }
    h2 {
        font-size: 12.5pt;
        border-bottom: 1.5px solid #cbd5e1;
        padding-bottom: 4px;
        color: #1e40af;
        margin-top: 1.5em;
    }
    h3 {
        font-size: 10.5pt;
        color: #334155;
        margin-top: 1.1em;
    }
    p, li {
        color: #334155;
        text-align: justify;
    }
    ul, ol {
        margin-top: 0.3em;
        margin-bottom: 0.6em;
        padding-left: 1.4em;
    }
    li {
        margin-bottom: 0.35em;
    }
    .header-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
    .header-box table {
        width: 100%;
        border-collapse: collapse;
        margin: 0;
    }
    .header-box td {
        padding: 2px 6px;
        font-size: 9pt;
        border: none;
        background: transparent;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 7.8pt;
        font-weight: 700;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .badge-scratch {
        background-color: #dbeafe;
        color: #1d4ed8;
        border: 1px solid #93c5fd;
    }
    .badge-no-pretrain {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .badge-no-saved {
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
        margin: 10px 0 14px 0;
        font-size: 8.5pt;
    }
    tr {
        page-break-inside: avoid;
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
        margin: 10px 0 12px 0;
        page-break-inside: avoid;
    }
    .card-title {
        font-weight: 700;
        color: #1e3a8a;
        font-size: 10pt;
        margin-bottom: 4px;
    }
    .formula-box {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 5px;
        padding: 7px 11px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 8.5pt;
        color: #0f172a;
        margin: 6px 0;
    }
    code {
        font-family: 'Consolas', 'Courier New', monospace;
        background-color: #f1f5f9;
        padding: 1px 4px;
        border-radius: 3px;
        font-size: 8.5pt;
        color: #0f172a;
    }
    .alert-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-top: 1px solid #dbeafe;
        border-right: 1px solid #dbeafe;
        border-bottom: 1px solid #dbeafe;
        padding: 9px 13px;
        margin: 10px 0;
        font-size: 8.8pt;
        page-break-inside: avoid;
    }
    .conclusion-box {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        border-top: 1px solid #bbf7d0;
        border-right: 1px solid #bbf7d0;
        border-bottom: 1px solid #bbf7d0;
        border-radius: 0 6px 6px 0;
        padding: 12px 16px;
        margin: 14px 0;
        page-break-inside: avoid;
    }
    .workflow-container {
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin: 12px 0;
        page-break-inside: avoid;
    }
    .flow-step {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 8px 12px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .flow-num {
        background-color: #1e3a8a;
        color: #ffffff;
        font-weight: 700;
        border-radius: 50%;
        width: 22px;
        height: 22px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 8pt;
        flex-shrink: 0;
    }
    .flow-text {
        font-size: 8.8pt;
    }
    .flow-arrow {
        text-align: center;
        font-size: 10pt;
        color: #64748b;
        margin: -3px 0;
    }
    .page-break {
        page-break-before: always;
    }
    .footer-note {
        font-size: 7.8pt;
        color: #64748b;
        text-align: center;
        margin-top: 22px;
        border-top: 1px solid #e2e8f0;
        padding-top: 6px;
    }
</style>
</head>
<body>

<!-- Header Banner -->
<h1>AML Forensic Analytics & Risk Scoring System</h1>
<p style="font-size: 11pt; color: #475569; margin-top: -6px; margin-bottom: 12px;">
    <strong>Special Investigation Report: Machine Learning Model Training Lifecycle & Pretrained Weights Audit</strong>
</p>

<div class="header-box">
    <table>
        <tr>
            <td style="width: 22%;"><strong>Project Name:</strong></td>
            <td style="width: 44%;">Transaction Monitoring & Risk Scoring for AML Detection</td>
            <td style="width: 16%;"><strong>Inspection Date:</strong></td>
            <td style="width: 18%;">October 2026</td>
        </tr>
        <tr>
            <td><strong>Investigation Focus:</strong></td>
            <td>Are models pretrained, trained from scratch, or loaded from disk?</td>
            <td><strong>Inspection Type:</strong></td>
            <td><span class="badge badge-scratch">Source Code & Disk Audit</span></td>
        </tr>
        <tr>
            <td><strong>Evaluated Models:</strong></td>
            <td>Isolation Forest (Unsupervised) & XGBoost Classifier (Supervised)</td>
            <td><strong>Audit Verdict:</strong></td>
            <td><span class="badge badge-no-pretrain">Trained From Scratch at Runtime</span></td>
        </tr>
    </table>
</div>

<!-- Core Question Answer Banner -->
<div class="conclusion-box">
    <h3 style="margin-top: 0; color: #166534; font-size: 11.5pt;">🎯 The Direct Answer to the Core Question</h3>
    <p style="font-size: 9.5pt; margin-bottom: 6px; line-height: 1.5;">
        <strong>"Are we using externally pretrained models, training our own models from scratch, or loading previously saved trained models?"</strong>
    </p>
    <ul style="margin-bottom: 0;">
        <li><strong>We are NOT using externally pretrained models:</strong> There are zero pre-existing model weights, checkpoints, neural network files, or third-party banking models bundled in the repository.</li>
        <li><strong>We are NOT loading previously saved trained models:</strong> There are zero <code>.joblib</code>, <code>.pkl</code>, <code>.pickle</code>, <code>.bin</code>, <code>.onnx</code>, or <code>.model</code> files anywhere in the workspace. No serialization or deserialization calls (like <code>joblib.load()</code> or <code>pickle.load()</code>) exist in the code.</li>
        <li><strong>We ARE training our own models from scratch dynamically in memory:</strong> Every single time the application runs or a new CSV file is loaded, both the <code>IsolationForest</code> and <code>XGBClassifier</code> instances are newly initialized in memory and fitted (<code>.fit()</code> / <code>.fit_predict()</code>) directly on the transaction records present in that specific uploaded dataset.</li>
    </ul>
</div>

<!-- Section 1: Pretrained vs Application-Trained -->
<h2>1. Conceptual Clarity: Pretrained vs. Saved vs. Scratch-Trained Models</h2>
<p>
To defend this project during a college presentation or technical viva, it is essential to distinguish between the three distinct machine learning lifecycle paradigms:
</p>

<table>
    <thead>
        <tr>
            <th style="width: 20%;">Paradigm</th>
            <th style="width: 32%;">What it Means in Theory</th>
            <th style="width: 28%;">Typical Industry Example</th>
            <th style="width: 20%;">Used in This Project?</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>1. Externally Pretrained Model</strong></td>
            <td>A model trained by a third party on a massive external dataset. The learned weights/parameters are bundled into the software. The application only performs forward inference (prediction), never calling <code>.fit()</code>.</td>
            <td>BERT, GPT, ResNet, or an AML model trained by a vendor on 50 million SWIFT wires.</td>
            <td><span class="badge badge-no-pretrain">NO (Zero Files)</span></td>
        </tr>
        <tr>
            <td><strong>2. Previously Saved Model (Offline Trained)</strong></td>
            <td>A model that the project authors trained earlier on their computer, saved as a binary file (e.g., <code>model.pkl</code> via Joblib/Pickle), and the web app loads it at startup using <code>joblib.load('model.pkl')</code> to avoid retraining.</td>
            <td>A fraud model trained offline once on Kaggle data, exported, and reloaded into Streamlit.</td>
            <td><span class="badge badge-no-saved">NO (Zero Files)</span></td>
        </tr>
        <tr>
            <td><strong>3. Application-Trained From Scratch (Dynamic)</strong></td>
            <td>A model instantiated with default parameters inside the running application and fitted directly on the currently loaded data via <code>.fit()</code>. If a new CSV is uploaded, a brand new model is trained on the spot.</td>
            <td>Interactive analytics tool that adapts directly to the user's uploaded transactions in real time.</td>
            <td><span class="badge badge-scratch">YES (100% of ML)</span></td>
        </tr>
    </tbody>
</table>

<!-- Section 2: Detailed Model Audit -->
<h2>2. Detailed Inspection of Isolation Forest & XGBoost</h2>

<div class="card">
    <div class="card-title">2.1 Isolation Forest (Unsupervised Anomaly Detection)</div>
    <p>
        <strong>Source File:</strong> <code>src/ml/isolation_forest.py</code> | <strong>Function:</strong> <code>train_isolation_forest()</code> (lines 14–86)
    </p>
    <ul>
        <li><strong>What it is:</strong> An unsupervised tree-based algorithm that isolates anomalies rather than profiling normal data. Because fraudulent or anomalous transactions have unusual feature values, they are isolated in fewer random splits (yielding shorter tree path lengths).</li>
        <li><strong>Is it pretrained?</strong> <strong>NO.</strong> A brand new <code>sklearn.ensemble.IsolationForest(n_estimators=100, contamination=0.08, random_state=42)</code> is instantiated in memory at line 40.</li>
        <li><strong>Is a saved model loaded?</strong> <strong>NO.</strong> No file-reading operations exist in the module.</li>
        <li><strong>How it trains:</strong>
            <ul>
                <li>Scales the 16 features: <code>scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)</code> (line 37).</li>
                <li>Trains from scratch: <code>preds = iso_model.fit_predict(X_scaled)</code> (line 48).</li>
                <li>Computes decision scores: <code>raw_decision = iso_model.decision_function(X_scaled)</code> (line 53).</li>
                <li>Normalizes decision scores to a $0\text{--}100$ scale and returns the fitted model object in memory.</li>
            </ul>
        </li>
        <li><strong>Behavior on CSV Upload:</strong> Whenever a new CSV is uploaded, the existing model in memory is replaced. A new <code>IsolationForest</code> is fitted on the new CSV's transactions.</li>
    </ul>
</div>

<div class="card">
    <div class="card-title">2.2 XGBoost Classifier (Supervised Laundering Classification)</div>
    <p>
        <strong>Source File:</strong> <code>src/ml/xgboost_model.py</code> | <strong>Functions:</strong> <code>train_xgboost()</code> (lines 50–148) & <code>detect_ground_truth_label()</code> (lines 23–48)
    </p>
    <ul>
        <li><strong>What it is:</strong> A supervised gradient boosting decision tree ensemble that minimizes logistic loss across the 16 engineered features to calculate class probabilities of illicit activity.</li>
        <li><strong>Is it pretrained?</strong> <strong>NO.</strong> An instance of <code>xgboost.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, scale_pos_weight=...)</code> is instantiated at line 93.</li>
        <li><strong>Is a saved model loaded?</strong> <strong>NO.</strong> No calls to <code>xgb.Booster.load_model()</code>, <code>joblib.load()</code>, or <code>pickle.load()</code> exist anywhere in the code.</li>
        <li><strong>How it trains (When labels exist):</strong>
            <ul>
                <li>Detects ground-truth column (e.g. <code>is_laundering</code>).</li>
                <li>Performs a 75/25 stratified holdout train/test split: <code>train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)</code> (lines 78–80).</li>
                <li>Computes class weighting: <code>scale_pos_weight = neg_count / max(1, pos_count)</code> (line 90).</li>
                <li>Trains from scratch: <code>model.fit(X_train, y_train)</code> (line 103).</li>
                <li>Evaluates holdout metrics: <code>y_pred = model.predict(X_test)</code> and <code>y_prob = model.predict_proba(X_test)[:, 1]</code> (lines 106–107).</li>
                <li>Infers full risk scores: <code>all_probs = model.predict_proba(X)[:, 1] * 100.0</code> (line 130).</li>
            </ul>
        </li>
        <li><strong>Behavior on Unlabeled CSV Upload:</strong> If an uploaded CSV has no genuine labels, XGBoost training is <strong>completely skipped</strong>. No model is loaded, no fake predictions are made, and its risk score is set to <code>NaN</code>.</li>
    </ul>
</div>

<!-- Section 3: Summary Table -->
<div class="page-break"></div>
<h2>3. Model Training & Persistence Matrix</h2>
<p>
The table below details the exact runtime behavior and persistence characteristics of every analytical model in the system:
</p>

<table>
    <thead>
        <tr>
            <th style="width: 16%;">Model Name</th>
            <th style="width: 13%;">Paradigm</th>
            <th style="width: 14%;">Trained From Scratch?</th>
            <th style="width: 13%;">Saved Model Loaded?</th>
            <th style="width: 22%;">Dataset Used for Training</th>
            <th style="width: 22%;">Behavior After CSV Upload</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Isolation Forest</strong></td>
            <td>Unsupervised ML</td>
            <td><strong style="color: #16a34a;">YES</strong><br>(via <code>fit_predict</code>)</td>
            <td><strong style="color: #dc2626;">NO</strong><br>(Zero files)</td>
            <td>16 engineered features from currently loaded CSV</td>
            <td><strong>Automatically retrained from scratch</strong> on the newly uploaded data.</td>
        </tr>
        <tr>
            <td><strong>XGBoost (Labeled Data)</strong></td>
            <td>Supervised ML</td>
            <td><strong style="color: #16a34a;">YES</strong><br>(via <code>model.fit</code>)</td>
            <td><strong style="color: #dc2626;">NO</strong><br>(Zero files)</td>
            <td>75% stratified holdout split of current CSV with <code>is_laundering</code></td>
            <td><strong>Retrained from scratch</strong> using the uploaded CSV's ground-truth labels.</td>
        </tr>
        <tr>
            <td><strong>XGBoost (Unlabeled Data)</strong></td>
            <td>Supervised ML</td>
            <td><strong style="color: #475569;">SKIPPED</strong><br>(No labels)</td>
            <td><strong style="color: #dc2626;">NO</strong><br>(Zero files)</td>
            <td>None (Awaits verified audit labels)</td>
            <td><strong>Bypassed safely.</strong> Weight redistributed to Forensics and IsoForest.</td>
        </tr>
        <tr>
            <td><strong>Forensic Analytics Suite</strong></td>
            <td>Deterministic & Statistical Rules</td>
            <td><strong>N/A</strong><br>(Rule-based)</td>
            <td><strong>N/A</strong><br>(No weights)</td>
            <td>Full transaction records of currently loaded CSV</td>
            <td><strong>Recalculates all forensic tests</strong> (Benford, RSF, Duplicates, Velocity).</td>
        </tr>
        <tr>
            <td><strong>Composite Risk Engine</strong></td>
            <td>Hybrid Convex Combination</td>
            <td><strong>N/A</strong><br>(Mathematical)</td>
            <td><strong>N/A</strong><br>(No weights)</td>
            <td>Enriched outputs from Forensics and ML layers</td>
            <td><strong>Recalculates final composite scores</strong> ($0\text{--}100$) and 4-tier risk categories.</td>
        </tr>
    </tbody>
</table>

<!-- Section 4: Exact Code Evidence -->
<h2>4. Executable Code Evidence (File Paths & Line Numbers)</h2>
<p>
The findings above are verified directly against executable statements in the source code:
</p>

<table>
    <thead>
        <tr>
            <th style="width: 28%;">File Path</th>
            <th style="width: 10%;">Line(s)</th>
            <th style="width: 32%;">Executable Code Statement</th>
            <th style="width: 30%;">Technical Significance</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>src/ml/isolation_forest.py</code></td>
            <td>Line 37</td>
            <td><code>X_scaled = scaler.fit_transform(X)</code></td>
            <td>Scales feature matrix in memory; not loading saved scaler.</td>
        </tr>
        <tr>
            <td><code>src/ml/isolation_forest.py</code></td>
            <td>Line 40</td>
            <td><code>iso_model = IsolationForest(...)</code></td>
            <td>Initializes brand-new model instance from scikit-learn.</td>
        </tr>
        <tr>
            <td><code>src/ml/isolation_forest.py</code></td>
            <td>Line 48</td>
            <td><code>preds = iso_model.fit_predict(X_scaled)</code></td>
            <td><strong>Trains from scratch</strong> on the active feature matrix.</td>
        </tr>
        <tr>
            <td><code>src/ml/xgboost_model.py</code></td>
            <td>Lines 78–85</td>
            <td><code>train_test_split(X, y, test_size=0.25, ...)</code></td>
            <td>Splits active dataset into 75% train and 25% test partitions.</td>
        </tr>
        <tr>
            <td><code>src/ml/xgboost_model.py</code></td>
            <td>Line 93</td>
            <td><code>model = xgb.XGBClassifier(...)</code></td>
            <td>Initializes brand-new XGBoost model instance.</td>
        </tr>
        <tr>
            <td><code>src/ml/xgboost_model.py</code></td>
            <td>Line 103</td>
            <td><code>model.fit(X_train, y_train)</code></td>
            <td><strong>Trains from scratch</strong> using the training split.</td>
        </tr>
        <tr>
            <td><code>src/ml/xgboost_model.py</code></td>
            <td>Lines 106–107</td>
            <td><code>model.predict(X_test); model.predict_proba(...)</code></td>
            <td>Computes genuine holdout validation metrics on test set.</td>
        </tr>
        <tr>
            <td><code>src/ml/engine.py</code></td>
            <td>Lines 42–45</td>
            <td><code>train_isolation_forest(features_df, ...)</code></td>
            <td>Calls training pipeline inside the execution engine.</td>
        </tr>
        <tr>
            <td><code>src/ml/engine.py</code></td>
            <td>Lines 53–57</td>
            <td><code>train_xgboost(features_df, target_series, ...)</code></td>
            <td>Calls supervised training if and only if labels exist.</td>
        </tr>
        <tr>
            <td><code>app.py</code></td>
            <td>Lines 283–289</td>
            <td><code>run_ml_pipeline(df=df_loaded, ...)</code></td>
            <td>Streamlit entry point: passes uploaded data into pipeline.</td>
        </tr>
        <tr>
            <td><em>Entire Workspace</em></td>
            <td>All files</td>
            <td><code>joblib.load</code>, <code>pickle.load</code>, <code>.pkl</code>, <code>.model</code></td>
            <td><strong>0 matches found across the entire repository.</strong></td>
        </tr>
    </tbody>
</table>

<!-- Section 5: End-to-End Workflow Diagram -->
<h2>5. End-to-End System Workflow Architecture</h2>
<p>
The complete lifecycle from CSV upload to final risk scoring operates strictly through a dynamic, linear, in-memory pipeline:
</p>

<div class="workflow-container">
    <div class="flow-step">
        <div class="flow-num">1</div>
        <div class="flow-text">
            <strong>Data Ingestion & Validation:</strong> User uploads a transaction CSV via Streamlit (or selects the built-in 218-record dataset). <code>load_csv()</code> cleans headers and validates required schema columns.
        </div>
    </div>
    <div class="flow-arrow">▼</div>
    <div class="flow-step">
        <div class="flow-num">2</div>
        <div class="flow-text">
            <strong>Stage 2 Forensic Auditing Suite:</strong> <code>run_forensic_suite()</code> executes Benford's Law (1st/2nd digit), RSF, Duplicate detection, Velocity burst analysis, and Typology heuristics (Structuring, Fan-in/out, Layering). Produces enriched DataFrame.
        </div>
    </div>
    <div class="flow-arrow">▼</div>
    <div class="flow-step">
        <div class="flow-num">3</div>
        <div class="flow-text">
            <strong>Feature Engineering (16 Features):</strong> <code>extract_ml_features()</code> merges raw transaction amounts, log amounts, Benford digit flags, account historical baselines (means, maxes, Z-scores), and Stage 2 forensic scores into a 16-column numerical matrix.
        </div>
    </div>
    <div class="flow-arrow">▼</div>
    <div class="flow-step">
        <div class="flow-num">4</div>
        <div class="flow-text">
            <strong>Dynamic Model Training From Scratch (Stage 3):</strong>
            <br>&bull; <em>Isolation Forest:</em> Standard-scales features and calls <code>iso_model.fit_predict(X_scaled)</code> in memory.
            <br>&bull; <em>XGBoost:</em> Checks for labels. If present, splits 75/25 and calls <code>model.fit(X_train, y_train)</code>. If absent, bypassed safely.
        </div>
    </div>
    <div class="flow-arrow">▼</div>
    <div class="flow-step">
        <div class="flow-num">5</div>
        <div class="flow-text">
            <strong>Inference & Anomaly Score Generation:</strong> <code>decision_function()</code> normalized to $0\text{--}100$ for Isolation Forest; <code>predict_proba()</code> scaled to $0\text{--}100$ for XGBoost.
        </div>
    </div>
    <div class="flow-arrow">▼</div>
    <div class="flow-step">
        <div class="flow-num">6</div>
        <div class="flow-text">
            <strong>Stage 4 Multi-Pillar Composite Risk Scoring:</strong> <code>calculate_composite_risk()</code> combines scores (40% Forensics, 30% IsoForest, 30% XGBoost). If XGBoost is unlabeled, its weight is proportionally redistributed (57.1% Forensics, 42.9% IsoForest). Assigns Low/Medium/High/Critical tiers.
        </div>
    </div>
</div>

<!-- Section 6: Viva Concept Distinctions -->
<div class="page-break"></div>
<h2>6. Key Concepts for Presentation Defense</h2>
<p>
Examiners frequently ask students to clarify the operational differences between the three analytical layers in this system:
</p>

<table>
    <thead>
        <tr>
            <th style="width: 22%;">Analytical Concept</th>
            <th style="width: 26%;">How it Works in This Project</th>
            <th style="width: 26%;">Primary Strengths</th>
            <th style="width: 26%;">Primary Limitations</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Anomaly Detection (Isolation Forest)</strong></td>
            <td>Unsupervised. Identifies geometric outliers across 16 features without knowing whether they are legal or illegal.</td>
            <td>Catches zero-day, previously unseen money laundering patterns without requiring historical training labels.</td>
            <td>High false-positive rate. Cannot distinguish a legitimate billionaire transfer from money laundering.</td>
        </tr>
        <tr>
            <td><strong>Supervised Classification (XGBoost)</strong></td>
            <td>Supervised. Learns specific non-linear decision boundaries separating confirmed laundering from legitimate transfers.</td>
            <td>Highly accurate at detecting known typologies; provides feature importances and class probabilities.</td>
            <td>Useless on unlabeled banking data. Blind to novel laundering techniques not present in the training labels.</td>
        </tr>
        <tr>
            <td><strong>Composite Risk Scoring (Stage 4)</strong></td>
            <td>Hybrid Convex Combination. Synthesizes Forensic red flags ($40\%$), IsoForest ($30\%$), and XGBoost ($30\%$) into a $0\text{--}100$ score.</td>
            <td>Eliminates blind spots. Balances legal forensic rules, novel anomaly discovery, and supervised pattern matching.</td>
            <td>Requires careful weight calibration. Needs graceful redistribution when supervised labels are absent.</td>
        </tr>
    </tbody>
</table>

<!-- Section 7: Limitations & Future Improvements -->
<h2>7. Limitations & Architectural Observations</h2>
<ul>
    <li><strong>In-Memory Retraining Scalability:</strong> Because the system fits <code>IsolationForest</code> and <code>XGBClassifier</code> from scratch upon every upload, execution is fast on college-scale datasets (218 records takes &lt; 2 seconds). However, on an institutional production dataset with 50 million transactions, retraining from scratch in Streamlit would cause timeout bottlenecks. A production architecture would save a trained model using Joblib and execute fast inference (<code>predict_proba</code>) only.</li>
    <li><strong>Absence of Model Checkpoints:</strong> If the user closes the Streamlit browser, the trained model instance is garbage-collected. The next user session must re-fit the model.</li>
    <li><strong>Lookahead Account Baseline Calculation:</strong> In <code>src/ml/feature_engineering.py</code>, account historical baselines (mean, max, frequency) are computed across the entire DataFrame before the 75/25 split. In a banking production environment, out-of-time sliding windows should be applied to prevent lookahead leakage.</li>
    <li><strong>Missing SHAP Package:</strong> Although <code>shap&gt;=0.45.0</code> is listed in <code>requirements.txt</code>, it is not currently installed in the environment (<code>ModuleNotFoundError</code>). It must be installed before implementing Stage 5.</li>
</ul>

<!-- Final Conclusion -->
<h2>8. Final Verification Conclusion</h2>
<div class="conclusion-box">
    <p style="margin: 0; font-size: 10pt; line-height: 1.6;">
        <strong>Definitive Audit Conclusion:</strong><br>
        The AML Forensic Analytics project <strong>strictly trains its own models from scratch dynamically at runtime</strong> on whatever dataset is active.
        <br><br>
        1. <strong>Pretrained Models:</strong> <strong>NONE.</strong> No pretrained models or weights exist in the repository.<br>
        2. <strong>Saved / Reused Models:</strong> <strong>NONE.</strong> No model files exist on disk, and no loading code exists.<br>
        3. <strong>Runtime Retraining:</strong> <strong>CONFIRMED.</strong> Every time a CSV is uploaded, both Isolation Forest and XGBoost (if labels exist) execute <code>.fit()</code> and train fresh models directly on that data.
    </p>
</div>

<div class="footer-note">
    AML Forensic Analytics System &copy; 2026 | Special ML Model Inspection Report | Generated via Headless Chromium Engine
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
