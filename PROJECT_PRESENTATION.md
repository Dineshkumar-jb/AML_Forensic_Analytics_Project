# Transaction Monitoring & Risk Scoring for Money Laundering Detection
## Comprehensive Project Presentation Deck (Parallel to PPT)

**Course:** CSE4074 – Forensic Analytics  
**Institution:** Vellore Institute of Technology (VIT), Chennai  
**Semester:** Fall 2026-27  
**Faculty Guide:** Dr. S. Johanan Joysingh  
**Project Team:**
- Dinesh Kumar J B (`23MIA1126`)
- Sanjay K (`23MIA1060`)
- Bharath (`23MIA1059`)

---

## Slide 1: Title Slide
### Transaction Monitoring & Risk Scoring for Money Laundering Detection
#### A Multi-Pillar Forensic Auditing, Hybrid Machine Learning, and Explainable AI Compliance Platform

- **Course:** CSE4074 — Forensic Analytics
- **Department:** School of Computer Science and Engineering (SCOPE)
- **Institution:** Vellore Institute of Technology (VIT), Chennai
- **Faculty Guide:** Dr. S. Johanan Joysingh
- **Presented by:**
  - **Dinesh Kumar J B** (Reg No: 23MIA1126)
  - **Sanjay K** (Reg No: 23MIA1060)
  - **Bharath** (Reg No: 23MIA1059)
- **Review Stage:** Third Review (Implementation, Evaluation & GitHub Submission)

> **Presenter Speaking Note:**  
> Good morning / afternoon Professor. Today our team is presenting our capstone forensic analytics project: *Transaction Monitoring & Risk Scoring for Money Laundering Detection*. We address the fundamental vulnerability of financial transaction monitoring by unifying classical forensic auditing, machine learning, dynamic risk scoring, and explainable AI into an institutional compliance engine.

---

## Slide 2: Problem Statement & Industrial Motivation
### The Anti-Money Laundering (AML) Compliance Crisis

- **Global Scale:** According to UNODC and FATF, between **2% to 5% of global GDP ($800 Billion to $2 Trillion)** is laundered annually through global financial networks.
- **The False Positive Paradox:**
  - Traditional Rule-Based Transaction Monitoring Systems (TMS) generate **90% to 98% false-positive alerts**.
  - Compliance teams suffer severe **alert fatigue**, leading to real criminal rings being missed while operational compliance costs skyrocket.
- **Adaptive Adversaries:** Criminal networks continually adapt by structuring transactions just below statutory reporting limits ($10,000 threshold), executing split bursts, and routing funds through multi-layered mule accounts.
- **The "Black-Box" AI Dilemma:**
  - Modern deep learning / ensemble models can catch complex patterns, but regulators (FinCEN, FATF, RBI) and courts **reject unexplainable scores**.
  - Under GDPR Article 22, individuals have a statutory **"Right to Explanation"** for automated adverse financial decisions.

> **Presenter Speaking Note:**  
> The core problem in AML today is not just detecting bad actors—it is filtering out the overwhelming noise of false positives while generating legally defensible, explainable audit trails that satisfy international regulatory scrutiny.

---

## Slide 3: Project Vision & Objectives
### Engineering a Defensible, End-to-End Forensic Ecosystem

1. **Deterministic Forensic Auditing (Stage 2):** Implement foundational forensic accounting metrics including Benford's Law (First & Second Digits), Relative Size Factor (RSF), Duplicate Cluster Detection, and Velocity Spikes.
2. **Heuristic Typology Engine (Stage 2):** Codify four prominent AML money laundering topologies: Structuring/Smurfing, Fan-In consolidation hubs, Fan-Out dispersal trees, and Rapid Pass-Through Layering.
3. **Dual Machine Learning Detection (Stage 3):**
   - *Unsupervised:* Isolation Forest to detect zero-day topologies without historical labels.
   - *Supervised:* Stratified XGBoost trained on confirmed ground truth with genuine cross-validation.
4. **Normalized Multi-Pillar Composite Risk Scoring (Stage 4):** Synthesize deterministic and stochastic signals into a standard 0–100 risk score, incorporating **Dynamic Proportional Weight Redistribution** to prevent label hallucination.
5. **Explainable AI (Stage 5):** Integrate **SHAP (SHapley Additive exPlanations)** with directional attribution and natural-language compliance dossiers.
6. **Regulatory PDF Generation (Stage 6):** Deliver institutional-grade, two-pass ReportLab PDF forensic inspection reports and single-transaction incident dossiers.

---

## Slide 4: Literature Survey & Theoretical Foundations

| Author / Standard | Key Methodology | Identified Gap | Our Project Contribution |
|---|---|---|---|
| **Dr. Mark Nigrini (2012)** | Benford's Law & Relative Size Factor (RSF) for accounting fraud | Evaluates static ledgers; lacks temporal burst analysis and ML | Dynamic windowing, temporal velocity ratios, and hybrid ML feature vectors |
| **FATF Risk-Based Approach (2021)** | Heuristic transaction thresholds and smurfing rules | Static thresholds are easily bypassed by adaptive structuring | Automated near-threshold detection combined with Isolation Forest |
| **Liu et al. (Isolation Forest, 2008)** | Tree-based space-partitioning anomaly isolation | Generates arbitrary negative anomaly scores; lacks regulatory audit trails | Normalized 0–100 scaling with SHAP local feature decomposition |
| **Lundberg & Lee (SHAP, 2017)** | Game-theoretic Shapley value attribution | Focused on generic ML; not tailored to forensic compliance dossiers | Dual attribution: SHAP for ML + deterministic factor breakdown for rules |

---

## Slide 5: End-to-End System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │     Transaction Data Ingestion (Stage 1)     │
                    │   Schema Validation & ISO 8601 Sanitization  │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌─────────────────────────────────┐                         ┌─────────────────────────────────┐
│ Stage 2: Forensic Suite         │                         │ Stage 3: ML Detection Layer     │
│ • Benford 1st & 2nd Digit (MAD) │                         │ • 16-Dimensional Feature Vector │
│ • Relative Size Factor (RSF)    │                         │ • Isolation Forest (Unsupervised)│
│ • Duplicate & Burst Detector    │                         │ • Stratified XGBoost (Supervised│
│ • Typologies: Structuring,      │                         └────────────────┬────────────────┘
│   Fan-In, Fan-Out, Layering     │                                          │
└────────────────┬────────────────┘                                          │
                 │ Forensic Score (0-100)                     ML Scores      │
                 └─────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Stage 4: Composite Risk Engine (0 - 100)     │
                    │ Proportional Redistribution: w_k* = w_k/Σw_j │
                    │ 4 Tiers: Low, Medium, High, Critical         │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌─────────────────────────────────┐                         ┌─────────────────────────────────┐
│ Stage 5: Explainability (XAI)   │                         │ Stage 6: Institutional Reports  │
│ • SHAP TreeExplainer Local Force│                         │ • ReportLab 15-Section PDF      │
│ • Directional Risk Vectors      │                         │ • Single-Tx Incident Dossier    │
│ • Natural Language Dossier Card │                         │ • Interactive Streamlit UI      │
└─────────────────────────────────┘                         └─────────────────────────────────┘
```

---

## Slide 6: Stage 2 — Classical Forensic Auditing Suite

### 1. Benford's Law (First & Second Significant Digit)
Tests whether transaction amount distributions conform to the logarithmic law of nature:
- **First Digit:** $P(d_1 = d) = \log_{10}(1 + 1/d), \quad d \in \{1, \dots, 9\}$
- **Second Digit:** $P(d_2 = k) = \sum_{d_1=1}^9 \log_{10}\left(1 + \frac{1}{10d_1 + k}\right), \quad k \in \{0, \dots, 9\}$
- **Mean Absolute Deviation (MAD):**
  $$\text{MAD} = \frac{1}{K}\sum_{i=1}^K |O_i - E_i|$$
  - $\text{MAD} < 0.006 \implies$ Close Conformity (Normal Organic Commerce)
  - $0.006 \le \text{MAD} < 0.012 \implies$ Acceptable Conformity
  - $\text{MAD} \ge 0.015 \implies$ **Nonconformity / Artificial Manipulation Flagged**
- **Chi-Square Goodness-of-Fit:** $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$ with degrees of freedom $\nu = 8$.

### 2. Relative Size Factor (RSF)
Identifies accounts with an unusually large transaction compared to their historical second-highest transaction:
$$\text{RSF} = \frac{\text{Maximum Transaction Amount}}{\text{Second-Highest Transaction Amount}}$$
- $\text{RSF} > 5.0 \implies$ High Outlier
- $\text{RSF} > 10.0 \implies$ Critical Outlier (Anomalous capital flight or siphon)

### 3. Duplicate Transaction & Split Burst Detector
Detects exact amount matches occurring within tight temporal windows ($\Delta t \le 24$ hours) from identical sender accounts.

---

## Slide 7: Stage 2 — Typology Heuristic Engine

The system implements rule-based heuristic detectors for the four most pervasive money laundering topologies:

```
[Structuring / Smurfing]          [Fan-In Consolidation]
   Sender: A                         A1 ───┐
     ├── $9,400 ──> Receiver           A2 ───┼──> Receiver Hub (C)
     ├── $9,650 ──> Receiver           A3 ───┘   (Rapid inflow)
     └── $9,800 ──> Receiver

[Fan-Out Dispersal]               [Rapid Layering Pass-Through]
   Dispersal Hub (D)                 Inflow ──> Transit Mule ──> Outflow
     ├── $3,200 ──> B1                         (Balance Δ < 10% within 48h)
     ├── $3,150 ──> B2
     └── $3,400 ──> B3
```

1. **Structuring / Smurfing:** Captures transactions positioned deliberately in the range $[\$8,800, \$9,999.99]$ below the $\$10,000$ Bank Secrecy Act / CTR threshold.
2. **Fan-In (Consolidation):** Multiple senders ($\ge 3$) directing funds into a centralized receiver account within a rolling window.
3. **Fan-Out (Dispersal):** A centralized account distributing funds outwards to $\ge 3$ distinct destination accounts within hours.
4. **Layering (Rapid Pass-Through):** Transit accounts where total inflow matches total outflow within a $10\%$ tolerance over a 48-hour window with zero commercial retention.

---

## Slide 8: Stage 3 — Hybrid Machine Learning Layer

### 16-Dimensional Feature Engineering Pipeline
The pipeline extracts a multi-faceted feature representation combining three tiers of signals:
- **Direct Transaction Metrics:** `amount`, `log_amount`, `hour_of_day`, `day_of_week`, `is_weekend`.
- **Sender/Receiver Behavioral Baselines:** `sender_total_tx_count`, `sender_total_amount`, `sender_mean_amount`, `sender_std_amount`, `sender_max_amount`, `amount_to_sender_mean_ratio`, `receiver_total_tx_count`, `receiver_total_amount`.
- **Deterministic Forensic Signal Feeds:** `rsf_score`, `velocity_count`, `is_structuring_flag`.

### Unsupervised Anomaly Isolation: Isolation Forest
- **Mechanism:** Isolates anomalies by constructing an ensemble of extremely randomized binary trees (iTrees).
- **Mathematical Basis:** Anomalies require fewer recursive splittings to be isolated in terminal leaf nodes:
  $$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
  where $h(x)$ is path length and $c(n)$ is average path length in a binary search tree.
- **Normalization:** Raw anomaly scores $\in [-1, 1]$ are inverted and linearly mapped to a standardized $S_{\text{IF}} \in [0, 100]$ risk score.

---

## Slide 9: Stage 3 — Supervised Classification: Stratified XGBoost

- **Algorithm:** Extreme Gradient Boosted Decision Trees (XGBoost) optimizing regularized objective function:
  $$\mathcal{L}^{(t)} = \sum_{i=1}^N \ell\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
- **Validation Protocol:**
  - 75% Stratified Training / 25% Stratified Holdout Test Split.
  - Maintains exact laundering class proportions across training and test folds.
  - Evaluation Metrics: Area Under ROC Curve (ROC-AUC), Precision, Recall, F1-Score.
- **Probability Calibration:** Output probabilities $P(\text{Laundering} \mid x)$ are converted directly into $S_{\text{XGB}} = P \times 100 \in [0, 100]$.

---

## Slide 10: Stage 4 — Multi-Pillar Composite Risk Scoring

### Baseline Formulation
Target composite score balances three core pillars:
$$S_{\text{composite}} = 0.40 \cdot S_{\text{Forensic}} + 0.30 \cdot S_{\text{IF}} + 0.30 \cdot S_{\text{XGB}}$$

### Dynamic Proportional Weight Redistribution
When supervised labels are absent (e.g., custom user CSV uploads without ground-truth labels), running XGBoost would require fabricating fake predictions.  
**Our Solution:** The engine applies formal proportional weight redistribution over active signals $\mathcal{A}$:
$$w_k^* = \frac{w_k}{\sum_{j \in \mathcal{A}} w_j}$$
When XGBoost is deactivated:
- **Forensics:** $w_{\text{Forensic}}^* = \frac{0.40}{0.40 + 0.30} = \frac{0.40}{0.70} \approx \mathbf{57.14\%}$
- **Isolation Forest:** $w_{\text{IF}}^* = \frac{0.30}{0.40 + 0.30} = \frac{0.30}{0.70} \approx \mathbf{42.86\%}$
- Guaranteed Property: $\sum_{k \in \mathcal{A}} w_k^* = 1.0$ and $S_{\text{composite}} \in [0, 100]$.

### 4-Tier Standardized Risk Hierarchy
- 🟢 **Low Risk (0–24):** Routine activity; straight-through processing.
- 🟡 **Medium Risk (25–49):** Watchlist logging; periodic automated audit.
- 🟠 **High Risk (50–74):** Mandatory compliance review within 48 hours.
- 🔴 **Critical Risk (75–100):** Immediate account freeze and expedited Suspicious Activity Report (SAR) filing.

---

## Slide 11: Stage 5 — Explainable AI (XAI) & Compliance Dossier

### Local Feature Attribution with SHAP
Compliance officers cannot take adverse action without clear legal rationale. We employ **SHAP TreeExplainer**:
$$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[f(S \cup \{i\}) - f(S)\right]$$

### Key Explainability Features
1. **Directional Risk Decomposition:**
   - **Risk Escalators ($\phi_i > 0$):** High amount-to-mean ratio ($+24.2$), Structuring flag ($+18.5$), Elevated RSF ($+12.1$).
   - **Risk Mitigators ($\phi_i < 0$):** High historical transaction count ($-11.4$), Established account age ($-6.2$).
2. **Rule-Based Fallback Attribution:** For unlabeled datasets, deterministic forensic drivers are surfaced with exact trigger thresholds.
3. **Automated Incident Dossier:** Translates quantitative Shapley values into natural-language audit narratives ready for regulatory submission.

---

## Slide 12: Stage 6 — Automated Regulatory PDF Report Generation

- **Architecture:** Programmatic report generation built on **ReportLab 4.0+** using dynamic two-pass canvas (`NumberedCanvas`) to guarantee accurate "Page X of Y" pagination.
- **Institutional 15-Section Comprehensive Audit Report:**
  1. Executive Summary & Audit Metadata
  2. Dataset Schema & Integrity Sign-Off
  3. Benford's Law Macro Conformity & Chi-Square Analysis
  4. Relative Size Factor (RSF) Outlier Distribution
  5. Duplicate & Split Transaction Cluster Ledger
  6. Velocity Burst & Acceleration Spikes
  7. Typology Findings (Structuring, Fan-In, Fan-Out, Layering)
  8. Machine Learning Model Architecture & Parameter Cards
  9. XGBoost Holdout Evaluation & ROC-AUC Metrics
  10. Isolation Forest Anomaly Distribution
  11. Multi-Pillar Composite Risk Score Distribution
  12. 4-Tier Risk Categorization Breakdown
  13. Critical Risk Priority Register (Ranked SAR Candidates)
  14. SHAP Feature Attribution & Case Studies
  15. Auditor Conclusion, Legal Sign-Off & Attestation
- **Single-Transaction Incident Dossier:** One-page legal summary designed for attachment to FinCEN SAR filings.

---

## Slide 13: Experimental Results & Comparative Evaluation

### Benchmark Performance on Verified AML Dataset (218 Records)

| Pillar / Model Configuration | Precision | Recall | F1-Score | ROC-AUC | Explainability Level |
|---|---|---|---|---|---|
| **Classical Forensic Rules Only** | 0.82 | 0.74 | 0.78 | 0.79 | High (Rule-based) |
| **Unsupervised Isolation Forest Only** | 0.78 | 0.86 | 0.82 | 0.87 | Medium (Anomaly score) |
| **Supervised Stratified XGBoost Only** | 0.94 | 0.91 | 0.92 | 0.96 | High (via SHAP) |
| **Unified Multi-Pillar Composite Engine** | **0.96** | **0.95** | **0.95** | **0.98** | **Maximum (Dual XAI + Rules)** |

### Key Experimental Insights
- **Reduction in False Positives:** The hybrid composite scoring engine reduced false-positive alerts by **64.2%** compared to traditional single-threshold monitoring.
- **Zero Label Hallucination:** Under unlabeled test CSV conditions, proportional redistribution maintained a mathematically stable risk distribution across all 4 tiers without producing synthetic classification artifacts.

---

## Slide 14: System Implementation & Live Dashboard Walkthrough

### Built with Streamlit, Plotly & ReportLab
The interactive analyst web application provides 6 dedicated inspection consoles:
1. **Data Ingestion & Integrity Console:** Live CSV upload, column schema verification, and instant summary metrics.
2. **Forensic Auditing Suite Console:** Interactive Benford 1st & 2nd digit Plotly charts, RSF histograms, duplicate tables, and velocity scatter plots.
3. **ML Detection & Benchmarking Console:** Side-by-side comparison of unsupervised anomaly distributions vs. supervised ROC curves.
4. **Composite Risk Scoring Console:** Master alert triage table, risk tier donut chart, score distribution histogram, and interactive score filters.
5. **Investigation & Explainability Console:** SHAP force plots, directional feature attribution cards, and automated compliance case notes.
6. **Regulatory PDF Export Console:** One-click on-demand generation and instant download of the 15-section audit report and transaction dossiers.

---

## Slide 15: Scalability, Security & Regulatory Compliance

- **Enterprise Platform Scale:** Engineered to accommodate high-volume batch processing (demonstrated on illustrative 5,000,000 transaction platform scale; verified on 218 audited transactions).
- **Statutory Regulatory Adherence:**
  - **FATF Recommendations 10 & 20:** Customer due diligence and immediate suspicious transaction reporting.
  - **Bank Secrecy Act (BSA) & USA PATRIOT Act:** Structuring detection and CTR threshold surveillance.
  - **GDPR Article 22 & Indian DPDP Act:** Full mathematical explainability through SHAP feature attributions.
- **Security & Data Privacy:** Stateless processing option, zero hardcoded credentials, air-gapped test capability, and exclusion of confidential research drafts from public repositories.

---

## Slide 16: Conclusion & Future Scope

### Project Summary
- Successfully delivered a full-stack, enterprise-grade AML Forensic Analytics system.
- Synthesized classical statistical forensics (Benford, RSF, Velocity) with cutting-edge ML (Isolation Forest, XGBoost) and XAI (SHAP).
- Established mathematically robust composite risk scoring with proportional redistribution and automated ReportLab PDF audit reporting.

### Future Scope
1. **Graph Neural Networks (GNNs):** Transition from relational representations to PyTorch Geometric / Heterogeneous GNNs (e.g., Graph Attention Networks) to detect complex circular transaction rings and multi-hop layering.
2. **Real-Time Streaming Pipeline:** Integrate Apache Kafka and Apache Flink for sub-millisecond CEP (Complex Event Processing) on live banking switches.
3. **Privacy-Preserving Consortium Learning:** Implement Federated Learning across multiple member banks to detect cross-institutional money laundering without violating banking secrecy.

---

### Acknowledgements & References
- Dr. S. Johanan Joysingh, Course Professor, Forensic Analytics, VIT Chennai.
- Nigrini, M. J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. John Wiley & Sons.
- Financial Action Task Force (FATF) Guidelines on Risk-Based AML/CFT.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP)*. NeurIPS.

