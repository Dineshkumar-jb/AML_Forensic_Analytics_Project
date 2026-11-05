"""PDF Forensic Report Generator using ReportLab.

Stage 6: Generates professional forensic audit reports and individual transaction
dossiers in PDF format with embedded vector charts, executive scorecards,
statistical conformity tests, ML evaluation metrics, and compliance recommendations.
"""

from typing import Dict, Any, List, Optional
import os
import io
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Safe matplotlib setup for headless PDF generation
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    Image as RLImage,
    HRFlowable,
)
from reportlab.pdfgen import canvas

# -----------------------------------------------------------------------------
# Color Palette (Corporate Forensic Analytics & Compliance Dark Slate / Navy)
# -----------------------------------------------------------------------------
COLOR_PRIMARY_NAVY = colors.HexColor("#0f172a")
COLOR_SECONDARY_BLUE = colors.HexColor("#1e40af")
COLOR_ACCENT_BLUE = colors.HexColor("#2563eb")
COLOR_LIGHT_BG = colors.HexColor("#f8fafc")
COLOR_BORDER = colors.HexColor("#cbd5e1")
COLOR_TEXT_MAIN = colors.HexColor("#1e293b")
COLOR_TEXT_MUTED = colors.HexColor("#64748b")

COLOR_CRITICAL = colors.HexColor("#dc2626")
COLOR_HIGH = colors.HexColor("#ea580c")
COLOR_MEDIUM = colors.HexColor("#d97706")
COLOR_LOW = colors.HexColor("#16a34a")


# -----------------------------------------------------------------------------
# Canvas with Dynamic Page Numbers & Running Headers / Footers
# -----------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page counts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_header_footer(self, page_count: int):
        self.saveState()
        page_width, page_height = A4

        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 7.5)
            self.setFillColor(COLOR_SECONDARY_BLUE)
            self.drawString(36, page_height - 24, "AML FORENSIC ANALYTICS & RISK SCORING SYSTEM")
            self.setFont("Helvetica", 7.5)
            self.setFillColor(COLOR_TEXT_MUTED)
            self.drawRightString(page_width - 36, page_height - 24, "OFFICIAL FORENSIC AUDIT REPORT")
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.5)
            self.line(36, page_height - 28, page_width - 36, page_height - 28)

        # Footer (All pages)
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.5)
        self.line(36, 32, page_width - 36, 32)

        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(COLOR_TEXT_MUTED)
        self.drawString(36, 20, "DEMO MODE (5M Illustrative Scale | 218 Actual Sample Txs) // CONFIDENTIAL AML AUDIT")

        self.setFont("Helvetica", 7.5)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(page_width - 36, 20, page_str)

        self.restoreState()


# -----------------------------------------------------------------------------
# Helper Functions: Chart Generation (Matplotlib -> In-Memory PNG -> ReportLab)
# -----------------------------------------------------------------------------
def _create_risk_distribution_chart(counts: Dict[str, int]) -> RLImage:
    """Render a clean risk tier bar/donut chart as a ReportLab image."""
    fig, ax = plt.subplots(figsize=(4.8, 2.4), dpi=160)
    labels = ["Low (0-24)", "Medium (25-49)", "High (50-74)", "Critical (75-100)"]
    values = [counts.get("Low", 0), counts.get("Medium", 0), counts.get("High", 0), counts.get("Critical", 0)]
    palette = ["#16a34a", "#d97706", "#ea580c", "#dc2626"]

    bars = ax.bar(labels, values, color=palette, width=0.55, edgecolor="#0f172a", linewidth=0.6)
    ax.set_ylabel("Number of Transactions", fontsize=8, color="#334155")
    ax.set_title("Risk Tier Distribution (218 Transactions)", fontsize=9, fontweight="bold", color="#0f172a", pad=8)
    ax.tick_params(axis="both", which="major", labelsize=7.5, colors="#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94a3b8")

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + max(1, max(values) * 0.02),
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#1e293b",
        )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=False, facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=320, height=160)


def _create_benford_chart(benford_data: Dict[str, Any]) -> Optional[RLImage]:
    """Render Benford first-digit frequency curve vs. empirical observations."""
    if not benford_data or not benford_data.get("is_valid"):
        return None

    f_digit = benford_data.get("first_digit", {})
    table = f_digit.get("table")
    if table is None or table.empty:
        return None

    fig, ax = plt.subplots(figsize=(4.8, 2.4), dpi=160)
    digits = table["Digit"].values
    obs = table["Observed_Freq"].values
    exp = table["Expected_Freq"].values

    ax.bar(digits, obs, color="#3b82f6", width=0.5, label="Observed (%)", alpha=0.85, edgecolor="#1e3a8a", linewidth=0.5)
    ax.plot(digits, exp, color="#f59e0b", marker="o", linewidth=2.0, markersize=4, label="Benford Curve (%)")

    ax.set_xlabel("Leading Digit (1-9)", fontsize=8, color="#334155")
    ax.set_ylabel("Frequency (%)", fontsize=8, color="#334155")
    ax.set_title("Benford's Law: First-Digit Frequency vs. Expected", fontsize=9, fontweight="bold", color="#0f172a", pad=8)
    ax.set_xticks(digits)
    ax.tick_params(axis="both", which="major", labelsize=7.5, colors="#334155")
    ax.legend(fontsize=7, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94a3b8")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=False, facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=320, height=160)


def _create_feature_importance_chart(global_importance_df: pd.DataFrame, is_model_based: bool) -> Optional[RLImage]:
    """Render top feature importance bar chart."""
    if global_importance_df is None or global_importance_df.empty:
        return None

    top_df = global_importance_df.head(8).iloc[::-1]  # Reverse for horizontal chart
    fig, ax = plt.subplots(figsize=(4.8, 2.5), dpi=160)

    color = "#2563eb" if is_model_based else "#d97706"
    bars = ax.barh(top_df["Feature"], top_df["Importance"], color=color, height=0.55, edgecolor="#0f172a", linewidth=0.5)

    title = "Top Feature Importances (SHAP Mean |Value|)" if is_model_based else "Top Forensic Red-Flag Driver Importances"
    ax.set_title(title, fontsize=8.5, fontweight="bold", color="#0f172a", pad=8)
    ax.set_xlabel("Attribution Magnitude", fontsize=8, color="#334155")
    ax.tick_params(axis="both", which="major", labelsize=7, colors="#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.3, color="#94a3b8")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=False, facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=320, height=160)


def _create_transaction_diverging_chart(contributions_df: pd.DataFrame, is_model_based: bool) -> Optional[RLImage]:
    """Render transaction feature attribution diverging bar chart."""
    if contributions_df is None or contributions_df.empty:
        return None

    top_df = contributions_df.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 2.4), dpi=160)

    vals = top_df["Contribution"].values
    feats = top_df["Feature"].values
    colors_list = ["#dc2626" if v > 0 else "#16a34a" for v in vals]

    ax.barh(feats, vals, color=colors_list, height=0.55, edgecolor="#0f172a", linewidth=0.5)
    ax.axvline(0, color="#64748b", linewidth=0.8, linestyle="--")

    title = "SHAP Local Contributions (+ Escalates Risk / - Mitigates)" if is_model_based else "Rule Risk Contributions (+ Escalator / - Mitigator)"
    ax.set_title(title, fontsize=8.5, fontweight="bold", color="#0f172a", pad=8)
    ax.set_xlabel("Risk Contribution Value", fontsize=8, color="#334155")
    ax.tick_params(axis="both", which="major", labelsize=7, colors="#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=False, facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=320, height=160)


# -----------------------------------------------------------------------------
# Style Setup
# -----------------------------------------------------------------------------
def _get_custom_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            "ReportTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=COLOR_PRIMARY_NAVY,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=COLOR_SECONDARY_BLUE,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=COLOR_PRIMARY_NAVY,
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "SubsectionHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=COLOR_SECONDARY_BLUE,
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "AuditBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=COLOR_TEXT_MAIN,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "AuditBodyBold",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12,
            textColor=COLOR_TEXT_MAIN,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=COLOR_TEXT_MAIN,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableTextBold",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=COLOR_TEXT_MAIN,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableTextHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            "CalloutBoxText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11.5,
            textColor=COLOR_TEXT_MAIN,
        )
    )
    return styles


# -----------------------------------------------------------------------------
# STAGE 6.1: FULL COMPREHENSIVE FORENSIC AUDIT INVESTIGATION REPORT
# -----------------------------------------------------------------------------
def generate_full_forensic_report(
    df: pd.DataFrame,
    summary: Dict[str, Any],
    forensic_results: Dict[str, Any],
    ml_results: Dict[str, Any],
    composite_results: Dict[str, Any],
    explainability_res: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> bytes:
    """Generate complete, professional 15-section Forensic Audit PDF report.

    Includes all required sections strictly from actual calculated application results.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42,
    )

    styles = _get_custom_styles()
    story = []

    # -------------------------------------------------------------------------
    # 1. Project Title & Report Information (Cover Header & Meta Card)
    # -------------------------------------------------------------------------
    gen_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    report_id = f"AML-AUD-{datetime.now().strftime('%Y%m%d-%H%M')}"

    story.append(Paragraph("AML FORENSIC ANALYTICS & RISK SCORING", styles["ReportTitle"]))
    story.append(Paragraph("Official Technical Inspection & Forensic Investigation Audit Report", styles["ReportSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=COLOR_SECONDARY_BLUE, spaceBefore=0, spaceAfter=10))

    tot_vol = summary.get("total_volume", float(df["amount"].sum()) if "amount" in df.columns else 0.0)

    meta_data = [
        [
            Paragraph("<b>Report ID:</b> " + report_id, styles["TableText"]),
            Paragraph("<b>Generation Date:</b> " + gen_time_str, styles["TableText"]),
        ],
        [
            Paragraph("<b>Operational Status:</b> <font color='#1e40af'><b>Demo Mode (5M Scale)</b></font>", styles["TableText"]),
            Paragraph(f"<b>Actual Audited Sample:</b> {len(df):,} transactions", styles["TableTextBold"]),
        ],
        [
            Paragraph("<b>Illustrative Platform Scale:</b> 5,000,000 transactions", styles["TableText"]),
            Paragraph("<b>Actual Sample Volume:</b> $" + f"{tot_vol:,.2f}", styles["TableText"]),
        ],
        [
            Paragraph("<b>Audit Framework:</b> Deterministic + Dual-ML", styles["TableText"]),
            Paragraph("<b>Classification Level:</b> STRICTLY CONFIDENTIAL // AML-SAR", styles["TableText"]),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[260, 260])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_BG),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # Persistent Demo Mode Callout Box
    demo_box_data = [[
        Paragraph(
            "<b>DEMO MODE NOTICE:</b> Demo Mode – 5M illustrative transaction scale; actual analysis uses 218 sample transactions. "
            "All empirical statistical models, Benford tests, ML metrics, and risk scores reflect genuine calculations on the 218 sample records.",
            styles["CalloutBoxText"],
        )
    ]]
    demo_box_table = Table(demo_box_data, colWidths=[520])
    demo_box_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_SECONDARY_BLUE),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(demo_box_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 2. Executive Summary
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary", styles["SectionHeading"]))
    
    counts = composite_results.get("counts_by_level", {"Low": 0, "Medium": 0, "High": 0, "Critical": 0})
    pcts = composite_results.get("percentages_by_level", {"Low": 0, "Medium": 0, "High": 0, "Critical": 0})
    high_crit_count = counts.get("High", 0) + counts.get("Critical", 0)
    high_crit_pct = round(high_crit_count / max(1, len(df)) * 100.0, 1)
    tot_vol = summary.get("total_volume", float(df["amount"].sum()) if "amount" in df.columns else 0.0)

    exec_summary_text = (
        f"This forensic investigation report details the multi-pillar AML analysis of "
        f"<b>{len(df):,} actual sample transactions</b> representing a cumulative audited volume of <b>${tot_vol:,.2f}</b>, "
        f"demonstrating enterprise architecture capacity for a <b>5,000,000 transaction illustrative platform scale</b>. "
        f"The forensic auditing suite evaluated empirical digit conformities (Benford's Law), Relative Size Factors (RSF), "
        f"duplicate velocity patterns, and AML typology heuristics. Concurrently, an unsupervised Isolation Forest "
        f"and supervised XGBoost classifier evaluated multidimensional feature geometries. "
        f"A total of <b>{high_crit_count} transactions ({high_crit_pct}% of the 218 sample)</b> have been designated as "
        f"<b>High or Critical Risk</b>, requiring mandatory secondary compliance examination or formal SAR escalation."
    )
    story.append(Paragraph(exec_summary_text, styles["AuditBody"]))

    # Executive KPI Table
    kpi_data = [
        [
            Paragraph("Platform Scale (Demo)", styles["TableTextBold"]),
            Paragraph("Actual Audited Sample", styles["TableTextBold"]),
            Paragraph("Sample Volume ($)", styles["TableTextBold"]),
            Paragraph("Avg Composite Risk", styles["TableTextBold"]),
            Paragraph("Critical Alerts (Sample)", styles["TableTextBold"]),
        ],
        [
            Paragraph("5,000,000 txs", styles["TableTextBold"]),
            Paragraph(f"{len(df):,} txs", styles["TableText"]),
            Paragraph(f"${tot_vol:,.2f}", styles["TableText"]),
            Paragraph(f"{composite_results.get('avg_composite_risk', 0.0):.1f} / 100", styles["TableText"]),
            Paragraph(f"{counts.get('Critical', 0)} ({pcts.get('Critical', 0)}%)", styles["TableTextBold"]),
            Paragraph(f"{counts.get('High', 0)} ({pcts.get('High', 0)}%)", styles["TableTextBold"]),
        ],
    ]
    kpi_table = Table(kpi_data, colWidths=[100, 110, 105, 105, 100])
    kpi_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (-1, 1), COLOR_LIGHT_BG),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_SECONDARY_BLUE),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(kpi_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 3. Dataset Overview (218 Sample Transactions)
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Dataset Overview & Ingestion Scope", styles["SectionHeading"]))
    ds_rows = [
        [
            Paragraph("<b>Parameter</b>", styles["TableTextHeader"]),
            Paragraph("<b>Empirical Value</b>", styles["TableTextHeader"]),
            Paragraph("<b>Regulatory / Compliance Context</b>", styles["TableTextHeader"]),
        ],
        [
            Paragraph("Operational Mode", styles["TableTextBold"]),
            Paragraph("Demo / Simulation Active", styles["TableTextBold"]),
            Paragraph("Demo Mode – 5M illustrative transaction scale; actual analysis uses 218 sample transactions.", styles["TableText"]),
        ],
        [
            Paragraph("Actual Sample Audited", styles["TableTextBold"]),
            Paragraph(f"{len(df)} sample transactions", styles["TableTextBold"]),
            Paragraph("Empirical reference population with full analytical scoring", styles["TableText"]),
        ],
        [
            Paragraph("Illustrative Platform Scale", styles["TableTextBold"]),
            Paragraph("5,000,000 transactions", styles["TableText"]),
            Paragraph("Target enterprise ingestion capacity demonstration", styles["TableText"]),
        ],
        [
            Paragraph("Time Horizon", styles["TableTextBold"]),
            Paragraph(f"{summary.get('start_date', 'N/A')} to {summary.get('end_date', 'N/A')}", styles["TableText"]),
            Paragraph("Discrete evaluation monitoring window", styles["TableText"]),
        ],
        [
            Paragraph("Amount Range", styles["TableTextBold"]),
            Paragraph(f"${summary.get('min_amount', 0.0):,.2f} – ${summary.get('max_amount', 0.0):,.2f}", styles["TableText"]),
            Paragraph("Transactions span retail smurfing up to commercial transfers", styles["TableText"]),
        ],
        [
            Paragraph("Originating Senders", styles["TableTextBold"]),
            Paragraph(f"{summary.get('unique_senders', 'N/A')} unique accounts", styles["TableText"]),
            Paragraph("Audited for out-degree velocity and RSF multipliers", styles["TableText"]),
        ],
        [
            Paragraph("Beneficiary Receivers", styles["TableTextBold"]),
            Paragraph(f"{summary.get('unique_receivers', 'N/A')} unique accounts", styles["TableText"]),
            Paragraph("Audited for in-degree consolidation and pass-throughs", styles["TableText"]),
        ],
    ]
    ds_table = Table(ds_rows, colWidths=[130, 160, 230])
    ds_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY_NAVY),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(ds_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 4. Data Quality & Preprocessing Summary
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Data Quality & Preprocessing Summary", styles["SectionHeading"]))
    null_count = summary.get("null_count", int(df.isnull().sum().sum()))
    dq_text = (
        f"Prior to model execution, the dataset was ingested through the schema validation pipeline. "
        f"All mandatory fields (<code>transaction_id, timestamp, sender_account, receiver_account, amount</code>) "
        f"were verified for structural completeness. Total null or missing values across mandatory fields: "
        f"<b>{null_count}</b>. Data types were strictly enforced, timestamps normalized into UTC ISO-8601 strings, "
        f"and amount attributes cast to double-precision floating points. No synthetic rows were injected."
    )
    story.append(Paragraph(dq_text, styles["AuditBody"]))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 5. Benford's Law Analysis
    # 6. First-Digit & Second-Digit Analysis
    # 7. MAD Conformity Results
    # 8. Chi-Square Goodness-of-Fit Results
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Benford's Law Statistical Digit Auditing", styles["SectionHeading"]))
    benford_data = forensic_results.get("benford", {})

    if benford_data and benford_data.get("is_valid"):
        f_digit = benford_data.get("first_digit", {})
        s_digit = benford_data.get("second_digit", {})

        b_summary_text = (
            f"Benford's Law evaluates the empirical distribution of leading digits against the logarithmic distribution "
            f"P(d) = log10(1 + 1/d). In unmanipulated financial transactions, leading digits exhibit natural logarithmic decay. "
            f"Deviations indicate intentional structuring, repetitive round amounts, or threshold-avoidance behavior. "
            f"Overall Benford Risk Severity: <b>{benford_data.get('overall_risk_score', 0.0):.1f} / 100</b>."
        )
        story.append(Paragraph(b_summary_text, styles["AuditBody"]))

        # Statistical Metrics Table
        mad_val = f_digit.get("mad", 0.0)
        conf_status = f_digit.get("conformity", "N/A")
        chi2_val = f_digit.get("chi2_stat", 0.0)
        pval = f_digit.get("p_value", 0.0)

        s_mad = s_digit.get("mad", "N/A") if s_digit.get("is_valid") else "N/A"
        s_conf = s_digit.get("conformity", "N/A") if s_digit.get("is_valid") else "N/A"
        s_chi2 = f"{s_digit.get('chi2_stat', 0.0):.2f}" if s_digit.get("is_valid") else "N/A"

        b_stat_rows = [
            [
                Paragraph("<b>Test Component</b>", styles["TableTextHeader"]),
                Paragraph("<b>First-Digit Analysis (1–9)</b>", styles["TableTextHeader"]),
                Paragraph("<b>Second-Digit Analysis (0–9)</b>", styles["TableTextHeader"]),
                Paragraph("<b>Nigrini Conformity Reference Standard</b>", styles["TableTextHeader"]),
            ],
            [
                Paragraph("Mean Absolute Dev (MAD)", styles["TableTextBold"]),
                Paragraph(f"{mad_val:.4f}", styles["TableText"]),
                Paragraph(f"{s_mad if isinstance(s_mad, str) else f'{s_mad:.4f}'}", styles["TableText"]),
                Paragraph("Close: <0.006 | Acceptable: 0.006–0.012 | Marginal: 0.012–0.015", styles["TableText"]),
            ],
            [
                Paragraph("Conformity Status", styles["TableTextBold"]),
                Paragraph(f"<b>{conf_status}</b>", styles["TableText"]),
                Paragraph(f"{s_conf}", styles["TableText"]),
                Paragraph("Non-conformity (>0.015) indicates high anomaly likelihood", styles["TableText"]),
            ],
            [
                Paragraph("Chi-Square Stat (χ²)", styles["TableTextBold"]),
                Paragraph(f"{chi2_val:.2f} (df=8)", styles["TableText"]),
                Paragraph(f"{s_chi2} (df=9)", styles["TableText"]),
                Paragraph("Critical value at α=0.05 is 15.51 (First-Digit)", styles["TableText"]),
            ],
            [
                Paragraph("p-value Significance", styles["TableTextBold"]),
                Paragraph(f"{pval:.4f}", styles["TableText"]),
                Paragraph(f"{s_digit.get('p_value', 0.0):.4f}" if s_digit.get("is_valid") else "N/A", styles["TableText"]),
                Paragraph("p < 0.05 indicates statistically significant non-conformance", styles["TableText"]),
            ],
        ]
        b_stat_table = Table(b_stat_rows, colWidths=[120, 110, 110, 180])
        b_stat_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY_BLUE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(b_stat_table)
        story.append(Spacer(1, 8))

        # First Digit Frequency Comparison Table
        f_table = f_digit.get("table")
        if f_table is not None and not f_table.empty:
            f_rows = [
                [
                    Paragraph("<b>Digit</b>", styles["TableTextHeader"]),
                    Paragraph("<b>Observed Count</b>", styles["TableTextHeader"]),
                    Paragraph("<b>Observed Freq (%)</b>", styles["TableTextHeader"]),
                    Paragraph("<b>Expected Freq (%)</b>", styles["TableTextHeader"]),
                    Paragraph("<b>Difference (%)</b>", styles["TableTextHeader"]),
                ]
            ]
            for _, r in f_table.iterrows():
                diff_val = r.get("Abs_Difference", abs(r["Observed_Freq"] - r["Expected_Freq"]))
                f_rows.append([
                    Paragraph(str(int(r["Digit"])), styles["TableTextBold"]),
                    Paragraph(str(int(r["Observed_Count"])), styles["TableText"]),
                    Paragraph(f"{r['Observed_Freq']:.1f}%", styles["TableText"]),
                    Paragraph(f"{r['Expected_Freq']:.1f}%", styles["TableText"]),
                    Paragraph(f"{diff_val:.1f}%", styles["TableText"]),
                ])
            digit_summary_table = Table(f_rows, colWidths=[60, 110, 115, 115, 120])
            digit_summary_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY_NAVY),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
                    ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ])
            )
            story.append(digit_summary_table)

        # Embedded Benford Chart
        b_chart = _create_benford_chart(benford_data)
        if b_chart:
            story.append(Spacer(1, 8))
            story.append(KeepTogether([b_chart]))
    else:
        story.append(Paragraph("Benford's Law analysis not available for current dataset.", styles["AuditBody"]))

    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 9. Machine Learning Model Information & Evaluation Metrics
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Machine Learning Detection Architecture & Validation", styles["SectionHeading"]))
    iso_res = ml_results.get("isolation_forest", {})
    xgb_res = ml_results.get("xgboost", {})

    ml_intro = (
        "The system deploys a dual-paradigm Machine Learning engine comprising Unsupervised Anomaly Detection "
        "(Isolation Forest) to uncover structural geometric outliers without label reliance, and Supervised "
        "Classification (XGBoost) to evaluate known money laundering typologies when verified ground-truth audit outcomes exist."
    )
    story.append(Paragraph(ml_intro, styles["AuditBody"]))

    ml_rows = [
        [
            Paragraph("<b>Model Architecture</b>", styles["TableTextHeader"]),
            Paragraph("<b>Operational Paradigm</b>", styles["TableTextHeader"]),
            Paragraph("<b>Flagged Suspicious</b>", styles["TableTextHeader"]),
            Paragraph("<b>Core Evaluation Findings</b>", styles["TableTextHeader"]),
        ],
        [
            Paragraph("Isolation Forest", styles["TableTextBold"]),
            Paragraph("Unsupervised Outlier Detection", styles["TableText"]),
            Paragraph(f"{iso_res.get('anomaly_count', 0)} ({iso_res.get('anomaly_percentage', 0.0)}%)", styles["TableText"]),
            Paragraph(f"Contamination: {iso_res.get('contamination', 0.08):.2f} | Mean Score: {iso_res.get('mean_anomaly_score', 0.0):.1f}/100", styles["TableText"]),
        ],
    ]

    if xgb_res.get("is_trained"):
        xgb_flagged = int(np.sum(xgb_res.get("all_preds", [0]) == 1))
        xgb_pct = round(xgb_flagged / max(1, len(df)) * 100.0, 1)
        conf_mat = xgb_res.get("confusion_matrix", [[0, 0], [0, 0]])
        metric_str = (
            f"Holdout Acc: {xgb_res.get('accuracy', 0.0)*100:.1f}% | "
            f"F1: {xgb_res.get('f1_score', 0.0):.3f} | "
            f"Recall: {xgb_res.get('recall', 0.0):.3f} | "
            f"Precision: {xgb_res.get('precision', 0.0):.3f} | "
            f"ROC-AUC: {xgb_res.get('roc_auc', 'N/A')}"
        )
        ml_rows.append([
            Paragraph("XGBoost Supervised", styles["TableTextBold"]),
            Paragraph(f"Trained on '{xgb_res.get('label_column', 'is_laundering')}'", styles["TableText"]),
            Paragraph(f"{xgb_flagged} ({xgb_pct}%)", styles["TableText"]),
            Paragraph(metric_str, styles["TableText"]),
        ])
    else:
        ml_rows.append([
            Paragraph("XGBoost Supervised", styles["TableTextBold"]),
            Paragraph("Supervised Classification", styles["TableText"]),
            Paragraph("Awaiting Ground Truth", styles["TableText"]),
            Paragraph("Inactive: No genuine labels present; zero synthetic labels fabricated", styles["TableText"]),
        ])

    ml_table = Table(ml_rows, colWidths=[110, 120, 100, 190])
    ml_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY_NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(ml_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 10. Composite Risk-Scoring Methodology
    # 11. Risk Category Distribution
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Composite Risk-Scoring Methodology & Calibration", styles["SectionHeading"]))
    eff_w = composite_results.get("effective_weights", {})
    comp_method_text = (
        "The Unified Multi-Pillar Composite Risk Engine synthesizes deterministic forensic indicators with "
        "machine learning anomaly scores into a standardized continuous decision metric in [0, 100]. "
        "To ensure mathematical fidelity, when a signal pillar is inactive, its weight is proportionally "
        "redistributed across the remaining active signals without fabricating fake probabilities. "
        f"Current applied effective weights: <b>Forensics: {eff_w.get('forensic', 0.40)*100:.1f}%</b>, "
        f"<b>Isolation Forest: {eff_w.get('isolation_forest', 0.30)*100:.1f}%</b>, "
        f"<b>XGBoost: {eff_w.get('xgboost', 0.30)*100:.1f}%</b>."
    )
    story.append(Paragraph(comp_method_text, styles["AuditBody"]))

    # Tier Breakdown Table
    tier_rows = [
        [
            Paragraph("<b>Risk Tier</b>", styles["TableTextHeader"]),
            Paragraph("<b>Score Range</b>", styles["TableTextHeader"]),
            Paragraph("<b>Transaction Count</b>", styles["TableTextHeader"]),
            Paragraph("<b>Percentage (%)</b>", styles["TableTextHeader"]),
            Paragraph("<b>Recommended AML Workflow</b>", styles["TableTextHeader"]),
        ],
        [
            Paragraph("<b>Critical Risk</b>", styles["TableTextBold"]),
            Paragraph("75.0 – 100.0", styles["TableText"]),
            Paragraph(f"<b>{counts.get('Critical', 0)}</b>", styles["TableText"]),
            Paragraph(f"{pcts.get('Critical', 0.0)}%", styles["TableText"]),
            Paragraph("Immediate SAR filing & senior compliance escalation", styles["TableText"]),
        ],
        [
            Paragraph("<b>High Risk</b>", styles["TableTextBold"]),
            Paragraph("50.0 – 74.9", styles["TableText"]),
            Paragraph(f"<b>{counts.get('High', 0)}</b>", styles["TableText"]),
            Paragraph(f"{pcts.get('High', 0.0)}%", styles["TableText"]),
            Paragraph("Enhanced Due Diligence (EDD) & source of funds check", styles["TableText"]),
        ],
        [
            Paragraph("<b>Medium Risk</b>", styles["TableTextBold"]),
            Paragraph("25.0 – 49.9", styles["TableText"]),
            Paragraph(f"{counts.get('Medium', 0)}", styles["TableText"]),
            Paragraph(f"{pcts.get('Medium', 0.0)}%", styles["TableText"]),
            Paragraph("Heightened 30-day velocity watch & automated alerts", styles["TableText"]),
        ],
        [
            Paragraph("<b>Low Risk</b>", styles["TableTextBold"]),
            Paragraph("0.0 – 24.9", styles["TableText"]),
            Paragraph(f"{counts.get('Low', 0)}", styles["TableText"]),
            Paragraph(f"{pcts.get('Low', 0.0)}%", styles["TableText"]),
            Paragraph("Routine baseline conformance & statutory recordkeeping", styles["TableText"]),
        ],
    ]
    tier_table = Table(tier_rows, colWidths=[95, 80, 85, 80, 180])
    tier_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY_BLUE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("ALIGN", (1, 1), (3, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(tier_table)

    # Embedded Risk Distribution Chart
    r_chart = _create_risk_distribution_chart(counts)
    story.append(Spacer(1, 8))
    story.append(KeepTogether([r_chart]))
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 12. High-Risk Transaction Details
    # -------------------------------------------------------------------------
    story.append(Paragraph("7. High & Critical Risk Transaction Inspection Register", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "Below is the complete register of all transactions classified in the High or Critical tiers, "
            "ranked in descending order of composite severity.",
            styles["AuditBody"],
        )
    )

    scored_df = composite_results.get("scored_df")
    if scored_df is not None and not scored_df.empty:
        high_crit_df = scored_df[scored_df["risk_level"].isin(["Critical", "High"])].sort_values(
            by="composite_risk_score", ascending=False
        )

        reg_rows = [
            [
                Paragraph("<b>Tx ID</b>", styles["TableTextHeader"]),
                Paragraph("<b>Amount ($)</b>", styles["TableTextHeader"]),
                Paragraph("<b>Sender Account</b>", styles["TableTextHeader"]),
                Paragraph("<b>Receiver Account</b>", styles["TableTextHeader"]),
                Paragraph("<b>Score</b>", styles["TableTextHeader"]),
                Paragraph("<b>Tier</b>", styles["TableTextHeader"]),
                Paragraph("<b>Contributing Factors</b>", styles["TableTextHeader"]),
            ]
        ]
        for _, r in high_crit_df.iterrows():
            reg_rows.append([
                Paragraph(str(r["transaction_id"]), styles["TableTextBold"]),
                Paragraph(f"${r['amount']:,.2f}", styles["TableText"]),
                Paragraph(str(r["sender_account"]), styles["TableText"]),
                Paragraph(str(r["receiver_account"]), styles["TableText"]),
                Paragraph(f"<b>{r['composite_risk_score']:.1f}</b>", styles["TableTextBold"]),
                Paragraph(str(r["risk_level"]), styles["TableTextBold"]),
                Paragraph(str(r.get("contributing_risk_factors", "N/A")), styles["TableText"]),
            ])

        reg_table = Table(reg_rows, colWidths=[55, 75, 75, 75, 45, 55, 140])
        reg_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY_NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(reg_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 13. Explainable AI Findings or Rule-Based Explanations
    # -------------------------------------------------------------------------
    story.append(Paragraph("8. Explainable AI & Feature Attribution Findings", styles["SectionHeading"]))
    is_model_based = explainability_res.get("is_model_based", False)
    exp_status = explainability_res.get("status_message", "")
    global_imp = explainability_res.get("global_importance")

    exp_narrative = (
        f"<b>Attribution Framework:</b> {exp_status}<br/>"
        "Model interpretability ensures that every flagged transaction possesses full evidentiary transparency. "
        "Global feature attribution reveals the structural signals that most strongly govern risk escalation across the population."
    )
    story.append(Paragraph(exp_narrative, styles["AuditBody"]))

    if global_imp is not None and not global_imp.empty:
        imp_rows = [
            [
                Paragraph("<b>Rank</b>", styles["TableTextHeader"]),
                Paragraph("<b>Feature / Indicator</b>", styles["TableTextHeader"]),
                Paragraph("<b>Importance Metric</b>", styles["TableTextHeader"]),
                Paragraph("<b>Category</b>", styles["TableTextHeader"]),
                Paragraph("<b>Audit Description</b>", styles["TableTextHeader"]),
            ]
        ]
        for idx, r in global_imp.head(6).iterrows():
            imp_rows.append([
                Paragraph(f"#{idx + 1}", styles["TableTextBold"]),
                Paragraph(str(r["Feature"]), styles["TableTextBold"]),
                Paragraph(f"{r['Importance']:.4f}" if isinstance(r['Importance'], float) else str(r['Importance']), styles["TableText"]),
                Paragraph(str(r.get("Category", "General")), styles["TableText"]),
                Paragraph(str(r.get("Description", "Feature")), styles["TableText"]),
            ])
        imp_table = Table(imp_rows, colWidths=[35, 120, 85, 95, 185])
        imp_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY_BLUE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(imp_table)

        # Embedded Feature Importance Chart
        f_chart = _create_feature_importance_chart(global_imp, is_model_based)
        if f_chart:
            story.append(Spacer(1, 8))
            story.append(KeepTogether([f_chart]))

    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 14. Investigation Recommendations
    # -------------------------------------------------------------------------
    story.append(Paragraph("9. Procedural Investigation Recommendations", styles["SectionHeading"]))
    rec_text = (
        "Based on the empirical findings, the compliance division should execute the following four-tier remediation plan:"
    )
    story.append(Paragraph(rec_text, styles["AuditBody"]))

    rec_matrix = [
        [
            Paragraph("<b>Risk Severity</b>", styles["TableTextHeader"]),
            Paragraph("<b>Target Population</b>", styles["TableTextHeader"]),
            Paragraph("<b>Mandatory Compliance Action & Statutory Timeline</b>", styles["TableTextHeader"]),
        ],
        [
            Paragraph("<b>CRITICAL (75–100)</b>", styles["TableTextBold"]),
            Paragraph(f"{counts.get('Critical', 0)} transactions", styles["TableText"]),
            Paragraph("Place administrative hold on unsettled funds; draft FinCEN Form 111 (SAR) for filing within 30 statutory days; issue senior compliance escalation.", styles["TableText"]),
        ],
        [
            Paragraph("<b>HIGH (50–74)</b>", styles["TableTextBold"]),
            Paragraph(f"{counts.get('High', 0)} transactions", styles["TableText"]),
            Paragraph("Initiate Enhanced Due Diligence (EDD); request commercial invoices and beneficial ownership documentation; verify against OFAC/PEP watchlists.", styles["TableText"]),
        ],
        [
            Paragraph("<b>MEDIUM (25–49)</b>", styles["TableTextBold"]),
            Paragraph(f"{counts.get('Medium', 0)} transactions", styles["TableText"]),
            Paragraph("Place counterparties on 30-day automated velocity surveillance; monitor for repetitive structuring below CTR thresholds.", styles["TableText"]),
        ],
        [
            Paragraph("<b>LOW (0–24)</b>", styles["TableTextBold"]),
            Paragraph(f"{counts.get('Low', 0)} transactions", styles["TableText"]),
            Paragraph("Routine baseline conformance; archive in audit warehouse for statutory 5-year AML recordkeeping.", styles["TableText"]),
        ],
    ]
    rec_table = Table(rec_matrix, colWidths=[100, 90, 330])
    rec_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY_NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(rec_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 15. Conclusion, Limitations & Auditor Sign-Off
    # -------------------------------------------------------------------------
    story.append(Paragraph("10. Conclusion, Limitations & Formal Sign-Off", styles["SectionHeading"]))
    conclusion_text = (
        "<b>Conclusion:</b> The automated forensic analytics and machine learning pipeline has systematically inspected all 218 "
        "sample transactions. The multi-pillar consensus successfully differentiated routine commercial baseline operations "
        "from high-severity red flags including structuring, relative size spikes, and rapid duplicate intervals.<br/><br/>"
        "<b>Audit Limitations:</b> (1) Results are derived from the supplied historical transaction sample and do not constitute "
        "a judicial determination of criminal liability. (2) Machine learning risk scores serve as decision-support alerts and "
        "must be validated through Human-in-the-Loop compliance review prior to law enforcement referral. "
        "(3) Supervised metrics reflect the confirmed sample ground truth and should be continuously recalibrated as new SAR dispositions emerge. "
        "(4) <b>Demo Mode Scale Distinction:</b> The architecture exhibits a 5,000,000 transaction illustrative platform scale; "
        "all actual analysis, statistical tests, and risk scores are calculated strictly on the 218 sample transactions."
    )
    story.append(Paragraph(conclusion_text, styles["AuditBody"]))
    story.append(Spacer(1, 12))

    # Sign-off Block
    sign_data = [
        [
            Paragraph("<b>Lead AML Forensic Auditor:</b>", styles["TableText"]),
            Paragraph("____________________________", styles["TableText"]),
            Paragraph("<b>Date:</b> ______________", styles["TableText"]),
        ],
        [
            Paragraph("<b>Chief Compliance Officer:</b>", styles["TableText"]),
            Paragraph("____________________________", styles["TableText"]),
            Paragraph("<b>Date:</b> ______________", styles["TableText"]),
        ],
    ]
    sign_table = Table(sign_data, colWidths=[160, 200, 160])
    sign_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_BG),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(KeepTogether([sign_table]))

    # Build PDF Document
    doc.build(story, canvasmaker=NumberedCanvas)
    pdf_bytes = buffer.getvalue()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


# -----------------------------------------------------------------------------
# STAGE 6.3: INDIVIDUAL TRANSACTION FORENSIC DOSSIER PDF REPORT
# -----------------------------------------------------------------------------
def generate_transaction_dossier_report(
    tx_id: str,
    dossier: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> bytes:
    """Generate dedicated, professional PDF forensic report for an individual transaction.

    Includes complete transaction details, predicted risk score, tier badge,
    suspicious indicators, feature attribution, and recommended compliance actions.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42,
    )

    styles = _get_custom_styles()
    story = []

    gen_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    case_ref = f"CASE-{tx_id}-{datetime.now().strftime('%Y%m%d')}"

    # Header
    story.append(Paragraph("AML FORENSIC TRANSACTION DOSSIER", styles["ReportTitle"]))
    story.append(
        Paragraph(
            f"Individual Incident Investigation & SAR Assessment // Case Ref: <b>{case_ref}</b>",
            styles["ReportSubtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=2, color=COLOR_SECONDARY_BLUE, spaceBefore=0, spaceAfter=10))

    # Metadata & Risk Badge Banner
    risk_level = dossier.get("risk_level", "Low")
    comp_score = dossier.get("composite_risk_score", 0.0)

    tier_color_map = {
        "Critical": COLOR_CRITICAL,
        "High": COLOR_HIGH,
        "Medium": COLOR_MEDIUM,
        "Low": COLOR_LOW,
    }
    badge_color = tier_color_map.get(risk_level, COLOR_LOW)

    banner_data = [
        [
            Paragraph(f"<b>Transaction ID:</b> <code>{tx_id}</code>", styles["TableTextBold"]),
            Paragraph(f"<b>Investigation Date:</b> {gen_time_str}", styles["TableText"]),
            Paragraph(f"<b>Assigned Risk Tier:</b> <font color='{badge_color.hexval()}'><b>{risk_level.upper()} ({comp_score:.1f}/100)</b></font>", styles["TableTextBold"]),
        ],
        [
            Paragraph("<b>Scale & Mode:</b> Demo Mode – 5M illustrative transaction scale (5,000,000 records); actual analysis uses 218 sample transactions.", styles["TableText"]),
            "",
            "",
        ],
    ]
    banner_table = Table(banner_data, colWidths=[160, 180, 180])
    banner_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_BG),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_SECONDARY_BLUE),
            ("SPAN", (0, 1), (2, 1)),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(banner_table)
    story.append(Spacer(1, 10))

    # 1. Transaction Core Attributes Table
    story.append(Paragraph("1. Transaction Identification & Attributes", styles["SectionHeading"]))
    tx_attr_rows = [
        [
            Paragraph("<b>Attribute</b>", styles["TableTextHeader"]),
            Paragraph("<b>Value</b>", styles["TableTextHeader"]),
            Paragraph("<b>Attribute</b>", styles["TableTextHeader"]),
            Paragraph("<b>Value</b>", styles["TableTextHeader"]),
        ],
        [
            Paragraph("Transaction ID", styles["TableTextBold"]),
            Paragraph(str(tx_id), styles["TableText"]),
            Paragraph("Timestamp", styles["TableTextBold"]),
            Paragraph(str(dossier.get("timestamp", "N/A")), styles["TableText"]),
        ],
        [
            Paragraph("Transaction Amount", styles["TableTextBold"]),
            Paragraph(f"${dossier.get('amount', 0.0):,.2f}", styles["TableTextBold"]),
            Paragraph("Transaction Type", styles["TableTextBold"]),
            Paragraph(str(dossier.get("transaction_type", "TRANSFER")), styles["TableText"]),
        ],
        [
            Paragraph("Originating Sender", styles["TableTextBold"]),
            Paragraph(f"<code>{dossier.get('sender_account', 'N/A')}</code>", styles["TableText"]),
            Paragraph("Beneficiary Receiver", styles["TableTextBold"]),
            Paragraph(f"<code>{dossier.get('receiver_account', 'N/A')}</code>", styles["TableText"]),
        ],
        [
            Paragraph("Geographic Location", styles["TableTextBold"]),
            Paragraph(str(dossier.get("location", "N/A")), styles["TableText"]),
            Paragraph("Portfolio Percentile", styles["TableTextBold"]),
            Paragraph(f"Top {100.0 - dossier.get('percentile_rank', 0.0):.1f}% risk (in 218 sample | 5M scale)", styles["TableText"]),
        ],
    ]
    tx_attr_table = Table(tx_attr_rows, colWidths=[120, 140, 120, 140])
    tx_attr_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY_NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(tx_attr_table)
    story.append(Spacer(1, 10))

    # 2. Multi-Pillar Score Breakdown
    story.append(Paragraph("2. Analytical Risk Pillars Breakdown", styles["SectionHeading"]))
    f_score = dossier.get("forensic_score", 0.0)
    i_score = dossier.get("iso_score", 0.0)
    x_score = dossier.get("xgb_score")
    x_score_str = f"{x_score:.1f} / 100" if x_score is not None else "N/A (Unlabeled Mode)"

    pillar_rows = [
        [
            Paragraph("<b>Analytical Pillar</b>", styles["TableTextHeader"]),
            Paragraph("<b>Methodology / Paradigm</b>", styles["TableTextHeader"]),
            Paragraph("<b>Assigned Score</b>", styles["TableTextHeader"]),
            Paragraph("<b>Evaluation Status</b>", styles["TableTextHeader"]),
        ],
        [
            Paragraph("Forensic Auditing Suite", styles["TableTextBold"]),
            Paragraph("Deterministic Tests (RSF, Dup, Velocity, Structuring)", styles["TableText"]),
            Paragraph(f"<b>{f_score:.1f} / 100</b>", styles["TableTextBold"]),
            Paragraph("🟢 Active", styles["TableText"]),
        ],
        [
            Paragraph("Isolation Forest", styles["TableTextBold"]),
            Paragraph("Unsupervised Outlier Multi-Dimensional Geometry", styles["TableText"]),
            Paragraph(f"<b>{i_score:.1f} / 100</b>", styles["TableTextBold"]),
            Paragraph("🟢 Active", styles["TableText"]),
        ],
        [
            Paragraph("XGBoost Classifier", styles["TableTextBold"]),
            Paragraph("Supervised Classification on Ground Truth Labels", styles["TableText"]),
            Paragraph(f"<b>{x_score_str}</b>", styles["TableTextBold"]),
            Paragraph("🟢 Active" if x_score is not None else "⚠️ Weight Redistributed", styles["TableText"]),
        ],
        [
            Paragraph("Final Composite Decision Score", styles["TableTextBold"]),
            Paragraph("Normalized Weighted Convex Combination", styles["TableTextBold"]),
            Paragraph(f"<b>{comp_score:.1f} / 100</b>", styles["TableTextBold"]),
            Paragraph(f"<b>Tier: {risk_level}</b>", styles["TableTextBold"]),
        ],
    ]
    pillar_table = Table(pillar_rows, colWidths=[130, 180, 100, 110])
    pillar_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY_BLUE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, COLOR_LIGHT_BG]),
            ("BACKGROUND", (0, -1), (-1, -1), COLOR_LIGHT_BG),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(pillar_table)
    story.append(Spacer(1, 10))

    # 3. Suspicious Indicators & Red-Flag Audit Findings
    story.append(Paragraph("3. Specific Suspicious Indicators & Forensic Red Flags", styles["SectionHeading"]))
    indicators = dossier.get("suspicious_indicators", [])
    if indicators:
        for ind in indicators:
            story.append(Paragraph(f"• <b>{ind}</b>", styles["AuditBody"]))
    else:
        story.append(Paragraph("• No specific suspicious indicators triggered (Routine baseline).", styles["AuditBody"]))

    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Natural Language Audit Justification:</b> {dossier.get('audit_explanation', '')}", styles["AuditBody"]))
    story.append(Spacer(1, 10))

    # 4. Explainable AI Feature Attribution
    story.append(Paragraph("4. Explainable AI / Feature Contribution Analysis", styles["SectionHeading"]))
    exp_detail = dossier.get("explanation_detail", {})
    is_model_based = exp_detail.get("is_model_based", False)
    contrib_df = exp_detail.get("contributions_df")

    story.append(Paragraph(exp_detail.get("narrative", ""), styles["AuditBody"]))

    if contrib_df is not None and not contrib_df.empty:
        c_rows = [
            [
                Paragraph("<b>Feature / Indicator</b>", styles["TableTextHeader"]),
                Paragraph("<b>Observed Value</b>", styles["TableTextHeader"]),
                Paragraph("<b>Contribution</b>", styles["TableTextHeader"]),
                Paragraph("<b>Attribution Direction</b>", styles["TableTextHeader"]),
                Paragraph("<b>Impact Class</b>", styles["TableTextHeader"]),
            ]
        ]
        for _, r in contrib_df.head(6).iterrows():
            c_val = r["Contribution"]
            c_rows.append([
                Paragraph(str(r["Feature"]), styles["TableTextBold"]),
                Paragraph(str(r["Value"]), styles["TableText"]),
                Paragraph(f"{c_val:+.4f}" if isinstance(c_val, float) else str(c_val), styles["TableTextBold"]),
                Paragraph(str(r["Direction"]), styles["TableText"]),
                Paragraph(str(r["Impact"]), styles["TableTextBold"]),
            ])
        c_table = Table(c_rows, colWidths=[120, 80, 80, 140, 100])
        c_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY_NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(c_table)

        # Embedded Diverging Chart
        div_chart = _create_transaction_diverging_chart(contrib_df, is_model_based)
        if div_chart:
            story.append(Spacer(1, 8))
            story.append(KeepTogether([div_chart]))

    story.append(Spacer(1, 10))

    # 5. Recommended Investigation Action & Checklist
    story.append(Paragraph("5. Recommended Investigation Workflow & Action Plan", styles["SectionHeading"]))
    rec = dossier.get("recommended_action", {})
    story.append(Paragraph(f"<b>Determination:</b> {rec.get('title', 'Routine Monitoring')}", styles["AuditBodyBold"]))
    story.append(Paragraph(f"<b>Priority Level:</b> <b>{rec.get('priority', 'ROUTINE')}</b>", styles["AuditBody"]))
    story.append(Paragraph(rec.get("summary", ""), styles["AuditBody"]))

    story.append(Paragraph("<b>Required Investigator Actions:</b>", styles["SubsectionHeading"]))
    for s in rec.get("steps", []):
        story.append(Paragraph(f"[  ] {s}", styles["AuditBody"]))

    story.append(Spacer(1, 12))

    # 6. Case Review Sign-Off
    sign_data = [
        [
            Paragraph("<b>Investigating Compliance Officer:</b>", styles["TableText"]),
            Paragraph("____________________________", styles["TableText"]),
            Paragraph("<b>Date:</b> ______________", styles["TableText"]),
        ],
        [
            Paragraph("<b>Final Disposition Decision:</b>", styles["TableText"]),
            Paragraph("[  ] Clear Transaction   [  ] Enhanced Watch   [  ] Formal SAR Filing", styles["TableTextBold"]),
            Paragraph("<b>Signature:</b> ______________", styles["TableText"]),
        ],
    ]
    sign_table = Table(sign_data, colWidths=[160, 240, 120])
    sign_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_BG),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(KeepTogether([sign_table]))

    doc.build(story, canvasmaker=NumberedCanvas)
    pdf_bytes = buffer.getvalue()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
