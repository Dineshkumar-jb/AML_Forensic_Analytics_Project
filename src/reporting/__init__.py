"""Reporting module for AML Forensic Analytics.

Stage 6: Generates professional forensic audit reports and individual transaction
dossiers in PDF format using ReportLab with embedded charts, executive scorecards,
and compliance recommendations.
"""

from src.reporting.pdf_generator import (
    generate_full_forensic_report,
    generate_transaction_dossier_report,
)

__all__ = [
    "generate_full_forensic_report",
    "generate_transaction_dossier_report",
]

