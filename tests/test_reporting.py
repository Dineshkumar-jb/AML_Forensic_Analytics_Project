"""Unit and integration tests for Stage 6: Forensic Audit Report Generation."""

import pytest
import PyPDF2
import io
from pathlib import Path

from src.utils.data_loader import load_csv, get_dataset_summary
from src.forensic import run_forensic_suite
from src.ml import run_ml_pipeline
from src.scoring import calculate_composite_risk
from src.explainability import compute_explainability, get_investigation_dossier
from src.reporting import generate_full_forensic_report, generate_transaction_dossier_report
from src.config import SAMPLE_DATA_PATH


@pytest.fixture(scope="module")
def prepared_audit_data():
    """Load sample data and execute complete analytical pipeline."""
    df, _ = load_csv(SAMPLE_DATA_PATH)
    summary = get_dataset_summary(df)
    f_res = run_forensic_suite(df)
    ml_res = run_ml_pipeline(df, f_res.get("enriched_df"))
    comp_res = calculate_composite_risk(df, f_res, ml_res)
    exp_res = compute_explainability(
        ml_results=ml_res,
        features_df=ml_res["features_df"],
        feature_names=ml_res["feature_names"],
        composite_results=comp_res,
    )
    return df, summary, f_res, ml_res, comp_res, exp_res


def test_full_forensic_report_pdf_generation(prepared_audit_data):
    """Verify that complete 15-section Forensic Audit PDF report compiles properly."""
    df, summary, f_res, ml_res, comp_res, exp_res = prepared_audit_data

    pdf_bytes = generate_full_forensic_report(
        df=df,
        summary=summary,
        forensic_results=f_res,
        ml_results=ml_res,
        composite_results=comp_res,
        explainability_res=exp_res,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 20000  # Report contains tables and embedded charts
    assert pdf_bytes.startswith(b"%PDF-")

    # Read with PyPDF2
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    num_pages = len(reader.pages)
    assert num_pages >= 3

    # Check text content of first page
    first_page_text = reader.pages[0].extract_text()
    assert "AML FORENSIC ANALYTICS" in first_page_text or "FORENSIC" in first_page_text

    # Verify Demo Mode and scale distinction in full report
    all_text = " ".join([page.extract_text() for page in reader.pages])
    assert "DEMO MODE" in all_text or "Demo Mode" in all_text
    assert "5,000,000" in all_text
    assert "218" in all_text


def test_transaction_dossier_pdf_generation(prepared_audit_data):
    """Verify that individual transaction dossier PDF compiles properly."""
    df, summary, f_res, ml_res, comp_res, exp_res = prepared_audit_data

    target_tx = "TX_1218"
    dossier = get_investigation_dossier(
        tx_id=target_tx,
        scored_df=comp_res["scored_df"],
        features_df=ml_res["features_df"],
        explainability_res=exp_res,
        forensic_results=f_res,
        ml_results=ml_res,
    )

    pdf_bytes = generate_transaction_dossier_report(
        tx_id=target_tx,
        dossier=dossier,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 10000
    assert pdf_bytes.startswith(b"%PDF-")

    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    page1_text = reader.pages[0].extract_text()
    assert target_tx in page1_text
    assert "DEMO MODE" in page1_text or "Demo Mode" in page1_text
    assert "5,000,000" in page1_text


def test_pdf_generation_unlabeled_dataset():
    """Verify report generation handles unlabeled dataset without failing or inventing data."""
    df, _ = load_csv(SAMPLE_DATA_PATH)
    df_unlabeled = df.drop(columns=["is_laundering"])
    summary = get_dataset_summary(df_unlabeled)
    f_res = run_forensic_suite(df_unlabeled)
    ml_res = run_ml_pipeline(df_unlabeled, f_res.get("enriched_df"))
    comp_res = calculate_composite_risk(df_unlabeled, f_res, ml_res)
    exp_res = compute_explainability(
        ml_results=ml_res,
        features_df=ml_res["features_df"],
        feature_names=ml_res["feature_names"],
        composite_results=comp_res,
    )

    pdf_bytes = generate_full_forensic_report(
        df=df_unlabeled,
        summary=summary,
        forensic_results=f_res,
        ml_results=ml_res,
        composite_results=comp_res,
        explainability_res=exp_res,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 20000
    assert pdf_bytes.startswith(b"%PDF-")

