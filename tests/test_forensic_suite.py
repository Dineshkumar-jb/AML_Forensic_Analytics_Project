"""Automated validation tests for Stage 2 Forensic Analytics Suite."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import generate_sample_dataset
from src.forensic.benford import analyze_benford
from src.forensic.rsf import calculate_rsf
from src.forensic.duplicates import detect_duplicates
from src.forensic.velocity import analyze_velocity
from src.forensic.typologies import analyze_all_typologies
from src.forensic.engine import run_forensic_suite


def test_forensic_modules():
    print("--- 1. Generating / Loading Sample Data ---")
    df = generate_sample_dataset(save_to_disk=False)
    assert len(df) > 0, "DataFrame is empty!"
    print(f"Sample data records: {len(df)}")

    print("\n--- 2. Testing Benford's Law ---")
    b_res = analyze_benford(df)
    assert b_res["is_valid"], f"Benford test failed: {b_res.get('error')}"
    f_res = b_res["first_digit"]
    print(f"Benford First-Digit MAD: {f_res['mad']:.4f}")
    print(f"Benford Conformity: {f_res['conformity']}")
    print(f"Benford Risk Score (0-100): {b_res['overall_risk_score']}")
    assert 0 <= b_res["overall_risk_score"] <= 100

    print("\n--- 3. Testing Relative Size Factor (RSF) ---")
    r_res = calculate_rsf(df)
    assert r_res["is_valid"], f"RSF test failed: {r_res.get('error')}"
    print(f"Max RSF: {r_res['max_rsf']:.2f}x")
    print(f"Anomalous accounts count: {r_res['anomaly_count']}")
    print(f"RSF Risk Score (0-100): {r_res['risk_score']}")
    assert r_res["anomaly_count"] > 0, "Expected at least one seeded RSF anomaly!"
    assert 0 <= r_res["risk_score"] <= 100

    print("\n--- 4. Testing Duplicate Detection ---")
    d_res = detect_duplicates(df)
    assert d_res["is_valid"], f"Duplicate test failed: {d_res.get('error')}"
    print(f"Duplicate transactions count: {d_res['duplicate_count']}")
    print(f"Duplicate volume ($): {d_res['duplicate_volume']}")
    print(f"Duplicate Risk Score (0-100): {d_res['risk_score']}")
    assert d_res["duplicate_count"] > 0, "Expected at least one seeded duplicate transaction!"
    assert 0 <= d_res["risk_score"] <= 100

    print("\n--- 5. Testing Transaction Velocity ---")
    v_res = analyze_velocity(df)
    assert v_res["is_valid"], f"Velocity test failed: {v_res.get('error')}"
    print(f"Burst accounts count: {v_res['burst_account_count']}")
    print(f"Max burst count: {v_res['max_burst_count']}")
    print(f"Velocity Risk Score (0-100): {v_res['risk_score']}")
    assert 0 <= v_res["risk_score"] <= 100

    print("\n--- 6. Testing Typology Indicators ---")
    t_res = analyze_all_typologies(df)
    print(f"Structuring count: {t_res['structuring']['detected_count']}")
    print(f"Fan-in hubs: {t_res['fan_in']['hub_count']}")
    print(f"Fan-out sources: {t_res['fan_out']['source_count']}")
    print(f"Layering transits: {t_res['layering']['detected_transit_accounts']}")
    print(f"Overall Typology Score (0-100): {t_res['overall_risk_score']}")
    assert t_res["structuring"]["detected_count"] > 0, "Expected seeded structuring transactions!"
    assert t_res["fan_in"]["hub_count"] > 0, "Expected seeded fan-in hubs!"

    print("\n--- 7. Testing Forensic Orchestrator & Scorecard ---")
    suite_res = run_forensic_suite(df)
    assert suite_res["is_valid"]
    print("Aggregate Forensic Pillar Score:", suite_res["aggregate_forensic_score"])
    print("\nExecutive Scorecard:")
    print(suite_res["scorecard"].to_string())

    print("\n[SUCCESS] ALL FORENSIC AUDITING TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_forensic_modules()
