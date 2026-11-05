"""Benford's Law forensic module for AML transaction analysis.

Evaluates first-digit and second-digit distributions against Benford's logarithmic law:
P(d) = log10(1 + 1/d)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def get_first_digit(val: float) -> int:
    """Extract first non-zero digit from a numeric amount."""
    try:
        s = f"{val:.6f}".replace(".", "").lstrip("0")
        return int(s[0]) if s else 0
    except (ValueError, IndexError):
        return 0


def get_second_digit(val: float) -> int:
    """Extract second significant digit from a numeric amount."""
    try:
        s = f"{val:.6f}".replace(".", "").lstrip("0")
        return int(s[1]) if len(s) > 1 else -1
    except (ValueError, IndexError):
        return -1


def calculate_benford_first_digit(amounts: pd.Series) -> Dict[str, Any]:
    """Calculate first-digit observed vs expected distribution, MAD, and Chi-Square."""
    valid_amounts = amounts[amounts > 0].dropna()
    total_count = len(valid_amounts)

    if total_count < 10:
        return {
            "error": "Insufficient data points for Benford analysis (requires at least 10 positive amounts).",
            "is_valid": False,
        }

    first_digits = valid_amounts.apply(get_first_digit)
    first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]

    counts = first_digits.value_counts().to_dict()

    digits = list(range(1, 10))
    expected_probs = [np.log10(1 + 1 / d) for d in digits]
    observed_counts = [counts.get(d, 0) for d in digits]
    total_valid = len(first_digits)
    observed_probs = [c / total_valid for c in observed_counts]
    expected_counts = [p * total_valid for p in expected_probs]

    # Mean Absolute Deviation (MAD)
    abs_diffs = [abs(o - e) for o, e in zip(observed_probs, expected_probs)]
    mad = float(np.mean(abs_diffs))

    # Chi-Square Test
    # Add a small epsilon to avoid zero expected counts
    chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

    # Nigrini Conformity Standard for First-Digit:
    # Close conformity: MAD <= 0.006
    # Acceptable conformity: 0.006 < MAD <= 0.012
    # Marginally acceptable: 0.012 < MAD <= 0.015
    # Nonconformity: MAD > 0.015
    if mad <= 0.006:
        conformity = "Close Conformity"
        conformity_grade = "Low Deviation"
    elif mad <= 0.012:
        conformity = "Acceptable Conformity"
        conformity_grade = "Moderate Deviation"
    elif mad <= 0.015:
        conformity = "Marginally Acceptable"
        conformity_grade = "Elevated Deviation"
    else:
        conformity = "Nonconformity"
        conformity_grade = "Severe Anomaly"

    # Normalize to 0-100 risk score
    # MAD of 0 -> 0 score; MAD >= 0.035 -> 100 score
    normalized_score = min(100.0, max(0.0, (mad / 0.035) * 100.0))

    df_results = pd.DataFrame({
        "Digit": digits,
        "Observed_Count": observed_counts,
        "Expected_Count": [round(c, 1) for c in expected_counts],
        "Observed_Freq": [round(p * 100, 2) for p in observed_probs],
        "Expected_Freq": [round(p * 100, 2) for p in expected_probs],
        "Abs_Difference": [round(d * 100, 2) for d in abs_diffs],
    })

    return {
        "is_valid": True,
        "table": df_results,
        "sample_size": total_valid,
        "mad": mad,
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "conformity": conformity,
        "conformity_grade": conformity_grade,
        "risk_score": round(normalized_score, 1),
    }


def calculate_benford_second_digit(amounts: pd.Series) -> Dict[str, Any]:
    """Calculate second-digit distribution (0-9) according to Benford's Law."""
    valid_amounts = amounts[amounts >= 10].dropna()
    total_count = len(valid_amounts)

    if total_count < 15:
        return {
            "error": "Insufficient records >= 10 for second-digit analysis.",
            "is_valid": False,
        }

    second_digits = valid_amounts.apply(get_second_digit)
    second_digits = second_digits[(second_digits >= 0) & (second_digits <= 9)]

    counts = second_digits.value_counts().to_dict()
    digits = list(range(0, 10))

    # Second digit expected probability: sum_{k=1}^9 log10(1 + 1 / (10k + d2))
    expected_probs = []
    for d2 in digits:
        p = sum(np.log10(1 + 1 / (10 * k + d2)) for k in range(1, 10))
        expected_probs.append(p)

    total_valid = len(second_digits)
    observed_counts = [counts.get(d, 0) for d in digits]
    observed_probs = [c / total_valid for c in observed_counts]
    expected_counts = [p * total_valid for p in expected_probs]

    abs_diffs = [abs(o - e) for o, e in zip(observed_probs, expected_probs)]
    mad = float(np.mean(abs_diffs))

    chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

    # Nigrini Conformity for Second Digit:
    # Close: <= 0.008, Acceptable: <= 0.018, Nonconformity: > 0.022
    if mad <= 0.008:
        conformity = "Close Conformity"
    elif mad <= 0.018:
        conformity = "Acceptable Conformity"
    else:
        conformity = "Nonconformity"

    normalized_score = min(100.0, max(0.0, (mad / 0.030) * 100.0))

    df_results = pd.DataFrame({
        "Digit": digits,
        "Observed_Count": observed_counts,
        "Expected_Count": [round(c, 1) for c in expected_counts],
        "Observed_Freq": [round(p * 100, 2) for p in observed_probs],
        "Expected_Freq": [round(p * 100, 2) for p in expected_probs],
        "Abs_Difference": [round(d * 100, 2) for d in abs_diffs],
    })

    return {
        "is_valid": True,
        "table": df_results,
        "sample_size": total_valid,
        "mad": mad,
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "conformity": conformity,
        "risk_score": round(normalized_score, 1),
    }


def analyze_benford(df: pd.DataFrame, amount_col: str = "amount") -> Dict[str, Any]:
    """Execute complete Benford analysis suite (First and Second digit)."""
    if amount_col not in df.columns:
        return {"is_valid": False, "error": f"Column '{amount_col}' not found."}

    first_digit_res = calculate_benford_first_digit(df[amount_col])
    second_digit_res = calculate_benford_second_digit(df[amount_col])

    # Overall Benford Risk Score: 70% first digit, 30% second digit (or 100% first if second unavailable)
    if first_digit_res.get("is_valid"):
        f_score = first_digit_res["risk_score"]
        if second_digit_res.get("is_valid"):
            s_score = second_digit_res["risk_score"]
            overall_score = round(0.70 * f_score + 0.30 * s_score, 1)
        else:
            overall_score = f_score
    else:
        overall_score = 0.0

    return {
        "is_valid": first_digit_res.get("is_valid", False),
        "first_digit": first_digit_res,
        "second_digit": second_digit_res,
        "overall_risk_score": overall_score,
    }
