"""Configuration constants and schema definitions for the AML Forensic Analytics System."""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_DATA_PATH = DATA_DIR / "sample_transactions.csv"

# Application Metadata
APP_NAME = "Transaction Monitoring & Risk Scoring for AML Detection"
APP_SUBTITLE = "Forensic Analytics, ML Anomaly Detection & Typology Engine"
APP_VERSION = "v1.0 (Stage 6 Full Release - Demo Mode)"

# Demo / Simulation Mode Constants
DEMO_MODE_ACTIVE = True
ILLUSTRATIVE_PLATFORM_SCALE = 5_000_000
ACTUAL_SAMPLE_SIZE = 218
DEMO_MODE_BANNER_TEXT = "Demo Mode – 5M illustrative transaction scale; actual analysis uses 218 sample transactions."

# Schema Definition
REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "sender_account",
    "receiver_account",
    "amount",
]

OPTIONAL_COLUMNS = [
    "transaction_type",
    "sender_balance_before",
    "receiver_balance_before",
    "location",
    "is_laundering",  # Optional ground truth for benchmarking
]

# Risk Scoring Tier Boundaries (Stage 4 Standard: 4 tiers)
RISK_TIERS_STAGE4 = {
    "Low": {"min": 0, "max": 24, "color": "#22c55e", "badge": "🟢 Low (0-24)"},
    "Medium": {"min": 25, "max": 49, "color": "#f59e0b", "badge": "🟡 Medium (25-49)"},
    "High": {"min": 50, "max": 74, "color": "#f97316", "badge": "🟠 High (50-74)"},
    "Critical": {"min": 75, "max": 100, "color": "#ef4444", "badge": "🔴 Critical (75-100)"},
}

# Standard RISK_TIERS alias maintaining backwards compatibility
RISK_TIERS = RISK_TIERS_STAGE4

# Default Weights for Composite Risk Score (Stage 4)
DEFAULT_COMPOSITE_WEIGHTS = {
    "forensic": 0.40,           # Forensic auditing signals (RSF, Duplicates, Velocity, Typologies)
    "isolation_forest": 0.30,   # Unsupervised Isolation Forest anomaly score
    "xgboost": 0.30,            # Supervised XGBoost probability (when valid labels exist)
}

# Backward compatibility alias for legacy references
DEFAULT_WEIGHTS = DEFAULT_COMPOSITE_WEIGHTS

# Reporting Thresholds
DEFAULT_CURRENCY_SYMBOL = "$"
STRUCTURING_THRESHOLD = 10000.0  # Common CTR (Currency Transaction Report) threshold

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "REPORTS_DIR",
    "SAMPLE_DATA_PATH",
    "APP_NAME",
    "APP_SUBTITLE",
    "APP_VERSION",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "RISK_TIERS_STAGE4",
    "RISK_TIERS",
    "DEFAULT_COMPOSITE_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "DEFAULT_CURRENCY_SYMBOL",
    "STRUCTURING_THRESHOLD",
    "DEMO_MODE_ACTIVE",
    "ILLUSTRATIVE_PLATFORM_SCALE",
    "ACTUAL_SAMPLE_SIZE",
    "DEMO_MODE_BANNER_TEXT",
]
