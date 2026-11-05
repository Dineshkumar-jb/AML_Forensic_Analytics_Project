"""Data loading, validation, and sample dataset generation utilities."""

import io
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

from src.config import REQUIRED_COLUMNS, OPTIONAL_COLUMNS, SAMPLE_DATA_PATH


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """Validate DataFrame against required schema and basic integrity rules.

    Returns:
        (is_valid, missing_columns, warnings)
    """
    if df is None or df.empty:
        return False, ["Dataset is empty"], []

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, missing_cols, []

    warnings = []

    # Check for nulls in required columns
    for col in REQUIRED_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            warnings.append(f"Column '{col}' has {null_count} null/missing values.")

    # Check amounts validity
    if not pd.api.types.is_numeric_dtype(df["amount"]):
        warnings.append("Column 'amount' contains non-numeric values that will need coercion.")
    else:
        negative_amounts = (df["amount"] <= 0).sum()
        if negative_amounts > 0:
            warnings.append(f"Found {negative_amounts} transactions with zero or negative amounts.")

    return True, [], warnings


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, strip whitespace, and cast core data types."""
    df = df.copy()

    # Strip column name whitespaces
    df.columns = [c.strip().lower() for c in df.columns]

    # Coerce amounts to float
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # Coerce timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Coerce accounts and IDs to string
    for col in ["transaction_id", "sender_account", "receiver_account"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def load_csv(file_input: Any) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load CSV from either a file path, string buffer, or Streamlit UploadedFile.

    Returns:
        (DataFrame, error_message)
    """
    try:
        if hasattr(file_input, "read"):
            df = pd.read_csv(file_input)
        else:
            df = pd.read_csv(file_input)

        if df.empty:
            return None, "The uploaded CSV file is empty."

        df = clean_dataframe(df)
        is_valid, missing_cols, _ = validate_dataframe(df)

        if not is_valid:
            return None, f"Missing required columns: {', '.join(missing_cols)}"

        return df, None

    except pd.errors.EmptyDataError:
        return None, "The CSV file appears to be empty."
    except pd.errors.ParserError as e:
        return None, f"Malformed CSV structure: {str(e)}"
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level summary metrics of the loaded transactions."""
    if df is None or df.empty:
        return {}

    ts = df["timestamp"].dropna()
    start_date = ts.min().strftime("%Y-%m-%d %H:%M") if not ts.empty else "N/A"
    end_date = ts.max().strftime("%Y-%m-%d %H:%M") if not ts.empty else "N/A"

    return {
        "total_records": len(df),
        "total_volume": float(df["amount"].sum()),
        "average_amount": float(df["amount"].mean()),
        "max_amount": float(df["amount"].max()),
        "min_amount": float(df["amount"].min()),
        "unique_senders": int(df["sender_account"].nunique()),
        "unique_receivers": int(df["receiver_account"].nunique()),
        "start_date": start_date,
        "end_date": end_date,
        "null_count": int(df[REQUIRED_COLUMNS].isnull().sum().sum()),
    }


def generate_sample_dataset(num_records: int = 250, save_to_disk: bool = True) -> pd.DataFrame:
    """Generate a realistic synthetic transaction dataset seeded with AML typologies.

    Includes:
    - Normal peer-to-peer / retail transactions
    - Structuring (transactions just below $10,000)
    - Duplicate rapid transactions
    - Fan-in pattern (multiple senders to single mule)
    - Fan-out pattern (single account dispersing to multiple targets)
    - Extreme relative size outliers (RSF)
    """
    np.random.seed(42)
    base_time = datetime(2026, 3, 1, 9, 0, 0)

    records: List[Dict[str, Any]] = []
    tx_counter = 1000

    # 1. Normal transactions (~75% of data)
    normal_senders = [f"ACC_NORM_{i:03d}" for i in range(1, 30)]
    normal_receivers = [f"ACC_RECV_{i:03d}" for i in range(1, 30)]
    tx_types = ["TRANSFER", "WIRE", "PAYMENT", "CASH_DEPOSIT"]

    for i in range(int(num_records * 0.72)):
        tx_counter += 1
        delta_mins = int(np.random.exponential(scale=180)) + (i * 15)
        # Lognormal distribution for natural monetary amounts (median around $150-$500)
        amt = round(float(np.random.lognormal(mean=5.5, sigma=1.2)), 2)
        amt = max(10.0, min(amt, 8500.0))

        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": (base_time + timedelta(minutes=delta_mins)).strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": np.random.choice(normal_senders),
            "receiver_account": np.random.choice(normal_receivers),
            "amount": amt,
            "transaction_type": np.random.choice(tx_types),
            "location": np.random.choice(["New York", "Chicago", "Dallas", "London", "Toronto"]),
            "is_laundering": 0,
        })

    # 2. Structuring / Smurfing Typology: Amounts just below $10,000 regulatory limit
    structuring_sender = "ACC_SUSP_SMURF_01"
    structuring_receiver = "ACC_SHELL_CORP_01"
    smurf_time = base_time + timedelta(days=2, hours=10)

    for i in range(8):
        tx_counter += 1
        # Amounts clustered between $9,400 and $9,950
        amt = round(float(np.random.uniform(9400, 9950)), 2)
        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": (smurf_time + timedelta(minutes=i * 22)).strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": structuring_sender,
            "receiver_account": structuring_receiver,
            "amount": amt,
            "transaction_type": "CASH_DEPOSIT",
            "location": "Miami",
            "is_laundering": 1,
        })

    # 3. Duplicate Transaction Typology: Identical amount & accounts within minutes
    dup_sender = "ACC_DUP_SEND_02"
    dup_receiver = "ACC_DUP_RECV_02"
    dup_time = base_time + timedelta(days=3, hours=14)
    dup_amt = 4850.00

    for i in range(4):
        tx_counter += 1
        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": (dup_time + timedelta(minutes=i * 2)).strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": dup_sender,
            "receiver_account": dup_receiver,
            "amount": dup_amt,
            "transaction_type": "TRANSFER",
            "location": "Las Vegas",
            "is_laundering": 1,
        })

    # 4. Fan-in Pattern: 10 distinct senders transferring into 1 central collector
    mule_collector = "ACC_MULE_HUB_03"
    fan_time = base_time + timedelta(days=4, hours=8)

    for i in range(12):
        tx_counter += 1
        feeder = f"ACC_FEEDER_{i:02d}"
        amt = round(float(np.random.uniform(2500, 4800)), 2)
        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": (fan_time + timedelta(minutes=i * 12)).strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": feeder,
            "receiver_account": mule_collector,
            "amount": amt,
            "transaction_type": "WIRE",
            "location": "Panama City",
            "is_laundering": 1,
        })

    # 5. Fan-out / Dispersal: The collector disperses funds out immediately
    fan_out_time = fan_time + timedelta(hours=4)
    for i in range(8):
        tx_counter += 1
        receiver_dest = f"ACC_OFFSHORE_{i:02d}"
        amt = round(float(np.random.uniform(4000, 5500)), 2)
        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": (fan_out_time + timedelta(minutes=i * 15)).strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": mule_collector,
            "receiver_account": receiver_dest,
            "amount": amt,
            "transaction_type": "WIRE",
            "location": "Zurich",
            "is_laundering": 1,
        })

    # 6. High RSF Outlier: Account with historic average of $80 suddenly moves $92,000
    rsf_sender = "ACC_DORMANT_SPIKE"
    rsf_receiver = "ACC_UNKNOWN_ENTITY"
    # historic small ones
    for i in range(5):
        tx_counter += 1
        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": (base_time + timedelta(days=i, hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": rsf_sender,
            "receiver_account": normal_receivers[0],
            "amount": round(float(np.random.uniform(40, 95)), 2),
            "transaction_type": "PAYMENT",
            "location": "Boston",
            "is_laundering": 0,
        })
    # Sudden giant spike
    tx_counter += 1
    records.append({
        "transaction_id": f"TX_{tx_counter}",
        "timestamp": (base_time + timedelta(days=5, hours=15)).strftime("%Y-%m-%d %H:%M:%S"),
        "sender_account": rsf_sender,
        "receiver_account": rsf_receiver,
        "amount": 92450.00,
        "transaction_type": "WIRE",
        "location": "Cayman Islands",
        "is_laundering": 1,
    })

    df = pd.DataFrame(records)
    # Sort chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    if save_to_disk:
        SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SAMPLE_DATA_PATH, index=False)

    return df
