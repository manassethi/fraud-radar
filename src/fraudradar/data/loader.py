"""Load the IEEE-CIS Fraud Detection data (Kaggle / Vesta) into one frame.

Expected files in `data/raw/` (from https://www.kaggle.com/c/ieee-fraud-detection/data):
    train_transaction.csv   one row per transaction, label `isFraud`
    train_identity.csv      optional device / network info, joined on TransactionID

Only the Kaggle *train* files carry labels, so the whole project works on them and
splits by time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudradar.config import DAY, RAW_DIR

# Columns we always want, if present. V1..V339 are Vesta's engineered features;
# they are large and partially redundant, so they are opt-in.
BASE_TRANSACTION_COLS = (
    ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
    + [f"card{i}" for i in range(1, 7)]
    + ["addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"M{i}" for i in range(1, 10)]
)
V_COLS = [f"V{i}" for i in range(1, 340)]

CATEGORICAL_COLS = (
    ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
    + [f"M{i}" for i in range(1, 10)]
    + [f"id_{i}" for i in range(12, 39)]
)


def load_raw(raw_dir: Path = RAW_DIR, include_v: bool = False, nrows: int | None = None) -> pd.DataFrame:
    """Read and merge transaction + identity tables, sorted by time."""
    txn_path = raw_dir / "train_transaction.csv"
    if not txn_path.exists():
        raise FileNotFoundError(
            f"{txn_path} not found. Download the IEEE-CIS data from Kaggle into "
            f"{raw_dir}, or run `python -m fraudradar.cli synth` for synthetic data."
        )

    wanted = set(BASE_TRANSACTION_COLS) | (set(V_COLS) if include_v else set())
    txn = pd.read_csv(txn_path, usecols=lambda c: c in wanted, nrows=nrows)

    id_path = raw_dir / "train_identity.csv"
    if id_path.exists():
        ident = pd.read_csv(id_path)
        # Kaggle's test identity file uses `id-01`; normalise so both work.
        ident.columns = [c.replace("-", "_") for c in ident.columns]
        txn = txn.merge(ident, on="TransactionID", how="left")

    return _clean(txn)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)

    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Seconds since Vesta's (undisclosed) reference point; `day` is the integer day.
    df["ts"] = df["TransactionDT"].astype(np.int64)
    df["day"] = df["ts"] // DAY
    df["isFraud"] = df["isFraud"].astype(np.int8)
    return df
