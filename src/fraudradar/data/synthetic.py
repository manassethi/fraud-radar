"""Synthetic data in the IEEE-CIS file format.

Lets the whole pipeline (and CI) run without the Kaggle download. It mimics the
structure that matters for this project, not the real distributions:

  * accounts have a fixed card-start day, so D1n identity reconstruction works;
  * account takeover: after a compromise time, a burst of larger transactions on
    a fraudster's device, and every later transaction of that account is labelled
    fraud (Vesta's account-level labelling);
  * fraud rings: one fraudster device hits several victim accounts, so graph
    features carry signal;
  * a few very common "generic" device fingerprints, to exercise the hub cap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudradar.config import DAY, RAW_DIR

EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com", "outlook.com", "icloud.com"]
OSES = ["Windows 10", "Windows 7", "iOS 11.2.1", "Android 7.0", "Mac OS X 10_12_6"]
BROWSERS = ["chrome 63.0", "mobile safari 11.0", "ie 11.0 for desktop", "firefox 57.0", "edge 16.0"]
SCREENS = ["1920x1080", "1366x768", "2208x1242", "1334x750", "2560x1440"]


def _device(rng: np.random.Generator, generic: bool) -> tuple[str, str, str, str]:
    if generic:  # a fingerprint thousands of people share
        return ("Windows", "Windows 10", "chrome 63.0", "1920x1080")
    return (
        f"SM-G{rng.integers(100, 999)}",
        str(rng.choice(OSES)),
        str(rng.choice(BROWSERS)),
        str(rng.choice(SCREENS)),
    )


def generate(
    n_accounts: int = 4000,
    n_days: int = 180,
    fraud_account_rate: float = 0.04,
    ring_size: int = 4,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, identity) frames with IEEE-CIS column names."""
    rng = np.random.default_rng(seed)
    card1_pool = rng.integers(1000, 18000, size=max(50, n_accounts // 5))

    n_fraud_accounts = int(n_accounts * fraud_account_rate)
    n_rings = max(1, n_fraud_accounts // ring_size)
    ring_devices = [_device(rng, generic=False) for _ in range(n_rings)]

    txns: list[dict] = []
    ident: list[dict] = []
    tid = 2_987_000  # real IEEE-CIS IDs start here

    for a in range(n_accounts):
        start_day = int(rng.integers(-300, n_days - 20))
        card = {
            "card1": int(rng.choice(card1_pool)),
            "card2": float(rng.integers(100, 600)),
            "card3": 150.0,
            "card4": str(rng.choice(["visa", "mastercard", "american express", "discover"], p=[0.65, 0.3, 0.03, 0.02])),
            "card5": float(rng.integers(100, 230)),
            "card6": str(rng.choice(["debit", "credit"], p=[0.75, 0.25])),
            "addr1": float(rng.integers(100, 540)),
            "addr2": 87.0,
            "P_emaildomain": str(rng.choice(EMAILS)),
        }
        own_device = _device(rng, generic=rng.random() < 0.3)
        typical_amt = float(rng.lognormal(3.8, 0.7))
        rate = float(rng.uniform(0.03, 0.4))  # transactions per day

        first, last = max(0, start_day), n_days
        n_legit = rng.poisson(rate * (last - first))
        times = np.sort(rng.uniform(first, last, size=n_legit)) * DAY

        compromised_at = None
        if a < n_fraud_accounts and last - first > 5:
            compromised_at = rng.uniform(first + 2, last - 1) * DAY
            ring_dev = ring_devices[a % n_rings]
            # Burst: several fast, larger transactions within ~2 days of takeover.
            burst = compromised_at + np.sort(rng.exponential(6 * 3600, size=rng.integers(3, 12)))
            times = np.sort(np.concatenate([times, burst[burst < n_days * DAY]]))
            burst_set = set(burst.tolist())

        for t in times:
            t = int(t) + DAY  # IEEE-CIS TransactionDT starts at 86400
            is_fraud = compromised_at is not None and t - DAY >= compromised_at
            in_burst = is_fraud and (t - DAY) in burst_set
            amt = typical_amt * float(rng.lognormal(0, 0.5)) * (3.0 if in_burst else 1.0)
            day = t // DAY
            tid += 1
            txns.append(
                {
                    "TransactionID": tid,
                    "isFraud": int(is_fraud),
                    "TransactionDT": t,
                    "TransactionAmt": round(amt, 3 if in_burst and rng.random() < 0.5 else 2),
                    "ProductCD": str(rng.choice(["W", "C", "H", "R", "S"], p=[0.7, 0.12, 0.08, 0.06, 0.04])),
                    **card,
                    "R_emaildomain": card["P_emaildomain"] if rng.random() < 0.3 else None,
                    "C1": float(rng.poisson(3 if in_burst else 1) + 1),
                    "D1": float(day - start_day),
                }
            )
            has_identity = rng.random() < (0.85 if in_burst else 0.3)
            if has_identity:
                dev = ring_dev if in_burst else own_device
                ident.append(
                    {"TransactionID": tid, "DeviceType": "mobile" if "SM-" in dev[0] else "desktop",
                     "DeviceInfo": dev[0], "id_30": dev[1], "id_31": dev[2], "id_33": dev[3]}
                )

    txn_df = pd.DataFrame(txns).sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    return txn_df, pd.DataFrame(ident)


def write(raw_dir: Path = RAW_DIR, **kwargs) -> Path:
    txn, ident = generate(**kwargs)
    raw_dir.mkdir(parents=True, exist_ok=True)
    txn.to_csv(raw_dir / "train_transaction.csv", index=False)
    ident.to_csv(raw_dir / "train_identity.csv", index=False)
    return raw_dir
