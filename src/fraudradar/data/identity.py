"""Identity reconstruction: recover card / account / device entities from IEEE-CIS.

The dataset has no customer or card ID. Fraud is labelled at the *account* level
(once an account charges back, Vesta marks its later transactions as fraud too),
so recovering who-is-who is the single most useful piece of feature engineering.

Keys built here, from coarse to fine:

    card_uid     card1..card6 (issuer / BIN-like card attributes)
    account_uid  card_uid + addr1 + D1n, where D1n = day - D1. D1 is "days since
                 the card was first seen", so day - D1 is the card's (constant)
                 start day -- a near-unique fingerprint for one cardholder.
                 This is the trick popularised by the Kaggle 1st-place solution.
    device_uid   DeviceInfo + OS (id_30) + browser (id_31) + screen (id_33)

On top of those, `add_graph_features` links accounts that share a device into
clusters (an incremental, point-in-time union-find -- the same entity-resolution
idea as linking business hierarchies, applied to fraud rings).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CARD_COLS = [f"card{i}" for i in range(1, 7)]
DEVICE_COLS = ["DeviceInfo", "id_30", "id_31", "id_33"]


def _concat(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Join columns into one string key; missing components become 'nan'."""
    present = [c for c in cols if c in df.columns]
    out = df[present[0]].astype("string").fillna("nan")
    for c in present[1:]:
        out = out + "_" + df[c].astype("string").fillna("nan")
    return out


def add_entity_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add card_uid, account_uid and device_uid columns (string dtype, <NA> if unknown)."""
    df = df.copy()
    df["card_uid"] = _concat(df, CARD_COLS)

    # D1n: the card's start day. Constant across all transactions of one account.
    df["D1n"] = (df["day"] - df["D1"]).astype("Float64") if "D1" in df else pd.NA
    acct = _concat(df, ["card_uid", "addr1"]) + "_" + df["D1n"].astype("string")
    df["account_uid"] = acct.where(df["D1n"].notna(), pd.NA)

    if "DeviceInfo" in df.columns:
        dev = _concat(df, DEVICE_COLS)
        df["device_uid"] = dev.where(df["DeviceInfo"].notna(), pd.NA)
    else:
        df["device_uid"] = pd.Series(pd.NA, index=df.index, dtype="string")
    return df


class _UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.n_accounts: list[int] = []
        self.n_devices: list[int] = []

    def add(self, is_account: bool) -> int:
        self.parent.append(len(self.parent))
        self.n_accounts.append(int(is_account))
        self.n_devices.append(int(not is_account))
        return len(self.parent) - 1

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.n_accounts[ra] + self.n_devices[ra] < self.n_accounts[rb] + self.n_devices[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.n_accounts[ra] += self.n_accounts[rb]
        self.n_devices[ra] += self.n_devices[rb]
        return ra


def add_graph_features(df: pd.DataFrame, max_hub_degree: int = 20) -> pd.DataFrame:
    """Point-in-time account<->device graph features.

    Rows are processed in time order. Each row first links its account to its
    device, then records the state of the graph *including* this transaction
    (the scoring system sees the current transaction, so that is not leakage).

    Features:
        device_n_accounts   distinct accounts seen on this device so far
        cluster_n_accounts  accounts in this account's connected component
        cluster_n_devices   devices in that component

    A device already used by `max_hub_degree` accounts stops linking new ones,
    so generic fingerprints don't collapse everything into one giant component.
    """
    assert df["ts"].is_monotonic_increasing, "sort by time first"
    uf = _UnionFind()
    node_of: dict[tuple[str, str], int] = {}
    device_accounts: dict[str, set[str]] = {}

    n = len(df)
    dev_deg = np.full(n, np.nan, dtype=np.float32)
    cl_acc = np.full(n, np.nan, dtype=np.float32)
    cl_dev = np.full(n, np.nan, dtype=np.float32)

    accounts = df["account_uid"].to_numpy(dtype=object, na_value=None)
    devices = df["device_uid"].to_numpy(dtype=object, na_value=None)

    for i in range(n):
        acct, dev = accounts[i], devices[i]
        if acct is None:
            continue
        a = node_of.get(("a", acct))
        if a is None:
            a = node_of[("a", acct)] = uf.add(is_account=True)

        if dev is not None:
            seen = device_accounts.setdefault(dev, set())
            if acct in seen or len(seen) < max_hub_degree:
                d = node_of.get(("d", dev))
                if d is None:
                    d = node_of[("d", dev)] = uf.add(is_account=False)
                uf.union(a, d)
            seen.add(acct)
            dev_deg[i] = len(seen)

        root = uf.find(a)
        cl_acc[i] = uf.n_accounts[root]
        cl_dev[i] = uf.n_devices[root]

    df = df.copy()
    df["device_n_accounts"] = dev_deg
    df["cluster_n_accounts"] = cl_acc
    df["cluster_n_devices"] = cl_dev
    return df
