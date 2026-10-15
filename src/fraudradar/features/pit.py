"""Point-in-time (PIT) entity features.

Every feature for a transaction at time t uses only:
  * transactions of the same entity strictly before t, and
  * labels of those transactions that would already be known at t
    (i.e. the transaction happened at least `label_delay` seconds earlier).

That is exactly what an online scoring service can compute from its feature store,
so offline training and online serving see the same distribution. The property is
enforced by `tests/test_pit_features.py`, which mutates the future and checks that
past features don't move.

Implementation: fully vectorised. Rows of one entity are made contiguous by sorting
on (entity, ts), and windows become `searchsorted` lookups on the composite key
entity * BIG + ts, with prefix sums for amounts and labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudradar.config import HOUR, FeatureConfig
from fraudradar.data.identity import add_entity_keys, add_graph_features

_BIG = np.int64(10**10)  # > any TransactionDT (~1.6e7 s) plus the longest window


def entity_history_features(
    df: pd.DataFrame,
    key: str,
    windows: dict[str, int],
    label_delay: int,
    amount_col: str = "TransactionAmt",
    label_col: str = "isFraud",
) -> pd.DataFrame:
    """History features for one entity column. Returns a frame aligned to df.index."""
    n = len(df)
    valid = df[key].notna().to_numpy()
    rows = np.flatnonzero(valid)

    codes, _ = pd.factorize(df[key].to_numpy()[valid])
    ts = df["ts"].to_numpy(dtype=np.int64)[valid]
    amt = df[amount_col].to_numpy(dtype=np.float64)[valid]
    y = df[label_col].to_numpy(dtype=np.float64)[valid]

    order = np.lexsort((ts, codes))  # stable: ties keep time order
    rows, g, t, amt, y = rows[order], codes[order].astype(np.int64), ts[order], amt[order], y[order]
    comp = g * _BIG + t

    cs_amt = np.concatenate([[0.0], np.cumsum(amt)])
    cs_y = np.concatenate([[0.0], np.cumsum(y)])

    start = np.searchsorted(comp, g * _BIG, side="left")  # first row of this entity
    cur = np.searchsorted(comp, comp, side="left")  # first row at this exact second
    n_prior = cur - start

    feats: dict[str, np.ndarray] = {}
    p = f"{key}__"
    feats[p + "n_prior"] = n_prior.astype(np.float32)

    for name, w in windows.items():
        lo = np.maximum(np.searchsorted(comp, comp - w, side="left"), start)
        feats[p + f"cnt_{name}"] = (cur - lo).astype(np.float32)
        feats[p + f"amt_sum_{name}"] = (cs_amt[cur] - cs_amt[lo]).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        has_prior = n_prior > 0
        mean_prior = np.where(has_prior, (cs_amt[cur] - cs_amt[start]) / n_prior, np.nan)
        feats[p + "amt_mean_prior"] = mean_prior.astype(np.float32)
        feats[p + "amt_ratio"] = (amt / mean_prior).astype(np.float32)

        prev_t = t[np.maximum(cur - 1, 0)]
        first_t = t[start]
        feats[p + "secs_since_prev"] = np.where(has_prior, t - prev_t, np.nan).astype(np.float32)
        feats[p + "secs_since_first"] = np.where(has_prior, t - first_t, np.nan).astype(np.float32)

        # Labels known at time t: transactions at or before t - label_delay.
        hi = np.clip(np.searchsorted(comp, comp - label_delay, side="right"), start, cur)
        n_known = hi - start
        n_known_fraud = cs_y[hi] - cs_y[start]
        feats[p + "n_known_fraud"] = n_known_fraud.astype(np.float32)
        feats[p + "fraud_rate_known"] = np.where(
            n_known > 0, n_known_fraud / n_known, np.nan
        ).astype(np.float32)

    def scatter(vals: np.ndarray) -> np.ndarray:
        arr = np.full(n, np.nan, dtype=np.float32)
        arr[rows] = vals
        return arr

    return pd.DataFrame({col: scatter(v) for col, v in feats.items()}, index=df.index)


def transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless features of the transaction itself."""
    amt = df["TransactionAmt"].astype(np.float64)
    return pd.DataFrame(
        {
            "hour": ((df["ts"] // HOUR) % 24).astype(np.float32),
            "dow": ((df["ts"] // (24 * HOUR)) % 7).astype(np.float32),
            "amt_log": np.log1p(amt).astype(np.float32),
            # Fraction of a dollar: foreign-currency conversions leave odd cents.
            "amt_cents": (np.round(amt * 100) % 100).astype(np.float32),
        },
        index=df.index,
    )


def build_pit_features(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> tuple[pd.DataFrame, list[str]]:
    """Add identity keys, graph and history features. Returns (frame, new feature names)."""
    df = add_entity_keys(df)
    df = add_graph_features(df, max_hub_degree=cfg.max_hub_degree)
    parts = [transaction_features(df)]
    for key in cfg.entities:
        parts.append(entity_history_features(df, key, cfg.windows, cfg.label_delay))
    new = pd.concat(parts, axis=1)
    graph_cols = ["device_n_accounts", "cluster_n_accounts", "cluster_n_devices"]
    feature_names = list(new.columns) + graph_cols
    return pd.concat([df, new], axis=1), feature_names
