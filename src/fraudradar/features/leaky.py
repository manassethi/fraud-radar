"""Deliberately *wrong* features, kept as the control arm of the leakage experiment.

These are the features you get from the obvious `groupby().transform()` approach:

  * counts / mean amounts per entity over the *whole* dataset (they see the future),
  * entity fraud rate computed over all training rows, including the row's own
    label and labels that wouldn't have arrived yet (target leakage).

They look great on a random validation split and degrade out-of-time. Comparing
them to `pit.py` quantifies how much offline metrics would mislead you.
Never use these for a model you intend to serve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudradar.config import FeatureConfig
from fraudradar.data.identity import add_entity_keys
from fraudradar.features.pit import transaction_features


def build_leaky_features(
    df: pd.DataFrame, train_mask: np.ndarray, cfg: FeatureConfig = FeatureConfig()
) -> tuple[pd.DataFrame, list[str]]:
    df = add_entity_keys(df)
    parts = [transaction_features(df)]
    train_y = df["isFraud"].where(train_mask)  # NaN outside train, so test labels aren't used directly
    for key in cfg.entities:
        grp = df.groupby(key, dropna=True)
        parts.append(
            pd.DataFrame(
                {
                    f"{key}__cnt_all": grp["ts"].transform("count"),
                    f"{key}__amt_mean_all": grp["TransactionAmt"].transform("mean"),
                    f"{key}__fraud_rate_train": train_y.groupby(df[key]).transform("mean"),
                },
                index=df.index,
            ).astype(np.float32)
        )
    new = pd.concat(parts, axis=1)
    return pd.concat([df, new], axis=1), list(new.columns)
