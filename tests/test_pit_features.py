import numpy as np
import pandas as pd
import pytest

from fraudradar.config import DAY, HOUR, FeatureConfig
from fraudradar.data import synthetic
from fraudradar.data.loader import _clean
from fraudradar.features.pit import build_pit_features, entity_history_features

WINDOWS = {"1h": HOUR, "24h": DAY}


def _frame(rows):
    df = pd.DataFrame(rows, columns=["ts", "key", "TransactionAmt", "isFraud"])
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def test_windows_exclude_current_and_same_second():
    df = _frame([(0, "a", 10.0, 0), (60, "a", 20.0, 0), (60, "a", 5.0, 0), (2 * HOUR, "a", 40.0, 0)])
    f = entity_history_features(df, "key", WINDOWS, label_delay=DAY)
    assert f["key__n_prior"].tolist() == [0, 1, 1, 3]  # same-second rows don't see each other
    assert f["key__cnt_1h"].tolist() == [0, 1, 1, 0]
    assert f["key__amt_sum_24h"].tolist() == [0, 10, 10, 35]
    assert f["key__secs_since_prev"].iloc[3] == 2 * HOUR - 60
    assert np.isnan(f["key__amt_mean_prior"].iloc[0])
    assert f["key__amt_ratio"].iloc[3] == pytest.approx(40 / (35 / 3))


def test_entities_are_isolated_and_missing_keys_are_nan():
    df = _frame([(0, "a", 10.0, 1), (10, "b", 99.0, 1), (20, "a", 1.0, 0), (30, None, 5.0, 0)])
    f = entity_history_features(df, "key", WINDOWS, label_delay=0)
    assert f["key__n_prior"].tolist()[:3] == [0, 0, 1]
    assert f["key__amt_sum_1h"].iloc[2] == 10.0  # doesn't include b's 99
    assert f.iloc[3].isna().all()


def test_labels_only_visible_after_delay():
    df = _frame([(0, "a", 1.0, 1), (DAY - 1, "a", 1.0, 0), (DAY, "a", 1.0, 0)])
    f = entity_history_features(df, "key", WINDOWS, label_delay=DAY)
    assert f["key__n_known_fraud"].tolist() == [0, 0, 1]
    assert np.isnan(f["key__fraud_rate_known"].iloc[1])
    assert f["key__fraud_rate_known"].iloc[2] == 1.0


def test_features_do_not_depend_on_the_future():
    """Rewrite everything after a cutoff; features before it must not change."""
    txn, ident = synthetic.generate(n_accounts=600, n_days=60, seed=1)
    base = _clean(txn.merge(ident, on="TransactionID", how="left"))
    cutoff = base["ts"].quantile(0.6)

    rng = np.random.default_rng(0)
    future = base["ts"] > cutoff
    mutated = base.copy()
    mutated.loc[future, "TransactionAmt"] = rng.uniform(1, 5000, future.sum()).astype(np.float32)
    mutated.loc[future, "isFraud"] = 1 - mutated.loc[future, "isFraud"]
    mutated.loc[future, "DeviceInfo"] = "Windows"  # also rewires the device graph
    extra = mutated[future].sample(500, random_state=0).assign(TransactionID=lambda d: d.TransactionID + 10**7)
    mutated = pd.concat([mutated, extra]).sort_values(["ts", "TransactionID"], kind="stable").reset_index(drop=True)

    cfg = FeatureConfig()
    f_base, names = build_pit_features(base, cfg)
    f_mut, _ = build_pit_features(mutated, cfg)

    key = ["TransactionID"]
    past_base = f_base.loc[f_base["ts"] <= cutoff, key + names].set_index("TransactionID").sort_index()
    past_mut = f_mut.loc[f_mut["ts"] <= cutoff, key + names].set_index("TransactionID").sort_index()
    pd.testing.assert_frame_equal(past_base, past_mut)
