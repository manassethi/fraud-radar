"""LightGBM baseline with an out-of-time split.

Feature sets:
    raw    the dataset's own columns only
    pit    raw + point-in-time identity, velocity, history and graph features
    leaky  raw + naive full-data groupby features (control arm, see features/leaky.py)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraudradar.config import ARTIFACT_DIR, FeatureConfig, SplitConfig, TrainConfig
from fraudradar.features.leaky import build_leaky_features
from fraudradar.features.pit import build_pit_features
from fraudradar.models.metrics import evaluate

# Never fed to the model: IDs, raw time, labels, and string entity keys
# (high-cardinality IDs only let the model memorise individual accounts).
EXCLUDE = {
    "TransactionID", "TransactionDT", "ts", "day", "isFraud",
    "card_uid", "account_uid", "device_uid", "D1n",
}


@dataclass
class Split:
    fit: np.ndarray  # boolean masks over df rows
    val: np.ndarray
    test: np.ndarray


def time_split(df: pd.DataFrame, split: SplitConfig, label_delay: int, val_frac: float = 0.15) -> Split:
    """fit | val | gap | test along the time axis. val = the last `val_frac` of train."""
    ts = df["ts"].to_numpy()
    t0, span = ts.min(), ts.max() - ts.min()
    train_end = t0 + split.train_end * span
    if split.drop_immature_labels:
        # At the train cutoff, chargebacks for the most recent transactions haven't arrived.
        train_end -= label_delay
    val_start = t0 + (train_end - t0) * (1 - val_frac)
    test_start = t0 + split.test_start * span
    return Split(fit=ts < val_start, val=(ts >= val_start) & (ts <= train_end), test=ts >= test_start)


def _to_model_matrix(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    raw = [c for c in df.columns if c not in EXCLUDE and c not in extra]
    X = df[raw + extra].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    return X


def build_features(df: pd.DataFrame, feature_set: str, split: Split, fcfg: FeatureConfig) -> pd.DataFrame:
    if feature_set == "raw":
        return _to_model_matrix(df, [])
    if feature_set == "pit":
        full, names = build_pit_features(df, fcfg)
        return _to_model_matrix(full, names)
    if feature_set == "leaky":
        full, names = build_leaky_features(df, split.fit | split.val, fcfg)
        return _to_model_matrix(full, names)
    raise ValueError(f"unknown feature set {feature_set!r}")


def train(
    df: pd.DataFrame,
    feature_set: str = "pit",
    fcfg: FeatureConfig = FeatureConfig(),
    scfg: SplitConfig = SplitConfig(),
    tcfg: TrainConfig = TrainConfig(),
    out_dir: Path | None = None,
) -> dict:
    split = time_split(df, scfg, fcfg.label_delay)
    X = build_features(df, feature_set, split, fcfg)
    y = df["isFraud"].to_numpy()
    amt = df["TransactionAmt"].to_numpy()

    dfit = lgb.Dataset(X[split.fit], y[split.fit])
    dval = lgb.Dataset(X[split.val], y[split.val], reference=dfit)
    params = {**tcfg.params, "seed": tcfg.seed, "metric": "auc"}
    booster = lgb.train(
        params,
        dfit,
        num_boost_round=tcfg.num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(tcfg.early_stopping_rounds, verbose=False)],
    )

    result = {
        "feature_set": feature_set,
        "n_features": X.shape[1],
        "best_iteration": booster.best_iteration,
        "val": evaluate(y[split.val], booster.predict(X[split.val]), amt[split.val]),
        "test": evaluate(y[split.test], booster.predict(X[split.test]), amt[split.test]),
        "config": {"features": asdict(fcfg), "split": asdict(scfg)},
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(out_dir / "model.txt"))
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        imp = pd.DataFrame(
            {"feature": booster.feature_name(), "gain": booster.feature_importance("gain")}
        ).sort_values("gain", ascending=False)
        imp.to_csv(out_dir / "feature_importance.csv", index=False)
        result["top_features"] = imp.head(15)["feature"].tolist()
    return result


def default_out_dir(feature_set: str) -> Path:
    return ARTIFACT_DIR / f"lgbm_{feature_set}"
