"""Turn ranking scores into probabilities.

The decision policy multiplies P(fraud) by dollar amounts, so it needs probabilities that
mean what they say, not just a good ranking. Calibrators are fit on the validation fold
(which the booster never trained on) and judged on the out-of-time test fold.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_calibrator(score: np.ndarray, y: np.ndarray, method: str = "isotonic") -> Callable[[np.ndarray], np.ndarray]:
    """Return a monotone map score -> P(fraud) fitted on (score, y)."""
    score = np.asarray(score, dtype=np.float64)
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(score, y)
        return lambda s: iso.predict(np.asarray(s, dtype=np.float64))
    if method == "platt":
        # Large C = effectively unregularised; Platt scaling has only 2 parameters.
        lr = LogisticRegression(C=1e6).fit(_logit(score)[:, None], y)
        return lambda s: lr.predict_proba(_logit(np.asarray(s, dtype=np.float64))[:, None])[:, 1]
    raise ValueError(f"unknown calibration method {method!r}")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - y) ** 2))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted probability: mean prediction vs observed fraud rate.

    Equal-count (not equal-width) bins because almost all transactions have p < 0.05;
    equal-width bins would put ~95% of rows in the first bin.
    """
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    bins = np.empty(len(p), dtype=np.int64)
    bins[order] = np.arange(len(p)) * n_bins // len(p)
    t = (
        pd.DataFrame({"bin": bins, "p": p, "y": y})
        .groupby("bin")
        .agg(n=("y", "size"), p_min=("p", "min"), p_max=("p", "max"), mean_pred=("p", "mean"), fraud_rate=("y", "mean"))
    )
    t["gap"] = t["mean_pred"] - t["fraud_rate"]
    return t


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    t = reliability_table(y, p, n_bins)
    return float((t["n"] * t["gap"].abs()).sum() / t["n"].sum())
