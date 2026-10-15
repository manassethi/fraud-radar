"""Fraud metrics. AUC alone hides what matters: how much fraud (in dollars) is
caught at a false-positive rate the business can tolerate."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    """Score threshold that flags `fpr` of legitimate transactions."""
    legit = score[y == 0]
    return float(np.quantile(legit, 1 - fpr))


def evaluate(y: np.ndarray, score: np.ndarray, amount: np.ndarray, fprs=(0.005, 0.01, 0.05)) -> dict:
    y = np.asarray(y)
    score = np.asarray(score)
    amount = np.asarray(amount, dtype=np.float64)
    out = {
        "n": int(len(y)),
        "fraud_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }
    fraud_dollars = amount[y == 1].sum()
    for f in fprs:
        thr = threshold_at_fpr(y, score, f)
        flagged = score > thr
        tag = f"{f * 100:g}pct_fpr"
        out[f"recall@{tag}"] = float(flagged[y == 1].mean())
        out[f"dollar_recall@{tag}"] = float(amount[flagged & (y == 1)].sum() / fraud_dollars)
        out[f"precision@{tag}"] = float(y[flagged].mean()) if flagged.any() else float("nan")
    return out
