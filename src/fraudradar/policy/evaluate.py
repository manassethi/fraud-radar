"""Compare decision policies by realised dollar cost on the out-of-time test fold.

Realised cost uses the true label; the 3DS outcome is replaced by its expectation
(catch rate, drop-off rate) because we cannot observe what a challenge would have done.
Every tunable choice (calibrator, thresholds) is fit on the validation fold only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from fraudradar.config import CostConfig
from fraudradar.policy.decision import (
    ALLOW,
    BLOCK,
    CHALLENGE,
    decide,
    outcome_costs,
    threshold_actions,
    tune_thresholds,
)


@dataclass
class Fold:
    p: np.ndarray  # calibrated P(fraud)
    y: np.ndarray
    amount: np.ndarray


def summarize(actions: np.ndarray, y: np.ndarray, amount: np.ndarray, cfg: CostConfig) -> dict:
    """Total realised cost of `actions`, split by source, plus the action mix."""
    y = np.asarray(y)
    if_fraud, if_legit = outcome_costs(amount, cfg)
    rows = np.arange(len(y))
    fraud_cost = if_fraud[rows, actions] * (y == 1)
    legit_cost = if_legit[rows, actions] * (y == 0)
    a = np.asarray(amount, dtype=np.float64)
    stopped = np.where(actions == BLOCK, 1.0, np.where(actions == CHALLENGE, cfg.challenge_catch_rate, 0.0))
    total = fraud_cost.sum() + legit_cost.sum()
    return {
        "total_cost": total,
        "cost_per_txn": total / len(y),
        "fraud_loss": fraud_cost.sum(),
        "false_decline_cost": legit_cost[actions == BLOCK].sum(),
        "challenge_friction_cost": legit_cost[actions == CHALLENGE].sum(),
        "allow_rate": float(np.mean(actions == ALLOW)),
        "challenge_rate": float(np.mean(actions == CHALLENGE)),
        "block_rate": float(np.mean(actions == BLOCK)),
        "fraud_$_stopped": float((stopped * a)[y == 1].sum() / a[y == 1].sum()),
        "legit_blocked_rate": float(np.mean(actions[y == 0] == BLOCK)),
    }


def policy_actions(val: Fold, test: Fold, cfg: CostConfig, raw_test_p: np.ndarray | None = None) -> dict:
    """Test-fold actions of each policy, keyed by name. Thresholds are tuned on `val`."""
    t_single, _ = tune_thresholds(val.p, val.y, val.amount, cfg, challenge=False)
    t_block, t_chal = tune_thresholds(val.p, val.y, val.amount, cfg, challenge=True)
    out = {
        "allow_all": np.full(len(test.y), ALLOW),
        "block_at_0.5": threshold_actions(test.p, 0.5),
        "tuned_threshold": threshold_actions(test.p, t_single),
        "tuned_2_thresholds": threshold_actions(test.p, t_block, t_chal),
        "expected_cost": decide(test.p, test.amount, cfg),
    }
    if raw_test_p is not None:
        # Ablation: same rule, uncalibrated scores. Shows what calibration buys.
        out["expected_cost_uncalibrated"] = decide(raw_test_p, test.amount, cfg)
    return out


def compare_policies(val: Fold, test: Fold, cfg: CostConfig, raw_test_p: np.ndarray | None = None) -> pd.DataFrame:
    acts = policy_actions(val, test, cfg, raw_test_p)
    t = pd.DataFrame({name: summarize(a, test.y, test.amount, cfg) for name, a in acts.items()}).T
    t["savings_vs_allow_all"] = 1 - t["total_cost"] / t.loc["allow_all", "total_cost"]
    t.index.name = "policy"
    return t


def sensitivity(
    val: Fold,
    test: Fold,
    base: CostConfig,
    dispute_fees=(0.0, 15.0, 30.0, 50.0),
    margins=(0.05, 0.10, 0.20, 0.40),
) -> pd.DataFrame:
    """Re-tune and re-evaluate under each (dispute_fee, margin). No retraining needed:
    the model's probabilities do not depend on costs, only the decisions do."""
    rows = []
    for fee in dispute_fees:
        for m in margins:
            cfg = replace(base, dispute_fee=fee, margin=m)
            t = compare_policies(val, test, cfg)
            ec = t.loc["expected_cost"]
            rows.append({
                "dispute_fee": fee,
                "margin": m,
                "allow_all": t.loc["allow_all", "total_cost"],
                "tuned_threshold": t.loc["tuned_threshold", "total_cost"],
                "tuned_2_thresholds": t.loc["tuned_2_thresholds", "total_cost"],
                "expected_cost": ec["total_cost"],
                "ec_vs_tuned_threshold": 1 - ec["total_cost"] / t.loc["tuned_threshold", "total_cost"],
                "ec_vs_tuned_2_thresholds": 1 - ec["total_cost"] / t.loc["tuned_2_thresholds", "total_cost"],
                "ec_challenge_rate": ec["challenge_rate"],
                "ec_block_rate": ec["block_rate"],
            })
    return pd.DataFrame(rows)

