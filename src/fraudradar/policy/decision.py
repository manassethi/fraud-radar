"""Bayes decision rule: pick the action with the lowest expected dollar cost.

For a transaction with amount A and calibrated fraud probability p, each action has a cost
under each outcome (CostConfig defines the parameters):

                 fraud (p)                    legit (1 - p)
    allow        A + dispute_fee              0
    challenge    (1 - catch) (A + fee)        dropoff * margin * A
    block        0                            margin * A + churn_rate * LTV

E[cost | action] = p * cost_if_fraud + (1 - p) * cost_if_legit, and the policy takes the
argmin. Because the costs scale with A, this is an amount-dependent threshold policy: e.g.
block beats allow when p / (1 - p) > (margin * A + churn * LTV) / (A + fee).
"""

from __future__ import annotations

import numpy as np

from fraudradar.config import CostConfig

# Ordered from least to most customer friction; np.argmin breaks ties toward the first,
# so an exact tie never adds friction.
ACTIONS = ("allow", "challenge", "block")
ALLOW, CHALLENGE, BLOCK = range(3)


def outcome_costs(amount: np.ndarray, cfg: CostConfig) -> tuple[np.ndarray, np.ndarray]:
    """(cost_if_fraud, cost_if_legit), each of shape (n, 3) with columns in ACTIONS order."""
    a = np.asarray(amount, dtype=np.float64)
    zero = np.zeros_like(a)
    chargeback = a + cfg.dispute_fee
    if_fraud = np.column_stack([chargeback, (1 - cfg.challenge_catch_rate) * chargeback, zero])
    if_legit = np.column_stack(
        [zero, cfg.challenge_dropoff * cfg.margin * a, cfg.margin * a + cfg.false_decline_fixed]
    )
    return if_fraud, if_legit


def expected_costs(p: np.ndarray, amount: np.ndarray, cfg: CostConfig) -> np.ndarray:
    """E[cost] of each action, shape (n, 3). Pass true labels as `p` to get realised costs."""
    p = np.asarray(p, dtype=np.float64)[:, None]
    if_fraud, if_legit = outcome_costs(amount, cfg)
    return p * if_fraud + (1 - p) * if_legit


def decide(p: np.ndarray, amount: np.ndarray, cfg: CostConfig) -> np.ndarray:
    """Index into ACTIONS of the minimum-expected-cost action for each transaction."""
    return np.argmin(expected_costs(p, amount, cfg), axis=1)


def threshold_actions(p: np.ndarray, t_block: float, t_challenge: float = np.inf) -> np.ndarray:
    """Score-only policy: block if p >= t_block, else challenge if p >= t_challenge, else allow."""
    p = np.asarray(p)
    return np.where(p >= t_block, BLOCK, np.where(p >= t_challenge, CHALLENGE, ALLOW))


def tune_thresholds(
    p: np.ndarray, y: np.ndarray, amount: np.ndarray, cfg: CostConfig, challenge: bool = False
) -> tuple[float, float]:
    """Cost-minimising (t_block, t_challenge) for `threshold_actions` on labelled data.

    Exact search over every cut point in O(n log n). Sort by p descending; a policy is two
    cut points kb <= kc: the top kb rows are blocked, rows kb..kc challenged, the rest allowed.
    With prefix sums B, C, A of each action's realised cost,
        cost(kb, kc) = B[kb] + (C[kc] - C[kb]) + (A[n] - A[kc]),
    so for each kc the best kb is a running minimum of B - C. Cuts are only allowed between
    distinct p values, since a threshold cannot split tied scores.
    With challenge=False the challenge band is empty (kb = kc), i.e. a single allow/block cut.
    """
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    cost = expected_costs(np.asarray(y)[order], np.asarray(amount)[order], cfg)
    pre = np.vstack([np.zeros(3), np.cumsum(cost, axis=0)])  # pre[k] = cost of rows < k
    A, C, B = pre[:, ALLOW], pre[:, CHALLENGE], pre[:, BLOCK]

    n = len(ps)
    valid = np.ones(n + 1, dtype=bool)
    valid[1:n] = ps[:-1] != ps[1:]

    if challenge:
        head = np.where(valid, B - C, np.inf)
        best_kb = _running_argmin(head)
        total = head[best_kb] + C + (A[n] - A)
    else:
        best_kb = np.arange(n + 1)
        total = B + (A[n] - A)
    total = np.where(valid, total, np.inf)
    kc = int(np.argmin(total))
    kb = int(best_kb[kc])

    def cut(k: int) -> float:  # threshold that selects exactly the top-k rows
        return float(ps[k - 1]) if k > 0 else np.inf

    return cut(kb), (cut(kc) if challenge else np.inf)


def _running_argmin(x: np.ndarray) -> np.ndarray:
    """out[i] = argmin(x[: i + 1])."""
    running = np.minimum.accumulate(x)
    idx = np.where(x == running, np.arange(len(x)), 0)
    return np.maximum.accumulate(idx)
