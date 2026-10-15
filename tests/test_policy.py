"""Decision rule checks with costs worked out by hand.

Default CostConfig: dispute_fee F = 15, margin m = 0.2, churn term L = 0.05 * 100 = 5,
3DS drop-off d = 0.1, 3DS catch rate c = 0.8. For amount A and fraud probability p:
    allow      p (A + F)
    challenge  p (1 - c)(A + F) + (1 - p) d m A
    block      (1 - p)(m A + L)
"""

from dataclasses import replace

import numpy as np
import pytest

from fraudradar.config import CostConfig, TrainConfig
from fraudradar.data import synthetic
from fraudradar.data.loader import load_raw
from fraudradar.models.calibration import fit_calibrator, reliability_table
from fraudradar.policy import ALLOW, BLOCK, CHALLENGE, decide, expected_costs
from fraudradar.policy.decision import threshold_actions, tune_thresholds
from fraudradar.policy.evaluate import summarize

CFG = CostConfig()


@pytest.mark.parametrize(
    "p, amount, costs, action",
    [
        # allow 0.01*115 = 1.15; challenge 0.01*0.2*115 + 0.99*0.1*0.2*100 = 0.23 + 1.98; block 0.99*25
        (0.01, 100.0, (1.15, 2.21, 24.75), ALLOW),
        # allow 11.5; challenge 0.1*23 + 0.9*2 = 4.1; block 0.9*25 = 22.5
        (0.10, 100.0, (11.5, 4.1, 22.5), CHALLENGE),
        # allow 69; challenge 0.6*23 + 0.4*2 = 14.6; block 0.4*25 = 10
        (0.60, 100.0, (69.0, 14.6, 10.0), BLOCK),
        # Same p, $1 payment: blocking still risks the $5 churn term, so challenge wins.
        # allow 0.6*16 = 9.6; challenge 0.6*0.2*16 + 0.4*0.1*0.2*1 = 1.928; block 0.4*(0.2 + 5) = 2.08
        (0.60, 1.0, (9.6, 1.928, 2.08), CHALLENGE),
        # Tiny p, $2 payment: the $15 dispute fee dwarfs the amount, so even 0.5% risk is worth a challenge.
        # allow 0.005*17 = 0.085; challenge 0.005*0.2*17 + 0.995*0.1*0.2*2 = 0.0568; block 0.995*5.4
        (0.005, 2.0, (0.085, 0.0568, 5.373), CHALLENGE),
        # ... but not on a $1000 payment, where 10% drop-off on $200 of margin dominates.
        # allow 0.005*1015 = 5.075; challenge 0.005*203 + 0.995*20 = 20.915; block 0.995*205
        (0.005, 1000.0, (5.075, 20.915, 203.975), ALLOW),
    ],
)
def test_hand_computed_costs_and_decisions(p, amount, costs, action):
    got = expected_costs(np.array([p]), np.array([amount]), CFG)[0]
    np.testing.assert_allclose(got, costs, rtol=1e-12)
    assert decide(np.array([p]), np.array([amount]), CFG)[0] == action


def test_allow_block_breakeven_without_3ds():
    # With a useless challenge (catches nothing) the rule reduces to one amount-dependent
    # threshold: block iff p > (mA + L) / (mA + L + A + F). For A = 100: 25 / 140.
    cfg = replace(CFG, challenge_catch_rate=0.0)
    p_star = 25 / 140
    acts = decide(np.array([p_star - 1e-9, p_star + 1e-9]), np.array([100.0, 100.0]), cfg)
    assert acts.tolist() == [ALLOW, BLOCK]
    # Larger amount -> breakeven approaches m / (1 + m) from above: easier to justify a block.
    # A = 1000: 205 / 1220 = 0.168 < 0.1786.
    assert decide(np.array([0.17]), np.array([1000.0]), cfg)[0] == BLOCK


def test_ties_go_to_least_friction():
    # No drop-off and no catch: challenge costs exactly what allow costs.
    cfg = replace(CFG, challenge_dropoff=0.0, challenge_catch_rate=0.0)
    assert decide(np.array([0.05]), np.array([50.0]), cfg)[0] == ALLOW


def test_realised_costs_are_expected_costs_at_the_label():
    y = np.array([1, 0])
    a = np.array([100.0, 100.0])
    np.testing.assert_allclose(expected_costs(y, a, CFG), [[115.0, 23.0, 0.0], [0.0, 2.0, 25.0]])


def test_summarize_hand_case():
    y = np.array([1, 0, 1, 0])
    a = np.array([100.0, 100.0, 50.0, 10.0])
    acts = np.array([BLOCK, CHALLENGE, ALLOW, BLOCK])
    s = summarize(acts, y, a, CFG)
    # fraud: blocked 0 + allowed 50 + 15 = 65; legit: challenged 2, blocked 0.2*10 + 5 = 7
    assert s["fraud_loss"] == pytest.approx(65.0)
    assert s["challenge_friction_cost"] == pytest.approx(2.0)
    assert s["false_decline_cost"] == pytest.approx(7.0)
    assert s["total_cost"] == pytest.approx(74.0)
    assert s["fraud_$_stopped"] == pytest.approx(100 / 150)
    assert (s["allow_rate"], s["challenge_rate"], s["block_rate"]) == (0.25, 0.25, 0.5)


def _brute_force(p, y, a, cfg, challenge):
    cuts = np.r_[np.inf, np.unique(p)]
    if challenge:
        candidates = [(tb, tc) for tb in cuts for tc in cuts if tc <= tb]
    else:
        candidates = [(t, np.inf) for t in cuts]
    return min(summarize(threshold_actions(p, tb, tc), y, a, cfg)["total_cost"] for tb, tc in candidates)


@pytest.mark.parametrize("challenge", [False, True])
def test_tune_thresholds_matches_brute_force(challenge):
    rng = np.random.default_rng(0)
    n = 60
    p = np.round(rng.beta(0.5, 5, n), 2)  # rounding creates ties the tuner must not split
    y = (rng.random(n) < p).astype(int)
    a = rng.lognormal(4, 1, n)
    tb, tc = tune_thresholds(p, y, a, CFG, challenge=challenge)
    got = summarize(threshold_actions(p, tb, tc), y, a, CFG)["total_cost"]
    assert got == pytest.approx(_brute_force(p, y, a, CFG, challenge))
    if not challenge:
        assert tc == np.inf


def test_isotonic_calibration_fixes_a_distorted_score():
    rng = np.random.default_rng(1)
    p_true = rng.beta(0.3, 6, 20_000)
    y = (rng.random(len(p_true)) < p_true).astype(int)
    score = np.sqrt(p_true)  # monotone, so same ranking, but badly overconfident
    cal = fit_calibrator(score[:10_000], y[:10_000], "isotonic")
    t_raw = reliability_table(y[10_000:], score[10_000:])
    t_cal = reliability_table(y[10_000:], cal(score[10_000:]))
    assert t_cal["gap"].abs().max() < 0.03 < t_raw["gap"].abs().max()


def test_policy_pipeline_on_synthetic(tmp_path):
    from fraudradar.policy import pipeline

    synthetic.write(tmp_path / "raw", n_accounts=800, n_days=90, seed=3)
    df = load_raw(tmp_path / "raw")
    tables = pipeline.run(df, tcfg=TrainConfig(num_boost_round=50, early_stopping_rounds=10), out_dir=tmp_path)
    assert (tmp_path / "policy_results.csv").exists()
    assert (tmp_path / "policy_sensitivity.csv").exists()
    r = tables["policy_results"]
    assert r.loc["expected_cost", "total_cost"] < r.loc["allow_all", "total_cost"]
    mix = r[["allow_rate", "challenge_rate", "block_rate"]].sum(axis=1)
    np.testing.assert_allclose(mix, 1.0)
