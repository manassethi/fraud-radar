"""End to end: train the pit model, calibrate on validation, compare policies on test."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraudradar.config import CalibrationConfig, CostConfig, FeatureConfig, SplitConfig, TrainConfig
from fraudradar.models.calibration import (
    brier,
    expected_calibration_error,
    fit_calibrator,
    reliability_table,
)
from fraudradar.models.train import fit
from fraudradar.policy.evaluate import Fold, compare_policies, sensitivity


def run(
    df: pd.DataFrame,
    costs: CostConfig = CostConfig(),
    ccfg: CalibrationConfig = CalibrationConfig(),
    fcfg: FeatureConfig = FeatureConfig(),
    scfg: SplitConfig = SplitConfig(),
    tcfg: TrainConfig = TrainConfig(),
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    m = fit(df, "pit", fcfg, scfg, tcfg)
    sv, st = m.split.val, m.split.test
    raw_val, raw_test = m.predict(sv), m.predict(st)
    y_val, y_test = m.y[sv], m.y[st]

    # Calibration report on test: raw booster output vs each calibrator fit on validation.
    probs = {"raw": raw_test}
    for method in ("platt", "isotonic"):
        probs[method] = fit_calibrator(raw_val, y_val, method)(raw_test)
    calibration = pd.DataFrame({
        name: {"brier": brier(y_test, p), "ece": expected_calibration_error(y_test, p, ccfg.n_bins),
               "mean_pred": p.mean(), "fraud_rate": y_test.mean()}
        for name, p in probs.items()
    }).T
    calibration.index.name = "scores"
    reliability = pd.concat(
        {name: reliability_table(y_test, p, ccfg.n_bins) for name, p in probs.items()}, names=["scores"]
    )

    cal = fit_calibrator(raw_val, y_val, ccfg.method)
    val = Fold(cal(raw_val), y_val, m.amount[sv])
    test = Fold(probs[ccfg.method], y_test, m.amount[st])
    results = compare_policies(val, test, costs, raw_test_p=raw_test)
    sweep = sensitivity(val, test, costs)

    tables = {"policy_results": results, "policy_sensitivity": sweep,
              "calibration": calibration, "calibration_reliability": reliability}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, t in tables.items():
            t.to_csv(out_dir / f"{name}.csv", index=name != "policy_sensitivity")
    return tables
