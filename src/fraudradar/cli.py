"""Command line entry point.

    python -m fraudradar.cli synth                 # write synthetic IEEE-CIS-format data
    python -m fraudradar.cli train --features pit  # train + evaluate one feature set
    python -m fraudradar.cli leakage               # raw vs pit vs leaky comparison table
    python -m fraudradar.cli policy                # calibrate + cost-based allow/3DS/block policy
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from fraudradar.config import ARTIFACT_DIR, RAW_DIR
from fraudradar.data import synthetic
from fraudradar.data.loader import load_raw
from fraudradar.models.train import default_out_dir, train
from fraudradar.policy import pipeline


def _summary_row(r: dict) -> dict:
    return {
        "features": r["feature_set"],
        "val_auc": r["val"]["roc_auc"],
        "test_auc": r["test"]["roc_auc"],
        "val_minus_test": r["val"]["roc_auc"] - r["test"]["roc_auc"],
        "test_pr_auc": r["test"]["pr_auc"],
        "test_recall@1%fpr": r["test"]["recall@1pct_fpr"],
        "test_$recall@1%fpr": r["test"]["dollar_recall@1pct_fpr"],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fraudradar")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate synthetic data into data/raw")
    s.add_argument("--accounts", type=int, default=4000)
    s.add_argument("--seed", type=int, default=0)

    for name in ("train", "leakage", "policy"):
        p = sub.add_parser(name)
        p.add_argument("--include-v", action="store_true", help="load Vesta's V1..V339 columns")
        p.add_argument("--nrows", type=int, default=None)
    sub.choices["train"].add_argument("--features", choices=["raw", "pit", "leaky"], default="pit")

    args = ap.parse_args(argv)

    if args.cmd == "synth":
        path = synthetic.write(RAW_DIR, n_accounts=args.accounts, seed=args.seed)
        print(f"wrote synthetic data to {path}")
        return

    df = load_raw(include_v=args.include_v, nrows=args.nrows)
    print(f"loaded {len(df):,} transactions, fraud rate {df['isFraud'].mean():.2%}")

    if args.cmd == "train":
        r = train(df, args.features, out_dir=default_out_dir(args.features))
        print(json.dumps({k: r[k] for k in ("feature_set", "best_iteration", "test")}, indent=2))
        print("top features:", r["top_features"])
        return

    if args.cmd == "policy":
        tables = pipeline.run(df, out_dir=ARTIFACT_DIR)
        with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 200,
                               "display.max_columns", None):
            for name in ("calibration", "policy_results", "policy_sensitivity"):
                print(f"\n== {name} ==")
                print(tables[name])
        print(f"\nwrote CSVs to {ARTIFACT_DIR}")
        return

    rows = [_summary_row(train(df, fs, out_dir=default_out_dir(fs))) for fs in ("raw", "pit", "leaky")]
    table = pd.DataFrame(rows).set_index("features")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(ARTIFACT_DIR / "leakage_experiment.csv")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200,
                           "display.max_columns", None):
        print(table)


if __name__ == "__main__":
    main()
