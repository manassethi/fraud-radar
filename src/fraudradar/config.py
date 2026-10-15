"""Central configuration. Everything time-related is in seconds, matching TransactionDT."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DAY = 86_400
HOUR = 3_600

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Set FRAUDRADAR_RAW_DIR to keep the (large) Kaggle CSVs outside the repo.
RAW_DIR = Path(os.environ.get("FRAUDRADAR_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class FeatureConfig:
    # Entities we compute velocity / history features for. Each must be a column
    # produced by `identity.add_entity_keys`.
    entities: tuple[str, ...] = ("card_uid", "account_uid", "device_uid")
    # Look-back windows for velocity features.
    windows: dict[str, int] = field(
        default_factory=lambda: {"1h": HOUR, "24h": DAY, "7d": 7 * DAY}
    )
    # A transaction's label (chargeback) is assumed to become known this long after
    # the transaction. Entity fraud-history features may only use labels known by
    # the time of scoring. 30 days is a conservative chargeback horizon.
    label_delay: int = 30 * DAY
    # Attribute values (e.g. a device fingerprint) already shared by this many
    # distinct cards stop linking new cards into a cluster. Prevents hubs like
    # "Windows / chrome" from merging the whole graph into one component.
    max_hub_degree: int = 20


@dataclass(frozen=True)
class SplitConfig:
    # Fractions of the time axis (by TransactionDT), not of rows.
    train_end: float = 0.70
    # Labels for the last `label_delay` of training would not be known yet at the
    # train cutoff in production; drop them so evaluation mimics deployment.
    drop_immature_labels: bool = True
    test_start: float = 0.75


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    params: dict = field(
        default_factory=lambda: {
            "objective": "binary",
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 100,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbose": -1,
        }
    )


@dataclass(frozen=True)
class CalibrationConfig:
    # "isotonic" (non-parametric, monotone) or "platt" (logistic on the score's logit).
    method: str = "isotonic"
    # Equal-count bins for the reliability table.
    n_bins: int = 10


@dataclass(frozen=True)
class CostConfig:
    """Dollar costs of each (action, outcome) pair, from the merchant's point of view.

    Costs are relative to the ideal outcome for each transaction (legit approved, fraud
    stopped), so allowing a legitimate payment costs 0 and blocking a fraudulent one costs 0.
    """

    # Allowed fraud becomes a chargeback: the merchant refunds the amount and pays a fee.
    dispute_fee: float = 15.0
    # Gross margin on a sale. A false decline loses margin * amount, not the whole amount.
    margin: float = 0.20
    # A declined good customer may not come back: P(churn | false decline) * lifetime value.
    decline_churn_rate: float = 0.05
    customer_ltv: float = 100.0
    # 3DS challenge: this share of legitimate customers abandon the checkout ...
    challenge_dropoff: float = 0.10
    # ... and this share of fraud attempts fail authentication and are stopped.
    challenge_catch_rate: float = 0.80

    @property
    def false_decline_fixed(self) -> float:
        return self.decline_churn_rate * self.customer_ltv
