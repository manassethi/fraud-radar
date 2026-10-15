# fraud-radar

Payment fraud detection built the way a production system has to be: every feature is
**point-in-time correct**, labels arrive **late** (chargebacks), and evaluation is
**out-of-time**. Built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
dataset (~590K real card-not-present e-commerce transactions, 3.5% fraud).

> Status: **phase 1** — identity reconstruction, point-in-time features, LightGBM baseline,
> leakage experiment. Planned: real-time scoring service (FastAPI + Redis), cost-optimal
> allow / challenge / block policy, counterfactual evaluation under selective labels, drift monitoring.

## Why this is hard

1. **No customer ID.** IEEE-CIS hides who is who. Fraud is labelled at the *account* level,
   so recovering accounts is most of the signal (`data/identity.py`).
2. **Features must not see the future.** `groupby().transform("mean")` silently uses tomorrow's
   transactions and today's label. Offline metrics then look great and the model fails in production.
3. **Labels are late.** A chargeback can arrive 30+ days after the transaction. At scoring time,
   an account's recent fraud is not known yet.

## Identity reconstruction

| Key | Built from | Meaning |
|---|---|---|
| `card_uid` | card1–card6 | card/issuer attributes |
| `account_uid` | card_uid + addr1 + **D1n** | `D1n = day − D1` is the card's start day, constant per cardholder |
| `device_uid` | DeviceInfo + OS + browser + screen | device fingerprint |
| clusters | online union-find over account ↔ device edges | fraud rings sharing devices; hub cap stops generic fingerprints from merging everything |

## Point-in-time features (`features/pit.py`)

For each entity and each transaction at time *t*, using only rows strictly before *t*:
velocity counts and amount sums (1h / 24h / 7d), prior mean amount and the amount ratio,
time since the previous and first transaction, and **fraud history using only labels known
by t** (`label_delay`, default 30 days). Graph features record the cluster size *as of t*.

Fully vectorised: rows of an entity are made contiguous and windows become `searchsorted`
lookups over `entity * BIG + ts` with prefix sums. `tests/test_pit_features.py` rewrites
every transaction after a cutoff (amounts, labels, devices, plus injected rows) and asserts
that no feature before the cutoff changes.

## Leakage experiment

`python -m fraudradar.cli leakage` trains the same LightGBM on three feature sets with an
out-of-time split (fit | validation | gap | test):

| features | val AUC | test AUC | val − test | test recall @1% FPR | test $-recall @1% FPR |
|---|---|---|---|---|---|
| raw | 0.777 | 0.686 | 0.091 | 0.273 | 0.267 |
| **pit** | 0.952 | **0.912** | **0.040** | **0.788** | **0.789** |
| leaky | **0.998** | 0.696 | 0.302 | 0.461 | 0.475 |

*Numbers above are from the **synthetic** data (`cli synth`), not IEEE-CIS. Replace them
with your real-data run.* The leaky model has the best validation score and the worst
generalisation: a 0.30 AUC gap that a random split would never show.

## Quickstart

```bash
pip install -e ".[dev]"
python -m fraudradar.cli synth            # or put Kaggle's train_*.csv in data/raw/
python -m fraudradar.cli train --features pit
python -m fraudradar.cli leakage
pytest
```

Real data: accept the competition rules on Kaggle, then
`kaggle competitions download -c ieee-fraud-detection -p data/raw` and unzip.
Add `--include-v` to load Vesta's V1–V339 columns (more memory).

Outputs go to `artifacts/<run>/`: `model.txt`, `metrics.json`, `feature_importance.csv`.

## Layout

```
src/fraudradar/
  config.py            windows, label delay, split, LightGBM params
  data/loader.py       read + merge IEEE-CIS transaction/identity tables
  data/identity.py     entity keys + point-in-time account-device graph
  data/synthetic.py    IEEE-CIS-format generator (account takeover, fraud rings)
  features/pit.py      point-in-time entity history features
  features/leaky.py    the wrong way, kept as a control
  models/train.py      out-of-time split, LightGBM, artifacts
  models/metrics.py    ROC/PR AUC, recall and dollar recall at fixed FPR
  cli.py
tests/
```

## Notes and limitations

- The synthetic generator is for running the pipeline and CI, not for drawing conclusions.
  It makes card attributes nearly unique per account, so `card_uid` features dominate there;
  on IEEE-CIS `account_uid` should matter much more.
- A fixed `label_delay` is a simplification. Real chargebacks have a delay distribution, and
  some fraud is confirmed sooner (customer reports).
- `drop_immature_labels` (off by default) also removes training rows whose labels would not
  exist yet at the training cutoff.
