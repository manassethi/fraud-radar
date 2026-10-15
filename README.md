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
out-of-time split (fit | validation | gap | test). Results on the IEEE-CIS train set
(590,540 transactions; validation 51,488 rows; test = last 25% of the timeline, 128,823 rows,
3.41% fraud):

| features | val AUC | test AUC | val − test | test PR-AUC | test recall @1% FPR | test $-recall @1% FPR |
|---|---|---|---|---|---|---|
| raw | 0.906 | 0.885 | 0.020 | 0.460 | 0.379 | 0.266 |
| **pit** | 0.926 | **0.931** | **−0.005** | **0.595** | **0.527** | **0.423** |
| leaky | **0.999** | 0.802 | 0.197 | 0.359 | 0.332 | 0.201 |

- **Point-in-time features help where it matters.** Versus raw columns, test PR-AUC rises from
  0.46 to 0.59 and the share of fraud *dollars* caught at a 1% false-positive rate from 27% to 42%.
  `account_uid__fraud_rate_known` (reconstructed account, labels delayed 30 days) is the
  third most important feature.
- **Leakage is invisible offline and costly out-of-time.** The leaky model scores 0.999
  validation AUC, the best of the three, yet is the *worst* on the future test period
  (0.802, below even the raw baseline). Its validation-to-test gap is 0.20 AUC; the PIT model's
  is essentially zero, so its offline numbers can be trusted.

## Quickstart

```bash
pip install -e ".[dev]"
python -m fraudradar.cli synth            # or put Kaggle's train_*.csv in data/raw/
python -m fraudradar.cli train --features pit
python -m fraudradar.cli leakage
pytest
```

Real data: accept the competition rules on Kaggle, save your API token to `~/.kaggle/access_token`, then
download and unzip the two train files (only they have labels):

```bash
python -m kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/raw
python -m kaggle competitions download -c ieee-fraud-detection -f train_identity.csv -p data/raw
```

The CSVs are ~700 MB. To keep them outside the repo (e.g. a synced folder), set
`FRAUDRADAR_RAW_DIR=/path/to/csvs`. Add `--include-v` to load Vesta's V1–V339 columns (more memory).

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
