# fraud-radar

Payment fraud detection built the way a production system has to be: every feature is
**point-in-time correct**, labels arrive **late** (chargebacks), and evaluation is
**out-of-time**. Built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
dataset (~590K real card-not-present e-commerce transactions, 3.5% fraud).

> Status: **phase 1** — identity reconstruction, point-in-time features, LightGBM baseline,
> leakage experiment. **Phase 2** — calibrated, cost-optimal allow / 3DS challenge / block policy.
> Planned: counterfactual evaluation under selective labels, real-time scoring service
> (FastAPI + Redis), drift monitoring.

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

## Decision policy: allow / 3DS challenge / block

A score is not a decision. `python -m fraudradar.cli policy` turns the pit model's scores into
actions by **minimising expected dollar cost per transaction** (`policy/decision.py`).

**Cost matrix** (merchant's view, relative to the ideal outcome; all parameters in `CostConfig`):

| action | if fraud (prob. p) | if legit (1 − p) |
|---|---|---|
| allow | A + F (chargeback + dispute fee, F = $15) | 0 |
| 3DS challenge | (1 − c)(A + F), c = 80% of fraud fails 3DS | d · m · A, d = 10% abandon, m = 20% margin |
| block | 0 | m · A + L, L = 5% churn × $100 LTV |

The policy takes `argmin_a  p · cost(a, fraud) + (1 − p) · cost(a, legit)`. That is a threshold
rule whose thresholds **depend on the amount**: block beats allow when
p/(1 − p) > (mA + L)/(A + F). Small payments are dominated by the fixed dispute fee and get
challenged at low risk. On large ones a 10% checkout drop-off is expensive, so challenges are rarer.

**Calibration first.** The rule multiplies p by dollars, so p must be a probability, not just a
ranking. Calibrators (isotonic, Platt) are fit on the validation fold and judged on test:

| scores | Brier | ECE | mean p | test fraud rate |
|---|---|---|---|---|
| raw LightGBM | **0.0198** | **0.0040** | 3.04% | 3.41% |
| Platt | 0.0199 | 0.0042 | 3.78% | 3.41% |
| isotonic (used) | 0.0200 | 0.0050 | 3.77% | 3.41% |

LightGBM trained on log-loss without class reweighting is already close to calibrated, so
calibration does not help here. The calibrators learn validation's 3.64% fraud rate and slightly
over-predict on the test period. Calibration matters much more once training downsamples or reweights
the negatives, which production fraud models often do. The full reliability table is in
`artifacts/calibration_reliability.csv`.

**Results on the out-of-time test set** (128,823 transactions; every threshold is tuned on validation
under the same costs):

| policy | total cost | $/txn | allow / 3DS / block | fraud $ stopped | good customers blocked | saving vs allow-all |
|---|---|---|---|---|---|---|
| allow all | $734,582 | 5.70 | 100 / 0 / 0 % | 0% | 0% | — |
| block if p ≥ 0.5 | $502,527 | 3.90 | 98.1 / 0 / 1.9 % | 33.1% | 0.45% | 31.6% |
| best single threshold | $471,154 | 3.66 | 95.7 / 0 / 4.3 % | 52.6% | 2.34% | 35.9% |
| best two thresholds (allow/3DS/block) | $276,589 | 2.15 | 82.2 / 15.8 / 2.0 % | 74.6% | 0.46% | 62.3% |
| **expected-cost policy** | **$271,466** | **2.11** | 77.9 / 20.2 / 1.9 % | **77.5%** | **0.42%** | **63.0%** |
| expected-cost, uncalibrated scores | $276,954 | 2.15 | 80.1 / 18.3 / 1.6 % | 73.7% | 0.30% | 62.3% |

- **Most of the value comes from having a middle action.** A binary allow/block threshold can only
  trade fraud losses for false declines. Adding 3DS cuts cost from $471K to $277K, because a
  challenge stops 80% of fraud at ~2% of the margin on a legitimate sale.
- **Amount-aware decisions add a further ~2%** over the best two-threshold policy ($5.1K here).
  The gain is real but modest. The cost model's structure (which actions exist) matters more than per-transaction tuning.
- **The tuned single threshold blocks 2.3% of good customers** to catch 53% of fraud dollars. The
  expected-cost policy blocks 0.4% and stops 78%.

**Sensitivity** (`artifacts/policy_sensitivity.csv`; dispute fee $0–50 × margin 5–40%, no
retraining, since only the decisions depend on costs): the expected-cost policy beats the best single
threshold by 29–54% in all 16 settings. It beats the best two-threshold policy in 15 of 16, by 0.4–4.6%,
and loses by 0.15% at fee $50 / margin 5%. Its challenge rate goes from 12% (40% margin, $0 fee)
to 55% (5% margin, $50 fee). When margins are thin, friction is cheap, so challenging is cheap too.

Caveats: a 20% challenge rate is high for real checkout flows, and the result is sensitive to
`challenge_dropoff` and `challenge_catch_rate`, which are assumptions. 3DS outcomes are scored at their
expected value because we cannot observe what a challenge would have done. Under 3DS liability shift,
authenticated fraud is the issuer's loss, which would make challenges even cheaper for the merchant
(not modelled). The test labels exist only for transactions that were historically approved
(selective labels). That is the next phase: counterfactual evaluation with IPS / doubly robust estimators.

## Quickstart

```bash
pip install -e ".[dev]"
python -m fraudradar.cli synth            # or put Kaggle's train_*.csv in data/raw/
python -m fraudradar.cli train --features pit
python -m fraudradar.cli leakage
python -m fraudradar.cli policy           # calibration + allow/3DS/block policy comparison
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
  models/calibration.py  isotonic / Platt calibration, reliability table, Brier, ECE
  policy/decision.py   cost matrix, argmin decision rule, exact threshold tuner
  policy/evaluate.py   realised cost and action mix per policy, sensitivity sweep
  policy/pipeline.py   train -> calibrate on val -> compare policies on test
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
