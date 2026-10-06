# Task: ICU in-hospital mortality from the first 48 hours (PhysioNet 2012)

Predict **in-hospital death** for ICU stays from the first 48 hours of
routinely collected data. This is a RETROSPECTIVE stay-level prediction: you
see the full 48-hour window, so whole-window aggregates are fine (no
real-time constraint).

## Data format (long time series)

`train_series.parquet` / `test_features.parquet`, one row per observation:

- `record_id` — the ICU stay you predict for
- `minutes` — minutes since ICU admission (0..2880)
- `parameter` — what was measured (about 37 time-varying signals: vitals,
  labs, urine output; plus the general descriptors `Age`, `Gender`, `Height`,
  `ICUType`, `Weight` recorded at minute 0)
- `value` — the measurement (missing static descriptors are NaN)

Measurements are irregular and sparse — most parameters are observed only a
few times per stay. `train_labels.csv` maps `record_id` to the label
(1 = died in hospital, ~14% of stays).

## Files

| file | contents |
|---|---|
| `train_series.parquet` + `train_labels.csv` | 4,000 labeled ICU stays |
| `test_features.parquet` | 8,000 held-out stays (the challenge's two test cohorts, merged) |
| `sample_submission.csv` | valid null: training prevalence, never predicting death |
| `validate.py` | format checker — `python validate.py submission.csv` |

## What to submit

`./submission.csv` with exactly: `record_id, PredictedProbability,
PredictedLabel` — probability of in-hospital death in [0,1] and your binary
prediction in {0,1} (the official score uses the binary column), one row per
test stay.

## Scoring (hidden labels, from your file alone)

- **`score1_min_se_ppv`** — primary; the challenge's official event-1 score:
  min(sensitivity, positive predictive value) of your binary predictions.
  Never predicting death scores 0.

## Environment

One machine: 8 CPU threads (4 physical cores), 16 GB of memory, no GPU. The
memory limit is enforced: a process that exceeds it is killed. `python` (3.11)
with numpy, pandas, pyarrow, scipy, scikit-learn, lightgbm, xgboost, torch
(CPU), torchtuples, pycox, lifelines, scikit-survival and matplotlib (the
example datasets bundled with lifelines, scikit-survival and pycox have been
removed). No network access; additional packages cannot be installed. Use only
the files provided here.

## Rules

- Work entirely inside this directory; do not read or write files outside it.
- Do not attempt to locate the hidden test labels.
- The run has a fixed wall-clock budget of 12 hours (720 minutes). The
  `submission.csv` on disk when the budget ends is what gets graded; an
  invalid or missing submission is a failed run.
