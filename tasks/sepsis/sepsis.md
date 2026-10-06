# Task: Early prediction of sepsis (PhysioNet/CinC Challenge 2019 data)

You are building an **hourly early-warning model for sepsis** in ICU patients.
The data are hourly records from two hospital systems (column `hospital`: A/B):
8 vital signs, 26 laboratory values, and 6 demographics per hour (40 feature
columns total, heavy missingness — labs are measured infrequently), plus:

- `patient`: patient identifier (each patient is one ICU stay)
- `ICULOS`: hour counter since ICU admission (1, 2, 3, ...)
- `SepsisLabel` (train only): becomes 1 starting **6 hours before** clinical
  sepsis onset and stays 1 afterwards; always 0 for non-septic patients.
  About 7% of patients ever become septic.

## The real-time rule (critical)

This is an early-warning task: your prediction for patient p at hour t must be
computable **using only that patient's rows with ICULOS ≤ t** (plus anything
learned from the training set). Later hours of the same test patient must not
inform earlier predictions. Carrying values forward and backward-looking
rolling statistics are allowed; centered windows and whole-stay aggregates are
not.

## Files in this directory

| file | contents |
|---|---|
| `train.parquet` | ~32k patients (~1.2M rows) WITH `SepsisLabel` |
| `test_features.parquet` | ~8k held-out patients, features only |
| `sample_submission.csv` | all-zero predictions — a valid "never alarm" null (scores exactly 0) |
| `utility_score.py` | the official challenge scoring code, with a `score_cohort(labels_df, pred_df)` helper |
| `validate.py` | format checker — run `python validate.py submission.csv` |

## What to submit

Write `./submission.csv` with exactly these columns:

```
patient, ICULOS, PredictedProbability, PredictedLabel
```

one row for every (patient, ICULOS) pair in `test_features.parquet`:
`PredictedProbability` in [0,1], `PredictedLabel` in {0,1} (your alarm
decision; the utility score is computed from these binary labels).

## How you are scored

- **`utility_norm`** — primary. The challenge's official clinical utility,
  normalized so that never alarming = 0 and clinically optimal alarming = 1.
  Per patient-hour: a true alarm earns up to +1 (maximum when raised 6h before
  onset, decaying to 0 at 12h early or 3h late), a false alarm costs −0.05,
  a missed septic hour after onset costs up to −2. Negative totals are
  possible. Exact definition: `utility_score.py`.

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
- Honor the real-time rule above; code and submissions are reviewed for it.
- The run has a fixed wall-clock budget of 12 hours (720 minutes). The
  `submission.csv` on disk when the budget ends is what gets graded; an
  invalid or missing submission is a failed run.
