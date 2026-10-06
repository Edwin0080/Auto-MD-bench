# Task: Early prediction of sepsis — FEW-SHOT adaptation to a new hospital

You are building an **hourly early-warning model for sepsis** in ICU patients.
**Your training data are all of hospital system A plus only 500 labeled
patients from hospital system B — and your model will be evaluated on other
hospital-B patients.** Sites differ in measurement practice and patient mix.

The data are hourly ICU records (column `hospital`: A/B): 8 vital signs, 26
laboratory values, 6 demographics per hour (40 feature columns, heavy
missingness), plus:

- `patient`, `ICULOS` (hour counter), and `SepsisLabel` (train only):
  becomes 1 starting **6 hours before** sepsis onset; ~7% of patients ever
  become septic.

## The real-time rule (critical)

Predictions for patient p at hour t may use only that patient's rows with
ICULOS ≤ t (plus anything learned from training). Carrying values forward
and backward-looking rolling statistics are allowed; centered windows and
whole-stay aggregates are not.

## Files

| file | contents |
|---|---|
| `train.parquet` | 20,336 hospital-A patients + 500 hospital-B patients, WITH `SepsisLabel` |
| `test_features.parquet` | 4,000 other hospital-B patients, features only |
| `sample_submission.csv` | all-zero predictions — valid "never alarm" null (scores exactly 0) |
| `utility_score.py` | the official challenge scoring code + `score_cohort` helper |
| `validate.py` | format checker — `python validate.py submission.csv` |

## What to submit

`./submission.csv` with exactly: `patient, ICULOS, PredictedProbability,
PredictedLabel` — one row for every (patient, ICULOS) pair in
`test_features.parquet`; the utility score is computed from the binary
alarms.

## Scoring

- **`utility_norm`** — primary; official clinical utility on the hospital-B
  test patients (never alarm = 0, clinically optimal = 1; false alarm −0.05,
  late/missed alarms down to −2). Exact definition: `utility_score.py`.

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
