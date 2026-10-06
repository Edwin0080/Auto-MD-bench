# Task: All-cause mortality after a national health examination (NHANES)

You are building a **survival model** for all-cause mortality in US adults,
using data from the National Health and Nutrition Examination Survey (NHANES)
linked to the National Death Index. Participants were examined between 2011 and
2016 and followed through the end of 2019, so follow-up runs from about 3 to 8
years and most people are still alive at the end — they are **right-censored**,
not event-free.

`duration` is months from the examination to death or to the end of follow-up;
`event` is 1 if the participant died, 0 if they were censored.

## This is survey data, not a hospital cohort

NHANES is a **complex probability sample** of the US population, not a
convenience sample. Three design columns ship with the data:

| column | meaning |
|---|---|
| `survey_weight` | examination weight — how many people in the population this participant represents |
| `survey_stratum` | sampling stratum |
| `survey_psu` | primary sampling unit (cluster) within the stratum |

Selection probabilities are deliberately unequal: some groups were oversampled
by design.

## Features

Demographics (`age`, `sex` (1 = male), `race_eth`, `education`,
`income_poverty_ratio`), body measures (`bmi`, `waist_cm`), blood pressure
(`sbp`, `dbp` — averaged over repeated readings), lipids
(`total_cholesterol`, `hdl_cholesterol`), `hba1c`, blood counts (`wbc`,
`hemoglobin`, `platelets`), biochemistry (`creatinine`, `albumin`, `glucose`,
`alk_phos`, `ast`), and `ever_smoked_100` (smoked 100+ cigarettes in life).
`cycle` is the survey cycle the participant came from.

Missing values are frequent and vary by component — not everyone got every lab.

## Files

| file | contents |
|---|---|
| `train.csv` | visible participants WITH `duration` / `event` |
| `test_features.csv` | held-out participants, features only |
| `time_grid.csv` | evaluation times (quantile-spaced): submission column ↔ time in months |
| `sample_submission.csv` | valid null example: the training-set Kaplan-Meier curve for everyone |
| `validate.py` | format checker — `python validate.py submission.csv` |

## What to submit

`./submission.csv` with columns `patient_id, t_000, t_001, ...` (every column
in `time_grid.csv`, in order): `t_k` = predicted probability of surviving
beyond the mapped time ([0,1], non-increasing along each row), one row per
participant in `test_features.csv`.

## Scoring (hidden labels, from your file alone)

- **`concordance_td`** (Antolini) — primary, higher better

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
