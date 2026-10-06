# Task: Breast-cancer-specific survival (SEER registry)

You are building a **cancer-specific survival** model on SEER (the US national
cancer registry program, Research data, 17 registries): female breast cancer,
first malignant primary, diagnosed 2010-2015. One row per patient.

**Outcome**: time from diagnosis to **death attributable to breast cancer**
(`duration` in months, `event` = 1). Deaths from other causes are censored at
the death time.

**Covariates** (values are SEER's own codes):
`age` (years, 90 = 90+), `year_dx`, `race`, `hispanic_origin`,
`histology_icdo3` (ICD-O-3 histology code), `grade`, `laterality`,
`summary_stage`, `ajcc_t/n/m/stage` (AJCC 7th ed.), `surgery_code`
(RX Summ--Surg Prim Site), `tumor_size_cs` (mm; 989+ are special codes,
999 = unknown), `nodes_examined` / `nodes_positive` (90+ are special codes),
`er_status`, `pr_status`, `her2_status`, `breast_subtype`,
`median_household_income` (county-level band), `rural_urban_code`.
Treatment fields in SEER are known to be under-ascertained (official caveat).

## Files

| file | contents |
|---|---|
| `train.csv` | 232,625 patients WITH labels |
| `test_features.csv` | 49,291 patients, features only |
| `time_grid.csv` | evaluation times (quantile-spaced): submission column ↔ time |
| `sample_submission.csv` | valid null example: train-set Kaplan-Meier for every patient |
| `validate.py` | format checker — `python validate.py submission.csv` |

## What to submit

`./submission.csv` with columns `patient_id, t_000, t_001, ...` (every column
in `time_grid.csv`, in order): `t_k` = predicted probability of being alive
(or dead only of other causes) beyond the mapped time ([0,1], non-increasing
per row), one row per test patient.

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
