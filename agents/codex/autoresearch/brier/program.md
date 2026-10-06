# Research program: greedy iterative improvement

Your objective is the task in `description.md`: minimize its primary metric
on the hidden test via `./submission.csv`.

## Phase 0 — working baseline first

Write your own experiment script that goes end-to-end as simply as possible:
load the provided data, hold out an internal validation split from the
training data, fit a deliberately simple model, print your validation
score(s), and write a format-valid `./submission.csv`. Confirm
`python validate.py submission.csv` passes before any improvement work — a
valid submission on disk is your safety net.

## Loop (repeat until the time budget is spent)

1. Propose ONE focused change (features, preprocessing, model class,
   hyperparameters — one idea at a time).
2. Run the script; read the validation score.
3. Append one line to `journal.md`: what you changed, the numbers, and
   KEEP or REVERT.
4. KEEP the change only if the validation primary metric beat the best so
   far; otherwise revert it exactly.

## Hard rules

- Your experiment script(s) and `journal.md` are yours to edit; do not modify
  the provided data files, `validate.py`, or this file.
- Tune only against your internal validation split, never against the test
  features.
- Keep individual runs short enough to iterate many times; a long training
  needs a clear reason in `journal.md`.
- The final `./submission.csv` must come from your best-validated version
  (re-run it if you experimented after your best result).
- Everything happens inside this directory; no network access.
