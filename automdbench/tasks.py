"""The 15 tasks."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    dataset: str              # folder under tasks/ that builds the package
    kind: str                 # "survival", "binary" or "sepsis"
    metric: str               # primary metric
    id_col: str = "patient_id"
    need_label: bool = False  # the submission also carries PredictedLabel


TASKS = {
    # survival analysis
    "support": Task("support", "survival", "concordance_td"),
    "metabric": Task("metabric", "survival", "concordance_td"),
    "rotterdam": Task("rotterdam", "survival", "concordance_td"),
    "rotterdam_gbsg": Task("rotterdam", "survival", "concordance_td"),
    "seer": Task("seer", "survival", "concordance_td"),
    "seer_later_period": Task("seer", "survival", "concordance_td"),
    "nhanes": Task("nhanes", "survival", "concordance_td"),
    # classification
    "uci_diabetes": Task("uci_diabetes", "binary", "auroc", "encounter_id"),
    "physionet2012": Task("physionet2012", "binary", "score1_min_se_ppv", "record_id", need_label=True),
    "mimiciv": Task("mimiciv", "binary", "auroc", "stay_id"),
    "eicu": Task("eicu", "binary", "auroc", "stay_id"),
    "eicu_new_hospitals": Task("eicu", "binary", "auroc", "stay_id"),
    # sequential prediction
    "sepsis": Task("sepsis", "sepsis", "utility_norm", "patient", need_label=True),
    "sepsis_new_hospital": Task("sepsis", "sepsis", "utility_norm", "patient", need_label=True),
    "sepsis_500_labels": Task("sepsis", "sepsis", "utility_norm", "patient", need_label=True),
}

# Calibration control (paper appendix): UCI Diabetes with the scoring section of the
# description changed to Brier score, or to AUROC and ECE. Same data and labels.
VARIANTS = {
    "uci_diabetes_brier": TASKS["uci_diabetes"],
    "uci_diabetes_ece": TASKS["uci_diabetes"],
}

# Score of an uninformative prediction.
NULL_SCORE = {"concordance_td": 0.5, "auroc": 0.5, "score1_min_se_ppv": 0.0, "utility_norm": 0.0}


def get(task):
    return TASKS.get(task) or VARIANTS[task]


def normalize(metric, x):
    """(x - null) / (1 - null), clipped to [0, 1]."""
    null = NULL_SCORE[metric]
    return min(1.0, max(0.0, (x - null) / (1.0 - null)))
