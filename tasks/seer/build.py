"""SEER breast cancer, cancer-specific survival (SEER Research 17, diagnoses 2010-2015).

    seer                train on all years except the test patients
    seer_later_period   train on 2010-2013 diagnoses only (external validation)
Both share one test set: a random half of the 2014-2015 diagnoses. Deaths from other causes are
censored; deaths with unknown cause and incomplete follow-up dates are excluded.

    python tasks/seer/build.py CASE_LISTING.csv [--out packages]
CASE_LISTING.csv is a SEER*Stat case listing with the columns named in RENAME and OUTCOME below.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from automdbench import metrics, package

HERE = Path(__file__).resolve().parent

RENAME = {
    "Patient ID": "patient_id",
    "Age recode with single ages and 90+": "age",
    "Year of diagnosis": "year_dx",
    "Race recode (W, B, AI, API)": "race",
    "Origin recode NHIA (Hispanic, Non-Hisp)": "hispanic_origin",
    "Histologic Type ICD-O-3": "histology_icdo3",
    "Grade Recode (thru 2017)": "grade",
    "Laterality": "laterality",
    "Combined Summary Stage with Expanded Regional Codes (2004+)": "summary_stage",
    "Derived AJCC T, 7th ed (2010-2015)": "ajcc_t",
    "Derived AJCC N, 7th ed (2010-2015)": "ajcc_n",
    "Derived AJCC M, 7th ed (2010-2015)": "ajcc_m",
    "Derived AJCC Stage Group, 7th ed (2010-2015)": "ajcc_stage",
    "RX Summ--Surg Prim Site (1998-2022)": "surgery_code",
    "CS tumor size (2004-2015)": "tumor_size_cs",
    "Regional nodes examined (1988+)": "nodes_examined",
    "Regional nodes positive (1988+)": "nodes_positive",
    "ER Status Recode Breast Cancer (1990+)": "er_status",
    "PR Status Recode Breast Cancer (1990+)": "pr_status",
    "Derived HER2 Recode (2010+)": "her2_status",
    "Breast Subtype (2010+)": "breast_subtype",
    "Median household income inflation adj to 2024": "median_household_income",
    "Rural-Urban Continuum Code": "rural_urban_code",
}
OUTCOME = {
    "SEER cause-specific death classification": "css_class",
    "Vital status recode (study cutoff used)": "vital",
    "Survival months": "surv_months",
    "Survival months flag": "surv_flag",
}


def main(raw, out):
    df = pd.read_csv(raw).rename(columns={**RENAME, **OUTCOME})
    df = df[df["css_class"] != "Dead (missing/unknown COD)"]
    df = df[df["surv_flag"] == "Complete dates are available and there are more than 0 days of survival"]
    df["age"] = df["age"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    df["duration"] = pd.to_numeric(df["surv_months"], errors="raise").astype(float)
    df["event"] = (df["css_class"] == "Dead (attributable to this cancer dx)").astype(int)
    df = df[list(RENAME.values()) + ["duration", "event"]].reset_index(drop=True)

    recent = df[df["year_dx"] >= 2014]
    test_ids = set(np.random.default_rng(42).permutation(recent["patient_id"].to_numpy())[: len(recent) // 2].tolist())
    is_test = df["patient_id"].isin(test_ids)
    later = df[~is_test & (df["year_dx"] <= 2013)].sort_values("patient_id").reset_index(drop=True)
    test = df[is_test]
    ids = package.opaque_ids(test["patient_id"].tolist(), "s")
    test = test.assign(patient_id=test["patient_id"].map(ids)).sort_values("patient_id").reset_index(drop=True)
    # one time grid for both tasks, from the smaller training set
    grid = metrics.make_time_grid(later["duration"], later["event"])
    package.survival(out, "seer", HERE / "seer.md",
                     df[~is_test].sort_values("patient_id").reset_index(drop=True), test, grid)
    package.survival(out, "seer_later_period", HERE / "seer_later_period.md", later, test, grid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", default="packages")
    main(**vars(ap.parse_args()))
