"""eICU-CRD 2.0: in-hospital death from the first 24 h of an ICU stay (adults, ICU stay >= 1 day).

Hospitals are split about 80/20 by number of stays; half of the patients of the held-out hospitals
form the shared test set.
    eicu                 train on every patient not in the test set, from all hospitals
    eicu_new_hospitals   train on patients of the other hospitals only (external validation)

    bash tasks/eicu/extract.sh EICU_DIR WORK_DIR
    python tasks/eicu/build.py EICU_DIR WORK_DIR [--out packages]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from automdbench import icu, package

HERE = Path(__file__).resolve().parent
CHUNK = 4_000_000
EXTRA = ["gcs", "vent", "urine_out", "pao2", "fio2", "pre_icu_los_days", "elective_surgery", "aids",
         "hepatic_failure", "lymphoma", "metastatic_cancer", "leukemia", "immunosuppression", "cirrhosis",
         "diabetes"]
VITALS = {"heartrate": "heart_rate", "sbp": "sbp", "dbp": "dbp", "map": "map", "respiration": "resp_rate",
          "temperature": "temp_c", "sao2": "spo2"}
LAB_NAMES = {"glucose": "glucose", "bedside glucose": "glucose", "creatinine": "creatinine", "BUN": "bun",
             "sodium": "sodium", "potassium": "potassium", "chloride": "chloride", "bicarbonate": "bicarbonate",
             "Hgb": "hemoglobin", "Hct": "hematocrit", "WBC x 1000": "wbc", "platelets x 1000": "platelets",
             "lactate": "lactate", "total bilirubin": "bilirubin", "albumin": "albumin"}
CHRONIC = {"aids": "aids", "hepaticfailure": "hepatic_failure", "lymphoma": "lymphoma",
           "metastaticcancer": "metastatic_cancer", "leukemia": "leukemia", "immunosuppression": "immunosuppression",
           "cirrhosis": "cirrhosis", "diabetes": "diabetes"}


def recorded(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s >= 0)  # -1 = not recorded


def cohort(eicu):
    pv = pd.read_csv(eicu / "apachePredVar.csv.gz",
                     usecols=["patientunitstayid", "gender", "age", "diedinhospital", "electivesurgery", *CHRONIC])
    pr = (pd.read_csv(eicu / "apachePatientResult.csv.gz", usecols=["patientunitstayid", "actualiculos"])
          .groupby("patientunitstayid", as_index=False)["actualiculos"].max())
    aps = pd.read_csv(eicu / "apacheApsVar.csv.gz", usecols=["patientunitstayid", "eyes", "motor", "verbal",
                                                             "vent", "urine", "pao2", "fio2"])
    pat = pd.read_csv(eicu / "patient.csv.gz",
                      usecols=["patientunitstayid", "uniquepid", "hospitalid", "hospitaladmitoffset"])
    df = pv.merge(pr, on="patientunitstayid").merge(aps, on="patientunitstayid", how="left").merge(
        pat, on="patientunitstayid")
    df = df[(df["age"] >= 18) & (df["gender"] >= 0) & (df["diedinhospital"] >= 0) & (df["actualiculos"] >= 1.0)]
    eyes, motor, verbal = (recorded(df[c]) for c in ("eyes", "motor", "verbal"))
    df["gcs"] = (eyes + motor + verbal).where(eyes.notna() & motor.notna() & verbal.notna())
    df.loc[(df["gcs"] < 3) | (df["gcs"] > 15), "gcs"] = np.nan
    df["vent"] = recorded(df["vent"])
    df["urine_out"] = recorded(df["urine"])
    df["pao2"] = recorded(df["pao2"]).where(lambda s: (s > 20) & (s < 700))
    fio2 = recorded(df["fio2"])
    df["fio2"] = np.where(fio2 <= 1.0, fio2 * 100.0, fio2)  # percent; a few rows hold fractions
    df.loc[(df["fio2"] < 21) | (df["fio2"] > 100), "fio2"] = np.nan
    df["pre_icu_los_days"] = (-pd.to_numeric(df["hospitaladmitoffset"], errors="coerce") / 1440.0).clip(lower=0)
    df["elective_surgery"] = recorded(df["electivesurgery"])
    for src, dst in CHRONIC.items():
        df[dst] = recorded(df[src])
    df = df.rename(columns={"patientunitstayid": "stay_id", "diedinhospital": "label"})
    return df[["stay_id", "uniquepid", "hospitalid", "age", "gender", "label"] + EXTRA].reset_index(drop=True)


def features(eicu, work):
    df = cohort(eicu)
    keep = set(df["stay_id"])
    acc = None
    for c in pd.read_csv(work / "vitals_raw.csv", chunksize=CHUNK):
        c = c[c["patientunitstayid"].isin(keep) & (c["offset"] >= 0) & (c["offset"] <= 24 * 60)]
        for src, name in VITALS.items():
            v = pd.to_numeric(c[src], errors="coerce")
            m = v.notna().to_numpy()
            if m.any():
                acc = icu.fold(acc, c["patientunitstayid"].to_numpy()[m], np.full(int(m.sum()), name),
                               v.to_numpy()[m])
    vit = icu.finalize(acc, "stay_id")
    acc = None
    for c in pd.read_csv(work / "labs_raw.csv", chunksize=CHUNK):
        c = c[c["patientunitstayid"].isin(keep)]
        name, v = c["labname"].map(LAB_NAMES), pd.to_numeric(c["labresult"], errors="coerce")
        m = (name.notna() & v.notna()).to_numpy()
        if m.any():
            acc = icu.fold(acc, c["patientunitstayid"].to_numpy()[m], name.to_numpy()[m], v.to_numpy()[m])
    df = df.merge(vit, on="stay_id", how="left").merge(icu.finalize(acc, "stay_id"), on="stay_id", how="left")
    for c in icu.feature_columns() + EXTRA:
        if c not in df.columns:
            df[c] = np.nan
    return icu.clip_implausible(df)


def split(df):
    rng = np.random.default_rng(42)
    counts = df.groupby("hospitalid")["stay_id"].size()
    hospitals = np.array(sorted(counts.index))
    rng.shuffle(hospitals)
    test_hosp, n = set(), 0
    for h in hospitals:
        if n >= 0.20 * len(df):
            break
        test_hosp.add(int(h))
        n += int(counts[h])
    in_test_hosp = df["hospitalid"].isin(test_hosp)
    patients = np.array(sorted(df.loc[in_test_hosp, "uniquepid"].unique()))
    rng.shuffle(patients)
    test_pat = set(patients[: int(round(0.5 * len(patients)))])
    other = ~df["uniquepid"].isin(test_pat)  # every stay of a test patient leaves both training sets
    return df[other], df[other & ~df["hospitalid"].isin(test_hosp)], df[in_test_hosp & ~other]


def main(eicu, work, out):
    within, new_hosp, test = split(features(Path(eicu), Path(work)))
    cols = icu.feature_columns() + EXTRA
    for task, train, prefix in (("eicu", within, "ei"), ("eicu_new_hospitals", new_hosp, "ec")):
        ids = package.opaque_ids(test["stay_id"].tolist(), prefix)
        t = test.assign(stay_id=test["stay_id"].map(ids)).sort_values("stay_id").reset_index(drop=True)
        tr = train.reset_index(drop=True)[cols + ["label"]]
        tr.insert(0, "stay_id", [f"tr{i:06d}" for i in range(len(tr))])
        package.binary(out, task, HERE / f"{task}.md", tr, t[["stay_id"] + cols + ["label"]], "stay_id")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eicu", help="eICU-CRD 2.0 directory with the .csv.gz tables")
    ap.add_argument("work", help="output directory of extract.sh")
    ap.add_argument("--out", default="packages")
    main(**vars(ap.parse_args()))
