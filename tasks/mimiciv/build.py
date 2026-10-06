"""MIMIC-IV 3.1: in-hospital death from the first 24 h of the first ICU stay (adults, ICU stay >= 24 h).

    bash tasks/mimiciv/extract.sh MIMIC_ROOT WORK_DIR
    python tasks/mimiciv/build.py MIMIC_ROOT WORK_DIR [--out packages]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from automdbench import icu, package

HERE = Path(__file__).resolve().parent
CHUNK = 4_000_000
EXTRA = ["gcs", "vent", "urine_out", "pre_icu_los_days", "elective_surgery", "metastatic_cancer",
         "heme_malignancy", "aids"]
VITAL_ITEMS = {220045: "heart_rate", 220179: "sbp", 220050: "sbp", 220180: "dbp", 220051: "dbp",
               220181: "map", 220052: "map", 220210: "resp_rate", 223762: "temp_c", 223761: "temp_f",
               220277: "spo2"}
LAB_ITEMS = {50931: "glucose", 50912: "creatinine", 51006: "bun", 50983: "sodium", 50971: "potassium",
             50902: "chloride", 50882: "bicarbonate", 51222: "hemoglobin", 51221: "hematocrit", 51301: "wbc",
             51265: "platelets", 50813: "lactate", 50885: "bilirubin", 50862: "albumin"}
GCS_ITEMS = {220739: "eye", 223900: "verbal", 223901: "motor"}
VENT_ITEMS = {225792, 225794}
URINE_ITEMS = {226557, 226558, 226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584, 227489}


def cohort(root):
    stays = pd.read_csv(root / "icu" / "icustays.csv.gz",
                        usecols=["subject_id", "hadm_id", "stay_id", "intime", "los"], parse_dates=["intime"])
    adm = pd.read_csv(root / "hosp" / "admissions.csv.gz",
                      usecols=["hadm_id", "admittime", "admission_type", "hospital_expire_flag"],
                      parse_dates=["admittime"])
    pat = pd.read_csv(root / "hosp" / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    stays = stays.sort_values(["subject_id", "intime"]).groupby("subject_id", as_index=False).first()
    df = stays[stays["los"] >= 1.0].merge(adm, on="hadm_id").merge(pat, on="subject_id")
    df = df[df["anchor_age"] >= 18].rename(columns={"anchor_age": "age", "hospital_expire_flag": "label"})
    df["gender"] = (df["gender"] == "M").astype(int)
    df["pre_icu_los_days"] = ((df["intime"] - df["admittime"]).dt.total_seconds() / 86400.0).clip(lower=0)
    df["elective_surgery"] = df["admission_type"].isin(["ELECTIVE", "SURGICAL SAME DAY ADMISSION"]).astype(int)
    return df[["subject_id", "hadm_id", "stay_id", "intime", "age", "gender", "pre_icu_los_days",
               "elective_surgery", "label"]].reset_index(drop=True)


def events(path, key, start, lo_h, hi_h):
    """Chunks of an extracted event file restricted to [lo_h, hi_h] hours after ICU admission."""
    for chunk in pd.read_csv(path, chunksize=CHUNK, parse_dates=["charttime"],
                             dtype={key: "float64", "itemid": "int32", "valuenum": "float64"}):
        chunk = chunk[chunk[key].isin(start.keys())]
        hrs = (chunk["charttime"] - chunk[key].map(start)).dt.total_seconds() / 3600.0
        chunk = chunk[(hrs >= lo_h) & (hrs <= hi_h)]
        if not chunk.empty:
            yield chunk


def vitals(path, co):
    acc = None
    for c in events(path, "stay_id", dict(zip(co["stay_id"], co["intime"].to_numpy())), 0, 24):
        names = c["itemid"].map(VITAL_ITEMS)
        v = c["valuenum"].to_numpy(dtype=float).copy()
        f = (names == "temp_f").to_numpy()
        v[f] = (v[f] - 32.0) * 5.0 / 9.0
        names = names.replace({"temp_f": "temp_c"})
        ok = names.notna().to_numpy()
        acc = icu.fold(acc, c["stay_id"].to_numpy()[ok], names.to_numpy()[ok], v[ok])
    return icu.finalize(acc, "stay_id")


def labs(path, co):
    acc = None
    for c in events(path, "hadm_id", dict(zip(co["hadm_id"], co["intime"].to_numpy())), -6, 24):
        names = c["itemid"].map(LAB_ITEMS)
        ok = names.notna().to_numpy()
        acc = icu.fold(acc, c["hadm_id"].to_numpy()[ok], names.to_numpy()[ok],
                       c["valuenum"].to_numpy(dtype=float)[ok])
    return icu.finalize(acc, "hadm_id")


def gcs(path, co):
    """Lowest total GCS (eye + verbal + motor charted together) in the first 24 h."""
    ev = pd.concat(events(path, "stay_id", dict(zip(co["stay_id"], co["intime"].to_numpy())), 0, 24))
    ev["comp"] = ev["itemid"].map(GCS_ITEMS)
    wide = ev.groupby(["stay_id", "charttime", "comp"])["valuenum"].max().unstack("comp")
    total = wide.dropna(subset=["eye", "verbal", "motor"]).sum(axis=1)
    out = total.groupby(level="stay_id").min().rename("gcs").reset_index()
    out.loc[(out["gcs"] < 3) | (out["gcs"] > 15), "gcs"] = np.nan
    return out


def vent(root, co):
    pe = pd.read_csv(root / "icu" / "procedureevents.csv.gz", usecols=["stay_id", "starttime", "itemid"],
                     parse_dates=["starttime"])
    m = pe[pe["itemid"].isin(VENT_ITEMS)].merge(co[["stay_id", "intime"]], on="stay_id")
    hrs = (m["starttime"] - m["intime"]).dt.total_seconds() / 3600.0
    on = set(m.loc[(hrs >= -6) & (hrs <= 24), "stay_id"])
    return pd.DataFrame({"stay_id": co["stay_id"], "vent": co["stay_id"].isin(on).astype(int)})


def urine(root, co):
    oe = pd.read_csv(root / "icu" / "outputevents.csv.gz", usecols=["stay_id", "charttime", "itemid", "value"],
                     parse_dates=["charttime"])
    m = oe[oe["itemid"].isin(URINE_ITEMS)].merge(co[["stay_id", "intime"]], on="stay_id")
    hrs = (m["charttime"] - m["intime"]).dt.total_seconds() / 3600.0
    m = m[(hrs >= 0) & (hrs <= 24)]
    m["value"] = pd.to_numeric(m["value"], errors="coerce").clip(lower=0)
    out = m.groupby("stay_id")["value"].sum().rename("urine_out").reset_index()
    out.loc[out["urine_out"] > 30000, "urine_out"] = np.nan
    return out


def chronic(root, co):
    """Chronic conditions from discharge diagnosis codes (ICD-9 and ICD-10)."""
    dx = pd.read_csv(root / "hosp" / "diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"],
                     dtype={"icd_code": str})
    dx = dx[dx["hadm_id"].isin(set(co["hadm_id"]))]
    code, v9, v10 = dx["icd_code"].str.strip(), dx["icd_version"] == 9, dx["icd_version"] == 10
    return pd.DataFrame({
        "hadm_id": dx["hadm_id"],
        "metastatic_cancer": ((v9 & code.str.match(r"19[6-9]")) | (v10 & code.str.match(r"C7[7-9]"))).astype(int),
        "heme_malignancy": ((v9 & code.str.match(r"20[0-8]")) | (v10 & code.str.match(r"C(8[1-9]|9[0-6])"))).astype(int),
        "aids": ((v9 & code.str.match(r"042")) | (v10 & code.str.match(r"B2[0-4]"))).astype(int),
    }).groupby("hadm_id", as_index=False).max()


def main(root, work, out):
    root, work = Path(root), Path(work)
    co = cohort(root)
    df = (co.merge(vitals(work / "vitals_raw.csv", co), on="stay_id", how="left")
          .merge(labs(work / "labs_raw.csv", co), on="hadm_id", how="left")
          .merge(gcs(work / "gcs_raw.csv", co), on="stay_id", how="left")
          .merge(vent(root, co), on="stay_id", how="left")
          .merge(urine(root, co), on="stay_id", how="left")
          .merge(chronic(root, co), on="hadm_id", how="left"))
    for c in ("metastatic_cancer", "heme_malignancy", "aids"):
        df[c] = df[c].fillna(0).astype(int)
    cols = icu.feature_columns() + EXTRA
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = icu.clip_implausible(df)

    train, test = train_test_split(df, test_size=0.20, random_state=42, shuffle=True, stratify=df["label"])
    ids = package.opaque_ids(test["stay_id"].tolist(), "mi")
    test = test.assign(stay_id=test["stay_id"].map(ids)).sort_values("stay_id").reset_index(drop=True)
    train = train.reset_index(drop=True)[["stay_id"] + cols + ["label"]]
    train["stay_id"] = [f"tr{i:06d}" for i in range(len(train))]
    package.binary(out, "mimiciv", HERE / "mimiciv.md", train, test[["stay_id"] + cols + ["label"]], "stay_id")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="MIMIC-IV 3.1 directory with hosp/ and icu/")
    ap.add_argument("work", help="output directory of extract.sh")
    ap.add_argument("--out", default="packages")
    main(**vars(ap.parse_args()))
