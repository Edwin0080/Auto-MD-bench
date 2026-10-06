"""First-24-hour ICU features shared by the MIMIC-IV and eICU builds."""
import numpy as np
import pandas as pd

VITALS = ["heart_rate", "sbp", "dbp", "map", "resp_rate", "temp_c", "spo2"]
LABS = ["glucose", "creatinine", "bun", "sodium", "potassium", "chloride", "bicarbonate", "hemoglobin",
        "hematocrit", "wbc", "platelets", "lactate", "bilirubin", "albumin"]
AGGS = ["mean", "min", "max"]

# values outside these ranges are documentation artefacts and are dropped
PLAUSIBLE = {
    "heart_rate": (10, 300), "sbp": (20, 300), "dbp": (5, 225), "map": (10, 250), "resp_rate": (0, 90),
    "temp_c": (25, 45), "spo2": (30, 100), "glucose": (5, 2000), "creatinine": (0.01, 50), "bun": (1, 300),
    "sodium": (80, 200), "potassium": (0.5, 15), "chloride": (50, 200), "bicarbonate": (2, 60),
    "hemoglobin": (1, 30), "hematocrit": (5, 80), "wbc": (0.05, 300), "platelets": (1, 2000),
    "lactate": (0.1, 40), "bilirubin": (0.05, 80), "albumin": (0.5, 8),
}


def feature_columns():
    return ["age", "gender"] + [f"{v}_{a}" for v in VITALS + LABS for a in AGGS]


def clip_implausible(df):
    for base, (lo, hi) in PLAUSIBLE.items():
        for a in AGGS:
            c = f"{base}_{a}"
            if c in df.columns:
                df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    return df


def fold(acc, ids, feats, vals):
    """Add one chunk of (id, feature, value) rows to running sum / count / min / max."""
    g = (pd.DataFrame({"id": ids, "feat": feats, "v": vals}).dropna()
         .groupby(["id", "feat"])["v"].agg(["sum", "count", "min", "max"]))
    if acc is None:
        return g
    j = acc.join(g, how="outer", rsuffix="_n").fillna({"sum": 0, "count": 0, "sum_n": 0, "count_n": 0})
    return pd.DataFrame({"sum": j["sum"] + j["sum_n"], "count": j["count"] + j["count_n"],
                         "min": j[["min", "min_n"]].min(axis=1), "max": j[["max", "max_n"]].max(axis=1)})


def finalize(acc, id_name):
    """Running aggregates -> one row per id with <feature>_mean / _min / _max columns."""
    acc = acc.reset_index()
    acc["mean"] = acc["sum"] / acc["count"].replace(0, np.nan)
    wide = acc.pivot(index="id", columns="feat", values=["mean", "min", "max"])
    wide.columns = [f"{feat}_{stat}" for stat, feat in wide.columns]
    return wide.reset_index().rename(columns={"id": id_name})
