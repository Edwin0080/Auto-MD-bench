"""Rotterdam and GBSG breast-cancer cohorts, recurrence-free survival (R `survival` package).

    rotterdam        Rotterdam, 80/20 split
    rotterdam_gbsg   train on all of Rotterdam, test on GBSG (external validation)

    python tasks/rotterdam/build.py RAW_DIR [--out packages]    RAW_DIR holds rotterdam.csv, gbsg.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from automdbench import package

HERE = Path(__file__).resolve().parent


def ids(prefix, n, seed):
    # random ids: the source row order is related to the outcome
    return [f"{prefix}{k:04d}" for k in np.random.default_rng(seed).permutation(n)]


def frame(df, sizecat, duration, event, prefix, seed):
    out = pd.DataFrame({"age": df["age"], "meno": df["meno"], "sizecat": sizecat, "grade": df["grade"],
                        "nodes": df["nodes"], "pgr": df["pgr"], "er": df["er"], "hormon": df["hormon"],
                        "duration": duration.astype(float), "event": event.astype(int)})
    out.insert(0, "patient_id", ids(prefix, len(out), seed))
    return out


def main(raw, out):
    r = pd.read_csv(Path(raw) / "rotterdam.csv")
    size = r["size"].astype(str)
    rot = frame(r, np.where(size.str.contains("<=20"), 1, np.where(size.str.contains(">50"), 3, 2)),
                np.minimum(r["rtime"], r["dtime"]), (r["recur"] == 1) | (r["death"] == 1), "r", 7)
    g = pd.read_csv(Path(raw) / "gbsg.csv")
    gbsg = frame(g, np.where(g["size"] <= 20, 1, np.where(g["size"] > 50, 3, 2)),
                 g["rfstime"], g["status"], "g", 11)

    train, test = train_test_split(rot, test_size=0.20, random_state=42, shuffle=True, stratify=rot["event"])
    package.survival(out, "rotterdam", HERE / "rotterdam.md",
                     train.sort_values("patient_id").reset_index(drop=True),
                     test.sort_values("patient_id").reset_index(drop=True))
    package.survival(out, "rotterdam_gbsg", HERE / "rotterdam_gbsg.md",
                     rot, gbsg.sort_values("patient_id").reset_index(drop=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", default="packages")
    main(**vars(ap.parse_args()))
