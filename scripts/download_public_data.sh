#!/bin/bash
# Download the raw data of the 10 public tasks into DATA_DIR (default data/raw).
#   usage: bash scripts/download_public_data.sh [DATA_DIR]
set -eu
D="${1:-data/raw}"
mkdir -p "$D"/{support,metabric,rotterdam,uci_diabetes,physionet2012,sepsis,nhanes}

# SUPPORT and METABRIC: the copies distributed with pycox
python - "$D" <<'PY'
import sys
from pycox import datasets
for name in ("support", "metabric"):
    getattr(datasets, name).read_df().reset_index(drop=True).to_feather(f"{sys.argv[1]}/{name}/{name}.feather")
PY

# Rotterdam and GBSG: R package `survival`, via Rdatasets
for f in rotterdam gbsg; do
  curl -fsSL -o "$D/rotterdam/$f.csv" "https://vincentarelbundock.github.io/Rdatasets/csv/survival/$f.csv"
done

# UCI Diabetes 130-US hospitals (UCI ML Repository id 296)
curl -fsSL -o "$D/uci_diabetes/uci.zip" "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
unzip -o -q "$D/uci_diabetes/uci.zip" -d "$D/uci_diabetes"
find "$D/uci_diabetes" -name "*.zip" ! -name uci.zip -exec unzip -o -q {} -d "$D/uci_diabetes" \;
find "$D/uci_diabetes" -mindepth 2 -name diabetic_data.csv -exec mv {} "$D/uci_diabetes/" \;

# PhysioNet/CinC Challenge 2012: sets A, B, C and outcomes
for s in a b c; do
  curl -fsSL "https://physionet.org/files/challenge-2012/1.0.0/set-$s.tar.gz" | tar -xz -C "$D/physionet2012"
  curl -fsSL -o "$D/physionet2012/Outcomes-$s.txt" "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-$s.txt"
done

# PhysioNet/CinC Challenge 2019: training sets A and B (40,336 .psv files), from PhysioNet's S3 mirror
python - "$D/sepsis" <<'PY'
import re, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S3, PREFIX = "https://physionet-open.s3.amazonaws.com", "challenge-2019/1.0.0/training/"
keys, token = [], None
while True:
    query = {"list-type": 2, "prefix": PREFIX, **({"continuation-token": token} if token else {})}
    page = urllib.request.urlopen(f"{S3}/?{urllib.parse.urlencode(query)}").read().decode()
    keys += [k for k in re.findall(r"<Key>([^<]+)</Key>", page) if k.endswith(".psv")]
    token = (re.findall(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", page) or [None])[0]
    if not token:
        break


def get(key):  # files already present are skipped, so an interrupted download can be rerun
    dest = Path(sys.argv[1]) / key[len(PREFIX):]
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{S3}/{key}", dest.with_suffix(".part"))
    dest.with_suffix(".part").replace(dest)


with ThreadPoolExecutor(16) as pool:
    list(pool.map(get, keys))
print(f"sepsis: {len(keys)} files")
PY

# NHANES 2011-2016 (cycles G, H, I) and the NCHS public-use linked mortality files
declare -A CYCLES=([G]=2011 [H]=2013 [I]=2015)
for L in G H I; do
  Y=${CYCLES[$L]}
  for F in DEMO BMX BPX TCHOL HDL GHB CBC BIOPRO SMQ; do
    curl -fsSL -o "$D/nhanes/${F}_$L.xpt" "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/$Y/DataFiles/${F}_$L.xpt"
  done
  curl -fsSL -o "$D/nhanes/NHANES_${Y}_$((Y + 1))_MORT_2019_PUBLIC.dat" \
    "https://ftp.cdc.gov/pub/HEALTH_STATISTICS/NCHS/datalinkage/linked_mortality/NHANES_${Y}_$((Y + 1))_MORT_2019_PUBLIC.dat"
done
echo "done: $D"
