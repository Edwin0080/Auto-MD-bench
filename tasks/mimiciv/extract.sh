#!/bin/bash
# Pull the chart and lab items used by build.py out of MIMIC-IV's two large event tables.
#   usage: extract.sh MIMIC_ROOT WORK_DIR
set -eu
ROOT="$1"; OUT="$2"
mkdir -p "$OUT"
# chartevents: subject_id,hadm_id,stay_id,caregiver_id,charttime,storetime,itemid,value,valuenum,...
# vitals: HR, SBP, DBP, MAP (non-invasive and arterial), RR, temperature C/F, SpO2; GCS eye/verbal/motor
gzip -dc "$ROOT/icu/chartevents.csv.gz" | awk -F',' -v V="$OUT/vitals_raw.csv" -v G="$OUT/gcs_raw.csv" '
  BEGIN { split("220045 220179 220050 220180 220051 220181 220052 220210 223762 223761 220277", a, " ")
          for (i in a) vit[a[i]] = 1
          split("220739 223900 223901", b, " "); for (i in b) gcs[b[i]] = 1
          print "stay_id,charttime,itemid,valuenum" > V; print "stay_id,charttime,itemid,valuenum" > G }
  NR > 1 && $9 != "" { if ($7 in vit) print $3","$5","$7","$9 > V; else if ($7 in gcs) print $3","$5","$7","$9 > G }'
# labevents: labevent_id,subject_id,hadm_id,specimen_id,itemid,order_provider_id,charttime,storetime,value,valuenum,...
gzip -dc "$ROOT/hosp/labevents.csv.gz" | awk -F',' '
  BEGIN { split("50931 50912 51006 50983 50971 50902 50882 51222 51221 51301 51265 50813 50885 50862", a, " ")
          for (i in a) keep[a[i]] = 1; print "hadm_id,charttime,itemid,valuenum" }
  NR > 1 && ($5 in keep) && $10 != "" { print $3","$7","$5","$10 }' > "$OUT/labs_raw.csv"
wc -l "$OUT"/*.csv
