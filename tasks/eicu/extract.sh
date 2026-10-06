#!/bin/bash
# Pull the first-day vitals and labs used by build.py out of eICU-CRD.
#   usage: extract.sh EICU_DIR WORK_DIR
set -eu
DIR="$1"; OUT="$2"
mkdir -p "$OUT"
# vitalPeriodic: vitalperiodicid,patientunitstayid,observationoffset,temperature,sao2,heartrate,respiration,
#                cvp,etco2,systemicsystolic,systemicdiastolic,systemicmean,...   (first 24 h)
gzip -dc "$DIR/vitalPeriodic.csv.gz" | awk -F',' '
  BEGIN { print "patientunitstayid,offset,temperature,sao2,heartrate,respiration,sbp,dbp,map" }
  NR > 1 && $3 >= 0 && $3 <= 1440 { print $2","$3","$4","$5","$6","$7","$10","$11","$12 }' > "$OUT/vitals_raw.csv"
# lab: labid,patientunitstayid,labresultoffset,labtypeid,labname,labresult,...   (1 h before to 24 h after admission)
gzip -dc "$DIR/lab.csv.gz" | awk -F',' '
  BEGIN { n = split("glucose|bedside glucose|creatinine|BUN|sodium|potassium|chloride|bicarbonate|Hgb|Hct|WBC x 1000|platelets x 1000|lactate|total bilirubin|albumin", a, "|")
          for (i = 1; i <= n; i++) keep[a[i]] = 1; print "patientunitstayid,offset,labname,labresult" }
  NR > 1 && ($5 in keep) && $3 >= -60 && $3 <= 1440 && $6 != "" { print $2","$3",\""$5"\","$6 }' > "$OUT/labs_raw.csv"
wc -l "$OUT"/*.csv
