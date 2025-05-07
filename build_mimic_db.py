import sqlite3
import pandas as pd
from google.cloud import bigquery
from tqdm import tqdm
import os

PROJECT = "physionet-data"
DATASET = "mimiciii_clinical"
client = bigquery.Client(project="linear-freehold-354813")
con = sqlite3.connect("mimic.db")

def run_query(sql):
    return client.query(sql).to_dataframe()

# 1. Build DEMOGRAPHIC
print("Building DEMOGRAPHIC...")
demo_sql = f"""
SELECT
    adm.hadm_id AS HADM_ID,
    pat.subject_id AS SUBJECT_ID,
    'John Doe' AS NAME,
    'Married' AS MARITAL_STATUS,
    ROUND(DATE_DIFF(adm.admittime, pat.dob, YEAR), 1) AS AGE,
    pat.dob AS DOB,
    pat.gender AS GENDER,
    'English' AS LANGUAGE,
    'Catholic' AS RELIGION,
    adm.admission_type AS ADMISSION_TYPE,
    ROUND(DATE_DIFF(adm.dischtime, adm.admittime, DAY), 1) AS DAYS_STAY,
    adm.insurance AS INSURANCE,
    adm.ethnicity AS ETHNICITY,
    CASE WHEN pat.dod IS NULL THEN 0 ELSE 1 END AS EXPIRE_FLAG,
    adm.admission_location AS ADMISSION_LOCATION,
    adm.discharge_location AS DISCHARGE_LOCATION,
    'Heart Failure' AS DIAGNOSIS,
    pat.dod AS DOD,
    EXTRACT(YEAR FROM pat.dob) AS DOB_YEAR,
    EXTRACT(YEAR FROM pat.dod) AS DOD_YEAR,
    adm.admittime AS ADMITTIME,
    adm.dischtime AS DISCHTIME,
    EXTRACT(YEAR FROM adm.admittime) AS ADMITYEAR
FROM `{PROJECT}.{DATASET}.admissions` adm
JOIN `{PROJECT}.{DATASET}.patients` pat USING (subject_id)
LIMIT 5000
"""

df = run_query(demo_sql)
df.to_sql("Demographic", con, if_exists="replace", index=False)

# 2. Diagnoses
print("Building DIAGNOSES...")
dx_sql = f"""
SELECT
    subject_id AS SUBJECT_ID,
    hadm_id AS HADM_ID,
    icd9_code AS ICD9_CODE,
    d.short_title AS SHORT_TITLE,
    d.long_title AS LONG_TITLE
FROM `{PROJECT}.{DATASET}.diagnoses_icd` dx
LEFT JOIN `{PROJECT}.{DATASET}.d_icd_diagnoses` d
USING (icd9_code)
LIMIT 5000
"""

df = run_query(dx_sql)
df.to_sql("Diagnoses", con, if_exists="replace", index=False)

# 3. Procedures
print("Building PROCEDURES...")
px_sql = f"""
SELECT
    subject_id AS SUBJECT_ID,
    hadm_id AS HADM_ID,
    icd9_code AS ICD9_CODE,
    p.short_title AS SHORT_TITLE,
    p.long_title AS LONG_TITLE
FROM `{PROJECT}.{DATASET}.procedures_icd` px
LEFT JOIN `{PROJECT}.{DATASET}.d_icd_procedures` p
USING (icd9_code)
LIMIT 5000
"""

df = run_query(px_sql)
df.to_sql("Procedures", con, if_exists="replace", index=False)

# 4. Prescriptions
print("Building PRESCRIPTIONS...")
presc_sql = """
SELECT
  subject_id AS SUBJECT_ID,
  hadm_id AS HADM_ID,
  icustay_id AS ICUSTAY_ID,
  drug_type AS DRUG_TYPE,
  drug AS DRUG,
  formulary_drug_cd AS FORMULARY_DRUG_CD,
  route AS ROUTE,
  dose_val_rx AS DRUG_DOSE
FROM `physionet-data.mimiciii_clinical.prescriptions`
WHERE drug IS NOT NULL
LIMIT 5000
"""

presc_df = run_query(presc_sql)
presc_df.to_sql("PRESCRIPTIONS", con, if_exists="replace", index=False)


# 5. Lab
print("Building LAB...")
lab_sql = """
SELECT
  l.subject_id AS SUBJECT_ID,
  l.hadm_id AS HADM_ID,
  l.itemid AS ITEMID,
  l.charttime AS CHARTTIME,
  l.flag AS FLAG,
  l.valuenum AS VALUE_UNIT,
  d.label AS LABEL,
  d.fluid AS FLUID,
  d.category AS CATEGORY
FROM `physionet-data.mimiciii_clinical.labevents` l
LEFT JOIN `physionet-data.mimiciii_clinical.d_labitems` d
ON l.itemid = d.itemid
WHERE l.valuenum IS NOT NULL
LIMIT 5000
"""

lab_df = run_query(lab_sql)
lab_df.to_sql("LAB", con, if_exists="replace", index=False)

con.execute("VACUUM")
con.close()
print("✓ mimic.db built with 5 standard MIMICSQL tables")
