# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH U.S. Empirical Upgrade Pipeline v2
Fixes empty-analytic-dataset failure in v1.

Main fix:
- Use the FINAL OxCGRT subnational dataset:
  OxCGRT_compact_subnational_v1.csv
- Keep only U.S. state rows.
- Fail early with diagnostics if signals / policies / episodes are empty.
- Never allow an empty DataFrame to reach analytic["signal_date"].

Study design remains:
- Unit: U.S. state x surge episode
- Study period: 2020-07-15 to 2022-12-31
- Signal: 7-day admissions >= 50% above 28-day rolling median
- Response escalation: >= 5-point rise in OxCGRT StringencyIndex_Average within 21 days
- Exposure: signal-to-response latency in days
- Primary outcome: subsequent 28-day excess deaths
"""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ==============================================================
# 0. PATHS
# ==============================================================

ROOT = Path(r"C:\Users\SIMONEY\Disease\AJPH_US_Empirical_Upgrade")
RAW = ROOT / "01_raw"
CLEAN = ROOT / "02_clean"
DERIVED = ROOT / "03_derived"
OUTPUT = ROOT / "04_outputs"
DOC = ROOT / "05_documentation"

for p in [RAW, CLEAN, DERIVED, OUTPUT, DOC]:
    p.mkdir(parents=True, exist_ok=True)

# ==============================================================
# 1. DATA URLS
# ==============================================================

HHS_URL = "https://healthdata.gov/resource/g62h-syeh.csv?$limit=100000"

CDC_EXCESS_URL = "https://data.cdc.gov/resource/xkkf-xrst.csv?$limit=50000"

# IMPORTANT FIX:
# final OxCGRT subnational file, not national file.
OXCGRT_URL = (
    "https://raw.githubusercontent.com/OxCGRT/"
    "covid-policy-dataset/main/data/"
    "OxCGRT_compact_subnational_v1.csv"
)

# ==============================================================
# 2. CONSTANTS / HELPERS
# ==============================================================

US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}

STATE_NAME_TO_ABBR = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District of Columbia":"DC",
    "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
    "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
    "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI",
    "Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT",
    "Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ",
    "New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND",
    "Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA",
    "Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN",
    "Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA",
    "West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
}

def download_csv(url: str, timeout: int = 180) -> pd.DataFrame:
    print("Downloading:")
    print(url)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)

def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def save_inventory(df, filename):
    pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }).to_csv(DOC / filename, index=False)

def stop_if_empty(df, name):
    if df is None or df.empty:
        raise RuntimeError(
            f"{name} is EMPTY. Pipeline stopped here deliberately. "
            f"Check the diagnostic CSVs in {DOC}."
        )

# ==============================================================
# 3. HHS HOSPITAL DATA
# ==============================================================

print("\n[1/7] HHS HOSPITAL DATA")
hhs = download_csv(HHS_URL)
hhs.to_csv(RAW / "hhs_state_hospital_raw.csv", index=False)
save_inventory(hhs, "hhs_columns.csv")

if "state" not in hhs.columns or "date" not in hhs.columns:
    raise ValueError("HHS schema changed: 'state' or 'date' is missing.")

hhs["state"] = hhs["state"].astype(str).str.upper().str.strip()
hhs["date"] = pd.to_datetime(hhs["date"], errors="coerce")

hhs = hhs[hhs["state"].isin(US_STATES)].copy()
hhs = hhs[
    (hhs["date"] >= "2020-07-15") &
    (hhs["date"] <= "2022-12-31")
].copy()

admit_col = first_existing(
    hhs.columns,
    ["previous_day_admission_adult_covid_confirmed"]
)
inpat_col = first_existing(
    hhs.columns,
    ["inpatient_beds_used_covid"]
)
icu_used_col = first_existing(
    hhs.columns,
    ["staffed_icu_adult_patients_confirmed_covid"]
)
icu_total_col = first_existing(
    hhs.columns,
    ["total_staffed_adult_icu_beds"]
)
staff_col = first_existing(
    hhs.columns,
    ["critical_staffing_shortage_today_yes"]
)

if admit_col is None:
    raise ValueError("Could not identify HHS adult COVID admissions column.")

keep = ["state", "date"] + [
    c for c in [admit_col, inpat_col, icu_used_col, icu_total_col, staff_col]
    if c is not None
]

hhs2 = hhs[keep].copy()

rename = {admit_col: "covid_admissions"}
if inpat_col:
    rename[inpat_col] = "covid_inpatients"
if icu_used_col:
    rename[icu_used_col] = "covid_icu_patients"
if icu_total_col:
    rename[icu_total_col] = "icu_beds_total"
if staff_col:
    rename[staff_col] = "hospitals_staffing_shortage"

hhs2 = hhs2.rename(columns=rename)

for c in hhs2.columns:
    if c not in ["state", "date"]:
        hhs2[c] = pd.to_numeric(hhs2[c], errors="coerce")

if {"covid_icu_patients", "icu_beds_total"}.issubset(hhs2.columns):
    hhs2["icu_covid_share"] = (
        hhs2["covid_icu_patients"] /
        hhs2["icu_beds_total"].replace(0, np.nan)
    )

hhs2 = hhs2.sort_values(["state", "date"]).reset_index(drop=True)

hhs2["admissions_7dma"] = (
    hhs2.groupby("state")["covid_admissions"]
        .transform(lambda s: s.rolling(7, min_periods=4).mean())
)

hhs2["admissions_baseline_28d"] = (
    hhs2.groupby("state")["admissions_7dma"]
        .transform(lambda s: s.shift(1).rolling(28, min_periods=14).median())
)

hhs2["admissions_rel_change"] = (
    hhs2["admissions_7dma"] /
    hhs2["admissions_baseline_28d"].replace(0, np.nan)
    - 1
)

stop_if_empty(hhs2, "HHS cleaned panel")

print("HHS rows:", len(hhs2))
print("HHS states:", hhs2["state"].nunique())
print("HHS date range:", hhs2["date"].min(), "to", hhs2["date"].max())

hhs2.to_csv(CLEAN / "hhs_state_daily_clean.csv", index=False)

# ==============================================================
# 4. OXCGRT U.S. STATE POLICY DATA
# ==============================================================

print("\n[2/7] OXCGRT U.S. STATE POLICY DATA")

ox = download_csv(OXCGRT_URL)
ox.to_csv(RAW / "oxcgrt_subnational_raw.csv", index=False)
save_inventory(ox, "oxcgrt_columns.csv")

print("Raw OxCGRT rows:", len(ox))
print("OxCGRT columns:", list(ox.columns[:20]))

country_code_col = first_existing(
    ox.columns,
    ["CountryCode", "country_code"]
)
region_name_col = first_existing(
    ox.columns,
    ["RegionName", "region_name"]
)
region_code_col = first_existing(
    ox.columns,
    ["RegionCode", "region_code"]
)
date_col = first_existing(
    ox.columns,
    ["Date", "date"]
)
stringency_col = first_existing(
    ox.columns,
    [
        "StringencyIndex_Average",
        "StringencyIndex",
        "stringency_index",
    ]
)

if country_code_col is None:
    raise ValueError("OxCGRT CountryCode column not found.")
if date_col is None:
    raise ValueError("OxCGRT Date column not found.")
if region_name_col is None and region_code_col is None:
    raise ValueError("OxCGRT RegionName/RegionCode columns not found.")

# United States only.
ox = ox[
    ox[country_code_col].astype(str).str.upper().eq("USA")
].copy()

print("OxCGRT USA rows:", len(ox))

# Convert date robustly.
date_as_str = ox[date_col].astype(str).str.replace(r"\.0$", "", regex=True)

ox["date"] = pd.to_datetime(
    date_as_str,
    format="%Y%m%d",
    errors="coerce"
)

# State mapping.
if region_name_col is not None:
    ox["state"] = ox[region_name_col].map(STATE_NAME_TO_ABBR)
else:
    ox["state"] = (
        ox[region_code_col]
        .astype(str)
        .str.extract(r"([A-Z]{2})$")[0]
    )

# Save mapping diagnostic before filtering.
mapping_diag = (
    ox[[c for c in [region_name_col, region_code_col] if c is not None] + ["state"]]
    .drop_duplicates()
    .sort_values("state", na_position="last")
)
mapping_diag.to_csv(DOC / "oxcgrt_US_state_mapping.csv", index=False)

ox = ox[ox["state"].isin(US_STATES)].copy()

print("OxCGRT mapped state rows:", len(ox))
print("OxCGRT mapped states:", ox["state"].nunique())
print("States present:", sorted(ox["state"].dropna().unique().tolist()))

ox = ox[
    (ox["date"] >= "2020-07-15") &
    (ox["date"] <= "2022-12-31")
].copy()

stop_if_empty(ox, "OxCGRT U.S. state panel")

if stringency_col is None:
    raise ValueError(
        "No StringencyIndex field found in final OxCGRT compact subnational file. "
        "Check oxcgrt_columns.csv."
    )

ox["response_index"] = pd.to_numeric(
    ox[stringency_col],
    errors="coerce"
)

ox2 = (
    ox[["state", "date", "response_index"]]
    .dropna(subset=["state", "date"])
    .sort_values(["state", "date"])
    .drop_duplicates(["state", "date"], keep="last")
    .reset_index(drop=True)
)

print("OxCGRT clean rows:", len(ox2))
print("OxCGRT clean states:", ox2["state"].nunique())
print("OxCGRT date range:", ox2["date"].min(), "to", ox2["date"].max())
print("Response index nonmissing:", ox2["response_index"].notna().mean())

stop_if_empty(ox2, "OxCGRT cleaned panel")

ox2.to_csv(CLEAN / "oxcgrt_state_daily_clean.csv", index=False)

# ==============================================================
# 5. CDC EXCESS DEATHS
# ==============================================================

print("\n[3/7] CDC EXCESS DEATHS")

cdc = download_csv(CDC_EXCESS_URL)
cdc.to_csv(RAW / "cdc_excess_deaths_raw.csv", index=False)
save_inventory(cdc, "cdc_excess_columns.csv")

date_cdc = first_existing(
    cdc.columns,
    ["week_ending_date", "weekendingdate"]
)
state_cdc = first_existing(
    cdc.columns,
    ["state", "jurisdiction"]
)
obs_col = first_existing(
    cdc.columns,
    ["observed_number", "observed_deaths", "observed"]
)
exp_col = first_existing(
    cdc.columns,
    ["average_expected_count", "expected_count", "expected"]
)
excess_col = first_existing(
    cdc.columns,
    ["excess_estimate", "excess_deaths"]
)

if date_cdc is None or state_cdc is None:
    raise ValueError("CDC state/date fields not found.")

cdc["week_ending_date"] = pd.to_datetime(
    cdc[date_cdc],
    errors="coerce"
)

raw_state = cdc[state_cdc].astype(str)
cdc["state"] = raw_state.map(STATE_NAME_TO_ABBR).fillna(raw_state.str.upper())

cdc = cdc[cdc["state"].isin(US_STATES)].copy()
cdc = cdc[
    (cdc["week_ending_date"] >= "2020-07-15") &
    (cdc["week_ending_date"] <= "2022-12-31")
].copy()

if excess_col:
    cdc["excess_deaths"] = pd.to_numeric(
        cdc[excess_col],
        errors="coerce"
    )
elif obs_col and exp_col:
    cdc["excess_deaths"] = (
        pd.to_numeric(cdc[obs_col], errors="coerce") -
        pd.to_numeric(cdc[exp_col], errors="coerce")
    )
else:
    raise ValueError("CDC excess-death fields not found.")

cdc2 = (
    cdc.dropna(subset=["excess_deaths"])
       .groupby(["state", "week_ending_date"], as_index=False)
       .agg(excess_deaths=("excess_deaths", "mean"))
)

stop_if_empty(cdc2, "CDC weekly excess-death panel")

print("CDC rows:", len(cdc2))
print("CDC states:", cdc2["state"].nunique())

cdc2.to_csv(
    CLEAN / "cdc_state_weekly_excess_deaths_clean.csv",
    index=False
)

# ==============================================================
# 6. SURGE DETECTION
# ==============================================================

print("\n[4/7] SURGE DETECTION")

REL_THRESHOLD = 0.50
MIN_ADMISSIONS = 25
MIN_GAP_DAYS = 42

signals = []

for state, g in hhs2.groupby("state"):
    g = g.sort_values("date").copy()

    cand = g[
        (g["admissions_rel_change"] >= REL_THRESHOLD) &
        (g["admissions_7dma"] >= MIN_ADMISSIONS)
    ].copy()

    last = None
    episode_id = 0

    for _, row in cand.iterrows():
        d = pd.Timestamp(row["date"])

        if last is None or (d - last).days >= MIN_GAP_DAYS:
            episode_id += 1

            signals.append({
                "state": state,
                "episode_id": episode_id,
                "signal_date": d,
                "signal_admissions_7dma": row["admissions_7dma"],
                "signal_rel_change": row["admissions_rel_change"],
            })

            last = d

signals = pd.DataFrame(signals)

print("Detected signal episodes:", len(signals))
if not signals.empty:
    print("States with signals:", signals["state"].nunique())
    print(signals.head(10))

signals.to_csv(
    DERIVED / "state_surge_signals.csv",
    index=False
)

if signals.empty:
    raise RuntimeError(
        "ZERO SURGE SIGNALS detected. "
        "The previous KeyError would also occur in this situation. "
        "Inspect hhs_state_daily_clean.csv and lower MIN_ADMISSIONS "
        "or REL_THRESHOLD if necessary."
    )

# ==============================================================
# 7. RESPONSE ESCALATION + LATENCY
# ==============================================================

print("\n[5/7] RESPONSE ESCALATION / LATENCY")

SEARCH_DAYS = 21
ESCALATION_POINTS = 5.0

episodes = []
missing_policy_states = []

for _, ep in signals.iterrows():

    state = ep["state"]
    signal_date = pd.Timestamp(ep["signal_date"])

    state_policy = ox2[
        ox2["state"].eq(state)
    ].copy()

    if state_policy.empty:
        missing_policy_states.append(state)
        continue

    baseline_rows = state_policy[
        state_policy["date"] <= signal_date
    ].dropna(subset=["response_index"]).sort_values("date")

    future = state_policy[
        (state_policy["date"] >= signal_date) &
        (state_policy["date"] <= signal_date + pd.Timedelta(days=SEARCH_DAYS))
    ].dropna(subset=["response_index"]).sort_values("date")

    if baseline_rows.empty or future.empty:
        continue

    baseline_index = float(
        baseline_rows.iloc[-1]["response_index"]
    )

    future = future.copy()
    future["delta_from_signal"] = (
        future["response_index"] - baseline_index
    )

    escalated = future[
        future["delta_from_signal"] >= ESCALATION_POINTS
    ]

    if not escalated.empty:
        response_date = pd.Timestamp(
            escalated.iloc[0]["date"]
        )
        latency_days = int(
            (response_date - signal_date).days
        )
        response_detected = 1
    else:
        response_date = pd.NaT

        # Keep censoring explicit.
        latency_days = SEARCH_DAYS + 1
        response_detected = 0

    row = ep.to_dict()

    row.update({
        "baseline_response_index": baseline_index,
        "response_date": response_date,
        "response_detected_21d": response_detected,
        "latency_days": latency_days,
    })

    episodes.append(row)

episodes = pd.DataFrame(episodes)

print("Latency episodes:", len(episodes))

if missing_policy_states:
    print(
        "States without OxCGRT policy rows:",
        sorted(set(missing_policy_states))
    )

if not episodes.empty:
    print("Latency states:", episodes["state"].nunique())
    print("Response detected fraction:",
          episodes["response_detected_21d"].mean())
    print("Latency distribution:")
    print(episodes["latency_days"].describe())

episodes.to_csv(
    DERIVED / "state_surge_latency.csv",
    index=False
)

if episodes.empty:
    raise RuntimeError(
        "SURGE SIGNALS WERE FOUND, BUT ZERO SIGNALS COULD BE MATCHED "
        "TO OxCGRT STATE POLICY DATA.\n"
        "This is almost certainly the cause of your original "
        "KeyError: 'signal_date'.\n"
        "Open these diagnostics:\n"
        f"  {DOC / 'oxcgrt_US_state_mapping.csv'}\n"
        f"  {CLEAN / 'oxcgrt_state_daily_clean.csv'}\n"
        f"  {DERIVED / 'state_surge_signals.csv'}"
    )

# ==============================================================
# 8. 28-DAY OUTCOMES
# ==============================================================

print("\n[6/7] OUTCOME CONSTRUCTION")

analytic_rows = []

for _, ep in episodes.iterrows():

    state = ep["state"]
    signal_date = pd.Timestamp(ep["signal_date"])

    mortality_window = cdc2[
        (cdc2["state"] == state) &
        (
            cdc2["week_ending_date"] >=
            signal_date + pd.Timedelta(days=7)
        ) &
        (
            cdc2["week_ending_date"] <=
            signal_date + pd.Timedelta(days=35)
        )
    ].copy()

    hospital_window = hhs2[
        (hhs2["state"] == state) &
        (hhs2["date"] > signal_date) &
        (
            hhs2["date"] <=
            signal_date + pd.Timedelta(days=28)
        )
    ].copy()

    row = ep.to_dict()

    row["excess_deaths_next_28d"] = (
        mortality_window["excess_deaths"].sum(min_count=1)
    )

    row["covid_admissions_next_28d"] = (
        hospital_window["covid_admissions"].sum(min_count=1)
    )

    if "icu_covid_share" in hospital_window.columns:
        row["max_icu_covid_share_next_28d"] = (
            hospital_window["icu_covid_share"].max()
        )

    if "hospitals_staffing_shortage" in hospital_window.columns:
        row["mean_staffing_shortage_next_28d"] = (
            hospital_window["hospitals_staffing_shortage"].mean()
        )

    analytic_rows.append(row)

analytic = pd.DataFrame(analytic_rows)

# Fail clearly BEFORE trying analytic["signal_date"].
if analytic.empty or "signal_date" not in analytic.columns:
    raise RuntimeError(
        "Analytic dataset is empty after outcome construction. "
        "This is now trapped explicitly instead of generating KeyError."
    )

analytic["signal_date"] = pd.to_datetime(
    analytic["signal_date"],
    errors="coerce"
)

analytic["signal_month"] = (
    analytic["signal_date"]
    .dt.to_period("M")
    .astype(str)
)

analytic["signal_year"] = (
    analytic["signal_date"].dt.year
)

analytic.to_csv(
    OUTPUT / "US_state_surge_analytic_dataset.csv",
    index=False
)

print("Analytic rows:", len(analytic))
print("Analytic states:", analytic["state"].nunique())
print("Nonmissing excess-death outcome:",
      analytic["excess_deaths_next_28d"].notna().sum())

# ==============================================================
# 9. SUMMARY
# ==============================================================

print("\n[7/7] PIPELINE SUMMARY")

summary = {
    "hhs_states": int(hhs2["state"].nunique()),
    "oxcgrt_states": int(ox2["state"].nunique()),
    "cdc_states": int(cdc2["state"].nunique()),
    "surge_episodes": int(len(signals)),
    "latency_episodes": int(len(episodes)),
    "analytic_episodes": int(len(analytic)),
    "analytic_states": int(analytic["state"].nunique()),
    "median_latency_days": float(
        episodes["latency_days"].median()
    ),
    "p25_latency_days": float(
        episodes["latency_days"].quantile(0.25)
    ),
    "p75_latency_days": float(
        episodes["latency_days"].quantile(0.75)
    ),
    "response_detected_fraction": float(
        episodes["response_detected_21d"].mean()
    ),
}

(OUTPUT / "pipeline_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print(json.dumps(summary, indent=2))

print("\nCOMPLETE")
print(
    "Primary analytic dataset:\n",
    OUTPUT / "US_state_surge_analytic_dataset.csv"
)
