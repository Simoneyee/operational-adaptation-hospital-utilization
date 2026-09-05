# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH v17d
Substate HSA Mortality Bridge — SEER Population + CDC FIPS Fix + Coverage Audit

This version fixes BOTH failures seen in v17b/v17c:

1) Census access failures:
   - No Census API.
   - No www2.census.gov.
   - County population is taken from NCI SEER's official U.S. county population file
     on the same seer.cancer.gov domain already used successfully for the HSA crosswalk.
   - File used:
     https://seer.cancer.gov/popdata/yr1969_2024.20ages/
     us.1969_2024.20ages.adjusted.txt.gz
   - Fixed-width format follows the official SEER population data dictionary.
   - 2020 county population is obtained by summing race x sex x age cells.

2) Empty HSA mortality model:
   - The CDC county-week dataset's `state` field is a state NAME, not necessarily a
     two-letter abbreviation. Earlier code filtered it against abbreviations and could
     remove all rows.
   - v17d derives state from the county FIPS code instead.
   - HSA IDs are normalized consistently before all merges.

Additional validity improvement:
- CDC suppresses county-week COVID death counts 1-9.
- v17d does NOT silently treat suppressed counts as zero.
- For each HSA-week it calculates the fraction of HSA population represented by counties
  with non-suppressed death counts.
- Mortality windows are retained only if mean population coverage across included weeks
  is >= 80% (configurable).
- The model stage prints explicit diagnostics and raises an informative error if the
  analytic sample is empty rather than crashing inside np.max().

Locked hospital exposure:
    v13 post/pre log response residuals

Primary downstream outcome:
    HSA COVID-19 mortality per 100,000, +14 to +42 days

Secondary:
    +21 to +49 days

Interpretation remains associative, not causal.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

ROOT = Path(r"C:\Users\SIMONEY\Disease\AJPH_US_Empirical_Upgrade")
RAW = ROOT / "01_raw"
CLEAN = ROOT / "02_clean"
OUTPUT = ROOT / "04_outputs"

for p in [RAW, CLEAN, OUTPUT]:
    p.mkdir(parents=True, exist_ok=True)

HOSPITAL_PANEL = OUTPUT / "AJPH_v13_hospital_log_response_panel.csv"

SEER_HSA_URL = (
    "https://seer.cancer.gov/seerstat/variables/"
    "countyattribs/Health.Service.Areas.xls"
)

SEER_POP_URL = (
    "https://seer.cancer.gov/popdata/yr1969_2024.20ages/"
    "us.1969_2024.20ages.adjusted.txt.gz"
)

CDC_DATASET_ID = "ite7-j2w7"
CDC_CSV_URL = f"https://data.cdc.gov/resource/{CDC_DATASET_ID}.csv"

MIN_MORTALITY_POP_COVERAGE = 0.80
WILD_REPS = 999
SEED = 20260905

VALID_US = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}

FIPS2STATE = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE",
    "11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA",
    "20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN",
    "28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM",
    "36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
    "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA",
    "54":"WV","55":"WI","56":"WY"
}

if not HOSPITAL_PANEL.exists():
    raise FileNotFoundError(HOSPITAL_PANEL)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 AJPH-research-script/1.0",
    "Accept": "*/*",
})

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def normalize_digits(x, width=None):
    s = pd.Series(x, copy=False).astype(str).str.extract(r"(\d+)", expand=False)
    if width is not None:
        s = s.str.zfill(width)
    return s

def find_col(cols, patterns, required=True):
    nmap = {norm(c): c for c in cols}
    for p in patterns:
        pn = norm(p)
        for nc, orig in nmap.items():
            if nc == pn or pn in nc:
                return orig
    if required:
        raise ValueError(
            f"Could not find column matching {patterns}\n"
            f"Available columns:\n{list(cols)}"
        )
    return None

def get_response(url, timeout=180, retries=5):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            last = r
            if r.status_code == 200:
                return r
        except Exception as e:
            last = e
        time.sleep(1.5 * (i + 1))
    if hasattr(last, "status_code"):
        raise RuntimeError(
            f"HTTP failure for {url}\n"
            f"Status: {last.status_code}\n"
            f"Content-Type: {last.headers.get('content-type')}\n"
            f"Body preview: {last.text[:500]}"
        )
    raise RuntimeError(f"Request failed for {url}: {last}")

def iterative_absorb(df, cols, fe_cols, tol=1e-10, max_iter=500):
    if len(df) == 0:
        raise ValueError(
            "iterative_absorb received 0 rows. Check the diagnostics printed immediately "
            "before the model stage; this is a data/merge issue, not a regression issue."
        )
    z = df[cols].astype(float).copy()
    if z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError(f"Empty absorption matrix: shape={z.shape}")
    for _ in range(max_iter):
        old = z.to_numpy(copy=True)
        for fe in fe_cols:
            z = z - z.groupby(df[fe], sort=False).transform("mean")
        diff = np.abs(z.to_numpy() - old)
        if diff.size == 0:
            raise ValueError("Absorption matrix became empty.")
        if np.nanmax(diff) < tol:
            break
    return z

def cluster_fit(y, X, groups):
    return sm.OLS(
        np.asarray(y, float),
        np.asarray(X, float)
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": np.asarray(groups, dtype=object)}
    )

def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.average(x[m], weights=w[m]))

def weighted_median_date(dates, weights):
    x = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "w": np.asarray(weights, float)
    }).dropna()
    x = x[x["w"] > 0].sort_values("date")
    if x.empty:
        return pd.NaT
    cs = x["w"].cumsum()
    return pd.Timestamp(x.loc[cs >= x["w"].sum()/2, "date"].iloc[0])

def wild_bootstrap(d, rz, outcome, exposure, controls, reps=999, seed=SEED):
    y = rz[outcome].to_numpy(float)
    Xu = rz[[exposure] + controls].to_numpy(float)
    groups = d["state"].astype(str).to_numpy(object)

    unres = cluster_fit(y, Xu, groups)
    obs_beta = float(unres.params[0])
    obs_se = float(unres.bse[0])
    obs_t = obs_beta / obs_se

    Xr = rz[controls].to_numpy(float)
    restricted = sm.OLS(y, Xr).fit()
    yhat = np.asarray(restricted.fittedvalues, float)
    resid = np.asarray(restricted.resid, float)

    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    bt, failed = [], 0

    for b in range(reps):
        signs = {g:rng.choice([-1.0,1.0]) for g in ug}
        ys = yhat + resid * np.array([signs[g] for g in groups], float)
        try:
            mb = cluster_fit(ys, Xu, groups)
            bb = float(mb.params[0])
            bs = float(mb.bse[0])
            if np.isfinite(bb) and np.isfinite(bs) and bs > 0:
                bt.append(bb/bs)
            else:
                failed += 1
        except Exception:
            failed += 1

        if (b+1) % 100 == 0:
            print(f"  bootstrap {b+1}/{reps}; valid={len(bt)}, failed={failed}")

    bt = np.asarray(bt, float)
    p = (np.sum(np.abs(bt) >= abs(obs_t)) + 1) / (len(bt) + 1)

    return {
        "reps_requested": reps,
        "reps_valid": int(len(bt)),
        "reps_failed": int(failed),
        "observed_beta": obs_beta,
        "observed_cluster_se": obs_se,
        "observed_t": float(obs_t),
        "wild_cluster_bootstrap_p_two_sided": float(p),
        "bootstrap_t_p025": float(np.quantile(bt,.025)),
        "bootstrap_t_median": float(np.quantile(bt,.5)),
        "bootstrap_t_p975": float(np.quantile(bt,.975)),
    }


# ==============================================================
# 1. HSA crosswalk
# ==============================================================

print("\n[1/10] NCI SEER HSA CROSSWALK")

hsa_xls = RAW / "SEER_Health_Service_Areas.xls"

if not hsa_xls.exists():
    r = get_response(SEER_HSA_URL, timeout=90)
    hsa_xls.write_bytes(r.content)

hsa_raw = pd.read_excel(hsa_xls)

print("Crosswalk shape:", hsa_raw.shape)
print("Crosswalk columns:", list(hsa_raw.columns))

fips_col = find_col(hsa_raw.columns, ["fips"])
hsa_mod_col = find_col(
    hsa_raw.columns,
    ["hsa # (nci modified)", "hsa nci modified", "health service area nci modified"]
)
hsa_name_col = find_col(
    hsa_raw.columns,
    [
        "health service area (nci modified) description",
        "nci modified description",
        "health service area description"
    ],
    required=False
)

hsa = pd.DataFrame({
    "county_fips": normalize_digits(hsa_raw[fips_col], 5),
    "hsa_id": normalize_digits(hsa_raw[hsa_mod_col]),
})
hsa["hsa_name"] = (
    hsa_raw[hsa_name_col].astype(str)
    if hsa_name_col else hsa["hsa_id"]
)
hsa = (
    hsa.dropna(subset=["county_fips","hsa_id"])
       .drop_duplicates("county_fips")
)

# Remove unknown-county placeholder FIPS ending 999.
hsa = hsa[~hsa["county_fips"].str.endswith("999")].copy()

print("Counties mapped:", hsa["county_fips"].nunique())
print("HSAs:", hsa["hsa_id"].nunique())


# ==============================================================
# 2. SEER county population (2020)
# ==============================================================

print("\n[2/10] NCI SEER 2020 COUNTY POPULATION")

pop_gz = RAW / "us.1969_2024.20ages.adjusted.txt.gz"

if not pop_gz.exists():
    print("Downloading SEER population file (~75 MB compressed; one-time only)...")
    r = get_response(SEER_POP_URL, timeout=300, retries=5)
    pop_gz.write_bytes(r.content)

# Official SEER fixed-width format:
# year 1-4; state abbr 5-6; state FIPS 7-8; county FIPS 9-11;
# race 14; origin 15; sex 16; age 17-18; population 19-26.
widths = [4,2,2,3,2,1,1,1,2,8]
names = [
    "year","state_abbr","state_fips","county_fips3","registry",
    "race","origin","sex","age","population"
]

with gzip.open(pop_gz, "rt", encoding="ascii", errors="replace") as fh:
    pop_raw = pd.read_fwf(
        fh,
        widths=widths,
        names=names,
        dtype=str
    )

pop_raw["year"] = pd.to_numeric(pop_raw["year"], errors="coerce")
pop2020 = pop_raw[pop_raw["year"] == 2020].copy()

pop2020["county_fips"] = (
    pop2020["state_fips"].astype(str).str.zfill(2)
    + pop2020["county_fips3"].astype(str).str.zfill(3)
)
pop2020["population"] = pd.to_numeric(pop2020["population"], errors="coerce")

county_pop = (
    pop2020.groupby("county_fips", as_index=False)["population"]
    .sum(min_count=1)
)

# Sanity check national total; broad bounds intentionally loose.
national_pop = float(county_pop["population"].sum())
print("2020 county rows:", len(county_pop))
print("2020 summed population:", int(national_pop))

if not (300_000_000 <= national_pop <= 360_000_000):
    raise ValueError(
        "SEER population parse failed sanity check. "
        f"Summed 2020 population={national_pop:,.0f}, expected roughly 300-360 million."
    )

hsa_pop = (
    hsa.merge(county_pop, on="county_fips", how="left")
       .groupby(["hsa_id","hsa_name"], as_index=False)["population"]
       .sum(min_count=1)
)

print("HSA population rows:", len(hsa_pop))
print("HSA population missing:", int(hsa_pop["population"].isna().sum()))


# ==============================================================
# 3. CDC county-week COVID deaths
# ==============================================================

print("\n[3/10] CDC COUNTY-WEEK COVID DEATHS")

cdc_cache = RAW / "CDC_ite7_j2w7_full.csv"

if cdc_cache.exists():
    cdc = pd.read_csv(cdc_cache, low_memory=False)
    print("Using cached CDC file:", cdc_cache)
else:
    parts = []
    offset = 0
    limit = 50000

    while True:
        url = f"{CDC_CSV_URL}?$limit={limit}&$offset={offset}"
        r = get_response(url, timeout=180)

        preview = r.text.lstrip()[:100].lower()
        if preview.startswith("<html") or preview.startswith("<!doctype"):
            raise RuntimeError(
                "CDC endpoint returned HTML instead of CSV.\n"
                f"URL: {url}\n"
                f"Status: {r.status_code}\n"
                f"Body preview: {r.text[:500]}"
            )

        part = pd.read_csv(io.StringIO(r.text), low_memory=False)
        if part.empty:
            break

        parts.append(part)
        offset += len(part)
        print(f"  downloaded rows: {offset}")

        if len(part) < limit:
            break

    if not parts:
        raise RuntimeError("CDC download returned no rows.")

    cdc = pd.concat(parts, ignore_index=True)
    cdc.to_csv(cdc_cache, index=False)

print("CDC rows:", len(cdc))
print("CDC columns:", list(cdc.columns))

county_fips_col = find_col(
    cdc.columns,
    ["fips_code", "county fips", "fips"]
)
week_col = find_col(
    cdc.columns,
    ["week_ending_date", "week ending date", "end week", "week end"]
)
death_col = find_col(
    cdc.columns,
    ["covid_19_deaths", "covid-19 deaths", "covid deaths"]
)

cdc["county_fips"] = normalize_digits(cdc[county_fips_col], 5)
cdc["week_end"] = pd.to_datetime(cdc[week_col], errors="coerce")
cdc["covid_deaths"] = pd.to_numeric(cdc[death_col], errors="coerce")

# IMPORTANT: derive state abbreviation from FIPS, not CDC's text state field.
cdc["state_fips"] = cdc["county_fips"].str[:2]
cdc["state"] = cdc["state_fips"].map(FIPS2STATE)

cdc = cdc[
    cdc["state"].isin(VALID_US)
    & cdc["county_fips"].notna()
    & cdc["week_end"].notna()
].copy()

print("CDC usable rows after FIPS/state normalization:", len(cdc))
print("CDC states:", cdc["state"].nunique())
print("CDC counties:", cdc["county_fips"].nunique())
print("COVID death count nonmissing fraction:",
      float(cdc["covid_deaths"].notna().mean()))

# Attach county population and HSA.
county_week = cdc[
    ["county_fips","state","week_end","covid_deaths"]
].drop_duplicates(
    ["county_fips","week_end"]
).merge(
    county_pop,
    on="county_fips",
    how="left"
).merge(
    hsa[["county_fips","hsa_id","hsa_name"]],
    on="county_fips",
    how="left"
)

county_week["death_observed"] = county_week["covid_deaths"].notna().astype(int)
county_week["observed_population"] = np.where(
    county_week["covid_deaths"].notna(),
    county_week["population"],
    0.0
)

# Because modified HSA does not cross state boundaries, keep state in key.
hsa_week = (
    county_week.dropna(subset=["hsa_id"])
    .groupby(["hsa_id","hsa_name","state","week_end"], as_index=False)
    .agg(
        covid_deaths=("covid_deaths", lambda x: x.sum(min_count=1)),
        observed_population=("observed_population","sum")
    )
)

hsa_week = hsa_week.merge(
    hsa_pop[["hsa_id","population"]].drop_duplicates("hsa_id"),
    on="hsa_id",
    how="left"
)

hsa_week["population_coverage"] = (
    hsa_week["observed_population"] / hsa_week["population"]
)

print("HSA-week rows:", len(hsa_week))
print("HSA-week median population coverage:",
      float(hsa_week["population_coverage"].median()))

hsa_week.to_csv(
    CLEAN / "CDC_HSA_weekly_covid_deaths_v17d.csv",
    index=False
)


# ==============================================================
# 4. Map locked v13 hospital episodes to HSA
# ==============================================================

print("\n[4/10] MAPPING V13 HOSPITAL EPISODES TO HSA")

hp = pd.read_csv(HOSPITAL_PANEL, low_memory=False)
hp["state"] = hp["state"].astype(str).str.upper().str.strip()
hp = hp[hp["state"].isin(VALID_US)].copy()
hp["signal_week"] = pd.to_datetime(hp["signal_week"], errors="coerce")

hospital_fips_col = find_col(
    hp.columns,
    ["fips_code","county_fips","fips"]
)

hp["county_fips"] = normalize_digits(hp[hospital_fips_col], 5)

hp = hp.merge(
    hsa[["county_fips","hsa_id","hsa_name"]],
    on="county_fips",
    how="left"
)

mapping_fraction = float(hp["hsa_id"].notna().mean())

print("Hospital episodes:", len(hp))
print("Mapped to HSA:", int(hp["hsa_id"].notna().sum()))
print("Mapped fraction:", mapping_fraction)


# ==============================================================
# 5. Form HSA surge episodes
# ==============================================================

print("\n[5/10] FORMING HSA SURGE EPISODES")

required = [
    "post_log_response_residual_cf",
    "pre_log_response_residual_cf",
    "baseline_capacity_quality",
    "signal_admissions_7d",
    "hospital_id"
]
missing = [c for c in required if c not in hp.columns]
if missing:
    raise ValueError("Missing required v13 columns:\n" + "\n".join(missing))

hp = hp.dropna(
    subset=[
        "hsa_id","signal_week",
        "post_log_response_residual_cf",
        "pre_log_response_residual_cf",
        "baseline_capacity_quality"
    ]
).copy()

episode_rows = []

for (hsa_id, state), g in hp.groupby(["hsa_id","state"]):
    g = g.sort_values("signal_week").copy()
    current = []
    last_date = None
    episode_n = 0

    def finalize(rows, n):
        if not rows:
            return None
        x = pd.DataFrame(rows)
        w = x["baseline_capacity_quality"].to_numpy(float)
        post = weighted_mean(x["post_log_response_residual_cf"], w)
        pre = weighted_mean(x["pre_log_response_residual_cf"], w)

        return {
            "hsa_id": str(hsa_id),
            "state": str(state),
            "hsa_episode_number": int(n),
            "signal_date": weighted_median_date(x["signal_week"], w),
            "hospital_episode_count": int(len(x)),
            "hospital_count": int(x["hospital_id"].nunique()),
            "represented_capacity": float(np.nansum(w)),
            "hsa_post_log_response": post,
            "hsa_pre_log_response": pre,
            "hsa_log_response_innovation": (
                post-pre if np.isfinite(post) and np.isfinite(pre) else np.nan
            ),
            "signal_admissions_weighted": weighted_mean(
                x["signal_admissions_7d"], w
            )
        }

    for _, row in g.iterrows():
        dt = pd.Timestamp(row["signal_week"])
        if last_date is None or (dt-last_date).days <= 21:
            current.append(row.to_dict())
        else:
            episode_n += 1
            rec = finalize(current, episode_n)
            if rec:
                episode_rows.append(rec)
            current = [row.to_dict()]
        last_date = dt

    episode_n += 1
    rec = finalize(current, episode_n)
    if rec:
        episode_rows.append(rec)

hep = pd.DataFrame(episode_rows)

# normalize HSA ids on both sides
hep["hsa_id"] = normalize_digits(hep["hsa_id"])
hsa_pop["hsa_id"] = normalize_digits(hsa_pop["hsa_id"])
hsa_week["hsa_id"] = normalize_digits(hsa_week["hsa_id"])

hep = hep.merge(
    hsa_pop[["hsa_id","population"]].drop_duplicates("hsa_id"),
    on="hsa_id",
    how="left"
)

print("HSA surge episodes:", len(hep))
print("HSAs represented:", hep["hsa_id"].nunique())
print("Median hospitals/episode:", float(hep["hospital_count"].median()))
print("Episodes missing population:", int(hep["population"].isna().sum()))


# ==============================================================
# 6. Attach mortality with suppression coverage requirement
# ==============================================================

print("\n[6/10] ATTACHING HSA MORTALITY OUTCOMES")

death_lookup = {
    (str(h),str(s)): g.sort_values("week_end")
    for (h,s),g in hsa_week.groupby(["hsa_id","state"])
}

def mortality_window(row, d1, d2):
    key = (str(row["hsa_id"]), str(row["state"]))
    g = death_lookup.get(key)

    if g is None:
        return pd.Series({
            "rate":np.nan,
            "coverage":np.nan,
            "weeks":0
        })

    start = pd.Timestamp(row["signal_date"]) + pd.Timedelta(days=d1)
    end = pd.Timestamp(row["signal_date"]) + pd.Timedelta(days=d2)

    x = g[(g["week_end"] >= start) & (g["week_end"] <= end)].copy()

    if x.empty:
        return pd.Series({
            "rate":np.nan,
            "coverage":np.nan,
            "weeks":0
        })

    coverage = float(x["population_coverage"].mean())
    n_weeks = int(len(x))

    if (
        not np.isfinite(coverage)
        or coverage < MIN_MORTALITY_POP_COVERAGE
    ):
        return pd.Series({
            "rate":np.nan,
            "coverage":coverage,
            "weeks":n_weeks
        })

    deaths = x["covid_deaths"].sum(min_count=1)
    popn = row["population"]

    if not np.isfinite(deaths) or not np.isfinite(popn) or popn <= 0:
        rate = np.nan
    else:
        rate = float(deaths / popn * 100000)

    return pd.Series({
        "rate":rate,
        "coverage":coverage,
        "weeks":n_weeks
    })

m1 = hep.apply(mortality_window, axis=1, args=(14,42))
m2 = hep.apply(mortality_window, axis=1, args=(21,49))

hep["covid_deaths_14_42_per100k"] = m1["rate"]
hep["mortality_coverage_14_42"] = m1["coverage"]
hep["mortality_weeks_14_42"] = m1["weeks"]

hep["covid_deaths_21_49_per100k"] = m2["rate"]
hep["mortality_coverage_21_49"] = m2["coverage"]
hep["mortality_weeks_21_49"] = m2["weeks"]

hep["signal_month"] = (
    pd.to_datetime(hep["signal_date"])
    .dt.to_period("M").astype(str).astype(object)
)
hep["signal_quarter"] = (
    pd.to_datetime(hep["signal_date"])
    .dt.to_period("Q").astype(str).astype(object)
)
hep["calendar_day"] = (
    pd.to_datetime(hep["signal_date"])
    - pd.to_datetime(hep["signal_date"]).min()
).dt.days.astype(float)

def assign_wave(d):
    d = pd.Timestamp(d)
    if d < pd.Timestamp("2020-10-01"): return "2020_summer"
    if d < pd.Timestamp("2021-03-01"): return "2020_21_winter"
    if d < pd.Timestamp("2021-07-01"): return "2021_spring"
    if d < pd.Timestamp("2021-11-01"): return "delta"
    if d < pd.Timestamp("2022-03-01"): return "omicron_BA1"
    if d < pd.Timestamp("2022-07-01"): return "omicron_BA2"
    return "omicron_later"

hep["wave"] = hep["signal_date"].apply(assign_wave).astype(object)
hep["hsa_id"] = hep["hsa_id"].astype(str).astype(object)
hep["state"] = hep["state"].astype(str).astype(object)

hep.to_csv(
    OUTPUT / "AJPH_v17d_HSA_mortality_bridge.csv",
    index=False
)

print("Primary mortality nonmissing:",
      int(hep["covid_deaths_14_42_per100k"].notna().sum()))
print("Secondary mortality nonmissing:",
      int(hep["covid_deaths_21_49_per100k"].notna().sum()))
print("Primary median coverage:",
      float(hep["mortality_coverage_14_42"].median(skipna=True)))


# ==============================================================
# 7. HSA mortality models + explicit diagnostics
# ==============================================================

print("\n[7/10] HSA MORTALITY MODELS")

OUTCOME = "covid_deaths_14_42_per100k"
EXP = "hsa_log_response_innovation"

BASE_CONTROLS = [
    "hsa_pre_log_response",
    "hospital_count",
    "represented_capacity",
    "signal_admissions_weighted",
]

def fit_model(time_mode, outcome=OUTCOME):
    need = [outcome,EXP,"hsa_id","state"] + BASE_CONTROLS

    if time_mode == "wave":
        need.append("wave")
        fe = ["hsa_id","wave"]
        controls = BASE_CONTROLS
    elif time_mode == "quarter":
        need.append("signal_quarter")
        fe = ["hsa_id","signal_quarter"]
        controls = BASE_CONTROLS
    elif time_mode == "month":
        need.append("signal_month")
        fe = ["hsa_id","signal_month"]
        controls = BASE_CONTROLS
    elif time_mode == "linear":
        need.append("calendar_day")
        fe = ["hsa_id"]
        controls = BASE_CONTROLS + ["calendar_day"]
    else:
        raise ValueError(time_mode)

    before = len(hep)
    d = hep.dropna(subset=need).copy()
    after_nonmissing = len(d)

    counts = d["hsa_id"].value_counts()
    valid_hsa = counts[counts >= 2].index
    d = d[d["hsa_id"].isin(valid_hsa)].copy()

    print(
        f"  {outcome} | {time_mode}: "
        f"total={before}, complete={after_nonmissing}, "
        f"after >=2 episodes/HSA={len(d)}, "
        f"HSAs={d['hsa_id'].nunique()}, states={d['state'].nunique()}"
    )

    if len(d) == 0:
        raise ValueError(
            f"Zero analytic rows for {outcome}, {time_mode}.\n"
            "Check: mortality coverage, HSA ID normalization, CDC FIPS mapping, "
            "and the >=2 episodes/HSA requirement."
        )

    rz = iterative_absorb(
        d,
        [outcome,EXP] + controls,
        fe
    )

    # Remove columns with numerically zero residual variance after FE.
    kept_controls = []
    for c in controls:
        if np.nanstd(rz[c].to_numpy(float)) > 1e-12:
            kept_controls.append(c)

    xcols = [EXP] + kept_controls

    if np.nanstd(rz[EXP].to_numpy(float)) <= 1e-12:
        raise ValueError(
            f"Exposure has no within-FE variation for {time_mode}."
        )

    m = cluster_fit(
        rz[outcome],
        rz[xcols],
        d["state"]
    )

    ci = m.conf_int()[0]
    sx = d[EXP].std(ddof=1)
    sy = d[outcome].std(ddof=1)

    return {
        "time_control":time_mode,
        "outcome":outcome,
        "n":int(len(d)),
        "hsas":int(d["hsa_id"].nunique()),
        "states":int(d["state"].nunique()),
        "coef":float(m.params[0]),
        "se_cluster":float(m.bse[0]),
        "p_value":float(m.pvalues[0]),
        "ci_low":float(ci[0]),
        "ci_high":float(ci[1]),
        "std_beta":float(m.params[0]*sx/sy) if sy > 0 else np.nan,
        "controls_used":";".join(kept_controls)
    }, d, rz, kept_controls

model_rows = []

for tm in ["wave","quarter","month","linear"]:
    r,*_ = fit_model(tm)
    model_rows.append(r)

for tm in ["wave","quarter","month","linear"]:
    r,*_ = fit_model(
        tm,
        outcome="covid_deaths_21_49_per100k"
    )
    model_rows.append(r)

results = pd.DataFrame(model_rows)
print("\nMODEL RESULTS")
print(results.to_string(index=False))

results.to_csv(
    OUTPUT / "AJPH_v17d_HSA_mortality_models.csv",
    index=False
)


# ==============================================================
# 8. LOSO
# ==============================================================

print("\n[8/10] HSA PRIMARY MODEL LOSO BY STATE")

base, base_d, base_rz, base_controls = fit_model("wave")
loso_rows = []

for i, st in enumerate(sorted(base_d["state"].astype(str).unique()),1):
    d0 = hep[hep["state"].astype(str) != st].copy()

    need = [OUTCOME,EXP,"hsa_id","state","wave"] + BASE_CONTROLS
    d = d0.dropna(subset=need).copy()
    counts = d["hsa_id"].value_counts()
    d = d[d["hsa_id"].isin(counts[counts >= 2].index)].copy()

    if len(d) == 0:
        continue

    rz = iterative_absorb(
        d,
        [OUTCOME,EXP] + BASE_CONTROLS,
        ["hsa_id","wave"]
    )

    kept = [
        c for c in BASE_CONTROLS
        if np.nanstd(rz[c].to_numpy(float)) > 1e-12
    ]

    m = cluster_fit(
        rz[OUTCOME],
        rz[[EXP] + kept],
        d["state"]
    )

    loso_rows.append({
        "excluded_state":st,
        "coef":float(m.params[0]),
        "p_value":float(m.pvalues[0]),
        "n":int(len(d)),
        "hsas":int(d["hsa_id"].nunique())
    })

    if i % 10 == 0:
        print(f"  LOSO {i}")

loso = pd.DataFrame(loso_rows)
loso.to_csv(
    OUTPUT / "AJPH_v17d_HSA_LOSO.csv",
    index=False
)

loso_summary = {
    "n_models":int(len(loso)),
    "fraction_negative":float((loso["coef"]<0).mean()),
    "median_coef":float(loso["coef"].median()),
    "min_coef":float(loso["coef"].min()),
    "max_coef":float(loso["coef"].max()),
    "fraction_p_lt_0_05":float((loso["p_value"]<.05).mean())
}
print(json.dumps(loso_summary, indent=2))


# ==============================================================
# 9. Wild cluster bootstrap
# ==============================================================

print("\n[9/10] HSA PRIMARY WILD CLUSTER BOOTSTRAP")

wild = wild_bootstrap(
    base_d,
    base_rz,
    OUTCOME,
    EXP,
    base_controls,
    reps=WILD_REPS,
    seed=SEED
)
print(json.dumps(wild, indent=2))


# ==============================================================
# 10. Final summary
# ==============================================================

print("\n[10/10] FINAL SUMMARY")

summary = {
    "population_source":"NCI SEER U.S. County Population Data, 2020",
    "mapping":{
        "counties_in_hsa_crosswalk":int(hsa["county_fips"].nunique()),
        "hsas_in_crosswalk":int(hsa["hsa_id"].nunique()),
        "hospital_episode_mapping_fraction":mapping_fraction
    },
    "cdc":{
        "usable_rows":int(len(cdc)),
        "states":int(cdc["state"].nunique()),
        "counties":int(cdc["county_fips"].nunique()),
        "death_nonmissing_fraction":float(cdc["covid_deaths"].notna().mean())
    },
    "hsa_episodes":{
        "n":int(len(hep)),
        "hsas":int(hep["hsa_id"].nunique()),
        "median_hospitals_per_episode":float(hep["hospital_count"].median()),
        "mortality_primary_nonmissing":int(hep[OUTCOME].notna().sum()),
        "primary_median_population_coverage":float(
            hep["mortality_coverage_14_42"].median(skipna=True)
        )
    },
    "models":model_rows,
    "loso":loso_summary,
    "wild_cluster_bootstrap":wild
}

(OUTPUT / "AJPH_v17d_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print(json.dumps(summary, indent=2))

print("\nCOMPLETE")
print("Bridge:", OUTPUT/"AJPH_v17d_HSA_mortality_bridge.csv")
print("Models:", OUTPUT/"AJPH_v17d_HSA_mortality_models.csv")
print("LOSO:", OUTPUT/"AJPH_v17d_HSA_LOSO.csv")
print("Summary:", OUTPUT/"AJPH_v17d_summary.json")
