# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH U.S. Empirical Upgrade v18b
Federal Military Medical Deployment Validation — No External Fuzzy-Match Dependency
Matched Stacked Event Study Using Official U.S. Army North Deployment Dates

Official treatment source
-------------------------
U.S. Army North (Fifth Army), Fact Sheet:
"U.S. Army North COVID-19 Hospital Support from August 2021 to March 2022"
As of March 29, 2022.

The fact sheet reports 68 military medical response teams supporting 62 civilian
hospitals in 30 states and the Navajo Nation.

Design
------
This is NOT assumed exogenous by construction: military teams were deployed to
stressed hospitals. Therefore the script explicitly treats this as an
external-intervention validation and tests pre-trends.

For each treated hospital:
1. Match to the HHS facility-week panel by hospital name within state.
2. Select never-treated control hospitals in the same state.
3. Match controls on PRE-deployment:
   - mean inpatient occupancy
   - occupancy slope
   - mean COVID admissions
   - admissions slope
   - staffed bed capacity
4. Keep up to 3 nearest controls.
5. Build a stacked event-study panel, event weeks -6 ... +6.
6. Estimate:
      stack×hospital FE
    + calendar-week FE
    + event-week FE
    + treated × event-week interactions
   with event week -1 as reference.
7. State-clustered SE.
8. Joint pre-period test (-6:-2).
9. Joint post-period test (+1:+4).
10. 999-rep wild cluster bootstrap for average post effect.

Primary outcome:
    inpatient occupancy = occupied adult beds / staffed adult beds

Secondary mechanism outcome:
    log staffed adult bed capacity

Interpretation
--------------
Only if pre-period treated-control differences are approximately null should the
post-period contrast be described as quasi-experimental evidence. Otherwise,
report it as intervention-aligned observational validation.

No result-dependent trimming or treatment redefinition is performed.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from difflib import SequenceMatcher


ROOT = Path(r"C:\Users\SIMONEY\Disease\AJPH_US_Empirical_Upgrade")
CLEAN = ROOT / "02_clean"
OUTPUT = ROOT / "04_outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

CANDIDATE_FILES = [
    CLEAN / "hhs_facility_weekly_selected_clean_v9.csv",
    CLEAN / "hhs_facility_weekly_selected_clean.csv",
    OUTPUT / "AJPH_v14b_reconstructed_event_time_panel.csv",
]

EVENT_WEEKS = list(range(-6, 7))
REF_WEEK = -1
PRE_TEST_WEEKS = [-6,-5,-4,-3,-2]
POST_TEST_WEEKS = [1,2,3,4]

MATCH_SCORE_MIN = 85
MATCH_GAP_MIN = 3
N_CONTROLS = 3
WILD_REPS = 999
SEED = 20260905

VALID_US = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}


# ------------------------------------------------------------------
# Official deployment inventory, transcribed from Army North fact sheet
# ------------------------------------------------------------------

DEPLOYMENTS = [
    # state, hospital, city, start, end, teams, primary_clinical_support
    ("AL","Southeast Health","Dothan","2021-08-29","2021-10-27",1,1),
    ("AL","Dale Medical Center","Ozark","2021-09-08","2021-11-07",1,1),

    ("AZ","Yuma Regional Medical Center","Yuma","2022-01-01","2022-03-08",1,1),
    ("AZ","Canyon Vista Medical Center","Sierra Vista","2022-02-16","2022-03-16",1,1),
    ("AZ","Valleywise Health Medical Center","Phoenix","2022-02-19","2022-03-20",1,1),

    ("AR","University of Arkansas for Medical Sciences","Little Rock","2021-09-10","2021-10-11",1,1),

    ("CA","Emanate Health Queen of the Valley Hospital","West Covina","2022-02-07","2022-03-08",1,1),

    ("CO","UCHealth Poudre Valley Hospital","Fort Collins","2021-11-25","2021-12-23",1,1),
    # mAb infusion team: retained in inventory but excluded from primary
    ("CO","Denver Health Federico F. Pena Southwest Family Health Center and Urgent Care","Denver","2021-12-11","2022-01-09",1,0),

    ("CT","Yale New Haven Hospital","New Haven","2022-02-11","2022-03-11",1,1),
    ("CT","Saint Francis Hospital","Hartford","2022-02-13","2022-03-14",1,1),
    ("CT","Hartford Hospital","Hartford","2022-02-15","2022-03-16",1,1),

    ("ID","Kootenai Health","Coeur d'Alene","2021-09-07","2021-12-06",1,1),

    ("IN","Indiana University Health Methodist Hospital","Indianapolis","2021-12-24","2022-02-21",1,1),

    ("LA","Ochsner Lafayette General Medical Center","Lafayette","2021-08-20","2021-10-19",1,1),
    ("LA","Our Lady of the Lake Regional Medical Center","Baton Rouge","2021-08-26","2021-10-25",1,1),
    ("LA","Rapides Regional Medical Center","Alexandria","2021-08-30","2021-10-29",1,1),
    ("LA","St. Francis Medical Center","Monroe","2022-02-09","2022-03-10",2,1),

    ("ME","Central Maine Medical Center","Lewiston","2022-02-01","2022-03-01",1,1),
    ("ME","Northern Light Eastern Maine Medical Center","Bangor","2022-02-19","2022-03-20",2,1),

    ("MD","Adventist HealthCare Alternate Care Site","Takoma Park","2022-02-04","2022-03-05",1,1),

    ("MA","Lawrence General Hospital","Lawrence","2022-02-13","2022-03-14",1,1),
    ("MA","Signature Healthcare Brockton Hospital","Brockton","2022-02-15","2022-03-16",1,1),

    ("MI","Beaumont Hospital Dearborn","Dearborn","2021-12-04","2022-02-01",1,1),
    ("MI","Spectrum Health Butterworth Hospital","Grand Rapids","2021-12-06","2022-02-03",1,1),
    ("MI","Mercy Health Muskegon","Muskegon","2022-01-02","2022-01-31",1,1),
    ("MI","Henry Ford Wyandotte Hospital","Wyandotte","2022-01-24","2022-02-22",1,1),
    ("MI","Covenant Medical Center Harrison","Saginaw","2021-12-12","2022-02-23",1,1),
    ("MI","Sparrow Hospital","Lansing","2022-02-08","2022-03-09",1,1),

    ("MN","Hennepin County Medical Center","Minneapolis","2021-11-25","2022-01-21",1,1),
    # Fact sheet says "Providence St. Patrick Hospital in St. Cloud"; likely naming anomaly.
    # It is retained exactly as source and must pass fuzzy/state match audit before use.
    ("MN","Providence St. Patrick Hospital","St. Cloud","2021-11-29","2022-01-27",1,1),
    ("MN","Abbott Northwestern Hospital","Minneapolis","2022-02-01","2022-03-16",1,1),

    ("MS","University of Mississippi Medical Center","Jackson","2021-08-23","2021-10-20",1,1),
    ("MS","North Mississippi Medical Center Tupelo","Tupelo","2021-08-27","2021-10-23",1,1),

    ("MO","Christian Hospital","St. Louis","2022-02-01","2022-03-02",1,1),

    ("MT","Billings Clinic Hospital","Billings","2021-11-11","2021-12-10",1,1),
    ("MT","Benefis Health System","Great Falls","2021-11-19","2021-12-18",1,1),
    ("MT","Providence St. Patrick Hospital","Missoula","2021-11-25","2021-12-23",1,1),

    ("NH","Elliot Hospital","Manchester","2022-01-07","2022-03-07",1,1),

    ("NJ","University Hospital","Newark","2022-01-23","2022-02-20",1,1),

    # Two overlapping teams at same hospital; use first deployment start and final end, teams=2.
    ("NM","San Juan Regional Medical Center","Farmington","2021-12-06","2022-02-04",2,1),
    ("NM","University of New Mexico Hospital","Albuquerque","2022-01-23","2022-03-07",1,1),

    ("NY","Coney Island Hospital","Brooklyn","2022-01-24","2022-02-22",1,1),
    ("NY","North Central Bronx Hospital","Bronx","2022-01-31","2022-02-26",1,1),
    ("NY","Erie County Medical Center","Buffalo","2022-01-10","2022-03-10",1,1),
    ("NY","SUNY Upstate Medical University Hospital","Syracuse","2022-02-11","2022-03-12",2,1),
    ("NY","University of Rochester Medical Center Strong Memorial Hospital","Rochester","2022-02-16","2022-03-16",2,1),

    ("OH","Cleveland Clinic","Cleveland","2022-01-21","2022-02-19",1,1),
    ("OH","Summa Health System Akron Campus","Akron","2022-02-01","2022-03-02",1,1),

    ("OK","OU Health University of Oklahoma Medical Center","Oklahoma City","2022-02-07","2022-03-09",1,1),
    ("OK","Integris Baptist Medical Center","Oklahoma City","2022-02-09","2022-03-10",2,1),

    ("PA","WellSpan Surgery and Rehabilitation Hospital","York","2022-01-03","2022-03-03",1,1),
    ("PA","Regional Hospital of Scranton","Scranton","2022-01-04","2022-03-04",1,1),

    ("RI","Rhode Island Hospital","Providence","2022-01-23","2022-03-06",1,1),

    ("TN","University of Tennessee Medical Center","Knoxville","2021-09-25","2021-10-23",1,1),

    ("TX","Northwest Texas Healthcare System","Amarillo","2022-01-28","2022-02-26",1,1),

    # mAb infusion team: excluded from primary
    ("UT","St. George Regional Hospital","St. George","2021-11-06","2021-12-03",1,0),
    ("UT","University of Utah Hospital","Salt Lake City","2022-03-05","2022-03-29",1,1),

    ("WA","Providence Sacred Heart Medical Center","Spokane","2021-10-19","2021-11-17",1,1),
    ("WA","Confluence Health","Wenatchee","2021-10-20","2021-11-22",1,1),

    ("WI","Bellin Hospital","Green Bay","2021-12-30","2022-02-27",1,1),

    # Navajo Nation support; NM location. Retained but excluded from primary state-matched design
    # because the source classifies it under Navajo Nation rather than ordinary state deployment.
    ("NM","Northern Navajo Medical Center","Shiprock","2022-01-26","2022-02-27",1,0),
]

deploy = pd.DataFrame(
    DEPLOYMENTS,
    columns=[
        "state","official_hospital_name","city","start_date","end_date",
        "team_count","primary_clinical_support"
    ]
)

deploy["start_date"] = pd.to_datetime(deploy["start_date"])
deploy["end_date"] = pd.to_datetime(deploy["end_date"])

print("\n[1/10] OFFICIAL DEPLOYMENT INVENTORY")
print("Inventory hospitals:", len(deploy))
print("Primary clinical-support hospitals:",
      int(deploy["primary_clinical_support"].sum()))
print("States represented:", deploy["state"].nunique())
print("Total teams represented:", int(deploy["team_count"].sum()))

deploy.to_csv(
    OUTPUT / "AJPH_v18_ARNORTH_official_deployment_inventory.csv",
    index=False
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def norm_name(x):
    x = str(x).lower()
    x = x.replace("&", " and ")
    x = re.sub(r"\buniversity\b", " univ ", x)
    x = re.sub(r"\bmedical center\b", " med center ", x)
    x = re.sub(r"\bmedical centre\b", " med center ", x)
    x = re.sub(r"\bhospital\b", " hosp ", x)
    x = re.sub(r"\bhealthcare\b", " health care ", x)
    x = re.sub(r"\bsaint\b", " st ", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return " ".join(x.split())


def token_set_ratio(a, b):
    """
    Standard-library token-set similarity score, 0-100.
    Replacement for rapidfuzz.fuzz.token_set_ratio.
    """
    a_tokens = set(str(a).split())
    b_tokens = set(str(b).split())

    if not a_tokens and not b_tokens:
        return 100.0
    if not a_tokens or not b_tokens:
        return 0.0

    inter = a_tokens & b_tokens
    a_only = a_tokens - inter
    b_only = b_tokens - inter

    s_inter = " ".join(sorted(inter))
    s_a = " ".join(sorted(inter | a_only))
    s_b = " ".join(sorted(inter | b_only))

    scores = [SequenceMatcher(None, s_a, s_b).ratio()]
    if s_inter:
        scores.append(SequenceMatcher(None, s_inter, s_a).ratio())
        scores.append(SequenceMatcher(None, s_inter, s_b).ratio())

    return 100.0 * max(scores)

def find_col(cols, candidates, required=True):
    normed = {
        re.sub(r"[^a-z0-9]+","_",str(c).lower()).strip("_"): c
        for c in cols
    }
    for cand in candidates:
        cc = re.sub(r"[^a-z0-9]+","_",cand.lower()).strip("_")
        for nc, orig in normed.items():
            if nc == cc or cc in nc:
                return orig
    if required:
        raise ValueError(
            f"Could not find column matching {candidates}\n"
            f"Available columns:\n{list(cols)}"
        )
    return None

def iterative_absorb(df, cols, fe_cols, tol=1e-10, max_iter=1000):
    if len(df) == 0:
        raise ValueError("Cannot absorb FE on an empty dataset.")
    z = df[cols].astype(float).copy()
    for _ in range(max_iter):
        old = z.to_numpy(copy=True)
        for fe in fe_cols:
            z = z - z.groupby(df[fe], sort=False).transform("mean")
        diff = np.abs(z.to_numpy() - old)
        if diff.size == 0:
            raise ValueError("FE residualization became empty.")
        if np.nanmax(diff) < tol:
            break
    return z

def cluster_fit(y, X, groups):
    return sm.OLS(
        np.asarray(y,float),
        np.asarray(X,float)
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups":np.asarray(groups,dtype=object)}
    )

def fit_slope(g, ycol):
    x = g["rel_pre_week"].to_numpy(float)
    y = g[ycol].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])

def nearest_sunday(dt):
    dt = pd.Timestamp(dt)
    # HHS collection_week usually Sunday.
    return dt - pd.Timedelta(days=(dt.weekday()+1) % 7)

def wild_avg_post(y, Xfull, Xr, groups, idx, reps=WILD_REPS, seed=SEED):
    fit = cluster_fit(y, Xfull, groups)
    beta = np.asarray(fit.params,float)
    cov = np.asarray(fit.cov_params(),float)

    a = np.zeros(len(beta))
    a[idx] = 1/len(idx)
    obs_eff = float(a @ beta)
    obs_se = float(np.sqrt(a @ cov @ a))
    obs_t = obs_eff / obs_se

    rfit = sm.OLS(y, Xr).fit()
    yhat = np.asarray(rfit.fittedvalues,float)
    resid = np.asarray(rfit.resid,float)

    ug = np.unique(groups)
    rng = np.random.default_rng(seed)

    bt=[]
    failed=0
    for b in range(reps):
        signs = {g:rng.choice([-1.,1.]) for g in ug}
        ys = yhat + resid*np.array([signs[g] for g in groups],float)
        try:
            mb = cluster_fit(ys, Xfull, groups)
            bvec=np.asarray(mb.params,float)
            bcov=np.asarray(mb.cov_params(),float)
            eff=float(a@bvec)
            se=float(np.sqrt(a@bcov@a))
            if np.isfinite(eff) and np.isfinite(se) and se>0:
                bt.append(eff/se)
            else:
                failed += 1
        except Exception:
            failed += 1
        if (b+1)%100==0:
            print(f"  bootstrap {b+1}/{reps}; valid={len(bt)}, failed={failed}")

    bt=np.asarray(bt,float)
    p=(np.sum(np.abs(bt)>=abs(obs_t))+1)/(len(bt)+1)

    return {
        "observed_average_effect":obs_eff,
        "observed_cluster_se":obs_se,
        "observed_t":float(obs_t),
        "wild_cluster_bootstrap_p_two_sided":float(p),
        "reps_valid":int(len(bt)),
        "reps_failed":int(failed),
        "bootstrap_t_p025":float(np.quantile(bt,.025)),
        "bootstrap_t_median":float(np.quantile(bt,.5)),
        "bootstrap_t_p975":float(np.quantile(bt,.975)),
    }


# ------------------------------------------------------------------
# 2. Load HHS facility-week data
# ------------------------------------------------------------------

print("\n[2/10] LOADING HHS FACILITY-WEEK DATA")

hhs_file = None
for f in CANDIDATE_FILES:
    if f.exists():
        hhs_file = f
        break

if hhs_file is None:
    # fallback search
    hits = list(ROOT.rglob("*facility*weekly*.csv"))
    if hits:
        hhs_file = hits[0]

if hhs_file is None:
    raise FileNotFoundError(
        "Could not find an HHS facility-week CSV. Expected something like:\n"
        "02_clean/hhs_facility_weekly_selected_clean_v9.csv"
    )

print("HHS file:", hhs_file)

hhs = pd.read_csv(hhs_file, low_memory=False)
print("Rows:", len(hhs))
print("Columns:", list(hhs.columns))

state_col = find_col(hhs.columns, ["state"])
id_col = find_col(
    hhs.columns,
    ["hospital_id","hospital_pk","ccn","facility_id","provider_id"]
)
name_col = find_col(
    hhs.columns,
    ["hospital_name","facility_name","hospital name","facility"]
)
week_col = find_col(
    hhs.columns,
    ["collection_week","week_start","week","calendar_week"]
)
beds_col = find_col(
    hhs.columns,
    [
        "all_adult_hospital_inpatient_beds_7_day_avg",
        "adult_inpatient_beds",
        "staffed_adult_beds",
        "inpatient_beds"
    ]
)
occ_col = find_col(
    hhs.columns,
    [
        "all_adult_hospital_inpatient_bed_occupied_7_day_avg",
        "adult_inpatient_bed_occupied",
        "occupied_adult_beds",
        "inpatient_bed_occupied"
    ]
)

adm_col = find_col(
    hhs.columns,
    [
        "previous_day_admission_adult_covid_confirmed_7_day_sum",
        "adult_covid_admissions",
        "covid_admissions",
        "signal_admissions"
    ],
    required=False
)

city_col = find_col(
    hhs.columns,
    ["city","hospital_city","facility_city"],
    required=False
)

hhs["state_std"] = hhs[state_col].astype(str).str.upper().str.strip()
hhs["hospital_id_std"] = hhs[id_col].astype(str).str.strip()
hhs["hospital_name_std"] = hhs[name_col].astype(str).str.strip()
hhs["week"] = pd.to_datetime(hhs[week_col], errors="coerce")
hhs["staffed_beds"] = pd.to_numeric(hhs[beds_col], errors="coerce")
hhs["occupied_beds"] = pd.to_numeric(hhs[occ_col], errors="coerce")
hhs["occupancy"] = hhs["occupied_beds"] / hhs["staffed_beds"]
hhs.loc[
    (~np.isfinite(hhs["occupancy"]))
    | (hhs["staffed_beds"] <= 0)
    | (hhs["occupancy"] < 0)
    | (hhs["occupancy"] > 2),
    "occupancy"
] = np.nan

if adm_col:
    hhs["covid_admissions"] = pd.to_numeric(hhs[adm_col], errors="coerce")
else:
    hhs["covid_admissions"] = np.nan
    print("WARNING: no COVID admissions field detected; admission matching covariates will be omitted.")

if city_col:
    hhs["city_std"] = hhs[city_col].astype(str).str.strip()
else:
    hhs["city_std"] = ""

hhs = hhs[
    hhs["state_std"].isin(VALID_US)
    & hhs["week"].notna()
].copy()

facility_master = (
    hhs.sort_values("week")
       .groupby("hospital_id_std", as_index=False)
       .agg(
           state=("state_std","first"),
           hospital_name=("hospital_name_std","first"),
           city=("city_std","first")
       )
)

facility_master["name_norm"] = facility_master["hospital_name"].map(norm_name)


# ------------------------------------------------------------------
# 3. Match official deployment hospitals to HHS hospitals
# ------------------------------------------------------------------

print("\n[3/10] MATCHING OFFICIAL DEPLOYMENTS TO HHS FACILITIES")

audit_rows=[]

for _, r in deploy.iterrows():
    pool = facility_master[facility_master["state"] == r["state"]].copy()
    target = norm_name(r["official_hospital_name"])

    if pool.empty:
        audit_rows.append({
            **r.to_dict(),
            "matched_hospital_id":np.nan,
            "matched_hospital_name":np.nan,
            "match_score":np.nan,
            "second_score":np.nan,
            "score_gap":np.nan,
            "match_ok":False,
            "match_reason":"no facilities in state"
        })
        continue

    scored=[]
    for _, p in pool.iterrows():
        score = token_set_ratio(target, p["name_norm"])

        # Small supportive city bonus, never enough to rescue a poor name match.
        if (
            str(r["city"]).strip()
            and str(p["city"]).strip()
            and norm_name(r["city"]) == norm_name(p["city"])
        ):
            score = min(100, score + 2)

        scored.append((score,p["hospital_id_std"],p["hospital_name"],p["city"]))

    scored.sort(reverse=True, key=lambda z:z[0])
    best=scored[0]
    second=scored[1][0] if len(scored)>1 else np.nan
    gap=best[0]-second if np.isfinite(second) else np.inf

    ok=(best[0]>=MATCH_SCORE_MIN) and (gap>=MATCH_GAP_MIN)

    audit_rows.append({
        **r.to_dict(),
        "matched_hospital_id":best[1],
        "matched_hospital_name":best[2],
        "matched_city":best[3],
        "match_score":best[0],
        "second_score":second,
        "score_gap":gap,
        "match_ok":bool(ok),
        "match_reason":"accepted" if ok else "manual review required"
    })

audit=pd.DataFrame(audit_rows)
audit.to_csv(
    OUTPUT / "AJPH_v18_deployment_HHS_match_audit.csv",
    index=False
)

print("Inventory:",len(audit))
print("Automatically accepted:",int(audit["match_ok"].sum()))
print("Primary accepted:",int(
    ((audit["primary_clinical_support"]==1)&audit["match_ok"]).sum()
))

print("\nLOW/AMBIGUOUS MATCHES")
print(
    audit.loc[
        ~audit["match_ok"],
        [
            "state","official_hospital_name","city",
            "matched_hospital_name","match_score","second_score","score_gap"
        ]
    ].to_string(index=False)
)

treated = audit[
    (audit["primary_clinical_support"]==1)
    & audit["match_ok"]
].copy()

if len(treated) < 20:
    raise ValueError(
        f"Only {len(treated)} primary deployments matched automatically. "
        "Review AJPH_v18_deployment_HHS_match_audit.csv before causal analysis."
    )

treated_ids=set(treated["matched_hospital_id"].astype(str))


# ------------------------------------------------------------------
# 4. Pre-period matching covariates
# ------------------------------------------------------------------

print("\n[4/10] CONSTRUCTING MATCHED CONTROL SETS")

match_rows=[]

for cohort_id, tr in treated.reset_index(drop=True).iterrows():
    start_week = nearest_sunday(tr["start_date"])

    # State-level candidate pool, never-treated.
    candidates = facility_master[
        (facility_master["state"]==tr["state"])
        & (~facility_master["hospital_id_std"].isin(treated_ids))
    ].copy()

    # Include treated hospital itself for pre covariate extraction.
    ids = [str(tr["matched_hospital_id"])] + candidates["hospital_id_std"].astype(str).tolist()

    pre = hhs[
        hhs["hospital_id_std"].isin(ids)
        & (hhs["week"] >= start_week - pd.Timedelta(weeks=4))
        & (hhs["week"] <= start_week - pd.Timedelta(weeks=1))
    ].copy()

    pre["rel_pre_week"] = (
        (pre["week"] - start_week).dt.days / 7
    )

    feat=[]

    for hid,g in pre.groupby("hospital_id_std"):
        occ_mean=g["occupancy"].mean()
        occ_slope=fit_slope(g,"occupancy")
        bed_mean=g["staffed_beds"].mean()

        if adm_col:
            adm_mean=g["covid_admissions"].mean()
            adm_slope=fit_slope(g,"covid_admissions")
        else:
            adm_mean=np.nan
            adm_slope=np.nan

        feat.append({
            "hospital_id_std":str(hid),
            "occ_pre_mean":occ_mean,
            "occ_pre_slope":occ_slope,
            "beds_pre_mean":bed_mean,
            "adm_pre_mean":adm_mean,
            "adm_pre_slope":adm_slope,
            "pre_weeks":g["week"].nunique()
        })

    feat=pd.DataFrame(feat)

    treated_id=str(tr["matched_hospital_id"])

    if treated_id not in set(feat["hospital_id_std"].astype(str)):
        continue

    tf=feat[feat["hospital_id_std"]==treated_id].iloc[0]

    if tf["pre_weeks"] < 3 or not np.isfinite(tf["occ_pre_mean"]):
        continue

    cf=feat[
        (feat["hospital_id_std"]!=treated_id)
        & (feat["pre_weeks"]>=3)
        & feat["occ_pre_mean"].notna()
        & feat["beds_pre_mean"].notna()
    ].copy()

    if cf.empty:
        continue

    match_vars=["occ_pre_mean","occ_pre_slope","beds_pre_mean"]
    if adm_col:
        usable_adm = (
            np.isfinite(tf["adm_pre_mean"])
            and cf["adm_pre_mean"].notna().sum() >= 5
        )
        if usable_adm:
            match_vars += ["adm_pre_mean","adm_pre_slope"]

    combined=pd.concat([
        pd.DataFrame([tf]),
        cf
    ],ignore_index=True)

    # standardized distance using candidate+treated cohort distribution
    for v in match_vars:
        med=combined[v].median(skipna=True)
        combined[v]=combined[v].fillna(med)
        sd=combined[v].std(ddof=1)
        if not np.isfinite(sd) or sd<=1e-12:
            combined[v+"_z"]=0.0
        else:
            combined[v+"_z"]=(combined[v]-combined[v].mean())/sd

    trow=combined[combined["hospital_id_std"]==treated_id].iloc[0]

    candidate_rows=combined[combined["hospital_id_std"]!=treated_id].copy()
    candidate_rows["distance"]=0.0

    for v in match_vars:
        candidate_rows["distance"] += (
            candidate_rows[v+"_z"] - trow[v+"_z"]
        )**2

    candidate_rows["distance"]=np.sqrt(candidate_rows["distance"])
    candidate_rows=candidate_rows.sort_values("distance").head(N_CONTROLS)

    if candidate_rows.empty:
        continue

    # treated row
    match_rows.append({
        "cohort_id":cohort_id,
        "deployment_start":start_week,
        "state":tr["state"],
        "hospital_id":treated_id,
        "hospital_name":tr["matched_hospital_name"],
        "treated":1,
        "match_distance":0.0,
        "official_name":tr["official_hospital_name"]
    })

    for _, c in candidate_rows.iterrows():
        nm=facility_master.loc[
            facility_master["hospital_id_std"]==str(c["hospital_id_std"]),
            "hospital_name"
        ]
        match_rows.append({
            "cohort_id":cohort_id,
            "deployment_start":start_week,
            "state":tr["state"],
            "hospital_id":str(c["hospital_id_std"]),
            "hospital_name":nm.iloc[0] if len(nm) else "",
            "treated":0,
            "match_distance":float(c["distance"]),
            "official_name":tr["official_hospital_name"]
        })

matches=pd.DataFrame(match_rows)

if matches.empty:
    raise ValueError("No matched cohorts could be constructed.")

cohort_counts=matches.groupby("cohort_id")["treated"].agg(["sum","count"])
valid_cohorts=cohort_counts[
    (cohort_counts["sum"]==1)&(cohort_counts["count"]>=2)
].index

matches=matches[matches["cohort_id"].isin(valid_cohorts)].copy()

print("Matched cohorts:",matches["cohort_id"].nunique())
print("Treated hospitals:",int(matches["treated"].sum()))
print("Control assignments:",int((matches["treated"]==0).sum()))
print("Median control distance:",
      float(matches.loc[matches["treated"]==0,"match_distance"].median()))

matches.to_csv(
    OUTPUT / "AJPH_v18_matched_deployment_controls.csv",
    index=False
)


# ------------------------------------------------------------------
# 5. Build stacked event panel
# ------------------------------------------------------------------

print("\n[5/10] BUILDING STACKED EVENT PANEL")

stack_parts=[]

for cid,g in matches.groupby("cohort_id"):
    start=pd.Timestamp(g["deployment_start"].iloc[0])

    ids=g["hospital_id"].astype(str).tolist()

    x=hhs[
        hhs["hospital_id_std"].isin(ids)
        & (hhs["week"] >= start - pd.Timedelta(weeks=6))
        & (hhs["week"] <= start + pd.Timedelta(weeks=6))
    ].copy()

    if x.empty:
        continue

    map_treat=dict(zip(g["hospital_id"].astype(str),g["treated"]))
    map_name=dict(zip(g["hospital_id"].astype(str),g["hospital_name"]))

    x["cohort_id"]=int(cid)
    x["treated"]=x["hospital_id_std"].astype(str).map(map_treat).astype(float)
    x["matched_name"]=x["hospital_id_std"].astype(str).map(map_name)

    x["event_week"]=np.round(
        (x["week"]-start).dt.days/7
    ).astype(int)

    x=x[x["event_week"].isin(EVENT_WEEKS)].copy()

    x["stack_hospital_fe"]=(
        x["cohort_id"].astype(str)+"__"+x["hospital_id_std"].astype(str)
    ).astype(object)
    x["calendar_week_fe"]=x["week"].dt.to_period("W-SUN").astype(str).astype(object)
    x["event_week_fe"]=x["event_week"].astype(str).astype(object)

    x["log_staffed_beds"]=np.where(
        x["staffed_beds"]>0,
        np.log(x["staffed_beds"]),
        np.nan
    )

    stack_parts.append(x)

stack=pd.concat(stack_parts,ignore_index=True)

print("Stacked rows:",len(stack))
print("Cohorts:",stack["cohort_id"].nunique())
print("Unique hospitals:",stack["hospital_id_std"].nunique())
print("States:",stack["state_std"].nunique())

stack.to_csv(
    OUTPUT / "AJPH_v18_stacked_deployment_event_panel.csv",
    index=False
)


# ------------------------------------------------------------------
# 6. Event-study function
# ------------------------------------------------------------------

def event_study(df, outcome, label):
    print(f"\n--- {label}: {outcome} ---")

    interaction_cols=[]
    work=df.copy()

    for k in EVENT_WEEKS:
        if k==REF_WEEK:
            continue
        c=f"treated_x_event_{k}"
        work[c]=work["treated"]*(work["event_week"]==k).astype(float)
        interaction_cols.append(c)

    need=[
        outcome,"stack_hospital_fe","calendar_week_fe",
        "event_week_fe","state_std"
    ]+interaction_cols

    d=work.dropna(subset=need).copy()

    # Require both treated and control observations in each cohort overall.
    ct=d.groupby("cohort_id")["treated"].agg(["min","max"])
    valid=ct[(ct["min"]==0)&(ct["max"]==1)].index
    d=d[d["cohort_id"].isin(valid)].copy()

    print(
        "n=",len(d),
        "cohorts=",d["cohort_id"].nunique(),
        "hospitals=",d["hospital_id_std"].nunique(),
        "states=",d["state_std"].nunique()
    )

    rz=iterative_absorb(
        d,
        [outcome]+interaction_cols,
        ["stack_hospital_fe","calendar_week_fe","event_week_fe"]
    )

    X=rz[interaction_cols].to_numpy(float)
    y=rz[outcome].to_numpy(float)
    groups=d["state_std"].astype(str).to_numpy(object)

    rank=np.linalg.matrix_rank(X)
    print("Interaction matrix:",X.shape,"rank=",rank)

    if rank < X.shape[1]:
        raise ValueError(
            f"{label}: interaction design rank deficient "
            f"({rank}/{X.shape[1]})."
        )

    m=cluster_fit(y,X,groups)

    rows=[]
    for j,c in enumerate(interaction_cols):
        k=int(c.split("_")[-1])
        ci=m.conf_int()[j]
        rows.append({
            "outcome":outcome,
            "event_week":k,
            "coef":float(m.params[j]),
            "se_cluster":float(m.bse[j]),
            "p_value":float(m.pvalues[j]),
            "ci_low":float(ci[0]),
            "ci_high":float(ci[1])
        })

    tab=pd.DataFrame(rows).sort_values("event_week")
    print(tab.to_string(index=False))

    pre_idx=[
        interaction_cols.index(f"treated_x_event_{k}")
        for k in PRE_TEST_WEEKS
    ]
    post_idx=[
        interaction_cols.index(f"treated_x_event_{k}")
        for k in POST_TEST_WEEKS
    ]

    Rpre=np.zeros((len(pre_idx),len(interaction_cols)))
    for i,j in enumerate(pre_idx):
        Rpre[i,j]=1

    Rpost=np.zeros((len(post_idx),len(interaction_cols)))
    for i,j in enumerate(post_idx):
        Rpost[i,j]=1

    pre=m.wald_test(Rpre,scalar=True)
    post=m.wald_test(Rpost,scalar=True)

    joint={
        "outcome":outcome,
        "pre_wald_stat":float(pre.statistic),
        "pre_wald_p":float(pre.pvalue),
        "post_wald_stat":float(post.statistic),
        "post_wald_p":float(post.pvalue),
    }

    print("Joint tests:")
    print(json.dumps(joint,indent=2))

    restricted=[
        c for c in interaction_cols
        if c not in [f"treated_x_event_{k}" for k in POST_TEST_WEEKS]
    ]

    boot=wild_avg_post(
        y=y,
        Xfull=X,
        Xr=rz[restricted].to_numpy(float),
        groups=groups,
        idx=post_idx,
        reps=WILD_REPS,
        seed=SEED
    )

    print("Wild bootstrap average post:")
    print(json.dumps(boot,indent=2))

    return tab,joint,boot,d


# ------------------------------------------------------------------
# 7. Primary occupancy event-study
# ------------------------------------------------------------------

print("\n[6/10] PRIMARY OCCUPANCY EVENT STUDY")

occ_tab,occ_joint,occ_boot,occ_d=event_study(
    stack,
    "occupancy",
    "PRIMARY"
)

occ_tab.to_csv(
    OUTPUT / "AJPH_v18_occupancy_event_study.csv",
    index=False
)


# ------------------------------------------------------------------
# 8. Capacity mechanism event-study
# ------------------------------------------------------------------

print("\n[7/10] CAPACITY MECHANISM EVENT STUDY")

cap_tab,cap_joint,cap_boot,cap_d=event_study(
    stack,
    "log_staffed_beds",
    "CAPACITY"
)

cap_tab.to_csv(
    OUTPUT / "AJPH_v18_capacity_event_study.csv",
    index=False
)


# ------------------------------------------------------------------
# 9. Treated vs control change-score validation
# ------------------------------------------------------------------

print("\n[8/10] PRE/POST CHANGE-SCORE VALIDATION")

change_rows=[]

for (cid,hid),g in stack.groupby(["cohort_id","hospital_id_std"]):
    pre=g[g["event_week"].isin([-4,-3,-2])]["occupancy"].mean()
    post=g[g["event_week"].isin([1,2,3,4])]["occupancy"].mean()

    if not np.isfinite(pre) or not np.isfinite(post):
        continue

    change_rows.append({
        "cohort_id":cid,
        "hospital_id":hid,
        "state":g["state_std"].iloc[0],
        "treated":float(g["treated"].iloc[0]),
        "pre_occ":float(pre),
        "post_occ":float(post),
        "delta_occ":float(post-pre)
    })

chg=pd.DataFrame(change_rows)

# Cohort FE absorbs treated-deployment-specific severity/time.
rz=iterative_absorb(
    chg,
    ["delta_occ","treated"],
    ["cohort_id"]
)

cm=cluster_fit(
    rz["delta_occ"],
    rz[["treated"]],
    chg["state"]
)

cci=cm.conf_int()[0]

change_result={
    "n":int(len(chg)),
    "cohorts":int(chg["cohort_id"].nunique()),
    "states":int(chg["state"].nunique()),
    "coef_treated":float(cm.params[0]),
    "se_cluster":float(cm.bse[0]),
    "p_value":float(cm.pvalues[0]),
    "ci_low":float(cci[0]),
    "ci_high":float(cci[1])
}

print(json.dumps(change_result,indent=2))

chg.to_csv(
    OUTPUT / "AJPH_v18_change_score_panel.csv",
    index=False
)


# ------------------------------------------------------------------
# 10. Final summary
# ------------------------------------------------------------------

print("\n[9/10] DESIGN DIAGNOSTICS")

diagnostics={
    "official_inventory_hospitals":int(len(deploy)),
    "official_primary_clinical_support":int(deploy["primary_clinical_support"].sum()),
    "official_team_count":int(deploy["team_count"].sum()),
    "automatic_name_matches":int(audit["match_ok"].sum()),
    "primary_accepted_matches":int(
        ((audit["primary_clinical_support"]==1)&audit["match_ok"]).sum()
    ),
    "matched_cohorts":int(matches["cohort_id"].nunique()),
    "treated_in_matched_design":int(matches["treated"].sum()),
    "control_assignments":int((matches["treated"]==0).sum()),
    "stacked_rows":int(len(stack)),
    "states_in_stack":int(stack["state_std"].nunique())
}

print(json.dumps(diagnostics,indent=2))

print("\n[10/10] FINAL SUMMARY")

summary={
    "design_diagnostics":diagnostics,
    "occupancy_joint_tests":occ_joint,
    "occupancy_wild_bootstrap":occ_boot,
    "capacity_joint_tests":cap_joint,
    "capacity_wild_bootstrap":cap_boot,
    "change_score":change_result,
    "interpretation_rule":{
        "quasi_experimental_upgrade_requires":
            "occupancy pre-period interactions jointly compatible with zero, "
            "with post-period occupancy contrast negative and supported by clustered/wild inference",
        "otherwise":
            "retain as intervention-aligned observational validation only"
    }
}

(OUTPUT / "AJPH_v18_summary.json").write_text(
    json.dumps(summary,indent=2),
    encoding="utf-8"
)

print(json.dumps(summary,indent=2))

print("\nCOMPLETE")
print("Official inventory:",
      OUTPUT/"AJPH_v18_ARNORTH_official_deployment_inventory.csv")
print("Name-match audit:",
      OUTPUT/"AJPH_v18_deployment_HHS_match_audit.csv")
print("Matched controls:",
      OUTPUT/"AJPH_v18_matched_deployment_controls.csv")
print("Stacked panel:",
      OUTPUT/"AJPH_v18_stacked_deployment_event_panel.csv")
print("Occupancy ES:",
      OUTPUT/"AJPH_v18_occupancy_event_study.csv")
print("Capacity ES:",
      OUTPUT/"AJPH_v18_capacity_event_study.csv")
print("Summary:",
      OUTPUT/"AJPH_v18_summary.json")
