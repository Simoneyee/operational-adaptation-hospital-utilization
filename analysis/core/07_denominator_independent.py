# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH U.S. Empirical Upgrade v22
Denominator-Independent Hospital-Strain Sensitivity

Purpose
-------
Address the key reviewer concern that the primary occupancy outcome

    occupied adult beds / staffed adult beds

shares a changing staffed-bed denominator with the capacity-based response
innovation exposure.

This script DOES NOT replace the locked v19 primary outcome. It adds
denominator-independent / fixed-denominator sensitivity outcomes.

Locked treatment
----------------
hospital_log_response_innovation

Locked treatment residual
-------------------------
treatment_resid
from v19 hospital-grouped cross-fitted orthogonalization.

Windows
-------
Pre:  event weeks -4,-3,-2
Post: event weeks +2,+3,+4

Sensitivity outcomes
--------------------
1) fixed_denominator_occupied_burden_change
      = (mean occupied beds post - mean occupied beds pre)
        / mean staffed beds pre

   This holds the denominator at PRE-surge capacity and therefore cannot
   decline mechanically because staffed capacity increases after the signal.

2) log_occupied_bed_census_change
      = 100 * [log(mean occupied beds post) - log(mean occupied beds pre)]

   Numerator-only proportional change in total occupied adult beds.

3) fixed_denominator_covid_census_change
      = (mean adult COVID inpatient census post - pre)
        / mean staffed beds pre

4) log1p_covid_census_change
      = 100 * [log(1 + post COVID census) - log(1 + pre COVID census)]

Estimator
---------
For each outcome:
- same v19 pretreatment covariates
- hospital-grouped 5-fold cross-fitting for the OUTCOME nuisance function
- locked v19 treatment residual (no treatment redefinition)
- residualized outcome ~ residualized treatment
- state-clustered standard errors
- prespecified 99% absolute-treatment-residual support sensitivity
- effect-size translation per 1 SD of the locked treatment residual

Interpretation
--------------
If numerator-only/fixed-denominator outcomes remain negative, this directly
addresses denominator coupling.

If they are null or positive, DO NOT describe that as disproving the primary
occupancy association. Capacity expansion can relieve proportional occupancy
while accommodating a stable or larger absolute patient census. Instead,
report that the primary signal is principally a capacity-relative strain
measure and explicitly acknowledge denominator coupling.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error


ROOT = Path(r"C:\Users\SIMONEY\Disease\AJPH_US_Empirical_Upgrade")
CLEAN = ROOT / "02_clean"
OUTPUT = ROOT / "04_outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

V19 = OUTPUT / "AJPH_v19_target_trial_crossfit_panel.csv"

HHS_CANDIDATES = [
    CLEAN / "hhs_facility_weekly_selected_clean_v9.csv",
    CLEAN / "hhs_facility_weekly_selected_clean.csv",
]

EXP = "hospital_log_response_innovation"
TRES = "treatment_resid"

PRE_WEEKS = [-4,-3,-2]
POST_WEEKS = [2,3,4]

N_SPLITS = 5
RF_TREES = 400
SEED = 20260905

NUM_COLS = [
    "occupancy_pre_mean",
    "occupancy_pre_slope",
    "occupancy_pre_volatility",
    "occupancy_week_m2",
    "admissions_pre_mean",
    "admissions_pre_slope",
    "staffed_beds_pre_mean",
    "icu_pre_mean",
    "icu_pre_slope",
    "signal_admissions_v13",
]

CAT_COLS = [
    "hospital_subtype",
    "region",
    "signal_quarter",
    "wave",
]


def norm(s):
    return re.sub(r"[^a-z0-9]+","_",str(s).lower()).strip("_")

def find_col(cols, candidates, required=True):
    nmap={norm(c):c for c in cols}
    for cand in candidates:
        cn=norm(cand)
        for nc,orig in nmap.items():
            if nc==cn or cn in nc:
                return orig
    if required:
        raise ValueError(
            f"Could not find column matching {candidates}\n"
            f"Available columns:\n{list(cols)}"
        )
    return None

def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_rf_pipeline(num_cols, cat_cols, seed):
    prep=ColumnTransformer([
        ("num", Pipeline([
            ("imputer",SimpleImputer(strategy="median"))
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",make_onehot())
        ]), cat_cols),
    ], remainder="drop")

    rf=RandomForestRegressor(
        n_estimators=RF_TREES,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=seed,
    )

    return Pipeline([
        ("prep",prep),
        ("rf",rf),
    ])

def cluster_fit(y, X, groups):
    y=np.asarray(y,float).reshape(-1)
    X=np.asarray(X,float)
    if X.ndim==1:
        X=X.reshape(-1,1)
    g=np.asarray(groups,object).reshape(-1)

    m=np.isfinite(y)&np.isfinite(X).all(axis=1)&pd.notna(g)
    y=y[m]; X=X[m]; g=g[m]

    if len(y)==0:
        raise ValueError("No finite observations.")
    if len(np.unique(g))<2:
        raise ValueError("Need >=2 state clusters.")

    return sm.OLS(y,X).fit(
        cov_type="cluster",
        cov_kwds={"groups":g}
    )

def crossfit_outcome(df, outcome):
    X=df[NUM_COLS+CAT_COLS].copy()
    y=df[outcome].to_numpy(float)
    groups=df["hospital_id"].astype(str).to_numpy()

    n_splits=min(N_SPLITS,len(np.unique(groups)))
    gkf=GroupKFold(n_splits=n_splits)
    pred=np.full(len(df),np.nan)

    for fold,(tr,te) in enumerate(gkf.split(X,y,groups=groups),1):
        pipe=make_rf_pipeline(
            NUM_COLS,CAT_COLS,
            SEED+1000+fold
        )
        pipe.fit(X.iloc[tr],y[tr])
        pred[te]=pipe.predict(X.iloc[te])

    if np.isnan(pred).any():
        raise ValueError(f"Missing predictions for {outcome}")

    return pred

def estimate(df, outcome, support_mask=None, label="primary"):
    d=df.copy()
    if support_mask is not None:
        d=d.loc[support_mask].copy()

    yres=d[f"{outcome}_resid"].to_numpy(float)
    tres=d[TRES].to_numpy(float)

    m=cluster_fit(
        yres,
        tres.reshape(-1,1),
        d["state"]
    )

    ci=m.conf_int()[0]
    sd_t=float(d[TRES].std(ddof=1))

    return {
        "label":label,
        "outcome":outcome,
        "n":int(len(d)),
        "hospitals":int(d["hospital_id"].nunique()),
        "states":int(d["state"].nunique()),
        "theta":float(m.params[0]),
        "se_cluster":float(m.bse[0]),
        "p_value":float(m.pvalues[0]),
        "ci_low":float(ci[0]),
        "ci_high":float(ci[1]),
        "treatment_resid_sd":sd_t,
        "effect_per_1SD_outcome_units":float(m.params[0]*sd_t),
        "ci_per_1SD_low":float(ci[0]*sd_t),
        "ci_per_1SD_high":float(ci[1]*sd_t),
    }


print("\n[1/7] LOADING V19 + HHS")

if not V19.exists():
    raise FileNotFoundError(f"{V19}\nRun v19 first.")

hhs_file=next((p for p in HHS_CANDIDATES if p.exists()),None)
if hhs_file is None:
    hits=list(ROOT.rglob("*facility*weekly*.csv"))
    if hits:
        hhs_file=hits[0]
if hhs_file is None:
    raise FileNotFoundError("Could not find HHS facility-week data.")

v19=pd.read_csv(V19,low_memory=False)
hhs=pd.read_csv(hhs_file,low_memory=False)

v19["hospital_id"]=v19["hospital_id"].astype(str).str.strip()
v19["state"]=v19["state"].astype(str).str.upper().str.strip()
v19["signal_week"]=pd.to_datetime(v19["signal_week"],errors="coerce")

id_col=find_col(hhs.columns,["hospital_pk","hospital_id","ccn","facility_id"])
week_col=find_col(hhs.columns,["collection_week","week"])
staffed_col=find_col(
    hhs.columns,
    ["all_adult_hospital_inpatient_beds_7_day_avg","staffed_adult_beds"]
)
occupied_col=find_col(
    hhs.columns,
    ["all_adult_hospital_inpatient_bed_occupied_7_day_avg","occupied_adult_beds"]
)
covid_census_col=find_col(
    hhs.columns,
    [
        "total_adult_patients_hospitalized_confirmed_covid_7_day_avg",
        "adult_covid_inpatient",
        "covid_inpatient"
    ],
    required=False
)

hhs["hospital_id"]=hhs[id_col].astype(str).str.strip()
hhs["week"]=pd.to_datetime(hhs[week_col],errors="coerce")
hhs["staffed_beds"]=pd.to_numeric(hhs[staffed_col],errors="coerce")
hhs["occupied_beds"]=pd.to_numeric(hhs[occupied_col],errors="coerce")
if covid_census_col:
    hhs["covid_census"]=pd.to_numeric(hhs[covid_census_col],errors="coerce")
else:
    hhs["covid_census"]=np.nan

print("V19 episodes:",len(v19))
print("Hospitals:",v19["hospital_id"].nunique())
print("States/DC:",v19["state"].nunique())
print("HHS file:",hhs_file)
print("COVID census field:",covid_census_col)


print("\n[2/7] CONSTRUCTING DENOMINATOR-INDEPENDENT OUTCOMES")

hhs_index={
    hid:g.sort_values("week")
    for hid,g in hhs.groupby("hospital_id")
}

rows=[]

for _,r in v19.iterrows():
    hid=str(r["hospital_id"])
    sig=pd.Timestamp(r["signal_week"])

    g=hhs_index.get(hid)
    if g is None or pd.isna(sig):
        continue

    x=g[
        (g["week"]>=sig-pd.Timedelta(weeks=4))
        &(g["week"]<=sig+pd.Timedelta(weeks=4))
    ].copy()

    if x.empty:
        continue

    x["event_week"]=np.round(
        (x["week"]-sig).dt.days/7
    ).astype(int)

    pre=x[x["event_week"].isin(PRE_WEEKS)]
    post=x[x["event_week"].isin(POST_WEEKS)]

    if pre["week"].nunique()<2 or post["week"].nunique()<2:
        continue

    pre_staff=pre["staffed_beds"].mean()
    pre_occ=pre["occupied_beds"].mean()
    post_occ=post["occupied_beds"].mean()

    pre_covid=pre["covid_census"].mean()
    post_covid=post["covid_census"].mean()

    fixed_occ=(
        (post_occ-pre_occ)/pre_staff
        if np.isfinite(pre_staff) and pre_staff>0
        and np.isfinite(pre_occ) and np.isfinite(post_occ)
        else np.nan
    )

    log_occ=(
        100.0*(np.log(post_occ)-np.log(pre_occ))
        if np.isfinite(pre_occ) and np.isfinite(post_occ)
        and pre_occ>0 and post_occ>0
        else np.nan
    )

    fixed_covid=(
        (post_covid-pre_covid)/pre_staff
        if np.isfinite(pre_staff) and pre_staff>0
        and np.isfinite(pre_covid) and np.isfinite(post_covid)
        else np.nan
    )

    log_covid=(
        100.0*(np.log1p(post_covid)-np.log1p(pre_covid))
        if np.isfinite(pre_covid) and np.isfinite(post_covid)
        and pre_covid>=0 and post_covid>=0
        else np.nan
    )

    rows.append({
        "hospital_id":hid,
        "signal_week":sig,
        "fixed_denominator_occupied_burden_change":fixed_occ,
        "log_occupied_bed_census_change":log_occ,
        "fixed_denominator_covid_census_change":fixed_covid,
        "log1p_covid_census_change":log_covid,
        "pre_staffed_beds_reconstructed":pre_staff,
        "pre_occupied_beds":pre_occ,
        "post_occupied_beds":post_occ,
        "pre_covid_census":pre_covid,
        "post_covid_census":post_covid,
    })

outcomes=pd.DataFrame(rows)

d=v19.merge(
    outcomes,
    on=["hospital_id","signal_week"],
    how="left"
)

OUTCOMES=[
    "fixed_denominator_occupied_burden_change",
    "log_occupied_bed_census_change",
]

if covid_census_col:
    OUTCOMES += [
        "fixed_denominator_covid_census_change",
        "log1p_covid_census_change",
    ]

print("Constructed outcome availability:")
for o in OUTCOMES:
    print(f"  {o}: {int(d[o].notna().sum())}")


print("\n[3/7] CROSS-FITTING OUTCOME NUISANCE FUNCTIONS")

results=[]
diag=[]

for o in OUTCOMES:
    need=[
        o,TRES,"hospital_id","state"
    ]+NUM_COLS+CAT_COLS

    od=(
        d.replace([np.inf,-np.inf],np.nan)
         .dropna(subset=[o,TRES,"hospital_id","state"])
         .copy()
    )

    # Numeric/categorical nuisance models impute covariates internally.
    # Ensure outcome itself finite.
    od=od[np.isfinite(pd.to_numeric(od[o],errors="coerce"))].copy()

    print(f"\nOutcome: {o}")
    print("  n:",len(od),
          "hospitals:",od["hospital_id"].nunique(),
          "states:",od["state"].nunique())

    pred=crossfit_outcome(od,o)
    od[f"{o}_hat"]=pred
    od[f"{o}_resid"]=od[o]-pred

    r2=float(r2_score(od[o],pred))
    rmse=float(math.sqrt(mean_squared_error(od[o],pred)))

    diag.append({
        "outcome":o,
        "n":int(len(od)),
        "hospitals":int(od["hospital_id"].nunique()),
        "states":int(od["state"].nunique()),
        "outcome_nuisance_r2":r2,
        "outcome_nuisance_rmse":rmse,
    })

    est=estimate(od,o,label="full_support")
    results.append(est)
    print(json.dumps(est,indent=2))

    p99=float(np.quantile(np.abs(od[TRES]),.99))
    keep=np.abs(od[TRES].to_numpy(float))<=p99

    est99=estimate(
        od,o,
        support_mask=keep,
        label="99pct_treatment_residual_support"
    )
    results.append(est99)
    print("  99% support:")
    print(json.dumps(est99,indent=2))

    # Save outcome-specific analytic panel.
    od.to_csv(
        OUTPUT/f"AJPH_v22_{o}_analytic_panel.csv",
        index=False
    )


print("\n[4/7] EFFECT-SIZE TRANSLATION")

res=pd.DataFrame(results)
diag_df=pd.DataFrame(diag)

# For fixed-denominator outcomes, effect_per_1SD_outcome_units is a proportion.
# Translate to percentage points.
for _,r in res.iterrows():
    if r["outcome"].startswith("fixed_denominator_"):
        print(
            f'{r["outcome"]} | {r["label"]}: '
            f'{100*r["effect_per_1SD_outcome_units"]:.3f} percentage points '
            f'per 1-SD response innovation '
            f'(95% CI {100*r["ci_per_1SD_low"]:.3f} to '
            f'{100*r["ci_per_1SD_high"]:.3f})'
        )
    else:
        print(
            f'{r["outcome"]} | {r["label"]}: '
            f'{r["effect_per_1SD_outcome_units"]:.3f} log-percent units '
            f'per 1-SD response innovation '
            f'(95% CI {r["ci_per_1SD_low"]:.3f} to '
            f'{r["ci_per_1SD_high"]:.3f})'
        )


print("\n[5/7] PRIMARY-OCCUPANCY REFERENCE")

primary_ref={
    "v19_theta":-0.0035222953999331486,
    "v19_ci_low":-0.004586335213029398,
    "v19_ci_high":-0.0024582555868368984,
    "v19_treatment_resid_sd":8.328461785754635,
}

primary_ref["effect_per_1SD_occupancy_proportion"]=(
    primary_ref["v19_theta"]*primary_ref["v19_treatment_resid_sd"]
)
primary_ref["effect_per_1SD_percentage_points"]=(
    100*primary_ref["effect_per_1SD_occupancy_proportion"]
)
primary_ref["ci_per_1SD_percentage_points"]=[
    100*primary_ref["v19_ci_low"]*primary_ref["v19_treatment_resid_sd"],
    100*primary_ref["v19_ci_high"]*primary_ref["v19_treatment_resid_sd"],
]

print(json.dumps(primary_ref,indent=2))


print("\n[6/7] REVIEWER-DECISION RULE")

rule={
    "if_fixed_denominator_occupied_burden_negative":
        "Denominator-independent evidence supports a genuine reduction in occupied-bed burden relative to fixed pre-surge capacity; denominator coupling is unlikely to fully explain the primary occupancy association.",
    "if_fixed_denominator_null_but_log_occupied_negative":
        "Absolute proportional census declined despite removal of the dynamic staffed-bed denominator; this still addresses the core coupling concern.",
    "if_both_occupied_bed_outcomes_null_or_positive":
        "Do not claim the primary occupancy result reflects fewer occupied beds. Interpret it as capacity-relative strain relief, acknowledge that part of the association may operate mechanically through the staffed-bed denominator, and retain these null sensitivity results transparently.",
    "covid_outcomes":
        "COVID census outcomes are secondary because all-cause adult bed occupancy can change for reasons beyond COVID admissions."
}
print(json.dumps(rule,indent=2))


print("\n[7/7] FINAL SUMMARY")

summary={
    "diagnostics":diag,
    "estimates":results,
    "primary_occupancy_reference":primary_ref,
    "interpretation_rule":rule,
}

res.to_csv(
    OUTPUT/"AJPH_v22_denominator_independent_estimates.csv",
    index=False
)
diag_df.to_csv(
    OUTPUT/"AJPH_v22_denominator_independent_diagnostics.csv",
    index=False
)
(OUTPUT/"AJPH_v22_denominator_independent_summary.json").write_text(
    json.dumps(summary,indent=2),
    encoding="utf-8"
)

print(json.dumps(summary,indent=2))
print("\nCOMPLETE")
print("Estimates:",
      OUTPUT/"AJPH_v22_denominator_independent_estimates.csv")
print("Diagnostics:",
      OUTPUT/"AJPH_v22_denominator_independent_diagnostics.csv")
print("Summary:",
      OUTPUT/"AJPH_v22_denominator_independent_summary.json")
