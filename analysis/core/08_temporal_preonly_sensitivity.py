# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH U.S. Empirical Upgrade v23
Temporal-Separation + Pretreatment-Only Response Sensitivity

Purpose
-------
Address two remaining reviewer concerns:

A. Exposure/outcome temporal overlap
   Locked response innovation uses post-surge staffed-capacity information from
   event weeks +1 and +2, whereas prior outcomes used weeks +2:+4.
   v23 moves outcomes strictly later to +3:+5 and +3:+6.

B. Post-signal demand in expected-response construction
   The locked v13 response innovation partially conditions expected post-capacity
   response on contemporaneous +1/+2 demand change. v23 constructs a second,
   pretreatment-only expected-response index using only information available at
   or before the surge signal.

This is a sensitivity analysis. It does not replace the locked v19 primary
analysis unless interpretation materially changes.

Inputs
------
1) 04_outputs/AJPH_v19_target_trial_crossfit_panel.csv
2) 04_outputs/AJPH_v13_hospital_log_response_panel.csv
3) HHS facility-week panel

Primary sensitivity outcomes
----------------------------
Strictly lagged windows:
- occupancy ratio change: pre (-4:-2) to post (+3:+5)
- fixed-precapacity occupied-bed burden: +3:+5
- log occupied-bed census change: +3:+5
- fixed-precapacity COVID census change: +3:+5
- log1p COVID census change: +3:+5

Extended window:
- same outcomes using +3:+6

Pretreatment-only response sensitivity
--------------------------------------
Post log-capacity response is residualized using ONLY:
- baseline demand
- baseline staffed capacity
- baseline admissions
- pre-surge demand slope
- signal admissions
- relative signal change
- calendar time
- signal quarter / wave
NO +1/+2 demand-change variable enters the expected-response model.

Pre response residual is constructed analogously using only pretreatment
predictors. Innovation = post residual - pre residual.

Inference
---------
- hospital-grouped cross-fitting for outcome nuisance functions
- state-clustered standard errors
- 99% absolute-treatment-residual support sensitivity
- 1-SD effect translations

Interpretation
--------------
A robust pattern would be:
- locked response: lagged occupancy ratio remains negative
- locked response: lagged absolute census/fixed-precapacity burden remains positive
- pretreatment-only response: same qualitative pattern

That pattern would support "association consistent with greater absorptive
capacity" while preserving observational language.
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
V13 = OUTPUT / "AJPH_v13_hospital_log_response_panel.csv"

HHS_CANDIDATES = [
    CLEAN / "hhs_facility_weekly_selected_clean_v9.csv",
    CLEAN / "hhs_facility_weekly_selected_clean.csv",
]

LOCKED_EXP = "hospital_log_response_innovation"
PREONLY_EXP = "hospital_log_response_innovation_preonly"

PRE_WEEKS = [-4,-3,-2]
POST_WINDOWS = {
    "lag35":[3,4,5],
    "lag36":[3,4,5,6],
}

N_SPLITS = 5
RF_TREES = 400
SEED = 20260906

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
        ("num",Pipeline([
            ("imputer",SimpleImputer(strategy="median"))
        ]),num_cols),
        ("cat",Pipeline([
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",make_onehot())
        ]),cat_cols),
    ],remainder="drop")

    rf=RandomForestRegressor(
        n_estimators=RF_TREES,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=seed,
    )

    return Pipeline([("prep",prep),("rf",rf)])

def cluster_fit(y,X,groups):
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
        raise ValueError("Need >=2 clusters.")

    return sm.OLS(y,X).fit(
        cov_type="cluster",
        cov_kwds={"groups":g}
    )

def grouped_crossfit_predict(df,target,num_cols,cat_cols,group_col,seed):
    X=df[num_cols+cat_cols].copy()
    y=pd.to_numeric(df[target],errors="coerce").to_numpy(float)
    groups=df[group_col].astype(str).to_numpy()

    gkf=GroupKFold(n_splits=min(N_SPLITS,len(np.unique(groups))))
    pred=np.full(len(df),np.nan)

    for fold,(tr,te) in enumerate(gkf.split(X,y,groups=groups),1):
        model=make_rf_pipeline(num_cols,cat_cols,seed+fold)
        model.fit(X.iloc[tr],y[tr])
        pred[te]=model.predict(X.iloc[te])

    if np.isnan(pred).any():
        raise ValueError(f"Crossfit failed for {target}")
    return pred

def loso_linear_expected(df,ycol,predictors,state_col="state"):
    """
    Leave-one-state-out expected value from linear model with numeric/categorical
    predictors encoded using pandas get_dummies on each training fold.
    """
    out=np.full(len(df),np.nan)

    for st in sorted(df[state_col].dropna().astype(str).unique()):
        te=(df[state_col].astype(str)==st).to_numpy()
        tr=~te

        train=df.loc[tr,[ycol]+predictors].copy()
        test=df.loc[te,predictors].copy()

        train=train.replace([np.inf,-np.inf],np.nan)
        test=test.replace([np.inf,-np.inf],np.nan)

        # Numeric median imputation using train statistics.
        Xtr=pd.DataFrame(index=train.index)
        Xte=pd.DataFrame(index=test.index)

        for c in predictors:
            if pd.api.types.is_numeric_dtype(df[c]):
                med=pd.to_numeric(train[c],errors="coerce").median()
                Xtr[c]=pd.to_numeric(train[c],errors="coerce").fillna(med)
                Xte[c]=pd.to_numeric(test[c],errors="coerce").fillna(med)
            else:
                trc=train[c].astype("object").where(train[c].notna(),"Missing").astype(str)
                tec=test[c].astype("object").where(test[c].notna(),"Missing").astype(str)
                comb=pd.concat([trc,tec],axis=0)
                dum=pd.get_dummies(comb,prefix=c,dtype=float)
                Xtr=pd.concat([Xtr,dum.loc[trc.index]],axis=1)
                Xte=pd.concat([Xte,dum.loc[tec.index]],axis=1)

        ytr=pd.to_numeric(train[ycol],errors="coerce")
        good=ytr.notna() & np.isfinite(ytr)

        Xtr=Xtr.loc[good]
        ytr=ytr.loc[good]

        # align columns and constant
        Xtr,Xte=Xtr.align(Xte,join="left",axis=1,fill_value=0)
        Xtr=sm.add_constant(Xtr,has_constant="add")
        Xte=sm.add_constant(Xte,has_constant="add")

        model=sm.OLS(ytr.to_numpy(float),Xtr.to_numpy(float)).fit()
        out[te]=model.predict(Xte.to_numpy(float))

    return out

def estimate_orthogonal(df,exp_col,outcome_col,label):
    d=df.replace([np.inf,-np.inf],np.nan).dropna(
        subset=[exp_col,outcome_col,"hospital_id","state"]
    ).copy()

    yhat=grouped_crossfit_predict(
        d,outcome_col,NUM_COLS,CAT_COLS,"hospital_id",
        SEED+2000
    )
    d["yres"]=d[outcome_col]-yhat

    # If using locked exposure, reuse v19 residual when available.
    if exp_col==LOCKED_EXP and "treatment_resid" in d.columns:
        d["tres"]=d["treatment_resid"]
    else:
        that=grouped_crossfit_predict(
            d,exp_col,NUM_COLS,CAT_COLS,"hospital_id",
            SEED+4000
        )
        d["tres"]=d[exp_col]-that

    fit=cluster_fit(
        d["yres"],
        d[["tres"]],
        d["state"]
    )
    ci=fit.conf_int()[0]
    sd_t=float(d["tres"].std(ddof=1))

    full={
        "label":label,
        "exposure":exp_col,
        "outcome":outcome_col,
        "n":int(len(d)),
        "hospitals":int(d["hospital_id"].nunique()),
        "states":int(d["state"].nunique()),
        "theta":float(fit.params[0]),
        "se_cluster":float(fit.bse[0]),
        "p_value":float(fit.pvalues[0]),
        "ci_low":float(ci[0]),
        "ci_high":float(ci[1]),
        "treatment_resid_sd":sd_t,
        "effect_per_1SD":float(fit.params[0]*sd_t),
        "ci_per_1SD_low":float(ci[0]*sd_t),
        "ci_per_1SD_high":float(ci[1]*sd_t),
        "outcome_nuisance_r2":float(r2_score(d[outcome_col],yhat)),
        "outcome_nuisance_rmse":float(
            math.sqrt(mean_squared_error(d[outcome_col],yhat))
        ),
    }

    # 99% treatment-residual support
    cutoff=float(np.quantile(np.abs(d["tres"]),.99))
    keep=np.abs(d["tres"].to_numpy(float))<=cutoff
    ds=d.loc[keep].copy()

    fit2=cluster_fit(
        ds["yres"],
        ds[["tres"]],
        ds["state"]
    )
    ci2=fit2.conf_int()[0]
    sd2=float(ds["tres"].std(ddof=1))

    support={
        "label":label+"_99support",
        "exposure":exp_col,
        "outcome":outcome_col,
        "n":int(len(ds)),
        "hospitals":int(ds["hospital_id"].nunique()),
        "states":int(ds["state"].nunique()),
        "theta":float(fit2.params[0]),
        "se_cluster":float(fit2.bse[0]),
        "p_value":float(fit2.pvalues[0]),
        "ci_low":float(ci2[0]),
        "ci_high":float(ci2[1]),
        "treatment_resid_sd":sd2,
        "effect_per_1SD":float(fit2.params[0]*sd2),
        "ci_per_1SD_low":float(ci2[0]*sd2),
        "ci_per_1SD_high":float(ci2[1]*sd2),
        "support_cutoff_abs_tres":cutoff,
    }

    return full,support,d


print("\n[1/8] LOADING INPUTS")

if not V19.exists():
    raise FileNotFoundError(V19)
if not V13.exists():
    raise FileNotFoundError(V13)

hhs_file=next((p for p in HHS_CANDIDATES if p.exists()),None)
if hhs_file is None:
    hits=list(ROOT.rglob("*facility*weekly*.csv"))
    if hits:
        hhs_file=hits[0]
if hhs_file is None:
    raise FileNotFoundError("No HHS facility-week panel found.")

v19=pd.read_csv(V19,low_memory=False)
v13=pd.read_csv(V13,low_memory=False)
hhs=pd.read_csv(hhs_file,low_memory=False)

for d in [v19,v13]:
    d["hospital_id"]=d["hospital_id"].astype(str).str.strip()
    d["state"]=d["state"].astype(str).str.upper().str.strip()
    d["signal_week"]=pd.to_datetime(d["signal_week"],errors="coerce")

print("v19 episodes:",len(v19))
print("v13 episodes:",len(v13))
print("HHS file:",hhs_file)


print("\n[2/8] BUILDING PRETREATMENT-ONLY RESPONSE INNOVATION")

# Use the v13 panel because it contains the locked response-construction ingredients.
post_y=find_col(
    v13.columns,
    ["post_log_capacity_change","capacity_post_log_change","observed_post_log_capacity_change"],
    required=False
)
pre_y=find_col(
    v13.columns,
    ["pre_log_capacity_change","capacity_pre_log_change","observed_pre_log_capacity_change"],
    required=False
)

if post_y is None:
    # v13 known fallback
    candidates=[c for c in v13.columns if "post" in c.lower() and "log" in c.lower() and "capacity" in c.lower()]
    if candidates: post_y=candidates[0]
if pre_y is None:
    candidates=[c for c in v13.columns if "pre" in c.lower() and "log" in c.lower() and "capacity" in c.lower()]
    if candidates: pre_y=candidates[0]

if post_y is None or pre_y is None:
    raise ValueError(
        "Could not identify observed post/pre log-capacity-change columns in v13.\n"
        "Inspect AJPH_v13_hospital_log_response_panel.csv columns."
    )

# Candidate pretreatment predictors using flexible discovery.
pred_map={}
searches={
    "baseline_demand":["baseline_demand","baseline_occupancy","baseline_occ"],
    "baseline_capacity":["baseline_capacity","baseline_staffed","staffed_beds_pre"],
    "baseline_admissions":["baseline_admissions","baseline_admit"],
    "pre_demand_slope":["pre_demand_slope","demand_pre_slope","admissions_pre_slope"],
    "signal_admissions":["signal_admissions","signal_admissions_7d"],
    "relative_signal_change":["relative_signal_change","signal_change","relative_change"],
    "calendar_time":["calendar_week_index","time_index","signal_time_index"],
    "signal_quarter":["signal_quarter","quarter"],
    "wave":["wave"],
}
for key,cands in searches.items():
    pred_map[key]=find_col(v13.columns,cands,required=False)

predictors=[c for c in pred_map.values() if c is not None]
if len(predictors)<4:
    raise ValueError(
        f"Too few pretreatment predictors found: {pred_map}\n"
        "Need at least baseline capacity/demand/signal/calendar variables."
    )

print("Observed post outcome:",post_y)
print("Observed pre outcome:",pre_y)
print("Pretreatment-only predictors:")
for k,v in pred_map.items():
    print(" ",k,":",v)

v13["expected_post_preonly"]=loso_linear_expected(
    v13,post_y,predictors,"state"
)
v13["expected_pre_preonly"]=loso_linear_expected(
    v13,pre_y,predictors,"state"
)

v13["post_resid_preonly"]=(
    pd.to_numeric(v13[post_y],errors="coerce")
    - v13["expected_post_preonly"]
)
v13["pre_resid_preonly"]=(
    pd.to_numeric(v13[pre_y],errors="coerce")
    - v13["expected_pre_preonly"]
)
v13[PREONLY_EXP]=v13["post_resid_preonly"]-v13["pre_resid_preonly"]

print(v13[PREONLY_EXP].describe(percentiles=[.01,.05,.5,.95,.99]))

preonly_keep=v13[
    ["hospital_id","signal_week",PREONLY_EXP]
].copy()

d=v19.merge(
    preonly_keep,
    on=["hospital_id","signal_week"],
    how="left"
)

print("Merged pre-only exposure nonmissing:",
      int(d[PREONLY_EXP].notna().sum()))


print("\n[3/8] PREPARING HHS WEEKLY OUTCOMES")

hid_col=find_col(hhs.columns,["hospital_pk","hospital_id","ccn","facility_id"])
week_col=find_col(hhs.columns,["collection_week","week"])
staffed_col=find_col(
    hhs.columns,
    ["all_adult_hospital_inpatient_beds_7_day_avg","staffed_adult_beds"]
)
occupied_col=find_col(
    hhs.columns,
    ["all_adult_hospital_inpatient_bed_occupied_7_day_avg","occupied_adult_beds"]
)
covid_col=find_col(
    hhs.columns,
    [
        "total_adult_patients_hospitalized_confirmed_covid_7_day_avg",
        "adult_covid_inpatient",
        "covid_inpatient"
    ],
    required=False
)

hhs["hospital_id"]=hhs[hid_col].astype(str).str.strip()
hhs["week"]=pd.to_datetime(hhs[week_col],errors="coerce")
hhs["staffed_beds"]=pd.to_numeric(hhs[staffed_col],errors="coerce")
hhs["occupied_beds"]=pd.to_numeric(hhs[occupied_col],errors="coerce")
if covid_col:
    hhs["covid_census"]=pd.to_numeric(hhs[covid_col],errors="coerce")
else:
    hhs["covid_census"]=np.nan

hhs_index={
    hid:g.sort_values("week")
    for hid,g in hhs.groupby("hospital_id")
}

rows=[]

for _,r in d.iterrows():
    hid=str(r["hospital_id"])
    sig=pd.Timestamp(r["signal_week"])
    g=hhs_index.get(hid)

    if g is None or pd.isna(sig):
        continue

    x=g[
        (g["week"]>=sig-pd.Timedelta(weeks=4))
        &(g["week"]<=sig+pd.Timedelta(weeks=6))
    ].copy()
    if x.empty:
        continue

    x["event_week"]=np.round(
        (x["week"]-sig).dt.days/7
    ).astype(int)

    pre=x[x["event_week"].isin(PRE_WEEKS)]
    pre_staff=pre["staffed_beds"].mean()
    pre_occbed=pre["occupied_beds"].mean()
    pre_covid=pre["covid_census"].mean()
    pre_ratio=(
        pre_occbed/pre_staff
        if np.isfinite(pre_occbed) and np.isfinite(pre_staff) and pre_staff>0
        else np.nan
    )

    rr={"hospital_id":hid,"signal_week":sig}

    for tag,weeks in POST_WINDOWS.items():
        post=x[x["event_week"].isin(weeks)]
        post_staff=post["staffed_beds"].mean()
        post_occbed=post["occupied_beds"].mean()
        post_covid=post["covid_census"].mean()
        post_ratio=(
            post_occbed/post_staff
            if np.isfinite(post_occbed) and np.isfinite(post_staff) and post_staff>0
            else np.nan
        )

        rr[f"{tag}_occupancy_ratio_change"]=(
            post_ratio-pre_ratio
            if np.isfinite(post_ratio) and np.isfinite(pre_ratio)
            else np.nan
        )
        rr[f"{tag}_fixed_pre_capacity_occupied_burden"]=(
            (post_occbed-pre_occbed)/pre_staff
            if np.isfinite(post_occbed) and np.isfinite(pre_occbed)
            and np.isfinite(pre_staff) and pre_staff>0
            else np.nan
        )
        rr[f"{tag}_log_occupied_census_change"]=(
            100*(np.log(post_occbed)-np.log(pre_occbed))
            if np.isfinite(post_occbed) and np.isfinite(pre_occbed)
            and post_occbed>0 and pre_occbed>0
            else np.nan
        )
        rr[f"{tag}_fixed_pre_capacity_covid_census"]=(
            (post_covid-pre_covid)/pre_staff
            if np.isfinite(post_covid) and np.isfinite(pre_covid)
            and np.isfinite(pre_staff) and pre_staff>0
            else np.nan
        )
        rr[f"{tag}_log1p_covid_census_change"]=(
            100*(np.log1p(post_covid)-np.log1p(pre_covid))
            if np.isfinite(post_covid) and np.isfinite(pre_covid)
            and post_covid>=0 and pre_covid>=0
            else np.nan
        )

    rows.append(rr)

lag=pd.DataFrame(rows)
d=d.merge(lag,on=["hospital_id","signal_week"],how="left")

print("Lagged outcome availability:")
for tag in POST_WINDOWS:
    for suffix in [
        "occupancy_ratio_change",
        "fixed_pre_capacity_occupied_burden",
        "log_occupied_census_change",
        "fixed_pre_capacity_covid_census",
        "log1p_covid_census_change",
    ]:
        c=f"{tag}_{suffix}"
        print(" ",c,":",int(d[c].notna().sum()))


print("\n[4/8] LOCKED-EXPOSURE STRICT-LAG ANALYSIS")

results=[]
analytic_panels=[]

for tag in POST_WINDOWS:
    for suffix in [
        "occupancy_ratio_change",
        "fixed_pre_capacity_occupied_burden",
        "log_occupied_census_change",
        "fixed_pre_capacity_covid_census",
        "log1p_covid_census_change",
    ]:
        outcome=f"{tag}_{suffix}"
        if d[outcome].notna().sum()<500:
            continue

        full,support,panel=estimate_orthogonal(
            d,LOCKED_EXP,outcome,
            f"locked_{tag}"
        )
        results += [full,support]
        analytic_panels.append(panel.assign(model_tag=f"locked_{tag}_{suffix}"))
        print(json.dumps(full,indent=2))
        print(json.dumps(support,indent=2))


print("\n[5/8] PRETREATMENT-ONLY EXPOSURE ANALYSIS")

for tag in POST_WINDOWS:
    for suffix in [
        "occupancy_ratio_change",
        "fixed_pre_capacity_occupied_burden",
        "log_occupied_census_change",
        "fixed_pre_capacity_covid_census",
        "log1p_covid_census_change",
    ]:
        outcome=f"{tag}_{suffix}"
        if d[outcome].notna().sum()<500:
            continue

        full,support,panel=estimate_orthogonal(
            d,PREONLY_EXP,outcome,
            f"preonly_{tag}"
        )
        results += [full,support]
        analytic_panels.append(panel.assign(model_tag=f"preonly_{tag}_{suffix}"))
        print(json.dumps(full,indent=2))
        print(json.dumps(support,indent=2))


print("\n[6/8] QUALITATIVE PATTERN CHECK")

res=pd.DataFrame(results)

def direction(theta):
    if theta<0: return "negative"
    if theta>0: return "positive"
    return "zero"

pattern_rows=[]
for exp in [LOCKED_EXP,PREONLY_EXP]:
    for tag in POST_WINDOWS:
        tmp=res[
            (res["exposure"]==exp)
            &(res["label"].str.endswith(tag))
        ]
        # full support labels only; support labels end with _99support
        if len(tmp)==0:
            continue

        p={}
        for _,r in tmp.iterrows():
            p[r["outcome"]]=direction(r["theta"])

        pattern_rows.append({
            "exposure":exp,
            "window":tag,
            "occupancy_ratio_direction":p.get(f"{tag}_occupancy_ratio_change"),
            "fixed_burden_direction":p.get(f"{tag}_fixed_pre_capacity_occupied_burden"),
            "occupied_census_direction":p.get(f"{tag}_log_occupied_census_change"),
            "covid_fixed_direction":p.get(f"{tag}_fixed_pre_capacity_covid_census"),
            "covid_log_direction":p.get(f"{tag}_log1p_covid_census_change"),
        })

pattern=pd.DataFrame(pattern_rows)
print(pattern.to_string(index=False))


print("\n[7/8] SAVING OUTPUTS")

res.to_csv(
    OUTPUT/"AJPH_v23_temporal_preonly_estimates.csv",
    index=False
)
pattern.to_csv(
    OUTPUT/"AJPH_v23_qualitative_pattern.csv",
    index=False
)

# Save one merged panel for audit.
d.to_csv(
    OUTPUT/"AJPH_v23_temporal_preonly_episode_panel.csv",
    index=False
)

summary={
    "pretreatment_only_predictor_map":pred_map,
    "observed_post_log_capacity_col":post_y,
    "observed_pre_log_capacity_col":pre_y,
    "results":results,
    "qualitative_pattern":pattern_rows,
    "interpretation_rule":{
        "strong_support":
            "Lagged occupancy ratio remains negative while fixed-denominator/absolute census remain positive under both locked and pretreatment-only exposure definitions.",
        "partial_support":
            "Pattern persists only for one exposure definition or one lag window; report as sensitivity with narrower language.",
        "failure":
            "Direction materially reverses or becomes unstable; do not use absorptive-capacity language as the principal empirical interpretation."
    }
}
(OUTPUT/"AJPH_v23_summary.json").write_text(
    json.dumps(summary,indent=2),
    encoding="utf-8"
)

print("Saved:")
print(OUTPUT/"AJPH_v23_temporal_preonly_estimates.csv")
print(OUTPUT/"AJPH_v23_qualitative_pattern.csv")
print(OUTPUT/"AJPH_v23_temporal_preonly_episode_panel.csv")
print(OUTPUT/"AJPH_v23_summary.json")


print("\n[8/8] FINAL REVIEWER DECISION RULE")
print(json.dumps(summary["interpretation_rule"],indent=2))
print("\nCOMPLETE")
