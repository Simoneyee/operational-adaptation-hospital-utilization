# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH U.S. Empirical Upgrade v19
Target-Trial Emulation + Cross-Fitted Orthogonal Estimation (Continuous Treatment)

Purpose
-------
Strengthen causal interpretation WITHOUT pretending to have a natural experiment.

This analysis emulates a hypothetical trial at the hospital × surge-episode level.

Eligibility:
    Hospital has a pre-defined surge signal episode already present in the locked v13 panel.

Time zero:
    Surge signal week.

Treatment:
    hospital_log_response_innovation
    (LOCKED from v13; never redefined in this script)

Primary outcome:
    Delta inpatient occupancy =
        mean occupancy weeks +2:+4
      - mean occupancy weeks -4:-2

Pretreatment covariates:
    - pre occupancy mean
    - pre occupancy slope
    - pre occupancy volatility
    - week -2 occupancy
    - pre COVID admissions mean
    - pre COVID admissions slope
    - pre staffed adult bed capacity
    - pre ICU occupancy mean
    - pre ICU occupancy slope
    - signal admissions
    - hospital subtype (if available)
    - census region
    - signal quarter / epidemic wave

Estimator:
    Partial-linear orthogonal / double machine learning:
        T = m(X) + v
        Y = g(X) + u
        u = theta * v + error

    Cross-fitting is grouped by HOSPITAL to prevent the same hospital appearing in
    both train and validation folds.

    m(X), g(X) are estimated using RandomForestRegressor pipelines with one-hot
    encoding for categorical covariates and median imputation for numeric covariates.

Inference:
    Final orthogonal score regression uses state-clustered standard errors.

Primary falsification tests:
    A) Negative-control outcome:
       pre-surge occupancy slope
    B) Temporal placebo:
       pre-surge mean occupancy
    C) Exposure residual balance:
       residualized treatment should have weak association with measured pretreatment covariates

Overlap / support audit:
    - reports distribution of cross-fitted treatment residual v
    - reports model R² for treatment and outcome nuisance models
    - prespecified sensitivity excludes ONLY observations with |v| above the 99th
      percentile of |v|, defined before looking at outcome estimates
    - no result-dependent trimming

Secondary learner sensitivity:
    HistGradientBoostingRegressor for numeric-only covariate representation.
    This is reported as a learner sensitivity, not selected based on significance.

Interpretation
--------------
This is still observational. The estimand can be interpreted causally only under
conditional exchangeability, consistency, positivity, and correct time ordering.
The script does not claim those assumptions are guaranteed.
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
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(r"C:\Users\SIMONEY\Disease\AJPH_US_Empirical_Upgrade")
CLEAN = ROOT / "02_clean"
OUTPUT = ROOT / "04_outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

V13_PANEL = OUTPUT / "AJPH_v13_hospital_log_response_panel.csv"

HHS_CANDIDATES = [
    CLEAN / "hhs_facility_weekly_selected_clean_v9.csv",
    CLEAN / "hhs_facility_weekly_selected_clean.csv",
]

EXP = "hospital_log_response_innovation"

PRE_WEEKS = [-4, -3, -2]
POST_WEEKS = [2, 3, 4]

N_SPLITS = 5
RF_TREES = 400
SEED = 20260905

VALID_US = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}

REGION = {
    "CT":"Northeast","ME":"Northeast","MA":"Northeast","NH":"Northeast","RI":"Northeast","VT":"Northeast",
    "NJ":"Northeast","NY":"Northeast","PA":"Northeast",
    "IN":"Midwest","IL":"Midwest","MI":"Midwest","OH":"Midwest","WI":"Midwest",
    "IA":"Midwest","KS":"Midwest","MN":"Midwest","MO":"Midwest","NE":"Midwest","ND":"Midwest","SD":"Midwest",
    "DE":"South","DC":"South","FL":"South","GA":"South","MD":"South","NC":"South","SC":"South","VA":"South","WV":"South",
    "AL":"South","KY":"South","MS":"South","TN":"South","AR":"South","LA":"South","OK":"South","TX":"South",
    "AZ":"West","CO":"West","ID":"West","MT":"West","NV":"West","NM":"West","UT":"West","WY":"West",
    "AK":"West","CA":"West","HI":"West","OR":"West","WA":"West"
}

if not V13_PANEL.exists():
    raise FileNotFoundError(V13_PANEL)

hhs_file = next((p for p in HHS_CANDIDATES if p.exists()), None)
if hhs_file is None:
    hits = list(ROOT.rglob("*facility*weekly*.csv"))
    if hits:
        hhs_file = hits[0]
if hhs_file is None:
    raise FileNotFoundError("Could not find HHS facility-week panel.")


# ==============================================================
# Helpers
# ==============================================================

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def find_col(cols, candidates, required=True):
    nmap = {norm(c): c for c in cols}
    for cand in candidates:
        cn = norm(cand)
        for nc, orig in nmap.items():
            if nc == cn or cn in nc:
                return orig
    if required:
        raise ValueError(
            f"Could not find column matching {candidates}\n"
            f"Available columns:\n{list(cols)}"
        )
    return None

def slope_from_xy(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])

def cluster_fit(y, X, groups):
    return sm.OLS(
        np.asarray(y, float),
        np.asarray(X, float)
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": np.asarray(groups, dtype=object)}
    )

def make_wave(d):
    d = pd.Timestamp(d)
    if d < pd.Timestamp("2020-10-01"):
        return "2020_summer"
    if d < pd.Timestamp("2021-03-01"):
        return "2020_21_winter"
    if d < pd.Timestamp("2021-07-01"):
        return "2021_spring"
    if d < pd.Timestamp("2021-11-01"):
        return "delta"
    if d < pd.Timestamp("2022-03-01"):
        return "omicron_BA1"
    if d < pd.Timestamp("2022-07-01"):
        return "omicron_BA2"
    return "omicron_later"

def make_onehot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

def make_rf_pipeline(num_cols, cat_cols, target_kind):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder())
    ])

    prep = ColumnTransformer(
        [
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols)
        ],
        remainder="drop"
    )

    # Separate random seeds for nuisance functions.
    rs = SEED + (11 if target_kind == "treatment" else 29)

    rf = RandomForestRegressor(
        n_estimators=RF_TREES,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=rs
    )

    return Pipeline([
        ("prep", prep),
        ("rf", rf)
    ])

def crossfit_rf(df, ycol, xcols_num, xcols_cat, group_col):
    X = df[xcols_num + xcols_cat].copy()
    y = df[ycol].to_numpy(float)
    groups = df[group_col].astype(str).to_numpy()

    unique_groups = np.unique(groups)
    n_splits = min(N_SPLITS, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Too few groups for cross-fitting.")

    gkf = GroupKFold(n_splits=n_splits)
    pred = np.full(len(df), np.nan)

    fold_rows = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        pipe = make_rf_pipeline(
            xcols_num,
            xcols_cat,
            "treatment" if ycol == EXP else "outcome"
        )
        pipe.fit(X.iloc[tr], y[tr])
        pred[te] = pipe.predict(X.iloc[te])

        fold_rows.append({
            "fold": fold,
            "train_n": int(len(tr)),
            "test_n": int(len(te)),
            "train_hospitals": int(len(np.unique(groups[tr]))),
            "test_hospitals": int(len(np.unique(groups[te]))),
        })

    if np.isnan(pred).any():
        raise ValueError(f"Cross-fitting failed for {ycol}: missing predictions.")

    return pred, pd.DataFrame(fold_rows)

def orthogonal_estimate(df, outcome_col, trim_mask=None, label="main"):
    d = df.copy()
    if trim_mask is not None:
        d = d.loc[trim_mask].copy()

    yres = d[f"{outcome_col}_resid"].to_numpy(float)
    tres = d["treatment_resid"].to_numpy(float)

    fit = cluster_fit(
        yres,
        tres.reshape(-1,1),
        d["state"]
    )

    ci = fit.conf_int()[0]

    return {
        "label": label,
        "outcome": outcome_col,
        "n": int(len(d)),
        "hospitals": int(d["hospital_id"].nunique()),
        "states": int(d["state"].nunique()),
        "theta": float(fit.params[0]),
        "se_cluster": float(fit.bse[0]),
        "p_value": float(fit.pvalues[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "treatment_resid_sd": float(np.std(tres, ddof=1)),
        "outcome_resid_sd": float(np.std(yres, ddof=1)),
        "std_theta": float(
            fit.params[0] * np.std(tres, ddof=1) / np.std(yres, ddof=1)
        ) if np.std(yres, ddof=1) > 0 else np.nan
    }

def numeric_only_hgb_crossfit(df, ycol, num_cols, group_col):
    X = df[num_cols].copy()
    y = df[ycol].to_numpy(float)
    groups = df[group_col].astype(str).to_numpy()

    # median imputation outside model but fit within fold
    unique_groups = np.unique(groups)
    n_splits = min(N_SPLITS, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.full(len(df), np.nan)

    for tr, te in gkf.split(X, y, groups=groups):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X.iloc[tr])
        Xte = imp.transform(X.iloc[te])

        model = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=SEED
        )
        model.fit(Xtr, y[tr])
        pred[te] = model.predict(Xte)

    if np.isnan(pred).any():
        raise ValueError(f"HGB cross-fitting failed for {ycol}.")
    return pred


# ==============================================================
# 1. Load v13 and HHS
# ==============================================================

print("\n[1/10] LOADING LOCKED V13 + HHS")

v13 = pd.read_csv(V13_PANEL, low_memory=False)
hhs = pd.read_csv(hhs_file, low_memory=False)

print("V13 rows:", len(v13))
print("HHS rows:", len(hhs))
print("HHS file:", hhs_file)

for d in [v13, hhs]:
    if "state" in d.columns:
        d["state"] = d["state"].astype(str).str.upper().str.strip()

v13 = v13[v13["state"].isin(VALID_US)].copy()
v13["hospital_id"] = v13["hospital_id"].astype(str).str.strip()
v13["signal_week"] = pd.to_datetime(v13["signal_week"], errors="coerce")

hhs_state = find_col(hhs.columns, ["state"])
hhs_id = find_col(hhs.columns, ["hospital_pk","hospital_id","ccn","facility_id"])
hhs_week = find_col(hhs.columns, ["collection_week","week"])
hhs_beds = find_col(
    hhs.columns,
    ["all_adult_hospital_inpatient_beds_7_day_avg","adult_inpatient_beds"]
)
hhs_occ_beds = find_col(
    hhs.columns,
    ["all_adult_hospital_inpatient_bed_occupied_7_day_avg","occupied_adult_beds"]
)
hhs_adm = find_col(
    hhs.columns,
    ["previous_day_admission_adult_covid_confirmed_7_day_sum","covid_admissions"]
)
hhs_icu_beds = find_col(
    hhs.columns,
    ["total_staffed_adult_icu_beds_7_day_avg","staffed_adult_icu_beds"],
    required=False
)
hhs_icu_occ = find_col(
    hhs.columns,
    ["staffed_adult_icu_bed_occupancy_7_day_avg","icu_bed_occupancy"],
    required=False
)
hhs_subtype = find_col(
    hhs.columns,
    ["hospital_subtype","hospital_type"],
    required=False
)

hhs["state"] = hhs[hhs_state].astype(str).str.upper().str.strip()
hhs["hospital_id"] = hhs[hhs_id].astype(str).str.strip()
hhs["week"] = pd.to_datetime(hhs[hhs_week], errors="coerce")
hhs["staffed_beds"] = pd.to_numeric(hhs[hhs_beds], errors="coerce")
hhs["occupied_beds"] = pd.to_numeric(hhs[hhs_occ_beds], errors="coerce")
hhs["occupancy"] = hhs["occupied_beds"] / hhs["staffed_beds"]
hhs.loc[
    (~np.isfinite(hhs["occupancy"]))
    | (hhs["staffed_beds"] <= 0)
    | (hhs["occupancy"] < 0)
    | (hhs["occupancy"] > 2),
    "occupancy"
] = np.nan

hhs["covid_admissions"] = pd.to_numeric(hhs[hhs_adm], errors="coerce")

if hhs_icu_beds and hhs_icu_occ:
    hhs["icu_beds"] = pd.to_numeric(hhs[hhs_icu_beds], errors="coerce")
    hhs["icu_occupied"] = pd.to_numeric(hhs[hhs_icu_occ], errors="coerce")
    # In the HHS file this field may already be a count, despite the name.
    # If values are mostly <=1.5, treat as ratio; otherwise occupied count / beds.
    q95 = hhs["icu_occupied"].quantile(.95)
    if np.isfinite(q95) and q95 <= 1.5:
        hhs["icu_occupancy"] = hhs["icu_occupied"]
    else:
        hhs["icu_occupancy"] = hhs["icu_occupied"] / hhs["icu_beds"]
    hhs.loc[
        (~np.isfinite(hhs["icu_occupancy"]))
        | (hhs["icu_occupancy"] < 0)
        | (hhs["icu_occupancy"] > 2),
        "icu_occupancy"
    ] = np.nan
else:
    hhs["icu_occupancy"] = np.nan

if hhs_subtype:
    hhs["hospital_subtype"] = hhs[hhs_subtype].astype(str)
else:
    hhs["hospital_subtype"] = "Unknown"


# ==============================================================
# 2. Construct pretreatment features and outcome
# ==============================================================

print("\n[2/10] CONSTRUCTING TARGET-TRIAL EPISODE PANEL")

# Merge each hospital episode to HHS weeks -4:+4.
episode_rows = []

hhs_index = {
    hid: g.sort_values("week")
    for hid, g in hhs.groupby("hospital_id")
}

for i, r in v13.iterrows():
    hid = str(r["hospital_id"])
    sig = pd.Timestamp(r["signal_week"])

    g = hhs_index.get(hid)
    if g is None or pd.isna(sig):
        continue

    x = g[
        (g["week"] >= sig - pd.Timedelta(weeks=4))
        & (g["week"] <= sig + pd.Timedelta(weeks=4))
    ].copy()

    if x.empty:
        continue

    x["event_week"] = np.round(
        (x["week"] - sig).dt.days / 7
    ).astype(int)

    pre = x[x["event_week"].isin(PRE_WEEKS)].copy()
    post = x[x["event_week"].isin(POST_WEEKS)].copy()

    if len(pre) < 2 or len(post) < 2:
        continue

    occ_pre_mean = pre["occupancy"].mean()
    occ_post_mean = post["occupancy"].mean()

    if not np.isfinite(occ_pre_mean) or not np.isfinite(occ_post_mean):
        continue

    occ_pre_slope = slope_from_xy(
        pre["event_week"],
        pre["occupancy"]
    )

    adm_pre_mean = pre["covid_admissions"].mean()
    adm_pre_slope = slope_from_xy(
        pre["event_week"],
        pre["covid_admissions"]
    )

    icu_pre_mean = pre["icu_occupancy"].mean()
    icu_pre_slope = slope_from_xy(
        pre["event_week"],
        pre["icu_occupancy"]
    )

    subtype_mode = pre["hospital_subtype"].mode()
    subtype = subtype_mode.iloc[0] if len(subtype_mode) else "Unknown"

    week_m2 = pre.loc[
        pre["event_week"] == -2,
        "occupancy"
    ].mean()

    episode_rows.append({
        "hospital_id": hid,
        "state": r["state"],
        "signal_week": sig,
        EXP: pd.to_numeric(r.get(EXP), errors="coerce"),
        "delta_occupancy": occ_post_mean - occ_pre_mean,
        "occupancy_pre_mean": occ_pre_mean,
        "occupancy_pre_slope": occ_pre_slope,
        "occupancy_pre_volatility": pre["occupancy"].std(ddof=1),
        "occupancy_week_m2": week_m2,
        "admissions_pre_mean": adm_pre_mean,
        "admissions_pre_slope": adm_pre_slope,
        "staffed_beds_pre_mean": pre["staffed_beds"].mean(),
        "icu_pre_mean": icu_pre_mean,
        "icu_pre_slope": icu_pre_slope,
        "signal_admissions_v13": pd.to_numeric(
            r.get("signal_admissions_7d", np.nan),
            errors="coerce"
        ),
        "hospital_subtype": subtype,
        "region": REGION.get(r["state"], "Unknown"),
        "signal_quarter": str(sig.to_period("Q")),
        "wave": make_wave(sig),
    })

epi = pd.DataFrame(episode_rows)

epi = epi.dropna(
    subset=[
        EXP,
        "delta_occupancy",
        "occupancy_pre_mean",
        "occupancy_pre_slope",
        "occupancy_pre_volatility",
        "occupancy_week_m2",
        "staffed_beds_pre_mean",
        "state",
        "hospital_id"
    ]
).copy()

print("Eligible episodes:", len(epi))
print("Hospitals:", epi["hospital_id"].nunique())
print("States:", epi["state"].nunique())

epi.to_csv(
    OUTPUT / "AJPH_v19_target_trial_episode_panel.csv",
    index=False
)


# ==============================================================
# 3. Define pretreatment covariates
# ==============================================================

print("\n[3/10] DEFINING PRETREATMENT COVARIATES")

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

print("Numeric covariates:", NUM_COLS)
print("Categorical covariates:", CAT_COLS)


# ==============================================================
# 4. Cross-fit nuisance models: treatment and primary outcome
# ==============================================================

print("\n[4/10] CROSS-FITTING RANDOM-FOREST NUISANCE MODELS")

t_hat, t_folds = crossfit_rf(
    epi, EXP, NUM_COLS, CAT_COLS, "hospital_id"
)

y_hat, y_folds = crossfit_rf(
    epi, "delta_occupancy", NUM_COLS, CAT_COLS, "hospital_id"
)

epi["treatment_hat"] = t_hat
epi["outcome_hat"] = y_hat
epi["treatment_resid"] = epi[EXP] - epi["treatment_hat"]
epi["delta_occupancy_resid"] = (
    epi["delta_occupancy"] - epi["outcome_hat"]
)

t_r2 = r2_score(epi[EXP], epi["treatment_hat"])
y_r2 = r2_score(epi["delta_occupancy"], epi["outcome_hat"])

t_rmse = math.sqrt(mean_squared_error(epi[EXP], epi["treatment_hat"]))
y_rmse = math.sqrt(
    mean_squared_error(epi["delta_occupancy"], epi["outcome_hat"])
)

print("Treatment nuisance cross-fit R2:", t_r2)
print("Treatment nuisance RMSE:", t_rmse)
print("Outcome nuisance cross-fit R2:", y_r2)
print("Outcome nuisance RMSE:", y_rmse)

t_folds.to_csv(
    OUTPUT / "AJPH_v19_crossfit_folds_treatment.csv",
    index=False
)
y_folds.to_csv(
    OUTPUT / "AJPH_v19_crossfit_folds_outcome.csv",
    index=False
)


# ==============================================================
# 5. Primary orthogonal estimate
# ==============================================================

print("\n[5/10] PRIMARY ORTHOGONAL ESTIMATE")

primary = orthogonal_estimate(
    epi,
    "delta_occupancy",
    label="RF_primary"
)

print(json.dumps(primary, indent=2))


# ==============================================================
# 6. Prespecified overlap audit + sensitivity
# ==============================================================

print("\n[6/10] OVERLAP / SUPPORT AUDIT")

abs_v = np.abs(epi["treatment_resid"].to_numpy(float))
v99 = float(np.quantile(abs_v, .99))

overlap = {
    "treatment_resid_mean": float(epi["treatment_resid"].mean()),
    "treatment_resid_sd": float(epi["treatment_resid"].std(ddof=1)),
    "abs_resid_p50": float(np.quantile(abs_v,.50)),
    "abs_resid_p90": float(np.quantile(abs_v,.90)),
    "abs_resid_p95": float(np.quantile(abs_v,.95)),
    "abs_resid_p99": v99,
    "abs_resid_max": float(np.max(abs_v)),
    "prespecified_keep_fraction": float((abs_v <= v99).mean()),
}

print(json.dumps(overlap, indent=2))

keep99 = abs_v <= v99

primary_overlap = orthogonal_estimate(
    epi,
    "delta_occupancy",
    trim_mask=keep99,
    label="RF_overlap_99pct_abs_treatment_residual"
)

print("Prespecified overlap sensitivity:")
print(json.dumps(primary_overlap, indent=2))


# ==============================================================
# 7. Falsification outcomes
# ==============================================================

print("\n[7/10] FALSIFICATION TESTS")

falsification_results = []

for outcome in [
    "occupancy_pre_slope",
    "occupancy_pre_mean",
    "occupancy_pre_volatility",
]:
    pred, _ = crossfit_rf(
        epi,
        outcome,
        NUM_COLS,
        CAT_COLS,
        "hospital_id"
    )

    epi[f"{outcome}_hat"] = pred
    epi[f"{outcome}_resid"] = epi[outcome] - pred

    rr = orthogonal_estimate(
        epi,
        outcome,
        label=f"negative_control_{outcome}"
    )
    falsification_results.append(rr)

    print(json.dumps(rr, indent=2))


# ==============================================================
# 8. Residual balance audit
# ==============================================================

print("\n[8/10] RESIDUALIZED EXPOSURE BALANCE AUDIT")

balance_rows = []

for c in NUM_COLS:
    x = epi[c].copy()
    m = x.notna() & epi["treatment_resid"].notna()

    if m.sum() < 50 or x[m].std(ddof=1) <= 0:
        continue

    fit = cluster_fit(
        epi.loc[m, "treatment_resid"],
        ((x[m] - x[m].mean()) / x[m].std(ddof=1)).to_numpy().reshape(-1,1),
        epi.loc[m, "state"]
    )

    ci = fit.conf_int()[0]

    balance_rows.append({
        "covariate": c,
        "coef_per_1sd": float(fit.params[0]),
        "se_cluster": float(fit.bse[0]),
        "p_value": float(fit.pvalues[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    })

balance = pd.DataFrame(balance_rows)

print(balance.to_string(index=False))

balance.to_csv(
    OUTPUT / "AJPH_v19_treatment_residual_balance.csv",
    index=False
)


# ==============================================================
# 9. Alternative learner sensitivity: HGB numeric-only
# ==============================================================

print("\n[9/10] ALTERNATIVE LEARNER SENSITIVITY")

# HGB is numeric-only here; categorical information is intentionally omitted
# as a learner sensitivity rather than used to redefine the primary estimator.
t_hat_hgb = numeric_only_hgb_crossfit(
    epi, EXP, NUM_COLS, "hospital_id"
)
y_hat_hgb = numeric_only_hgb_crossfit(
    epi, "delta_occupancy", NUM_COLS, "hospital_id"
)

epi["treatment_resid_hgb"] = epi[EXP] - t_hat_hgb
epi["delta_occupancy_resid_hgb"] = (
    epi["delta_occupancy"] - y_hat_hgb
)

hgb_fit = cluster_fit(
    epi["delta_occupancy_resid_hgb"],
    epi[["treatment_resid_hgb"]],
    epi["state"]
)
hci = hgb_fit.conf_int()[0]

hgb_result = {
    "label":"HGB_numeric_only_sensitivity",
    "n":int(len(epi)),
    "hospitals":int(epi["hospital_id"].nunique()),
    "states":int(epi["state"].nunique()),
    "theta":float(hgb_fit.params[0]),
    "se_cluster":float(hgb_fit.bse[0]),
    "p_value":float(hgb_fit.pvalues[0]),
    "ci_low":float(hci[0]),
    "ci_high":float(hci[1]),
    "treatment_nuisance_r2":float(r2_score(epi[EXP],t_hat_hgb)),
    "outcome_nuisance_r2":float(
        r2_score(epi["delta_occupancy"],y_hat_hgb)
    ),
}

print(json.dumps(hgb_result, indent=2))


# ==============================================================
# 10. Final summary
# ==============================================================

print("\n[10/10] FINAL SUMMARY")

epi.to_csv(
    OUTPUT / "AJPH_v19_target_trial_crossfit_panel.csv",
    index=False
)

summary = {
    "sample":{
        "episodes":int(len(epi)),
        "hospitals":int(epi["hospital_id"].nunique()),
        "states":int(epi["state"].nunique())
    },
    "locked_exposure":EXP,
    "primary_outcome":"delta_occupancy",
    "pretreatment_numeric_covariates":NUM_COLS,
    "pretreatment_categorical_covariates":CAT_COLS,
    "crossfit":{
        "grouping":"hospital_id",
        "n_splits":N_SPLITS,
        "treatment_nuisance_r2":float(t_r2),
        "treatment_nuisance_rmse":float(t_rmse),
        "outcome_nuisance_r2":float(y_r2),
        "outcome_nuisance_rmse":float(y_rmse)
    },
    "primary":primary,
    "overlap_audit":overlap,
    "overlap_sensitivity":primary_overlap,
    "falsification_results":falsification_results,
    "alternative_learner":hgb_result,
    "interpretation":{
        "causal_language_allowed_only_if":
            "conditional exchangeability, consistency, positivity, and correct time ordering are plausible; "
            "falsification outcomes are approximately null; results are stable across nuisance learners and overlap sensitivity",
        "default_wording":
            "cross-fitted orthogonal observational estimate"
    }
}

(OUTPUT / "AJPH_v19_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print(json.dumps(summary, indent=2))

print("\nCOMPLETE")
print("Episode panel:",
      OUTPUT/"AJPH_v19_target_trial_episode_panel.csv")
print("Cross-fit panel:",
      OUTPUT/"AJPH_v19_target_trial_crossfit_panel.csv")
print("Residual balance:",
      OUTPUT/"AJPH_v19_treatment_residual_balance.csv")
print("Summary:",
      OUTPUT/"AJPH_v19_summary.json")
