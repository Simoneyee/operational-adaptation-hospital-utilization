# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH U.S. Empirical Upgrade v20b
Mechanism + Heterogeneity + Unmeasured-Confounding Sensitivity + DML Replication Suite

This script is designed to be run AFTER v19.

Inputs
------
1) 04_outputs/AJPH_v19_target_trial_crossfit_panel.csv
2) 02_clean/hhs_facility_weekly_selected_clean_v9.csv
   (or a compatible HHS facility-week CSV)

Locked treatment
----------------
hospital_log_response_innovation

Locked primary outcome
----------------------
delta_occupancy =
    mean occupancy weeks +2:+4
  - mean occupancy weeks -4:-2

What v20 adds
-------------
A. Mechanism bridge
   A1. Response innovation -> early staffed-bed adaptation (+1:+2)
   A2. Response innovation -> early ICU-bed adaptation (+1:+2), if available
   A3. Early capacity adaptation -> later occupancy change (+2:+4 vs -4:-2),
       conditioning on response innovation and pre-surge covariates
   A4. Product-of-coefficients descriptive pathway decomposition.
       This is NOT labeled causal mediation.

B. Prespecified heterogeneity
   B1. Baseline occupancy (continuous standardized interaction)
   B2. Hospital size / staffed beds (continuous standardized interaction)
   B3. Epidemic wave (joint interaction test)
   B4. Hospital subtype, if informative (joint interaction test)
   Subgroup estimates are descriptive; interaction tests are primary.

C. Unmeasured-confounding sensitivity
   C1. Partial R^2 of treatment with outcome after cross-fitted nuisance adjustment
   C2. Cluster-based equal-strength omitted-confounder robustness benchmark
       using df = number of state clusters - 1.
   C3. Observed-covariate benchmark table: partial associations of each measured
       pretreatment covariate with treatment residual and outcome residual.

D. DML implementation replication
   D1. Repeated hospital-grouped cross-fitting (20 random hospital-fold assignments)
       with RandomForest nuisance models.
   D2. Reports distribution of theta across repetitions.
   D3. Optional DoubleML package replication if package is installed; script does
       not require DoubleML and will not fail if it is absent.

Interpretation
--------------
All analyses remain observational. Causal interpretation requires conditional
exchangeability, consistency, positivity, correct time ordering, and adequate
measurement of confounders.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score


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
Y = "delta_occupancy"

PRE_WEEKS = [-4, -3, -2]
EARLY_POST = [1, 2]
LATE_POST = [2, 3, 4]

N_REPEATS = 20
N_FOLDS = 5
RF_TREES = 300
SEED = 20260905

if not V19.exists():
    raise FileNotFoundError(
        f"{V19}\nRun v19 first."
    )

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

def cluster_fit(y, X, groups):
    """
    Cluster-robust OLS with explicit finite-row filtering.

    This prevents statsmodels MissingDataError when any residualized
    covariate contains NaN/inf. The same row mask is applied to y, X,
    and cluster groups.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    g_arr = np.asarray(groups, dtype=object).reshape(-1)

    finite = np.isfinite(y_arr) & np.isfinite(X_arr).all(axis=1)
    finite &= pd.notna(g_arr)

    y_arr = y_arr[finite]
    X_arr = X_arr[finite]
    g_arr = g_arr[finite]

    if len(y_arr) == 0:
        raise ValueError("cluster_fit received zero finite observations.")
    if len(np.unique(g_arr)) < 2:
        raise ValueError(
            f"cluster_fit requires >=2 clusters; got {len(np.unique(g_arr))}."
        )

    return sm.OLS(y_arr, X_arr).fit(
        cov_type="cluster",
        cov_kwds={"groups": g_arr}
    )

def iterative_absorb(df, cols, fe_cols, tol=1e-10, max_iter=500):
    if len(df) == 0:
        raise ValueError("Cannot residualize an empty dataset.")
    z = df[cols].astype(float).copy()
    for _ in range(max_iter):
        old = z.to_numpy(copy=True)
        for fe in fe_cols:
            z = z - z.groupby(df[fe], sort=False).transform("mean")
        diff = np.abs(z.to_numpy() - old)
        if np.nanmax(diff) < tol:
            break
    return z

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd

def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def rf_pipeline(num_cols, cat_cols, random_state):
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median"))
                ]),
                num_cols
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_onehot())
                ]),
                cat_cols
            )
        ],
        remainder="drop"
    )

    rf = RandomForestRegressor(
        n_estimators=RF_TREES,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=random_state
    )

    return Pipeline([
        ("prep", prep),
        ("rf", rf)
    ])

def random_group_folds(groups, n_folds, seed):
    groups = np.asarray(groups).astype(str)
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = ug.copy()
    rng.shuffle(shuffled)

    fold_map = {
        g: i % n_folds
        for i, g in enumerate(shuffled)
    }
    folds = np.array([fold_map[g] for g in groups], dtype=int)
    return folds

def repeated_crossfit_predictions(
    df, target, num_cols, cat_cols, group_col, seed
):
    groups = df[group_col].astype(str).to_numpy()
    folds = random_group_folds(groups, N_FOLDS, seed)
    pred = np.full(len(df), np.nan, float)

    X = df[num_cols + cat_cols].copy()
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(float)

    for fold in range(N_FOLDS):
        te = folds == fold
        tr = ~te

        model = rf_pipeline(
            num_cols,
            cat_cols,
            seed + 100 + fold
        )
        model.fit(X.loc[tr], y[tr])
        pred[te] = model.predict(X.loc[te])

    if np.isnan(pred).any():
        raise ValueError(f"Missing cross-fit predictions for {target}")
    return pred

def partial_r2_from_t(t, df):
    t = float(t)
    df = float(df)
    return (t*t) / (t*t + df)

def equal_strength_robustness_value(t, cluster_df):
    """
    Equal-strength omitted-confounder benchmark.

    Let f^2 = t^2 / df.
    Solve r^2 / (1-r) = f^2 for r in (0,1):
        r = (-f^2 + sqrt(f^4 + 4 f^2)) / 2

    This is reported as a cluster-based robustness benchmark, using
    df = number of clusters - 1. It is not a substitute for a full
    sensemakr implementation.
    """
    f2 = (float(t)**2) / float(cluster_df)
    r = (-f2 + math.sqrt(f2*f2 + 4*f2)) / 2
    return float(r)


# ==============================================================
# 1. Load v19 + HHS and construct early mediators
# ==============================================================

print("\n[1/9] LOADING V19 PANEL + HHS")

d = pd.read_csv(V19, low_memory=False)
hhs = pd.read_csv(hhs_file, low_memory=False)

d["hospital_id"] = d["hospital_id"].astype(str).str.strip()
d["state"] = d["state"].astype(str).str.upper().str.strip()
d["signal_week"] = pd.to_datetime(d["signal_week"], errors="coerce")

print("V19 episodes:", len(d))
print("Hospitals:", d["hospital_id"].nunique())
print("States:", d["state"].nunique())
print("HHS:", hhs_file)

hid_col = find_col(
    hhs.columns,
    ["hospital_pk","hospital_id","ccn","facility_id"]
)
week_col = find_col(
    hhs.columns,
    ["collection_week","week"]
)
beds_col = find_col(
    hhs.columns,
    ["all_adult_hospital_inpatient_beds_7_day_avg","adult_inpatient_beds"]
)
occ_beds_col = find_col(
    hhs.columns,
    [
        "all_adult_hospital_inpatient_bed_occupied_7_day_avg",
        "occupied_adult_beds"
    ]
)
icu_beds_col = find_col(
    hhs.columns,
    [
        "total_staffed_adult_icu_beds_7_day_avg",
        "staffed_adult_icu_beds"
    ],
    required=False
)

hhs["hospital_id"] = hhs[hid_col].astype(str).str.strip()
hhs["week"] = pd.to_datetime(hhs[week_col], errors="coerce")
hhs["staffed_beds"] = pd.to_numeric(hhs[beds_col], errors="coerce")
hhs["occupied_beds"] = pd.to_numeric(hhs[occ_beds_col], errors="coerce")

if icu_beds_col:
    hhs["staffed_icu_beds"] = pd.to_numeric(
        hhs[icu_beds_col],
        errors="coerce"
    )
else:
    hhs["staffed_icu_beds"] = np.nan

hhs_index = {
    hid: g.sort_values("week")
    for hid, g in hhs.groupby("hospital_id")
}

med_rows = []

for _, r in d.iterrows():
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

    pre = x[x["event_week"].isin(PRE_WEEKS)]
    early = x[x["event_week"].isin(EARLY_POST)]

    pre_beds = pre["staffed_beds"].mean()
    early_beds = early["staffed_beds"].mean()

    pre_icu = pre["staffed_icu_beds"].mean()
    early_icu = early["staffed_icu_beds"].mean()

    med_rows.append({
        "hospital_id": hid,
        "signal_week": sig,
        "early_log_bed_change": (
            np.log(early_beds) - np.log(pre_beds)
            if np.isfinite(pre_beds) and np.isfinite(early_beds)
            and pre_beds > 0 and early_beds > 0
            else np.nan
        ),
        "early_log_icu_bed_change": (
            np.log(early_icu) - np.log(pre_icu)
            if np.isfinite(pre_icu) and np.isfinite(early_icu)
            and pre_icu > 0 and early_icu > 0
            else np.nan
        )
    })

med = pd.DataFrame(med_rows)

d = d.merge(
    med,
    on=["hospital_id","signal_week"],
    how="left"
)

print("Early bed mediator nonmissing:",
      int(d["early_log_bed_change"].notna().sum()))
print("Early ICU-bed mediator nonmissing:",
      int(d["early_log_icu_bed_change"].notna().sum()))


# ==============================================================
# 2. Mechanism bridge
# ==============================================================

print("\n[2/9] MECHANISM BRIDGE")

mechanism_rows = []

# Use v19 orthogonalized treatment residual.
TRES = "treatment_resid"
YRES = "delta_occupancy_resid"

# A1 response -> mediator
for mediator in [
    "early_log_bed_change",
    "early_log_icu_bed_change"
]:
    md = d.dropna(
        subset=[mediator, TRES, YRES, "state"]
    ).copy()

    if len(md) < 500:
        continue

    # Residualize mediator on pre-treatment numeric covariates + quarter FE
    controls = [
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
    controls = [c for c in controls if c in md.columns]

    # IMPORTANT: iterative demeaning preserves NaNs, so use a complete-case
    # mechanism sample for the mediator model. This fixes the v20
    # MissingDataError caused by NaN/inf values in exogenous variables.
    mech_need = [mediator, TRES, YRES, "state", "signal_quarter"] + controls
    md = md.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(dict.fromkeys(mech_need))
    ).copy()

    if len(md) < 500:
        print(
            f"  {mediator}: skipped after complete-case filtering; "
            f"n={len(md)}"
        )
        continue

    md["signal_quarter_fe"] = md["signal_quarter"].astype(str)

    rz = iterative_absorb(
        md,
        [mediator, TRES] + controls,
        ["signal_quarter_fe"]
    )

    X = rz[[TRES] + controls]
    m1 = cluster_fit(
        rz[mediator],
        X,
        md["state"]
    )

    ci = m1.conf_int()[0]

    a = float(m1.params[0])

    # A2 mediator -> primary outcome residual, conditioning on T residual.
    # Residualize mediator itself against same pre-treatment controls.
    med_resid = rz[mediator].to_numpy(float)
    tres = rz[TRES].to_numpy(float)

    # align primary outcome residual from original rows
    yres = md[YRES].to_numpy(float)

    X2 = np.column_stack([tres, med_resid])
    m2 = cluster_fit(
        yres,
        X2,
        md["state"]
    )

    b = float(m2.params[1])
    b_ci = m2.conf_int()[1]

    total_theta = float(
        cluster_fit(
            yres,
            tres.reshape(-1,1),
            md["state"]
        ).params[0]
    )

    product = a * b
    fraction = (
        product / total_theta
        if np.isfinite(total_theta) and abs(total_theta) > 1e-12
        else np.nan
    )

    mechanism_rows.append({
        "mediator": mediator,
        "n": int(len(md)),
        "states": int(md["state"].nunique()),
        "a_response_to_mediator": a,
        "a_se": float(m1.bse[0]),
        "a_p": float(m1.pvalues[0]),
        "a_ci_low": float(ci[0]),
        "a_ci_high": float(ci[1]),
        "b_mediator_to_outcome_cond_response": b,
        "b_se": float(m2.bse[1]),
        "b_p": float(m2.pvalues[1]),
        "b_ci_low": float(b_ci[0]),
        "b_ci_high": float(b_ci[1]),
        "product_ab_descriptive": float(product),
        "total_theta_same_sample": float(total_theta),
        "descriptive_fraction_ab_over_total": float(fraction)
            if np.isfinite(fraction) else np.nan
    })

mechanism = pd.DataFrame(mechanism_rows)
print(mechanism.to_string(index=False))

mechanism.to_csv(
    OUTPUT / "AJPH_v20_mechanism_bridge.csv",
    index=False
)


# ==============================================================
# 3. Heterogeneity: baseline occupancy + size
# ==============================================================

print("\n[3/9] PRESPECIFIED CONTINUOUS HETEROGENEITY")

het_rows = []

for mod in [
    "occupancy_pre_mean",
    "staffed_beds_pre_mean"
]:
    hd = (
        d.replace([np.inf, -np.inf], np.nan)
         .dropna(subset=[YRES, TRES, mod, "state"])
         .copy()
    )

    hd["modifier_z"] = zscore(hd[mod])
    hd["interaction"] = hd[TRES] * hd["modifier_z"]

    X = np.column_stack([
        hd[TRES].to_numpy(float),
        hd["interaction"].to_numpy(float)
    ])

    m = cluster_fit(
        hd[YRES],
        X,
        hd["state"]
    )

    ci0 = m.conf_int()[0]
    ci1 = m.conf_int()[1]

    het_rows.append({
        "modifier": mod,
        "n": int(len(hd)),
        "theta_at_modifier_mean": float(m.params[0]),
        "theta_main_p": float(m.pvalues[0]),
        "interaction_beta": float(m.params[1]),
        "interaction_se": float(m.bse[1]),
        "interaction_p": float(m.pvalues[1]),
        "interaction_ci_low": float(ci1[0]),
        "interaction_ci_high": float(ci1[1]),
        "theta_at_minus1sd": float(m.params[0] - m.params[1]),
        "theta_at_plus1sd": float(m.params[0] + m.params[1]),
    })

het_cont = pd.DataFrame(het_rows)
print(het_cont.to_string(index=False))


# ==============================================================
# 4. Heterogeneity: epidemic wave + hospital subtype
# ==============================================================

print("\n[4/9] CATEGORICAL HETEROGENEITY")

cat_het_rows = []

for mod in ["wave","hospital_subtype"]:
    if mod not in d.columns:
        continue

    hd = (
        d.replace([np.inf, -np.inf], np.nan)
         .dropna(subset=[YRES, TRES, mod, "state"])
         .copy()
    )

    # Collapse rare subtype categories.
    if mod == "hospital_subtype":
        counts = hd[mod].value_counts()
        keep = counts[counts >= 200].index
        hd[mod] = np.where(
            hd[mod].isin(keep),
            hd[mod].astype(str),
            "Other"
        )

    levels = sorted(hd[mod].astype(str).unique())
    if len(levels) < 2:
        continue

    ref = levels[0]
    cols = [hd[TRES].to_numpy(float)]
    names = ["treatment_resid"]

    for lev in levels[1:]:
        inter = (
            hd[TRES].to_numpy(float)
            * (hd[mod].astype(str).to_numpy() == lev).astype(float)
        )
        cols.append(inter)
        names.append(f"T_x_{mod}_{lev}")

    X = np.column_stack(cols)
    m = cluster_fit(
        hd[YRES],
        X,
        hd["state"]
    )

    if len(levels) > 1:
        R = np.zeros((len(levels)-1, X.shape[1]))
        for i in range(len(levels)-1):
            R[i, i+1] = 1
        wald = m.wald_test(R, scalar=True)
        joint_p = float(wald.pvalue)
        joint_stat = float(wald.statistic)
    else:
        joint_p = np.nan
        joint_stat = np.nan

    # implied theta by level
    base = float(m.params[0])
    for i, lev in enumerate(levels):
        if i == 0:
            theta = base
        else:
            theta = base + float(m.params[i])

        cat_het_rows.append({
            "modifier": mod,
            "reference_level": ref,
            "level": lev,
            "n": int(len(hd)),
            "theta_implied": float(theta),
            "joint_interaction_stat": joint_stat,
            "joint_interaction_p": joint_p
        })

cat_het = pd.DataFrame(cat_het_rows)
print(cat_het.to_string(index=False))

het_cont.to_csv(
    OUTPUT / "AJPH_v20_heterogeneity_continuous.csv",
    index=False
)
cat_het.to_csv(
    OUTPUT / "AJPH_v20_heterogeneity_categorical.csv",
    index=False
)


# ==============================================================
# 5. Unmeasured-confounding sensitivity
# ==============================================================

print("\n[5/9] UNMEASURED-CONFOUNDING SENSITIVITY")

sd = (
    d.replace([np.inf, -np.inf], np.nan)
     .dropna(subset=[YRES, TRES, "state"])
     .copy()
)

main = cluster_fit(
    sd[YRES],
    sd[[TRES]],
    sd["state"]
)

theta = float(main.params[0])
se = float(main.bse[0])
tstat = theta / se
G = int(sd["state"].nunique())
cluster_df = G - 1

partial_r2 = partial_r2_from_t(
    tstat,
    cluster_df
)
rv = equal_strength_robustness_value(
    tstat,
    cluster_df
)

sens = {
    "theta": theta,
    "cluster_se": se,
    "t_stat": float(tstat),
    "clusters": G,
    "cluster_df_used": cluster_df,
    "partial_r2_treatment_outcome_conditional":
        float(partial_r2),
    "equal_strength_cluster_based_robustness_benchmark":
        float(rv),
    "note":
        "Robustness benchmark uses state-cluster df and equal-strength omitted-confounder assumption; descriptive sensitivity metric, not a full sensemakr replacement."
}

print(json.dumps(sens, indent=2))


# ==============================================================
# 6. Observed-covariate benchmark table
# ==============================================================

print("\n[6/9] OBSERVED-COVARIATE BENCHMARKS")

benchmark_rows = []

num_covars = [
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

for c in num_covars:
    if c not in d.columns:
        continue

    bd = (
        d.replace([np.inf, -np.inf], np.nan)
         .dropna(subset=[c,TRES,YRES,"state"])
         .copy()
    )

    if len(bd) < 500:
        continue

    cz = zscore(bd[c]).to_numpy(float)

    mt = cluster_fit(
        bd[TRES],
        cz.reshape(-1,1),
        bd["state"]
    )
    my = cluster_fit(
        bd[YRES],
        cz.reshape(-1,1),
        bd["state"]
    )

    benchmark_rows.append({
        "covariate": c,
        "treatment_resid_beta_per1sd": float(mt.params[0]),
        "treatment_resid_p": float(mt.pvalues[0]),
        "outcome_resid_beta_per1sd": float(my.params[0]),
        "outcome_resid_p": float(my.pvalues[0]),
    })

bench = pd.DataFrame(benchmark_rows)
print(bench.to_string(index=False))

bench.to_csv(
    OUTPUT / "AJPH_v20_unmeasured_confounding_benchmarks.csv",
    index=False
)


# ==============================================================
# 7. Repeated DML implementation replication
# ==============================================================

print("\n[7/9] REPEATED DML REPLICATION")

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

rd = (
    d.replace([np.inf, -np.inf], np.nan)
     .dropna(subset=[EXP,Y,"hospital_id","state"])
     .copy()
)

rep_rows = []

for rep in range(N_REPEATS):
    seed = SEED + 1000*rep

    t_hat = repeated_crossfit_predictions(
        rd, EXP, NUM_COLS, CAT_COLS,
        "hospital_id", seed
    )
    y_hat = repeated_crossfit_predictions(
        rd, Y, NUM_COLS, CAT_COLS,
        "hospital_id", seed + 17
    )

    tres = rd[EXP].to_numpy(float) - t_hat
    yres = rd[Y].to_numpy(float) - y_hat

    fit = cluster_fit(
        yres,
        tres.reshape(-1,1),
        rd["state"]
    )

    ci = fit.conf_int()[0]

    rep_rows.append({
        "rep": rep + 1,
        "seed": seed,
        "theta": float(fit.params[0]),
        "se_cluster": float(fit.bse[0]),
        "p_value": float(fit.pvalues[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "treatment_nuisance_r2": float(
            r2_score(rd[EXP], t_hat)
        ),
        "outcome_nuisance_r2": float(
            r2_score(rd[Y], y_hat)
        ),
    })

    print(
        f"  rep {rep+1:02d}/{N_REPEATS}: "
        f"theta={fit.params[0]:.6f}, p={fit.pvalues[0]:.3g}"
    )

rep = pd.DataFrame(rep_rows)

rep_summary = {
    "n_repeats": int(len(rep)),
    "theta_mean": float(rep["theta"].mean()),
    "theta_median": float(rep["theta"].median()),
    "theta_sd_across_repeats": float(rep["theta"].std(ddof=1)),
    "theta_min": float(rep["theta"].min()),
    "theta_max": float(rep["theta"].max()),
    "fraction_negative": float((rep["theta"] < 0).mean()),
    "fraction_p_lt_0_05": float((rep["p_value"] < .05).mean()),
    "mean_treatment_nuisance_r2":
        float(rep["treatment_nuisance_r2"].mean()),
    "mean_outcome_nuisance_r2":
        float(rep["outcome_nuisance_r2"].mean()),
}

print(json.dumps(rep_summary, indent=2))

rep.to_csv(
    OUTPUT / "AJPH_v20_repeated_DML_replication.csv",
    index=False
)


# ==============================================================
# 8. Optional DoubleML replication
# ==============================================================

print("\n[8/9] OPTIONAL DOUBLEML PACKAGE REPLICATION")

doubleml_result = {
    "available": False,
    "ran": False
}

try:
    import doubleml as dml
    from sklearn.ensemble import RandomForestRegressor

    # Numeric-only to avoid package-version-specific categorical handling.
    dd = rd[
        [Y,EXP,"state","hospital_id"] + NUM_COLS
    ].copy()

    for c in NUM_COLS:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")
        dd[c] = dd[c].fillna(dd[c].median())

    data_dml = dml.DoubleMLData(
        dd,
        y_col=Y,
        d_cols=EXP,
        x_cols=NUM_COLS
    )

    ml_l = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=SEED
    )
    ml_m = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=SEED+1
    )

    obj = dml.DoubleMLPLR(
        data_dml,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=5,
        n_rep=1
    )
    obj.fit()

    doubleml_result = {
        "available": True,
        "ran": True,
        "coef": float(np.asarray(obj.coef).ravel()[0]),
        "se": float(np.asarray(obj.se).ravel()[0]),
    }

except ImportError:
    doubleml_result = {
        "available": False,
        "ran": False,
        "note":
            "doubleml package not installed; internal repeated DML replication completed successfully and is the required analysis."
    }
except Exception as e:
    doubleml_result = {
        "available": True,
        "ran": False,
        "error": repr(e),
        "note":
            "Optional package replication failed; required internal DML replication remains valid."
    }

print(json.dumps(doubleml_result, indent=2))


# ==============================================================
# 9. Final summary
# ==============================================================

print("\n[9/9] FINAL SUMMARY")

summary = {
    "mechanism": mechanism_rows,
    "heterogeneity_continuous":
        het_cont.to_dict(orient="records"),
    "heterogeneity_categorical":
        cat_het.to_dict(orient="records"),
    "unmeasured_confounding_sensitivity":
        sens,
    "repeated_dml":
        rep_summary,
    "optional_doubleml":
        doubleml_result,
    "interpretation_rules":{
        "mechanism":
            "Product-of-coefficients results are descriptive pathway decomposition, not causal mediation.",
        "heterogeneity":
            "Interaction tests are primary; subgroup point estimates are descriptive.",
        "unmeasured_confounding":
            "Sensitivity metrics quantify robustness under stated assumptions but cannot rule out unmeasured confounding.",
        "dml":
            "Repeated cross-fitting is an implementation/stability replication, not a new independent identification design."
    }
}

(OUTPUT / "AJPH_v20_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print(json.dumps(summary, indent=2))

print("\nCOMPLETE")
print("Mechanism:",
      OUTPUT/"AJPH_v20_mechanism_bridge.csv")
print("Continuous heterogeneity:",
      OUTPUT/"AJPH_v20_heterogeneity_continuous.csv")
print("Categorical heterogeneity:",
      OUTPUT/"AJPH_v20_heterogeneity_categorical.csv")
print("Confounding benchmarks:",
      OUTPUT/"AJPH_v20_unmeasured_confounding_benchmarks.csv")
print("Repeated DML:",
      OUTPUT/"AJPH_v20_repeated_DML_replication.csv")
print("Summary:",
      OUTPUT/"AJPH_v20_summary.json")
