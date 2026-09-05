# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH U.S. Empirical Upgrade v21
Publication-Grade Sensitivity + Standard DoubleML Replication

Run AFTER v19/v20.

What this script does
---------------------
1) Standard Cinelli-Hazlett / sensemakr-style sensitivity statistics
   applied to the orthogonal score regression:
       Y_resid = theta * T_resid + error

   IMPORTANT:
   - The publication-grade sensemakr statistics are computed using the
     conventional linear-model standard error and residual degrees of freedom,
     exactly matching the assumptions of the Cinelli-Hazlett formulas.
   - The state-clustered t statistic is reported separately as the primary
     inferential result, but is NOT substituted into the standard sensemakr
     formula.

2) Optional package verification with PySensemakr if installed.
   The script also implements the official formulas directly, so PySensemakr
   is not required.

3) Standard DoubleMLPLR replication.
   - Uses the official DoubleML package.
   - Continuous treatment.
   - RandomForestRegressor nuisance learners.
   - 5-fold hospital-grouped external sample splitting.
   - 10 repeated hospital-grouped partitions.
   - Coefficient replication is the main purpose.
   - Primary state-clustered inference remains the v19 orthogonal estimate.

If DoubleML is missing, the script prints the exact install command instead of
crashing. Set AUTO_INSTALL_DOUBLEML=True below if you want the script to install
it automatically into the current Python environment.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(r"C:\Users\SIMONEY\Disease\AJPH_US_Empirical_Upgrade")
OUTPUT = ROOT / "04_outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

V19 = OUTPUT / "AJPH_v19_target_trial_crossfit_panel.csv"

EXP = "hospital_log_response_innovation"
Y = "delta_occupancy"

N_FOLDS = 5
N_REP = 10
SEED = 20260905

AUTO_INSTALL_DOUBLEML = False

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

if not V19.exists():
    raise FileNotFoundError(f"{V19}\nRun v19 first.")


# ==============================================================
# Helpers
# ==============================================================

def cluster_fit(y, X, groups):
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1,1)
    g = np.asarray(groups, object)

    m = np.isfinite(y) & np.isfinite(X).all(axis=1) & pd.notna(g)
    y = y[m]
    X = X[m]
    g = g[m]

    return sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": g}
    )

def iid_fit(y, X):
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1,1)

    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return sm.OLS(y[m], X[m]).fit()

def partial_r2(t_statistic, dof):
    t_statistic = float(t_statistic)
    dof = float(dof)
    return (t_statistic**2) / (t_statistic**2 + dof)

def extreme_robustness_value(t_statistic, dof, q=1.0, alpha=1.0):
    """
    Exact translation of sensemakr::extreme_robustness_value.numeric
    for a scalar t statistic.
    """
    t_statistic = float(t_statistic)
    dof = float(dof)

    fq = q * abs(t_statistic / math.sqrt(dof))
    fq2 = fq**2

    if alpha >= 1:
        fcrit2 = 0.0
    else:
        tcrit = abs(student_t.ppf(alpha/2, df=dof-1))
        fcrit2 = (tcrit / math.sqrt(dof-1))**2

    f1 = fq2
    f2 = fcrit2
    xrv = (f1 - f2) / (1 + f1)

    if f1 <= f2:
        return 0.0
    return float(xrv)

def robustness_value(t_statistic, dof, q=1.0, alpha=1.0):
    """
    Exact scalar translation of sensemakr::robustness_value.numeric.

    RV is the minimum equal-strength partial R^2 that an omitted confounder
    would need with BOTH treatment and outcome to reduce the estimate by q.
    """
    t_statistic = float(t_statistic)
    dof = float(dof)

    fq = q * abs(t_statistic / math.sqrt(dof))

    if alpha >= 1:
        fcrit = 0.0
    else:
        fcrit = abs(student_t.ppf(alpha/2, df=dof-1)) / math.sqrt(dof-1)

    f1 = fq
    f2 = fcrit
    fqa = f1 - f2

    if fqa < 0:
        return 0.0

    if abs(fqa) < 1e-15:
        rv = 0.0
    else:
        rv = 2 / (1 + math.sqrt(1 + 4/(fqa**2)))

    xrv = extreme_robustness_value(
        t_statistic=t_statistic,
        dof=dof,
        q=q,
        alpha=alpha
    )

    # sensemakr "constraint not binding" case.
    if f2 > 0 and (fqa > 0) and (f1 > 1/f2):
        return float(xrv)

    return float(rv)

def random_group_splits(groups, n_folds, n_rep, seed):
    """
    Returns nested DoubleML-compatible sample splitting:
      outer list = repetitions
      inner list = (train_idx, test_idx) folds

    Hospitals never occur in both train and test within a fold.
    """
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)

    all_smpls = []

    for rep in range(n_rep):
        rng = np.random.default_rng(seed + rep * 1009)
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)

        fold_map = {
            g: i % n_folds
            for i, g in enumerate(shuffled)
        }

        fold_id = np.array([fold_map[g] for g in groups], int)

        smpls_rep = []
        for k in range(n_folds):
            te = np.where(fold_id == k)[0]
            tr = np.where(fold_id != k)[0]
            smpls_rep.append((tr, te))

        all_smpls.append(smpls_rep)

    return all_smpls


# ==============================================================
# 1. Load locked v19 cross-fit panel
# ==============================================================

print("\n[1/7] LOADING LOCKED V19 CROSS-FIT PANEL")

d = pd.read_csv(V19, low_memory=False)

need = [
    EXP,
    Y,
    "treatment_resid",
    "delta_occupancy_resid",
    "hospital_id",
    "state",
] + NUM_COLS + CAT_COLS

missing = [c for c in need if c not in d.columns]
if missing:
    raise ValueError(f"Missing required v19 columns: {missing}")

d = d.replace([np.inf,-np.inf], np.nan).copy()
d = d.dropna(
    subset=[
        EXP,Y,"treatment_resid",
        "delta_occupancy_resid",
        "hospital_id","state"
    ]
).copy()

print("Episodes:", len(d))
print("Hospitals:", d["hospital_id"].nunique())
print("States:", d["state"].nunique())


# ==============================================================
# 2. Primary orthogonal regression: IID vs state-cluster inference
# ==============================================================

print("\n[2/7] ORTHOGONAL SCORE REGRESSION")

yres = d["delta_occupancy_resid"].to_numpy(float)
tres = d["treatment_resid"].to_numpy(float)

iid = iid_fit(yres, tres)
clu = cluster_fit(yres, tres, d["state"])

theta = float(iid.params[0])
iid_se = float(iid.bse[0])
iid_t = float(iid.tvalues[0])
iid_df = int(iid.df_resid)

cluster_theta = float(clu.params[0])
cluster_se = float(clu.bse[0])
cluster_t = float(clu.tvalues[0])

orthogonal = {
    "theta_iid": theta,
    "iid_se": iid_se,
    "iid_t": iid_t,
    "iid_residual_df": iid_df,
    "theta_state_clustered": cluster_theta,
    "state_clustered_se": cluster_se,
    "state_clustered_t": cluster_t,
    "state_clusters": int(d["state"].nunique())
}

print(json.dumps(orthogonal, indent=2))


# ==============================================================
# 3. Publication-grade Cinelli-Hazlett sensitivity statistics
# ==============================================================

print("\n[3/7] CINELLI-HAZLETT / SENSEMAKR SENSITIVITY")

pr2 = partial_r2(iid_t, iid_df)

rv_zero = robustness_value(
    iid_t,
    iid_df,
    q=1.0,
    alpha=1.0
)

rv_ci = robustness_value(
    iid_t,
    iid_df,
    q=1.0,
    alpha=0.05
)

rv_half = robustness_value(
    iid_t,
    iid_df,
    q=0.5,
    alpha=1.0
)

xrv_zero = extreme_robustness_value(
    iid_t,
    iid_df,
    q=1.0,
    alpha=1.0
)

sense = {
    "framework":
        "Cinelli-Hazlett sensemakr formulas applied to conventional OLS orthogonal-score regression",
    "estimate": theta,
    "standard_error_iid": iid_se,
    "t_statistic_iid": iid_t,
    "residual_dof": iid_df,
    "partial_R2_treatment_with_outcome_given_nuisance":
        float(pr2),
    "RV_q1_alpha1_reduce_point_estimate_to_zero":
        float(rv_zero),
    "RV_q1_alpha005_make_95CI_include_zero":
        float(rv_ci),
    "RV_q05_alpha1_reduce_point_estimate_by_half":
        float(rv_half),
    "extreme_RV_q1_alpha1":
        float(xrv_zero),
    "warning":
        "These standard sensitivity statistics use conventional linear-model SE/df, as required by the published sensemakr formulas. Primary inference in the paper remains state-cluster robust."
}

print(json.dumps(sense, indent=2))


# ==============================================================
# 4. Optional PySensemakr verification
# ==============================================================

print("\n[4/7] OPTIONAL PYSENSEMAKR FORMULA VERIFICATION")

pysense = {
    "available": False,
    "verified": False
}

try:
    import sensemakr as smkr

    # API differs slightly across releases; use direct numeric functions.
    rv_pkg_zero = float(np.asarray(
        smkr.robustness_value(
            t_statistic=iid_t,
            dof=iid_df,
            q=1,
            alpha=1
        )
    ).ravel()[0])

    rv_pkg_ci = float(np.asarray(
        smkr.robustness_value(
            t_statistic=iid_t,
            dof=iid_df,
            q=1,
            alpha=0.05
        )
    ).ravel()[0])

    pr2_pkg = float(np.asarray(
        smkr.partial_r2(
            t_statistic=iid_t,
            dof=iid_df
        )
    ).ravel()[0])

    pysense = {
        "available": True,
        "verified": (
            abs(rv_pkg_zero-rv_zero) < 1e-8
            and abs(rv_pkg_ci-rv_ci) < 1e-8
            and abs(pr2_pkg-pr2) < 1e-8
        ),
        "package_partial_R2": pr2_pkg,
        "package_RV_q1_alpha1": rv_pkg_zero,
        "package_RV_q1_alpha005": rv_pkg_ci,
        "manual_partial_R2": pr2,
        "manual_RV_q1_alpha1": rv_zero,
        "manual_RV_q1_alpha005": rv_ci,
    }

except ImportError:
    pysense = {
        "available": False,
        "verified": False,
        "note":
            "PySensemakr is not installed. Manual values use the official published sensemakr formulas exactly."
    }
except Exception as e:
    pysense = {
        "available": True,
        "verified": False,
        "error": repr(e)
    }

print(json.dumps(pysense, indent=2))


# ==============================================================
# 5. Prepare standard DoubleML data
# ==============================================================

print("\n[5/7] PREPARING STANDARD DOUBLEML DATA")

dd = d[
    [Y,EXP,"hospital_id","state"] + NUM_COLS + CAT_COLS
].copy()

# Numeric median imputation.
for c in NUM_COLS:
    dd[c] = pd.to_numeric(dd[c], errors="coerce")
    dd[c] = dd[c].fillna(dd[c].median())

# Categorical missing handling + one-hot.
for c in CAT_COLS:
    dd[c] = dd[c].astype("object")
    dd[c] = dd[c].where(dd[c].notna(), "Missing").astype(str)

X_cat = pd.get_dummies(
    dd[CAT_COLS],
    prefix=CAT_COLS,
    drop_first=False,
    dtype=float
)

X_num = dd[NUM_COLS].astype(float).reset_index(drop=True)
X_cat = X_cat.reset_index(drop=True)

model_df = pd.concat(
    [
        dd[[Y,EXP]].astype(float).reset_index(drop=True),
        X_num,
        X_cat
    ],
    axis=1
)

# DoubleML requires unique x column names.
model_df.columns = [
    str(c).replace(" ", "_").replace("[","").replace("]","")
    for c in model_df.columns
]

x_cols = [
    c for c in model_df.columns
    if c not in [Y,EXP]
]

groups = dd["hospital_id"].astype(str).to_numpy()

smpls = random_group_splits(
    groups,
    n_folds=N_FOLDS,
    n_rep=N_REP,
    seed=SEED
)

print("Model rows:", len(model_df))
print("Covariates after one-hot:", len(x_cols))
print("Hospital-grouped folds:", N_FOLDS)
print("Repeated partitions:", N_REP)


# ==============================================================
# 6. Standard DoubleMLPLR replication
# ==============================================================

print("\n[6/7] STANDARD DOUBLEMLPLR REPLICATION")

doubleml_result = {
    "available": False,
    "ran": False
}

try:
    import doubleml as dml

except ImportError:
    if AUTO_INSTALL_DOUBLEML:
        print("DoubleML missing; installing into current Python environment...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "doubleml"]
        )
        import doubleml as dml
    else:
        dml = None

if dml is None:
    doubleml_result = {
        "available": False,
        "ran": False,
        "install_command":
            f'"{sys.executable}" -m pip install doubleml',
        "note":
            "Set AUTO_INSTALL_DOUBLEML=True and rerun, or execute install_command once."
    }
    print(json.dumps(doubleml_result, indent=2))

else:
    ml_l = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=SEED
    )

    ml_m = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=SEED+1
    )

    data_dml = dml.DoubleMLData(
        model_df,
        y_col=Y,
        d_cols=EXP,
        x_cols=x_cols
    )

    obj = dml.DoubleMLPLR(
        data_dml,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=N_FOLDS,
        n_rep=N_REP,
        score="partialling out",
        draw_sample_splitting=False
    )

    obj.set_sample_splitting(smpls)
    obj.fit()

    coef = float(np.asarray(obj.coef).ravel()[0])
    se = float(np.asarray(obj.se).ravel()[0])

    all_coef = np.asarray(obj.all_coef).reshape(-1)
    all_se = np.asarray(obj.all_se).reshape(-1)

    doubleml_result = {
        "available": True,
        "ran": True,
        "package_version":
            getattr(dml, "__version__", "unknown"),
        "score": "partialling out",
        "n_folds": N_FOLDS,
        "n_rep": N_REP,
        "custom_split_unit": "hospital_id",
        "coef_aggregate": coef,
        "se_package": se,
        "all_coef_mean": float(np.mean(all_coef)),
        "all_coef_median": float(np.median(all_coef)),
        "all_coef_min": float(np.min(all_coef)),
        "all_coef_max": float(np.max(all_coef)),
        "all_coef_sd": float(np.std(all_coef, ddof=1))
            if len(all_coef) > 1 else 0.0,
        "all_negative": bool(np.all(all_coef < 0)),
        "difference_from_v19_theta":
            float(coef - cluster_theta),
        "relative_difference_from_v19":
            float((coef-cluster_theta)/cluster_theta)
            if abs(cluster_theta) > 1e-15 else np.nan,
        "note":
            "DoubleML coefficient is an implementation replication. Primary inference remains the v19 state-clustered orthogonal estimate."
    }

    print(json.dumps(doubleml_result, indent=2))


# ==============================================================
# 7. Final manuscript-use summary
# ==============================================================

print("\n[7/7] FINAL MANUSCRIPT-USE SUMMARY")

summary = {
    "orthogonal_regression": orthogonal,
    "publication_grade_sensemakr": sense,
    "pysensemakr_verification": pysense,
    "standard_doubleml": doubleml_result,
    "manuscript_rules":{
        "primary_inference":
            "Use state-clustered v19 coefficient, SE, CI and p-value.",
        "sensitivity":
            "Report standard Cinelli-Hazlett robustness values computed from the conventional orthogonal-score regression; explicitly state that clustered inference was used for the primary estimate.",
        "doubleml":
            "Use DoubleML only as implementation replication; do not replace the locked v19 primary estimate based on minor package differences."
    }
}

out_json = OUTPUT / "AJPH_v21_final_methodological_validation.json"
out_json.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print(json.dumps(summary, indent=2))
print("\nSaved:", out_json)

print("\nCOMPLETE")
