# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH U.S. Empirical Upgrade v16
Pre-Trend-Adjusted Occupancy Change Model

Purpose
-------
Address the remaining v15 concern: residual pre-surge differential trajectory,
especially at event week -2.

This script keeps the locked v13 exposure:
    hospital_log_response_innovation

Primary outcome
---------------
delta_occupancy_post_minus_pre =
    mean occupancy weeks +2:+4
  - mean occupancy weeks -4:-2

Primary adjustment set
----------------------
- pre-surge occupancy level: mean weeks -4:-2
- pre-surge occupancy slope: (occupancy_-2 - occupancy_-4) / 2
- pre-surge occupancy volatility: SD weeks -4:-2
- week -2 occupancy level
- hospital FE
- signal-quarter FE

Secondary robustness
--------------------
1) baseline + slope only
2) baseline + slope + volatility
3) baseline + slope + volatility + week -2
4) month FE
5) linear calendar-time control
6) LOSO by state
7) 999-rep wild cluster bootstrap
8) residualized post outcome formulation:
      post occupancy ~ response innovation
                     + pre occupancy level
                     + pre slope
                     + volatility
                     + hospital FE
                     + quarter FE

Interpretation remains associative, not causal.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(r"C:\Users\SIMONEY\Disease\AJPH_US_Empirical_Upgrade")
OUTPUT = ROOT / "04_outputs"

EVENT_PANEL = OUTPUT / "AJPH_v14b_reconstructed_event_time_panel.csv"
V13_PANEL = OUTPUT / "AJPH_v13_hospital_log_response_panel.csv"

for f in [EVENT_PANEL, V13_PANEL]:
    if not f.exists():
        raise FileNotFoundError(f)

VALID_US = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}

EXP = "hospital_log_response_innovation"
OUTCOME = "inpatient_occupancy"
PRE_WEEKS = [-4,-3,-2]
POST_WEEKS = [2,3,4]

WILD_REPS = 999
SEED = 20260905


def iterative_absorb(df, cols, fe_cols, tol=1e-10, max_iter=1000):
    z = df[cols].astype(float).copy()
    for _ in range(max_iter):
        old = z.to_numpy(copy=True)
        for fe in fe_cols:
            z = z - z.groupby(df[fe], sort=False).transform("mean")
        if np.max(np.abs(z.to_numpy() - old)) < tol:
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


def fit_change_model(df, controls, fe_cols):
    need = [
        "delta_occupancy_post_minus_pre",
        EXP,
        "state",
        "hospital_id",
    ] + controls + fe_cols

    d = df.dropna(subset=list(dict.fromkeys(need))).copy()

    rz = iterative_absorb(
        d,
        ["delta_occupancy_post_minus_pre", EXP] + controls,
        fe_cols
    )

    xcols = [EXP] + controls
    m = cluster_fit(
        rz["delta_occupancy_post_minus_pre"],
        rz[xcols],
        d["state"]
    )

    j = 0
    ci = m.conf_int()[j]
    sx = d[EXP].std(ddof=1)
    sy = d["delta_occupancy_post_minus_pre"].std(ddof=1)

    return {
        "n": int(len(d)),
        "hospitals": int(d["hospital_id"].nunique()),
        "states": int(d["state"].nunique()),
        "coef": float(m.params[j]),
        "se_cluster": float(m.bse[j]),
        "p_value": float(m.pvalues[j]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "std_beta": float(m.params[j] * sx / sy) if sy > 0 else np.nan,
    }, m, d, rz, xcols


def wild_bootstrap(d, rz, controls, reps=WILD_REPS, seed=SEED):
    y = rz["delta_occupancy_post_minus_pre"].to_numpy(float)
    Xu = rz[[EXP] + controls].to_numpy(float)
    groups = d["state"].astype(str).to_numpy(object)

    unres = cluster_fit(y, Xu, groups)
    obs_beta = float(unres.params[0])
    obs_se = float(unres.bse[0])
    obs_t = obs_beta / obs_se

    Xr = rz[controls].to_numpy(float)
    rfit = sm.OLS(y, Xr).fit()
    yhat = np.asarray(rfit.fittedvalues, float)
    resid = np.asarray(rfit.resid, float)

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
        "reps_requested": int(reps),
        "reps_valid": int(len(bt)),
        "reps_failed": int(failed),
        "observed_beta": obs_beta,
        "observed_cluster_se": obs_se,
        "observed_t": float(obs_t),
        "wild_cluster_bootstrap_p_two_sided": float(p),
        "bootstrap_t_p025": float(np.quantile(bt, .025)),
        "bootstrap_t_median": float(np.quantile(bt, .5)),
        "bootstrap_t_p975": float(np.quantile(bt, .975)),
    }


# ==============================================================
# 1. Load
# ==============================================================

print("\n[1/8] LOADING DATA")

panel = pd.read_csv(EVENT_PANEL, low_memory=False)
v13 = pd.read_csv(V13_PANEL, low_memory=False)

for d in [panel, v13]:
    d["state"] = d["state"].astype(str).str.upper().str.strip()
    d["hospital_id"] = d["hospital_id"].astype(str).str.strip()
    d["signal_week"] = pd.to_datetime(d["signal_week"], errors="coerce")

panel = panel[panel["state"].isin(VALID_US)].copy()
v13 = v13[v13["state"].isin(VALID_US)].copy()

panel["event_week"] = pd.to_numeric(panel["event_week"], errors="coerce").astype("Int64")

print("Event rows:", len(panel))
print("V13 episodes:", len(v13))


# ==============================================================
# 2. Construct episode pre-trend features
# ==============================================================

print("\n[2/8] CONSTRUCTING PRE-TREND FEATURES")

keys = ["hospital_id","state","signal_week"]

wide = (
    panel[panel["event_week"].isin(PRE_WEEKS + POST_WEEKS)]
    .pivot_table(
        index=keys,
        columns="event_week",
        values=OUTCOME,
        aggfunc="mean"
    )
    .reset_index()
)

for k in PRE_WEEKS + POST_WEEKS:
    if k not in wide.columns:
        wide[k] = np.nan

wide["occupancy_pre_mean"] = wide[PRE_WEEKS].mean(axis=1)
wide["occupancy_post_mean"] = wide[POST_WEEKS].mean(axis=1)
wide["occupancy_pre_slope"] = (wide[-2] - wide[-4]) / 2.0
wide["occupancy_pre_volatility"] = wide[PRE_WEEKS].std(axis=1, ddof=1)
wide["occupancy_week_m2"] = wide[-2]
wide["delta_occupancy_post_minus_pre"] = (
    wide["occupancy_post_mean"] - wide["occupancy_pre_mean"]
)

keep = keys + [
    "occupancy_pre_mean",
    "occupancy_post_mean",
    "occupancy_pre_slope",
    "occupancy_pre_volatility",
    "occupancy_week_m2",
    "delta_occupancy_post_minus_pre",
]

epi = v13.merge(wide[keep], on=keys, how="left")

epi["signal_quarter"] = (
    epi["signal_week"].dt.to_period("Q").astype(str).astype(object)
)
epi["signal_month"] = (
    epi["signal_week"].dt.to_period("M").astype(str).astype(object)
)
epi["calendar_week_index"] = (
    (epi["signal_week"] - epi["signal_week"].min()).dt.days / 7
).astype(float)
epi["hospital_id"] = epi["hospital_id"].astype(str).astype(object)
epi["state"] = epi["state"].astype(str).astype(object)

print(epi[
    [
        "occupancy_pre_mean",
        "occupancy_pre_slope",
        "occupancy_pre_volatility",
        "occupancy_week_m2",
        "delta_occupancy_post_minus_pre"
    ]
].describe())


# ==============================================================
# 3. Main model sequence
# ==============================================================

print("\n[3/8] PRE-TREND-ADJUSTED CHANGE MODELS")

specs = [
    (
        "baseline_plus_slope",
        ["occupancy_pre_mean","occupancy_pre_slope"],
        ["hospital_id","signal_quarter"]
    ),
    (
        "plus_volatility",
        ["occupancy_pre_mean","occupancy_pre_slope","occupancy_pre_volatility"],
        ["hospital_id","signal_quarter"]
    ),
    (
        "plus_week_m2_primary",
        [
            "occupancy_pre_mean",
            "occupancy_pre_slope",
            "occupancy_pre_volatility",
            "occupancy_week_m2"
        ],
        ["hospital_id","signal_quarter"]
    ),
    (
        "month_FE",
        [
            "occupancy_pre_mean",
            "occupancy_pre_slope",
            "occupancy_pre_volatility",
            "occupancy_week_m2"
        ],
        ["hospital_id","signal_month"]
    ),
    (
        "linear_time",
        [
            "occupancy_pre_mean",
            "occupancy_pre_slope",
            "occupancy_pre_volatility",
            "occupancy_week_m2",
            "calendar_week_index"
        ],
        ["hospital_id"]
    ),
]

rows = []
models = {}

for name, controls, fes in specs:
    r, m, d, rz, xcols = fit_change_model(epi, controls, fes)
    r["spec"] = name
    rows.append(r)
    models[name] = (r,m,d,rz,xcols,controls,fes)

res = pd.DataFrame(rows)

print(res[
    ["spec","n","hospitals","states","coef","se_cluster",
     "p_value","ci_low","ci_high","std_beta"]
].to_string(index=False))

res.to_csv(
    OUTPUT / "AJPH_v16_pretrend_adjusted_change_models.csv",
    index=False
)


# ==============================================================
# 4. Primary wild bootstrap
# ==============================================================

print("\n[4/8] PRIMARY WILD CLUSTER BOOTSTRAP")

primary = models["plus_week_m2_primary"]
pr, pm, pdta, prz, pxcols, pcontrols, pfes = primary

wild = wild_bootstrap(
    pdta,
    prz,
    pcontrols,
    reps=WILD_REPS,
    seed=SEED
)

print(json.dumps(wild, indent=2))


# ==============================================================
# 5. LOSO
# ==============================================================

print("\n[5/8] PRIMARY MODEL LOSO")

loso_rows = []

for i, st in enumerate(sorted(pdta["state"].astype(str).unique()), 1):
    temp = epi[epi["state"].astype(str) != st].copy()

    rr, *_ = fit_change_model(
        temp,
        pcontrols,
        pfes
    )

    rr["excluded_state"] = st
    loso_rows.append(rr)

    if i % 10 == 0:
        print(f"  LOSO {i}")

loso = pd.DataFrame(loso_rows)
loso.to_csv(
    OUTPUT / "AJPH_v16_primary_LOSO.csv",
    index=False
)

loso_summary = {
    "n_models": int(len(loso)),
    "fraction_negative": float((loso["coef"] < 0).mean()),
    "median_coef": float(loso["coef"].median()),
    "min_coef": float(loso["coef"].min()),
    "max_coef": float(loso["coef"].max()),
    "fraction_p_lt_0_05": float((loso["p_value"] < .05).mean()),
}

print(json.dumps(loso_summary, indent=2))


# ==============================================================
# 6. Post-outcome residualized formulation
# ==============================================================

print("\n[6/8] POST-OCCUPANCY FORMULATION")

POST_CONTROLS = [
    "occupancy_pre_mean",
    "occupancy_pre_slope",
    "occupancy_pre_volatility",
    "occupancy_week_m2",
]

need = [
    "occupancy_post_mean",
    EXP,
    "hospital_id",
    "signal_quarter",
    "state"
] + POST_CONTROLS

postd = epi.dropna(subset=need).copy()

postrz = iterative_absorb(
    postd,
    ["occupancy_post_mean", EXP] + POST_CONTROLS,
    ["hospital_id","signal_quarter"]
)

postm = cluster_fit(
    postrz["occupancy_post_mean"],
    postrz[[EXP] + POST_CONTROLS],
    postd["state"]
)

pci = postm.conf_int()[0]
post_result = {
    "n": int(len(postd)),
    "hospitals": int(postd["hospital_id"].nunique()),
    "states": int(postd["state"].nunique()),
    "coef": float(postm.params[0]),
    "se_cluster": float(postm.bse[0]),
    "p_value": float(postm.pvalues[0]),
    "ci_low": float(pci[0]),
    "ci_high": float(pci[1]),
}

print(json.dumps(post_result, indent=2))


# ==============================================================
# 7. Pre-trend descriptive association
# ==============================================================

print("\n[7/8] RESPONSE VS PRE-TREND DESCRIPTIVE CHECK")

check_rows = []

for outcome in [
    "occupancy_pre_mean",
    "occupancy_pre_slope",
    "occupancy_pre_volatility",
    "occupancy_week_m2"
]:
    need = [
        outcome, EXP,
        "hospital_id","signal_quarter","state"
    ]
    x = epi.dropna(subset=need).copy()

    rz = iterative_absorb(
        x,
        [outcome,EXP],
        ["hospital_id","signal_quarter"]
    )

    mm = cluster_fit(
        rz[outcome],
        rz[[EXP]],
        x["state"]
    )

    ci = mm.conf_int()[0]

    check_rows.append({
        "outcome": outcome,
        "coef": float(mm.params[0]),
        "se_cluster": float(mm.bse[0]),
        "p_value": float(mm.pvalues[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    })

check = pd.DataFrame(check_rows)

print(check.to_string(index=False))

check.to_csv(
    OUTPUT / "AJPH_v16_pretrend_descriptive_checks.csv",
    index=False
)


# ==============================================================
# 8. Final summary
# ==============================================================

print("\n[8/8] FINAL SUMMARY")

summary = {
    "primary_model": pr,
    "wild_cluster_bootstrap": wild,
    "loso": loso_summary,
    "post_occupancy_formulation": post_result,
    "pretrend_checks": check_rows,
    "all_specs": rows,
}

(OUTPUT / "AJPH_v16_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print(json.dumps(summary, indent=2))

print("\nCOMPLETE")
print("Main models:", OUTPUT / "AJPH_v16_pretrend_adjusted_change_models.csv")
print("LOSO:", OUTPUT / "AJPH_v16_primary_LOSO.csv")
print("Pretrend checks:", OUTPUT / "AJPH_v16_pretrend_descriptive_checks.csv")
print("Summary:", OUTPUT / "AJPH_v16_summary.json")
