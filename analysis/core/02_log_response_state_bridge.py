# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH U.S. Empirical Upgrade v13 FINAL SPECIFICATION
Objective HHS Quality + Log-Scale Hospital Response + Symmetric State Innovation

Purpose
-------
This is the final empirical specification intended to resolve the two remaining issues:

1) Hospital-level proportional response instability:
   Percentage changes can explode for small baseline hospitals.
   v13 uses LOG capacity change instead:
       100 * [log(post capacity) - log(baseline capacity)]
   and constructs a cross-fitted post response residual.

2) State-level placebo specificity:
   v12 aggregated hospital-level innovation but compared it against a separately aggregated placebo.
   v13 constructs symmetric state-level post and pre response measures first, then defines:
       state_response_innovation = state_post_response - state_pre_response
   and includes state_pre_response as a covariate.

Primary hospital exposure
-------------------------
hospital_log_response_innovation
    = cross-fitted residualized post-surge log capacity response
      - cross-fitted residualized pre-surge log capacity response

Primary hospital outcome
------------------------
inpatient_occupancy_w2_4

Primary hospital model
----------------------
Outcome ~ hospital_log_response_innovation
        + post demand change
        + pre demand change
        + baseline COVID demand
        + pre demand slope
        + signal admissions
        + hospital FE
        + quarter FE

Primary state exposure
----------------------
state_log_response_innovation
    = capacity-weighted state post log response residual
      - capacity-weighted state pre log response residual

State mortality model
---------------------
Excess mortality ~ state_log_response_innovation
                 + state_pre_log_response
                 + signal admissions
                 + number of contributing hospitals
                 + state FE
                 + time controls

Additional checks
-----------------
- hospital quarter / month / linear-time sensitivity
- hospital LOSO
- hospital wild cluster bootstrap
- state wave / quarter / month / linear-time sensitivity
- state LOSO
- state wild cluster bootstrap
- state placebo-control model via explicit pre-response covariate
- no percentile trimming is used in the primary specification

Interpretation remains associative, not causal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(r"C:\Users\SIMONEY\Disease\AJPH_US_Empirical_Upgrade")
OUTPUT = ROOT / "04_outputs"

HOSPITAL_Q = OUTPUT / "AJPH_v12_hospital_quality_filtered_panel.csv"
STATE_BRIDGE_SOURCE = OUTPUT / "AJPH_v12_state_hospital_response_bridge.csv"
STATE_EP_SOURCE = OUTPUT / "AJPH_v8_crossfitted_episode_panel.csv"

for f in [HOSPITAL_Q, STATE_EP_SOURCE]:
    if not f.exists():
        raise FileNotFoundError(f)

VALID_US = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}

WILD_BOOT_REPS = 999
SEED = 20260905


# ==============================================================
# Helpers
# ==============================================================

def iterative_absorb(df, cols, fe_cols, tol=1e-10, max_iter=300):
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

def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.average(x[m], weights=w[m]))

def fit_absorbed(df, outcome, exposure, controls, fe_cols):
    need = list(dict.fromkeys(
        [outcome, exposure, "state", "hospital_id"] + controls + fe_cols
    ))
    d = df.dropna(subset=need).copy()

    rz = iterative_absorb(
        d,
        [outcome, exposure] + controls,
        fe_cols
    )

    y = rz[outcome].to_numpy(float)
    Xcols = [exposure] + controls
    X = rz[Xcols].to_numpy(float)

    m = cluster_fit(
        y, X,
        d["state"].astype(str).to_numpy(object)
    )

    ci = m.conf_int()[0]
    sx = d[exposure].std(ddof=1)
    sy = d[outcome].std(ddof=1)

    return {
        "n": int(len(d)),
        "hospitals": int(d["hospital_id"].nunique()),
        "states": int(d["state"].nunique()),
        "coef": float(m.params[0]),
        "se_cluster": float(m.bse[0]),
        "p_value": float(m.pvalues[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "std_beta": float(m.params[0] * sx / sy) if sy > 0 else np.nan,
    }, m, d, rz, Xcols

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
    yhat0 = np.asarray(restricted.fittedvalues, float)
    u0 = np.asarray(restricted.resid, float)

    ug = np.unique(groups)
    rng = np.random.default_rng(seed)

    bt = []
    failed = 0

    for b in range(reps):
        signs = {g: rng.choice([-1.0, 1.0]) for g in ug}
        ys = yhat0 + u0 * np.array([signs[g] for g in groups], float)

        try:
            mb = cluster_fit(ys, Xu, groups)
            bb = float(mb.params[0])
            bs = float(mb.bse[0])
            if np.isfinite(bb) and np.isfinite(bs) and bs > 0:
                bt.append(bb / bs)
            else:
                failed += 1
        except Exception:
            failed += 1

        if (b + 1) % 100 == 0:
            print(
                f"  bootstrap {b+1}/{reps}; "
                f"valid={len(bt)}, failed={failed}"
            )

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
# 1. Load objective-quality hospital panel
# ==============================================================

print("\n[1/9] LOADING OBJECTIVE-QUALITY HOSPITAL PANEL")

hq = pd.read_csv(HOSPITAL_Q, low_memory=False)

hq["state"] = hq["state"].astype(str).str.upper().str.strip()
hq = hq[hq["state"].isin(VALID_US)].copy()

hq["hospital_id"] = hq["hospital_id"].astype(str).astype(object)
hq["state"] = hq["state"].astype(str).astype(object)

hq["signal_week"] = pd.to_datetime(hq["signal_week"], errors="coerce")
hq["signal_quarter"] = (
    hq["signal_week"].dt.to_period("Q").astype(str).astype(object)
)
hq["signal_month"] = (
    hq["signal_week"].dt.to_period("M").astype(str).astype(object)
)
hq["calendar_week_index"] = (
    (hq["signal_week"] - hq["signal_week"].min()).dt.days / 7
).astype(float)

print("Rows:", len(hq))
print("Hospitals:", hq["hospital_id"].nunique())
print("States/DC:", hq["state"].nunique())


# ==============================================================
# 2. Construct post/pre LOG capacity changes
# ==============================================================

print("\n[2/9] CONSTRUCTING LOG-SCALE CAPACITY RESPONSE")

required = [
    "baseline_capacity_quality",
    "capacity_week1",
    "capacity_week2",
    "placebo_capacity_pre_pp",
    "expected_placebo_capacity_pre_pp",
    "placebo_capacity_residual_cf_pp",
    "sustained_demand_change_w1_2_pp",
    "baseline_demand_quality",
    "baseline_admissions",
    "pre_demand_slope",
    "signal_admissions_7d",
    "signal_rel_change",
]

missing = [c for c in required if c not in hq.columns]
if missing:
    raise ValueError("Missing required columns:\n" + "\n".join(missing))

# Post sustained capacity = mean weeks +1 and +2
hq["post_capacity_mean"] = (
    hq[["capacity_week1","capacity_week2"]].mean(axis=1)
)

# Log response (x100 = approx percent change for small changes)
hq["post_log_capacity_change"] = np.where(
    (hq["baseline_capacity_quality"] > 0) &
    (hq["post_capacity_mean"] > 0),
    100.0 * (
        np.log(hq["post_capacity_mean"]) -
        np.log(hq["baseline_capacity_quality"])
    ),
    np.nan
)

# Reconstruct an approximate pre capacity change from placebo percentage change:
# placebo_capacity_pre_pp = 100 * (pre/base_prior - 1)
# therefore log ratio = log(1 + p/100)
hq["pre_log_capacity_change"] = np.where(
    (1.0 + hq["placebo_capacity_pre_pp"]/100.0) > 0,
    100.0 * np.log(
        1.0 + hq["placebo_capacity_pre_pp"]/100.0
    ),
    np.nan
)

print("\nPost log-change summary:")
print(hq["post_log_capacity_change"].describe(
    percentiles=[.01,.025,.05,.95,.975,.99]
))
print("\nPre log-change summary:")
print(hq["pre_log_capacity_change"].describe(
    percentiles=[.01,.025,.05,.95,.975,.99]
))


# ==============================================================
# 3. Cross-fit expected POST log capacity response
# ==============================================================

print("\n[3/9] CROSS-FITTING POST LOG RESPONSE")

POST_TARGET = "post_log_capacity_change"

post_num = [
    "sustained_demand_change_w1_2_pp",
    "baseline_demand_quality",
    "baseline_capacity_quality",
    "baseline_admissions",
    "pre_demand_slope",
    "signal_admissions_7d",
    "signal_rel_change",
    "calendar_week_index",
]
post_cat = ["signal_quarter"]

pw = hq.dropna(
    subset=[POST_TARGET, "state"] + post_num + post_cat
).copy()

pred_rows = []

for i, st in enumerate(sorted(pw["state"].astype(str).unique()), 1):
    tr = pw[pw["state"].astype(str) != st].copy()
    te = pw[pw["state"].astype(str) == st].copy()

    if len(tr) < 100 or len(te) == 0:
        continue

    prep = ColumnTransformer([
        ("num", "passthrough", post_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), post_cat),
    ])

    pipe = Pipeline([
        ("prep", prep),
        ("reg", LinearRegression())
    ])

    pipe.fit(tr[post_num + post_cat], tr[POST_TARGET])
    phat = pipe.predict(te[post_num + post_cat])

    for idx, val in zip(te.index, phat):
        pred_rows.append({
            "index": idx,
            "expected_post_log_capacity_change": float(val)
        })

    if i % 10 == 0:
        print(f"  post cross-fit {i}")

post_pred = pd.DataFrame(pred_rows).set_index("index")
hq = hq.join(post_pred, how="left")

hq["post_log_response_residual_cf"] = (
    hq["post_log_capacity_change"] -
    hq["expected_post_log_capacity_change"]
)


# ==============================================================
# 4. Cross-fit expected PRE log capacity response
# ==============================================================

print("\n[4/9] CROSS-FITTING PRE LOG RESPONSE")

PRE_TARGET = "pre_log_capacity_change"

pre_num = [
    "placebo_demand_pre_pp",
    "baseline_demand_quality",
    "baseline_capacity_quality",
    "baseline_admissions",
    "signal_admissions_7d",
    "signal_rel_change",
    "calendar_week_index",
]
pre_cat = ["signal_quarter"]

prew = hq.dropna(
    subset=[PRE_TARGET, "state"] + pre_num + pre_cat
).copy()

pred_rows = []

for i, st in enumerate(sorted(prew["state"].astype(str).unique()), 1):
    tr = prew[prew["state"].astype(str) != st].copy()
    te = prew[prew["state"].astype(str) == st].copy()

    if len(tr) < 100 or len(te) == 0:
        continue

    prep = ColumnTransformer([
        ("num", "passthrough", pre_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), pre_cat),
    ])

    pipe = Pipeline([
        ("prep", prep),
        ("reg", LinearRegression())
    ])

    pipe.fit(tr[pre_num + pre_cat], tr[PRE_TARGET])
    phat = pipe.predict(te[pre_num + pre_cat])

    for idx, val in zip(te.index, phat):
        pred_rows.append({
            "index": idx,
            "expected_pre_log_capacity_change": float(val)
        })

    if i % 10 == 0:
        print(f"  pre cross-fit {i}")

pre_pred = pd.DataFrame(pred_rows).set_index("index")
hq = hq.join(pre_pred, how="left")

hq["pre_log_response_residual_cf"] = (
    hq["pre_log_capacity_change"] -
    hq["expected_pre_log_capacity_change"]
)

hq["hospital_log_response_innovation"] = (
    hq["post_log_response_residual_cf"] -
    hq["pre_log_response_residual_cf"]
)

print("\nHospital log-response innovation summary:")
print(hq["hospital_log_response_innovation"].describe(
    percentiles=[.01,.025,.05,.95,.975,.99]
))

print("\nPost/pre residual correlation:")
print(
    hq[
        ["post_log_response_residual_cf",
         "pre_log_response_residual_cf",
         "hospital_log_response_innovation"]
    ].corr().round(4)
)

hq.to_csv(
    OUTPUT / "AJPH_v13_hospital_log_response_panel.csv",
    index=False
)


# ==============================================================
# 5. Hospital models
# ==============================================================

print("\n[5/9] HOSPITAL-LEVEL FINAL MODELS")

H_OUT = "inpatient_occupancy_w2_4"
H_EXP = "hospital_log_response_innovation"
H_CONTROLS = [
    "sustained_demand_change_w1_2_pp",
    "placebo_demand_pre_pp",
    "baseline_demand_quality",
    "pre_demand_slope",
    "signal_admissions_7d",
]

hospital_rows = []

r, *_ = fit_absorbed(
    hq, H_OUT, H_EXP, H_CONTROLS,
    ["hospital_id","signal_quarter"]
)
r["spec"] = "hospital_FE_plus_quarter_FE"
hospital_rows.append(r)

r, *_ = fit_absorbed(
    hq, H_OUT, H_EXP, H_CONTROLS,
    ["hospital_id","signal_month"]
)
r["spec"] = "hospital_FE_plus_month_FE"
hospital_rows.append(r)

r, *_ = fit_absorbed(
    hq, H_OUT, H_EXP,
    H_CONTROLS + ["calendar_week_index"],
    ["hospital_id"]
)
r["spec"] = "hospital_FE_plus_linear_time"
hospital_rows.append(r)

hospital_results = pd.DataFrame(hospital_rows)

print(hospital_results[
    ["spec","n","hospitals","states","coef","se_cluster",
     "p_value","ci_low","ci_high","std_beta"]
].to_string(index=False))

hospital_results.to_csv(
    OUTPUT / "AJPH_v13_hospital_final_models.csv",
    index=False
)

# Primary hospital quarter-FE model
h_primary, hm, hd, hrz, _ = fit_absorbed(
    hq, H_OUT, H_EXP, H_CONTROLS,
    ["hospital_id","signal_quarter"]
)

# LOSO
print("\nHospital LOSO")
hloso_rows = []
for i, st in enumerate(sorted(hd["state"].astype(str).unique()), 1):
    temp = hq[hq["state"].astype(str) != st].copy()
    try:
        rr, *_ = fit_absorbed(
            temp, H_OUT, H_EXP, H_CONTROLS,
            ["hospital_id","signal_quarter"]
        )
        rr["excluded_state"] = st
        rr["status"] = "ok"
    except Exception as e:
        rr = {"excluded_state":st,"status":"failed","error":repr(e)}
    hloso_rows.append(rr)
    if i % 10 == 0:
        print(f"  hospital LOSO {i}")

hloso = pd.DataFrame(hloso_rows)
hloso.to_csv(OUTPUT / "AJPH_v13_hospital_LOSO.csv", index=False)

hok = hloso[hloso["status"]=="ok"].copy()
hospital_loso = {
    "n_models": int(len(hok)),
    "fraction_negative": float((hok["coef"] < 0).mean()),
    "median_coef": float(hok["coef"].median()),
    "min_coef": float(hok["coef"].min()),
    "max_coef": float(hok["coef"].max()),
    "fraction_p_lt_0_05": float((hok["p_value"] < .05).mean()),
}
print(json.dumps(hospital_loso, indent=2))

print("\nHospital wild cluster bootstrap")
hospital_wild = wild_bootstrap(
    hd, hrz, H_OUT, H_EXP, H_CONTROLS
)
print(json.dumps(hospital_wild, indent=2))


# ==============================================================
# 6. Build symmetric state post/pre aggregates
# ==============================================================

print("\n[6/9] BUILDING SYMMETRIC STATE POST/PRE RESPONSE")

sep = pd.read_csv(STATE_EP_SOURCE, low_memory=False)
sep["signal_date"] = pd.to_datetime(sep["signal_date"], errors="coerce")
sep["state"] = sep["state"].astype(str).str.upper().str.strip()
sep = sep[sep["state"].isin(VALID_US)].copy()

bridge_rows = []

for idx, s in sep.iterrows():
    st = s["state"]
    sd = pd.Timestamp(s["signal_date"])

    matches = hq[
        (hq["state"].astype(str) == st) &
        (hq["signal_week"] >= sd - pd.Timedelta(days=14)) &
        (hq["signal_week"] <= sd + pd.Timedelta(days=14))
    ].dropna(
        subset=[
            "post_log_response_residual_cf",
            "pre_log_response_residual_cf",
            "baseline_capacity_quality"
        ]
    ).copy()

    if matches.empty:
        bridge_rows.append({"_sep_index":idx})
        continue

    w = matches["baseline_capacity_quality"].to_numpy(float)

    state_post = weighted_mean(
        matches["post_log_response_residual_cf"].to_numpy(float), w
    )
    state_pre = weighted_mean(
        matches["pre_log_response_residual_cf"].to_numpy(float), w
    )

    bridge_rows.append({
        "_sep_index": idx,
        "state_hospital_count": int(matches["hospital_id"].nunique()),
        "state_hospital_episode_count": int(len(matches)),
        "state_capacity_weight": float(np.nansum(w)),
        "state_post_log_response": state_post,
        "state_pre_log_response": state_pre,
        "state_log_response_innovation": (
            state_post - state_pre
            if np.isfinite(state_post) and np.isfinite(state_pre)
            else np.nan
        )
    })

bridge = pd.DataFrame(bridge_rows).set_index("_sep_index")
sep = sep.join(bridge, how="left")

sep["signal_month"] = sep["signal_date"].dt.to_period("M").astype(str).astype(object)
sep["signal_quarter"] = sep["signal_date"].dt.to_period("Q").astype(str).astype(object)
sep["calendar_day"] = (
    sep["signal_date"] - sep["signal_date"].min()
).dt.days.astype(float)
sep["state"] = sep["state"].astype(str).astype(object)
sep["wave"] = sep["wave"].astype(str).astype(object)

sep.to_csv(
    OUTPUT / "AJPH_v13_state_log_response_bridge.csv",
    index=False
)

print("State episodes with symmetric bridge:",
      int(sep["state_log_response_innovation"].notna().sum()))
print("Median contributing hospitals:",
      float(sep["state_hospital_count"].median()))


# ==============================================================
# 7. State mortality models
# ==============================================================

print("\n[7/9] STATE-LEVEL FINAL MORTALITY MODELS")

S_OUT = "excess_deaths_14_42_per100k"
S_EXP = "state_log_response_innovation"
S_CONTROLS = [
    "state_pre_log_response",
    "signal_admissions_7dma",
    "state_hospital_count",
]

def fit_state(time_mode):
    need = [S_OUT,S_EXP,"state"] + S_CONTROLS
    if time_mode=="wave":
        need.append("wave")
        fe=["state","wave"]
        controls=S_CONTROLS
    elif time_mode=="quarter":
        need.append("signal_quarter")
        fe=["state","signal_quarter"]
        controls=S_CONTROLS
    elif time_mode=="month":
        need.append("signal_month")
        fe=["state","signal_month"]
        controls=S_CONTROLS
    elif time_mode=="linear":
        need.append("calendar_day")
        fe=["state"]
        controls=S_CONTROLS+["calendar_day"]
    else:
        raise ValueError(time_mode)

    d = sep.dropna(subset=need).copy()

    rz = iterative_absorb(
        d,
        [S_OUT,S_EXP] + controls,
        fe
    )

    m = cluster_fit(
        rz[S_OUT],
        rz[[S_EXP] + controls],
        d["state"]
    )

    ci = m.conf_int()[0]
    sx=d[S_EXP].std(ddof=1)
    sy=d[S_OUT].std(ddof=1)

    return {
        "time_control":time_mode,
        "n":int(len(d)),
        "states":int(d["state"].nunique()),
        "coef":float(m.params[0]),
        "se_cluster":float(m.bse[0]),
        "p_value":float(m.pvalues[0]),
        "ci_low":float(ci[0]),
        "ci_high":float(ci[1]),
        "std_beta":float(m.params[0]*sx/sy) if sy>0 else np.nan,
    }, d, rz, controls, m

state_rows=[]
for tm in ["wave","quarter","month","linear"]:
    rr,*_=fit_state(tm)
    state_rows.append(rr)

state_results=pd.DataFrame(state_rows)
print(state_results.to_string(index=False))
state_results.to_csv(
    OUTPUT / "AJPH_v13_state_final_models.csv",
    index=False
)


# ==============================================================
# 8. State LOSO + wild bootstrap
# ==============================================================

print("\n[8/9] STATE ROBUSTNESS")

s_primary, sd, srz, scontrols, smod = fit_state("wave")

sloso_rows=[]
for i, st in enumerate(sorted(sd["state"].astype(str).unique()),1):
    temp=sep[sep["state"].astype(str)!=st].copy()
    need=[S_OUT,S_EXP,"state","wave"]+S_CONTROLS
    td=temp.dropna(subset=need).copy()
    trz=iterative_absorb(
        td,[S_OUT,S_EXP]+S_CONTROLS,
        ["state","wave"]
    )
    tm=cluster_fit(
        trz[S_OUT],
        trz[[S_EXP]+S_CONTROLS],
        td["state"]
    )
    ci=tm.conf_int()[0]
    sloso_rows.append({
        "excluded_state":st,
        "coef":float(tm.params[0]),
        "p_value":float(tm.pvalues[0]),
        "ci_low":float(ci[0]),
        "ci_high":float(ci[1]),
    })
    if i%10==0:
        print(f"  state LOSO {i}")

sloso=pd.DataFrame(sloso_rows)
sloso.to_csv(OUTPUT/"AJPH_v13_state_LOSO.csv",index=False)

state_loso={
    "n_models":int(len(sloso)),
    "fraction_negative":float((sloso["coef"]<0).mean()),
    "median_coef":float(sloso["coef"].median()),
    "min_coef":float(sloso["coef"].min()),
    "max_coef":float(sloso["coef"].max()),
    "fraction_p_lt_0_05":float((sloso["p_value"]<.05).mean()),
}
print("\nSTATE LOSO SUMMARY")
print(json.dumps(state_loso,indent=2))

print("\nSTATE WILD CLUSTER BOOTSTRAP")
state_wild=wild_bootstrap(
    sd,srz,S_OUT,S_EXP,scontrols
)
print(json.dumps(state_wild,indent=2))


# ==============================================================
# 9. Final summary
# ==============================================================

print("\n[9/9] FINAL SUMMARY")

summary={
    "hospital_primary":h_primary,
    "hospital_time_models":hospital_rows,
    "hospital_loso":hospital_loso,
    "hospital_wild":hospital_wild,
    "state_bridge":{
        "episodes_with_bridge":int(sep["state_log_response_innovation"].notna().sum()),
        "median_hospitals":float(sep["state_hospital_count"].median())
    },
    "state_models":state_rows,
    "state_loso":state_loso,
    "state_wild":state_wild
}

(OUTPUT/"AJPH_v13_summary.json").write_text(
    json.dumps(summary,indent=2),
    encoding="utf-8"
)

print(json.dumps(summary,indent=2))
print("\nCOMPLETE")
print("Hospital final models:", OUTPUT/"AJPH_v13_hospital_final_models.csv")
print("Hospital LOSO:", OUTPUT/"AJPH_v13_hospital_LOSO.csv")
print("State bridge:", OUTPUT/"AJPH_v13_state_log_response_bridge.csv")
print("State final models:", OUTPUT/"AJPH_v13_state_final_models.csv")
print("State LOSO:", OUTPUT/"AJPH_v13_state_LOSO.csv")
print("Summary:", OUTPUT/"AJPH_v13_summary.json")
