# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH U.S. Empirical Upgrade v12
Objective HHS Data-Quality Primary Sample + State-Level Mortality Bridge

This script addresses two remaining weaknesses:

A) Hospital-level primary specification
---------------------------------------
Replace percentile-based trimming with an outcome-independent, source-based HHS quality rule:
1. Restrict to 50 states + DC.
2. Exclude suppressed/missing core facility reports.
3. Require >=4 reporting days for the relevant weekly capacity/demand fields, consistent
   with HHS coverage definitions and suppression rules.
4. Require a sustained response signal across weeks +1 and +2 rather than a single-week spike.
5. Require baseline capacity to be observed in >=3 of 4 baseline weeks.
6. Exclude transient reporting discontinuities by requiring that the week +1/+2 capacity
   observations are both finite and that their absolute disagreement is not larger than the
   larger of:
       (a) 100% of the baseline capacity, or
       (b) 50 beds.
   This is a structural persistence rule, not an outcome-based percentile rule.

The primary hospital exposure is then re-estimated by leave-one-state-out cross-fitting:
    sustained_capacity_responsiveness_cf_pp

B) State-level mortality bridge
-------------------------------
Aggregate the objectively quality-filtered hospital response innovations within each
pre-existing state surge episode. Hospitals are matched to a state surge when the hospital
surge signal falls within +/- 14 days of the state surge signal. Hospital contributions are
weighted by baseline staffed inpatient capacity.

This produces a population-facing exposure:
    state_weighted_hospital_response_innovation_pp

The state mortality model then relates this hospital-derived operational response index to
subsequent state excess mortality, retaining:
- state FE
- wave / quarter / month / linear-time variants
- state surge admissions severity
- hospital coverage/count controls
- placebo aggregate
- LOSO
- wild cluster bootstrap

Interpretation remains associative, not causal.

Inputs expected from prior versions:
- 02_clean/hhs_facility_weekly_selected_clean_v9.csv
- 04_outputs/AJPH_v9_hospital_episode_panel.csv
- 04_outputs/AJPH_v8_crossfitted_episode_panel.csv
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
CLEAN = ROOT / "02_clean"
OUTPUT = ROOT / "04_outputs"

FACILITY_WEEKLY = CLEAN / "hhs_facility_weekly_selected_clean_v9.csv"
HOSPITAL_EP = OUTPUT / "AJPH_v9_hospital_episode_panel.csv"
STATE_EP = OUTPUT / "AJPH_v8_crossfitted_episode_panel.csv"

for f in [FACILITY_WEEKLY, HOSPITAL_EP, STATE_EP]:
    if not f.exists():
        raise FileNotFoundError(f)

VALID_US = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}

MIN_COVERAGE_DAYS = 4
WILD_BOOT_REPS = 999
SEED = 20260905


# ==============================================================
# Helpers
# ==============================================================

def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

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
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.average(np.asarray(x)[m], weights=np.asarray(w)[m]))

def wild_bootstrap_absorbed(d, rz, outcome, exposure, controls, reps=999, seed=SEED):
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
    u = np.asarray(restricted.resid, float)

    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    bt, failed = [], 0

    for b in range(reps):
        signs = {g: rng.choice([-1.0, 1.0]) for g in ug}
        ys = yhat + u * np.array([signs[g] for g in groups], float)
        try:
            mb = cluster_fit(ys, Xu, groups)
            bb, bs = float(mb.params[0]), float(mb.bse[0])
            if np.isfinite(bb) and np.isfinite(bs) and bs > 0:
                bt.append(bb / bs)
            else:
                failed += 1
        except Exception:
            failed += 1

        if (b + 1) % 100 == 0:
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
    }


# ==============================================================
# 1. Load sources and identify HHS fields
# ==============================================================

print("\n[1/10] LOADING SOURCES")

fw = pd.read_csv(FACILITY_WEEKLY, low_memory=False)
hep = pd.read_csv(HOSPITAL_EP, low_memory=False)
sep = pd.read_csv(STATE_EP, low_memory=False)

hid = first_existing(fw.columns, ["hospital_pk", "ccn"])
date = first_existing(fw.columns, ["collection_week"])
statecol = first_existing(fw.columns, ["state"])

cap = first_existing(
    fw.columns,
    ["all_adult_hospital_inpatient_beds_7_day_avg", "inpatient_beds_7_day_avg"]
)
cap_cov = first_existing(
    fw.columns,
    ["all_adult_hospital_inpatient_beds_7_day_coverage", "inpatient_beds_7_day_coverage"]
)

dem = first_existing(
    fw.columns,
    ["total_adult_patients_hospitalized_confirmed_covid_7_day_avg",
     "inpatient_beds_used_covid_7_day_avg"]
)
dem_cov = first_existing(
    fw.columns,
    ["total_adult_patients_hospitalized_confirmed_covid_7_day_coverage",
     "inpatient_beds_used_covid_7_day_coverage"]
)

if None in [hid, date, statecol, cap, dem]:
    raise ValueError("Could not resolve required HHS fields.")

print("Hospital ID:", hid)
print("Capacity:", cap)
print("Capacity coverage:", cap_cov)
print("Demand:", dem)
print("Demand coverage:", dem_cov)

fw[date] = pd.to_datetime(fw[date], errors="coerce")
fw[statecol] = fw[statecol].astype(str).str.upper().str.strip()
fw[hid] = fw[hid].astype(str).str.strip()
fw = fw[fw[statecol].isin(VALID_US)].copy()

for c in [cap, cap_cov, dem, dem_cov]:
    if c is not None:
        fw[c] = pd.to_numeric(fw[c], errors="coerce")
        fw.loc[fw[c] <= -999000, c] = np.nan

hep["signal_week"] = pd.to_datetime(hep["signal_week"], errors="coerce")
hep["state"] = hep["state"].astype(str).str.upper().str.strip()
hep["hospital_id"] = hep["hospital_id"].astype(str).str.strip()
hep = hep[hep["state"].isin(VALID_US)].copy()

sep["signal_date"] = pd.to_datetime(sep["signal_date"], errors="coerce")
sep["state"] = sep["state"].astype(str).str.upper().str.strip()
sep = sep[sep["state"].isin(VALID_US)].copy()

print("Facility weeks:", len(fw))
print("Hospital episodes:", len(hep))
print("State episodes:", len(sep))


# ==============================================================
# 2. Objective HHS episode-quality audit
# ==============================================================

print("\n[2/10] APPLYING OBJECTIVE HHS QUALITY RULES")

lookup = {
    k: g.sort_values(date).copy()
    for k, g in fw.groupby(hid)
}

quality_rows = []

for idx, r in hep.iterrows():
    hospital = r["hospital_id"]
    sw = pd.Timestamp(r["signal_week"])
    g = lookup.get(hospital)

    if g is None:
        continue

    x = g[
        (g[date] >= sw - pd.Timedelta(weeks=4)) &
        (g[date] <= sw + pd.Timedelta(weeks=4))
    ].copy()

    if x.empty:
        continue

    x["rel_week"] = ((x[date] - sw).dt.days / 7).round().astype(int)

    base = x[x["rel_week"].between(-4, -1)].copy()
    post = x[x["rel_week"].isin([1, 2])].copy()
    outw = x[x["rel_week"].isin([2, 3, 4])].copy()

    # reporting coverage rules
    def cov_ok(df, cov_col):
        if cov_col is None:
            return np.ones(len(df), dtype=bool)
        return (df[cov_col] >= MIN_COVERAGE_DAYS).to_numpy()

    base_good = (
        np.isfinite(base[cap].to_numpy(float)) &
        np.isfinite(base[dem].to_numpy(float)) &
        cov_ok(base, cap_cov) &
        cov_ok(base, dem_cov)
    )

    post_good = (
        np.isfinite(post[cap].to_numpy(float)) &
        np.isfinite(post[dem].to_numpy(float)) &
        cov_ok(post, cap_cov) &
        cov_ok(post, dem_cov)
    )

    n_base_good = int(base_good.sum())
    n_post_good = int(post_good.sum())

    bcap = float(np.nanmedian(base.loc[base_good, cap])) if n_base_good >= 3 else np.nan
    bdem = float(np.nanmedian(base.loc[base_good, dem])) if n_base_good >= 3 else np.nan

    post_caps = post.loc[post_good, cap].to_numpy(float)
    post_dems = post.loc[post_good, dem].to_numpy(float)

    # sustained 2-week response requires both weeks +1 and +2
    has_sustained_post = (n_post_good == 2)

    if has_sustained_post and np.isfinite(bcap) and bcap > 0:
        cap_w1 = float(post_caps[0])
        cap_w2 = float(post_caps[1])
        sustained_capacity_change_pp = (
            ((cap_w1 + cap_w2) / 2) / bcap - 1
        ) * 100

        # structural discontinuity rule:
        # if adjacent response weeks disagree by more than max(100% baseline, 50 beds),
        # the change is not treated as a sustained operational response.
        max_allowed_adjacent_difference = max(bcap, 50.0)
        persistent_capacity_ok = (
            abs(cap_w2 - cap_w1) <= max_allowed_adjacent_difference
        )
    else:
        cap_w1 = cap_w2 = np.nan
        sustained_capacity_change_pp = np.nan
        persistent_capacity_ok = False

    if has_sustained_post and np.isfinite(bdem) and bdem > 0:
        sustained_demand_change_pp = (
            np.nanmean(post_dems) / bdem - 1
        ) * 100
    else:
        sustained_demand_change_pp = np.nan

    # objective primary-quality flag
    quality_ok = (
        n_base_good >= 3 and
        has_sustained_post and
        np.isfinite(bcap) and bcap > 0 and
        np.isfinite(bdem) and bdem > 0 and
        persistent_capacity_ok
    )

    quality_rows.append({
        "_hep_index": idx,
        "n_baseline_good_weeks": n_base_good,
        "n_post_good_weeks": n_post_good,
        "baseline_capacity_quality": bcap,
        "baseline_demand_quality": bdem,
        "capacity_week1": cap_w1,
        "capacity_week2": cap_w2,
        "persistent_capacity_ok": bool(persistent_capacity_ok),
        "objective_quality_ok": bool(quality_ok),
        "sustained_capacity_change_w1_2_pp": sustained_capacity_change_pp,
        "sustained_demand_change_w1_2_pp": sustained_demand_change_pp,
    })

q = pd.DataFrame(quality_rows).set_index("_hep_index")
hep = hep.join(q, how="left")

print("Objective quality pass:", int(hep["objective_quality_ok"].fillna(False).sum()))
print("Pass fraction:", float(hep["objective_quality_ok"].fillna(False).mean()))

hep.to_csv(
    OUTPUT / "AJPH_v12_hospital_objective_quality_audit.csv",
    index=False
)


# ==============================================================
# 3. Cross-fit sustained expected capacity response
# ==============================================================

print("\n[3/10] CROSS-FITTING SUSTAINED CAPACITY RESPONSE")

hq = hep[hep["objective_quality_ok"] == True].copy()

hq["signal_quarter"] = hq["signal_week"].dt.to_period("Q").astype(str)
hq["calendar_week_index"] = (
    (hq["signal_week"] - hq["signal_week"].min()).dt.days / 7
).astype(float)

target = "sustained_capacity_change_w1_2_pp"
num_features = [
    "sustained_demand_change_w1_2_pp",
    "baseline_demand_quality",
    "baseline_capacity_quality",
    "baseline_admissions",
    "pre_demand_slope",
    "signal_admissions_7d",
    "signal_rel_change",
    "calendar_week_index",
]
cat_features = ["signal_quarter"]

work = hq.dropna(
    subset=[target, "state"] + num_features + cat_features
).copy()

preds = []

for i, st in enumerate(sorted(work["state"].unique()), 1):
    tr = work[work["state"] != st].copy()
    te = work[work["state"] == st].copy()

    if len(tr) < 100 or len(te) == 0:
        continue

    prep = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ])

    pipe = Pipeline([
        ("prep", prep),
        ("reg", LinearRegression())
    ])

    pipe.fit(tr[num_features + cat_features], tr[target])
    yhat = pipe.predict(te[num_features + cat_features])

    for idx, v in zip(te.index, yhat):
        preds.append({
            "index": idx,
            "expected_sustained_capacity_change_pp": float(v)
        })

    if i % 10 == 0:
        print(f"  cross-fit {i}")

pred = pd.DataFrame(preds).set_index("index")
hq = hq.join(pred, how="left")

hq["sustained_capacity_responsiveness_cf_pp"] = (
    hq[target] - hq["expected_sustained_capacity_change_pp"]
)

# immediate pre-pattern from v9 cross-fitted placebo
hq["sustained_response_innovation_pp"] = (
    hq["sustained_capacity_responsiveness_cf_pp"]
    - hq["placebo_capacity_residual_cf_pp"]
)

print("\nObjective-quality innovation summary:")
print(hq["sustained_response_innovation_pp"].describe(
    percentiles=[.01,.025,.05,.95,.975,.99]
))

hq.to_csv(
    OUTPUT / "AJPH_v12_hospital_quality_filtered_panel.csv",
    index=False
)


# ==============================================================
# 4. Hospital-level primary objective-quality model
# ==============================================================

print("\n[4/10] HOSPITAL-LEVEL OBJECTIVE-QUALITY MODEL")

H_OUT = "inpatient_occupancy_w2_4"
H_EXP = "sustained_response_innovation_pp"
H_CONTROLS = [
    "sustained_demand_change_w1_2_pp",
    "placebo_demand_pre_pp",
    "baseline_demand_quality",
    "pre_demand_slope",
    "signal_admissions_7d",
]

need = [H_OUT, H_EXP, "hospital_id", "state", "signal_quarter"] + H_CONTROLS
hd = hq.dropna(subset=need).copy()
hd["hospital_id"] = hd["hospital_id"].astype(str).astype(object)
hd["state"] = hd["state"].astype(str).astype(object)
hd["signal_quarter"] = hd["signal_quarter"].astype(str).astype(object)

rz = iterative_absorb(
    hd,
    [H_OUT, H_EXP] + H_CONTROLS,
    ["hospital_id", "signal_quarter"]
)

hm = cluster_fit(
    rz[H_OUT],
    rz[[H_EXP] + H_CONTROLS],
    hd["state"]
)

h_ci = hm.conf_int()[0]
h_std = (
    hm.params[0] * hd[H_EXP].std(ddof=1) / hd[H_OUT].std(ddof=1)
)

hospital_primary = {
    "n": int(len(hd)),
    "hospitals": int(hd["hospital_id"].nunique()),
    "states": int(hd["state"].nunique()),
    "coef": float(hm.params[0]),
    "se_cluster": float(hm.bse[0]),
    "p_value": float(hm.pvalues[0]),
    "ci_low": float(h_ci[0]),
    "ci_high": float(h_ci[1]),
    "std_beta": float(h_std),
}

print(json.dumps(hospital_primary, indent=2))

print("\nHospital wild cluster bootstrap:")
hospital_wild = wild_bootstrap_absorbed(
    hd, rz, H_OUT, H_EXP, H_CONTROLS
)
print(json.dumps(hospital_wild, indent=2))


# ==============================================================
# 5. Build state-level hospital response bridge
# ==============================================================

print("\n[5/10] BUILDING STATE-LEVEL HOSPITAL RESPONSE INDEX")

bridge_rows = []

for idx, s in sep.iterrows():
    st = s["state"]
    sd = pd.Timestamp(s["signal_date"])

    matches = hq[
        (hq["state"] == st) &
        (hq["signal_week"] >= sd - pd.Timedelta(days=14)) &
        (hq["signal_week"] <= sd + pd.Timedelta(days=14)) &
        hq["sustained_response_innovation_pp"].notna() &
        hq["baseline_capacity_quality"].notna()
    ].copy()

    if matches.empty:
        bridge_rows.append({"_sep_index": idx})
        continue

    weights = matches["baseline_capacity_quality"].to_numpy(float)
    innov = matches["sustained_response_innovation_pp"].to_numpy(float)
    placebo = matches["placebo_capacity_residual_cf_pp"].to_numpy(float)

    bridge_rows.append({
        "_sep_index": idx,
        "state_hospital_episode_count": int(len(matches)),
        "state_hospital_count": int(matches["hospital_id"].nunique()),
        "state_hospital_capacity_weight": float(np.nansum(weights)),
        "state_weighted_hospital_response_innovation_pp":
            weighted_mean(innov, weights),
        "state_weighted_hospital_placebo_pp":
            weighted_mean(placebo, weights),
        "state_unweighted_hospital_response_innovation_pp":
            float(np.nanmean(innov)),
    })

bridge = pd.DataFrame(bridge_rows).set_index("_sep_index")
sep2 = sep.join(bridge, how="left")

sep2.to_csv(
    OUTPUT / "AJPH_v12_state_hospital_response_bridge.csv",
    index=False
)

print("State episodes with hospital bridge:",
      int(sep2["state_weighted_hospital_response_innovation_pp"].notna().sum()))
print("Median contributing hospitals:",
      sep2["state_hospital_count"].median())


# ==============================================================
# 6. State mortality models with hospital-derived exposure
# ==============================================================

print("\n[6/10] STATE-LEVEL MORTALITY MODELS")

S_OUT = "excess_deaths_14_42_per100k"
S_EXP = "state_weighted_hospital_response_innovation_pp"

sep2["signal_month"] = sep2["signal_date"].dt.to_period("M").astype(str).astype(object)
sep2["signal_quarter"] = sep2["signal_date"].dt.to_period("Q").astype(str).astype(object)
sep2["calendar_day"] = (
    sep2["signal_date"] - sep2["signal_date"].min()
).dt.days.astype(float)
sep2["state"] = sep2["state"].astype(str).astype(object)
if "wave" in sep2.columns:
    sep2["wave"] = sep2["wave"].astype(str).astype(object)

S_CONTROLS = [
    "signal_admissions_7dma",
    "state_hospital_count",
]

def state_model(time_mode):
    need = [S_OUT, S_EXP, "state"] + S_CONTROLS

    if time_mode == "wave":
        need.append("wave")
    elif time_mode == "quarter":
        need.append("signal_quarter")
    elif time_mode == "month":
        need.append("signal_month")
    elif time_mode == "linear":
        need.append("calendar_day")

    d = sep2.dropna(subset=need).copy()

    if time_mode == "wave":
        fe = ["state", "wave"]
        controls = S_CONTROLS
    elif time_mode == "quarter":
        fe = ["state", "signal_quarter"]
        controls = S_CONTROLS
    elif time_mode == "month":
        fe = ["state", "signal_month"]
        controls = S_CONTROLS
    elif time_mode == "linear":
        fe = ["state"]
        controls = S_CONTROLS + ["calendar_day"]
    else:
        raise ValueError(time_mode)

    rz = iterative_absorb(
        d, [S_OUT, S_EXP] + controls, fe
    )

    m = cluster_fit(
        rz[S_OUT],
        rz[[S_EXP] + controls],
        d["state"]
    )

    ci = m.conf_int()[0]
    stdb = m.params[0] * d[S_EXP].std(ddof=1) / d[S_OUT].std(ddof=1)

    return {
        "time_control": time_mode,
        "n": int(len(d)),
        "states": int(d["state"].nunique()),
        "coef": float(m.params[0]),
        "se_cluster": float(m.bse[0]),
        "p_value": float(m.pvalues[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "std_beta": float(stdb),
    }, d, rz, controls

state_rows = []
for tm in ["wave","quarter","month","linear"]:
    r, *_ = state_model(tm)
    state_rows.append(r)

state_results = pd.DataFrame(state_rows)
print(state_results.to_string(index=False))
state_results.to_csv(
    OUTPUT / "AJPH_v12_state_mortality_models.csv",
    index=False
)


# ==============================================================
# 7. State placebo
# ==============================================================

print("\n[7/10] STATE-LEVEL PLACEBO")

P_EXP = "state_weighted_hospital_placebo_pp"
pneed = [
    S_OUT, P_EXP, "signal_admissions_7dma",
    "state_hospital_count", "state", "wave"
]
pdta = sep2.dropna(subset=pneed).copy()

pcontrols = ["signal_admissions_7dma","state_hospital_count"]
prz = iterative_absorb(
    pdta,
    [S_OUT, P_EXP] + pcontrols,
    ["state","wave"]
)
pm = cluster_fit(
    prz[S_OUT],
    prz[[P_EXP] + pcontrols],
    pdta["state"]
)
pci = pm.conf_int()[0]

state_placebo = {
    "n": int(len(pdta)),
    "states": int(pdta["state"].nunique()),
    "coef": float(pm.params[0]),
    "se_cluster": float(pm.bse[0]),
    "p_value": float(pm.pvalues[0]),
    "ci_low": float(pci[0]),
    "ci_high": float(pci[1]),
}
print(json.dumps(state_placebo, indent=2))


# ==============================================================
# 8. State LOSO
# ==============================================================

print("\n[8/10] STATE-LEVEL LOSO")

base_r, base_d, base_rz, base_controls = state_model("wave")
loso_rows = []

for i, st in enumerate(sorted(base_d["state"].astype(str).unique()), 1):
    temp = sep2[sep2["state"].astype(str) != st].copy()

    need = [S_OUT,S_EXP,"state","wave"] + S_CONTROLS
    d = temp.dropna(subset=need).copy()
    rz2 = iterative_absorb(
        d, [S_OUT,S_EXP] + S_CONTROLS,
        ["state","wave"]
    )
    m = cluster_fit(
        rz2[S_OUT],
        rz2[[S_EXP] + S_CONTROLS],
        d["state"]
    )
    ci = m.conf_int()[0]

    loso_rows.append({
        "excluded_state": st,
        "coef": float(m.params[0]),
        "p_value": float(m.pvalues[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    })

    if i % 10 == 0:
        print(f"  LOSO {i}")

loso = pd.DataFrame(loso_rows)
loso.to_csv(OUTPUT / "AJPH_v12_state_LOSO.csv", index=False)

state_loso = {
    "n_models": int(len(loso)),
    "fraction_negative": float((loso["coef"] < 0).mean()),
    "median_coef": float(loso["coef"].median()),
    "min_coef": float(loso["coef"].min()),
    "max_coef": float(loso["coef"].max()),
    "fraction_p_lt_0_05": float((loso["p_value"] < 0.05).mean()),
}
print(json.dumps(state_loso, indent=2))


# ==============================================================
# 9. State wild cluster bootstrap
# ==============================================================

print("\n[9/10] STATE-LEVEL WILD CLUSTER BOOTSTRAP")

state_wild = wild_bootstrap_absorbed(
    base_d, base_rz, S_OUT, S_EXP, base_controls
)
print(json.dumps(state_wild, indent=2))


# ==============================================================
# 10. Summary
# ==============================================================

print("\n[10/10] FINAL SUMMARY")

summary = {
    "objective_hospital_quality": {
        "episodes_total": int(len(hep)),
        "episodes_pass": int(hep["objective_quality_ok"].fillna(False).sum()),
        "pass_fraction": float(hep["objective_quality_ok"].fillna(False).mean()),
    },
    "hospital_primary": hospital_primary,
    "hospital_wild": hospital_wild,
    "state_bridge": {
        "state_episodes_with_bridge": int(
            sep2["state_weighted_hospital_response_innovation_pp"].notna().sum()
        ),
        "median_contributing_hospitals": float(
            sep2["state_hospital_count"].median()
        ),
    },
    "state_models": state_rows,
    "state_placebo": state_placebo,
    "state_loso": state_loso,
    "state_wild": state_wild,
}

(OUTPUT / "AJPH_v12_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print(json.dumps(summary, indent=2))
print("\nCOMPLETE")
print("Hospital quality audit:", OUTPUT / "AJPH_v12_hospital_objective_quality_audit.csv")
print("Hospital filtered panel:", OUTPUT / "AJPH_v12_hospital_quality_filtered_panel.csv")
print("State bridge:", OUTPUT / "AJPH_v12_state_hospital_response_bridge.csv")
print("State mortality models:", OUTPUT / "AJPH_v12_state_mortality_models.csv")
print("Summary:", OUTPUT / "AJPH_v12_summary.json")
