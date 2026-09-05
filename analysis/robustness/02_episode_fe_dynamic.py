# PUBLIC REPOSITORY NOTE:
# These scripts were developed on Windows and may contain historical absolute-path defaults.
# Before running, set ROOT / data paths to your local clone and downloaded public datasets.
# Raw HHS/CDC/NCHS source data are not redistributed in this repository.
#

"""
AJPH U.S. Empirical Upgrade v15
Episode-Fixed-Effects Dynamic Event Study + Pre/Post Change Model
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

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

EVENT_WEEKS = [-4,-3,-2,-1,0,1,2,3,4,5,6]
REF_WEEK = -1
PRE_WEEKS = [-4,-3,-2]
POST_WEEKS = [2,3,4]
EXP = "hospital_log_response_innovation"
OUTCOME = "inpatient_occupancy"
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
    return sm.OLS(np.asarray(y,float), np.asarray(X,float)).fit(
        cov_type="cluster",
        cov_kwds={"groups": np.asarray(groups,dtype=object)}
    )

def matrix_rank_info(X, names):
    X = np.asarray(X,float)
    r = int(np.linalg.matrix_rank(X))
    return {"n_columns":int(X.shape[1]),"rank":r,"full_rank":bool(r==X.shape[1]),"columns":list(names)}

def wild_cluster_average_effect(y, X_full, X_restricted, groups, effect_idx, reps=WILD_REPS, seed=SEED):
    ufit = cluster_fit(y, X_full, groups)
    beta = np.asarray(ufit.params,float)
    cov = np.asarray(ufit.cov_params(),float)
    a = np.zeros(len(beta),float)
    a[effect_idx] = 1.0/len(effect_idx)
    obs_eff = float(a@beta)
    obs_se = float(np.sqrt(a@cov@a))
    obs_t = obs_eff/obs_se

    rfit = sm.OLS(y, X_restricted).fit()
    yhat0 = np.asarray(rfit.fittedvalues,float)
    u0 = np.asarray(rfit.resid,float)

    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    bt, failed = [], 0

    for b in range(reps):
        signs = {g:rng.choice([-1.0,1.0]) for g in ug}
        ystar = yhat0 + u0*np.array([signs[g] for g in groups],float)
        try:
            mb = cluster_fit(ystar, X_full, groups)
            bvec = np.asarray(mb.params,float)
            bcov = np.asarray(mb.cov_params(),float)
            eff = float(a@bvec)
            se = float(np.sqrt(a@bcov@a))
            if np.isfinite(eff) and np.isfinite(se) and se>0:
                bt.append(eff/se)
            else:
                failed += 1
        except Exception:
            failed += 1
        if (b+1)%100==0:
            print(f"  bootstrap {b+1}/{reps}; valid={len(bt)}, failed={failed}")

    bt = np.asarray(bt,float)
    p = (np.sum(np.abs(bt)>=abs(obs_t))+1)/(len(bt)+1)
    return {
        "reps_requested":int(reps),
        "reps_valid":int(len(bt)),
        "reps_failed":int(failed),
        "observed_average_effect":obs_eff,
        "observed_cluster_se":obs_se,
        "observed_t":float(obs_t),
        "wild_cluster_bootstrap_p_two_sided":float(p),
        "bootstrap_t_p025":float(np.quantile(bt,.025)),
        "bootstrap_t_median":float(np.quantile(bt,.5)),
        "bootstrap_t_p975":float(np.quantile(bt,.975))
    }, bt

print("\n[1/8] LOADING EVENT PANEL")
panel = pd.read_csv(EVENT_PANEL, low_memory=False)
v13 = pd.read_csv(V13_PANEL, low_memory=False)

for d in [panel,v13]:
    d["state"] = d["state"].astype(str).str.upper().str.strip()
    d["hospital_id"] = d["hospital_id"].astype(str).str.strip()
    d["signal_week"] = pd.to_datetime(d["signal_week"], errors="coerce")

panel = panel[panel["state"].isin(VALID_US)].copy()
v13 = v13[v13["state"].isin(VALID_US)].copy()

panel["event_week"] = pd.to_numeric(panel["event_week"], errors="coerce")
panel = panel[panel["event_week"].isin(EVENT_WEEKS)].copy()
panel["event_week"] = panel["event_week"].astype(int)
panel["calendar_week"] = pd.to_datetime(panel["calendar_week"], errors="coerce")
panel["calendar_week_fe"] = panel["calendar_week"].dt.to_period("W-SUN").astype(str).astype(object)
panel["episode_id"] = (panel["hospital_id"].astype(str)+"__"+panel["signal_week"].dt.strftime("%Y-%m-%d")).astype(object)
panel["event_week_fe"] = panel["event_week"].astype(str).astype(object)
panel["state"] = panel["state"].astype(str).astype(object)

print("Rows:", len(panel))
print("Episodes:", panel["episode_id"].nunique())
print("Hospitals:", panel["hospital_id"].nunique())
print("States/DC:", panel["state"].nunique())

print("\n[2/8] EPISODE-FE EVENT-STUDY")
interaction_cols=[]
for k in EVENT_WEEKS:
    if k==REF_WEEK:
        continue
    c=f"resp_x_event_{k}"
    panel[c]=panel[EXP]*(panel["event_week"]==k).astype(float)
    interaction_cols.append(c)

need=[OUTCOME,"episode_id","calendar_week_fe","event_week_fe","state"]+interaction_cols
d=panel.dropna(subset=need).copy()
rz=iterative_absorb(d,[OUTCOME]+interaction_cols,["episode_id","calendar_week_fe","event_week_fe"])

y=rz[OUTCOME].to_numpy(float)
X=rz[interaction_cols].to_numpy(float)
groups=d["state"].astype(str).to_numpy(object)

rank_info=matrix_rank_info(X, interaction_cols)
print("Design rank:")
print(json.dumps(rank_info,indent=2))
if not rank_info["full_rank"]:
    raise ValueError("Event-study interaction matrix is not full rank after FE absorption.")

m=cluster_fit(y,X,groups)
coef_rows=[]
for j,c in enumerate(interaction_cols):
    k=int(c.split("_")[-1])
    ci=m.conf_int()[j]
    coef_rows.append({
        "event_week":k,
        "coef":float(m.params[j]),
        "se_cluster":float(m.bse[j]),
        "p_value":float(m.pvalues[j]),
        "ci_low":float(ci[0]),
        "ci_high":float(ci[1]),
    })
event_res=pd.DataFrame(coef_rows).sort_values("event_week")
print(event_res.to_string(index=False))
event_res.to_csv(OUTPUT/"AJPH_v15_episodeFE_event_study_coefficients.csv",index=False)

print("\n[3/8] JOINT PRE/POST TESTS")
pre_idx=[interaction_cols.index(f"resp_x_event_{k}") for k in PRE_WEEKS]
post_idx=[interaction_cols.index(f"resp_x_event_{k}") for k in POST_WEEKS]

Rpre=np.zeros((len(pre_idx),len(interaction_cols)))
for i,j in enumerate(pre_idx): Rpre[i,j]=1.0
Rpost=np.zeros((len(post_idx),len(interaction_cols)))
for i,j in enumerate(post_idx): Rpost[i,j]=1.0

pre_wald=m.wald_test(Rpre,scalar=True)
post_wald=m.wald_test(Rpost,scalar=True)
joint={
    "pre_weeks":PRE_WEEKS,
    "pre_wald_stat":float(pre_wald.statistic),
    "pre_wald_p":float(pre_wald.pvalue),
    "post_weeks":POST_WEEKS,
    "post_wald_stat":float(post_wald.statistic),
    "post_wald_p":float(post_wald.pvalue)
}
print(json.dumps(joint,indent=2))

print("\n[4/8] WILD CLUSTER BOOTSTRAP: AVG POST EFFECT")
restricted_cols=[c for c in interaction_cols if c not in [f"resp_x_event_{k}" for k in POST_WEEKS]]
Xr=rz[restricted_cols].to_numpy(float)
post_boot,bt=wild_cluster_average_effect(y,X,Xr,groups,post_idx)
print(json.dumps(post_boot,indent=2))
pd.DataFrame({"bootstrap_t":bt}).to_csv(OUTPUT/"AJPH_v15_episodeFE_post_average_bootstrap_t.csv",index=False)

print("\n[5/8] PRE-TO-POST OCCUPANCY CHANGE MODEL")
keys=["hospital_id","state","signal_week"]
pre=(panel[panel["event_week"].isin(PRE_WEEKS)].groupby(keys,as_index=False)[OUTCOME].mean()
     .rename(columns={OUTCOME:"occupancy_pre_m4_m2"}))
post=(panel[panel["event_week"].isin(POST_WEEKS)].groupby(keys,as_index=False)[OUTCOME].mean()
      .rename(columns={OUTCOME:"occupancy_post_p2_p4"}))
epi=v13.merge(pre,on=keys,how="left").merge(post,on=keys,how="left")
epi["delta_occupancy_post_minus_pre"]=epi["occupancy_post_p2_p4"]-epi["occupancy_pre_m4_m2"]
epi["signal_quarter"]=epi["signal_week"].dt.to_period("Q").astype(str).astype(object)
epi["hospital_id"]=epi["hospital_id"].astype(str).astype(object)
epi["state"]=epi["state"].astype(str).astype(object)

change_need=["delta_occupancy_post_minus_pre",EXP,"hospital_id","signal_quarter","state"]
cd=epi.dropna(subset=change_need).copy()
crz=iterative_absorb(cd,["delta_occupancy_post_minus_pre",EXP],["hospital_id","signal_quarter"])
cm=cluster_fit(crz["delta_occupancy_post_minus_pre"],crz[[EXP]],cd["state"])
cci=cm.conf_int()[0]
change_result={
    "n":int(len(cd)),
    "hospitals":int(cd["hospital_id"].nunique()),
    "states":int(cd["state"].nunique()),
    "coef":float(cm.params[0]),
    "se_cluster":float(cm.bse[0]),
    "p_value":float(cm.pvalues[0]),
    "ci_low":float(cci[0]),
    "ci_high":float(cci[1]),
    "std_beta":float(cm.params[0]*cd[EXP].std(ddof=1)/cd["delta_occupancy_post_minus_pre"].std(ddof=1))
}
print(json.dumps(change_result,indent=2))

print("\n[6/8] CHANGE-MODEL LOSO")
loso_rows=[]
for i,st in enumerate(sorted(cd["state"].astype(str).unique()),1):
    temp=epi[epi["state"].astype(str)!=st].dropna(subset=change_need).copy()
    trz=iterative_absorb(temp,["delta_occupancy_post_minus_pre",EXP],["hospital_id","signal_quarter"])
    tm=cluster_fit(trz["delta_occupancy_post_minus_pre"],trz[[EXP]],temp["state"])
    loso_rows.append({"excluded_state":st,"coef":float(tm.params[0]),"p_value":float(tm.pvalues[0])})
    if i%10==0: print(f"  LOSO {i}")

loso=pd.DataFrame(loso_rows)
loso.to_csv(OUTPUT/"AJPH_v15_change_model_LOSO.csv",index=False)
loso_summary={
    "n_models":int(len(loso)),
    "fraction_negative":float((loso["coef"]<0).mean()),
    "median_coef":float(loso["coef"].median()),
    "min_coef":float(loso["coef"].min()),
    "max_coef":float(loso["coef"].max()),
    "fraction_p_lt_0_05":float((loso["p_value"]<.05).mean())
}
print(json.dumps(loso_summary,indent=2))

print("\n[7/8] RANK-SAFE SPLINE ON CHANGE OUTCOME")
sd=cd.copy()
basis=dmatrix("0 + bs(x, df=4, degree=3, include_intercept=False)",{"x":sd[EXP].to_numpy(float)},return_type="dataframe")
basis.columns=[f"spline_{i+1}" for i in range(basis.shape[1])]
for c in basis.columns: sd[c]=basis[c].to_numpy(float)
spline_cols=list(basis.columns)

srz=iterative_absorb(sd,["delta_occupancy_post_minus_pre"]+spline_cols,["hospital_id","signal_quarter"])
SX=srz[spline_cols].to_numpy(float)
spline_rank=matrix_rank_info(SX,spline_cols)
print("Spline rank:")
print(json.dumps(spline_rank,indent=2))

if not spline_rank["full_rank"]:
    keep=[]
    current=np.empty((len(sd),0),float)
    current_rank=0
    for c in spline_cols:
        cand=np.column_stack([current,srz[c].to_numpy(float)])
        r=np.linalg.matrix_rank(cand)
        if r>current_rank:
            keep.append(c)
            current=cand
            current_rank=r
    spline_cols=keep
    SX=srz[spline_cols].to_numpy(float)

sm_spline=cluster_fit(srz["delta_occupancy_post_minus_pre"],SX,sd["state"])
R=np.eye(len(spline_cols))
swald=sm_spline.wald_test(R,scalar=True)
spline_summary={
    "n":int(len(sd)),
    "hospitals":int(sd["hospital_id"].nunique()),
    "states":int(sd["state"].nunique()),
    "basis_columns_used":spline_cols,
    "basis_rank":int(np.linalg.matrix_rank(SX)),
    "global_wald_stat":float(swald.statistic),
    "global_p_value":float(swald.pvalue)
}
print(json.dumps(spline_summary,indent=2))

print("\n[8/8] FINAL SUMMARY")
figure_table=pd.DataFrame({"event_week":EVENT_WEEKS}).merge(event_res,on="event_week",how="left")
figure_table.loc[figure_table["event_week"]==REF_WEEK,["coef","se_cluster","p_value","ci_low","ci_high"]]=[0.0,np.nan,np.nan,0.0,0.0]
figure_table.to_csv(OUTPUT/"AJPH_v15_episodeFE_event_study_figure_table.csv",index=False)

summary={
    "sample":{
        "rows":int(len(d)),
        "episodes":int(d["episode_id"].nunique()),
        "hospitals":int(d["hospital_id"].nunique()),
        "states":int(d["state"].nunique())
    },
    "design_rank":rank_info,
    "joint_tests":joint,
    "post_average_wild_bootstrap":post_boot,
    "change_model":change_result,
    "change_model_loso":loso_summary,
    "spline":spline_summary
}
(OUTPUT/"AJPH_v15_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
print("\nCOMPLETE")
print("Event coefficients:", OUTPUT/"AJPH_v15_episodeFE_event_study_coefficients.csv")
print("Figure table:", OUTPUT/"AJPH_v15_episodeFE_event_study_figure_table.csv")
print("Change-model LOSO:", OUTPUT/"AJPH_v15_change_model_LOSO.csv")
print("Summary:", OUTPUT/"AJPH_v15_summary.json")
