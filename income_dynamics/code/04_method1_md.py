import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Minimum-distance estimation of the income process, by family background.

Sample: child-year rows of panel_resid with a non-missing winsorized residual
y_tilde_w and a non-missing HighParentOcc14 (children aged 22 to 55, waves
2014, 2016, 2018, 2020 and 2022, at least three waves with positive income).
Group H is the advantaged children (HighParentOcc14 = 1: at least one parent in
an official, managerial, or professional occupation when the child was
fourteen); group L is the less advantaged children (HighParentOcc14 = 0).

Outcome: y_tilde_w, the residual of log income winsorized at the 1st and 99th
percentiles of each wave.

Steps:
  1. Demean y_tilde_w within group and year.
  2. Compute the empirical autocovariances for every pair of waves on the
     unbalanced panel, separately for the pooled sample, group H and group L.
  3. Fit two models of the autocovariances, for each group:
        (FE)  permanent effect plus AR(1) transitory component
              y = eta_i + v_it,   v_it = rho * v_{i,t-1} + eps_it
              parameters (sigma_eta^2, rho, sigma_eps^2)
        (RW)  random-walk permanent component plus AR(1) transitory component
              eta_it = eta_{i,0} + sum of u, eta_{i,0} with variance sigma_eta0^2
              parameters (sigma_eta0^2, sigma_u^2, rho, sigma_eps^2)
     Each model is fitted by minimum distance in two steps: equal weights
     first, then weights proportional to the number of children behind each
     autocovariance. The overidentification statistic J is reported.
  4. Hall-Mishkin ratio of differenced autocovariances
        rho_v = (cov_1 - cov_2) / (cov_0 - cov_1)
  5. Share of the permanent effect in the variance of the residual,
     sigma_eta^2 / Var(y), by group.
  6. Standard errors of the differences between groups from a bootstrap that
     resamples children (1,000 replications).

The same steps are repeated on the four-wave panel without the 2016 wave.

Output:
  data/intermediate/method1_results.csv          (five-wave panel, both models)
  data/intermediate/method1_autocov_pooled.csv
  data/intermediate/method1_autocov_H.csv
  data/intermediate/method1_autocov_L.csv
  data/intermediate/method1_bootstrap.csv         (bootstrap standard errors)
  data/intermediate/method1_drop2016_results.csv  (four-wave panel without 2016)
  and a summary printed to the console
"""
import os, math, time
import numpy as np
import pandas as pd
import pyreadstat
from scipy.optimize import minimize

BASE_INT = _WSROOT + "/data/intermediate"

_pkl = os.path.join(BASE_INT, "panel_resid.pkl")
_dta = os.path.join(BASE_INT, "panel_resid.dta")
if os.path.exists(_pkl) and os.path.getsize(_pkl) > 1000:
    df = pd.read_pickle(_pkl)
else:
    df, _ = pyreadstat.read_dta(_dta)

# Keep the rows with a non-missing y_tilde_w and a non-missing HighParentOcc14
df = df.dropna(subset=["y_tilde_w", "HighParentOcc14"]).copy()
df["pid"]   = df["pid"].astype(np.int64)
df["year"]  = df["year"].astype(int)
df["HP14"]  = df["HighParentOcc14"].astype(int)

WAVES = sorted(df["year"].unique().tolist())
T = len(WAVES)
print(f"loaded panel_resid (HighParentOcc14 non-missing): {len(df):,} rows, "
      f"{df['pid'].nunique():,} children, T={T} waves: {WAVES}")
print(f"  N_H = {df.drop_duplicates('pid')['HP14'].sum()}, "
      f"N_L = {(df.drop_duplicates('pid')['HP14']==0).sum()}")
print(f"  Outcome: y_tilde_w (winsorized at within-year p1/p99)")

# ---------------------------------------------------------------------------
# 1. Demean within group and year: yc
# ---------------------------------------------------------------------------
def add_demean(d, group_col, ycol="y_tilde_w"):
    out = d.copy()
    out["yc"] = out[ycol] - out.groupby([group_col, "year"])[ycol].transform("mean")
    return out

df_g = add_demean(df, "HP14")

# ---------------------------------------------------------------------------
# 2. Empirical autocovariances for every pair of waves
# ---------------------------------------------------------------------------
def pairwise_cov(d, waves, ycol="yc"):
    rows = []
    for t in waves:
        for s in waves:
            if s < t: continue
            yt = d.loc[d["year"]==t, ["pid", ycol]].rename(columns={ycol:"yt"})
            ys = d.loc[d["year"]==s, ["pid", ycol]].rename(columns={ycol:"ys"})
            both = yt.merge(ys, on="pid", how="inner")
            if len(both) == 0: continue
            rows.append((t, s, len(both), float((both["yt"]*both["ys"]).mean())))
    return pd.DataFrame(rows, columns=["t","s","N_ts","omega_hat"])


# ---------------------------------------------------------------------------
# 3. Model autocovariances and the minimum-distance fit
#    model_cov_FE: permanent effect plus AR(1) transitory component
#    model_cov_RW: random-walk permanent component plus AR(1) transitory component
#    Waves are two years apart, so the lag in waves is |s - t| / 2.
# ---------------------------------------------------------------------------
def model_cov_FE(theta, t, s):
    sigma_eta2, rho, sigma_eps2 = theta
    if abs(rho) >= 0.9999 or sigma_eta2 < 0 or sigma_eps2 < 0: return np.nan
    k = abs(s - t) // 2
    return sigma_eta2 + (rho**k) * sigma_eps2 / (1 - rho**2)

def model_cov_RW(theta, t, s, t0):
    sigma_eta0_2, sigma_u2, rho, sigma_eps2 = theta
    if abs(rho) >= 0.9999 or any(p < 0 for p in [sigma_eta0_2, sigma_u2, sigma_eps2]):
        return np.nan
    tw = (t - t0) // 2 + 1
    sw = (s - t0) // 2 + 1
    k = abs(s - t) // 2
    return sigma_eta0_2 + min(tw, sw) * sigma_u2 + (rho**k) * sigma_eps2 / (1 - rho**2)

def md_objective(theta, cov_df, model, weight=None, t0=None):
    res = []
    for _, r in cov_df.iterrows():
        h = model(theta, r["t"], r["s"]) if t0 is None else model(theta, r["t"], r["s"], t0)
        if not np.isfinite(h): return 1e10
        res.append(r["omega_hat"] - h)
    res = np.array(res)
    if weight is not None:
        return float(res @ np.diag(weight) @ res)
    return float(res @ res)

def fit_md(cov_df, model_name="FE"):
    cov_df = cov_df.copy()
    if model_name == "FE":
        x0 = [0.3, 0.5, 0.3]
        bounds = [(1e-6, 5.0), (-0.95, 0.95), (1e-6, 5.0)]
        model = model_cov_FE; names = ["sigma_eta2","rho","sigma_eps2"]; kwargs = {}
    else:
        x0 = [0.3, 0.05, 0.3, 0.3]
        bounds = [(1e-6,5.0),(1e-6,1.0),(-0.95,0.95),(1e-6,5.0)]
        model = model_cov_RW; names = ["sigma_eta0_2","sigma_u2","rho","sigma_eps2"]
        kwargs = {"t0": int(cov_df["t"].min())}

    res1 = minimize(md_objective, x0, args=(cov_df, model, None) + tuple(kwargs.values()),
                    method="L-BFGS-B", bounds=bounds)
    weight = cov_df["N_ts"].values / cov_df["N_ts"].sum()
    res2 = minimize(md_objective, res1.x, args=(cov_df, model, weight) + tuple(kwargs.values()),
                    method="L-BFGS-B", bounds=bounds)
    theta_hat = res2.x

    N_total = cov_df["N_ts"].sum()
    obj_val = md_objective(theta_hat, cov_df, model, weight, **kwargs)
    J = float(N_total * obj_val)
    df_overid = len(cov_df) - len(theta_hat)

    out = {names[i]: float(theta_hat[i]) for i in range(len(names))}
    out["J"] = J; out["df_overid"] = df_overid
    if model_name == "FE":
        out["var_total"] = out["sigma_eta2"] + out["sigma_eps2"] / (1 - out["rho"]**2)
        out["s_perm"]    = out["sigma_eta2"] / out["var_total"]
    else:
        Tw = T
        perm_var_endsample = out["sigma_eta0_2"] + Tw * out["sigma_u2"]
        out["var_total_endsample"] = perm_var_endsample + out["sigma_eps2"] / (1 - out["rho"]**2)
        out["s_perm_endsample"]    = perm_var_endsample / out["var_total_endsample"]
    return out


def diag_lag_moments(cov_df):
    diag = cov_df.loc[cov_df["t"]==cov_df["s"]]
    var_avg = (diag["omega_hat"] * diag["N_ts"]).sum() / diag["N_ts"].sum()
    cov1 = cov_df.loc[cov_df["s"]==cov_df["t"]+2]
    cov2 = cov_df.loc[cov_df["s"]==cov_df["t"]+4]
    c1 = (cov1["omega_hat"]*cov1["N_ts"]).sum()/cov1["N_ts"].sum() if len(cov1) else np.nan
    c2 = (cov2["omega_hat"]*cov2["N_ts"]).sum()/cov2["N_ts"].sum() if len(cov2) else np.nan
    return dict(var=var_avg, cov1=c1, cov2=c2,
                r1=c1/var_avg if not np.isnan(c1) else np.nan,
                r2=c2/var_avg if not np.isnan(c2) else np.nan)


# ---------------------------------------------------------------------------
# 4. Five-wave panel 2014 to 2022
# ---------------------------------------------------------------------------
def run_panel(d, waves, label):
    """Run the minimum-distance fits on the panel of the given waves; return the fits and the autocovariance tables."""
    sub = d[d["year"].isin(waves)].copy()
    sub_g = add_demean(sub, "HP14")
    cov_pooled = pairwise_cov(add_demean(sub, group_col="HP14")[["pid","year","yc"]]
                                 if False else sub_g, waves)
    # Pooled sample: demeaned by year only
    sub_p = sub.copy()
    sub_p["yc"] = sub_p["y_tilde_w"] - sub_p.groupby("year")["y_tilde_w"].transform("mean")
    cov_pooled = pairwise_cov(sub_p, waves)
    cov_H = pairwise_cov(sub_g.loc[sub_g["HP14"]==1], waves)
    cov_L = pairwise_cov(sub_g.loc[sub_g["HP14"]==0], waves)
    print(f"\n[{label}] pairs: pooled={len(cov_pooled)}, H={len(cov_H)}, L={len(cov_L)}")
    print(f"  smallest H pair (N): {cov_H['N_ts'].min()}; smallest L pair (N): {cov_L['N_ts'].min()}")

    fits_FE, fits_RW = {}, {}
    for lab, c in [("pooled", cov_pooled), ("H", cov_H), ("L", cov_L)]:
        fits_FE[lab] = fit_md(c, "FE")
        fits_RW[lab] = fit_md(c, "RW")

    # Hall-Mishkin ratio
    HM = {}
    for lab, c in [("pooled", cov_pooled), ("H", cov_H), ("L", cov_L)]:
        m = diag_lag_moments(c)
        HM[lab] = (m["cov1"] - m["cov2"]) / (m["var"] - m["cov1"]) \
                  if not np.isnan(m["cov2"]) else np.nan

    return dict(cov_pooled=cov_pooled, cov_H=cov_H, cov_L=cov_L,
                fits_FE=fits_FE, fits_RW=fits_RW, HM=HM, label=label,
                N_chi=sub["pid"].nunique(),
                N_H=int(sub.drop_duplicates("pid")["HP14"].sum()))


# Five-wave panel
print("\n" + "="*72)
print("Five-wave panel 2014-2022 (HighParentOcc14, y_tilde_w)")
print("="*72)
res5 = run_panel(df, WAVES, "5-WAVE HEADLINE")
for lab in ["pooled", "H", "L"]:
    f = res5["fits_FE"][lab]
    print(f"  FE {lab:>6s}: σ²_η={f['sigma_eta2']:.4f}, ρ={f['rho']:+.4f}, "
          f"σ²_ε={f['sigma_eps2']:.4f}, s_perm={f['s_perm']:.4f}, "
          f"J={f['J']:.2f} (df={f['df_overid']})")
print(f"  HM ρ_v: H={res5['HM']['H']:+.4f}, L={res5['HM']['L']:+.4f}, "
      f"H-L gap={res5['HM']['H']-res5['HM']['L']:+.4f}")
print(f"  s_perm gap (H-L) = {res5['fits_FE']['H']['s_perm'] - res5['fits_FE']['L']['s_perm']:+.4f}")
print(f"  ρ_v gap (H-L) (FE-MD) = {res5['fits_FE']['H']['rho'] - res5['fits_FE']['L']['rho']:+.4f}")

# Write the autocovariance tables and the estimates of the five-wave panel
res5["cov_pooled"].to_csv(os.path.join(BASE_INT, "method1_autocov_pooled.csv"), index=False)
res5["cov_H"].to_csv(os.path.join(BASE_INT, "method1_autocov_H.csv"), index=False)
res5["cov_L"].to_csv(os.path.join(BASE_INT, "method1_autocov_L.csv"), index=False)

rows = []
for lab in ["pooled","H","L"]:
    for model in ["FE","RW"]:
        f = (res5["fits_FE"] if model=="FE" else res5["fits_RW"])[lab].copy()
        f["group"] = lab; f["model"] = model
        rows.append(f)
pd.DataFrame(rows).to_csv(os.path.join(BASE_INT, "method1_results.csv"), index=False)

# ---------------------------------------------------------------------------
# 5. Four-wave panel without 2016
# ---------------------------------------------------------------------------
print("\n" + "="*72)
print("Four-wave panel without 2016")
print("="*72)
W4 = [w for w in WAVES if w != 2016]
res4 = run_panel(df, W4, "DROP-2016")
for lab in ["pooled", "H", "L"]:
    f = res4["fits_FE"][lab]
    print(f"  FE {lab:>6s}: σ²_η={f['sigma_eta2']:.4f}, ρ={f['rho']:+.4f}, "
          f"σ²_ε={f['sigma_eps2']:.4f}, s_perm={f['s_perm']:.4f}, "
          f"J={f['J']:.2f} (df={f['df_overid']})")
print(f"  HM ρ_v: H={res4['HM']['H']:+.4f}, L={res4['HM']['L']:+.4f}, "
      f"H-L gap={res4['HM']['H']-res4['HM']['L']:+.4f}")
print(f"  s_perm gap (H-L) = {res4['fits_FE']['H']['s_perm'] - res4['fits_FE']['L']['s_perm']:+.4f}")

rows4 = []
for lab in ["pooled","H","L"]:
    for model in ["FE","RW"]:
        f = (res4["fits_FE"] if model=="FE" else res4["fits_RW"])[lab].copy()
        f["group"] = lab; f["model"] = model
        rows4.append(f)
pd.DataFrame(rows4).to_csv(os.path.join(BASE_INT, "method1_drop2016_results.csv"), index=False)

# ---------------------------------------------------------------------------
# 6. Bootstrap standard errors of the differences between groups (five-wave
#    panel). Children are resampled with replacement; 1,000 replications take
#    about 25 seconds. Results are written to method1_bootstrap.csv.
# ---------------------------------------------------------------------------
print("\n" + "="*72)
print("BOOTSTRAP SE (five-wave panel, B=1000, children resampled)")
print("="*72)
year_to_idx = {y: i for i, y in enumerate(WAVES)}
df_b = df.copy()
df_b["wave_idx"] = df_b["year"].map(year_to_idx)
df_b["yc"] = df_b["y_tilde_w"] - df_b.groupby(["HP14","year"])["y_tilde_w"].transform("mean")

pair_lists = []  # for each child, an array of (wave index t, wave index s, product of the two residuals)
groups_arr = []
for pid_, sub in df_b.groupby("pid"):
    g = int(sub["HP14"].iloc[0])
    yrs = sub["wave_idx"].values
    yvs = sub["yc"].values
    pairs = []
    for i in range(len(yrs)):
        for j in range(i, len(yrs)):
            ti, sj = (yrs[i], yrs[j]) if yrs[i] <= yrs[j] else (yrs[j], yrs[i])
            pairs.append((ti, sj, yvs[i]*yvs[j]))
    pair_lists.append(np.array(pairs, dtype=np.float64))
    groups_arr.append(g)
groups_arr = np.array(groups_arr, dtype=np.int8)
N_pid = len(pair_lists)


def fit_g_fast(sums, counts, T):
    cov_pairs = []
    for t in range(T):
        for s in range(t, T):
            if counts[t,s] >= 5:
                cov_pairs.append((t, s, counts[t,s], sums[t,s]/counts[t,s]))
    if len(cov_pairs) < 6: return np.nan, np.nan, np.nan
    cov = np.array(cov_pairs, dtype=np.float64)
    is_var = cov[:,0] == cov[:,1]
    is_c1 = (cov[:,1] - cov[:,0]) == 1
    is_c2 = (cov[:,1] - cov[:,0]) == 2
    var_avg = float((cov[is_var,3]*cov[is_var,2]).sum() / cov[is_var,2].sum())
    c1 = float((cov[is_c1,3]*cov[is_c1,2]).sum() / cov[is_c1,2].sum()) if is_c1.any() else np.nan
    c2 = float((cov[is_c2,3]*cov[is_c2,2]).sum() / cov[is_c2,2].sum()) if is_c2.any() else np.nan
    HM = (c1 - c2) / (var_avg - c1) if not np.isnan(c2) else np.nan

    def model(theta, t, s):
        s2, rho, e2 = theta
        if abs(rho) >= 0.9999 or s2 < 0 or e2 < 0: return np.nan
        return s2 + (rho**abs(s-t)) * e2 / (1 - rho**2)
    def obj(theta, w):
        res = np.array([cov[i,3] - model(theta, cov[i,0], cov[i,1]) for i in range(len(cov))])
        if not np.isfinite(res).all(): return 1e10
        return float(res @ np.diag(w) @ res)
    w_arr = cov[:,2] / cov[:,2].sum()
    try:
        r = minimize(obj, [0.3,0.5,0.3], args=(w_arr,),
                     method="L-BFGS-B", bounds=[(1e-6,5),(-0.95,0.95),(1e-6,5)])
        s2, rho, e2 = r.x
        var_total = s2 + e2/(1-rho**2)
        s_perm = s2/var_total
    except Exception:
        return HM, np.nan, np.nan
    return HM, rho, s_perm


def boot_one(idx_sample, T):
    sums = [np.zeros((T,T)), np.zeros((T,T))]
    counts = [np.zeros((T,T), dtype=np.int64), np.zeros((T,T), dtype=np.int64)]
    for i in idx_sample:
        g = groups_arr[i]
        for row in pair_lists[i]:
            t, s, prod = int(row[0]), int(row[1]), row[2]
            sums[g][t,s] += prod; counts[g][t,s] += 1
    out = {g: fit_g_fast(sums[g], counts[g], T) for g in [0,1]}
    return (out[1][0]-out[0][0], out[1][1]-out[0][1], out[1][2]-out[0][2])


idx_full = np.arange(N_pid)
HM_pt, rho_pt, sp_pt = boot_one(idx_full, T)
print(f"  Point: HM gap={HM_pt:+.4f}, FE-MD ρ_v gap={rho_pt:+.4f}, s_perm gap={sp_pt:+.4f}")

np.random.seed(20260501)
B = 1000
t0 = time.time()
boot = np.full((B,3), np.nan)
for b in range(B):
    idx = np.random.randint(0, N_pid, size=N_pid)
    try:
        boot[b] = boot_one(idx, T)
    except Exception:
        pass
    if (b+1) % 200 == 0:
        print(f"    b={b+1}/{B}, elapsed={time.time()-t0:.1f}s, "
              f"HM_gap sd={np.nanstd(boot[:b+1,0]):.4f}")

def summ(arr, name, point):
    arr = arr[np.isfinite(arr)]
    se = arr.std(ddof=1)
    ci_lo, ci_hi = np.percentile(arr, [2.5, 97.5])
    z = abs(point/se) if se > 0 else np.nan
    p = 2*(1 - 0.5*(1+math.erf(z/np.sqrt(2)))) if not np.isnan(z) else np.nan
    star = "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else ""))
    print(f"  {name:<22}  point={point:+.4f}  SE={se:.4f}  "
          f"95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  p={p:.4f} {star}")
    return dict(point=point, se=se, ci_lo=ci_lo, ci_hi=ci_hi, p=p, B=len(arr))

print("\nBootstrap results:")
b_HM  = summ(boot[:,0], "HM ρ_v gap (H-L)",  HM_pt)
b_rho = summ(boot[:,1], "FE-MD ρ_v gap",     rho_pt)
b_sp  = summ(boot[:,2], "s_perm gap (H-L)",  sp_pt)
pd.DataFrame([
    {"metric":"HM_gap",     **b_HM},
    {"metric":"rho_gap",    **b_rho},
    {"metric":"s_perm_gap", **b_sp},
]).to_csv(os.path.join(BASE_INT, "method1_bootstrap.csv"), index=False)
print(f"\nsaved: method1_results.csv, method1_drop2016_results.csv, method1_bootstrap.csv, "
      f"method1_autocov_*.csv")
