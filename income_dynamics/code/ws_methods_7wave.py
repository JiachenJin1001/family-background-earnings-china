import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Estimates of the income process and of the premium on the seven-wave panel.

Sample: child-year rows of ws/panel_resid_7w.pkl (written by
ws_panel_7wave.py: children aged 22 to 55, waves 2010 to 2022 with income from
jobs reconstructed for 2010 and 2012, at least three waves with positive
income) with non-missing y_tilde_w and HighParentOcc14. Group H is the
advantaged children (HP14 = 1) and group L the less advantaged children (HP14 = 0).

Estimators, as in the scripts for the five-wave panel 2014 to 2022:
  1. Minimum distance with a permanent effect plus AR(1) transitory
     component, and the Hall-Mishkin ratio, by group (04_method1_md.py).
  2. Two-step Arellano-Bond GMM with collapsed instruments lagged two to four
     waves, with bootstrap standard errors (05_method2_abgmm.py).
  3. Random-coefficient model with child-specific trends and the regression
     of the trends on HP14 and ParentInc_bar (06_mini_method3_randcoef.py).
  4. Correlated random effects regression of log income (07_cre.py).

Output (data/intermediate/ws/): ws_method1_7wave.csv, ws_method2_7wave.csv,
ws_method3_7wave.csv, ws_cre_7wave.csv
"""
import os, math, time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

WS_OUT = _WSROOT + "/data/intermediate/ws"

df = pd.read_pickle(os.path.join(WS_OUT, "panel_resid_7w.pkl"))
df = df.dropna(subset=["y_tilde_w","HighParentOcc14"]).copy()
df["pid"] = df["pid"].astype(np.int64)
df["year"] = df["year"].astype(int)
df["HP14"] = df["HighParentOcc14"].astype(int)

WAVES = sorted(df["year"].unique().tolist())
T = len(WAVES)
print(f"Seven-wave panel: {len(df):,} rows, {df['pid'].nunique():,} children, T={T} waves: {WAVES}")
print(f"  N_H = {df.drop_duplicates('pid')['HP14'].sum()}, "
      f"N_L = {(df.drop_duplicates('pid')['HP14']==0).sum()}")
print(f"  T_i mean = {df.groupby('pid').size().mean():.2f}, "
      f"T_i ≥ 5: {(df.groupby('pid').size() >= 5).sum()}")

# ==========================================================
# 1. Minimum distance (permanent effect plus AR(1)) and Hall-Mishkin ratio
# ==========================================================
def add_demean_g(d, hp_col="HP14", y_col="y_tilde_w"):
    out = d.copy()
    out["yc"] = out[y_col] - out.groupby([hp_col,"year"])[y_col].transform("mean")
    return out

def pairwise_cov(d, waves, ycol="yc"):
    rows = []
    for t in waves:
        for s in waves:
            if s < t: continue
            yt = d.loc[d["year"]==t, ["pid", ycol]].rename(columns={ycol:"yt"})
            ys = d.loc[d["year"]==s, ["pid", ycol]].rename(columns={ycol:"ys"})
            both = yt.merge(ys, on="pid", how="inner")
            if len(both): rows.append((t, s, len(both), float((both["yt"]*both["ys"]).mean())))
    return pd.DataFrame(rows, columns=["t","s","N_ts","omega_hat"])

def diag_lag_moments(cov_df):
    diag = cov_df.loc[cov_df["t"]==cov_df["s"]]
    var = (diag["omega_hat"]*diag["N_ts"]).sum()/diag["N_ts"].sum()
    cov1 = cov_df.loc[cov_df["s"]==cov_df["t"]+2]
    cov2 = cov_df.loc[cov_df["s"]==cov_df["t"]+4]
    c1 = (cov1["omega_hat"]*cov1["N_ts"]).sum()/cov1["N_ts"].sum() if len(cov1) else np.nan
    c2 = (cov2["omega_hat"]*cov2["N_ts"]).sum()/cov2["N_ts"].sum() if len(cov2) else np.nan
    return dict(var=var, cov1=c1, cov2=c2)

def model_cov_FE(theta, t, s):
    se2, rho, ee2 = theta
    if abs(rho) >= 0.9999 or se2 < 0 or ee2 < 0: return np.nan
    k = abs(s-t)//2
    return se2 + (rho**k) * ee2 / (1 - rho**2)

def md_obj(theta, cov_df, w=None):
    res = np.array([cov_df.iloc[i]["omega_hat"] - model_cov_FE(theta, cov_df.iloc[i]["t"], cov_df.iloc[i]["s"])
                     for i in range(len(cov_df))])
    if not np.isfinite(res).all(): return 1e10
    return float(res @ np.diag(w) @ res) if w is not None else float(res @ res)

def fit_md_FE(cov_df):
    if len(cov_df) < 3: return None
    x0 = [0.3, 0.5, 0.3]; bounds = [(1e-6,5),(-0.95,0.95),(1e-6,5)]
    res1 = minimize(md_obj, x0, args=(cov_df,), method="L-BFGS-B", bounds=bounds)
    w = cov_df["N_ts"].values / cov_df["N_ts"].sum()
    res2 = minimize(md_obj, res1.x, args=(cov_df, w), method="L-BFGS-B", bounds=bounds)
    se2, rho, ee2 = res2.x
    var_total = se2 + ee2/(1-rho**2)
    return dict(sigma_eta2=se2, rho=rho, sigma_eps2=ee2,
                s_perm=se2/var_total, var_total=var_total,
                J=float(cov_df["N_ts"].sum() * md_obj(res2.x, cov_df, w)),
                df=len(cov_df)-3)

print("\n=== Minimum distance, seven waves ===")
df_g = add_demean_g(df)
cov_H = pairwise_cov(df_g[df_g["HP14"]==1], WAVES)
cov_L = pairwise_cov(df_g[df_g["HP14"]==0], WAVES)
m_H = diag_lag_moments(cov_H); m_L = diag_lag_moments(cov_L)
HM_H = (m_H["cov1"]-m_H["cov2"])/(m_H["var"]-m_H["cov1"])
HM_L = (m_L["cov1"]-m_L["cov2"])/(m_L["var"]-m_L["cov1"])
fH = fit_md_FE(cov_H); fL = fit_md_FE(cov_L)
print(f"  pairs: H={len(cov_H)}, L={len(cov_L)}; smallest H pair N={cov_H['N_ts'].min()}, smallest L pair N={cov_L['N_ts'].min()}")
print(f"  HM ρ_v: H={HM_H:+.4f}, L={HM_L:+.4f}, H-L={HM_H-HM_L:+.4f}")
print(f"  FE-MD: H ρ_v={fH['rho']:+.4f}, L ρ_v={fL['rho']:+.4f}, gap={fH['rho']-fL['rho']:+.4f}")
print(f"  s_perm: H={fH['s_perm']:.4f}, L={fL['s_perm']:.4f}, gap={fH['s_perm']-fL['s_perm']:+.4f}")
print(f"  J-stat: H={fH['J']:.1f} (df={fH['df']}), L={fL['J']:.1f} (df={fL['df']})")

# Write the estimates
m1_7w = pd.DataFrame([
    {"group":"H", **fH, "HM": HM_H},
    {"group":"L", **fL, "HM": HM_L},
])
m1_7w.to_csv(os.path.join(WS_OUT, "ws_method1_7wave.csv"), index=False)


# ==========================================================
# 2. Two-step Arellano-Bond GMM
# ==========================================================
df["wave_idx"] = df["year"].map({y:i+1 for i,y in enumerate(WAVES)})
y_wide  = df.pivot(index="pid", columns="wave_idx", values="y_tilde_w")
hp_wide = df.groupby("pid")["HP14"].first()

def cluster_score_sum(Z, resid, pid_arr):
    pids = np.unique(pid_arr); M = np.zeros((Z.shape[1], Z.shape[1]))
    for p in pids:
        m = pid_arr == p; s = Z[m].T @ resid[m]; M += np.outer(s, s)
    return M, len(pids)

def build_AB(y_wide, hp_wide, T_local, max_lag=4):
    rows = []
    for i in y_wide.index:
        yi = y_wide.loc[i].values
        hp = int(hp_wide.loc[i])
        for t in range(3, T_local+1):
            y_t,y_tm1,y_tm2 = yi[t-1],yi[t-2],yi[t-3]
            if any(np.isnan([y_t,y_tm1,y_tm2])): continue
            Dy = y_t - y_tm1
            Dy_lag = y_tm1 - y_tm2
            instr = []
            for lag in range(2, max_lag+1):
                lev = yi[t-1-lag] if (t-1-lag) >= 0 else np.nan
                instr.append(0.0 if np.isnan(lev) else float(lev))
            Z_row = instr + [v*hp for v in instr]
            rows.append({"pid":i, "Dy":Dy, "Dy_lag":Dy_lag, "Dy_lag_x_hp":Dy_lag*hp, "Z":Z_row})
    return dict(
        Dy=np.array([r["Dy"] for r in rows]),
        Dy_lag=np.array([r["Dy_lag"] for r in rows]),
        Dy_lag_x_hp=np.array([r["Dy_lag_x_hp"] for r in rows]),
        Z=np.array([r["Z"] for r in rows]),
        pid=np.array([r["pid"] for r in rows]),
    )

def ab_gmm_2step(d):
    X = np.column_stack([d["Dy_lag"], d["Dy_lag_x_hp"]])
    Z = d["Z"]; y = d["Dy"]
    ZX = Z.T @ X; Zy = Z.T @ y
    W = np.linalg.pinv(Z.T @ Z)
    XZWZX = ZX.T @ W @ ZX
    b1 = np.linalg.solve(XZWZX, ZX.T @ W @ Zy)
    r1 = y - X @ b1
    score, n_cl = cluster_score_sum(Z, r1, d["pid"])
    W2 = np.linalg.pinv(score)
    XZW2ZX = ZX.T @ W2 @ ZX
    b2 = np.linalg.solve(XZW2ZX, ZX.T @ W2 @ Zy)
    r2 = y - X @ b2
    score2, n_cl2 = cluster_score_sum(Z, r2, d["pid"])
    inv = np.linalg.pinv(XZW2ZX)
    Var = (inv @ ZX.T @ W2 @ score2 @ W2 @ ZX @ inv) * (n_cl2/(n_cl2-1))
    se = np.sqrt(np.diag(Var))
    g_sum = Z.T @ r2
    J = float(g_sum @ W2 @ g_sum)
    return dict(rho_0=float(b2[0]), rho_1=float(b2[1]),
                se_rho_0=float(se[0]), se_rho_1=float(se[1]),
                J=J, df_J=Z.shape[1]-X.shape[1], n_eq=len(y))

print("\n=== Arellano-Bond GMM, seven waves (two-step, max_lag=4) ===")
d_ab = build_AB(y_wide, hp_wide, T_local=T, max_lag=4)
print(f"  AB equations: N_eq = {len(d_ab['Dy']):,}, instruments = {d_ab['Z'].shape[1]}")
fit2 = ab_gmm_2step(d_ab)
print(f"  ρ_0 = {fit2['rho_0']:+.4f} (SE {fit2['se_rho_0']:.4f})")
print(f"  ρ_1 = {fit2['rho_1']:+.4f} (SE {fit2['se_rho_1']:.4f})")
print(f"  Hansen J = {fit2['J']:.3f} (df = {fit2['df_J']})")

# Bootstrap that resamples children, 500 replications
np.random.seed(20260501)
B = 500
unique_pids = np.unique(d_ab["pid"])
pid_to_rows = {p: np.where(d_ab["pid"]==p)[0] for p in unique_pids}
N_pid_AB = len(unique_pids)
boot = np.full((B, 2), np.nan)
t0 = time.time()
for b in range(B):
    pid_sample = np.random.choice(unique_pids, size=N_pid_AB, replace=True)
    parts = [pid_to_rows[p] for p in pid_sample]
    idx = np.concatenate(parts)
    rep_lengths = [len(pid_to_rows[p]) for p in pid_sample]
    new_pid = np.repeat(np.arange(N_pid_AB), rep_lengths)
    d_b = dict(Dy=d_ab["Dy"][idx], Dy_lag=d_ab["Dy_lag"][idx],
               Dy_lag_x_hp=d_ab["Dy_lag_x_hp"][idx], Z=d_ab["Z"][idx], pid=new_pid)
    try:
        r = ab_gmm_2step(d_b)
        boot[b] = [r["rho_0"], r["rho_1"]]
    except Exception:
        # The replication failed (for example, a singular weight matrix in
        # the resample); boot[b] stays missing.
        pass
boot_se_r0 = np.nanstd(boot[:,0], ddof=1)
boot_se_r1 = np.nanstd(boot[:,1], ddof=1)
ci_r0 = np.nanpercentile(boot[:,0], [2.5, 97.5])
ci_r1 = np.nanpercentile(boot[:,1], [2.5, 97.5])
p_r0 = 2*(1 - 0.5*(1+math.erf(abs(fit2['rho_0']/boot_se_r0)/math.sqrt(2))))
p_r1 = 2*(1 - 0.5*(1+math.erf(abs(fit2['rho_1']/boot_se_r1)/math.sqrt(2))))
print(f"\n  Bootstrap (B={B}, elapsed={time.time()-t0:.1f}s):")
print(f"    ρ_0: SE={boot_se_r0:.4f}, 95% CI=[{ci_r0[0]:+.4f}, {ci_r0[1]:+.4f}], p={p_r0:.4f}")
print(f"    ρ_1: SE={boot_se_r1:.4f}, 95% CI=[{ci_r1[0]:+.4f}, {ci_r1[1]:+.4f}], p={p_r1:.4f}")

m2_7w = pd.DataFrame([
    {"coef":"rho_0", "estimate":fit2['rho_0'], "se_analytic":fit2['se_rho_0'],
     "se_boot":boot_se_r0, "ci_lo":ci_r0[0], "ci_hi":ci_r0[1], "p_boot":p_r0},
    {"coef":"rho_1", "estimate":fit2['rho_1'], "se_analytic":fit2['se_rho_1'],
     "se_boot":boot_se_r1, "ci_lo":ci_r1[0], "ci_hi":ci_r1[1], "p_boot":p_r1},
])
m2_7w.to_csv(os.path.join(WS_OUT, "ws_method2_7wave.csv"), index=False)


# ==========================================================
# 3. Random-coefficient model and the within-child (FE) interaction model
# ==========================================================
def per_indiv(d, min_T=3):
    rows = []
    waves = sorted(d["year"].unique())
    d = d.copy(); d["t"] = d["year"].map({y:i+1 for i,y in enumerate(waves)})
    for i, sub in d.groupby("pid"):
        if len(sub) < min_T: continue
        t = sub["t"].values.astype(float)
        y = sub["y_tilde_w"].values.astype(float)
        if t.std() < 1e-9: continue
        Xi = np.column_stack([np.ones_like(t), t])
        try: ab = np.linalg.solve(Xi.T@Xi, Xi.T@y)
        except np.linalg.LinAlgError: continue
        e = y - Xi@ab
        s2 = float(e@e/max(len(t)-2,1))
        rows.append({"pid":i, "alpha":float(ab[0]), "beta_hat":float(ab[1]),
                     "var_indiv":s2*float(np.linalg.inv(Xi.T@Xi)[1,1]), "T_i":len(t)})
    return pd.DataFrame(rows)

print("\n=== Random-coefficient model, seven waves ===")
m3 = df.dropna(subset=["y_tilde_w","HP14","ParentInc_bar"]).copy()
bp = per_indiv(m3, min_T=3)
info = m3.groupby("pid").agg(HP=("HP14","first"), PInc=("ParentInc_bar","first")).reset_index()
bp = bp.merge(info, on="pid", how="left").dropna()
print(f"  Children with T_i ≥ 3: {len(bp):,}, T_i mean = {bp['T_i'].mean():.2f}")

var_naive = bp["beta_hat"].var(ddof=1)
var_corr = max(var_naive - bp["var_indiv"].mean(), 0)
X = sm.add_constant(bp[["HP","PInc"]].astype(float), has_constant="add")
res = sm.OLS(bp["beta_hat"].astype(float), X).fit(cov_type="HC3")
print(f"  Var(β_i) raw = {var_naive:.6f}; O(1/T)-corrected = {var_corr:.6f}")
print(f"  λ_1 (CRC HP14) = {res.params['HP']:+.4f} (SE {res.bse['HP']:.4f}, p={res.pvalues['HP']:.3f})")
print(f"  λ_2 (CRC PInc) = {res.params['PInc']:+.4f} (SE {res.bse['PInc']:.4f})")

# gamma: within-child (FE) model with the interaction of HP14 and the wave index
m3w = m3.copy()
m3w["t"] = m3w["year"].map({y:i+1 for i,y in enumerate(WAVES)})
m3w["HPt"] = m3w["HP14"]*m3w["t"]
my = m3w.groupby("pid")["y_tilde_w"].transform("mean")
mhpt = m3w.groupby("pid")["HPt"].transform("mean")
g_ols = sm.OLS(m3w["y_tilde_w"]-my, sm.add_constant(m3w["HPt"]-mhpt, has_constant="add")).fit(
    cov_type="cluster", cov_kwds={"groups":m3w["pid"].values})
print(f"  γ (FE-interaction) = {g_ols.params.iloc[1]:+.4f} (SE {g_ols.bse.iloc[1]:.4f}, p={g_ols.pvalues.iloc[1]:.3f})")

m3_7w = pd.DataFrame([
    {"obj":"Var(beta_i) raw", "val":var_naive},
    {"obj":"Var(beta_i) O(1/T)", "val":var_corr},
    {"obj":"lambda_1", "val":res.params["HP"], "se":res.bse["HP"], "p":res.pvalues["HP"]},
    {"obj":"lambda_2", "val":res.params["PInc"], "se":res.bse["PInc"], "p":res.pvalues["PInc"]},
    {"obj":"gamma_FE", "val":g_ols.params.iloc[1], "se":g_ols.bse.iloc[1], "p":g_ols.pvalues.iloc[1]},
])
m3_7w.to_csv(os.path.join(WS_OUT, "ws_method3_7wave.csv"), index=False)


# ==========================================================
# 4. Correlated random effects, outcome log income
# ==========================================================
print("\n=== Correlated random effects, seven waves (log income) ===")
keep = (df["log_income"].notna() & df["age"].notna() & df["female"].notna() &
        df["urban"].notna() & df["provcd"].notna() &
        df["HP14"].notna() & df["ParentInc_bar"].notna())
r = df.loc[keep].copy()
r["age2"] = r["age"]**2
for v in ["age","age2","urban"]:
    r[f"{v}_bar"] = r.groupby("pid")[v].transform("mean")
prov_d = pd.get_dummies(r["provcd"].astype(int), prefix="prov", drop_first=True, dtype=float)
year_d = pd.get_dummies(r["year"].astype(int),    prefix="yr",   drop_first=True, dtype=float)
X = pd.concat([r[["age","age2","urban","age_bar","age2_bar","urban_bar","female",
                  "HP14","ParentInc_bar"]].astype(float), prov_d, year_d], axis=1)
X = sm.add_constant(X, has_constant="add")
y = r["log_income"].astype(float)
fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups":r["pid"].values})
X2 = X.drop(columns=["ParentInc_bar"])
fit2_ = sm.OLS(y, X2).fit(cov_type="cluster", cov_kwds={"groups":r["pid"].values})
print(f"  N obs = {int(fit.nobs):,}, N children = {r['pid'].nunique():,}")
print(f"  δ_1 (with PInc) = {fit.params['HP14']:+.4f} (SE {fit.bse['HP14']:.4f}, p={fit.pvalues['HP14']:.3f})")
print(f"  δ_2 (PInc) = {fit.params['ParentInc_bar']:+.4f} (SE {fit.bse['ParentInc_bar']:.4f}, p={fit.pvalues['ParentInc_bar']:.3f})")
print(f"  δ_1 (no PInc) = {fit2_.params['HP14']:+.4f}; change from controlling for PInc = {(1-fit.params['HP14']/fit2_.params['HP14'])*100:+.1f}%")

cre_7w = pd.DataFrame([
    {"coef":"delta_1", "est":fit.params["HP14"], "se":fit.bse["HP14"], "p":fit.pvalues["HP14"]},
    {"coef":"delta_2", "est":fit.params["ParentInc_bar"], "se":fit.bse["ParentInc_bar"], "p":fit.pvalues["ParentInc_bar"]},
    {"coef":"delta_1_noinc", "est":fit2_.params["HP14"], "se":fit2_.bse["HP14"], "p":fit2_.pvalues["HP14"]},
])
cre_7w.to_csv(os.path.join(WS_OUT, "ws_cre_7wave.csv"), index=False)
print(f"\nsaved CSVs to {WS_OUT}/")
