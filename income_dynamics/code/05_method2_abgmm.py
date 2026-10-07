import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Dynamic panel model of residual log income, estimated by Arellano-Bond GMM.

Sample: child-year rows of panel_resid with a non-missing winsorized residual
y_tilde_w and a non-missing HighParentOcc14 (children aged 22 to 55, waves 2014
to 2022, at least three waves with positive income).

Contents:
  - Within-groups estimate of the AR(1) coefficient of y_tilde_w, which is
    subject to the Nickell bias in a short panel.
  - Arellano-Bond GMM in first differences of
       y_tilde_w_it = rho_0 * y_tilde_w_{i,t-1}
                    + rho_1 * (y_tilde_w_{i,t-1} * HighParentOcc14_i)
                    + alpha_i + u_it
    Instruments: the levels of y_tilde_w lagged two and three waves, collapsed
    (one column per lag, as in Roodman 2009), and their interactions with
    HighParentOcc14. One-step and two-step estimates with standard errors
    clustered by child.
  - Hansen J test of the overidentifying restrictions and a test of
    second-order serial correlation in the differenced residuals.
  - F statistic of the first stage for the interaction term, computed after
    partialling out the instruments that are not interacted.
  - Number of children with at least three valid waves.
  - Comparison of rho_1 with the value implied by the minimum-distance
    estimates in method1_results.csv:
       Delta_share = (1 - rho_v_pooled) * (s_H - s_L),
    where s_H and s_L are the shares of the permanent effect in the variance
    of the residual for advantaged and less advantaged children.

Output:
  data/intermediate/method2_results.csv
  and a summary printed to the console

The GMM estimator is coded here directly. The Stata version of the same
specification is in stata/05_method2_abgmm.do.
"""
import os, json
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm

BASE_INT = _WSROOT + "/data/intermediate"

_pkl = os.path.join(BASE_INT, "panel_resid.pkl")
_dta = os.path.join(BASE_INT, "panel_resid.dta")
if os.path.exists(_pkl) and os.path.getsize(_pkl) > 1000:
    df = pd.read_pickle(_pkl)
else:
    df, _ = pyreadstat.read_dta(_dta)
df = df.dropna(subset=["y_tilde_w","HighParentOcc14"]).copy()
df["pid"]  = df["pid"].astype(np.int64)
df["year"] = df["year"].astype(int)
df["HighParentOcc14"] = df["HighParentOcc14"].astype(int)

WAVES = sorted(df["year"].unique().tolist())
T = len(WAVES)
print(f"loaded panel_resid: {len(df):,} rows, {df['pid'].nunique():,} children, T={T} waves")

# ---------------------------------------------------------------------------
# Wide format: one row per child, one column per wave, for y_tilde_w; and the
# child-level HighParentOcc14
# ---------------------------------------------------------------------------
df["wave_idx"] = df["year"].map({y: i+1 for i, y in enumerate(WAVES)})
y_wide  = df.pivot(index="pid", columns="wave_idx", values="y_tilde_w")
hp_wide = df.groupby("pid")["HighParentOcc14"].first()

# ---------------------------------------------------------------------------
# 1. Within-groups estimate of the AR(1) coefficient
# ---------------------------------------------------------------------------
# Demean within child, then regress y_it on y_{i,t-1}
df_sorted = df.sort_values(["pid","year"]).reset_index(drop=True)
df_sorted["y_lag"] = df_sorted.groupby("pid")["y_tilde_w"].shift(1)
within = df_sorted.dropna(subset=["y_lag"]).copy()
mean_y    = within.groupby("pid")["y_tilde_w"].transform("mean")
mean_ylag = within.groupby("pid")["y_lag"].transform("mean")
within["yd"]    = within["y_tilde_w"] - mean_y
within["ylag_d"] = within["y_lag"]  - mean_ylag
ols = sm.OLS(within["yd"], sm.add_constant(within["ylag_d"], has_constant="add")).fit(
    cov_type="cluster", cov_kwds={"groups": within["pid"].values}
)
rho_WG = float(ols.params["ylag_d"])
print(f"\n[Within-Groups]   rho_WG = {rho_WG:.4f}  "
      f"(SE {ols.bse['ylag_d']:.4f}, N obs={int(ols.nobs):,})")
print(f"  Nickell approximation of the bias for true rho=0.5, T={T}:  "
      f"-(1+0.5)/(T-1) = {-1.5/(T-1):.4f}")

# ---------------------------------------------------------------------------
# 2. Arellano-Bond GMM with the interaction of the lag and HighParentOcc14
# ---------------------------------------------------------------------------
def build_AB_data(y_wide, hp_wide, max_lag=3):
    """Build the first-differenced equations with collapsed lagged-level instruments.

    Returns a dict with the arrays used by the GMM functions:
       Dy          : (N_eq,) first difference of y at wave t
       Dy_lag      : (N_eq,) first difference of y at wave t-1
       Dy_lag_x_hp : (N_eq,) Dy_lag times HighParentOcc14_i
       Z           : (N_eq, m) instrument matrix, one row per equation
       pid         : (N_eq,) child identifier, used for clustering
    """
    rows = []
    pids = list(y_wide.index)
    for i in pids:
        yi = y_wide.loc[i].values   # length T, missing where the wave is not observed
        hp = int(hp_wide.loc[i])
        # Equations for waves 3 to T; waves t, t-1 and t-2 must all be observed
        for t in range(3, T+1):
            y_t   = yi[t-1]
            y_tm1 = yi[t-2]
            y_tm2 = yi[t-3] if t >= 3 else np.nan
            if np.isnan(y_t) or np.isnan(y_tm1) or np.isnan(y_tm2):
                continue
            Dy     = y_t   - y_tm1
            Dy_lag = y_tm1 - y_tm2
            # Instruments: levels lagged 2 to max_lag waves, one column per lag;
            # a lag that is not observed enters as zero
            instr = []
            for lag in range(2, max_lag+1):
                yi_lev = yi[t-1-lag] if (t-1-lag) >= 0 else np.nan
                instr.append(0.0 if np.isnan(yi_lev) else float(yi_lev))
            # Each instrument interacted with HighParentOcc14
            instr_hp = [v*hp for v in instr]
            rows.append({
                "pid": i,
                "Dy": Dy,
                "Dy_lag": Dy_lag,
                "Dy_lag_x_hp": Dy_lag * hp,
                "Z": instr + instr_hp,
            })
    if not rows:
        return None
    Dy           = np.array([r["Dy"] for r in rows])
    Dy_lag       = np.array([r["Dy_lag"] for r in rows])
    Dy_lag_x_hp  = np.array([r["Dy_lag_x_hp"] for r in rows])
    Z            = np.array([r["Z"] for r in rows])
    pids_arr     = np.array([r["pid"] for r in rows])
    return dict(Dy=Dy, Dy_lag=Dy_lag, Dy_lag_x_hp=Dy_lag_x_hp, Z=Z, pid=pids_arr)


def _cluster_score_sum(Z, resid, pid_arr):
    """Return the sum over children of (Z_i' u_i)(u_i' Z_i), and the number of children."""
    pids_unique = np.unique(pid_arr)
    score_outer = np.zeros((Z.shape[1], Z.shape[1]))
    for p in pids_unique:
        m = pid_arr == p
        s = Z[m].T @ resid[m]
        score_outer += np.outer(s, s)
    return score_outer, len(pids_unique)


def ab_gmm_one_step(d):
    """One-step Arellano-Bond GMM with weight matrix (Z'Z)^-1.

    Standard errors clustered by child:
        V = (X'ZWZ'X)^-1 X'ZW [sum_i Z_i'u_i u_i'Z_i] WZ'X (X'ZWZ'X)^-1
    multiplied by G/(G-1), where G is the number of children.
    """
    X = np.column_stack([d["Dy_lag"], d["Dy_lag_x_hp"]])
    Z = d["Z"]
    y = d["Dy"]
    ZX = Z.T @ X
    Zy = Z.T @ y
    W  = np.linalg.pinv(Z.T @ Z)
    XZWZX = ZX.T @ W @ ZX
    XZWZy = ZX.T @ W @ Zy
    beta = np.linalg.solve(XZWZX, XZWZy)
    resid = y - X @ beta
    score_sum, n_clusters = _cluster_score_sum(Z, resid, d["pid"])
    XZWZinv = np.linalg.pinv(XZWZX)
    # score_sum is a sum over children, not an average, so it is not divided by G
    Var = XZWZinv @ ZX.T @ W @ score_sum @ W @ ZX @ XZWZinv * (n_clusters / (n_clusters - 1))
    se = np.sqrt(np.diag(Var))
    return dict(beta=beta, se=se, resid=resid, X=X, Z=Z, y=y, n_clusters=n_clusters,
                W=W, score_sum=score_sum, ZX=ZX)


def ab_gmm_two_step(d, first):
    """Two-step Arellano-Bond GMM; the weight matrix is the inverse of the sum over children of the outer products of the one-step scores."""
    X = first["X"]
    Z = first["Z"]
    y = first["y"]
    # Weight matrix: inverse of the sum of outer products of the scores
    W2 = np.linalg.pinv(first["score_sum"])
    ZX = Z.T @ X
    Zy = Z.T @ y
    XZWZX = ZX.T @ W2 @ ZX
    XZWZy = ZX.T @ W2 @ Zy
    beta = np.linalg.solve(XZWZX, XZWZy)
    resid = y - X @ beta
    # Scores at the two-step estimate, then the sandwich formula
    score_sum_new, n_clusters = _cluster_score_sum(Z, resid, d["pid"])
    XZWZinv = np.linalg.pinv(XZWZX)
    Var = XZWZinv @ ZX.T @ W2 @ score_sum_new @ W2 @ ZX @ XZWZinv * (n_clusters / (n_clusters - 1))
    se = np.sqrt(np.diag(Var))
    # Hansen J: g = sum_i Z_i'u_i, J = g' W2 g
    g_sum = Z.T @ resid
    J = float(g_sum @ W2 @ g_sum)
    df_J = Z.shape[1] - X.shape[1]
    return dict(beta=beta, se=se, resid=resid, J=J, df_J=df_J,
                n_clusters=n_clusters, n_obs=len(y))


print("\n=== Building AB-GMM equations (collapsed lagged-level instruments, max_lag=3) ===")
d = build_AB_data(y_wide, hp_wide, max_lag=3)
print(f"  AB equations: N_eq = {len(d['Dy']):,}, instrument cols = {d['Z'].shape[1]}, "
      f"unique children = {len(np.unique(d['pid'])):,}")

eff_N = (df.groupby("pid").size() >= 3).sum()
print(f"  Effective N for AB-GMM (children with >=3 valid waves): {eff_N:,}")

print("\n=== AB-GMM: one-step ===")
fit1 = ab_gmm_one_step(d)
print(f"  rho_0 (lag y_tilde_w):        {fit1['beta'][0]:+.4f}  (SE {fit1['se'][0]:.4f})")
print(f"  rho_1 (lag y_tilde_w x HP):   {fit1['beta'][1]:+.4f}  (SE {fit1['se'][1]:.4f})")

print("\n=== AB-GMM: two-step (standard errors clustered by child) ===")
fit2 = ab_gmm_two_step(d, fit1)
print(f"  rho_0:                      {fit2['beta'][0]:+.4f}  (SE {fit2['se'][0]:.4f})")
print(f"  rho_1:                      {fit2['beta'][1]:+.4f}  (SE {fit2['se'][1]:.4f})")
print(f"  Hansen J:                   {fit2['J']:.4f}  (df = {fit2['df_J']})")

# ---------------------------------------------------------------------------
# Test of second-order serial correlation in the differenced residuals
# ---------------------------------------------------------------------------
# The statistic is the mean of the product of the differenced residual and its
# second lag, divided by the standard error of that mean
resid = fit2["resid"]
# The equations are listed again in the order of build_AB_data, to attach the
# child identifier and the wave to each residual
ab_rows = []
ridx = 0
pids_list = list(y_wide.index)
for i in pids_list:
    yi = y_wide.loc[i].values
    for t in range(3, T+1):
        y_t   = yi[t-1]
        y_tm1 = yi[t-2]
        y_tm2 = yi[t-3]
        if np.isnan(y_t) or np.isnan(y_tm1) or np.isnan(y_tm2):
            continue
        ab_rows.append({"pid": i, "t": t, "Du_resid": resid[ridx]})
        ridx += 1
ab_df = pd.DataFrame(ab_rows)
ab_df = ab_df.sort_values(["pid","t"]).reset_index(drop=True)
ab_df["Du_lag2"] = ab_df.groupby("pid")["Du_resid"].shift(2)
sub = ab_df.dropna(subset=["Du_lag2"])
if len(sub) > 0:
    cov2 = (sub["Du_resid"] * sub["Du_lag2"]).mean()
    se2  = (sub["Du_resid"] * sub["Du_lag2"]).std() / np.sqrt(len(sub))
    z2   = cov2 / se2 if se2 > 0 else np.nan
    print(f"\n[AR(2) test]  Cov(Δu_t, Δu_{{t-2}}) = {cov2:+.5f}  z = {z2:+.3f}  (N pairs={len(sub):,})")
    print(f"  First-order correlation of the differenced residuals holds by construction; second-order correlation would invalidate the moment conditions.")

# ---------------------------------------------------------------------------
# First-stage F statistic for the interaction term
# ---------------------------------------------------------------------------
# Regress Dy_lag_x_hp on the instruments and test the instruments that are
# interacted with HighParentOcc14
Z = d["Z"]
n_inst = Z.shape[1]
m = n_inst // 2  # number of instruments that are not interacted
X_endog = d["Dy_lag_x_hp"]
# Partial out the non-interacted instruments from the interaction term and from the interacted instruments
Z_main = Z[:, :m]
Z_inter = Z[:, m:]
# Residual of the interaction term on the non-interacted instruments
b1, _, _, _ = np.linalg.lstsq(Z_main, X_endog, rcond=None)
X_resid = X_endog - Z_main @ b1
# Residual of the interacted instruments on the non-interacted instruments
B2, _, _, _ = np.linalg.lstsq(Z_main, Z_inter, rcond=None)
Z_inter_resid = Z_inter - Z_main @ B2
# OLS of the first residual on the second; its F statistic is reported
ols_inter = sm.OLS(X_resid, Z_inter_resid).fit()
print(f"\n[First-stage F on interaction instruments, non-interacted instruments partialled out]")
print(f"  R^2 = {ols_inter.rsquared:.4f}, F = {ols_inter.fvalue:.2f}, "
      f"df_resid = {int(ols_inter.df_resid)}, df_model = {int(ols_inter.df_model)}")
if ols_inter.fvalue < 10:
    print("  WARNING: F < 10; the interaction instruments are weak.")

# ---------------------------------------------------------------------------
# Value of rho_1 implied by the minimum-distance estimates:
# Delta_share = (1 - rho_v) * (s_H - s_L)
# ---------------------------------------------------------------------------
print("\n=== rho_1 implied by the minimum-distance variance shares ===")
m1 = pd.read_csv(os.path.join(BASE_INT, "method1_results.csv"))
m1_FE = m1[m1["model"]=="FE"].set_index("group")
rho_v_pooled = float(m1_FE.loc["pooled","rho"])
s_H = float(m1_FE.loc["H","s_perm"])
s_L = float(m1_FE.loc["L","s_perm"])
Delta_share = (1 - rho_v_pooled) * (s_H - s_L)
rho_1_hat = float(fit2["beta"][1])
excess = rho_1_hat - Delta_share
print(f"  Minimum-distance inputs:  rho_v_pooled = {rho_v_pooled:.4f}, s_H = {s_H:.4f}, s_L = {s_L:.4f}")
print(f"  Implied by the shares:    Delta_share = (1-rho_v)*(s_H-s_L) = {Delta_share:+.4f}")
print(f"  Arellano-Bond estimate:   rho_1_hat = {rho_1_hat:+.4f}")
print(f"  Difference:               rho_1_hat - Delta_share = {excess:+.4f}")
print(f"  Reading: {'rho_1 differs from the value implied by the shares by more than half of that value' if abs(excess) > abs(Delta_share)*0.5 else 'rho_1 is within half of the value implied by the shares'}")

# ---------------------------------------------------------------------------
# Write the estimates
# ---------------------------------------------------------------------------
results = pd.DataFrame([
    {"estimator":"WG", "coef":"rho","est":rho_WG, "se":float(ols.bse['ylag_d']), "n_obs":int(ols.nobs)},
    {"estimator":"AB1","coef":"rho_0","est":fit1["beta"][0],"se":fit1["se"][0],"n_obs":int(fit2["n_obs"])},
    {"estimator":"AB1","coef":"rho_1","est":fit1["beta"][1],"se":fit1["se"][1],"n_obs":int(fit2["n_obs"])},
    {"estimator":"AB2","coef":"rho_0","est":fit2["beta"][0],"se":fit2["se"][0],"n_obs":int(fit2["n_obs"])},
    {"estimator":"AB2","coef":"rho_1","est":fit2["beta"][1],"se":fit2["se"][1],"n_obs":int(fit2["n_obs"])},
    {"estimator":"AB2","coef":"Hansen_J","est":fit2["J"],"se":np.nan,"n_obs":int(fit2["df_J"])},
    {"estimator":"benchmark","coef":"Delta_share","est":Delta_share,"se":np.nan,"n_obs":eff_N},
    {"estimator":"benchmark","coef":"excess","est":excess,"se":np.nan,"n_obs":eff_N},
])
out = os.path.join(BASE_INT, "method2_results.csv")
results.to_csv(out, index=False)
print(f"\nsaved {out}")
