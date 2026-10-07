import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Random-coefficient model of residual log income: child-specific intercept and
child-specific linear trend, related to family background by a correlated
random coefficients regression, with a split-panel jackknife.

Sample: child-year rows of panel_resid with a non-missing winsorized residual
y_tilde_w and a non-missing HighParentOcc14 (children aged 22 to 55, waves 2014
to 2022, at least three waves with positive income).

Model:
   y_tilde_w_it = alpha_i + beta_i * t + v_it
where t is the wave index (1 for the first wave of the panel, 2 for the second,
and so on).

Estimation:
   1. OLS by child of y_tilde_w on a constant and t, for each child with at
      least three valid waves, which gives (alpha_i, beta_i). There are no
      other regressors, because y_tilde_w is already a residual.
   2. Variance of beta_i across children, corrected by subtracting the average
      sampling variance of the child-level slope estimates.
   3. Regression of the estimated beta_i on HighParentOcc14_i and
      ParentInc_bar_i, with heteroskedasticity-consistent (HC3) standard
      errors. lambda_1 is the coefficient on HighParentOcc14.
   4. Split-panel jackknife (Dhaene and Jochmans 2015) of the variance of
      beta_i and of lambda_1, with two splits: halves that share the 2018 wave
      (2014, 2016, 2018 and 2018, 2020, 2022) and halves that do not overlap
      (2014, 2016 and 2020, 2022).

Comparison:
   gamma from the model with child-specific intercepts and an interaction,
   y_tilde_w_it = alpha_i + gamma * (HighParentOcc14_i * t) + e_it,
   estimated after demeaning within child, is reported next to lambda_1.

Output:
  data/intermediate/method3_mini_results.csv
  data/intermediate/method3_mini_betahat_panel.csv
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
df["HighParentOcc14"]  = df["HighParentOcc14"].astype(int)
df["ParentInc_bar"]  = df["ParentInc_bar"].astype(float)

WAVES = sorted(df["year"].unique().tolist())
T = len(WAVES)
df["t"] = df["year"].map({y: i+1 for i, y in enumerate(WAVES)})  # wave index 1 to T

print(f"loaded panel_resid: {len(df):,} rows, {df['pid'].nunique():,} children")
print(f"WAVES = {WAVES}, T = {T}")


# ---------------------------------------------------------------------------
# OLS by child of y_tilde_w on a constant and t
# ---------------------------------------------------------------------------
def per_individual_betahat(d, min_T=3):
    """Return a DataFrame (pid, T_i, alpha_hat, beta_hat, sigma_v_i_2,
    var_beta_indiv) with one row for each child that has at least min_T valid
    waves and variation in t."""
    rows = []
    for i, sub in d.groupby("pid"):
        if len(sub) < min_T:
            continue
        t = sub["t"].values.astype(float)
        y = sub["y_tilde_w"].values.astype(float)
        if t.std() < 1e-9:
            continue
        # OLS of y on a constant and t
        Xi = np.column_stack([np.ones_like(t), t])
        XtX = Xi.T @ Xi
        Xty = Xi.T @ y
        try:
            ab = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            continue
        e = y - Xi @ ab
        sigma2 = float(e @ e / max(len(t) - 2, 1))
        var_beta_indiv = sigma2 * float(np.linalg.inv(XtX)[1, 1])
        rows.append({
            "pid": i,
            "T_i": len(t),
            "alpha_hat": float(ab[0]),
            "beta_hat": float(ab[1]),
            "sigma_v_i_2": sigma2,
            "var_beta_indiv": var_beta_indiv,
        })
    return pd.DataFrame(rows)


# HighParentOcc14 and ParentInc_bar at the child level
hp_inc = (df.groupby("pid")
            .agg(HighParentOcc14=("HighParentOcc14","first"),
                 ParentInc_bar=("ParentInc_bar","first"))
            .reset_index())

print("\n=== OLS by child: (alpha_hat, beta_hat) ===")
beta_panel_full = per_individual_betahat(df, min_T=3)
beta_panel_full = beta_panel_full.merge(hp_inc, on="pid", how="left")
print(f"  Children with T_i>=3 and t-variation: {len(beta_panel_full):,}")
print(f"  beta_hat distribution: mean={beta_panel_full['beta_hat'].mean():.4f}, "
      f"sd={beta_panel_full['beta_hat'].std():.4f}, T_i mean={beta_panel_full['T_i'].mean():.2f}")

# ---------------------------------------------------------------------------
# Variance of beta_i across children, corrected for the sampling variance of
# the child-level estimates
# ---------------------------------------------------------------------------
def var_beta_corrected(panel):
    """Return the variance of beta_hat across children, the variance net of the average sampling variance (not below zero), and the average sampling variance."""
    var_naive = panel["beta_hat"].var(ddof=1)
    avg_within = panel["var_beta_indiv"].mean()
    var_corrected = var_naive - avg_within
    return var_naive, max(var_corrected, 0.0), avg_within


def crc_regression(panel):
    """Regress beta_hat on HighParentOcc14 and ParentInc_bar, with HC3 standard errors."""
    sub = panel.dropna(subset=["beta_hat","HighParentOcc14","ParentInc_bar"]).copy()
    X = sm.add_constant(sub[["HighParentOcc14","ParentInc_bar"]].astype(float),
                        has_constant="add")
    y = sub["beta_hat"].astype(float)
    res = sm.OLS(y, X).fit(cov_type="HC3")
    return {
        "lambda_0": float(res.params["const"]),
        "lambda_1": float(res.params["HighParentOcc14"]),
        "lambda_2": float(res.params["ParentInc_bar"]),
        "se_lambda_1": float(res.bse["HighParentOcc14"]),
        "se_lambda_2": float(res.bse["ParentInc_bar"]),
        "wald_joint_p": float(res.f_pvalue),
        "n": int(res.nobs),
    }


# All waves
var_naive, var_corrected, avg_within = var_beta_corrected(beta_panel_full)
print(f"\n=== Var(beta_i) and its correction ===")
print(f"  raw Var(beta_hat):         {var_naive:.6f}")
print(f"  avg within-i sampling var: {avg_within:.6f}")
print(f"  O(1/T)-corrected Var:      {var_corrected:.6f}")

crc_full = crc_regression(beta_panel_full)
print(f"\n=== CRC regression on full panel ===")
print(f"  lambda_1 (HighParentOcc14):  {crc_full['lambda_1']:+.4f}  (SE {crc_full['se_lambda_1']:.4f})")
print(f"  lambda_2 (ParentInc_bar):  {crc_full['lambda_2']:+.4f}  (SE {crc_full['se_lambda_2']:.4f})")
print(f"  N: {crc_full['n']:,}")

# By group
print(f"\n=== beta_hat distribution by HighParentOcc14 ===")
for g in [0, 1]:
    sub = beta_panel_full.loc[beta_panel_full["HighParentOcc14"]==g, "beta_hat"]
    print(f"  HighParentOcc14={g}: N={len(sub):,}, mean={sub.mean():+.4f}, sd={sub.std():.4f}")

# ---------------------------------------------------------------------------
# Split-panel jackknife (Dhaene and Jochmans 2015)
# ---------------------------------------------------------------------------
def fit_half_panel(d, half_waves, min_T_half=2):
    sub = d.loc[d["year"].isin(half_waves)].copy()
    panel = per_individual_betahat(sub, min_T=min_T_half)
    panel = panel.merge(hp_inc, on="pid", how="left")
    var_naive, var_corr, _ = var_beta_corrected(panel)
    crc = crc_regression(panel)
    return {
        "n_panel": len(panel),
        "var_naive": var_naive,
        "var_corrected": var_corr,
        "lambda_1": crc["lambda_1"],
        "lambda_2": crc["lambda_2"],
    }


print(f"\n=== Split-panel jackknife (Dhaene-Jochmans 2015) ===")
# Five-wave panel (2014 to 2022).
# Halves that share the 2018 wave: three waves each
H1_overlap = [2014,2016,2018]
H2_overlap = [2018,2020,2022]
fit1_o = fit_half_panel(df, H1_overlap, min_T_half=2)
fit2_o = fit_half_panel(df, H2_overlap, min_T_half=2)
# Halves that do not overlap: two waves each, the smallest number that identifies a slope
H1_nonov = [2014,2016]
H2_nonov = [2020,2022]
fit1_n = fit_half_panel(df, H1_nonov, min_T_half=2)
fit2_n = fit_half_panel(df, H2_nonov, min_T_half=2)

# Jackknife estimate: 2 * theta_full - (theta_1 + theta_2) / 2
def spj(theta_full, theta1, theta2):
    return 2*theta_full - 0.5*(theta1 + theta2)

theta_full_var = var_corrected
theta_full_lam1 = crc_full["lambda_1"]

print(f"\n  Overlapping halves (three waves each, both contain 2018):")
print(f"    Half 1 ({H1_overlap}): N={fit1_o['n_panel']:,}, Var_corr={fit1_o['var_corrected']:.6f}, "
      f"lambda_1={fit1_o['lambda_1']:+.4f}")
print(f"    Half 2 ({H2_overlap}): N={fit2_o['n_panel']:,}, Var_corr={fit2_o['var_corrected']:.6f}, "
      f"lambda_1={fit2_o['lambda_1']:+.4f}")

spj_var_o  = spj(theta_full_var,  fit1_o["var_corrected"], fit2_o["var_corrected"])
spj_lam1_o = spj(theta_full_lam1, fit1_o["lambda_1"],     fit2_o["lambda_1"])

print(f"\n  Non-overlapping halves (two waves each, 2018 not used):")
print(f"    Half 1 ({H1_nonov}):    N={fit1_n['n_panel']:,}, Var_corr={fit1_n['var_corrected']:.6f}, "
      f"lambda_1={fit1_n['lambda_1']:+.4f}")
print(f"    Half 2 ({H2_nonov}):    N={fit2_n['n_panel']:,}, Var_corr={fit2_n['var_corrected']:.6f}, "
      f"lambda_1={fit2_n['lambda_1']:+.4f}")

spj_var_n  = spj(theta_full_var,  fit1_n["var_corrected"], fit2_n["var_corrected"])
spj_lam1_n = spj(theta_full_lam1, fit1_n["lambda_1"],     fit2_n["lambda_1"])

print(f"\n=== SPJ-corrected estimates ===")
print(f"  Var(beta_i):")
print(f"    raw                    {var_naive:.6f}")
print(f"    O(1/T)-corrected       {var_corrected:.6f}")
print(f"    SPJ overlapping        {spj_var_o:.6f}")
print(f"    SPJ non-overlapping    {spj_var_n:.6f}")
print(f"  CRC lambda_1 (HighParentOcc14):")
print(f"    raw                    {crc_full['lambda_1']:+.6f}")
print(f"    SPJ overlapping        {spj_lam1_o:+.6f}")
print(f"    SPJ non-overlapping    {spj_lam1_n:+.6f}")

# ---------------------------------------------------------------------------
# Comparison: within-child (FE) model with the interaction of HighParentOcc14 and t
# ---------------------------------------------------------------------------
print(f"\n=== Comparison: gamma from the within-child (FE) interaction model ===")
# y_tilde_w_it = alpha_i + gamma * (HighParentOcc14_i * t) + e_it
# estimated after demeaning within child
df["HPt"] = df["HighParentOcc14"] * df["t"]
mean_y    = df.groupby("pid")["y_tilde_w"].transform("mean")
mean_HPt  = df.groupby("pid")["HPt"].transform("mean")
df["yd"]   = df["y_tilde_w"] - mean_y
df["HPtd"] = df["HPt"]    - mean_HPt
ols_gamma = sm.OLS(df["yd"], sm.add_constant(df["HPtd"], has_constant="add")).fit(
    cov_type="cluster", cov_kwds={"groups": df["pid"].values}
)
gamma_hat = float(ols_gamma.params["HPtd"])
print(f"  gamma_hat = {gamma_hat:+.5f}  (SE {ols_gamma.bse['HPtd']:.5f}, N obs={int(ols_gamma.nobs):,})")
print(f"  CRC lambda_1 (raw) = {crc_full['lambda_1']:+.5f}")
print(f"  Sign agreement: {'YES' if np.sign(gamma_hat) == np.sign(crc_full['lambda_1']) else 'NO'}")

# ---------------------------------------------------------------------------
# Write the estimates
# ---------------------------------------------------------------------------
results = pd.DataFrame([
    {"object":"Var(beta_i)","variant":"raw",                "estimate":var_naive,    "n":len(beta_panel_full)},
    {"object":"Var(beta_i)","variant":"O(1/T)-corrected",   "estimate":var_corrected,"n":len(beta_panel_full)},
    {"object":"Var(beta_i)","variant":"SPJ overlapping",    "estimate":spj_var_o,    "n":len(beta_panel_full)},
    {"object":"Var(beta_i)","variant":"SPJ non-overlapping","estimate":spj_var_n,    "n":len(beta_panel_full)},
    {"object":"lambda_1",   "variant":"raw",                "estimate":crc_full["lambda_1"], "n":crc_full["n"]},
    {"object":"lambda_1",   "variant":"SPJ overlapping",    "estimate":spj_lam1_o,    "n":crc_full["n"]},
    {"object":"lambda_1",   "variant":"SPJ non-overlapping","estimate":spj_lam1_n,    "n":crc_full["n"]},
    {"object":"lambda_2",   "variant":"raw",                "estimate":crc_full["lambda_2"], "n":crc_full["n"]},
    {"object":"gamma_FE",   "variant":"FE-interaction",     "estimate":gamma_hat,     "n":int(ols_gamma.nobs)},
])
out = os.path.join(BASE_INT, "method3_mini_results.csv")
results.to_csv(out, index=False)
print(f"\nsaved {out}")

beta_panel_full.to_csv(os.path.join(BASE_INT, "method3_mini_betahat_panel.csv"), index=False)
print(f"saved method3_mini_betahat_panel.csv (per-child beta_hat)")
