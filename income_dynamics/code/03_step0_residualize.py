import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Residual log income: remove age, gender, residence, province and year effects
from the log income of the child sample in panel_analysis (children aged 22 to
45, waves 2014 to 2022, at least three waves with positive income).

  Pooled specification (residual y_tilde):
    log_income_it = age polynomial of order 4
                  + female + female x age polynomial
                  + urban
                  + province indicators
                  + year indicators
                  + e_it
    y_tilde_it is the OLS residual.

  Group-specific age profile (residual y_tilde_grouphet):
    the pooled specification plus the interactions of the age polynomial with
    HighParentOcc. A joint Wald test of the four interaction coefficients is
    printed.

  y_tilde_w is y_tilde winsorized at the 1st and 99th percentiles of each wave.
  It is the outcome used by the minimum-distance, Arellano-Bond and
  random-coefficient scripts.

  The minimum-distance script additionally demeans the residual within group
  and year before computing autocovariances; that step is not done here.

Output: panel_resid.pkl and panel_resid.dta with columns
  pid, year, income, log_income,
  y_tilde, y_tilde_w, y_tilde_grouphet,
  age, female, urban, provcd,
  pid_father, pid_mother, occ_father, occ_mother, fd_father, fd_mother,
  HighParentOcc, HighParentOcc14, has_qv14,
  ParentInc_bar, eduy
"""
import os
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm

BASE_INT = _WSROOT + "/data/intermediate"

_pkl = os.path.join(BASE_INT, "panel_analysis.pkl")
_dta = os.path.join(BASE_INT, "panel_analysis.dta")
if os.path.exists(_pkl) and os.path.getsize(_pkl) > 1000:
    df = pd.read_pickle(_pkl)
    print(f"loaded panel_analysis from pkl: {len(df):,} rows, {df['pid'].nunique():,} unique pids")
else:
    df, _ = pyreadstat.read_dta(_dta)
    print(f"loaded panel_analysis from dta: {len(df):,} rows, {df['pid'].nunique():,} unique pids")

# Variable types
df["age"]    = df["age"].astype(float)
df["female"] = df["female"].astype(float)
df["urban"]  = df["urban"].astype(float)
df["provcd"] = df["provcd"].astype(float)
df["year"]   = df["year"].astype(int)
df["HighParentOcc"] = df["HighParentOcc"].astype(float)

# Rows with a missing regressor are left out of the regression; their residual is missing
keep = (df["age"].notna() & df["female"].notna() & df["urban"].notna() &
        df["provcd"].notna() & df["HighParentOcc"].notna())
print(f"Step-0 regression eligible rows: {keep.sum():,} / {len(df):,}")
reg = df.loc[keep].copy()

# Regressors: powers of age, their interactions with female and with HighParentOcc
for k in range(1, 5):
    reg[f"age{k}"] = reg["age"]**k
    reg[f"f_age{k}"] = reg["female"] * reg["age"]**k
    reg[f"hp_age{k}"] = reg["HighParentOcc"] * reg["age"]**k  # used by the group-specific specification

prov_dum = pd.get_dummies(reg["provcd"].astype(int), prefix="prov",
                          drop_first=True, dtype=float)
year_dum = pd.get_dummies(reg["year"].astype(int),   prefix="yr",
                          drop_first=True, dtype=float)


def fit_step0(extra_cols=None):
    """OLS of log income on the pooled regressors, optionally with extra columns; standard errors clustered by child."""
    base_cols = ["age1","age2","age3","age4","female",
                 "f_age1","f_age2","f_age3","f_age4","urban"]
    if extra_cols is not None:
        cols = base_cols + list(extra_cols)
    else:
        cols = base_cols
    X = pd.concat([reg[cols], prov_dum, year_dum], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = reg["log_income"].astype(float)
    return sm.OLS(y, X, hasconst=True).fit(
        cov_type="cluster", cov_kwds={"groups": reg["pid"].values}
    )


# ----------------------------------------------------------------------
# (A) Pooled specification
# ----------------------------------------------------------------------
mh = fit_step0(extra_cols=None)
print(f"\n[pooled]   N={int(mh.nobs):,}  R^2={mh.rsquared:.4f}  adj R^2={mh.rsquared_adj:.4f}")
df["y_tilde"] = np.nan
df.loc[keep, "y_tilde"] = mh.resid.values

# ----------------------------------------------------------------------
# (B) Group-specific age profile (Guvenen 2009; Arellano, Blundell and Bonhomme 2017)
# ----------------------------------------------------------------------
mg = fit_step0(extra_cols=["hp_age1","hp_age2","hp_age3","hp_age4"])
print(f"[group-specific age profile] N={int(mg.nobs):,}  R^2={mg.rsquared:.4f}  adj R^2={mg.rsquared_adj:.4f}")
df["y_tilde_grouphet"] = np.nan
df.loc[keep, "y_tilde_grouphet"] = mg.resid.values

# Joint Wald test of the four coefficients on age power x HighParentOcc
from statsmodels.stats.contrast import ContrastResults
hp_names = ["hp_age1","hp_age2","hp_age3","hp_age4"]
n_params = len(mg.params)
hp_idx = [list(mg.params.index).index(n) for n in hp_names]
R = np.zeros((4, n_params))
for r, j in enumerate(hp_idx):
    R[r, j] = 1.0
wald = mg.wald_test(R, scalar=True)
print(f"\n=== Joint Wald test: ψ_1=ψ_2=ψ_3=ψ_4=0  (age × HighParentOcc interactions) ===")
print(f"  F = {float(wald.statistic):.4f}, p = {float(wald.pvalue):.4f}")
print(f"  Interpretation: {'reject the pooled specification' if float(wald.pvalue) < 0.10 else 'cannot reject the pooled specification'}")

# ----------------------------------------------------------------------
# Compare the two residuals
# ----------------------------------------------------------------------
both = df.loc[keep, ["y_tilde","y_tilde_grouphet"]].dropna()
print(f"\n=== Pooled vs. group-specific residuals ===")
print(f"  correlation              : {both.corr().iloc[0,1]:.4f}")
print(f"  std of difference        : {(both['y_tilde']-both['y_tilde_grouphet']).std():.4f}")
print(f"  max |difference|         : {(both['y_tilde']-both['y_tilde_grouphet']).abs().max():.4f}")

# ----------------------------------------------------------------------
# (C) Winsorize y_tilde at the 1st and 99th percentiles of each wave.
#     y_tilde has a heavy left tail. The winsorized residual y_tilde_w is
#     the outcome of the income-process estimators; y_tilde is kept in the
#     output file.
# ----------------------------------------------------------------------
def winsorize_within_year(s, year, p_lo=0.01, p_hi=0.99):
    """Winsorize the series s at the p_lo and p_hi quantiles of each year."""
    out = s.copy()
    for y in year.dropna().unique():
        m = (year == y) & s.notna()
        if m.sum() < 10: continue
        lo = s[m].quantile(p_lo)
        hi = s[m].quantile(p_hi)
        out.loc[m] = s[m].clip(lower=lo, upper=hi)
    return out

df["y_tilde_w"] = winsorize_within_year(df["y_tilde"], df["year"])
n_w = ((df["y_tilde"] != df["y_tilde_w"]) & df["y_tilde"].notna()).sum()
print(f"\n=== Winsorize y_tilde at within-year p1/p99 ===")
print(f"  obs winsorized: {n_w:,} ({n_w/df['y_tilde'].notna().sum()*100:.2f}% of valid y_tilde)")
print(f"  y_tilde   range: [{df['y_tilde'].min():+.2f}, {df['y_tilde'].max():+.2f}]"
      f"   skew={df['y_tilde'].skew():+.2f}, kurt={df['y_tilde'].kurtosis():+.2f}")
print(f"  y_tilde_w range: [{df['y_tilde_w'].min():+.2f}, {df['y_tilde_w'].max():+.2f}]"
      f"   skew={df['y_tilde_w'].skew():+.2f}, kurt={df['y_tilde_w'].kurtosis():+.2f}")

# Variance and mean of the winsorized residual by wave and by HighParentOcc14
print("\nvar(y_tilde_w) by year:")
for yr in sorted(df["year"].unique()):
    sub = df.loc[df["year"]==yr, "y_tilde_w"].dropna()
    print(f"  {yr}: N={len(sub):,}, var={sub.var():.4f}, mean={sub.mean():.4f}")

print("\nvar(y_tilde_w) by HighParentOcc14 (within-group):")
for g in [0, 1]:
    sub = df.loc[df["HighParentOcc14"]==g, "y_tilde_w"].dropna()
    print(f"  HighParentOcc14={g}: N={len(sub):,}, var={sub.var():.4f}, mean={sub.mean():.4f}")

# ----------------------------------------------------------------------
# Write the file
# ----------------------------------------------------------------------
cols_out = [
    "pid","year","income","log_income",
    "y_tilde","y_tilde_w","y_tilde_grouphet",
    "age","female","urban","provcd",
    "pid_father","pid_mother","occ_father","occ_mother",
    "fd_father","fd_mother",
    "HighParentOcc","HighParentOcc14","has_qv14",
    "ParentInc_bar","eduy",
]
out_df = df[cols_out].copy()
for c in cols_out:
    if c not in ("pid","year"):
        out_df[c] = out_df[c].astype(float)
pkl_path = os.path.join(BASE_INT, "panel_resid.pkl")
out_df.to_pickle(pkl_path)
print(f"\nsaved {pkl_path}")
out_path = os.path.join(BASE_INT, "panel_resid.dta")
out_tmp = out_path + ".tmp"
try:
    pyreadstat.write_dta(out_df, out_tmp)
    os.replace(out_tmp, out_path)
    print(f"saved {out_path}")
except Exception as e:
    print(f"WARN: dta write failed ({e}); use the .pkl file")
