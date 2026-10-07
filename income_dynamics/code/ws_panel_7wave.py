import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Seven-wave panel 2010 to 2022.

The income variable of the waves 2014 to 2022 is income from jobs. The 2010
and 2012 waves report personal income from all sources, so for these two
waves income from jobs is reconstructed from the components in the adult
questionnaire, and the rows are added to the five-wave panel.

Steps:
  1. Read panel_resid.pkl (children aged 22 to 55, waves 2014 to 2022, at
     least three waves with positive income), which carries HighParentOcc14,
     ParentInc_bar, the parents' pids and the child's characteristics.
  2. 2010 income from jobs = qk101*12 + qk102 + qk105 + qk106 (monthly wage
     of the main job times 12, annual bonus, annual self-employment income,
     annual income from secondary jobs).
  3. 2012 income from jobs = sum of qg417_a_1 to qg417_a_10, qg5141_a_1 to
     qg5141_a_4, ksa106 and qg305.
  4. Keep the 2010 and 2012 rows of the children of the five-wave panel and
     attach their family-background variables.
  5. On the combined panel: age 22 to 55, positive income, log income at
     least ln(1000) and at most the 99th percentile of its wave, at least
     three waves per child.
  6. Residual of log income from the regression on a quartic in age, female,
     their interactions, urban, province indicators and year indicators
     (y_tilde), and its winsorized version (y_tilde_w).
  7. Write ws/panel_resid_7w.pkl and ws/panel_resid_7w.dta; print the number
     of rows and the variance of log income by wave.
"""
import os
import numpy as np
import pandas as pd
import pyreadstat

BASE_RAW = _WSROOT + "/data/raw"
BASE_INT = _WSROOT + "/data/intermediate"
WS_OUT   = os.path.join(BASE_INT, "ws")

# ----------------------------------------------------------------------
# 1. Five-wave panel 2014 to 2022
# ----------------------------------------------------------------------
locked = pd.read_pickle(os.path.join(BASE_INT, "panel_resid.pkl"))
locked["pid"]  = locked["pid"].astype(np.int64)
locked["year"] = locked["year"].astype(int)
print(f"Five-wave 2014-2022 panel: {len(locked):,} rows, {locked['pid'].nunique():,} children, "
      f"years = {sorted(locked['year'].unique())}")

# Variables that are constant within child, to be attached to the 2010 and 2012 rows
child_invariant = locked.drop_duplicates("pid")[
    ["pid","female","urban","provcd","pid_father","pid_mother",
     "occ_father","occ_mother","fd_father","fd_mother",
     "HighParentOcc","HighParentOcc14","has_qv14","ParentInc_bar","eduy"]
].copy()
print(f"Child-level variables extracted for {len(child_invariant):,} children")

# ----------------------------------------------------------------------
# 2. 2010 income from jobs (the gender variable of the 2010 file is `gender`)
# ----------------------------------------------------------------------
print("\nReconstructing 2010 income from jobs ...")
need_2010 = ["pid","qk101","qk102","qk105","qk106","qk2","qa1age","gender","urban","provcd"]
hd, _ = pyreadstat.read_dta(os.path.join(BASE_RAW, "CFPS2010/ecfps2010adult_201906.dta"), row_limit=1)
have_2010 = [c for c in need_2010 if c in hd.columns]
df10, _ = pyreadstat.read_dta(os.path.join(BASE_RAW, "CFPS2010/ecfps2010adult_201906.dta"),
                                usecols=have_2010)
for c in ["qk101","qk102","qk105","qk106","qk2"]:
    df10[c] = pd.to_numeric(df10.get(c, 0), errors="coerce").fillna(0)
    df10.loc[df10[c] < 0, c] = 0
# CFPS 2010 questionnaire: qk101 = monthly wage of the main job, qk102 =
# bonus (annual), qk105 = annual self-employment income, qk106 = annual
# income from secondary jobs, qk2 = annual transfers.
# Income from jobs = qk101*12 + qk102 + qk105 + qk106. Transfers (qk2) are
# not part of income from jobs and are left out.
df10["income"] = df10["qk101"]*12 + df10["qk102"] + df10["qk105"] + df10["qk106"]
df10.loc[df10["income"] <= 0, "income"] = np.nan
df10["year"] = 2010
df10["age"] = pd.to_numeric(df10.get("qa1age"), errors="coerce")
df10["gender_raw"] = pd.to_numeric(df10.get("gender"), errors="coerce")  # gender variable of the 2010 file
df10["log_income"] = np.log(df10["income"])
print(f"  2010 income from jobs: N positive income = {df10['income'].notna().sum():,}, "
      f"median = {df10['income'].dropna().median():,.0f}, "
      f"log Var = {df10['log_income'].dropna().var():.4f}")

# ----------------------------------------------------------------------
# 3. 2012 income from jobs
# ----------------------------------------------------------------------
print("\nReconstructing 2012 income from jobs ...")
inc_components_12 = [f"qg417_a_{k}" for k in range(1,11)] + \
                     [f"qg5141_a_{k}" for k in range(1,5)] + \
                     ["ksa106","qg305"]
need_2012 = ["pid","cfps2012_age","cfps2012_gender_best","urban12","provcd"] + inc_components_12
hd12, _ = pyreadstat.read_dta(os.path.join(BASE_RAW, "CFPS2012/ecfps2012adult_202505.dta"), row_limit=1)
have_2012 = [c for c in need_2012 if c in hd12.columns]
df12, _ = pyreadstat.read_dta(os.path.join(BASE_RAW, "CFPS2012/ecfps2012adult_202505.dta"),
                                usecols=have_2012)
have_inc_12 = [c for c in inc_components_12 if c in df12.columns]
for c in have_inc_12:
    df12[c] = pd.to_numeric(df12[c], errors="coerce").fillna(0)
    df12.loc[df12[c] < 0, c] = 0
# qg417_a_1 to qg417_a_10: wages by job. qg5141_a_1 to qg5141_a_4:
# self-employment income. ksa106: income from secondary jobs. qg305: annual
# job income. Income from jobs is the sum of these components.
df12["income"] = df12[have_inc_12].sum(axis=1)
df12.loc[df12["income"] <= 0, "income"] = np.nan
df12["year"] = 2012
df12["age"] = pd.to_numeric(df12.get("cfps2012_age"), errors="coerce")
df12["gender_raw"] = pd.to_numeric(df12.get("cfps2012_gender_best"), errors="coerce")
df12["urban"] = pd.to_numeric(df12.get("urban12"), errors="coerce")
df12["provcd"] = pd.to_numeric(df12.get("provcd"), errors="coerce")
df12["log_income"] = np.log(df12["income"])
print(f"  2012 income from jobs: N positive income = {df12['income'].notna().sum():,}, "
      f"median = {df12['income'].dropna().median():,.0f}, "
      f"log Var = {df12['log_income'].dropna().var():.4f}")

# In the 2010 file the residence and province variables are `urban` and `provcd`
df10["urban"] = pd.to_numeric(df10.get("urban"), errors="coerce")
df10["provcd"] = pd.to_numeric(df10.get("provcd"), errors="coerce")

# Female indicator from the gender code, as in 01_build_panel.py: 0 is female, 1 is male
def to_female(s):
    return np.where(s==0, 1.0, np.where(s==1, 0.0, np.nan))
df10["female"] = to_female(df10["gender_raw"])
df12["female"] = to_female(df12["gender_raw"])

# Share of women in the 2010 and 2012 files
print(f"\n  2010 female share (raw): {df10['female'].mean():.3f}")
print(f"  2012 female share (raw): {df12['female'].mean():.3f}")

# Columns kept
df10s = df10[["pid","year","income","log_income","age","female","urban","provcd"]].copy()
df12s = df12[["pid","year","income","log_income","age","female","urban","provcd"]].copy()
df10s["pid"] = pd.to_numeric(df10s["pid"], errors="coerce")
df12s["pid"] = pd.to_numeric(df12s["pid"], errors="coerce")
df10s = df10s.dropna(subset=["pid"]).astype({"pid": np.int64})
df12s = df12s.dropna(subset=["pid"]).astype({"pid": np.int64})

# ----------------------------------------------------------------------
# 4. Attach the child-level variables (HP14, ParentInc_bar and the others)
#    to the 2010 and 2012 rows. A 2010 or 2012 row enters the seven-wave
#    panel only if the child is in the five-wave panel, which is where the
#    parent linkage comes from.
# ----------------------------------------------------------------------
# female, urban and provcd are taken from the 2010 or 2012 wave when
# reported there, and from the five-wave panel otherwise.
for src in [df10s, df12s]:
    src["female"] = src["female"]
# female, urban and provcd are dropped from the child-level table before the
# merge, so that the values reported in the 2010 or 2012 wave are kept
ci_drop = child_invariant.drop(columns=["female","urban","provcd"]).copy()
df10s_full = df10s.merge(ci_drop, on="pid", how="inner")
df12s_full = df12s.merge(ci_drop, on="pid", how="inner")

# Where female, urban or provcd is missing in the 2010 or 2012 wave, the
# value of the child in the five-wave panel is used.
ci_fem = child_invariant.set_index("pid")["female"]
ci_urb = child_invariant.set_index("pid")["urban"]
ci_pro = child_invariant.set_index("pid")["provcd"]
for src in [df10s_full, df12s_full]:
    src["female"] = src["female"].where(src["female"].notna(), src["pid"].map(ci_fem))
    src["urban"]  = src["urban"].where(src["urban"].notna(),  src["pid"].map(ci_urb))
    src["provcd"] = src["provcd"].where(src["provcd"].notna(), src["pid"].map(ci_pro))

print(f"\n  After merging on the children of the five-wave panel:")
print(f"    2010 rows: {len(df10s_full):,}, with valid income: {df10s_full['income'].notna().sum():,}")
print(f"    2012 rows: {len(df12s_full):,}, with valid income: {df12s_full['income'].notna().sum():,}")

# ----------------------------------------------------------------------
# 5. Combine with the 2014 to 2022 rows and apply the same trimming
# ----------------------------------------------------------------------
locked_keep_cols = ["pid","year","income","log_income","age","female","urban","provcd",
                     "pid_father","pid_mother","occ_father","occ_mother",
                     "fd_father","fd_mother","HighParentOcc","HighParentOcc14",
                     "has_qv14","ParentInc_bar","eduy"]
combined_cols = locked_keep_cols
df10_done = df10s_full[combined_cols].copy()
df12_done = df12s_full[combined_cols].copy()

panel7 = pd.concat([df10_done, df12_done, locked[combined_cols].copy()],
                    ignore_index=True)
panel7["age"] = pd.to_numeric(panel7["age"], errors="coerce")

# Age 22 to 55, as in the five-wave panel
n_before_age = len(panel7)
panel7 = panel7.loc[panel7["age"].between(22, 55)].copy()
print(f"\n  After age filter 22-55: {len(panel7):,} rows  (dropped {n_before_age-len(panel7):,})")

# Drop rows without log income
panel7 = panel7.dropna(subset=["log_income"]).copy()
print(f"  With valid log_income: {len(panel7):,}")

# Trimming: drop log income below ln(1000) and above the 99th percentile of the wave
LO = np.log(1000)
mask_lo = panel7["log_income"] >= LO
top1pct = panel7.groupby("year")["log_income"].transform(lambda s: s.quantile(0.99))
mask_hi = panel7["log_income"] <= top1pct
n_lo, n_hi = (~mask_lo).sum(), (~mask_hi).sum()
panel7 = panel7.loc[mask_lo & mask_hi].copy()
print(f"  Trim: drop log_y < ln(1000): {n_lo}; drop top 1% within year: {n_hi}")
print(f"  After trim: {len(panel7):,} rows, {panel7['pid'].nunique():,} children")

# At least three waves per child, as in the five-wave panel
counts = panel7.groupby("pid").size()
keep_pids = counts[counts >= 3].index
panel7 = panel7.loc[panel7["pid"].isin(keep_pids)].copy()
print(f"  After T_i >= 3: {len(panel7):,} rows, {panel7['pid'].nunique():,} children")

# Rows, variance of log income and number of advantaged children by wave
print(f"\n  Seven-wave panel by year:")
for yr in sorted(panel7["year"].unique()):
    sub = panel7.loc[panel7["year"]==yr]
    print(f"    {int(yr)}: N obs = {len(sub):>5,}, "
          f"log Var = {sub['log_income'].var():.4f}, "
          f"HP14=1: {(sub['HighParentOcc14']==1).sum():>3,}")

# ----------------------------------------------------------------------
# 6. Residual of log income on the seven-wave panel
# ----------------------------------------------------------------------
import statsmodels.api as sm

reg_keep = (panel7["log_income"].notna() & panel7["age"].notna() & panel7["female"].notna() &
             panel7["urban"].notna() & panel7["provcd"].notna() &
             panel7["HighParentOcc14"].notna())
print(f"\n  Rows in the residual regression (HP14, controls and log income non-missing): {reg_keep.sum():,}")
reg = panel7.loc[reg_keep].copy()
for k in range(1, 5):
    reg[f"age{k}"] = reg["age"]**k
    reg[f"f_age{k}"] = reg["female"] * reg["age"]**k
prov_dum = pd.get_dummies(reg["provcd"].astype(int), prefix="prov", drop_first=True, dtype=float)
year_dum = pd.get_dummies(reg["year"].astype(int),   prefix="yr",   drop_first=True, dtype=float)
X = pd.concat([reg[["age1","age2","age3","age4","female",
                     "f_age1","f_age2","f_age3","f_age4","urban"]],
                prov_dum, year_dum], axis=1)
X = sm.add_constant(X, has_constant="add")
y = reg["log_income"].astype(float)
m0 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": reg["pid"].values})
panel7["y_tilde"] = np.nan
panel7.loc[reg_keep, "y_tilde"] = m0.resid.values
print(f"  Residual regression: N = {int(m0.nobs):,}, R² = {m0.rsquared:.4f}")
print(f"  var(y_tilde) by year:")
for yr in sorted(panel7["year"].unique()):
    sub = panel7.loc[panel7["year"]==yr, "y_tilde"].dropna()
    print(f"    {int(yr)}: N={len(sub):,}, var={sub.var():.4f}, mean={sub.mean():+.4f}")

# Winsorize y_tilde at the 1st and 99th percentiles of each wave
panel7["y_tilde_w"] = panel7["y_tilde"]
for yr in panel7["year"].dropna().unique():
    m_yr = (panel7["year"]==yr) & panel7["y_tilde"].notna()
    if m_yr.sum() < 10: continue
    p1 = panel7.loc[m_yr, "y_tilde"].quantile(0.01)
    p99 = panel7.loc[m_yr, "y_tilde"].quantile(0.99)
    panel7.loc[m_yr, "y_tilde_w"] = panel7.loc[m_yr, "y_tilde"].clip(lower=p1, upper=p99)
n_w = ((panel7["y_tilde"] != panel7["y_tilde_w"]) & panel7["y_tilde"].notna()).sum()
print(f"\n  Winsorized {n_w:,} obs ({n_w/panel7['y_tilde'].notna().sum()*100:.2f}% of valid y_tilde)")

# Write the seven-wave panel
out_pkl = os.path.join(WS_OUT, "panel_resid_7w.pkl")
panel7.to_pickle(out_pkl)
print(f"\nsaved -> {out_pkl}")
try:
    pyreadstat.write_dta(panel7, os.path.join(WS_OUT, "panel_resid_7w.dta"))
    print(f"saved -> {os.path.join(WS_OUT, 'panel_resid_7w.dta')}")
except Exception as e:
    print(f"  WARN: dta write failed ({e})")
