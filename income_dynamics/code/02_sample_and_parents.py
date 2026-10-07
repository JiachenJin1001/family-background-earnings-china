import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Link children to their parents, classify family background, measure parental
income, and select the child sample of the income-dynamics analysis.

Variables built here (all constant within child):
  - HighParentOcc14  1 if the father or the mother worked in an official,
                     managerial, or professional occupation (occupation code
                     with first digit 1 or 2, CSCO major group 1 or 2) when the
                     child was fourteen. Source: the retrospective questions
                     qv103code_best (father) and qv203code_best (mother) of the
                     2012 wave and qv103code and qv203code of the 2020 and 2022
                     waves. Children with HighParentOcc14 = 1 are the
                     advantaged children of the paper; children with
                     HighParentOcc14 = 0 are the less advantaged children.
  - has_qv14         1 if the child has at least one valid retrospective
                     occupation code for the father or the mother.
  - HighParentOcc    the same indicator built from the occupation code that
                     the parents report for themselves as adult respondents
                     (first non-missing code across waves, or the code in the
                     2010 family-relationship file).
  - ParentInc_bar    average over father and mother of the parent's mean log
                     family income across the waves in which the parent's
                     family is observed.

Sample written to panel_analysis.pkl and panel_analysis.dta (one row per child
and wave): waves 2014 to 2022, age 22 to 55, positive income, HighParentOcc
non-missing, log income at least ln(1000) and at most the 99th percentile of
its wave, and at least three such rows for the child.

Because the filter requires HighParentOcc to be non-missing, about 420 children
with a valid HighParentOcc14 and no occupation code from the parents' own adult
records are not in the sample.
"""
import os, sys
import numpy as np
import pandas as pd
import pyreadstat

BASE_RAW = _WSROOT + "/data/raw"
BASE_INT = _WSROOT + "/data/intermediate"

# Files with the retrospective questions on the parents' occupation when the
# respondent was fourteen. The 2012 wave provides qv103code_best (father) and
# qv203code_best (mother); the 2020 and 2022 waves provide qv103code and
# qv203code. The 2010, 2014, 2016 and 2018 waves do not ask these questions.
QV14_FILES = [
    ("CFPS2012/ecfps2012adult_202505.dta",  "qv103code_best", "qv203code_best"),
    ("CFPS2020/ecfps2020person_202306.dta", "qv103code",      "qv203code"),
    ("CFPS2022/ecfps2022person_202410.dta", "qv103code",      "qv203code"),
]

FAMCONF = {
    2010: ("CFPS2010/ecfps2010famconf_nat072016.dta", "pid_f", "pid_m"),
    2012: ("CFPS2012/ecfps2012famconf_092015.dta", "pid_f", "pid_m"),
    2014: ("CFPS2014/ecfps2014famconf_170630.dta", "pid_f", "pid_m"),
    2016: ("CFPS2016/ecfps2016famconf_201804.dta", "pid_f", "pid_m"),
    2018: ("CFPS2018/ecfps2018famconf_202008.dta", "pid_a_f", "pid_a_m"),
    2020: ("CFPS2020/ecfps2020famconf_202306.dta", "pid_a_f", "pid_a_m"),
    2022: ("CFPS2022/ecfps2022famconf_202410.dta", "pid_f", "pid_m"),
}

# 1. Load the person-year panel. The .pkl copy is read when it exists (it loads
# faster); otherwise the .dta file is read.
_cache = os.path.join(BASE_INT, "_panel_long_cache.pkl")
_dta = os.path.join(BASE_INT, "panel_long.dta")
if os.path.exists(_cache) and os.path.getsize(_cache) > 1000:
    long = pd.read_pickle(_cache)
    print(f"loaded panel_long from pkl cache: {len(long):,} rows, {long['pid'].nunique():,} unique pids")
else:
    long, _ = pyreadstat.read_dta(_dta)
    long["pid"] = long["pid"].astype(np.int64)
    print(f"loaded panel_long from dta: {len(long):,} rows, {long['pid'].nunique():,} unique pids")

# 2. Parent linkage: for each child, the first non-missing father pid and
# mother pid across the family-relationship files of the seven waves
frames = []
for yr, (rel, pidf, pidm) in FAMCONF.items():
    df, _ = pyreadstat.read_dta(os.path.join(BASE_RAW, rel), usecols=["pid", pidf, pidm])
    df = df.rename(columns={pidf:"pid_father", pidm:"pid_mother"})
    df = df.dropna(subset=["pid"])
    df["pid"] = df["pid"].astype(np.int64)
    for c in ["pid_father","pid_mother"]:
        df.loc[df[c] <= 0, c] = np.nan
    df["famconf_year"] = yr
    frames.append(df[["pid","pid_father","pid_mother","famconf_year"]])
pool = pd.concat(frames, ignore_index=True)
# sort by wave, so that the earliest non-missing value is the one kept
pool = pool.sort_values(["pid","famconf_year"])
# first non-missing father and mother per child
link = pool.groupby("pid").agg(
    pid_father=("pid_father", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
    pid_mother=("pid_mother", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
).reset_index()
print(f"parent linkage: {len(link):,} pids; father link: {link['pid_father'].notna().sum():,}; mother link: {link['pid_mother'].notna().sum():,}")

# 3. Each person's own occupation code: first positive occ_code across waves
own = long.loc[long["occ_code"].notna() & (long["occ_code"]>0)].sort_values(["pid","year"])
own_first = own.drop_duplicates(subset=["pid"])[["pid","occ_code"]].rename(columns={"occ_code":"own_occ"})
print(f"own-occ lookup: {len(own_first):,} pids")

# 4. Occupation codes of the parents as recorded in the 2010
# family-relationship file, for parents without an adult record of their own
fb, _ = pyreadstat.read_dta(os.path.join(BASE_RAW, FAMCONF[2010][0]),
                            usecols=["pid_f","pid_m","tb5_code_a_f","tb5_code_a_m"])
parts = []
for ppid, pocc in [("pid_f","tb5_code_a_f"),("pid_m","tb5_code_a_m")]:
    d = fb[[ppid,pocc]].rename(columns={ppid:"pid",pocc:"fb_occ"}).copy()
    d = d.loc[d["pid"].notna() & (d["pid"]>0) & d["fb_occ"].notna() & (d["fb_occ"]>0)]
    parts.append(d)
fb2 = pd.concat(parts, ignore_index=True).astype({"pid":np.int64})
fb2 = fb2.drop_duplicates(subset=["pid"])
print(f"2010-famconf fallback occ lookup: {len(fb2):,} pids")

# The parent's own code is used when available, the 2010 code otherwise.
occ_pool = own_first.merge(fb2, on="pid", how="outer")
occ_pool["occ_final"] = occ_pool["own_occ"].where(occ_pool["own_occ"].notna(), occ_pool["fb_occ"])
print(f"combined occ lookup: {len(occ_pool):,} pids, occ_final non-null: {occ_pool['occ_final'].notna().sum():,}")

# 5. Child-level variables
df = long.merge(link, on="pid", how="left")

# Occupation code of the father and of the mother
pf = occ_pool[["pid","occ_final"]].rename(columns={"pid":"pid_father","occ_final":"occ_father"})
pm = occ_pool[["pid","occ_final"]].rename(columns={"pid":"pid_mother","occ_final":"occ_mother"})
# numeric parent pids, so that the merge keys have the same type
for c in ["pid_father","pid_mother"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.merge(pf, on="pid_father", how="left")
df = df.merge(pm, on="pid_mother", how="left")

def first_digit_vec(s):
    s = pd.to_numeric(s, errors="coerce")
    # first digit of positive codes; missing otherwise
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s > 0
    out.loc[mask] = (s.loc[mask].astype(np.int64).astype(str).str[0].astype(int)).values
    return out

df["fd_father"] = first_digit_vec(df["occ_father"])
df["fd_mother"] = first_digit_vec(df["occ_mother"])
df["any_parent_occ"] = df["fd_father"].notna() | df["fd_mother"].notna()
df["HighParentOcc"] = (
    df["fd_father"].isin([1,2]) | df["fd_mother"].isin([1,2])
).astype(int)
df.loc[~df["any_parent_occ"], "HighParentOcc"] = np.nan

# ----------------------------------------------------------------------
# 5b. HighParentOcc14: occupation of the parents when the child was fourteen
# ----------------------------------------------------------------------
# Father's code (qv103code_best, qv103code) and mother's code (qv203code_best,
# qv203code) from the 2012, 2020 and 2022 waves. The child is classified as
# advantaged if any of these codes, for either parent in any wave, has first
# digit 1 or 2.
qv_frames = []
for rel, fcol, mcol in QV14_FILES:
    src = os.path.join(BASE_RAW, rel)
    if not os.path.exists(src):
        print(f"  WARN: missing {src}; skipping wave for qv14")
        continue
    qv, _ = pyreadstat.read_dta(src, usecols=["pid", fcol, mcol])
    qv = qv.rename(columns={fcol:"qv14_f", mcol:"qv14_m"})
    qv = qv.dropna(subset=["pid"])
    qv["pid"] = qv["pid"].astype(np.int64)
    for c in ["qv14_f","qv14_m"]:
        qv.loc[qv[c] <= 0, c] = np.nan  # CFPS codes refusals and not-applicable as negative values
    qv_frames.append(qv[["pid","qv14_f","qv14_m"]])
qv_pool = pd.concat(qv_frames, ignore_index=True) if qv_frames else \
          pd.DataFrame(columns=["pid","qv14_f","qv14_m"])
print(f"qv14 retrospective rows pooled: {len(qv_pool):,}; unique pids with at least one obs: {qv_pool['pid'].nunique():,}")

# First digit of each code, then the indicator for first digit 1 or 2 by row
qv_pool["qvfd_f"] = first_digit_vec(qv_pool["qv14_f"])
qv_pool["qvfd_m"] = first_digit_vec(qv_pool["qv14_m"])
qv_pool["high14_father"] = qv_pool["qvfd_f"].isin([1,2]).astype(int)
qv_pool["high14_mother"] = qv_pool["qvfd_m"].isin([1,2]).astype(int)
qv_pool.loc[qv_pool["qvfd_f"].isna(), "high14_father"] = 0
qv_pool.loc[qv_pool["qvfd_m"].isna(), "high14_mother"] = 0

# Per child: maximum across waves (a code with first digit 1 or 2 in any wave
# classifies the child as advantaged), and an indicator for having at least
# one valid retrospective code for the father or the mother
qv_child = qv_pool.groupby("pid").agg(
    HighParentOcc14=("high14_father", "max"),
    HighParentOcc14_m=("high14_mother", "max"),
    has_qv14_f=("qvfd_f", lambda s: s.notna().any()),
    has_qv14_m=("qvfd_m", lambda s: s.notna().any()),
).reset_index()
qv_child["HighParentOcc14"] = (qv_child[["HighParentOcc14","HighParentOcc14_m"]].max(axis=1)).astype(float)
qv_child["has_qv14"] = (qv_child["has_qv14_f"] | qv_child["has_qv14_m"]).astype(int)
qv_child = qv_child[["pid","HighParentOcc14","has_qv14"]]
# Without any valid retrospective code, HighParentOcc14 is missing
qv_child.loc[qv_child["has_qv14"] == 0, "HighParentOcc14"] = np.nan

print(f"qv14 child-level pids: {len(qv_child):,}, with valid father-or-mother at-14 occ: {(qv_child['has_qv14']==1).sum():,}")
df = df.merge(qv_child, on="pid", how="left")
df["has_qv14"] = df["has_qv14"].fillna(0).astype(int)

# ----------------------------------------------------------------------
# 5c. ParentInc_bar from family income in the seven waves
#
#     Sources:
#       2010: faminc_net        (net family income as defined by CFPS)
#       2012 to 2018: fincome2  (net family income comparable to 2010)
#       2020 and 2022: finc     (total family income; fincome2 is not provided)
#
#     Each person is mapped to a family (fid of the wave) with the
#     family-relationship file, and the family to its income with the
#     family-economy file. For each parent, log family income is averaged
#     over the waves in which the parent's family is observed.
# ----------------------------------------------------------------------
FAMECON = [
    (2010, "CFPS2010/ecfps2010famecon_201906.dta",
            "CFPS2010/ecfps2010famconf_nat072016.dta", "fid",   "faminc_net"),
    (2012, "CFPS2012/ecfps2012famecon_201906.dta",
            "CFPS2012/ecfps2012famconf_092015.dta",   "fid12", "fincome2"),
    (2014, "CFPS2014/ecfps2014famecon_201906.dta",
            "CFPS2014/ecfps2014famconf_170630.dta",   "fid14", "fincome2"),
    (2016, "CFPS2016/ecfps2016famecon_201807.dta",
            "CFPS2016/ecfps2016famconf_201804.dta",   "fid16", "fincome2"),
    (2018, "CFPS2018/ecfps2018famecon_202101.dta",
            "CFPS2018/ecfps2018famconf_202008.dta",   "fid18", "fincome2"),
    (2020, "CFPS2020/ecfps2020famecon_202306.dta",
            "CFPS2020/ecfps2020famconf_202306.dta",   "fid20", "finc"),
    (2022, "CFPS2022/ecfps2022famecon_202410.dta",
            "CFPS2022/ecfps2022famconf_202410.dta",   "fid22", "finc"),
]

fam_inc_frames = []
for yr, econ_fn, conf_fn, fid_col, inc_col in FAMECON:
    econ_path = os.path.join(BASE_RAW, econ_fn)
    conf_path = os.path.join(BASE_RAW, conf_fn)
    if not (os.path.exists(econ_path) and os.path.exists(conf_path)):
        print(f"  WARN: missing files for {yr} family income; skipping")
        continue
    # family-economy file: family id and family income (positive values only)
    econ, _ = pyreadstat.read_dta(econ_path, usecols=[fid_col, inc_col])
    econ = econ.rename(columns={fid_col:"fid_y", inc_col:"fam_inc"})
    econ["fid_y"] = pd.to_numeric(econ["fid_y"], errors="coerce")
    econ = econ.loc[econ["fam_inc"].notna() & (econ["fam_inc"] > 0)]
    # family-relationship file: person id and family id of the wave
    conf, _ = pyreadstat.read_dta(conf_path, usecols=["pid", fid_col])
    conf = conf.rename(columns={fid_col:"fid_y"})
    conf["pid"]   = pd.to_numeric(conf["pid"],   errors="coerce")
    conf["fid_y"] = pd.to_numeric(conf["fid_y"], errors="coerce")
    conf = conf.dropna(subset=["pid","fid_y"]).astype({"pid": np.int64})
    # family income of each person in this wave
    pid_inc = conf.merge(econ, on="fid_y", how="inner")
    pid_inc["year"] = yr
    pid_inc = pid_inc[["pid","year","fam_inc"]]
    pid_inc["log_fam_inc"] = np.log(pid_inc["fam_inc"])
    fam_inc_frames.append(pid_inc)
    print(f"  {yr} family income: N pid-year={len(pid_inc):,}, "
          f"median={pid_inc['fam_inc'].median():,.0f}, log Var={pid_inc['log_fam_inc'].var():.3f}")

fam_inc = pd.concat(fam_inc_frames, ignore_index=True)
print(f"  pooled across 7 waves: {len(fam_inc):,} pid-year rows, "
      f"{fam_inc['pid'].nunique():,} unique pids")

# For each parent, the mean of log family income over the waves observed
parent_fam_avg = (fam_inc.groupby("pid")["log_fam_inc"]
                          .mean()
                          .reset_index()
                          .rename(columns={"log_fam_inc":"parent_log_fam_inc_mean"}))
print(f"  parent-level long-run family income: {len(parent_fam_avg):,} pids contribute")

# Attach the father's and the mother's mean to the child rows
df = df.merge(parent_fam_avg.rename(columns={"pid":"pid_father",
                                              "parent_log_fam_inc_mean":"finc_fam_mean"}),
              on="pid_father", how="left")
df = df.merge(parent_fam_avg.rename(columns={"pid":"pid_mother",
                                              "parent_log_fam_inc_mean":"minc_fam_mean"}),
              on="pid_mother", how="left")
df["ParentInc_bar"] = df[["finc_fam_mean","minc_fam_mean"]].mean(axis=1)
print(f"  ParentInc_bar non-null on full df: "
      f"{df['ParentInc_bar'].notna().sum():,} / {len(df):,} rows "
      f"({df.loc[df['ParentInc_bar'].notna(),'pid'].nunique():,} unique children)")

# Sample filter
# The children with at least three waves at ages 22 to 55 (Sections 3 and 4 of the paper) are selected by
# unified/code/d0_dynamics_panel.py with the rule of unified/code/sample_mature_age.py,
# which reads the child-wave frame `df` built above and writes
# data/intermediate/panel_analysis.pkl. Nothing is filtered here.
