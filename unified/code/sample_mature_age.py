"""The sample of the paper, built by one rule, and the earnings measures taken from it.

Sample: children with a linked parent, a retrospective report of parental
occupation at age fourteen, the family income of the parental household, and
positive labor income in at least two waves of 2014 to 2022 at ages 22 to 55,
who live in a province with a measure of expansion intensity. Wave by wave,
incomes below 1,000 yuan and the top one percent of log income are dropped.
A child's birth year, province, gender and urban residence are those of the
first wave in which the child is observed in the window. The national exposure
series starts with the cohort born in 1964; the few children born earlier get
the mean exposure ratio of the cohorts born 1964 to 1980, all of whom were of
college age before the reform.

What each section takes from the sample:
  Sections 3 and 4   the children with at least three waves, because the dynamic
                     panel estimators need three (DYNAMICS = (22, 55, 3))
  Section 5          the level of earnings at ages 30 and over: the mean, over a
                     child's waves at 30 and over, of log labor income net of
                     survey-year effects, for the children with at least two
                     such waves (build_sample(), child_waves())
  Sections 6 and 7   the two rows per child of two_row(): that mean, and the
                     same mean over the waves below 30

  prefilter_frame()              child-wave rows of income_dynamics/code/02_sample_and_parents.py
                                 before any sample filter (needs the survey files; cached)
  build_sample(level_from=None)  all children of the sample, one row each (log_y: mean log income over all waves)
  build_sample()                 children with at least two waves at 30 and over; log_y is the level of earnings at
                                 30 and over defined above; age_first and waves refer to the waves at 30 and over
  child_waves()                  the child-wave rows at 30 and over of those children
  child_waves(level_from=None)   all child-wave rows of the children of the sample
  two_row()                     see the function
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _locate import sibling
P = sibling("college_expansion", __file__); M = sibling("income_dynamics", __file__)
EXT = os.path.join(P, "data", "external")
LO, HI, MIN_WAVES = 22, 55, 2          # the sample: lowest age, highest age, minimum number of waves
LEVEL_FROM = 30                        # the level of earnings is measured at ages 30 and over
OLD_MIN, YOUNG_MIN = 2, 1              # waves behind the mean at 30 and over (a multi-year average) and behind the mean below 30
DYNAMICS = (LO, HI, 3)                 # children with at least three waves (Sections 3 and 4)


def prefilter_frame():
    """The child-wave frame of 02_sample_and_parents.py up to its sample filter (needs the survey files).
    Cached as a person-level file next to the panels of the pipeline that holds the survey files."""
    for pipe in (P, M):
        if os.path.isdir(os.path.join(pipe, "data", "raw", "CFPS2012")):
            break
    else:
        sys.exit("the survey files (data/raw/CFPS2012 and so on) were not found in either pipeline folder")
    cache = os.path.join(pipe, "data", "intermediate", "ws", "child_waves_before_sample_filter.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    script = os.path.join(pipe, "code", "02_sample_and_parents.py")
    src = open(script, encoding="utf-8").read()
    src = src[:src.index("# Sample filter")]
    g = {"__file__": script, "__name__": "prefilter"}
    exec(compile(src, script, "exec"), g)
    df = g["df"]; os.makedirs(os.path.dirname(cache), exist_ok=True); df.to_pickle(cache)
    return df


def _waves(lo=LO, hi=HI, min_waves=MIN_WAVES):
    df = prefilter_frame()
    d = df[(df.year >= 2014) & df.age.between(lo, hi) & (df.income > 0) & df.HighParentOcc14.notna() & df.ParentInc_bar.notna()
           & df.female.notna() & df.urban.notna() & df.provcd.notna()].copy()
    d["log_income"] = np.log(d.income)
    d = d[d.log_income >= np.log(1000)]
    d = d[d.log_income <= d.groupby("year").log_income.transform(lambda s: s.quantile(0.99))]
    n = d.groupby("pid").size()
    return d[d.pid.isin(n[n >= min_waves].index)].copy()


def _children(d):
    c = d.groupby("pid", as_index=False).agg(log_y=("log_income", "mean"), HP14=("HighParentOcc14", "first"), ParentInc=("ParentInc_bar", "first"),
                                             female=("female", "first"), urban=("urban", "first"), provcd=("provcd", "first"),
                                             age_first=("age", "min"), year_first=("year", "min"), waves=("year", "size"))
    c["birth_year"] = c.year_first - c.age_first
    pl = pd.read_pickle(os.path.join(P, "data", "intermediate", "_panel_long_cache.pkl"))
    c = c.merge(pl.groupby("pid").eduy.max().rename("eduy_max"), left_on="pid", right_index=True, how="left")
    occ = pl.loc[pl.occ_code.notna() & (pl.occ_code > 0) & pl.pid.isin(c.pid), ["pid", "occ_code"]].copy()
    occ["prof"] = (occ.occ_code.astype(int).astype(str).str[0].astype(int) <= 2).astype(float)
    gp = occ.groupby("pid").prof
    c = c.merge(pd.DataFrame({"own_prof": gp.max(), "own_maj": (gp.mean() > 0.5).astype(float)}), left_on="pid", right_index=True, how="left")
    c[["own_prof", "own_maj"]] = c[["own_prof", "own_maj"]].fillna(0.0)
    prov = pd.read_csv(os.path.join(EXT, "expansion_intensity_real.csv")); prov["flow_per_worker"] = prov.predict_flow_nomig / prov.employ2005
    canon = pd.read_csv(os.path.join(EXT, "expansion_intensity_canonical.csv"))[["provcd", "int_placebo_log_1998_1997"]]
    stock = pd.read_csv(os.path.join(EXT, "census_hs_stock.csv")).rename(columns={"provcd": "prov_merge"})
    coh = pd.read_csv(os.path.join(EXT, "cohort_exposure.csv")).rename(columns={"birth_year_age18": "birth_year"})[["birth_year", "exposure_ratio"]]
    pre = float(coh[coh.birth_year <= 1980].exposure_ratio.mean())        # cohorts born before the series starts: the pre-reform mean
    coh = pd.concat([pd.DataFrame({"birth_year": range(1930, int(coh.birth_year.min())), "exposure_ratio": pre}), coh])
    trade = pd.read_csv(os.path.join(EXT, "expgdp2000.csv"))[["provcd", "expgdp"]]
    c["prov_merge"] = c.provcd.replace({50: 51})
    c = (c.merge(prov[["provcd", "pro_predict_growth", "flow_per_worker", "log_growth_implied"]], on="provcd", how="left")
          .merge(canon, on="provcd", how="left").merge(stock[["prov_merge", "hs2000"]], on="prov_merge", how="left")
          .merge(coh, on="birth_year", how="left").merge(trade, on="provcd", how="left"))
    c = c.dropna(subset=["pro_predict_growth", "exposure_ratio", "eduy_max"]).reset_index(drop=True)
    c["college"] = (c.eduy_max >= 16).astype(float); c["by"] = c.birth_year.astype(int); c["pc"] = c.provcd.astype(int)
    c["Z"] = c.pro_predict_growth * c.exposure_ratio
    c["coastal"] = c.provcd.isin({11, 12, 13, 21, 31, 32, 33, 35, 37, 44, 45, 46}).astype(float)
    return c


def _net_of_year(d, col="log_income"):
    """The column net of survey-year effects (regression on survey-year indicators over the rows of d), recentered at its mean."""
    Y = pd.get_dummies(d.year.astype(int), drop_first=True).astype(float).assign(const=1.0).values
    v = d[col].astype(float).values
    return v - Y @ np.linalg.lstsq(Y, v, rcond=None)[0] + v.mean()


def build_sample(lo=LO, hi=HI, min_waves=MIN_WAVES, level_from=LEVEL_FROM):
    d = _waves(lo, hi, min_waves); c = _children(d)
    if level_from is None:
        return c
    d = d[d.pid.isin(c.pid)].copy(); d["yr"] = _net_of_year(d)
    o = d[d.age >= level_from]
    lv = o.groupby("pid").agg(log_y=("yr", "mean"), age_first=("age", "min"), waves=("year", "size"))
    lv = lv[lv.waves >= OLD_MIN]
    c = c[c.pid.isin(lv.index)].drop(columns=["log_y", "age_first", "waves"]).merge(lv, left_on="pid", right_index=True)
    return c.reset_index(drop=True)


def child_waves(lo=LO, hi=HI, min_waves=MIN_WAVES, level_from=LEVEL_FROM):
    """Child-wave rows (log labor income, age, year and the fixed characteristics) of the children of build_sample()."""
    c = build_sample(lo, hi, min_waves, level_from); d = _waves(lo, hi, min_waves)
    d = d[d.pid.isin(c.pid)]
    if level_from is not None:
        d = d[d.age >= level_from]
    return d.reset_index(drop=True)


def process_parameters(group="pooled"):
    """Permanent variance, transitory innovation variance and transitory persistence of the Section 3 earnings process
    (minimum distance, income_dynamics/output/tables/m7_md_cellw_boot.csv)."""
    m7 = pd.read_csv(os.path.join(M, "output", "tables", "m7_md_cellw_boot.csv")).set_index("group").loc[group]
    return float(m7.eta2), float(m7.eps2), float(m7.rho)


def two_row(A=LEVEL_FROM, weights="none", outcome="log_income", c=None, d=None, old_min=OLD_MIN, young_min=YOUNG_MIN):
    """The rows of the Section 6 design: for each child of c, the mean of the outcome net of survey-year effects over
    the child's waves below age A and over the waves at A and over. The row at A and over is kept if it averages at
    least old_min waves, the row below A if it averages at least young_min waves.
    Columns: the child-level variables of build_sample(level_from=None); old (1 for the row at A and over); n (waves
    behind the mean); y (the mean); w (weight). weights='none' gives every row weight one; weights='process' gives the
    precision implied by the pooled earnings process of Section 3: a mean of n waves has variance
    sigma_eta^2 + sigma_v^2 / n with sigma_v^2 = sigma_eps^2 / (1 - rho^2), and the weight is its inverse."""
    old_min = OLD_MIN if old_min is None else old_min; young_min = YOUNG_MIN if young_min is None else young_min
    c = build_sample(level_from=None) if c is None else c
    d = _waves() if d is None else d
    d = d[d.pid.isin(c.pid)].copy()
    d["yr"] = _net_of_year(d, outcome); d["old"] = (d.age >= A).astype(int)
    s = d.groupby(["pid", "old"]).agg(y=("yr", "mean"), n=("yr", "size")).reset_index()
    s = s[((s.old == 1) & (s.n >= old_min)) | ((s.old == 0) & (s.n >= young_min))]
    s = s.merge(c.drop(columns=["log_y", "waves"]), on="pid").reset_index(drop=True)
    if weights == "process":
        eta2, eps2, rho = process_parameters(); s["w"] = 1.0 / (eta2 + eps2 / (1 - rho ** 2) / s.n)
    else:
        s["w"] = 1.0
    return s
