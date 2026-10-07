"""The sample of the paper (Section 2): means, counts and the facts quoted in the text.

One sample (sample_mature_age.py): children aged 22 to 55 with at least two waves of positive labor income in 2014
to 2022. Three groups of children are described, each defined by what an analysis needs:
  'all children'            every child of the sample
  'three waves'             the children with at least three waves (Sections 3 and 4)
  'level at 30 and over'    the children with at least two waves at ages 30 and over, for whom the level of earnings
                            is measured (Section 5; the rows at 30 and over of Sections 6 and 7)

  A  means and standard deviations by family background for all children and for the children with a level at 30 and
     over (Table 2); for the latter, log labor income is the level of earnings at 30 and over (net of survey-year
     effects) and age and waves refer to the waves at 30 and over
  B  facts: counts, child-waves, birth years, provinces; children with a row below 30 and with both rows; children
     born before 1964; among children with a retrospective report observed at ages 22 to 55, the share who enter
     the sample and the three-wave group and the share of person-waves with positive labor income, by family
     background; agreement of the retrospective indicator with the indicator from the parents' current occupations;
     birth years of the linked parents; child-waves on the household roster of a linked parent; children with a
     report for one parent only (children with three waves; all children); province at age twelve against the province
     assigned, by family background; survey-year means of log income at 30 and over
  C  first stages at the child level on all children: each intensity measure, the placebo, and by family background
     in the specification of u12_returns_by_family_background.py (wild cluster bootstrap, 9,999 draws). These include
     the children observed only below 30 and are not the first stages of the returns of Section 6
  D  least squares return to college by age band on the child-waves of all children (family background, parental
     income, gender, urban residence, province, survey-year and age effects; standard errors clustered on child)
The rows of B that read the survey files are copied from the existing output if the files are not present.
Outputs: output/u14_A_sample_means.csv, u14_B_sample_facts.csv, u14_C_first_stages_all_children.csv, u14_D_return_by_age.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..")); OUT = os.path.join(ROOT, "output"); sys.path.insert(0, HERE)
REPS = int(os.environ.get("U14_REPS", "9999")); os.environ["U9_REPS"] = str(REPS)
src = open(os.path.join(HERE, "u9_mature_age_earnings.py"), encoding="utf-8").read(); src = src[:src.index("# ------------------------------------------------------------------ the children, the rows and the two-row design")]
u9 = {"__file__": os.path.join(HERE, "u9_mature_age_earnings.py"), "__name__": "u9"}; exec(compile(src, "u9", "exec"), u9)
Design, X_fe, cl, dummies, INTENSITIES = u9["Design"], u9["X_fe"], u9["cl"], u9["dummies"], u9["INTENSITIES"]
from _locate import sibling
from sample_mature_age import build_sample, child_waves, prefilter_frame, two_row, DYNAMICS, LO, HI, MIN_WAVES, LEVEL_FROM
M = sibling("income_dynamics", __file__); P = sibling("college_expansion", __file__)
RAW = next((d for d in [os.path.join(M, "data", "raw"), os.path.join(P, "data", "raw")] if os.path.isdir(os.path.join(d, "CFPS2012"))), None)
ALL, THREE, LEVEL = "all children", "three waves", "level at 30 and over"
S = {ALL: (build_sample(level_from=None), child_waves(level_from=None)), THREE: (build_sample(*DYNAMICS, level_from=None), child_waves(*DYNAMICS, level_from=None)),
     LEVEL: (build_sample(), child_waves())}
ca, wa = S[ALL]; c3, w3 = S[THREE]; cl_, wl = S[LEVEL]

# ---------------------------------------------------------------- A. means by family background
A = []
for name in (ALL, LEVEL):
    c, w = S[name]
    k = c.merge(w.groupby("pid").age.mean().rename("age_mean"), left_on="pid", right_index=True)
    k = k.merge(w.sort_values("year").groupby("pid").urban.first().rename("urban_first"), left_on="pid", right_index=True)
    for var, col in [("Mean log labor income", "log_y"), ("Log family income of the parental household", "ParentInc"), ("Years of schooling", "eduy_max"),
                     ("College (16 or more years)", "college"), ("Female", "female"), ("Urban (first wave)", "urban_first"), ("Age over the waves", "age_mean"),
                     ("Birth year", "birth_year"), ("Waves with labor income", "waves")]:
        a, o = k.loc[k.HP14 == 1, col], k.loc[k.HP14 == 0, col]
        A.append(dict(sample=name, variable=var, advantaged_mean=a.mean(), advantaged_sd=a.std(), less_advantaged_mean=o.mean(), less_advantaged_sd=o.std()))
pd.DataFrame(A).to_csv(os.path.join(OUT, "u14_A_sample_means.csv"), index=False)

# ---------------------------------------------------------------- B. facts
B = []
for name, (c, w) in S.items():
    B += [dict(item=f"{name}: children", value=len(c)), dict(item=f"{name}: advantaged children", value=int(c.HP14.sum())), dict(item=f"{name}: child-waves", value=len(w)),
          dict(item=f"{name}: waves per child", value=len(w) / len(c)), dict(item=f"{name}: first birth year", value=int(c.birth_year.min())),
          dict(item=f"{name}: last birth year", value=int(c.birth_year.max())), dict(item=f"{name}: provinces", value=c.provcd.nunique()),
          dict(item=f"{name}: share advantaged", value=c.HP14.mean()), dict(item=f"{name}: raw gap in mean log earnings", value=c[c.HP14 == 1].log_y.mean() - c[c.HP14 == 0].log_y.mean()),
          dict(item=f"{name}: college completion, advantaged", value=c[c.HP14 == 1].college.mean()), dict(item=f"{name}: college completion, less advantaged", value=c[c.HP14 == 0].college.mean()),
          dict(item=f"{name}: mean age at the first wave", value=float(w.groupby("pid").age.min().mean()))]
rows = two_row(c=ca); below = set(rows[rows.old == 0].pid); above = set(rows[rows.old == 1].pid)
B += [dict(item="children with a row at 30 and over", value=len(above)), dict(item="children with a row below 30", value=len(below)), dict(item="children with both rows", value=len(above & below)),
      dict(item="children with a row below 30 only", value=len(below - above)), dict(item="children with three waves and a level at 30 and over", value=int(c3.pid.isin(cl_.pid).sum())),
      dict(item="first birth year among the rows below 30", value=int(rows[rows.old == 0].by.min())), dict(item="last birth year among the rows at 30 and over", value=int(rows[rows.old == 1].by.max())),
      dict(item="children born before 1964", value=int((ca.by < 1964).sum())), dict(item="rows at 30 and over of children born before 1964", value=int(((rows.old == 1) & (rows.by < 1964)).sum())),
      dict(item="rows at 30 and over of children born 1981 or later", value=int(((rows.old == 1) & (rows.by >= 1981)).sum()))]
for y_, v_ in wl.groupby("year").log_income.mean().items(): B.append(dict(item=f"mean log income at 30 and over, survey year {int(y_)}", value=float(v_)))
# who enters the sample, and who has earnings, among children with a retrospective report (same provinces, family income observed)
df = prefilter_frame(); prov_ok = set(ca.provcd.unique())
e = df[(df.year >= 2014) & df.age.between(LO, HI) & df.HighParentOcc14.notna() & df.ParentInc_bar.notna() & df.provcd.isin(prov_ok)].copy()
kid = e.groupby("pid").HighParentOcc14.first()
for h, lab in [(1, "advantaged"), (0, "less advantaged")]:
    ids = kid[kid == h].index; rows_h = e[e.pid.isin(ids)]
    B += [dict(item=f"children with a retrospective report observed at ages {LO} to {HI} in 2014-2022, {lab}", value=len(ids)),
          dict(item=f"share of them in the sample, {lab}", value=float(pd.Index(ids).isin(ca.pid).mean())),
          dict(item=f"share of them with three waves, {lab}", value=float(pd.Index(ids).isin(c3.pid).mean())),
          dict(item=f"share of their person-waves with positive labor income, {lab}", value=float((rows_h.income > 0).mean()))]
# residual waves (income-dynamics pipeline)
res = pd.read_pickle(os.path.join(M, "data", "intermediate", "panel_resid.pkl")).dropna(subset=["HighParentOcc14"])
nres = res.dropna(subset=["y_tilde_w"]).groupby("pid").size().reindex(c3.pid).fillna(0)
B += [dict(item="children with three waves and fewer than three residual waves", value=int((nres < 3).sum())), dict(item="residual person-waves of the children with three waves", value=int(res.y_tilde_w.notna().sum()))]
# agreement of the two indicators
for name, w in [(ALL, wa), (THREE, w3), (LEVEL, wl)]:
    one = w.dropna(subset=["HighParentOcc14", "HighParentOcc"]).drop_duplicates("pid")
    B += [dict(item=f"{name}: children with both indicators", value=len(one)), dict(item=f"{name}: indicators agree", value=float((one.HighParentOcc14 == one.HighParentOcc).mean())),
          dict(item=f"{name}: advantaged by the retrospective indicator only", value=float(((one.HighParentOcc14 == 1) & (one.HighParentOcc == 0)).mean())),
          dict(item=f"{name}: advantaged by the current-occupation indicator only", value=float(((one.HighParentOcc14 == 0) & (one.HighParentOcc == 1)).mean()))]
# birth years of the linked parents (long panel of the college-expansion pipeline; a parent's birth year is the median of survey year minus age)
_long = pd.read_pickle(os.path.join(P, "data", "intermediate", "_panel_long_cache.pkl")); _by = (_long.year - _long.age).groupby(_long.pid).median()
_par = pd.concat([wa.pid_father, wa.pid_mother]).dropna().unique(); _pby = _by.reindex(_par).dropna()
B += [dict(item="linked parents with a birth year", value=len(_pby)), dict(item="parents' birth year, 5th percentile", value=float(_pby.quantile(0.05))),
      dict(item="parents' birth year, median", value=float(_pby.median())), dict(item="parents' birth year, 95th percentile", value=float(_pby.quantile(0.95)))]
NEED_RAW = [f"{ALL}: share of child-waves on the household roster of a linked parent", f"{THREE}: share of child-waves on the household roster of a linked parent",
            f"{LEVEL}: share of child-waves on the household roster of a linked parent", "children with three waves and a retrospective report for one parent only",
            "all children: retrospective report for one parent only", "all children: province at age twelve recorded in the 2010 wave",
            "all children: province at age twelve differs from the province assigned", "share with a different province at age twelve, advantaged",
            "share with a different province at age twelve, less advantaged"]
if RAW is None:
    old = pd.read_csv(os.path.join(OUT, "u14_B_sample_facts.csv")); B += old[old["item"].isin(NEED_RAW)].to_dict("records")
else:
    import pyreadstat
    CONF = [(2014, "CFPS2014/ecfps2014famconf_170630.dta", "fid14"), (2016, "CFPS2016/ecfps2016famconf_201804.dta", "fid16"), (2018, "CFPS2018/ecfps2018famconf_202008.dta", "fid18"),
            (2020, "CFPS2020/ecfps2020famconf_202306.dta", "fid20"), (2022, "CFPS2022/ecfps2022famconf_202410.dta", "fid22")]
    fr = []
    for year, f, fid in CONF:
        q, _ = pyreadstat.read_dta(os.path.join(RAW, f), usecols=["pid", fid]); q = q.rename(columns={fid: "fid"})
        q["pid"] = pd.to_numeric(q.pid, errors="coerce"); q["fid"] = pd.to_numeric(q.fid, errors="coerce"); q = q.dropna().astype({"pid": np.int64}); q["year"] = year; fr.append(q)
    conf = pd.concat(fr).drop_duplicates(["pid", "year"]).set_index(["pid", "year"]).fid
    for (name, (c, w)), key in zip(S.items(), NEED_RAW[:3]):
        x = w[["pid", "year", "pid_father", "pid_mother"]].copy()
        own = pd.Series(conf.reindex(pd.MultiIndex.from_frame(x[["pid", "year"]])).values, index=x.index)
        same = np.zeros(len(x), bool)
        for par in ("pid_father", "pid_mother"):
            ok = x[par].notna(); idx = pd.MultiIndex.from_arrays([x.loc[ok, par].astype(np.int64), x.loc[ok, "year"]])
            pf = pd.Series(conf.reindex(idx).values, index=x.index[ok])
            same[ok.values] |= (pf.notna() & own[ok].notna() & (pf == own[ok])).values
        B.append(dict(item=key, value=float(same.mean())))
    s8 = open(os.path.join(HERE, "u8_premium_specifications.py"), encoding="utf-8").read(); s8 = s8[:s8.index("from sample_mature_age import build_sample, child_waves")]
    u8 = {"__file__": os.path.join(HERE, "u8_premium_specifications.py"), "__name__": "u8"}; exec(compile(s8, "u8", "exec"), u8)
    rep = u8["retrospective_reports"](); rep = rep[rep.pid.isin(c3.pid)]
    has_f = rep.assign(v=rep.occ_f.where(rep.occ_f >= 0).notna()).groupby("pid").v.max(); has_m = rep.assign(v=rep.occ_m.where(rep.occ_m >= 0).notna()).groupby("pid").v.max()
    B.append(dict(item=NEED_RAW[3], value=int((has_f.reindex(c3.pid).fillna(False) ^ has_m.reindex(c3.pid).fillna(False)).sum())))
    rep = u8["retrospective_reports"](); rep = rep[rep.pid.isin(ca.pid)]
    has_f = rep.assign(v=rep.occ_f.where(rep.occ_f >= 0).notna()).groupby("pid").v.max(); has_m = rep.assign(v=rep.occ_m.where(rep.occ_m >= 0).notna()).groupby("pid").v.max()
    B.append(dict(item=NEED_RAW[4], value=int((has_f.reindex(ca.pid).fillna(False) ^ has_m.reindex(ca.pid).fillna(False)).sum())))
    # province of residence at age twelve (2010 adult file), as in block C of u9_mature_age_earnings.py
    a12, _ = pyreadstat.read_dta(os.path.join(RAW, "CFPS2010", "ecfps2010adult_201906.dta"), usecols=["pid", "qa4", "qa401acode", "qa102acode"]); a12 = a12.dropna(subset=["pid"]); a12["pid"] = a12.pid.astype(np.int64)
    num = lambda v: pd.to_numeric(v, errors="coerce").where(lambda x: x > 0); p12 = num(a12.qa401acode).where(num(a12.qa401acode).notna(), num(a12.qa102acode).where(num(a12.qa4) == 1))
    p12 = pd.Series(p12.values, index=a12.pid).loc[lambda x: x.between(11, 82)]; p12 = p12[~p12.index.duplicated()]; pp = ca.pid.map(p12); diff = (pp.notna() & (pp != ca.pc))
    B += [dict(item=NEED_RAW[5], value=int(pp.notna().sum())), dict(item=NEED_RAW[6], value=int(diff.sum())),
          dict(item=NEED_RAW[7], value=float(diff[pp.notna() & (ca.HP14 == 1)].mean())), dict(item=NEED_RAW[8], value=float(diff[pp.notna() & (ca.HP14 == 0)].mean()))]
pd.DataFrame(B).to_csv(os.path.join(OUT, "u14_B_sample_facts.csv"), index=False); print(pd.DataFrame(B).round(4).to_string(index=False))

# ---------------------------------------------------------------- C. first stages at the child level on all children
rng = np.random.default_rng(20261014); c = ca.reset_index(drop=True); C = []
for col, lab in INTENSITIES:
    d = c.dropna(subset=[col]).reset_index(drop=True); D = Design(d, (d[col] * d.exposure_ratio).values, X_fe(d), rng); f = D.rf(d.college.values, wild=(col == "pro_predict_growth"))
    C.append(dict(row=f"all children: {lab}", children=len(d), fs=f["coef"], fs_se=f["se"], F=f["t"] ** 2, fs_p_wild=f["p_wild"]))
Dp = Design(c, (c.int_placebo_log_1998_1997 * c.exposure_ratio).values, X_fe(c), rng); f = Dp.rf(c.college.values, wild=False)
C.append(dict(row="all children: placebo, registrations 1998/1997", children=len(c), fs=f["coef"], fs_se=f["se"], F=f["t"] ** 2))
H = c.HP14.values.astype(float)
Xg = pd.concat([c[["ParentInc", "female", "urban"]].astype(float), dummies(c.pc, "p"), dummies(c.by, "b"), c[["HP14"]].astype(float), dummies(c.by, "hb").multiply(c.HP14, axis=0)], axis=1)
Xg = Xg.loc[:, Xg.std() > 0].assign(const=1.0).values; Zg = {"advantaged": c.Z.values * H, "less advantaged": c.Z.values * (1 - H)}
for grp, other in [("less advantaged", "advantaged"), ("advantaged", "less advantaged")]:
    D = Design(c, Zg[grp], np.column_stack([Zg[other], Xg]), rng); f = D.rf(c.college.values)
    C.append(dict(row=f"{grp} children, instrument entering once for each group", children=int((H == (grp == "advantaged")).sum()), fs=f["coef"], fs_se=f["se"], F=f["t"] ** 2, fs_p_wild=f["p_wild"]))
pd.DataFrame(C).to_csv(os.path.join(OUT, "u14_C_first_stages_all_children.csv"), index=False); print(pd.DataFrame(C).round(4).to_string(index=False))

# ---------------------------------------------------------------- D. least squares return to college by age band
w = wa.merge(ca[["pid", "college", "pc"]], on="pid"); w["a"] = w.age.astype(int); Dr = []
for lo_, hi_ in [(22, 25), (26, 29), (22, 29), (30, 34), (35, HI), (30, HI)]:
    d = w[(w.a >= lo_) & (w.a <= hi_)]
    X = pd.concat([d[["college", "HighParentOcc14", "ParentInc_bar", "female", "urban"]].astype(float), dummies(d.pc, "p"), dummies(d.year, "y"), dummies(d.a, "a")], axis=1).assign(const=1.0)
    r = sm.OLS(d.log_income.values.astype(float), X.values.astype(float)).fit(cov_type="cluster", cov_kwds={"groups": d.pid.values})
    Dr.append(dict(ages=f"{lo_}-{hi_}", coef=r.params[0], se=r.bse[0], child_waves=len(d), children=d.pid.nunique()))
pd.DataFrame(Dr).to_csv(os.path.join(OUT, "u14_D_return_by_age.csv"), index=False); print(pd.DataFrame(Dr).round(3).to_string(index=False))
