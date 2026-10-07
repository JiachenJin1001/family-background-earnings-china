"""The premium across specifications (Section 5 and Figure 2 of the paper).

Children: those of the paper's sample (sample_mature_age.py: ages 22 to 55, at
least two waves of positive labor income in 2014 to 2022, a linked parent, a
retrospective report of parental occupation, the family income of the parental
household, a province with an expansion intensity) who have at least two waves
at ages 30 and over, where the level of earnings is measured.

Every row is the coefficient on the indicator for advantaged children (at
least one parent in CSCO major group 1 or 2 when the child was fourteen) in a
regression of the child's log annual labor income that holds parental income
fixed. No specification controls for parental education.

Panel rows (correlated random effects): child-wave observations at ages 30 to
55, waves 2014 to 2022. Controls: parental income, age, urban residence, the
within-child means of age and urban residence, gender, survey-year effects,
province effects. Standard errors clustered on child. Each row changes one
element:
  the income control (none; family income of the parental household; the
      parents' own labor income; the family income of the 2010 and 2012 waves
      only, which precede the earnings window; the family income of the waves
      in which the child was at most 40 or at most 35 years old),
  the dependent variable (income deflated by the provincial consumer price
      index),
  the indicator (most frequent retrospective report; CSCO major groups 1 to
      3; CSCO major group 1 only; the occupations the parents hold during
      2010 to 2022),
  the geographic effects (three regions in place of provinces),
  the sample (children recorded as urban in most of their waves; children
      recorded as rural in most of their waves).
Cross-section rows: one observation per child (the mean, across the child's
waves at 30 and over, of log income net of survey-year effects, as in the
entropy balancing and in Sections 6 and 7), controls gender, urban residence, age at first
observation and its square, province effects, parental income;
heteroskedasticity-robust standard errors. One row adds parental party
membership when the child was fourteen, one adds the parents' state-sector
employment in 2012 (children with a parent working in 2012).
Entropy balancing (Hainmueller 2012): less advantaged children reweighted to
the means of advantaged children in parental income and age at first
observation (and their variances), gender, urban residence, five-year birth
cohorts and province (the fifteen largest provinces separately); the premium
and the gap in college completion under the same weights, with percentile
intervals from 200 bootstrap draws of children within group.

The rows that read the survey files (folder data/raw) use the retrospective
questions, the 2012 work-unit question, or the family income of each wave. If
the survey files are not present, those rows are copied from the existing
output file.

Outputs: output/u8_premium_specifications.csv, output/u8_indicator_agreement.csv (and the same
agreement on the analysis sample of Sections 3 and 4, output/u8_indicator_agreement_analysis_sample.csv),
output/u8_income_measure_waves.csv (number of waves behind the family income
measure and behind the parents' own labor income, per linked parent),
output/u8_entropy_balancing.csv, output/u8_level_sample.csv (the sample: counts,
means by family background, families and siblings)
"""
import os, sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from _locate import sibling
P = sibling("college_expansion", __file__)
M = sibling("income_dynamics", __file__)
INT = os.path.join(P, "data", "intermediate")
OUT = os.path.join(ROOT, "output", "u8_premium_specifications.csv")
RAW = next((d for d in [os.path.join(M, "data", "raw"), os.path.join(P, "data", "raw"),
                        os.path.join(os.path.dirname(P), "data", "raw")]
            if os.path.isdir(os.path.join(d, "CFPS2012"))), None)

REGION = {"East": [11, 12, 13, 21, 31, 32, 33, 35, 37, 44, 46],
          "Central": [14, 22, 23, 34, 36, 41, 42, 43],
          "West": [15, 45, 50, 51, 52, 53, 54, 61, 62, 63, 64, 65]}
PROV_TO_REGION = {p: r for r, ps in REGION.items() for p in ps}


RETROSPECTIVE = [("CFPS2012", "ecfps2012adult_202505.dta", "qv103code_best", "qv203code_best", "qv104", "qv204"),
                 ("CFPS2020", "ecfps2020person_202306.dta", "qv103code", "qv203code", "qv104", "qv204"),
                 ("CFPS2022", "ecfps2022person_202410.dta", "qv103code", "qv203code", "qv104", "qv204")]


def retrospective_reports():
    """One row per respondent and wave: occupation codes and political status of father and mother
    when the respondent was fourteen (2012, 2020 and 2022 waves)."""
    import pyreadstat
    out = []
    for folder, name, occ_f, occ_m, pol_f, pol_m in RETROSPECTIVE:
        df, _ = pyreadstat.read_dta(os.path.join(RAW, folder, name), usecols=["pid", occ_f, occ_m, pol_f, pol_m])
        df = df.rename(columns={occ_f: "occ_f", occ_m: "occ_m", pol_f: "pol_f", pol_m: "pol_m"})
        out.append(df.assign(wave=int(folder[-4:])))
    return pd.concat(out, ignore_index=True)


def occupation_indicator(reports, groups, rule="any"):
    """Indicator by child from the first digit of the parents' occupation codes.
    rule "any": one if any report places a parent in the groups (the paper's rule);
    rule "most": one if more reports place a parent in the groups than outside them."""
    q = reports.copy()
    for s in ("f", "m"):
        code = q["occ_" + s].where(q["occ_" + s] >= 0)
        digit = code.apply(lambda x: np.nan if pd.isna(x) else int(str(int(x))[0]))
        q["high_" + s] = np.where(digit.isna(), np.nan, digit.isin(groups).astype(float))
    q["high"] = q[["high_f", "high_m"]].max(axis=1)
    g = q.groupby("pid")["high"]
    n1, n0, n = g.apply(lambda s: (s == 1).sum()), g.apply(lambda s: (s == 0).sum()), g.count()
    ind = (n1 >= 1) if rule == "any" else (n1 > n0)
    return ind.astype(float).where(n > 0)


def party_membership(reports):
    """One if either parent was a member of the Communist Party, a democratic party or the Youth League."""
    q = reports.copy()
    for s in ("f", "m"):
        v = q["pol_" + s].where(q["pol_" + s] > 0)
        q["party_" + s] = np.where(v.isna(), np.nan, v.isin([1, 2, 3]).astype(float))
    q["party"] = q[["party_f", "party_m"]].max(axis=1)
    return q.groupby("pid")["party"].max()


FAMILY_INCOME = [(2010, "CFPS2010/ecfps2010famecon_201906.dta", "CFPS2010/ecfps2010famconf_nat072016.dta", "fid", "faminc_net"),
                 (2012, "CFPS2012/ecfps2012famecon_201906.dta", "CFPS2012/ecfps2012famconf_092015.dta", "fid12", "fincome2"),
                 (2014, "CFPS2014/ecfps2014famecon_201906.dta", "CFPS2014/ecfps2014famconf_170630.dta", "fid14", "fincome2"),
                 (2016, "CFPS2016/ecfps2016famecon_201807.dta", "CFPS2016/ecfps2016famconf_201804.dta", "fid16", "fincome2"),
                 (2018, "CFPS2018/ecfps2018famecon_202101.dta", "CFPS2018/ecfps2018famconf_202008.dta", "fid18", "fincome2"),
                 (2020, "CFPS2020/ecfps2020famecon_202306.dta", "CFPS2020/ecfps2020famconf_202306.dta", "fid20", "finc"),
                 (2022, "CFPS2022/ecfps2022famecon_202410.dta", "CFPS2022/ecfps2022famconf_202410.dta", "fid22", "finc")]


def family_income_by_wave():
    """One row per person and wave: log income of the family the person belongs to in that wave, from the
    same files and variables as the family income measure of the paper
    (income_dynamics/code/02_sample_and_parents.py), which averages these rows over all waves."""
    import pyreadstat
    out = []
    for year, econ_file, conf_file, fid, var in FAMILY_INCOME:
        econ, _ = pyreadstat.read_dta(os.path.join(RAW, econ_file), usecols=[fid, var])
        econ = econ.rename(columns={fid: "fid", var: "family_income"})
        econ["fid"] = pd.to_numeric(econ["fid"], errors="coerce")
        econ = econ[econ["family_income"].notna() & (econ["family_income"] > 0)]
        conf, _ = pyreadstat.read_dta(os.path.join(RAW, conf_file), usecols=["pid", fid])
        conf = conf.rename(columns={fid: "fid"})
        conf["pid"] = pd.to_numeric(conf["pid"], errors="coerce"); conf["fid"] = pd.to_numeric(conf["fid"], errors="coerce")
        conf = conf.dropna(subset=["pid", "fid"]).astype({"pid": np.int64})
        m = conf.merge(econ, on="fid", how="inner")
        out.append(pd.DataFrame(dict(pid=m["pid"].values, year=year, log_family_income=np.log(m["family_income"].values))))
    return pd.concat(out, ignore_index=True).drop_duplicates(["pid", "year"])


def state_sector_2012():
    """One if the person's work unit in 2012 was government, party or military, a public institution, or a
    state-owned enterprise; zero for any other work unit; missing if not working."""
    import pyreadstat
    q, _ = pyreadstat.read_dta(os.path.join(RAW, "CFPS2012", "ecfps2012adult_202505.dta"), usecols=["pid", "qg703"])
    unit = pd.to_numeric(q["qg703"], errors="coerce"); unit = unit.where(unit > 0)
    q["state"] = np.where(unit.isin([1, 2, 3]), 1.0, np.where(unit.notna(), 0.0, np.nan))
    return q.assign(pid=q["pid"].astype(np.int64)).set_index("pid")["state"]


def cre(panel, label, family, treat="HighParentOcc14", inc="ParentInc_bar", dep="log_income",
        geography="province", source=""):
    df = panel.dropna(subset=[dep, treat, inc, "provcd", "year", "age", "urban", "female"]).copy()
    for v in ["age", "urban"]:
        df[v + "_bar"] = df.groupby("pid")[v].transform("mean")
    if geography == "province":
        geo = pd.get_dummies(df["provcd"].astype(int), prefix="prv", drop_first=True).astype(float)
    else:
        geo = pd.get_dummies(df["provcd"].map(PROV_TO_REGION).fillna("Other"), prefix="reg", drop_first=True).astype(float)
    yr = pd.get_dummies(df["year"].astype(int), prefix="yr", drop_first=True).astype(float)
    X = pd.concat([df[[treat, inc, "age", "urban", "age_bar", "urban_bar", "female"]].astype(float), yr, geo], axis=1)
    X["const"] = 1.0
    m = sm.OLS(df[dep].astype(float).values, X.values).fit(cov_type="cluster", cov_kwds={"groups": df["pid"].values})
    j = X.columns.get_loc(treat); k = X.columns.get_loc(inc)
    one = df.drop_duplicates("pid")
    return dict(specification=label, estimator=family, estimate=m.params[j], se=m.bse[j], p=m.pvalues[j],
                lo=m.params[j] - 1.96 * m.bse[j], hi=m.params[j] + 1.96 * m.bse[j],
                observations=int(m.nobs), children=int(one.shape[0]), advantaged=int((one[treat] == 1).sum()),
                source=source, income_coef=m.params[k], income_se=m.bse[k], income_p=m.pvalues[k])


def cross_section(est, cols, label, report, source):
    prov = pd.get_dummies(est["provcd"].astype(int), prefix="p", drop_first=True).astype(float)
    base = pd.concat([est[["female", "urban", "age_first"]].astype(float), prov], axis=1)
    base["age2"] = est["age_first"].astype(float) ** 2; base["const"] = 1.0
    X = pd.concat([est[cols].astype(float), base], axis=1)
    m = sm.OLS(est["log_y"].astype(float), X).fit(cov_type="HC1")
    return dict(specification=label, estimator="Cross-section, one observation per child",
                estimate=m.params[report], se=m.bse[report], p=m.pvalues[report],
                lo=m.params[report] - 1.96 * m.bse[report], hi=m.params[report] + 1.96 * m.bse[report],
                observations=int(len(est)), children=int(len(est)), advantaged=int((est["HP14"] == 1).sum()),
                source=source)


from sample_mature_age import build_sample, child_waves
kid = build_sample()                      # one row per child
panel = child_waves()                     # child-wave rows of the same children
panel["has_qv14"] = 1
pl = pd.read_pickle(os.path.join(INT, "_panel_long_cache.pkl"))
rows = []

# --- income control
rows.append(cre(panel, "baseline: family income of the parental household", "Correlated random effects"))
# the parents' own labor income: mean log positive labor income of each linked parent over the waves in which it is
# observed, averaged over the two parents
parent_labor = np.log(pl["income"].where(pl["income"] > 0)).groupby(pl["pid"]).mean()
lab = panel.copy()
lab["ParentInc_labor"] = pd.concat([lab["pid_father"].map(parent_labor), lab["pid_mother"].map(parent_labor)], axis=1).mean(axis=1)
rows.append(cre(lab, "parents' own labor income as the income control", "Correlated random effects", inc="ParentInc_labor"))
panel["no_income_control"] = 0.0
r0 = cre(panel.assign(ParentInc_none=panel["ParentInc_bar"] * 0.0), "no income control (not in Figure 2)", "Correlated random effects", inc="ParentInc_none")
r0.update(income_coef=np.nan, income_se=np.nan, income_p=np.nan); rows.append(r0)

# --- income deflated by the provincial consumer price index
cpi = pd.read_csv(os.path.join(P, "data", "external", "cpi_province_year.csv"))
cpi = cpi.melt(id_vars=["provcd", "province_name"], var_name="v", value_name="cpi")
cpi["year"] = cpi["v"].str.replace("cpi_", "").astype(int)
d = panel.merge(cpi[["provcd", "year", "cpi"]], on=["provcd", "year"], how="left")
d["log_real_income"] = np.log((d["income"] / (d["cpi"] / 100.0)).where(d["income"] > 0))
rows.append(cre(d, "income deflated by the provincial consumer price index", "Correlated random effects", dep="log_real_income"))

# --- geographic effects, subsamples, indicator from the parents' current occupations
rows.append(cre(panel, "three regions in place of province effects", "Correlated random effects", geography="region"))
share = panel.groupby("pid")["urban"].transform("mean")
rows.append(cre(panel[share > 0.5], "children recorded as urban in most of their waves", "Correlated random effects"))
rows.append(cre(panel[share <= 0.5], "children recorded as rural in most of their waves", "Correlated random effects"))
rows.append(cre(panel[panel["HighParentOcc14"].notna()],
                "indicator from the occupations the parents hold during 2010 to 2022 (not in Figure 2)",
                "Correlated random effects", treat="HighParentOcc"))

# --- the family income of the parental household from a restricted set of waves (needs the survey files):
#     the 2010 and 2012 waves only, which precede the 2014 to 2022 earnings window, and the waves in which the child
#     was at most 40 or at most 35 years old (child age in a wave is the survey year minus the birth year)
kids = panel.groupby("pid").first().reset_index()[["pid", "pid_father", "pid_mother"]]
link = pd.concat([kids.rename(columns={"pid": "child", c: "parent"})[["child", "parent"]].dropna()
                  for c in ("pid_father", "pid_mother")]).drop_duplicates()
WINDOW_ROWS = {"early": "parental income from the 2010 and 2012 waves, before the earnings window",
               40: "parental income from waves with the child aged 40 or less",
               35: "parental income from waves with the child aged 35 or less"}
WAVES = os.path.join(ROOT, "output", "u8_income_measure_waves.csv")

# --- rows that read the survey files
NEED_RAW = ["most frequent retrospective report of parental occupation",
            "indicator widened to CSCO major groups 1 to 3",
            "indicator narrowed to CSCO major group 1",
            "parental party membership added (cross-section)",
            "parental party membership in place of the occupation indicator (not in Figure 2)",
            "parental state-sector employment added (cross-section)",
            "parental state-sector employment alone (not in Figure 2)",
            "state-sector employment coefficient with occupation and income (not in Figure 2)",
            "party membership coefficient with occupation and income (not in Figure 2)",
            "occupation indicator before party membership is added (not in Figure 2)"] + list(WINDOW_ROWS.values())
STATE_OCC_ALONE = "occupation indicator alone, children with a parent working in 2012 (not in Figure 2)"
NEED_RAW.append(STATE_OCC_ALONE)
if RAW is None:
    old = pd.read_csv(OUT)
    rows += old[old.specification.isin(NEED_RAW)].to_dict("records")
    print("survey files not found: rows that need them are copied from", OUT)
else:
    # family income of the parental household by wave, restricted by the age of the child
    fam = family_income_by_wave().merge(link, left_on="pid", right_on="parent")
    birth = panel.assign(b=panel["year"] - panel["age"]).groupby("pid")["b"].median().round().rename("birth_year")
    fam = fam.merge(birth, left_on="child", right_index=True)
    fam["child_age"] = fam["year"] - fam["birth_year"]
    for key, sel in [("early", fam.year <= 2012), (40, fam.child_age <= 40), (35, fam.child_age <= 35)]:
        col = f"ParentInc_{key}"
        w = fam[sel].groupby(["child", "parent"])["log_family_income"].mean().groupby("child").mean().rename(col)
        rows.append(cre(panel.merge(w, left_on="pid", right_index=True, how="left"), WINDOW_ROWS[key],
                        "Correlated random effects", inc=col))
    # number of waves behind each of the two income measures of Table 6, per linked parent, averaged within
    # child and then over the children of each regression
    lab_children = lab.dropna(subset=["log_income", "HighParentOcc14", "ParentInc_labor"])["pid"].unique()
    fam_children = panel.dropna(subset=["log_income", "HighParentOcc14", "ParentInc_bar"])["pid"].unique()
    n_fam = fam.groupby(["child", "parent"]).size().groupby("child").mean()
    n_lab = pl[pl["income"] > 0].merge(link, left_on="pid", right_on="parent").groupby(["child", "parent"]).size().groupby("child").mean()
    pd.DataFrame([dict(measure="family income of the parental household", children=len(fam_children),
                       mean_waves_per_parent=n_fam.reindex(fam_children).mean(), median_waves_per_parent=n_fam.reindex(fam_children).median()),
                  dict(measure="parents' own labor income", children=len(lab_children),
                       mean_waves_per_parent=n_lab.reindex(lab_children).mean(), median_waves_per_parent=n_lab.reindex(lab_children).median())]
                 ).to_csv(WAVES, index=False)

    reports = retrospective_reports()
    modal = occupation_indicator(reports, [1, 2], rule="most").rename("HP14_modal")
    pm = panel.merge(modal, left_on="pid", right_index=True, how="left")
    r = cre(pm, NEED_RAW[0], "Correlated random effects", treat="HP14_modal")
    one = pm.dropna(subset=["HighParentOcc14", "HP14_modal", "ParentInc_bar", "log_income"]).drop_duplicates("pid")
    r["children_coded_differently_from_baseline"] = int((one.HighParentOcc14 != one.HP14_modal).sum())
    rows.append(r)
    for groups, labl in [([1, 2, 3], NEED_RAW[1]), ([1], NEED_RAW[2])]:
        h = occupation_indicator(reports, groups).rename("HP14_alt")
        rows.append(cre(panel.merge(h, left_on="pid", right_index=True, how="left"), labl,
                        "Correlated random effects", treat="HP14_alt"))

    child = panel[panel["has_qv14"] == 1].groupby("pid", as_index=False).agg(
        log_y=("log_income", "mean"), HP14=("HighParentOcc14", "first"), ParentInc=("ParentInc_bar", "first"),
        female=("female", "first"), urban=("urban", "first"), provcd=("provcd", "first"), age_first=("age", "min"),
        pid_father=("pid_father", "first"), pid_mother=("pid_mother", "first"))
    child["log_y"] = child["pid"].map(kid.set_index("pid")["log_y"])          # the level of earnings at 30 and over, net of survey-year effects
    e = child.merge(party_membership(reports).rename("party"), left_on="pid", right_index=True, how="left")
    e = e.dropna(subset=["HP14", "ParentInc", "log_y", "party"])
    rows.append(cross_section(e, ["HP14", "ParentInc"], NEED_RAW[9], "HP14", "children with a report of parental party membership"))
    rows.append(cross_section(e, ["HP14", "ParentInc", "party"], NEED_RAW[3], "HP14", ""))
    rows.append(cross_section(e, ["HP14", "ParentInc", "party"], NEED_RAW[8], "party", ""))
    rows.append(cross_section(e, ["party", "ParentInc"], NEED_RAW[4], "party", ""))
    rows[-1]["party_share_advantaged"] = e.loc[e.HP14 == 1, "party"].mean()
    rows[-1]["party_share_ordinary"] = e.loc[e.HP14 == 0, "party"].mean()

    state = state_sector_2012()
    s = child.copy()
    s["state"] = pd.concat([s["pid_father"].map(state), s["pid_mother"].map(state)], axis=1).max(axis=1)
    share_with_state = s["state"].notna().mean()
    s = s.dropna(subset=["state", "HP14", "log_y", "ParentInc"])
    rows.append(cross_section(s, ["state", "ParentInc"], NEED_RAW[6], "state", "children with a parent working in 2012"))   # parental income controlled, as in every row of the table
    rows[-1]["state_share_advantaged"] = s.loc[s.HP14 == 1, "state"].mean()      # share of children with a parent in the state sector, by family background
    rows[-1]["state_share_ordinary"] = s.loc[s.HP14 == 0, "state"].mean()
    rows.append(cross_section(s, ["HP14", "ParentInc"], STATE_OCC_ALONE, "HP14", "children with a parent working in 2012"))
    rows.append(cross_section(s, ["HP14", "state", "ParentInc"], NEED_RAW[5], "HP14", ""))
    rows[-1]["share_of_children_in_subsample"] = share_with_state
    rows.append(cross_section(s, ["HP14", "state", "ParentInc"], NEED_RAW[7], "state", ""))

# --- agreement of the retrospective indicator with the indicator from current occupations
one = panel.dropna(subset=["HighParentOcc14", "HighParentOcc"]).drop_duplicates("pid")
one = one[one.pid.isin(panel.dropna(subset=["log_income", "HighParentOcc14", "ParentInc_bar"]).pid)]
agreement = dict(children_with_both_indicators=len(one),
                 share_agree=(one.HighParentOcc14 == one.HighParentOcc).mean(),
                 share_retrospective_only=((one.HighParentOcc14 == 1) & (one.HighParentOcc == 0)).mean(),
                 share_current_only=((one.HighParentOcc14 == 0) & (one.HighParentOcc == 1)).mean())
pd.DataFrame([agreement]).to_csv(os.path.join(ROOT, "output", "u8_indicator_agreement.csv"), index=False)
print(agreement)
# the same agreement on the analysis sample of Sections 3 and 4 (reported where Section 2.2 introduces the two indicators)
pa = pd.read_pickle(os.path.join(INT, "panel_analysis.pkl"))
one_a = pa.dropna(subset=["HighParentOcc14", "HighParentOcc"]).drop_duplicates("pid")
one_a = one_a[one_a.pid.isin(pa.dropna(subset=["log_income", "HighParentOcc14", "ParentInc_bar"]).pid)]
pd.DataFrame([dict(children_with_both_indicators=len(one_a), share_agree=(one_a.HighParentOcc14 == one_a.HighParentOcc).mean(),
                   share_retrospective_only=((one_a.HighParentOcc14 == 1) & (one_a.HighParentOcc == 0)).mean(),
                   share_current_only=((one_a.HighParentOcc14 == 0) & (one_a.HighParentOcc == 1)).mean())]
             ).to_csv(os.path.join(ROOT, "output", "u8_indicator_agreement_analysis_sample.csv"), index=False)

# --- entropy balancing on the children of the sample, with the covariates and the algorithm of
#     college_expansion/code/ws_07b_entropy_balance.py
sys.path.insert(0, os.path.join(P, "code"))
from ws_07b_entropy_balance import entropy_balance
k = kid.copy()
k["cohort"] = (k["birth_year"] // 5) * 5
top = k["provcd"].value_counts().head(15).index.tolist()
k["provcd_top"] = np.where(k["provcd"].isin(top), k["provcd"], -1)
cont = k[["ParentInc", "age_first"]].astype(float).values
Xfull = np.hstack([cont, (cont - cont.mean(0)) ** 2, k[["female", "urban"]].astype(float).values,
                   pd.get_dummies(k["cohort"].astype(int), prefix="coh", drop_first=True).astype(float).values,
                   pd.get_dummies(k["provcd_top"].astype(int), prefix="prv", drop_first=True).astype(float).values])
t, c_ = (k["HP14"] == 1).values, (k["HP14"] == 0).values
y, m = k["log_y"].values, k["college"].values
w_eb, _ = entropy_balance(Xfull[c_], Xfull[t].mean(0))
tau, gap = y[t].mean() - (y[c_] * w_eb).sum(), m[t].mean() - (m[c_] * w_eb).sum()
rng = np.random.default_rng(20260520); it, ic = np.where(t)[0], np.where(c_)[0]; bt_, bg_ = [], []
for _ in range(200):
    a_, b_ = rng.choice(it, size=len(it), replace=True), rng.choice(ic, size=len(ic), replace=True)
    try:
        wb, _ = entropy_balance(Xfull[b_], Xfull[a_].mean(0), max_iter=200, tol=1e-7)
        tb, gb = y[a_].mean() - (y[b_] * wb).sum(), m[a_].mean() - (m[b_] * wb).sum()
        if np.isfinite(tb) and np.isfinite(gb):
            bt_.append(tb); bg_.append(gb)
    except Exception:
        continue
assert len(bt_) >= 150, len(bt_)
eb = dict(tau_hat=tau, tau_ci_lo=np.quantile(bt_, 0.025), tau_ci_hi=np.quantile(bt_, 0.975), tau_se=np.std(bt_, ddof=1),
          gap_hat=gap, gap_ci_lo=np.quantile(bg_, 0.025), gap_ci_hi=np.quantile(bg_, 0.975), gap_se=np.std(bg_, ddof=1),
          N_treat=int(t.sum()), N_ctrl=int(c_.sum()), n_moments_matched=int(Xfull.shape[1]), n_boot=len(bt_),
          completion_advantaged=m[t].mean(), completion_less_advantaged_reweighted=(m[c_] * w_eb).sum(), tau_naive=y[t].mean() - y[c_].mean())
pd.DataFrame([eb]).to_csv(os.path.join(ROOT, "output", "u8_entropy_balancing.csv"), index=False)
print("entropy balancing:", {k_: round(float(v), 4) for k_, v in eb.items()})
rows.append(dict(specification="entropy balancing", estimator="Entropy balancing", estimate=eb["tau_hat"],
                 lo=eb["tau_ci_lo"], hi=eb["tau_ci_hi"], children=int(eb["N_treat"] + eb["N_ctrl"]), advantaged=int(eb["N_treat"]),
                 source="bootstrap percentile interval"))

# --- the sample: counts, means by family background, families and siblings, standard error clustered on family
S = [dict(item="children", value=len(kid)), dict(item="advantaged children", value=int((kid.HP14 == 1).sum())),
     dict(item="child-waves", value=len(panel)), dict(item="waves per child", value=len(panel) / len(kid)),
     dict(item="first birth year", value=int(kid.birth_year.min())), dict(item="last birth year", value=int(kid.birth_year.max())),
     dict(item="raw gap in mean log earnings", value=kid[kid.HP14 == 1].log_y.mean() - kid[kid.HP14 == 0].log_y.mean())]
kk = kid.merge(panel.groupby("pid")[["age"]].mean().rename(columns={"age": "age_mean"}), left_on="pid", right_index=True)
for h, labh in [(1, "advantaged"), (0, "less advantaged")]:
    g = kk[kk.HP14 == h]
    for col, name in [("log_y", "mean log labor income"), ("ParentInc", "log family income of the parental household"), ("eduy_max", "years of schooling"),
                      ("college", "college completion"), ("female", "female"), ("urban", "urban residence"), ("age_mean", "age over the waves in the window"),
                      ("birth_year", "birth year"), ("waves", "waves in the window")]:
        S.append(dict(item=f"{name}, {labh} children", value=g[col].mean(), sd=g[col].std()))
fam_id = kids.assign(f=kids.pid_father.fillna(-1).astype(np.int64).astype(str) + "_" + kids.pid_mother.fillna(-1).astype(np.int64).astype(str)).set_index("pid")["f"]
sizes = fam_id.value_counts()
S += [dict(item="families", value=int(sizes.size)), dict(item="families with two or more children in the sample", value=int((sizes >= 2).sum()))]
dfb = panel.dropna(subset=["log_income", "HighParentOcc14", "ParentInc_bar", "provcd", "year", "age", "urban", "female"]).copy()
for v in ["age", "urban"]:
    dfb[v + "_bar"] = dfb.groupby("pid")[v].transform("mean")
Xb = pd.concat([dfb[["HighParentOcc14", "ParentInc_bar", "age", "urban", "age_bar", "urban_bar", "female"]].astype(float),
                pd.get_dummies(dfb["year"].astype(int), prefix="yr", drop_first=True).astype(float),
                pd.get_dummies(dfb["provcd"].astype(int), prefix="prv", drop_first=True).astype(float)], axis=1).assign(const=1.0)
se_child = sm.OLS(dfb["log_income"].values, Xb.values).fit(cov_type="cluster", cov_kwds={"groups": dfb["pid"].values}).bse[0]
se_family = sm.OLS(dfb["log_income"].values, Xb.values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dfb["pid"].map(fam_id))[0]}).bse[0]
S += [dict(item="baseline standard error clustered on child", value=se_child), dict(item="baseline standard error clustered on family", value=se_family)]
pd.DataFrame(S).to_csv(os.path.join(ROOT, "output", "u8_level_sample.csv"), index=False)

out = pd.DataFrame(rows)
out["in_figure_2"] = ~out.specification.str.contains("not in Figure 2")
out.to_csv(OUT, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(out[["specification", "estimate", "se", "p", "lo", "hi", "children", "advantaged"]].round(4).to_string())
print("rows in Figure 2:", int(out.in_figure_2.sum()))
