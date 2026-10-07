"""Census test of differential trends in college completion before the 1999
college expansion, by provincial intensity.

Data: IPUMS International extract ipumsi_00001.csv.gz (China 1990 and 2000,
1 percent samples, ages 24 to 44). GEO1_CN minus 156000 is the GB province
code (29 provinces with consistent boundaries; Chongqing is part of Sichuan).
College completion is EDATTAIN == 4 (university completed). The detailed
codes of these samples have no separate category for completed junior
college, so this is the narrow definition of tertiary education.

Provincial intensity: the log ratio of the number of college entrance
examination registrants in 2000 to the number in 1998 (int_log_2000_1998),
from the intensity file in data/external.

Design: in the 2000 census, the birth cohorts 1956 to 1976 all passed college
age before the expansion. College completion rates by province and cohort are
regressed on a linear cohort trend, an indicator for provinces in the top
third of intensity, and their interaction, for provinces in the top and the
bottom third, weighted by population and with standard errors clustered by
province. The interaction is the difference in cohort slopes. The 1990 census
(cohorts 1946 to 1966) repeats the test on an earlier window. For the cohorts
1956 to 1966, observed in both censuses, the national completion rates
measured in 1990 and in 2000 are compared.

Outputs: output/tables/p16_census_pretrend.csv,
output/figures/fig_census_pretrend.png
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
EXT = os.path.join(BASE, "data", "external")

d = pd.read_csv(os.path.join(EXT, "ipumsi_00001.csv.gz"),
                usecols=["YEAR","GEO1_CN","AGE","EDATTAIN","PERWT"])
d["prov"] = d["GEO1_CN"] - 156000
d["cohort"] = d["YEAR"] - d["AGE"]
d["college"] = (d["EDATTAIN"] == 4).astype(float)

canon = pd.read_csv(os.path.join(EXT, "expansion_intensity_canonical.csv"))
canon = canon.rename(columns={"provcd": "prov"})
inten = canon[["prov", "int_log_2000_1998"]].dropna()

def cell_rates(df):
    g = df.groupby(["prov","cohort"]).apply(
        lambda x: pd.Series({"rate": np.average(x["college"], weights=x["PERWT"]),
                             "w": x["PERWT"].sum()}), include_groups=False).reset_index()
    return g

def pretrend(df, label, c_lo, c_hi):
    cells = cell_rates(df[(df.cohort>=c_lo)&(df.cohort<=c_hi)])
    m = cells.merge(inten, on="prov", how="inner")
    terc = m.groupby("prov")["int_log_2000_1998"].first()
    hi = terc[terc >= terc.quantile(2/3)].index
    lo = terc[terc <= terc.quantile(1/3)].index
    t = m[m.prov.isin(hi) | m.prov.isin(lo)].copy()
    t["hi"] = t.prov.isin(hi).astype(float)
    t["c0"] = t.cohort - c_lo
    t["hi_c"] = t.hi * t.c0
    X = sm.add_constant(t[["c0","hi","hi_c"]])
    r = sm.WLS(t.rate, X, weights=t.w).fit(cov_type="cluster",
                                           cov_kwds={"groups": t.prov})
    print(f"{label}: slope diff (hi x cohort) = {r.params['hi_c']:+.5f} "
          f"(se {r.bse['hi_c']:.5f}, p = {r.pvalues['hi_c']:.3f})  "
          f"[{len(hi)} hi / {len(lo)} lo provinces, cohorts {c_lo}-{c_hi}]")
    return dict(census=label, slope_diff=r.params["hi_c"], se=r.bse["hi_c"],
                p=r.pvalues["hi_c"], n_cells=len(t)), m, hi, lo

res = []
r2000, m2000, hi, lo = pretrend(d[d.YEAR==2000], "2000 census", 1956, 1976)
res.append(r2000)
r1990, m1990, _, _ = pretrend(d[d.YEAR==1990], "1990 census", 1946, 1966)
res.append(r1990)

# national completion rate of the cohorts 1956 to 1966 as measured in each census
ov = []
for yr, (a,b) in [(1990,(1956,1966)), (2000,(1956,1966))]:
    cells = cell_rates(d[(d.YEAR==yr)&(d.cohort>=a)&(d.cohort<=b)])
    nat = np.average(cells.rate, weights=cells.w)
    ov.append(nat)
    print(f"national college rate, cohorts 1956-66, measured {yr}: {nat:.4f}")
res.append(dict(census="overlap 1956-66", slope_diff=ov[1]-ov[0], se=np.nan,
                p=np.nan, n_cells=np.nan))
pd.DataFrame(res).to_csv(os.path.join(BASE, "output", "tables", "p16_census_pretrend.csv"), index=False)

# figure: cohort profiles by intensity group, both censuses
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, (yr, (a,b), mm) in zip(axes, [(1990,(1946,1966),m1990), (2000,(1956,1976),m2000)]):
    for grp, ids, c in [("high-intensity", hi, "#d62728"), ("low-intensity", lo, "#1f77b4")]:
        sub = mm[mm.prov.isin(ids)]
        prof = sub.groupby("cohort").apply(
            lambda x: np.average(x.rate, weights=x.w), include_groups=False)
        ax.plot(prof.index, prof.values, "o-", ms=3.5, color=c, label=grp)
    ax.set_title(f"{yr} census (cohorts {a}--{b})", fontsize=10)
    ax.set_xlabel("Birth cohort"); ax.grid(alpha=.25)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
axes[0].set_ylabel("University completion rate")
axes[0].legend(fontsize=9)
fig.suptitle("Pre-reform cohort profiles of college completion, by 1999-expansion intensity tercile",
             fontsize=10)
fig.savefig(os.path.join(BASE, "output", "figures", "fig_census_pretrend.png"),
            dpi=200, bbox_inches="tight")
print("wrote figure + p16_census_pretrend.csv")
