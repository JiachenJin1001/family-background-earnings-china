"""Income process of the parents.

The attenuation schedule of the paper is estimated from children aged 22 to
45 and is applied to income measures of parents. This script estimates the
same variance decomposition on the parents' own earnings.

Sample: every parent linked to a child of the analysis sample (pid_father and
pid_mother in panel_analysis), the parent's own income records in the
person-year panel for the waves 2014 to 2022 at age 60 or younger, with
positive income in at least three waves.

Log income is regressed on a quartic in age, its interaction with a female
indicator, the female indicator and year indicators (the specification used
for the children, without the urban and province indicators). From the
autocovariances of the residual:

    sigma2_eta = average autocovariance at lags of two waves or more
    permanent share = sigma2_eta / variance
    rho_v = (cov1 - sigma2_eta) / (var - sigma2_eta)

The standard error and the 95 percent interval of the permanent share come
from a bootstrap that resamples parents (200 replications).

Output: output/tables/m14_parent_process.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

long = pd.read_pickle(os.path.join(ROOT,
    "data/intermediate/_panel_long_cache.pkl"))
an = pd.read_pickle(os.path.join(ROOT, "data/intermediate/panel_analysis.pkl"))
parents = pd.unique(pd.concat([an["pid_father"], an["pid_mother"]]).dropna())

d = long[long["pid"].isin(parents)].copy()
d = d[(d["year"] >= 2014) & (d["year"] <= 2022)]
d["income"] = pd.to_numeric(d["income"], errors="coerce")
d["age"] = pd.to_numeric(d["age"], errors="coerce")
d = d[(d["age"] <= 60) & (d["income"] > 0)].copy()
d["logy"] = np.log(d["income"])
d = d.dropna(subset=["logy", "age", "female"])
nw = d.groupby("pid")["year"].nunique()
keep = nw[nw >= 3].index
d = d[d["pid"].isin(keep)].copy()
print(f"linked parents, aged<=60, 3+ waves of positive income: "
      f"{d['pid'].nunique():,} parents, {len(d):,} person-waves")

# residual of log income on a quartic in age, its interaction with female, female, and year indicators
X = [np.ones(len(d))]
a = (d["age"] - d["age"].mean()).values
for k in range(1, 5):
    X.append(a**k)
    X.append((a**k) * d["female"].values)
X.append(d["female"].values)
for y in sorted(d["year"].unique())[1:]:
    X.append((d["year"] == y).astype(float).values)
X = np.column_stack(X)
b = np.linalg.lstsq(X, d["logy"].values, rcond=None)[0]
d["res"] = d["logy"].values - X @ b

def moments(df):
    wide = df.pivot_table(index="pid", columns="year", values="res")
    years = sorted(wide.columns)
    var = np.nanmean([wide[y].var() for y in years])
    covs = {}
    for i, y1 in enumerate(years):
        for y2 in years[i+1:]:
            k = (y2 - y1) // 2
            pair = wide[[y1, y2]].dropna()
            if len(pair) >= 30:
                covs.setdefault(k, []).append(pair[y1].cov(pair[y2]))
    cov1 = np.mean(covs.get(1, [np.nan]))
    covhi = np.mean(sum([covs[k] for k in covs if k >= 2], []))
    share = covhi / var
    rho = (cov1 - covhi) / (var - covhi)
    return var, cov1, covhi, share, rho

var, cov1, covhi, share, rho = moments(d)
print(f"variance {var:.3f}  cov(lag1) {cov1:.3f}  cov(lag>=2) {covhi:.3f}")
print(f"permanent share {share:.3f}   implied rho_v {rho:.3f}")

rng = np.random.default_rng(20260722)
pids = d["pid"].unique()
dd = d.set_index("pid")
sizes = dd.groupby(level=0).size()
boots = []
for r in range(200):
    draw = rng.choice(pids, size=len(pids), replace=True)
    bs = dd.loc[draw].reset_index(drop=True)
    # each drawn parent receives a new identifier, so that a parent drawn
    # twice enters as two separate panel units
    bs["pid"] = np.repeat(np.arange(len(draw)), sizes.loc[draw].values)
    try:
        boots.append(moments(bs)[3])
    except Exception:
        pass
boots = np.array([x for x in boots if np.isfinite(x)])
print(f"share bootstrap SE {boots.std(ddof=1):.3f}  "
      f"95% CI [{np.quantile(boots,0.025):.3f}, {np.quantile(boots,0.975):.3f}]  "
      f"(n_ok={len(boots)})")

pd.DataFrame([dict(n_parents=d["pid"].nunique(), n_obs=len(d), var=var,
                   cov1=cov1, cov_hi=covhi, share=share, rho=rho,
                   share_se=boots.std(ddof=1),
                   share_lo=np.quantile(boots, 0.025),
                   share_hi=np.quantile(boots, 0.975))]).to_csv(
    os.path.join(ROOT, "output/tables/m14_parent_process.csv"), index=False)
print("wrote m14_parent_process.csv")
