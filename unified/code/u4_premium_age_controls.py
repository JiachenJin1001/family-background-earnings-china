"""The premium under alternative age controls.

Correlated random-effects regression of Section 5 of the paper (family-income
control) on the child-waves at ages 30 and over of the children with a level of
earnings at those ages (sample_mature_age.child_waves()), with the age
control linear (the paper's specification), quadratic, cubic, and a full set
of age indicators; each time-varying control enters with its within-child
mean. Standard errors clustered on child.
Output: output/u4_premium_age_controls.csv (read by the online appendix builder).
"""
import os, sys
import numpy as np, pandas as pd, statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from _locate import sibling
P = sibling("college_expansion", __file__)

from sample_mature_age import child_waves
panel = child_waves()
d = panel.dropna(subset=["log_income", "HighParentOcc14", "ParentInc_bar",
                         "provcd", "year", "age", "urban", "female"]).copy()
d["age2"] = d["age"] ** 2; d["age3"] = d["age"] ** 3
for v in ["age", "age2", "age3", "urban"]:
    d[f"{v}_bar"] = d.groupby("pid")[v].transform("mean")
prv = pd.get_dummies(d["provcd"].astype(int), prefix="prv", drop_first=True).astype(float)
yr = pd.get_dummies(d["year"].astype(int), prefix="yr", drop_first=True).astype(float)
Y = d["log_income"].astype(float).values; pid = d["pid"].values
print(f"sample: {len(d):,} child-waves, {d.pid.nunique():,} children")


def cre(X, label):
    X = X.copy(); X["const"] = 1.0
    m = sm.OLS(Y, X.values).fit(cov_type="cluster", cov_kwds={"groups": pid})
    print(f"  {label:32s} {m.params[0]:+.4f} (SE {m.bse[0]:.4f})")
    return m.params[0], m.bse[0]


base = ["HighParentOcc14", "ParentInc_bar"]
res = {}
res["linear"] = cre(pd.concat([d[base + ["age", "urban", "age_bar", "urban_bar", "female"]].astype(float), yr, prv], axis=1), "linear age")
res["quad"] = cre(pd.concat([d[base + ["age", "age2", "urban", "age_bar", "age2_bar", "urban_bar", "female"]].astype(float), yr, prv], axis=1), "quadratic age")
res["cubic"] = cre(pd.concat([d[base + ["age", "age2", "age3", "urban", "age_bar", "age2_bar", "age3_bar", "urban_bar", "female"]].astype(float), yr, prv], axis=1), "cubic age")
agd = pd.get_dummies(d["age"].astype(int), prefix="a", drop_first=True).astype(float)
agd_bar = agd.groupby(d["pid"].values).transform("mean"); agd_bar.columns = [c + "_bar" for c in agd.columns]
res["dummies"] = cre(pd.concat([d[base + ["urban", "urban_bar", "female"]].astype(float), agd, agd_bar, yr, prv], axis=1), "age indicators")
pd.DataFrame([dict(spec=k, b=v[0], se=v[1]) for k, v in res.items()]).to_csv(
    os.path.join(ROOT, "output", "u4_premium_age_controls.csv"), index=False)
print("wrote output/u4_premium_age_controls.csv")
