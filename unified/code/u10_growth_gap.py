"""Differential earnings growth by family background (equation eq:fe of the paper).

Fixed-effects regression of residual log earnings on the interaction of the
advantaged-family indicator with a linear wave trend, with and without the
time-varying covariates (age, urban residence), on the five-wave analysis panel
(panel_resid of the income-dynamics pipeline). Standard errors clustered on the
child. The seven-wave estimate is copied from ws_method3_7wave.csv.
Output: output/u10_growth_gap.csv
"""
import os, sys
import numpy as np, pandas as pd, statsmodels.api as sm, pyreadstat
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _locate import sibling
M = sibling("income_dynamics", __file__)
df, _ = pyreadstat.read_dta(os.path.join(M, "data", "intermediate", "panel_resid.dta"))
df = df.dropna(subset=["y_tilde_w", "HighParentOcc14"]).copy()
df["pid"] = df.pid.astype(np.int64); df["year"] = df.year.astype(int)
waves = sorted(df.year.unique()); df["t"] = df.year.map({y: i + 1 for i, y in enumerate(waves)}).astype(float)
df["HPt"] = df.HighParentOcc14.astype(float) * df.t
def within(d, cols):
    return d[cols] - d.groupby("pid")[cols].transform("mean")
rows = []
for lab, z in [("no covariates", []), ("age and urban residence", ["age", "urban"])]:
    d = df.dropna(subset=z) if z else df
    W = within(d, ["y_tilde_w", "HPt"] + z)
    X = sm.add_constant(W[["HPt"] + z].astype(float), has_constant="add")
    r = sm.OLS(W["y_tilde_w"].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": d.pid.values})
    rows.append(dict(panel="2014-2022 (five waves)", covariates=lab, gamma=r.params["HPt"], se=r.bse["HPt"], p=r.pvalues["HPt"],
                     observations=int(r.nobs), children=int(d.pid.nunique())))
    print(f"{lab:26s} gamma {r.params['HPt']:+.4f} (SE {r.bse['HPt']:.4f}, p {r.pvalues['HPt']:.3f}) N obs {int(r.nobs):,}, children {d.pid.nunique():,}")
w7 = pd.read_csv(os.path.join(M, "data", "intermediate", "ws", "ws_method3_7wave.csv")).set_index("obj").loc["gamma_FE"]
rows.append(dict(panel="2010-2022 (reconstructed, seven waves)", covariates="no covariates", gamma=w7.val, se=w7.se, p=w7.p, observations=np.nan, children=np.nan))
out = os.path.join(HERE, "..", "output", "u10_growth_gap.csv"); pd.DataFrame(rows).to_csv(out, index=False); print("wrote", out)
