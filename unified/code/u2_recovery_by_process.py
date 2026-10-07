"""Recovery fractions for the one published study that uses the same survey
(Fan, Yi and Zhang 2021). That study uses the 2010, 2012, 2014 and 2016 waves of
the CFPS and averages income over at least two of them, that is over two to
four biennial waves ("two to six years over the four waves" in their words).
Its raw least squares estimates are 0.166 for the cohort born 1970 to 1980 and
0.176 for the cohort born 1981 to 1988; its preferred estimates, 0.390 and
0.442, are least squares on income predicted from education and demographics
with a selection correction. The fractions are computed under two earnings
processes: the one estimated on the 2,523 children observed in at least three
waves and the one estimated on the 542 linked parents aged sixty or younger
with three or more waves of positive labor income.

The upper end is a four-wave average, the longest the study offers.

Inputs (income-dynamics pipeline):
  output/tables/m7_md_cellw_boot.csv    children: permanent share and persistence (pooled sample)
  output/tables/m14_parent_process.csv  linked parents: permanent share and persistence
Output: output/u2_recovery_by_process.csv
"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
import sys; sys.path.insert(0, HERE)
from _locate import sibling
M = os.path.join(sibling("income_dynamics", __file__), "output", "tables")
PUBLISHED_LO, PUBLISHED_HI = 0.166, 0.176      # raw least squares estimates of the study (early and late cohort)
WAVES_LO, WAVES_HI = 2, 4                      # the study averages income over two to four biennial waves
child = pd.read_csv(os.path.join(M, "m7_md_cellw_boot.csv")).set_index("group").loc["pooled"]
par = pd.read_csv(os.path.join(M, "m14_parent_process.csv")).iloc[0]


def recovery(share, rho, T):
    """Share of a coefficient on permanent income recovered by a T-wave
    average, with the exact variance of the average of an AR(1) transitory
    component (total variance normalized to one)."""
    k = np.arange(1, T)
    var_avg = (1 - share) * (T + 2 * np.sum((T - k) * rho ** k)) / T ** 2
    return share / (share + var_avg)


rows = []
for name, share, rho in [("children", child.sperm, child.rho), ("linked parents", par.share, par.rho)]:
    lo, hi = recovery(share, rho, WAVES_LO), recovery(share, rho, WAVES_HI)
    rows.append(dict(process=name, share=share, persistence=rho, recovery_one_wave=recovery(share, rho, 1),
                     waves_lo=WAVES_LO, waves_hi=WAVES_HI, recovery_lo=lo, recovery_hi=hi, implied_lo=PUBLISHED_LO / hi, implied_hi=PUBLISHED_HI / lo))
out = pd.DataFrame(rows); out.to_csv(os.path.join(ROOT, "output", "u2_recovery_by_process.csv"), index=False)
print(out.round(3).to_string(index=False))
