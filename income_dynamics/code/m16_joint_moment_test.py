"""Joint Wald test that the autocovariances of residual log income are equal
for advantaged and less advantaged children.

Sample: child-year rows of panel_resid (children aged 22 to 55, waves 2014 to
2022, at least three waves with positive income), split by HighParentOcc14.

The winsorized residual y_tilde_w is demeaned within group and year. For each
group the 15 variances and autocovariances of the five waves are computed from
all children observed in both waves of a pair. The test statistic is the
quadratic form of the 15 differences (advantaged minus ordinary) in the
inverse of their covariance matrix, which is estimated by a bootstrap that
resamples children within each group (2,000 replications). The p-value is
reported from the chi-squared distribution with 15 degrees of freedom and
from the bootstrap distribution of the statistic.

Output: data/intermediate/m16_joint_moment_test.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "..")

resid = pd.read_pickle(os.path.join(M, "data", "intermediate", "panel_resid.pkl"))
gcol = "HP14" if "HP14" in resid.columns else "HighParentOcc14"
ycol = "y_tilde_w" if "y_tilde_w" in resid.columns else \
       [c for c in resid.columns if "tilde" in c or c.startswith("resid")][0]
WAVES = sorted(resid["year"].unique())
print("group col:", gcol, "| y col:", ycol, "| waves:", WAVES, "| rows:", len(resid))

def pivot(group):
    return group.pivot_table(index="pid", columns="year", values=ycol).values

PH = pivot(resid[resid[gcol] == 1])
PL = pivot(resid[resid[gcol] == 0])
print(f"children: H {PH.shape[0]}, L {PL.shape[0]}")

def moments(P):
    Q = P - np.nanmean(P, axis=0, keepdims=True)   # demean within group and year
    k = Q.shape[1]
    out = []
    for a in range(k):
        for b in range(a, k):
            prod = Q[:, a] * Q[:, b]
            out.append(np.nanmean(prod))
    return np.array(out)

gap = moments(PH) - moments(PL)
n_m = len(gap)

rng = np.random.default_rng(20260822)
B = 2000
draws = np.empty((B, n_m))
for i in range(B):
    bH = PH[rng.integers(0, PH.shape[0], PH.shape[0])]
    bL = PL[rng.integers(0, PL.shape[0], PL.shape[0])]
    draws[i] = moments(bH) - moments(bL)

V = np.cov(draws.T)
W = float(gap @ np.linalg.solve(V, gap))
p = 1 - stats.chi2.cdf(W, n_m)
se = np.sqrt(np.diag(V))
z = gap / se
print(f"\nJoint Wald on {n_m} moment gaps: chi2 = {W:.1f}, df = {n_m}, p = {p:.3f}")
print(f"max |z| = {np.abs(z).max():.2f}; number with |z| > 2: {(np.abs(z) > 2).sum()}")
# p-value from the bootstrap distribution: the draws are centered at the estimated differences and their statistics compared with W
Wb = np.einsum("ij,jk,ik->i", draws - gap, np.linalg.inv(V), draws - gap)
print(f"bootstrap p = {(Wb >= W).mean():.3f}")

out = pd.DataFrame([dict(chi2=W, df=n_m, p=p, p_boot=(Wb >= W).mean(),
                         max_abs_z=float(np.abs(z).max()), B=B)])
out.to_csv(os.path.join(M, "data", "intermediate", "m16_joint_moment_test.csv"), index=False)
print("wrote m16_joint_moment_test.csv")
