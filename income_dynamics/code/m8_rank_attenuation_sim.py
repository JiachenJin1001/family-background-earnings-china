"""Attenuation of rank-rank slopes, simulated from the fitted income process.

The attenuation schedule eta_T of the paper gives the fraction of a
coefficient in levels that is recovered when parental income is measured by
an average over T waves. It depends only on the ratio of variances. Rank-rank
slopes are attenuated as well, by a factor that depends on the whole
distribution. This script simulates that factor from the fitted process
(permanent effect plus AR(1) transitory component, pooled minimum-distance
estimates written by m7_md_cellweighted_boot.py):

    sigma2_eta   variance of the permanent effect (0.125 in the pooled sample)
    rho          AR(1) coefficient of the transitory component between waves
                 two years apart (0.16)
    share        permanent share (0.27), so sigma2_v = sigma2_eta*(1/share - 1)

Design: the parent's permanent log income alpha is normal with variance
sigma2_eta. The child's permanent income is b * alpha + e with b = 0.30 and
the variance of e set so that the child's permanent income has the same
variance as the parent's. The value of b does not affect the ratio reported.
Measured parental income is the mean over T waves of alpha + v_t, with v_t a
stationary AR(1). The factor reported is

    slope( rank(child) on rank(mean over T waves) ) / slope( rank(child) on rank(alpha) ),

the fraction of the true rank-rank slope recovered with T waves. A second
factor adds transitory noise to the child's income (mean of three waves). All
components are normal, and ranks are computed in the simulated population of
400,000 parent-child pairs. The simulation is repeated with innovations from a
t distribution with 5 degrees of freedom, scaled to unit variance, for T = 1,
3 and 7.

Output: output/tables/m8_rank_attenuation.csv (T, eta_T, rank_factor,
rank_factor_2sided) and output/tables/m8_rank_attenuation_t5.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output", "tables",
                   "m8_rank_attenuation.csv")

import pandas as _pd, os as _os2
_m7 = _pd.read_csv(_os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), "..", "output", "tables", "m7_md_cellw_boot.csv")).set_index("group").loc["pooled"]
S2_ETA, RHO, SHARE = float(_m7["eta2"]), float(_m7["rho"]), float(_m7["sperm"])   # pooled minimum-distance estimates of this run
S2_V = S2_ETA * (1.0 / SHARE - 1.0)
N = 400_000
B_TRUE = 0.30
rng = np.random.default_rng(20260716)

alpha = rng.normal(0.0, np.sqrt(S2_ETA), N)
child = B_TRUE * alpha + rng.normal(
    0.0, np.sqrt(S2_ETA - B_TRUE**2 * S2_ETA), N)

# transitory component: stationary AR(1), waves two years apart
Tmax = 7
v = np.empty((N, Tmax))
v[:, 0] = rng.normal(0.0, np.sqrt(S2_V), N)
for t in range(1, Tmax):
    v[:, t] = RHO * v[:, t - 1] + rng.normal(
        0.0, np.sqrt(S2_V * (1 - RHO**2)), N)

def rank01(x):
    r = np.argsort(np.argsort(x)).astype(float)
    return r / (len(x) - 1)

r_child = rank01(child)
r_alpha = rank01(alpha)
slope_true = np.polyfit(r_alpha, r_child, 1)[0]

def eta_T(T):
    if T == 1:
        return S2_ETA / (S2_ETA + S2_V)
    # variance of the mean of T draws from a stationary AR(1)
    ac = sum((T - k) * RHO**k for k in range(1, T))
    var_mean = S2_V * (T + 2 * ac) / T**2
    return S2_ETA / (S2_ETA + var_mean)

# transitory component of the child's income, for the factor with noise on
# both sides (same process, independent draws; the child is measured with
# the mean of three waves)
vc = np.empty((N, 3))
vc[:, 0] = rng.normal(0.0, np.sqrt(S2_V), N)
for t in range(1, 3):
    vc[:, t] = RHO * vc[:, t - 1] + rng.normal(
        0.0, np.sqrt(S2_V * (1 - RHO**2)), N)
r_child_meas = rank01(child + vc.mean(axis=1))
slope_true_2s = np.polyfit(r_alpha, r_child_meas, 1)[0] / slope_true

rows = []
for T in range(1, Tmax + 1):
    meas = alpha + v[:, :T].mean(axis=1)
    rk = rank01(meas)
    slope_T = np.polyfit(rk, r_child, 1)[0]
    slope_T2 = np.polyfit(rk, r_child_meas, 1)[0]
    rows.append(dict(T=T, eta_T=eta_T(T),
                     rank_factor=slope_T / slope_true,
                     rank_factor_2sided=slope_T2 / slope_true))
    print(f"T={T}:  eta_T = {eta_T(T):.3f}   rank factor = "
          f"{slope_T / slope_true:.3f}   two-sided (child 3 waves) = "
          f"{slope_T2 / slope_true:.3f}")

pd.DataFrame(rows).to_csv(OUT, index=False)
print("wrote", os.path.abspath(OUT))

# The same simulation with innovations from a t distribution with 5 degrees
# of freedom, scaled to unit variance
def t5(size):
    x = rng.standard_t(5, size)
    return x / np.sqrt(5.0 / 3.0)

alpha_t = np.sqrt(S2_ETA) * t5(N)
child_t = B_TRUE * alpha_t + np.sqrt(S2_ETA - B_TRUE**2 * S2_ETA) * t5(N)
vt = np.empty((N, Tmax))
vt[:, 0] = np.sqrt(S2_V) * t5(N)
for t in range(1, Tmax):
    vt[:, t] = RHO * vt[:, t - 1] + np.sqrt(S2_V * (1 - RHO**2)) * t5(N)
rc_t = rank01(child_t)
slope_true_t = np.polyfit(rank01(alpha_t), rc_t, 1)[0]
out_t = []
for T in [1, 3, 7]:
    st = np.polyfit(rank01(alpha_t + vt[:, :T].mean(axis=1)), rc_t, 1)[0]
    out_t.append((T, st / slope_true_t))
    print(f"t(5) tails, T={T}: rank factor = {st / slope_true_t:.3f}")
pd.DataFrame(out_t, columns=["T", "rank_factor_t5"]).to_csv(
    OUT.replace(".csv", "_t5.csv"), index=False)
