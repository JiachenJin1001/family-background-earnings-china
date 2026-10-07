"""Minimum-distance estimation of the income process with diagonal weights.

Sample: child-year rows of panel_resid with non-missing y_tilde_w and
HighParentOcc14 (children aged 22 to 55, waves 2014 to 2022, at least three
waves with positive income); pooled, advantaged children (H) and ordinary
children (L). The residual is demeaned within group and year.

Model: permanent effect plus AR(1) transitory component, fitted to the 15
variances and autocovariances of the five waves.

For each group:
  - standard errors of the 15 moments from a bootstrap that resamples
    children (200 replications);
  - minimum-distance estimates with equal weights, then with each moment
    weighted by the inverse of its bootstrap variance;
  - the overidentification statistic of the weighted fit, with a p-value from
    a parametric bootstrap: 200 panels are simulated from the fitted model
    with normal draws, the observed pattern of missing waves is imposed, and
    the weighted fit is repeated on each.

Outputs:
  output/tables/m5_weighted_md.csv      weighted estimates, overidentification
                                        statistic and its bootstrap p-value
  output/tables/m5_moment_se.csv        moments and their bootstrap standard
                                        errors, by group
  output/figures/fig_ridge.png          bootstrap draws of the AR(1)
                                        coefficient and the permanent share,
                                        advantaged and less advantaged children
"""
import os as _os, time
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_pickle(_os.path.join(_WSROOT, "data/intermediate/panel_resid.pkl"))
d = d.dropna(subset=["y_tilde_w","HighParentOcc14"]).copy()
d["HP14"] = d["HighParentOcc14"].astype(int)
years = sorted(d["year"].unique()); T = len(years)
piv = d.pivot_table(index="pid", columns="year", values="y_tilde_w")
hp = d.groupby("pid")["HP14"].first().reindex(piv.index)
for g in [0,1]:
    piv.loc[hp==g] = piv.loc[hp==g] - piv[hp==g].mean()
Y = piv.values
PAIRS = [(i,j) for i in range(T) for j in range(i,T)]

def cov_vec(Yg):
    out = np.full(len(PAIRS), np.nan)
    for k,(i,j) in enumerate(PAIRS):
        prod = Yg[:,i]*Yg[:,j]; ok = ~np.isnan(prod)
        if ok.sum() >= 20: out[k] = prod[ok].mean()
    return out

def model_vec(theta):
    se2, r, sg2 = theta
    return np.array([se2 + (r**(j-i))*sg2/(1-r**2) for i,j in PAIRS])

def fit(omega, w=None):
    mask = ~np.isnan(omega)
    ww = np.ones(mask.sum()) if w is None else w[mask]
    def obj(theta):
        if not (0<theta[0]<2 and -0.95<theta[1]<0.95 and 0<theta[2]<2): return 1e6
        e = (omega - model_vec(theta))[mask]
        return float(e @ (ww * e))
    best=None
    for x0 in [(0.1,0.2,0.3),(0.05,0.4,0.35),(0.15,0.05,0.3)]:
        r = minimize(obj, x0, method="Nelder-Mead",
                     options=dict(maxiter=2000, xatol=1e-7, fatol=1e-12))
        if best is None or r.fun < best.fun: best = r
    return best.x, best.fun

rng = np.random.default_rng(20260708)
B = 200
groups = {"pooled": np.ones(len(Y),bool), "H": (hp==1).values, "L": (hp==0).values}
rows, mrows, draws = [], [], {}
for gname, mask in groups.items():
    Yg = Y[mask]; n = Yg.shape[0]
    om0 = cov_vec(Yg)
    # bootstrap of the moments, which gives their variances, and of the equal-weight estimates
    boots_om, boots_th = [], []
    for b in range(B):
        idx = rng.integers(0, n, n)
        omb = cov_vec(Yg[idx]); boots_om.append(omb)
        th,_ = fit(omb); boots_th.append(th)
    boots_om = np.array(boots_om); boots_th = np.array(boots_th)
    draws[gname] = boots_th
    mse = np.nanstd(boots_om, 0)
    for k,(i,j) in enumerate(PAIRS):
        mrows.append(dict(group=gname, t=years[i], s=years[j],
                          omega=om0[k], se=mse[k]))
    # step 1: equal weights
    th1, J1 = fit(om0)
    # step 2: diagonal weights, the inverse of the bootstrap variance of each moment
    w = 1.0/np.maximum(mse**2, 1e-8)
    th2, J2 = fit(om0, w=w)
    # parametric bootstrap of the weighted statistic under the fitted model, with the observed pattern of missing waves
    se2, r_, sg2 = th2
    missmask = np.isnan(Yg)
    sims = 200; Jsims = []
    sd_v = np.sqrt(sg2/(1-r_**2)); sd_e = np.sqrt(sg2); sd_eta = np.sqrt(se2)
    for sdx in range(sims):
        eta = rng.normal(0, sd_eta, size=(n,1))
        v = np.empty((n,T)); v[:,0] = rng.normal(0, sd_v, n)
        for t in range(1,T):
            v[:,t] = r_*v[:,t-1] + rng.normal(0, sd_e, n)
        Ysim = eta + v; Ysim[missmask] = np.nan
        Ysim = Ysim - np.nanmean(Ysim, 0)
        oms = cov_vec(Ysim)
        _, Js = fit(oms, w=w)
        Jsims.append(Js)
    Jsims = np.array(Jsims)
    pJ = float((Jsims >= J2).mean())
    th2_se = boots_th.std(0)   # standard deviation of the equal-weight bootstrap estimates
    sperm2 = se2/(se2 + sg2/(1-r_**2))
    rows.append(dict(group=gname, eta2_w=se2, rho_w=r_, eps2_w=sg2,
                     sperm_w=sperm2, J_w=J2, J_p_calibrated=pJ,
                     J_eq=J1))
    print(f"{gname:>6}: two-step eta2={se2:.3f} rho={r_:.2f} eps2={sg2:.3f} "
          f"s_perm={sperm2:.3f}  J_w={J2:.1f}  bootstrap p={pJ:.3f}")

pd.DataFrame(rows).to_csv(_os.path.join(_WSROOT,"output/tables/m5_weighted_md.csv"), index=False)
pd.DataFrame(mrows).to_csv(_os.path.join(_WSROOT,"output/tables/m5_moment_se.csv"), index=False)

# figure: bootstrap draws of the AR(1) coefficient and the permanent share
fig, ax = plt.subplots(figsize=(7.2,5))
for gname, c in [("H","#d62728"), ("L","#1f77b4")]:
    th = draws[gname]
    sperm = th[:,0]/(th[:,0] + th[:,2]/(1-th[:,1]**2))
    ax.scatter(th[:,1], sperm, s=9, alpha=0.35, color=c,
               label=f"{'advantaged' if gname=='H' else 'comparison'} group draws")
ax.set_xlabel(r"Transitory persistence $\rho_v$")
ax.set_ylabel("Permanent share")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.savefig(_os.path.join(_WSROOT,"output/figures/fig_ridge.png"), dpi=200, bbox_inches="tight")
print("wrote figure fig_ridge.png")
