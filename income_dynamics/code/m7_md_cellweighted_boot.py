"""Bootstrap standard errors of the minimum-distance estimates of the income
process (permanent effect plus AR(1) transitory component).

Sample: child-year rows of panel_resid with non-missing y_tilde_w and
HighParentOcc14 (children aged 22 to 55, waves 2014 to 2022, at least three
waves with positive income); pooled, advantaged children (H) and ordinary
children (L).

The point estimator is the one of 04_method1_md.py: each autocovariance is
weighted by the number of children observed in both waves, the residual is
demeaned by year within the group being estimated (within year only for the
pooled sample). Children are resampled with replacement (1,000 replications),
and the demeaning is repeated in every replication.

Outputs:
  output/tables/m7_md_cellw_boot.csv   estimates and bootstrap standard errors
  output/figures/fig_ridge.png         bootstrap draws of the AR(1) coefficient
                                       and the permanent share, by group
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
hp = d.groupby("pid")["HP14"].first().reindex(piv.index).values
Yraw = piv.values
PAIRS = [(i,j) for i in range(T) for j in range(i,T)]

def demean(Y, groups):
    Y = Y.copy()
    if groups is None:
        Y -= np.nanmean(Y, 0)
    else:
        for g in np.unique(groups):
            Y[groups==g] -= np.nanmean(Y[groups==g], 0)
    return Y

def moments(Y):
    om = np.full(len(PAIRS), np.nan); n = np.zeros(len(PAIRS))
    for k,(i,j) in enumerate(PAIRS):
        p = Y[:,i]*Y[:,j]; ok = ~np.isnan(p)
        c = ok.sum()
        if c >= 2: om[k] = p[ok].mean(); n[k] = c
    return om, n

def fit(om, w):
    mask = ~np.isnan(om); ww = w[mask]
    def model(th):
        se2, r, sg2 = th
        return np.array([se2 + (r**(j-i))*sg2/(1-r**2) for i,j in PAIRS])
    def obj(th):
        if not (0<th[0]<2 and -0.95<th[1]<0.95 and 0<th[2]<2): return 1e9
        e = (om - model(th))[mask]
        return float(e @ (ww*e))
    best=None
    for x0 in [(0.1,0.2,0.3),(0.05,0.4,0.35),(0.15,0.05,0.3)]:
        r = minimize(obj, x0, method="Nelder-Mead",
                     options=dict(maxiter=3000, xatol=1e-7, fatol=1e-12))
        if best is None or r.fun < best.fun: best = r
    se2, r_, sg2 = best.x
    return se2, r_, sg2, se2/(se2+sg2/(1-r_**2))

rng = np.random.default_rng(20260715); B = 1000
out, draws = [], {}
scopes = [("pooled", np.ones(len(Yraw),bool), None),
          ("H", hp==1, None), ("L", hp==0, None)]
for name, mask, _ in scopes:
    Ys = Yraw[mask]
    grp = None if name!="pooled" else None
    # Ys holds the children of one group (or all children), so demeaning by
    # year is demeaning within group and year (within year for the pooled sample)
    Y0 = demean(Ys, None)
    om0, n0 = moments(Y0)
    pt = fit(om0, n0)
    bs = np.empty((B,4)); n = Ys.shape[0]
    t0=time.time()
    for b in range(B):
        idx = rng.integers(0,n,n)
        Yb = demean(Ys[idx], None)
        omb, nb = moments(Yb)
        bs[b] = fit(omb, nb)
    se = bs.std(0)
    draws[name] = bs
    out.append(dict(group=name, eta2=pt[0], eta2_se=se[0], rho=pt[1], rho_se=se[1],
                    eps2=pt[2], eps2_se=se[2], sperm=pt[3], sperm_se=se[3], B=B))
    print(f"{name:>6}: eta2={pt[0]:.3f}({se[0]:.3f}) rho={pt[1]:.2f}({se[1]:.3f}) "
          f"eps2={pt[2]:.3f}({se[2]:.3f}) share={pt[3]:.3f}({se[3]:.3f})  [{time.time()-t0:.0f}s]")

res = pd.DataFrame(out)
res.to_csv(_os.path.join(_WSROOT,"output/tables/m7_md_cellw_boot.csv"), index=False)
gH = res[res.group=="H"].iloc[0]; gL = res[res.group=="L"].iloc[0]
for par in ["rho","sperm"]:
    gap = gH[par]-gL[par]; z = gap/np.hypot(gH[par+"_se"], gL[par+"_se"])
    print(f"H-L {par} gap = {gap:+.3f}  z = {z:+.2f}")

fig, ax = plt.subplots(figsize=(7.2,5))
for g, c, lab in [("H","#d62728","advantaged group draws"), ("L","#1f77b4","comparison group draws")]:
    bs = draws[g]
    ax.scatter(bs[:,1], bs[:,3], s=6, alpha=0.25, color=c, label=lab)
ax.set_xlabel(r"Transitory persistence $\rho_v$")
ax.set_ylabel("Permanent share")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig.savefig(_os.path.join(_WSROOT,"output/figures/fig_ridge.png"), dpi=200, bbox_inches="tight")
print("wrote figure fig_ridge.png (B=1000)")
