"""Computations behind Part II of the online appendix (formal results).

Every number printed in Part II that is not copied from the paper comes from
this script. Outputs (all CSV, in ../output/):

  A_eta_bootstrap.csv      share of the coefficient on permanent income
                           recovered by an average of T waves (independence
                           approximation and exact formula) at the pooled
                           minimum-distance estimates, with percentile
                           intervals from a bootstrap that resamples children
                           (1,000 replications), T = 1 to 10
  A_eta_Tstar.csv          number of waves needed to recover two thirds of the
                           coefficient under both formulas
  A_rank_arcsin.csv        closed form for the attenuation of rank-rank slopes,
                           arcsin(r sqrt(share)/2) / arcsin(r/2), against the
                           simulated factors of m8_rank_attenuation.csv
  B_J_decomposition.csv    the minimum-distance criterion split into a
                           stationarity part (moments at the same lag that
                           differ across calendar dates) and a shape part
                           (the five lag means against the fitted curve), by
                           group
  B_ridge_curvature.csv    h_k(rho) = 1 - rho^k - k rho^(k-1)(1 - rho), the
                           slope of the lag-k autocorrelation along the level
                           set of the lag-1 autocorrelation, at the group
                           estimates

The minimum-distance estimator is the one of m7_md_cellweighted_boot.py
(pooled sample: residuals demeaned within year; weights equal to the number of
children behind each moment; Nelder-Mead from three starting values), so the
point estimates reproduce the table of the paper to the printed digit.
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, ".."))
WS = os.path.abspath(os.path.join(APP, ".."))
WSM = next(os.path.join(WS, n) for n in ("income_dynamics", "income_dynamics") if os.path.isdir(os.path.join(WS, n)))
OUT = os.path.join(APP, "output")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260908)
B = 1000

# --------------------------------------------------------------------------
# Data: residual panel, five waves 2014-2022, locked sample
# --------------------------------------------------------------------------
d = pd.read_pickle(os.path.join(WSM, "data/intermediate/panel_resid.pkl"))
d = d.dropna(subset=["y_tilde_w", "HighParentOcc14"]).copy()
d["HP14"] = d["HighParentOcc14"].astype(int)
years = sorted(d["year"].unique())
T = len(years)
piv = d.pivot_table(index="pid", columns="year", values="y_tilde_w")
hp = d.groupby("pid")["HP14"].first().reindex(piv.index).values
Yraw = piv.values
PAIRS = [(i, j) for i in range(T) for j in range(i, T)]
LAGS = np.array([j - i for i, j in PAIRS])
print(f"panel: {Yraw.shape[0]} children, waves {years}, {len(PAIRS)} pairwise moments")


def demean(Y):
    return Y - np.nanmean(Y, 0)


def moments(Y):
    om = np.full(len(PAIRS), np.nan)
    n = np.zeros(len(PAIRS))
    for k, (i, j) in enumerate(PAIRS):
        p = Y[:, i] * Y[:, j]
        ok = ~np.isnan(p)
        c = ok.sum()
        if c >= 2:
            om[k] = p[ok].mean()
            n[k] = c
    return om, n


def model(th):
    se2, r, sg2 = th
    return np.array([se2 + (r ** (j - i)) * sg2 / (1 - r ** 2) for i, j in PAIRS])


def fit(om, w):
    mask = ~np.isnan(om)
    ww = w[mask]

    def obj(th):
        if not (0 < th[0] < 2 and -0.95 < th[1] < 0.95 and 0 < th[2] < 2):
            return 1e9
        e = (om - model(th))[mask]
        return float(e @ (ww * e))

    best = None
    for x0 in [(0.1, 0.2, 0.3), (0.05, 0.4, 0.35), (0.15, 0.05, 0.3)]:
        r = minimize(obj, x0, method="Nelder-Mead",
                     options=dict(maxiter=3000, xatol=1e-7, fatol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


# --------------------------------------------------------------------------
# Attenuation formulas
# --------------------------------------------------------------------------
def f_T(rho, t):
    """Variance inflation factor of a T-wave mean of a stationary AR(1),
    relative to sigma_v^2 / T: f_T = 1 + 2 sum_{k<T} (1 - k/T) rho^k."""
    return 1.0 + 2.0 * sum((1.0 - k / t) * rho ** k for k in range(1, t))


def f_T_closed(rho, t):
    """Closed form: (1+rho)/(1-rho) - 2 rho (1 - rho^T) / (T (1-rho)^2)."""
    return (1 + rho) / (1 - rho) - 2 * rho * (1 - rho ** t) / (t * (1 - rho) ** 2)


def eta(se2, sg2, rho, t, exact):
    sv2 = sg2 / (1 - rho ** 2)
    f = f_T(rho, t) if exact else 1.0
    return se2 / (se2 + sv2 * f / t)


def t_star(se2, sg2, rho, c, exact, tmax=60):
    for t in range(1, tmax + 1):
        if eta(se2, sg2, rho, t, exact) >= c:
            return t
    return np.nan


# check closed form against the sum
_m7all = pd.read_csv(os.path.join(WSM, "output/tables/m7_md_cellw_boot.csv")).set_index("group")
_arma = pd.read_csv(os.path.join(WSM, "data/intermediate/ws/ws_method1_arma11.csv")).set_index("group")
RHO_POINTS = {"pooled": round(float(_m7all.loc["pooled", "rho"]), 2), "H": round(float(_m7all.loc["H", "rho"]), 2), "L": round(float(_m7all.loc["L", "rho"]), 2),
              "ARMA root": round(float(_arma.loc["pooled", "rho_arma"]), 2)}   # the estimates of this run, to two decimals
for rho in (RHO_POINTS["pooled"], RHO_POINTS["H"], RHO_POINTS["ARMA root"]):
    for t in (1, 2, 5, 9):
        assert abs(f_T(rho, t) - f_T_closed(rho, t)) < 1e-12
print("closed form for f_T verified")

# --------------------------------------------------------------------------
# A. eta_T bootstrap (pooled, cell-weighted MD)
# --------------------------------------------------------------------------
Y0 = demean(Yraw)
om0, n0 = moments(Y0)
pt, J0 = fit(om0, n0)
se2_hat, rho_hat, sg2_hat = pt
print(f"pooled MD point: sigma_eta2={se2_hat:.4f} rho={rho_hat:.4f} "
      f"sigma_eps2={sg2_hat:.4f} share={se2_hat/(se2_hat+sg2_hat/(1-rho_hat**2)):.4f}")

TMAX = 10
Tgrid = np.arange(1, TMAX + 1)
draws = np.empty((B, 3))
t0 = time.time()
n = Yraw.shape[0]
for b in range(B):
    idx = rng.integers(0, n, n)
    omb, nb = moments(demean(Yraw[idx]))
    draws[b] = fit(omb, nb)[0]
    if (b + 1) % 250 == 0:
        print(f"  bootstrap {b+1}/{B}  {time.time()-t0:.0f}s")

rows = []
for t in Tgrid:
    for exact in (False, True):
        e_pt = eta(se2_hat, sg2_hat, rho_hat, int(t), exact)
        e_b = np.array([eta(x[0], x[2], x[1], int(t), exact) for x in draws])
        lo, hi = np.percentile(e_b, [2.5, 97.5])
        rows.append(dict(T=int(t), formula="exact" if exact else "approx",
                         eta=e_pt, eta_lo=lo, eta_hi=hi, eta_se=e_b.std(ddof=1),
                         inflation=1 / e_pt))
eta_df = pd.DataFrame(rows)
ts_rows = []
for exact in (False, True):
    ts_pt = t_star(se2_hat, sg2_hat, rho_hat, 2 / 3, exact)
    ts_b = np.array([t_star(x[0], x[2], x[1], 2 / 3, exact) for x in draws])
    ts_rows.append(dict(formula="exact" if exact else "approx", T_star=ts_pt,
                        T_star_p025=np.nanpercentile(ts_b, 2.5),
                        T_star_p975=np.nanpercentile(ts_b, 97.5),
                        share_of_draws_at_or_below_point=float(np.mean(ts_b <= ts_pt))))
eta_df.to_csv(os.path.join(OUT, "A_eta_bootstrap.csv"), index=False)
pd.DataFrame(ts_rows).to_csv(os.path.join(OUT, "A_eta_Tstar.csv"), index=False)
print(eta_df.pivot(index="T", columns="formula", values="eta").round(3))
print(pd.DataFrame(ts_rows))
# closed-form T* under the approximation
c = 2 / 3
sv2_hat = sg2_hat / (1 - rho_hat ** 2)
T_c = c * sv2_hat / ((1 - c) * se2_hat)
print(f"closed-form T_c (approx) = {T_c:.3f} -> ceil = {int(np.ceil(T_c))}")

# --------------------------------------------------------------------------
# A. rank-rank attenuation: arcsin formula versus m8 simulation
# --------------------------------------------------------------------------
m8 = pd.read_csv(os.path.join(WSM, "output/tables/m8_rank_attenuation.csv"))
r_true = 0.30           # m8 design: child permanent = 0.3 * parent permanent + e, equal variances
_m7 = pd.read_csv(os.path.join(WSM, "output/tables/m7_md_cellw_boot.csv")).set_index("group").loc["pooled"]   # the estimates the simulation of m8 uses
eta3_exact = eta(_m7["eta2"], _m7["eta2"] * (1 / _m7["sperm"] - 1) * (1 - _m7["rho"] ** 2), _m7["rho"], 3, True)
rows = []
for _, m in m8.iterrows():
    e_T = m["eta_T"]  # m8 already uses the exact variance of the mean
    one = np.arcsin(r_true * np.sqrt(e_T) / 2) / np.arcsin(r_true / 2)
    two = np.arcsin(r_true * np.sqrt(e_T) * np.sqrt(eta3_exact) / 2) / np.arcsin(r_true / 2)
    rows.append(dict(T=int(m["T"]), eta_T=e_T, sqrt_eta=np.sqrt(e_T),
                     arcsin_one_sided=one, sim_one_sided=m["rank_factor"],
                     arcsin_two_sided=two, sim_two_sided=m["rank_factor_2sided"]))
rank_df = pd.DataFrame(rows)
rank_df.to_csv(os.path.join(OUT, "A_rank_arcsin.csv"), index=False)
print(rank_df.round(3))
print("max |arcsin - simulation|, one-sided:",
      float((rank_df.arcsin_one_sided - rank_df.sim_one_sided).abs().max()),
      " two-sided:", float((rank_df.arcsin_two_sided - rank_df.sim_two_sided).abs().max()))

# --------------------------------------------------------------------------
# B. J decomposition: stationarity part + shape part, by group
# --------------------------------------------------------------------------
def decompose(om, w):
    th, Jtot = fit(om, w)
    mask = ~np.isnan(om)
    omv, wv, lag = om[mask], w[mask], LAGS[mask]
    Ntot = wv.sum()
    # cell weights are pair counts; the criterion in the paper is
    # sum_p (n_p / N) (omega_p - m_p)^2, so J = N * criterion = sum_p n_p e_p^2
    J_total = float(np.sum(wv * (omv - model(th)[mask]) ** 2))
    J_stat, J_shape = 0.0, 0.0
    lagmeans = {}
    for k in np.unique(lag):
        sel = lag == k
        Wk = wv[sel].sum()
        obar = np.sum(wv[sel] * omv[sel]) / Wk
        lagmeans[int(k)] = (obar, Wk)
        J_stat += float(np.sum(wv[sel] * (omv[sel] - obar) ** 2))
        mk = th[0] + (th[1] ** k) * th[2] / (1 - th[1] ** 2)
        J_shape += float(Wk * (obar - mk) ** 2)
    return th, J_total, J_stat, J_shape, lagmeans


rows = []
for name, mask in [("pooled", np.ones(n, bool)), ("H", hp == 1), ("L", hp == 0)]:
    om, w = moments(demean(Yraw[mask]))
    th, Jt, Js, Jsh, lm = decompose(om, w)
    assert abs(Jt - (Js + Jsh)) < 1e-6 * max(1.0, Jt), (Jt, Js, Jsh)
    rows.append(dict(group=name, N=int(mask.sum()), sigma_eta2=th[0], rho=th[1],
                     sigma_eps2=th[2], J_total=Jt, J_stationarity=Js, J_shape=Jsh,
                     share_stationarity=Js / Jt, n_moments=int((~np.isnan(om)).sum()),
                     df_stationarity=int((~np.isnan(om)).sum()) - len(lm),
                     df_shape=len(lm) - 3,
                     **{f"lagmean_{k}": v[0] for k, v in lm.items()}))
Jdf = pd.DataFrame(rows)
Jdf.to_csv(os.path.join(OUT, "B_J_decomposition.csv"), index=False)
print(Jdf[["group", "N", "sigma_eta2", "rho", "sigma_eps2", "J_total",
           "J_stationarity", "J_shape", "share_stationarity", "df_stationarity", "df_shape"]].round(3))

# --------------------------------------------------------------------------
# B. ridge curvature h_k(rho) at the group estimates
# --------------------------------------------------------------------------
def h(rho, k):
    return 1 - rho ** k - k * rho ** (k - 1) * (1 - rho)


rows = []
for name, rho in RHO_POINTS.items():
    rows.append(dict(group=name, rho=rho, h2=h(rho, 2), h3=h(rho, 3), h4=h(rho, 4)))
    assert abs(h(rho, 2) - (1 - rho) ** 2) < 1e-12
    assert abs(h(rho, 3) - (1 - rho) ** 2 * (1 + 2 * rho)) < 1e-12
    assert abs(h(rho, 4) - (1 - rho) ** 2 * (1 + 2 * rho + 3 * rho ** 2)) < 1e-12
pd.DataFrame(rows).to_csv(os.path.join(OUT, "B_ridge_curvature.csv"), index=False)
print(pd.DataFrame(rows).round(3))

print("done")
