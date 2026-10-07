"""Arellano-Bond estimates of earnings persistence by family background.

Model, for the winsorized residual of log income y_tilde_w:
    y_it = rho_0 * y_{i,t-1} + rho_1 * (y_{i,t-1} * HighParentOcc14_i) + alpha_i + u_it
HighParentOcc14 = 1 for advantaged children (at least one parent in an
official, managerial, or professional occupation when the child was fourteen)
and 0 for less advantaged children.

Samples: child-year rows with non-missing y_tilde_w and HighParentOcc14 from
  - panel_resid.pkl: children aged 22 to 55, waves 2014 to 2022, at least
    three waves with positive income (five-wave panel);
  - ws/panel_resid_7w.pkl: the panel that adds the 2010 and 2012 waves, with
    income from jobs reconstructed for those two waves by ws_panel_7wave.py
    (seven-wave panel).

Estimates:
  1. Five-wave panel, two-step difference GMM. Instruments: levels lagged two
     and three waves, collapsed, and their interactions with HighParentOcc14
     (4 columns). Standard errors clustered by child, Hansen J test. Same
     estimator as 05_method2_abgmm.py.
  2. Five-wave panel, two-step system GMM: adds the equations in levels,
     instrumented by the lagged first difference, collapsed, and its
     interaction with HighParentOcc14. The instrument matrix is block diagonal
     (4 + 2 = 6 columns). The two-step weight matrix is the inverse of the sum
     over children of the outer products of the scores. Hansen J test.
  3. Seven-wave panel, two-step difference GMM with levels lagged two to four
     waves, collapsed (6 columns), as in ws_methods_7wave.py.
  4. Anderson-Rubin type confidence set for rho_1 on the moments of estimate
     1: the two-step overidentification test is inverted over a grid of
     imposed values of rho_1. With rho_1 imposed the model has one free
     parameter, so the statistic is compared with the chi-squared distribution
     with 3 degrees of freedom.

Output: output/tables/m10_abgmm_table3.csv, and a summary printed to the console
"""
import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
from scipy import stats

BASE_INT = _os.path.join(_WSROOT, "data", "intermediate")
OUT = _os.path.join(_WSROOT, "output", "tables", "m10_abgmm_table3.csv")


def load_panel(path):
    df = pd.read_pickle(path)
    df = df.dropna(subset=["y_tilde_w", "HighParentOcc14"]).copy()
    df["pid"] = df["pid"].astype(np.int64)
    df["year"] = df["year"].astype(int)
    df["HighParentOcc14"] = df["HighParentOcc14"].astype(int)
    waves = sorted(df["year"].unique().tolist())
    df["wave_idx"] = df["year"].map({y: i + 1 for i, y in enumerate(waves)})
    y_wide = df.pivot(index="pid", columns="wave_idx", values="y_tilde_w")
    hp_wide = df.groupby("pid")["HighParentOcc14"].first()
    return y_wide, hp_wide, len(waves)


def build_diff(y_wide, hp_wide, T, max_lag):
    """First-differenced equations with collapsed lagged-level instruments, lags 2 to max_lag, and their interactions with HighParentOcc14."""
    rows = []
    for i in y_wide.index:
        yi = y_wide.loc[i].values
        hp = int(hp_wide.loc[i])
        for t in range(3, T + 1):
            y_t, y_tm1, y_tm2 = yi[t - 1], yi[t - 2], yi[t - 3]
            if np.isnan(y_t) or np.isnan(y_tm1) or np.isnan(y_tm2):
                continue
            instr = []
            for lag in range(2, max_lag + 1):
                lev = yi[t - 1 - lag] if (t - 1 - lag) >= 0 else np.nan
                instr.append(0.0 if np.isnan(lev) else float(lev))
            rows.append(dict(pid=i, y=y_t - y_tm1,
                             x0=y_tm1 - y_tm2, x1=(y_tm1 - y_tm2) * hp,
                             Z=instr + [v * hp for v in instr]))
    return rows


def build_level(y_wide, hp_wide, T):
    """Equations in levels, instrumented by the lagged first difference and its interaction with HighParentOcc14."""
    rows = []
    for i in y_wide.index:
        yi = y_wide.loc[i].values
        hp = int(hp_wide.loc[i])
        for t in range(3, T + 1):
            y_t, y_tm1, y_tm2 = yi[t - 1], yi[t - 2], yi[t - 3]
            if np.isnan(y_t) or np.isnan(y_tm1) or np.isnan(y_tm2):
                continue
            dlag = y_tm1 - y_tm2
            rows.append(dict(pid=i, y=y_t, x0=y_tm1, x1=y_tm1 * hp,
                             Z=[dlag, dlag * hp]))
    return rows


def stack(rows_diff, rows_lev=None):
    """Stack the equation rows into arrays; with equations in levels, the instrument matrix is block diagonal."""
    md = len(rows_diff[0]["Z"])
    ml = len(rows_lev[0]["Z"]) if rows_lev else 0
    rows = []
    for r in rows_diff:
        rows.append((r["pid"], r["y"], r["x0"], r["x1"], r["Z"] + [0.0] * ml))
    if rows_lev:
        for r in rows_lev:
            rows.append((r["pid"], r["y"], r["x0"], r["x1"], [0.0] * md + r["Z"]))
    pid = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    X = np.column_stack([[r[2] for r in rows], [r[3] for r in rows]])
    Z = np.array([r[4] for r in rows])
    return dict(pid=pid, y=y, X=X, Z=Z)


def cluster_score_sum(Z, resid, pid):
    order = np.argsort(pid, kind="stable")
    Zs, rs, ps = Z[order], resid[order], pid[order]
    score = np.zeros((Z.shape[1], Z.shape[1]))
    start = 0
    for k in range(1, len(ps) + 1):
        if k == len(ps) or ps[k] != ps[start]:
            s = Zs[start:k].T @ rs[start:k]
            score += np.outer(s, s)
            start = k
    return score, len(np.unique(pid))


def gmm_two_step(d):
    X, Z, y, pid = d["X"], d["Z"], d["y"], d["pid"]
    ZX, Zy = Z.T @ X, Z.T @ y
    W1 = np.linalg.pinv(Z.T @ Z)
    b1 = np.linalg.solve(ZX.T @ W1 @ ZX, ZX.T @ W1 @ Zy)
    score1, _ = cluster_score_sum(Z, y - X @ b1, pid)
    W2 = np.linalg.pinv(score1)
    A = ZX.T @ W2 @ ZX
    b2 = np.linalg.solve(A, ZX.T @ W2 @ Zy)
    resid = y - X @ b2
    score2, G = cluster_score_sum(Z, resid, pid)
    Ainv = np.linalg.pinv(A)
    V = Ainv @ ZX.T @ W2 @ score2 @ W2 @ ZX @ Ainv * (G / (G - 1))
    se = np.sqrt(np.diag(V))
    g = Z.T @ resid
    J = float(g @ W2 @ g)
    df_J = Z.shape[1] - X.shape[1]
    return dict(beta=b2, se=se, J=J, df_J=df_J, pJ=1 - stats.chi2.cdf(J, df_J),
                n_inst=Z.shape[1], G=G, n_eq=len(y))


def ar_interval_rho1(d, grid=np.arange(-0.60, 0.6001, 0.005), level=0.05):
    """Invert the two-step overidentification test over imposed values of rho_1 (one free parameter)."""
    X, Z, y, pid = d["X"], d["Z"], d["y"], d["pid"]
    x0, x1 = X[:, 0], X[:, 1]
    crit = stats.chi2.ppf(1 - level, Z.shape[1] - 1)
    keep = []
    for c in grid:
        yr = y - c * x1
        Zx, Zy = Z.T @ x0, Z.T @ yr
        W1 = np.linalg.pinv(Z.T @ Z)
        b1 = (Zx @ W1 @ Zy) / (Zx @ W1 @ Zx)
        s1, _ = cluster_score_sum(Z, yr - b1 * x0, pid)
        W2 = np.linalg.pinv(s1)
        b2 = (Zx @ W2 @ Zy) / (Zx @ W2 @ Zx)
        g = Z.T @ (yr - b2 * x0)
        if float(g @ W2 @ g) <= crit:
            keep.append(c)
    if not keep:
        return np.nan, np.nan, False
    edge = (min(keep) <= grid[0] + 1e-9) or (max(keep) >= grid[-1] - 1e-9)
    contiguous = np.allclose(np.diff(keep), grid[1] - grid[0], atol=1e-9)
    if not contiguous:
        print("  WARNING: the confidence set is disconnected; its smallest and largest points are reported.")
    if edge:
        print("  WARNING: the confidence set reaches the edge of the grid.")
    return float(min(keep)), float(max(keep)), contiguous


rows_out = []

def record(label, fit):
    rows_out.append(dict(panel=label, rho0=fit["beta"][0], rho0_se=fit["se"][0],
                         rho1=fit["beta"][1], rho1_se=fit["se"][1],
                         J=fit["J"], df_J=fit["df_J"], pJ=fit["pJ"],
                         n_inst=fit["n_inst"], n_children=fit["G"],
                         n_eq=fit["n_eq"]))
    print(f"[{label}]  rho0 = {fit['beta'][0]:+.4f} ({fit['se'][0]:.4f})   "
          f"rho1 = {fit['beta'][1]:+.4f} ({fit['se'][1]:.4f})   "
          f"Hansen J = {fit['J']:.2f} (df {fit['df_J']}, p = {fit['pJ']:.3f})   "
          f"instruments = {fit['n_inst']}, children = {fit['G']:,}, eqs = {fit['n_eq']:,}")


# 1. Five-wave panel, difference GMM
y5, hp5, T5 = load_panel(_os.path.join(BASE_INT, "panel_resid.pkl"))
d5 = stack(build_diff(y5, hp5, T5, max_lag=3))
fit_d5 = gmm_two_step(d5)
record("diff GMM, 2014-2022", fit_d5)

# 2. Five-wave panel, system GMM
dsys = stack(build_diff(y5, hp5, T5, max_lag=3), build_level(y5, hp5, T5))
fit_sys = gmm_two_step(dsys)
record("system GMM, 2014-2022", fit_sys)

# 3. Seven-wave panel, difference GMM
y7, hp7, T7 = load_panel(_os.path.join(BASE_INT, "ws", "panel_resid_7w.pkl"))
d7 = stack(build_diff(y7, hp7, T7, max_lag=4))
fit_d7 = gmm_two_step(d7)
record("diff GMM, 2010-2022 reconstructed", fit_d7)

# 4. Confidence set for rho_1 (five-wave panel, difference moments)
lo, hi, contig = ar_interval_rho1(d5)
print(f"[confidence set for rho_1, five-wave diff]  95% set = [{lo:+.2f}, {hi:+.2f}]"
      f"{'' if contig else '  (smallest and largest points of a disconnected set)'}")
rows_out.append(dict(panel="AR interval rho1 (diff, 5-wave)", rho0=np.nan,
                     rho0_se=np.nan, rho1=lo, rho1_se=hi, J=np.nan,
                     df_J=np.nan, pJ=np.nan, n_inst=d5["Z"].shape[1],
                     n_children=fit_d5["G"], n_eq=fit_d5["n_eq"]))

pd.DataFrame(rows_out).to_csv(OUT, index=False)
print(f"\nwrote {OUT}")
