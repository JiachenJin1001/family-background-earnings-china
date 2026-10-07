"""Serial-correlation tests and deeper-lag instruments for the Arellano-Bond
estimates of earnings persistence.

The Arellano-Bond estimate of the persistence coefficient rho_0 is smaller
than the minimum-distance estimate of the AR(1) coefficient of the transitory
component. This script reports two diagnostics, on the five-wave panel
(panel_resid.pkl, waves 2014 to 2022) and on the seven-wave panel
(ws/panel_resid_7w.pkl, waves 2010 to 2022):
  1. The Arellano-Bond m1 and m2 tests of first-order and second-order serial
     correlation in the differenced residuals. Second-order correlation would
     invalidate the instruments lagged two waves and would indicate a moving
     average or measurement error component in levels.
  2. Difference GMM with instruments lagged three waves or more only. These
     instruments remain valid under classical measurement error or an MA(1)
     component in levels.

The GMM functions are those of m10_abgmm_table3.py (collapsed instruments,
two-step estimator, standard errors clustered by child).

Output: data/intermediate/m17_ab_diagnostics.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
BASE_INT = _os.path.join(_WSROOT, "data", "intermediate")


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


def build_diff(y_wide, hp_wide, T, min_lag, max_lag):
    rows = []
    for i in y_wide.index:
        yi = y_wide.loc[i].values
        hp = int(hp_wide.loc[i])
        for t in range(3, T + 1):
            y_t, y_tm1, y_tm2 = yi[t - 1], yi[t - 2], yi[t - 3]
            if np.isnan(y_t) or np.isnan(y_tm1) or np.isnan(y_tm2):
                continue
            instr = []
            for lag in range(min_lag, max_lag + 1):
                lev = yi[t - 1 - lag] if (t - 1 - lag) >= 0 else np.nan
                instr.append(0.0 if np.isnan(lev) else float(lev))
            rows.append(dict(pid=i, t=t, y=y_t - y_tm1,
                             x0=y_tm1 - y_tm2, x1=(y_tm1 - y_tm2) * hp,
                             Z=instr + [v * hp for v in instr]))
    return rows


def stack(rows):
    return dict(pid=np.array([r["pid"] for r in rows]),
                t=np.array([r["t"] for r in rows]),
                y=np.array([r["y"] for r in rows]),
                X=np.column_stack([[r["x0"] for r in rows],
                                   [r["x1"] for r in rows]]),
                Z=np.array([r["Z"] for r in rows]))


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
    return dict(beta=b2, se=se, resid=resid, J=J, df_J=df_J,
                pJ=1 - stats.chi2.cdf(J, df_J) if df_J > 0 else np.nan,
                n_inst=Z.shape[1], G=G, n_eq=len(y))


def m_tests(d, resid):
    """m_j statistics with clustering by child: z statistic of the sums by
    child of resid_t * resid_{t-j}, where j counts waves of the differenced
    equations."""
    df = pd.DataFrame(dict(pid=d["pid"], t=d["t"], u=resid))
    df = df.sort_values(["pid", "t"]).reset_index(drop=True)
    out = {}
    for j in (1, 2):
        lag = df.groupby("pid")["u"].shift(j)
        tlag = df.groupby("pid")["t"].shift(j)
        ok = lag.notna() & (df["t"] - tlag == j)
        prod = (df["u"] * lag)[ok]
        pids = df.loc[ok, "pid"]
        s = prod.groupby(pids).sum()
        z = s.sum() / np.sqrt((s ** 2).sum()) if len(s) else np.nan
        out[j] = (z, 2 * (1 - stats.norm.cdf(abs(z))), int(ok.sum()))
    return out


def report(label, fit):
    jtxt = (f"J = {fit['J']:.2f} (df {fit['df_J']}, p = {fit['pJ']:.3f})"
            if fit["df_J"] > 0 else "just identified")
    print(f"[{label}]")
    print(f"  rho0 = {fit['beta'][0]:+.4f} ({fit['se'][0]:.4f})   "
          f"rho1 = {fit['beta'][1]:+.4f} ({fit['se'][1]:.4f})   {jtxt}")
    print(f"  instruments = {fit['n_inst']}, children = {fit['G']:,}, "
          f"eqs = {fit['n_eq']:,}")


rows_out = []
for name, path, maxlag in [
        ("5-wave 2014-2022", os.path.join(BASE_INT, "panel_resid.pkl"), 3),
        ("7-wave 2010-2022", os.path.join(BASE_INT, "ws", "panel_resid_7w.pkl"), 4)]:
    yw, hw, T = load_panel(path)
    print(f"\n===== {name} (T = {T}) =====")
    base = stack(build_diff(yw, hw, T, 2, maxlag))
    fb = gmm_two_step(base)
    report(f"baseline, lags 2-{maxlag}", fb)
    mt = m_tests(base, fb["resid"])
    for j, (z, p, n) in mt.items():
        print(f"  m{j}: z = {z:+.2f}  (p = {p:.3f}, {n:,} pairs)")
    d3 = stack(build_diff(yw, hw, T, 3, maxlag))
    keep = (np.abs(d3["Z"]).sum(axis=1) > 0)
    print(f"  lag>=3 usable equations: {keep.sum():,} of {len(keep):,}")
    f3 = gmm_two_step(d3)
    report(f"lag >= 3 only (3-{maxlag})", f3)
    rows_out.append(dict(panel=name,
                         rho0_base=fb["beta"][0], se_base=fb["se"][0],
                         rho0_lag3=f3["beta"][0], se_lag3=f3["se"][0],
                         m1_z=mt[1][0], m1_p=mt[1][1],
                         m2_z=mt[2][0], m2_p=mt[2][1]))

out = os.path.join(BASE_INT, "m17_ab_diagnostics.csv")
pd.DataFrame(rows_out).to_csv(out, index=False)
print(f"\nwrote {out}")
