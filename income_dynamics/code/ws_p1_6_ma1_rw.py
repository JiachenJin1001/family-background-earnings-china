import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Minimum-distance estimation of two alternative models of the income process.

Input: the autocovariances of the demeaned residual y_tilde_w written by
04_method1_md.py (method1_autocov_pooled.csv, method1_autocov_H.csv for
advantaged children, method1_autocov_L.csv for less advantaged children). Both
models are fitted by minimising the sum of squared deviations weighted by the
number of children behind each autocovariance.

(1) Permanent effect plus MA(1) transitory component:
    v_it = eps_it + psi * eps_{i,t-1}.
    Autocovariances:
       Var(v) = sigma_eps^2 * (1 + psi^2)
       Cov(v_t, v_{t-1}) = psi * sigma_eps^2
       Cov(v_t, v_{t-k}) = 0  for k >= 2.

(2) Random-walk permanent component: eta_it = eta_{i,t-1} + u_it, with a
    transitory component that is uncorrelated over time:
       Cov(y_t, y_s) = sigma_eta0^2 + min(t, s) * sigma_u^2 + I{t=s}*sigma_eps^2.

Output: data/intermediate/ws/ws_p1_6_ma1_rw.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

_MAC = _WSROOT + ""
_LIN = _WSROOT + ""
BASE = _MAC if os.path.isdir(_MAC) else _LIN
BASE_INT = os.path.join(BASE, "data/intermediate")
WS_DIR   = os.path.join(BASE_INT, "ws")


def ma1_model(k, sigma_eta2, sigma_eps2, psi):
    """Autocovariance at distance k (in waves, two years apart) implied by the permanent effect plus MA(1) transitory component."""
    if k == 0:
        return sigma_eta2 + sigma_eps2 * (1 + psi ** 2)
    if k == 1:
        return sigma_eta2 + psi * sigma_eps2
    return sigma_eta2  # k >= 2


def rw_model(t, s, sigma_eta0_2, sigma_u2, sigma_eps2):
    """Autocovariance at waves t and s implied by the random-walk permanent component.

    Cov(y_t, y_s) = sigma_eta0^2 + min(t, s) * sigma_u^2
                    + I{t = s} * sigma_eps^2

    t and s are wave indices from 0 to T-1: 2014 is 0, 2016 is 1, and so on.
    """
    return sigma_eta0_2 + min(t, s) * sigma_u2 + (sigma_eps2 if t == s else 0.0)


def fit_ma1(autocov):
    df = autocov.copy()
    df["k"] = ((df["s"] - df["t"]) // 2).astype(int)
    df = df[df.k >= 0]
    Nts = df["N_ts"].astype(float).values
    omega = df["omega_hat"].astype(float).values
    k_vals = df["k"].astype(int).values

    def loss(theta):
        s_eta2, s_eps2, psi = theta
        if s_eta2 < 0 or s_eps2 < 0 or abs(psi) >= 2:
            return 1e8
        pred = np.array([ma1_model(k, s_eta2, s_eps2, psi) for k in k_vals])
        return float(np.sum(Nts * (omega - pred) ** 2))

    best = None
    for s0 in [(0.10, 0.30, 0.20), (0.05, 0.40, -0.10), (0.15, 0.25, 0.30)]:
        r = minimize(loss, s0, method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 5000})
        if best is None or r.fun < best.fun:
            best = r
    return {"sigma_eta2": best.x[0], "sigma_eps2": best.x[1],
            "psi": best.x[2], "J": best.fun, "n_mom": len(df)}


def fit_rw(autocov):
    """Fit of the random-walk model. The wave indices t and s run from 0
    (2014) to 4 (2022)."""
    df = autocov.copy()
    df["t_idx"] = ((df["t"] - 2014) // 2).astype(int)
    df["s_idx"] = ((df["s"] - 2014) // 2).astype(int)
    Nts = df["N_ts"].astype(float).values
    omega = df["omega_hat"].astype(float).values
    t_idx = df["t_idx"].astype(int).values
    s_idx = df["s_idx"].astype(int).values

    def loss(theta):
        s_eta0_2, s_u2, s_eps2 = theta
        if any(x < 0 for x in theta):
            return 1e8
        pred = np.array([rw_model(t, s, s_eta0_2, s_u2, s_eps2)
                         for t, s in zip(t_idx, s_idx)])
        return float(np.sum(Nts * (omega - pred) ** 2))

    best = None
    for s0 in [(0.08, 0.02, 0.30), (0.05, 0.05, 0.20), (0.10, 0.01, 0.40)]:
        r = minimize(loss, s0, method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 5000})
        if best is None or r.fun < best.fun:
            best = r
    return {"sigma_eta0_2": best.x[0], "sigma_u2": best.x[1],
            "sigma_eps2": best.x[2], "J": best.fun, "n_mom": len(df)}


def main():
    rows = []
    for gtype in ["pooled", "H", "L"]:
        fp = os.path.join(BASE_INT, f"method1_autocov_{gtype}.csv")
        if not os.path.exists(fp):
            continue
        autocov = pd.read_csv(fp)
        print(f"\n=== Group {gtype}  (n moments = {len(autocov)}) ===")
        r_ma  = fit_ma1(autocov)
        r_rw  = fit_rw(autocov)
        print(f"  MA(1) transitory:")
        print(f"    σ_η²={r_ma['sigma_eta2']:.4f}  σ_ε²={r_ma['sigma_eps2']:.4f}  "
              f"ψ={r_ma['psi']:.4f}  J={r_ma['J']:.3f}")
        print(f"  Random-walk permanent:")
        print(f"    σ_η₀²={r_rw['sigma_eta0_2']:.4f}  σ_u²={r_rw['sigma_u2']:.4f}  "
              f"σ_ε²={r_rw['sigma_eps2']:.4f}  J={r_rw['J']:.3f}")
        rows.append({
            "group": gtype,
            "MA1_sigma_eta2": r_ma["sigma_eta2"], "MA1_sigma_eps2": r_ma["sigma_eps2"],
            "MA1_psi": r_ma["psi"], "MA1_J": r_ma["J"],
            "RW_sigma_eta0_2": r_rw["sigma_eta0_2"], "RW_sigma_u2": r_rw["sigma_u2"],
            "RW_sigma_eps2": r_rw["sigma_eps2"], "RW_J": r_rw["J"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(WS_DIR, "ws_p1_6_ma1_rw.csv"), index=False)
    print(f"\nWrote: {os.path.join(WS_DIR, 'ws_p1_6_ma1_rw.csv')}")

    # Criterion J of the three models (the smaller the value, the closer the fit)
    print("\n========= Criterion J by model (smaller = closer fit) =========")
    fe_j = {"pooled": 29.65, "H": 2.75, "L": 30.34}
    for r in rows:
        gt = r["group"]
        print(f"  {gt}: FE+AR(1) J={fe_j.get(gt,'?'):>6.2f}  vs  FE+MA(1) J={r['MA1_J']:>6.2f}  "
              f"vs  RW permanent J={r['RW_J']:>6.2f}")


if __name__ == "__main__":
    main()
