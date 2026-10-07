import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Minimum-distance estimation of the income process with an ARMA(1,1)
transitory component, for comparison with the AR(1) transitory component.

Input: the autocovariances of the demeaned residual y_tilde_w written by
04_method1_md.py (method1_autocov_pooled.csv, method1_autocov_H.csv for
advantaged children, method1_autocov_L.csv for less advantaged children).

Model, with waves two years apart:
    y_it = eta_i + v_it,
    v_it = rho * v_{i,t-1} + eps_it + psi * eps_{i,t-1}.

Implied autocovariances (k = distance in waves):
    gamma_0  = sigma_eps^2 * (1 + psi^2 + 2*rho*psi) / (1 - rho^2)
    gamma_1  = rho * gamma_0 + psi * sigma_eps^2
    gamma_k  = rho^(k-1) * gamma_1   for k >= 1
    omega_k  = sigma_eta^2 + gamma_k  for k >= 0

The criterion is the sum of squared deviations weighted by the number of
children behind each autocovariance,
    J = sum_{t,s} N_ts * (omega_hat_ts - omega_k)^2,  k = (s-t)/2.
The model with psi = 0 (AR(1) transitory component) is fitted with the same
criterion, and the difference between the two values of J is reported.

Output:
    data/intermediate/ws/ws_method1_arma11.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

_MAC = _WSROOT + "/data/intermediate"
_LIN = _WSROOT + "/data/intermediate"
BASE_INT = _MAC if os.path.isdir(_MAC) else _LIN
OUT_DIR  = os.path.join(BASE_INT, "ws")
os.makedirs(OUT_DIR, exist_ok=True)


def model_omega(k, sigma_eta2, rho, sigma_eps2, psi):
    """Autocovariance at distance k (in waves) implied by the permanent effect plus ARMA(1,1) transitory component."""
    if abs(rho) >= 0.999:
        return np.nan
    g0 = sigma_eps2 * (1.0 + psi**2 + 2.0 * rho * psi) / (1.0 - rho**2)
    if k == 0:
        return sigma_eta2 + g0
    g1 = rho * g0 + psi * sigma_eps2
    gk = (rho ** (k - 1)) * g1
    return sigma_eta2 + gk


def fit_arma11(df_autocov):
    """Weighted minimum-distance fit of the ARMA(1,1) model and of the AR(1) model to one autocovariance table (pooled, H or L)."""
    df = df_autocov.copy()
    df["k"] = ((df["s"] - df["t"]) // 2).astype(int)
    # Keep the pairs with k >= 0 (the tables hold only these)
    df = df.loc[df["k"] >= 0].reset_index(drop=True)
    Nts = df["N_ts"].astype(float).values
    omega_hat = df["omega_hat"].astype(float).values
    k_vals = df["k"].astype(int).values

    def loss(theta):
        sigma_eta2, rho, sigma_eps2, psi = theta
        # Parameters outside the admissible region receive a large value of the criterion
        if sigma_eta2 < 0 or sigma_eps2 < 0 or abs(rho) >= 0.99 or abs(psi) >= 2.0:
            return 1e8
        pred = np.array([model_omega(k, sigma_eta2, rho, sigma_eps2, psi) for k in k_vals])
        if np.any(np.isnan(pred)):
            return 1e8
        resid = omega_hat - pred
        return float(np.sum(Nts * resid ** 2))

    # Several starting points; the fit with the smallest criterion is kept
    starts = [
        (0.10, 0.30, 0.30, 0.00),
        (0.05, 0.20, 0.35, 0.20),
        (0.15, 0.10, 0.25, -0.20),
        (0.10, 0.50, 0.30, 0.10),
        (0.08, 0.05, 0.40, 0.50),
    ]
    best = None
    first_arma_error = None
    for s in starts:
        try:
            res = minimize(
                loss, s, method="Nelder-Mead",
                options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 5000},
            )
            if best is None or res.fun < best.fun:
                best = res
        except Exception as e:
            if first_arma_error is None:
                first_arma_error = repr(e)
            continue
    if best is None:
        raise RuntimeError(
            "ws_method1_arma11: ARMA(1,1) fit failed at every starting point. "
            f"First exception: {first_arma_error}"
        )
    sigma_eta2, rho, sigma_eps2, psi = best.x
    # The AR(1) model (psi = 0) with the same criterion
    def loss_ar1(theta):
        s_eta, rho_, s_eps = theta
        if s_eta < 0 or s_eps < 0 or abs(rho_) >= 0.99:
            return 1e8
        return loss((s_eta, rho_, s_eps, 0.0))

    best_ar1 = None
    first_ar1_error = None
    for s in [(0.10, 0.20, 0.30), (0.05, 0.05, 0.40), (0.15, 0.40, 0.25)]:
        try:
            r = minimize(loss_ar1, s, method="Nelder-Mead",
                         options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 5000})
            if best_ar1 is None or r.fun < best_ar1.fun:
                best_ar1 = r
        except Exception as e:
            if first_ar1_error is None:
                first_ar1_error = repr(e)
    if best_ar1 is None:
        raise RuntimeError(
            "ws_method1_arma11: AR(1) comparison fit failed at every starting point. "
            f"First exception: {first_ar1_error}"
        )
    sigma_eta2_ar, rho_ar, sigma_eps2_ar = best_ar1.x

    return {
        "sigma_eta2_arma": sigma_eta2,
        "rho_arma":        rho,
        "sigma_eps2_arma": sigma_eps2,
        "psi_arma":        psi,
        "J_arma":          best.fun,
        "sigma_eta2_ar1":  sigma_eta2_ar,
        "rho_ar1":         rho_ar,
        "sigma_eps2_ar1":  sigma_eps2_ar,
        "J_ar1":           best_ar1.fun,
        "delta_J":         best_ar1.fun - best.fun,
        "n_moments":       int(len(df)),
    }


def main():
    rows = []
    for gtype in ["pooled", "H", "L"]:
        fp = os.path.join(BASE_INT, f"method1_autocov_{gtype}.csv")
        if not os.path.exists(fp):
            print(f"  Missing: {fp}")
            continue
        autocov = pd.read_csv(fp)
        res = fit_arma11(autocov)
        res["group"] = gtype
        rows.append(res)
        print(f"[{gtype}]")
        print(f"  FE+AR(1):   sigma_eta2={res['sigma_eta2_ar1']:.4f}, rho={res['rho_ar1']:.4f},"
              f" sigma_eps2={res['sigma_eps2_ar1']:.4f}, J={res['J_ar1']:.3f}")
        print(f"  FE+ARMA(1,1): sigma_eta2={res['sigma_eta2_arma']:.4f}, rho={res['rho_arma']:.4f},"
              f" sigma_eps2={res['sigma_eps2_arma']:.4f}, psi={res['psi_arma']:.4f}, J={res['J_arma']:.3f}")
        print(f"  ΔJ (AR1 - ARMA) = {res['delta_J']:.3f}   "
              f"(positive: the ARMA(1,1) criterion is smaller)")
        print()

    out = pd.DataFrame(rows)
    cols = ["group", "sigma_eta2_ar1", "rho_ar1", "sigma_eps2_ar1", "J_ar1",
            "sigma_eta2_arma", "rho_arma", "sigma_eps2_arma", "psi_arma", "J_arma",
            "delta_J", "n_moments"]
    out = out[cols]
    out_path = os.path.join(OUT_DIR, "ws_method1_arma11.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
