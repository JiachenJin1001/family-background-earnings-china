import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Stochastic EM estimator of Arellano, Blundell and Bonhomme (2017) for the
nonlinear persistence of residual log income.

Sample: child-year rows of panel_resid (children aged 22 to 55, waves 2014 to
2022) with non-missing y_tilde_w and HighParentOcc14, children with at least
three such rows. Estimated for the pooled sample, advantaged children (H) and
less advantaged children (L). A group with more than 500 children is reduced to a
random subsample of 500 children.

State-space model
-----------------
    y_tilde_it = eta_it + nu_it,                    nu_it normal (0, sigma_nu^2)
    eta_it     = Q_eta(eta_{i,t-1}, U_it),          U_it uniform on (0, 1)

Sieve representation
--------------------
    Q_eta(eta, tau)  = sum_{k=0..K} sum_{l=0..L}  a_kl * phi_k(eta) * psi_l(tau)

  phi_k(eta) = normalised Hermite polynomials in the standardized eta
  psi_l(tau) = Bernstein basis polynomials on [0, 1]

Algorithm
---------
Starting values: the standard deviation of nu is half the standard deviation
of y_tilde_w, and Q_eta(eta, tau) = eta.

In each of 6 iterations:
   E-step:
     For each child, a Metropolis-Hastings chain of 30 steps on the path
     (eta_i1, ..., eta_iT_i), started at the observed y. Each step proposes a
     normal random-walk change of one wave chosen at random. The transition
     density of eta is approximated by a normal density with the mean and
     three times the variance of the fitted quantiles at the five values of
     tau. The last 4 states of the chain are kept.
   M-step:
     Q_eta is fitted again by quantile regression on the sieve, pooling the
     pairs (eta_{t-1}, eta_t) of the kept paths, at tau in
     {0.1, 0.3, 0.5, 0.7, 0.9}. sigma_nu is the root mean square of y - eta.

The sieve has K = 2 and L = 3 (12 coefficients a_kl) and the grid of tau has
five points.

Output: data/intermediate/ws/ws_05b_abb_stochasticEM_surface.csv, the local
persistence rho(eta, tau), the derivative of Q_eta with respect to eta, by
group.
"""

import os
import math
import numpy as np
import pandas as pd
from scipy.special import eval_hermitenorm
from scipy.optimize import minimize

_MAC = _WSROOT + "/data/intermediate"
_LIN = _WSROOT + "/data/intermediate"
BASE_INT = _MAC if os.path.isdir(_MAC) else _LIN
WS_DIR = os.path.join(BASE_INT, "ws")
os.makedirs(WS_DIR, exist_ok=True)


# ============================================================
# Sieve basis
# ============================================================

def hermite_basis(eta, K, mu=0.0, sd=1.0):
    """Normalised Hermite polynomials of degree 0 to K, evaluated at the
    standardized input (eta - mu)/sd.  Returns an (N, K+1) array.
    """
    x = (eta - mu) / sd
    out = np.zeros((len(np.atleast_1d(eta)), K + 1))
    for k in range(K + 1):
        out[:, k] = eval_hermitenorm(k, x) / np.sqrt(math.factorial(k))
    return out


def bernstein_basis(tau, L):
    """Bernstein polynomials of degree L at tau in [0,1].  Returns an (N, L+1) array."""
    from math import comb
    t = np.atleast_1d(tau)
    out = np.zeros((len(t), L + 1))
    for l in range(L + 1):
        out[:, l] = comb(L, l) * (t ** l) * ((1 - t) ** (L - l))
    return out


def sieve_design(eta, tau, K, L, mu, sd):
    """Design matrix of the tensor-product sieve at the points (eta_n, tau_n).
    Returns an (N, (K+1)*(L+1)) matrix.
    """
    phi = hermite_basis(eta, K, mu, sd)   # (N, K+1)
    psi = bernstein_basis(tau, L)          # (N, L+1)
    N = phi.shape[0]
    A = (phi[:, :, None] * psi[:, None, :]).reshape(N, -1)
    return A


# ============================================================
# E-step: Metropolis-Hastings sampling of the latent paths of eta
# ============================================================

def predictive_moments_vec(eta_prev_arr, a_kl, K, L, mu_eta, sd_eta,
                           tau_grid=np.array([0.1, 0.3, 0.5, 0.7, 0.9])):
    """Mean and variance of the normal approximation to the transition
    density, for an array of values of eta in the preceding wave: the mean of
    the fitted quantiles over tau_grid and three times their variance (not
    below 1e-4). Returns mu_pred, var_pred, each of shape (len(eta_prev_arr),).
    """
    n = len(eta_prev_arr)
    G = len(tau_grid)
    eta_long = np.repeat(eta_prev_arr, G)
    tau_long = np.tile(tau_grid, n)
    A = sieve_design(eta_long, tau_long, K, L, mu_eta, sd_eta)
    q_vals = (A @ a_kl).reshape(n, G)
    mu_pred = q_vals.mean(axis=1)
    var_pred = np.maximum(1e-4, q_vals.var(axis=1) * 3.0)
    return mu_pred, var_pred


def sample_eta_paths(y, theta, K, L, mu_eta, sd_eta, n_paths=4,
                     mh_steps=30, prop_sd=None, rng=None):
    """Metropolis-Hastings sampler of the path of eta given y for one child.
    Each step proposes a normal random-walk change of one wave chosen at
    random. Returns the last n_paths states of the chain.
    """
    T = len(y)
    if prop_sd is None:
        prop_sd = max(0.10, 0.30 * sd_eta)
    if rng is None:
        rng = np.random.default_rng(0)
    sigma_nu = theta["sigma_nu"]
    a_kl     = theta["a_kl"]
    inv_2s2  = 1.0 / (2 * sigma_nu ** 2)

    def log_target_vec(eta_path):
        m = -inv_2s2 * np.sum((y - eta_path) ** 2)
        if T >= 2:
            mu_p, var_p = predictive_moments_vec(
                eta_path[:-1], a_kl, K, L, mu_eta, sd_eta)
            m += -np.sum((eta_path[1:] - mu_p) ** 2 / (2 * var_p))
            m += -0.5 * np.sum(np.log(var_p))
        return m

    eta = y.copy()
    cur_ll = log_target_vec(eta)
    paths = []
    for step in range(mh_steps):
        # Random walk on one wave, chosen uniformly at random
        t_idx = rng.integers(0, T)
        prop_eta = eta.copy()
        prop_eta[t_idx] += rng.normal(0.0, prop_sd)
        new_ll = log_target_vec(prop_eta)
        if np.log(rng.uniform()) < new_ll - cur_ll:
            eta = prop_eta
            cur_ll = new_ll
        if step >= mh_steps - n_paths:
            paths.append(eta.copy())
    return np.array(paths)  # (n_paths, T_i)


# ============================================================
# M-step: quantile regression on the sieve
# ============================================================

def m_step_fit_sieve(eta_lag, eta_cur, tau_grid, K, L, mu_eta, sd_eta,
                     weight_per_path=1.0):
    """Fit a_kl by minimising the mean check-function loss jointly over all
    tau in tau_grid; tau enters the design through the Bernstein basis."""
    # Stacked design: the pairs (eta_lag, eta_cur) are repeated for every tau
    N = len(eta_lag)
    tau_long = np.repeat(tau_grid, N)
    eta_lag_long = np.tile(eta_lag, len(tau_grid))
    eta_cur_long = np.tile(eta_cur, len(tau_grid))
    A = sieve_design(eta_lag_long, tau_long, K, L, mu_eta, sd_eta)

    def loss(a):
        u = eta_cur_long - A @ a
        # check-function loss
        return float(np.mean(np.maximum(tau_long * u, (tau_long - 1) * u)))

    a0 = np.zeros(A.shape[1])
    a0[0] = eta_cur.mean()  # starting value: mean of eta on the first basis term
    res = minimize(loss, a0, method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-7})
    return res.x


# ============================================================
# Estimation for one group
# ============================================================

def run_abb(panel, group_label, group_mask, K=2, L=3,
            n_iter=8, n_paths=4, mh_steps=40, seed=20260520,
            max_children=500):
    rng = np.random.default_rng(seed)
    sub = panel.loc[group_mask].sort_values(["pid", "year"])
    # One sequence of residuals per child; children with fewer than 3 valid waves are dropped
    child_y = sub.groupby("pid")["y_tilde_w"].apply(list).to_dict()
    child_y = {k: np.array(v) for k, v in child_y.items() if len(v) >= 3}
    # Random subsample of max_children children when the group is larger
    if len(child_y) > max_children:
        keys = list(child_y.keys())
        sel = rng.choice(len(keys), size=max_children, replace=False)
        child_y = {keys[i]: child_y[keys[i]] for i in sel}
    N_i = len(child_y); T_total = sum(len(v) for v in child_y.values())
    print(f"  [{group_label}] N children = {N_i}, T_total = {T_total}")

    # Mean and standard deviation used to standardize eta
    y_all = np.concatenate(list(child_y.values()))
    mu_eta, sd_eta = float(y_all.mean()), float(y_all.std())

    # Starting values
    sigma_nu = 0.5 * sd_eta   # the variance of nu is one quarter of the variance of y
    a_kl     = np.zeros((K + 1) * (L + 1))
    # Starting value of the sieve: eta_t = eta_{t-1}.
    # The coefficient a_kl is stored at position k*(L+1) + l, since
    # A[n, k*(L+1)+l] = phi_k(eta) * psi_l(tau); k = 1 is the term linear in eta.
    # The Bernstein polynomials sum to one over l, so a_{1,l} equal to the
    # same constant for all l gives a function of eta that does not vary with tau.
    for l in range(L + 1):
        a_kl[1 * (L + 1) + l] = sd_eta  # phi_1 is the standardized eta
    theta = {"a_kl": a_kl, "sigma_nu": sigma_nu}

    tau_grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    history = []
    for it in range(n_iter):
        # E-step: sample the latent paths
        all_lag, all_cur = [], []
        nu_resids = []
        for pid, y in child_y.items():
            paths = sample_eta_paths(
                y, theta, K, L, mu_eta, sd_eta,
                n_paths=n_paths, mh_steps=mh_steps, rng=rng,
            )
            for path in paths:
                all_lag.append(path[:-1])
                all_cur.append(path[1:])
                nu_resids.append(y - path)
        eta_lag = np.concatenate(all_lag)
        eta_cur = np.concatenate(all_cur)
        nu_resids = np.concatenate(nu_resids)

        # M-step: fit the sieve and sigma_nu
        a_new   = m_step_fit_sieve(eta_lag, eta_cur, tau_grid, K, L, mu_eta, sd_eta)
        s_new   = max(1e-3, float(np.sqrt(np.mean(nu_resids ** 2))))

        change = np.linalg.norm(a_new - theta["a_kl"])
        theta["a_kl"]     = a_new
        theta["sigma_nu"] = s_new
        history.append({"iter": it, "a_norm_change": change,
                        "sigma_nu": s_new, "n_pairs": len(eta_lag)})
        print(f"    iter {it:2d}:  σ_ν={s_new:.4f},  Δa_kl L2={change:.4f},"
              f"  n_pairs={len(eta_lag)}")

    return {
        "theta":    theta,
        "history":  history,
        "mu_eta":   mu_eta,
        "sd_eta":   sd_eta,
        "K":        K,
        "L":        L,
        "tau_grid": tau_grid,
    }


def extract_persistence_surface(fit, y_pcts=(10, 25, 50, 75, 90), taus=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """Local persistence rho(eta, tau), the derivative of Q_eta with respect
    to eta, by central differences of the fitted sieve. The points of
    evaluation are mu_eta + sd_eta * c, where c is the percentile (y_pcts) of
    an evenly spaced grid on [-2, 2]."""
    K, L = fit["K"], fit["L"]
    mu_eta, sd_eta = fit["mu_eta"], fit["sd_eta"]
    a_kl = fit["theta"]["a_kl"]
    eta_grid = np.array([mu_eta + (sd_eta * np.percentile(np.linspace(-2, 2, 1000), p))
                          for p in y_pcts])
    rows = []
    h = 0.05 * sd_eta
    for tau in taus:
        for j, eta_v in enumerate(eta_grid):
            A_plus  = sieve_design(np.array([eta_v + h]), np.array([tau]), K, L, mu_eta, sd_eta)
            A_minus = sieve_design(np.array([eta_v - h]), np.array([tau]), K, L, mu_eta, sd_eta)
            rho_loc = float((A_plus - A_minus) @ a_kl / (2 * h))
            rows.append({"tau": tau, "y_percentile": y_pcts[j],
                         "eta_value": float(eta_v), "rho_local": rho_loc})
    return pd.DataFrame(rows)


def main():
    panel = pd.read_pickle(os.path.join(BASE_INT, "panel_resid.pkl"))
    hp_col = "HighParentOcc14"
    panel = panel.dropna(subset=["y_tilde_w", "year", "pid", hp_col]).copy()
    panel["year"] = panel["year"].astype(int)

    all_surfs = []
    for gname, mask in [
        ("pooled", np.ones(len(panel), dtype=bool)),
        ("H",      (panel[hp_col] == 1).values),
        ("L",      (panel[hp_col] == 0).values),
    ]:
        print(f"\n--- ABB stochastic-EM, group = {gname} ---")
        fit = run_abb(panel, gname, mask, K=2, L=3,
                      n_iter=6, n_paths=4, mh_steps=30,
                      seed=20260520 + (1 if gname == "H" else (2 if gname == "L" else 0)))
        surf = extract_persistence_surface(fit)
        surf["group"] = gname
        surf["sigma_nu"] = fit["theta"]["sigma_nu"]
        all_surfs.append(surf)
        print(f"  Final σ_ν = {fit['theta']['sigma_nu']:.4f}   "
              f"(ratio of the variance of nu to the variance of y: {fit['theta']['sigma_nu']**2 / (fit['sd_eta']**2):.3f})")

    out = pd.concat(all_surfs, ignore_index=True)
    out_path = os.path.join(WS_DIR, "ws_05b_abb_stochasticEM_surface.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")

    # Table of rho by group and tau
    pivot = out.pivot_table(index=["group", "tau"],
                            columns="y_percentile",
                            values="rho_local", aggfunc="first").round(3)
    print("\nStochastic-EM local persistence ρ(y_pct, τ):")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
