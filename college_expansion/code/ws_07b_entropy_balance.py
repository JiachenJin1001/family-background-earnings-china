import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Entropy balancing estimate (Hainmueller 2012) of the premium: the earnings
difference between advantaged and less advantaged children after the less advantaged
children are reweighted to match the advantaged children in parental income
and the other covariates below.

The function entropy_balance() is imported by unified/code/u8_premium_specifications.py
and unified/code/u16_identity_in_balanced_comparison.py, which apply it to the sample
of the paper. Run on its own, the script uses one row per child of panel_analysis
with non-missing HighParentOcc14, ParentInc_bar and log income.

Outcome: the child's mean log income over the waves observed.
Treatment: HighParentOcc14 (1 for advantaged children, 0 for less advantaged
children).

Balancing constraints (the weighted moments of the less advantaged children are set
equal to the moments of the advantaged children):
  - ParentInc_bar: mean and squared deviation from the sample mean
  - age of the child at the first wave observed: mean and squared deviation
  - female
  - urban residence
  - indicators for the birth cohort of the child in five-year groups
  - indicators for the province of the child, for the 15 provinces with the
    most children, with the other provinces as one group

Reports:
  tau_hat: the average effect on the treated (the premium), with a standard
           error and a 95 percent interval from a bootstrap that resamples
           children within group (200 replications)
  the difference in unweighted means
  standardized mean differences of the continuous covariates before and
  after weighting
  delta_1 from the correlated random effects regression (cre_results.csv),
  for comparison

Output: data/intermediate/ws/ws_07b_ebal_results.csv
"""

import os
import numpy as np
import pandas as pd

_MAC = _WSROOT + "/data/intermediate"
_LIN = _WSROOT + "/data/intermediate"
BASE_INT = _MAC if os.path.isdir(_MAC) else _LIN
WS_DIR = os.path.join(BASE_INT, "ws")
os.makedirs(WS_DIR, exist_ok=True)


def entropy_balance(X_ctrl, m_target, max_iter=500, tol=1e-8):
    """Entropy balancing weights, computed from the dual problem with L-BFGS-B.

    Dual objective: min over lam of log( sum_i q_i exp(Z_i' lam) ),
    where Z_i = X_ctrl_i - m_target. The gradient is w'Z, where w are the
    weights at the current lam.

    Each column of Z is divided by its standard deviation in the control
    group for numerical stability; the weights do not depend on this scaling.
    """
    from scipy.optimize import minimize
    n = X_ctrl.shape[0]
    Z_raw = X_ctrl - m_target[None, :]
    sd = Z_raw.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = Z_raw / sd  # columns with unit standard deviation
    q = np.ones(n) / n

    def w_of(lam_):
        u = Z @ lam_
        u = u - u.max()
        w_un = q * np.exp(u)
        s = w_un.sum()
        return w_un / s

    def obj_grad(lam_):
        u = Z @ lam_
        umax = u.max()
        wu = q * np.exp(u - umax)
        s = wu.sum()
        loss = umax + np.log(s)
        w = wu / s
        g = w @ Z
        return loss, g

    lam0 = np.zeros(Z.shape[1])
    res = minimize(obj_grad, lam0, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": tol, "gtol": tol})
    lam = res.x
    w = w_of(lam)
    return w, lam


def smd(treat_x, ctrl_x, w_ctrl=None):
    if w_ctrl is None:
        w = np.ones(len(ctrl_x)) / len(ctrl_x)
    else:
        w = w_ctrl
    mt = treat_x.mean()
    mc = (ctrl_x * w).sum()
    vt = treat_x.var()
    vc = ((ctrl_x - mc) ** 2 * w).sum()
    return (mt - mc) / np.sqrt(0.5 * (vt + vc) + 1e-12)


def main():
    panel = pd.read_pickle(os.path.join(BASE_INT, "panel_analysis.pkl"))
    # One row per child
    child = panel.groupby("pid", as_index=False).agg(
        log_income_bar = ("log_income", "mean"),
        HP14           = ("HighParentOcc14", "first"),
        ParentInc      = ("ParentInc_bar", "first"),
        female         = ("female", "first"),
        urban          = ("urban", "first"),
        provcd         = ("provcd", "first"),
        age_first      = ("age", "min"),
        year_first     = ("year", "min"),
        eduy           = ("eduy", "max"),  # the child's own years of schooling, maximum across waves
    )
    # Drop children with missing HP14, ParentInc or log income
    child = child.dropna(subset=["HP14", "ParentInc", "log_income_bar"])
    print(f"Sample with non-missing HP14, ParentInc, log income: N={len(child)}, N_H={(child.HP14==1).sum()},"
          f" N_L={(child.HP14==0).sum()}")
    # Birth cohort in five-year groups (birth year = first year observed minus age in that year)
    child["birth_year"] = child["year_first"] - child["age_first"]
    child["cohort"] = (child["birth_year"] // 5) * 5
    # Province: the 15 provinces with the most children, the others as one group
    top = child["provcd"].value_counts().head(15).index.tolist()
    child["provcd_top"] = np.where(child["provcd"].isin(top), child["provcd"], -1)

    # Covariate matrix: means and variances of the continuous covariates,
    # means of the indicators. The child's own years of schooling are
    # determined after family background and are the mediator in
    # ws_07c_iv_robustness.py, so they do not enter the balancing constraints.
    cont_vars = ["ParentInc", "age_first"]
    child[cont_vars] = child[cont_vars].fillna(child[cont_vars].median())

    # Indicators for cohort and province
    coh = pd.get_dummies(child["cohort"].astype(int), prefix="coh", drop_first=True).astype(float)
    prv = pd.get_dummies(child["provcd_top"].astype(int), prefix="prv", drop_first=True).astype(float)
    bin_vars = pd.concat([
        child[["female", "urban"]].astype(float).fillna(0),
        coh, prv,
    ], axis=1)
    # First moments for the indicators; first and second moments for the continuous covariates
    X_cont_lvl = child[cont_vars].astype(float).values
    X_cont_sq  = (X_cont_lvl - X_cont_lvl.mean(0)) ** 2  # squared deviation from the mean, to match the variance
    X_full = np.hstack([X_cont_lvl, X_cont_sq, bin_vars.values])
    col_names = (cont_vars + [f"{v}_sq" for v in cont_vars] + bin_vars.columns.tolist())

    treat_mask = (child["HP14"] == 1).values
    ctrl_mask  = (child["HP14"] == 0).values

    m_target = X_full[treat_mask].mean(axis=0)
    X_ctrl   = X_full[ctrl_mask]
    print(f"Solving entropy balancing for {X_full.shape[1]} moment constraints,"
          f" N_treat={treat_mask.sum()}, N_ctrl={ctrl_mask.sum()}")
    w, lam = entropy_balance(X_ctrl, m_target)

    # Average effect on the treated
    y = child["log_income_bar"].values
    y_t = y[treat_mask].mean()
    y_c_w = (y[ctrl_mask] * w).sum()
    tau_hat = y_t - y_c_w
    print(f"\nATT (entropy-balanced, mean log income): {tau_hat:+.4f}")

    # Difference in unweighted means
    tau_naive = y_t - y[ctrl_mask].mean()
    print(f"Difference in unweighted means:          {tau_naive:+.4f}")

    # Standard error: bootstrap that resamples children within group, 200 replications
    rng = np.random.default_rng(20260520)
    boot_tau = []
    first_boot_err = None
    n_treat = treat_mask.sum(); n_ctrl = ctrl_mask.sum()
    idx_t = np.where(treat_mask)[0]; idx_c = np.where(ctrl_mask)[0]
    n_reps = 200
    for _ in range(n_reps):
        bt = rng.choice(idx_t, size=n_treat, replace=True)
        bc = rng.choice(idx_c, size=n_ctrl, replace=True)
        Xt_b = X_full[bt]; Xc_b = X_full[bc]; y_b_t = y[bt]; y_b_c = y[bc]
        m_b = Xt_b.mean(0)
        try:
            w_b, _ = entropy_balance(Xc_b, m_b, max_iter=200, tol=1e-7)
            tau_b = y_b_t.mean() - (y_b_c * w_b).sum()
            if np.isfinite(tau_b):
                boot_tau.append(tau_b)
        except Exception as e:
            if first_boot_err is None:
                first_boot_err = repr(e)
            continue
    boot_tau = np.array(boot_tau)
    # With fewer than a quarter of the replications (and fewer than 50)
    # completed, the standard error and the percentile interval are not
    # computed and the script stops with an error.
    min_ok = max(50, n_reps // 4)
    if len(boot_tau) < min_ok:
        raise RuntimeError(
            f"ws_07b: only {len(boot_tau)}/{n_reps} bootstrap replicates succeeded "
            f"(need >= {min_ok}). Likely cause: near-collinear covariates in "
            f"subsampled control group. First exception: {first_boot_err}"
        )
    se_tau = boot_tau.std(ddof=1)
    ci_lo = np.quantile(boot_tau, 0.025); ci_hi = np.quantile(boot_tau, 0.975)
    print(f"  bootstrap SE = {se_tau:.4f}   95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  (n boot ok = {len(boot_tau)} / {n_reps})")

    # Balance table
    print("\nBalance (SMD treat vs ctrl):")
    print(f"  {'covariate':<22} {'pre':>8} {'post':>8}")
    for j, name in enumerate(col_names[: len(cont_vars) + 2]):  # the continuous covariates and their squared deviations
        pre  = smd(X_full[treat_mask, j], X_full[ctrl_mask, j])
        post = smd(X_full[treat_mask, j], X_full[ctrl_mask, j], w)
        print(f"  {name:<22} {pre:>+8.3f} {post:>+8.3f}")

    # delta_1 from the correlated random effects regression, for comparison
    cre = pd.read_csv(os.path.join(BASE_INT, "cre_results.csv"))
    d1 = float(cre.loc[cre["coef"] == "delta_1", "est"].iloc[0]) if "coef" in cre.columns else None
    if d1 is not None:
        print(f"\nComparison: CRE delta_1 = {d1:+.4f}")
    print(f"            entropy balancing tau_hat = {tau_hat:+.4f}")
    print(f"            (tau_hat is a difference in weighted means of the child's mean log income; "
          "delta_1 is a coefficient of the panel regression)")

    # Write the estimates
    out = {
        "tau_hat": tau_hat, "tau_se": float(se_tau),
        "tau_ci_lo": float(ci_lo), "tau_ci_hi": float(ci_hi),
        "tau_naive": tau_naive,
        "N_treat": int(treat_mask.sum()), "N_ctrl": int(ctrl_mask.sum()),
        "n_moments_matched": int(X_full.shape[1]),
        "n_boot": int(len(boot_tau)),
    }
    pd.DataFrame([out]).to_csv(os.path.join(WS_DIR, "ws_07b_ebal_results.csv"), index=False)
    print(f"\nWrote: {os.path.join(WS_DIR, 'ws_07b_ebal_results.csv')}")


if __name__ == "__main__":
    main()
