import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Instrumental-variable estimate of the return to college and of the premium,
with alternative samples, a placebo instrument, and inference that remains
valid with a weak instrument.

The functions of this script are imported by unified/code/u3_online_appendix.py,
which applies them to the sample of the paper. Run on its own, the script uses
one row per child of panel_analysis with non-missing HighParentOcc14, ParentInc_bar, log income, years of schooling, provincial
intensity and cohort exposure ratio.

Specification: the child's mean log income on a college indicator (16 or more
years of schooling, treated as endogenous), HP14 (1 for advantaged children),
ParentInc, female, urban, age at the first wave observed, an indicator for
birth year 1981 or later, and province indicators. The excluded instrument is
the provincial intensity of the 1999 college expansion (pro_predict_growth)
times the cohort exposure ratio (exposure_ratio). beta_1 is the coefficient
on college; beta_2 is the coefficient on HP14.

Estimates
---------
  B0. Full sample.
  B1. Birth years 1975 to 1985 only.
  B2. Placebo instrument: provincial intensity times an indicator for birth
      year 1995 or later, in place of the cohort exposure ratio.
  B3. 2SLS and LIML side by side.
  B4. Anderson-Rubin 95 percent confidence interval for beta_1, with
      standard errors clustered by province.
  B5. Effective F statistic of Montiel Olea and Pflueger (2013) for the
      first stage, with standard errors clustered by province.

Output
------
  data/intermediate/ws/ws_07c_iv_robustness.csv
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

_MAC = _WSROOT + ""
_LIN = _WSROOT + ""
BASE = _MAC if os.path.isdir(_MAC) else _LIN
BASE_INT = os.path.join(BASE, "data/intermediate")
WS_DIR   = os.path.join(BASE_INT, "ws")
EXT_DIR  = os.path.join(BASE, "data/external")


def build_child_with_intensity():
    panel = pd.read_pickle(os.path.join(BASE_INT, "panel_analysis.pkl"))
    child = panel.groupby("pid", as_index=False).agg(
        log_y     = ("log_income",   "mean"),
        HP14      = ("HighParentOcc14", "first"),
        ParentInc = ("ParentInc_bar",   "first"),
        female    = ("female",          "first"),
        urban     = ("urban",           "first"),
        provcd    = ("provcd",          "first"),
        age_first = ("age",             "min"),
        year_first= ("year",            "min"),
        eduy_max  = ("eduy",            "max"),
    )
    child = child.dropna(subset=["HP14", "ParentInc", "log_y", "eduy_max"]).copy()
    child["birth_year"] = child["year_first"] - child["age_first"]
    child["post1981"]   = (child["birth_year"] >= 1981).astype(int)
    child["post1995"]   = (child["birth_year"] >= 1995).astype(int)  # used by the placebo instrument
    child["college"]    = (child["eduy_max"] >= 16).astype(int)

    prov = pd.read_csv(os.path.join(EXT_DIR, "expansion_intensity_real.csv"))
    coh  = pd.read_csv(os.path.join(EXT_DIR, "cohort_exposure.csv"))
    coh = coh.rename(columns={"birth_year_age18": "birth_year"})
    child = child.merge(prov[["provcd","pro_predict_growth"]],
                        on="provcd", how="left")
    child = child.merge(coh[["birth_year","exposure_ratio"]],
                        on="birth_year", how="left")
    return child


def make_exog(df):
    prv = pd.get_dummies(df["provcd"].astype(int), prefix="prv",
                         drop_first=True).astype(float)
    X = pd.concat([
        df[["HP14","ParentInc","female","urban","age_first","post1981"]].astype(float),
        prv,
    ], axis=1)
    X["const"] = 1.0
    return X


def liml_estimator(Y, X, Z, n_endog=1):
    """Limited-information maximum likelihood (LIML) as a k-class estimator.

    Construction (Anderson and Rubin 1949; Davidson and MacKinnon 2004,
    ch. 12): let Y1 = [y, X_endog] and X1 = the included exogenous regressors
    (a subset of Z).  kappa is the smallest eigenvalue of
        (Y1' M_Z Y1)^{-1} (Y1' M_X1 Y1),   kappa >= 1,
    and  b_LIML = (X'X - kappa X'M_Z X)^{-1} (X'y - kappa X'M_Z y).
    In a just-identified model kappa = 1 and LIML equals 2SLS.
    X = [X_endog, X_exog] must hold the endogenous regressors in the first
    n_endog columns, and Z must hold X_exog and the excluded instruments.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n = len(Y)

    def _resid(A, B):
        """Residual of A from the OLS projection on B."""
        coef, *_ = np.linalg.lstsq(B, A, rcond=None)
        return A - B @ coef

    X1 = X[:, n_endog:]                       # included exogenous regressors
    Y1 = np.column_stack([Y, X[:, :n_endog]]) # [y, endogenous regressors]
    W_z  = Y1.T @ _resid(Y1, Z)               # Y1' M_Z  Y1
    W_x1 = Y1.T @ _resid(Y1, X1)              # Y1' M_X1 Y1
    eigvals = np.linalg.eigvals(np.linalg.solve(W_z, W_x1))
    eigvals = np.real(eigvals[np.abs(np.imag(eigvals)) < 1e-8])
    kappa = float(eigvals[eigvals >= 1 - 1e-8].min())

    MzX = _resid(X, Z)
    MzY = _resid(Y, Z)
    XmX = X.T @ X - kappa * (X.T @ MzX)
    XmY = X.T @ Y - kappa * (X.T @ MzY)
    b_liml = np.linalg.solve(XmX, XmY)
    # Standard errors under homoskedasticity
    resid = Y - X @ b_liml
    sigma2 = float(resid.T @ resid / (n - X.shape[1]))
    cov_b = sigma2 * np.linalg.inv(XmX)
    return b_liml, np.sqrt(np.diag(cov_b)), kappa


def anderson_rubin_ci(Y, X_endog, X_exog, Z_excl, beta_grid, cluster_groups):
    """Anderson-Rubin confidence interval for the coefficient beta on a single endogenous regressor.

    For each candidate beta in the grid, the squared t statistic of the
    excluded instrument Z_excl is computed in the regression of
       (Y - beta * X_endog)  on  X_exog and Z_excl
    with clustered standard errors. The 95 percent interval runs from the
    smallest to the largest beta at which the statistic is below 3.84.
    """
    cis = []
    for b in beta_grid:
        Yres = Y - b * X_endog
        XX = np.column_stack([X_exog, Z_excl])
        model = sm.OLS(Yres, XX).fit(
            cov_type="cluster", cov_kwds={"groups": cluster_groups})
        # squared t statistic of the last coefficient (Z_excl)
        t = model.tvalues[-1]
        F = t ** 2
        if F < 3.84:  # 95 percent critical value of the chi-squared distribution with 1 degree of freedom
            cis.append(b)
    if not cis:
        return None, None
    return float(min(cis)), float(max(cis))


def olea_pflueger_F(df, z_col, endog_col):
    """Effective F statistic of Montiel Olea and Pflueger (2013).

    With a single excluded instrument it equals (pi/SE)^2, where pi is the
    first-stage coefficient and SE its standard error clustered by province.
    """
    X_exog = make_exog(df)
    X = pd.concat([df[[z_col]].astype(float), X_exog], axis=1)
    fs = sm.OLS(df[endog_col].astype(float).values, X.values).fit(
        cov_type="cluster", cov_kwds={"groups": df["provcd"].values})
    idx = X.columns.get_loc(z_col)
    return (fs.params[idx] / fs.bse[idx]) ** 2


def run_spec(df, z_col, endog_col):
    """2SLS, LIML and the Anderson-Rubin interval on one sample. Returns a dict."""
    X_exog = make_exog(df)
    # 2SLS
    Z_full = pd.concat([X_exog, df[[z_col]].astype(float)], axis=1).values
    X_full = pd.concat([df[[endog_col]].astype(float), X_exog], axis=1).values
    Y = df["log_y"].astype(float).values
    iv = IV2SLS(Y, X_full, Z_full).fit()
    cols = pd.concat([df[[endog_col]].astype(float), X_exog], axis=1).columns.tolist()
    idx_e = cols.index(endog_col)
    idx_h = cols.index("HP14")
    res = {
        "N":           int(len(df)),
        "iv_beta1":    float(iv.params[idx_e]),
        "iv_beta1_se": float(iv.bse[idx_e]),
        "iv_beta2":    float(iv.params[idx_h]),
        "iv_beta2_se": float(iv.bse[idx_h]),
    }
    # First-stage effective F (clustered by province)
    res["op_F"] = olea_pflueger_F(df, z_col, endog_col)
    # LIML
    try:
        b_liml, se_liml, lam = liml_estimator(Y, X_full, Z_full)
        res["liml_beta1"]    = float(b_liml[idx_e])
        res["liml_beta1_se"] = float(se_liml[idx_e])
        res["liml_beta2"]    = float(b_liml[idx_h])
        res["liml_k"]        = float(lam)
    except Exception as e:
        res["liml_beta1"] = res["liml_beta1_se"] = np.nan
        res["liml_beta2"] = res["liml_k"] = np.nan
    # Anderson-Rubin interval on a grid of 6 standard errors around the 2SLS estimate
    grid = np.linspace(res["iv_beta1"] - 6 * res["iv_beta1_se"],
                       res["iv_beta1"] + 6 * res["iv_beta1_se"], 121)
    lo, hi = anderson_rubin_ci(
        Y, df[endog_col].astype(float).values,
        X_exog.values, df[z_col].astype(float).values, grid,
        df["provcd"].values)
    res["ar_ci_lo"] = lo
    res["ar_ci_hi"] = hi
    return res


def main():
    child = build_child_with_intensity()
    child["Z"] = child["pro_predict_growth"] * child["exposure_ratio"]
    n0 = len(child)
    child = child.dropna(subset=["Z", "exposure_ratio", "pro_predict_growth", "college"]).copy()
    print(f"Full sample N = {len(child)} (dropped {n0 - len(child)} children with missing provincial intensity or cohort exposure ratio)")
    print()

    rows = []

    # B0. Full sample
    print("=" * 60)
    print("B0. FULL SAMPLE  instrument: provincial intensity x cohort exposure ratio")
    print("=" * 60)
    r = run_spec(child, "Z", "college")
    r["spec"] = "B0_headline_fullsample"
    print_res(r)
    rows.append(r)

    # B1. Birth years 1975 to 1985
    print("=" * 60)
    print("B1. BANDWIDTH  birth_year in [1975, 1985]")
    print("=" * 60)
    sub = child.loc[(child.birth_year >= 1975) & (child.birth_year <= 1985)].copy()
    sub["Z"] = sub["pro_predict_growth"] * sub["exposure_ratio"]
    r = run_spec(sub, "Z", "college")
    r["spec"] = "B1_bandwidth_1975_1985"
    print_res(r)
    rows.append(r)

    # B2. Placebo instrument: indicator for birth year 1995 or later
    print("=" * 60)
    print("B2. PLACEBO  instrument: pro_predict_growth × 1{birth>=1995}")
    print("=" * 60)
    child["Z_placebo"] = child["pro_predict_growth"] * child["post1995"].astype(float)
    r = run_spec(child, "Z_placebo", "college")
    r["spec"] = "B2_placebo_post1995"
    print_res(r)
    rows.append(r)

    # B3. 2SLS and LIML (both are in the B0 row)
    print("=" * 60)
    print("B3. 2SLS and LIML (both are in the B0 results)")
    print("=" * 60)
    print("    See the B0 row: iv_beta1 is the 2SLS estimate and")
    print("    liml_beta1 is the LIML estimate.")
    print()

    # B4. Anderson-Rubin interval
    print("=" * 60)
    print("B4. ANDERSON-RUBIN  95% CI for β_1 (B0 sample)")
    print(f"    AR 95% CI = [{rows[0]['ar_ci_lo']:.3f}, {rows[0]['ar_ci_hi']:.3f}]")
    print("=" * 60)
    print()

    # B5. Effective F statistic
    print("=" * 60)
    print("B5. Effective F of Montiel Olea and Pflueger (clustered by province)")
    print(f"    op_F (full sample) = {rows[0]['op_F']:.2f}")
    print(f"    Threshold for comparison = 23.11 (one endogenous regressor, one excluded instrument)")
    print("=" * 60)
    print()

    out = pd.DataFrame(rows)
    out_path = os.path.join(WS_DIR, "ws_07c_iv_robustness.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")

    # Summary table
    print("\n========== SUMMARY TABLE ==========")
    cols = ["spec", "N", "op_F", "iv_beta1", "iv_beta1_se",
            "liml_beta1", "liml_beta1_se", "iv_beta2", "ar_ci_lo", "ar_ci_hi"]
    out2 = out[cols].copy()
    out2["op_F"] = out2["op_F"].round(2)
    for c in ["iv_beta1","iv_beta1_se","liml_beta1","liml_beta1_se","iv_beta2","ar_ci_lo","ar_ci_hi"]:
        out2[c] = out2[c].round(3)
    print(out2.to_string(index=False))


def print_res(r):
    print(f"  N        = {r['N']}")
    print(f"  op_F     = {r['op_F']:.2f}")
    print(f"  2SLS β_1 = {r['iv_beta1']:+.3f}  (SE {r['iv_beta1_se']:.3f})")
    print(f"  LIML β_1 = {r['liml_beta1']:+.3f}  (SE {r['liml_beta1_se']:.3f}, k={r['liml_k']:.3f})")
    print(f"  2SLS β_2 = {r['iv_beta2']:+.3f}  (coefficient on HP14)")
    print(f"  AR 95% CI on β_1 = [{r['ar_ci_lo']:.3f}, {r['ar_ci_hi']:.3f}]"
          if r['ar_ci_lo'] is not None else "  AR CI: unbounded")
    print()


if __name__ == "__main__":
    main()
