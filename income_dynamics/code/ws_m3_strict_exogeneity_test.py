import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Test of strict exogeneity by adding a lead of the regressor (Wooldridge 2010,
Section 10.7.1), for the model of residual log income with child-specific
intercepts and a trend that differs by family background.

Sample: child-year rows of panel_resid (children aged 22 to 55, waves 2014 to
2022, at least three waves with positive income) of children with a valid
retrospective report of the parents' occupation (has_qv14 = 1), restricted to
rows with non-missing y_tilde_w and urban and with the child observed in the
following wave.

Model:  y_it = alpha_i + gamma*(HP14_i x t) + delta*urban_it + e_it,
        estimated after demeaning within child, standard errors clustered by
        child; t is the wave index (0 for 2014, 1 for 2016, and so on).
Test:   urban_{i,t+1} is added; under strict exogeneity its coefficient is
        zero. gamma is reported with and without the lead.

Output: data/intermediate/ws/ws_m3_strict_exogeneity.csv
"""
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

_MAC = _WSROOT + ""
BASE = _MAC if os.path.isdir(_MAC) else os.environ.get("WS_BASE", _MAC)
INT = os.path.join(BASE, "data/intermediate")


def main():
    d = pd.read_pickle(os.path.join(INT, "panel_resid.pkl"))
    d = d[d["has_qv14"] == 1].copy().sort_values(["pid", "year"])
    d["t"] = (d["year"] - 2014) // 2
    d["hp_t"] = d["HighParentOcc14"] * d["t"]
    d["urban_lead"] = d.groupby("pid")["urban"].shift(-1)
    est = d.dropna(subset=["y_tilde_w", "urban", "urban_lead"]).copy()
    for c in ["y_tilde_w", "hp_t", "urban", "urban_lead"]:
        est[c + "_dm"] = est[c] - est.groupby("pid")[c].transform("mean")

    X1 = sm.add_constant(est[["hp_t_dm", "urban_dm", "urban_lead_dm"]].astype(float))
    m1 = sm.OLS(est["y_tilde_w_dm"].astype(float), X1).fit(
        cov_type="cluster", cov_kwds={"groups": est["pid"]})
    X0 = sm.add_constant(est[["hp_t_dm", "urban_dm"]].astype(float))
    m0 = sm.OLS(est["y_tilde_w_dm"].astype(float), X0).fit(
        cov_type="cluster", cov_kwds={"groups": est["pid"]})

    out = pd.DataFrame([{
        "n_obs": len(est), "n_pid": est["pid"].nunique(),
        "lead_coef": m1.params["urban_lead_dm"],
        "lead_t": m1.tvalues["urban_lead_dm"],
        "lead_p": m1.pvalues["urban_lead_dm"],
        "gamma_no_lead": m0.params["hp_t_dm"], "gamma_no_lead_se": m0.bse["hp_t_dm"],
        "gamma_with_lead": m1.params["hp_t_dm"], "gamma_with_lead_se": m1.bse["hp_t_dm"],
    }])
    path = os.path.join(INT, "ws", "ws_m3_strict_exogeneity.csv")
    out.to_csv(path, index=False)
    print(out.T)
    print(f"\nWrote: {path}")
    verdict = "NOT rejected" if m1.pvalues["urban_lead_dm"] > 0.05 else "REJECTED"
    print(f"Strict exogeneity {verdict} (lead p = {m1.pvalues['urban_lead_dm']:.3f})")


if __name__ == "__main__":
    main()
