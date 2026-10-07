"""Numbers for the online appendix that the paper's producers do not already
write (expansion design under birth-year fixed effects, and the
decomposition).

Sample: the children of the paper's sample (sample_mature_age.py: ages 22 to 55,
at least two waves of positive labor income) who have a level of earnings at
ages 30 and over (at least two waves at those ages), one observation per child.
Blocks B and C list the birth years and provinces of all children of the sample.
Specification: the regression on the rows at 30 and over only (family background,
parental income, gender, urban residence, province effects, one indicator per
birth year). Standard errors clustered on province; "wild p" is a wild cluster
bootstrap p-value (Rademacher weights by province, null imposed, 9,999
replications).

Outputs (output/):
  u3_A_occupation_definitions.csv  reduced form of each definition of the
                                   child's professional occupation, each
                                   measured on the waves at ages 30 and over
  u3_B_exposure_series.csv         cohort exposure ratio, every birth year
  u3_C_intensity_by_province.csv   the provincial intensity measures
  u3_D_decomposition_grid.csv      college share of the premium for each
                                   estimate of the premium and each return
  u3_E_selection_sensitivity.csv   least squares return adjusted for
                                   selection on unobservables (Oster 2019)
"""
import os, sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WS = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, HERE)
from _locate import sibling
P = sibling("college_expansion", __file__)
sys.path.insert(0, os.path.join(P, "code"))
import ws_07c_iv_robustness as R

OUT = os.path.join(ROOT, "output")
EXT = os.path.join(P, "data", "external")
INT = os.path.join(P, "data", "intermediate")
rng = np.random.default_rng(20260929)
REPS = int(os.environ.get("U3_REPS", "9999"))
DEMO = ["HP14", "ParentInc", "female", "urban"]
_eb = pd.read_csv(os.path.join(OUT, "u8_entropy_balancing.csv")).iloc[0]
DM, TAU = float(_eb.gap_hat), float(_eb.tau_hat)   # entropy-balanced gap in college completion and premium (Section 5 of the paper)

from sample_mature_age import build_sample, child_waves
c = build_sample().reset_index(drop=True)                      # children with a level of earnings at 30 and over
c_all = build_sample(level_from=None).reset_index(drop=True)   # all children of the sample


def dummies(s, prefix):
    return pd.get_dummies(s.astype(int), prefix=prefix, drop_first=True).astype(float)


X = pd.concat([c[DEMO].astype(float), dummies(c.pc, "p"), dummies(c.by, "b")], axis=1)
X["const"] = 1.0
Xv = X.values; pinv = np.linalg.pinv(Xv)
M = lambda v: v - Xv @ (pinv @ v)
zt = M(c.Z.values.astype(float)); szz = zt @ zt
g = c.pc.values; ks = np.unique(g); gi = np.searchsorted(ks, g)
Gm = np.zeros((len(g), len(ks))); Gm[np.arange(len(g)), gi] = 1.0
G, n, k = len(ks), len(c), Xv.shape[1] + 1
W = rng.choice([-1.0, 1.0], size=(G, REPS))


def coef_se(V):
    V = V.reshape(n, -1)
    b = zt @ V / szz
    U = V - np.outer(zt, b)
    S = Gm.T @ (U * zt[:, None])
    adj = G / (G - 1) * (n - 1) / (n - k)
    return b, np.sqrt(adj * (S ** 2).sum(0)) / szz


def rf(y):
    r0 = M(np.asarray(y, float)); b, se = coef_se(r0)
    E = r0[:, None] * W[gi, :]; E = E - Xv @ (pinv @ E)
    bb, ss = coef_se(E)
    return b[0], se[0], (np.sum(np.abs(bb / ss) >= abs(b[0] / se[0])) + 1) / (REPS + 1)


# ---------------------------------------------------------------- A
pl = child_waves()                                               # the child-waves at ages 30 and over of those children
occ = pl.loc[pl.occ_code.notna() & (pl.occ_code > 0) & pl.pid.isin(c.pid)].copy()
occ["prof"] = (occ.occ_code.astype(int).astype(str).str[0].astype(int) <= 2).astype(float)
gp = occ.groupby("pid").prof
defs = pd.DataFrame({
    "ever": gp.max(),
    "majority of waves": (gp.mean() > 0.5).astype(float),
    "share of waves": gp.mean(),
    "last observed wave": occ.sort_values("year").groupby("pid").prof.last(),
    "two or more waves": (gp.sum() >= 2).astype(float),
})
d = c.merge(defs, left_on="pid", right_index=True, how="left").fillna({k_: 0.0 for k_ in defs.columns})
fs = rf(c.college.values)
rowsA = []
for name in defs.columns:
    b, se, p = rf(d[name].values)
    rowsA.append(dict(definition=name, mean=d[name].mean(), rf=b, se=se, p_wild=p, effect_of_college=b / fs[0]))
    print(f"  {name:22s} mean {d[name].mean():.3f}  RF {b:+.3f} (SE {se:.3f}, wild p {p:.4f})")
pd.DataFrame(rowsA).to_csv(os.path.join(OUT, "u3_A_occupation_definitions.csv"), index=False)

# ---------------------------------------------------------------- B
coh = pd.read_csv(os.path.join(EXT, "cohort_exposure.csv")).rename(columns={"birth_year_age18": "birth_year"})
cnt = c_all.groupby("by").size().rename("children")
B = coh[(coh.birth_year >= c_all.by.min()) & (coh.birth_year <= c_all.by.max())][["birth_year", "year", "recruitment_M", "projected_M", "exposure_ratio"]]
B = B.merge(cnt, left_on="birth_year", right_index=True, how="left")
B.to_csv(os.path.join(OUT, "u3_B_exposure_series.csv"), index=False)
print(B.round(3).to_string(index=False))

# ---------------------------------------------------------------- C
prov = pd.read_csv(os.path.join(EXT, "expansion_intensity_real.csv"))
prov["flow_per_worker"] = prov.predict_flow_nomig / prov.employ2005
canon = pd.read_csv(os.path.join(EXT, "expansion_intensity_canonical.csv"))
stock = pd.read_csv(os.path.join(EXT, "census_hs_stock.csv"))
C = canon[["provcd", "province_en", "int_placebo_log_1998_1997"]].merge(
    prov[["provcd", "pro_predict_growth", "flow_per_worker", "log_growth_implied"]], on="provcd", how="left")
stock = stock.copy(); stock.loc[stock.provcd == 51, "provcd"] = 51
C = C.merge(stock[["provcd", "hs2000"]], on="provcd", how="left")
C = C[C.provcd.isin(c_all.provcd.unique())].sort_values("pro_predict_growth", ascending=False)
C.to_csv(os.path.join(OUT, "u3_C_intensity_by_province.csv"), index=False)
print(C.round(3).to_string(index=False))
cc = C[["pro_predict_growth", "flow_per_worker", "log_growth_implied", "hs2000", "int_placebo_log_1998_1997"]].corr()
cc.to_csv(os.path.join(OUT, "u3_C_intensity_correlations.csv"))
print(cc.round(2).to_string())

# ---------------------------------------------------------------- D
S8 = pd.read_csv(os.path.join(OUT, "u8_premium_specifications.csv")).set_index("specification")
ls_all = float(pd.read_csv(os.path.join(OUT, "u13_A_least_squares.csv")).set_index("item").loc["least squares return, all children", "coef"])
m9 = pd.read_csv(os.path.join(OUT, "u9_B_main.csv")); iv_all = float(m9[m9.intensity.str.contains("main measure")].earn_iv.iloc[0])
rowsD = []
for tau_lab, tau in [("correlated random effects, family income", float(S8.loc["baseline: family income of the parental household", "estimate"])),
                     ("correlated random effects, parental labor income", float(S8.loc["parents' own labor income as the income control", "estimate"])),
                     ("entropy balancing", TAU)]:
    for ret_lab, b in [("least squares", ls_all), ("instrumented, all children", iv_all)]:
        rowsD.append(dict(premium_estimate=tau_lab, premium=tau, return_used=ret_lab, value=b,
                          college_channel=b * DM, direct=tau - b * DM, college_share=b * DM / tau,
                          return_for_half=0.5 * tau / DM, return_for_all=tau / DM))
pd.DataFrame(rowsD).to_csv(os.path.join(OUT, "u3_D_decomposition_grid.csv"), index=False)
print(pd.DataFrame(rowsD).round(3).to_string(index=False))

# ---------------------------------------------------------------- E
y = c.log_y.values.astype(float)
r0 = sm.OLS(y, sm.add_constant(c[["college"]].astype(float))).fit()
Xf = pd.concat([c[["college"] + DEMO].astype(float), dummies(c.pc, "p"), dummies(c.by, "b")], axis=1); Xf["const"] = 1.0
r1 = sm.OLS(y, Xf.values.astype(float)).fit()
b0, R0, b1, R1 = r0.params["college"], r0.rsquared, r1.params[0], r1.rsquared
rowsE = [dict(item="no controls", premium=b0, r2=R0), dict(item="full controls", premium=b1, r2=R1)]
for mult in [1.3, 1.5, 2.0]:
    Rmax = min(1.0, mult * R1)
    bstar = b1 - (b0 - b1) * (Rmax - R1) / (R1 - R0)
    rowsE.append(dict(item=f"adjusted, maximum R2 = {mult} times the controlled R2", premium=bstar, r2=Rmax,
                      college_share=bstar * DM / TAU))
pd.DataFrame(rowsE).to_csv(os.path.join(OUT, "u3_E_selection_sensitivity.csv"), index=False)
print(pd.DataFrame(rowsE).round(3).to_string(index=False))
print("\nwrote u3_*.csv")
