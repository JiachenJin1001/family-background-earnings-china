"""The expansion design: earnings at ages 30 and over (Section 6 of the paper).

Sample: the children of the paper's sample (sample_mature_age.py: ages 22 to 55, at least two waves of positive
labor income in 2014 to 2022).

Rows (two_row() of sample_mature_age.py): each child contributes the mean of log labor income, net of survey-year
effects, over the waves at ages 30 and over, and the same mean over the waves below 30. The row at 30 and over is
kept only if it averages at least two waves.

Two-row specification (main): on the rows r of child i,
    y_ir = b_old M_i 1[r at 30+] + b_young M_i 1[r below 30] + controls + e_ir,
with M college completion, instrumented by Z_i 1[r at 30+] and Z_i 1[r below 30], Z provincial intensity times the
cohort exposure ratio. Controls: family background, parental income, gender, urban residence, province effects, one
indicator per birth year, an indicator for the row at 30 and over, and family background times that indicator.
Every coefficient reported is the one of the row at 30 and over: the first stage (college x 1[30+] on Z x 1[30+]),
the reduced form, the instrumented return b_old, the least squares return. The coefficient of the row below 30 is
estimated only on cohorts born after the reform (nobody born before it is observed below 30 in 2014 to 2022); it is
not identified by the reform and is written to the output for completeness only.
Rows at 30 and over only: the same regression without the rows below 30 (one row per child).
Weights: unweighted in the main specification. With E_WEIGHTS=process each row is weighted by the precision implied
by the earnings process of Section 3 (the inverse of sigma_eta^2 + sigma_v^2 / n for a mean of n waves) and the output
files carry the suffix _weighted.
Inference: standard errors clustered on province; wild cluster bootstrap p-values (Rademacher weights by province,
null imposed, 9,999 replications); Anderson-Rubin set by grid inversion. The columns ar_lo and ar_hi of u9_B_main.csv use a
short grid (-0.5 to 4.0, step 0.05) and are superseded for the main measure by block G (grid -6 to 6), which the paper prints.

Outputs (output/, with the suffix _weighted for the weighted run):
  u9_A_sample.csv       counts of children and rows, completion rates before and after the reform by family
                        background, least squares returns at 30 and over and below 30
  u9_B_main.csv         first stage, earnings and occupation reduced forms, instrumented return, Anderson-Rubin set,
                        under each provincial intensity measure; the placebo intensity
  u9_C_checks.csv       children with nine or fewer years of schooling; trade controls; cohort trends by region and by
                        province; pre-reform trends; leaving one province out; the instrument assigned by the province
                        at age twelve; rows at 30 and over only; the exposure ratio of the year the cohort turned 17 or 19;
                        the cohorts born 1973 to 1976 left out
  u9_D_cohort_bins.csv  coefficients on intensity by birth-cohort bin for college completion and earnings
  paper/figures/fig_mature_age.png   the cohort-bin coefficients (main run only)
Blocks A to D can be run separately with U9_BLOCKS; block G (the Anderson-Rubin p-value of the return for all
children at every grid point) runs only when asked for, U9_BLOCKS=G.
Block R (randomization inference: provincial intensity reassigned among all provinces, u9_R_randomization_inference.csv;
among the provinces of the same region with a set for the return, u9_R_randomization_within_region.csv)
also runs only when asked for, U9_BLOCKS=R.
"""
import os, sys, warnings, importlib.util
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from _locate import sibling
P = sibling("college_expansion", __file__); M = sibling("income_dynamics", __file__)
EXT = os.path.join(P, "data", "external"); OUT = os.path.join(ROOT, "output")
SEED = 20261001; REPS = int(os.environ.get("U9_REPS", "9999")); BLOCKS = os.environ.get("U9_BLOCKS", "ABCD")
DEMO = ["HP14", "ParentInc", "female", "urban"]
INTENSITIES = [("pro_predict_growth", "predicted provincial growth (main measure)"), ("flow_per_worker", "predicted flow per 2005 worker"),
               ("log_growth_implied", "log implied 2010 stock"), ("hs2000", "census senior-high stock in 2000")]


# ------------------------------------------------------------------ the sample (sample_mature_age.py)
from sample_mature_age import prefilter_frame, build_sample, two_row, _waves, LO as S_LO, HI as S_HI, YOUNG_MIN


def dummies(s, prefix):
    return pd.get_dummies(s.astype(int), prefix=prefix, drop_first=True).astype(float)


def X_fe(d, demo=DEMO, extra=None):
    parts = [d[demo].astype(float), dummies(d.pc, "p"), dummies(d.by, "b")]
    if extra is not None:
        parts.append(extra.astype(float))
    X = pd.concat(parts, axis=1); X["const"] = 1.0
    return X.values


class Design:
    """One instrument, one control matrix; wild cluster bootstrap by partialling the controls out."""
    def __init__(self, d, z, X, rng):
        self.X = X; self.pinv = np.linalg.pinv(X)
        self.g = d.pc.values; self.ks = np.unique(self.g); self.gi = np.searchsorted(self.ks, self.g)
        self.Gm = np.zeros((len(d), len(self.ks))); self.Gm[np.arange(len(d)), self.gi] = 1.0
        self.zt = self.M(np.asarray(z, float)); self.szz = self.zt @ self.zt
        self.W = rng.choice([-1.0, 1.0], size=(len(self.ks), REPS))
        self.G = len(self.ks); self.n = len(d); self.k = X.shape[1] + 1

    def M(self, v):
        return v - self.X @ (self.pinv @ v)

    def coef_t(self, V):
        V = V.reshape(self.n, -1); beta = self.zt @ V / self.szz; U = V - np.outer(self.zt, beta)
        S = self.Gm.T @ (U * self.zt[:, None]); adj = self.G / (self.G - 1) * (self.n - 1) / (self.n - self.k)
        return beta, np.sqrt(adj * (S ** 2).sum(0)) / self.szz

    def rf(self, y, wild=True):
        r0 = self.M(np.asarray(y, float)); b, se = self.coef_t(r0)
        out = dict(coef=b[0], se=se[0], t=b[0] / se[0], p_wild=np.nan)
        if wild:
            E = r0[:, None] * self.W[self.gi, :]; E = E - self.X @ (self.pinv @ E); bb, ss = self.coef_t(E)
            out["p_wild"] = (np.sum(np.abs(bb / ss) >= abs(out["t"])) + 1) / (REPS + 1)
        return out

    def ar_interval(self, y, m, grid, level=0.05):
        keep = [b0 for b0 in grid if self.rf(np.asarray(y, float) - b0 * np.asarray(m, float))["p_wild"] >= level]
        return (min(keep), max(keep)) if keep else (np.nan, np.nan)


def cl(y, X, g):
    return sm.OLS(np.asarray(y, float), np.asarray(X, float)).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(g)})


# ------------------------------------------------------------------ the children, the rows and the two-row design
c = build_sample(level_from=None)
print(f"sample (ages {S_LO}-{S_HI}): {len(c)} children, {int(c.HP14.sum())} advantaged, provinces {c.pc.nunique()}, birth years {c.by.min()}-{c.by.max()}")
WEIGHTS = os.environ.get("E_WEIGHTS", "none"); SFX = "_weighted" if WEIGHTS == "process" else ""
W0 = _waves(); W0 = W0[W0.pid.isin(c.pid)]

def rows_frame(cc, A=30, outcome="log_income", d=None, young_min=None):
    return two_row(A=A, weights=WEIGHTS, outcome=outcome, c=cc, d=W0 if d is None else d, young_min=young_min)

def tb_X(s, extra=None, group_fe=False):
    parts = [s[DEMO].astype(float), dummies(s.pc, "p"), dummies(s.by, "b")]
    if group_fe: parts.append(dummies(s.by, "hb").multiply(s.HP14, axis=0))
    if extra is not None: parts.append(extra.astype(float))
    X = pd.concat(parts, axis=1); X["old"] = s.old.astype(float); X["H_old"] = (s.HP14 * s.old).astype(float)   # band indicator and family background x band
    X = X.loc[:, X.std() > 0]; X["const"] = 1.0
    return X.values

class TB:
    """Two-row design on a rows frame s, weighted by s.w. Tested instrument: Z x 1[30+]; Z x 1[<30] is a control."""
    def __init__(self, s, z, rng, extra=None):
        self.s = s; sw = np.sqrt(s.w.values.astype(float)); self.sw = sw; old = s.old.values.astype(float); self.old = old
        X = tb_X(s, extra); self.X = X
        self.z_old, self.z_young = z * old, z * (1 - old)
        self.D = Design(s, sw * self.z_old, sw[:, None] * np.column_stack([self.z_young, X]), rng)
    def rf(self, y, wild=True):
        return self.D.rf(self.sw * y, wild=wild)
    def fs(self, m, wild=True):                       # first stage of college x 1[30+] on Z x 1[30+] in the rows
        return self.D.rf(self.sw * m * self.old, wild=wild)
    def iv(self, y, m, both=False):
        sw, old = self.sw, self.old; W = lambda a: sw[:, None] * a
        Zm = W(np.column_stack([self.z_old, self.z_young, self.X])); En = W(np.column_stack([m * old, m * (1 - old)]))
        Enh = Zm @ np.linalg.lstsq(Zm, En, rcond=None)[0]; b = np.linalg.lstsq(np.column_stack([Enh, W(self.X)]), sw * y, rcond=None)[0]
        return (float(b[0]), float(b[1])) if both else float(b[0])
    def ols(self, y, m):
        sw, old = self.sw, self.old
        r = cl(sw * y, sw[:, None] * np.column_stack([m * old, m * (1 - old), self.X]), self.s.pc.values); return float(r.params[0]), float(r.bse[0]), float(r.params[1]), float(r.bse[1])
    def ar(self, y, m, grid, level=0.05):
        keep = [b0 for b0 in grid if self.rf(y - b0 * m * self.old)["p_wild"] >= level]
        return (min(keep), max(keep)) if keep else (np.nan, np.nan)

def occ_rows(cc):
    d = W0[W0.pid.isin(cc.pid) & W0.occ_code.notna() & (W0.occ_code > 0)].copy()
    d["prof"] = (d.occ_code.astype(int).astype(str).str[0].astype(int) <= 2).astype(float)
    return rows_frame(cc, outcome="prof", d=d)
s = rows_frame(c)
print(f"rows: {len(s)} ({int(s.old.sum())} at 30 and over, of which {int((s.old * s.HP14).sum())} advantaged; {int((1 - s.old).sum())} below 30); children with a row {s.pid.nunique()}; weights {WEIGHTS}")

# ------------------------------------------------------------------ A. the sample
if "A" in BLOCKS:
    so_ = s[s.old == 1]
    rows = [dict(item="children", value=len(c)), dict(item="advantaged children", value=int(c.HP14.sum())),
            dict(item="child-waves", value=int(c.waves.sum())), dict(item="waves per child", value=c.waves.mean()),
            dict(item="children with a row", value=int(s.pid.nunique())), dict(item="rows at 30 and over", value=int(s.old.sum())), dict(item="advantaged rows at 30 and over", value=int((s.old * s.HP14).sum())),
            dict(item="rows below 30", value=int((1 - s.old).sum())), dict(item="waves behind a row at 30 and over, mean", value=so_.n.mean()),
            dict(item="children born 1981 or later", value=int((c.by >= 1981).sum())), dict(item="rows at 30 and over, born 1981 or later", value=int((so_.by >= 1981).sum())),
            dict(item="birth years", value=f"{c.by.min()}-{c.by.max()}"), dict(item="birth years of the rows at 30 and over", value=f"{so_.by.min()}-{so_.by.max()}"), dict(item="provinces", value=c.pc.nunique())]
    pre = c.by <= 1980
    for h, lab in [(1, "advantaged"), (0, "ordinary")]:
        rows += [dict(item=f"college completion, {lab} children, born 1980 or earlier", value=c[(c.HP14 == h) & pre].college.mean()),
                 dict(item=f"college completion, {lab} children, born 1981 or later", value=c[(c.HP14 == h) & ~pre].college.mean()),
                 dict(item=f"{lab} children born 1980 or earlier", value=int(((c.HP14 == h) & pre).sum()))]
    T = TB(s, s.Z.values, np.random.default_rng(SEED)); b, se, by_, bys = T.ols(s.y.values, s.college.values)
    rows += [dict(item="least squares return to college", value=b, se=se), dict(item="least squares return to college below 30", value=by_, se=bys)]
    q25, q75 = c.pro_predict_growth.quantile(0.25), c.pro_predict_growth.quantile(0.75); e_late = c[c.by >= 1989].exposure_ratio.mean()
    rows += [dict(item="provincial intensity, 25th percentile of children", value=q25), dict(item="provincial intensity, 75th percentile of children", value=q75),
             dict(item="mean exposure ratio, born 1989 or later", value=e_late), dict(item="difference in the instrument", value=(q75 - q25) * e_late)]
    # the part of that difference that the reform created: the exposure ratio of the cohorts born in 1980 or earlier is near one and is
    # absorbed by the province effects, so the reform moves the instrument by the interquartile range times the rise in the exposure ratio
    e_pre = c[c.by <= 1980].exposure_ratio.mean()
    rows += [dict(item="mean exposure ratio, born 1980 or earlier", value=e_pre), dict(item="difference in the instrument created by the reform", value=(q75 - q25) * (e_late - e_pre))]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"u9_A_sample{SFX}.csv"), index=False); print(pd.DataFrame(rows).to_string(index=False))

GRID = np.round(np.arange(-0.5, 4.001, 0.05), 2)
# ------------------------------------------------------------------ B. main estimates under each intensity
if "B" in BLOCKS:
    rng = np.random.default_rng(SEED + 1); rows = []; so = occ_rows(c)
    for col, lab in INTENSITIES:
        cc = c.dropna(subset=[col]).reset_index(drop=True); ss = s[s[col].notna()].reset_index(drop=True); oo = so[so[col].notna()].reset_index(drop=True)
        zc = (cc[col] * cc.exposure_ratio).values; fc = Design(cc, zc, X_fe(cc), rng).rf(cc.college.values, wild=False)
        T = TB(ss, (ss[col] * ss.exposure_ratio).values, rng); fs = T.fs(ss.college.values); e = T.rf(ss.y.values); biv, biv_y = T.iv(ss.y.values, ss.college.values, both=True)
        lo, hi = T.ar(ss.y.values, ss.college.values, GRID) if col == "pro_predict_growth" else (np.nan, np.nan)
        To = TB(oo, (oo[col] * oo.exposure_ratio).values, rng); o2 = To.rf(oo.y.values); oiv = To.iv(oo.y.values, oo.college.values)
        sp = ss[ss.int_placebo_log_1998_1997.notna()].reset_index(drop=True); Tp = TB(sp, (sp.int_placebo_log_1998_1997 * sp.exposure_ratio).values, rng); bp = Tp.fs(sp.college.values, wild=False); ep = Tp.rf(sp.y.values)
        ols_, ols_se, olsy, _ = T.ols(ss.y.values, ss.college.values)
        rows.append(dict(intensity=lab, n=int(ss.pid.nunique()), rows_old=int(ss.old.sum()), ar_grid_hi=float(GRID[-1]), fs=fs["coef"], fs_se=fs["se"], F=fs["t"] ** 2, fs_p_wild=fs["p_wild"], F_child_level_all_children=fc["t"] ** 2,
                         earn=e["coef"], earn_se=e["se"], earn_p_wild=e["p_wild"], earn_iv=biv, earn_iv_below_30=biv_y, ar_lo=lo, ar_hi=hi, ols=ols_, ols_se=ols_se, ols_below_30=olsy,
                         occ_maj=o2["coef"], occ_maj_se=o2["se"], occ_maj_p_wild=o2["p_wild"], occ_maj_iv=oiv,
                         placebo_F=bp["t"] ** 2, placebo_earn=ep["coef"], placebo_earn_se=ep["se"], placebo_earn_p_wild=ep["p_wild"]))
        print(f"  {lab:44s} F {fs['t']**2:5.1f} (p {fs['p_wild']:.3f}) | earnings {e['coef']:+.3f} ({e['se']:.3f}) p {e['p_wild']:.3f} return {biv:+.2f} (below 30 {biv_y:+.2f}) AR [{lo}, {hi}] ls {ols_:.3f} | professional share {o2['coef']:+.3f} p {o2['p_wild']:.3f} | placebo F {bp['t']**2:.1f} earn p {ep['p_wild']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"u9_B_main{SFX}.csv"), index=False)

# ------------------------------------------------------------------ G. Anderson-Rubin test of the return for all children at every grid point
# Main measure of intensity. The file holds the wild-cluster p-value of the test that the return equals b, for b from -6.0 to 6.0
# in steps of 0.05 (the grid of u12_returns_by_family_background.py), so that the set can be read at any level and its ends are
# inside the grid. As b grows without bound the test becomes the test of a zero first stage, whose p-value is in u9_B_main.csv
# (fs_p_wild): a set is unbounded at a level only if that p-value is at or above the level. The block also writes the first stage
# of three-year completion (exactly 15 years of schooling), which the completion indicator does not count. Run with U9_BLOCKS=G.
if "G" in BLOCKS:
    rng = np.random.default_rng(SEED + 7); T = TB(s, (s.pro_predict_growth * s.exposure_ratio).values, rng); yv, mv = s.y.values, s.college.values
    grid_all = np.round(np.arange(-6.0, 6.001, 0.05), 2)
    pd.DataFrame([dict(group="all children", b=float(b0), p_wild=T.rf(yv - b0 * mv * T.old)["p_wild"]) for b0 in grid_all]).to_csv(os.path.join(OUT, f"u9_anderson_rubin_grid{SFX}.csv"), index=False)
    print("wrote u9_anderson_rubin_grid" + SFX + ".csv")
    e15 = s.pid.map(c.set_index("pid").eduy_max).eq(15).astype(float).values if "eduy_max" not in s.columns else (s.eduy_max == 15).astype(float).values
    f15, f15p, f16 = T.fs(e15), T.fs(np.maximum(e15, mv)), T.fs(mv)
    pd.DataFrame([dict(outcome="exactly 15 years of schooling (three-year degree)", coef=f15["coef"], se=f15["se"], p_wild=f15["p_wild"], share_of_rows_at_30_and_over=float(e15[T.old == 1].mean())),
                  dict(outcome="15 or more years of schooling", coef=f15p["coef"], se=f15p["se"], p_wild=f15p["p_wild"], share_of_rows_at_30_and_over=float(np.maximum(e15, mv)[T.old == 1].mean())),
                  dict(outcome="16 or more years of schooling (college completion)", coef=f16["coef"], se=f16["se"], p_wild=f16["p_wild"], share_of_rows_at_30_and_over=float(mv[T.old == 1].mean()))]
                 ).to_csv(os.path.join(OUT, f"u9_G_three_year_completion{SFX}.csv"), index=False)
    print(f"three-year completion on the instrument: {f15['coef']:+.4f} ({f15['se']:.4f}) wild p {f15['p_wild']:.4f}")

# ------------------------------------------------------------------ R. randomization inference over provinces
# The 29 provincial values of the main intensity measure are reassigned at random among the 29 provinces (4,999
# reassignments); cohort exposure, the children and every control stay as they are. For each reassignment the two-row
# specification is re-estimated and the cluster t-statistics of the earnings reduced form and of the first stage (both on
# the instrument x row at 30 and over) are stored. The permutation p-value is the share of reassignments whose |t| is
# at least the |t| of the actual assignment. It does not rely on the number of provinces being large. Run with U9_BLOCKS=R.
if "R" in BLOCKS:
    rng = np.random.default_rng(SEED + 11); NPERM = int(os.environ.get("U9_PERMUTATIONS", "4999"))
    sw = np.sqrt(s.w.values.astype(float)); old = s.old.values.astype(float); X = sw[:, None] * tb_X(s); Xp = np.linalg.pinv(X)
    MX = lambda v: v - X @ (Xp @ v)
    g = s.pc.values; ks = np.unique(g); gi = np.searchsorted(ks, g); Gm = np.zeros((len(s), len(ks))); Gm[np.arange(len(s)), gi] = 1.0
    G, n, k = len(ks), len(s), X.shape[1] + 2; adj = G / (G - 1) * (n - 1) / (n - k)
    ry, rm = MX(sw * s.y.values), MX(sw * s.college.values * old)
    def tstats(z):
        zo, zy = MX(sw * z * old), MX(sw * z * (1 - old)); zt = zo - zy * (zy @ zo) / (zy @ zy); szz = zt @ zt; out = []
        for r in (ry, rm):
            r2 = r - zy * (zy @ r) / (zy @ zy); b = zt @ r2 / szz; u = r2 - zt * b; out.append(b / (np.sqrt(adj * ((Gm.T @ (u * zt)) ** 2).sum()) / szz))
        return out
    prov = s.groupby("pc").pro_predict_growth.first(); t_rf, t_fs = tstats(s.Z.values); T0 = TB(s, s.Z.values, np.random.default_rng(SEED))
    assert abs(t_rf - T0.rf(s.y.values, wild=False)["t"]) < 1e-6 and abs(t_fs - T0.fs(s.college.values, wild=False)["t"]) < 1e-6
    tt = np.array([tstats((s.pc.map(pd.Series(rng.permutation(prov.values), index=prov.index)) * s.exposure_ratio).values) for _ in range(NPERM)])
    R = pd.DataFrame([dict(statistic="earnings reduced form", t=t_rf, permutation_p=(np.sum(np.abs(tt[:, 0]) >= abs(t_rf)) + 1) / (NPERM + 1), reassignments=NPERM, provinces=G),
                      dict(statistic="first stage", t=t_fs, permutation_p=(np.sum(np.abs(tt[:, 1]) >= abs(t_fs)) + 1) / (NPERM + 1), reassignments=NPERM, provinces=G)])
    R.to_csv(os.path.join(OUT, f"u9_R_randomization_inference{SFX}.csv"), index=False); print(R.round(4).to_string(index=False))
    # Reassignment only among the provinces of the same region (east, center, west: the regions of the regional-trend check),
    # and a set for the return to college by inverting the within-region test: for each b on the grid the outcome is
    # earnings minus b times college completion on the rows at 30 and over, and b is kept when its permutation p-value is at
    # least the level. The test at b assumes the same return b for every child.
    REG_R = {11: "E", 12: "E", 13: "E", 21: "E", 31: "E", 32: "E", 33: "E", 35: "E", 37: "E", 44: "E", 46: "E", 14: "C", 22: "C", 23: "C", 34: "C", 36: "C", 41: "C", 42: "C", 43: "C"}
    region = pd.Series({pv: REG_R.get(int(pv), "W") for pv in prov.index}); grid_r = np.round(np.arange(-6.0, 6.001, 0.05), 2)
    def t_all(z):
        zo, zy = MX(sw * z * old), MX(sw * z * (1 - old)); zt = zo - zy * (zy @ zo) / (zy @ zy); szz = zt @ zt; cs = []
        for r in (ry, rm):
            r2 = r - zy * (zy @ r) / (zy @ zy); cc = zt @ r2 / szz; cs += [cc, Gm.T @ ((r2 - zt * cc) * zt)]
        cy, Sy, cm, Sm = cs; se = lambda S: np.sqrt(adj * (S ** 2).sum(0)) / szz
        return cy / se(Sy), cm / se(Sm), (cy - grid_r * cm) / se(Sy[:, None] - Sm[:, None] * grid_r[None, :])
    a_rf, a_fs, a_b = t_all(s.Z.values); assert abs(a_rf - t_rf) < 1e-8 and abs(a_fs - t_fs) < 1e-8
    rng = np.random.default_rng(SEED + 12); P_rf = P_fs = 0; P_b = np.zeros(len(grid_r))
    for _ in range(NPERM):
        perm = prov.copy()
        for rg in ("E", "C", "W"):
            idx = region.index[region == rg]; perm.loc[idx] = rng.permutation(prov.loc[idx].values)
        q_rf, q_fs, q_b = t_all((s.pc.map(perm) * s.exposure_ratio).values); P_rf += abs(q_rf) >= abs(a_rf); P_fs += abs(q_fs) >= abs(a_fs); P_b += np.abs(q_b) >= np.abs(a_b)
    p_b = (P_b + 1) / (NPERM + 1)
    def ends(level):
        keep = grid_r[p_b >= level]; assert len(keep) and np.all(np.diff(keep) < 0.051), "the set is not one interval"; return float(keep.min()), float(keep.max())
    (l95, h95), (l90, h90) = ends(0.05), ends(0.10)
    Rw = pd.DataFrame([dict(reassignment="within region (east, center, west)", provinces_by_region=str(region.value_counts().sort_index().to_dict()), reassignments=NPERM,
                            permutation_p_reduced_form=(P_rf + 1) / (NPERM + 1), permutation_p_first_stage=(P_fs + 1) / (NPERM + 1), set95_lo=l95, set95_hi=h95, set90_lo=l90, set90_hi=h90, grid_lo=float(grid_r.min()), grid_hi=float(grid_r.max()))])
    Rw.to_csv(os.path.join(OUT, f"u9_R_randomization_within_region{SFX}.csv"), index=False); print(Rw.T.to_string())
    pd.DataFrame({"b": grid_r, "permutation_p": p_b}).to_csv(os.path.join(OUT, f"u9_R_randomization_p_by_return{SFX}.csv"), index=False)

# ------------------------------------------------------------------ C. checks (main intensity)
if "C" in BLOCKS:
    rng = np.random.default_rng(SEED + 2); rows = []; so = occ_rows(c)
    def add(lab, T, ss, extra=None, **kw):
        e = T.rf(ss.y.values); f = T.fs(ss.college.values, wild=False); rows.append(dict(check=lab, n=int(ss.pid.nunique()), rows_old=int(ss.old.sum()), coef=e["coef"], se=e["se"], p_wild=e["p_wild"], F=f["t"] ** 2, iv=T.iv(ss.y.values, ss.college.values), **kw))
    for cut in (9, 12):
        ss = rows_frame(c[c.eduy_max <= cut].reset_index(drop=True)); e = TB(ss, ss.Z.values, rng).rf(ss.y.values)
        rows.append(dict(check=f"earnings, children with {cut} or fewer years of schooling", n=int(ss.pid.nunique()), rows_old=int(ss.old.sum()), coef=e["coef"], se=e["se"], p_wild=e["p_wild"]))
        oo = so[so.eduy_max <= cut].reset_index(drop=True); o = TB(oo, oo.Z.values, rng).rf(oo.y.values)
        rows.append(dict(check=f"professional occupation, children with {cut} or fewer years of schooling", n=int(oo.pid.nunique()), coef=o["coef"], se=o["se"], p_wild=o["p_wild"]))
    for lab, cols in [("coastal x exposure", ["coastal"]), ("export share x exposure", ["expgdp"]), ("both", ["coastal", "expgdp"])]:
        ss = s.dropna(subset=cols).reset_index(drop=True); ex = pd.DataFrame({k + "_x_E": ss[k] * ss.exposure_ratio for k in cols})
        T = TB(ss, ss.Z.values, rng, extra=ex); e = T.rf(ss.y.values); f = T.fs(ss.college.values, wild=False)
        sw = np.sqrt(ss.w.values); old = ss.old.values.astype(float); Xe = tb_X(ss, ex); W = lambda a: sw[:, None] * a
        Zm = W(np.column_stack([ss.Z.values * old, ss.Z.values * (1 - old), Xe])); En = W(np.column_stack([ss.college.values * old, ss.college.values * (1 - old)]))
        bb = np.linalg.lstsq(np.column_stack([Zm @ np.linalg.lstsq(Zm, En, rcond=None)[0], W(Xe)]), sw * ss.y.values, rcond=None)[0]
        rows.append(dict(check=f"trade control: {lab}", n=int(ss.pid.nunique()), coef=e["coef"], se=e["se"], p_wild=e["p_wild"], F=f["t"] ** 2, iv=float(bb[0])))
    REG = {11: "E", 12: "E", 13: "E", 21: "E", 31: "E", 32: "E", 33: "E", 35: "E", 37: "E", 44: "E", 46: "E", 14: "C", 22: "C", 23: "C", 34: "C", 36: "C", 41: "C", 42: "C", 43: "C"}
    def trends(d, kind):
        t = (d.by - 1980).astype(float)
        if kind == "region":
            rg = d.pc.map(lambda k: REG.get(int(k), "W")); return pd.DataFrame({f"t_{k}": (rg == k).astype(float) * t for k in ("C", "W")}, index=d.index)
        return pd.DataFrame({f"t_{k}": (d.pc == k).astype(float) * t for k in np.unique(d.pc)[1:]}, index=d.index)
    for kind, lab in [("region", "region-specific linear cohort trends (three regions)"), ("province", "province-specific linear cohort trends")]:
        T = TB(s, s.Z.values, rng, extra=trends(s, kind)); e = T.rf(s.y.values); f = T.fs(s.college.values, wild=False)
        sw = np.sqrt(s.w.values); old = s.old.values.astype(float); Xe = tb_X(s, trends(s, kind)); W = lambda a: sw[:, None] * a
        Zm = W(np.column_stack([s.Z.values * old, s.Z.values * (1 - old), Xe])); En = W(np.column_stack([s.college.values * old, s.college.values * (1 - old)]))
        bb = np.linalg.lstsq(np.column_stack([Zm @ np.linalg.lstsq(Zm, En, rcond=None)[0], W(Xe)]), sw * s.y.values, rcond=None)[0]
        rows.append(dict(check=lab, n=int(s.pid.nunique()), coef=e["coef"], se=e["se"], p_wild=e["p_wild"], F=f["t"] ** 2, iv=float(bb[0])))
    # pre-reform trends: cohorts born 1980 or earlier, intensity x (birth year - 1980), on the rows at 30 and over and on completion
    pre = s[(s.by <= 1980) & (s.old == 1)].reset_index(drop=True); sw = np.sqrt(pre.w.values); xt = (pre.pro_predict_growth * (pre.by - 1980)).values
    Xp = pd.concat([pre[DEMO].astype(float), dummies(pre.pc, "p"), dummies(pre.by, "b")], axis=1).assign(const=1.0).values; Dp = Design(pre, sw * xt, sw[:, None] * Xp, rng)
    e = Dp.rf(sw * pre.y.values); f = Dp.rf(sw * pre.college.values)
    rows.append(dict(check="pre-reform cohorts: earnings on intensity x birth year", n=len(pre), coef=e["coef"], se=e["se"], p_wild=e["p_wild"]))
    rows.append(dict(check="pre-reform cohorts: completion on intensity x birth year", n=len(pre), coef=f["coef"], se=f["se"], p_wild=f["p_wild"]))
    loo = []
    for pcode in np.unique(c.pc):
        ss = s[s.pc != pcode].reset_index(drop=True); T = TB(ss, ss.Z.values, rng); loo.append((T.rf(ss.y.values, wild=False)["coef"], T.iv(ss.y.values, ss.college.values)))
    rows.append(dict(check="leave one province out: earnings reduced form, smallest and largest", coef=min(x[0] for x in loo), se=max(x[0] for x in loo)))
    rows.append(dict(check="leave one province out: instrumented return, smallest and largest", coef=min(x[1] for x in loo), se=max(x[1] for x in loo)))
    # instrument by the province at age twelve where the 2010 survey records it
    import pyreadstat
    raw = next(os.path.join(p, "data", "raw", "CFPS2010", "ecfps2010adult_201906.dta") for p in (P, M) if os.path.exists(os.path.join(p, "data", "raw", "CFPS2010", "ecfps2010adult_201906.dta")))
    a, _ = pyreadstat.read_dta(raw, usecols=["pid", "qa4", "qa401acode", "qa102acode"]); a = a.dropna(subset=["pid"]); a["pid"] = a.pid.astype(np.int64)
    num = lambda v: pd.to_numeric(v, errors="coerce").where(lambda x: x > 0); p12 = num(a.qa401acode).where(num(a.qa401acode).notna(), num(a.qa102acode).where(num(a.qa4) == 1))
    p12 = pd.Series(p12.values, index=a.pid).loc[lambda x: x.between(11, 82)]; p12 = p12[~p12.index.duplicated()]
    prov = pd.read_csv(os.path.join(EXT, "expansion_intensity_real.csv")).set_index("provcd").pro_predict_growth
    c12 = c.copy(); pp = c12.pid.map(p12); c12["pc"] = np.where(pp.notna() & pp.map(prov).notna(), pp, c12.pc).astype(int); c12["pro_predict_growth"] = c12.pc.map(prov); c12["Z"] = c12.pro_predict_growth * c12.exposure_ratio
    ss = rows_frame(c12); add("instrument by the province at age twelve", TB(ss, ss.Z.values, rng), ss, changed=int((pp.notna() & (pp != c.pc)).sum()), with_report=int(pp.notna().sum()))
    # the other minimum for the row below 30, and band thresholds 28 and 29
    ss = rows_frame(c, young_min=3 - YOUNG_MIN); add(f"row below 30 with at least {3 - YOUNG_MIN} wave(s)", TB(ss, ss.Z.values, rng), ss)
    ss = s[s.old == 1].reset_index(drop=True); sw = np.sqrt(ss.w.values); Xo = pd.concat([ss[DEMO].astype(float), dummies(ss.pc, "p"), dummies(ss.by, "b")], axis=1).assign(const=1.0).values
    Do = Design(ss, sw * ss.Z.values, sw[:, None] * Xo, rng); e = Do.rf(sw * ss.y.values); f = Do.rf(sw * ss.college.values, wild=False)
    rows.append(dict(check="rows at 30 and over only (no row below 30)", n=len(ss), rows_old=len(ss), coef=e["coef"], se=e["se"], p_wild=e["p_wild"], F=f["t"] ** 2, iv=e["coef"] / f["coef"]))
    for A in (28, 29):
        ss = rows_frame(c, A=A); add(f"band threshold {A}", TB(ss, ss.Z.values, rng), ss)
    # the age at which a cohort is taken to reach college: the exposure ratio of the year it turned 17 or 19 in place of 18
    # (cohorts whose year falls before the start of the admissions series keep the pre-reform mean ratio, as in the sample rule)
    coh_ = pd.read_csv(os.path.join(EXT, "cohort_exposure.csv")); by_year = coh_.set_index("year").exposure_ratio; pre_mean = float(coh_[coh_.birth_year_age18 <= 1980].exposure_ratio.mean())
    for age in (17, 19):
        e_age = s.by.add(age).map(by_year).fillna(pre_mean); add(f"exposure ratio of the year the cohort turned {age}", TB(s, (s.pro_predict_growth * e_age).values, rng), s)
    # the pre-reform cohorts born 1973 to 1976 left out: in the cohort-bin figure their completion gradient is the pre-reform
    # coefficient furthest from zero (block D), so the return is re-estimated without them
    ss = s[~s.by.between(1973, 1976)].reset_index(drop=True); add("cohorts born 1973 to 1976 left out", TB(ss, ss.Z.values, rng), ss)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"u9_C_checks{SFX}.csv"), index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))

# ------------------------------------------------------------------ D. cohort bins (completion at the child level; earnings on the rows at 30 and over)
if "D" in BLOCKS:
    bins = [(1950, 1972, "1972 or earlier"), (1973, 1976, "1973-1976"), (1981, 1984, "1981-1984"), (1985, 1988, "1985-1988"), (1989, 1997, "1989 or later")]   # reference: 1977-1980
    rows = []
    so = s[s.old == 1].reset_index(drop=True); sw = np.sqrt(so.w.values)
    K = pd.DataFrame({lab: ((so.by >= a) & (so.by <= b)).astype(float) * so.pro_predict_growth for a, b, lab in bins})
    Xb = pd.concat([so[DEMO].astype(float), dummies(so.pc, "p"), dummies(so.by, "b")], axis=1).assign(const=1.0)
    for out_, lab_ in [("college", "college completion (children with a row at 30 and over)"), ("y", "mean log earnings at 30 and over")]:
        r = cl(sw * so[out_].values, sw[:, None] * pd.concat([K, Xb], axis=1).values, so.pc)
        for j, k in enumerate(K.columns): rows.append(dict(outcome=lab_, bin=k, children=int((K[k] != 0).sum()), coef=r.params[j], se=r.bse[j], p=r.pvalues[j]))
    Dd = pd.DataFrame(rows); Dd.to_csv(os.path.join(OUT, f"u9_D_cohort_bins{SFX}.csv"), index=False); print(Dd.round(3).to_string(index=False))
    if SFX == "":
        from fig_style import fs, setup, tr, outpath
        setup()
        fig, axes = plt.subplots(1, 2, figsize=fs((11.0, 4.2)), sharex=True)
        labels = ["1972" + tr(" or\nearlier"), "1973\u201376", "1977\u201380", "1981\u201384", "1985\u201388", "1989" + tr(" or\nlater")]   # the short form of the figure of first stages
        keys = [bins[0][2], bins[1][2], None, bins[2][2], bins[3][2], bins[4][2]]
        for ax, (out_, title) in zip(axes, [("college completion (children with a row at 30 and over)", "(a) College completion"), ("mean log earnings at 30 and over", "(b) Level of earnings")]):
            sd = Dd[Dd.outcome == out_].set_index("bin"); xs, ys, es = [], [], []
            for i, k in enumerate(keys):
                if k is None: ax.plot(i, 0, "D", mfc="none", mec="#08519c", ms=6)
                else: xs.append(i); ys.append(sd.loc[k, "coef"]); es.append(1.96 * sd.loc[k, "se"])
            ax.errorbar(xs, ys, yerr=es, fmt="o", color="#08519c", capsize=2.5, ms=4, lw=1.2)
            ax.axhline(0, color="k", lw=.8); ax.axvline(2.5, color="grey", ls="--", lw=.8)
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.5); ax.set_title(tr(title)); ax.set_xlabel(tr("Birth cohort"))
            ax.set_ylabel(tr("Interaction with provincial intensity") + "\n" + tr("(completion rate per unit)" if "completion" in out_ else "(log points per unit)"), fontsize=7.5)
        axes[0].plot([], [], "D", mfc="none", mec="#08519c", ls="", label=tr("reference bin (omitted)")); axes[0].legend(fontsize=7.5, loc="upper left")
        fig.tight_layout(); from fig_style import save; save(fig, "fig_mature_age.png")
print("wrote u9_*.csv")
