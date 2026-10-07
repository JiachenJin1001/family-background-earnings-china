"""Write every table and figure of Part I of the online appendix from the
pipeline outputs, so that no number in the appendix is typed by hand.

Reads output CSVs of the three pipelines (unified, income dynamics, college
expansion) and writes
  online_appendix/tables/*.tex    tabular bodies, \\input by online_appendix.tex
  online_appendix/figures/*.png   two figures redrawn with labels in words
Run from anywhere:  python3 online_appendix/build/build_tables.py
"""
import os, sys, csv, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.abspath(os.path.join(HERE, ".."))


def sibling(*names):
    cur = OA
    for _ in range(8):
        for nm in names:
            if os.path.isdir(os.path.join(cur, nm)):
                return os.path.join(cur, nm)
        cur = os.path.dirname(cur)
    raise FileNotFoundError(names)


U = sibling("unified", "unified")
M = sibling("income_dynamics", "income_dynamics")
P = sibling("college_expansion", "college_expansion")
TAB = os.environ.get("OA_TABLES_DIR", os.path.join(OA, "tables"))
FIG = os.environ.get("OA_FIGURES_DIR", os.path.join(OA, "figures"))
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def u(name): return pd.read_csv(os.path.join(U, "output", name))
def m(*parts): return pd.read_csv(os.path.join(M, *parts))
def p(*parts): return pd.read_csv(os.path.join(P, *parts))


def num(x, d=3):
    """Number with d decimals, minus sign set in math mode."""
    s = f"{x:.{d}f}"
    if float(s) == 0: s = s.replace("-", "")
    return f"$-${s[1:]}" if s.startswith("-") else s


def intc(x): return f"{int(round(x)):,}"


# fragments that the appendix no longer prints (their code is kept; nothing is written for them)
NUMWORDS = {15: "fifteen"}   # small counts are spelled out in the text
NOT_PRINTED = {"oa_maturedesc", "oa_selection"}


def frag(name, text):
    """A piece of text read with \\input inside a sentence; the closing % keeps a space from following it."""
    open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8").write(text + "%\n")


def write(name, header, rows, colspec, note=None):
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    lines += [r if r.startswith("\\") else r + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if name in NOT_PRINTED:
        return
    open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", name)


# ------------------------------------------------------------ A. data
F14 = u("u14_B_sample_facts.csv").set_index("item")["value"]
adv, less = "advantaged", "less advantaged"
n1 = {g: F14[f"children with a retrospective report observed at ages 22 to 55 in 2014-2022, {g}"] for g in (adv, less)}
n2 = {adv: F14["all children: advantaged children"], less: F14["all children: children"] - F14["all children: advantaged children"]}
n3 = {adv: F14["three waves: advantaged children"], less: F14["three waves: children"] - F14["three waves: advantaged children"]}
n4 = {adv: F14["level at 30 and over: advantaged children"], less: F14["level at 30 and over: children"] - F14["level at 30 and over: advantaged children"]}
rows = [f"Retrospective report, family income of the parental household, observed at ages 22--55 in a 2014--2022 wave & {intc(n1[adv])} & {intc(n1[less])} & 100.0 & 100.0",
        f"Trimming and at least two waves of positive labor income at ages 22--55 (the sample of the paper) & {intc(n2[adv])} & {intc(n2[less])} & {100 * F14[f'share of them in the sample, {adv}']:.1f} & {100 * F14[f'share of them in the sample, {less}']:.1f}",
        f"Of these, at least three waves (Sections 3 and 4 of the paper) & {intc(n3[adv])} & {intc(n3[less])} & {100 * F14[f'share of them with three waves, {adv}']:.1f} & {100 * F14[f'share of them with three waves, {less}']:.1f}",
        f"Of the sample, at least two waves at ages 30 and over (level of earnings, Sections 5 to 7 of the paper) & {intc(n4[adv])} & {intc(n4[less])} & {100 * n4[adv] / n1[adv]:.1f} & {100 * n4[less] / n1[less]:.1f}"]
write("oa_screens", " & \\multicolumn{2}{c}{Children} & \\multicolumn{2}{c}{Percent of first row} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nScreen & Advantaged & Less advantaged & Advantaged & Less advantaged",
      rows, "@{}p{6.0cm}cccc@{}")
frag("oa_employment",
    f"{100 * F14[f'share of their person-waves with positive labor income, {adv}']:.1f} percent of the person-waves of advantaged and {100 * F14[f'share of their person-waves with positive labor income, {less}']:.1f} percent of the person-waves of less advantaged children")

# ------------------------------------------------------------ B. earnings process
j = m("data", "intermediate", "m16_joint_moment_test.csv").iloc[0]
frag("oa_jointtest",
    f"The statistic is {j.chi2:.1f} on {NUMWORDS.get(int(j.df), str(int(j.df)))} degrees of freedom ($p = {j.p:.2f}$); a bootstrap-calibrated version that compares the statistic with the distribution of centered bootstrap draws gives $p = {j.p_boot:.3f}$ with {intc(j.B)} draws")
# variances and first-order autocovariances by wave (the table that the paper's figure of autocovariances stands for)
m5 = m("output", "tables", "m5_moment_se.csv"); m5["t"] = m5.t.astype(int); m5["s"] = m5.s.astype(int)
M5 = {(r.group, r.t, r.s): (r.omega, r.se) for r in m5.itertuples()}; yrs = [2014, 2016, 2018, 2020, 2022]
def _mrow(lab, g, pairs, pad): return f"{lab} & " + " & ".join(f"{M5[(g, a, b)][0]:.2f} ({M5[(g, a, b)][1]:.2f})" for a, b in pairs) + pad
_grp = [("Pooled", "pooled"), ("Advantaged", "H"), ("Less advantaged", "L")]
_lines = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule", "& \\multicolumn{5}{c}{Variances} \\\\", "\\cmidrule(lr){2-6}", "& " + " & ".join(str(y) for y in yrs) + " \\\\", "\\midrule"]
_lines += [_mrow(lab, g, [(y, y) for y in yrs], " \\\\") for lab, g in _grp]
_lines += ["\\midrule", "& \\multicolumn{5}{c}{First-order autocovariances} \\\\", "\\cmidrule(lr){2-6}", "& " + " & ".join(f"{a}, {b}" for a, b in zip(yrs[:-1], yrs[1:])) + " & \\\\", "\\midrule"]
_lines += [_mrow(lab, g, list(zip(yrs[:-1], yrs[1:])), " & \\\\") for lab, g in _grp]
open(os.path.join(TAB, "oa_autocov.tex"), "w", encoding="utf-8").write("\n".join(_lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"); print("wrote oa_autocov")
_z = {(a, b): abs(M5[("H", a, b)][0] - M5[("L", a, b)][0]) / math.hypot(M5[("H", a, b)][1], M5[("L", a, b)][1]) for (g, a, b) in M5 if g == "H"}
_k = max(_z, key=_z.get); frag("oa_autocovmax", f"the largest standardized gap is {_z[_k]:.2f}, at the ({_k[0]},\\,{_k[1]}) covariance")
mde = m("output", "tables", "m13_mde.csv")
rows = []
for _, r in mde.iterrows():
    gap = "---" if pd.isna(r.gap) else num(r.gap)
    lab = {"permanent variance": "Permanent variance", "transitory variance": "Variance of transitory shocks",
           "transitory persistence": "Transitory persistence", "permanent share": "Permanent share",
           "AB persistence interaction": "Arellano--Bond persistence interaction"}[r.param]
    rows.append(f"{lab} & {gap} & {r.se_gap:.3f} & {r.mde80:.2f}")
write("oa_mde", " & Gap, advantaged minus & Standard & Minimum detectable \\\\\nParameter & less advantaged children & error & difference", rows, "@{}lccc@{}")

ab = m("data", "intermediate", "m17_ab_diagnostics.csv")
rows = [f"{'Five waves, 2014--2022' if r.panel.startswith('5') else 'Seven waves, 2010--2022 (reconstructed)'} & {r.rho0_base:.3f} ({r.se_base:.3f}) & {num(r.m1_z, 2)} & {num(r.m2_z, 2)} [{r.m2_p:.2f}] & {num(r.rho0_lag3, 2)} ({r.se_lag3:.2f})"
        for _, r in ab.iterrows()]
write("oa_abdiag", " & Persistence, & & & Persistence, instruments \\\\\n & main & & & lagged three or \\\\\nPanel & estimate (SE) & $m_1$ ($z$) & $m_2$ ($z$ [$p$]) & more waves (SE)", rows, "@{}p{4.6cm}cccc@{}")

# criterion values under alternative error structures
jc = u("u7_error_structures.csv").set_index("sample")
rows = [f"{lab} & {jc.loc[k,'criterion_ar1']:.1f} & {jc.loc[k,'criterion_arma11']:.1f} & {jc.loc[k,'criterion_ma1']:.1f} & {jc.loc[k,'criterion_random_walk']:.1f}"
        for k, lab in [("pooled", "Pooled"), ("advantaged children", "Advantaged children"), ("ordinary children", "Less advantaged children")]]
write("oa_criterion", " & Permanent effect & Permanent effect & Permanent effect & Random-walk \\\\\nSample & plus AR(1) & plus ARMA(1,1) & plus MA(1) & permanent component",
      rows, "@{}lcccc@{}")

# ------------------------------------------------------------ C. measurement
rk = m("output", "tables", "m8_rank_attenuation.csv")
rows = [f"{int(r['T'])} & {r.eta_T:.2f} & {r.rank_factor:.2f} & {r.rank_factor_2sided:.2f}" for _, r in rk.iterrows()]
write("oa_rank", "Waves & Share recovered, & Rank-rank slope, child's rank & Rank-rank slope, child's rank \\\\\naveraged & levels (exact) & from permanent income & from three waves",
      rows, "@{}cccc@{}")
par = m("output", "tables", "m14_parent_process.csv").iloc[0]
rows = [f"Parents (person-waves) & {intc(par.n_parents)} ({intc(par.n_obs)})",
        f"Variance of residual log earnings & {par['var']:.3f}",
        f"First-order autocovariance & {par.cov1:.3f}",
        f"Autocovariance at longer lags & {par.cov_hi:.3f}",
        f"Permanent share & {par.share:.2f} [{par.share_lo:.2f}, {par.share_hi:.2f}]",
        f"Transitory persistence & {par.rho:.2f}"]
write("oa_parent", "Quantity & Value", rows, "@{}lc@{}")

# attenuation figure, redrawn
_m7 = m("output", "tables", "m7_md_cellw_boot.csv").set_index("group").loc["pooled"]   # pooled estimates of the paper's minimum-distance table
s2e, s2eps, rho = float(_m7.eta2), float(_m7.eps2), float(_m7.rho)
s2v = s2eps / (1 - rho ** 2)
T = np.arange(1, 11)
approx = s2e / (s2e + s2v / T)
fT = np.array([(t + 2 * sum((t - k) * rho ** k for k in range(1, t))) / t ** 2 for t in T])
exact = s2e / (s2e + s2v * fT)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(T, approx, "o-", color="#1f77b4", label="independence approximation")
ax.plot(T, exact, "s--", color="#d62728", label="exact (serially correlated waves)")
ax.axhline(2 / 3, color="grey", lw=.8, ls="--")
ax.text(1.05, 2 / 3 + 0.015, "two thirds of the true coefficient", fontsize=8, color="grey")
ax.set_xticks(T); ax.set_ylim(0, 1); ax.set_xlabel("Number of waves averaged, $T$")
ax.set_ylabel("Share of the true coefficient recovered"); ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=.25)
fig.savefig(os.path.join(FIG, "oa_attenuation.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------ D. premium
LABELS = [("baseline: family income of the parental household", "Correlated random effects, family-income control (baseline)"),
          ("parents' own labor income as the income control", "Correlated random effects, parents' own labor income as the control"),
          ("income deflated by the provincial consumer price index", "Correlated random effects, income deflated by provincial CPI"),
          ("most frequent retrospective report of parental occupation", "Correlated random effects, indicator from the most frequent retrospective report"),
          ("indicator widened to CSCO major groups 1 to 3", "Correlated random effects, indicator widened to CSCO groups 1--3"),
          ("indicator narrowed to CSCO major group 1", "Correlated random effects, indicator narrowed to CSCO group 1"),
          ("three regions in place of province effects", "Correlated random effects, region effects in place of province effects"),
          ("parental income from the 2010 and 2012 waves, before the earnings window", "Correlated random effects, family income from the 2010 and 2012 waves only"),
          ("parental income from waves with the child aged 40 or less", "Correlated random effects, family income from waves with the child aged 40 or less"),
          ("parental income from waves with the child aged 35 or less", "Correlated random effects, the same with the child aged 35 or less"),
          ("children recorded as rural in most of their waves", "Correlated random effects, children recorded as rural in most of their waves"),
          ("children recorded as urban in most of their waves", "Correlated random effects, children recorded as urban in most of their waves"),
          ("parental party membership added (cross-section)", "Cross-section, parental party membership added"),
          ("parental state-sector employment added (cross-section)", "Cross-section, parental state-sector employment added"),
          ("entropy balancing", "Entropy balancing")]
S8 = u("u8_premium_specifications.csv").set_index("specification")
assert set(S8.index[S8.in_figure_2]) == {k for k, _ in LABELS}
rows = [f"{lab} & {num(S8.loc[k, 'estimate'])} & [{num(S8.loc[k, 'lo'])}, {num(S8.loc[k, 'hi'])}] & {intc(S8.loc[k, 'children'])}" for k, lab in LABELS]
write("oa_grid", "Specification & Estimate & 95 percent interval & Children", rows, "@{}p{9.4cm}ccc@{}")

age = u("u4_premium_age_controls.csv").set_index("spec")
rows = [f"{lab} & {age.loc[k,'b']:.3f} & ({age.loc[k,'se']:.3f})" for k, lab in
        [("linear", "Linear in age (the paper's specification)"), ("quad", "Quadratic in age"),
         ("cubic", "Cubic in age"), ("dummies", "One indicator per year of age")]]
write("oa_age", "Age control & Coefficient on the indicator for advantaged children & SE", rows, "@{}lcc@{}")

# ------------------------------------------------------------ E. expansion design
ex = u("u3_B_exposure_series.csv")
half = (len(ex) + 1) // 2
rows = []
for i in range(half):
    l = ex.iloc[i]; r = ex.iloc[i + half] if i + half < len(ex) else None
    left = f"{int(l.birth_year)} & {int(l.year)} & {l.recruitment_M:.2f} & {l.projected_M:.2f} & {l.exposure_ratio:.2f}"
    right = f"{int(r.birth_year)} & {int(r.year)} & {r.recruitment_M:.2f} & {r.projected_M:.2f} & {r.exposure_ratio:.2f}" if r is not None else " & & & & "
    rows.append(left + " & & " + right)
hdr = "Birth & Year at & Admissions & Trend & Exposure & & Birth & Year at & Admissions & Trend & Exposure \\\\\nyear & age 18 & (millions) & projection & ratio & & year & age 18 & (millions) & projection & ratio"
write("oa_exposure", hdr, rows, "@{}ccccc@{\\hspace{14pt}}c@{}ccccc@{}")

pv = u("u3_C_intensity_by_province.csv")
NAMES = {"InnerMongolia": "Inner Mongolia"}
rows = [f"{NAMES.get(r.province_en, r.province_en)} & {r.pro_predict_growth:.3f} & {r.flow_per_worker:.3f} & {r.log_growth_implied:.3f} & {r.hs2000:.3f} & {num(r.int_placebo_log_1998_1997)}"
        for _, r in pv.iterrows()]
write("oa_intensity", " & Inflow per 2005 & Inflow per & Log growth of the & Share with senior & Registration growth \\\\\n & college-educated & 2005 & college-educated & secondary schooling & 1998/1997 \\\\\nProvince & worker & worker & stock to 2010 & (2000 census) & (placebo)",
      rows, "@{}lccccc@{}")

A3 = u("u3_A_occupation_definitions.csv")
LAB = {"ever": "Ever observed in a professional occupation", "majority of waves": "In a professional occupation in most observed waves",
       "share of waves": "Share of observed waves in a professional occupation", "last observed wave": "In a professional occupation at the last observed wave",
       "two or more waves": "In a professional occupation in two or more waves"}
rows = [f"{LAB[r.definition]} & {r['mean']:.2f} & {r.rf:.3f} ({r.se:.3f}) & {r.p_wild:.3f}" for _, r in A3.iterrows()]
write("oa_occdefs", "Definition of the outcome & Mean & Reduced form (SE) & Wild-cluster $p$", rows, "@{}p{8.2cm}ccc@{}")

D = u("u14_D_return_by_age.csv").set_index("ages")
rows = [f"{k.replace('-', '--')} & {num(D.loc[k, 'coef'])} & ({D.loc[k, 'se']:.3f}) & {intc(D.loc[k, 'child_waves'])}" for k in ("22-25", "26-29", "30-34", "35-55")]
write("oa_premiumage", "Age at observation & Least squares return to college & SE & Child-wave observations", rows, "@{}lccc@{}")

cen = p("output", "tables", "p16_census_pretrend.csv")
cen = cen.dropna(subset=["n_cells"])
rows = [f"{r.census.replace(' census','')} & {r.slope_diff:.5f} & {r.se:.5f} & {r.p:.3f} & {intc(r.n_cells)}" for _, r in cen.iterrows()]
write("oa_census", "Census & Difference in cohort slope & SE & $p$ & Province-cohort cells", rows, "@{}lcccc@{}")


# ------------------------------------------------------------ E. earnings at ages 30 and over: the two-row regression, its variants and checks
MAIN = "predicted provincial growth (main measure)"
B9 = u("u9_B_main.csv").set_index("intensity").loc[MAIN]; B9w = u("u9_B_main_weighted.csv").set_index("intensity").loc[MAIN]
C9 = u("u9_C_checks.csv").set_index("check"); C9w = u("u9_C_checks_weighted.csv").set_index("check")
def vrow(lab, n, rows_old, F, b, se, pw, iv):
    return f"{lab} & {intc(n)} & {intc(rows_old)} & {F:.1f} & {num(b)} ({se:.3f}) [{pw:.3f}] & {num(iv, 2)}"
o_, y2 = "rows at 30 and over only (no row below 30)", "row below 30 with at least 2 wave(s)"
rows = [vrow("Two rows per child, unweighted (the paper's main specification)", B9.n, B9.rows_old, B9.F, B9.earn, B9.earn_se, B9.earn_p_wild, B9.earn_iv),
        vrow("Rows at 30 and over only", C9.loc[o_, "n"], C9.loc[o_, "rows_old"], C9.loc[o_, "F"], C9.loc[o_, "coef"], C9.loc[o_, "se"], C9.loc[o_, "p_wild"], C9.loc[o_, "iv"]),
        vrow("Two rows per child, the row below 30 kept only if it averages two or more waves", C9.loc[y2, "n"], C9.loc[y2, "rows_old"], C9.loc[y2, "F"], C9.loc[y2, "coef"], C9.loc[y2, "se"], C9.loc[y2, "p_wild"], C9.loc[y2, "iv"]),
        vrow("Two rows per child, weighted by the precision implied by the earnings process", B9w.n, B9w.rows_old, B9w.F, B9w.earn, B9w.earn_se, B9w.earn_p_wild, B9w.earn_iv),
        vrow("Rows at 30 and over only, weighted in the same way", C9w.loc[o_, "n"], C9w.loc[o_, "rows_old"], C9w.loc[o_, "F"], C9w.loc[o_, "coef"], C9w.loc[o_, "se"], C9w.loc[o_, "p_wild"], C9w.loc[o_, "iv"])]
write("oa_variants", " & & Rows at 30 & First-stage & Earnings reduced form & Return to \\\\\nSpecification & Children & and over & $F$ & (SE) [wild $p$] & college", rows, "@{}p{6.2cm}ccccc@{}")
def crow(lab, key, iv=True):
    out = [lab, intc(C9.loc[key, "n"]) if pd.notna(C9.loc[key, "n"]) else "---"]
    for C in (C9, C9w):
        r = C.loc[key]; out += [f"{num(r.coef)} ({r.se:.3f}) [{r.p_wild:.3f}]", num(r.iv, 2) if iv else "---"]
    return " & ".join(out)
rows = [crow("Children with nine or fewer years of schooling", "earnings, children with 9 or fewer years of schooling", iv=False),
        crow("Children with twelve or fewer years of schooling", "earnings, children with 12 or fewer years of schooling", iv=False),
        crow("Children with nine or fewer years of schooling, professional occupation as the outcome", "professional occupation, children with 9 or fewer years of schooling", iv=False),
        crow("Children with twelve or fewer years of schooling, professional occupation as the outcome", "professional occupation, children with 12 or fewer years of schooling", iv=False),
        crow("Coastal province $\\times$ exposure added", "trade control: coastal x exposure"),
        crow("Export share $\\times$ exposure added", "trade control: export share x exposure"),
        crow("Both added", "trade control: both"),
        crow("Linear cohort trends for each of three regions added", "region-specific linear cohort trends (three regions)"),
        crow("Linear cohort trend for each province added", "province-specific linear cohort trends"),
        crow("Instrument assigned by the province at age twelve", "instrument by the province at age twelve"),
        crow("Cohort exposure taken at age seventeen instead of eighteen", "exposure ratio of the year the cohort turned 17"),
        crow("Cohort exposure taken at age nineteen instead of eighteen", "exposure ratio of the year the cohort turned 19"),
        crow("Cohorts born 1973 to 1976 left out", "cohorts born 1973 to 1976 left out")]
write("oa_maturechecks", " & & \\multicolumn{2}{c}{Unweighted rows} & \\multicolumn{2}{c}{Rows weighted by precision} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n & & Reduced form & Return to & Reduced form & Return to \\\\\nSample or added control & Children & (SE) [wild $p$] & college & (SE) [wild $p$] & college", rows, "@{}p{4.6cm}ccccc@{}")
_loo = 'leave one province out: instrumented return, smallest and largest'
frag("oa_matureloo", f"$[{C9.loc[_loo, 'coef']:.2f}, {C9.loc[_loo, 'se']:.2f}]$")
frag("oa_maturetrendF", f"{C9.loc['province-specific linear cohort trends', 'F']:.1f}")
frag("oa_pretrendp", f"{C9.loc['pre-reform cohorts: earnings on intensity x birth year', 'p_wild']:.2f} for earnings and {C9.loc['pre-reform cohorts: completion on intensity x birth year', 'p_wild']:.2f} for college completion")
# returns by family background, unweighted and weighted rows
G12 = u("u12_returns_by_family_background.csv").set_index("group"); G12w = u("u12_returns_by_family_background_weighted.csv").set_index("group")
GRID12 = {"": u("u12_anderson_rubin_grid.csv"), "w": u("u12_anderson_rubin_grid_weighted.csv")}   # wild-cluster p-value at every grid point
def ar(grid, grp, level):
    """The grid points not rejected at the level, as a union of intervals; an end at the edge of the grid is printed as infinite."""
    d = grid[grid.group == grp].sort_values("b"); bs = d.b.round(2).tolist(); step = round(bs[1] - bs[0], 4)
    acc = d[d.p_wild >= level].b.round(2).tolist(); segs = []
    for x in acc:
        if segs and abs(x - segs[-1][1] - step) < 1e-6: segs[-1][1] = x
        else: segs.append([x, x])
    parts = [("(-\\infty" if lo <= bs[0] else f"[{lo:.2f}") + ", " + ("\\infty)" if hi >= bs[-1] else f"{hi:.2f}]") for lo, hi in segs]
    if len(parts) == 1 and segs[0][0] <= bs[0] and segs[0][1] >= bs[-1]: return "unbounded"
    return "$" + " \\cup ".join(parts) + "$"
# the regression for all children (Table 10 of the paper, main measure), unweighted and weighted, with its grid of p-values
B9A = {"": u("u9_B_main.csv").set_index("intensity"), "w": u("u9_B_main_weighted.csv").set_index("intensity")}
GRID9A = {"": u("u9_anderson_rubin_grid.csv"), "w": u("u9_anderson_rubin_grid_weighted.csv")}
rows = []
for lab, G, gk in [("Unweighted rows", G12, ""), ("Rows weighted by precision", G12w, "w")]:
    rows.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{lab}}}}} \\\\")
    _b = B9A[gk].loc["predicted provincial growth (main measure)"]
    rows.append(f"All children & {_b.F:.1f} & {num(_b.earn)} ({_b.earn_se:.3f}) [{_b.earn_p_wild:.3f}] & {num(_b.earn_iv, 2)} & {ar(GRID9A[gk], 'all children', 0.05)} & {ar(GRID9A[gk], 'all children', 0.10)}")
    for g, gl in [("less advantaged", "Less advantaged children"), ("advantaged", "Advantaged children")]:
        r = G.loc[g]; rows.append(f"{gl} & {r.F:.1f} & {num(r.rf)} ({r.rf_se:.3f}) [{r.rf_p_wild:.3f}] & {num(r.beta, 2)} ({r.beta_se:.2f}) & {ar(GRID12[gk], g, 0.05)} & {ar(GRID12[gk], g, 0.10)}")
write("oa_groups", " & First-stage & Earnings reduced form & Return to & \\multicolumn{2}{c}{Anderson--Rubin set} \\\\\n\\cmidrule(lr){5-6}\nFamily background & $F$ & (SE) [wild $p$] & college (SE) & 95 percent & 90 percent", rows, "@{}lccccc@{}")

# ------------------------------------------------------------ the children with a level of earnings at 30 and over, by family background
LS = u("u8_level_sample.csv").set_index("item")
rows = []
for name, lab, d in [("mean log labor income", "Level of earnings at ages 30 and over (mean log labor income net of survey-year effects)", 2), ("log family income of the parental household", "Log family income of the parental household", 2),
                     ("years of schooling", "Years of schooling", 2), ("college completion", "College (16 or more years)", 2), ("female", "Female", 2),
                     ("urban residence", "Urban residence (first wave at 30 and over)", 2), ("age over the waves in the window", "Age over the waves at 30 and over", 1),
                     ("birth year", "Birth year", 1), ("waves in the window", "Waves with labor income at 30 and over", 2)]:
    a_, b_ = LS.loc[f"{name}, advantaged children"], LS.loc[f"{name}, less advantaged children"]
    rows.append(f"{lab} & {a_.value:.{d}f} & {a_.sd:.{d}f} & {b_.value:.{d}f} & {b_.sd:.{d}f}")
nA = int(LS.loc["advantaged children", "value"]); nL = int(LS.loc["children", "value"]) - nA
write("oa_maturedesc", f" & \\multicolumn{{2}}{{c}}{{Advantaged ($N = {nA}$)}} & \\multicolumn{{2}}{{c}}{{Less advantaged ($N = {nL:,}$)}} \\\\\n\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}\n & Mean & SD & Mean & SD", rows, "@{}lcccc@{}")

# ------------------------------------------------------------ F. decomposition
Dg = u("u3_D_decomposition_grid.csv")
LABP = {"correlated random effects, family income": "Correlated random effects, family income",
        "correlated random effects, parental labor income": "Correlated random effects, parents' labor income",
        "entropy balancing": "Entropy balancing"}
rows = []
for k in LABP:
    d = Dg[Dg.premium_estimate == k]
    a_, b_ = d.iloc[0], d.iloc[1]
    rows.append(f"{LABP[k]} & {a_.premium:.3f} & {a_.college_share:.2f} & {b_.college_share:.2f} & {a_.return_for_half:.3f} & {a_.return_for_all:.3f}")
write("oa_decomp", " & & \\multicolumn{2}{c}{College share at the} & \\multicolumn{2}{c}{Return at which} \\\\\n & & \\multicolumn{2}{c}{return to college} & \\multicolumn{2}{c}{college explains} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\nEstimate of the premium & Premium & Least squares & Instrumented & Half & All",
      rows, "@{}p{5.4cm}ccccc@{}")
S = u("u3_E_selection_sensitivity.csv")
rows = [f"No controls & {S.iloc[0].premium:.3f} & {S.iloc[0].r2:.3f} & ---",
        f"Family background, parental income, gender, urban residence, province and birth-year effects & {S.iloc[1].premium:.3f} & {S.iloc[1].r2:.3f} & ---"]
for _, r in S.iloc[2:].iterrows():
    mult = r["item"].split("= ")[1].split(" times")[0]
    rows.append(f"Adjusted, maximum $R^2$ equal to {mult} times the controlled $R^2$ & {r.premium:.3f} & {r.r2:.3f} & {r.college_share:.2f}")
write("oa_selection", " & Return to & & College share \\\\\nRegression & college & $R^2$ & of the premium", rows, "@{}p{8.6cm}ccc@{}")
print("done")


# ------------------------------------------------------------ randomization inference (Section E.7)
_ra = u("u9_R_randomization_inference.csv").set_index("statistic"); _rw = u("u9_R_randomization_within_region.csv").iloc[0]
_n = int(_ra.loc["first stage", "reassignments"]); _cnt = lambda pv: int(round(pv * (_n + 1))) - 1
_pf = lambda pv: f"{pv:.4f}"
rows = [f"Among all {int(_ra.loc['first stage', 'provinces'])} provinces & {_cnt(_ra.loc['earnings reduced form', 'permutation_p'])} & {_pf(_ra.loc['earnings reduced form', 'permutation_p'])} & {_cnt(_ra.loc['first stage', 'permutation_p'])} & {_pf(_ra.loc['first stage', 'permutation_p'])} & --- & ---",
        f"Within region (east, center, west) & {_cnt(_rw['permutation_p_reduced_form'])} & {_pf(_rw['permutation_p_reduced_form'])} & {_cnt(_rw['permutation_p_first_stage'])} & {_pf(_rw['permutation_p_first_stage'])} & $[{_rw['set95_lo']:.2f}, {_rw['set95_hi']:.2f}]$ & $[{_rw['set90_lo']:.2f}, {_rw['set90_hi']:.2f}]$"]
write("oa_randomization", " & \\multicolumn{2}{c}{Earnings reduced form} & \\multicolumn{2}{c}{First stage} & \\multicolumn{2}{c}{Set for the return to college} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\nReassignment of provincial intensity & As large as actual & $p$ & As large as actual & $p$ & 95 percent & 90 percent", rows, "@{}lcccccc@{}")


# ------------------------------------------------------------ nonlinear persistence: the run of the paper and ten runs (Section B)
_ab = m("data", "intermediate", "ws", "ws_05b_abb_stochasticEM_surface.csv"); _main = _ab.groupby("group").rho_local.mean()
_r15 = u("u15_nonlinear_persistence_summary.csv").set_index("sample")
rows = [f"{lab} & {_main[k]:.2f} & {_r15.loc[r, 'mean']:.2f} & {_r15.loc[r, 'min']:.2f} & {_r15.loc[r, 'max']:.2f}"
        for lab, k, r in [("All children with at least three waves", "pooled", "pooled"), ("Advantaged children", "H", "advantaged children"), ("Less advantaged children", "L", "less advantaged children")]]
write("oa_nonlinear", " & Run reported & \\multicolumn{3}{c}{Ten further runs} \\\\\n\\cmidrule(lr){3-5}\nChildren & in the paper & Mean & Smallest & Largest", rows, "@{}lcccc@{}")
