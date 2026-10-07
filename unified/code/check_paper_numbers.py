"""Checks of the numbers printed in the paper.

Each check reads a file of estimates, formats the value as the paper prints
it, and looks for that string in paper/main.tex or in the table fragments
that the paper inputs.

    python3 unified/code/check_paper_numbers.py
"""
import os, re, sys, io, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WS = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, HERE)
from _locate import sibling
PKG = {"income_dynamics": sibling("income_dynamics", __file__), "college_expansion": sibling("college_expansion", __file__)}
TEX = open(os.path.join(ROOT, "paper", "main.tex")).read()
# the tables of estimates are generated fragments (code/build_paper_tables.py); they are part of the text checked here
# (only the fragments that main.tex inputs, so that no assertion can be satisfied by a fragment the paper does not print)
for _f in re.findall(r"\\input\{(tables/en/[^}]+)\}", TEX):
    TEX += "\n" + open(os.path.join(ROOT, "paper", _f + ".tex"), encoding="utf-8").read()
TEXN = re.sub(r"\s+", " ", TEX)

fails = 0

# ====================================================================================================================
# Sentences of the unified paper, each rebuilt from the files of estimates. One sample: ages 22 to 55, at least two
# waves (all children); the children with at least three waves (Sections 3 and 4); the level of earnings at ages 30
# and over (Sections 5 to 7); the two-row regression of Section 6.
# ====================================================================================================================
import csv, math
MP = PKG["income_dynamics"]; FR = sibling("formal_results", __file__)


def rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))
def u1(name):
    return rd(os.path.join(ROOT, "output", name))
def mt(name):
    return rd(os.path.join(MP, "output", "tables", name))
def mi(*name):
    return rd(os.path.join(MP, "data", "intermediate", *name))
def chk(lbl, snip, count=None):
    """count: exact number of occurrences required (default: at least one)."""
    global fails
    n = TEXN.count(re.sub(r"\s+", " ", snip))
    ok = n >= 1 if count is None else n == count
    print(("  ok   " if ok else "  FAIL ") + f"{lbl:58s} {snip!r}")
    fails += (not ok)
def need(cond, lbl):
    global fails
    print(("  ok   " if cond else "  FAIL ") + f"{lbl}")
    fails += (not cond)
pct = lambda v: 100 * (math.exp(v) - 1)
sg = lambda v, d=3: (f"$-{abs(v):.{d}f}$" if round(v, d) < 0 else f"{v:.{d}f}")
pf = lambda p: "$p < 0.001$" if p < 0.0005 else f"$p = {p:.3f}$"

F14 = {r["item"]: float(r["value"]) for r in u1("u14_B_sample_facts.csv")}
ALL, THREE, LEVEL = "all children", "three waves", "level at 30 and over"
n_all, a_all, n3, a3, nl, al = [int(F14[k]) for k in (f"{ALL}: children", f"{ALL}: advantaged children", f"{THREE}: children", f"{THREE}: advantaged children", f"{LEVEL}: children", f"{LEVEL}: advantaged children")]

print("=== Section 2: the sample")
chk("sample size", f"The sample has {n_all:,} children ({a_all} advantaged, {100 * F14[f'{ALL}: share advantaged']:.1f} percent) and {int(F14[f'{ALL}: child-waves']):,} child-waves, born {int(F14[f'{ALL}: first birth year'])} to {int(F14[f'{ALL}: last birth year'])}.")
chk("children with three waves", f"and {n3:,} children ({a3} advantaged) have them, with {int(F14[f'{THREE}: child-waves']):,} child-waves")
chk("children with a level of earnings", f"{nl:,} children ({al} advantaged) have one, and {int(F14['children with three waves and a level at 30 and over']):,} children have both three waves and a level")
chk("all children in Section 6", f"Section~\\ref{{sec:expansion}} keeps all {n_all:,} children")
chk("provinces", f"the sample of Section~\\ref{{sec:sample}} spans {int(F14[f'{ALL}: provinces'])} provinces")
D14 = {r["ages"]: r for r in u1("u14_D_return_by_age.csv")}; d = lambda k, c="coef": float(D14[k][c])
chk("return to college by age", f"is {sg(d('22-25'), 2)} log points at ages 22 to 25 (SE {d('22-25', 'se'):.2f}), {d('26-29'):.2f} at 26 to 29, {d('30-34'):.2f} at 30 to 34, and {d('35-55'):.2f} at 35 to 55")
chk("survey-year means", f"rose from {F14['mean log income at 30 and over, survey year 2014']:.2f} in the 2014 wave to {F14['mean log income at 30 and over, survey year 2022']:.2f} in the 2022 wave")
chk("survey-year means in words", "(by more than four fifths in nominal terms)"); need(0.80 < pct(F14['mean log income at 30 and over, survey year 2022'] - F14['mean log income at 30 and over, survey year 2014']) / 100 < 0.90, "nominal income at 30 and over rose by more than four fifths between the 2014 and 2022 waves")
chk("share entering the sample", f"{100 * F14['share of them in the sample, advantaged']:.0f} percent satisfy every screen and enter the sample; among less advantaged children the figure is {100 * F14['share of them in the sample, less advantaged']:.0f} percent")
chk("share with three waves", f"For the children with three waves the figures are {100 * F14['share of them with three waves, advantaged']:.0f} and {100 * F14['share of them with three waves, less advantaged']:.0f} percent")
chk("positive earnings", f"earnings are positive in {100 * F14['share of their person-waves with positive labor income, advantaged']:.0f} percent of the person-waves of advantaged children and in {100 * F14['share of their person-waves with positive labor income, less advantaged']:.0f} percent")
chk("one-parent reports", f"For {int(F14['all children: retrospective report for one parent only'])} children in the sample only one parent's occupation is reported")
chk("agreement of the two indicators", f"agree for {100 * F14[f'{ALL}: indicators agree']:.0f} percent of the children in the sample for whom both exist")
chk("disagreement, moved down", f"{100 * F14[f'{ALL}: advantaged by the retrospective indicator only']:.1f} percent of children have parents who held such an occupation")
chk("disagreement, moved up", f"against {100 * F14[f'{ALL}: advantaged by the current-occupation indicator only']:.1f} percent who moved up")
_rd = [r for r in u1("u5_ridge_draws.csv") if r["group"] == "advantaged"]; need(0 < sum(float(r["permanent_share"]) < 1e-6 for r in _rd) < 0.05 * len(_rd), "ridge: some bootstrap draws of advantaged children lie on the boundary, fewer than 5 percent")
need(0.60 < F14[f"{ALL}: share of child-waves on the household roster of a linked parent"] < 0.68, "co-residence: nearly two thirds of the child-waves")
A14 = {(r["sample"], r["variable"]): r for r in u1("u14_A_sample_means.csv")}
gapA = lambda smp, var: float(A14[(smp, var)]["advantaged_mean"]) - float(A14[(smp, var)]["less_advantaged_mean"])
chk("Table 2 in the text", "In both, advantaged children earn about 0.2 log points more on average; they have parents with more family income (by more than 0.4 log points)")
need(all(0.19 < gapA(s_, "Mean log labor income") < 0.215 and gapA(s_, "Log family income of the parental household") > 0.4 for s_ in (ALL, LEVEL)), "Table 2: earnings gaps of about 0.2 and family income gaps above 0.4 log points")
need(all(gapA(s_, "Years of schooling") > 2 and float(A14[(s_, "College (16 or more years)")]["advantaged_mean"]) > 2 * float(A14[(s_, "College (16 or more years)")]["less_advantaged_mean"]) for s_ in (ALL, LEVEL)), "Table 2: more than two years of schooling and more than twice the completion rate")
chk("Table 2 notes", f"All children: the {n_all:,} children with at least two waves")
chk("Table 2 notes, level", f"the {nl:,} with at least two waves at those ages")
chk("retirement ages", "55 is the middle of the three statutory retirement ages in force over the period (50 for women in worker posts, 55 for women in cadre posts, and 60 for men)")

print("=== Section 3: the earnings process (children with at least three waves)")
M7 = {r["group"]: {k: float(v) for k, v in r.items() if k != "group"} for r in mt("m7_md_cellw_boot.csv")}
P7, H7, L7 = M7["pooled"], M7["H"], M7["L"]
chk("pooled parameters, note of the attenuation table", f"$\\sigma^2_\\eta = {P7['eta2']:.3f}$, $\\sigma^2_\\varepsilon = {P7['eps2']:.3f}$, $\\rho_v = {P7['rho']:.2f}$")
chk("smaller group, weighting matrix", f"since with {a3} children in the smaller group")
chk("smaller group, table note", f"so its small value for advantaged children reflects a group of {a3} children, not better fit")
chk("pooled persistence in the text", f"Transitory shocks, which in survey data include reporting error, have a serial correlation of {P7['rho']:.2f} at the biennial")
need(0.25 < P7["sperm"] < 0.30, "pooled permanent share: a little over a quarter")
need(abs(math.sqrt(P7["eta2"]) - 0.35) < 0.01, "standard deviation of the permanent component about 0.35")
U7 = {r["sample"]: r for r in u1("u7_error_structures.csv")}["pooled"]
chk("criteria of the richer error structures", f"pooled criterion values of {float(U7['criterion_arma11']):.1f}, {float(U7['criterion_ma1']):.1f}, and {float(U7['criterion_random_walk']):.1f} against the baseline {float(U7['criterion_ar1']):.1f}")
W5 = {r["group"]: r for r in mt("m5_weighted_md.csv")}
chk("inverse-variance reweighting", f"moves the permanent share from {P7['sperm']:.3f} to {float(W5['pooled']['sperm_w']):.3f}")
chk("parametric bootstrap of the reweighted criterion", f"not among advantaged children ($p = {float(W5['H']['J_p_calibrated']):.2f}$); the two-component model is an approximation"); need(float(W5["pooled"]["J_p_calibrated"]) < 0.01 and float(W5["L"]["J_p_calibrated"]) < 0.01, "the calibrated criterion rejects in the pooled sample and among less advantaged children")
EXPO = {int(float(r["year"])): float(r["recruitment_M"]) for r in u1("u3_B_exposure_series.csv")}; chk("admissions in 1988 and 1997", f"from roughly {EXPO[1988]:.1f} million admissions in 1988 to one million in 1997"); need(abs(EXPO[1997] - 1.0) < 0.05 and 0.42 < EXPO[1999] / EXPO[1998] - 1 < 0.44, "admissions: one million in 1997; up 43 percent in 1999")
M5 = {(r["group"], r["t"], r["s"]): (float(r["omega"]), float(r["se"])) for r in mt("m5_moment_se.csv")}
yrs = ["2014", "2016", "2018", "2020", "2022"]
for g_, lab in [("pooled", "Pooled"), ("H", "Advantaged"), ("L", "Less advantaged")]:
    need(all((g_, a, b) in M5 for a in yrs for b in yrs if a <= b), f"fifteen moments of the 2014 to 2022 panel on file, {lab} (the table of moments by wave is in the online appendix)")
chk("variance decline", f"falls from {M5[('pooled', '2014', '2014')][0]:.2f} in 2014 to {M5[('pooled', '2022', '2022')][0]:.2f} in 2022")
need(0.14 < 1 - M5[("pooled", "2022", "2022")][0] / M5[("pooled", "2014", "2014")][0] < 0.19, "variance decline: about a sixth")
zs = {(t, s_): abs(M5[("H", t, s_)][0] - M5[("L", t, s_)][0]) / math.hypot(M5[("H", t, s_)][1], M5[("L", t, s_)][1]) for (g_, t, s_) in M5 if g_ == "H"}
need(len(zs) == 15 and max(zs.values()) < 2 and max(zs, key=zs.get) == ("2014", "2022"), "no moment gap above two standard errors; the largest at (2014, 2022)")
chk("largest standardized gap, text", f"(the largest standardized difference is {max(zs.values()):.2f})")
mom = sorted([r for r in u1("u7_moments.csv") if r["sample"] == "pooled" and r["t"] == r["s"]], key=lambda r: r["t"]); nw = [f"{int(r['children']):,}" for r in mom]
chk("wave sizes", f"(wave sizes {', '.join(nw[:-1])}, and {nw[-1]})"); need(sum(int(r["children"]) for r in mom) == int(F14[f"{THREE}: child-waves"]), "wave sizes sum to the child-waves of the children with three waves")
J16 = mi("m16_joint_moment_test.csv")[0]
chk("joint test of the fifteen gaps", f"does not reject equality ($p = {float(J16['p']):.2f}$)"); chk("joint test, introduction", f"(joint test $p = {float(J16['p']):.2f}$)")
fitv = lambda g: g["eta2"] + g["eps2"] / (1 - g["rho"] ** 2)
chk("fitted single-wave variances", f"the fitted single-wave variances are close, {fitv(H7):.3f} against {fitv(L7):.3f}")
chk("fitted variances, introduction", f"The fitted variances are {fitv(H7):.2f} and {fitv(L7):.2f}")
chk("fitted split", f"a permanent share of {H7['sperm']:.2f} against {L7['sperm']:.2f}, and transitory persistence of {H7['rho']:.2f} against {L7['rho']:.2f}")
chk("fitted split, conclusion", "The one difference the fitted model suggests runs the wrong way. Shocks fade more slowly for advantaged children, the opposite of protection."); need(H7["rho"] > L7["rho"] and H7["sperm"] < L7["sperm"], "advantaged children: more persistent transitory shocks and a smaller permanent share in the fitted split")
M13 = {r["param"]: r for r in mt("m13_mde.csv")}; g13 = lambda k, c: float(M13[k][c])
chk("gaps in standard errors", f"The two gaps are {abs(g13('permanent share', 'gap')) / g13('permanent share', 'se_gap'):.1f} and {abs(g13('transitory persistence', 'gap')) / g13('transitory persistence', 'se_gap'):.1f} bootstrap standard errors from zero")
chk("minimum detectable differences", f"group differences of {100 * g13('permanent share', 'mde80'):.0f} percentage points in the permanent share and of {g13('transitory persistence', 'mde80'):.2f} in transitory persistence")
need(abs(g13("transitory persistence", "gap")) < g13("transitory persistence", "mde80") and abs(g13("permanent share", "gap")) < 0.6 * g13("permanent share", "mde80"), "the persistence gap just below and the share gap well below the detectable difference")
AB = {r["panel"]: r for r in mt("m10_abgmm_table3.csv")}; ab5, abs_, ab7, abar = AB["diff GMM, 2014-2022"], AB["system GMM, 2014-2022"], AB["diff GMM, 2010-2022 reconstructed"], AB["AR interval rho1 (diff, 5-wave)"]
D17 = {r["panel"]: r for r in mi("m17_ab_diagnostics.csv")}
chk("Arellano-Bond, five waves", f"$\\hat\\rho_0 = {float(ab5['rho0']):.3f}$ (SE {float(ab5['rho0_se']):.3f}; Table~\\ref{{tab:ab}})")
chk("second-order serial correlation", f"($m_2$ $p = {float(D17['5-wave 2014-2022']['m2_p']):.2f}$)")
chk("Arellano-Bond, seven waves", f"The reconstructed seven-wave panel gives {float(ab7['rho0']):.3f}")
chk("system GMM", f"moves the point estimate to {float(abs_['rho0']):.3f}, and the Hansen test rejects the enlarged moment set ($p < 0.001$)"); need(float(abs_["pJ"]) < 0.001 and float(ab5["pJ"]) > 0.1, "Hansen: rejects the system moments, not the difference moments")
ab_lo, ab_hi = float(ab5["rho0"]) - 1.96 * float(ab5["rho0_se"]), float(ab5["rho0"]) + 1.96 * float(ab5["rho0_se"]); chk("minimum distance inside the Arellano-Bond interval", f"for less advantaged children, {L7['rho']:.2f}, lies inside the 95 percent interval of the Arellano--Bond estimate, $[{sg(ab_lo, 2)[1:-1]}, {ab_hi:.2f}]$"); need(ab_lo < L7["rho"] < ab_hi, "the minimum-distance estimate lies inside the Arellano-Bond interval")
need(0.2 <= round(math.sqrt(float(ab5["rho0"])), 1) and round(math.sqrt(P7["rho"]), 1) <= 0.4, "annual coefficients roughly 0.2 to 0.4")
chk("observed-earnings persistence", f"with a permanent share near {P7['sperm']:.2f}, the biennial autocorrelation of observed residual earnings implied by the minimum-distance fit is about ${P7['rho']:.2f} + {1 - round(P7['rho'], 2):.2f} \\times {P7['sperm']:.2f} \\approx {P7['rho'] + (1 - P7['rho']) * P7['sperm']:.2f}$")
chk("pass-through of a shock", f"raises them by about {10 * float(ab5['rho0']):.1f} percent in the next")
r1, s1 = float(ab5["rho1"]), float(ab5["rho1_se"])
chk("interaction", f"at {sg(r1)} with a standard error of {s1:.3f} and a Wald interval of $[{r1 - 1.96 * s1:.2f}, {r1 + 1.96 * s1:.2f}]$")
chk("robust set of the interaction", f"gives a set of $[{float(abar['rho1']):.2f}, {float(abar['rho1_se']):.2f}]$")
chk("detectable interaction", f"detects with 80 percent power is {g13('AB persistence interaction', 'mde80'):.2f})")
chk("Arellano-Bond table note", f"$N = {n3 // 1000}{{,}}{n3 % 1000:03d}$ children in the five-wave panel ({int(float(ab5['n_eq'])):,} usable differenced equations from {int(float(ab5['n_children'])):,} children); the reconstructed seven-wave panel applies the same screens to seven waves ({int(float(ab7['n_eq'])):,} equations from {int(float(ab7['n_children'])):,} children)")
abb = mi("ws", "ws_05b_abb_stochasticEM_surface.csv"); avg = {g: sum(float(r["rho_local"]) for r in abb if r["group"] == g) / sum(1 for r in abb if r["group"] == g) for g in ("pooled", "H", "L")}
need(all(sum(1 for r in abb if r["group"] == g) == 25 for g in avg), "nonlinear persistence: 25 lattice points per group")
chk("nonlinear persistence", f"is {avg['pooled']:.2f} in the pooled sample, {avg['H']:.2f} for advantaged children, and {avg['L']:.2f} for less advantaged children")
# ten runs of the stochastic estimator (u15_nonlinear_persistence_runs.py): range of the pooled figure and the ordering of the two groups
R15 = {r["sample"]: r for r in u1("u15_nonlinear_persistence_summary.csv")}
chk("nonlinear persistence, pooled range over ten runs", f"across ten runs of the stochastic estimator the pooled figure lies between {float(R15['pooled']['min']):.2f} and {float(R15['pooled']['max']):.2f}")
need(all(int(float(r["runs"])) == 10 for r in R15.values()) and int(float(R15["pooled"]["runs_advantaged_above_less_advantaged"])) == 10, "nonlinear persistence: advantaged above less advantaged in each of ten runs")
need(float(R15["advantaged children"]["min"]) > float(R15["less advantaged children"]["max"]), "nonlinear persistence: ranges of the two groups do not overlap")
need(float(R15["pooled"]["min"]) <= avg["pooled"] <= float(R15["pooled"]["max"]), "nonlinear persistence: the reported pooled figure lies inside the range over ten runs")
chk("nonlinear persistence, ordering in every run", "the figure for advantaged children is the higher of the two in every run")
G10 = u1("u10_growth_gap.csv"); g5 = [r for r in G10 if r["panel"].startswith("2014") and r["covariates"] != "no covariates"][0]; g7 = [r for r in G10 if not r["panel"].startswith("2014")][0]
chk("growth regression", f"gives $\\hat\\gamma = {float(g5['gamma']):.3f}$ per wave (cluster SE {float(g5['se']):.3f}) on the five-wave panel and {float(g7['gamma']):.3f} (SE {float(g7['se']):.3f}) on the reconstructed panel without the covariates. The point estimates imply a divergence of {5 * float(g5['gamma']):.2f} and {5 * float(g7['gamma']):.2f} log points over a decade, and neither differs from zero at the 5 percent level")
need(float(g5["p"]) > 0.05 and float(g7["p"]) > 0.05, "growth gaps: neither different from zero at the 5 percent level")
chk("growth, introduction", f"The earnings gap between them grows by {float(g5['gamma']):.2f} log points per two-year wave (SE {float(g5['se']):.3f}), which is not distinguishable from zero")
m3 = mi("method3_mini_results.csv"); vb = [float(r["estimate"]) for r in m3 if r["object"] == "Var(beta_i)" and r["variant"] == "O(1/T)-corrected"][0]; raw = [float(r["estimate"]) for r in m3 if r["object"] == "Var(beta_i)" and r["variant"] == "raw"][0]
chk("absent: the variance of individual growth slopes", "variance of individual growth slopes", count=0)
se3 = mi("ws", "ws_m3_strict_exogeneity.csv")[0]
chk("strict exogeneity", f"lead enters with coefficient {float(se3['lead_coef']):.3f} and a $t$-statistic of {float(se3['lead_t']):.2f}"); need(f"{float(se3['gamma_no_lead']):.3f}" == f"{float(se3['gamma_with_lead']):.3f}", "gamma unchanged at the third decimal with the lead")
age1 = F14[f"{THREE}: mean age at the first wave"]
chk("age at entry", f"Children enter the panel at {age1:.0f} on average. The least squares return to college is {d('26-29'):.2f} at ages 26 to 29 and {d('30-34'):.2f} at ages 30 to 34 (Section~\\ref{{sec:sample}}), so for a child first observed at {age1:.0f}")
chk("the level, not the variance or the growth", "not with the variance of earnings around that level or with its growth, so whatever advantage these families confer must already be in the level")
chk("introduction, children with three waves", f"Among the children observed in at least three waves, {a3} are advantaged and {n3 - a3:,} less advantaged")
chk("conclusion: the scope sentence", "and the comparison of earnings risk is conditional on having earnings.")

print("=== Section 4: measurement")
ETA = {(r["formula"], int(r["T"])): r for r in rd(os.path.join(FR, "output", "A_eta_bootstrap.csv"))}; TS = {r["formula"]: r for r in rd(os.path.join(FR, "output", "A_eta_Tstar.csv"))}
e = lambda f_, T, c="eta": float(ETA[(f_, T)][c])
chk("schedule, approximation", "independence approximation & " + " & ".join(f"{e('approx', T):.2f}" for T in range(1, 8)) + r" \\")
chk("schedule, exact", "exact & " + " & ".join(f"{e('exact', T):.2f}" for T in range(1, 8)) + r" \\")
chk("schedule, inflation", "$1/\\lambda_T$ & " + " & ".join(f"{e('approx', T, 'inflation'):.1f}" for T in range(1, 8)) + r" \\")
chk("factor at one wave", f"the reliability ratio $\\lambda_T$ is {e('approx', 1):.2f} at $T = 1$")
chk("footnote on the exact calculation", f"With biennial persistence $\\rho_v = {P7['rho']:.2f}$ the exact variance of the $T$-wave average carries cross terms. The exact calculation, derived in Part II of the Online Appendix, lowers $\\lambda_5$ from {e('approx', 5):.2f} to {e('exact', 5):.2f}")
need(int(float(TS["approx"]["T_star"])) == 6 and int(float(TS["exact"]["T_star"])) == 8, "two thirds recovered at six waves (approximation) and eight (exact)")
U2 = {r["process"]: {k: (float(v) if k != "process" else v) for k, v in r.items()} for r in u1("u2_recovery_by_process.csv")}; uc, up = U2["children"], U2["linked parents"]
chk("recovery table, children", f"Children's process (permanent share {uc['share']:.2f}) & {uc['recovery_lo']:.2f}--{uc['recovery_hi']:.2f} & {uc['implied_lo']:.2f}--{uc['implied_hi']:.2f} &")
chk("recovery table, parents", f"Linked parents' process (permanent share {up['share']:.2f}) & {up['recovery_lo']:.2f}--{up['recovery_hi']:.2f} & {up['implied_lo']:.2f}--{up['implied_hi']:.2f} &")
chk("implied range, children", f"implies a true coefficient between {uc['implied_lo']:.2f} and {uc['implied_hi']:.2f}"); need(uc["implied_lo"] < 0.390 and uc["implied_hi"] > 0.442, "the children's range contains the authors' preferred estimates")
chk("implied range, parents", f"$[{up['implied_lo']:.2f}, {up['implied_hi']:.2f}]$")
sh = [(up["implied_lo"] - 0.166) / (0.390 - 0.166), (up["implied_hi"] - 0.176) / (0.442 - 0.176)]; need(0.24 < sh[0] < 0.28 and 0.43 < sh[1] < 0.49, "share of the distance closed: a quarter to just under one half")
chk("share of the distance closed", "between a quarter and just under one half", count=2); chk("the published study: waves", "use its 2010 to 2016 waves"); chk("the published study: averaging", "with income averaged over two to\nfour biennial waves".replace("\n", " ")); need(int(uc["waves_lo"]) == 2 and int(uc["waves_hi"]) == 4, "recovery fractions computed for two to four waves")
PAR = mt("m14_parent_process.csv")[0]
chk("linked parents", f"The {int(float(PAR['n_parents']))} linked parents aged 60 or younger with three or more waves of positive labor income have a permanent share of {float(PAR['share']):.2f} (bootstrap interval $[{float(PAR['share_lo']):.2f}, {float(PAR['share_hi']):.2f}]$) against {P7['sperm']:.2f} for the children")
chk("parents, introduction", "under the earnings process estimated for the children, and a little over two fifths under the one estimated for their parents"); need(0.40 < float(PAR["share"]) < 0.45, "parents: a single wave recovers a little over two fifths"); chk("parents' persistence, table note", f"the transitory persistence of the linked parents is {float(PAR['rho']):.2f}")
need(0.40 < float(PAR["share"]) < 0.46 and 0.23 < uc["recovery_one_wave"] < 0.28, "two fifths under the parents' process and a quarter under the children's")
M8 = {int(r["T"]): r for r in mt("m8_rank_attenuation.csv")}
chk("rank-rank recovery", f"rank-rank slope at {float(M8[1]['rank_factor']):.2f}, roughly $\\sqrt{{\\lambda_1}}$, when the child's rank is measured without error, and at {float(M8[1]['rank_factor_2sided']):.2f} when the child's rank is measured from three waves")
need(abs(float(M8[1]["rank_factor"]) - math.sqrt(e("approx", 1))) < 0.02, "rank recovery close to the square root of the level recovery")

print("=== Section 5: the level premium")
S8 = {r["specification"]: r for r in u1("u8_premium_specifications.csv")}; s8 = lambda k, c="estimate": float(S8[k][c])
EB = {k: float(v) for k, v in u1("u8_entropy_balancing.csv")[0].items()}; L8 = {r["item"]: float(r["value"]) for r in u1("u8_level_sample.csv")}; W8 = {r["measure"]: r for r in u1("u8_income_measure_waves.csv")}
b_, l_, nin = "baseline: family income of the parental household", "parents' own labor income as the income control", "no income control (not in Figure 2)"
raw5, tau, gapM = L8["raw gap in mean log earnings"], EB["tau_hat"], EB["gap_hat"]; cH, cL = 100 * L8["college completion, advantaged children"], 100 * L8["college completion, less advantaged children"]
need(abs(raw5 - EB["tau_naive"]) < 1e-9 and abs(raw5 - F14[f"{LEVEL}: raw gap in mean log earnings"]) < 1e-9 and int(L8["children"]) == nl, "raw gap and sample size agree across the output files")
chk("opening of Section 5", f"Of the {n_all:,} children in the sample, {nl:,} ({al} advantaged) have one"); chk("raw gap at the opening of Section 5", f"Before any controls, advantaged children earn {raw5:.3f} log points ({pct(raw5):.0f} percent) more than less advantaged children, and {cH:.0f} percent of them completed college, compared with {cL:.0f} percent")
chk("raw gap in percent, abstract and introduction", f"earn {pct(raw5):.0f} percent more than other children", count=2)
chk("raw gap, introduction", f"children is {raw5:.3f} log points ({pct(raw5):.0f} percent). Holding")
chk("premium range, introduction", f"what remains, {tau:.2f} to {s8(l_):.2f} log points or an earnings difference of {pct(tau):.0f} to {pct(s8(l_)):.0f} percent, is a premium")
need(0.18 < (raw5 - s8(l_)) / raw5 < 0.23 and 0.60 < (raw5 - tau) / raw5 < 0.66, "controls remove between a fifth and just over three fifths of the raw gap")
chk("premium in percent, abstract", f"put the premium of advantaged children at {pct(tau):.0f} to {pct(s8(l_)):.0f} percent"); chk("premium in percent, Table 1", f"{pct(tau):.0f} to {pct(s8(l_)):.0f} percent with parental income held fixed; party membership")
chk("premium in percent, introduction", f"or an earnings difference of {pct(tau):.0f} to {pct(s8(l_)):.0f} percent")
chk("completion ratio, introduction", "Advantaged children complete college at more than twice the rate of less advantaged children"); need(cL > 0 and 2.0 < cH / cL < 3.0, "completion of advantaged children more than twice that of less advantaged children (children with a level of earnings)"); chk("completion rates at the same parental income, introduction", f"is {100 * gapM:.1f} percentage points. Valued")
pb = lambda p: "$p < 0.001$" if p < 0.001 else f"$p = {p:.3f}$"
chk("no income control", f"the occupation coefficient is {s8(nin):.3f} (SE {s8(nin, 'se'):.3f})")
chk("baseline premium", f"lowers the coefficient by nearly half, to {s8(b_):.3f} ({pb(s8(b_, 'p'))})"); need(0.42 < 1 - s8(b_) / s8(nin) < 0.50, "family income lowers the coefficient by nearly half")
chk("parents' labor income", f"falls from {s8(b_, 'income_coef'):.3f} to {s8(l_, 'income_coef'):.3f} and the occupation coefficient rises to {s8(l_):.3f} ({pb(s8(l_, 'p'))})")
chk("waves behind the income measures", f"compared with {float(W8['family income of the parental household']['mean_waves_per_parent']):.1f} for family income, so by the schedule of"); chk("waves behind the parents' labor income", f"{float(W8[l_.split(' as ')[0]]['mean_waves_per_parent']):.1f} waves per parent on average compared with")
chk("regression bracket", f"therefore lies between {s8(b_):.3f} and {s8(l_):.3f} log points ({pct(s8(b_)):.0f} and {pct(s8(l_)):.0f} percent)")
chk("family clustering", f"by {abs(L8['baseline standard error clustered on family'] - L8['baseline standard error clustered on child']):.4f} ({int(L8['families with two or more children in the sample'])} of the {int(L8['families']):,} families")
chk("entropy balancing premium", f"earnings is {tau:.3f}, or {pct(tau):.0f} percent, with a 200-replication bootstrap percentile interval of $[{EB['tau_ci_lo']:.2f}, {EB['tau_ci_hi']:.2f}]$")
chk("entropy balancing completion gap", f"gap in college completion between the groups is {gapM:.3f}, or {100 * gapM:.1f} percentage points ({100 * EB['completion_advantaged']:.2f} percent of advantaged children against {100 * EB['completion_less_advantaged_reweighted']:.2f} percent of reweighted less advantaged children; interval $[{EB['gap_ci_lo']:.2f}, {EB['gap_ci_hi']:.2f}]$)")
chk("entropy balancing against the regression", f"same premium, {s8(b_):.3f} and {tau:.3f}"); chk("all children with a level", f"Both figures are for all {nl:,} children")
p0, p1, pc_, pa = "occupation indicator before party membership is added (not in Figure 2)", "parental party membership added (cross-section)", "party membership coefficient with occupation and income (not in Figure 2)", "parental party membership in place of the occupation indicator (not in Figure 2)"
s0, s1_, sc = "parental state-sector employment alone (not in Figure 2)", "parental state-sector employment added (cross-section)", "state-sector employment coefficient with occupation and income (not in Figure 2)"
chk("party shares", f"{100 * float(S8[pa]['party_share_advantaged']):.0f} percent of advantaged children had a parent who was a party member, compared with {100 * float(S8[pa]['party_share_ordinary']):.0f} percent of less advantaged children.")
chk("state sector, share of children", f"available for the {100 * float(S8[s1_]['share_of_children_in_subsample']):.0f} percent of children")
so_ = "occupation indicator alone, children with a parent working in 2012 (not in Figure 2)"
chk("markers on their own", f"party membership carries a coefficient of {s8(pa):.3f} and state-sector employment one of {s8(s0):.3f} (columns 2 and 5), a little over half of the occupation coefficient on the same children ({s8(p0):.3f} and {s8(so_):.3f}), and neither is distinguishable from zero")
chk("markers on their own, introduction", "On its own, each has a coefficient a little over half that of parental occupation and is not distinguishable from zero")
need(0.5 < s8(pa) / s8(p0) < 0.6 and 0.5 < s8(s0) / s8(so_) < 0.6 and s8(pa, "p") > 0.1 and s8(s0, "p") > 0.1 and s8(pc_, "p") > 0.1 and s8(sc, "p") > 0.1 and int(s8(so_, "children")) == int(s8(s0, "children")) == int(s8(s1_, "children")), "markers: ratios, significance and common sample as worded")
need(abs(s8(p1) - s8(p0)) < 0.01 and abs(s8(s1_) - s8(so_)) < 0.01, "the occupation coefficient barely moves when a marker is added")
chk("conditional coefficients", f"the party coefficient falls to {s8(pc_):.3f} and the state-sector coefficient to {s8(sc):.3f}, while the occupation coefficient barely moves, to {s8(p1):.3f} and {s8(s1_):.3f}")
chk("interval upper ends", f"reach {float(S8[pc_]['hi']):.2f} for party membership and {float(S8[sc]['hi']):.2f} for state-sector employment ({pct(float(S8[pc_]['hi'])):.0f} and {pct(float(S8[sc]['hi'])):.0f} percent)")
AG = u1("u8_indicator_agreement.csv")[0]
chk("disagreement of the indicators", f"disagree for {100 * (1 - F14[f'{ALL}: indicators agree']):.0f} percent of the children in the sample for whom both exist ({100 * (1 - float(AG['share_agree'])):.0f} percent among those with a level of earnings)")
chk("lower bound", f"that assumption {s8(b_):.3f} is a lower bound"); need(int(float(S8["most frequent retrospective report of parental occupation"]["children_coded_differently_from_baseline"])) == 0, "most frequent report codes every child as the baseline does")
cur = "indicator from the occupations the parents hold during 2010 to 2022 (not in Figure 2)"; chk("indicator from current occupations", f"the estimate is {s8(cur):.3f} ($p = {s8(cur, 'p'):.2f}$); it is not among the fifteen")
win = ["parental income from the 2010 and 2012 waves, before the earnings window", "parental income from waves with the child aged 40 or less", "parental income from waves with the child aged 35 or less"]
# Section 5.5 does not print the single estimates of the fifteen specifications; Figure 3 shows them, and the figure file is drawn from the same csv
need(all(S8[w]["in_figure_2"] == "True" for w in win) and all(S8[k]["in_figure_2"] == "True" for k in ("indicator widened to CSCO major groups 1 to 3", "indicator narrowed to CSCO major group 1", "most frequent retrospective report of parental occupation")), "the income-window and indicator-definition specifications are among the fifteen of the figure")
chk("what the fifteen specifications vary", "the definition of the indicator (the most frequent retrospective report, which classifies every child as the baseline does; CSCO group 3 added; group 1 alone)")
need(abs(s8("income deflated by the provincial consumer price index") - s8(b_)) < 0.001, "CPI deflation moves the coefficient by less than 0.001")
rg, ur, ru = "three regions in place of province effects", "children recorded as urban in most of their waves", "children recorded as rural in most of their waves"
need(all(S8[k]["in_figure_2"] == "True" for k in (rg, ur, ru)), "region effects and the urban and rural subsamples are among the fifteen of the figure")
chk("the subsamples of the figure", "the subsample (children recorded as urban or as rural in most of their waves; the specification with state-sector employment uses the children with a parent working in 2012)")
fig = [r for r in S8.values() if r["in_figure_2"] == "True"]; est = [float(r["estimate"]) for r in fig]
need(len(fig) == 15 and min(fig, key=lambda r: float(r["estimate"]))["specification"] == ru and max(fig, key=lambda r: float(r["estimate"]))["specification"] == l_, "fifteen specifications; smallest rural, largest parents' labor income")
need(sum((float(r["lo"]) > 0) or (float(r["hi"]) < 0) for r in fig) == 11 and sum(0.06 <= v < 0.09 for v in est) == 10, "eleven intervals exclude zero; ten estimates between 0.06 and 0.09")
chk("range of the fifteen", f"from {sg(min(est))} (children recorded as rural in most waves) to {max(est):.3f} (the parents' own labor income as the income control); ten of the fifteen lie between 0.06 and 0.09, and eleven intervals exclude zero")
pby = (F14["parents' birth year, 5th percentile"], F14["parents' birth year, 95th percentile"]); need(1935 <= pby[0] <= 1939 and 1970 <= pby[1] <= 1973, "parents born mostly between the late 1930s and the early 1970s"); chk("parents' birth years", "born mostly between the late 1930s and the early 1970s")

print("=== Section 6: the expansion design (two-row regression)")
U9A = {r["item"]: r for r in u1("u9_A_sample.csv")}; U9B = {r["intensity"]: r for r in u1("u9_B_main.csv")}; U9C = {r["check"]: r for r in u1("u9_C_checks.csv")}; U9D = {(r["outcome"], r["bin"]): r for r in u1("u9_D_cohort_bins.csv")}
W9B = {r["intensity"]: r for r in u1("u9_B_main_weighted.csv")}; W9C = {r["check"]: r for r in u1("u9_C_checks_weighted.csv")}
MAIN = "predicted provincial growth (main measure)"; m = {k: (float(v) if v not in ("", MAIN) else v) for k, v in U9B[MAIN].items()}; mw = {k: (float(v) if v not in ("", MAIN) else v) for k, v in W9B[MAIN].items()}
OTH = [U9B[k] for k in ("predicted flow per 2005 worker", "log implied 2010 stock", "census senior-high stock in 2000")]
a9 = lambda k: float(U9A[k]["value"]); c9 = lambda k, c="coef": float(U9C[k][c]); w9 = lambda k, c="coef": float(W9C[k][c])
C14 = {r["row"]: r for r in u1("u14_C_first_stages_all_children.csv")}; fa = C14[f"all children: {MAIN}"]; fL, fH = C14["less advantaged children, instrument entering once for each group"], C14["advantaged children, instrument entering once for each group"]
C13 = {r["item"]: r for r in u1("u13_C_design_facts.csv")}; cv = lambda k: float(C13[k]["value"]); E13 = {r["item"]: r for r in u1("u13_E_added_degrees.csv")}
# Section 2.1: the variance of log income by wave (u17), and Section 6.1: the census measure in 1990 and in 2000
_v = {int(r["wave"]): float(r["variance_of_log_income"]) for r in u1("u17_income_variance_by_wave.csv") if r["respondents"].startswith("all respondents")}
chk("variance of log income at the change of concept", f"it is {_v[2012]:.2f} in the 2012 wave and between {min(_v[y] for y in (2014, 2016, 2018, 2020, 2022)):.2f} and {max(_v[y] for y in (2014, 2016, 2018, 2020, 2022)):.2f} in each of the five waves from 2014")
_hs = rd(os.path.join(PKG["college_expansion"], "data", "external", "census_hs_stock.csv")); _a = [float(r["hs1990"]) for r in _hs]; _b = [float(r["hs2000"]) for r in _hs]
_ma, _mb = sum(_a) / len(_a), sum(_b) / len(_b); _r = sum((x - _ma) * (y - _mb) for x, y in zip(_a, _b)) / math.sqrt(sum((x - _ma) ** 2 for x in _a) * sum((y - _mb) ** 2 for y in _b))
chk("census measure, 1990 against 2000", f"(correlation {_r:.2f} between the 1990 and 2000")
COR = {r[""]: r for r in u1("u3_C_intensity_correlations.csv")}; chk("census measure and the main measure", f"correlates {float(COR['hs2000']['pro_predict_growth']):.2f} with the main measure")
EXP = {int(r["birth_year"]): float(r["exposure_ratio"]) for r in u1("u3_B_exposure_series.csv")}; pre_ = [EXP[y_] for y_ in (1977, 1978, 1979, 1980)]
chk("exposure ratio, cohorts 1977 to 1980", f"between {min(pre_):.2f} and {max(pre_):.2f} for those born 1977 to 1980"); chk("exposure ratio, cohort 1981", f"rises to {EXP[1981]:.2f} for the 1981 cohort")
need(abs(EXP[1990] - 4) < 0.05 and min(EXP) == 1964, "exposure ratio reaches 4 around 1990; the series starts with the cohort born in 1964"); need(0.85 < min(EXP[y] for y in range(1964, 1977)) < 0.95 and 1.55 < max(EXP[y] for y in range(1964, 1977)) < 1.65, "exposure ratio between 0.9 and 1.6 for the cohorts born 1964 to 1976"); chk("exposure ratio, earliest cohorts", "lies between 0.9 and 1.6 for the cohorts born 1964 to 1976"); chk("exposure ratio around 1990", "reaches 4 for the cohorts born around 1990")
need(int(a9("children")) == n_all and int(a9("rows at 30 and over")) == nl and int(a9("advantaged rows at 30 and over")) == al and int(a9("children with a row")) == n_all, "rows of the two-row regression agree with the sample counts")
chk("conventional regression: one observation per child with a level of earnings", f"one observation for each of the {nl:,} children who have a level of earnings"); chk("children left out by the conventional regression", f"out the {n_all - nl:,} children in the sample who are not yet 30"); chk("advantaged children added by the rows below 30", f"the added rows bring {a_all - al} more advantaged children into the estimation of those effects"); chk("restriction of the two-row specification stated", "the province effects, the birth-year effects, and the coefficients on $X_i$ are common to the two rows"); need(int(F14["children with a row below 30 only"]) == n_all - nl, "children with a row below 30 only")
chk("identity: same outcome and children", f"All three refer to the level of earnings and to the same {nl:,} children"); chk("return settles from age 30", "it is still rising through the late twenties")
need(d("22-25") < 0.1 and d("26-29") < d("30-34") - 0.05 and abs(d("30-34") - d("35-55")) < 0.08, "least squares return by age: near zero at 22 to 25, lower at 26 to 29 than from 30, level from 30")
chk("rows", f"The {n_all:,} children contribute {nl:,} rows at 30 and over and {int(a9('rows below 30')):,} rows below 30; {int(F14['children with both rows'])} children contribute both")
chk("cohorts below 30", f"was born in {int(F14['first birth year among the rows below 30'])} or later and reached college age well after the expansion had begun")
need(int(F14['children born before 1964']) == 38, "38 children born before the admissions series starts")
sv = P7["eps2"] / (1 - P7["rho"] ** 2); wt = lambda n: 1 / (P7["eta2"] + sv / n)
chk("weights", f"A row that averages one wave has weight {wt(1):.1f} and a row that averages five waves {wt(5):.1f}.")
chk("pooled parameters behind the weights", f"evaluated at the pooled estimates of Table~\\ref{{tab:md}} ($\\hat\\sigma^2_\\eta = {P7['eta2']:.3f}$, $\\hat\\sigma^2_\\varepsilon = {P7['eps2']:.3f}$, $\\hat\\rho_v = {P7['rho']:.2f}$)")
chk("weights: the variance inverted is the denominator of the attenuation formula", "This variance is the denominator of equation~\\eqref{eq:eta} with $n_{ir}$ in place of $T$")
chk("Section 3.3 points forward to the weights", "The pooled estimates are used twice later in the paper")
chk("Section 4 points forward to the weights", "its inverse is the precision weight of the weighted expansion regression (equation~\\eqref{eq:weights})")
chk("province at age twelve, agreement", f"of the {int(F14['all children: province at age twelve recorded in the 2010 wave']):,} children for whom the 2010 wave records the province of residence at age twelve, {100 * F14['share with a different province at age twelve, advantaged']:.0f} percent of advantaged children and {100 * F14['share with a different province at age twelve, less advantaged']:.0f} percent of less advantaged children are assigned a different province")
need(int(c9("instrument by the province at age twelve", "with_report")) == int(F14["all children: province at age twelve recorded in the 2010 wave"]) and int(c9("instrument by the province at age twelve", "changed")) == int(F14["all children: province at age twelve differs from the province assigned"]), "province at age twelve: the two scripts agree")
need(int(cv("provinces")) == 29 and int(F14[f"{ALL}: provinces"]) == 29, "twenty-nine provinces"); chk("clusters", "inference rests on 29 clusters")
chk("instrument variance between cohorts", f"{100 * (1 - cv(chr(115) + 'hare of the instrument' + chr(39) + 's variance, net of province effects, within birth cohorts')):.0f} percent of the instrument's variance")
urb = C13["reduced form of urban residence on the instrument"]; need(float(urb['p_wild']) > 0.5, "urban residence does not respond to the instrument"); chk("supply of graduates in the exclusion paragraph", "to the extent that graduates work where they studied, their lower relative wages push the earnings reduced form toward zero"); chk("urban residence, threats table", f"Reduced form of the child's adult urban residence: {float(urb['value']):.3f} (SE {float(urb['se']):.3f})")
chk("placebo, threats table", f"Placebo from pre-reform registration growth: first-stage $F$ of {m['placebo_F']:.1f}"); chk("placebo against the main measure, Section 6.2", f"($F = {m['placebo_F']:.1f}$ against {m['F']:.1f})")
chk("placebo against the main measure, introduction", f"does not predict college completion (first-stage $F = {m['placebo_F']:.1f}$, compared with {m['F']:.1f} for the instrument)"); need(m["placebo_earn_p_wild"] > 0.3, "placebo earnings reduced form is null")
H0, L0, H1, L1 = [100 * cv(q) for q in ("college completion, advantaged children, born 1980 or earlier", "college completion, less advantaged children, born 1980 or earlier", "college completion, advantaged children, born 1981 or later", "college completion, less advantaged children, born 1981 or later")]
chk("completion before the reform, introduction", f"before the reform {H0:.1f} percent of advantaged children and {L0:.1f} percent of less advantaged children completed college"); chk("completion after the reform, introduction", f"{H1:.0f} percent of advantaged children and {L1:.0f} percent of less advantaged children completed college. Part of the rise from the pre-reform rates is a national trend")
chk("completion before and after the reform, Table 1", f"Completion rose in both groups (advantaged {H0:.0f} to {H1:.0f} percent, less advantaged {L0:.0f} to {L1:.0f})")
chk("completion before the reform, introduction", f"Yet before the reform {H0:.1f} percent of advantaged children and {L0:.1f} percent of less advantaged children completed college.")
chk("completion rates after the reform, introduction", f"In the cohorts born in 1981 or later, {H1:.0f} percent of advantaged children and {L1:.0f} percent of less advantaged children completed college")
chk("completion rates, Section 6.3", f"Advantaged children complete college at a rate of {H0:.1f} percent in the pre-reform cohorts and {H1:.1f} percent in the post-reform cohorts; less advantaged children start at {L0:.1f} percent and reach {L1:.1f}.")
chk("rise in completion", f"by about {cv('rise in completion, advantaged children, percentage points'):.0f} percentage points among advantaged children and by {cv('rise in completion, less advantaged children, percentage points'):.0f} among less advantaged children")
chk("share of all children, introduction", f"Less advantaged children, {100 * cv('less advantaged share of children'):.0f} percent of all children,")
need(0.55 < cv("less advantaged share of graduates, born 1980 or earlier") < 0.62 and 0.72 < cv("less advantaged share of the added graduates") < 0.78, "less advantaged children: three fifths of graduates before, three quarters of the added graduates")
chk("shares of graduates, introduction", f"Less advantaged children, {100 * cv('less advantaged share of children'):.0f} percent of all children, were {100 * cv('less advantaged share of graduates, born 1980 or earlier'):.0f} percent of the graduates in the cohorts before the reform and {100 * cv('less advantaged share of the added graduates'):.0f} percent of the additional graduates in the cohorts after it")
chk("incidence, conclusion", f"Completion among less advantaged children rose from {L0:.0f} to {L1:.0f} percent", count=1)
chk("absent: three quarters of the added graduates without the population share", "three quarters of the graduates the expansion added were less advantaged children", count=0)
chk("ratio of completion rates, introduction", f"fell from {cv('completion ratio, born 1980 or earlier'):.1f} in the cohorts born before 1981, who reached college age before the reform, to {cv('completion ratio, born 1981 or later'):.1f} in those born in 1981 or later")
chk("ratio, introduction and Section 6.3", f"fell from {cv('completion ratio, born 1980 or earlier'):.1f} to {cv('completion ratio, born 1981 or later'):.1f}", count=1); chk("ratio, conclusion", f"measured as a ratio, it shrank, from {cv('completion ratio, born 1980 or earlier'):.1f} to {cv('completion ratio, born 1981 or later'):.1f}."); chk("gap in percentage points, Section 6.3", f"so the gap widened from {cv('completion gap in percentage points, born 1980 or earlier'):.0f} to {cv('completion gap in percentage points, born 1981 or later'):.0f} percentage points")
need(2.0 <= cv("completion ratio, born 1981 or later") < cv("completion ratio, born 1980 or earlier") <= 4.0, "completion ratio between two and four in the cohorts before and after the reform")
chk("gap in percentage points, conclusion", f"Measured in percentage points, the advantage of family background in college completion grew, from {cv('completion gap in percentage points, born 1980 or earlier'):.0f} to {cv('completion gap in percentage points, born 1981 or later'):.0f}, as earlier work on the expansion reports")
FL = {(r["group"], r["cohort_group"]): r for r in u1("fig_first_stage_left.csv")}; ratio = lambda c_: float(FL[("Advantaged children", c_)]["completion_rate"]) / float(FL[("Less advantaged children", c_)]["completion_rate"])
c_first = "1972 or earlier"; c_last = [q[1] for q in FL if q[1].startswith("1989")][0]
chk("ratio by cohort, Section 6.3", f"falls from {ratio(c_first):.1f} in the cohorts born in 1972 or earlier to {ratio(c_last):.1f} in those born 1989 to 1997"); need(int(F14[f"{ALL}: last birth year"]) == 1997, "last birth year 1997")
chk("first stage on all children", f"delivers a first-stage coefficient of {float(fa['fs']):.3f} (SE {float(fa['fs_se']):.3f}) with $F = {float(fa['F']):.1f}$")
Fo = [float(C14[f"all children: {k}"]["F"]) for k in ("predicted flow per 2005 worker", "log implied 2010 stock", "census senior-high stock in 2000")]
chk("first-stage F of the other measures", f"statistics between {min(Fo):.0f} and {max(Fo):.0f}")
dz = a9("difference in the instrument created by the reform")   # interquartile range of intensity times the rise in the exposure ratio from the cohorts born in 1980 or earlier to those born in 1989 or later
chk("interquartile effect on completion", f"the college completion of a cohort born in 1989 or later by {100 * dz * float(fa['fs']):.0f} percentage points")
chk("interquartile effect on completion, introduction", f"raises the completion of the cohorts born in 1989 or later, who faced admissions about four times the pre-reform trend, by {100 * dz * float(fL['fs']):.0f} percentage points among less advantaged children and by {100 * dz * float(fH['fs']):.0f} among advantaged children"); need(3.8 < a9("mean exposure ratio, born 1989 or later") < 4.2, "cohorts born 1989 or later: admissions about four times the pre-reform trend")
chk("first stage at 30 and over, Section 6.3", f"at 30 and over: {m['fs']:.3f} (SE {m['fs_se']:.3f}) with $F = {m['F']:.1f}$")
chk("children at 30 and over born after the reform", f"of whom {int(a9('rows at 30 and over, born 1981 or later')):,} were born in 1981 or later")
chk("advantaged children per province", f"Only {al} advantaged children have a level of earnings, fewer than 20 in {int(cv('provinces with fewer than twenty advantaged children'))} of the {int(cv('provinces'))} provinces")
dgap, dse = float(E13["difference (advantaged minus less advantaged)"]["value"]), float(E13["difference (advantaged minus less advantaged)"]["se"])
chk("first stages by group", f"is {float(fL['fs']):.3f} (SE {float(fL['fs_se']):.3f}, $F = {float(fL['F']):.1f}$) for less advantaged children and {float(fH['fs']):.3f} (SE {float(fH['fs_se']):.3f}, $F = {float(fH['F']):.1f}$) for advantaged children, and the difference between them, {sg(dgap)} (SE {dse:.3f}), is not distinguishable from zero")
need(abs(dgap) < 1.96 * dse and abs(dgap - (float(fH["fs"]) - float(fL["fs"]))) < 1e-6, "difference of the two first stages")
chk("interquartile effect by group, Section 6.3", f"({100 * dz * float(fL['fs']):.0f} and {100 * dz * float(fH['fs']):.0f} for the same interquartile difference)")
# the first stages by group are asserted for Section 6.3 above and Section 6.4 below
rel = (float(fL["fs"]) / (L0 / 100)) / (float(fH["fs"]) / (H0 / 100)); need(3.8 < rel < 4.8 and 4.8 < L1 / L0 < 5.6, "relative response four times as large; completion rose fivefold")
chk("relative response", f"Relative to pre-reform rates of {L0:.1f} and {H0:.1f} percent, the response of less advantaged children is four times as large.")
# 6.4
chk("first stage at 30 and over", f"has $F = {m['F']:.1f}$ (wild-cluster $p = {m['fs_p_wild']:.2f}$)")
chk("earnings reduced form", f"and the earnings reduced form is {m['earn']:.3f} (SE {m['earn_se']:.3f}, wild-cluster $p = {m['earn_p_wild']:.3f}$)")
chk("interquartile effect on earnings", f"an interquartile difference in intensity ({dz:.2f} in $Z$, which on the rows at 30 and over raises completion by {100 * m['fs'] * dz:.0f} percentage points) raises the level of earnings of a cohort born in 1989 or later by about {dz * m['earn']:.3f} log points ({pct(dz * m['earn']):.0f} percent)")
chk("return and least squares", f"The two-stage least squares estimate of the return to college is {m['earn_iv']:.2f} log points, against a least squares return of {m['ols']:.2f}.")
# Anderson-Rubin sets, read from the grids of p-values (grid -6 to 6); the limit of the test as b grows is the test of a zero first stage
def _segs(rows_, lvl):
    acc = sorted(float(r["b"]) for r in rows_ if float(r["p_wild"]) >= lvl); out = []
    for b_0 in acc:
        if out and abs(b_0 - out[-1][1] - 0.05) < 1e-6: out[-1][1] = b_0
        else: out.append([b_0, b_0])
    return out
_own = {r["group"]: float(r["fs_own_regressor_p_wild"]) for r in u1("u12_returns_by_family_background.csv")}
G9g = u1("u9_anderson_rubin_grid.csv"); G12g = u1("u12_anderson_rubin_grid.csv"); gridL = [r for r in G12g if r["group"] == "less advantaged"]; gridH = [r for r in G12g if r["group"] == "advantaged"]
sA95, sA90, sL95, sL90, sH95, sH90 = _segs(G9g, .05), _segs(G9g, .10), _segs(gridL, .05), _segs(gridL, .10), _segs(gridH, .05), _segs(gridH, .10)
need(len(sA95) == 1 and len(sA90) == 1 and len(sL95) == 1 and len(sL90) == 1 and len(sH95) == 2 and len(sH90) == 2, "Anderson-Rubin sets: one interval for all children and for less advantaged children, two rays for advantaged children")
need(-5.99 < sA95[0][0] and sA95[0][1] < 5.99 and sL95[0][1] < 5.99 and m["fs_p_wild"] < 0.05 and _own["less advantaged"] < 0.05, "95 percent sets of all children and of less advantaged children end inside the grid, and the first stage rejects zero at 5 percent")
need(sH95[0][0] <= -5.99 and sH95[1][1] >= 5.99 and sH90[0][0] <= -5.99 and sH90[1][1] >= 5.99 and _own["advantaged"] >= 0.10, "advantaged children: both sets are unions of two rays, and the test of a zero first stage does not reject at 10 percent")
chk("Anderson-Rubin set", f"The 95 percent wild-cluster Anderson--Rubin set is $[{sA95[0][0]:.2f}, {sA95[0][1]:.2f}]$: it excludes zero and the least squares return."); need(sA95[0][0] > m["ols"] and abs(sA95[0][0] - m["ar_lo"]) < 1e-9, "the 95 percent set of all children starts above the least squares return and agrees with u9_B_main.csv")
need(m["ar_hi"] >= 3.99 and m["ols"] < m["ar_lo"] < m["earn_iv"], "the set starts above the least squares return and reaches the end of the grid")
chk("other intensity measures", f"returns of {min(float(r['earn_iv']) for r in OTH):.2f} to {max(float(r['earn_iv']) for r in OTH):.2f} with wild-cluster $p$ between {min(float(r['earn_p_wild']) for r in OTH):.3f} and {max(float(r['earn_p_wild']) for r in OTH):.3f}")
chk("placebo intensity", f"the placebo intensity does not predict completion ($F = {m['placebo_F']:.1f}$)")
lw = "earnings, children with 9 or fewer years of schooling"; need(c9(lw, 'p_wild') > 0.5, "low-schooling children: earnings reduced form not distinguishable from zero (not printed in the paper)")
_lo9 = "professional occupation, children with 9 or fewer years of schooling"; chk("low-schooling placebo, occupation first", f"has a reduced form of {c9(_lo9):.3f} (SE {c9(_lo9, 'se'):.3f}), against {m['occ_maj']:.3f} for all children")
tr = [c9(f"trade control: {k}") for k in ("coastal x exposure", "export share x exposure", "both")]; chk("trade controls", f"leave the reduced form at {min(tr):.3f} to {max(tr):.3f}")
loo = "leave one province out: instrumented return, smallest and largest"; chk("leave one province out", f"gives returns between {c9(loo):.2f} and {c9(loo, 'se'):.2f}")
rt = "region-specific linear cohort trends (three regions)"; p12 = "instrument by the province at age twelve"; pt = "province-specific linear cohort trends"
a17, a19 = "exposure ratio of the year the cohort turned 17", "exposure ratio of the year the cohort turned 19"
need(c9(a17, "p_wild") < 0.05 and c9(a19, "p_wild") < 0.05 and abs(c9(a17, "iv") - m["earn_iv"]) < 0.1 and abs(c9(a19, "iv") - m["earn_iv"]) < 0.1, "college age seventeen or nineteen gives the same results (reduced forms significant, returns within 0.1 of the main return)")
chk("trend checks and province at age twelve", f"It is {c9(rt, 'iv'):.2f} with linear cohort trends for each of three regions (east, center, and west) and {c9(p12, 'iv'):.2f} with the instrument assigned by the province at age twelve (wild-cluster $p = {c9(rt, 'p_wild'):.3f}$ for both reduced forms). It is {c9(a17, 'iv'):.2f} and {c9(a19, 'iv'):.2f} when cohort exposure is taken at age seventeen or nineteen instead of eighteen.")
need(f"{c9(rt, 'p_wild'):.3f}" == f"{c9(p12, 'p_wild'):.3f}" and c9(pt, "p_wild") > 0.1, "region trends and province at twelve have the same printed p; province trends not distinguishable from zero")
# (province trends: asserted in the sentence above)
chk("pre-reform trends", f"(wild-cluster $p = {c9('pre-reform cohorts: earnings on intensity x birth year', 'p_wild'):.2f}$ and ${c9('pre-reform cohorts: completion on intensity x birth year', 'p_wild'):.2f}$)")
# (province at age twelve: asserted in the sentence above)
oo = "rows at 30 and over only (no row below 30)"
chk("rows at 30 and over only", f"Dropping the second row (equation~\\eqref{{eq:oldonly}}, {int(c9(oo, 'n')):,} children) gives a return of {c9(oo, 'iv'):.2f}, with a first-stage $F$ of {c9(oo, 'F'):.1f} and a reduced form of {c9(oo):.3f} (SE {c9(oo, 'se'):.3f}, wild-cluster $p = {c9(oo, 'p_wild'):.3f}$)"); chk("conventional regression against the main specification, Section 6.2", f"the conventional regression gives nearly the same value ({c9(oo, 'iv'):.2f} against {m['earn_iv']:.2f}")
chk("weighted rows", f"gives {mw['earn_iv']:.2f} ($F = {mw['F']:.1f}$; reduced form {mw['earn']:.3f}, SE {mw['earn_se']:.3f}, wild-cluster $p = {mw['earn_p_wild']:.3f}$), and the Anderson--Rubin set again starts at {mw['ar_lo']:.2f}")
chk("weighted return, introduction", f"When I weight each average by the precision it implies, the return is {mw['earn_iv']:.2f}")
chk("absent in the introduction: the regression without the rows below 30", "when the rows below 30 are dropped", count=0)
chk("absent: weighted return in the abstract", "For all children the return is", count=0)
chk("return per year of college", f"{m['earn_iv']:.2f} is about {m['earn_iv'] / 4:.2f} log points ({pct(m['earn_iv'] / 4):.0f} percent) per year of college, close to what other designs find")
chk("return per year, literature", "The instrumented return per year of college is close to"); chk("return per year, Section 6.4", f"the instrumented return of {m['earn_iv']:.2f} is about {m['earn_iv'] / 4:.2f} log points ({pct(m['earn_iv'] / 4):.0f} percent) per year of college")
bins = ("1972 or earlier", "1973-1976", "1981-1984", "1985-1988", "1989 or later"); eb = {b: float(U9D[("mean log earnings at 30 and over", b)]["coef"]) for b in bins}; cb = {b: float(U9D[("college completion (children with a row at 30 and over)", b)]["coef"]) for b in bins}
chk("cohort bins, earnings", f"is {sg(eb[bins[0]], 2)} and {eb[bins[1]]:.2f} in the two pre-reform bins and rises to {eb[bins[2]]:.2f}, {eb[bins[3]]:.2f}, and {eb[bins[4]]:.2f} for the cohorts born 1981 to 1984, 1985 to 1988, and 1989 or later")
chk("cohort bins, completion", f"The completion gradient is {sg(cb[bins[0]], 2)} and {cb[bins[1]]:.2f} in the two pre-reform bins, neither distinguishable from zero at the 5 percent level, and {cb[bins[2]]:.2f}, {cb[bins[3]]:.2f}, and {cb[bins[4]]:.2f} in the three post-reform bins, where it rises with the exposure of the cohort")
need(all(float(U9D[("college completion (children with a row at 30 and over)", b_)]["p"]) > 0.05 for b_ in bins[:2]), "pre-reform completion gradients not different from zero at the 5 percent level")
chk("return in the introduction", f"The instrumented return to college is {m['earn_iv']:.2f} log points, or {m['earn_iv'] / 4:.2f} per year of college ({pct(m['earn_iv'] / 4):.0f} percent per year), with an Anderson--Rubin confidence set that excludes returns below {m['ar_lo']:.2f}; the least squares return is {m['ols']:.2f}.", count=1)
chk("return in the abstract", f"college raised earnings by {m['earn_iv']:.2f} log points, or {m['earn_iv'] / 4:.2f} per year of college; by family background the return is")
chk("table note, sample", f"All {n_all:,} children in the sample ({a_all} advantaged), two-row specification")
chk("table note, rows", f"({nl:,} rows, of which {int(a9('rows at 30 and over, born 1981 or later')):,} belong to children born in 1981 or later)")
G12 = {r["group"]: {k: (float(v) if k not in ("sample", "group") else v) for k, v in r.items()} for r in u1("u12_returns_by_family_background.csv")}; gL, gH = G12["less advantaged"], G12["advantaged"]
WG = {r["group"]: r for r in u1("u12_returns_by_family_background_weighted.csv")}
chk("first stages and reduced forms by group", f"($F = {gL['F']:.1f}$ for less advantaged and {gH['F']:.1f} for advantaged children), and both earnings reduced forms are positive ({gL['rf']:.3f}, wild-cluster $p = {gL['rf_p_wild']:.3f}$, and {gH['rf']:.3f}, $p = {gH['rf_p_wild']:.3f}$)")
chk("returns by group", f"The instrumented return is {gL['beta']:.2f} for less advantaged children and {gH['beta']:.2f} for advantaged children; the least squares returns are {gL['ls']:.2f} and {gH['ls']:.2f}. Both 95 percent Anderson--Rubin sets exclude zero. The set for less advantaged children is $[{sL95[0][0]:.2f}, {sL95[0][1]:.2f}]$, and the 90 percent set for advantaged children excludes every return between {sg(sH90[0][1], 2)} and {sH90[1][0]:.2f}."); need(gL["rf_p_wild"] < 0.05 and gH["rf_p_wild"] < 0.05, "both 95 percent Anderson-Rubin sets exclude zero (the test at zero is the reduced-form test)")
need(gL["F"] > 10 and gH["F"] > 10 and gL["rf"] > 0 and gH["rf"] > 0, "both first stages by group have F above 10 and both reduced forms are positive")
need(abs(gH["fs"] - 0.189) < 0.002 and abs(gL["fs"] - 0.182) < 0.002, "the first stage by group is the regression of the child's completion on the group's instrument")
chk("equal returns", f"($z = {abs(gL['equal_returns_z']):.2f}$)"); need(abs(gL["equal_returns_z"]) < 1.645 and gL["beta"] > gL["ls"] and gH["beta"] > gH["ls"], "returns not distinguishable; neither below the least squares return")
chk("return in Table 1", f"return {m['earn_iv']:.2f} log points ({pct(m['earn_iv'] / 4):.0f} percent per year of college), {gH['beta']:.2f} for advantaged and {gL['beta']:.2f} for less advantaged children")
chk("abstract: completion in both groups", f"Completion rose in both groups, from {H0:.0f} to {H1:.0f} percent among advantaged children and from {L0:.0f} to {L1:.0f} percent among less advantaged children, and added seats raised it by a similar number of percentage points in each.")
chk("returns by group, abstract", f"by family background the return is {gL['beta']:.2f} for less advantaged children and {gH['beta']:.2f} for advantaged children."); chk("returns by group, introduction", f"By family background, it is {gL['beta']:.2f} for less advantaged children and {gH['beta']:.2f} for advantaged children ({pct(gL['beta'] / 4):.0f} and {pct(gH['beta'] / 4):.0f} percent per year of college), both above the least squares return of {m['ols']:.2f}"); chk("less advantaged return, conclusion", f"the return for those whom the added seats admitted is {gL['beta']:.2f} log points ({pct(gL['beta'] / 4):.0f} percent per year of college), above the least squares return of {gL['ls']:.2f}.")
chk("returns by group, conclusion", f"The return to college is {gH['beta']:.2f} for advantaged children and {gL['beta']:.2f} for less advantaged children. If the difference holds as more of the post-reform cohorts reach their thirties, part of the advantage of family background moves from who completes college to what college returns."); chk("absent: the negative opening of the sentence on the two returns", "Nor do the estimates show", count=0)
chk("returns, Section 7", f"{m['earn_iv']:.2f} for all children, {gL['beta']:.2f} for less advantaged children, and {gH['beta']:.2f} for advantaged children. All three point estimates lie above the least squares returns"); need(min(m["earn_iv"], gL["beta"], gH["beta"]) > max(gL["ls"], gH["ls"], m["ols"]), "all three instrumented returns lie above the least squares returns")
need(5.5 < nl / al < 6.5, "the returns for all children rest on six times as many children")
chk("margin between the set and least squares", f"For all children, the point estimate ({m['earn_iv']:.2f}) and the lower end of the 95 percent Anderson--Rubin set ({sA95[0][0]:.2f}) lie above the least squares return ({m['ols']:.2f})"); need(m["earn_iv"] > m["ols"] and m["ar_lo"] > m["ols"], "point estimate and lower end of the 95 percent set above the least squares return")
chk("occupation", f"Its reduced form on the instrument is {m['occ_maj']:.3f} (SE {m['occ_maj_se']:.3f}, wild-cluster $p = {m['occ_maj_p_wild']:.3f}$")
chk("occupation, table note", f"the instrumented effect of college on that share under the main measure is {m['occ_maj_iv']:.2f}"); chk("occupation, instrumented", f"by {m['occ_maj_iv']:.2f}, or {100 * m['occ_maj_iv']:.0f} percentage points")
op = [float(r["occ_maj_p_wild"]) for r in OTH]; chk("occupation, other measures", f"with wild-cluster $p$ between {min(op):.3f} and {max(op):.3f}"); need(all(float(r["occ_maj"]) > 0 for r in OTH), "occupation reduced forms of the other measures are positive")
lo9 = "professional occupation, children with 9 or fewer years of schooling"; need("show no response (" not in TEX, "the low-schooling occupation result is stated once in the text (Section 6.4) and in the table of threats")
# the table of assumptions, threats, diagnostics and results (Table 11): every number is the one asserted for the text above
_Fo = [float(r["F"]) for r in OTH]; _cen = [r for r in OTH if "census" in r["intensity"]][0]
chk("threats table: relevance", f"First-stage $F$ of {m['F']:.1f} for all children, {gL['F']:.1f} and {gH['F']:.1f} by family background, and {min(_Fo):.1f} to {max(_Fo):.1f} under the three other measures")
chk("threats table: exclusion", f"Trade-exposure controls: reduced form {min(tr):.3f} to {max(tr):.3f}, against {m['earn']:.3f}. Children with nine or fewer years of schooling: occupation {c9(lo9):.3f} (SE {c9(lo9, 'se'):.3f}).")
chk("threats table: exogeneity", f"Predetermined census measure: return {float(_cen['earn_iv']):.2f}. Cohorts born in 1980 or earlier: no trend in earnings or completion ($p = {c9('pre-reform cohorts: earnings on intensity x birth year', 'p_wild'):.2f}$ and ${c9('pre-reform cohorts: completion on intensity x birth year', 'p_wild'):.2f}$). Linear cohort trends by region: return {c9(rt, 'iv'):.2f}.")
_ri = {r["statistic"]: r for r in u1("u9_R_randomization_inference.csv")}
chk("randomization inference", f"I reassign the {int(_ri['first stage']['provinces'])} provincial values of intensity at random among the provinces {int(_ri['first stage']['reassignments']):,} times"); chk("randomization inference, counts", "No reassignment yields an earnings reduced-form $t$-statistic as large in absolute value as the actual one, and two yield a first-stage $t$-statistic as large ($p < 0.001$ for both).")
_g3 = {r["outcome"]: r for r in u1("u9_G_three_year_completion.csv")}; need(float(_g3["exactly 15 years of schooling (three-year degree)"]["coef"]) < 0, "the point estimate of three-year completion on the instrument is negative"); chk("three-year counterfactual", "If some compliers would otherwise have held a three-year degree, 0.68 compares a four-year degree with a mix of no college and a three-year degree.")
_own = {r["group"]: float(r["fs_own_regressor_p_wild"]) for r in u1("u12_returns_by_family_background.csv") if r.get("fs_own_regressor_p_wild") not in (None, "")}
need(_own["advantaged"] >= 0.10 and _own["less advantaged"] < 0.05, "own-completion first stage: advantaged children do not reject zero at 10 percent (sets unbounded), less advantaged children reject at 5 percent (sets bounded)"); chk("rule for unbounded sets", "a zero coefficient on the group's instrument in the first stage for the group's own college completion")
_rw = u1("u9_R_randomization_within_region.csv")[0]
need(float(_rw["permutation_p_reduced_form"]) < 0.001 and float(_rw["permutation_p_first_stage"]) < 0.001, "randomization inference within regions: p below 0.001 for the reduced form and the first stage")
chk("randomization inference within regions", "Reassigning intensity only among the provinces of the same region (east, center, and west) again gives $p < 0.001$ for both.")
chk("set from randomization inference within regions", f"Inverting the randomization test under a common return, with reassignment within regions, gives a narrower 95 percent set, $[{float(_rw['set95_lo']):.2f}, {float(_rw['set95_hi']):.2f}]$.")
need(float(_rw["grid_lo"]) < float(_rw["set95_lo"]) and float(_rw["set95_hi"]) < float(_rw["grid_hi"]), "the randomization set lies inside the grid")
_st = [r for r in u1("u8_premium_specifications.csv") if r.get("state_share_advantaged") not in (None, "")][0]
chk("state-sector shares by family background", f"among them, {100 * float(_st['state_share_advantaged']):.0f} percent of advantaged children have a parent in the state sector, compared with {100 * float(_st['state_share_ordinary']):.0f} percent of less advantaged children")
_urb = C13["reduced form of urban residence on the instrument"]; chk("urban residence in the text", f"does not move with the instrument (reduced form {float(_urb['value']):.3f}, SE {float(_urb['se']):.3f}, one observation per child)")
_cnt = {k: round(float(r["permutation_p"]) * (int(r["reassignments"]) + 1)) - 1 for k, r in _ri.items()}; need(_cnt == {"earnings reduced form": 0, "first stage": 2} and all(float(r["permutation_p"]) < 0.001 for r in _ri.values()), "randomization inference: none and two reassignments reach the actual |t|, p below 0.001")
if "the most demanding specification" in TEX or "by province." in TEX.split("Linear cohort trends")[1][:60]: fails += 1; print("  FAIL the province-trend result belongs to the Online Appendix, not to the body")
chk("intensity: 1982 shares", "by their shares of national college enrollment in the 1982 census"); chk("absent: the repository address in the title-page footnote", "replication files are at", count=0)
chk("figure caption, children", f"the {nl:,} children with a level of earnings;")

print("=== Section 7: the decomposition")
B13 = {r["row"]: r for r in u1("u13_B_identity.csv")}; B = lambda r_, c_="college_channel": float(B13[r_][c_]); A13 = {r["item"]: r for r in u1("u13_A_least_squares.csv")}; WB = {r["row"]: r for r in u1("u13_B_identity_weighted.csv")}
need(abs(B("premium (entropy balancing)") - tau) < 1e-9 and abs(B("completion gap (entropy balancing)") - gapM) < 1e-9, "premium and completion gap of the identity are those of Section 5")
chk("premium and gap", f"The completion gap is $\\Delta M = {gapM:.3f}$ ({100 * gapM:.1f} percentage points)")
chk("completion in the balanced comparison", f"{100 * EB['completion_advantaged']:.2f} percent of advantaged children hold a degree against {100 * EB['completion_less_advantaged_reweighted']:.2f} percent of less advantaged children ({cL:.1f} percent before reweighting)")
ls = lambda k, c="coef": float(A13[k][c]); la, lh, ll = "least squares return, all children", "least squares return, advantaged children", "least squares return, less advantaged children"
chk("least squares returns", f"It is {ls(la):.3f} (SE {ls(la, 'se'):.3f}) for all children, and {ls(lh):.3f} (SE {ls(lh, 'se'):.3f}) for advantaged children against {ls(ll):.3f} (SE {ls(ll, 'se'):.3f}) for less advantaged children")
need(abs(ls(lh) - ls(ll)) < 1.96 * math.hypot(ls(lh, "se"), ls(ll, "se")), "least squares returns of the two groups not distinguishable")
chk("family background coefficient with and without college", f"falls from {ls('coefficient on family background before college completion enters'):.3f} to {ls('coefficient on family background after college completion enters'):.3f}")
chk("the same, introduction", f"falls from {ls('coefficient on family background before college completion enters'):.2f} to {ls('coefficient on family background after college completion enters'):.2f}")
need(1 - ls("coefficient on family background after college completion enters") / ls("coefficient on family background before college completion enters") > 2 / 3, "college accounts statistically for more than two thirds of the coefficient")
rows = {"Return at which college explains half of the premium": "return at which college accounts for half of the premium", "Least squares, all children": la, "Least squares, advantaged children": lh,
        "Return at which college explains all of the premium": "return at which college accounts for all of the premium", "Instrumented, all children": "instrumented return, all children", "Instrumented, advantaged children": "instrumented return, advantaged children"}
for lab, key in rows.items():
    chk(f"decomposition table: {lab}", f"{lab} & {B(key, 'return_to_college'):.3f} & {B(key):.3f} & {sg(B(key, 'direct'))} & {B(key, 'college_share'):.2f} \\\\")
iv = "instrumented return, all children"; arl = "lower end of the 95 percent Anderson-Rubin set, all children"
ivh = "instrumented return, advantaged children"; arh = "lower end of the 90 percent Anderson-Rubin set, advantaged children"
chk("instrumented returns above the return at which college explains all", f"the two instrumented returns, {B(iv, 'return_to_college'):.2f} and {B(ivh, 'return_to_college'):.2f}, lie above {B(rows['Return at which college explains all of the premium'], 'return_to_college'):.2f}, and the 90 percent Anderson--Rubin set for advantaged children excludes every return between {sg(sH90[0][1], 2)} and {sH90[1][0]:.2f}"); need(abs(sH90[1][0] - B(arh, "return_to_college")) < 1e-9 and sH90[1][0] > B(rows["Return at which college explains all of the premium"], "return_to_college"), "the positive part of the 90 percent set of advantaged children starts above the return at which college explains all"); need(B(iv) > tau and B(ivh) > tau, "the college channel exceeds the premium at both instrumented returns")
need(B(arh, "college_share") >= 1 and B(ivh, "college_share") > 1, "college accounts for all of the premium at every return of the 90 percent set of advantaged children")
chk("channel at the least squares returns", f"At the least squares returns the completion gap accounts for {100 * B(la, 'college_share'):.0f} percent of the premium (the return for all children) and {100 * B(lh, 'college_share'):.0f} percent (the return for advantaged children), and leaves a direct effect of {B(la, 'direct'):.3f} and {B(lh, 'direct'):.3f}")
chk("half and all", f"College explains half of the premium at a return of {B(rows['Return at which college explains half of the premium'], 'return_to_college'):.2f} and all of it at {B(rows['Return at which college explains all of the premium'], 'return_to_college'):.2f}")
chk("lower end of the set", f"At {B(arl, 'return_to_college'):.2f}, the lower end of the 95 percent Anderson--Rubin set for all children, the completion gap accounts for {100 * B(arl, 'college_share'):.0f} percent")
chk("absent: shares above 100 percent read as shares", "and the shares 107 and 182 percent", count=0)
I16 = {r["item"]: float(r["value"]) for r in u1("u16_identity_in_balanced_comparison.csv")}
t1, t2, t3 = I16["part 1: college channel (return of advantaged children x completion gap)"], I16["part 2: same degrees worth more (completion of less advantaged children x difference in returns)"], I16["part 3: gap between the children without a degree"]
rH16, rL16 = I16["return, advantaged children (graduates minus children without a degree)"], I16["return, less advantaged children (reweighted)"]
need(abs(t1 + t2 + t3 - I16["premium"]) < 1e-9 and abs(I16["premium"] - tau) < 1e-6 and abs(I16["completion gap"] - gapM) < 1e-6 and int(I16["children"]) == nl, "the three measured parts sum to the premium of the balanced comparison, on the same children")
chk("returns inside the balanced comparison", f"Graduates out-earn children without a degree by {rH16:.3f} log points among advantaged children and by {rL16:.3f} among reweighted less advantaged children, and the premium of {tau:.3f} has three parts")
chk("first part", f"The first, {t1:.3f}, comes from the degrees that advantaged children hold in excess of less advantaged children, valued at the return for advantaged children; it is the college channel, {100 * t1 / tau:.0f} percent of the premium")
chk("second part", f"The second, {t2:.3f}, comes from the degrees that both groups hold ({100 * EB['completion_less_advantaged_reweighted']:.0f} percent of each), which are worth {rH16 - rL16:.3f} log points more to advantaged children")
chk("third part", f"The third, {t3:.3f}, is the gap between children of the two groups without a degree, which is negligible"); need(abs(t3) < 0.005 and 0.75 < t1 / tau <= 1.0, "no gap between the children without a degree; the measured college channel lies between three quarters and all of the premium")
need(0.10 < B(lh, "direct") / tau < 0.15 and 0.22 < B(la, "direct") / tau < 0.27 and f"{B(lh, 'direct'):.2f}" == "0.01" and f"{B(la, 'direct'):.2f}" == "0.02", "direct effect: an eighth to a quarter of the premium, 0.01 to 0.02 log points")
chk("direct effect, Section 7", "the direct effect is between an eighth and a quarter of the premium, 0.01 to 0.02 log points; at the instrumented returns nothing is left for it"); chk("direct effect, conclusion", "At the least squares returns it is an eighth to a quarter of the premium."); chk("share of the premium, conclusion", f"valued at the return to college, accounts for between three quarters and all of the premium at its entropy-balanced value of {pct(tau):.0f} percent")
chk("ability reading, introduction", "their return is no lower than the least squares return"); chk("ability reading, first version: earnings at equal schooling (introduction)", f"earn the same (a difference of {t3:.3f} log points), and so, nearly, do advantaged and less advantaged graduates ({I16['gap between the graduates of the two groups']:.3f})."); need(abs(t3) < 0.005 and I16["gap between the graduates of the two groups"] < 0.03, "at equal schooling the two groups earn the same or nearly the same in the balanced comparison"); chk("ability reading, closing of the introduction", "Most of the premium therefore comes through access to college, not through ability rewarded outside it"); need(m["earn_iv"] > m["ols"] and m["ar_lo"] > m["ols"], "the return of the children the expansion admitted is no lower than the least squares return"); need(max(B(la, "direct") / tau, ls("coefficient on family background after college completion enters") / ls("coefficient on family background before college completion enters")) < 0.30, "at given college completion no more than about a quarter of the premium remains (identity and regression)")
chk("ability reading, introduction, closing", "I find no sign of that: their return is no lower than the least squares return."); need(B(la, "college_share") > 0.7, "college accounts for most of the premium even at the least squares return")
chk("ability reading, conclusion", "runs through access to college rather than through ability rewarded outside it, and the returns of the newly admitted suggest that access itself turned on preparation more than on ability."); chk("ability reading, conclusion, equal schooling", "among children without a degree the two groups earn the same, and among graduates nearly the same")
chk("raw gap split, Section 7", f"Of the {raw5:.3f} log points ({pct(raw5):.0f} percent) by which advantaged children out-earn less advantaged children, {B('raw gap: parental income and the demographic and provincial controls'):.3f} reflects parental income")
chk("raw gap split, remainder", f"and the remaining {tau:.3f} ({pct(tau):.0f} percent) is the premium $\\tau$ with parental income held fixed")
chk("premium and gap on the same comparison, introduction", f"the premium is {tau:.3f} log points ({pct(tau):.0f} percent) and the gap in college completion is {100 * gapM:.1f} percentage points"); chk("the split of the raw gap is stated once", "0.123 reflects parental income", count=1)
need(B(iv, "college_share") > 1 and 0.74 < B(la, "college_share") < 0.78, "college accounts for all of the premium at the instrumented return and three quarters at the least squares return")
chk("abstract: decomposition", f"At the entropy-balanced premium of {pct(tau):.0f} percent, the gap in college completion, valued at the return to college, accounts for between three quarters and all of the premium."); need(min(B(la, "college_share"), B(lh, "college_share"), B(iv, "college_share"), B(ivh, "college_share")) >= 0.75, "the completion gap accounts for at least three quarters of the premium at every return of the table")
chk("Table 1: decomposition", "All at the instrumented returns; three quarters at the least squares return")
chk("introduction: decomposition", f"Valued at the instrumented return for all children, {m['earn_iv']:.2f}, the completion gap accounts for all of the premium; valued at the least squares return, a lower benchmark, it accounts for three quarters.")
chk("decomposition table note", f"The premium with parental income held fixed is {tau:.3f} and the completion gap {gapM:.3f}")
chk("conclusion: premium", f"by {pct(tau):.0f} to {pct(s8(l_)):.0f} percent with parental income held fixed")
chk("conclusion: cohorts", f"born between {int(F14[f'{ALL}: first birth year'])} and {int(F14[f'{ALL}: last birth year'])}")

# census check of trends before the reform (one sentence of Section 6.4; the table is in the online appendix)
CEN = [r for r in rd(os.path.join(sibling("college_expansion", __file__), "output", "tables", "p16_census_pretrend.csv")) if r["n_cells"]]
cen_max = max(float(r["slope_diff"]) for r in CEN)
chk("census check of trends before the reform", f"by at most {100 * cen_max:.3f} percentage points per birth year, under one percentage point over 20 cohorts"); need(len(CEN) == 2 and 0 < 20 * 100 * cen_max < 1, "census: two censuses; the largest slope difference cumulates to under one percentage point over 20 cohorts")
chk("absent: the subsection that held the census check", "\\subsection{Robustness of the design}", count=0)
# the return for all children beside the returns by family background (first row of the table of returns): its 90 percent Anderson-Rubin set
G9 = sorted((float(r["b"]), float(r["p_wild"])) for r in u1("u9_anderson_rubin_grid.csv")); a90 = [b_ for b_, pv in G9 if pv >= 0.10]; a95 = [b_ for b_, pv in G9 if pv >= 0.05]
need(len(a90) == round((max(a90) - min(a90)) / 0.05) + 1, "90 percent set of the return for all children is one interval")
chk("90 percent set, all children", f"The 90 percent set for all children is $[{min(a90):.2f}, {max(a90):.2f}]$, above the least squares return of {m['ols']:.2f}"); need(min(a90) > m["ols"], "90 percent set of all children starts above the least squares return")
need(abs(min(a95) - m["ar_lo"]) < 1e-9 and max(a95) >= 3.99, "95 percent set of all children: the grid file agrees with u9_B_main.csv")
_abs = TEX[TEX.index("\\begin{abstract}"):TEX.index("\\textbf{JEL:}")]; need(len(re.sub(r"\\\\[a-z]+|[{}\\\\]|\\[4pt\\]", " ", _abs).split()) <= 290, "abstract of at most 290 words")
chk("title-page footnote points to the Online Appendix and the replication files", "The Online Appendix (supplementary results, proofs of the formal results, and a replication guide) and the replication files are available at \\href{\\repourl}{\\nolinkurl{github.com/JiachenJin1001/family-background-earnings-china}}."); chk("pointers to the Online Appendix: the footnote and three sentences in the body (fifteen specifications, checks of Section 6.4, college share at the regression estimates)", "The Online Appendix", count=4); chk("pointer to Part II of the Online Appendix for the exact attenuation formula", "derived in Part II of the Online Appendix", count=1); need("\\newcommand{\\repourl}{https://github.com/JiachenJin1001/family-background-earnings-china}" in TEX, "the repository address of the footnote")
G3 = {r["outcome"]: r for r in u1("u9_G_three_year_completion.csv")}; g15 = G3["exactly 15 years of schooling (three-year degree)"]
chk("three-year completion does not rise with the instrument", f"does not rise with the instrument (coefficient {sg(float(g15['coef']), 2)}, SE {float(g15['se']):.2f}, wild-cluster $p = {float(g15['p_wild']):.2f}$)"); need(float(g15["coef"]) < 0 and float(g15["p_wild"]) > 0.10, "three-year completion: negative coefficient, not distinguishable from zero")
chk("absent: 0.68 called an upper benchmark", "upper benchmark", count=0); chk("absent: sets printed as infinite at the edge of the grid", "printed as infinite", count=0)
for lbl_, snip_ in [("local average treatment effect named", "a local average treatment effect in the sense of \\citet{ImbensAngrist1994}: a weighted average of the returns to college for compliers"), ("Oaxaca-Blinder named", "the identity is an Oaxaca--Blinder decomposition with college completion as the only covariate"),
                    ("reliability ratio named", "the reliability ratio $\\lambda_T$ is"), ("errors-in-variables named", "the errors-in-variables problem that \\citet{Solon1992}"), ("difference-in-differences named", "the comparison is a difference-in-differences"),
                    ("parallel trends and exclusion named", "parallel trends (provinces that added more seats were not already on different trends) and exclusion"), ("event study named", "reports the event study"), ("intergenerational elasticity named", "intergenerational income elasticity"),
                    ("identity against valuation", "Evaluated at a return estimated elsewhere, by regression or by the instrument, it prices the completion gap and leaves Direct as the remainder: the premium that would remain if"),
                    ("the conclusion ends on the answer", "the advantage that parental occupation confers is settled at the university gate, and the children who came through it when seats were added earned as much from college as those before them.")]:
    chk(lbl_, snip_)
need(TEXN.rindex("settled at the university gate, and the children who came through it") > TEXN.rindex("Studies that compare intergenerational persistence"), "the last paragraph of the conclusion is the one on the expansion")
print("=== wording that must appear, and the table fragments")
for lbl, snip in [("abstract: instrument defined", "an instrument that interacts the growth of university seats in the province with each birth cohort's exposure"), ("abstract: the answer", "It is in place before work begins."),
                  ("title", "Family Background, Earnings Dynamics, and the Return to College}"), ("the two columns bracket the premium", "The two columns err in opposite directions, so they bracket the premium."), ("entropy balancing has its own subsection", "\\subsection{Entropy balancing}"),
                  ("below-30 term not interpreted", "I do not report or interpret $\\beta_b$: every child observed below 30"), ("low-schooling check in the introduction is the occupation result", "did not move into official, managerial, or professional occupations where seats grew faster"), ("first sentence is about the panel", "In a national panel of Chinese households, children of officials"), ("keyword matches the title", "intergenerational mobility; earnings dynamics;"), ("advantage means occupation, not income", "so an advantaged family here need not be a high-income family"), ("the same parameters for attenuation and for the weights", "The same three parameters do two jobs")]:
    chk(lbl, snip)
# the table fragments on disk must be what the files of estimates imply
import subprocess as _sp
_r = _sp.run([sys.executable, os.path.join(ROOT, "code", "build_paper_tables.py"), "--check"], capture_output=True, text=True)
_msg = _r.stdout.strip().splitlines()[-1] if _r.stdout.strip() else _r.stderr.strip()[-200:]
need(_r.returncode == 0, f"table fragments regenerate identically: {_msg}")

if fails:
    print(f"{fails} check(s) FAILED")
    sys.exit(1)
print("all checks of the paper passed")
