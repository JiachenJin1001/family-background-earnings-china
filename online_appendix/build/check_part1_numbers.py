"""Checks of Part I and Part III of the online appendix.

1. Every table of Part I is rebuilt in memory from the estimation output and
   compared with the file in tables/; a stale table fails.
2. Numbers typed in the running text of online_appendix.tex are asserted
   against the files of estimates they come from.
3. Every file that the appendix links to exists in the repository.
4. The source contains no Chinese character and no em dash.

    python3 online_appendix/build/check_part1_numbers.py
"""
import os, re, sys, math, subprocess, tempfile, shutil, filecmp
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(OA, "..", "code"))
sys.path.insert(0, os.path.join(OA, "..", "unified", "code"))
from _locate import sibling
U = sibling("unified", __file__)
M = sibling("income_dynamics", __file__)
P = sibling("college_expansion", __file__)
BASE = os.path.dirname(U)

TEX = open(os.path.join(OA, "online_appendix.tex"), encoding="utf-8").read()
PART3 = open(os.path.join(OA, "part3_replication.tex"), encoding="utf-8").read()
PART2 = open(os.path.join(OA, "part2_formal_results.tex"), encoding="utf-8").read()
TEXN = re.sub(r"\s+", " ", TEX)
fails = 0


def check(label, ok, detail=""):
    global fails
    print(("  ok   " if ok else "  FAIL ") + f"{label:70s} {detail}")
    fails += (not ok)


def text(label, snippet):
    check(label, re.sub(r"\s+", " ", snippet) in TEXN, repr(snippet))


# 1. tables are up to date
saved = os.path.join(OA, "tables")
with tempfile.TemporaryDirectory() as tmp:
    env = dict(os.environ, OA_TABLES_DIR=tmp, OA_FIGURES_DIR=tmp, MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, os.path.join(HERE, "build_tables.py")], env=env,
                       capture_output=True, text=True)
    check("build_tables.py runs", r.returncode == 0, r.stderr[-300:])
    built = sorted(f for f in os.listdir(tmp) if f.endswith(".tex"))
    for f in built:
        same = os.path.exists(os.path.join(saved, f)) and filecmp.cmp(os.path.join(tmp, f), os.path.join(saved, f), shallow=False)
        check(f"table file up to date: {f}", same)
    used = set(re.findall(r"\\input\{tables/(oa_[a-z]+)\}", TEX))
    check("every table read by the appendix is built by the script", used <= {f[:-4] for f in built},
          str(sorted(used - {f[:-4] for f in built})))

# 2. numbers in the running text
D = pd.read_csv(os.path.join(U, "output", "u3_D_decomposition_grid.csv"))
_lab = D[D.premium_estimate.str.contains("parental labor income")]
check("college share is smallest at the premium that controls for the parents' labor income", D.college_share.min() == _lab.college_share.min()
      and f"{100 * _lab.college_share.iloc[0]:.0f} percent at the least squares return and {100 * _lab.college_share.iloc[1]:.0f} percent at the instrumented return" in TEXN,
      f"minimum {D.college_share.min():.3f}")
_age = pd.read_csv(os.path.join(U, "output", "u4_premium_age_controls.csv"))
check("age controls: no row differs from the linear specification by more than 0.002", ((_age.b - _age.b.iloc[0]).abs() < 0.002).all()
      and "No row differs from the linear specification by more than 0.002" in TEXN, str(_age.b.round(4).tolist()))

_m5 = pd.read_csv(os.path.join(M, "output", "tables", "m5_moment_se.csv")); _frag = open(os.path.join(OA, "tables", "oa_autocov.tex"), encoding="utf-8").read()
_m5v = {(r.group, int(r.t), int(r.s)): (r.omega, r.se) for r in _m5.itertuples()}
check("table of moments by wave: every printed variance and first-order autocovariance matches the file of moments",
      all(f"{_m5v[(g, a, b)][0]:.2f} ({_m5v[(g, a, b)][1]:.2f})" in _frag for g in ("pooled", "H", "L") for a, b in [(y, y) for y in range(2014, 2023, 2)] + [(y, y + 2) for y in range(2014, 2021, 2)])
      and len(_m5v) == 45 and "\\input{tables/oa_autocov}" in TEX, f"{len(_m5v)} moments on file")
_zz = {k[1:]: abs(_m5v[k][0] - _m5v[("L",) + k[1:]][0]) / math.hypot(_m5v[k][1], _m5v[("L",) + k[1:]][1]) for k in _m5v if k[0] == "H"}
check("table of moments by wave: no gap between the groups above two standard errors, the largest at (2014, 2022)",
      max(_zz.values()) < 2 and max(_zz, key=_zz.get) == (2014, 2022) and f"the largest standardized gap is {max(_zz.values()):.2f}" in open(os.path.join(OA, "tables", "oa_autocovmax.tex"), encoding="utf-8").read(), f"{max(_zz.values()):.2f}")
cen = pd.read_csv(os.path.join(P, "output", "tables", "p16_census_pretrend.csv")).dropna(subset=["n_cells"])
check("pre-reform slope difference cumulates to under one point over twenty cohorts",
      (20 * cen.slope_diff.abs() < 0.01).all() and "the difference cumulates to under one percentage point" in TEXN,
      str((20 * cen.slope_diff).round(4).tolist()))

ab = pd.read_csv(os.path.join(M, "data", "intermediate", "m17_ab_diagnostics.csv"))
check("second-order serial correlation not rejected on either panel", (ab.m2_p > 0.10).all(),
      str(ab.m2_p.round(2).tolist()))

J = pd.read_csv(os.path.join(U, "output", "u7_error_structures.csv")).set_index("sample")
check("random walk fits worse in every sample", (J.criterion_random_walk > J.criterion_ar1).all())
check("richer transitory structures leave the criterion within one unit",
      ((J.criterion_arma11 - J.criterion_ar1).abs() < 1).all() and ((J.criterion_ma1 - J.criterion_ar1).abs() < 1).all())

O = pd.read_csv(os.path.join(U, "output", "u3_A_occupation_definitions.csv"))
check("occupation reduced form positive under every definition", (O.rf > 0).all())
check("smallest under 'ever'", O.sort_values("rf").definition.iloc[0] == "ever" and "it is smallest for an occupation held in any single wave" in TEXN, str(O.sort_values("rf").definition.tolist()))

paper = open(os.path.join(U, "paper", "main.tex"), encoding="utf-8").read()
# 3. linked files exist
links = set(re.findall(r"\\(?:code|folder)\{([^}]*)\}", TEX + PART2 + PART3))
for rel in sorted(links):
    here = os.path.join(BASE, rel)
    alt = rel
    for pub, work in [("unified/", "unified/"), ("income_dynamics/", "income_dynamics/"),
                      ("college_expansion/", "college_expansion/"),
                      ("formal_results/", "formal_results/"),
                      ("online_appendix/", "unified/online_appendix/")]:
        if rel.startswith(pub) or rel == pub[:-1]:
            alt = work + rel[len(pub):] if rel != pub[:-1] else work[:-1]
    found = os.path.exists(here) or os.path.exists(os.path.join(BASE, alt)) or rel in ("requirements.txt", "run_all.sh")
    check(f"linked file exists: {rel}", found)

# 3a. the Arellano-Bond estimates in Python, Mata and xtabond2 (Section B.5): the parity file written from the Stata logs
_sp = pd.read_csv(os.path.join(M, "output", "tables", "stata_abgmm_parity.csv")).set_index(["model", "implementation"])
_m10 = pd.read_csv(os.path.join(M, "output", "tables", "m10_abgmm_table3.csv")).set_index("panel")
_py, _mata, _xt = (_sp.loc[("difference GMM, five waves", k)] for k in ("Python", "Mata", "xtabond2"))
check("Stata parity: the Python row of the parity file is the row of Table 4", abs(_py.rho0 - _m10.loc["diff GMM, 2014-2022", "rho0"]) < 1e-9 and abs(_py.hansen_J - _m10.loc["diff GMM, 2014-2022", "J"]) < 1e-9)
check("Stata parity: Python and Mata agree to four decimals", all(abs(_py[k] - _mata[k]) < 5e-5 for k in ("rho0", "rho0_se", "rho1", "rho1_se", "hansen_J")))
text("Stata parity: xtabond2 difference GMM", f"\\texttt{{xtabond2}} gives {_xt.rho0:.3f} (SE {_xt.rho0_se:.3f}) and {_xt.rho1:.3f} (SE {_xt.rho1_se:.3f}) for the two coefficients of Table~\\ref{{P-tab:ab}} of the paper, and a Hansen statistic of {_xt.hansen_J:.2f} against {_py.hansen_J:.2f}.")
_pys, _xts = _sp.loc[("system GMM, five waves", "Python")], _sp.loc[("system GMM, five waves", "xtabond2")]
text("Stata parity: system GMM", f"Python and \\texttt{{xtabond2}} give persistence of {_pys.rho0:.3f} and {_xts.rho0:.3f}, and the Hansen test rejects in both ({_pys.hansen_J:.2f} and {_xts.hansen_J:.2f} on four degrees of freedom)")
check("Stata parity: both system GMM Hansen tests reject at 1 percent", _pys.hansen_p < 0.01 and _xts.hansen_p < 0.01 and int(_pys.df) == 4 and int(_xts.df) == 4)

# 3a2. the cohorts left out in the last row of the table of checks are the pre-reform bin with the larger completion gradient
_bins = pd.read_csv(os.path.join(U, "output", "u9_D_cohort_bins.csv")); _bins = _bins[_bins.outcome.str.startswith("college")].set_index("bin").coef
check("pre-reform bin with the largest completion gradient is 1973-1976", abs(_bins["1973-1976"]) > abs(_bins["1972 or earlier"]) and "leaves out the cohorts born 1973 to 1976, the pre-reform cohorts whose college completion has the largest gradient" in TEXN, f"{_bins['1973-1976']:.3f} against {_bins['1972 or earlier']:.3f}")
_c9 = pd.read_csv(os.path.join(U, "output", "u9_C_checks.csv")).set_index("check").loc["cohorts born 1973 to 1976 left out"]
check("return without the cohorts born 1973 to 1976 stays above the least squares return", _c9.iv > 0.45 and _c9.p_wild < 0.01, f"return {_c9.iv:.2f}, wild p {_c9.p_wild:.3f}")

# 3b. earnings at ages 30 and over: variants of the specification, checks, sample counts cited in the notes
MAIN = "predicted provincial growth (main measure)"
B9 = pd.read_csv(os.path.join(U, "output", "u9_B_main.csv")).set_index("intensity").loc[MAIN]; B9w = pd.read_csv(os.path.join(U, "output", "u9_B_main_weighted.csv")).set_index("intensity").loc[MAIN]
C9 = pd.read_csv(os.path.join(U, "output", "u9_C_checks.csv")).set_index("check"); C9w = pd.read_csv(os.path.join(U, "output", "u9_C_checks_weighted.csv")).set_index("check")
o_, y2 = "rows at 30 and over only (no row below 30)", "row below 30 with at least 2 wave(s)"
_iv = [B9.earn_iv, C9.loc[o_, "iv"], C9.loc[y2, "iv"], B9w.earn_iv, C9w.loc[o_, "iv"]]; _p = [B9.earn_p_wild, C9.loc[o_, "p_wild"], C9.loc[y2, "p_wild"], B9w.earn_p_wild, C9w.loc[o_, "p_wild"]]
text("variants: range of the return", f"between {min(_iv):.2f} and {max(_iv):.2f} in every row")
check("variants: wild p of the reduced form below 0.005", max(_p) < 0.005 and "is below 0.005" in TEXN, str([round(x, 4) for x in _p]))
_keep = ["trade control: coastal x exposure", "trade control: export share x exposure", "trade control: both", "region-specific linear cohort trends (three regions)", "instrument by the province at age twelve"]
check("checks: cohort exposure at age seventeen or nineteen leaves the reduced form significant", all(C.loc[f"exposure ratio of the year the cohort turned {k}", "p_wild"] < 0.05 for C in (C9, C9w) for k in (17, 19)))
check("checks: trade controls, region trends and the province at age twelve leave the reduced form significant", all(C.loc[k, "p_wild"] < 0.01 for C in (C9, C9w) for k in _keep))
check("checks: low-schooling children do not respond", all(C.loc[f"earnings, children with {k} or fewer years of schooling", "p_wild"] > 0.2 for C in (C9, C9w) for k in (9, 12)))
_pt = C9.loc["province-specific linear cohort trends"]
text("checks: linear cohort trend for each province", f"the earnings reduced form is {_pt['coef']:.3f} (SE {_pt['se']:.3f}), and the return to college is {_pt['iv']:.2f}.")
_g3 = pd.read_csv(os.path.join(U, "output", "u9_G_three_year_completion.csv")).set_index("outcome")
_a, _b, _c = (_g3.loc[k] for k in ("exactly 15 years of schooling (three-year degree)", "15 or more years of schooling", "16 or more years of schooling (college completion)"))
text("three-year completion on the instrument", f"is $-{abs(_a['coef']):.3f}$ (SE {_a['se']:.3f}, wild-cluster $p = {_a['p_wild']:.2f}$) when the outcome is exactly fifteen years of schooling, the three-year degree, {_b['coef']:.3f} (SE {_b['se']:.3f}) when it is fifteen or more years, and {_c['coef']:.3f} (SE {_c['se']:.3f}) for the completion indicator of the paper")
_v = pd.read_csv(os.path.join(U, "output", "u17_income_variance_by_wave.csv")); _v = _v[_v.respondents.str.startswith("all respondents")].set_index("wave").variance_of_log_income
text("variance of log income by wave", f"is {_v[2010]:.2f} in the 2010 wave and {_v[2012]:.2f} in the 2012 wave, against {_v[2014]:.2f}, {_v[2016]:.2f}, {_v[2018]:.2f}, {_v[2020]:.2f}, and {_v[2022]:.2f} in the five waves from 2014 to 2022")
_i = pd.read_csv(os.path.join(U, "output", "u16_identity_in_balanced_comparison.csv")).set_index("item")["value"]
text("three parts inside the balanced comparison", f"by {_i['return, advantaged children (graduates minus children without a degree)']:.3f} log points among advantaged children and by {_i['return, less advantaged children (reweighted)']:.3f} among reweighted less advantaged children; {100 * _i['completion, advantaged children']:.2f} percent of the former and {100 * _i['completion, less advantaged children (reweighted)']:.2f} percent of the latter hold a degree.")
text("three parts, values", f"account for {_i['part 1: college channel (return of advantaged children x completion gap)']:.3f}; the degrees held in both groups, worth {_i['return, advantaged children (graduates minus children without a degree)'] - _i['return, less advantaged children (reweighted)']:.3f} more to advantaged children, account for {_i['part 2: same degrees worth more (completion of less advantaged children x difference in returns)']:.3f}; and the earnings gap between the two groups among children without a degree is {_i['part 3: gap between the children without a degree']:.3f}.")
_F14 = pd.read_csv(os.path.join(U, "output", "u14_B_sample_facts.csv")).set_index("item")["value"]
text("children born before the admissions series", f"the {int(_F14['children born before 1964'])} children of the sample born before it are assigned the mean ratio of the cohorts born 1964 to 1980")
_r15 = pd.read_csv(os.path.join(U, "output", "u15_nonlinear_persistence_summary.csv")).set_index("sample")
check("nonlinear persistence: advantaged above less advantaged in each of ten runs", int(_r15.loc["pooled", "runs"]) == 10 and int(_r15.loc["pooled", "runs_advantaged_above_less_advantaged"]) == 10)
_rw = pd.read_csv(os.path.join(U, "output", "u9_R_randomization_within_region.csv")).iloc[0]
check("randomization inference: 4,999 reassignments; 10, 8 and 11 provinces by region", int(_rw["reassignments"]) == 4999 and _rw["provinces_by_region"] == "{'C': 8, 'E': 10, 'W': 11}" and "(ten provinces in the east, eight in the center, and eleven in the west)" in TEXN)
F14 = pd.read_csv(os.path.join(U, "output", "u14_B_sample_facts.csv")).set_index("item")["value"]
text("sample size in the notes", f"({int(F14['all children: children']):,} children observed in at least two waves at ages 22 to 55)")
text("children with a level of earnings, notes", f"The {int(F14['level at 30 and over: children']):,} children with a level of earnings at ages 30 and over")
text("child-waves at 30 and over, notes", f"the same {int(F14['level at 30 and over: child-waves']):,} child-wave observations at ages 30 and over")
text("provinces", f"The {int(F14['all children: provinces'])} provinces of the sample"); text("clusters", f"inference rests on {int(F14['all children: provinces'])} clusters")
_eb = pd.read_csv(os.path.join(U, "output", "u8_entropy_balancing.csv")).iloc[0]; text("completion gap in the notes", f"gap in college completion of {_eb.gap_hat:.3f}")

# 4. language
for name, src in [("online_appendix.tex", TEX), ("part3_replication.tex", PART3)]:
    check(f"no Chinese character in {name}", re.search(r"[\u4e00-\u9fff]", src) is None)
    prose = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("%"))
    check(f"no em dash in {name}", "\u2014" not in prose and re.search(r"(?<!-)---(?!-)", prose) is None)

print()
if fails:
    print(f"{fails} check(s) FAILED"); sys.exit(1)
print("all checks of Parts I and III passed")
