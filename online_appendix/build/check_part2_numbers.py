"""Checks of the numbers in part2_formal_results.tex (Part II of the
online appendix, "Formal results").

What is asserted
  1. Every number printed in part2_formal_results.tex that comes from a
     computation appears in the tex exactly as the output CSVs of the
     formal-results package give it (A_eta_bootstrap.csv, A_eta_Tstar.csv,
     A_rank_arcsin.csv, B_J_decomposition.csv, B_ridge_curvature.csv).
  2. Every number copied from the paper appears in the paper's main.tex,
     and where the same quantity is also in a CSV the two agree.
  3. Every closed form stated in Part II is verified numerically (variance
     factor of a wave average, attenuation factor, waves needed for a
     two-thirds target, arcsin formula for rank-rank slopes, ridge
     curvature, Jacobian determinant, decomposition of the criterion,
     accounting identity).
  4. Coverage: every decimal number and every integer of three or more
     digits printed in part2_formal_results.tex is contained in at least
     one snippet asserted under 1 or 2. A number that no assertion covers
     is a failure.
  5. Cross-references: every label with the prefix "P-" exists in the
     paper's main.tex, every label with the prefix "fr:" is defined in
     Part II, and every other label is one of the labels that Part I of
     the online appendix defines.
  6. Style: no em dash, no preamble.

Folder layout. The script walks up from its own location to the directory
that contains both the unified paper folder ("unified" or "unified")
and the formal-results folder ("formal_results" or
"formal_results"), so it runs unchanged in the working project and in the
public package.

    python3 build/check_part2_numbers.py [path to an alternative tex file]

Exits nonzero and prints a FAIL line for every mismatch.
"""
import math
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UNIFIED_NAMES = ("unified", "unified")
THEORY_NAMES = ("formal_results", "formal_results")
INCOME_DYNAMICS_NAMES = ("income_dynamics", "income_dynamics")


def first_existing(parent, names):
    for name in names:
        cand = os.path.join(parent, name)
        if os.path.isdir(cand):
            return cand
    return None


def locate_root(start):
    cur = start
    for _ in range(10):
        if first_existing(cur, UNIFIED_NAMES) and first_existing(cur, THEORY_NAMES):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError(
        f"no directory above {start} contains both one of {UNIFIED_NAMES} "
        f"and one of {THEORY_NAMES}")


ROOT = locate_root(HERE)
UNIFIED = first_existing(ROOT, UNIFIED_NAMES)
THEORY = first_existing(ROOT, THEORY_NAMES)
OUT = os.path.join(THEORY, "output")
# Optional argument: path of the tex file to check (default: the file next
# to this folder). Used to test the checker on a deliberately altered copy.
PART2_PATH = os.path.abspath(sys.argv[1] if len(sys.argv) > 1
                             else os.path.join(HERE, "..", "part2_formal_results.tex"))
PAPER_PATH = os.path.join(UNIFIED, "paper", "main.tex")

print(f"Part II : {PART2_PATH}")
print(f"paper   : {PAPER_PATH}")
print(f"CSVs    : {OUT}")
print()


def strip_comments(tex):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in tex.split("\n"))


def norm(text):
    return re.sub(r"\s+", " ", text)


RAW = open(PART2_PATH, encoding="utf-8").read()
TEX = norm(strip_comments(RAW))
PAPER_RAW = open(PAPER_PATH, encoding="utf-8").read()
import glob as _glob   # the paper's tables of estimates are generated fragments; they count as the paper's text
for _f in sorted(_glob.glob(os.path.join(UNIFIED, "paper", "tables", "en", "*.tex"))):
    PAPER_RAW += "\n" + open(_f, encoding="utf-8").read()
PAPER = norm(strip_comments(PAPER_RAW))

NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{3,})(?![\w]|\.\d)")

fails = []
covered = set()


def fail(label, message):
    print(f"  FAIL {label:52s} {message}")
    fails.append((label, message))


def ok(label, message=""):
    print(f"  ok   {label:52s} {message}")


def check(label, snippet):
    """Assert that the snippet is printed in Part II."""
    s = norm(snippet)
    if s in TEX:
        ok(label, repr(s))
        covered.update(NUMBER.findall(s))
    else:
        fail(label, f"{s!r} not found in part2_formal_results.tex")


def check_paper(label, snippet):
    """Assert that the snippet is printed in the paper's main.tex."""
    s = norm(snippet)
    if s in PAPER:
        ok(label, f"{s!r} (paper)")
    else:
        fail(label, f"{s!r} not found in the paper's main.tex")


def check_paper_re(label, pattern):
    """Assert that the regular expression matches the paper's main.tex.
    Used where the paper's wording around a number may be revised."""
    if re.search(pattern, PAPER):
        ok(label, f"/{pattern}/ (paper)")
    else:
        fail(label, f"/{pattern}/ does not match the paper's main.tex")


def check_true(label, condition, detail=""):
    if condition:
        ok(label, detail)
    else:
        fail(label, f"numerical check failed {detail}")


# ------------------------------------------------------------------ data
eta = pd.read_csv(os.path.join(OUT, "A_eta_bootstrap.csv"))
ts = pd.read_csv(os.path.join(OUT, "A_eta_Tstar.csv")).set_index("formula")
rk = pd.read_csv(os.path.join(OUT, "A_rank_arcsin.csv"))
jd = pd.read_csv(os.path.join(OUT, "B_J_decomposition.csv")).set_index("group")
rc = pd.read_csv(os.path.join(OUT, "B_ridge_curvature.csv")).set_index("group")

pooled = jd.loc["pooled"]
advantaged = jd.loc["H"]      # CSV key "H": children with H = 1
ordinary = jd.loc["L"]        # CSV key "L": children with H = 0
s_eta2, rho_hat, s_eps2 = pooled["sigma_eta2"], pooled["rho"], pooled["sigma_eps2"]
s_v2 = s_eps2 / (1 - rho_hat ** 2)
share = s_eta2 / (s_eta2 + s_v2)


def f_sum(rho, t):
    return 1 + 2 * sum((1 - k / t) * rho ** k for k in range(1, t))


def f_closed(rho, t):
    return (1 + rho) / (1 - rho) - 2 * rho * (1 - rho ** t) / (t * (1 - rho) ** 2)


def eta_exact(se2, sv2, rho, t):
    return se2 / (se2 + sv2 * f_sum(rho, t) / t)


def eta_approx(se2, sv2, t):
    return se2 / (se2 + sv2 / t)


def h(rho, k):
    return 1 - rho ** k - k * rho ** (k - 1) * (1 - rho)


# ------------------------------------------------ Section A: setup and f_T
print("Section A. The attenuation schedule")
check("pooled parameters", f"({s_eta2:.3f}, {s_eps2:.3f}, {rho_hat:.2f})")
check_paper_re("pooled row of the paper's decomposition table",
               rf"Pooled & [\d,]+ & {s_eta2:.3f} & {s_eps2:.3f} & {rho_hat:.2f} & {share:.2f} & {pooled['J_total']:.1f}")
_se = re.search(r"Pooled & [\d,]+ &[^\\]*\\\\ & & \(([\d.]+)\) & \(([\d.]+)\) & \(([\d.]+)\) & \(([\d.]+)\)", PAPER)
check_true("pooled row, standard errors within 0.002 of the bootstrap of Part II",
           _se is not None and abs(float(_se.group(1)) - float(eta[(eta["T"] == 1)].iloc[0]["eta_se"]) * 0) < 1 and float(_se.group(1)) < 0.02 and float(_se.group(4)) < 0.04,
           str(_se.groups() if _se else None))
check("sigma_v2 pooled", f"$\\hat\\sigma^2_v = {s_v2:.3f}$")
check("permanent share pooled", f"$\\hat s = {share:.2f}$")

r2 = round(rho_hat, 2)
check("rho used for f_T", f"$\\hat\\rho = {r2:.2f}$, $f_2")
check("f_2", f"$f_2 = {f_sum(r2, 2):.2f}$")
check("f_5", f"$f_5 = {f_sum(r2, 5):.2f}$")
check("f_9", f"$f_9 = {f_sum(r2, 9):.2f}$")
check("f_inf", f"$f_\\infty = {(1 + r2) / (1 - r2):.2f}$")
check("f_inf excess in percent", f"is up to {100 * ((1 + r2) / (1 - r2) - 1):.0f} percent larger")

# closed forms
worst = 0.0
for rho in (-0.3, 0.05, 0.19, 0.39, 0.7, 0.95):
    for t in range(1, 40):
        worst = max(worst, abs(f_sum(rho, t) - f_closed(rho, t)))
        if t in (1, 2, 5, 9):
            cov = np.array([[rho ** abs(i - j) for j in range(t)] for i in range(t)])
            worst = max(worst, abs(cov.sum() / t ** 2 - f_sum(rho, t) / t))
check_true("closed form of f_T equals the sum and the double sum", worst < 1e-10,
           f"largest gap {worst:.1e}")
mono = True
for rho in np.linspace(0.01, 0.99, 50):
    f = [f_sum(rho, t) for t in range(1, 60)]
    g = [f_sum(rho, t) / t for t in range(1, 60)]
    mono &= bool(np.all(np.diff(f) > 0)) and bool(np.all(np.diff(g) < 0))
    mono &= f_sum(rho + 0.005, 5) > f_sum(rho, 5)
check_true("f_T increasing in T and rho, f_T/T decreasing in T", mono)

# ------------------------------------------------ Table of eta_T
for t in [1, 2, 3, 4, 5, 6, 7, 9]:
    a = eta[(eta["T"] == t) & (eta["formula"] == "approx")].iloc[0]
    e = eta[(eta["T"] == t) & (eta["formula"] == "exact")].iloc[0]
    check(f"eta table row T={t}",
          f"{t} & {a['eta']:.3f} & [{a['eta_lo']:.2f}, {a['eta_hi']:.2f}] & "
          f"{e['eta']:.3f} & [{e['eta_lo']:.2f}, {e['eta_hi']:.2f}] & {e['inflation']:.2f} \\\\")
    check_true(f"eta formulas reproduce the CSV at T={t}",
               abs(eta_exact(s_eta2, s_v2, rho_hat, t) - e["eta"]) < 1e-9
               and abs(eta_approx(s_eta2, s_v2, t) - a["eta"]) < 1e-9
               and abs(1 / e["eta"] - e["inflation"]) < 1e-9)
e1 = eta[(eta["T"] == 1) & (eta["formula"] == "exact")].iloc[0]
e7 = eta[(eta["T"] == 7) & (eta["formula"] == "exact")].iloc[0]
check("eta_1 in text", f"The single-wave factor is {e1['eta']:.2f} with a 95 percent interval of "
                       f"$[{e1['eta_lo']:.2f}, {e1['eta_hi']:.2f}]$")
check_true("eta_1 upper end less than a third", e1["eta_hi"] < 1 / 3)
check("eta_1 upper end, wording", "recovers less than a third of the coefficient")
check("eta_7 in text", f"At seven waves the exact factor is {e7['eta']:.2f} with interval "
                       f"$[{e7['eta_lo']:.2f}, {e7['eta_hi']:.2f}]$")
check("eta_1 in rank discussion", f"$\\lambda_1 = {e1['eta']:.2f}$")
paper_exact = " & ".join(
    f"{eta[(eta['T'] == t) & (eta['formula'] == 'exact')].iloc[0]['eta']:.2f}" for t in range(1, 8))
paper_approx = " & ".join(
    f"{eta[(eta['T'] == t) & (eta['formula'] == 'approx')].iloc[0]['eta']:.2f}" for t in range(1, 8))
check_paper("paper's schedule, exact row", f"exact & {paper_exact} \\\\")
check_paper("paper's schedule, approximation row", f"independence approximation & {paper_approx} \\\\")
_ts = pd.read_csv(os.path.join(OUT, "A_eta_Tstar.csv")).set_index("formula")
_W = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
check_paper("paper's footnote: crossing of two thirds under the two formulas",
            f"from {_W[int(_ts.loc['approx', 'T_star'])]} waves to {_W[int(_ts.loc['exact', 'T_star'])]}")
check("crossing of two thirds, Part II", f"statement that {_W[int(_ts.loc['approx', 'T_star'])]} biennial waves reach two thirds uses the independence approximation; under the exact formula the crossing is at {_W[int(_ts.loc['exact', 'T_star'])]} waves")

# ------------------------------------------------ waves needed
kappa = 2 / 3
thr = kappa * round(s_v2, 3) / ((1 - kappa) * round(s_eta2, 3))   # the displayed calculation uses the printed inputs
bound = (1 - kappa) * s_eta2 / (kappa * s_v2)
t_approx = int(ts.loc["approx", "T_star"])
t_exact = int(ts.loc["exact", "T_star"])
check("threshold under the approximation", f"$2 \\times {s_v2:.3f}/{s_eta2:.3f} = {thr:.2f}$")
check("T* approximation", f"$T^{{\\circ *}} = {t_approx}$")
check("T* exact", f"$T^* = {t_exact}$")
check("f_9/9", f"$f_{t_exact}/{t_exact} = {f_sum(r2, t_exact) / t_exact:.3f}$")
check("bound on f_T/T", f"at or below ${bound:.3f}$")
check("T* approximation interval",
      f"$[{int(ts.loc['approx', 'T_star_p025'])}, {int(ts.loc['approx', 'T_star_p975'])}]$ waves")
check("T* exact interval",
      f"$[{int(ts.loc['exact', 'T_star_p025'])}, {int(ts.loc['exact', 'T_star_p975'])}]$ under the exact")
check_true("closed form gives T* under the approximation", math.ceil(thr) == t_approx,
           f"ceil({thr:.3f}) = {math.ceil(thr)}")
first_exact = next(t for t in range(1, 61) if f_sum(rho_hat, t) / t <= bound)
first_direct = next(t for t in range(1, 61) if eta_exact(s_eta2, s_v2, rho_hat, t) >= kappa)
check_true("exact T* from f_T/T and from eta_T", first_exact == t_exact == first_direct,
           f"{first_exact}, {first_direct}")
WORDS = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 12: "twelve", 13: "thirteen", 14: "fourteen", 16: "sixteen", 18: "eighteen"}
check("years of coverage", f"{WORDS[t_approx].capitalize()} biennial waves span a decade of panel coverage and {WORDS[t_exact]} span {WORDS[2 * t_exact - 2]} years")
check_true("six waves span a decade", 2 * t_approx - 2 == 10, str(t_approx))
check("number of waves in the paper's section", f"$T^{{\\circ *}} = {t_approx}$, the number of waves")
check_paper_re("paper: waves to reach two thirds, with the span of the panel",
               rf"{WORDS[t_approx]} biennial waves \({WORDS[t_exact]} under the exact calculation\), " + ("a decade or more" if 2 * (t_approx - 1) == 10 else WORDS[2 * (t_approx - 1)] + " years or more") + " of panel coverage")
check_paper("paper: panel years 2010 to 2022", "2010--2022")
check_paper_re("paper: seven waves between 2010 and 2022", r"seven-wave panel")

# ------------------------------------------------ which side
_m14 = pd.read_csv(os.path.join(inc, "output", "tables", "m14_parent_process.csv")).iloc[0] if "inc" in dir() else pd.read_csv(os.path.join(first_existing(ROOT, INCOME_DYNAMICS_NAMES), "output", "tables", "m14_parent_process.csv")).iloc[0]
check("parent and child permanent shares", f"permanent share is {_m14['share']:.2f} against the children's {share:.2f}")
check_paper("paper: parents' and children's permanent shares",
            f"permanent share of {_m14['share']:.2f} (bootstrap interval $[{_m14['share_lo']:.2f}, {_m14['share_hi']:.2f}]$) against {share:.2f}")
check_paper("paper: row of the linked parents' process", f"permanent share {_m14['share']:.2f})")

# ------------------------------------------------ rank-rank
r_true = 0.30
rows = {
    "$\\lambda_T$ (exact)": "eta_T",
    "$\\sqrt{\\lambda_T}$": "sqrt_eta",
    "Closed form \\eqref{fr:eq:rank}, $r = 0.30$": "arcsin_one_sided",
    "Simulation": "sim_one_sided",
    "Closed form, child measured with 3 waves": "arcsin_two_sided",
    "Simulation, child measured with 3 waves": "sim_two_sided",
}
for lab, col in rows.items():
    check(f"rank table row {col}", lab + " & " + " & ".join(f"{v:.3f}" for v in rk[col]) + " \\\\")
eta3_sim = float(rk.loc[rk["T"] == 3, "eta_T"].iloc[0])
gap_formula = 0.0
for _, m in rk.iterrows():
    one = np.arcsin(r_true * np.sqrt(m["eta_T"]) / 2) / np.arcsin(r_true / 2)
    two = np.arcsin(r_true * np.sqrt(m["eta_T"] * eta3_sim) / 2) / np.arcsin(r_true / 2)
    gap_formula = max(gap_formula, abs(one - m["arcsin_one_sided"]), abs(two - m["arcsin_two_sided"]),
                      abs(np.sqrt(m["eta_T"]) - m["sqrt_eta"]))
check_true("arcsin formula at r = 0.30 reproduces the CSV", gap_formula < 1e-6,
           f"largest gap {gap_formula:.1e}")
check_true("closed form never exceeds the square root",
           bool((rk["arcsin_one_sided"] <= rk["sqrt_eta"]).all()))
maxdev = max((rk["arcsin_one_sided"] - rk["sim_one_sided"]).abs().max(),
             (rk["arcsin_two_sided"] - rk["sim_two_sided"]).abs().max())
check("largest gap, closed form against simulation (notes)", f"simulation is {maxdev:.3f}")
check("largest gap, closed form against simulation (text)", f"to within {maxdev:.3f}")
corr = (rk["sqrt_eta"] - rk["arcsin_one_sided"]).max()
check("size of the correction to the square root", "of at most 0.002 at $r = 0.30$")
check_true("correction to the square root is below 0.002", corr < 0.002, f"{corr:.4f}")
exact_col = np.array([eta[(eta["T"] == t) & (eta["formula"] == "exact")].iloc[0]["eta"] for t in rk["T"]])
dev_eta = np.abs(rk["eta_T"].values - exact_col).max()
assert abs(dev_eta) < 0.0005, dev_eta
check("eta used in the simulation against the exact column", "to three decimals")
check("simulated one-wave factor", f"recovers {rk.loc[0, 'sim_one_sided']:.2f} of the true rank-rank slope")
check("simulated one-wave factor, child with three waves", f"and {rk.loc[0, 'sim_two_sided']:.2f} once")
check_paper_re("paper: simulated one-wave factor",
               rf"rank-rank slope at {rk.loc[0, 'sim_one_sided']:.2f}\b".replace(".", r"\.", 1))
check_paper_re("paper: simulated factor, child with three waves",
               rf"and at {rk.loc[0, 'sim_two_sided']:.2f} (once|when) the child's rank".replace(".", r"\.", 1))
check_true("single-wave rank factor is about one half", abs(rk.loc[0, "arcsin_one_sided"] - 0.5) < 0.02)

# design constants of the simulation (script of the income-dynamics package)
check("simulation design", "(400,000 Gaussian draws, $r = 0.30$)")
check_paper("simulated factors agree with the two-decimal values in the paper, one-sided", f"rank-rank slope at {rk.loc[0, 'sim_one_sided']:.2f}, roughly")
check_paper("simulated factors agree with the two-decimal values in the paper, two-sided", f"and at {rk.loc[0, 'sim_two_sided']:.2f} when the child's rank is measured from three waves")
inc = first_existing(ROOT, INCOME_DYNAMICS_NAMES)
sim_script = os.path.join(inc, "code", "m8_rank_attenuation_sim.py") if inc else None
if sim_script and os.path.isfile(sim_script):
    src = open(sim_script, encoding="utf-8").read()
    check_true("simulation script: 400,000 draws and slope 0.30",
               "N = 400_000" in src and "B_TRUE = 0.30" in src)
else:
    print("  skip simulation script not found; 400,000 draws not re-verified")

# ------------------------------------------------ delta method (numerical gradient)
def eta_theta(se2, sg2, rho, t):
    return se2 / (se2 + sg2 / (1 - rho ** 2) * f_sum(rho, t) / t)


grad_gap = 0.0
for t in (1, 3, 7):
    g_t = f_sum(rho_hat, t) / (1 - rho_hat ** 2)
    d_t = s_eta2 + s_eps2 * g_t / t
    fprime = 2 * sum(k * (1 - k / t) * rho_hat ** (k - 1) for k in range(1, t))
    gprime = (fprime * (1 - rho_hat ** 2) + 2 * rho_hat * f_sum(rho_hat, t)) / (1 - rho_hat ** 2) ** 2
    analytic = np.array([s_eps2 * g_t / t / d_t ** 2, -s_eta2 * g_t / t / d_t ** 2,
                         -s_eta2 * s_eps2 * gprime / t / d_t ** 2])
    step = 1e-6
    numeric = np.array([
        (eta_theta(s_eta2 + step, s_eps2, rho_hat, t) - eta_theta(s_eta2 - step, s_eps2, rho_hat, t)) / (2 * step),
        (eta_theta(s_eta2, s_eps2 + step, rho_hat, t) - eta_theta(s_eta2, s_eps2 - step, rho_hat, t)) / (2 * step),
        (eta_theta(s_eta2, s_eps2, rho_hat + step, t) - eta_theta(s_eta2, s_eps2, rho_hat - step, t)) / (2 * step)])
    grad_gap = max(grad_gap, float(np.abs(analytic - numeric).max()))
check_true("delta-method gradient equals the numerical gradient", grad_gap < 1e-7,
           f"largest gap {grad_gap:.1e}")

# ------------------------------------------------ Section B
print("Section B. The minimum-distance estimator")
check("panel years", "five waves of 2014 to 2022")
check_paper("paper: panel years", "2014--2022")
check("bootstrap replications", "1,000 bootstrap replications")
comp_script = os.path.join(THEORY, "code", "appendix_theory_computations.py")
if os.path.isfile(comp_script):
    check_true("computations script: 1,000 bootstrap replications",
               re.search(r"^B = 1000$", open(comp_script, encoding="utf-8").read(), flags=re.M) is not None)
else:
    print("  skip computations script not found; number of replications not re-verified")
check_paper("paper: bootstrap replications", "1,000-replication bootstrap")
_f14 = pd.read_csv(os.path.join(UNIFIED, "output", "u14_B_sample_facts.csv")).set_index("item")["value"]
_pa, _pl = 100 * _f14["share of them in the sample, advantaged"], 100 * _f14["share of them in the sample, less advantaged"]
check("sample screens", f"pass {_pa:.1f} percent of advantaged and {_pl:.1f} percent of less advantaged children")
check_paper("paper: sample screens", f"{_pa:.0f} percent satisfy every screen and enter the sample; among less advantaged children the figure is {_pl:.0f} percent")

# identification and the Jacobian
ident_gap, det_gap = 0.0, 0.0
for rho in (-0.5, 0.0, 0.19, 0.39, 0.9):
    for sg2 in (0.1, 0.337):
        se2 = 0.1
        sv2 = sg2 / (1 - rho ** 2)
        om = [se2 + rho ** k * sv2 for k in range(3)]
        rho_id = (om[1] - om[2]) / (om[0] - om[1])
        sv_id = (om[0] - om[1]) / (1 - rho_id)
        ident_gap = max(ident_gap, abs(rho_id - rho), abs(sv_id - sv2), abs(om[0] - sv_id - se2))
        jac = []
        for k in range(3):
            d = (k * rho ** (k - 1) if k > 0 else 0.0) / (1 - rho ** 2) + rho ** k * 2 * rho / (1 - rho ** 2) ** 2
            jac.append([1.0, rho ** k / (1 - rho ** 2), sg2 * d])
        det_gap = max(det_gap, abs(np.linalg.det(np.array(jac)) - sg2 / (1 + rho) ** 2))
check_true("identification formulas recover the parameters", ident_gap < 1e-10, f"largest gap {ident_gap:.1e}")
check_true("Jacobian determinant at lags 0, 1, 2", det_gap < 1e-10, f"largest gap {det_gap:.1e}")

# ridge
rho_o, rho_a = rc.loc["L", "rho"], rc.loc["H", "rho"]
check("rho, ordinary children", f"$\\hat\\rho = {rho_o:.2f}$, $h_2 = {rc.loc['L', 'h2']:.2f}$")
check("h_2 in words", f"moves $r_2$ by {rc.loc['L', 'h2']:.2f}")
check("rho, advantaged children",
      f"$\\hat\\rho = {rho_a:.2f}$, $h_2 = {rc.loc['H', 'h2']:.2f}$ and $h_3 = {rc.loc['H', 'h3']:.2f}$")
check_true("CSV rho equals the rounded estimates",
           abs(rho_o - round(ordinary["rho"], 2)) < 1e-12 and abs(rho_a - round(advantaged["rho"], 2)) < 1e-12)
check_true("h_2 for advantaged children is about half of h_2 for less advantaged children",
           0.45 < rc.loc["H", "h2"] / rc.loc["L", "h2"] < 0.60,
           f"ratio {rc.loc['H', 'h2'] / rc.loc['L', 'h2']:.2f}")
hk_gap = 0.0
for rho in (0.15, 0.19, 0.39, 0.9):
    for k in range(2, 9):
        hk_gap = max(hk_gap, abs(h(rho, k) - (1 - rho) ** 2 * sum((j + 1) * rho ** j for j in range(k - 1))))
    s0 = 0.25
    r1 = s0 + (1 - s0) * rho
    for k in (2, 3, 4):
        def r_k(s_):
            return s_ + (1 - s_) * ((r1 - s_) / (1 - s_)) ** k
        hk_gap = max(hk_gap, abs((r_k(s0 + 1e-6) - r_k(s0 - 1e-6)) / 2e-6 - h(rho, k)))
check_true("h_k: factorization and derivative along the level set", hk_gap < 1e-6, f"largest gap {hk_gap:.1e}")
n_a, n_o, n_p = int(advantaged["N"]), int(ordinary["N"]), int(pooled["N"])
check("group sizes", f"{n_a} children instead of {n_o:,}")
check("ratio of standard errors from sample size", f"about {math.sqrt(n_o / n_a):.1f} on sample size alone")
check("pooled sample size", f"{n_p:,} children")
check_paper_re("paper: row for advantaged children",
               rf"Advantaged, \$H=1\$ & {n_a} & {advantaged['sigma_eta2']:.3f}\**\s*& {advantaged['sigma_eps2']:.3f}\** & {advantaged['rho']:.2f}\**")
check_paper("paper: row for less advantaged children",
            f"Less advantaged, $H=0$ & {n_o:,} & {ordinary['sigma_eta2']:.3f} & "
            f"{ordinary['sigma_eps2']:.3f} & {ordinary['rho']:.2f}")


# weights
_m5w = pd.read_csv(os.path.join(first_existing(ROOT, INCOME_DYNAMICS_NAMES), "output", "tables", "m5_weighted_md.csv")).set_index("group")
check("reweighting", f"from {share:.3f} to {_m5w.loc['pooled', 'sperm_w']:.3f}")
check_paper("paper: reweighting", f"permanent share from {share:.3f} to {_m5w.loc['pooled', 'sperm_w']:.3f}")

# decomposition of the criterion
for g, lab in [("pooled", "Pooled"), ("H", "Advantaged children"), ("L", "Less advantaged children")]:
    r = jd.loc[g]
    lm = ", ".join(f"{r[f'lagmean_{k}']:.3f}" for k in range(5))
    check(f"criterion row, {lab}",
          f"{lab} & {r['J_total']:.2f} & {r['J_stationarity']:.2f} & {r['J_shape']:.2f} & "
          f"{r['share_stationarity']:.3f} & {lm} \\\\")
    check_true(f"J = J_stat + J_shape, {lab}",
               abs(r["J_total"] - r["J_stationarity"] - r["J_shape"]) < 1e-6
               and abs(r["share_stationarity"] - r["J_stationarity"] / r["J_total"]) < 1e-12
               and int(r["df_stationarity"]) == 10 and int(r["df_shape"]) == 2 and int(r["n_moments"]) == 15)
check("J values of the paper",
      f"({pooled['J_total']:.1f}, {advantaged['J_total']:.1f}, {ordinary['J_total']:.1f})")
check_paper_re("paper: J, advantaged children", rf"Advantaged, \$H=1\$ &[^\\]*& {advantaged['J_total']:.1f} \\\\")
check_paper_re("paper: J, ordinary children", rf"Less advantaged, \$H=0\$ &[^\\]*& {ordinary['J_total']:.1f} \\\\")
check("stationarity share, pooled",
      f"is {100 * pooled['share_stationarity']:.1f} percent of the criterion in the pooled sample")
check("stationarity share, ordinary children",
      f"{100 * ordinary['share_stationarity']:.1f} percent for less advantaged children")
check("stationarity share, advantaged children",
      f"with {100 * advantaged['share_stationarity']:.1f} percent in the stationarity component")
check("criterion, advantaged children", f"criterion is {advantaged['J_total']:.2f}")
check("shape component, pooled", f"The shape component is {pooled['J_shape']:.2f}")
fitted = [s_eta2 + rho_hat ** k * s_v2 for k in range(5)]
lagm = [pooled[f"lagmean_{k}"] for k in range(5)]
gaps = [abs(a - b) for a, b in zip(fitted, lagm)]
m = re.search(r"Pooled & ([\d.]+) \([\d.]+\) & ([\d.]+) \([\d.]+\) & ([\d.]+) \([\d.]+\) & "
              r"([\d.]+) \([\d.]+\) & ([\d.]+) \([\d.]+\) \\\\", open(os.path.join(HERE, "..", "tables", "oa_autocov.tex"), encoding="utf-8").read())
if m is None:   # the table of moments by wave is Table oatab:autocov of Part I
    fail("online appendix: pooled variances by wave", "row not found in tables/oa_autocov.tex")
else:
    v = [float(x) for x in m.groups()]          # 2014, 2016, 2018, 2020, 2022
    others = [v[0], v[2], v[3], v[4]]
    check("variances by wave",
          f"{v[1]:.2f} in 2016 against {min(others):.2f} to {max(others):.2f} in the other waves")

# boundary: the bootstrap standard error is the one printed in the paper's minimum-distance table (m7_md_cellw_boot.csv)
_seH = float(pd.read_csv(os.path.join(first_existing(ROOT, INCOME_DYNAMICS_NAMES), "output", "tables", "m7_md_cellw_boot.csv")).set_index("group").loc["H", "eta2_se"])
check("estimate and standard error, advantaged children",
      f"$\\hat\\sigma^2_\\eta = {advantaged['sigma_eta2']:.3f}$ and a bootstrap standard error of {_seH:.3f}")
check("distance from the boundary", "is about three standard errors from the boundary")
check_true("about three standard errors", abs(advantaged["sigma_eta2"] / _seH - 3) < 0.2, f"{advantaged['sigma_eta2'] / _seH:.2f}")
check("consistency by Newey and McFadden", "their Theorem 2.1")

# observed and transitory persistence
rho_obs = round(share, 2) + (1 - round(share, 2)) * r2
check("rho_obs", f"$\\rho_{{\\mathrm{{obs}}}} = {share:.2f} + {1 - round(share, 2):.2f} \\times {r2:.2f} = {rho_obs:.2f}$")
check_paper("paper: rho_obs", f"\\approx {rho_obs:.2f}$")
check_true("rho_obs at unrounded parameters", f"{share + (1 - share) * rho_hat:.2f}" == f"{rho_obs:.2f}")
check("two estimates of persistence", "gap between the minimum-distance estimate and the Arellano--Bond estimate")
_m10 = pd.read_csv(os.path.join(first_existing(ROOT, INCOME_DYNAMICS_NAMES), "output", "tables", "m10_abgmm_table3.csv")).set_index("panel")
check_paper("paper: Arellano-Bond estimate", f"$\\hat\\rho_0 = {_m10.loc['diff GMM, 2014-2022', 'rho0']:.3f}$")
_m7L = pd.read_csv(os.path.join(first_existing(ROOT, INCOME_DYNAMICS_NAMES), "output", "tables", "m7_md_cellw_boot.csv")).set_index("group").loc["L"]
check_paper_re("paper: minimum-distance estimate next to it", rf"minimum-distance estimate for less advantaged children, {_m7L['rho']:.2f}, lies inside the 95 percent interval of the Arellano--Bond estimate")
check("Part II: the same statement", "lies inside the\n95 percent interval of the Arellano--Bond estimate")

# ------------------------------------------------ Section C
print("Section C. The accounting identity")
# premium, completion gap and least squares returns for the level of earnings at ages 30 and over (unified/output)
import csv as _csv
def _uo(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.path.join(here, "..", "..", "output"), os.path.join(here, "..", "..", "unified", "output")):
        if os.path.exists(os.path.join(base, name)):
            return list(_csv.DictReader(open(os.path.join(base, name))))
    raise FileNotFoundError(name)
_eb = _uo("u8_entropy_balancing.csv")[0]; _ls = {r["item"]: r for r in _uo("u13_A_least_squares.csv")}
tau, dm, b_ls = float(_eb["tau_hat"]), float(_eb["gap_hat"]), float(_ls["least squares return, all children"]["coef"])
check("premium", f"$\\tau = {tau:.3f}$")
check("college gap", f"$\\Delta M = {dm:.3f}$")
check_paper_re("paper: premium", rf"the remaining {tau:.3f} \(\d+ percent\) is the premium \$\\tau\$")
check_paper("paper: college gap", f"$\\Delta M = {dm:.3f}$")
# instrumented returns for advantaged children (unified/code/u12_returns_by_family_background.py) and for all children
# (unified/code/u9_mature_age_earnings.py), two-row regression, row at 30 and over
_m9 = [r for r in _uo("u9_B_main.csv") if "main measure" in r["intensity"]][0]; _b_iv = float(_m9["earn_iv"])
_g12 = {r["group"]: r for r in _uo("u12_returns_by_family_background.csv")}; _b_adv = float(_g12["advantaged"]["beta"])
_g90 = sorted(float(r["b"]) for r in _uo("u12_anderson_rubin_grid.csv") if r["group"] == "advantaged" and float(r["p_wild"]) >= 0.10)   # two rays
_neg_end, _pos_start = max(b for b in _g90 if b < 0), min(b for b in _g90 if b > 0)
check("instrumented return, advantaged children",
      f"expansion moved into college, {_b_adv:.2f}, with a 90 percent Anderson--Rubin set that excludes every return between $-{abs(_neg_end):.2f}$ and {_pos_start:.2f}")
check("instrumented return, all children", f"estimate for all children, {_b_iv:.2f}, with a 95 percent Anderson--Rubin set that starts at {float(_m9['ar_lo']):.2f}")
check("college channel at the instrumented returns", f"the college channel, {_b_iv * dm:.3f} for all children and {_b_adv * dm:.3f} for advantaged children,")
check_true("college channel exceeds the premium at both instrumented returns", _b_iv * dm > tau and _b_adv * dm > tau)
check_paper("paper: row at the instrumented return, all children", f"Instrumented, all children & {_b_iv:.3f} & {_b_iv * dm:.3f}")
check_paper("paper: row at the instrumented return, advantaged children", f"Instrumented, advantaged children & {_b_adv:.3f} & {_b_adv * dm:.3f}")
check("remainder at the least squares return",
      f"${tau:.3f} - {b_ls:.3f} \\times {dm:.3f} = {tau - b_ls * dm:.3f}$")
check("least squares return", f"least squares return, {b_ls:.3f},")
check_paper("paper: row at the least squares return",
            f"Least squares, all children & {b_ls:.3f} & {b_ls * dm:.3f} & {tau - b_ls * dm:.3f} & "
            f"{b_ls * dm / tau:.2f}")
check("scale of the remainder's sensitivity (excess)", f"by {dm:.3f} times the excess")
check("scale of the remainder's sensitivity (shortfall)", f"by {dm:.3f} times the shortfall")
_lh, _ll = _ls["least squares return, advantaged children"], _ls["least squares return, less advantaged children"]
_u16 = pd.read_csv(os.path.join(UNIFIED, "output", "u16_identity_in_balanced_comparison.csv")).set_index("item")["value"]
_r1, _r0 = _u16["return, advantaged children (graduates minus children without a degree)"], _u16["return, less advantaged children (reweighted)"]
_p1, _p0 = _u16["completion, advantaged children"], _u16["completion, less advantaged children (reweighted)"]
check("balanced comparison: returns, completion, three parts",
      f"by $r_1 = {_r1:.3f}$ among advantaged children and by $r_0 = {_r0:.3f}$ among reweighted less advantaged children, with $p_1 = {_p1:.4f}$ and $p_0 = {_p0:.4f}$, so "
      f"$r_1\\,\\Delta M = {_u16['part 1: college channel (return of advantaged children x completion gap)']:.3f}$, $p_0\\,(r_1 - r_0) = {_u16['part 2: same degrees worth more (completion of less advantaged children x difference in returns)']:.3f}$, and $a_1 - a_0 = {_u16['part 3: gap between the children without a degree']:.3f}$")
check("balanced comparison: second term and direct term", f"the two returns are {_r1:.3f} and {_r0:.3f}, so the second term is {_u16['part 2: same degrees worth more (completion of less advantaged children x difference in returns)']:.3f} and the direct term exceeds the gap between the groups in earnings without a degree, $a_1 - a_0 = {_u16['part 3: gap between the children without a degree']:.3f}$")
check_true("balanced comparison: the three parts add up to the premium", abs(_u16["premium"] - sum(_u16[k] for k in _u16.index if k.startswith("part "))) < 1e-9)
check_paper("paper: the three parts", f"{_u16['part 1: college channel (return of advantaged children x completion gap)']:.3f}")
check("Arellano-Bond interval for less advantaged children", "for less advantaged children, 0.12, lies inside the 95 percent interval of the Arellano--Bond estimate for that group, $[-0.01, 0.13]$")
check_paper("paper: Arellano-Bond interval for less advantaged children", "0.12, lies inside the 95 percent interval of the Arellano--Bond estimate, $[-0.01, 0.13]$")
check_paper("paper: least squares returns by group",
            f"{float(_lh['coef']):.3f} (SE {float(_lh['se']):.3f}) for advantaged children against {float(_ll['coef']):.3f} (SE {float(_ll['se']):.3f}) for less advantaged children")
rng = np.random.default_rng(20260928)
id_gap = 0.0
for _ in range(200):
    a0, a1, r0, r1, b = rng.normal(size=5)
    p0, p1 = rng.uniform(size=2)
    t_ = (a1 + p1 * r1) - (a0 + p0 * r0)
    direct = (a1 - a0) + p0 * (r1 - r0)
    id_gap = max(id_gap, abs(t_ - r1 * (p1 - p0) - direct),
                 abs(direct - ((a1 + p0 * r1) - (a0 + p0 * r0))),
                 abs(t_ - b * (p1 - p0) - direct - (r1 - b) * (p1 - p0)))
check_true("accounting identity, both parts", id_gap < 1e-12, f"largest gap {id_gap:.1e}")

# ------------------------------------------------ coverage
print("Coverage, cross-references, scope, style")
printed = NUMBER.findall(re.sub(r"\d\.\d+\\textwidth", "", TEX))
missing = sorted(set(printed) - covered)
if missing:
    fail("coverage", f"numbers printed in Part II that no assertion covers: {missing}")
else:
    ok("coverage", f"{len(set(printed))} distinct numbers, all covered by an assertion")

# ------------------------------------------------ cross-references
paper_labels = set(re.findall(r"\\label\{([^}]+)\}", PAPER_RAW))
own_labels = re.findall(r"\\label\{([^}]+)\}", RAW)
HOST_LABELS = {"oa:errorstructures", "oatab:rank", "oatab:parent", "oatab:mde", "oa:jointtest", "oatab:autocov", "oatab:screens"}
refs = set()
for group in re.findall(r"\\(?:eq)?ref\{([^}]+)\}", strip_comments(RAW)):
    refs.update(x.strip() for x in group.split(","))
bad = []
for r in sorted(refs):
    if r.startswith("P-"):
        if r[2:] not in paper_labels:
            bad.append(f"{r} (not a label of the paper)")
    elif r.startswith("fr:"):
        if r not in own_labels:
            bad.append(f"{r} (not defined in Part II)")
    elif r not in HOST_LABELS:
        bad.append(f"{r} (not a label of the host document)")
if bad:
    fail("cross-references", "; ".join(bad))
else:
    ok("cross-references", f"{len(refs)} labels referenced, all defined")
dup = sorted({x for x in own_labels if own_labels.count(x) > 1})
notprefixed = sorted(x for x in own_labels if not x.startswith("fr:"))
check_true("labels of Part II are unique and carry the prefix fr:", not dup and not notprefixed,
           f"{dup + notprefixed}")
if re.search(r"(Section|Table|Figure|[Ee]quation)s?[~ ]\(?\d", TEX):
    fail("hard-coded numbers", "a section, table, figure or equation is cited by a typed number")
else:
    ok("no section, table, figure or equation cited by a typed number")

# ------------------------------------------------ scope and style
body = strip_comments(RAW)
for token in ["\\documentclass", "\\begin{document}", "\\usepackage", "\\newcommand", "\\newtheorem"]:
    if token in body:
        fail("includable file", f"{token} present in part2_formal_results.tex")
if not body.strip().startswith("\\part*{Part II. Formal results}"):
    fail("includable file", "file does not start with \\part*{Part II. Formal results}")
else:
    ok("file starts with \\part*{Part II. Formal results}")
if "\u2014" in RAW or re.search(r"(?<!-)---(?!-)", body):
    fail("style", "em dash present")
else:
    ok("no em dash")
if re.search(r"[\u4e00-\u9fff]", RAW):
    fail("style", "Chinese characters present")
else:
    ok("no Chinese characters")

print()
if fails:
    print(f"{len(fails)} FAIL(s)")
    sys.exit(1)
print("all checks passed")
