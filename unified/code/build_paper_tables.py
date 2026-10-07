"""Tables of estimates in the paper, written from the files of estimates.

Each table of the paper that reports estimates with standard errors is written
here as a tabular fragment in paper/tables/en, which the paper \\input{}s.
Coefficients carry significance stars (* 10, ** 5, *** 1 percent) and their
standard errors sit beneath them in parentheses; where the paper's inference
is the wild cluster bootstrap, the bootstrap p-value follows in brackets and
the stars are computed from it.

Fragments and sources:
  tab_md        income_dynamics/output/tables/m7_md_cellw_boot.csv, unified/output/u7_error_structures.csv, unified/output/u14_B_sample_facts.csv
  tab_ab        income_dynamics/output/tables/m10_abgmm_table3.csv
  tab_summary   m7, m10, ws_05b_abb_stochasticEM_surface.csv, unified/output/u10_growth_gap.csv
  tab_cre       unified/output/u8_premium_specifications.csv, unified/output/u8_entropy_balancing.csv
  tab_markers   unified/output/u8_premium_specifications.csv
  tab_fs        unified/output/u9_B_main.csv, u12_returns_by_family_background.csv (rows at 30 and over), u14_C_first_stages_all_children.csv
  tab_desc      unified/output/u14_A_sample_means.csv, u14_B_sample_facts.csv
  tab_mature    unified/output/u9_B_main.csv
  tab_groups    unified/output/u12_returns_by_family_background.csv
Run with --check to compare the fragments on disk with what the files of
estimates imply (exit code 1 on any difference).
"""
import os, sys, csv, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _locate import sibling
U = os.path.abspath(os.path.join(HERE, "..")); OUT = os.path.join(U, "output")
M = sibling("income_dynamics", __file__); P = sibling("college_expansion", __file__)

def rows(path): return list(csv.DictReader(open(path, encoding="utf-8")))
def by(path, key): return {r[key]: r for r in rows(path)}
def fl(x): return float(x)

def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
def num(v, d=3):
    return f"$-{abs(v):.{d}f}$" if round(v, d) < 0 else f"{v:.{d}f}"
def pnorm(z): return math.erfc(abs(z) / math.sqrt(2))
def signed(v, d):
    """A gap with its sign inside math mode, e.g. $+0.020$ or $-0.017$."""
    return f"$+{v:.{d}f}$" if round(v, d) >= 0 else f"$-{abs(v):.{d}f}$"
def half_up(v, d=1):
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(v)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
def est(c, se, p=None, d=3, wild=None, mark=True):
    """Cells of one estimate: coefficient with stars, standard error, optional wild p.
    mark=False prints no stars: used for variances, shares and persistence parameters of the minimum-distance fit, whose
    null of zero lies on the boundary of the parameter space and whose bootstrap distribution is skewed."""
    pv = p if p is not None else pnorm(c / se)
    cells = [num(c, d) + (rl(stars(pv)) if mark else ""), f"({se:.{d}f})"]
    if wild is not None: cells.append(f"[{wild:.3f}]")
    return cells

def rl(st):
    """Significance stars in a zero-width box: the number stays centred in its column, above its standard error."""
    return "\\rlap{" + st + "}" if st else ""

def tabular(spec, head, blocks, pre=""):
    """blocks: list of lists of row-lists; each row-list is a list of lines (each a list of cells)."""
    out = [pre + "\\begin{tabular}{" + spec + "}", "\\toprule"] + head + ["\\midrule"]
    for b, block in enumerate(blocks):
        if b: out.append("\\addlinespace")
        for row in block:
            for line in row: out.append(" & ".join(line) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"

def stacked(label, groups, ncol, extra=None, span=False):
    """One estimate per column group; groups is a list of (col_index, cells) with cells stacked vertically.
    extra: dict col_index -> single cell printed on the first line only. span: the label is a top-aligned
    multirow cell over all lines (for paragraph-type label columns, so a wrapped label stays aligned)."""
    depth = max(len(c) for _, c in groups) if groups else 1
    lines = []
    for k in range(depth):
        head = label if k == 0 else ""
        if span and depth > 1 and k == 0: head = "\\multirow[t]{" + str(depth) + "}{=}{" + label + "}"
        line = [head] + [""] * ncol
        for j, cells in groups:
            if k < len(cells): line[j] = cells[k]
        if k == 0 and extra:
            for j, v in extra.items(): line[j] = v
        lines.append(line)
    return lines

L = {  # labels by language
 "en": dict(pooled="Pooled", adv="Advantaged, $H=1$", less="Less advantaged, $H=0$", fitted="Fitted total variance",
            advc="Advantaged children", lessc="Less advantaged children", adv_s="Advantaged", less_s="Less advantaged",
            diff="Difference GMM", sys="System GMM", recon="2010--2022 (reconstructed)",
            md_head=["& & Permanent & Shock & Transitory & Permanent & \\\\",
                     "& $N$ & variance $\\hat\\sigma^2_\\eta$ & variance $\\hat\\sigma^2_\\varepsilon$ & persistence $\\hat\\rho_v$ & share & Criterion $J$ \\\\"],
            ab_head=["& & Less advantaged & Difference for & Hansen & \\\\",
                     "Panel & Estimator & children $\\hat\\rho_0$ & advantaged $\\hat\\rho_1$ & $J$ [$p$] & Instruments \\\\"],
            sum_head=["Moment & Advantaged & Less advantaged & Gap \\\\"],
            sum_joint="All fifteen variances and autocovariances, joint test of equality",
            sum_rows=["Fitted total variance (minimum distance)", "Transitory persistence, Arellano--Bond",
                      "Nonlinear persistence, average over quantiles", "Differential growth per wave, fixed effects",
                      "\\emph{Split of the common variance (weakly identified for advantaged children)}",
                      "Permanent share (minimum distance)", "Transitory persistence $\\rho_v$ (minimum distance)"],
            cre_head=["& (1) & (2) & (3) \\\\", "Estimator & Correlated & Correlated & Entropy \\\\",
                      "& random effects & random effects & balancing \\\\",
                      "Income control & Family income & Parental labor income & Family income \\\\"],
            cre_rows=["Advantaged, $H$ (parent in CSCO 1--2 at child age 14)", "Coefficient on parental income", "Children"],
            mk_head=["& \\multicolumn{3}{c}{Coefficient on} \\\\", "\\cmidrule(lr){2-4}", "& & Party & State-sector \\\\",
                     "Regressors (with parental income and controls) & Advantaged, $H$ & membership & employment \\\\"],
            mk_panel1="\\emph{Parental party membership when the child was fourteen (N = NPARTY; NMISS children lack a report)}",
            mk_panel2="\\emph{Parental state-sector employment in 2012 (N = NSTATE)}",
            mk_rows=["Occupation indicator only", "Party membership only", "Both", "State-sector employment only", "Both"],
            iv_head=["& & Reduced form & Effect of & AR 95\\% & Least \\\\", "Outcome & Mean & on $Z$ & college & interval & squares \\\\"],
            iv_rows=["College completion (first stage)", "Ever in a professional occupation", "Professional occupation, most waves", "Log earnings, averaged across waves"],
            iv_F="First-stage $F$ statistic",
            norm_head=["& First-stage & \\multicolumn{2}{c}{Professional occupation (ever)} & Log earnings \\\\", "\\cmidrule(lr){3-4}",
                       "Intensity & $F$ & Reduced form & Effect of college & Reduced form \\\\"],
            norm_panels=["\\emph{Shift-share measures}", "\\emph{Predetermined measure}", "\\emph{Placebo (yearbooks)}"],
            int_rows={"predicted provincial growth (main measure)": "Predicted inflow per 2005 college-educated worker (main measure)",
                      "predicted flow per 2005 worker": "Predicted inflow per 2005 worker",
                      "log implied 2010 stock": "Log implied growth of the college-educated stock",
                      "census senior-high stock in 2000": "Census share with senior secondary",
                      "placebo: registrations 1998/1997": "Pre-reform registration growth 1998/1997"},
            mat_head=["& First-stage & Earnings & Instrumented & Occupation \\\\",
                      "Intensity & $F$ & reduced form & return & reduced form \\\\"],
            mat_placebo="Pre-reform registration growth 1998/1997 (placebo)",
            fs_head=["& All & Less advantaged & Advantaged \\\\", "& children & children & children \\\\"],
            fs_panelA="\\emph{A. First-stage coefficient on the instrument, main measure of intensity}",
            fs_rows=["Earnings at 30 and over: the first stage of the returns", "All children in the sample, one observation per child"],
            desc_head=["& \\multicolumn{2}{c}{All children, ages 22 to 55} & \\multicolumn{2}{c}{Children with earnings at 30 and over} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "& Advantaged & Less advantaged & Advantaged & Less advantaged \\\\"],
            desc_rows={"Mean log labor income": "Mean log labor income", "Log family income of the parental household": "$\\pinc$ (log family income)", "Years of schooling": "Years of schooling", "College (16 or more years)": "College (16$+$ years)", "Female": "Female", "Urban (first wave)": "Urban (first wave)", "Age over the waves": "Age over the waves", "Birth year": "Birth year", "Waves with labor income": "Waves with labor income"}, desc_n="Children",
            fs_panelB="\\emph{B. First-stage $F$ statistic by measure of intensity: all children in the sample, one observation per child}",
            mk_alt_head=["& \\multicolumn{3}{c}{Parental party membership} & \\multicolumn{3}{c}{Parental state-sector} \\\\",
                         "& \\multicolumn{3}{c}{when the child was fourteen} & \\multicolumn{3}{c}{employment in 2012} \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                         "& (1) & (2) & (3) & (4) & (5) & (6) \\\\"],
            mk_alt_rows=["Advantaged, $H$", "Party membership", "State-sector employment", "Children"],
            grp_alt_head=["& & First & Earnings & Instrumented & \\multicolumn{2}{c}{Anderson--Rubin set} & Least \\\\", "\\cmidrule(lr){6-7}",
                          "& Children & stage & reduced form & return & 95 percent & 90 percent & squares \\\\"],
            mat2_head=["& \\multicolumn{2}{c}{First-stage $F$} & & & \\\\", "\\cmidrule(lr){2-3}", "& Rows at 30 & One observation & Earnings & Instrumented & Occupation \\\\",
                       "Intensity & and over & per child & reduced form & return & reduced form \\\\"],
            fs2_head=["& All & Less advantaged & Advantaged \\\\", "& children & children & children \\\\"],
            fs2_main="Main measure of intensity: coefficient on the instrument", fs2_other="\\emph{First-stage $F$ statistic under the other measures of intensity}",
            grp_all="All children", grp_alt_rows={"less advantaged": "Less advantaged", "advantaged": "Advantaged"},
            fs_placebo="Pre-reform registration growth 1998/1997 (placebo)",
            grp_head=["& & First & Earnings & Instrumented & \\multicolumn{2}{c}{Anderson--Rubin set} \\\\", "\\cmidrule(lr){6-7}",
                      "Family background & Children & stage & reduced form & return & 95 percent & 90 percent \\\\"],
            grp_rows={"less advantaged": "Less advantaged children", "advantaged": "Advantaged children"}, unbounded="unbounded"),
}
SEP = {"en": ": "}
try:   # the Chinese version of the paper (not part of the public package)
    from _cn_labels import TABLES_CN, SEP_CN; L["cn"] = TABLES_CN; SEP["cn"] = SEP_CN
except ImportError:
    pass
TAB = {lang: os.path.join(U, "paper", "tables", lang) for lang in L}

def build(lang):
    T = L[lang]; frags = {}
    # ---------------------------------------------------------------- tab_md
    m7 = by(os.path.join(M, "output", "tables", "m7_md_cellw_boot.csv"), "group")
    def fitted(g): r = m7[g]; return fl(r["eta2"]) + fl(r["eps2"]) / (1 - fl(r["rho"]) ** 2)
    _u7 = by(os.path.join(OUT, "u7_error_structures.csv"), "sample")   # fit statistic of the permanent--AR(1) model (code/u7_error_structures.py)
    J = {"pooled": fl(_u7["pooled"]["criterion_ar1"]), "H": fl(_u7["advantaged children"]["criterion_ar1"]), "L": fl(_u7["ordinary children"]["criterion_ar1"])}
    FA = {r["item"]: fl(r["value"]) for r in rows(os.path.join(OUT, "u14_B_sample_facts.csv"))}
    nD, nDH = int(FA["three waves: children"]), int(FA["three waves: advantaged children"])   # children with at least three waves (Sections 3 and 4)
    N = {"pooled": f"{nD:,}", "H": f"{nDH:,}", "L": f"{nD - nDH:,}"}
    blk = []
    for g, lab in [("pooled", T["pooled"]), ("H", T["adv"]), ("L", T["less"])]:
        r = m7[g]
        blk.append(stacked(lab, [(2, est(fl(r["eta2"]), fl(r["eta2_se"]), mark=False)), (3, est(fl(r["eps2"]), fl(r["eps2_se"]), mark=False)),
                                 (4, est(fl(r["rho"]), fl(r["rho_se"]), d=2, mark=False)), (5, est(fl(r["sperm"]), fl(r["sperm_se"]), d=2, mark=False))],
                           6, extra={1: N[g], 6: half_up(J[g])}))
    sep = SEP[lang]
    fit_line = [[T["fitted"], "", f"\\multicolumn{{2}}{{c}}{{{T['adv_s']}{sep}{fitted('H'):.3f}}}", f"\\multicolumn{{2}}{{c}}{{{T['less_s']}{sep}{fitted('L'):.3f}}}", ""]]
    frags["tab_md"] = tabular("@{}lcccccc@{}", T["md_head"], [blk, [fit_line]], pre="\\setlength{\\tabcolsep}{4pt}\n")
    # ---------------------------------------------------------------- tab_ab
    m10 = by(os.path.join(M, "output", "tables", "m10_abgmm_table3.csv"), "panel")
    blk = []
    for key, panel, estr in [("diff GMM, 2014-2022", "2014--2022", T["diff"]), ("system GMM, 2014-2022", "2014--2022", T["sys"]),
                             ("diff GMM, 2010-2022 reconstructed", T["recon"], T["diff"])]:
        r = m10[key]
        blk.append(stacked(panel, [(2, est(fl(r["rho0"]), fl(r["rho0_se"]))), (3, est(fl(r["rho1"]), fl(r["rho1_se"]))),
                                   (4, [f"{fl(r['J']):.2f}", f"[{fl(r['pJ']):.3f}]"])], 5, extra={1: estr, 5: str(int(fl(r["n_inst"])))}))
    frags["tab_ab"] = tabular("@{}llcccc@{}", T["ab_head"], [blk], pre="\\setlength{\\tabcolsep}{3pt}\n")
    # ---------------------------------------------------------------- tab_summary
    abb = rows(os.path.join(M, "data", "intermediate", "ws", "ws_05b_abb_stochasticEM_surface.csv"))
    avg = {g: sum(fl(r["rho_local"]) for r in abb if r["group"] == g) / sum(1 for r in abb if r["group"] == g) for g in ("H", "L")}
    g10 = [r for r in rows(os.path.join(OUT, "u10_growth_gap.csv")) if r["panel"].startswith("2014") and r["covariates"] != "no covariates"][0]
    d5 = m10["diff GMM, 2014-2022"]; H, Lr = m7["H"], m7["L"]
    def gapcell(h, l, se_h, se_l, d):
        g = h - l; se = math.hypot(se_h, se_l)
        return [signed(g, d), f"({se:.{d}f})"]          # no stars: see est(mark=False)
    j16 = rows(os.path.join(M, "data", "intermediate", "m16_joint_moment_test.csv"))[0]
    r0 = [[T["sum_joint"], "\\multicolumn{2}{c}{---}", f"$p = {fl(j16['p']):.2f}$"]]
    r1 = [[T["sum_rows"][0], f"{fitted('H'):.3f}", f"{fitted('L'):.3f}", num(fitted('H') - fitted('L'), 3)]]
    ab_gap = fl(d5["rho1"]); ab_se = fl(d5["rho1_se"])
    r2 = stacked(T["sum_rows"][1], [(3, [signed(ab_gap, 3) + stars(pnorm(ab_gap / ab_se)), f"({ab_se:.3f})"])], 3,
                 extra={1: f"{fl(d5['rho0']) + ab_gap:.3f}", 2: f"{fl(d5['rho0']):.3f}"})
    r3 = [[T["sum_rows"][2], f"{avg['H']:.2f}", f"{avg['L']:.2f}", signed(avg['H'] - avg['L'], 2)]]
    gg, gs = fl(g10["gamma"]), fl(g10["se"])
    r4 = stacked(T["sum_rows"][3], [(3, [signed(gg, 3) + stars(fl(g10["p"])), f"({gs:.3f})"])], 3, extra={1: "\\multicolumn{2}{c}{---}", 2: None})
    r4 = [[c for c in line if c is not None] for line in r4]
    panel = [["\\multicolumn{4}{@{}l}{" + T["sum_rows"][4] + "}"]]
    r5 = stacked(T["sum_rows"][5], [(1, est(fl(H["sperm"]), fl(H["sperm_se"]), d=2, mark=False)), (2, est(fl(Lr["sperm"]), fl(Lr["sperm_se"]), d=2, mark=False)),
                                   (3, gapcell(fl(H["sperm"]), fl(Lr["sperm"]), fl(H["sperm_se"]), fl(Lr["sperm_se"]), 2))], 3)
    r6 = stacked(T["sum_rows"][6], [(1, est(fl(H["rho"]), fl(H["rho_se"]), d=2, mark=False)), (2, est(fl(Lr["rho"]), fl(Lr["rho_se"]), d=2, mark=False)),
                                   (3, gapcell(fl(H["rho"]), fl(Lr["rho"]), fl(H["rho_se"]), fl(Lr["rho_se"]), 2))], 3)
    frags["tab_summary"] = tabular("@{}lccc@{}", T["sum_head"], [[r0, r1, r2, r3, r4], [panel, r5, r6]])
    # ---------------------------------------------------------------- tab_cre
    S8 = by(os.path.join(OUT, "u8_premium_specifications.csv"), "specification")
    b, l = S8["baseline: family income of the parental household"], S8["parents' own labor income as the income control"]
    eb = rows(os.path.join(OUT, "u8_entropy_balancing.csv"))[0]
    eb_star = "**" if fl(eb["tau_ci_lo"]) > 0 or fl(eb["tau_ci_hi"]) < 0 else ""
    r1 = stacked(T["cre_rows"][0], [(1, est(fl(b["estimate"]), fl(b["se"]), fl(b["p"]))), (2, est(fl(l["estimate"]), fl(l["se"]), fl(l["p"]))),
                                   (3, [f"{fl(eb['tau_hat']):.3f}{rl(eb_star)}", f"[{fl(eb['tau_ci_lo']):.2f}, {fl(eb['tau_ci_hi']):.2f}]"])], 3, span=True)
    r2 = stacked(T["cre_rows"][1], [(1, est(fl(b["income_coef"]), fl(b["income_se"]), fl(b["income_p"]))),
                                   (2, est(fl(l["income_coef"]), fl(l["income_se"]), fl(l["income_p"])))], 3, extra={3: "---"}, span=True)
    r3 = [[T["cre_rows"][2], f"{int(b['children']):,}", f"{int(l['children']):,}", f"{int(eb['N_treat']) + int(eb['N_ctrl']):,}"]]
    frags["tab_cre"] = tabular("@{}>{\\raggedright\\arraybackslash}p{5.6cm}ccc@{}", T["cre_head"], [[r1, r2, r3]])
    # ---------------------------------------------------------------- tab_markers
    k = dict(p0="occupation indicator before party membership is added (not in Figure 2)", p1="parental party membership added (cross-section)",
             pc="party membership coefficient with occupation and income (not in Figure 2)", pa="parental party membership in place of the occupation indicator (not in Figure 2)",
             s0="parental state-sector employment alone (not in Figure 2)", s1="parental state-sector employment added (cross-section)",
             sc="state-sector employment coefficient with occupation and income (not in Figure 2)",
             so="occupation indicator alone, children with a parent working in 2012 (not in Figure 2)")
    e8 = {n: est(fl(S8[key]["estimate"]), fl(S8[key]["se"]), fl(S8[key]["p"])) for n, key in k.items()}
    n_party, n_state, n_all = int(fl(S8[k["p1"]]["children"])), int(fl(S8[k["s1"]]["children"])), int(fl(b["children"]))
    panel1 = T["mk_panel1"].replace("NPARTY", f"{n_party:,}").replace("NMISS", str(n_all - n_party)); panel2 = T["mk_panel2"].replace("NSTATE", f"{n_state:,}")
    blk1 = [[["\\multicolumn{4}{@{}l}{" + panel1 + "}"]], stacked(T["mk_rows"][0], [(1, e8["p0"])], 3, span=True),
            stacked(T["mk_rows"][1], [(2, e8["pa"])], 3, span=True), stacked(T["mk_rows"][2], [(1, e8["p1"]), (2, e8["pc"])], 3, span=True)]
    blk2 = [[["\\multicolumn{4}{@{}l}{" + panel2 + "}"]], stacked(T["mk_rows"][0], [(1, e8["so"])], 3, span=True), stacked(T["mk_rows"][3], [(3, e8["s0"])], 3, span=True),
            stacked(T["mk_rows"][4], [(1, e8["s1"]), (3, e8["sc"])], 3, span=True)]
    frags["tab_markers"] = tabular("@{}>{\\raggedright\\arraybackslash}p{4.6cm}ccc@{}", T["mk_head"], [blk1, blk2], pre="\\setlength{\\tabcolsep}{5pt}\n")
    # the same eight estimates with one regression per column (the layout of a regression table)
    A = lambda lab, cols: stacked(lab, cols, 6)
    alt = [[A(T["mk_alt_rows"][0], [(1, e8["p0"]), (3, e8["p1"]), (4, e8["so"]), (6, e8["s1"])]),
            A(T["mk_alt_rows"][1], [(2, e8["pa"]), (3, e8["pc"])]), A(T["mk_alt_rows"][2], [(5, e8["s0"]), (6, e8["sc"])])],
           [[[T["mk_alt_rows"][3]] + [f"{n_party:,}"] * 3 + [f"{n_state:,}"] * 3]]]
    frags["tab_markers_by_regression"] = tabular("@{}lcccccc@{}", T["mk_alt_head"], alt, pre="\\setlength{\\tabcolsep}{6pt}\n")
    # ---------------------------------------------------------------- tab_fs
    C14 = by(os.path.join(OUT, "u14_C_first_stages_all_children.csv"), "row")   # child level, all children in the sample
    fs = C14["all children: predicted provincial growth (main measure)"]
    B9 = by(os.path.join(OUT, "u9_B_main.csv"), "intensity"); m9 = B9["predicted provincial growth (main measure)"]
    G12 = by(os.path.join(OUT, "u12_returns_by_family_background.csv"), "group")   # rows at 30 and over of the two-row design
    def fscell(c, se, Fv, p): return [num(c) + rl(stars(p)), f"({se:.3f})", f"$F$ = {Fv:.1f}"]
    def grow(label, allc, key):
        if key is None:   # all children, child level
            l, h = C14["less advantaged children, instrument entering once for each group"], C14["advantaged children, instrument entering once for each group"]
        else:
            l, h = G12["less advantaged"], G12["advantaged"]
        return stacked(label, [(1, allc), (2, fscell(fl(l["fs"]), fl(l["fs_se"]), fl(l["F"]), fl(l["fs_p_wild"]))),
                               (3, fscell(fl(h["fs"]), fl(h["fs_se"]), fl(h["F"]), fl(h["fs_p_wild"])))], 3, span=True)
    pA = [[["\\multicolumn{4}{@{}l}{" + T["fs_panelA"] + "}"]],
          grow(T["fs_rows"][0], fscell(fl(m9["fs"]), fl(m9["fs_se"]), fl(m9["F"]), fl(m9["fs_p_wild"])), "rows"),
          grow(T["fs_rows"][1], fscell(fl(fs["fs"]), fl(fs["fs_se"]), fl(fs["F"]), fl(fs["fs_p_wild"])), None)]
    pB = [[["\\multicolumn{4}{@{}l}{" + T["fs_panelB"] + "}"]]]
    for key in ("predicted provincial growth (main measure)", "predicted flow per 2005 worker", "log implied 2010 stock", "census senior-high stock in 2000"):
        pB.append([[T["int_rows"][key], f"{fl(C14['all children: ' + key]['F']):.1f}", "", ""]])
    pB.append([[T["fs_placebo"], f"{fl(C14['all children: placebo, registrations 1998/1997']['F']):.1f}", "", ""]])
    # each panel carries its own column headings, so that the headings of panel A do not appear to label panel B
    _ln = lambda block: [" & ".join(line) + " \\\\" for row in block for line in row]
    frags["tab_fs"] = "\n".join(["\\setlength{\\tabcolsep}{6pt}\n\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{6.2cm}ccc@{}}", "\\toprule"] + _ln(pA[:1]) + T["fs_head"] + ["\\midrule"] + _ln(pA[1:])
                                 + ["\\midrule"] + _ln(pB[:1]) + ["\\midrule"] + _ln(pB[1:]) + ["\\bottomrule", "\\end{tabular}"]) + "\n"
    # ---------------------------------------------------------------- Anderson-Rubin sets (used by tab_mature and tab_groups)
    GRID12 = rows(os.path.join(OUT, "u12_anderson_rubin_grid.csv")) + rows(os.path.join(OUT, "u9_anderson_rubin_grid.csv"))
    LIMIT = {"all children": fl(m9["fs_p_wild"]), "less advantaged": fl(G12["less advantaged"]["fs_own_regressor_p_wild"]), "advantaged": fl(G12["advantaged"]["fs_own_regressor_p_wild"])}
    def arsegs(grp, level):
        """The grid points not rejected at the level, as a list of [lo, hi, open_below, open_above]. As b grows without bound the test
        becomes the test of a zero first stage (p-value LIMIT[grp]); a segment that reaches the edge of the grid is unbounded on that
        side if LIMIT[grp] >= level, and the function stops if a segment reaches the edge when the limit says the set is bounded
        (the grid is then too short)."""
        pts = sorted((fl(r["b"]), fl(r["p_wild"])) for r in GRID12 if r["group"] == grp); step = round(pts[1][0] - pts[0][0], 4)
        acc = [b for b, pv in pts if pv >= level]; segs = []
        for b in acc:
            if segs and abs(b - segs[-1][1] - step) < 1e-6: segs[-1][1] = b
            else: segs.append([b, b])
        glo, ghi = pts[0][0], pts[-1][0]; out = []
        for lo, hi in segs:
            ob, oa = lo <= glo, hi >= ghi
            assert not (ob or oa) or LIMIT[grp] >= level, (grp, level, lo, hi, "the set reaches the edge of the grid although the first stage rejects zero at this level")
            out.append([lo, hi, ob, oa])
        return out
    def arset(grp, level):
        parts = [("(-\\infty" if ob else f"[{lo:.2f}") + ", " + ("\\infty)" if oa else f"{hi:.2f}]") for lo, hi, ob, oa in arsegs(grp, level)]
        return "$" + " \\cup ".join(parts) + "$"
    # ---------------------------------------------------------------- tab_mature
    def matrow(key):
        r = B9[key]
        return stacked(T["int_rows"][key], [(2, est(fl(r["earn"]), fl(r["earn_se"]), fl(r["earn_p_wild"]), wild=fl(r["earn_p_wild"]))),
                                           (4, est(fl(r["occ_maj"]), fl(r["occ_maj_se"]), fl(r["occ_maj_p_wild"]), wild=fl(r["occ_maj_p_wild"])))], 4,
                       extra={1: f"{fl(r['F']):.1f}", 3: f"{fl(r['earn_iv']):.2f}"}, span=True)
    def matrow_main(key):
        """The main measure: the 95 percent Anderson-Rubin set of the return is printed under it (read from the grid of p-values of u9, block G)."""
        out = matrow(key); out[1][3] = arset("all children", 0.05)
        return out
    plc = stacked(T["mat_placebo"], [(2, est(fl(m9["placebo_earn"]), fl(m9["placebo_earn_se"]), fl(m9["placebo_earn_p_wild"]), wild=fl(m9["placebo_earn_p_wild"])))], 4,
                  extra={1: f"{fl(m9['placebo_F']):.1f}", 3: "---", 4: "---"}, span=True)
    frags["tab_mature"] = tabular("@{}>{\\raggedright\\arraybackslash}p{4.3cm}cccc@{}", T["mat_head"],
                                  [[matrow_main("predicted provincial growth (main measure)")] + [matrow(k) for k in ("predicted flow per 2005 worker", "log implied 2010 stock", "census senior-high stock in 2000")] + [plc]],
                                  pre="\\setlength{\\tabcolsep}{4pt}\n")
    # Two ways of removing what tab_fs prints twice (its first row is the first-stage column of the table of returns).
    # (i) tab_mature_with_child_level_F: tab_mature with one more column, the first-stage F with one observation per child, so that tab_fs can go.
    CK = {"predicted provincial growth (main measure)": "all children: predicted provincial growth (main measure)", "predicted flow per 2005 worker": "all children: predicted flow per 2005 worker",
          "log implied 2010 stock": "all children: log implied 2010 stock", "census senior-high stock in 2000": "all children: census senior-high stock in 2000"}
    def widen(block, Fchild):
        out = [[line[0], line[1], ""] + line[2:] for line in block]; out[0][2] = f"{Fchild:.1f}"; return out
    rows2 = [widen(matrow_main(k) if "main" in k else matrow(k), fl(C14[CK[k]]["F"])) for k in CK] + [widen(plc, fl(C14["all children: placebo, registrations 1998/1997"]["F"]))]
    frags["tab_mature_with_child_level_F"] = tabular("@{}>{\\raggedright\\arraybackslash}p{4.0cm}ccccc@{}", T["mat2_head"], [[r_] for r_ in rows2], pre="\\setlength{\\tabcolsep}{4pt}\n")
    # (ii) tab_fs_short: tab_fs without the row that the table of returns repeats; all entries are first stages with one observation per child.
    short = [grow(T["fs2_main"], fscell(fl(fs["fs"]), fl(fs["fs_se"]), fl(fs["F"]), fl(fs["fs_p_wild"])), None)]
    other = [[["\\multicolumn{4}{@{}l}{" + T["fs2_other"] + "}"]]] + [[[T["int_rows"][k], f"{fl(C14[CK[k]]['F']):.1f}", "", ""]] for k in list(CK)[1:]] + [[[T["fs_placebo"], f"{fl(C14['all children: placebo, registrations 1998/1997']['F']):.1f}", "", ""]]]
    frags["tab_fs_short"] = "\n".join(["\\setlength{\\tabcolsep}{6pt}\n\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{6.2cm}ccc@{}}", "\\toprule"] + T["fs2_head"] + ["\\midrule"] + _ln(short)
                                      + ["\\midrule"] + _ln(other) + ["\\bottomrule", "\\end{tabular}"]) + "\n"
    # ---------------------------------------------------------------- tab_groups
    blk = []
    for g_ in ("less advantaged", "advantaged"):
        r = G12[g_]
        blk.append(stacked(T["grp_rows"][g_], [(2, fscell(fl(r["fs"]), fl(r["fs_se"]), fl(r["F"]), fl(r["fs_p_wild"]))),
                                               (3, est(fl(r["rf"]), fl(r["rf_se"]), fl(r["rf_p_wild"]), wild=fl(r["rf_p_wild"])))], 6,
                           extra={1: f"{int(fl(r['children'])):,}", 4: f"{fl(r['beta']):.2f}", 5: arset(g_, 0.05), 6: arset(g_, 0.10)}))
    # a set made of two rays is printed on two lines of its cell, so that the column stays narrow
    for row in blk:
        for c_ in (5, 6):
            if "\\cup" in row[0][c_]:
                first, second = row[0][c_].strip("$").split(" \\cup ")
                row[0][c_] = "$" + first + "$"; row[1][c_] = "$\\cup\\ " + second + "$"
    frags["tab_groups"] = tabular("@{}lcccccc@{}", T["grp_head"], [blk], pre="\\setlength{\\tabcolsep}{5pt}\n")
    # the same table with a first row for all children (the regression of tab_mature, main measure) and a last column of least squares returns
    assert abs(arsegs("all children", 0.05)[0][0] - fl(m9["ar_lo"])) < 1e-9, "lower end of the 95 percent set of all children differs between the grid file and u9_B_main.csv"
    allrow = stacked(T["grp_all"], [(2, fscell(fl(m9["fs"]), fl(m9["fs_se"]), fl(m9["F"]), fl(m9["fs_p_wild"]))),
                                    (3, est(fl(m9["earn"]), fl(m9["earn_se"]), fl(m9["earn_p_wild"]), wild=fl(m9["earn_p_wild"]))),
                                    (7, [f"{fl(m9['ols']):.2f}", f"({fl(m9['ols_se']):.3f})"])], 7,
                     extra={1: f"{int(fl(m9['rows_old'])):,}", 4: f"{fl(m9['earn_iv']):.2f}", 5: arset("all children", 0.05), 6: arset("all children", 0.10)})
    blk2 = [allrow]
    for g_, row in zip(("less advantaged", "advantaged"), blk):
        r = G12[g_]; new = [line + [""] for line in row]; new[0][0] = T["grp_alt_rows"][g_]; new[0][7] = f"{fl(r['ls']):.2f}"; new[1][7] = f"({fl(r['ls_se']):.3f})"; blk2.append(new)
    frags["tab_groups_with_all_children"] = tabular("@{}lccccccc@{}", T["grp_alt_head"], [[b_] for b_ in blk2], pre="\\setlength{\\tabcolsep}{4pt}\n")
    # ---------------------------------------------------------------- tab_desc: all children and the children with earnings at 30 and over, by family background
    A14 = {(r["sample"], r["variable"]): r for r in rows(os.path.join(OUT, "u14_A_sample_means.csv"))}
    dec = {"Age over the waves": 1, "Birth year": 1}
    def cell(smp, var, grp):
        r = A14[(smp, var)]; d = dec.get(var, 2)
        return f"{fl(r[grp + '_mean']):.{d}f} ({fl(r[grp + '_sd']):.{d}f})"
    body = [[[T["desc_rows"][v]] + [cell(smp, v, grp) for smp in ("all children", "level at 30 and over") for grp in ("advantaged", "less_advantaged")]] for v in T["desc_rows"]]
    nA_, nAH = int(FA["all children: children"]), int(FA["all children: advantaged children"])
    nL_, nLH = int(FA["level at 30 and over: children"]), int(FA["level at 30 and over: advantaged children"])
    nrow = [[[T["desc_n"], f"{nAH:,}", f"{nA_ - nAH:,}", f"{nLH:,}", f"{nL_ - nLH:,}"]]]
    frags["tab_desc"] = tabular("@{}lcccc@{}", T["desc_head"], [body, nrow], pre="\\setlength{\\tabcolsep}{5pt}\n")
    return frags

# Layouts kept for comparison and not read by the paper: the competing markers with one regression per row, and the returns by
# family background without the row for all children (the paper reads tab_markers_by_regression and
# tab_groups_with_all_children).
NOT_IN_PAPER = {"tab_markers", "tab_groups", "tab_fs", "tab_fs_short", "tab_mature"}   # the last three are layouts of a first-stage table that the paper does not print

if __name__ == "__main__":
    check = "--check" in sys.argv; bad = 0
    for lang in L:
        if check and not os.path.isdir(TAB[lang]):
            continue                      # a language whose fragments are not kept here is not checked
        os.makedirs(TAB[lang], exist_ok=True)
        for name, text in build(lang).items():
            path = os.path.join(TAB[lang], name + ".tex")
            if check:
                if name in NOT_IN_PAPER and not os.path.exists(path):
                    continue              # an earlier layout that the paper no longer reads; the public package does not carry its file
                old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
                if old != text: bad += 1; print("DIFFERS:", path)
            else:
                open(path, "w", encoding="utf-8").write(text)
    if check:
        print("all table fragments match the files of estimates" if not bad else f"{bad} fragment(s) differ"); sys.exit(1 if bad else 0)
    print("wrote", len(build("en")) * 2, "fragments to", os.path.dirname(TAB["en"]))
