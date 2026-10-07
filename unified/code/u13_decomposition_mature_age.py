"""The decomposition of the premium (Section 7 of the paper) and facts about the design.

Reads the premium and the gap in college completion from the entropy balancing of u8_premium_specifications.py
(children with at least two waves at 30 and over), the instrumented return from u9_mature_age_earnings.py and the
returns by family background from u12_returns_by_family_background.py (two-row design, coefficients of the rows at
30 and over), and computes:

  A  least squares returns to college at 30 and over from the two-row rows (all children; by family background from
     u12), the least squares return below 30, and the coefficient on family background before and after college
     completion enters (one row per child with a level at 30 and over; family background, parental income, gender,
     urban residence, province effects, one indicator per birth year; standard errors clustered on province)
  B  the accounting identity, premium = return x completion gap + direct, at each return; the return at which college
     accounts for half and for all of the premium; the three parts at the least squares returns; the split of the raw
     earnings gap
  C  on all children of the sample: college completion before and after the reform by family background (rates, gap
     in percentage points, ratio, share of less advantaged children among graduates and among the added graduates);
     the reduced form of urban residence on the instrument (wild cluster bootstrap, 9,999 draws); the share of the
     instrument's variance, net of province effects, that lies within birth cohorts; the effect on completion at 30
     and over of moving from the 25th to the 75th percentile of provincial intensity for a cohort born 1989 or later
  E  how many degrees the expansion added to each group: completion on the instrument by family background among all
     children, the difference, and the difference in added degrees it implies
With E_WEIGHTS=process the returns are those of the weighted run and files A to C carry the suffix _weighted; the
premium and the completion gap are the same in both runs.
Outputs: output/u13_A_least_squares.csv, u13_B_identity.csv, u13_C_design_facts.csv, u13_E_added_degrees.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..")); OUT = os.path.join(ROOT, "output")
sys.path.insert(0, HERE)
REPS = int(os.environ.get("U13_REPS", "9999")); os.environ["U9_REPS"] = str(REPS)
src = open(os.path.join(HERE, "u9_mature_age_earnings.py"), encoding="utf-8").read(); src = src[:src.index("# ------------------------------------------------------------------ A. the sample")]
u9 = {"__file__": os.path.join(HERE, "u9_mature_age_earnings.py"), "__name__": "u9"}; exec(compile(src, "u9", "exec"), u9)
Design, X_fe, cl, dummies = u9["Design"], u9["X_fe"], u9["cl"], u9["dummies"]
from sample_mature_age import build_sample
c = build_sample().reset_index(drop=True); H = c.HP14.values.astype(float); g = c.pc.values      # children with a level of earnings at 30 and over
SFX = u9["SFX"]; cfull = u9["c"]; s2 = u9["s"]; T2 = u9["TB"](s2, s2.Z.values, np.random.default_rng(1)); LS_ALL = T2.ols(s2.y.values, s2.college.values)   # all children; two-row rows
g12_ = pd.read_csv(os.path.join(OUT, f"u12_returns_by_family_background{SFX}.csv")).set_index("group"); LS_L, LS_H = float(g12_.loc["less advantaged", "ls"]), float(g12_.loc["advantaged", "ls"])
rng = np.random.default_rng(20261013)

# ---------------------------------------------------------------- A. least squares
X = X_fe(c)                                                   # family background, parental income, gender, urban, province, birth year
pooled = cl(c.log_y, np.column_stack([c.college.values, X]), g)
Xg = X_fe(c, demo=["HP14", "ParentInc", "female", "urban"])
bygrp = cl(c.log_y, np.column_stack([c.college.values * (1 - H), c.college.values * H, Xg]), g)
before = cl(c.log_y, X, g); after = cl(c.log_y, np.column_stack([c.college.values, X]), g)
A = [dict(item="least squares return, all children", coef=LS_ALL[0], se=LS_ALL[1]),
     dict(item="least squares return, less advantaged children", coef=LS_L, se=float(g12_.loc["less advantaged", "ls_se"])),
     dict(item="least squares return, advantaged children", coef=LS_H, se=float(g12_.loc["advantaged", "ls_se"])),
     dict(item="least squares return below 30, all children", coef=LS_ALL[2], se=LS_ALL[3]),
     dict(item="coefficient on family background before college completion enters", coef=before.params[0], se=before.bse[0]),
     dict(item="coefficient on family background after college completion enters", coef=after.params[1], se=after.bse[1])]
pd.DataFrame(A).to_csv(os.path.join(OUT, "u13_A_least_squares.csv".replace(".csv", SFX + ".csv")), index=False); print(pd.DataFrame(A).round(4).to_string(index=False))

# ---------------------------------------------------------------- B. the identity
eb = pd.read_csv(os.path.join(OUT, "u8_entropy_balancing.csv")).iloc[0]; tau, gap = float(eb.tau_hat), float(eb.gap_hat)
main = pd.read_csv(os.path.join(OUT, f"u9_B_main{SFX}.csv")); iv_all = float(main[main.intensity.str.contains("main measure")].earn_iv.iloc[0])
g12 = g12_
B = []
for lab, b in [("return at which college accounts for half of the premium", 0.5 * tau / gap),
               ("least squares return, all children", LS_ALL[0]),
               ("least squares return, advantaged children", LS_H),
               ("return at which college accounts for all of the premium", tau / gap),
               ("instrumented return, less advantaged children", float(g12.loc["less advantaged", "beta"])),
               ("instrumented return, all children", iv_all),
               ("instrumented return, advantaged children", float(g12.loc["advantaged", "beta"])),
               ("lower end of the 90 percent Anderson-Rubin set, advantaged children", float(g12.loc["advantaged", "ar90_lo"])),
               ("lower end of the 95 percent Anderson-Rubin set, all children", float(main[main.intensity.str.contains("main measure")].ar_lo.iloc[0]))]:
    B.append(dict(row=lab, return_to_college=b, college_channel=b * gap, direct=tau - b * gap, college_share=b * gap / tau))
raw = c[c.HP14 == 1].log_y.mean() - c[c.HP14 == 0].log_y.mean(); ls = LS_ALL[0]
B += [dict(row="premium (entropy balancing)", return_to_college=np.nan, college_channel=tau), dict(row="completion gap (entropy balancing)", college_channel=gap),
      dict(row="completion rate, advantaged children", college_channel=float(eb.completion_advantaged)),
      dict(row="completion rate, less advantaged children reweighted", college_channel=float(eb.completion_less_advantaged_reweighted)),
      dict(row="direct effect: completion rate of less advantaged children times the difference in least squares returns", college_channel=float(eb.completion_less_advantaged_reweighted) * (LS_H - LS_L)),
      dict(row="three parts at the least squares returns: more degrees (completion gap times the return of advantaged children)", college_channel=gap * LS_H),
      dict(row="three parts at the least squares returns: gap between children without a degree", college_channel=tau - gap * LS_H - float(eb.completion_less_advantaged_reweighted) * (LS_H - LS_L)),
      dict(row="three parts at the least squares returns: share of the first two parts", college_channel=(gap * LS_H + float(eb.completion_less_advantaged_reweighted) * (LS_H - LS_L)) / tau),
      dict(row="raw gap in mean log earnings", college_channel=raw), dict(row="raw gap: parental income and the demographic and provincial controls", college_channel=raw - tau),
      dict(row="raw gap: college gap at the least squares return", college_channel=ls * gap), dict(row="raw gap: remainder", college_channel=tau - ls * gap)]
pd.DataFrame(B).to_csv(os.path.join(OUT, "u13_B_identity.csv".replace(".csv", SFX + ".csv")), index=False); print(pd.DataFrame(B).round(4).to_string(index=False))

# ---------------------------------------------------------------- C. design facts
pre, post = cfull[cfull.by <= 1980], cfull[cfull.by >= 1981]
rate = lambda d, h: d[d.HP14 == h].college.mean()
C = []
for lab, d in [("born 1980 or earlier", pre), ("born 1981 or later", post)]:
    C += [dict(item=f"college completion, advantaged children, {lab}", value=rate(d, 1)), dict(item=f"college completion, less advantaged children, {lab}", value=rate(d, 0)),
          dict(item=f"completion gap in percentage points, {lab}", value=100 * (rate(d, 1) - rate(d, 0))), dict(item=f"completion ratio, {lab}", value=rate(d, 1) / rate(d, 0)),
          dict(item=f"less advantaged share of graduates, {lab}", value=d[d.HP14 == 0].college.sum() / d.college.sum())]
add_l = post[post.HP14 == 0].college.sum() / len(post) - pre[pre.HP14 == 0].college.sum() / len(pre); add_h = post[post.HP14 == 1].college.sum() / len(post) - pre[pre.HP14 == 1].college.sum() / len(pre)
C += [dict(item="less advantaged share of the added graduates", value=add_l / (add_l + add_h)), dict(item="less advantaged share of children", value=(cfull.HP14 == 0).mean()),
      dict(item="rise in completion, advantaged children, percentage points", value=100 * (rate(post, 1) - rate(pre, 1))),
      dict(item="rise in completion, less advantaged children, percentage points", value=100 * (rate(post, 0) - rate(pre, 0)))]
Du = Design(cfull, cfull.Z.values, X_fe(cfull, demo=["HP14", "ParentInc", "female"]), rng); r = Du.rf(cfull.urban.values)
C.append(dict(item="reduced form of urban residence on the instrument", value=r["coef"], se=r["se"], p_wild=r["p_wild"]))
Zp = sm.OLS(cfull.Z.values, np.column_stack([dummies(cfull.pc, "p").values, np.ones(len(cfull))])).fit().resid
Zpb = sm.OLS(Zp, np.column_stack([dummies(cfull.by, "b").values, np.ones(len(cfull))])).fit().resid
C.append(dict(item="share of the instrument's variance, net of province effects, within birth cohorts", value=Zpb.var() / Zp.var()))
fs = float(main[main.intensity.str.contains("main measure")].fs.iloc[0]); q25, q75 = c.pro_predict_growth.quantile([0.25, 0.75]); ex = c[c.by >= 1989].exposure_ratio.mean()   # children with a level at 30 and over, the rows of the first stage
C += [dict(item="difference in the instrument, 25th to 75th percentile of intensity, born 1989 or later", value=(q75 - q25) * ex),
      dict(item="effect on completion of that difference", value=(q75 - q25) * ex * fs)]
n_adv = c[c.HP14 == 1].groupby("pc").size()   # advantaged children with a level at 30 and over, per province
C += [dict(item="provinces", value=c.pc.nunique()), dict(item="provinces with fewer than twenty advantaged children", value=int((n_adv.reindex(sorted(c.pc.unique())).fillna(0) < 20).sum()))]
pd.DataFrame(C).to_csv(os.path.join(OUT, "u13_C_design_facts.csv".replace(".csv", SFX + ".csv")), index=False); print(pd.DataFrame(C).round(4).to_string(index=False))

# ---------------------------------------------------------------- E. degrees the expansion added to each group (all children; main run only)
if SFX == "":
    Hf = cfull.HP14.values.astype(float); Xf_ = pd.concat([cfull[["ParentInc", "female", "urban", "HP14"]].astype(float), dummies(cfull.pc, "p"), dummies(cfull.by, "b"), dummies(cfull.by, "hb").multiply(cfull.HP14, axis=0)], axis=1)
    Xf_ = Xf_.loc[:, Xf_.std() > 0].assign(const=1.0).values; rr = cl(cfull.college.values, np.column_stack([cfull.Z.values * (1 - Hf), cfull.Z.values * Hf, Xf_]), cfull.pc.values); Vv = rr.cov_params()[:2, :2]
    dz = cfull[cfull.by >= 1981].Z.mean() - cfull[cfull.by <= 1980].Z.mean()
    E_ = [dict(item="completion on the instrument, less advantaged children", value=rr.params[0], se=rr.bse[0]), dict(item="completion on the instrument, advantaged children", value=rr.params[1], se=rr.bse[1]),
          dict(item="difference (advantaged minus less advantaged)", value=rr.params[1] - rr.params[0], se=float(np.sqrt(Vv[0, 0] + Vv[1, 1] - 2 * Vv[0, 1]))),
          dict(item="rise of the instrument from cohorts born 1980 or earlier to cohorts born 1981 or later", value=dz), dict(item="implied difference in the degrees the expansion added", value=(rr.params[1] - rr.params[0]) * dz)]
    pd.DataFrame(E_).to_csv(os.path.join(OUT, "u13_E_added_degrees.csv"), index=False); print(pd.DataFrame(E_).round(4).to_string(index=False))
