"""The variance of log personal income in each CFPS wave, before and after the change of income concept in 2014
(Section 2.1 of the paper).

Respondents: every adult respondent of the wave aged 22 to 55 with positive personal income, from the long panel
that income_dynamics/code/02_sample_and_parents.py caches (2010: income; 2012: income_adj; 2014 to 2018: income,
jobs only; 2020 and 2022: emp_income). No trimming. A second block gives the same variances for the children of the
paper's sample. Output: output/u17_income_variance_by_wave.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from _locate import sibling
from sample_mature_age import build_sample
M = sibling("income_dynamics", __file__)
d = pd.read_pickle(os.path.join(M, "data", "intermediate", "_panel_long_cache.pkl"))
d = d[(d.income > 0) & d.age.between(22, 55)].copy(); d["log_income"] = np.log(d.income)
c = build_sample(level_from=None)
rows = []
for lab, f in [("all respondents aged 22 to 55 with positive income", d), ("children of the paper's sample, waves at ages 22 to 55 with positive income", d[d.pid.isin(c.pid)])]:
    for year, g in f.groupby("year"):
        rows.append(dict(respondents=lab, wave=int(year), n=len(g), variance_of_log_income=float(g.log_income.var())))
out = pd.DataFrame(rows); out.to_csv(os.path.join(ROOT, "output", "u17_income_variance_by_wave.csv"), index=False); print(out.round(3).to_string(index=False))
