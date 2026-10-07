"""Write the child-wave panel for the income-dynamics pipeline (Sections 3 and 4 of the paper).

Sections 3 and 4 use the children of the paper's sample (sample_mature_age.py: ages 22 to 55, at least two waves of
positive labor income in 2014 to 2022) who have at least three waves, because the dynamic panel estimators need
three. This script saves their child-wave rows in the layout that income_dynamics/code/03_step0_residualize.py reads
(data/intermediate/panel_analysis.pkl of the income-dynamics pipeline).

The indicator built from the parents' current occupations (HighParentOcc) is missing for some of these children;
step 03 uses it only for an auxiliary residual that the paper does not report, and requires it to be present, so it
is filled with the retrospective indicator where missing. The untouched variable stays in the child-wave frame of
sample_mature_age.py.
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _locate import sibling
from sample_mature_age import child_waves, DYNAMICS
M = sibling("income_dynamics", __file__)
w = child_waves(*DYNAMICS, level_from=None)
w["HighParentOcc"] = w["HighParentOcc"].fillna(w["HighParentOcc14"])
cols = ["pid", "year", "income", "log_income", "age", "female", "urban", "provcd", "pid_father", "pid_mother", "occ_father", "occ_mother",
        "fd_father", "fd_mother", "HighParentOcc", "HighParentOcc14", "has_qv14", "ParentInc_bar", "eduy"]
save = w[cols].copy()
for c in cols:
    if c not in ("pid", "year"):
        save[c] = save[c].astype(float)
out = os.path.join(M, "data", "intermediate", "panel_analysis.pkl"); save.to_pickle(out)
print(f"children with at least three waves: {save.pid.nunique():,} children, {len(save):,} child-waves, ages {int(save.age.min())} to {int(save.age.max())}; wrote {out}")
