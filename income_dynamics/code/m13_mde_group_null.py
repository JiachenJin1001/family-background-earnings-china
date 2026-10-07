"""Minimum detectable differences between advantaged and less advantaged children.

For each parameter of the earnings process (permanent effect plus AR(1)
transitory component), the smallest true difference between advantaged
children (group H) and less advantaged children (group L) that a two-sided 5 percent
test detects with 80 percent power is 2.8 times the standard error of the
estimated difference (Bloom 1995). The standard error of a difference combines
the two group-specific bootstrap standard errors written by
m7_md_cellweighted_boot.py; the two groups are independent samples, so the
variances add. For the Arellano-Bond interaction rho_1 the two-step standard
error from m10_abgmm_table3.py is used.

Input: output/tables/m7_md_cellw_boot.csv, output/tables/m10_abgmm_table3.csv
Output: output/tables/m13_mde.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
m7 = pd.read_csv(os.path.join(ROOT, "output/tables/m7_md_cellw_boot.csv"))
g = m7.set_index("group")

rows = []
for par, lab in [("eta2", "permanent variance"),
                 ("eps2", "transitory variance"),
                 ("rho", "transitory persistence"),
                 ("sperm", "permanent share")]:
    se_gap = np.sqrt(g.loc["H", f"{par}_se"]**2 + g.loc["L", f"{par}_se"]**2)
    rows.append(dict(param=lab, gap=g.loc["H", par] - g.loc["L", par],
                     se_gap=se_gap, mde80=2.8 * se_gap))
    print(f"{lab:24s} gap {g.loc['H',par]-g.loc['L',par]:+.3f}  "
          f"SE {se_gap:.3f}  MDE(80%) {2.8*se_gap:.3f}")

# Arellano-Bond rho_1: two-step standard error of the five-wave difference GMM
m10 = pd.read_csv(os.path.join(ROOT, "output/tables/m10_abgmm_table3.csv"))
se1 = float(m10.set_index("panel").loc["diff GMM, 2014-2022", "rho1_se"])
rows.append(dict(param="AB persistence interaction", gap=np.nan,
                 se_gap=se1, mde80=2.8 * se1))
print(f"{'AB rho_1':24s} SE {se1:.3f}  MDE(80%) {2.8*se1:.3f}")

pd.DataFrame(rows).to_csv(os.path.join(ROOT, "output/tables/m13_mde.csv"),
                          index=False)
print("wrote m13_mde.csv")
