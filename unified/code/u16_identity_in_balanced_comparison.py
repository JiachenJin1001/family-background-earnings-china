"""The accounting identity of Section 7 inside the entropy-balanced comparison of Section 5.3.

Children: the 2,774 children with a level of earnings. Comparison: advantaged children against less advantaged
children reweighted by entropy balancing to the parental income and demographics of advantaged children (the weights
of u8_premium_specifications.py, same covariates and algorithm). In this comparison every term of the identity is a
mean that is measured directly, so the identity holds exactly:

    premium = (return of advantaged children) x (completion gap)                    the college channel
            + (completion of less advantaged children) x (difference in returns)    the same degrees worth more
            + (gap in mean earnings between the children without a degree)

where the return of a group is the difference in mean log earnings between its graduates and its children without a
four-year degree. Output: output/u16_identity_in_balanced_comparison.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
# the sample and the helper functions of u8_premium_specifications.py, up to the construction of the children's frame
src = open(os.path.join(HERE, "u8_premium_specifications.py"), encoding="utf-8").read()
head = src[:src.index("kid = build_sample()") + len("kid = build_sample()")]
u8 = {"__file__": os.path.join(HERE, "u8_premium_specifications.py"), "__name__": "u8"}; exec(compile(head, "u8", "exec"), u8)
kid, P = u8["kid"], u8["P"]; sys.path.insert(0, os.path.join(P, "code"))
from ws_07b_entropy_balance import entropy_balance
k = kid.copy(); k["cohort"] = (k["birth_year"] // 5) * 5
top = k["provcd"].value_counts().head(15).index.tolist(); k["provcd_top"] = np.where(k["provcd"].isin(top), k["provcd"], -1)
cont = k[["ParentInc", "age_first"]].astype(float).values
X = np.hstack([cont, (cont - cont.mean(0)) ** 2, k[["female", "urban"]].astype(float).values,
               pd.get_dummies(k["cohort"].astype(int), prefix="coh", drop_first=True).astype(float).values,
               pd.get_dummies(k["provcd_top"].astype(int), prefix="prv", drop_first=True).astype(float).values])
t, c_ = (k["HP14"] == 1).values, (k["HP14"] == 0).values; y, m = k["log_y"].values, k["college"].values
w, _ = entropy_balance(X[c_], X[t].mean(0)); wf = np.zeros(len(k)); wf[c_] = w
wmean = lambda v, mask: float((v[mask] * wf[mask]).sum() / wf[mask].sum())
tau = float(y[t].mean() - (y[c_] * w).sum()); m1, m0 = float(m[t].mean()), float((m[c_] * w).sum())
a1, a0 = float(y[t & (m == 0)].mean()), wmean(y, c_ & (m == 0)); g1, g0 = float(y[t & (m == 1)].mean()), wmean(y, c_ & (m == 1))
b1, b0 = g1 - a1, g0 - a0
parts = dict(college=b1 * (m1 - m0), returns=m0 * (b1 - b0), no_degree=a1 - a0)
assert abs(sum(parts.values()) - tau) < 1e-9, "the three parts do not sum to the premium"
rows = [("children", len(k)), ("premium", tau), ("completion, advantaged children", m1), ("completion, less advantaged children (reweighted)", m0), ("completion gap", m1 - m0),
        ("return, advantaged children (graduates minus children without a degree)", b1), ("return, less advantaged children (reweighted)", b0),
        ("part 1: college channel (return of advantaged children x completion gap)", parts["college"]),
        ("part 2: same degrees worth more (completion of less advantaged children x difference in returns)", parts["returns"]),
        ("part 3: gap between the children without a degree", parts["no_degree"]),
        ("share of the premium, part 1", parts["college"] / tau), ("gap between the graduates of the two groups", g1 - g0)]
pd.DataFrame(rows, columns=["item", "value"]).to_csv(os.path.join(ROOT, "output", "u16_identity_in_balanced_comparison.csv"), index=False)
for r in rows: print(f"{r[0]:95s} {r[1]:.4f}")
