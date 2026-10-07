"""Variances and autocovariances of residual log earnings by family background
(the figure next to the autocovariance table of Section 3).

Left panel: the variance of residual log earnings in each wave, advantaged
against less advantaged children. Right panel: the autocovariance at each lag,
averaged over the wave pairs at that lag with inverse-variance weights.
Intervals are 95 percent intervals from the bootstrap standard errors of the
moments (m5_md_weighted_and_ridge.py of the income-dynamics pipeline).
Output: paper/figures/fig_autocov.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from _locate import sibling
m = pd.read_csv(os.path.join(sibling("income_dynamics", __file__), "output", "tables", "m5_moment_se.csv"))
m["lag"] = (m.s - m.t) // 2

from fig_style import fs, setup, tr, outpath, ADV, LESS
setup()
fig, axes = plt.subplots(1, 2, figsize=fs((10.4, 4.3)))
GROUPS = [("H", tr("Advantaged children"), ADV, "s", 0.06), ("L", tr("Less advantaged children"), LESS, "o", -0.06)]
LS = {"s": "--", "o": "-"}   # dashed line for advantaged children, solid for less advantaged children
ax = axes[0]
for grp, lab, col, mk, off in GROUPS:
    d = m[(m.group == grp) & (m.lag == 0)]
    ax.errorbar(d.t + off * 8, d.omega, yerr=1.96 * d.se, fmt=mk + LS[mk], color=col, capsize=2.5, lw=1.2, ms=4, label=lab)
ax.set_xlabel(tr("Survey wave")); ax.set_ylabel(tr("Variance of residual log earnings"), fontsize=8.5)
ax.set_title(tr("(a) Variance in each wave")); ax.set_ylim(0, None); ax.set_xticks([2014, 2016, 2018, 2020, 2022])
ax = axes[1]
for grp, lab, col, mk, off in GROUPS:
    d = m[m.group == grp]
    rows = []
    for lag, x in d.groupby("lag"):
        w = 1 / x.se ** 2
        rows.append(dict(lag=lag, omega=np.average(x.omega, weights=w), se=np.sqrt(1 / w.sum())))
    g = pd.DataFrame(rows)
    ax.errorbar(g.lag + off, g.omega, yerr=1.96 * g.se, fmt=mk + LS[mk], color=col, capsize=2.5, lw=1.2, ms=4, label=lab)
ax.set_xlabel(tr("Lag (waves of two years)")); ax.set_ylabel(tr("Autocovariance of residual log earnings"), fontsize=8.5)
ax.set_title(tr("(b) Autocovariance by lag")); ax.set_xticks(range(5))
for ax in axes:
    ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.25)
axes[0].legend(loc="lower left")   # one legend for the two panels
fig.tight_layout()
out = outpath("fig_autocov.png")
from fig_style import save
out = save(fig, os.path.basename(out)); print("wrote", out)
