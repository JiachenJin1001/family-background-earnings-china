"""The identification ridge (Figure 1 of the paper), redrawn with the group
labels used in the paper and with the two point estimates marked.

Repeats the bootstrap of the income-dynamics pipeline's
m7_md_cellweighted_boot.py (same estimator, same seed, 1,000 replications that
resample children, re-demean within year, and re-solve the minimum-distance
problem with weights equal to the number of child pairs behind each moment),
and saves the draws so the figure can be redrawn without rerunning it.
Outputs: output/u5_ridge_draws.csv, paper/figures/fig_ridge.png
"""
import os, sys, runpy, io, contextlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from _locate import sibling
M = sibling("income_dynamics", __file__)
DRAWS = os.path.join(ROOT, "output", "u5_ridge_draws.csv")

if not os.path.exists(DRAWS) or os.environ.get("U5_REDO") == "1":
    # run the pipeline's script in a scratch namespace without letting it overwrite its own outputs
    src = open(os.path.join(M, "code", "m7_md_cellweighted_boot.py"), encoding="utf-8").read()
    src = src.replace('res.to_csv(_os.path.join(_WSROOT,"output/tables/m7_md_cellw_boot.csv"), index=False)', "pass")
    src = src.replace('fig.savefig(_os.path.join(_WSROOT,"output/figures/fig_ridge.png"), dpi=200, bbox_inches="tight")', "pass")
    g = {"__name__": "__ridge__", "__file__": os.path.join(M, "code", "m7_md_cellweighted_boot.py")}
    exec(compile(src, g["__file__"], "exec"), g)
    rows = []
    for key, lab in [("H", "advantaged"), ("L", "ordinary")]:
        for r in g["draws"][key]:
            rows.append(dict(group=lab, permanent_variance=r[0], persistence=r[1], shock_variance=r[2], permanent_share=r[3]))
    pd.DataFrame(rows).to_csv(DRAWS, index=False)
    pt = g["res"].set_index("group")
    pd.DataFrame([dict(group="advantaged", persistence=pt.loc["H", "rho"], permanent_share=pt.loc["H", "sperm"]),
                  dict(group="ordinary", persistence=pt.loc["L", "rho"], permanent_share=pt.loc["L", "sperm"])]).to_csv(
        os.path.join(ROOT, "output", "u5_ridge_points.csv"), index=False)

d = pd.read_csv(DRAWS); pts = pd.read_csv(os.path.join(ROOT, "output", "u5_ridge_points.csv")).set_index("group")
from fig_style import fs, setup, tr, outpath, ADV, LESS
setup()
fig, ax = plt.subplots(figsize=fs((6.4, 4.4)))
_facts = pd.read_csv(os.path.join(ROOT, "output", "u14_B_sample_facts.csv")).set_index("item")["value"]   # children with at least three waves
_nH = int(_facts["three waves: advantaged children"]); _nL = int(_facts["three waves: children"]) - _nH
for key, col, mk, lab, n in [("advantaged", ADV, "o", "Advantaged children", _nH), ("ordinary", LESS, "^", "Less advantaged children", _nL)]:
    s = d[d.group == key]
    ax.scatter(s.persistence, s.permanent_share, s=5, alpha=0.3, color=col, marker=mk, linewidths=0, label=f"{tr(lab)} ($N$ = {n:,}), {tr('bootstrap draws')}")
for key, mk in [("advantaged", "o"), ("ordinary", "^")]:
    ax.scatter(pts.loc[key, "persistence"], pts.loc[key, "permanent_share"], s=60, marker=mk, facecolor="white", edgecolor="k", linewidth=1.4, zorder=5)
ax.scatter([], [], s=60, marker="o", facecolor="white", edgecolor="k", linewidth=1.4, label=tr("Point estimates (circle: advantaged; triangle: less advantaged)"))
ax.set_xlabel(tr("Transitory persistence $\\rho_v$")); ax.set_ylabel(tr("Permanent share"))
# the legend sits above the axes, so that it covers none of the draws on the boundary at a share of zero
leg = ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, borderaxespad=0); ax.grid(alpha=0.25)
for h_ in leg.legend_handles[:2]: h_.set_alpha(1); h_.set_sizes([22])   # the legend markers of the bootstrap draws are drawn opaque and larger, so that they can be seen
from fig_style import save
save(fig, "fig_ridge.png")
print("wrote fig_ridge.png; point estimates:", pts.round(3).to_dict())
