"""The premium in fifteen specifications (Figure 2 of the paper).

Reads the estimates and 95 percent intervals written by
u8_premium_specifications.py (the rows marked as belonging to the figure).
The upper panel plots the coefficient on the indicator for advantaged
children, sorted by size; the lower panel marks how each specification
differs from the baseline.
Output: paper/figures/fig_spec_curve.png
"""
import os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

CRE, EB, XS = "Correlated random effects", "Entropy balancing", "Cross-section, one observation per child"
# what each specification changes relative to the baseline
CHANGES = {
    "baseline: family income of the parental household": "base",
    "parents' own labor income as the income control": "labor",
    "income deflated by the provincial consumer price index": "cpi",
    "most frequent retrospective report of parental occupation": "altdef",
    "indicator widened to CSCO major groups 1 to 3": "altdef",
    "indicator narrowed to CSCO major group 1": "altdef",
    "three regions in place of province effects": "region",
    "parental income from the 2010 and 2012 waves, before the earnings window": "window",
    "parental income from waves with the child aged 40 or less": "window",
    "parental income from waves with the child aged 35 or less": "window",
    "children recorded as rural in most of their waves": "subsample",
    "children recorded as urban in most of their waves": "subsample",
    "parental party membership added (cross-section)": "marker",
    "parental state-sector employment added (cross-section)": "marker+subsample",
    "entropy balancing": "",
}
FEATS = [("Baseline specification", "base"),
         ("Parents' own labor income as the income control", "labor"),
         ("Alternative definition of the parental occupation indicator", "altdef"),
         ("Family income from the 2010 and 2012 waves,\nor from waves with the child aged 40 or 35 at most", "window"),
         ("Urban, rural, or parent-still-working subsample", "subsample"),
         ("Parental party membership or state-sector employment added", "marker"),
         ("Region effects in place of province effects", "region"),
         ("Income deflated by provincial CPI", "cpi")]

d = pd.read_csv(os.path.join(ROOT, "output", "u8_premium_specifications.csv"))
d = d[d.in_figure_2].copy()
assert set(d.specification) == set(CHANGES), set(d.specification) ^ set(CHANGES)
for _, key in FEATS:
    d[key] = d.specification.map(lambda s: int(key in CHANGES[s].split("+")))
d = d.sort_values(["estimate", "base"]).reset_index(drop=True)
colors = {CRE: "#08519c", EB: "#d95f02", XS: "#555555"}   # the dark blue and the orange of the other figures, and a dark grey
marks = {CRE: "o", EB: "s", XS: "^"}
from fig_style import fs, setup, tr, outpath
setup()
fig, (a1, a2) = plt.subplots(2, 1, figsize=fs((9.6, 6.6)), gridspec_kw=dict(height_ratios=[2.0, 1.9], hspace=0.07))
for i, r in d.iterrows():
    a1.plot([i, i], [r.lo, r.hi], color=colors[r.estimator], alpha=.6, lw=1.3)
    a1.plot(i, r.estimate, marks[r.estimator], color=colors[r.estimator], ms=4.5)
    if r.base == 1:
        a1.plot(i, r.estimate, "D", mfc="none", mec="k", ms=9, mew=1.1)
        a1.annotate(tr("baseline"), (i, r.hi), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
a1.axhline(0, color="k", lw=.8)
a1.set_ylabel(tr("Premium (log points)"))
a1.set_xticks([])
a1.legend(handles=[plt.Line2D([], [], marker=marks[k], ls="", color=c, label=tr(k)) for k, c in colors.items()],
          fontsize=10, loc="upper left", framealpha=0.95)
lo_, hi_ = a1.get_ylim(); a1.set_ylim(lo_, hi_ + 0.35 * (hi_ - lo_))   # room for the legend above the estimates
for j, (nm, key) in enumerate(FEATS):
    on = d.index[d[key] == 1]
    a2.plot(on, [j] * len(on), "o", ms=3.2, color="#333")
a2.set_yticks(range(len(FEATS))); a2.set_yticklabels([tr(f[0]) for f in FEATS], fontsize=9.5, linespacing=0.95)
a2.set_ylim(-.7, len(FEATS) - .3); a2.invert_yaxis(); a2.set_xlim(a1.get_xlim())
a2.set_xticks([]); a2.set_xlabel(tr("Specifications, sorted by point estimate")); a2.grid(axis="x", alpha=.2)
out = outpath("fig_spec_curve.png")
from fig_style import save
out = save(fig, os.path.basename(out)); print("wrote", out, "specifications:", len(d))
print(d[["specification", "estimate", "lo", "hi"]].round(3).to_string())
print("intervals that exclude zero:", int(((d.lo > 0) | (d.hi < 0)).sum()))
