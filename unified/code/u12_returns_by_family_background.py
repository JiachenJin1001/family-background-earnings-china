"""First stages and returns to college by family background (Section 6 of the paper), in the two-row design.

Rows of u9_mature_age_earnings.py (main run unweighted; E_WEIGHTS=process for the weighted run). The instrument and
college completion enter once for each group and age band; province effects are common to the groups; birth-year
effects are specific to each group; family background, the indicator for the row at 30 and over and their product
are controlled. Everything reported is for the rows at 30 and over: the first stage by family background, which is the
coefficient on the group's instrument in the regression of the child's college completion x 1[30+] on the instruments
and controls (with wild p; the same definition as the first stages on all children in u14_sample_facts.py); the
coefficient on the group's instrument when the dependent variable is the group's own endogenous regressor, college x
group x 1[30+] (columns fs_own_regressor and F_own_regressor, kept for the record, not printed in the paper); the
reduced form (wild p), the instrumented return with its clustered standard
error, the least squares return, Anderson-Rubin sets at 95 and 90 percent (grid -6 to 6, step 0.05; a set that reaches the
edge of the grid is unbounded on that side, which happens only when the first stage of the group does not reject zero), and the test that the two returns are equal.
Wild cluster bootstrap by province, 9,999 Rademacher draws, null imposed.
Output: output/u12_returns_by_family_background.csv and output/u12_anderson_rubin_grid.csv, the p-value at every grid
point (suffix _weighted for the weighted run)
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPS = int(os.environ.get("U12_REPS", "9999")); os.environ["U9_REPS"] = str(REPS)
src = open(os.path.join(HERE, "u9_mature_age_earnings.py"), encoding="utf-8").read(); src = src[:src.index("# ------------------------------------------------------------------ A. the sample")]
u9 = {"__file__": os.path.join(HERE, "u9_mature_age_earnings.py"), "__name__": "u9"}; exec(compile(src, "u9", "exec"), u9)
Design, c, s, dummies, DEMO, cl, SFX, tb_X = u9["Design"], u9["c"], u9["s"], u9["dummies"], u9["DEMO"], u9["cl"], u9["SFX"], u9["tb_X"]
rng = np.random.default_rng(20261003); GRID = np.round(np.arange(-6.0, 6.001, 0.05), 2)   # wide enough for the ends of every bounded set
H = s.HP14.values.astype(float); old = s.old.values.astype(float); sw = np.sqrt(s.w.values); z, m, y, g = s.Z.values, s.college.values, s.y.values, s.pc.values
X = tb_X(s, group_fe=True)                      # includes the band indicator and family background x band
keys = [("less advantaged", 1), ("advantaged", 1), ("less advantaged", 0), ("advantaged", 0)]
sel = lambda grp, o: (old == o) * (H == (1.0 if grp == "advantaged" else 0.0))
Z4 = {k: z * sel(*k) for k in keys}; M4 = {k: m * sel(*k) for k in keys}; W = lambda a: sw[:, None] * a
Zm = W(np.column_stack([Z4[k] for k in keys] + [X])); En = W(np.column_stack([M4[k] for k in keys]))
Xh = np.column_stack([Zm @ np.linalg.lstsq(Zm, En, rcond=None)[0], W(X)]); b = np.linalg.lstsq(Xh, sw * y, rcond=None)[0]
u = sw * y - np.column_stack([En, W(X)]) @ b; bread = np.linalg.pinv(Xh.T @ Xh); meat = np.zeros((Xh.shape[1],) * 2)
for k in np.unique(g): sc = Xh[g == k].T @ u[g == k]; meat += np.outer(sc, sc)
G, n, kk = len(np.unique(g)), len(u), Xh.shape[1]; V = bread @ meat @ bread * G / (G - 1) * (n - 1) / (n - kk)
beta = {k: float(b[i]) for i, k in enumerate(keys)}; se = {k: float(np.sqrt(V[i, i])) for i, k in enumerate(keys)}
eq = (b[0] - b[1]) / np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])
ls = cl(sw * y, np.column_stack([En, W(X)]), g); lsb = {k: (float(ls.params[i]), float(ls.bse[i])) for i, k in enumerate(keys)}
rows = []; grid_rows = []   # grid_rows: the wild-cluster p-value of the Anderson-Rubin test at every grid point
for grp in ("less advantaged", "advantaged"):
    k1 = (grp, 1); D = Design(s, sw * Z4[k1], W(np.column_stack([Z4[k] for k in keys if k != k1] + [X])), rng)
    f = D.rf(sw * m * old); own = D.rf(sw * M4[k1]); e = D.rf(sw * y)   # f: the child's completion x 1[30+] on the group's instrument
    pv = np.array([D.rf(sw * (y - b0 * M4[k1]))["p_wild"] for b0 in GRID]); k95, k90 = GRID[pv >= 0.05], GRID[pv >= 0.10]
    r = dict(sample=f"rows at 30 and over of the two-row design, ages {u9['S_LO']}-{u9['S_HI']}", group=grp, children=int(sel(grp, 1).sum()), fs=f["coef"], fs_se=f["se"], F=f["t"] ** 2, fs_p_wild=f["p_wild"], fs_own_regressor=own["coef"], F_own_regressor=own["t"] ** 2, fs_own_regressor_p_wild=own["p_wild"],
             rf=e["coef"], rf_se=e["se"], rf_p_wild=e["p_wild"], beta=beta[k1], beta_se=se[k1], beta_below_30=beta[(grp, 0)], ls=lsb[k1][0], ls_se=lsb[k1][1],
             ar95_lo=float(k95.min()) if len(k95) else np.nan, ar95_hi=float(k95.max()) if len(k95) else np.nan, ar90_lo=float(k90.min()) if len(k90) else np.nan, ar90_hi=float(k90.max()) if len(k90) else np.nan,
             grid_lo=float(GRID[0]), grid_hi=float(GRID[-1]), equal_returns_z=float(eq))
    rows.append(r); grid_rows += [dict(group=grp, b=float(b0), p_wild=float(p0)) for b0, p0 in zip(GRID, pv)]
    print(f"{grp:16s} rows at 30+ {r['children']:5d} | fs {f['coef']:+.3f} ({f['se']:.3f}) F {r['F']:5.1f} p {f['p_wild']:.4f} | rf {e['coef']:+.3f} ({e['se']:.3f}) p {e['p_wild']:.4f} | return {r['beta']:+.2f} ({r['beta_se']:.2f}) AR95 [{r['ar95_lo']}, {r['ar95_hi']}] AR90 [{r['ar90_lo']}, {r['ar90_hi']}] | ls {r['ls']:.3f} ({r['ls_se']:.3f})", flush=True)
print(f"  test of equal returns z = {eq:.2f}")
pd.DataFrame(rows).to_csv(os.path.join(HERE, "..", "output", f"u12_returns_by_family_background{SFX}.csv"), index=False); print("wrote")
pd.DataFrame(grid_rows).to_csv(os.path.join(HERE, "..", "output", f"u12_anderson_rubin_grid{SFX}.csv"), index=False)
