"""Variation of the grid-averaged nonlinear persistence across runs of the stochastic EM (Section 3.5 of the paper).

The paper prints the grid-averaged local persistence of one run of
income_dynamics/code/ws_05b_abb_stochastic_em.py (pooled sample, advantaged children, less advantaged children). This
script reruns that estimator with ten seeds for each of the three samples, with the same settings (Hermite order 2,
Bernstein order 3, 6 iterations, 4 paths, 30 Metropolis-Hastings steps, at most 500 children drawn for each fit), and
records the average of the local persistence over the 5 x 5 lattice of state and shock-rank quantiles in each run. The
paper reports the range of the pooled figure over the ten runs.

Input (income-dynamics pipeline): data/intermediate/panel_resid.pkl
Output: output/u15_nonlinear_persistence_runs.csv (one line for each sample and run)
        output/u15_nonlinear_persistence_summary.csv (mean, standard deviation, smallest and largest by sample, and the
        number of runs in which the figure of advantaged children exceeds that of less advantaged children)
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from _locate import sibling
sys.path.insert(0, os.path.join(sibling("income_dynamics", __file__), "code"))
import ws_05b_abb_stochastic_em as abb

RUNS = int(os.environ.get("U15_RUNS", "10")); SEED = 20261004
panel = pd.read_pickle(os.path.join(abb.BASE_INT, "panel_resid.pkl")); hp = "HighParentOcc14"
panel = panel.dropna(subset=["y_tilde_w", "year", "pid", hp]).copy(); panel["year"] = panel["year"].astype(int)
SAMPLES = [("pooled", np.ones(len(panel), dtype=bool), 0), ("advantaged children", (panel[hp] == 1).values, 1), ("less advantaged children", (panel[hp] == 0).values, 2)]
rows = []
for name, mask, shift in SAMPLES:
    for k in range(RUNS):
        t0 = time.time()
        fit = abb.run_abb(panel, name, mask, K=2, L=3, n_iter=6, n_paths=4, mh_steps=30, seed=SEED + 100 * k + shift)
        surf = abb.extract_persistence_surface(fit)
        rows.append(dict(sample=name, run=k, average_persistence=float(surf.rho_local.mean()), seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.drop(columns="seconds").to_csv(os.path.join(ROOT, "output", "u15_nonlinear_persistence_runs.csv"), index=False)
s = d.groupby("sample").average_persistence.agg(["mean", "std", "min", "max"]).reset_index()
p = d.pivot(index="run", columns="sample", values="average_persistence")
s["runs"] = RUNS; s["runs_advantaged_above_less_advantaged"] = int((p["advantaged children"] > p["less advantaged children"]).sum())
s.to_csv(os.path.join(ROOT, "output", "u15_nonlinear_persistence_summary.csv"), index=False); print(s.round(3).to_string(index=False)); print("finished")
