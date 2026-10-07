"""Minimum-distance criterion under alternative error structures.

Sample: the 2,133 children of the analysis sample, waves 2014 to 2022, log
labor income residualized and winsorized within year at the 1st and 99th
percentiles, demeaned within year (pooled) or within family-background group
and year (advantaged children, ordinary children).
Moments: the fifteen variances and autocovariances, each from the children
observed in both waves of the pair.
Weights: the number of children behind each moment.

Four structures are fitted to the same moments:
  permanent effect plus AR(1) transitory component (the paper's model),
  permanent effect plus ARMA(1,1) transitory component,
  permanent effect plus MA(1) transitory component,
  random-walk permanent component plus white noise.

Outputs: output/u7_error_structures.csv (one row per sample) and
output/u7_moments.csv (the fifteen moments of each sample with the number of
children behind each).
"""
import os, sys, importlib.util
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from _locate import sibling
M = sibling("income_dynamics", __file__)


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(M, "code", name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


arma = load("ws_method1_arma11")
alt = load("ws_p1_6_ma1_rw")

df = pd.read_pickle(os.path.join(M, "data", "intermediate", "panel_resid.pkl"))
df = df.dropna(subset=["y_tilde_w", "HighParentOcc14"])
df = df[df["year"].isin([2014, 2016, 2018, 2020, 2022])].copy()
df["H"] = df["HighParentOcc14"].astype(int)
waves = sorted(df["year"].unique().tolist())


def moments(d, by):
    d = d.copy()
    d["yc"] = d["y_tilde_w"] - d.groupby(by)["y_tilde_w"].transform("mean")
    rows = []
    for i, t in enumerate(waves):
        for s in waves[i:]:
            a = d.loc[d.year == t, ["pid", "yc"]].rename(columns={"yc": "yt"})
            b = d.loc[d.year == s, ["pid", "yc"]].rename(columns={"yc": "ys"})
            both = a.merge(b, on="pid")
            rows.append((t, s, len(both), float((both.yt * both.ys).mean())))
    return pd.DataFrame(rows, columns=["t", "s", "N_ts", "omega_hat"])


samples = [("pooled", moments(df, ["year"])),
           ("advantaged children", moments(df[df.H == 1], ["year"])),
           ("ordinary children", moments(df[df.H == 0], ["year"]))]
pd.concat([cov.assign(sample=name) for name, cov in samples])[["sample", "t", "s", "N_ts", "omega_hat"]].rename(
    columns={"N_ts": "children", "omega_hat": "moment"}).to_csv(os.path.join(ROOT, "output", "u7_moments.csv"), index=False)
out = []
for name, cov in samples:
    a = arma.fit_arma11(cov); m = alt.fit_ma1(cov); r = alt.fit_rw(cov)
    out.append(dict(sample=name, moments=len(cov),
                    criterion_ar1=a["J_ar1"], criterion_arma11=a["J_arma"],
                    criterion_ma1=m["J"], criterion_random_walk=r["J"]))
out = pd.DataFrame(out)
out.to_csv(os.path.join(ROOT, "output", "u7_error_structures.csv"), index=False)
print(out.round(4).T.to_string())
