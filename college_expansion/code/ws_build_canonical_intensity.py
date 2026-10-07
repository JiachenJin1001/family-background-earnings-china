import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
ws_build_canonical_intensity.py
-------------------------------
Build canonical expansion intensity from real CNKI CEE-registrant data
(1997, 1998, 1999, 2000, 2002 by province).

The canonical normalizations:
  A. CEE response ratio (Che-Zhang style, registrant-based):
       intensity_p = log(cee_2000 / cee_1998)        ← post-expansion change
       intensity_p = log(cee_1999 / cee_1998)        ← onset-year change
       intensity_p = log(cee_2002 / cee_1998)        ← medium-run change

  B. Pre-expansion baseline:
       intensity_p = cee_1998_p / national_pop_19yr  (proxy)

The output `expansion_intensity_canonical.csv` merges with provcd so it
can be directly merged with our CFPS analysis at the child level.
"""
import os
import numpy as np
import pandas as pd

_MAC = _WSROOT + ""
_LIN = _WSROOT + ""
BASE = _MAC if os.path.isdir(_MAC) else _LIN
EXT = os.path.join(BASE, "data/external")


def main():
    long = pd.read_csv(os.path.join(EXT, "cee_registrants_province.csv"))
    print(f"Input shape: {long.shape}")
    print("Coverage by year:")
    print(long.groupby("year")["cee_total"].agg(["count", "sum"]).to_string())

    # Reshape to wide
    wide = long.pivot_table(index=["provcd","province_zh","province_en"],
                            columns="year", values="cee_total").reset_index()
    wide.columns = [f"cee_{c}" if isinstance(c, int) else c for c in wide.columns]
    print(f"\nWide shape: {wide.shape}, columns: {wide.columns.tolist()}")

    # Drop rows where critical years missing (1998 and 1999)
    have_critical = wide["cee_1998"].notna() & wide["cee_1999"].notna()
    wide_c = wide.loc[have_critical].copy()
    print(f"Provinces with 1998 + 1999 both observed: {len(wide_c)}")

    # Build canonical intensity measures
    # A1. log ratio 1999/1998 (onset year)
    wide_c["int_log_1999_1998"] = np.log(wide_c["cee_1999"] / wide_c["cee_1998"])
    # A2. log ratio 2000/1998 (1 year post)
    if "cee_2000" in wide_c.columns:
        m = wide_c["cee_2000"].notna()
        wide_c.loc[m, "int_log_2000_1998"] = np.log(wide_c.loc[m, "cee_2000"]
                                                    / wide_c.loc[m, "cee_1998"])
    # A3. log ratio 2002/1998 (medium-run)
    if "cee_2002" in wide_c.columns:
        m = wide_c["cee_2002"].notna()
        wide_c.loc[m, "int_log_2002_1998"] = np.log(wide_c.loc[m, "cee_2002"]
                                                    / wide_c.loc[m, "cee_1998"])
    # B1. Percent change 2000 vs 1998
    if "cee_2000" in wide_c.columns:
        m = wide_c["cee_2000"].notna()
        wide_c.loc[m, "int_pct_2000_1998"] = ((wide_c.loc[m, "cee_2000"]
                                               / wide_c.loc[m, "cee_1998"]) - 1)
    # B2. Pre-expansion (1997-1998) change as placebo (should be near 0)
    if "cee_1997" in wide_c.columns:
        m = wide_c["cee_1997"].notna()
        wide_c.loc[m, "int_placebo_log_1998_1997"] = np.log(
            wide_c.loc[m, "cee_1998"] / wide_c.loc[m, "cee_1997"])

    # Headline canonical intensity = log_2000_1998 (Che-Zhang style)
    wide_c["intensity_canonical"] = wide_c["int_log_2000_1998"]

    print("\nIntensity summary statistics:")
    cols = [c for c in wide_c.columns if c.startswith("int_") or c == "intensity_canonical"]
    print(wide_c[cols].describe().round(3).T.to_string())

    print("\nProvince-level intensity ranking (top/bottom 5):")
    by_p = wide_c[["province_en","cee_1998","cee_2000","intensity_canonical"]].sort_values(
        "intensity_canonical", ascending=False).reset_index(drop=True)
    print("Top 5 (largest CEE-response):")
    print(by_p.head(5).round(3).to_string(index=False))
    print("\nBottom 5 (smallest CEE-response):")
    print(by_p.tail(5).round(3).to_string(index=False))

    # Save
    out_cols = ["provcd","province_zh","province_en",
                "cee_1997","cee_1998","cee_1999","cee_2000","cee_2002",
                "int_log_1999_1998","int_log_2000_1998","int_log_2002_1998",
                "int_pct_2000_1998","int_placebo_log_1998_1997","intensity_canonical"]
    keep_cols = [c for c in out_cols if c in wide_c.columns]
    out_path = os.path.join(EXT, "expansion_intensity_canonical.csv")
    wide_c[keep_cols].to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
