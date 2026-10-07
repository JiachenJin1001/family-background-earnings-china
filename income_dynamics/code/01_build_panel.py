import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Build the person-year panel from the seven CFPS waves (2010, 2012, 2014, 2016,
2018, 2020, 2022).

One row per person (pid) and wave, for every record of the CFPS adult or
person files: income, age, gender, urban residence, province, occupation code,
years of schooling. Two copies are written to data/intermediate/:
  - panel_long.dta           readable in Stata
  - _panel_long_cache.pkl    read by 02_sample_and_parents.py and by the
                             scripts that need the parents' records

The next two steps are separate scripts: 02_sample_and_parents.py links
children to parents, builds the indicator for advantaged children from the
retrospective questions, and selects the analysis sample;
03_step0_residualize.py removes age, gender, residence, year and province
effects from log income.
"""
import os, sys
import numpy as np
import pandas as pd
import pyreadstat

BASE_RAW = _WSROOT + "/data/raw"
BASE_INT = _WSROOT + "/data/intermediate"
os.makedirs(BASE_INT, exist_ok=True)

# ======================================================================
# Variable names of each CFPS wave
# ======================================================================
WAVES = {
    2010: dict(
        ind="CFPS2010/ecfps2010adult_201906.dta",
        famconf="CFPS2010/ecfps2010famconf_nat072016.dta",
        income="income",
        age="qa1age",
        gender="gender",          # CFPS value labels: 0 = female, 1 = male
        urban="urban",
        provcd="provcd",
        occ="qg307code",
        eduy="cfps2010eduy_best",
    ),
    2012: dict(
        ind="CFPS2012/ecfps2012adult_202505.dta",
        famconf="CFPS2012/ecfps2012famconf_092015.dta",
        income="income_adj",
        age="cfps2012_age",
        gender="cfps2012_gender_best",
        urban="urban12",
        provcd="provcd",
        occ="qg411code_a_1",
        eduy="eduy2012",
    ),
    2014: dict(
        ind="CFPS2014/ecfps2014adult_201906.dta",
        famconf="CFPS2014/ecfps2014famconf_170630.dta",
        # The 2014 wave uses "income" (total income from all jobs) and not
        # "p_income" (income from all sources), so that the income concept
        # matches 2016 and 2018 ("income") and 2020 and 2022 ("emp_income"),
        # which record income from jobs only.
        income="income",
        age="cfps2014_age",
        gender="cfps_gender",
        urban="urban14",
        provcd="provcd14",
        occ="qg303code",
        eduy="cfps2014eduy_im",
    ),
    2016: dict(
        ind="CFPS2016/ecfps2016adult_201906.dta",
        famconf="CFPS2016/ecfps2016famconf_201804.dta",
        income="income",
        age="cfps_age",
        gender="cfps_gender",
        urban="urban16",
        provcd="provcd16",
        occ="qg303code",
        eduy="cfps2016eduy_im",
    ),
    2018: dict(
        ind="CFPS2018/ecfps2018person_202012.dta",
        famconf="CFPS2018/ecfps2018famconf_202008.dta",
        income="income",
        age="age",
        gender="gender",
        urban="urban18",
        provcd="provcd18",
        occ="qg303code",
        eduy="cfps2018eduy_im",
    ),
    2020: dict(
        ind="CFPS2020/ecfps2020person_202306.dta",
        famconf="CFPS2020/ecfps2020famconf_202306.dta",
        income="emp_income",
        age="age",
        gender="gender",
        urban="urban20",
        provcd="provcd20",
        occ="qg303code",
        eduy="cfps2020eduy_im",
    ),
    2022: dict(
        ind="CFPS2022/ecfps2022person_202410.dta",
        famconf="CFPS2022/ecfps2022famconf_202410.dta",
        income="emp_income",
        age="age",
        gender="gender",
        urban="urban22",
        provcd="provcd22",
        occ="qg303code",
        eduy="cfps2022eduy_im",
    ),
}
# The columns of the family-relationship file that hold the parents' pids are
# pid_a_f and pid_a_m in 2018 and 2020, and pid_f and pid_m in the other waves.
FAMCONF_PIDF = {yr: ("pid_a_f" if yr in (2018, 2020) else "pid_f") for yr in WAVES}
FAMCONF_PIDM = {yr: ("pid_a_m" if yr in (2018, 2020) else "pid_m") for yr in WAVES}


# ======================================================================
# Part 1: person-year panel
# ======================================================================
def build_long_panel() -> pd.DataFrame:
    rows = []
    for yr, spec in WAVES.items():
        path = os.path.join(BASE_RAW, spec["ind"])
        keep = ["pid", spec["income"], spec["age"], spec["gender"],
                spec["urban"], spec["provcd"], spec["occ"], spec["eduy"]]
        # keep only the columns that exist in the file of this wave
        df_head, _ = pyreadstat.read_dta(path, row_limit=1)
        keep = [c for c in keep if c in df_head.columns]
        df, meta = pyreadstat.read_dta(path, usecols=keep)
        rename = {spec["income"]:"income", spec["age"]:"age",
                  spec["gender"]:"gender_raw", spec["urban"]:"urban",
                  spec["provcd"]:"provcd", spec["occ"]:"occ_code",
                  spec["eduy"]:"eduy"}
        rename = {k:v for k,v in rename.items() if k in df.columns}
        df = df.rename(columns=rename)
        df["year"] = yr
        for c in ["income","age","urban","provcd","occ_code","eduy","gender_raw"]:
            if c not in df.columns:
                df[c] = np.nan
        rows.append(df[["pid","year","income","age","gender_raw","urban","provcd","occ_code","eduy"]])
        print(f"[01] wave {yr}: {len(df):,} rows loaded")
    long = pd.concat(rows, ignore_index=True)
    long = long.dropna(subset=["pid"]).copy()
    long["pid"] = long["pid"].astype(np.int64)

    # Indicator for female. CFPS value labels of the gender variable:
    # 0 = female, 1 = male.
    long["female"] = np.where(long["gender_raw"] == 0, 1,
                       np.where(long["gender_raw"] == 1, 0, np.nan))

    # urban: 0/1; negative codes (refused, do not know) are set to missing
    long.loc[long["urban"] < 0, "urban"] = np.nan

    # provcd: negative codes are set to missing
    long.loc[long["provcd"] < 0, "provcd"] = np.nan
    long["provcd"] = long["provcd"].astype("Int64")

    # occ_code: negative codes (refused, do not know) are set to missing
    long.loc[long["occ_code"] < 0, "occ_code"] = np.nan

    # income: negative codes (refused, do not know) are set to missing; zeros
    # are kept here and excluded later by the positive-income filter
    long.loc[long["income"] < 0, "income"] = np.nan

    # eduy: negative codes are set to missing
    long.loc[long["eduy"] < 0, "eduy"] = np.nan

    # age: values outside 0 to 120 are set to missing
    long.loc[(long["age"] < 0) | (long["age"] > 120), "age"] = np.nan

    print(f"[01] panel_long: {len(long):,} rows, {long['pid'].nunique():,} unique pids")
    return long


# ======================================================================
# Part 2: parents and child sample
# ======================================================================


def main():
    long = build_long_panel()
    save = long[["pid", "year", "income", "age", "female", "urban", "provcd", "occ_code", "eduy"]].copy()
    save["provcd"] = save["provcd"].astype(float)
    os.makedirs(BASE_INT, exist_ok=True)
    save.to_pickle(os.path.join(BASE_INT, "_panel_long_cache.pkl"))
    pyreadstat.write_dta(save, os.path.join(BASE_INT, "panel_long.dta"))
    print(f"wrote the long panel: {len(save):,} rows, {save['pid'].nunique():,} persons")


if __name__ == "__main__":
    main()
