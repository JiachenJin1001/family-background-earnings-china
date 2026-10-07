import os as _os
_WSROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
"""
Read the number of college entrance examination (CEE) registrants by province
from five spreadsheets of the China Education Statistical Yearbook downloaded
from CNKI, and write them as one table with a row per province and year.

Input: five password-protected .xls files, one for each of the years 1997,
1998, 1999, 2000 and 2002, in the CNKI folder under data/raw (folder name in
SRC, file names in main() below). Each file holds the yearbook table of
registrants for admission to regular institutions of higher education, by
province.

Output:
  data/external/cee_registrants_province.csv
    columns provcd, province_zh, province_en, year, cee_total, cee_male,
    cee_female, and, where the yearbook table of the year reports them,
    registrants by subject track (cee_science_eng, cee_literature) or by age
    group (cee_under18, cee_18to25, cee_over25)
"""
import io
import os
import msoffcrypto
import numpy as np
import pandas as pd

_MAC = _WSROOT + ""
_LIN = _WSROOT + ""
BASE = _MAC if os.path.isdir(_MAC) else _LIN
SRC  = os.path.join(BASE, "data/raw/CNKI_中国教育统计年鉴")
OUT  = os.path.join(BASE, "data/external")

# Province names in the yearbook tables, with the GB/T 2260 province code (the code used by the CFPS variable provcd) and the English name
PROV_MAP = {
    "北京": (11, "Beijing"),     "天津": (12, "Tianjin"),
    "河北": (13, "Hebei"),       "山西": (14, "Shanxi"),
    "内蒙古": (15, "InnerMongolia"),
    "辽宁": (21, "Liaoning"),    "吉林": (22, "Jilin"),
    "黑龙江": (23, "Heilongjiang"),
    "上海": (31, "Shanghai"),    "江苏": (32, "Jiangsu"),
    "浙江": (33, "Zhejiang"),    "安徽": (34, "Anhui"),
    "福建": (35, "Fujian"),      "江西": (36, "Jiangxi"),
    "山东": (37, "Shandong"),
    "河南": (41, "Henan"),       "湖北": (42, "Hubei"),
    "湖南": (43, "Hunan"),
    "广东": (44, "Guangdong"),   "广西": (45, "Guangxi"),
    "海南": (46, "Hainan"),
    "重庆": (50, "Chongqing"),
    "四川": (51, "Sichuan"),     "贵州": (52, "Guizhou"),
    "云南": (53, "Yunnan"),      "西藏": (54, "Tibet"),
    "陕西": (61, "Shaanxi"),     "甘肃": (62, "Gansu"),
    "青海": (63, "Qinghai"),     "宁夏": (64, "Ningxia"),
    "新疆": (65, "Xinjiang"),
}


def load_sheet(fp):
    """Decrypt a CNKI .xls file and return its sheets, without the sheet named CNKI."""
    with open(fp, "rb") as f:
        of = msoffcrypto.OfficeFile(f)
        of.load_key(password="VelvetSweatshop")
        buf = io.BytesIO(); of.decrypt(buf); buf.seek(0)
    d = pd.read_excel(buf, sheet_name=None, header=None)
    # All sheets other than the sheet named CNKI
    return {k: v for k, v in d.items() if k != "CNKI"}


def extract_province_rows(sheet, prov_col=0, total_col_candidates=(1, 2, 3, 4)):
    """Find the rows of a sheet whose first cell contains a province name.

    Returns a list with one dict per province row: provcd, province_zh,
    province_en, raw_row_idx (the row number in the sheet) and cells (the
    numeric cells of the row as pairs of column number and value).
    """
    rows = []
    for i in range(sheet.shape[0]):
        cell0 = sheet.iloc[i, 0]
        if pd.isna(cell0):
            continue
        s0 = str(cell0).strip()
        # Look for a province name in the first cell
        for zh, (code, en) in PROV_MAP.items():
            if s0.startswith(zh) or zh in s0:
                # All numeric cells of the row
                vals = []
                for c in range(1, sheet.shape[1]):
                    v = sheet.iloc[i, c]
                    if pd.notna(v):
                        try:
                            vals.append((c, float(v)))
                        except (TypeError, ValueError):
                            continue
                rows.append({
                    "provcd":        code,
                    "province_zh":   zh,
                    "province_en":   en,
                    "raw_row_idx":   i,
                    "cells":         vals,
                })
                break
    return rows


def parse_1997_1998(fp, year):
    """Files of 1997 and 1998: one sheet; the total number of registrants is in column 1 (1998) or column 2 (1997)."""
    sheets = load_sheet(fp)
    sheet = list(sheets.values())[0]
    prov_rows = extract_province_rows(sheet)

    # Row with the national total
    total_row = None
    for i in range(sheet.shape[0]):
        v = sheet.iloc[i, 0]
        if pd.notna(v) and any(k in str(v) for k in ["总计Total", "合计"]):
            total_row = i
            break
    print(f"  {os.path.basename(fp)}: total_row={total_row}, prov_rows={len(prov_rows)}")

    # 1998 file: column 1 = total registrants, column 2 = male, column 3 = female,
    #   column 4 = science and engineering track, column 5 = literature and history track
    # 1997 file: the same columns moved one place to the right, because of an empty column
    out = []
    for r in prov_rows:
        cells = dict(r["cells"])
        if year == 1998:
            total = cells.get(1)
            male  = cells.get(2)
            female= cells.get(3)
            # Subject tracks
            sci   = cells.get(4)
            lit   = cells.get(5)
        elif year == 1997:
            total = cells.get(2)
            male  = cells.get(3)
            female= cells.get(4)
            sci   = cells.get(5)
            lit   = cells.get(6)
        else:
            total = male = female = sci = lit = None
        out.append({
            "provcd":           r["provcd"],
            "province_zh":      r["province_zh"],
            "province_en":      r["province_en"],
            "year":             year,
            "cee_total":        total,
            "cee_male":         male,
            "cee_female":       female,
            "cee_science_eng":  sci,
            "cee_literature":   lit,
        })
    return pd.DataFrame(out)


def parse_2002(fp):
    """File of 2002: the sheet named '1-2' holds parts one and two of the table of registrants; the third part is in a separate sheet and is not read."""
    sheets = load_sheet(fp)
    # Sheet '1-2': the first 35 rows or so are part one (registrants by province), the rows after them are part two
    main = sheets.get("1-2") if "1-2" in sheets else list(sheets.values())[0]
    out = []
    for i in range(main.shape[0]):
        v = main.iloc[i, 0]
        if pd.isna(v):
            continue
        s = str(v).strip()
        for zh, (code, en) in PROV_MAP.items():
            if s == zh or s.startswith(zh) or zh in s:
                cells = {}
                for c in range(1, main.shape[1]):
                    val = main.iloc[i, c]
                    if pd.notna(val):
                        try:
                            cells[c] = float(val)
                        except (TypeError, ValueError):
                            continue
                # Part one of the 2002 table: column 2 = total registrants, column 3 = male, column 4 = female
                total = cells.get(2)
                out.append({
                    "provcd":      code,
                    "province_zh": zh,
                    "province_en": en,
                    "year":        2002,
                    "cee_total":   total,
                    "cee_male":    cells.get(3),
                    "cee_female":  cells.get(4),
                })
                break
    return pd.DataFrame(out).drop_duplicates(subset=["provcd"], keep="first")


def parse_1999(fp):
    """File of 1999: province in column 0, total registrants in column 1, male and female in columns 2 and 3,
    age groups in columns 4 to 6."""
    sheets = load_sheet(fp)
    sheet = list(sheets.values())[0]
    out = []
    for i in range(sheet.shape[0]):
        v = sheet.iloc[i, 0]
        if pd.isna(v): continue
        s = str(v).strip()
        for zh, (code, en) in PROV_MAP.items():
            if s.startswith(zh) or zh in s:
                def get(c):
                    vv = sheet.iloc[i, c]
                    if pd.isna(vv): return None
                    try: return float(vv)
                    except (TypeError, ValueError): return None
                out.append({
                    "provcd": code, "province_zh": zh, "province_en": en,
                    "year": 1999,
                    "cee_total":       get(1),
                    "cee_male":        get(2),
                    "cee_female":      get(3),
                    "cee_under18":     get(4),
                    "cee_18to25":      get(5),
                    "cee_over25":      get(6),
                })
                break
    return pd.DataFrame(out).drop_duplicates(subset=["provcd"], keep="first")


def parse_2000(fp):
    """File of 2000: same layout as the file of 1999. Province in column 0, total registrants in column 1,
    male in column 2, female in column 3, age groups in columns 4 to 6."""
    sheets = load_sheet(fp)
    sheet = list(sheets.values())[0]
    out = []
    for i in range(sheet.shape[0]):
        v = sheet.iloc[i, 0]
        if pd.isna(v): continue
        s = str(v).strip()
        for zh, (code, en) in PROV_MAP.items():
            if s.startswith(zh) or zh in s:
                def get(c):
                    vv = sheet.iloc[i, c]
                    if pd.isna(vv): return None
                    try: return float(vv)
                    except (TypeError, ValueError): return None
                out.append({
                    "provcd": code, "province_zh": zh, "province_en": en,
                    "year": 2000,
                    "cee_total":       get(1),
                    "cee_male":        get(2),
                    "cee_female":      get(3),
                    "cee_under18":     get(4),
                    "cee_18to25":      get(5),
                    "cee_over25":      get(6),
                })
                break
    return pd.DataFrame(out).drop_duplicates(subset=["provcd"], keep="first")


def main():
    files = {
        1997: os.path.join(SRC, "招生报名情况_1997_分省.xls"),
        1998: os.path.join(SRC, "招生报名情况_1998_分省.xls"),
        1999: os.path.join(SRC, "招生报名情况_1999_分省_一.xls"),
        2000: os.path.join(SRC, "招生报名情况_2000_分省_扩展.xls"),
        2002: os.path.join(SRC, "招生报名情况_2002_分省.xls"),
    }
    dfs = []
    for year, fp in files.items():
        print(f"\n=== Parsing {year}: {os.path.basename(fp)} ===")
        if year == 2002:
            df = parse_2002(fp)
        elif year == 1999:
            df = parse_1999(fp)
        elif year == 2000:
            df = parse_2000(fp)
        else:
            df = parse_1997_1998(fp, year)
        # Compare the sum over provinces with the national total
        total = df["cee_total"].sum()
        print(f"   Sum of cee_total across provinces = {total:,.0f}")
        print(f"   N provinces = {len(df)}")
        # National totals of registrants by year
        expected = {1997: 2_842_659, 1998: 3_202_197,
                    1999: 3_404_445, 2000: 3_884_823,
                    2002: 6_124_580}
        if year in expected:
            err = abs(total - expected[year]) / expected[year] * 100
            print(f"   Expected ≈ {expected[year]:,.0f}, error = {err:.2f}%")
        dfs.append(df)

    long = pd.concat(dfs, ignore_index=True)
    # 1997, Inner Mongolia: the total in the source file has an extra zero; the sum of male and female registrants is used
    mask = (long.year == 1997) & (long.provcd == 15)
    if mask.any():
        male_v   = long.loc[mask, "cee_male"].iloc[0]
        female_v = long.loc[mask, "cee_female"].iloc[0]
        corrected = (male_v if pd.notna(male_v) else 0) + (female_v if pd.notna(female_v) else 0)
        old = long.loc[mask, "cee_total"].iloc[0]
        long.loc[mask, "cee_total"] = corrected
        print(f"\nCorrection: 1997 Inner Mongolia cee_total {old:,.0f} replaced by {corrected:,.0f}")
    long = long.sort_values(["year", "provcd"]).reset_index(drop=True)
    out_path = os.path.join(OUT, "cee_registrants_province.csv")
    long.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")
    print(f"Shape: {long.shape}")
    print(long.head(10).to_string())

    # List province-year totals that are more than 2 standard deviations from the mean of the province
    print("\nMissing / suspect totals (>2 std from province mean):")
    by_prov = long.groupby("provcd")["cee_total"].agg(["mean", "std"]).reset_index()
    merged = long.merge(by_prov, on="provcd")
    merged["z"] = (merged["cee_total"] - merged["mean"]) / merged["std"].replace(0, np.nan)
    sus = merged.loc[merged["z"].abs() > 2]
    if len(sus) > 0:
        print(sus[["year","province_zh","cee_total","mean","z"]].to_string(index=False))


if __name__ == "__main__":
    main()
