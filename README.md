# Before the Labor Market or Inside It? Family Background, Earnings Dynamics, and the Return to College

Replication files and online appendix. Jiachen Jin, New York University.

The paper: [`unified/paper/main.pdf`](unified/paper/main.pdf). The online appendix:
[`online_appendix/online_appendix.pdf`](online_appendix/online_appendix.pdf).

## What is here

| Folder | Content |
|---|---|
| `unified/paper/` | The paper (`main.pdf`, `main.tex`), its figures and bibliography |
| `unified/code/` | Scripts that build the sample of the paper and its descriptive facts, estimate the premium (regressions, entropy balancing, the fifteen specifications), run the expansion design and the decomposition, and write the tables and figures of the paper; the check of the numbers printed in the paper |
| `unified/output/` | Estimates written by the scripts of `unified/code/` |
| `income_dynamics/` | Construction of the panel from the survey files, residualization, the earnings process, the attenuation schedule |
| `college_expansion/` | Provincial intensity measures, the cohort exposure series, the placebo from examination registrations, the census check of trends before the reform, and the entropy-balancing routine that `unified/code/` imports |
| `formal_results/` | Computations behind Part II of the online appendix |
| `online_appendix/` | The online appendix (`online_appendix.pdf`), its source, the script that writes its tables, and the checks of its numbers |

The online appendix has three parts: supplementary results in the order of the
paper, formal results with proofs, and a replication guide that names the script
and the output file behind each table and figure.

## Data

The repository contains every aggregate series and every file of estimates.
It does not contain person-level records, which their providers license to
registered users and do not allow to be redistributed.

| Data | In the repository | Access |
|---|---|---|
| China Family Panel Studies, waves 2010 to 2022 | No | Institute of Social Science Survey, Peking University, https://www.isss.pku.edu.cn/cfps/ |
| Census samples of 1990 and 2000 (one percent) | No | IPUMS International, https://international.ipums.org |
| Provincial intensity of the expansion, cohort exposure series | Yes | `college_expansion/data/external/` |
| Registrations for the college entrance examination by province | Yes | `college_expansion/data/external/cee_registrants_province.csv` |
| Exports and GDP by province in 2000, consumer prices by province | Yes | `college_expansion/data/external/` |

To run the pipeline from the raw data, place the survey files of each wave in
`income_dynamics/data/raw/CFPS2010`, ..., `CFPS2022` and the IPUMS International
extract, saved as `college_expansion/data/external/ipumsi_00001.csv.gz` with the
variables YEAR, GEO1_CN, AGE, EDATTAIN, and PERWT for the 1990 and 2000 samples,
then run `./run_all.sh --from-raw`. Run it once; the default run reuses the
person-level panels it wrote. Those files are listed in `.gitignore`.

## Running

Python 3.12 with the libraries in `requirements.txt` (`msoffcrypto-tool` is needed
only by `college_expansion/code/ws_extract_cnki_cee.py`, which parses the
yearbook tables behind the registration series). The dynamic-panel estimates
are also computed in Stata 18 with `xtabond2`; the two do-files in
`income_dynamics/code/stata/` are run by hand from that folder and are not part
of `run_all.sh`.

```bash
pip install -r requirements.txt
```

```bash
./run_all.sh
```

`run_all.sh` runs the estimation scripts in order (they need the person-level
files), rebuilds the tables of the appendix, and runs the three checks. The
checks alone need only the files in the repository and run in seconds; run them
with `./run_all.sh --checks` or one at a time:

```bash
python3 unified/code/check_paper_numbers.py
```

```bash
python3 online_appendix/build/check_part1_numbers.py
```

```bash
python3 online_appendix/build/check_part2_numbers.py
```

Each check reads the files of estimates, formats every value as the paper or the
appendix prints it, and fails if the printed string is absent from the source.

## Names in the code

Scripts, functions, and files of estimates keep names from earlier stages of the
project. `mature_age` and `old` refer to the level of earnings at ages 30 and
over; `band` to the two rows of the regression of Section 6 of the paper, one
row per child below 30 and one at 30 and over; `HP14` and `HighParentOcc14` to
the family-background indicator (a parent in a professional occupation when the
child was fourteen); `ordinary` in the files of estimates to less advantaged
children. `IGM_PROJECT_ROOT`, the environment variable that `_paths.py` reads,
carries the project's earlier name. `unified/code/sample_mature_age.py` defines
the sample and is the place to start reading.

## Compiling the documents

```bash
cd unified/paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

```bash
cd online_appendix && pdflatex online_appendix && bibtex online_appendix && pdflatex online_appendix && pdflatex online_appendix
```

The appendix reads the labels of the paper from `unified/paper/main.aux`, which
is kept in the repository for that purpose.

## Contact

jj4608@stern.nyu.edu
