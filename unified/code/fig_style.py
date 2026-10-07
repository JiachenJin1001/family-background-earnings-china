"""Shared style of the five figures of the paper.

The figures are drawn close to the size at which they are printed, so that the text inside them prints at about 7 to 9
points. The text is set in Latin Modern Roman, the typeface of the paper, and each figure is saved as a PNG and
as a vector PDF; the paper includes the PDF. The two family-background groups are drawn in an orange and a dark blue that differ in lightness, so
that they remain distinguishable in a black-and-white print, and they also differ in marker shape.

FIG_LANG=cn draws the same figure with Chinese text and writes it to paper/figures/cn/, when the module _cn_labels
(the labels of the Chinese version of the paper) is present.
"""
import os
import matplotlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
LANG = os.environ.get("FIG_LANG", "en")
ADV, LESS = "#d95f02", "#08519c"        # advantaged children (orange), less advantaged children (dark blue)
SHRINK = 0.64                           # canvas size in inches as a fraction of the nominal size


def fs(size):
    """Canvas size in inches: the nominal size times SHRINK."""
    return (size[0] * SHRINK, size[1] * SHRINK)


def setup():
    matplotlib.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                                "mathtext.fontset": "cm"})   # fonts are embedded as outlines (matplotlib default), which every viewer renders
    lm = [p for p in _latin_modern() if os.path.exists(p)]
    if lm:
        from matplotlib import font_manager
        for p in lm: font_manager.fontManager.addfont(p)
        matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"]})
    else:
        matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["CMU Serif", "Times New Roman", "DejaVu Serif"]})
    if LANG == "cn":
        matplotlib.rcParams.update({"font.family": ["Songti SC", "Heiti TC", "Arial Unicode MS"], "axes.unicode_minus": False, "mathtext.fontset": "stix",
                                    "font.weight": "regular", "axes.titleweight": "regular", "axes.labelweight": "regular"})


try:
    from _cn_labels import FIG_CN as CN
except ImportError:
    CN = {}


def tr(s):
    """The Chinese text of a label when FIG_LANG=cn; the label itself otherwise."""
    return CN.get(s, s) if LANG == "cn" else s


def _latin_modern():
    """Paths of the Latin Modern Roman font files of the TeX installation (empty if kpsewhich does not find them)."""
    import subprocess
    out = []
    for f in ("lmroman10-regular.otf", "lmroman10-bold.otf", "lmroman10-italic.otf"):
        try:
            p = subprocess.run(["kpsewhich", f], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            p = ""
        if p: out.append(p)
    return out


def save(fig, name):
    """Write the figure as name (a PNG at 300 dpi) and as a vector PDF beside it; returns the path of the PNG."""
    out = outpath(name)
    fig.savefig(out, dpi=300, bbox_inches="tight"); fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
    return out


def outpath(name):
    d = os.path.join(ROOT, "paper", "figures") if LANG == "en" else os.path.join(ROOT, "paper", "figures", "cn")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)
