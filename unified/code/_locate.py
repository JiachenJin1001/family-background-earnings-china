"""Locate the sibling folders of the package (unified, income_dynamics,
college_expansion, formal_results) from any script. `sibling` walks up from
the calling file until it finds a directory that contains the folder.
"""
import os

NAMES = {
    "unified": ("unified",),
    "income_dynamics": ("income_dynamics",),
    "college_expansion": ("college_expansion",),
    "formal_results": ("formal_results",),
}


def sibling(key, start=None):
    cur = os.path.dirname(os.path.abspath(start or __file__))
    for _ in range(8):
        for name in NAMES[key]:
            cand = os.path.join(cur, name)
            if os.path.isdir(cand):
                return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError(f"none of {NAMES[key]} found above {start or __file__}")
