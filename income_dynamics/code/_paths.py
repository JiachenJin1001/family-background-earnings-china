"""
_paths.py
Project-root resolver for the replication package.

The scripts of this folder import BASE_RAW and BASE_INT from this module
rather than hardcoding absolute paths. This makes the package
runnable from any clone of the repository.

Resolution order:
1. Environment variable IGM_PROJECT_ROOT (highest priority; IGM is the
   project's earlier name).
2. Auto-detect: walk up from this file's directory until we see a sibling
   `data/` directory.
3. Fallback: assume `code/` is one level below project root.
"""
import os

def _resolve_root():
    env = os.environ.get("IGM_PROJECT_ROOT")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(5):
        if os.path.isdir(os.path.join(cur, "data")) and \
           os.path.isdir(os.path.join(cur, "code")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # Fallback: parent of code/
    return os.path.dirname(here)


PROJECT_ROOT = _resolve_root()
BASE_RAW     = os.path.join(PROJECT_ROOT, "data", "raw")
BASE_INT     = os.path.join(PROJECT_ROOT, "data", "intermediate")
WS_INT       = os.path.join(BASE_INT, "ws")
TABLES_DIR   = os.path.join(PROJECT_ROOT, "output", "tables")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "output", "figures")

if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"BASE_RAW     = {BASE_RAW}")
    print(f"BASE_INT     = {BASE_INT}")
    print(f"WS_INT       = {WS_INT}")
    for d in [BASE_RAW, BASE_INT, TABLES_DIR, FIGURES_DIR]:
        ok = "OK " if os.path.isdir(d) else "MISSING"
        print(f"  [{ok}] {d}")
