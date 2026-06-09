"""Deprecated: use ``python inference.py stage3 --task <property> --split_csv ...``."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from inference import main  # noqa: E402

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--task" not in argv and "--property" in argv:
        pass
    elif "--task" not in argv:
        argv = ["--task", "dipole_moment", *argv]
    main(["stage3", *argv])
