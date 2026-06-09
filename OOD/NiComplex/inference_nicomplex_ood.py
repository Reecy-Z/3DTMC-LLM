"""Deprecated: use ``python inference.py stage3 --task nicomplex_ddg --ood_experiment ...``."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from inference import main  # noqa: E402

if __name__ == "__main__":
    argv = ["stage3", "--task", "nicomplex_ddg", *sys.argv[1:]]
    main(argv)
