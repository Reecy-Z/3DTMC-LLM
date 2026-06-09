"""
Backward-compatible entry point for TmQM property training.

Prefer:
  deepspeed Stage3.py --task homo_lumo_gap --mode single_token
  deepspeed Stage3.py --task dipole_moment --mode freeze_3d
"""
from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Stage3 import main, run_training  # noqa: E402
from task_datasets import PROPERTY_CONFIG, TmQMgSingleTokenUnimolDataset  # noqa: E402

__all__ = ["PROPERTY_CONFIG", "TmQMgSingleTokenUnimolDataset", "main", "run_training"]

if __name__ == "__main__":
    main()
