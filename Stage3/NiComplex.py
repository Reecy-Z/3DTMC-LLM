"""
Backward-compatible entry point for NiComplex ΔΔG training.

Prefer:
  deepspeed Stage3.py --task nicomplex_ddg --lmdb /path/data.lmdb --split_seed 38
"""
from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Stage3 import main  # noqa: E402
from task_datasets import (  # noqa: E402
    NI_INSTRUCTION,
    NiComplexDDGDataset,
    load_merged_valid_nicomplex_records,
    read_nicomplex_lmdb,
)

__all__ = [
    "NI_INSTRUCTION",
    "NiComplexDDGDataset",
    "load_merged_valid_nicomplex_records",
    "read_nicomplex_lmdb",
    "main",
]

if __name__ == "__main__":
    main(default_task="nicomplex_ddg")
