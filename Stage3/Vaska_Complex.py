"""
Backward-compatible entry point for Vaska barrier training.

Prefer:
  deepspeed Stage3.py --task vaska_barrier --lmdb /path/data.lmdb --split_seed 43
"""
from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Stage3 import main  # noqa: E402
from task_datasets import VASKA_INSTRUCTION, VaskaComplexDataset, read_vaska_lmdb  # noqa: E402

__all__ = ["VASKA_INSTRUCTION", "VaskaComplexDataset", "read_vaska_lmdb", "main"]

if __name__ == "__main__":
    main(default_task="vaska_barrier")
