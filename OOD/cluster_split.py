"""Cluster-based train/test split from tmQMg CSV(s) (CSD code keys).

Supports:
  - Single CSV with ``CSD code`` + ``split`` columns (e.g. cluster_split_k200.csv)
  - Directory with ``train.csv`` and ``test.csv`` (e.g. split_k200_far_from_train/)
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Set


def csd_from_lmdb_key(key_bytes) -> str:
    if isinstance(key_bytes, bytes):
        return key_bytes.decode("utf-8", errors="replace").strip()
    return str(key_bytes).strip()


def load_cluster_split(csv_path: str) -> Dict[str, str]:
    """Return mapping CSD code -> split label (e.g. train / test)."""
    split_map: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            csd = (row.get("CSD code") or row.get("csd_code") or "").strip()
            split = (row.get("split") or "").strip().lower()
            if csd and split:
                split_map[csd] = split
    if not split_map:
        raise RuntimeError(f"No split entries loaded from {csv_path}")
    return split_map


def csd_codes_for_split(split_map: Dict[str, str], split_name: str) -> Set[str]:
    want = split_name.strip().lower()
    return {csd for csd, split in split_map.items() if split == want}


def summarize_splits(split_map: Dict[str, str]) -> Dict[str, int]:
    from collections import Counter

    c = Counter(split_map.values())
    return dict(c)


def _csd_from_row(row: dict) -> str:
    return (row.get("CSD code") or row.get("csd_code") or "").strip()


def load_csd_codes_from_list_csv(csv_path: str) -> Set[str]:
    """Load CSD codes from a single-list CSV (no ``split`` column)."""
    codes: Set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            csd = _csd_from_row(row)
            if csd:
                codes.add(csd)
    if not codes:
        raise RuntimeError(f"No CSD codes loaded from {csv_path}")
    return codes


def is_split_directory(split_source: str) -> bool:
    if not os.path.isdir(split_source):
        return False
    return os.path.isfile(os.path.join(split_source, "train.csv")) and os.path.isfile(
        os.path.join(split_source, "test.csv")
    )


def load_csd_codes_for_split(split_source: str, split_name: str) -> Set[str]:
    """``split_source``: combined split CSV, or directory with train.csv / test.csv."""
    want = split_name.strip().lower()
    if want not in ("train", "test"):
        raise ValueError(f"split_name must be 'train' or 'test', got {split_name!r}")

    if is_split_directory(split_source):
        csv_path = os.path.join(split_source, f"{want}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing split file: {csv_path}")
        return load_csd_codes_from_list_csv(csv_path)

    split_map = load_cluster_split(split_source)
    return csd_codes_for_split(split_map, want)


def summarize_split_source(split_source: str) -> Dict[str, int]:
    if is_split_directory(split_source):
        return {
            "train": len(load_csd_codes_from_list_csv(os.path.join(split_source, "train.csv"))),
            "test": len(load_csd_codes_from_list_csv(os.path.join(split_source, "test.csv"))),
        }
    return summarize_splits(load_cluster_split(split_source))


def split_output_suffix(split_source: str) -> str:
    """Basename tag for checkpoint dirs (e.g. split_k200_far_from_train)."""
    if is_split_directory(split_source):
        return os.path.basename(os.path.normpath(split_source))
    base = os.path.basename(split_source)
    if base.endswith(".csv"):
        base = base[: -len(".csv")]
    return base or "cluster_split"
