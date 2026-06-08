"""OOD train/test splits for NiComplex ΔΔG (ligand scaffold LOLO + reaction-type OOD)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Set, Tuple

ALL_LIGAND_SCAFFOLDS = frozenset({"Box", "Biox", "Pybox", "Pyox", "Biim", "Diamine"})

# Five experiments requested: 3 ligand-scaffold holdouts + 2 reaction-type OOD splits.
OOD_EXPERIMENTS = (
    {
        "name": "train_rest_test_Pybox",
        "split_col": "L_Scaffold",
        "train_types": ALL_LIGAND_SCAFFOLDS - {"Pybox"},
        "test_types": {"Pybox"},
        "description": "Leave-one-ligand-scaffold-out: test Pybox",
    },
    {
        "name": "train_rest_test_Biox",
        "split_col": "L_Scaffold",
        "train_types": ALL_LIGAND_SCAFFOLDS - {"Biox"},
        "test_types": {"Biox"},
        "description": "Leave-one-ligand-scaffold-out: test Biox",
    },
    {
        "name": "train_rest_test_Biim",
        "split_col": "L_Scaffold",
        "train_types": ALL_LIGAND_SCAFFOLDS - {"Biim"},
        "test_types": {"Biim"},
        "description": "Leave-one-ligand-scaffold-out: test Biim",
    },
    {
        "name": "train_rest_test_Box",
        "split_col": "L_Scaffold",
        "train_types": ALL_LIGAND_SCAFFOLDS - {"Box"},
        "test_types": {"Box"},
        "description": "Leave-one-ligand-scaffold-out: test Box",
    },
    {
        "name": "train_rest_test_Pyox",
        "split_col": "L_Scaffold",
        "train_types": ALL_LIGAND_SCAFFOLDS - {"Pyox"},
        "test_types": {"Pyox"},
        "description": "Leave-one-ligand-scaffold-out: test Pyox",
    },
    {
        "name": "train_NiH+Csp2_test_Csp3",
        "split_col": "R_Type",
        "train_types": {"NiH", "C(sp3)-C(sp2)"},
        "test_types": {"C(sp3)-C(sp3)"},
        "description": "Train NiH + C(sp3)-C(sp2); test C(sp3)-C(sp3)",
    },
    {
        "name": "train_NiH+Csp3_test_Csp2",
        "split_col": "R_Type",
        "train_types": {"NiH", "C(sp3)-C(sp3)"},
        "test_types": {"C(sp3)-C(sp2)"},
        "description": "Train NiH + C(sp3)-C(sp3); test C(sp3)-C(sp2)",
    },
)

EXPERIMENT_NAMES = tuple(spec["name"] for spec in OOD_EXPERIMENTS)


def experiment_dirname(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_+-]+", "_", name.strip())
    return f"exp_{safe}"


def get_experiment(name: str) -> dict:
    for spec in OOD_EXPERIMENTS:
        if spec["name"] == name:
            return spec
    raise ValueError(f"Unknown experiment={name!r}; expected one of {EXPERIMENT_NAMES}")


def sample_matches_types(sample: dict, split_col: str, types: Set[str]) -> bool:
    value = sample.get(split_col)
    return value in types


def split_by_experiment(
    samples: List[dict],
    experiment_name: str,
) -> Tuple[List[dict], List[dict], dict]:
    spec = get_experiment(experiment_name)
    split_col = spec["split_col"]
    train_types = set(spec["train_types"])
    test_types = set(spec["test_types"])

    train = [s for s in samples if sample_matches_types(s, split_col, train_types)]
    ood_test = [s for s in samples if sample_matches_types(s, split_col, test_types)]
    if not train:
        raise RuntimeError(f"No train samples for experiment={experiment_name!r}")
    if not ood_test:
        raise RuntimeError(f"No OOD test samples for experiment={experiment_name!r}")
    return train, ood_test, spec


def summarize_field(samples: List[dict], field: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for sample in samples:
        value = sample.get(field)
        if value is not None:
            counts[str(value)] += 1
    return dict(counts)
